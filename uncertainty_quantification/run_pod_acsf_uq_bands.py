"""
run_pod_acsf_uq_bands.py
========================
Relaxes twisted bilayer graphene (TBLG) for a range of twist angles using an
MCMC ensemble of POD potentials, then computes tight-binding band structures
using a paired MCMC ensemble of ACSF linear hopping models.

Each of the 75 ensemble samples uses the *same* index for both the POD
potential and the ACSF hopping model (paired sampling).

Output structure
----------------
{pod_label}_{acsf_label}/
  theta_{angle}/
    sample_{j}/
      relaxed.traj    — ASE trajectory of the relaxed structure
      lbfgs.log       — LBFGS optimiser log
      bands.npz       — dict with keys: evals, kvec, k_dist, k_node

Configuration
-------------
Adjust the ALL_CAPS variables in the "Configuration" section below to change
temperature weights, basis sizes, cutoffs, convergence thresholds, etc.
"""

from __future__ import annotations

import os
import pickle
import sys

import ase.io
import flatgraphene as fg
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from ase.optimize import LBFGS

from blg_model_builder.potentials import PODASECalculator
from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
from blg_model_builder.tb_models import (
    get_acsf_hoppings,
    get_recip_cell,
    k_path,
)

# ---------------------------------------------------------------------------
# Configuration — edit these variables to select ensembles / adjust settings
# ---------------------------------------------------------------------------

# Temperature weights used to select the ensemble pkl files.
# E.g. POD_TEMPERATURE = 0.5 → POD_energy_ensemble_T_0.5.pkl
POD_TEMPERATURE: float = 1e-5
ACSF_TEMPERATURE: float = 0.0001

# ACSF hopping model basis parameters (must match the ensemble pkl filenames).
ACSF_M: int = 15   # number of Chebyshev radial basis functions
ACSF_W: int = 6   # number of angular (cos^w) exponents

# Real-space cutoffs (Å)
POD_RCUT: float = 5.0   # POD potential cutoff (matches training)
ACSF_RCUT: float = 6.0  # ACSF hopping cutoff (default in get_acsf_hopping_descriptors)

# Ensemble size
N_ENSEMBLES: int = 75

# Relaxation settings
FMAX: float = 1e-4    # force convergence threshold (eV/Å)
MAX_STEPS: int = 150  # maximum LBFGS steps

# Band structure settings
N_EIGS: int = 40  # number of eigenvalues to find near the Fermi level (E=0)
NK: int = 60       # k-points per high-symmetry segment

# Twist angles (degrees) to process
TWIST_ANGLES: np.ndarray = np.array([2.88,0.83, 0.88, 0.93, 0.99, 1.08, 1.12, 1.16, 1.2])

# Graphene lattice parameters (Å)
LAYER_SEP: float = 3.35  # interlayer separation
A_LAT: float = 2.46       # in-plane lattice constant

# Random seed for ensemble subsampling (reproducibility)
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# POD hyperparameters (fixed architecture matching the trained ensemble)
# Source: uncertainty_quantification/get_MCMC_inputs.py and tests/test_relaxation.py
# ---------------------------------------------------------------------------

POD_HYPERPARAMS: dict = {
    "bessel_polynomial_degree": 4,
    "inverse_polynomial_degree": 8,
    "twobody_number_radial_basis_functions": 10,
    "threebody_number_radial_basis_functions": 8,
    "threebody_angular_degree": 4,
    "fourbody_number_radial_basis_functions": 6,
    "fourbody_angular_degree": 3,
    "fivebody_number_radial_basis_functions": 4,
    "fivebody_angular_degree": 3,
    "sixbody_number_radial_basis_functions": 3,
    "sixbody_angular_degree": 2,
    "sevenbody_number_radial_basis_functions": 2,
    "sevenbody_angular_degree": 2,
}


# ---------------------------------------------------------------------------
# Derived labels / paths (do not edit)
# ---------------------------------------------------------------------------

def _pod_label() -> str:
    return f"POD_energy_T{POD_TEMPERATURE}"


def _acsf_label() -> str:
    return f"ACSF_hoppings_M{ACSF_M}_W{ACSF_W}_T{ACSF_TEMPERATURE}"


def _results_root() -> str:
    return f"{_pod_label()}_{_acsf_label()}"


def _pod_ensemble_path() -> str:
    return os.path.join(
        "ensembles", "POD_energy",
        f"POD_energy_ensemble_T_{POD_TEMPERATURE}.pkl",
    )


def _acsf_ensemble_path() -> str:
    return os.path.join(
        "ensembles", f"ACSF_hoppings_M_{ACSF_M}_W_{ACSF_W}",
        f"ACSF_hoppings_M_{ACSF_M}_W_{ACSF_W}_ensemble_T_{ACSF_TEMPERATURE}.pkl",
    )


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def get_twist_geom(theta: float) -> ase.Atoms:
    """Build an unrelaxed twisted-bilayer-graphene supercell for *theta* degrees."""
    p, q, _ = fg.twist.find_p_q(theta)
    atoms = fg.twist.make_graphene(
        cell_type="hex",
        n_layer=2,
        p=p,
        q=q,
        lat_con=A_LAT,
        sym=["C", "C"],
        mass=[12.01, 12.01],
        sep=LAYER_SEP,
        h_vac=20,
    )
    return atoms


# ---------------------------------------------------------------------------
# Band-structure builder
# ---------------------------------------------------------------------------

def build_band_structure(
    atoms: ase.Atoms,
    acsf_params: np.ndarray,
    kvec_cart: np.ndarray,
) -> np.ndarray:
    """
    Compute the ACSF tight-binding band structure along *kvec_cart*.

    Steps
    -----
    1. Compute ACSF descriptors and pair indices / bond vectors.
    2. Evaluate linear hopping model:  t_ij = descriptors @ acsf_params.
    3. For each k-point build a sparse Bloch Hamiltonian, symmetrize it,
       then find *N_EIGS* eigenvalues nearest to the Fermi level (E = 0).

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed TBLG structure.
    acsf_params : np.ndarray, shape (n_features,)
        ACSF linear model weights for one ensemble sample.
    kvec_cart : np.ndarray, shape (n_kpts, 3)
        k-points in Cartesian coordinates (Å⁻¹, with 2π).

    Returns
    -------
    evals : np.ndarray, shape (n_kpts, N_EIGS)
        Sorted real eigenvalues (eV) near the Fermi level at each k-point.
    """
    # --- Step 1: descriptors and pair geometry ---
    descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
        atoms,
        M=ACSF_M,
        W=ACSF_W,
        r_cut=ACSF_RCUT,
    )
    # pair_v : (n_pairs, 3) Cartesian bond vectors in Å (lattice offsets included)

    # --- Step 2: hopping amplitudes ---
    hoppings = get_acsf_hoppings(descriptors, acsf_params)  # (n_pairs,)

    N = len(atoms)
    evals_list: list[np.ndarray] = []

    # --- Step 3: Bloch Hamiltonian at each k-point ---
    for k_cart in kvec_cart:
        # Bloch phases: exp(i k · r_{ij})  — r_{ij} in Å, k in Å⁻¹
        phases = np.exp(1j * (pair_v @ k_cart))   # (n_pairs,)
        hop_vals = hoppings * phases                # complex (n_pairs,)

        # Build sparse upper-triangle Hamiltonian from the computed pairs
        H = scipy.sparse.coo_matrix(
            (hop_vals, (pair_i, pair_j)),
            shape=(N, N),
            dtype=np.complex128,
        ).tocsr()

        # NeighborList(bothways=False) is a half-list: each bond stored once.
        # Adding H† fills the missing triangle.  No factor of ½ — dividing
        # would halve hopping amplitudes and give 2× too-small bandwidth.
        H = H + H.conj().T

        # sigma=0 activates shift-invert mode: eigenvalues closest to E=0 are
        # returned first.  Guard k < N as required by ARPACK.
        k_req = min(N_EIGS, N - 2)
        vals, _ = scipy.sparse.linalg.eigsh(H, k=k_req, sigma=0.0)
        evals_list.append(np.sort(vals.real))

    return np.array(evals_list)  # (n_kpts, N_EIGS)


# ---------------------------------------------------------------------------
# Ensemble loading helpers
# ---------------------------------------------------------------------------

def _load_and_subsample(path: str, key: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Load an MCMC ensemble pkl and subsample *n* rows if necessary."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ensemble file not found: {path}\n"
            "Check that POD_TEMPERATURE / ACSF_TEMPERATURE / ACSF_M / ACSF_W "
            "match an existing file in the ensembles/ directory."
        )
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    arr = np.asarray(data["ensemble"][key], dtype=np.float64)
    if arr.shape[0] > n:
        idx = rng.choice(arr.shape[0], size=n, replace=False)
        arr = arr[idx]
    return arr  # (n, n_params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Load ensembles ---
    print(f"Loading POD ensemble from:  {_pod_ensemble_path()}")
    pod_ensemble = _load_and_subsample(
        _pod_ensemble_path(), key="energy", n=N_ENSEMBLES, rng=rng
    )
    print(f"  → {pod_ensemble.shape[0]} samples, {pod_ensemble.shape[1]} coefficients")

    print(f"Loading ACSF ensemble from: {_acsf_ensemble_path()}")
    acsf_ensemble = _load_and_subsample(
        _acsf_ensemble_path(), key="hopping", n=N_ENSEMBLES, rng=rng
    )
    print(f"  → {acsf_ensemble.shape[0]} samples, {acsf_ensemble.shape[1]} features")

    n_samples = min(pod_ensemble.shape[0], acsf_ensemble.shape[0])
    print(f"Using {n_samples} paired ensemble samples.\n")

    # --- k-path: K → Γ → M → K ---
    K      = [1 / 3, 2 / 3, 0]
    Gamma  = [0,     0,     0]
    M_pt   = [1 / 2, 0,     0]
    sym_pts = [K, Gamma, M_pt, K]
    kvec, k_dist, k_node = k_path(sym_pts, NK)
    # kvec : (n_kpts, 3) in reduced coordinates — converted to Cartesian per angle

    results_root = _results_root()
    os.makedirs(results_root, exist_ok=True)

    # --- Main loop: angles × ensemble samples ---
    for theta in TWIST_ANGLES:
        print(f"\n{'='*60}")
        print(f" Twist angle: {theta}°  ({n_samples} samples)")
        print(f"{'='*60}")

        atoms_init = get_twist_geom(theta)
        print(f"  Supercell: {len(atoms_init)} atoms")

        theta_dir = os.path.join(results_root, f"theta_{theta}")
        os.makedirs(theta_dir, exist_ok=True)

        # k → Cartesian conversion uses the *unrelaxed* reciprocal cell as a
        # reference; the actual conversion is done after relaxation per sample
        # because the cell is fixed (LBFGS with fixed cell relaxes only ions).

        for j in range(n_samples):
            sample_dir = os.path.join(theta_dir, f"sample_{j}")
            traj_path  = os.path.join(sample_dir, "relaxed.traj")
            bands_path = os.path.join(sample_dir, "bands.npz")

            # Resume: skip if both outputs already exist
            if os.path.exists(traj_path) and os.path.exists(bands_path):
                print(f"  [skip] sample {j:3d} — outputs already exist")
                continue

            os.makedirs(sample_dir, exist_ok=True)
            print(f"  sample {j:3d}/{n_samples-1}", end="  ", flush=True)

            # ----------------------------------------------------------------
            # Step 1: Relax with POD potential
            # ----------------------------------------------------------------
            pod_calc = PODASECalculator(
                POD_HYPERPARAMS,
                pod_ensemble[j],
                elements=["C"],
                cutoff=POD_RCUT,
            )
            atoms = atoms_init.copy()
            atoms.calc = pod_calc

            log_path = os.path.join(sample_dir, "lbfgs.log")
            dyn = LBFGS(
                atoms,
                logfile=log_path,
                trajectory=os.path.join(sample_dir, "lbfgs.traj"),
            )
            converged = dyn.run(fmax=FMAX, steps=MAX_STEPS)

            # Compute fmax while atoms still has the calculator attached
            fmax_actual = np.max(np.linalg.norm(atoms.get_forces(), axis=1))

            # Write a clean copy of the relaxed structure (without calculator)
            relaxed = atoms.copy()
            ase.io.write(traj_path, relaxed)

            print(f"relax ✓ (fmax={fmax_actual:.2e} eV/Å, converged={converged})", end="  ", flush=True)

            # ----------------------------------------------------------------
            # Step 2: Band structure with ACSF hoppings
            # ----------------------------------------------------------------
            cell  = np.array(relaxed.get_cell())
            recip = get_recip_cell(cell.T)         # cell.T: columns = lattice vectors
            kvec_cart = kvec @ recip               # (n_kpts, 3) in Å⁻¹ with 2π

            evals = build_band_structure(relaxed, acsf_ensemble[j], kvec_cart)

            np.savez(
                bands_path,
                evals=evals,        # (n_kpts, N_EIGS)
                kvec=kvec,          # (n_kpts, 3) reduced coordinates
                k_dist=k_dist,      # (n_kpts,)  accumulated distance for x-axis
                k_node=k_node,      # (4,)       high-symmetry point positions
                sym_pts=np.array(sym_pts),  # [[K],[Γ],[M],[K]]
            )
            print(f"bands ✓  evals shape={evals.shape}")

    print(f"\nAll done. Results in: {os.path.abspath(results_root)}")


if __name__ == "__main__":
    main()
