#!/usr/bin/env python3
"""
run_uq_propagation_flat_bands_U.py
==================================
Propagate tight-binding uncertainties into the moiré flat-band density–density
interaction energy ``U``.

For each LAMMPS model (``--models``) and twist angle (``--twist-angle``):

1. Load a TBLG structure (relaxed trajectory or ``--unrelaxed`` commensurate cell).
2. Load an MCMC ensemble of ACSF hopping parameters (``--tb-model``).
3. Diagonalize the Bloch Hamiltonian on a uniform k-point mesh (default 11×11×1).
4. Identify the two highest occupied bands at charge neutrality
   (indices ``N//2 - 2`` and ``N//2 - 1``, with ``E_F = evals[k, N//2]``).
5. Evaluate

   .. math::

       U = \\frac{1}{N_{kp}} \\sum_{n,k,i,j}
           |c_{i,n,k}|^2 |c_{j,n,k}|^2
           \\frac{e}{4\\pi\\varepsilon_0 \\varepsilon_r |R_i - R_j|}

   where ``c_{i,n,k}`` are eigenvector components of the flat bands and
   ``R_i`` are atomic positions (minimum-image convention in-plane).

6. Save ``U`` and metadata to ``bands/flat_band_U/``.

Examples
--------
::

    python run_uq_propagation_flat_bands_U.py --unrelaxed \\
        --twist-angle 9.43 --tb-model ACSF_hoppings_sk_M_10_W_6

    python run_uq_propagation_flat_bands_U.py \\
        --models 'POD_energy_POD_index*' --twist-angle 9.43
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# GPU / CPU array backend (mirrors run_uq_propagation_bands.py).
try:
    import cupy as xp
    import cupyx
    _GPU: bool = bool(xp.cuda.is_available())
    if not _GPU:
        xp = np
        cupyx = None  # type: ignore[assignment]
except (ImportError, Exception):
    xp = np
    cupyx = None  # type: ignore[assignment]
    _GPU = False

if _GPU:
    def _scatter_add(target, idx, src):
        if target.dtype.kind == "c":
            cupyx.scatter_add(target.real, idx, src.real)
            cupyx.scatter_add(target.imag, idx, src.imag)
        else:
            cupyx.scatter_add(target, idx, src)
    def _to_cpu(arr): return arr.get()
else:
    def _scatter_add(target, idx, src): xp.add.at(target, idx, src)
    def _to_cpu(arr): return np.asarray(arr)

HERE = os.path.dirname(os.path.abspath(__file__))

from blg_model_builder.tb_models import create_tb_model
from blg_model_builder.tb_descriptors import (
    get_acsf_hopping_descriptors,
    get_acsf_sk_hopping_descriptors,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    add_tb_model_arg,
    collect_workflow_hyperparams,
)

# Shared helpers / CLI defaults from the band-structure propagation script.
from run_uq_propagation_bands import (  # noqa: E402
    DEFAULT_CALIBRATION_METRICS_DIR,
    DEFAULT_ENSEMBLE_SHUFFLE_SEED,
    DEFAULT_TB_MODEL,
    DEFAULT_TB_RCUT,
    DEFAULT_TWIST_ANGLE,
    DEFAULT_TRAJECTORY_DIR,
    _discover_traj_files,
    _get_recip_cell,
    _parse_tb_model_name,
    _resolve_tb_ensemble,
    _safe_filename_part,
    _shuffle_ensemble,
)
from run_uq_propagation_relaxation import build_tblg_atoms  # noqa: E402

# e^2 / (4 pi epsilon_0) in eV·Å (|R| in Å → interaction energy in eV).
_COULOMB_EV_ANG: float = 14.3996

DEFAULT_OUTPUT_DIR = "bands/flat_band_U"
DEFAULT_K_GRID = (11, 11, 1)
DEFAULT_EPSILON_R = 12.0
UNRELAXED_MODEL_KEY = "unrelaxed_tblg"


# ---------------------------------------------------------------------------
# k-mesh
# ---------------------------------------------------------------------------

def _parse_k_grid(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in str(s).replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected k-grid 'nkx,nky,nkz' or 'nkx x nky x nkz', got {s!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _k_mesh(nkx: int, nky: int, nkz: int) -> np.ndarray:
    """Gamma-centred uniform mesh in fractional coordinates, shape (N_kp, 3)."""
    def _axis(n: int) -> np.ndarray:
        if n <= 1:
            return np.array([0.0])
        return (np.arange(n, dtype=float) + 0.5) / n - 0.5

    ku = _axis(nkx)
    kv = _axis(nky)
    kw = _axis(nkz)
    U, V, W = np.meshgrid(ku, kv, kw, indexing="ij")
    return np.stack([U.ravel(), V.ravel(), W.ravel()], axis=1)


# ---------------------------------------------------------------------------
# TB Hamiltonian → eigenpairs on k-mesh
# ---------------------------------------------------------------------------

def _prepare_tb_geometry(
    atoms,
    params: np.ndarray,
    M: int,
    W: int,
    r_cut: float,
    *,
    is_sk: bool,
    extra_hp: dict | None,
):
    extra_hp = dict(extra_hp or {})
    desc_kw = {}
    if "use_envelope" in extra_hp:
        desc_kw["use_envelope"] = extra_hp["use_envelope"]

    if is_sk:
        descriptors, (pair_i, pair_j, pair_v) = get_acsf_sk_hopping_descriptors(
            atoms, M=M, W=W, r_cut=r_cut, **desc_kw,
        )
    else:
        descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=r_cut, **desc_kw,
        )

    tb_name = f"ACSF_hoppings_sk_M_{M}_W_{W}" if is_sk else f"ACSF_hoppings_M_{M}_W_{W}"
    tb_model = create_tb_model(tb_name, {"M": M, "W": W, "r_cut": r_cut, **extra_hp})
    hoppings = tb_model(descriptors, params)
    return pair_i, pair_j, pair_v, hoppings


def _diagonalize_k_mesh(
    atoms,
    params: np.ndarray,
    kvec_cart: np.ndarray,
    M: int,
    W: int,
    r_cut: float,
    *,
    is_sk: bool = False,
    extra_hp: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return eigenvalues, eigenvectors, and flat-band indices.

    Returns
    -------
    evals : ndarray (n_kpts, N)
    evecs : ndarray (n_kpts, N, N) — columns are Bloch eigenvectors
    flat_band_indices : tuple[int, int]
        Indices of the two occupied flat bands (``nocc-2``, ``nocc-1``).
    """
    pair_i, pair_j, pair_v, hoppings = _prepare_tb_geometry(
        atoms, params, M, W, r_cut, is_sk=is_sk, extra_hp=extra_hp,
    )

    N = len(atoms)
    nocc = N // 2
    if nocc < 2:
        raise ValueError(f"Need at least 4 orbitals for two flat bands below E_F; got N={N}")

    pair_v_xp = xp.asarray(pair_v, dtype=xp.float64)
    hoppings_xp = xp.asarray(hoppings, dtype=xp.float64)
    pair_i_xp = xp.asarray(pair_i)
    pair_j_xp = xp.asarray(pair_j)
    kvec_xp = xp.asarray(kvec_cart, dtype=xp.float64)
    lin_idx = pair_i_xp * N + pair_j_xp

    evals_list: list[np.ndarray] = []
    evecs_list: list[np.ndarray] = []

    for k_cart in kvec_xp:
        phases = xp.exp(1j * (pair_v_xp @ k_cart))
        hop_vals = (hoppings_xp * phases).astype(xp.complex128)
        H = xp.zeros(N * N, dtype=np.complex128)
        _scatter_add(H, lin_idx, hop_vals)
        H = xp.asarray(H).reshape(N, N)
        H = H + H.conj().T
        w, v = xp.linalg.eigh(H)
        evals_list.append(_to_cpu(w).astype(np.float64))
        evecs_list.append(_to_cpu(v).astype(np.complex128))

    evals = np.stack(evals_list, axis=0)
    evecs = np.stack(evecs_list, axis=0)
    flat_band_indices = (nocc - 2, nocc - 1)
    return evals, evecs, flat_band_indices


# ---------------------------------------------------------------------------
# Flat-band interaction U
# ---------------------------------------------------------------------------

def _coulomb_matrix_ev(
    positions: np.ndarray,
    cell: np.ndarray,
    epsilon_r: float,
    *,
    pbc: Sequence[bool] = (True, True, False),
) -> np.ndarray:
    """Coulomb matrix V_ij in eV (diagonal set to zero)."""
    from ase.geometry import get_distances

    dist = get_distances(
        positions, positions, cell=cell, pbc=tuple(pbc),
    )[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        V = _COULOMB_EV_ANG / (epsilon_r * dist)
    np.fill_diagonal(V, 0.0)
    V[~np.isfinite(V)] = 0.0
    return V


def compute_flat_band_U(
    evecs: np.ndarray,
    flat_band_indices: Sequence[int],
    positions: np.ndarray,
    cell: np.ndarray,
    epsilon_r: float,
) -> float:
    """Density–density interaction U (eV) averaged over the k-mesh.

    The sum over all orbital pairs ``(i, j)`` counts each unordered pair
    twice because ``V`` and ``|c_i|^2 |c_j|^2`` are symmetric; the result is
    therefore divided by 2 (diagonal ``i = j`` terms are already zero).

    Parameters
    ----------
    evecs : ndarray (n_kpts, N, N)
        Bloch eigenvectors (columns) at each k-point.
    flat_band_indices : sequence of int
        Band indices ``n`` (two flat bands below ``E_F``).
    """
    V = _coulomb_matrix_ev(positions, cell, epsilon_r)
    n_kpts = evecs.shape[0]
    U_sum = 0.0

    for k in range(n_kpts):
        for band in flat_band_indices:
            c2 = np.abs(evecs[k, :, band]) ** 2
            # sum_{i,j} |c_i|^2 V_ij |c_j|^2 — each unordered i≠j pair appears twice
            U_sum += float(c2 @ V @ c2)

    return U_sum / (2.0 * n_kpts)


def fermi_energy_charge_neutral(evals: np.ndarray) -> np.ndarray:
    """Fermi energy at each k-point: ``evals[k, N//2]``."""
    nocc = evals.shape[1] // 2
    return evals[:, nocc]


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def _flat_u_output_path(
    output_dir: str,
    model_name: str,
    t_label: str,
    tb_model_canonical: str,
    tb_t_label: str,
    theta: float,
    sample_index: int,
) -> str:
    base = (
        f"{_safe_filename_part(model_name)}"
        f"_T{_safe_filename_part(t_label)}"
        f"_{_safe_filename_part(tb_model_canonical)}"
        f"_tbT{_safe_filename_part(tb_t_label)}"
        f"_theta{theta:g}deg"
        f"_sample{sample_index:04d}.npz"
    )
    return os.path.join(output_dir, base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Propagate hopping-parameter uncertainties into the moiré flat-band "
            "density–density interaction U."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    add_energy_models_arg(p, required=False)
    p.add_argument("--ensemble-dir", default="ensembles")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="POD ensemble temperature (for trajectory path resolution).",
    )
    p.add_argument(
        "--calibration-metrics-dir",
        default=DEFAULT_CALIBRATION_METRICS_DIR,
    )
    p.add_argument("--calibration-target", default="energy")
    p.add_argument("--calibration-technique", default="mcmc")

    add_tb_model_arg(p, default=DEFAULT_TB_MODEL)
    p.add_argument("--tb-temperature", type=float, default=None)
    p.add_argument("--tb-calibration-target", default="hopping")
    p.add_argument("--tb-rcut", type=float, default=DEFAULT_TB_RCUT)

    p.add_argument("--twist-angle", type=float, default=DEFAULT_TWIST_ANGLE)
    p.add_argument("--trajectory-dir", default=DEFAULT_TRAJECTORY_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    p.add_argument(
        "--k-grid",
        default="11,11,1",
        help="Monkhorst-Pack mesh nkx,nky,nkz in fractional coordinates (default: 11,11,1).",
    )
    p.add_argument(
        "--epsilon-r",
        type=float,
        default=DEFAULT_EPSILON_R,
        dest="epsilon_r",
        help=f"Relative dielectric constant ε_r in the Coulomb term (default: {DEFAULT_EPSILON_R}).",
    )
    p.add_argument(
        "--unrelaxed",
        action="store_true",
        help="Use a commensurate unrelaxed TBLG cell (flatgraphene) instead of relaxed trajectories.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_ENSEMBLE_SHUFFLE_SEED)
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit number of TB ensemble samples processed (default: all available).",
    )

    add_hyperparam_args(p)
    args, _unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, _unknown)
    if cli_hyperparams:
        print(f"TB CLI hyperparameters: {cli_hyperparams}", flush=True)

    os.chdir(HERE)

    from blg_model_builder.ensemble_io import (
        DEFAULT_CALIBRATION_METRICS_DIR as _PBF_CALIB_DIR,
        expand_model_patterns,
        load_ensemble_pickle,
        resolve_ensemble_pickle,
    )
    if args.calibration_metrics_dir == DEFAULT_CALIBRATION_METRICS_DIR:
        args.calibration_metrics_dir = _PBF_CALIB_DIR

    if not args.unrelaxed and not args.models:
        p.error("--models is required unless --unrelaxed is set.")

    try:
        tb_M, tb_W, tb_canonical = _parse_tb_model_name(args.tb_model)
    except ValueError as exc:
        p.error(str(exc))

    nkx, nky, nkz = _parse_k_grid(args.k_grid)
    kvec_red = _k_mesh(nkx, nky, nkz)
    n_kpts = len(kvec_red)
    print(
        f"\nTB model: {tb_canonical}  (M={tb_M}, W={tb_W}, r_cut={args.tb_rcut} Å)",
        flush=True,
    )
    print(f"k-mesh: {nkx}×{nky}×{nkz} = {n_kpts} points  ε_r={args.epsilon_r}", flush=True)

    try:
        tb_pkl_path, tb_t_used = _resolve_tb_ensemble(
            args.tb_model,
            args.ensemble_dir,
            args.tb_temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_technique=args.calibration_technique,
            calibration_target=args.tb_calibration_target,
        )
    except FileNotFoundError as exc:
        p.error(str(exc))

    tb_t_label = f"{tb_t_used:g}"
    tb_ens_dict = load_ensemble_pickle(tb_pkl_path)
    tb_ensemble = _shuffle_ensemble(
        np.asarray(tb_ens_dict["ensemble"]["hopping"], dtype=float),
        args.seed,
    )
    n_tb = tb_ensemble.shape[0]
    if args.n_samples is not None:
        n_tb = min(n_tb, int(args.n_samples))
        tb_ensemble = tb_ensemble[:n_tb]
    print(f"TB ensemble: {tb_pkl_path}  (T={tb_t_label}, {n_tb} samples)", flush=True)

    is_sk = tb_canonical.startswith("ACSF_hoppings_sk")
    extra_hp = cli_hyperparams or None

    if args.unrelaxed:
        model_list = [UNRELAXED_MODEL_KEY]
        t_labels = {UNRELAXED_MODEL_KEY: "unrelaxed"}
        print(f"\nUnrelaxed TBLG at θ={args.twist_angle:g}°", flush=True)
    else:
        model_list = expand_model_patterns(args.models, args.ensemble_dir)
        if not model_list:
            p.error("No models matched --models patterns.")
        t_labels = {}
        for model_name in model_list:
            try:
                _, t_used = resolve_ensemble_pickle(
                    model_name,
                    args.ensemble_dir,
                    args.temperature,
                    calibration_metrics_dir=args.calibration_metrics_dir,
                    calibration_technique=args.calibration_technique,
                    calibration_target=args.calibration_target,
                )
            except FileNotFoundError as exc:
                print(f"Warning: skipping {model_name!r}: {exc}", file=sys.stderr)
                continue
            t_labels[model_name] = f"{t_used:g}"
        model_list = [m for m in model_list if m in t_labels]
        if not model_list:
            p.error("No models with resolvable POD ensembles.")

    if args.unrelaxed:
        atoms_template = build_tblg_atoms(args.twist_angle)
        print(f"  n_atoms={len(atoms_template)}", flush=True)

    for model_name in model_list:
        print(f"\n{'=' * 60}", flush=True)
        print(f" Model: {model_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        t_label = t_labels[model_name]
        out_dir = os.path.join(
            args.output_dir,
            _safe_filename_part(model_name),
            f"T{t_label}",
            f"{_safe_filename_part(tb_canonical)}_tbT{tb_t_label}",
            f"theta{args.twist_angle:g}deg",
        )
        os.makedirs(out_dir, exist_ok=True)

        if args.unrelaxed:
            n_process = n_tb
        else:
            traj_files = _discover_traj_files(
                args.trajectory_dir, model_name, t_label, args.twist_angle,
            )
            n_trajs = len(traj_files)
            if n_trajs == 0:
                print(
                    f"  Warning: no trajectory files for {model_name!r} "
                    f"T={t_label} θ={args.twist_angle:g}°.",
                    file=sys.stderr,
                )
                continue
            print(f"  Found {n_trajs} trajectory file(s).", flush=True)
            n_process = n_trajs
            if n_tb < n_trajs:
                print(
                    f"  Warning: cycling TB ensemble ({n_tb} < {n_trajs} trajs).",
                    file=sys.stderr,
                )

        U_values: list[float] = []

        for sample_idx in range(n_process):
            npz_path = _flat_u_output_path(
                out_dir, model_name, t_label, tb_canonical,
                tb_t_label, args.twist_angle, sample_idx,
            )
            if os.path.isfile(npz_path):
                data = np.load(npz_path)
                U_values.append(float(data["U"]))
                print(f"  [skip] sample {sample_idx:04d} — exists, U={data['U']:.6f} eV", flush=True)
                continue

            if args.unrelaxed:
                atoms = atoms_template
            else:
                from ase.io import read as ase_read
                atoms = ase_read(traj_files[sample_idx], index=1)

            tb_params = tb_ensemble[sample_idx % n_tb]
            cell = np.array(atoms.get_cell(), dtype=float)
            positions = np.array(atoms.get_positions(), dtype=float)
            recip = _get_recip_cell(cell.T)
            kvec_cart = kvec_red @ recip

            evals, evecs, flat_idx = _diagonalize_k_mesh(
                atoms, tb_params, kvec_cart, tb_M, tb_W, args.tb_rcut,
                is_sk=is_sk, extra_hp=extra_hp,
            )
            E_F = fermi_energy_charge_neutral(evals)
            U = compute_flat_band_U(
                evecs, flat_idx, positions, cell, args.epsilon_r,
            )
            U_values.append(U)

            np.savez(
                npz_path,
                U=np.float64(U),
                evals=evals,
                fermi_level=E_F,
                flat_band_indices=np.array(flat_idx, dtype=np.int64),
                kvec=kvec_red,
                k_grid=np.array([nkx, nky, nkz], dtype=np.int64),
                epsilon_r=np.float64(args.epsilon_r),
                tb_params=tb_params,
                n_atoms=np.int64(len(atoms)),
                twist_angle=np.float64(args.twist_angle),
                unrelaxed=np.bool_(args.unrelaxed),
            )
            print(
                f"  sample {sample_idx:04d}: U = {U:.6f} eV  "
                f"(flat bands {flat_idx}, E_F[k=0]={E_F[0]:.4f} eV)  "
                f"→ {os.path.basename(npz_path)}",
                flush=True,
            )

        if U_values:
            arr = np.asarray(U_values, dtype=float)
            print(
                f"\n  U summary ({len(arr)} samples): "
                f"mean={arr.mean():.6f} eV  std={arr.std(ddof=min(1, len(arr)-1)):.6f} eV",
                flush=True,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
