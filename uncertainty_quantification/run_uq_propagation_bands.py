#!/usr/bin/env python3
"""
run_uq_propagation_bands.py
===========================
Compute tight-binding band structures for twisted bilayer graphene (TBLG)
structures previously relaxed by ``run_uq_propagation_relaxation.py``.

For each selected LAMMPS model (``--models``) and twist angle
(``--twist-angle``):

1. Locate the relaxed trajectory files written by
   ``run_uq_propagation_relaxation.py`` under ``--trajectory-dir``.
2. Load an MCMC ensemble of ACSF hopping model parameters
   (``--tb-model``, ``--tb-temperature``).
3. Pair each relaxed structure (sample *i*) with a **different** hopping
   ensemble draw (shuffled by ``--seed``).
4. Build the Bloch Hamiltonian at each k-point and diagonalize it with
   ``np.linalg.eigh``.
5. Save eigenvalues and k-path metadata to an ``.npz`` file per sample.

Supported model patterns for ``--models``:
    POD_energy*, TETB_POD*, Tersoff+DRIP, Tersoff+Kolmogorov_Crespi

Hopping ensemble default temperature
-------------------------------------
When ``--tb-temperature`` is omitted (default), the temperature that
minimises ``miscalibration_area`` in ``--calibration-metrics-dir`` is chosen
automatically for the TB model (same mechanism as ``--temperature`` for the
POD ensemble).  Falls back to the lowest available temperature on disk if no
calibration metrics exist for the TB model.

Examples
--------
::

    python run_uq_propagation_bands.py \\
        --models 'POD_energy_POD_index*' \\
        --temperature 0.8 \\
        --twist-angle 9.43

    python run_uq_propagation_bands.py \\
        --models 'POD_energy_POD_index*' \\
        --tb-model acsf_hoppings_M_15_W_6 \\
        --twist-angle 9.43 \\
        --n-kpts 80
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np   # always real NumPy (file I/O, plotting, non-GPU code)

# ── GPU / CPU array library ───────────────────────────────────────────────────
# xp = cupy when a GPU is present, numpy otherwise.
# _scatter_add / _to_cpu abstract the two genuine cupy/numpy differences.
try:
    import cupy as xp
    import cupyx
    _GPU: bool = bool(xp.cuda.is_available())
    if not _GPU:
        xp = np
        cupyx = None  # type: ignore[assignment]
except (ImportError, Exception):
    xp = np
    cupyx = None      # type: ignore[assignment]
    _GPU = False

if _GPU:
    def _scatter_add(target, idx, src):
        # cupyx.scatter_add only supports real dtypes (int/float).
        # For complex arrays, scatter real and imaginary parts separately via
        # float64 views — both .real and .imag are writeable CuPy views.
        if target.dtype.kind == 'c':
            cupyx.scatter_add(target.real, idx, src.real)
            cupyx.scatter_add(target.imag, idx, src.imag)
        else:
            cupyx.scatter_add(target, idx, src)
    def _to_cpu(arr): return arr.get().astype(np.float64)
    print("GPU available")
else:
    # numpy.add.at handles complex dtypes natively; no special casing needed.
    def _scatter_add(target, idx, src): xp.add.at(target, idx, src)
    def _to_cpu(arr): return np.asarray(arr, dtype=np.float64)
    print("GPU not available, using CPU instead")

HERE = os.path.dirname(os.path.abspath(__file__))

import blg_model_builder  # noqa: F401 — already installed
from blg_model_builder.tb_models import create_tb_model
from blg_model_builder.tb_descriptors import (
    get_acsf_hopping_descriptors,
    get_acsf_sk_hopping_descriptors,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args, collect_hyperparams
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    add_tb_model_arg,
    collect_workflow_hyperparams,
)
import types as _types

# ---------------------------------------------------------------------------
# Local imports — deferred to avoid triggering heavy dependency chains
# (matplotlib, ase, etc.) that may not be present in every conda env.
# These are imported lazily inside main() and the helper that uses them.
# ---------------------------------------------------------------------------

# Placeholder so type annotations resolve at module level.
DEFAULT_CALIBRATION_METRICS_DIR: str = "calibration_metrics"

# Inlined from run_uq_propagation_elasticity to avoid importing that module's
# heavy dependency chain (matplotlib, ase, etc.) into the band-structure environment.
DEFAULT_ENSEMBLE_SHUFFLE_SEED: int = 0


# ---------------------------------------------------------------------------
# Inlined TB utilities from blg_model_builder.tb_models
#
# tb_models.py unconditionally imports torch (and optionally cupy /
# matplotlib) at module level, none of which are required by the three
# functions we use.  Inlining avoids triggering those imports entirely.
# ---------------------------------------------------------------------------

def _get_recip_cell(cell: np.ndarray) -> np.ndarray:
    """Reciprocal cell from real-space cell (columns = lattice vectors)."""
    a1, a2, a3 = cell[:, 0], cell[:, 1], cell[:, 2]
    vol = float(np.dot(a1, np.cross(a2, a3)))
    b1 = 2 * np.pi * np.cross(a2, a3) / vol
    b2 = 2 * np.pi * np.cross(a3, a1) / vol
    b3 = 2 * np.pi * np.cross(a1, a2) / vol
    return np.array([b1, b2, b3])


def _k_path(sym_pts, nk: int):
    """Interpolate a k-path through *sym_pts* with *nk* points per segment."""
    k_list = np.array(sym_pts)
    n_nodes = k_list.shape[0]
    mesh_step = nk // (n_nodes - 1)
    step = np.arange(0, mesh_step, 1) / mesh_step
    kvec = np.zeros((0, 3))
    knode = np.zeros(n_nodes)
    for i in range(n_nodes - 1):
        n1, n2 = k_list[i], k_list[i + 1]
        segment = np.outer((n2 - n1), step).T + n1
        kvec = np.vstack((kvec, segment))
        knode[i + 1] = np.linalg.norm(n2 - n1) + knode[i]
    kvec = np.vstack((kvec, k_list[-1]))
    k_dist = np.zeros(len(kvec))
    for i in range(1, len(kvec)):
        k_dist[i] = np.linalg.norm(kvec[i] - kvec[i - 1]) + k_dist[i - 1]
    return kvec, k_dist, knode

def _shuffle_ensemble(ensemble: np.ndarray, seed: int) -> np.ndarray:
    """Return a full copy of the ensemble in random order."""
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 2:
        raise ValueError(f"Expected 2-D ensemble array, got shape {ensemble.shape}")
    n = ensemble.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    return ensemble[order]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TB_MODEL = "acsf_hoppings_M_10_W_6"
DEFAULT_TB_RCUT: float = 6.0
DEFAULT_TWIST_ANGLE: float = 9.43
DEFAULT_NK: int = 60
DEFAULT_TRAJECTORY_DIR = "trajectories/relaxation"
DEFAULT_OUTPUT_DIR = "bands/propagation"

# k-path nodes in reduced coordinates (hexagonal cell)
# a1 = [a, 0, 0]  a2 = [a/2, a√3/2, 0]
_K_NODE = [1 / 3, 2 / 3, 0.0]
_M_NODE = [1 / 2, 0.0, 0.0]
_GAMMA_NODE = [0.0, 0.0, 0.0]
_SYM_PTS = [_K_NODE, _GAMMA_NODE, _M_NODE, _K_NODE]
_SYM_LABELS = ["K", "\u0393", "M", "K"]


# ---------------------------------------------------------------------------
# TB model name helpers
# ---------------------------------------------------------------------------

_TB_NAME_RE = re.compile(
    r"(?i)acsf[_\-]hoppings?(?:[_\-]sk)?[_\-]m[_\-](\d+)[_\-]w[_\-](\d+)"
)


_TB_NAME_SK_RE = re.compile(r"(?i)acsf[_\-]hoppings[_\-]sk")


def _parse_tb_model_name(name: str) -> Tuple[int, int, str]:
    """Parse an ACSF hopping model name and return ``(M, W, canonical_name)``.

    Accepted formats (case-insensitive)::

        acsf_hoppings_M_10_W_6
        ACSF_hoppings_M_15_W_6
        acsf-hoppings-M-10-W-6
        ACSF_hoppings_sk_M_10_W_6

    Returns
    -------
    M : int
        Number of Chebyshev radial basis functions.
    W : int
        Number of angular (cos^w) exponents.
    canonical_name : str
        ``ACSF_hoppings_M_{M}_W_{W}`` or ``ACSF_hoppings_sk_M_{M}_W_{W}``
        — matches the ensemble directory name.

    Raises
    ------
    ValueError
        If *name* does not match the expected pattern.
    """
    m = _TB_NAME_RE.search(name)
    if m is None:
        raise ValueError(
            f"Cannot parse TB model name {name!r}.  "
            "Expected format: acsf_hoppings[_sk]_M_<M>_W_<W> "
            "(e.g. acsf_hoppings_M_10_W_6 or acsf_hoppings_sk_M_10_W_6)."
        )
    M = int(m.group(1))
    W = int(m.group(2))
    is_sk = bool(_TB_NAME_SK_RE.search(name))
    prefix = "ACSF_hoppings_sk" if is_sk else "ACSF_hoppings"
    canonical = f"{prefix}_M_{M}_W_{W}"
    return M, W, canonical


def _resolve_tb_ensemble(
    tb_model_name: str,
    ensemble_dir: str,
    temperature: Optional[float],
    *,
    calibration_metrics_dir: str,
    calibration_technique: str = "mcmc",
    calibration_target: str = "hopping",
) -> Tuple[str, float]:
    """Resolve the hopping ensemble pickle path and temperature used.

    Wraps :func:`resolve_ensemble_pickle` with the canonical TB model name
    and ``calibration_target="hopping"``.

    When *temperature* is ``None``, the temperature that minimises
    ``miscalibration_area`` is selected automatically (same logic as the POD
    ensemble temperature auto-select).
    """
    from blg_model_builder.ensemble_io import resolve_ensemble_pickle  # lazy import
    _, _, canonical = _parse_tb_model_name(tb_model_name)
    return resolve_ensemble_pickle(
        canonical,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_technique=calibration_technique,
        calibration_target=calibration_target,
    )


# ---------------------------------------------------------------------------
# Trajectory discovery
# ---------------------------------------------------------------------------

def _discover_traj_files(
    traj_dir: str,
    model_name: str,
    t_label: str,
    theta: float,
) -> List[str]:
    """Return sorted list of relaxed ``.traj`` files for *model_name* / *theta*.

    Scans::

        <traj_dir>/<safe_model>/T<t_label>/theta<theta>deg/*_sample*.traj

    Parameters
    ----------
    traj_dir : str
        Root directory written by ``run_uq_propagation_relaxation.py``
        (default ``"trajectories/relaxation"``).
    model_name : str
        LAMMPS model name (e.g. ``"POD_energy_POD_index_0_09fdb1c2b98eb30e"``).
    t_label : str
        Temperature label string used when relaxation was run (e.g. ``"0.8"``).
    theta : float
        Twist angle in degrees.

    Returns
    -------
    list of str
        Absolute paths of discovered ``.traj`` files, sorted lexicographically.
    """
    search_dir = os.path.join(
        traj_dir,
        _safe_filename_part(model_name),
        f"T{t_label}",
        f"theta{theta:g}deg",
    )
    pattern = os.path.join(search_dir, "*_sample*.traj")
    files = sorted(glob.glob(pattern))
    return files

def _build_bands(
    atoms,
    params: np.ndarray,
    kvec_cart: np.ndarray,
    M: int,
    W: int,
    r_cut: float,
    *,
    is_sk: bool = False,
    extra_hp: dict | None = None,
) -> Tuple[np.ndarray, float]:
    """Compute ACSF tight-binding band structure using ``np.linalg.eigh``.

    When *is_sk* is True, uses the Slater-Koster model with flat params
    ``[w_pi, w_sigma]`` of length ``2*n_feat``.

    The Hamiltonian is built from a **half neighbour-list** (each bond stored
    once, ``bothways=False``).  The Bloch matrix is therefore symmetrized as
    ``H_k = H_k + H_k†`` — no factor of ½ — before diagonalization.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed TBLG structure.
    params : np.ndarray, shape (n_features,)
        ACSF linear model weights for one ensemble sample.
    kvec_cart : np.ndarray, shape (n_kpts, 3)
        k-points in Cartesian coordinates (Å⁻¹, including 2π factor).
    M, W : int
        ACSF basis hyperparameters (must match the ensemble).
    r_cut : float
        Real-space hopping cutoff in Å.

    Returns
    -------
    evals : np.ndarray, shape (n_kpts, N)
        Eigenvalues in eV at each k-point, shifted so the Fermi level
        (midpoint between band ``nocc-1`` and ``nocc`` at the first k-point,
        where ``nocc = N // 2``) is at 0 eV.
    fermi_level : float
        Fermi energy (eV) that was subtracted.
    """

    extra_hp = dict(extra_hp or {})
    # Forward an optional ``use_envelope`` knob to the descriptor builders.
    _desc_kw = {}
    if "use_envelope" in extra_hp:
        _desc_kw["use_envelope"] = extra_hp["use_envelope"]

    # --- Step 1: descriptors and bond geometry ---
    if is_sk:
        # SK descriptors bake the (1-n²) / n² weighting into the columns;
        # shape (n_pairs, 2*n_feat).  pair_v is still returned for the
        # Hamiltonian construction below.
        descriptors, (pair_i, pair_j, pair_v) = get_acsf_sk_hopping_descriptors(
            atoms, M=M, W=W, r_cut=r_cut, **_desc_kw,
        )
    else:
        descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=r_cut, **_desc_kw,
        )
    # pair_v : (n_pairs, 3) — Cartesian bond vectors in Å (with lattice offsets)

    # --- Step 2: hopping amplitudes (class-based TB model) ---
    tb_name = f"ACSF_hoppings_sk_M_{M}_W_{W}" if is_sk else f"ACSF_hoppings_M_{M}_W_{W}"
    tb_model = create_tb_model(tb_name, {"M": M, "W": W, "r_cut": r_cut, **extra_hp})
    hoppings = tb_model(descriptors, params)

    N = len(atoms)
    nocc = N // 2
    n_kpts = len(kvec_cart)
    evals_list: list[np.ndarray] = []

    # --- Step 3: Bloch Hamiltonian at each k-point (unified xp path) ---
    # pair_v / hoppings already live on GPU when _GPU (from tb_descriptors/tb_models).
    # pair_i / pair_j are CPU int arrays; move them once.
    pair_v_xp   = xp.asarray(pair_v,   dtype=xp.float64)    # no-op if already xp
    hoppings_xp = xp.asarray(hoppings, dtype=xp.float64)    # no-op if already xp
    pair_i_xp   = xp.asarray(pair_i)
    pair_j_xp   = xp.asarray(pair_j)
    kvec_xp     = xp.asarray(kvec_cart, dtype=xp.float64)

    # Precompute linearized 1-D index for the N×N Hamiltonian scatter.
    # Using a 1-D index (instead of a 2-D tuple) avoids type-checking issues with
    # cupyx.scatter_add and is equally valid for numpy.add.at.
    lin_idx = pair_i_xp * N + pair_j_xp          # (n_pairs,) flat index into H.ravel()

    for k_cart in kvec_xp:
        phases   = xp.exp(1j * (pair_v_xp @ k_cart))        # (n_pairs,) complex128
        # Cast explicitly so scatter accepts complex even when hoppings is float64
        hop_vals = (hoppings_xp * phases).astype(xp.complex128)

        # Use np.complex128 (numpy dtype object) — accepted by both numpy and CuPy
        H      = xp.zeros(N * N, dtype=np.complex128)        # 1-D flat
        _scatter_add(H, lin_idx, hop_vals)
        H      = xp.asarray(H).reshape(N, N)                 # guard + 2-D view
        H      = H + H.conj().T                              # symmetrize

        w = xp.linalg.eigh(H)[0]                             # eigenvalues only
        evals_list.append(_to_cpu(w))

    evals = np.array(evals_list)  # (n_kpts, N)  — always real NumPy from here on

    # --- Fermi level: midpoint between the highest occupied and lowest
    #     unoccupied band at the first k-point (half-filling). ---
    
    fermi_level = float((evals[0, nocc] + evals[0, nocc - 1]) / 2.0)
    evals = evals - fermi_level

    # Keep only 50 bands centred on the Fermi level (25 below, 25 above).
    n_keep = 25
    band_lo = max(nocc - n_keep, 0)
    band_hi = min(nocc + n_keep, N)
    evals = evals[:, band_lo:band_hi]

    return evals, fermi_level


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_filename_part(s: str) -> str:
    return re.sub(r"[^\w.\-+]+", "_", str(s))


def _bands_output_path(
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
            "TBLG band-structure UQ propagation: pair pre-relaxed structures "
            "with MCMC ACSF hopping ensemble samples and diagonalize with "
            "pyELSI ELPA."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Model / ensemble selection (mirrors run_uq_propagation_relaxation.py) ---
    add_energy_models_arg(p)
    p.add_argument("--ensemble-dir", default="ensembles")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "POD ensemble temperature weight T (nearest match) used when the "
            "relaxation trajectories were produced.  Default: auto-select "
            "(minimises miscalibration_area)."
        ),
    )
    p.add_argument(
        "--calibration-metrics-dir",
        default=DEFAULT_CALIBRATION_METRICS_DIR,
        help="Directory with calibration_*.npz from plot_bayes_factor.py --calculate.",
    )
    p.add_argument(
        "--calibration-target",
        default="energy",
        help="Calibration target key for the POD ensemble (default: energy).",
    )
    p.add_argument(
        "--calibration-technique",
        default="mcmc",
        help="Calibration technique key (default: mcmc).",
    )

    # --- TB model ---
    add_tb_model_arg(p, default=DEFAULT_TB_MODEL)
    p.add_argument(
        "--tb-temperature",
        type=float,
        default=None,
        help=(
            "Hopping ensemble temperature weight T (nearest match).  "
            "Default: auto-select (minimises miscalibration_area for the TB "
            "model); falls back to lowest T on disk if no calibration metrics "
            "are available."
        ),
    )
    p.add_argument(
        "--tb-calibration-target",
        default="hopping",
        help="Calibration target key for the hopping ensemble (default: hopping).",
    )
    p.add_argument(
        "--tb-rcut",
        type=float,
        default=DEFAULT_TB_RCUT,
        help=f"ACSF real-space hopping cutoff in Å (default: {DEFAULT_TB_RCUT}).",
    )

    # --- Geometry ---
    p.add_argument(
        "--twist-angle",
        type=float,
        default=DEFAULT_TWIST_ANGLE,
        help=f"Twist angle in degrees (default: {DEFAULT_TWIST_ANGLE}).",
    )

    # --- I/O directories ---
    p.add_argument(
        "--trajectory-dir",
        default=DEFAULT_TRAJECTORY_DIR,
        help=(
            f"Root directory of relaxed trajectories (default: {DEFAULT_TRAJECTORY_DIR!r}).  "
            "Must match --output-dir used in run_uq_propagation_relaxation.py."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for band-structure .npz output (default: {DEFAULT_OUTPUT_DIR!r}).",
    )

    # --- Band-structure settings ---
    p.add_argument(
        "--n-kpts",
        type=int,
        default=DEFAULT_NK,
        help=f"k-points per high-symmetry segment (default: {DEFAULT_NK}).",
    )

    # --- Reproducibility ---
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_ENSEMBLE_SHUFFLE_SEED,
        help=(
            "RNG seed for hopping ensemble shuffle "
            f"(default: {DEFAULT_ENSEMBLE_SHUFFLE_SEED})."
        ),
    )

    add_hyperparam_args(p)
    args, _unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, _unknown)
    if cli_hyperparams:
        print(f"TB CLI hyperparameters: {cli_hyperparams}", flush=True)

    # Change to the script directory so relative paths (ensembles/, etc.) work.
    os.chdir(HERE)

    # -----------------------------------------------------------------------
    # Lazy imports of local modules (may pull in matplotlib / ase which are
    # not present in every env — defer until after arg parsing).
    # -----------------------------------------------------------------------
    from blg_model_builder.ensemble_io import (
        DEFAULT_CALIBRATION_METRICS_DIR as _PBF_CALIB_DIR,
        expand_model_patterns,
        load_ensemble_pickle,
        resolve_ensemble_pickle,
    )
    # Override placeholder with the real value now that plot_bayes_factor loaded.
    if args.calibration_metrics_dir == DEFAULT_CALIBRATION_METRICS_DIR:
        args.calibration_metrics_dir = _PBF_CALIB_DIR

    # -----------------------------------------------------------------------
    # Parse TB model
    # -----------------------------------------------------------------------
    try:
        tb_M, tb_W, tb_canonical = _parse_tb_model_name(args.tb_model)
    except ValueError as exc:
        p.error(str(exc))

    print(f"\nTB model: {tb_canonical}  (M={tb_M}, W={tb_W}, r_cut={args.tb_rcut} Å)",
          flush=True)

    # -----------------------------------------------------------------------
    # Resolve TB ensemble
    # -----------------------------------------------------------------------
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
    print(f"TB ensemble pickle: {tb_pkl_path}  (T={tb_t_label})", flush=True)

    tb_ens_dict = load_ensemble_pickle(tb_pkl_path)
    tb_ensemble_raw = np.asarray(tb_ens_dict["ensemble"]["hopping"], dtype=float)
    tb_ensemble = _shuffle_ensemble(tb_ensemble_raw, args.seed)
    print(
        f"TB ensemble (seed={args.seed}): "
        f"{tb_ensemble_raw.shape[0]} members × {tb_ensemble_raw.shape[1]} features",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Expand model patterns
    # -----------------------------------------------------------------------
    models = expand_model_patterns(args.models, args.ensemble_dir)
    if not models:
        p.error("No models matched --models patterns.")
    print(f"\nModels: {models}", flush=True)

    # -----------------------------------------------------------------------
    # k-path (same for all models / samples — cell is nearly identical)
    # -----------------------------------------------------------------------
    kvec_red, k_dist, k_node = _k_path(_SYM_PTS, args.n_kpts)
    # Cartesian conversion is done per-sample using the relaxed cell.
    print(
        f"\nk-path: K → Γ → M → K  "
        f"({args.n_kpts} pts/segment → {len(kvec_red)} total k-points)",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Main loop: models → traj files → band structures
    # -----------------------------------------------------------------------
    for model_name in models:
        print(f"\n{'=' * 60}", flush=True)
        print(f" Model: {model_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # --- Resolve POD ensemble temperature (to locate trajectory directory) ---
        try:
            _pkl_path, t_used = resolve_ensemble_pickle(
                model_name,
                args.ensemble_dir,
                args.temperature,
                calibration_metrics_dir=args.calibration_metrics_dir,
                calibration_technique=args.calibration_technique,
                calibration_target=args.calibration_target,
            )
        except FileNotFoundError as exc:
            print(
                f"  Warning: cannot resolve ensemble for {model_name!r}: {exc}\n"
                f"  Skipping.",
                file=sys.stderr,
            )
            continue

        t_label = f"{t_used:g}"
        print(f"  POD ensemble T = {t_label} (trajectory directory key)", flush=True)

        # --- Discover relaxed trajectory files ---
        traj_files = _discover_traj_files(
            args.trajectory_dir,
            model_name,
            t_label,
            args.twist_angle,
        )
        if not traj_files:
            print(
                f"  Warning: no trajectory files found for {model_name!r} "
                f"T={t_label} θ={args.twist_angle:g}°.\n"
                f"  Expected location: "
                f"{os.path.join(args.trajectory_dir, _safe_filename_part(model_name), f'T{t_label}', f'theta{args.twist_angle:g}deg', '*_sample*.traj')}",
                file=sys.stderr,
            )
            continue

        n_trajs = len(traj_files)
        n_tb = tb_ensemble.shape[0]
        if n_tb < n_trajs:
            print(
                f"  Warning: hopping ensemble has {n_tb} samples but {n_trajs} "
                f"traj files found.  Hopping samples will be cycled.",
                file=sys.stderr,
            )
        print(
            f"  Found {n_trajs} trajectory file(s); "
            f"{n_tb} TB ensemble samples available.",
            flush=True,
        )

        # --- Prepare output directory ---
        out_dir = os.path.join(
            args.output_dir,
            _safe_filename_part(model_name),
            f"T{t_label}",
            f"{_safe_filename_part(tb_canonical)}_tbT{tb_t_label}",
            f"theta{args.twist_angle:g}deg",
        )
        os.makedirs(out_dir, exist_ok=True)

        # --- Process each trajectory ---
        n_done = 0
        n_skipped = 0

        for sample_idx, traj_path in enumerate(traj_files):
            npz_path = _bands_output_path(
                out_dir,
                model_name,
                t_label,
                tb_canonical,
                tb_t_label,
                args.twist_angle,
                sample_idx,
            )

            if os.path.isfile(npz_path):
                print(
                    f"  [skip] sample {sample_idx:04d} — {os.path.basename(npz_path)} exists",
                    flush=True,
                )
                n_skipped += 1
                continue

            print(
                f"  sample {sample_idx:04d}/{n_trajs - 1}  traj={os.path.basename(traj_path)} …",
                end="  ",
                flush=True,
            )

            # --- Read relaxed structure (frame index 1) ---
            try:
                from ase.io import read as ase_read
                relaxed = ase_read(traj_path, index=1)
            except Exception as exc:
                print(
                    f"\n  Warning: failed to read {traj_path}: {exc}  Skipping.",
                    file=sys.stderr,
                )
                continue

            # --- Select hopping ensemble sample (cycle if necessary) ---
            tb_params = tb_ensemble[sample_idx % n_tb]

            # --- Cartesian k-path using the relaxed cell ---
            cell = np.array(relaxed.get_cell(), dtype=float)
            recip = _get_recip_cell(cell.T)        # (3, 3) Å⁻¹ with 2π
            kvec_cart = kvec_red @ recip           # (n_kpts, 3)

            # --- Compute band structure ---
            try:
                evals, fermi_level = _build_bands(
                    relaxed,
                    tb_params,
                    kvec_cart,
                    tb_M,
                    tb_W,
                    args.tb_rcut,
                    is_sk=tb_canonical.startswith("ACSF_hoppings_sk"),
                    extra_hp=cli_hyperparams or None,
                )
            except Exception as exc:
                print(
                    f"\n  Warning: band structure failed for sample {sample_idx:04d}: "
                    f"{type(exc).__name__}: {exc}  Skipping.",
                    file=sys.stderr,
                )
                continue

            # --- Save ---
            np.savez(
                npz_path,
                evals=evals,                    # (n_kpts, N_atoms) eV, E_F = 0
                kvec=kvec_red,                  # (n_kpts, 3) reduced coordinates
                k_dist=k_dist,                  # (n_kpts,)
                k_node=k_node,                  # (n_nodes,)
                sym_pts=np.array(_SYM_PTS),     # (4, 3)
                sym_labels=np.array(_SYM_LABELS),
                fermi_level=fermi_level,        # scalar eV (before subtraction)
                tb_params=tb_params,            # (n_features,) hopping weights used
                n_atoms=np.int64(len(relaxed)),
                twist_angle=np.float64(args.twist_angle),
            )

            n_done += 1
            n_bands = evals.shape[1]
            print(
                f"done  evals=({len(kvec_red)}, {n_bands})  "
                f"E_F={fermi_level:.4f} eV  → {os.path.basename(npz_path)}",
                flush=True,
            )

        print(
            f"\n  {model_name}: {n_done} computed, {n_skipped} skipped "
            f"(already existed), {n_trajs - n_done - n_skipped} failed.",
            flush=True,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
