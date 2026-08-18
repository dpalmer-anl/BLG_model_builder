#!/usr/bin/env python3
"""pod_hyperparameter_search.py — systematic POD descriptor hyperparameter search.

Grid search over POD descriptor hyperparameters for either:

* ``POD_energy`` — standalone :class:`~blg_model_builder.lammps_interface.PODLammpsCalculator`
  (LAMMPS ``fitpod`` on DFT totals), or
* ``TETB_POD`` — residual POD on top of TB + Ewald via
  :func:`model_fit.fit_tetb_residual_pod` /
  :class:`~blg_model_builder.lammps_interface.TETB_PODLammpsCalculator`
  (same POD grid and fit weights as the standalone path).

Toggle :data:`SEARCH_MODEL` at the top of this file.  For each candidate:

  1. Fit POD via LAMMPS ``fitpod`` on a fixed training split.
  2. Evaluate energy MAE / RMSE on a held-out test split (per atom).
  3. Relax TBLG at every angle in ``TBLG_THETAS`` with ASE FIRE.
  4. Record AB-style (min gap) and AA-style (max span) interlayer separations.

Configurations where ``ncoeff > n_train_configs`` are skipped (under-determined).

Results are written after **every** trial to
``pod_hyperparam_search.json`` and ``pod_hyperparam_search.csv`` (see
``_results_json_path`` / ``_results_csv_path``).  Fit coefficients go
under ``<CACHE_DIR>/<SEARCH_MODEL>/``.  The legacy broad-search JSON files are
not read or updated by this script.

The *best* model is the one with the lowest test-set energy RMSE among those
where **all** TBLG relaxations converged and interlayer separations satisfy
:data:`SEP_MIN`–:data:`SEP_MAX` (Å): AB and AA gaps in band, AB < AA, and
min/max layer separations ``2·min|zᵢ−⟨z⟩|``, ``2·max|zᵢ−⟨z⟩|`` in band.

Usage (from repo root or from this directory)::

    cd uncertainty_quantification/
    python pod_hyperparameter_search.py
    python pod_hyperparameter_search.py --fresh   # restart: refit all configs (clears cache)

Outputs (under ``uncertainty_quantification/pod_hyperparam_search/``):

* Results — ``pod_hyperparam_search.json`` and ``pod_hyperparam_search.csv``
* Fit cache — ``pod_hyperparam_search_cache/<SEARCH_MODEL>/<hash>.npz``

Edit :data:`SEARCH_MODEL` and the constants in the ``USER-EDITABLE CONFIGURATION``
section below.  ``INCLUDE_INTRALAYER`` applies only when ``SEARCH_MODEL == "POD_energy"``.

Interlayer separation conventions:

  AB-style  = min(z_top) − max(z_bottom)   (closest interlayer pair)
  AA-style  = max(z_top) − min(z_bottom)   (largest interlayer span)
  min layer sep = ``2·min|zᵢ − ⟨z⟩|``     (same as ``run_uq_propagation_relaxation``)
  max layer sep = ``2·max|zᵢ − ⟨z⟩|``

TBLG acceptance (all angles): converged; each separation in
``[SEP_MIN, SEP_MAX]``; ``AB < AA``.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — works whether run from repo root or from this directory
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_UQ_DIR = _THIS_DIR.parent
_REPO_ROOT = _UQ_DIR.parent
for _p in (str(_REPO_ROOT / "src"), str(_UQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ===========================================================================
# USER-EDITABLE CONFIGURATION
# ===========================================================================

# Which model to search: ``"POD_energy"`` (standalone POD) or ``"TETB_POD"``
# (residual POD coefficients for ``TETB_PODLammpsCalculator``).
SEARCH_MODEL: str = "POD_energy"

# --- TETB_POD only: fixed tight-binding / ACSF hopping knobs for the subtracted
# TB + Ewald baseline (same role as ``tb_M`` / ``tb_W`` / ``r_cut`` in
# ``get_MCMC_inputs`` / ``build_tetb_pod_hyperparams_from_data_kw``).  The POD
# grid below still controls the **residual** descriptor architecture.
TETB_TB_M: int = 10
TETB_TB_W: int = 6
TETB_TB_RCUT: float = 6.0
TETB_RIDGE_ACSF: float = 0.1

# Hyperparameter grid.  All combinations satisfying the ordering constraints
#   two_body ≥ three_body ≥ four_body ≥ five_body ≥ six_body ≥ seven_body
# (for radial counts) and angular ≤ radial within each body order are explored.
# ``bessel_polynomial_degree`` / ``inverse_polynomial_degree`` must appear here;
# they are passed through to the POD descriptor (see ``_build_hyperparams``) and
# hashed into the trial cache filename.
# Regularization is independent of architecture and is tried for every config.
#
# **Stricter phase-2 grid** (from ``pod_hyperparam_search_results_new.csv``):
# Among 134 / 2889 trials with ``meets_all_criteria``, lowest test RMSE and
# median RMSE cluster at:
#   bessel_polynomial_degree = 2
#   inverse_polynomial_degree = 8
#   four_body_radial = 2, four_body_angular = 2
# (``four_body_radial=0``, ``four_body_angular=0`` gives marginally lower test
# RMSE but drops four-body terms entirely; ``4``/``2`` matches POD_index_0.)
# Vary two- and three-body counts only; see ``_iter_valid_configs`` ordering rules.
#
# Estimated runtime: each trial takes ~5-15 s (fit) + up to ~30-60 s per TBLG angle
# (fewer if the first angle fails and the rest are skipped).
# Reduce the grid or set MAX_TRIALS to limit total wall time.
GRID: dict[str, list] = {
    "rcut":               [7.0],
    "two_body_radial":    list(range(6, 14, 1)),  # 6, 7, …, 14
    "three_body_radial":  [4, 6, 8,10],
    "three_body_angular": [4, 6,8],
    "four_body_radial":   [2], 
    "four_body_angular":  [2], 
    "five_body_radial":   [0], 
    "five_body_angular":  [0], 
    "six_body_radial":    [0],
    "six_body_angular":   [0],
    "seven_body_radial":  [0],
    "seven_body_angular": [0],
    "regularization":     [1e-12],
    "weight_energy":      [100.0],
    "weight_force":       [1.0],
    "bessel_polynomial_degree": [8],
    "inverse_polynomial_degree": [12],
}

# Set to an integer to cap total trials (configs are shuffled first so the
# budget covers a diverse sample).  None = run all valid configs.
MAX_TRIALS: Optional[int] = None

# Include intralayer monolayer graphene data in the fit (``POD_energy`` only;
# ``TETB_POD`` uses the default ``DataLoader`` interlayer + hopping bundle).
INCLUDE_INTRALAYER: bool = True

# DFT reference for bilayer energy training: ``"rVV10"`` (default),
# ``"MBD"`` (``data/bilayer_graphene_MBD.xyz``), or ``"QMC"``.
# Override at runtime with ``--level-of-theory``.
LEVEL_OF_THEORY: str = "rVV10"

# Fixed random seed and test-set fraction for the train / test split.
DATA_SPLIT_SEED: int = 42
TEST_FRACTION: float = 0.20

# TBLG twist angles (degrees) to test relaxation stability.
TBLG_THETAS: list[float] = [21.78, 9.43]

# TBLG relaxation parameters. ASE FIRE uses RELAX_FTOL as ``fmax`` and
# RELAX_MAX_STEPS as its maximum number of optimizer steps.
RELAX_ETOL: float = 1e-9     # retained for the common calculator API
RELAX_FTOL: float = 1e-3     # eV/Å
RELAX_MAX_STEPS: int = 1_000
RELAX_MAX_EVAL: int = 800_000

# Accepted interlayer separation range (Å) for relaxed TBLG structures.
SEP_MIN: float = 3.0
SEP_MAX: float = 4.0

# LAMMPS executable path (override here if not on PATH).
LAMMPS_EXEC: str = "/mnt/c/Users/Daniel/Documents/research/lammps/build/lmp"

# Cache directory root.  Fits are stored under ``<CACHE_DIR>/<SEARCH_MODEL>/``
# so standalone POD and TETB+POD residual searches never clobber each other.
CACHE_DIR: str = "pod_hyperparam_search_cache"

# Primary search outputs (JSON + CSV); consumed by ``pod_model_selection``.
RESULTS_STEM = "pod_hyperparam_search"


def _level_of_theory_tag() -> str:
    """Suffix for cache / results paths so different LOTs do not collide."""
    lot = str(LEVEL_OF_THEORY).strip()
    if lot.lower() == "rvv10":
        return ""
    return f"_{lot}"


def _cache_subdir() -> str:
    return f"{SEARCH_MODEL}{_level_of_theory_tag()}"


def _results_json_path() -> Path:
    return _THIS_DIR / f"{RESULTS_STEM}{_level_of_theory_tag()}.json"


def _results_csv_path() -> Path:
    return _THIS_DIR / f"{RESULTS_STEM}{_level_of_theory_tag()}.csv"


def _legacy_broad_results_json_path() -> Path:
    """Legacy broad-search JSON (not used for resume)."""
    return _THIS_DIR / f"pod_hyperparam_search_results_{SEARCH_MODEL}.json"


def _load_results_for_resume() -> list[dict]:
    """Load completed trials for resume."""
    path = _results_json_path()
    if not path.is_file():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    return list(data) if isinstance(data, list) else []


def _results_json_read_candidates() -> list[Path]:
    """Try legacy namespaced JSON first, then untagged legacy file (POD_energy only)."""
    paths = [_legacy_broad_results_json_path()]
    if SEARCH_MODEL == "POD_energy":
        legacy = _THIS_DIR / "pod_hyperparam_search_results.json"
        if legacy.resolve() != paths[0].resolve():
            paths.append(legacy)
    return paths


def _load_existing_results_for_resume() -> tuple[list[dict], Path, Optional[Path]]:
    """Return ``(results, canonical_write_path, path_actually_read)`` — legacy broad search."""
    canonical = _legacy_broad_results_json_path()
    for p in _results_json_read_candidates():
        if not p.is_file():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            continue
        src = p if p.resolve() != canonical.resolve() else None
        return list(data), canonical, src
    return [], canonical, None

# ===========================================================================
# Imports that require the path bootstrap above
# ===========================================================================
# NOTE: lammps_interface and potentials have a circular dependency at module
# level (potentials imports LammpsCalculatorBase from lammps_interface).
# Importing lammps_interface here at top level would trigger that cycle.
# We therefore defer PODLammpsCalculator to a lazy getter used inside functions.

from blg_model_builder.DataLoader import load_data_for_model  # noqa: E402
from blg_model_builder.potentials import ncoeff_from_params, pod_hyperparams_to_str  # noqa: E402
from blg_model_builder.model_fit import fit_pod, fit_tetb_residual_pod  # noqa: E402


def _get_pod_calculator_class():
    """Lazily import PODLammpsCalculator to avoid the circular import cycle."""
    from blg_model_builder.lammps_interface import PODLammpsCalculator  # noqa: PLC0415
    return PODLammpsCalculator


def _get_tetb_pod_calculator_class():
    from blg_model_builder.lammps_interface import TETB_PODLammpsCalculator  # noqa: PLC0415
    return TETB_PODLammpsCalculator


def _tb_hyperparams_dict() -> dict:
    """Fixed TB / ACSF descriptor dict for ``TETB_PODLammpsCalculator``."""
    return {"M": int(TETB_TB_M), "W": int(TETB_TB_W), "r_cut": float(TETB_TB_RCUT)}


def _make_energy_calculator(
    hyperparams: dict,
    pod_params: np.ndarray,
    rcut: float,
    *,
    tb_params: Optional[np.ndarray] = None,
) -> Any:
    """Construct the energy calculator used for test eval and TBLG relaxations."""
    if SEARCH_MODEL == "POD_energy":
        return _get_pod_calculator_class()(
            hyperparams, pod_params, elements=["C"], cutoff=float(rcut),
        )
    if SEARCH_MODEL == "TETB_POD":
        cls = _get_tetb_pod_calculator_class()
        hp = dict(hyperparams)
        if "species" not in hp:
            hp["species"] = ["C"]
        if tb_params is None:
            raise ValueError("TETB_POD requires tb_params for calculator construction")
        return cls(
            tb_params=np.asarray(tb_params, dtype=np.float64).ravel(),
            pod_params=np.asarray(pod_params, dtype=np.float64).ravel(),
            tb_hyperparams=_tb_hyperparams_dict(),
            pod_hyperparams=hp,
            pod_cutoff=float(rcut),
            elements=["C"],
            shift=0.0,
        )
    raise ValueError(f"Unknown SEARCH_MODEL {SEARCH_MODEL!r}")

# ===========================================================================
# Helpers
# ===========================================================================

_POD_INT_KEYS = (
    "two_body_radial",
    "three_body_radial", "three_body_angular",
    "four_body_radial",  "four_body_angular",
    "five_body_radial",  "five_body_angular",
    "six_body_radial",   "six_body_angular",
    "seven_body_radial", "seven_body_angular",
    "bessel_polynomial_degree",
    "inverse_polynomial_degree",
)


def _build_hyperparams(cfg: dict) -> dict:
    """Convert a flat ``cfg`` dict to the POD hyperparams dict.

    ``bessel_polynomial_degree`` and ``inverse_polynomial_degree`` must be
    present (every trial config comes from ``GRID``).
    """
    bessel = int(cfg["bessel_polynomial_degree"])
    inverse = int(cfg["inverse_polynomial_degree"])
    return {
        "species": ["C"],
        "bessel_polynomial_degree": bessel,
        "inverse_polynomial_degree": inverse,
        "twobody_number_radial_basis_functions":   cfg["two_body_radial"],
        "threebody_number_radial_basis_functions":  cfg["three_body_radial"],
        "threebody_angular_degree":                 cfg["three_body_angular"],
        "fourbody_number_radial_basis_functions":   cfg["four_body_radial"],
        "fourbody_angular_degree":                  cfg["four_body_angular"],
        "fivebody_number_radial_basis_functions":   cfg["five_body_radial"],
        "fivebody_angular_degree":                  cfg["five_body_angular"],
        "sixbody_number_radial_basis_functions":    cfg["six_body_radial"],
        "sixbody_angular_degree":                   cfg["six_body_angular"],
        "sevenbody_number_radial_basis_functions":  cfg["seven_body_radial"],
        "sevenbody_angular_degree":                 cfg["seven_body_angular"],
    }


def _config_hash(cfg: dict) -> str:
    """Deterministic SHA-256 hash of a config dict (architecture + fit params)."""
    payload: dict[str, Any] = {"SEARCH_MODEL": SEARCH_MODEL, "config": cfg}
    if SEARCH_MODEL == "TETB_POD":
        payload["TETB_TB"] = [TETB_TB_M, TETB_TB_W, TETB_TB_RCUT]
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _is_valid_config(cfg: dict) -> bool:
    """Return True if the body-order ordering constraints are satisfied."""
    r2 = cfg["two_body_radial"]
    r3, a3 = cfg["three_body_radial"], cfg["three_body_angular"]
    r4, a4 = cfg["four_body_radial"],  cfg["four_body_angular"]
    r5, a5 = cfg["five_body_radial"],  cfg["five_body_angular"]
    r6, a6 = cfg["six_body_radial"],   cfg["six_body_angular"]
    r7, a7 = cfg["seven_body_radial"], cfg["seven_body_angular"]
    # Radial chain: must be non-increasing
    if r3 > r2 or r4 > r3 or r5 > r4 or r6 > r5 or r7 > r6:
        return False
    # Angular ≤ radial within each body order, and must be 0 when radial is 0
    for r, a in ((r3, a3), (r4, a4), (r5, a5), (r6, a6), (r7, a7)):
        if r == 0 and a != 0:
            return False
        if r > 0 and a == 0:
            return False   # angular must be enabled if radial is
        if a > r:
            return False
    bessel = int(cfg["bessel_polynomial_degree"])
    inverse = int(cfg["inverse_polynomial_degree"])
    if bessel < 0 or inverse < 0:
        return False
    return True


def _iter_valid_configs() -> list[dict]:
    """Generate all valid hyperparameter configs from ``GRID``."""
    keys = list(GRID.keys())
    out: list[dict] = []
    for values in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, values))
        if _is_valid_config(cfg):
            out.append(cfg)
    return out


def _params_finite(
    pod_params: np.ndarray,
    tb_params: Optional[np.ndarray] = None,
) -> bool:
    pod_ok = np.all(np.isfinite(np.asarray(pod_params, dtype=float)))
    if not pod_ok:
        return False
    if tb_params is None:
        return True
    return bool(np.all(np.isfinite(np.asarray(tb_params, dtype=float))))


def _test_metrics_finite(err: dict[str, float]) -> bool:
    for key in ("energy_mae_per_atom_eV", "energy_rmse_per_atom_eV"):
        val = err.get(key)
        if val is None or not np.isfinite(float(val)):
            return False
    return True


def _attempt_fit(
    cfg: dict,
    hyperparams: dict,
    hyperparams_str: str,
    train_atoms: list,
    rcut: float,
    test_atoms: list,
    test_E: np.ndarray,
    tetb_aux: Optional[dict[str, Any]],
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict[str, float]],
    float,
    Optional[str],
]:
    """
    Fit once with the GRID regularization value in ``cfg``.

    Returns ``(pod_params, tb_params, test_err, fit_time_s, error_msg)``.
    On failure ``error_msg`` is set and the other fields are ``None`` / ``0``.
    """
    t0 = time.time()
    reg = float(cfg["regularization"])
    try:
        if SEARCH_MODEL == "POD_energy":
            pod_params = fit_pod(
                hyperparams_str,
                train_atoms,
                lammps_exec=LAMMPS_EXEC,
                regularization=reg,
                weight_energy=cfg["weight_energy"],
                weight_force=cfg["weight_force"],
            )
            tb_params = None
        else:
            xh = (tetb_aux or {}).get("x_hopping")
            yh = (tetb_aux or {}).get("y_hopping")
            e_train = np.array(
                [float(a.get_potential_energy()) for a in train_atoms],
                dtype=float,
            )
            f_train = [np.asarray(a.get_forces(), dtype=float) for a in train_atoms]
            tb_params, pod_params, _ = fit_tetb_residual_pod(
                train_atoms,
                e_train,
                f_train,
                M=int(TETB_TB_M),
                W=int(TETB_TB_W),
                r_cut=float(TETB_TB_RCUT),
                pod_hyperparams=hyperparams,
                pod_cutoff=float(rcut),
                xdata_hopping=xh,
                ydata_hopping=yh,
                lammps_exec=LAMMPS_EXEC,
                ridge_acsf=float(TETB_RIDGE_ACSF),
                regularization=reg,
                weight_energy=float(cfg["weight_energy"]),
                weight_force=float(cfg["weight_force"]),
            )
            pod_params = np.asarray(pod_params, dtype=float).ravel()
            tb_params = np.asarray(tb_params, dtype=float).ravel()
    except Exception as exc:
        fit_time = float(time.time() - t0)
        err_msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"  [fit] reg={reg:.0e} exception: {err_msg}", flush=True)
        return None, None, None, fit_time, err_msg

    if not _params_finite(pod_params, tb_params):
        fit_time = float(time.time() - t0)
        err_msg = "non-finite POD/TB coefficients"
        print(f"  [fit] reg={reg:.0e} {err_msg}", flush=True)
        return None, None, None, fit_time, err_msg

    try:
        err = _evaluate_test_error(
            hyperparams,
            pod_params,
            test_atoms,
            test_E,
            rcut,
            tb_params=tb_params,
        )
    except Exception as exc:
        fit_time = float(time.time() - t0)
        err_msg = f"test eval {type(exc).__name__}: {str(exc)[:200]}"
        print(f"  [fit] reg={reg:.0e} {err_msg}", flush=True)
        return None, None, None, fit_time, err_msg

    if not _test_metrics_finite(err):
        fit_time = float(time.time() - t0)
        err_msg = "non-finite test MAE/RMSE"
        print(f"  [fit] reg={reg:.0e} {err_msg}", flush=True)
        return None, None, None, fit_time, err_msg

    fit_time = float(time.time() - t0)
    return pod_params, tb_params, err, fit_time, None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_and_split() -> tuple[list, list, np.ndarray, int, Optional[dict[str, Any]]]:
    """Load training data and return a reproducible train / test split.

    For ``SEARCH_MODEL == "POD_energy"`` the last return value is ``None``.

    For ``SEARCH_MODEL == "TETB_POD"`` the last value is a dict with
    ``"x_hopping"`` / ``"y_hopping"`` lists (full dataset) passed to
    :func:`fit_tetb_residual_pod` for ACSF weight resolution when no cache exists.
    Energy train/test indices use the same RNG as the POD_energy path.

    Returns
    -------
    train_atoms_sp : list[ase.Atoms]
        Training atoms with SinglePointCalculator (ready for ``fit_pod`` / residual fit).
    test_atoms : list[ase.Atoms]
        Test atoms with SinglePointCalculator.
    test_energies : np.ndarray
        Reference total energies for the test set (eV).
    n_train : int
        Number of training configurations.
    tetb_aux : dict or None
        Only for ``TETB_POD``: hopping lists for TB pre-fit.
    """
    from ase.calculators.singlepoint import SinglePointCalculator as SPC

    model_key = "POD_energy" if SEARCH_MODEL == "POD_energy" else "TETB_POD"
    if SEARCH_MODEL not in ("POD_energy", "TETB_POD"):
        raise ValueError(
            f"SEARCH_MODEL must be 'POD_energy' or 'TETB_POD', got {SEARCH_MODEL!r}",
        )

    print(
        f"[search] Loading training data ({model_key}, level_of_theory={LEVEL_OF_THEORY})…",
        flush=True,
    )
    if SEARCH_MODEL == "POD_energy":
        _, _, xdata, _, _, ydata = load_data_for_model(
            "POD_energy",
            supercells=1,
            level_of_theory=LEVEL_OF_THEORY,
            include_intralayer=INCLUDE_INTRALAYER,
        )
        tetb_aux: Optional[dict[str, Any]] = None
    else:
        _, _, xdata, _, _, ydata = load_data_for_model(
            "TETB_POD", supercells=1, level_of_theory=LEVEL_OF_THEORY,
        )
        tetb_aux = {
            "x_hopping": xdata.get("hopping"),
            "y_hopping": ydata.get("hopping"),
        }

    atoms_all = list(xdata["energy"])
    E_all      = np.asarray(ydata["energy"], dtype=float)
    F_all      = list(ydata["forces"])
    n_total    = len(atoms_all)

    rng = np.random.default_rng(DATA_SPLIT_SEED)
    perm = rng.permutation(n_total)
    n_test  = max(1, int(round(n_total * TEST_FRACTION)))
    n_train = n_total - n_test
    idx_test  = sorted(perm[:n_test])
    idx_train = sorted(perm[n_test:])

    def _wrap(indices):
        out = []
        for i in indices:
            a = atoms_all[i].copy()
            a.calc = SPC(a, energy=float(E_all[i]),
                         forces=np.asarray(F_all[i], dtype=float))
            out.append(a)
        return out

    train_atoms = _wrap(idx_train)
    test_atoms  = _wrap(idx_test)
    test_E = E_all[idx_test]

    print(
        f"[search] n_total={n_total}  n_train={n_train}  n_test={n_test}  model={SEARCH_MODEL}",
        flush=True,
    )
    return train_atoms, test_atoms, test_E, n_train, tetb_aux


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def _evaluate_test_error(
    hyperparams: dict,
    pod_params: np.ndarray,
    test_atoms: list,
    test_E: np.ndarray,
    rcut: float,
    *,
    tb_params: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute per-atom energy MAE and RMSE on the test set."""
    calc = _make_energy_calculator(
        hyperparams, pod_params, rcut, tb_params=tb_params,
    )
    try:
        e_pred_list, _ = calc.evaluate_batch(test_atoms)
    finally:
        calc.close()

    e_pred = np.asarray(e_pred_list, dtype=float)
    n_atoms = np.array([len(a) for a in test_atoms], dtype=float)
    delta_pa = e_pred / n_atoms - test_E / n_atoms
    mae  = float(np.mean(np.abs(delta_pa)))
    rmse = float(np.sqrt(np.mean(delta_pa ** 2)))
    return {"energy_mae_per_atom_eV": mae, "energy_rmse_per_atom_eV": rmse}


# ---------------------------------------------------------------------------
# TBLG structure building and analysis
# ---------------------------------------------------------------------------

def _ensure_mol_id_from_z(atoms) -> None:
    """Assign ``mol-id`` ∈ {1, 2} by z-coordinate if not already set."""
    if atoms.has("mol-id"):
        return
    z = atoms.positions[:, 2]
    mid = float(np.median(z))
    mol = np.where(z < mid, np.int8(1), np.int8(2))
    z1 = float(np.mean(z[mol == 1]))
    z2 = float(np.mean(z[mol == 2]))
    if z1 > z2:
        mol = np.where(mol == 1, np.int8(2), np.int8(1))
    atoms.set_array("mol-id", mol)


def _layer_separation_min_max(atoms) -> tuple[float, float]:
    """``2·min|z−⟨z⟩|`` and ``2·max|z−⟨z⟩|`` (Å), same as relaxation UQ scripts."""
    z = np.asarray(atoms.get_positions(wrap=False), dtype=float)[:, 2]
    dz = np.abs(z - float(np.mean(z)))
    return 2.0 * float(np.min(dz)), 2.0 * float(np.max(dz))


def _layer_separations(atoms) -> dict[str, float]:
    """AB-style (min gap) and AA-style (max span) interlayer separations in Å."""
    _ensure_mol_id_from_z(atoms)
    pos = atoms.get_positions(wrap=False)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    u = np.unique(mol)
    if u.size != 2:
        raise ValueError(f"Expected 2 mol-id values, got {u!r}")
    z1 = pos[mol == int(u[0]), 2]
    z2 = pos[mol == int(u[1]), 2]
    z_bot, z_top = (z1, z2) if np.mean(z1) < np.mean(z2) else (z2, z1)
    ab = float(np.min(z_top) - np.max(z_bot))   # closest pair (min gap)
    aa = float(np.max(z_top) - np.min(z_bot))   # farthest pair (max span)
    return {"ab_separation": ab, "aa_separation": aa}


def _sep_in_band(value: Optional[float]) -> bool:
    if value is None or not np.isfinite(float(value)):
        return False
    v = float(value)
    return SEP_MIN <= v <= SEP_MAX


def _separation_flags(
    ab: Optional[float],
    aa: Optional[float],
    sep_min: Optional[float],
    sep_max: Optional[float],
) -> dict[str, bool]:
    """Per-metric band checks and combined TBLG separation acceptance."""
    ab_ok = _sep_in_band(ab)
    aa_ok = _sep_in_band(aa)
    # Legacy CSV rows may lack sep_min/sep_max; AB/AA are the gap metrics then.
    sep_min_ok = _sep_in_band(sep_min) if sep_min is not None else ab_ok
    sep_max_ok = _sep_in_band(sep_max) if sep_max is not None else aa_ok
    ab_lt_aa = (
        ab is not None
        and aa is not None
        and np.isfinite(ab)
        and np.isfinite(aa)
        and float(ab) < float(aa)
    )
    return {
        "ab_in_target": ab_ok,
        "aa_in_target": aa_ok,
        "sep_min_in_target": sep_min_ok,
        "sep_max_in_target": sep_max_ok,
        "ab_lt_aa": ab_lt_aa,
        "separations_ok": bool(
            ab_ok and aa_ok and sep_min_ok and sep_max_ok and ab_lt_aa
        ),
    }


def _build_tblg_atoms(theta_deg: float) -> Optional[object]:
    """Build a TBLG commensurate supercell using ``flatgraphene``.

    Returns ``None`` if ``flatgraphene`` is not installed.
    """
    try:
        import flatgraphene as fg
    except ImportError:
        return None

    p, q, _ = fg.twist.find_p_q(float(theta_deg), a_tol=0.1)
    atoms = fg.twist.make_graphene(
        cell_type="hex",
        n_layer=2,
        p=p,
        q=q,
        lat_con=2.46,
        sym=["C", "C"],
        mass=[12.01, 12.01],
        sep=3.35,
        h_vac=20.0,
    )
    _ensure_mol_id_from_z(atoms)
    return atoms


# ---------------------------------------------------------------------------
# TBLG relaxation
# ---------------------------------------------------------------------------

def _plot_tblg_cross_section(atoms, theta_deg: float) -> Path:
    """Scatter top-layer xy positions colored by interlayer separation.

    Interlayer separation per atom is ``2·|z − ⟨z⟩|`` (Å), where ``⟨z⟩`` is the
    mean z over all atoms.  Only top-layer atoms (``z > ⟨z⟩``) are plotted.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plotted = atoms.copy()
    plotted.wrap()
    pos = np.asarray(plotted.get_positions(wrap=False), dtype=float)

    mean_z = float(np.mean(pos[:, 2]))
    top_mask = pos[:, 2] > mean_z
    top_pos = pos[top_mask]
    if top_pos.shape[0] < 1:
        raise ValueError("No top-layer atoms found for scatter plot")

    separation = 2.0 * np.abs(top_pos[:, 2] - mean_z)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    sc = ax.scatter(
        top_pos[:, 0],
        top_pos[:, 1],
        c=separation,
        s=25,
        cmap="viridis",
        linewidths=0.0,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"interlayer separation $2\,|z_i - \langle z \rangle|$ (Å)")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(rf"$\theta = {float(theta_deg):g}^\circ$")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out = _THIS_DIR / f"tmp_tblg_cross_section_theta_{float(theta_deg):g}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _attempt_tblg_relax(
    atoms0,
    hyperparams: dict,
    pod_params: np.ndarray,
    rcut: float,
    *,
    tb_params: Optional[np.ndarray] = None,
    theta_deg: Optional[float] = None,
) -> dict[str, Any]:
    """Relax a TBLG structure with ASE FIRE and plot its cross section."""
    calc = _make_energy_calculator(
        hyperparams, pod_params, rcut, tb_params=tb_params,
    )
    atoms = atoms0.copy()
    atoms.calc = calc

    result: dict[str, Any] = {
        "n_atoms": len(atoms0),
        "converged": False,
        "fmax_initial": None,
        "fmax_after": None,
        "ab_separation": None,
        "aa_separation": None,
        "ab_in_target": None,
        "aa_in_target": None,
        "sep_min": None,
        "sep_max": None,
        "sep_min_in_target": None,
        "sep_max_in_target": None,
        "ab_lt_aa": None,
        "cross_section_plot": None,
        "cross_section_plot_error": None,
        "error": None,
    }

    try:
        # Mirror test_relaxation: check initial forces first.
        f0 = np.asarray(atoms.get_forces(), dtype=float)
        fmax0 = float(np.max(np.linalg.norm(f0, axis=1)))
        result["fmax_initial"] = fmax0
        if fmax0 > 1e8:
            result["error"] = f"Pre-relax force explosion: max|F|={fmax0:.2e} eV/Å"
            return result

        relaxed = calc.relax_structure(
            atoms,
            relax_backend="ase",
            etol=RELAX_ETOL,
            ftol=RELAX_FTOL,
            maxiter=RELAX_MAX_STEPS,
            maxeval=RELAX_MAX_EVAL,
        )

        _ensure_mol_id_from_z(relaxed)
        f = np.asarray(relaxed.get_forces(), dtype=float)
        fmax_after = float(np.max(np.linalg.norm(f, axis=1)))
        result["fmax_after"] = fmax_after
        result["converged"] = bool(fmax_after <= RELAX_FTOL * (1.0 + 1e-6))

        if theta_deg is not None:
            try:
                out = _plot_tblg_cross_section(relaxed, theta_deg)
                result["cross_section_plot"] = str(out)
                print(f"  [tblg θ={theta_deg:g}°] wrote cross section: {out}", flush=True)
            except Exception as exc:
                result["cross_section_plot_error"] = (
                    f"{type(exc).__name__}: {str(exc)[:280]}"
                )

        if result["converged"] or fmax_after < 10.0:
            seps = _layer_separations(relaxed)
            sep_min, sep_max = _layer_separation_min_max(relaxed)
            flags = _separation_flags(
                seps["ab_separation"],
                seps["aa_separation"],
                sep_min,
                sep_max,
            )
            result["ab_separation"] = seps["ab_separation"]
            result["aa_separation"] = seps["aa_separation"]
            result["sep_min"] = sep_min
            result["sep_max"] = sep_max
            result.update(flags)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:280]}"
    finally:
        calc.close()

    return result


def _tblg_relax_worker(
    atoms0,
    hyperparams: dict,
    pod_params: np.ndarray,
    rcut: float,
    tb_params,
    theta_deg: float,
    result_queue,
) -> None:
    """Subprocess target: run ``_attempt_tblg_relax`` and push the result dict."""
    try:
        res = _attempt_tblg_relax(
            atoms0,
            hyperparams,
            pod_params,
            rcut,
            tb_params=tb_params,
            theta_deg=theta_deg,
        )
    except Exception as exc:
        res = {
            "n_atoms": len(atoms0) if atoms0 is not None else 0,
            "converged": False,
            "fmax_initial": None,
            "fmax_after": None,
            "ab_separation": None,
            "aa_separation": None,
            "ab_in_target": None,
            "aa_in_target": None,
            "sep_min": None,
            "sep_max": None,
            "sep_min_in_target": None,
            "sep_max_in_target": None,
            "ab_lt_aa": None,
            "cross_section_plot": None,
            "cross_section_plot_error": None,
            "error": f"{type(exc).__name__}: {str(exc)[:280]}",
        }
    result_queue.put(res)


_CRASH_SIGNALS: dict[int, str] = {6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV"}


def _attempt_tblg_relax_safe(
    atoms0,
    hyperparams: dict,
    pod_params: np.ndarray,
    rcut: float,
    tb_params,
    theta_deg: float,
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Run ``_attempt_tblg_relax`` in a child process to isolate LAMMPS crashes.

    LAMMPS can segfault (SIGSEGV) for certain POD architectures on large TBLG
    cells — a hard C-level crash that Python ``try/except`` cannot intercept.
    Spawning a child process means the crash kills only the child; the parent
    reads the non-zero exit code and records the trial as failed instead of
    crashing the whole search.

    Uses ``fork`` start-method (Linux default) so no pickling overhead.
    """
    n_atoms = len(atoms0) if atoms0 is not None else 0

    def _failure(msg: str) -> dict[str, Any]:
        return {
            "n_atoms": n_atoms,
            "converged": False,
            "fmax_initial": None,
            "fmax_after": None,
            "ab_separation": None,
            "aa_separation": None,
            "ab_in_target": None,
            "aa_in_target": None,
            "sep_min": None,
            "sep_max": None,
            "sep_min_in_target": None,
            "sep_max_in_target": None,
            "ab_lt_aa": None,
            "cross_section_plot": None,
            "cross_section_plot_error": None,
            "error": msg,
        }

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=_tblg_relax_worker,
        args=(atoms0, hyperparams, pod_params, rcut, tb_params, theta_deg, q),
        daemon=True,
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join()
        return _failure(f"TBLG relax timed out after {timeout:.0f} s")

    if p.exitcode != 0:
        if p.exitcode is not None and p.exitcode < 0:
            sig = -p.exitcode
            name = _CRASH_SIGNALS.get(sig, f"signal {sig}")
        else:
            name = f"exitcode={p.exitcode}"
        return _failure(f"LAMMPS subprocess crashed ({name})")

    try:
        return q.get_nowait()
    except Exception:
        return _failure("Subprocess exited cleanly but produced no result")


def _tblg_criteria_met(tr: dict[str, Any]) -> bool:
    """True when this angle counts as success for ``meets_all_criteria``."""
    if tr.get("skipped") or tr.get("error"):
        return False
    return bool(tr.get("converged") and tr.get("separations_ok"))


def _tblg_skipped_after_prior_angle_failed(prior_theta_deg: float) -> dict[str, Any]:
    """Placeholder result: relaxation not run after an earlier angle failed."""
    return {
        "n_atoms": None,
        "converged": False,
        "fmax_initial": None,
        "fmax_after": None,
        "ab_separation": None,
        "aa_separation": None,
        "ab_in_target": False,
        "aa_in_target": False,
        "sep_min": None,
        "sep_max": None,
        "sep_min_in_target": False,
        "sep_max_in_target": False,
        "ab_lt_aa": False,
        "separations_ok": False,
        "error": (
            f"Skipped (marked failed): first TBLG angle θ={prior_theta_deg:g}° "
            "did not meet convergence / separation criteria; this angle was not attempted."
        ),
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_candidate_paths(h: str) -> list[Path]:
    """Primary cache file first; legacy flat ``<CACHE_DIR>/<hash>.npz`` last (POD_energy only)."""
    subdir = _cache_subdir()
    primary = _THIS_DIR / CACHE_DIR / subdir / f"{h}.npz"
    out = [primary]
    if SEARCH_MODEL == "POD_energy" and not _level_of_theory_tag():
        legacy = _THIS_DIR / CACHE_DIR / f"{h}.npz"
        if legacy.resolve() != primary.resolve():
            out.append(legacy)
        legacy_model = _THIS_DIR / CACHE_DIR / SEARCH_MODEL / f"{h}.npz"
        if legacy_model.resolve() != primary.resolve():
            out.append(legacy_model)
    return out


def _cache_path(h: str) -> Path:
    """Canonical path for **writing** new fit caches."""
    return _THIS_DIR / CACHE_DIR / _cache_subdir() / f"{h}.npz"


def _load_cached_fit(h: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(pod_params, tb_params)``.  ``tb_params`` is ``None`` for ``POD_energy`` caches."""
    for p in _cache_candidate_paths(h):
        if not p.is_file():
            continue
        z = np.load(str(p), allow_pickle=True)
        if "pod_params" in z.files:
            pod = np.asarray(z["pod_params"], dtype=float).ravel()
            tb = None
            if "tb_params" in z.files:
                tb = np.asarray(z["tb_params"], dtype=float).ravel()
            return pod, tb
        if "params" in z.files:
            return np.asarray(z["params"], dtype=float).ravel(), None
    return None, None


def _save_cached_fit(
    h: str,
    pod_params: np.ndarray,
    tb_params: Optional[np.ndarray] = None,
) -> None:
    p = _cache_path(h)
    p.parent.mkdir(parents=True, exist_ok=True)
    pod_arr = np.asarray(pod_params, dtype=float).ravel()
    if tb_params is None:
        np.savez(str(p), pod_params=pod_arr)
    else:
        np.savez(
            str(p),
            pod_params=pod_arr,
            tb_params=np.asarray(tb_params, dtype=float).ravel(),
        )


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(
    cfg: dict,
    train_atoms: list,
    test_atoms: list,
    test_E: np.ndarray,
    n_train: int,
    tblg_cache: dict[float, Optional[object]],
    completed_hashes: set[str],
    tetb_aux: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Fit and evaluate one hyperparameter config.

    Returns a result dict, or ``None`` if the trial was skipped (already done
    or ncoeff constraint violated).
    """
    h = _config_hash(cfg)
    if h in completed_hashes:
        return None

    hyperparams = _build_hyperparams(cfg)
    ncoeff = int(ncoeff_from_params(hyperparams))
    rcut = float(cfg["rcut"])

    if ncoeff > n_train:
        print(
            f"  [skip] ncoeff={ncoeff} > n_train={n_train}  hash={h}",
            flush=True,
        )
        return None

    print(
        f"\n{'='*68}\n"
        f"  config hash={h}  ncoeff={ncoeff}  rcut={rcut}"
        f"  reg={cfg['regularization']:.0e}\n"
        f"  bessel={hyperparams['bessel_polynomial_degree']}  "
        f"inverse={hyperparams['inverse_polynomial_degree']}\n"
        f"  2b={cfg['two_body_radial']}  3b={cfg['three_body_radial']}/{cfg['three_body_angular']}"
        f"  4b={cfg['four_body_radial']}/{cfg['four_body_angular']}"
        f"  5b={cfg['five_body_radial']}/{cfg['five_body_angular']}",
        flush=True,
    )

    result: dict[str, Any] = {
        "hash": h,
        "search_model": SEARCH_MODEL,
        "config": cfg,
        "ncoeff": ncoeff,
        "n_train": n_train,
        "fit_time_s": None,
        "error_from_fit": None,
        "test_energy_mae_per_atom_eV": None,
        "test_energy_rmse_per_atom_eV": None,
        "tblg": {},
        "meets_all_criteria": False,
    }

    # --- Fit (or load from cache) ------------------------------------------
    pod_params: Optional[np.ndarray] = None
    tb_params: Optional[np.ndarray] = None
    test_err_from_fit: Optional[dict[str, float]] = None
    pod_cached, tb_cached = _load_cached_fit(h)
    cache_ok = pod_cached is not None and (
        SEARCH_MODEL != "TETB_POD" or tb_cached is not None
    )
    if cache_ok:
        assert pod_cached is not None
        pod_params = pod_cached
        tb_params = tb_cached
        print(f"  [fit] loaded from cache ({SEARCH_MODEL})", flush=True)
        result["fit_time_s"] = 0.0
    else:
        hyperparams_str = pod_hyperparams_to_str(hyperparams, rcut, ["C"])
        pod_params, tb_params, test_err_from_fit, fit_time, fit_err = (
            _attempt_fit(
                cfg,
                hyperparams,
                hyperparams_str,
                train_atoms,
                rcut,
                test_atoms,
                test_E,
                tetb_aux,
            )
        )
        result["fit_time_s"] = fit_time
        if fit_err is not None:
            result["error_from_fit"] = fit_err
            print(f"  [fit] FAILED: {fit_err}", flush=True)
            return result
        assert pod_params is not None
        print(
            f"  [fit] done in {result['fit_time_s']:.1f}s  n_pod={pod_params.size}"
            + (
                f"  n_tb={tb_params.size}"
                if SEARCH_MODEL == "TETB_POD" and tb_params is not None
                else ""
            ),
            flush=True,
        )
        _save_cached_fit(h, pod_params, tb_params=tb_params)

    assert pod_params is not None

    # --- Test-set evaluation -----------------------------------------------
    if test_err_from_fit is not None:
        err = test_err_from_fit
        result["test_energy_mae_per_atom_eV"] = err["energy_mae_per_atom_eV"]
        result["test_energy_rmse_per_atom_eV"] = err["energy_rmse_per_atom_eV"]
        print(
            f"  [test] MAE={err['energy_mae_per_atom_eV']*1e3:.3f} meV/atom"
            f"  RMSE={err['energy_rmse_per_atom_eV']*1e3:.3f} meV/atom",
            flush=True,
        )
    else:
        try:
            err = _evaluate_test_error(
                hyperparams,
                pod_params,
                test_atoms,
                test_E,
                rcut,
                tb_params=tb_params,
            )
            result["test_energy_mae_per_atom_eV"] = err["energy_mae_per_atom_eV"]
            result["test_energy_rmse_per_atom_eV"] = err["energy_rmse_per_atom_eV"]
            print(
                f"  [test] MAE={err['energy_mae_per_atom_eV']*1e3:.3f} meV/atom"
                f"  RMSE={err['energy_rmse_per_atom_eV']*1e3:.3f} meV/atom",
                flush=True,
            )
        except Exception as exc:
            print(f"  [test] evaluation failed: {exc}", flush=True)
            result["error_from_fit"] = (result.get("error_from_fit") or "") + \
                f" | test eval: {type(exc).__name__}: {str(exc)[:200]}"

    # --- TBLG relaxation ---------------------------------------------------
    all_tblg_ok = True

    for theta in TBLG_THETAS:
        atoms0 = tblg_cache.get(theta)

        if atoms0 is None:
            result["tblg"][str(theta)] = {"skipped": "flatgraphene not installed"}
            all_tblg_ok = False
            continue

        print(f"  [tblg θ={theta}°] n_atoms={len(atoms0)} relaxing…", flush=True)
        t1 = time.time()
        tr = _attempt_tblg_relax_safe(
            atoms0, hyperparams, pod_params, rcut, tb_params, float(theta),
        )
        elapsed = time.time() - t1

        converged = tr["converged"]
        ab = tr["ab_separation"]
        aa = tr["aa_separation"]
        ab_ok = tr["ab_in_target"]
        aa_ok = tr["aa_in_target"]
        result["tblg"][str(theta)] = {**tr, "relax_time_s": elapsed}

        passed = _tblg_criteria_met(tr)

        status = "✓" if passed else "✗"
        sep_str = (
            f"AB={ab:.3f} AA={aa:.3f} min={tr.get('sep_min')} max={tr.get('sep_max')} Å"
            if ab is not None and aa is not None
            else "no separation (not converged)"
        )
        print(
            f"  [tblg θ={theta}°] {status}  fmax_after={tr['fmax_after']}  "
            f"{sep_str}  {elapsed:.1f}s",
            flush=True,
        )
        if tr.get("error"):
            print(f"    error: {tr['error']}", flush=True)

        if not passed:
            all_tblg_ok = False

    result["meets_all_criteria"] = bool(
        result["test_energy_rmse_per_atom_eV"] is not None
        and np.isfinite(result["test_energy_rmse_per_atom_eV"])
        and all_tblg_ok
    )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_results(results: list[dict], json_path: Path, csv_path: Path) -> None:
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if not results:
        return

    csv_keys = [
        "hash", "search_model", "ncoeff", "n_train",
        "test_energy_mae_per_atom_eV", "test_energy_rmse_per_atom_eV",
        "fit_time_s", "meets_all_criteria", "error_from_fit",
        "rcut", "regularization", "weight_energy", "weight_force",
        "bessel_polynomial_degree", "inverse_polynomial_degree",
        "two_body_radial",
        "three_body_radial", "three_body_angular",
        "four_body_radial",  "four_body_angular",
        "five_body_radial",  "five_body_angular",
        "six_body_radial",   "six_body_angular",
        "seven_body_radial", "seven_body_angular",
    ]
    # Add per-theta columns dynamically
    theta_keys: list[str] = []
    for theta in TBLG_THETAS:
        for k in (
            "converged", "fmax_initial", "fmax_after",
            "ab_separation", "aa_separation", "sep_min", "sep_max",
            "ab_in_target", "aa_in_target", "sep_min_in_target", "sep_max_in_target",
            "ab_lt_aa", "separations_ok", "error",
        ):
            theta_keys.append(f"tblg_{theta}_{k}")

    all_keys = csv_keys + theta_keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row: dict = {k: r.get(k) for k in csv_keys}
            cfg = r.get("config", {})
            for k in _POD_INT_KEYS + ("rcut", "regularization",
                                       "weight_energy", "weight_force"):
                row[k] = cfg.get(k)
            for theta in TBLG_THETAS:
                td = r.get("tblg", {}).get(str(theta), {})
                for k in (
                    "converged", "fmax_initial", "fmax_after",
                    "ab_separation", "aa_separation", "sep_min", "sep_max",
                    "ab_in_target", "aa_in_target", "sep_min_in_target",
                    "sep_max_in_target", "ab_lt_aa", "separations_ok", "error",
                ):
                    row[f"tblg_{theta}_{k}"] = td.get(k)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _find_best(results: list[dict]) -> Optional[dict]:
    """Return the result with the lowest test RMSE that meets all criteria."""
    candidates = [r for r in results if r.get("meets_all_criteria")]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["test_energy_rmse_per_atom_eV"])


def _tblg_relax_for_plot_worker(
    theta: float,
    hyperparams: dict,
    pod_params: np.ndarray,
    rcut: float,
    tb_params,
    result_queue,
) -> None:
    """Subprocess target: relax TBLG and return per-atom separation for plotting."""
    atoms0 = _build_tblg_atoms(float(theta))
    if atoms0 is None:
        result_queue.put({"ok": False, "error": "flatgraphene not installed"})
        return

    calc = _make_energy_calculator(
        hyperparams, pod_params, rcut, tb_params=tb_params,
    )
    atoms = atoms0.copy()
    atoms.calc = calc
    try:
        relaxed = calc.relax_structure(
            atoms,
            relax_backend="ase",
            etol=RELAX_ETOL,
            ftol=RELAX_FTOL,
            maxiter=RELAX_MAX_STEPS,
            maxeval=RELAX_MAX_EVAL,
        )
        _ensure_mol_id_from_z(relaxed)
        pos = np.asarray(relaxed.get_positions(wrap=False), dtype=float)
        mol = np.asarray(relaxed.get_array("mol-id"), dtype=int).ravel()

        z_mean = float(np.mean(pos[:, 2]))
        u = np.unique(mol)
        if u.size != 2:
            raise ValueError(f"Expected 2 layers via mol-id, got {u!r}")
        z0 = float(np.mean(pos[mol == int(u[0]), 2]))
        z1 = float(np.mean(pos[mol == int(u[1]), 2]))
        top = int(u[0]) if z0 > z1 else int(u[1])

        xy = pos[mol == top, :2]
        zt = pos[mol == top, 2]
        sep = 2.0 * (zt - z_mean)

        result_queue.put(
            {
                "ok": True,
                "theta": float(theta),
                "xy": np.asarray(xy, dtype=float),
                "sep": np.asarray(sep, dtype=float),
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:280]}"})
    finally:
        try:
            calc.close()
        except Exception:
            pass


def _plot_best_tblg_toplayer_separations(best: dict) -> None:
    """After finding the best model, save a per-atom top-layer separation plot for each TBLG."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"[search] matplotlib unavailable; skipping best-model TBLG separation plots: {exc}",
            flush=True,
        )
        return

    cfg = best.get("config") or {}
    h = best.get("hash")
    if not h or not isinstance(cfg, dict):
        print("[search] best result missing 'hash'/'config'; skipping plots.", flush=True)
        return

    pod_params, tb_params = _load_cached_fit(str(h))
    if pod_params is None:
        tried = ", ".join(str(p) for p in _cache_candidate_paths(str(h)))
        print(
            f"[search] best-model params cache missing (tried: {tried}); skipping plots.",
            flush=True,
        )
        return
    if SEARCH_MODEL == "TETB_POD" and tb_params is None:
        print(
            "[search] best-model TETB cache missing ``tb_params``; skipping plots.",
            flush=True,
        )
        return

    hyperparams = _build_hyperparams(cfg)
    rcut = float(cfg.get("rcut", 6.0))
    out_dir = _THIS_DIR / "figures" / "pod_hyperparam_search" / SEARCH_MODEL
    out_dir.mkdir(parents=True, exist_ok=True)

    for theta in TBLG_THETAS:
        print(
            f"[search] [best plot] relaxing θ={theta}° for per-atom separation field…",
            flush=True,
        )
        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(
            target=_tblg_relax_for_plot_worker,
            args=(float(theta), hyperparams, pod_params, rcut, tb_params, q),
            daemon=True,
        )
        p.start()
        p.join(timeout=900.0)

        if p.is_alive():
            p.kill()
            p.join()
            print(f"[search] [best plot] θ={theta}° timed out; skipping.", flush=True)
            continue

        if p.exitcode != 0:
            code = p.exitcode
            if code is not None and code < 0:
                sig = -code
                name = _CRASH_SIGNALS.get(sig, f"signal {sig}")
            else:
                name = f"exitcode={code}"
            print(
                f"[search] [best plot] θ={theta}° subprocess crashed ({name}); skipping.",
                flush=True,
            )
            continue

        try:
            r = q.get_nowait()
        except Exception:
            print(f"[search] [best plot] θ={theta}° no result produced; skipping.", flush=True)
            continue

        if not r.get("ok"):
            print(f"[search] [best plot] θ={theta}° failed: {r.get('error')}", flush=True)
            continue

        xy = np.asarray(r["xy"], dtype=float)
        sep = np.asarray(r["sep"], dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 5.6), dpi=160)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=sep,
            s=20,
            cmap="viridis",
            linewidths=0.0,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$2\,(z_i - \langle z \rangle)$ (Å)")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_title(
            f"Best {SEARCH_MODEL} model (hash={h}) — TBLG θ={theta}° top-layer separation field"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = out_dir / f"best_tblg_toplayer_separation_theta_{theta:g}_hash_{h}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"[search] [best plot] wrote: {out}", flush=True)


def _print_summary(results: list[dict]) -> None:
    n_done   = len(results)
    n_ok     = sum(1 for r in results if r.get("meets_all_criteria"))
    n_failed = sum(1 for r in results if r.get("error_from_fit"))

    print("\n" + "=" * 68)
    print(f"SEARCH COMPLETE: {n_done} trials, {n_ok} meet all criteria, "
          f"{n_failed} fit failures.")

    best = _find_best(results)
    if best is None:
        print("\nNo configuration met all criteria.")
        print(
            f"Consider relaxing [{SEP_MIN}, {SEP_MAX}] Å separation band, "
            "RELAX_FTOL, or changing the grid."
        )
        # Still print the best by test RMSE ignoring TBLG
        valid = [r for r in results if r.get("test_energy_rmse_per_atom_eV") is not None]
        if valid:
            runner_up = min(valid, key=lambda r: r["test_energy_rmse_per_atom_eV"])
            print(
                f"\nBest by test RMSE (ignoring TBLG stability):"
                f"\n  hash={runner_up['hash']}"
                f"\n  RMSE={runner_up['test_energy_rmse_per_atom_eV']*1e3:.3f} meV/atom"
                f"\n  ncoeff={runner_up['ncoeff']}"
                f"\n  config={runner_up['config']}"
            )
        return

    cfg = best["config"]
    print(
        f"\n{'*'*68}"
        f"\nBEST MODEL  hash={best['hash']}"
        f"\n  Test RMSE : {best['test_energy_rmse_per_atom_eV']*1e3:.4f} meV/atom"
        f"\n  Test MAE  : {best['test_energy_mae_per_atom_eV']*1e3:.4f} meV/atom"
        f"\n  ncoeff    : {best['ncoeff']}"
        f"\n  rcut      : {cfg['rcut']} Å"
        f"\n  reg       : {cfg['regularization']:.0e}"
        f"\n  bessel / inverse poly degree : "
        f"{cfg['bessel_polynomial_degree']} / {cfg['inverse_polynomial_degree']}"
        f"\n  2-body radial                : {cfg['two_body_radial']}"
        f"\n  3-body radial / angular      : {cfg['three_body_radial']} / {cfg['three_body_angular']}"
        f"\n  4-body radial / angular      : {cfg['four_body_radial']} / {cfg['four_body_angular']}"
        f"\n  5-body radial / angular      : {cfg['five_body_radial']} / {cfg['five_body_angular']}"
        f"\n  6-body radial / angular      : {cfg['six_body_radial']} / {cfg['six_body_angular']}"
        f"\n  7-body radial / angular      : {cfg['seven_body_radial']} / {cfg['seven_body_angular']}"
    )
    for theta in TBLG_THETAS:
        td = best.get("tblg", {}).get(str(theta), {})
        print(
            f"\n  TBLG θ={theta}°:"
            f"\n    converged       : {td.get('converged')}"
            f"\n    max |F| after   : {td.get('fmax_after')} eV/Å"
            f"\n    AB separation   : {td.get('ab_separation')} Å"
            f"\n    AA separation   : {td.get('aa_separation')} Å"
            f"\n    min layer sep   : {td.get('sep_min')} Å"
            f"\n    max layer sep   : {td.get('sep_max')} Å"
            f"\n      (band [{SEP_MIN}, {SEP_MAX}] Å, AB < AA)"
        )
    print(f"{'*'*68}\n")
    print(
        f"To use this {SEARCH_MODEL} model, set POD descriptor knobs in ``test_relaxation.py`` "
        "(or read degrees from JSON config):\n"
        f"  (POD descriptor) bessel_polynomial_degree = {cfg['bessel_polynomial_degree']}\n"
        f"  (POD descriptor) inverse_polynomial_degree = {cfg['inverse_polynomial_degree']}\n"
        f"  RELAXATION_POD_TWO_BODY_RADIAL   = {cfg['two_body_radial']}\n"
        f"  RELAXATION_POD_THREE_BODY_RADIAL  = {cfg['three_body_radial']}\n"
        f"  RELAXATION_POD_THREE_BODY_ANGULAR = {cfg['three_body_angular']}\n"
        f"  RELAXATION_POD_FOUR_BODY_RADIAL   = {cfg['four_body_radial']}\n"
        f"  RELAXATION_POD_FOUR_BODY_ANGULAR  = {cfg['four_body_angular']}\n"
        f"  RELAXATION_POD_FIVE_BODY_RADIAL   = {cfg['five_body_radial']}\n"
        f"  RELAXATION_POD_FIVE_BODY_ANGULAR  = {cfg['five_body_angular']}\n"
        f"  RELAXATION_POD_SIX_BODY_RADIAL    = {cfg['six_body_radial']}\n"
        f"  RELAXATION_POD_SIX_BODY_ANGULAR   = {cfg['six_body_angular']}\n"
        f"  RELAXATION_POD_SEVEN_BODY_RADIAL  = {cfg['seven_body_radial']}\n"
        f"  RELAXATION_POD_SEVEN_BODY_ANGULAR = {cfg['seven_body_angular']}\n"
        f"  RELAXATION_POD_REGULARIZATION     = {cfg['regularization']:.0e}\n"
    )
    if SEARCH_MODEL == "TETB_POD":
        print(
            "\nTETB_POD: the subtracted TB baseline used here matches "
            f"``TETB_TB_M={TETB_TB_M}``, ``TETB_TB_W={TETB_TB_W}``, "
            f"``TETB_TB_RCUT={TETB_TB_RCUT}`` (see ``get_MCMC_inputs`` / "
            "``build_tetb_pod_hyperparams_from_data_kw`` for MCMC parity).\n"
        )
    _plot_best_tblg_toplayer_separations(best)


# ---------------------------------------------------------------------------
# Re-evaluate stored CSV (separation band criteria)
# ---------------------------------------------------------------------------


def _parse_bool_csv(val: Any) -> bool:
    if val is None:
        return False
    try:
        if isinstance(val, float) and np.isnan(val):
            return False
    except TypeError:
        pass
    return str(val).strip().lower() in ("true", "1", "yes")


def _parse_float_csv(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _tblg_error_present(val: Any) -> bool:
    """True when a stored TBLG error cell has a non-empty message (not NaN/blank)."""
    if val is None:
        return False
    try:
        if isinstance(val, float) and np.isnan(val):
            return False
    except TypeError:
        pass
    try:
        import pandas as pd

        if pd.isna(val):
            return False
    except (TypeError, ValueError, ImportError):
        pass
    return bool(str(val).strip())


def _tblg_row_passes_csv(row: Any, theta: float, columns: set[str]) -> tuple[bool, dict[str, bool]]:
    """Apply current separation band rules to one angle's flattened CSV columns."""
    prefix = f"tblg_{theta}_"
    err = row.get(f"{prefix}error") if hasattr(row, "get") else row[f"{prefix}error"]
    if _tblg_error_present(err):
        return False, {}
    conv_key = f"{prefix}converged"
    if conv_key in columns and not _parse_bool_csv(
        row.get(conv_key) if hasattr(row, "get") else row[conv_key]
    ):
        return False, {}

    ab = _parse_float_csv(
        row.get(f"{prefix}ab_separation") if hasattr(row, "get") else row[f"{prefix}ab_separation"]
    )
    aa = _parse_float_csv(
        row.get(f"{prefix}aa_separation") if hasattr(row, "get") else row[f"{prefix}aa_separation"]
    )
    sep_min = (
        _parse_float_csv(row.get(f"{prefix}sep_min") if hasattr(row, "get") else row[f"{prefix}sep_min"])
        if f"{prefix}sep_min" in columns
        else None
    )
    sep_max = (
        _parse_float_csv(row.get(f"{prefix}sep_max") if hasattr(row, "get") else row[f"{prefix}sep_max"])
        if f"{prefix}sep_max" in columns
        else None
    )
    flags = _separation_flags(ab, aa, sep_min, sep_max)
    return bool(flags["separations_ok"]), flags


def reevaluate_meets_all_criteria_csv(csv_path: Path) -> tuple[int, int]:
    """
    Recompute ``meets_all_criteria`` and per-angle separation flags in a results CSV.

    Uses :data:`SEP_MIN` / :data:`SEP_MAX`, ``AB < AA``, and stored ``sep_min`` /
    ``sep_max`` when present (otherwise AB/AA stand in for legacy rows).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    columns = set(df.columns)
    n_ok = 0

    flag_cols = (
        "ab_in_target", "aa_in_target", "ab_lt_aa",
        "sep_min_in_target", "sep_max_in_target", "separations_ok",
    )
    for theta in TBLG_THETAS:
        for key in flag_cols:
            col = f"tblg_{theta}_{key}"
            if col not in df.columns:
                df[col] = False
            else:
                df[col] = df[col].astype(bool)

    for i in range(len(df)):
        rmse = _parse_float_csv(df.at[i, "test_energy_rmse_per_atom_eV"])
        meets = rmse is not None
        if meets:
            for theta in TBLG_THETAS:
                ok, flags = _tblg_row_passes_csv(df.iloc[i], theta, set(df.columns))
                prefix = f"tblg_{theta}_"
                for key in flag_cols:
                    df.at[i, f"{prefix}{key}"] = flags.get(key, False)
                if not ok:
                    meets = False
                    break
        df.at[i, "meets_all_criteria"] = meets
        if meets:
            n_ok += 1

    df.to_csv(csv_path, index=False)
    return n_ok, len(df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clear_fresh_run_state() -> None:
    """Remove results JSON/CSV and fit cache so every config is refit from scratch."""
    for path in (_results_json_path(), _results_csv_path()):
        if path.is_file():
            path.unlink()
            print(f"[search] removed {path.name}", flush=True)

    cache_dir = _THIS_DIR / CACHE_DIR / _cache_subdir()
    if cache_dir.is_dir():
        n_removed = sum(1 for p in cache_dir.glob("*.npz") if p.is_file())
        shutil.rmtree(cache_dir)
        print(
            f"[search] cleared fit cache {cache_dir.name}/ ({n_removed} file(s))",
            flush=True,
        )

    if SEARCH_MODEL == "POD_energy" and not _level_of_theory_tag():
        legacy_flat = _THIS_DIR / CACHE_DIR
        if legacy_flat.is_dir():
            legacy_npz = list(legacy_flat.glob("*.npz"))
            for p in legacy_npz:
                p.unlink()
            if legacy_npz:
                print(
                    f"[search] cleared {len(legacy_npz)} legacy flat cache file(s) "
                    f"in {CACHE_DIR}/",
                    flush=True,
                )


def main() -> None:
    global LEVEL_OF_THEORY

    import argparse

    parser = argparse.ArgumentParser(description="POD hyperparameter grid search (tightened phase).")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete results JSON/CSV and cached POD fits "
            f"({CACHE_DIR}/<model>_<LOT>/) before running."
        ),
    )
    parser.add_argument(
        "--level-of-theory",
        default=LEVEL_OF_THEORY,
        choices=["rVV10", "MBD", "QMC"],
        help=(
            "DFT reference for bilayer energy training "
            "(default: rVV10; MBD uses data/bilayer_graphene_MBD.xyz)."
        ),
    )
    args = parser.parse_args()

    LEVEL_OF_THEORY = str(args.level_of_theory)

    # DataLoader paths are relative to uncertainty_quantification/ (../data → repo data/).
    os.chdir(_UQ_DIR)

    print(
        f"[search] level_of_theory={LEVEL_OF_THEORY}  "
        f"cache={CACHE_DIR}/{_cache_subdir()}/  "
        f"results={_results_json_path().name}",
        flush=True,
    )

    if args.fresh:
        _clear_fresh_run_state()

    json_path = _results_json_path()
    csv_path = _results_csv_path()

    # Build TBLG structures once (expensive if flatgraphene initializes slowly)
    tblg_cache: dict[float, Optional[object]] = {}
    for theta in TBLG_THETAS:
        atoms = _build_tblg_atoms(theta)
        tblg_cache[theta] = atoms
        if atoms is None:
            print(
                f"[search] WARNING: flatgraphene not installed; "
                f"TBLG θ={theta}° will be skipped.",
                flush=True,
            )
        else:
            print(
                f"[search] Built TBLG θ={theta}°  n_atoms={len(atoms)}",
                flush=True,
            )

    # Load data
    train_atoms, test_atoms, test_E, n_train, tetb_aux = _load_and_split()

    # Resume from previous trials in pod_hyperparam_search.json / .csv.
    existing = _load_results_for_resume()
    completed_hashes = {r["hash"] for r in existing}
    results = list(existing)
    print(
        f"[search] {len(completed_hashes)} trial(s) already completed "
        f"(loaded from {json_path.name}).",
        flush=True,
    )

    # Generate and optionally shuffle / cap the config list
    all_configs = _iter_valid_configs()
    total_raw = sum(1 for _ in itertools.product(*[GRID[k] for k in GRID]))
    print(
        f"[search] {len(all_configs)} valid configs "
        f"(from {total_raw} raw combinations in grid).",
        flush=True,
    )

    rng = np.random.default_rng(DATA_SPLIT_SEED)
    order = rng.permutation(len(all_configs)).tolist()
    configs_to_run = [all_configs[i] for i in order]
    if MAX_TRIALS is not None:
        configs_to_run = configs_to_run[:MAX_TRIALS]

    # Search loop
    trial_count = 0
    for cfg in configs_to_run:
        h = _config_hash(cfg)
        if h in completed_hashes:
            continue

        trial_count += 1
        result = _run_trial(
            cfg, train_atoms, test_atoms, test_E, n_train,
            tblg_cache, completed_hashes, tetb_aux,
        )
        if result is not None:
            results.append(result)
            completed_hashes.add(result["hash"])
            _save_results(results, json_path, csv_path)

    if trial_count == 0:
        print("[search] All configs already completed.  Nothing new to run.", flush=True)

    _save_results(results, json_path, csv_path)
    _print_summary(results)
    print(f"[search] Tightened results: {json_path}", flush=True)
    print(f"[search] Tightened CSV     : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
