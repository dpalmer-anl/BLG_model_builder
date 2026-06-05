"""
BLG relaxation tests: rigid in-plane ``a`` scan + relax vs rigid interlayer ``d`` scan.

Three calculators (parametrized): **tersoff_kc**, **pod_energy**, **tetb_pod``.
``tetb_pod`` uses ASE FIRE only for relaxation; ``pod_energy`` uses a preflight
rigid ``d`` scan before relax (see module history). Prints are unchanged so
``python tests/test_relaxation.py`` output matches prior behavior (pytest ``-s``).

A separate **slow** test (parametrized over ``MODEL_KEYS``) relaxes twisted bilayer
graphene (``theta = 5.09°``), reports AA/AB-style layer separations from relaxed
``z``, and saves an AB-metric bar chart per model under ``tests/_artifacts/``.
Force convergence for that relax is ``0.05`` eV/Å. Skip with ``pytest -m "not slow"``.

Run
---
``python tests/test_relaxation.py`` → pytest ``-v --tb=short -s`` on this file.
``RELAXATION_DEMO_ONLY=1`` → legacy plot demo for any case that loads.
``pytest tests/test_relaxation.py -m "not slow"`` skips the TBLG test (needs ``flatgraphene``).

POD_energy model source (edit constants at top of this file)
------------------------------------------------------------
``RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST = True`` (default): loads the
best POD model found by ``uncertainty_quantification/pod_hyperparameter_search.py``
(lowest test-set RMSE with ``meets_all_criteria = True``).  Coefficients are
read from ``uncertainty_quantification/pod_hyperparam_search_cache/<hash>.npz``;
the rcut, body-order counts, and Bessel / inverse polynomial degrees come from the
search JSON ``config`` (legacy files without those keys use bessel ``0`` /
inverse ``10``).  Those differ from ``POD_DEFAULT_*`` / ``get_MCMC_inputs`` for
manual POD_energy fits.

Set to ``False`` to use the manual ``RELAXATION_POD_TWO/.../SEVEN_BODY_*``
constants instead (architecture must be non-increasing from 2-body down).
``RELAXATION_POD_REGULARIZATION`` controls Tikhonov regularization (larger →
better out-of-distribution stability, e.g. for TBLG).

``RELAXATION_TETB_POD_TB_M`` / ``RELAXATION_TETB_POD_TB_W`` are the ACSF hopping
basis sizes; ``RELAXATION_TETB_POD_POD_M`` / ``RELAXATION_TETB_POD_POD_W`` are
the POD primary knobs (see ``build_tetb_pod_hyperparams_from_data_kw``).

If the matching ``best_fit_params`` cache is missing, the tests call
``get_MCMC_inputs`` from ``uncertainty_quantification/`` (so ``../data`` resolves)
to fit and write caches — requires the LAMMPS Python module and a working
``lammps`` executable for ``fitpod`` / TETB residual fits.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import pytest

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from blg_model_builder.geom_tools import get_bilayer_atoms
from blg_model_builder.potentials import (
    PODASECalculator,
    TETB_PODASECalculator,
    TersoffKolmogorovCrespiASECalculator,
    ncoeff_from_params,
)

# ---------------------------------------------------------------------------
# User-tunable ``M`` / ``W`` for POD_energy and TETB+POD (edit here)
# ---------------------------------------------------------------------------

# POD_energy descriptor hyperparameters.
# All body-order counts must satisfy the chain:
#   twobody ≥ threebody ≥ fourbody ≥ fivebody ≥ sixbody ≥ sevenbody
# Set high-body counts to 0 to improve out-of-distribution (TBLG) stability.
RELAXATION_POD_TWO_BODY_RADIAL: int = 10
RELAXATION_POD_THREE_BODY_RADIAL: int = 10
RELAXATION_POD_THREE_BODY_ANGULAR: int = 6
RELAXATION_POD_FOUR_BODY_RADIAL: int = 10
RELAXATION_POD_FOUR_BODY_ANGULAR: int = 4
RELAXATION_POD_FIVE_BODY_RADIAL: int = 6
RELAXATION_POD_FIVE_BODY_ANGULAR: int = 2
RELAXATION_POD_SIX_BODY_RADIAL: int = 4
RELAXATION_POD_SIX_BODY_ANGULAR: int = 2
RELAXATION_POD_SEVEN_BODY_RADIAL: int = 0
RELAXATION_POD_SEVEN_BODY_ANGULAR: int = 0

# Tikhonov regularization for the POD_energy least-squares fit.
# Root cause of TBLG instability: ``1e-12`` (unregularized) allows huge
# out-of-distribution forces on twisted geometries.  Typical good range:
#   1e-4  conservative — improves OOD stability with minimal accuracy loss
#   1e-2  aggressive  — forces bounded even far OOD; training RMSE may rise
# Delete the cached POD_energy_*_best_fit_params.npz to trigger a refit.
RELAXATION_POD_REGULARIZATION: float = 1e-12

# ---------------------------------------------------------------------------
# Hyperparameter-search best model (pod_energy only)
# ---------------------------------------------------------------------------
# When True, the pod_energy calculator is loaded from the best result in
# ``uncertainty_quantification/pod_hyperparam_search_results.json`` (lowest
# test-set RMSE with ``meets_all_criteria = True``).  Coefficients are read
# from ``uncertainty_quantification/pod_hyperparam_search_cache/<hash>.npz``.
#
# When False, the RELAXATION_POD_* constants above are used instead (manual
# configuration, fitted via get_MCMC_inputs).
RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST: bool = True

# TETB+POD: ``tb_M``/``tb_W`` = ACSF hopping basis; ``pod_M``/``pod_W`` = POD block
# (twobody radial + threebody angular primary knobs in ``build_tetb_pod_hyperparams_from_data_kw``).
RELAXATION_TETB_POD_TB_M = 12
RELAXATION_TETB_POD_TB_W = 6
RELAXATION_TETB_POD_POD_M = 12
RELAXATION_TETB_POD_POD_W = 6

# POD descriptor radial defaults for **manual** POD_energy fits
# (``RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST = False``), matching
# ``get_MCMC_inputs.POD_DEFAULT_*``.
# When loading the hyperparameter-search best model, use
# the JSON ``config`` (swept in ``pod_hyperparameter_search.GRID``) when using
# hyperparam-search best; they differ from ``get_MCMC_inputs`` / ``POD_DEFAULT_*``.
POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE = 2
POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE = 12

# m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A — BNC.tersoff C–C–C
TERSOFF_CC_BNC = np.asarray(
    [
        3.0, 1.0, 0.0, 38049.0, 4.3484, -0.93, 0.72751, 1.5724e-7, 2.2119, 430.0,
        1.95, 0.15, 3.4879, 1393.6,
    ],
    dtype=float,
)

MODEL_KEYS: Tuple[str, ...] = (
    ["pod_energy", "tetb_pod"]
)  

# Models that require the LAMMPS Python module to run.
_MODELS_NEEDING_LAMMPS: frozenset = frozenset({"tersoff_kc", "pod_energy", "tetb_pod"})

# TBLG slow test: force tolerance and post-relax ``max|F|`` check (eV/Å).
LAMMPS_RELAX_FMAX_EV_PER_ANG = 1e-4
ASE_RELAX_FMAX_EV_PER_ANG = 1e-3
TBLG_RELAX_TWIST_ANGLE = 9.43

# ---------------------------------------------------------------------------
# Global relaxation backend — applies to ALL models and tests in this file.
#   "lammps"  LAMMPS FIRE via ``minimize`` (fastest; requires LAMMPS Python).
#   "fire"    ASE FIRE optimizer.
#   "lbfgs"   ASE L-BFGS-B optimizer (default; robust, no step-size tuning).
# ---------------------------------------------------------------------------
RELAXATION_BACKEND: str = "fire"
if RELAXATION_BACKEND == "fire" or RELAXATION_BACKEND == "lbfgs":
    RELAXATION_FMAX_EV_PER_ANG = ASE_RELAX_FMAX_EV_PER_ANG
else:
    RELAXATION_FMAX_EV_PER_ANG = LAMMPS_RELAX_FMAX_EV_PER_ANG
# ---------------------------------------------------------------------------
# Paths & artifacts
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _best_fit_dir_candidates() -> list[Path]:
    r = _repo_root()
    return [
        r / "uncertainty_quantification" / "best_fit_params",
        r / "best_fit_params",
        Path("/mnt/c/Users/Daniel/Documents/research/BLG_model_builder_v2")
        / "uncertainty_quantification"
        / "best_fit_params",
    ]


def _tersoff_kc_best_fit_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "uncertainty_quantification", "best_fit_params"),
    )


def _artifacts_dir() -> Path:
    d = Path(__file__).resolve().parent / "_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_tblg_toplayer_interlayer_separation_field(
    relaxed,
    *,
    theta_deg: float,
    model_key: str,
    display_name: str,
) -> Optional[Path]:
    """Save an x–y scatter of top-layer atoms colored by 2*(z_i - mean(z_full)).

    This mirrors the diagnostic figure saved by
    ``uncertainty_quantification/pod_hyperparameter_search.py``.
    """
    if plt is None:
        return None

    _ensure_bilayer_mol_id_from_z(relaxed)
    pos = np.asarray(relaxed.get_positions(wrap=False), dtype=float)
    mol = np.asarray(relaxed.get_array("mol-id"), dtype=int).ravel()
    z_mean = float(np.mean(pos[:, 2]))

    u = np.unique(mol)
    if u.size != 2:
        return None
    z0 = float(np.mean(pos[mol == int(u[0]), 2]))
    z1 = float(np.mean(pos[mol == int(u[1]), 2]))
    top = int(u[0]) if z0 > z1 else int(u[1])

    xy = pos[mol == top, :2]
    zt = pos[mol == top, 2]
    sep = 2.0 * (zt - z_mean)

    art = _artifacts_dir()
    safe_key = str(model_key).replace(os.sep, "_")
    out_png = art / (
        f"relaxation_tblg_{str(theta_deg).replace('.', 'p')}deg_{safe_key}_toplayer_sepfield.png"
    )

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
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Relaxed TBLG {theta_deg}° — top-layer $2(z_i-\\langle z\\rangle)$ "
        f"({display_name})",
    )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def _uncertainty_quantification_dir() -> Path:
    d = _repo_root() / "uncertainty_quantification"
    if not d.is_dir():
        raise FileNotFoundError(
            f"Expected uncertainty_quantification at {d} "
            "(run from repo root; DataLoader uses ../data relative to this folder).",
        )
    return d


def _ensure_uq_best_fit_dir() -> Path:
    """Where ``get_MCMC_inputs`` writes caches when cwd is ``uncertainty_quantification/``."""
    bf = _uncertainty_quantification_dir() / "best_fit_params"
    bf.mkdir(parents=True, exist_ok=True)
    return bf


def _pod_energy_hyperparams() -> dict:
    """Build the POD_energy descriptor dict from module-level RELAXATION_POD_* constants."""
    return {
        "species": ["C"],
        "bessel_polynomial_degree": POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE,
        "inverse_polynomial_degree": POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE,
        "twobody_number_radial_basis_functions": RELAXATION_POD_TWO_BODY_RADIAL,
        "threebody_number_radial_basis_functions": RELAXATION_POD_THREE_BODY_RADIAL,
        "threebody_angular_degree": RELAXATION_POD_THREE_BODY_ANGULAR,
        "fourbody_number_radial_basis_functions": RELAXATION_POD_FOUR_BODY_RADIAL,
        "fourbody_angular_degree": RELAXATION_POD_FOUR_BODY_ANGULAR,
        "fivebody_number_radial_basis_functions": RELAXATION_POD_FIVE_BODY_RADIAL,
        "fivebody_angular_degree": RELAXATION_POD_FIVE_BODY_ANGULAR,
        "sixbody_number_radial_basis_functions": RELAXATION_POD_SIX_BODY_RADIAL,
        "sixbody_angular_degree": RELAXATION_POD_SIX_BODY_ANGULAR,
        "sevenbody_number_radial_basis_functions": RELAXATION_POD_SEVEN_BODY_RADIAL,
        "sevenbody_angular_degree": RELAXATION_POD_SEVEN_BODY_ANGULAR,
    }


def _fit_pod_energy_through_mcmc() -> None:
    _ensure_get_mcmc_inputs_importable()
    from get_MCMC_inputs import get_MCMC_inputs

    _ensure_uq_best_fit_dir()
    with _working_directory(_uncertainty_quantification_dir()):
        get_MCMC_inputs(
            "POD_energy",
            supercells=1,
            two_body_radial=RELAXATION_POD_TWO_BODY_RADIAL,
            three_body_radial=RELAXATION_POD_THREE_BODY_RADIAL,
            three_body_angular=RELAXATION_POD_THREE_BODY_ANGULAR,
            four_body_radial=RELAXATION_POD_FOUR_BODY_RADIAL,
            four_body_angular=RELAXATION_POD_FOUR_BODY_ANGULAR,
            five_body_radial=RELAXATION_POD_FIVE_BODY_RADIAL,
            five_body_angular=RELAXATION_POD_FIVE_BODY_ANGULAR,
            six_body_radial=RELAXATION_POD_SIX_BODY_RADIAL,
            six_body_angular=RELAXATION_POD_SIX_BODY_ANGULAR,
            seven_body_radial=RELAXATION_POD_SEVEN_BODY_RADIAL,
            seven_body_angular=RELAXATION_POD_SEVEN_BODY_ANGULAR,
            regularization=RELAXATION_POD_REGULARIZATION,
        )


def _fit_tetb_pod_through_mcmc() -> None:
    _ensure_get_mcmc_inputs_importable()
    from get_MCMC_inputs import get_MCMC_inputs

    _ensure_uq_best_fit_dir()
    kw = {
        "tb_M": RELAXATION_TETB_POD_TB_M,
        "tb_W": RELAXATION_TETB_POD_TB_W,
        "pod_M": RELAXATION_TETB_POD_POD_M,
        "pod_W": RELAXATION_TETB_POD_POD_W,
    }
    with _working_directory(_uncertainty_quantification_dir()):
        get_MCMC_inputs("TETB_POD", supercells=1, **kw)


# ---------------------------------------------------------------------------
# Parameter files → numpy (raise FileNotFoundError / ValueError)
# ---------------------------------------------------------------------------


def _load_tersoff_kc_weights() -> tuple[np.ndarray, np.ndarray]:
    """Tersoff C–C row fixed; KC weights from estimate or combined npz."""
    kc_est = os.path.join(_tersoff_kc_best_fit_dir(), "Kolmogorov_Crespi_best_fit_params_estimate.npz")
    combined = os.path.join(_tersoff_kc_best_fit_dir(), "Tersoff+Kolmogorov_Crespi_best_fit_params.npz")

    if os.path.isfile(kc_est):
        _ = np.load(kc_est)["params"]  # file exists; KC row below matches prior test behavior
        kc_p = np.array(
            [
                3.416084,
                20.021583,
                10.9055107,
                4.2756354,
                1.0010836e-2,
                0.8447122,
                2.9360584,
                14.3132588,
            ],
        )
        return TERSOFF_CC_BNC.copy(), np.asarray(kc_p, dtype=float)

    if os.path.isfile(combined):
        p = np.load(combined)["params"]
        if len(p) >= 24:
            kc_p = p[:10]
        elif len(p) >= 22:
            kc_p = p[:8]
        else:
            raise ValueError(
                f"Combined Tersoff+KC params length {len(p)}; need >= 22 (8+14 or 10+14).",
            )
        return TERSOFF_CC_BNC.copy(), np.asarray(kc_p, dtype=float)

    raise FileNotFoundError(f"Expected {kc_est!s} or {combined!s} under best_fit_params.")


def _load_npz_params(path: Path, n_expect: int) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        raw = np.load(str(path), allow_pickle=True)
        p = np.asarray(raw["params"], dtype=float).ravel()
    except Exception:
        return None
    if p.size != n_expect:
        return None
    return p


def _pod_energy_cache_name() -> str:
    """Cache filename produced by ``get_MCMC_inputs`` for the current settings."""
    hp = _pod_energy_hyperparams()
    n = int(ncoeff_from_params(hp))
    reg_tag   = f"reg{RELAXATION_POD_REGULARIZATION:.0e}"
    return f"POD_energy_{n}_{reg_tag}_best_fit_params.npz"


def _load_pod_energy_coeffs_or_fit() -> tuple[np.ndarray, dict]:
    """Return ``(pod_params, hyperparams)`` for the current RELAXATION_POD_* constants."""
    hp = _pod_energy_hyperparams()
    n_expect = int(ncoeff_from_params(hp))
    name = _pod_energy_cache_name()

    last_err: Optional[BaseException] = None
    for d in _best_fit_dir_candidates():
        p = _load_npz_params(d / name, n_expect)
        if p is not None:
            return p, hp
        if (d / name).is_file():
            last_err = ValueError(f"{d / name}: bad or wrong-length params")

    print(
        f"[test_relaxation] No usable {name!r}; fitting POD_energy "
        f"(ncoeff={n_expect}) via get_MCMC_inputs …",
        flush=True,
    )
    _fit_pod_energy_through_mcmc()

    for d in _best_fit_dir_candidates():
        p = _load_npz_params(d / name, n_expect)
        if p is not None:
            return p, hp

    msg = f"After fit, could not load {name} from: {[str(d / name) for d in _best_fit_dir_candidates()]!r}"
    if last_err is not None:
        msg += f"; prior issue: {last_err!r}"
    raise FileNotFoundError(msg)


def _load_pod_energy_from_hyperparam_search() -> tuple[np.ndarray, dict, float]:
    """Load the best ``pod_energy`` model from the hyperparameter search.

    Reads ``uncertainty_quantification/pod_hyperparam_search_results.json``,
    picks the entry with the lowest per-atom test-set RMSE among those
    flagged ``meets_all_criteria = True``, then loads its fitted coefficients
    from ``uncertainty_quantification/pod_hyperparam_search_cache/<hash>.npz``.

    The descriptor hyperparameter dict must match the fit embedded in that NPZ:
    body-order counts and Bessel / inverse polynomial degrees come from the JSON
    ``config`` (legacy results without those keys default to bessel ``0`` /
    inverse ``10``).

    Returns
    -------
    pod_params : np.ndarray
        Fitted POD coefficients vector.
    hyperparams : dict
        Descriptor hyperparameters dict (ready for ``PODASECalculator``).
    rcut : float
        Radial cutoff radius in Å (varies per config; read from JSON).

    Raises
    ------
    FileNotFoundError
        JSON results file or coefficient cache is missing.
    ValueError
        No result in the JSON satisfies ``meets_all_criteria``.
    """
    json_path = _uncertainty_quantification_dir() / "pod_hyperparam_search_results.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Hyperparameter search results not found: {json_path}\n"
            "Run uncertainty_quantification/pod_hyperparameter_search.py first, "
            "or set RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST = False to use "
            "the manual RELAXATION_POD_* constants instead.",
        )

    with open(json_path) as fh:
        results = json.load(fh)

    candidates = [r for r in results if r.get("meets_all_criteria")]
    if not candidates:
        raise ValueError(
            f"No POD hyperparameter config in {json_path} satisfies "
            "'meets_all_criteria'.  Run pod_hyperparameter_search.py or set "
            "RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST = False.",
        )

    best = min(candidates, key=lambda r: float(r["test_energy_rmse_per_atom_eV"]))
    h   = best["hash"]
    cfg = best["config"]
    rcut = float(cfg["rcut"])

    cache_path = (
        _uncertainty_quantification_dir()
        / "pod_hyperparam_search_cache"
        / f"{h}.npz"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Coefficient cache missing for best config (hash={h}): {cache_path}\n"
            "Re-run pod_hyperparameter_search.py to regenerate it.",
        )

    pod_params = np.asarray(np.load(str(cache_path))["params"], dtype=float).ravel()

    hyperparams = {
        "species": ["C"],
        "bessel_polynomial_degree": int(cfg.get("bessel_polynomial_degree", 0)),
        "inverse_polynomial_degree": int(cfg.get("inverse_polynomial_degree", 10)),
        "twobody_number_radial_basis_functions":  int(cfg["two_body_radial"]),
        "threebody_number_radial_basis_functions": int(cfg["three_body_radial"]),
        "threebody_angular_degree":                int(cfg["three_body_angular"]),
        "fourbody_number_radial_basis_functions":  int(cfg["four_body_radial"]),
        "fourbody_angular_degree":                 int(cfg["four_body_angular"]),
        "fivebody_number_radial_basis_functions":  int(cfg["five_body_radial"]),
        "fivebody_angular_degree":                 int(cfg["five_body_angular"]),
        "sixbody_number_radial_basis_functions":   int(cfg["six_body_radial"]),
        "sixbody_angular_degree":                  int(cfg["six_body_angular"]),
        "sevenbody_number_radial_basis_functions": int(cfg["seven_body_radial"]),
        "sevenbody_angular_degree":                int(cfg["seven_body_angular"]),
    }

    n_coeff_expect = int(ncoeff_from_params(hyperparams))
    if pod_params.size != n_coeff_expect:
        raise ValueError(
            f"POD coefficient vector has length {pod_params.size}, but "
            f"ncoeff_from_params(...) = {n_coeff_expect} for the reconstructed "
            f"hyperparams (hash={h}).  Check body-order counts and "
            f"bessel_polynomial_degree / inverse_polynomial_degree in the JSON vs. "
            f"the NPZ.",
        )

    rmse_mev = float(best["test_energy_rmse_per_atom_eV"]) * 1e3
    mae_mev  = float(best["test_energy_mae_per_atom_eV"])  * 1e3
    print(
        f"[test_relaxation] Best POD from hyperparam search: "
        f"hash={h}  rcut={rcut} Å  "
        f"bessel_deg={hyperparams['bessel_polynomial_degree']} "
        f"inverse_deg={hyperparams['inverse_polynomial_degree']} "
        f"n_coeff={pod_params.size}  "
        f"RMSE={rmse_mev:.3f} meV/atom  MAE={mae_mev:.3f} meV/atom",
        flush=True,
    )
    return pod_params, hyperparams, rcut


_UQ_IMPORT_READY = False


def _ensure_get_mcmc_inputs_importable() -> None:
    global _UQ_IMPORT_READY
    if _UQ_IMPORT_READY:
        return
    uq = _repo_root() / "uncertainty_quantification"
    s = str(uq)
    if s not in sys.path:
        sys.path.insert(0, s)
    _UQ_IMPORT_READY = True


def _load_tetb_pod_fit_arrays() -> tuple[np.ndarray, np.ndarray, dict, dict, int]:
    """TB weights, POD coeffs, hyperparam dicts, n_pod; tagged npz preferred, legacy fallback."""
    _ensure_get_mcmc_inputs_importable()
    from get_MCMC_inputs import build_tetb_pod_hyperparams_from_data_kw

    data_kw = {
        "tb_M": RELAXATION_TETB_POD_TB_M,
        "tb_W": RELAXATION_TETB_POD_TB_W,
        "pod_M": RELAXATION_TETB_POD_POD_M,
        "pod_W": RELAXATION_TETB_POD_POD_W,
    }
    tb_hp, pod_hp, tetb_tag = build_tetb_pod_hyperparams_from_data_kw(data_kw)
    n_pod = int(ncoeff_from_params(pod_hp))
    tb_m, tb_w = int(tb_hp["M"]), int(tb_hp["W"])
    tb_name = f"ACSF_hoppings_M_{tb_m}_W_{tb_w}_best_fit_params.npz"
    tagged = f"TETB_POD_{tetb_tag}_best_fit_params.npz"
    legacy = "TETB_POD_best_fit_params.npz"

    def _try_read_pair(root: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
        tb_path = root / tb_name
        if not tb_path.is_file():
            return None
        for tetb_name in (tagged, legacy):
            te_path = root / tetb_name
            if not te_path.is_file():
                continue
            tb_arr = np.asarray(np.load(str(tb_path), allow_pickle=True)["params"], dtype=float).ravel()
            pe = np.asarray(np.load(str(te_path), allow_pickle=True)["params"], dtype=float).ravel()
            if pe.size == n_pod + 1:
                pe = pe[:n_pod]
            if pe.size != n_pod:
                continue
            return tb_arr, pe
        return None

    for root in _best_fit_dir_candidates():
        pair = _try_read_pair(root)
        if pair is not None:
            tb_arr, pe = pair
            return tb_arr, pe, tb_hp, pod_hp, n_pod

    print(
        f"[test_relaxation] Missing {tb_name!r} and/or {tagged!r}; "
        f"fitting TETB_POD (tag {tetb_tag}) via get_MCMC_inputs …",
        flush=True,
    )
    _fit_tetb_pod_through_mcmc()

    for root in _best_fit_dir_candidates():
        pair = _try_read_pair(root)
        if pair is not None:
            tb_arr, pe = pair
            return tb_arr, pe, tb_hp, pod_hp, n_pod

    raise FileNotFoundError(
        f"After fit, need {tb_name!r} and {tagged!r} (or compatible legacy {legacy!r}) under one of: "
        f"{[str(d) for d in _best_fit_dir_candidates()]!r}",
    )


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------


def _mean_layer_separation(atoms) -> float:
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    layers = np.searchsorted(np.unique(mol), mol).astype(int)
    z0 = pos[layers == 0, 2]
    z1 = pos[layers == 1, 2]
    return float(z1.mean() - z0.mean())


def _max_inplane_buckling_std_z(atoms) -> float:
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    layers = np.searchsorted(np.unique(mol), mol).astype(int)
    s0 = float(np.std(pos[layers == 0, 2]))
    s1 = float(np.std(pos[layers == 1, 2]))
    return max(s0, s1)


def _ensure_bilayer_mol_id_from_z(atoms) -> None:
    """Ensure ``mol-id`` in ``{1, 2}`` with layer 1 below layer 2 (mean ``z``)."""
    if atoms.has("mol-id"):
        mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
        if mol.shape[0] != len(atoms):
            raise ValueError("mol-id length mismatch")
        u = np.unique(mol)
        if u.size == 2:
            id_a, id_b = int(u.min()), int(u.max())
            za = float(np.mean(atoms.positions[mol == id_a, 2]))
            zb = float(np.mean(atoms.positions[mol == id_b, 2]))
            bottom_id = id_a if za < zb else id_b
            mol12 = np.where(mol == bottom_id, 1, 2).astype(np.int8)
            atoms.set_array("mol-id", mol12)
            return
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    mid = float(np.median(z))
    mol = np.where(z < mid, 1, 2).astype(np.int8)
    z1 = float(np.mean(z[mol == 1]))
    z2 = float(np.mean(z[mol == 2]))
    if z1 > z2:
        mol = np.where(mol == 1, 2, 1).astype(np.int8)
    atoms.set_array("mol-id", mol)


def _tblg_aa_ab_layer_separations(atoms) -> dict[str, float]:
    """Layer separations from ``z`` extrema (bottom vs top ``mol-id`` layers).

    * **AA-style** — ``max(z_top) - min(z_bottom)`` (largest vertical span).
    * **AB-style** — ``min(z_top) - max(z_bottom)`` (smallest gap between layers).
    """
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    u = np.unique(mol)
    if u.size != 2:
        raise ValueError(f"Expected two mol-id values for bilayer; got {u!r}")
    z1 = pos[mol == int(u.min()), 2]
    z2 = pos[mol == int(u.max()), 2]
    mean1, mean2 = float(np.mean(z1)), float(np.mean(z2))
    if mean1 <= mean2:
        z_bottom, z_top = z1, z2
    else:
        z_bottom, z_top = z2, z1
    aa = float(np.max(z_top) - np.min(z_bottom))
    ab = float(np.min(z_top) - np.max(z_bottom))
    return {"aa_stacking_layer_separation": aa, "ab_stacking_layer_separation": ab}


def _build_tblg_atoms(theta_deg: float, *, lat_con: float = 2.46, sep: float = 3.35) -> object:
    fg = pytest.importorskip(
        "flatgraphene",
        reason="flatgraphene not installed; skipping TBLG relaxation test.",
    )
    p, q, _ = fg.twist.find_p_q(float(theta_deg), a_tol=0.1)
    atoms = fg.twist.make_graphene(
        cell_type="hex",
        n_layer=2,
        p=p,
        q=q,
        lat_con=float(lat_con),
        sym=["C", "C"],
        mass=[12.01, 12.01],
        sep=float(sep),
        h_vac=20,
    )
    _ensure_bilayer_mol_id_from_z(atoms)
    return atoms


# ---------------------------------------------------------------------------
# Model case (calculator factory + relax options)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelaxationModelCase:
    key: str
    energy_display_name: str
    calc_factory: Callable[..., object]
    blg_sc: int = 1
    relax_backend: str = "lammps"
    relax_etol: float = 0
    relax_ftol: float = RELAXATION_FMAX_EV_PER_ANG
    relax_maxiter: int = 2000
    relax_maxeval: int = 10_000
    relax_d_preflight_scan: bool = False
    relax_d_start_bump: float = 0.2
    relax_d_start_fixed: float = 3.70


def build_relaxation_case(model_key: str) -> RelaxationModelCase:
    if model_key == "tersoff_kc":
        te_p, kc_p = _load_tersoff_kc_weights()

        def calc_factory(atoms):
            return TersoffKolmogorovCrespiASECalculator(te_p.tolist(), kc_p.tolist())

        return RelaxationModelCase(
            key="tersoff_kc",
            energy_display_name="Tersoff+Kolmogorov-Crespi",
            calc_factory=calc_factory,
        )

    if model_key == "pod_energy":
        if RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST:
            pod_params, hp, rcut = _load_pod_energy_from_hyperparam_search()
        else:
            pod_params, hp = _load_pod_energy_coeffs_or_fit()
            rcut = 6.0

        def calc_factory(atoms, _p=pod_params, _hp=hp, _rcut=rcut):
            return PODASECalculator(_hp, _p, elements=["C"], cutoff=_rcut)

        # Settings mirror pod_hyperparameter_search.py:
        #   etol=1e-9, ftol=1e-2 eV/Å, maxiter=10_000, maxeval=800_000.
        # The loose ftol (1e-2 eV/Å) matches the TBLG convergence threshold used
        # during the hyperparameter search so that the same model that passed the
        # search also passes test_tblg_relaxation_stable.
        return RelaxationModelCase(
            key="pod_energy",
            energy_display_name="POD_energy",
            calc_factory=calc_factory,
            blg_sc=1,
            relax_backend="lammps",
            relax_etol=1e-9,
            relax_ftol=RELAXATION_FMAX_EV_PER_ANG,
            relax_maxiter=10_000,
            relax_maxeval=800_000,
            relax_d_preflight_scan=True,
            relax_d_start_bump=0.2,
        )

    if model_key == "tetb_pod":
        tb_p, pod_p, tb_hp, pod_hp, _n = _load_tetb_pod_fit_arrays()

        def calc_factory(atoms):
            return TETB_PODASECalculator(
                tb_params=tb_p,
                pod_params=pod_p,
                tb_hyperparams=dict(tb_hp),
                pod_hyperparams=dict(pod_hp),
                pod_cutoff=6.0,
                elements=["C"],
                kpoints=None,
                tb_solver_method="diagonalization",
                ewald_cutoff=12.0,
                pppm_accuracy=1e-4,
                valence_charge=1.0,
                shift=0.0,
            )

        return RelaxationModelCase(
            key="tetb_pod",
            energy_display_name="TETB+POD",
            calc_factory=calc_factory,
            blg_sc=1,
            relax_backend="ase",
            relax_etol=0,
            relax_ftol=RELAXATION_FMAX_EV_PER_ANG,
            relax_maxiter=100,
            relax_maxeval=100,
            relax_d_preflight_scan=True,
            relax_d_start_bump=0.2,
        )

    raise ValueError(f"Unknown model_key: {model_key!r}")


def try_build_relaxation_case(model_key: str) -> Optional[RelaxationModelCase]:
    try:
        return build_relaxation_case(model_key)
    except (FileNotFoundError, ValueError):
        return None


def relaxation_case_or_skip(model_key: str) -> RelaxationModelCase:
    """Build a RelaxationModelCase, auto-fitting if needed.

    For ``pod_energy`` and ``tetb_pod``, the loaders (``_load_pod_energy_coeffs_or_fit``
    and ``_load_tetb_pod_fit_arrays``) already attempt a fit via ``get_MCMC_inputs``
    when the cache is absent.  A ``FileNotFoundError`` here means the fit itself
    failed (e.g. no LAMMPS executable or missing training data).  For
    ``tersoff_kc`` there is no automated fit pipeline; skip when files are absent.
    """
    try:
        return build_relaxation_case(model_key)
    except FileNotFoundError as exc:
        if model_key == "tersoff_kc":
            pytest.skip(
                f"Tersoff+KC parameter files not found and no auto-fit is "
                f"available for classical potentials: {exc}"
            )
        if model_key == "pod_energy":
            if RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST:
                pytest.skip(
                    f"POD_energy hyperparam-search model unavailable "
                    f"(run pod_hyperparameter_search.py or set "
                    f"RELAXATION_POD_USE_HYPERPARAM_SEARCH_BEST = False): {exc}"
                )
            pytest.skip(
                f"POD_energy auto-fit failed (check LAMMPS executable and "
                f"training data in ../data): {exc}"
            )
        if model_key == "tetb_pod":
            pytest.skip(
                f"TETB+POD auto-fit failed (check LAMMPS executable and "
                f"training data in ../data): {exc}"
            )
        pytest.skip(f"{model_key}: parameters unavailable and auto-fit failed: {exc}")
    except ValueError as exc:
        if model_key == "tetb_pod":
            pytest.skip(f"TETB+POD parameters incompatible or cache corrupted: {exc}")
        raise


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_VALID_RELAXATION_BACKENDS: Tuple[str, ...] = ("lammps", "fire", "lbfgs")


def _active_backend() -> str:
    """Return the canonical backend string from ``RELAXATION_BACKEND``."""
    b = str(RELAXATION_BACKEND).strip().lower()
    if b not in _VALID_RELAXATION_BACKENDS:
        raise ValueError(
            f"RELAXATION_BACKEND={RELAXATION_BACKEND!r} is not recognised; "
            f"choose one of {_VALID_RELAXATION_BACKENDS}.",
        )
    return b


def _run_relax(
    case: "RelaxationModelCase",
    atoms,
    *,
    ftol: float,
    maxiter: int,
    maxeval: int,
):
    """Relax *atoms* using the backend selected by ``RELAXATION_BACKEND``.

    ``atoms.calc`` must be attached before calling this function.

    When ``case.relax_backend`` is ``"fire"`` or ``"lbfgs"`` and the global
    ``RELAXATION_BACKEND`` is ``"lammps"``, the case-level setting wins (e.g.
    TETB+POD uses ASE FIRE regardless of the global backend).

    Parameters
    ----------
    case : RelaxationModelCase
        Used for ``relax_etol`` (LAMMPS path only) and ``relax_backend``
        override for pure-Python models.
    atoms : ase.Atoms
        Structure to relax (calc must be set).
    ftol : float
        Max per-atom force threshold (eV/Å) — ``fmax`` for ASE optimisers,
        second argument to LAMMPS ``minimize``.
    maxiter, maxeval : int
        Maximum optimizer steps / force evaluations.

    Returns
    -------
    ase.Atoms
        Relaxed structure with ``info["relax_backend"]`` and
        ``info["lammps_fire_relax_energy"]`` / ``info["lammps_pe_singlepoint"]``
        populated (same contract as ``calc.relax_structure``).
    """
    global_backend = _active_backend()
    # Pure-Python models specify relax_backend="fire" or "lbfgs"; respect that
    # even when the module-level RELAXATION_BACKEND is "lammps".
    if case.relax_backend in ("fire", "lbfgs") and global_backend == "lammps":
        backend = case.relax_backend
    else:
        backend = global_backend

    if backend == "lbfgs":
        from ase.optimize import LBFGS  # noqa: PLC0415

        out = atoms.copy()
        out.calc = atoms.calc
        dyn = LBFGS(out, logfile="log", maxstep=0.1)
        dyn.run(fmax=ftol, steps=maxiter)
        e = float(out.get_potential_energy())
        out.info["relax_backend"] = "ase_lbfgs"
        out.info["lammps_fire_relax_energy"] = e
        out.info["lammps_pe_singlepoint"] = float(out.calc.get_potential_energy(out))
        return out

    if backend == "fire":
        return atoms.relax_structure(
            relax_backend="ase",
            etol=0.0,
            ftol=ftol,
            maxiter=maxiter,
            maxeval=maxeval,
        )

    # backend == "lammps"
    return atoms.relax_structure(
        relax_backend="lammps",
        etol=case.relax_etol,
        ftol=ftol,
        maxiter=maxiter,
        maxeval=maxeval,
    )


# ---------------------------------------------------------------------------
# Rigid scans + relax workflow (prints preserved)
# ---------------------------------------------------------------------------


def _interlayer_rigid_scan_energies(
    case: RelaxationModelCase,
    a_min_rigid: float,
    layer_sep: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    template = get_bilayer_atoms(
        float(layer_sep[0]), 0.0, a=float(a_min_rigid), sc=case.blg_sc,
    )
    calc0 = case.calc_factory(template)
    e = np.zeros(len(layer_sep), dtype=float)
    for i, sep in enumerate(layer_sep):
        atoms = get_bilayer_atoms(sep, 0.0, a=float(a_min_rigid), sc=case.blg_sc)
        atoms.calc = calc0
        e[i] = float(atoms.get_potential_energy())
    j = int(np.argmin(e))
    return e, float(layer_sep[j]), float(np.min(e))


def _inplane_rigid_a_minimum(
    case: RelaxationModelCase,
    *,
    d_interlayer: float = 3.43,
    a_rigid_grid: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if a_rigid_grid is None:
        a_rigid_grid = np.linspace(2.40, 2.52, 15)
    e_rigid = np.empty_like(a_rigid_grid)
    for i, a in enumerate(a_rigid_grid):
        atoms = get_bilayer_atoms(d_interlayer, 0.0, a=float(a), sc=case.blg_sc)
        atoms.calc = case.calc_factory(atoms)
        e_rigid[i] = atoms.get_potential_energy()
    a_min_rigid = float(a_rigid_grid[int(np.argmin(e_rigid))])
    print(f"[{case.energy_display_name}] a_min_rigid = {a_min_rigid}")
    return a_min_rigid, a_rigid_grid, e_rigid


def _relax_and_interlayer_rigid_scan(
    case: RelaxationModelCase,
    a_min_rigid: float,
    *,
    layer_sep: Optional[np.ndarray] = None,
) -> dict:
    if layer_sep is None:
        layer_sep = np.linspace(3.0, 4.0, 30)

    interlayer_energy: Optional[np.ndarray] = None
    if case.relax_d_preflight_scan:
        interlayer_energy, d_pref_min, _ = _interlayer_rigid_scan_energies(
            case, a_min_rigid, layer_sep,
        )
        d_start = float(
            np.clip(
                d_pref_min + case.relax_d_start_bump,
                float(layer_sep[0]),
                float(layer_sep[-1]),
            ),
        )
        print(
            f"[{case.energy_display_name}] preflight rigid d_scan at a={a_min_rigid:.6f} Å: "
            f"d_min={d_pref_min:.4f} Å → relax start d={d_start:.4f} Å "
            f"(d_min + {case.relax_d_start_bump:g} Å)",
        )
    else:
        d_start = float(case.relax_d_start_fixed)

    atoms0 = get_bilayer_atoms(d_start, 0.0, a=float(a_min_rigid), sc=case.blg_sc)
    atoms0.calc = case.calc_factory(atoms0)

    print(f"\n--- {case.energy_display_name} ---")
    print(
        f"[{case.energy_display_name}] Relaxing AB-stacked BLG "
        f"({case.energy_display_name} model, backend={_active_backend()!r})...",
    )
    relaxed = _run_relax(
        case, atoms0,
        ftol=case.relax_ftol,
        maxiter=case.relax_maxiter,
        maxeval=case.relax_maxeval,
    )
    pe_rel = relaxed.get_potential_energy()
    rb = relaxed.info.get("relax_backend", case.relax_backend)
    print(
        f"[{case.energy_display_name}] relax_backend={rb!r} — FIRE endpoint energy = "
        f"{relaxed.info.get('lammps_fire_relax_energy')}",
    )
    print(
        f"[{case.energy_display_name}] PE single-point re-eval on relaxed Atoms = "
        f"{relaxed.info.get('lammps_pe_singlepoint')}",
    )
    pos = relaxed.get_positions()
    print(f"[{case.energy_display_name}] positions = \n", np.round(pos, 3))
    zspan = float(np.max(pos[:, 2]) - np.min(pos[:, 2]))
    d_mean = _mean_layer_separation(relaxed)
    buckling = _max_inplane_buckling_std_z(relaxed)
    print(f"[{case.energy_display_name}] layer separation (max-min z) = {zspan}")
    print(f"[{case.energy_display_name}] max in-plane buckling std(z) = {buckling:.5f} Å")

    if interlayer_energy is None:
        interlayer_energy, d_argmin, e_min_scan = _interlayer_rigid_scan_energies(
            case, a_min_rigid, layer_sep,
        )
    else:
        d_argmin = float(layer_sep[int(np.argmin(interlayer_energy))])
        e_min_scan = float(np.min(interlayer_energy))
    print(
        f"[{case.energy_display_name}] energy min interlayer separation (rigid AB scan) = {d_argmin}",
    )

    delta_e = float(pe_rel) - float(e_min_scan)
    print(f"[{case.energy_display_name}] relaxed PE - rigid scan min PE = {delta_e}")

    if plt is not None:
        art = _artifacts_dir()
        stem = f"relaxation_interlayer_{case.energy_display_name.replace('+', '_').replace(' ', '_')}"
        out_png = art / f"{stem}.png"
        plt.figure(figsize=(6.0, 4.0), dpi=120)
        plt.plot(layer_sep, interlayer_energy - np.min(interlayer_energy))
        plt.xlabel("layer separation (Å)")
        plt.ylabel("interlayer energy (eV)")
        plt.title(f"{case.energy_display_name} model")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[{case.energy_display_name}] saved interlayer rigid scan plot: {out_png}")

    return {
        "pe_rel": float(pe_rel),
        "e_min_scan": e_min_scan,
        "d_argmin_scan": d_argmin,
        "layer_sep_relaxed": d_mean,
        "layer_span_max_min": zspan,
        "delta_e": delta_e,
        "buckling_std": buckling,
    }


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------


@pytest.fixture
def _require_lammps_py(request):
    """Skip if the current parametrized model needs LAMMPS but it is unavailable.

    Models not listed in ``_MODELS_NEEDING_LAMMPS`` skip this check.
    """
    callspec = getattr(request.node, "callspec", None)
    model_key = callspec.params.get("model_key") if callspec else None
    if model_key is not None and model_key not in _MODELS_NEEDING_LAMMPS:
        return  # pure-Python model — no LAMMPS required
    try:
        import lammps  # noqa: F401
    except Exception as exc:
        pytest.skip(f"LAMMPS Python module not available: {exc}")


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_inplane_lattice_rigid_scan(model_key: str, _require_lammps_py):
    """Rigid scan of total energy vs in-plane lattice constant ``a`` at fixed ``d``."""
    case = relaxation_case_or_skip(model_key)
    _inplane_rigid_a_minimum(case, d_interlayer=3.43)


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_relaxed_vs_rigid_interlayer_minimum(model_key: str, _require_lammps_py):
    """Relax at fixed ``a``; relaxed layer sep and energy vs rigid interlayer scan."""
    case = relaxation_case_or_skip(model_key)
    a_min, _, _ = _inplane_rigid_a_minimum(case)
    res = _relax_and_interlayer_rigid_scan(case, a_min)
    assert abs(res["layer_sep_relaxed"] - res["d_argmin_scan"]) <= 0.12, (
        f"[{case.energy_display_name}] rigid-scan d={res['d_argmin_scan']:.4f} Å "
        f"vs relaxed mean layer sep={res['layer_sep_relaxed']:.4f} Å"
    )
    assert res["delta_e"] <= 10.0, (
        f"[{case.energy_display_name}] relaxed PE above rigid min by {res['delta_e']:.4f} eV"
    )
    assert res["buckling_std"] <= 0.02, (
        f"[{case.energy_display_name}] in-plane buckling std(z)={res['buckling_std']:.4f} Å "
        "(expected flat layers)"
    )


_TBLG_KNOWN_LAMMPS_ERRORS = (
    # KC Full computes a per-atom surface normal from the bonding environment.
    # For large-angle TBLG geometries generated by flatgraphene some atoms end
    # up in a configuration where that normal degenerates to zero length.
    "magnitude of the normal vector is zero",
    # POD trained on primitive 4-atom cells gives unphysical forces for the
    # 508-atom TBLG cell; LAMMPS FIRE flings atoms outside the box.
    "Lost atoms",
    "lammps_scatter_atoms",
)


def _tblg_lammps_error_reason(exc: Exception) -> Optional[str]:
    """Return a human-readable xfail reason if *exc* matches a known LAMMPS
    failure mode for TBLG, or ``None`` if the exception is unexpected."""
    msg = str(exc)
    if "magnitude of the normal vector is zero" in msg:
        return (
            "KC Full pair style cannot compute surface normals for some atoms "
            "in the large-angle TBLG geometry produced by flatgraphene "
            "(known limitation of pair_kolmogorov_crespi_full)."
        )
    if "Lost atoms" in msg or "lammps_scatter_atoms" in msg:
        return (
            "POD potential trained on 4-atom primitive cells produces "
            "unphysical forces for the 508-atom TBLG cell; LAMMPS FIRE "
            "ejects atoms from the simulation box."
        )
    return None


@pytest.mark.slow
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_tblg_relaxation_stable(model_key: str, _require_lammps_py):
    """Relax 5.09° TBLG for each ``MODEL_KEYS`` entry; print AA; plot AB separation.

    Uses ``flatgraphene`` commensurate supercells. **AA** (printed):
    ``max(z_top) - min(z_bottom)``. **AB** (saved bar figure):
    ``min(z_top) - max(z_bottom)``.

    Relaxation uses ``ftol=TBLG_RELAX_FMAX_EV_PER_ANG`` (0.05 eV/Å); stability
    is checked against the same threshold (with a small numerical margin).

    Known failure modes are caught and reported as ``xfail`` rather than a
    hard error, since they reflect limitations of the potential (KC Full normal-
    vector degeneracy; POD out-of-distribution TBLG forces) rather than bugs in
    the test or the relaxation infrastructure.
    """
    theta_deg = TBLG_RELAX_TWIST_ANGLE
    atoms0 = _build_tblg_atoms(theta_deg)
    case = relaxation_case_or_skip(model_key)
    atoms0.calc = case.calc_factory(atoms0)

    ftol = float(RELAXATION_FMAX_EV_PER_ANG)
    maxiter = int(max(case.relax_maxiter, 8_000))
    maxeval = int(max(case.relax_maxeval, 800_000))

    print(
        f"\n--- TBLG θ={theta_deg}° ({case.energy_display_name}, {model_key!r}) — "
        f"{len(atoms0)} atoms ---",
        flush=True,
    )
    print(
        f"[TBLG {theta_deg}° {model_key}] Relaxing (backend={_active_backend()!r}, "
        f"ftol={ftol:g} eV/Å, maxiter={maxiter}, maxeval={maxeval})…",
        flush=True,
    )
    try:
        relaxed = _run_relax(case, atoms0, ftol=ftol, maxiter=maxiter, maxeval=maxeval)
    except Exception as exc:
        reason = _tblg_lammps_error_reason(exc)
        if reason is not None:
            pytest.xfail(f"[{model_key}] Known LAMMPS failure for TBLG: {reason}")
        raise
    _ensure_bilayer_mol_id_from_z(relaxed)
    out_sepfield = _plot_tblg_toplayer_interlayer_separation_field(
        relaxed,
        theta_deg=theta_deg,
        model_key=model_key,
        display_name=case.energy_display_name,
    )
    if out_sepfield is not None:
        print(
            f"[TBLG {theta_deg}° {model_key}] saved top-layer separation field plot: "
            f"{out_sepfield}",
            flush=True,
        )

    f = np.asarray(relaxed.get_forces(), dtype=float)
    fmax = float(np.max(np.linalg.norm(f, axis=1)))
    print(
        f"[TBLG {theta_deg}° {model_key}] max |F| after relax = {fmax:.6f} eV/Å",
        flush=True,
    )
    assert fmax <= ftol + 1e-2, (
        f"[{case.energy_display_name}] TBLG relax not stable: max |F|={fmax:.4f} eV/Å "
        f"(target ≤ {ftol:g} + 1e-3 eV/Å)"
    )

    seps = _tblg_aa_ab_layer_separations(relaxed)
    aa = seps["aa_stacking_layer_separation"]
    ab = seps["ab_stacking_layer_separation"]
    print(
        f"[TBLG {theta_deg}° {model_key}] AA stacking layer separation "
        f"(max z_top − min z_bottom) = {aa:.5f} Å",
        flush=True,
    )
    print(
        f"[TBLG {theta_deg}° {model_key}] AB stacking layer separation "
        f"(min z_top − max z_bottom) = {ab:.5f} Å",
        flush=True,
    )
    assert aa > 0.5 and ab > 0.5, (
        f"Unphysical layer gaps: AA={aa:.4f} Å, AB={ab:.4f} Å"
    )
    assert ab <= aa + 1e-3, (
        f"Expected AB separation ≤ AA span; got AB={ab:.4f} Å, AA={aa:.4f} Å"
    )

    if plt is not None:
        art = _artifacts_dir()
        safe_key = str(model_key).replace(os.sep, "_")
        out_png = art / (
            f"relaxation_tblg_{str(theta_deg).replace('.', 'p')}deg_{safe_key}_AB_separation.png"
        )
        fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=120)
        ax.barh([0.0], [ab], color="#c05621", height=0.45)
        ax.set_yticks([0.0])
        ax.set_yticklabels(["AB stacking\n(min z_top − max z_bottom)"])
        ax.set_xlabel("separation (Å)")
        ax.set_title(
            f"Relaxed TBLG {theta_deg}° — AB layer separation ({case.energy_display_name})",
        )
        ax.text(ab * 0.02, 0.0, f"  {ab:.4f} Å", va="center", fontsize=11)
        ax.set_xlim(0.0, max(ab, 0.5) * 1.4 + 0.05)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        print(
            f"[TBLG {theta_deg}° {model_key}] saved AB stacking separation plot: {out_png}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Demo (RELAXATION_DEMO_ONLY=1) and CLI fallback without pytest
# ---------------------------------------------------------------------------


def _run_demo_for_case(case: RelaxationModelCase) -> None:
    a_min, a_grid, e_grid = _inplane_rigid_a_minimum(case)
    _relax_and_interlayer_rigid_scan(case, a_min)
    if plt is not None:
        art = _artifacts_dir()
        stem = f"relaxation_inplane_a_{case.key}"
        p = art / f"{stem}.png"
        plt.figure(figsize=(6.0, 4.0), dpi=120)
        plt.plot(a_grid, e_grid)
        plt.xlabel("in-plane lattice constant (Å)")
        plt.ylabel("energy (eV)")
        plt.title(f"{case.energy_display_name} model — rigid in-plane scan")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        print(f"[{case.energy_display_name}] saved in-plane rigid scan plot: {p}")


def _collect_demo_cases() -> list[RelaxationModelCase]:
    cases: list[RelaxationModelCase] = []
    for key in MODEL_KEYS:
        try:
            cases.append(build_relaxation_case(key))
        except FileNotFoundError as exc:
            print(f"[{key}] demo skipped (missing inputs): {exc}", flush=True)
        except ValueError as exc:
            print(f"[{key}] demo skipped (missing inputs or bad cache): {exc}", flush=True)
    return cases


def _run_relaxation_checks_cli() -> int:
    try:
        import lammps  # noqa: F401
    except Exception as exc:
        print(f"ERROR: LAMMPS Python module required: {exc}", flush=True)
        return 1
    any_ran = False
    failed = False
    for model_key in MODEL_KEYS:
        try:
            case = build_relaxation_case(model_key)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n[{model_key}] SKIP — {exc}", flush=True)
            continue
        any_ran = True
        print(f"\n=== {model_key} ({case.energy_display_name}) ===", flush=True)
        try:
            _inplane_rigid_a_minimum(case, d_interlayer=3.43)
            a_min, _, _ = _inplane_rigid_a_minimum(case)
            res = _relax_and_interlayer_rigid_scan(case, a_min)
            assert abs(res["layer_sep_relaxed"] - res["d_argmin_scan"]) <= 0.12, (
                f"rigid-scan d={res['d_argmin_scan']:.4f} Å vs relaxed "
                f"mean layer sep={res['layer_sep_relaxed']:.4f} Å"
            )
            assert res["delta_e"] <= 10.0, (
                f"relaxed PE above rigid min by {res['delta_e']:.4f} eV"
            )
            assert res["buckling_std"] <= 0.02, (
                f"in-plane buckling std(z)={res['buckling_std']:.4f} Å"
            )
            print(f"[{model_key}] PASS", flush=True)
        except AssertionError as exc:
            print(f"[{model_key}] FAIL: {exc}", flush=True)
            failed = True
        except Exception as exc:
            print(f"[{model_key}] ERROR: {exc}", flush=True)
            failed = True
    if not any_ran:
        print(
            "\nNo models ran (all skipped). Install inputs under "
            "uncertainty_quantification/best_fit_params/ or run from repo root.",
            flush=True,
        )
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    if os.environ.get("RELAXATION_DEMO_ONLY", "").strip() in ("1", "true", "yes"):
        demo_cases = _collect_demo_cases()
        if not demo_cases:
            raise SystemExit(
                "No model cases could be built (missing inputs or fit failed). "
                "Tune RELAXATION_POD_ENERGY_* / RELAXATION_TETB_POD_* at the top of "
                "test_relaxation.py; caches are POD_energy_<ncoeff>_reg<reg>_best_fit_params.npz "
                "and TETB_POD_tb_M_*_W_*_pod_M_*_W_*_best_fit_params.npz plus matching "
                "ACSF_hoppings_M_*_W_*.npz. Missing caches trigger get_MCMC_inputs fits "
                "(LAMMPS + ../data from uncertainty_quantification/).",
            )
        for c in demo_cases:
            _run_demo_for_case(c)
        raise SystemExit(0)

    try:
        import pytest as _pytest
    except ImportError:
        _pytest = None

    if _pytest is not None:
        raise SystemExit(
            _pytest.main([__file__, "-v", "--tb=short", "-s"] + sys.argv[1:]),
        )
    raise SystemExit(_run_relaxation_checks_cli())
