"""
Calibration metrics for MCMC and subsampling ensembles.

Computes (per control grid point: temperature *T* for MCMC, fraction *p* for SubSamp):

- ``model_partition_function`` — mean Boltzmann weight ``mean(exp(-SSE / (T * T0)))``
- ``crps`` — continuous ranked probability score (ensemble predictive)
- ``nll`` — mean Gaussian negative log-likelihood using ensemble mean / std
- ``miscalibration_area`` — PIT-based QQ miscalibration (see ``miscalibration_area``; scaled ×4)

Saves arrays to ``calibration_metrics/`` and can plot:

- Mean prediction ± ensemble std vs reference (parity-style; **energy** is
  total energy divided by **each** configuration's atom count from
  ``xdata['energy']``, same convention as PES parity; no shift to minimum
  reference energy)
- Ensemble std vs per-point absolute error
- Metrics vs *T* or *p*, with optional overlay for several models on the same dataset type.

When a pickle contains ``ypred_samples_test`` and ``ydata_test``, those are used
for metrics and diagnostics (test split); otherwise the full/train fields are used.

For ``--diagnostics`` with MCMC, use ``--temperatures <T>`` (first value) to select the
nearest temperature on the current grid (works with ``--auto-discover``).  For
subsamp, the first ``--p-values`` entry selects the nearest ``p``.

Examples
--------
Discover MCMC ensemble files and compute hopping metrics (default output dir)::

    cd uncertainty_quantification
    python visualizations/plot_bayes_factor.py --calculate \\
        --models ACSF_hoppings_sk_M_12_W_1 --technique mcmc --auto-discover

Compare several hopping models on one figure (same *T* grid from saved files)::

    python plot_bayes_factor.py --plot-metrics --compare \\
        --models ACSF_hoppings_M_8_W_3 ACSF_hoppings_M_10_W_4 \\
        --technique mcmc --target hopping

Wildcard model selection (matches subfolders under ``ensembles/``)::

    python plot_bayes_factor.py --calculate --models 'POD_energy_POD_index*' \\
        --technique mcmc --target energy --auto-discover

Subsampling vs *p* (glob ``*_SubSamp_ensemble_p_*.pkl``)::

    python plot_bayes_factor.py --calculate --models ACSF_hoppings_M_10_W_4 \\
        --technique subsamp --target hopping --auto-discover
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent

# -----------------------------------------------------------------------------
# Core metrics (NumPy)
# -----------------------------------------------------------------------------


def _trapz_yx(y: np.ndarray, x: np.ndarray) -> float:
    """∫ y dx with consistent spacing (NumPy 1.x ``trapz`` / 2.x ``trapezoid``)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def pit_values(observations: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    """Probability integral transform: PIT *u* = fraction of ensemble ≤ observation per site."""
    observations = np.asarray(observations, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(1, -1)
    if observations.size != forecasts.shape[0]:
        raise ValueError(
            f"PIT: obs length {observations.size} vs forecasts rows {forecasts.shape[0]}"
        )
    return np.mean(forecasts <= observations[:, np.newaxis], axis=1)


def miscalibration_area(observations: np.ndarray, forecasts: np.ndarray) -> float:
    """
    Area between the QQ curve and the Uniform(0,1) diagonal.

    Uses sorted PIT values vs ``linspace(0,1,n)`` (same construction as
    ``qq_plot_example.py``), integrates ``|u_sorted - q|`` with respect to *q*,
    then multiplies by 4 so a well-calibrated Uniform PIT has scale comparable
    to a unit-square deviation budget.
    """
    observations = np.asarray(observations, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(1, -1)
    n = observations.size
    if n == 0 or forecasts.shape[0] != n:
        return float("nan")
    u = pit_values(observations, forecasts)
    u_sorted = np.sort(u)
    uniform_quantiles = np.linspace(0.0, 1.0, n)
    area = _trapz_yx(np.abs(u_sorted - uniform_quantiles), uniform_quantiles)
    return float(4.0 * area)

def crps_ensemble(observations: np.ndarray, forecasts: np.ndarray) -> float:
    """
    CRPS for ensemble forecasts.

    Parameters
    ----------
    observations : (n_obs,)
    forecasts : (n_obs, n_ensemble_members)
    """
    observations = np.asarray(observations, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(1, -1)
    if observations.shape[0] != forecasts.shape[0]:
        raise ValueError(
            f"CRPS: obs shape {observations.shape} vs forecasts {forecasts.shape}"
        )
    abs_diff = np.mean(np.abs(forecasts - observations[:, np.newaxis]), axis=1)
    m = forecasts.shape[1]
    ensemble_diff = (2.0 / (m * m)) * np.sum(
        (2 * np.arange(m) - (m - 1)) * np.sort(forecasts, axis=1), axis=1
    )
    return float(np.mean(abs_diff - 0.5 * ensemble_diff))


def mean_gaussian_nll(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean over observations of Gaussian NLL with diagonal variance."""
    y = np.asarray(y, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.maximum(np.asarray(sigma, dtype=float).ravel(), eps)
    if not (y.size == mu.size == sigma.size):
        raise ValueError("NLL: y, mu, sigma must broadcast to same length")
    nll = 0.5 * np.log(2.0 * np.pi * sigma**2) + (y - mu) ** 2 / (2.0 * sigma**2)
    return float(np.mean(nll))


def _flatten_reference_hoppings(y_hop: Union[list, np.ndarray]) -> np.ndarray:
    if isinstance(y_hop, list):
        return np.concatenate([np.asarray(y, dtype=float).ravel() for y in y_hop])
    return np.asarray(y_hop, dtype=float).ravel()


def _ensemble_predictions_to_matrix(yp: Any) -> np.ndarray:
    """(n_ensemble, n_obs) from MK-style stack or ACSF ragged list output."""
    if isinstance(yp, list):
        rows = []
        for r in yp:
            if isinstance(r, list):
                flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in r])
            else:
                flat = np.asarray(r, dtype=float).ravel()
            rows.append(flat)
        lens = [row.size for row in rows]
        lmin = min(lens) if lens else 0
        if lmin and max(lens) != min(lens):
            rows = [row[:lmin] for row in rows]
        return np.vstack(rows) if rows else np.empty((0, 0), dtype=float)

    yp = np.asarray(yp)
    if yp.dtype == object:
        rows = []
        for r in yp:
            flat = np.asarray(r, dtype=float).ravel()
            rows.append(flat)
        lens = [row.size for row in rows]
        lmin = min(lens) if lens else 0
        if lmin and max(lens) != min(lens):
            rows = [row[:lmin] for row in rows]
        return np.vstack(rows) if rows else np.empty((0, 0), dtype=float)

    yp = np.asarray(yp, dtype=float)
    if yp.ndim == 1:
        return yp[:, np.newaxis]
    return yp


def _T0_global(C0: Optional[float], params: Any, calc_keys: Sequence[str]) -> float:
    """Match EMCEE: T0 = C0 * (2 / n_params) (single global scale)."""
    if C0 is None or not np.isfinite(float(C0)):
        return 1.0
    n_params = sum(len(params[k]) for k in calc_keys)
    if n_params <= 0:
        n_params = 1
    t0 = float(C0) * (2.0 / float(n_params))
    return t0 if np.isfinite(t0) and t0 > 0 else 1.0


# -----------------------------------------------------------------------------
# Ensemble pickle I/O and path discovery
# -----------------------------------------------------------------------------


def default_temperature_grid(target: str) -> np.ndarray:
    if target == "energy":
        return np.array(
            [
                1e-5,
                1e-3,
                0.01,
                0.1,
                0.2,
                0.5,
                1,
                1.5,
                2.0,
                3,
                4,
                5,
                7,
                10,
                15,
                20,
                30,
                50,
            ],
            dtype=float,
        )
    return np.array(
        [
            1e-3,
            0.01,
            0.1,
            0.5,
            1,
            1.5,
            2.0,
            3,
            5,
            10,
            20,
            30,
            50,
            100,
            150,
            200,
            300,
            500,
        ],
        dtype=float,
    )


def default_p_grid() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)


_RE_MCMC_T = re.compile(r"_ensemble_T_([^/]+)\.pkl$")
_RE_SUB_P = re.compile(r"_SubSamp_ensemble_p_([^/]+)\.pkl$")


def _similar_ensemble_folder_names(
    model_name: str,
    ensemble_dir: str,
    *,
    limit: int = 8,
) -> List[str]:
    """Return ensemble subfolder names that partially match ``model_name``."""
    root = Path(ensemble_dir)
    if not root.is_dir():
        return []
    needle = model_name.lower()
    scored: List[Tuple[int, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        low = name.lower()
        if needle in low or low in needle:
            scored.append((0, name))
            continue
        # Same M/W tag but different ACSF variant (e.g. missing ``_sk``).
        mw = re.search(r"m[_\-](\d+)[_\-]w[_\-](\d+)", needle, re.I)
        if mw and re.search(
            rf"m[_\-]{mw.group(1)}[_\-]w[_\-]{mw.group(2)}", low, re.I,
        ):
            scored.append((1, name))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [name for _, name in scored[:limit]]


def discover_mcmc_files(model_name: str, ensemble_dir: str = "ensembles") -> List[Tuple[float, str]]:
    """Return sorted (T_weight, path) for existing MCMC ensemble pickles."""
    folder = os.path.join(ensemble_dir, model_name)
    pattern = os.path.join(glob.escape(folder), f"{glob.escape(model_name)}_ensemble_T_*.pkl")
    out: List[Tuple[float, str]] = []
    for path in glob.glob(pattern):
        m = _RE_MCMC_T.search(path.replace("\\", "/"))
        if not m:
            continue
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        out.append((t, path))
    out.sort(key=lambda x: x[0])
    return out


def expand_model_patterns(
    model_patterns: Sequence[str],
    ensemble_dir: str = "ensembles",
) -> List[str]:
    """Expand ``--models`` entries that contain glob wildcards.

    Each pattern is matched against immediate subdirectories of ``ensemble_dir``
    (the ensemble folder names).  Patterns without ``*``, ``?``, or ``[`` are
    passed through unchanged.
    """
    expanded: List[str] = []
    for pattern in model_patterns:
        if any(ch in pattern for ch in "*?[]"):
            search = os.path.join(ensemble_dir, pattern)
            matches = sorted(
                os.path.basename(os.path.normpath(p))
                for p in glob.glob(search)
                if os.path.isdir(p)
            )
            if not matches:
                print(
                    f"Warning: no ensemble folders match pattern {pattern!r} "
                    f"under {ensemble_dir!r}",
                    file=sys.stderr,
                )
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    seen: set[str] = set()
    out: List[str] = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def discover_subsamp_files(model_name: str, ensemble_dir: str = "ensembles") -> List[Tuple[float, str]]:
    """Return sorted (p, path) for SubSamp ensemble pickles."""
    folder = os.path.join(ensemble_dir, model_name)
    pattern = os.path.join(glob.escape(folder), f"{glob.escape(model_name)}_SubSamp_ensemble_p_*.pkl")
    out: List[Tuple[float, str]] = []
    for path in glob.glob(pattern):
        m = _RE_SUB_P.search(path.replace("\\", "/"))
        if not m:
            continue
        try:
            p = float(m.group(1))
        except ValueError:
            continue
        out.append((p, path))
    out.sort(key=lambda x: x[0])
    return out


def load_ensemble_pickle(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as exc:
        # Bad file on disk (empty, truncated, or corrupt) — not a metrics-code bug.
        size = os.path.getsize(path) if os.path.isfile(path) else None
        print(
            "[pickle] failed to load ensemble pickle "
            "(empty, truncated, or corrupt — e.g. interrupted MCMC save):\n"
            f"  path: {path}\n"
            f"  size_bytes: {size}\n"
            f"  error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise


def _pick_ypred_ydata_xdata(d: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Prefer test-set ensemble predictions and matching references.

    Ensemble pickles from ``EMCEE_generate_ensemble`` / SubSamp store
    ``ypred_samples_test`` alongside ``ydata_test`` / ``xdata_test``.  When those
    are present, all calibration metrics and diagnostics use the **test** split
    so scores match the predictions.  Otherwise fall back to the full/train
    payload (``ypred_samples``, ``ydata``, ``xdata``).
    """
    yp_test = d.get("ypred_samples_test")
    if isinstance(yp_test, dict) and yp_test:
        yd_test = d.get("ydata_test")
        if isinstance(yd_test, dict) and yd_test:
            return yp_test, yd_test, d.get("xdata_test") or d.get("xdata")
    return d["ypred_samples"], d["ydata"], d.get("xdata")


def detect_target_from_pkl(path: str) -> Optional[str]:
    """Return 'energy' or 'hopping' by inspecting prediction keys in one pickle."""
    try:
        d = load_ensemble_pickle(path)
        yp_test = d.get("ypred_samples_test")
        keys = set(yp_test.keys()) if isinstance(yp_test, dict) and yp_test else set()
        if not keys:
            keys = set(d.get("ypred_samples", {}) or {})
        if "energy" in keys:
            return "energy"
        if "hopping" in keys:
            return "hopping"
    except Exception:
        pass
    return None


def detect_target_from_model_name(model_name: str) -> Optional[str]:
    """Heuristic target from ensemble folder name when pickles are unavailable."""
    if model_name.startswith("ACSF_hoppings"):
        return "hopping"
    if model_name.startswith("POD_energy"):
        return "energy"
    if model_name in ("MK",) or model_name.startswith(
        ("intralayer_LETB", "interlayer_LETB", "LETB_")
    ):
        return "hopping"
    if model_name.startswith(("Tersoff", "TETB_POD", "TETB_energy", "DRIP", "Kolmogorov")):
        return "energy"
    return None


def resolve_target_for_model(
    model_name: str,
    *,
    target_arg: Optional[str],
    ensemble_root: str,
    metrics_dir: str,
    technique: str,
) -> str:
    """Choose ``energy`` vs ``hopping`` for metrics I/O (matches ``--calculate`` auto-detect)."""
    if target_arg is not None:
        return target_arg
    for tgt in ("hopping", "energy"):
        if os.path.isfile(metrics_npz_path(metrics_dir, model_name, technique, tgt)):
            return tgt
    inferred = detect_target_from_model_name(model_name)
    if inferred is not None:
        return inferred
    if technique == "mcmc":
        discovered = discover_mcmc_files(model_name, ensemble_root)
    else:
        discovered = discover_subsamp_files(model_name, ensemble_root)
    for _, pkl_path in discovered:
        if os.path.isfile(pkl_path):
            tgt = detect_target_from_pkl(pkl_path)
            if tgt is not None:
                return tgt
    return "energy"


def discover_control_grid(
    model_name: str,
    technique: str,
    ensemble_root: str,
    *,
    auto_discover: bool,
    temperatures: Optional[str],
    p_values: Optional[str],
    target: str,
) -> Tuple[np.ndarray, List[str], str]:
    """Return (control_values, pickle_paths, control_name) for metric sweeps."""
    if technique == "mcmc":
        if auto_discover:
            discovered = discover_mcmc_files(model_name, ensemble_root)
            if not discovered:
                return np.array([], dtype=float), [], "temperature"
            control_values = np.array([t for t, _ in discovered], dtype=float)
            paths = [path for _, path in discovered]
        else:
            tv = _parse_float_list(temperatures)
            if tv is None:
                tv = default_temperature_grid(target)
            control_values = tv
            paths = [
                os.path.join(
                    ensemble_root,
                    model_name,
                    f"{model_name}_ensemble_T_{t}.pkl",
                )
                for t in control_values
            ]
        return control_values, paths, "temperature"

    if auto_discover:
        discovered = discover_subsamp_files(model_name, ensemble_root)
        if not discovered:
            return np.array([], dtype=float), [], "p"
        control_values = np.array([pp for pp, _ in discovered], dtype=float)
        paths = [path for _, path in discovered]
    else:
        pv = _parse_float_list(p_values)
        if pv is None:
            pv = default_p_grid()
        control_values = pv
        paths = [
            os.path.join(
                ensemble_root,
                model_name,
                f"{model_name}_SubSamp_ensemble_p_{pp}.pkl",
            )
            for pp in control_values
        ]
    return control_values, paths, "p"


def ensure_metrics_npz(
    model_name: str,
    technique: str,
    target: str,
    *,
    ensemble_root: str,
    metrics_dir: str,
    auto_discover: bool,
    temperatures: Optional[str],
    p_values: Optional[str],
    no_t0_fit: bool,
    z_reference_temperature: float,
) -> Optional[str]:
    """Create calibration metrics NPZ from ensemble pickles if it is missing."""
    out = metrics_npz_path(metrics_dir, model_name, technique, target)
    if os.path.isfile(out):
        return out

    control_values, paths, control_name = discover_control_grid(
        model_name,
        technique,
        ensemble_root,
        auto_discover=auto_discover,
        temperatures=temperatures,
        p_values=p_values,
        target=target,
    )
    if control_values.size == 0 or not any(os.path.isfile(p) for p in paths):
        print(
            f"Cannot create metrics file: no {technique} ensemble pickles for "
            f"{model_name!r} under {ensemble_root!r}",
            file=sys.stderr,
        )
        return None

    print(f"Missing metrics file: {out} — computing from ensembles …", flush=True)
    T0_ref = 1.0 if no_t0_fit else resolve_T0_ref(model_name, target, ensemble_root)
    run_calculate(
        model_name,
        technique,
        target,
        control_values,
        paths,
        control_name,
        T0_ref,
        metrics_dir,
        z_reference_temperature,
    )
    return out if os.path.isfile(out) else None


# -----------------------------------------------------------------------------
# Per-target metric computation
# -----------------------------------------------------------------------------


def _metrics_hopping(
    ypred_samples: Any,
    ydata: Dict[str, Any],
    T_weight: float,
    T0_ref: float,
) -> Dict[str, Any]:
    key = "hopping"
    y_flat = _flatten_reference_hoppings(ydata[key])
    Y = _ensemble_predictions_to_matrix(ypred_samples[key])
    if Y.ndim != 2 or Y.shape[1] == 0:
        return {
            "mae": np.nan,
            "crps": np.nan,
            "nll": np.nan,
            "miscalibration_area": np.nan,
            "model_partition_function": np.nan,
            "average_cost": np.nan,
            "std_cost": np.nan,
            "y_ref": y_flat,
            "y_mean": None,
            "y_std": None,
            "forecasts": None,
        }
    if y_flat.size != Y.shape[1]:
        L = int(min(y_flat.size, Y.shape[1]))
        y_flat = y_flat[:L]
        Y = Y[:, :L]
    mean_pred = np.mean(Y, axis=0)
    y_std = np.std(Y, axis=0)
    mae = float(np.mean(np.abs(mean_pred - y_flat)))
    F = Y.T
    crps = crps_ensemble(y_flat, F)
    nll = mean_gaussian_nll(y_flat, mean_pred, y_std)
    miscal = miscalibration_area(y_flat, F)
    diff = Y - y_flat
    cost_val = np.sum(diff**2, axis=1)
    Z = float(np.mean(np.exp(-cost_val / (T_weight * T0_ref))))
    return {
        "mae": mae,
        "crps": crps,
        "nll": nll,
        "miscalibration_area": miscal,
        "model_partition_function": Z,
        "average_cost": float(np.mean(diff**2)),
        "std_cost": float(np.std(diff**2)),
        "y_ref": y_flat,
        "y_mean": mean_pred,
        "y_std": y_std,
        "forecasts": F,
    }


def _energy_atom_counts_per_config(
    xdata: Optional[Dict[str, Any]],
    n_configs: int,
) -> np.ndarray:
    """Atoms per structure for ``ydata['energy'][:n_configs]`` (eV/atom normalization).

    ``DataLoader.train_test_split`` keeps ``xdata['energy'][i]`` aligned with
    ``ydata['energy'][i]``.  Using a single ``len(xdata['energy'][0])`` for every
    column is wrong when atom counts vary (e.g. supercell size) and can silently
    default to ``1.0`` if ``xdata['energy'][0]`` is not an ``Atoms`` instance,
    which leaves **total** energies on the plot (orders of magnitude too large).
    """
    nat = np.ones(max(int(n_configs), 0), dtype=float)
    if n_configs <= 0:
        return nat
    if xdata is None or "energy" not in xdata:
        return nat
    xv = xdata["energy"]
    if isinstance(xv, (list, tuple)):
        for i in range(min(int(n_configs), len(xv))):
            try:
                nat[i] = float(len(xv[i]))
            except Exception:
                nat[i] = 1.0
        return nat
    try:
        n0 = float(len(xv[0]))
        nat[:] = n0
    except Exception:
        pass
    return nat


def _metrics_energy(
    ypred_samples: Any,
    ydata: Dict[str, Any],
    xdata: Optional[Dict[str, Any]],
    T_weight: float,
    T0_ref: float,
) -> Dict[str, Any]:
    key = "energy"
    ye = np.asarray(ydata[key], dtype=float).ravel()
    yp = np.asarray(ypred_samples[key], dtype=float)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)
    # ``evaluate_ensemble`` stacks rows = ensemble members, cols = configurations.
    if yp.ndim == 2 and yp.shape[1] != ye.size and yp.shape[0] == ye.size:
        yp = yp.T
    n_x = 0
    if xdata is not None and "energy" in xdata:
        xv0 = xdata["energy"]
        if isinstance(xv0, (list, tuple)):
            n_x = len(xv0)
    L = int(min(ye.size, yp.shape[1], n_x)) if n_x else int(min(ye.size, yp.shape[1]))
    if L <= 0:
        return {
            "mae": np.nan,
            "crps": np.nan,
            "nll": np.nan,
            "miscalibration_area": np.nan,
            "model_partition_function": np.nan,
            "average_cost": np.nan,
            "std_cost": np.nan,
            "y_ref": np.array([], dtype=float),
            "y_mean": None,
            "y_std": None,
            "forecasts": None,
        }
    ye = ye[:L]
    yp = yp[:, :L]
    nat = _energy_atom_counts_per_config(xdata, L)
    if L > 5 and np.all(nat == 1.0):
        print(
            "[plot_bayes_factor] warning: energy metrics used N_atoms=1 for every "
            "configuration (check pickle has aligned ``xdata_test['energy']`` list).",
            file=sys.stderr,
        )
    # Per-atom totals; same convention as PES parity (total energy / N_atoms).
    ypred_per_atom = yp / nat[np.newaxis, :]
    ydata_per_atom = ye / nat
    mean_pred = np.mean(ypred_per_atom, axis=0)
    y_std = np.std(ypred_per_atom, axis=0)
    mae = float(np.mean(np.abs(mean_pred - ydata_per_atom)))
    F = ypred_per_atom.T
    crps = crps_ensemble(ydata_per_atom, F)
    nll = mean_gaussian_nll(ydata_per_atom, mean_pred, y_std)
    miscal = miscalibration_area(ydata_per_atom, F)
    cost_val = np.sum((ypred_per_atom - ydata_per_atom) ** 2, axis=1)
    Z = float(np.mean(np.exp(-cost_val / (T_weight * T0_ref))))
    return {
        "mae": mae,
        "crps": crps,
        "nll": nll,
        "miscalibration_area": miscal,
        "model_partition_function": Z,
        "average_cost": float(np.mean((ypred_per_atom - ydata_per_atom) ** 2)),
        "std_cost": float(np.std((ypred_per_atom - ydata_per_atom) ** 2)),
        "y_ref": ydata_per_atom,
        "y_mean": mean_pred,
        "y_std": y_std,
        "forecasts": F,
    }


def compute_metrics_one_file(
    path: str,
    target: str,
    technique: str,
    T_weight_for_z: float,
    T0_ref: float,
) -> Dict[str, Any]:
    """Load one pickle and return metric dict + raw arrays for diagnostics."""
    d = load_ensemble_pickle(path)
    ypred, ydata, xdata = _pick_ypred_ydata_xdata(d)
    if target == "hopping":
        return _metrics_hopping(ypred, ydata, T_weight_for_z, T0_ref)
    if target == "energy":
        return _metrics_energy(ypred, ydata, xdata, T_weight_for_z, T0_ref)
    raise ValueError(f"target must be 'energy' or 'hopping', got {target!r}")


def resolve_T0_ref(
    model_name: str,
    target: str,
    ensemble_dir: str = "ensembles",
) -> float:
    """Return T0 scaling for Z.

    Reads ``T0_ref.json`` written by ``EMCEE_generate_ensemble.py`` from the
    model's ensemble directory.  Falls back to ``1.0`` without invoking LAMMPS
    or any external calculation when the sidecar is absent.
    """
    sidecar = os.path.join(ensemble_dir, model_name, "T0_ref.json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as f:
                data = json.load(f)
            t0 = float(data["T0_ref"])
            if np.isfinite(t0) and t0 > 0:
                return t0
        except Exception as e:
            print(f"Warning: could not read {sidecar}: {e}", file=sys.stderr)
    return 1.0


# -----------------------------------------------------------------------------
# Sweeps: calculate + save
# -----------------------------------------------------------------------------


def metrics_npz_path(
    metrics_dir: str,
    model_name: str,
    technique: str,
    target: str,
) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", model_name)
    return os.path.join(
        metrics_dir,
        f"calibration_{safe}_{technique}_{target}.npz",
    )


def run_calculate(
    model_name: str,
    technique: str,
    target: str,
    control_values: np.ndarray,
    paths_by_control: Sequence[str],
    control_name: str,
    T0_ref: float,
    metrics_dir: str,
    z_reference_temperature: float,
) -> str:
    """Compute metrics along the grid and save compressed NPZ. Returns output path."""
    n = len(control_values)
    mae = np.full(n, np.nan)
    crps = np.full(n, np.nan)
    nll = np.full(n, np.nan)
    miscal = np.full(n, np.nan)
    Z = np.full(n, np.nan)
    avg_cost = np.full(n, np.nan)
    std_cost = np.full(n, np.nan)

    for i, (cv, path) in enumerate(zip(control_values, paths_by_control)):
        if not os.path.isfile(path):
            print(f"  skip missing: {path}", file=sys.stderr)
            continue
        if technique == "mcmc":
            T_for_z = float(cv)
        else:
            T_for_z = float(z_reference_temperature)
        try:
            m = compute_metrics_one_file(path, target, technique, T_for_z, T0_ref)
        except Exception as exc:
            print(
                f"  skip {os.path.basename(path)}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        mae[i] = m["mae"]
        crps[i] = m["crps"]
        nll[i] = m["nll"]
        miscal[i] = m["miscalibration_area"]
        Z[i] = m["model_partition_function"]
        avg_cost[i] = m["average_cost"]
        std_cost[i] = m["std_cost"]

    out = metrics_npz_path(metrics_dir, model_name, technique, target)
    os.makedirs(metrics_dir, exist_ok=True)
    meta = {
        "model_name": model_name,
        "technique": technique,
        "target": target,
        "control_name": control_name,
        "T0_ref": float(T0_ref),
        "z_reference_temperature": float(z_reference_temperature),
    }
    np.savez_compressed(
        out,
        control_values=control_values.astype(float),
        control_name=np.array(control_name),
        mae=mae,
        crps=crps,
        nll=nll,
        miscalibration_area=miscal,
        model_partition_function=Z,
        average_cost=avg_cost,
        std_cost=std_cost,
        meta_json=np.array(json.dumps(meta)),
    )
    print(f"Wrote {out}")
    return out


def _resolve_diagnostic_index(
    technique: str,
    control_values: np.ndarray,
    temperatures_arg: Optional[str],
    p_values_arg: Optional[str],
    diagnostic_index_arg: Optional[int],
) -> int:
    """Pick grid index for ``--diagnostics``.

    For MCMC, if ``--temperatures`` parses to at least one float, the first
    value selects the closest ``T`` in ``control_values`` (overrides
    ``--diagnostic-index``).  For ``subsamp``, the same applies to the first
    value in ``--p-values``.  Otherwise use ``diagnostic_index_arg`` or the
    middle of the grid.
    """
    cv = np.asarray(control_values, dtype=float)
    n = int(cv.size)
    if n <= 0:
        return 0
    if technique == "mcmc":
        tv = _parse_float_list(temperatures_arg)
        if tv is not None and tv.size > 0:
            T_req = float(tv[0])
            return int(np.argmin(np.abs(cv - T_req)))
    elif technique == "subsamp":
        pv = _parse_float_list(p_values_arg)
        if pv is not None and pv.size > 0:
            p_req = float(pv[0])
            return int(np.argmin(np.abs(cv - p_req)))
    if diagnostic_index_arg is not None:
        return int(np.clip(int(diagnostic_index_arg), 0, n - 1))
    return n // 2


def run_diagnostics_plots(
    model_name: str,
    technique: str,
    target: str,
    control_name: str,
    control_values: np.ndarray,
    paths_by_control: Sequence[str],
    T0_ref: float,
    figures_dir: str,
    z_reference_temperature: float,
    diagnostic_index: Optional[int] = None,
) -> None:
    """Parity, QQ (``qq_plot_example`` style), and std vs |error| at one grid point."""
    if diagnostic_index is None:
        diagnostic_index = len(control_values) // 2
    diagnostic_index = int(np.clip(diagnostic_index, 0, len(control_values) - 1))
    cv = control_values[diagnostic_index]
    path = paths_by_control[diagnostic_index]
    if not os.path.isfile(path):
        print(f"Diagnostics: file missing {path}", file=sys.stderr)
        return
    if technique == "mcmc":
        T_for_z = float(cv)
    else:
        T_for_z = float(z_reference_temperature)
    m = compute_metrics_one_file(path, target, technique, T_for_z, T0_ref)
    y_ref = m["y_ref"]
    y_mean = m["y_mean"]
    y_std = m["y_std"]
    if y_mean is None or y_std is None:
        print("Diagnostics: insufficient data for plots", file=sys.stderr)
        return

    sub = os.path.join(figures_dir, model_name)
    os.makedirs(sub, exist_ok=True)
    ctrl_tag = f"{technique}_{control_name}_{cv:g}".replace(".", "p")

    plt.figure(figsize=(6, 5))
    plt.errorbar(y_ref, y_mean, yerr=y_std, fmt="o", ms=3, alpha=0.7)
    lims = np.linspace(
        float(np.nanmin(y_ref)), float(np.nanmax(y_ref)), 50
    )
    plt.plot(lims, lims, "k-", lw=1)
    if target == "energy":
        plt.xlabel("Reference energy / atom (eV)")
        plt.ylabel("Ensemble mean ± std (eV / atom)")
    else:
        plt.xlabel("Reference")
        plt.ylabel("Ensemble mean ± std")
    plt.title(f"{model_name} ({target}) {control_name}={cv:g}")
    plt.tight_layout()
    p1 = os.path.join(sub, f"{model_name}_{ctrl_tag}_parity.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    abs_err = np.abs(y_mean - y_ref)
    plt.figure(figsize=(6, 5))
    plt.scatter(abs_err, y_std, s=8, alpha=0.6)
    mx = max(float(np.nanmax(abs_err)), float(np.nanmax(y_std)))
    if mx > 0:
        plt.plot([0, mx], [0, mx], "k--", lw=1)
    plt.xlabel("|error| (MAE per point)")
    plt.ylabel("Ensemble std")
    plt.title(f"{model_name} spread vs error ({control_name}={cv:g})")
    plt.tight_layout()
    p2 = os.path.join(sub, f"{model_name}_{ctrl_tag}_std_vs_abserr.png")
    plt.savefig(p2, dpi=150)
    plt.close()

    F = m.get("forecasts")
    p3 = None
    if F is not None and np.asarray(F).size > 0:
        u = pit_values(y_ref, F)
        u_sorted = np.sort(u)
        n_u = u_sorted.size
        uniform_quantiles = np.linspace(0.0, 1.0, n_u)
        plt.figure(figsize=(6, 5))
        plt.plot(uniform_quantiles, u_sorted, "o", ms=3, alpha=0.7, label="QQ")
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        plt.xlabel("Uniform quantiles")
        plt.ylabel("Empirical quantiles")
        plt.title(f"QQ — {model_name} ({target}) {control_name}={cv:g}")
        plt.legend()
        plt.tight_layout()
        p3 = os.path.join(sub, f"{model_name}_{ctrl_tag}_pit_qq.png")
        plt.savefig(p3, dpi=150)
        plt.close()

    lines = [f"           {p1}", f"           {p2}"]
    if p3:
        lines.append(f"           {p3}")
    print("Diagnostics:\n" + "\n".join(lines))


# -----------------------------------------------------------------------------
# Plot metrics vs control (single or compare)
# -----------------------------------------------------------------------------


def load_metrics_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"][()]))
    n = int(np.asarray(z["control_values"]).size)
    nan_vec = np.full(n, np.nan, dtype=float)
    out: Dict[str, Any] = {
        "control_values": np.asarray(z["control_values"]),
        "mae": np.asarray(z["mae"]),
        "crps": np.asarray(z["crps"]),
        "nll": np.asarray(z["nll"]),
        "model_partition_function": np.asarray(z["model_partition_function"]),
        "meta": meta,
    }
    if "miscalibration_area" in z.files:
        out["miscalibration_area"] = np.asarray(z["miscalibration_area"])
    else:
        out["miscalibration_area"] = nan_vec
    return out


DEFAULT_CALIBRATION_METRICS_DIR = "calibration_metrics"


def optimal_temperature_miscalibration(
    model_name: str,
    metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    technique: str = "mcmc",
    target: str = "energy",
) -> Optional[float]:
    """
    Temperature *T* that minimizes ``miscalibration_area`` on the saved calibration grid.

    Reads ``calibration_metrics/calibration_<model>_<technique>_<target>.npz``
    (from ``plot_bayes_factor.py --calculate``).  Returns ``None`` if the file is
    missing or has no finite miscalibration values.
    """
    path = metrics_npz_path(metrics_dir, model_name, technique, target)
    if not os.path.isfile(path):
        return None
    d = load_metrics_npz(path)
    cv = np.asarray(d["control_values"], dtype=float)
    miscal = np.asarray(d["miscalibration_area"], dtype=float)
    valid = np.isfinite(cv) & np.isfinite(miscal)
    if not np.any(valid):
        return None
    idx = int(np.argmin(np.where(valid, miscal, np.inf)))
    return float(cv[idx])


def resolve_ensemble_pickle(
    model_name: str,
    ensemble_dir: str = "ensembles",
    temperature: Optional[float] = None,
    *,
    calibration_metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    calibration_technique: str = "mcmc",
    calibration_target: str = "energy",
) -> Tuple[str, float]:
    """
    Path to the MCMC ensemble pickle and the temperature weight used.

    If ``temperature`` is ``None``, use the grid point that minimizes
    ``miscalibration_area`` in ``calibration_metrics_dir``; if that file is
    unavailable, use the lowest discovered ``T`` on disk.
    """
    discovered = discover_mcmc_files(model_name, ensemble_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No MCMC ensemble pickles for {model_name!r} under {ensemble_dir!r}"
        )

    t_req: Optional[float] = float(temperature) if temperature is not None else None
    if t_req is None:
        t_opt = optimal_temperature_miscalibration(
            model_name,
            calibration_metrics_dir,
            calibration_technique,
            calibration_target,
        )
        if t_opt is not None:
            t_req = float(t_opt)
            metrics_path = metrics_npz_path(
                calibration_metrics_dir,
                model_name,
                calibration_technique,
                calibration_target,
            )
            print(
                f"  Default T = {t_req:g} (min miscalibration_area in {metrics_path})",
                flush=True,
            )
        else:
            t_req = float(discovered[0][0])
            print(
                f"  Warning: no calibration metrics for {model_name!r} under "
                f"{calibration_metrics_dir!r}; using T={t_req:g} (lowest on disk).",
                file=sys.stderr,
                flush=True,
            )

    best = min(discovered, key=lambda x: abs(x[0] - t_req))
    return best[1], float(best[0])


_SAMPLING_LABEL = {
    "mcmc": "MCMC",
    "subsamp": "SubSamp",
}


def plot_metric_curves(
    entries: List[Tuple[str, str]],
    metric: str,
    figures_dir: str,
    x_scale: str,
    title: Optional[str] = None,
) -> str:
    """
    entries : list of (label, path_to_npz)
    metric : model_partition_function | crps | nll | mae | miscalibration_area
    x_scale : linear | log
    """
    plt.figure(figsize=(8, 5))
    technique_key: Optional[str] = None
    for label, path in entries:
        d = load_metrics_npz(path)
        if technique_key is None:
            technique_key = str(d["meta"].get("technique", "mcmc"))
        x = d["control_values"]
        y = np.asarray(d[metric], dtype=float)
        meta = d["meta"]
        cn = meta.get("control_name", "control")
        if x_scale == "log":
            if cn == "temperature":
                # ``control_values`` are the temperature weights from ``*_ensemble_T_<w>.pkl``.
                xplot = np.log10(np.maximum(x, 1e-300))
                xlab = r"$log(T/T_{0})$"
            else:
                xplot = np.log10(np.maximum(x, 1e-300))
                xlab = r"$\log_{10}(p)$"
        else:
            xplot = x
            xlab = cn
        plt.plot(xplot, y, marker="o", ms=4, label=label)

    plt.xlabel(xlab)
    ylab = {
        "model_partition_function": "model_partition_function (Z)",
        "miscalibration_area": r"$\mathcal{M}$",
    }.get(metric, metric)
    plt.ylabel(ylab)
    if metric == "miscalibration_area":
        plt.ylim(0.0, 1.0)
    tech_label = _SAMPLING_LABEL.get(technique_key or "mcmc", technique_key or "mcmc")
    if title:
        plt.title(f"{title} ({tech_label})")
    else:
        plt.title(f"{tech_label} — {metric}")
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    safe_metric = re.sub(r"[^\w]+", "_", metric)
    tech_slug = re.sub(r"[^\w]+", "_", technique_key or "mcmc")
    out = os.path.join(
        figures_dir, f"compare_{safe_metric}_{x_scale}_{tech_slug}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_float_list(s: Optional[str]) -> Optional[np.ndarray]:
    if s is None or not str(s).strip():
        return None
    parts = str(s).replace(",", " ").split()
    return np.array([float(p) for p in parts], dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Calibration metrics (Z, CRPS, NLL, miscalibration area) for MCMC / SubSamp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "One or more model_name strings (ensemble folder names under "
            "--ensemble-dir). Patterns with glob wildcards (*, ?, []) match "
            "subfolders, e.g. POD_energy_POD_index* selects every folder whose "
            "name starts with POD_energy_POD_index."
        ),
    )
    p.add_argument(
        "--technique",
        choices=("mcmc", "subsamp"),
        default="mcmc",
        help="mcmc: files *_ensemble_T_*.pkl ; subsamp: *_SubSamp_ensemble_p_*.pkl",
    )
    p.add_argument(
        "--target",
        choices=("hopping", "energy"),
        default=None,
        help="Which observable block to score. Auto-detected from model name, saved "
        "metrics, or the first pkl when omitted (used by --calculate and --plot-metrics).",
    )
    p.add_argument(
        "--calculate",
        action="store_true",
        help="Compute metrics from ensemble pickles and write calibration_metrics/*.npz",
    )
    p.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Plot metric curves from saved npz (use with --compare for overlay).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="When plotting, overlay all --models on one figure (same target/technique).",
    )
    p.add_argument(
        "--auto-discover",
        action="store_true",
        help="Use glob to find all T or p values with existing pickles (ignore manual lists).",
    )
    p.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Space-separated T weights for mcmc. With --auto-discover, still used "
             "by --diagnostics: first value picks the nearest T on the discovered grid. "
             "Without --auto-discover, this list is also the calculate grid.",
    )
    p.add_argument(
        "--p-values",
        dest="p_values",
        type=str,
        default=None,
        help="Space-separated p for subsamp. With --auto-discover, first value picks "
             "the nearest p for --diagnostics. Without --auto-discover, also the calculate grid.",
    )
    p.add_argument(
        "--metrics-dir",
        type=str,
        default="calibration_metrics",
        help="Directory for *.npz metric tables.",
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default="figures/calibration",
        help="Diagnostic and comparison figures.",
    )
    p.add_argument(
        "--ensemble-dir",
        type=str,
        default="ensembles",
        help="Root folder containing <model_name>/ ensemble pickles.",
    )
    p.add_argument(
        "--z-reference-temperature",
        type=float,
        default=1.0,
        help="For subsamp only: T in exp(-SSE/(T*T0)) when computing Z (MCMC uses each file's T).",
    )
    p.add_argument(
        "--no-t0-fit",
        action="store_true",
        help="Use T0_ref=1.0 instead of loading get_MCMC_inputs/get_C0.",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write parity, QQ, and std-vs-error plots for one grid point.",
    )
    p.add_argument(
        "--diagnostic-index",
        type=int,
        default=None,
        help="Index into sorted control grid for --diagnostics when --temperatures "
             "(mcmc) or --p-values (subsamp) is unset; default: middle.",
    )
    p.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric key: model_partition_function, crps, nll, mae, miscalibration_area. "
        "Repeatable. Default: Z, crps, nll, miscalibration_area.",
    )
    p.add_argument(
        "--x-scale",
        choices=("linear", "log"),
        default="log",
        help="X-axis for metric curves: log → log10(T) or log10(p).",
    )

    args = p.parse_args()
    if not args.calculate and not args.plot_metrics and not args.diagnostics:
        p.error("Select at least one of: --calculate, --plot-metrics, --diagnostics")

    # Paths such as ``ensembles/`` and ``calibration_metrics/`` are relative to
    # ``uncertainty_quantification/``, not ``visualizations/``.
    os.chdir(UQ_DIR)

    technique = args.technique
    ensemble_root = getattr(args, "ensemble_dir", "ensembles")
    models = expand_model_patterns(args.models, ensemble_root)
    if models != args.models:
        print(f"Expanded --models to {len(models)} ensemble folder(s): {models}", flush=True)
    if not models:
        p.error("No models to process (check --models patterns and --ensemble-dir).")

    for model_name in models:
        if args.calculate or args.diagnostics:
            # ── 1. Resolve target (before grid discovery) ───────────────────────
            target = resolve_target_for_model(
                model_name,
                target_arg=args.target,
                ensemble_root=ensemble_root,
                metrics_dir=args.metrics_dir,
                technique=technique,
            )
            if args.target is None:
                print(f"Auto-detected target '{target}' for {model_name}", flush=True)

            # ── 2. Discover pickle files ──────────────────────────────────────
            control_values, paths, control_name = discover_control_grid(
                model_name,
                technique,
                ensemble_root,
                auto_discover=args.auto_discover,
                temperatures=args.temperatures,
                p_values=args.p_values,
                target=target,
            )
            if control_values.size == 0 or not any(os.path.isfile(p) for p in paths):
                msg = (
                    f"No {technique} ensemble files for {model_name} under "
                    f"{ensemble_root!r} (cwd={os.getcwd()!r})"
                )
                similar = _similar_ensemble_folder_names(model_name, ensemble_root)
                if similar:
                    msg += "\n  Similar ensemble folders: " + ", ".join(similar)
                print(msg, file=sys.stderr)
                continue

            # ── 3. T0 reference (reads sidecar JSON, no LAMMPS) ──────────────
            T0_ref = 1.0 if args.no_t0_fit else resolve_T0_ref(model_name, target, ensemble_root)

            if args.calculate:
                run_calculate(
                    model_name,
                    technique,
                    target,
                    control_values,
                    paths,
                    control_name,
                    T0_ref,
                    args.metrics_dir,
                    args.z_reference_temperature,
                )

            if args.diagnostics:
                diag_idx = _resolve_diagnostic_index(
                    technique,
                    control_values,
                    args.temperatures,
                    args.p_values,
                    args.diagnostic_index,
                )
                tv_parsed = _parse_float_list(args.temperatures)
                if technique == "mcmc" and tv_parsed is not None and tv_parsed.size > 0:
                    print(
                        f"[diagnostics] {model_name}: T_request={float(tv_parsed[0]):g} → "
                        f"T={float(control_values[diag_idx]):g} (index {diag_idx})",
                        flush=True,
                    )
                pv_parsed = _parse_float_list(args.p_values)
                if technique == "subsamp" and pv_parsed is not None and pv_parsed.size > 0:
                    print(
                        f"[diagnostics] {model_name}: p_request={float(pv_parsed[0]):g} → "
                        f"p={float(control_values[diag_idx]):g} (index {diag_idx})",
                        flush=True,
                    )
                run_diagnostics_plots(
                    model_name,
                    technique,
                    target,
                    control_name,
                    control_values,
                    paths,
                    T0_ref,
                    args.figures_dir,
                    args.z_reference_temperature,
                    diagnostic_index=diag_idx,
                )

    if args.plot_metrics:
        metrics_to_plot = args.metric or [
            "model_partition_function",
            "crps",
            "nll",
            "miscalibration_area",
        ]
        if args.compare:
            entries = []
            for model_name in models:
                plot_target = resolve_target_for_model(
                    model_name,
                    target_arg=args.target,
                    ensemble_root=ensemble_root,
                    metrics_dir=args.metrics_dir,
                    technique=technique,
                )
                if args.target is None:
                    print(f"Auto-detected target '{plot_target}' for {model_name}", flush=True)
                path = ensure_metrics_npz(
                    model_name,
                    technique,
                    plot_target,
                    ensemble_root=ensemble_root,
                    metrics_dir=args.metrics_dir,
                    auto_discover=args.auto_discover,
                    temperatures=args.temperatures,
                    p_values=args.p_values,
                    no_t0_fit=args.no_t0_fit,
                    z_reference_temperature=args.z_reference_temperature,
                )
                if path is None:
                    continue
                entries.append((model_name, path))
            if len(entries) < 1:
                print("No metric files to plot.", file=sys.stderr)
            else:
                plot_target = resolve_target_for_model(
                    entries[0][0],
                    target_arg=args.target,
                    ensemble_root=ensemble_root,
                    metrics_dir=args.metrics_dir,
                    technique=technique,
                )
                for mname in metrics_to_plot:
                    plot_metric_curves(
                        entries,
                        mname,
                        args.figures_dir,
                        args.x_scale,
                        title=f"{plot_target} — {mname}",
                    )
        else:
            for model_name in models:
                plot_target = resolve_target_for_model(
                    model_name,
                    target_arg=args.target,
                    ensemble_root=ensemble_root,
                    metrics_dir=args.metrics_dir,
                    technique=technique,
                )
                if args.target is None:
                    print(f"Auto-detected target '{plot_target}' for {model_name}", flush=True)
                path = ensure_metrics_npz(
                    model_name,
                    technique,
                    plot_target,
                    ensemble_root=ensemble_root,
                    metrics_dir=args.metrics_dir,
                    auto_discover=args.auto_discover,
                    temperatures=args.temperatures,
                    p_values=args.p_values,
                    no_t0_fit=args.no_t0_fit,
                    z_reference_temperature=args.z_reference_temperature,
                )
                if path is None:
                    continue
                for mname in metrics_to_plot:
                    plot_metric_curves(
                        [(model_name, path)],
                        mname,
                        os.path.join(args.figures_dir, model_name),
                        args.x_scale,
                        title=f"{model_name} — {mname}",
                    )


if __name__ == "__main__":
    main()
