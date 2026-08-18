"""
Non-plotting utilities for ensemble pickle discovery, loading, and calibration metrics.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CALIBRATION_METRICS_DIR = "calibration_metrics"

_RE_MCMC_T = re.compile(r"_ensemble_T_([^/]+)\.pkl$")
_RE_SUB_P = re.compile(r"_SubSamp_ensemble_p_([^/]+)\.pkl$")


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


def expand_model_patterns(
    model_patterns: Sequence[str],
    ensemble_dir: str = "ensembles",
) -> List[str]:
    """Expand ``--models`` entries that contain glob wildcards."""
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


def load_ensemble_pickle(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as exc:
        size = os.path.getsize(path) if os.path.isfile(path) else None
        print(
            "[pickle] failed to load ensemble pickle:\n"
            f"  path: {path}\n"
            f"  size_bytes: {size}\n"
            f"  error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise


def detect_target_from_model_name(model_name: str) -> Optional[str]:
    """Heuristic target from ensemble folder name."""
    if model_name.startswith("ACSF_hoppings"):
        return "hopping"
    if model_name.startswith("MK") or model_name.startswith("LETB") or model_name.startswith("intralayer_LETB"):
        return "hopping"
    if model_name.startswith("POD_energy") or model_name.startswith("Allegro_energy") or model_name in (
        "Tersoff", "DRIP", "Kolmogorov_Crespi",
        "Tersoff+DRIP", "Tersoff+Kolmogorov_Crespi",
    ):
        return "energy"
    return None


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


def optimal_calibration_index(metrics: Dict[str, Any]) -> Optional[int]:
    """Grid index that minimizes ``miscalibration_area`` among finite control values."""
    cv = np.asarray(metrics["control_values"], dtype=float)
    miscal = np.asarray(metrics["miscalibration_area"], dtype=float)
    valid = np.isfinite(cv) & np.isfinite(miscal)
    if not np.any(valid):
        return None
    return int(np.argmin(np.where(valid, miscal, np.inf)))


def calibration_at_min_miscalibration(
    model_name: str,
    metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    technique: str = "mcmc",
    target: str = "energy",
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(control_value, nll)`` at the grid point minimizing miscalibration area."""
    path = metrics_npz_path(metrics_dir, model_name, technique, target)
    if not os.path.isfile(path):
        return None, None
    d = load_metrics_npz(path)
    idx = optimal_calibration_index(d)
    if idx is None:
        return None, None
    cv_arr = np.asarray(d["control_values"], dtype=float)
    nll_arr = np.asarray(d["nll"], dtype=float)
    cv = float(cv_arr[idx]) if idx < cv_arr.size else float("nan")
    nll = float(nll_arr[idx]) if idx < nll_arr.size else float("nan")
    return (
        cv if np.isfinite(cv) else None,
        nll if np.isfinite(nll) else None,
    )


def optimal_temperature_miscalibration(
    model_name: str,
    metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    technique: str = "mcmc",
    target: str = "energy",
) -> Optional[float]:
    """Temperature *T* that minimizes ``miscalibration_area`` on the saved calibration grid."""
    path = metrics_npz_path(metrics_dir, model_name, technique, target)
    if not os.path.isfile(path):
        return None
    d = load_metrics_npz(path)
    idx = optimal_calibration_index(d)
    if idx is None:
        return None
    return float(np.asarray(d["control_values"], dtype=float)[idx])


def resolve_ensemble_pickle(
    model_name: str,
    ensemble_dir: str = "ensembles",
    temperature: Optional[float] = None,
    *,
    calibration_metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    calibration_technique: str = "mcmc",
    calibration_target: str = "energy",
) -> Tuple[str, float]:
    """Path to the MCMC ensemble pickle and the temperature weight used."""
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
