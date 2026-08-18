"""
Calibration metrics for MCMC and subsampling ensembles.

Computes (per control grid point: temperature *T* for MCMC, fraction *p* for SubSamp):

- ``model_partition_function`` — mean Boltzmann weight ``mean(exp(-SSE / (T * T0)))``
- ``crps`` — continuous ranked probability score (ensemble predictive)
- ``nll`` — mean Gaussian negative log-likelihood using ensemble mean / std
  (raw per-point ensemble std; no relative σ floor)
- ``likelihood`` — mean Gaussian likelihood
  ``(2πσ)^(-1/2) exp(-(y-μ)²/σ²)`` using the same ensemble mean / std as NLL
- ``loo_log_predictive_likelihood`` — LOO-CV mean log predictive density
  ``(1/N) Σ_i log p(y_i | X, y_{-i})`` with Gaussian ``μ_i, σ_i`` from
  ensemble members reweighted by fit to ``y_{-i}``
- ``miscalibration_area`` — PIT-based QQ miscalibration (see ``miscalibration_area``; scaled ×4)
- ``standardized_miscalibration_area`` — standardized-residual QQ miscalibration (``[-3,3]`` quantiles)

Saves arrays to ``calibration_metrics/`` and can plot:

- Mean prediction ± ensemble std vs reference (parity-style; **energy** is
  total energy divided by **each** configuration's atom count from
  ``xdata['energy']``, same convention as PES parity; no shift to minimum
  reference energy)
- Ensemble std vs per-point absolute error
- PIT QQ plot (used for ``miscalibration_area``) and a separate standardized-residual
  QQ plot (per-point ``(y - μ_i) / σ_i``; excludes near-zero ``σ_i``; not used for
  miscalibration)
- Metrics vs *T* or *p*, with optional overlay for several models on the same dataset type.
- In ``--compare`` mode, **NLL** is shown as a bar chart at each model's control
  value (temperature *T* or subsampling fraction *p*) that minimizes
  ``miscalibration_area``; other metrics remain line plots.  Compare figures
  are named ``<family>_compare_<metric>_…png`` (e.g.
  ``POD_energy_POD_index_compare_crps_log_mcmc.png``).
- With ``--plot-nll-hyperparams``, ``--plot-likelihood-hyperparams``, and/or
  ``--plot-loo-hyperparams`` (and ``--plot-metrics``), additional figures plot
  NLL, likelihood, and/or LOO log predictive likelihood at best calibration
  vs hyperparameters (POD: vs 2-body radial count, **one figure per ``rcut``**;
  ACSF: vs ``M``).

When a pickle contains ``ypred_samples_test`` and ``ydata_test``, those are used
for metrics and diagnostics (test split); otherwise the full/train fields are used.

For ``--diagnostics`` with MCMC, use ``--temperatures <T>`` (first value) to select the
nearest temperature on the current grid (works with ``--auto-discover``).  For
subsamp, the first ``--p-values`` entry selects the nearest ``p``.  When neither
``--temperatures``/``--p-values`` nor ``--diagnostic-index`` is set, the grid point
with minimum ``miscalibration_area`` is used.

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

NLL vs hyperparameters (POD 2-body radial; ACSF ``M`` at fixed ``W``)::

    python plot_bayes_factor.py --plot-metrics --plot-nll-hyperparams --compare \\
        --models 'POD_energy_POD_index*' 'ACSF_hoppings_sk_M_*_W_*' \\
        --technique mcmc --target energy --auto-discover

Likelihood vs hyperparameters (same layout as NLL)::

    python plot_bayes_factor.py --plot-metrics --plot-likelihood-hyperparams --compare \\
        --models 'POD_energy_POD_index*' 'ACSF_hoppings_sk_M_*_W_*' \\
        --technique mcmc --target hopping --auto-discover
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

CSFONT = {"fontname": "sans-serif", "size": 20}
STANDARDIZED_QQ_LIM = 3.0
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": CSFONT["size"],
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
REPO_ROOT = UQ_DIR.parent

_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from blg_model_builder.pod_model_selection import (
        find_pod_energy_ensemble_folder,
        pod_row_for_index,
    )
except ImportError:
    find_pod_energy_ensemble_folder = None  # type: ignore[misc, assignment]
    pod_row_for_index = None  # type: ignore[misc, assignment]

# POD_energy hyperparameter grid (matches ``pod_hyperparameter_search.GRID``).
_POD_TWO_BODY_RADIAL_GRID = list(range(6, 14, 1))
_POD_THREE_BODY_RADIAL_GRID = (4, 6, 8, 10)
_POD_THREE_BODY_ANGULAR_GRID = (4, 6, 8)

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


def standardized_residual_qq_quantiles(
    observations: np.ndarray,
    forecasts: np.ndarray,
    quantile_probs: Optional[np.ndarray] = None,
    quantile_hi: float = 0.97,
    sigma_rel_floor: float = 1e-3,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    QQ quantiles for per-point standardized ensemble and reference residuals.

    At each observation *i*, ensemble members and the reference are centered on
    the ensemble mean and scaled by the ensemble std at that site::

        z_ij = (F_ij - μ_i) / σ_i ,   z_obs_i = (y_i - μ_i) / σ_i

    Points with degenerate ensemble spread (``σ_i`` below a relative floor) are
    excluded because standardization is ill-defined there and inflates ``z_obs``.

    Returns quantiles of the pooled standardized ensemble (x-axis) and
    standardized reference values (y-axis) at common interior probability
    levels (default upper limit 0.97, as in ``qq_plot_example.py``).
    """
    observations = np.asarray(observations, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(1, -1)
    if observations.size != forecasts.shape[0]:
        raise ValueError(
            "standardized_residual_qq_quantiles: obs length "
            f"{observations.size} vs forecasts rows {forecasts.shape[0]}"
        )
    mu = np.mean(forecasts, axis=1)
    sigma_raw = np.std(forecasts, axis=1, ddof=0)
    positive = sigma_raw[sigma_raw > 0]
    sigma_ref = float(np.median(positive)) if positive.size else 1.0
    sigma_min = max(eps, sigma_rel_floor * sigma_ref)
    valid = sigma_raw >= sigma_min
    if not np.any(valid):
        raise ValueError(
            "standardized_residual_qq_quantiles: no observations with finite "
            f"ensemble spread (sigma_min={sigma_min:g})"
        )
    sigma = np.maximum(sigma_raw[valid], sigma_min)
    z_obs = (observations[valid] - mu[valid]) / sigma
    z_ens = (forecasts[valid, :] - mu[valid, np.newaxis]) / sigma[:, np.newaxis]
    if quantile_probs is None:
        n_q = int(min(100, max(z_obs.size, 10)))
        quantile_probs = np.linspace(0.0, quantile_hi, n_q)
    else:
        quantile_probs = np.asarray(quantile_probs, dtype=float)
        quantile_probs = quantile_probs[
            (quantile_probs >= 0.0) & (quantile_probs <= quantile_hi)
        ]
    if quantile_probs.size == 0:
        raise ValueError("standardized_residual_qq_quantiles: empty quantile grid")
    observed_q = np.quantile(z_obs, quantile_probs)
    ensemble_q = np.quantile(z_ens.ravel(), quantile_probs)
    n_excluded = int(observations.size - z_obs.size)
    return ensemble_q, observed_q, quantile_probs, n_excluded


def standardized_miscalibration_area(
    observations: np.ndarray,
    forecasts: np.ndarray,
    qq_lim: float = STANDARDIZED_QQ_LIM,
    quantile_hi: float = 0.97,
    sigma_rel_floor: float = 1e-3,
    eps: float = 1e-12,
) -> float:
    """
    Area between the standardized-residual QQ curve and the diagonal.

    Uses ``standardized_residual_qq_quantiles`` (same construction as the
    diagnostic QQ plot), keeps only quantile pairs in ``[-qq_lim, qq_lim]``,
    integrates ``|obs_q - ens_q|`` with respect to quantile probability, and
    normalizes by ``2 * qq_lim`` so well-calibrated curves are O(1) on [0, 1].
    """
    observations = np.asarray(observations, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(1, -1)
    n = observations.size
    if n == 0 or forecasts.shape[0] != n:
        return float("nan")
    try:
        ens_q, obs_q, probs, _ = standardized_residual_qq_quantiles(
            observations,
            forecasts,
            quantile_hi=quantile_hi,
            sigma_rel_floor=sigma_rel_floor,
            eps=eps,
        )
    except ValueError:
        return float("nan")
    in_range = (
        (ens_q >= -qq_lim)
        & (ens_q <= qq_lim)
        & (obs_q >= -qq_lim)
        & (obs_q <= qq_lim)
    )
    ens_q = ens_q[in_range]
    obs_q = obs_q[in_range]
    probs = probs[in_range]
    if probs.size < 2:
        return float("nan")
    area = _trapz_yx(np.abs(obs_q - ens_q), probs)
    return float(area / (2.0 * qq_lim))


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


def mean_gaussian_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean over observations of Gaussian likelihood with diagonal variance.

    Per observation: ``(2πσ)^(-1/2) exp(-(y-μ)²/σ²)``.
    Uses the same ``σ`` (ensemble predictive std) as ``mean_gaussian_nll``.

    Computed in log-space (log-sum-exp) so tiny per-point values do not all
    underflow to zero before averaging.  Non-finite inputs/outputs are mapped
    with ``np.nan_to_num`` (NaN → 0).
    """
    y = np.nan_to_num(np.asarray(y, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(np.asarray(mu, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.maximum(
        np.nan_to_num(np.asarray(sigma, dtype=float).ravel(), nan=eps, posinf=eps, neginf=eps),
        eps,
    )
    if not (y.size == mu.size == sigma.size):
        raise ValueError("likelihood: y, mu, sigma must broadcast to same length")
    log_lik = -0.5 * np.log(2.0 * np.pi * sigma) - ((y - mu) ** 2) / (sigma**2)
    log_lik = np.nan_to_num(log_lik, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    finite = np.isfinite(log_lik)
    if not np.any(finite):
        return 0.0
    log_lik_f = log_lik[finite]
    log_max = float(np.max(log_lik_f))
    mean_lik = float(np.exp(log_max) * np.mean(np.exp(log_lik_f - log_max)))
    return float(np.nan_to_num(mean_lik, nan=0.0, posinf=0.0, neginf=0.0))


def _sanitize_likelihood(value: Any) -> float:
    """Map non-finite likelihood scalars to zero for plotting and NPZ storage."""
    return float(np.nan_to_num(float(value), nan=0.0, posinf=0.0, neginf=0.0))


LOO_LOG_PREDICTIVE_KEY = "loo_log_predictive_likelihood"


def mean_loo_gaussian_log_predictive_likelihood(
    y: np.ndarray,
    Y: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """LOO-CV mean log predictive density with diagonal Gaussian forecasts.

    For each observation ``i``, ensemble members are reweighted by their
    Gaussian likelihood on the reduced training targets ``y_{-i}`` (all
    observations except ``i``).  The predictive mean ``μ_i`` and standard
    deviation ``σ_i`` at ``i`` are the weighted moments of ``Y[:, i]``.
    The per-point log score is

    ``-½ log(σ_i²) - (y_i - μ_i)² / (2 σ_i²)``,

    (no ``-½ log(2π)`` normalization constant), and the returned value is
    ``(1/N) Σ_i`` of that expression.
    """
    y = np.nan_to_num(np.asarray(y, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if Y.ndim != 2:
        raise ValueError("LOO log predictive likelihood: Y must be 2D (n_ensemble, n_obs)")
    n_obs = int(Y.shape[1])
    if y.size != n_obs or n_obs < 2:
        return float("nan")

    resid = y[np.newaxis, :] - Y
    sse_total = np.sum(resid**2, axis=1)
    sse_minus_i = sse_total[:, np.newaxis] - resid**2

    tau = float(np.std(y - np.mean(Y, axis=0)))
    if not np.isfinite(tau) or tau <= 0.0:
        tau = eps
    tau = max(tau, eps)

    log_w = -0.5 * sse_minus_i / (tau**2)
    log_w_max = np.max(log_w, axis=0, keepdims=True)
    w = np.exp(log_w - log_w_max)
    w_sum = np.sum(w, axis=0, keepdims=True)
    w = np.where(w_sum > 0.0, w / w_sum, 1.0 / float(Y.shape[0]))

    mu = np.sum(w * Y, axis=0)
    var = np.sum(w * (Y - mu) ** 2, axis=0)
    sigma = np.maximum(np.sqrt(np.maximum(var, 0.0)), eps)

    log_p = -0.5 * np.log(sigma**2) - (y - mu) ** 2 / (2.0 * sigma**2)
    log_p = np.nan_to_num(log_p, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    finite = np.isfinite(log_p)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(log_p[finite]))


_BOOTSTRAP_SAMPLE_FRACTION = 0.05
_N_BOOTSTRAP_REPLICATES = 10
_BOOTSTRAP_SEED = 0

CALIBRATION_SCALAR_METRICS: Tuple[str, ...] = (
    "mae",
    "crps",
    "nll",
    "likelihood",
    LOO_LOG_PREDICTIVE_KEY,
    "miscalibration_area",
    "standardized_miscalibration_area",
    "model_partition_function",
    "average_cost",
    "std_cost",
)


def metric_std_key(metric: str) -> str:
    """Bootstrap standard deviation column name for a scalar calibration metric."""
    return f"{metric}_std"


def _empty_calibration_scalar_metrics() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in CALIBRATION_SCALAR_METRICS:
        out[key] = float("nan")
        out[metric_std_key(key)] = float("nan")
    return out


def _scalar_metrics_from_ensemble_matrix(
    Y: np.ndarray,
    y_ref: np.ndarray,
    T_weight: float,
    T0_ref: float,
) -> Dict[str, float]:
    """Scalar calibration metrics from one ensemble forecast matrix ``Y`` (n_members, n_obs)."""
    mean_pred = np.mean(Y, axis=0)
    y_std = np.std(Y, axis=0)
    F = Y.T
    diff = Y - y_ref
    cost_val = np.sum(diff**2, axis=1)
    return {
        "mae": float(np.mean(np.abs(mean_pred - y_ref))),
        "crps": crps_ensemble(y_ref, F),
        "nll": mean_gaussian_nll(y_ref, mean_pred, y_std),
        "likelihood": _sanitize_likelihood(
            mean_gaussian_likelihood(y_ref, mean_pred, y_std),
        ),
        LOO_LOG_PREDICTIVE_KEY: mean_loo_gaussian_log_predictive_likelihood(y_ref, Y),
        "miscalibration_area": miscalibration_area(y_ref, F),
        "standardized_miscalibration_area": standardized_miscalibration_area(y_ref, F),
        "model_partition_function": float(np.mean(np.exp(-cost_val / (T_weight * T0_ref)))),
        "average_cost": float(np.mean(diff**2)),
        "std_cost": float(np.std(diff**2)),
    }


def _bootstrap_calibration_scalar_metrics(
    Y: np.ndarray,
    y_ref: np.ndarray,
    T_weight: float,
    T0_ref: float,
    *,
    n_replicates: int = _N_BOOTSTRAP_REPLICATES,
    fraction: float = _BOOTSTRAP_SAMPLE_FRACTION,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Mean and std of scalar metrics over bootstrap subsamples of ensemble members."""
    n_members = int(Y.shape[0])
    if n_members <= 0:
        empty = _empty_calibration_scalar_metrics()
        return (
            {k: empty[k] for k in CALIBRATION_SCALAR_METRICS},
            {metric_std_key(k): empty[metric_std_key(k)] for k in CALIBRATION_SCALAR_METRICS},
        )

    if n_members == 1:
        single = _scalar_metrics_from_ensemble_matrix(
            Y, y_ref, T_weight, T0_ref,
        )
        return single, {metric_std_key(k): 0.0 for k in CALIBRATION_SCALAR_METRICS}

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    k = max(1, int(np.ceil(fraction * n_members)))
    replicates = [
        _scalar_metrics_from_ensemble_matrix(
            Y[rng.choice(n_members, size=k, replace=False)],
            y_ref,
            T_weight,
            T0_ref,
        )
        for _ in range(n_replicates)
    ]

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for key in CALIBRATION_SCALAR_METRICS:
        vals = np.array([rep[key] for rep in replicates], dtype=float)
        means[key] = float(np.nanmean(vals))
        n_finite = int(np.count_nonzero(np.isfinite(vals)))
        stds[metric_std_key(key)] = (
            float(np.nanstd(vals, ddof=1)) if n_finite > 1 else 0.0
        )
        if key == "likelihood":
            means[key] = _sanitize_likelihood(means[key])
    return means, stds


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
_RE_POD_INDEX_LABEL = re.compile(r"^POD_energy_POD_index_(\d+)_", re.I)
_RE_ACSF_MW = re.compile(r"^ACSF_hoppings(?:_sk)?_M_(\d+)_W_(\d+)$", re.I)


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


def common_model_label(model_names: Sequence[str]) -> str:
    """
    Shared underscore-delimited prefix across ensemble folder names.

    Examples
    --------
    >>> common_model_label([
    ...     "POD_energy_POD_index_0_09fdb1c2",
    ...     "POD_energy_POD_index_9_e5aa13cf",
    ... ])
    'POD_energy_POD_index'
    >>> common_model_label([
    ...     "ACSF_hoppings_sk_M_12_W_0",
    ...     "ACSF_hoppings_sk_M_10_W_6",
    ... ])
    'ACSF_hoppings_sk'
    """
    names = [str(n).strip() for n in model_names if str(n).strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    token_lists = [n.split("_") for n in names]
    common: List[str] = []
    for parts in zip(*token_lists):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    return "_".join(common) if common else names[0]


def _safe_filename_slug(s: str) -> str:
    """Filesystem-safe slug for model-family labels in figure names."""
    return re.sub(r"[^\w.\-]+", "_", str(s)).strip("_")


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


def _paths_for_stored_control_grid(
    model_name: str,
    technique: str,
    ensemble_root: str,
    control_values: np.ndarray,
    *,
    auto_discover: bool,
    temperatures: Optional[str],
    p_values: Optional[str],
    target: str,
) -> List[Optional[str]]:
    """Map each stored control value to its ensemble pickle path (or ``None``)."""
    grid_cv, grid_paths, _ = discover_control_grid(
        model_name,
        technique,
        ensemble_root,
        auto_discover=auto_discover,
        temperatures=temperatures,
        p_values=p_values,
        target=target,
    )
    path_by_cv = {float(cv): path for cv, path in zip(grid_cv, grid_paths)}
    return [path_by_cv.get(float(cv)) for cv in np.asarray(control_values, dtype=float)]


def npz_needs_metric_refresh(path: str, metric_key: str) -> bool:
    """True when saved metrics lack a usable array for ``metric_key``."""
    if not os.path.isfile(path):
        return False
    z = np.load(path, allow_pickle=True)
    if metric_key not in z.files:
        return True
    values = np.asarray(z[metric_key], dtype=float)
    return not np.any(np.isfinite(values))


def npz_needs_likelihood_refresh(path: str) -> bool:
    """Backward-compatible wrapper for likelihood column refresh checks."""
    return npz_needs_metric_refresh(path, "likelihood")


def npz_needs_bootstrap_std_refresh(path: str) -> bool:
    """True when saved metrics lack bootstrap standard-deviation columns."""
    if not os.path.isfile(path):
        return False
    z = np.load(path, allow_pickle=True)
    for key in CALIBRATION_SCALAR_METRICS:
        if metric_std_key(key) not in z.files:
            return True
    return False


def refresh_metric_column_in_npz(
    model_name: str,
    technique: str,
    target: str,
    metrics_path: str,
    metric_key: str,
    *,
    ensemble_root: str,
    auto_discover: bool,
    temperatures: Optional[str],
    p_values: Optional[str],
    no_t0_fit: bool,
    z_reference_temperature: float,
    sanitize: Optional[Any] = None,
) -> bool:
    """Fill or replace one metric column in an existing calibration NPZ."""
    if not os.path.isfile(metrics_path):
        return False

    z = np.load(metrics_path, allow_pickle=True)
    control_values = np.asarray(z["control_values"], dtype=float)
    paths = _paths_for_stored_control_grid(
        model_name,
        technique,
        ensemble_root,
        control_values,
        auto_discover=auto_discover,
        temperatures=temperatures,
        p_values=p_values,
        target=target,
    )
    if not any(p and os.path.isfile(p) for p in paths):
        print(
            f"Cannot refresh {metric_key} in {metrics_path}: no ensemble pickles.",
            file=sys.stderr,
        )
        return False

    meta = json.loads(str(z["meta_json"][()]))
    T0_ref = float(meta.get("T0_ref", 1.0))
    if not no_t0_fit:
        T0_ref = resolve_T0_ref(model_name, target, ensemble_root)

    values = np.full(control_values.size, np.nan, dtype=float)
    std_key = metric_std_key(metric_key)
    std_values = np.full(control_values.size, np.nan, dtype=float)
    n_ok = 0
    for i, (cv, path) in enumerate(zip(control_values, paths)):
        if path is None or not os.path.isfile(path):
            continue
        T_for_z = float(cv) if technique == "mcmc" else float(z_reference_temperature)
        try:
            m = compute_metrics_one_file(path, target, technique, T_for_z, T0_ref)
            val = m[metric_key]
            if sanitize is not None:
                val = sanitize(val)
            values[i] = float(val)
            std_values[i] = float(m.get(std_key, np.nan))
            n_ok += 1
        except Exception as exc:
            print(
                f"  skip {metric_key} refresh {os.path.basename(path)}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if n_ok == 0:
        print(f"No {metric_key} values refreshed for {metrics_path}", file=sys.stderr)
        return False

    skip_keys = {metric_key, std_key}
    arrays = {key: np.asarray(z[key]) for key in z.files if key not in skip_keys}
    if metric_key == "likelihood":
        arrays[metric_key] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        arrays[metric_key] = values
    arrays[std_key] = std_values
    np.savez_compressed(metrics_path, **arrays)
    print(
        f"Refreshed {metric_key} in {metrics_path} "
        f"({n_ok}/{control_values.size} grid points)",
        flush=True,
    )
    return True


def refresh_likelihood_in_npz(
    model_name: str,
    technique: str,
    target: str,
    metrics_path: str,
    *,
    ensemble_root: str,
    auto_discover: bool,
    temperatures: Optional[str],
    p_values: Optional[str],
    no_t0_fit: bool,
    z_reference_temperature: float,
) -> bool:
    """Fill or replace the ``likelihood`` column in an existing calibration NPZ."""
    return refresh_metric_column_in_npz(
        model_name,
        technique,
        target,
        metrics_path,
        "likelihood",
        ensemble_root=ensemble_root,
        auto_discover=auto_discover,
        temperatures=temperatures,
        p_values=p_values,
        no_t0_fit=no_t0_fit,
        z_reference_temperature=z_reference_temperature,
        sanitize=_sanitize_likelihood,
    )


def refresh_loo_log_predictive_likelihood_in_npz(
    model_name: str,
    technique: str,
    target: str,
    metrics_path: str,
    *,
    ensemble_root: str,
    auto_discover: bool,
    temperatures: Optional[str],
    p_values: Optional[str],
    no_t0_fit: bool,
    z_reference_temperature: float,
) -> bool:
    """Fill or replace the LOO log predictive likelihood column in a calibration NPZ."""
    return refresh_metric_column_in_npz(
        model_name,
        technique,
        target,
        metrics_path,
        LOO_LOG_PREDICTIVE_KEY,
        ensemble_root=ensemble_root,
        auto_discover=auto_discover,
        temperatures=temperatures,
        p_values=p_values,
        no_t0_fit=no_t0_fit,
        z_reference_temperature=z_reference_temperature,
    )


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
        refresh_kwargs = dict(
            ensemble_root=ensemble_root,
            auto_discover=auto_discover,
            temperatures=temperatures,
            p_values=p_values,
            no_t0_fit=no_t0_fit,
            z_reference_temperature=z_reference_temperature,
        )
        if npz_needs_bootstrap_std_refresh(out):
            print(
                f"Refreshing bootstrap metric uncertainties in {out} …",
                flush=True,
            )
            control_values, paths, control_name = discover_control_grid(
                model_name,
                technique,
                ensemble_root,
                auto_discover=auto_discover,
                temperatures=temperatures,
                p_values=p_values,
                target=target,
            )
            if control_values.size > 0 and any(os.path.isfile(p) for p in paths):
                T0_ref = 1.0 if no_t0_fit else resolve_T0_ref(
                    model_name, target, ensemble_root,
                )
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
        if npz_needs_likelihood_refresh(out):
            refresh_likelihood_in_npz(
                model_name, technique, target, out, **refresh_kwargs,
            )
        if npz_needs_metric_refresh(out, LOO_LOG_PREDICTIVE_KEY):
            refresh_loo_log_predictive_likelihood_in_npz(
                model_name, technique, target, out, **refresh_kwargs,
            )
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
        empty = _empty_calibration_scalar_metrics()
        return {
            **empty,
            "y_ref": y_flat,
            "y_mean": None,
            "y_std": None,
            "forecasts": None,
        }
    if y_flat.size != Y.shape[1]:
        L = int(min(y_flat.size, Y.shape[1]))
        print(
            f"  Warning: hopping reference length {y_flat.size} != "
            f"ensemble width {Y.shape[1]}; truncating to {L}",
            file=sys.stderr,
            flush=True,
        )
        y_flat = y_flat[:L]
        Y = Y[:, :L]
    mean_pred = np.mean(Y, axis=0)
    y_std = np.std(Y, axis=0)
    scalars, scalar_stds = _bootstrap_calibration_scalar_metrics(
        Y, y_flat, T_weight, T0_ref,
    )
    F = Y.T
    return {
        **scalars,
        **scalar_stds,
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
        empty = _empty_calibration_scalar_metrics()
        return {
            **empty,
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
    scalars, scalar_stds = _bootstrap_calibration_scalar_metrics(
        ypred_per_atom, ydata_per_atom, T_weight, T0_ref,
    )
    F = ypred_per_atom.T
    return {
        **scalars,
        **scalar_stds,
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
    metric_arrays: Dict[str, np.ndarray] = {}
    for key in CALIBRATION_SCALAR_METRICS:
        metric_arrays[key] = np.full(n, np.nan, dtype=float)
        metric_arrays[metric_std_key(key)] = np.full(n, np.nan, dtype=float)

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
        for key in CALIBRATION_SCALAR_METRICS:
            val = m[key]
            if key == "likelihood":
                val = _sanitize_likelihood(val)
            metric_arrays[key][i] = val
            metric_arrays[metric_std_key(key)][i] = m.get(metric_std_key(key), np.nan)

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
    save_payload: Dict[str, Any] = {
        "control_values": control_values.astype(float),
        "control_name": np.array(control_name),
        "meta_json": np.array(json.dumps(meta)),
    }
    for key in CALIBRATION_SCALAR_METRICS:
        save_payload[key] = metric_arrays[key]
        save_payload[metric_std_key(key)] = metric_arrays[metric_std_key(key)]
    np.savez_compressed(out, **save_payload)
    print(f"Wrote {out}")
    return out


def _miscalibration_on_discovered_grid(
    control_values: np.ndarray,
    paths_by_control: Sequence[str],
    technique: str,
    target: str,
    T0_ref: float,
    z_reference_temperature: float,
    metrics_path: Optional[str] = None,
) -> np.ndarray:
    """Per-grid-point miscalibration area aligned with ``control_values``."""
    cv = np.asarray(control_values, dtype=float)
    n = int(cv.size)
    miscal = np.full(n, np.nan, dtype=float)
    if metrics_path and os.path.isfile(metrics_path):
        d = load_metrics_npz(metrics_path)
        cv_saved = np.asarray(d["control_values"], dtype=float)
        miscal_saved = np.asarray(d["miscalibration_area"], dtype=float)
        from_saved = True
        for i, c in enumerate(cv):
            hits = np.where(np.isclose(cv_saved, c, rtol=0.0, atol=1e-6))[0]
            if hits.size != 1:
                from_saved = False
                break
            miscal[i] = miscal_saved[hits[0]]
        if from_saved and np.any(np.isfinite(miscal)):
            return miscal
        miscal[:] = np.nan

    for i, (c, path) in enumerate(zip(cv, paths_by_control)):
        if not os.path.isfile(path):
            continue
        T_for_z = float(c) if technique == "mcmc" else float(z_reference_temperature)
        try:
            m = compute_metrics_one_file(path, target, technique, T_for_z, T0_ref)
            miscal[i] = m["miscalibration_area"]
        except Exception as exc:
            print(
                f"  [diagnostics] skip {os.path.basename(path)}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return miscal


def _index_min_miscalibration(
    control_values: np.ndarray,
    paths_by_control: Sequence[str],
    technique: str,
    target: str,
    T0_ref: float,
    z_reference_temperature: float,
    metrics_path: Optional[str] = None,
) -> Optional[int]:
    """Grid index with smallest finite miscalibration area on the discovered grid."""
    cv = np.asarray(control_values, dtype=float)
    miscal = _miscalibration_on_discovered_grid(
        cv,
        paths_by_control,
        technique,
        target,
        T0_ref,
        z_reference_temperature,
        metrics_path=metrics_path,
    )
    valid = np.isfinite(cv) & np.isfinite(miscal)
    if not np.any(valid):
        return None
    return int(np.argmin(np.where(valid, miscal, np.inf)))


def _resolve_diagnostic_index(
    technique: str,
    control_values: np.ndarray,
    temperatures_arg: Optional[str],
    p_values_arg: Optional[str],
    diagnostic_index_arg: Optional[int],
    *,
    paths_by_control: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
    T0_ref: float = 1.0,
    z_reference_temperature: float = 1.0,
    metrics_path: Optional[str] = None,
) -> int:
    """Pick grid index for ``--diagnostics``.

    For MCMC, if ``--temperatures`` parses to at least one float, the first
    value selects the closest ``T`` in ``control_values`` (overrides
    ``--diagnostic-index``).  For ``subsamp``, the same applies to the first
    value in ``--p-values``.  Otherwise use ``diagnostic_index_arg`` if set,
    else the grid point with minimum ``miscalibration_area``, else the middle
    of the grid.
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
    if (
        paths_by_control is not None
        and target is not None
        and len(paths_by_control) == n
    ):
        idx = _index_min_miscalibration(
            cv,
            paths_by_control,
            technique,
            target,
            T0_ref,
            z_reference_temperature,
            metrics_path=metrics_path,
        )
        if idx is not None:
            return idx
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
    """Parity, PIT QQ, standardized-residual QQ, and std vs |error| at one grid point."""
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
    p4 = None
    if F is not None and np.asarray(F).size > 0:
        u = pit_values(y_ref, F)
        u_sorted = np.sort(u)
        n_u = u_sorted.size
        uniform_quantiles = np.linspace(0.0, 1.0, n_u)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(uniform_quantiles, u_sorted, "o", ms=3, alpha=0.7, label="QQ")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.fill_between(
            uniform_quantiles,
            uniform_quantiles,
            u_sorted,
            alpha=0.25,
            color="C0",
            linewidth=0,
        )
        ax.set_xlabel("Uniform quantiles", fontdict=CSFONT)
        ax.set_ylabel("empirical quantiles", fontdict=CSFONT)
        ax.legend(prop={"family": CSFONT["fontname"], "size": CSFONT["size"]})
        fig.tight_layout()
        p3 = os.path.join(sub, f"{model_name}_{ctrl_tag}_pit_qq.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)

        ens_q, obs_q, _, n_excl = standardized_residual_qq_quantiles(y_ref, F)
        if n_excl:
            print(
                f"  standardized QQ: excluded {n_excl}/{y_ref.size} points "
                "with degenerate ensemble spread",
                file=sys.stderr,
            )
        qq_lim = STANDARDIZED_QQ_LIM
        in_range = (
            (ens_q >= -qq_lim)
            & (ens_q <= qq_lim)
            & (obs_q >= -qq_lim)
            & (obs_q <= qq_lim)
        )
        ens_q_plot = ens_q[in_range]
        obs_q_plot = obs_q[in_range]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ens_q_plot, obs_q_plot, "o", ms=3, alpha=0.7, label="QQ")
        lims = np.linspace(-qq_lim, qq_lim, 50)
        ax.plot(lims, lims, "k--", lw=1, label="ideal")
        if ens_q_plot.size > 0:
            ax.fill_between(
                ens_q_plot, ens_q_plot, obs_q_plot, alpha=0.25, color="C0", linewidth=0
            )
        ax.set_xlim(-qq_lim, qq_lim)
        ax.set_ylim(-qq_lim, qq_lim)
        ax.set_xlabel("ensemble standardized quantiles", fontdict=CSFONT)
        ax.set_ylabel("empirical quantiles", fontdict=CSFONT)
        ax.legend(prop={"family": CSFONT["fontname"], "size": CSFONT["size"]})
        fig.tight_layout()
        p4 = os.path.join(sub, f"{model_name}_{ctrl_tag}_standardized_residual_qq.png")
        fig.savefig(p4, dpi=150)
        plt.close(fig)

    lines = [f"           {p1}", f"           {p2}"]
    if p3:
        lines.append(f"           {p3}")
    if p4:
        lines.append(f"           {p4}")
    print("Diagnostics:\n" + "\n".join(lines))


# -----------------------------------------------------------------------------
# Plot metrics vs control (single or compare)
# -----------------------------------------------------------------------------


def load_metrics_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"][()]))
    n = int(np.asarray(z["control_values"]).size)
    nan_vec = np.full(n, np.nan, dtype=float)
    zero_vec = np.zeros(n, dtype=float)
    out: Dict[str, Any] = {
        "control_values": np.asarray(z["control_values"]),
        "meta": meta,
    }
    for key in CALIBRATION_SCALAR_METRICS:
        if key in z.files:
            arr = np.asarray(z[key], dtype=float)
            if key == "likelihood":
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            out[key] = arr
        elif key == "likelihood":
            out[key] = zero_vec.copy()
        else:
            out[key] = nan_vec.copy()
        std_key = metric_std_key(key)
        if std_key in z.files:
            out[std_key] = np.asarray(z[std_key], dtype=float)
        else:
            out[std_key] = zero_vec.copy()
    return out


DEFAULT_CALIBRATION_METRICS_DIR = "calibration_metrics"


def optimal_calibration_index(metrics: Dict[str, np.ndarray]) -> Optional[int]:
    """
    Grid index that minimizes ``miscalibration_area`` among finite control values.

    Ensemble selection (compare bars, hyperparameter NLL plots,
    ``resolve_ensemble_pickle``) always uses this index — never min-NLL.
    """
    cv = np.asarray(metrics["control_values"], dtype=float)
    miscal = np.asarray(metrics["miscalibration_area"], dtype=float)
    valid = np.isfinite(cv) & np.isfinite(miscal)
    if not np.any(valid):
        return None
    return int(np.argmin(np.where(valid, miscal, np.inf)))


def optimal_temperature_miscalibration(
    model_name: str,
    metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    technique: str = "mcmc",
    target: str = "energy",
) -> Optional[float]:
    """
    Temperature *T* (or subsampling fraction *p*) that minimizes
    ``miscalibration_area`` on the saved calibration grid.

    Reads ``calibration_metrics/calibration_<model>_<technique>_<target>.npz``
    (from ``plot_bayes_factor.py --calculate``).  Returns ``None`` if the file is
    missing or has no finite miscalibration values.
    """
    path = metrics_npz_path(metrics_dir, model_name, technique, target)
    if not os.path.isfile(path):
        return None
    d = load_metrics_npz(path)
    idx = optimal_calibration_index(d)
    if idx is None:
        return None
    return float(np.asarray(d["control_values"], dtype=float)[idx])


def metrics_at_min_miscalibration(path: str) -> Tuple[float, float, float, float]:
    """
    Return ``(control_value, miscalibration_area, nll, nll_bootstrap_std)`` at the
    grid point with the smallest finite miscalibration area.
    """
    return metric_at_min_miscalibration(path, "nll")


def metric_at_min_miscalibration(path: str, metric: str) -> Tuple[float, float, float, float]:
    """
    Return ``(control_value, miscalibration_area, metric_value, metric_bootstrap_std)``
    at the grid point with the smallest finite miscalibration area.
    """
    d = load_metrics_npz(path)
    idx = optimal_calibration_index(d)
    if idx is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    cv = np.asarray(d["control_values"], dtype=float)
    miscal = np.asarray(d["miscalibration_area"], dtype=float)
    values = np.asarray(d[metric], dtype=float)
    std_values = np.asarray(d.get(metric_std_key(metric), np.zeros_like(values)), dtype=float)
    val = float(values[idx]) if idx < values.size else float("nan")
    std_val = float(std_values[idx]) if idx < std_values.size else float("nan")
    if metric == "likelihood":
        val = _sanitize_likelihood(val)
    elif not np.isfinite(val):
        val = float("nan")
    if not np.isfinite(std_val):
        std_val = float("nan")
    return float(cv[idx]), float(miscal[idx]), val, std_val


def format_compare_bar_label(model_name: str, family_label: str) -> str:
    """Bar-tick label ``family (POD-index)`` for POD models."""
    m = _RE_POD_INDEX_LABEL.match(model_name)
    if m:
        return f"{family_label} ({m.group(1)})"
    mw = _RE_ACSF_MW.match(model_name)
    if mw:
        return f"{family_label} (M={mw.group(1)}, W={mw.group(2)})"
    return model_name


def _pod_hyperparam_search_dir() -> Path:
    """POD hyperparameter-search artifacts (anchored to UQ dir, not CWD)."""
    return UQ_DIR / "pod_hyperparam_search"


def _pod_search_csv_paths() -> List[Path]:
    d = _pod_hyperparam_search_dir()
    out: List[Path] = []
    for name in (
        "pod_hyperparam_search.csv",
        "pod_hyperparam_search_results_tightened.csv",
        "pod_hyperparam_search_results.csv",
    ):
        p = d / name
        if p.is_file():
            out.append(p)
    return out


def _load_pod_search_dataframe() -> Optional[Any]:
    """Pandas DataFrame of POD hyperparameter-search results (tightened CSV first)."""
    try:
        import pandas as pd
    except ImportError:
        return None
    for path in _pod_search_csv_paths():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr, flush=True)
    return None


def _pod_index_from_model_name(model_name: str) -> int:
    m = _RE_POD_INDEX_LABEL.match(model_name)
    return int(m.group(1)) if m else 10**9


def _pod_valid_two_body_radial_values(
    three_body_radial: int,
    three_body_angular: int,
) -> List[int]:
    """Valid 2-body radial counts for a fixed (n3r, n3a) slice of the POD grid."""
    out: List[int] = []
    for n2 in _POD_TWO_BODY_RADIAL_GRID:
        if three_body_radial > n2:
            continue
        if three_body_angular > three_body_radial:
            continue
        out.append(int(n2))
    return out


def _pod_valid_hyperparameter_groups() -> List[Tuple[int, int]]:
    """All valid (three_body_radial, three_body_angular) pairs from the POD grid."""
    groups: List[Tuple[int, int]] = []
    for n3r in _POD_THREE_BODY_RADIAL_GRID:
        for n3a in _POD_THREE_BODY_ANGULAR_GRID:
            if _pod_valid_two_body_radial_values(n3r, n3a):
                groups.append((int(n3r), int(n3a)))
    return groups


def discover_pod_energy_models_from_search(
    ensemble_dir: str = "ensembles",
) -> List[str]:
    """
    Map every row of the tightened POD search CSV to an ensemble folder name.

    Uses ``find_pod_energy_ensemble_folder`` so models are not dropped when
    ``--models POD_energy_POD_index*`` glob expansion is incomplete.
    """
    if find_pod_energy_ensemble_folder is None:
        return []
    df = _load_pod_search_dataframe()
    if df is None or len(df) == 0:
        return []
    names: List[str] = []
    missing = 0
    for pod_idx, row in df.iterrows():
        folder = find_pod_energy_ensemble_folder(
            str(row["hash"]),
            ensemble_dir,
            pod_index=int(pod_idx),
        )
        if folder is None:
            missing += 1
            continue
        names.append(folder)
    if missing:
        print(
            f"  Warning: {missing}/{len(df)} POD search CSV row(s) have no ensemble "
            f"folder under {ensemble_dir!r}",
            file=sys.stderr,
            flush=True,
        )
    return names


def expand_pod_energy_hyperparam_models(
    models: Sequence[str],
    ensemble_dir: str = "ensembles",
) -> List[str]:
    """Union CLI model list with every POD_energy folder indexed by the search CSV."""
    if not any(name.startswith("POD_energy") for name in models):
        return list(models)
    discovered = discover_pod_energy_models_from_search(ensemble_dir)
    if not discovered:
        return list(models)
    merged: List[str] = []
    seen: set[str] = set()
    for name in list(models) + discovered:
        if name not in seen:
            seen.add(name)
            merged.append(name)
    merged.sort(key=lambda n: (_pod_index_from_model_name(n), n))
    if len(merged) != len(models):
        print(
            f"Expanded POD_energy hyperparameter models: {len(models)} -> {len(merged)}",
            flush=True,
        )
    return merged


def _pod_hash_from_model_name(model_name: str) -> Optional[str]:
    m = _RE_POD_INDEX_LABEL.match(model_name)
    if not m:
        return None
    suffix = str(model_name[m.end() :]).strip()
    if suffix and re.fullmatch(r"[0-9a-f]+", suffix, flags=re.I):
        return suffix
    return None


def _pod_row_dict_for_hash(pod_hash: str) -> Optional[dict]:
    df = _load_pod_search_dataframe()
    if df is None:
        return None
    target = str(pod_hash).strip().lower()
    hits = df[df["hash"].astype(str).str.lower() == target]
    if hits.empty:
        return None
    return hits.iloc[0].to_dict()


def _resolve_pod_hyperparam_row(model_name: str) -> Optional[dict]:
    """
    Resolve a POD hyperparameter-search CSV row for ``POD_energy_POD_index_*``.

    Prefer the hash embedded in the ensemble folder name; fall back to the
    ``POD_index`` row in the tightened CSV.
    """
    pod_hash = _pod_hash_from_model_name(model_name)
    if pod_hash:
        row = _pod_row_dict_for_hash(pod_hash)
        if row is not None:
            return row

    m = _RE_POD_INDEX_LABEL.match(model_name)
    if m is None:
        return None
    pod_idx = int(m.group(1))

    if pod_row_for_index is not None:
        try:
            return pod_row_for_index(pod_idx).to_dict()
        except Exception:
            pass
    return None


def pod_hyperparams_from_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return POD search hyperparameters for ``POD_energy_POD_index_*``.

    Keys: ``two_body_radial``, ``three_body_radial``, ``three_body_angular``,
    and ``rcut`` (Å) when present in the search CSV.
    """
    row = _resolve_pod_hyperparam_row(model_name)
    if row is None:
        pod_idx = _pod_index_from_model_name(model_name)
        df = _load_pod_search_dataframe()
        if df is not None and 0 <= pod_idx < len(df):
            row = df.iloc[int(pod_idx)].to_dict()
    if row is None:
        print(
            f"  Warning: could not resolve POD hyperparameters for {model_name!r}",
            file=sys.stderr,
            flush=True,
        )
        return None
    try:
        out: Dict[str, Any] = {
            "two_body_radial": int(row["two_body_radial"]),
            "three_body_radial": int(row["three_body_radial"]),
            "three_body_angular": int(row["three_body_angular"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        print(
            f"  Warning: invalid POD hyperparameter row for {model_name!r}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None
    if "rcut" in row and row["rcut"] is not None and str(row["rcut"]).strip() != "":
        try:
            out["rcut"] = float(row["rcut"])
        except (TypeError, ValueError):
            pass
    return out


def pod_basis_counts_from_model(model_name: str) -> Optional[Tuple[int, int, int]]:
    """
    Return ``(two_body_radial, three_body_radial, three_body_angular)`` for a
    ``POD_energy_POD_index_*`` ensemble folder name.
    """
    hp = pod_hyperparams_from_model(model_name)
    if hp is None:
        return None
    return (
        int(hp["two_body_radial"]),
        int(hp["three_body_radial"]),
        int(hp["three_body_angular"]),
    )


def _format_rcut_slug(rcut: float) -> str:
    """Filename / label slug for a POD cutoff (e.g. ``6`` or ``6.5``)."""
    r = float(rcut)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:g}"


def _pod_family_label_for_rcut(rcut: Optional[float]) -> str:
    """Figure family label; separate plots when rcut differs."""
    if rcut is None or not np.isfinite(float(rcut)):
        return "POD_energy"
    return f"POD_energy_rcut{_format_rcut_slug(float(rcut))}"


def _group_pod_records_by_rcut(
    pod_records: Sequence[Dict[str, Any]],
) -> List[Tuple[Optional[float], List[Dict[str, Any]]]]:
    """Group POD records by ``rcut`` (Å); ``None`` if missing. Sorted by rcut."""
    groups: Dict[Optional[float], List[Dict[str, Any]]] = {}
    for rec in pod_records:
        rcut_raw = rec.get("rcut", None)
        key: Optional[float]
        if rcut_raw is None or (
            isinstance(rcut_raw, float) and not np.isfinite(rcut_raw)
        ):
            key = None
        else:
            try:
                key = float(rcut_raw)
            except (TypeError, ValueError):
                key = None
        groups.setdefault(key, []).append(rec)
    def _sort_key(item: Tuple[Optional[float], List[Dict[str, Any]]]):
        rcut, _ = item
        return (rcut is None, float(rcut) if rcut is not None else 0.0)

    return sorted(groups.items(), key=_sort_key)


def acsf_mw_from_model(model_name: str) -> Optional[Tuple[int, int]]:
    """Return ``(M, W)`` from an ``ACSF_hoppings*_M_*_W_*`` ensemble folder name."""
    m = _RE_ACSF_MW.match(model_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def acsf_family_prefix(model_name: str) -> Optional[str]:
    if model_name.startswith("ACSF_hoppings_sk"):
        return "ACSF_hoppings_sk"
    if model_name.startswith("ACSF_hoppings"):
        return "ACSF_hoppings"
    return None


def collect_hyperparam_calibration_records(
    entries: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build POD and ACSF records with NLL and likelihood at min-miscalibration.

    Each record includes ``model_name``, ``nll``, ``likelihood``, hyperparameters,
    and the control value used.
    """
    pod_records: List[Dict[str, Any]] = []
    acsf_records: List[Dict[str, Any]] = []

    for model_name, path in entries:
        cv, miscal, nll, nll_std = metric_at_min_miscalibration(path, "nll")
        _, _, likelihood, likelihood_std = metric_at_min_miscalibration(path, "likelihood")
        _, _, loo_log_pred, loo_log_pred_std = metric_at_min_miscalibration(
            path, LOO_LOG_PREDICTIVE_KEY,
        )
        likelihood = _sanitize_likelihood(likelihood)
        if not np.isfinite(nll) and likelihood <= 0.0 and not np.isfinite(loo_log_pred):
            print(
                f"  Warning: no finite NLL/likelihood/LOO at min-miscalibration for "
                f"{model_name!r}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"  @ min miscalibration: {model_name}  control={cv:g}  "
            f"M={miscal:.4f}  NLL={nll:.4f}  likelihood={likelihood:.4e}  "
            f"LOO log pred={loo_log_pred:.4f}",
            flush=True,
        )

        if model_name.startswith("POD_energy"):
            hp = pod_hyperparams_from_model(model_name)
            if hp is None:
                continue
            rec: Dict[str, Any] = {
                "model_name": model_name,
                "nll": float(nll),
                "nll_std": float(nll_std),
                "likelihood": likelihood,
                "likelihood_std": float(likelihood_std),
                LOO_LOG_PREDICTIVE_KEY: float(loo_log_pred),
                f"{LOO_LOG_PREDICTIVE_KEY}_std": float(loo_log_pred_std),
                "control_value": float(cv),
                "miscalibration_area": float(miscal),
                "two_body_radial": int(hp["two_body_radial"]),
                "three_body_radial": int(hp["three_body_radial"]),
                "three_body_angular": int(hp["three_body_angular"]),
            }
            if "rcut" in hp:
                rec["rcut"] = float(hp["rcut"])
            pod_records.append(rec)
            continue

        family = acsf_family_prefix(model_name)
        if family is not None:
            mw = acsf_mw_from_model(model_name)
            if mw is None:
                continue
            m_val, w_val = mw
            acsf_records.append(
                {
                    "model_name": model_name,
                    "family": family,
                    "nll": float(nll),
                    "nll_std": float(nll_std),
                    "likelihood": likelihood,
                    "likelihood_std": float(likelihood_std),
                    LOO_LOG_PREDICTIVE_KEY: float(loo_log_pred),
                    f"{LOO_LOG_PREDICTIVE_KEY}_std": float(loo_log_pred_std),
                    "control_value": float(cv),
                    "miscalibration_area": float(miscal),
                    "M": m_val,
                    "W": w_val,
                },
            )

    return pod_records, acsf_records


def collect_nll_hyperparam_records(
    entries: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Backward-compatible alias for :func:`collect_hyperparam_calibration_records`."""
    return collect_hyperparam_calibration_records(entries)


def _plot_pod_metric_vs_two_body_radial(
    pod_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str,
    metric_key: str,
    ylabel: str,
    filename_suffix: str,
    log_y: bool = False,
    dpi: int = 150,
    show_title: bool = True,
    legend_outside: bool = False,
) -> Optional[str]:
    """Metric vs 2-body radial basis count, grouped by 3-body radial / angular."""
    if not pod_records:
        return None

    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in pod_records:
        key = (int(rec["three_body_radial"]), int(rec["three_body_angular"]))
        groups.setdefault(key, []).append(rec)

    expected_groups = _pod_valid_hyperparameter_groups()
    plotted_keys = {(int(r["three_body_radial"]), int(r["three_body_angular"])) for r in pod_records}
    missing_groups = [g for g in expected_groups if g not in plotted_keys]
    if missing_groups:
        print(
            "  POD hyperparameter plot: no models for grid group(s) "
            + ", ".join(f"(n3r={a}, l3a={b})" for a, b in missing_groups),
            flush=True,
        )

    fig_w = 14.0
    fig_h = 8.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for (n3r, n3a) in sorted(groups):
        pts = sorted(groups[(n3r, n3a)], key=lambda r: int(r["two_body_radial"]))
        x = np.array([int(p["two_body_radial"]) for p in pts], dtype=float)
        y = np.array([float(p[metric_key]) for p in pts], dtype=float)
        yerr = np.array(
            [float(p.get(metric_std_key(metric_key), 0.0)) for p in pts],
            dtype=float,
        )
        expected_x = _pod_valid_two_body_radial_values(n3r, n3a)
        missing_x = sorted(set(expected_x) - set(int(v) for v in x))
        if missing_x:
            print(
                f"  POD hyperparameter plot: group (n3r={n3r}, l3a={n3a}) "
                f"missing 2-body radial value(s) {missing_x}",
                flush=True,
            )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            lw=1.8,
            markersize=6,
            capsize=3,
            label=rf"$N_{{\mathrm{{3b}}}}={int(n3r) * int(n3a)}$",
        )

    ax.set_xticks(list(_POD_TWO_BODY_RADIAL_GRID))
    ax.set_xlim(min(_POD_TWO_BODY_RADIAL_GRID) - 0.5, max(_POD_TWO_BODY_RADIAL_GRID) + 0.5)
    tech_label = _SAMPLING_LABEL.get(technique, technique)
    ax.set_xlabel(r"$n_{\mathrm{rad}}$ (2-body radial basis functions)")
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(
            f"{family_label} — {plot_target}\n"
            f"{ylabel} at best-calibration {tech_label} vs 2-body radial count"
        )
    if log_y:
        all_y = np.concatenate(
            [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines],
        ) if ax.lines else np.array([], dtype=float)
        if np.any(all_y > 0):
            ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)
        fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    else:
        ax.legend(loc="best")
        fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    tech_slug = re.sub(r"[^\w]+", "_", technique)
    out = os.path.join(
        figures_dir,
        f"{_safe_filename_slug(family_label)}_{filename_suffix}_{plot_target}_{tech_slug}.png",
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(
        f"Wrote {out}  ({len(pod_records)} models, {len(groups)} grid group(s))",
        flush=True,
    )
    return out


def _plot_acsf_metric_vs_M(
    acsf_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str,
    metric_key: str,
    ylabel: str,
    filename_suffix: str,
    log_y: bool = False,
    dpi: int = 150,
    show_title: bool = True,
    legend_outside: bool = False,
) -> Optional[str]:
    """Metric vs ``M`` for ACSF hopping models, grouped by fixed ``W``."""
    if not acsf_records:
        return None

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for rec in acsf_records:
        groups.setdefault(int(rec["W"]), []).append(rec)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for w_val in sorted(groups):
        pts = sorted(groups[w_val], key=lambda r: int(r["M"]))
        x = np.array([int(p["M"]) for p in pts], dtype=float)
        y = np.array([float(p[metric_key]) for p in pts], dtype=float)
        yerr = np.array(
            [float(p.get(metric_std_key(metric_key), 0.0)) for p in pts],
            dtype=float,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            lw=1.8,
            markersize=6,
            capsize=3,
            label=rf"$W={w_val}$",
        )

    tech_label = _SAMPLING_LABEL.get(technique, technique)
    ax.set_xlabel(r"$M$ (radial basis functions)")
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(
            f"{family_label} — {plot_target}\n"
            f"{ylabel} at best-calibration {tech_label} vs $M$"
        )
    if log_y:
        all_y = np.concatenate(
            [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines],
        )
        if np.any(all_y > 0):
            ax.set_yscale("log")
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    else:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    tech_slug = re.sub(r"[^\w]+", "_", technique)
    out = os.path.join(
        figures_dir,
        f"{_safe_filename_slug(family_label)}_{filename_suffix}_{plot_target}_{tech_slug}.png",
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_pod_nll_vs_two_body_radial(
    pod_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str = "POD_energy",
    dpi: int = 150,
) -> Optional[str]:
    """NLL vs 2-body radial basis count, grouped by 3-body radial / angular."""
    return _plot_pod_metric_vs_two_body_radial(
        pod_records,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        family_label=family_label,
        metric_key="nll",
        ylabel="NLL",
        filename_suffix="nll_vs_two_body_radial",
        log_y=False,
        dpi=dpi,
        show_title=False,
        legend_outside=True,
    )


def plot_pod_likelihood_vs_two_body_radial(
    pod_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str = "POD_energy",
    dpi: int = 150,
) -> Optional[str]:
    """Likelihood vs 2-body radial basis count, grouped by 3-body radial / angular."""
    return _plot_pod_metric_vs_two_body_radial(
        pod_records,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        family_label=family_label,
        metric_key="likelihood",
        ylabel=r"$\langle \mathcal{L} \rangle$",
        filename_suffix="likelihood_vs_two_body_radial",
        log_y=True,
        dpi=dpi,
    )


def plot_acsf_nll_vs_M(
    acsf_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str,
    dpi: int = 150,
) -> Optional[str]:
    """NLL vs ``M`` for ACSF hopping models, grouped by fixed ``W``."""
    return _plot_acsf_metric_vs_M(
        acsf_records,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        family_label=family_label,
        metric_key="nll",
        ylabel="NLL",
        filename_suffix="nll_vs_M",
        log_y=False,
        dpi=dpi,
        show_title=False,
        legend_outside=True,
    )


def plot_acsf_likelihood_vs_M(
    acsf_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str,
    dpi: int = 150,
) -> Optional[str]:
    """Likelihood vs ``M`` for ACSF hopping models, grouped by fixed ``W``."""
    return _plot_acsf_metric_vs_M(
        acsf_records,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        family_label=family_label,
        metric_key="likelihood",
        ylabel=r"$\langle \mathcal{L} \rangle$",
        filename_suffix="likelihood_vs_M",
        log_y=True,
        dpi=dpi,
    )


def plot_pod_loo_log_predictive_vs_two_body_radial(
    pod_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str = "POD_energy",
    dpi: int = 150,
) -> Optional[str]:
    """LOO log predictive likelihood vs 2-body radial count."""
    return _plot_pod_metric_vs_two_body_radial(
        pod_records,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        family_label=family_label,
        metric_key=LOO_LOG_PREDICTIVE_KEY,
        ylabel=r"$L_{\mathrm{LOO}}$",
        filename_suffix="loo_log_predictive_vs_two_body_radial",
        log_y=False,
        dpi=dpi,
    )


def plot_acsf_loo_log_predictive_vs_M(
    acsf_records: Sequence[Dict[str, Any]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    family_label: str,
    dpi: int = 150,
) -> Optional[str]:
    """LOO log predictive likelihood vs ``M`` for ACSF hopping models."""
    return _plot_acsf_metric_vs_M(
        acsf_records,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        family_label=family_label,
        metric_key=LOO_LOG_PREDICTIVE_KEY,
        ylabel=r"$L_{\mathrm{LOO}}$",
        filename_suffix="loo_log_predictive_vs_M",
        log_y=False,
        dpi=dpi,
    )


def plot_calibration_hyperparam_figures(
    entries: Sequence[Tuple[str, str]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    plot_nll: bool = False,
    plot_likelihood: bool = False,
    plot_loo: bool = False,
    dpi: int = 150,
) -> None:
    """Generate POD/ACSF hyperparameter figures for NLL, likelihood, and/or LOO."""
    if not plot_nll and not plot_likelihood and not plot_loo:
        return

    pod_records, acsf_records = collect_hyperparam_calibration_records(entries)

    if plot_nll:
        pod_nll = [r for r in pod_records if np.isfinite(r.get("nll", np.nan))]
        if pod_nll:
            for rcut, recs in _group_pod_records_by_rcut(pod_nll):
                plot_pod_nll_vs_two_body_radial(
                    recs,
                    figures_dir,
                    plot_target=plot_target,
                    technique=technique,
                    family_label=_pod_family_label_for_rcut(rcut),
                    dpi=dpi,
                )
        elif pod_records:
            print(
                "No POD_energy models with finite NLL at min miscalibration "
                "for hyperparameter plot.",
                flush=True,
            )
        else:
            print(
                "No POD_energy models with resolvable hyperparameters for "
                "NLL hyperparameter plot.",
                flush=True,
            )

        acsf_by_family: Dict[str, List[Dict[str, Any]]] = {}
        for rec in acsf_records:
            acsf_by_family.setdefault(str(rec["family"]), []).append(rec)
        for family, recs in sorted(acsf_by_family.items()):
            plot_acsf_nll_vs_M(
                recs,
                figures_dir,
                plot_target=plot_target,
                technique=technique,
                family_label=family,
                dpi=dpi,
            )
        if not acsf_records:
            print("No ACSF_hoppings models with finite NLL for hyperparameter plot.", flush=True)

    if plot_likelihood:
        pod_lik = [r for r in pod_records if r.get("likelihood", 0.0) > 0.0]
        if pod_lik:
            for rcut, recs in _group_pod_records_by_rcut(pod_lik):
                plot_pod_likelihood_vs_two_body_radial(
                    recs,
                    figures_dir,
                    plot_target=plot_target,
                    technique=technique,
                    family_label=_pod_family_label_for_rcut(rcut),
                    dpi=dpi,
                )
        else:
            print(
                "No POD_energy models with finite likelihood for hyperparameter plot "
                "(rerun --calculate to populate likelihood arrays).",
                flush=True,
            )

        acsf_by_family = {}
        for rec in acsf_records:
            if rec.get("likelihood", 0.0) > 0.0:
                acsf_by_family.setdefault(str(rec["family"]), []).append(rec)
        for family, recs in sorted(acsf_by_family.items()):
            plot_acsf_likelihood_vs_M(
                recs,
                figures_dir,
                plot_target=plot_target,
                technique=technique,
                family_label=family,
                dpi=dpi,
            )
        if not acsf_by_family:
            print(
                "No ACSF_hoppings models with finite likelihood for hyperparameter plot "
                "(rerun --calculate to populate likelihood arrays).",
                flush=True,
            )

    if plot_loo:
        pod_loo = [
            r for r in pod_records
            if np.isfinite(r.get(LOO_LOG_PREDICTIVE_KEY, np.nan))
        ]
        if pod_loo:
            for rcut, recs in _group_pod_records_by_rcut(pod_loo):
                plot_pod_loo_log_predictive_vs_two_body_radial(
                    recs,
                    figures_dir,
                    plot_target=plot_target,
                    technique=technique,
                    family_label=_pod_family_label_for_rcut(rcut),
                    dpi=dpi,
                )
        else:
            print(
                "No POD_energy models with finite LOO log predictive likelihood "
                "(rerun --calculate to populate arrays).",
                flush=True,
            )

        acsf_by_family = {}
        for rec in acsf_records:
            if np.isfinite(rec.get(LOO_LOG_PREDICTIVE_KEY, np.nan)):
                acsf_by_family.setdefault(str(rec["family"]), []).append(rec)
        for family, recs in sorted(acsf_by_family.items()):
            plot_acsf_loo_log_predictive_vs_M(
                recs,
                figures_dir,
                plot_target=plot_target,
                technique=technique,
                family_label=family,
                dpi=dpi,
            )
        if not acsf_by_family:
            print(
                "No ACSF_hoppings models with finite LOO log predictive likelihood "
                "(rerun --calculate to populate arrays).",
                flush=True,
            )


def plot_nll_hyperparam_figures(
    entries: Sequence[Tuple[str, str]],
    figures_dir: str,
    *,
    plot_target: str,
    technique: str,
    dpi: int = 150,
) -> None:
    """Generate POD and ACSF NLL-vs-hyperparameter figures from metric NPZs."""
    plot_calibration_hyperparam_figures(
        entries,
        figures_dir,
        plot_target=plot_target,
        technique=technique,
        plot_nll=True,
        plot_likelihood=False,
        dpi=dpi,
    )


def plot_nll_at_min_miscalibration_bar(
    entries: List[Tuple[str, str]],
    figures_dir: str,
    *,
    family_label: str,
    plot_target: str,
    technique: str,
    dpi: int = 150,
) -> Optional[str]:
    """
    Bar chart of NLL at each model's best-calibration control value.

    The control value per model is the grid point that minimizes
    ``miscalibration_area`` on that model's saved metrics curve.
    """
    if not entries:
        return None

    labels: List[str] = []
    nll_vals: List[float] = []
    nll_stds: List[float] = []
    miscals: List[float] = []

    for model_name, path in entries:
        cv, miscal, nll, nll_std = metrics_at_min_miscalibration(path)
        if not np.isfinite(nll):
            print(
                f"  Warning: no finite NLL at min-miscalibration point for {model_name!r}",
                file=sys.stderr,
                flush=True,
            )
            continue
        labels.append(format_compare_bar_label(model_name, family_label))
        nll_vals.append(float(nll))
        nll_stds.append(float(nll_std) if np.isfinite(nll_std) else 0.0)
        miscals.append(float(miscal))
        print(
            f"  NLL @ best calibration: {labels[-1]}  "
            f"control={cv:g}  NLL={nll:.4f}  M={miscal:.4f}",
            flush=True,
        )

    if not labels:
        print("No NLL values at min-miscalibration points to plot.", file=sys.stderr)
        return None

    control_name = "T" if technique == "mcmc" else "p"
    tech_label = _SAMPLING_LABEL.get(technique, technique)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(labels)), 5.0))
    x = np.arange(len(labels))
    ax.bar(
        x,
        nll_vals,
        yerr=nll_stds,
        capsize=3,
        color="steelblue",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("NLL")
    ax.set_title(
        f"{family_label} — {plot_target}\n"
        f"NLL at {control_name} minimizing miscalibration ({tech_label})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    tech_slug = re.sub(r"[^\w]+", "_", technique)
    family_slug = _safe_filename_slug(family_label)
    out = os.path.join(
        figures_dir,
        f"{family_slug}_compare_nll_at_min_miscalibration_{tech_slug}.png",
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


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
    *,
    family_label: Optional[str] = None,
) -> str:
    """
    entries : list of (label, path_to_npz)
    metric : model_partition_function | crps | nll | mae | miscalibration_area
        | standardized_miscalibration_area
    x_scale : linear | log
    """
    plt.figure(figsize=(8, 5))
    technique_key: Optional[str] = None
    for _label, path in entries:
        d = load_metrics_npz(path)
        if technique_key is None:
            technique_key = str(d["meta"].get("technique", "mcmc"))
        x = d["control_values"]
        y = np.asarray(d[metric], dtype=float)
        yerr = np.asarray(d.get(metric_std_key(metric), np.zeros_like(y)), dtype=float)
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
        plt.errorbar(xplot, y, yerr=yerr, marker="o", ms=4, capsize=3)

    plt.xlabel(xlab)
    ylab = {
        "model_partition_function": "model_partition_function (Z)",
        "miscalibration_area": r"$\mathcal{M}$",
        "standardized_miscalibration_area": r"$\mathcal{M}_{\mathrm{std}}$",
    }.get(metric, metric)
    plt.ylabel(ylab)
    if metric in ("miscalibration_area", "standardized_miscalibration_area"):
        plt.ylim(0.0, 1.0)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    safe_metric = re.sub(r"[^\w]+", "_", metric)
    tech_slug = re.sub(r"[^\w]+", "_", technique_key or "mcmc")
    if family_label:
        family_slug = _safe_filename_slug(family_label)
        fname = f"{family_slug}_compare_{safe_metric}_{x_scale}_{tech_slug}.png"
    else:
        fname = f"compare_{safe_metric}_{x_scale}_{tech_slug}.png"
    out = os.path.join(figures_dir, fname)
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
             "(mcmc) or --p-values (subsamp) is unset; overrides min-miscalibration default.",
    )
    p.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric key: model_partition_function, crps, nll, mae, miscalibration_area, "
        "standardized_miscalibration_area. "
        "Repeatable. Default: Z, crps, nll, miscalibration_area, "
        "standardized_miscalibration_area.",
    )
    p.add_argument(
        "--x-scale",
        choices=("linear", "log"),
        default="log",
        help="X-axis for metric curves: log → log10(T) or log10(p).",
    )
    p.add_argument(
        "--plot-nll-hyperparams",
        action="store_true",
        help=(
            "Plot NLL at best calibration vs hyperparameters: POD_energy NLL vs "
            "2-body radial count (one figure per rcut; curves per 3-body "
            "radial/angular); ACSF_hoppings* NLL vs M (curves per W)."
        ),
    )

    p.add_argument(
        "--plot-likelihood-hyperparams",
        action="store_true",
        help=(
            "Plot mean Gaussian likelihood at best calibration vs hyperparameters: "
            "POD_energy vs 2-body radial count (one figure per rcut; curves per "
            "3-body radial/angular); ACSF_hoppings* vs M (curves per W). Requires "
            "likelihood in saved calibration NPZs (--calculate)."
        ),
    )

    p.add_argument(
        "--plot-loo-hyperparams",
        action="store_true",
        help=(
            "Plot LOO-CV mean log predictive likelihood at best calibration vs "
            "hyperparameters (same layout as NLL/likelihood; POD one figure per "
            "rcut). Requires loo_log_predictive_likelihood in saved calibration "
            "NPZs (--calculate)."
        ),
    )

    args = p.parse_args()
    if (
        not args.calculate
        and not args.plot_metrics
        and not args.diagnostics
        and not args.plot_nll_hyperparams
        and not args.plot_likelihood_hyperparams
        and not args.plot_loo_hyperparams
    ):
        p.error(
            "Select at least one of: --calculate, --plot-metrics, "
            "--diagnostics, --plot-nll-hyperparams, --plot-likelihood-hyperparams, "
            "--plot-loo-hyperparams"
        )

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
                metrics_path = metrics_npz_path(
                    args.metrics_dir, model_name, technique, target,
                )
                diag_idx = _resolve_diagnostic_index(
                    technique,
                    control_values,
                    args.temperatures,
                    args.p_values,
                    args.diagnostic_index,
                    paths_by_control=paths,
                    target=target,
                    T0_ref=T0_ref,
                    z_reference_temperature=args.z_reference_temperature,
                    metrics_path=metrics_path,
                )
                tv_parsed = _parse_float_list(args.temperatures)
                pv_parsed = _parse_float_list(args.p_values)
                if technique == "mcmc" and tv_parsed is not None and tv_parsed.size > 0:
                    print(
                        f"[diagnostics] {model_name}: T_request={float(tv_parsed[0]):g} → "
                        f"T={float(control_values[diag_idx]):g} (index {diag_idx})",
                        flush=True,
                    )
                elif technique == "subsamp" and pv_parsed is not None and pv_parsed.size > 0:
                    print(
                        f"[diagnostics] {model_name}: p_request={float(pv_parsed[0]):g} → "
                        f"p={float(control_values[diag_idx]):g} (index {diag_idx})",
                        flush=True,
                    )
                elif args.diagnostic_index is None:
                    miscal = _miscalibration_on_discovered_grid(
                        control_values,
                        paths,
                        technique,
                        target,
                        T0_ref,
                        args.z_reference_temperature,
                        metrics_path=metrics_path,
                    )
                    print(
                        f"[diagnostics] {model_name}: min miscalibration → "
                        f"{control_name}={float(control_values[diag_idx]):g} "
                        f"(miscalibration_area={float(miscal[diag_idx]):g}, "
                        f"index {diag_idx})",
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
            "standardized_miscalibration_area",
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
                compare_label = common_model_label([name for name, _ in entries])
                metrics_for_curves = [m for m in metrics_to_plot if m != "nll"]
                for mname in metrics_for_curves:
                    plot_metric_curves(
                        entries,
                        mname,
                        args.figures_dir,
                        args.x_scale,
                        family_label=compare_label,
                    )
                if "nll" in metrics_to_plot:
                    plot_nll_at_min_miscalibration_bar(
                        entries,
                        args.figures_dir,
                        family_label=compare_label,
                        plot_target=plot_target,
                        technique=technique,
                        dpi=150,
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
                    )

    if args.plot_nll_hyperparams or args.plot_likelihood_hyperparams or args.plot_loo_hyperparams:
        hp_models = expand_pod_energy_hyperparam_models(models, ensemble_root)
        hp_entries: List[Tuple[str, str]] = []
        for model_name in hp_models:
            plot_target = resolve_target_for_model(
                model_name,
                target_arg=args.target,
                ensemble_root=ensemble_root,
                metrics_dir=args.metrics_dir,
                technique=technique,
            )
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
            hp_entries.append((model_name, path))
        if not hp_entries:
            print(
                "No calibration metrics for hyperparameter plots.",
                file=sys.stderr,
            )
        else:
            plot_target = resolve_target_for_model(
                hp_entries[0][0],
                target_arg=args.target,
                ensemble_root=ensemble_root,
                metrics_dir=args.metrics_dir,
                technique=technique,
            )
            plot_calibration_hyperparam_figures(
                hp_entries,
                args.figures_dir,
                plot_target=plot_target,
                technique=technique,
                plot_nll=args.plot_nll_hyperparams,
                plot_likelihood=args.plot_likelihood_hyperparams,
                plot_loo=args.plot_loo_hyperparams,
            )


if __name__ == "__main__":
    main()
