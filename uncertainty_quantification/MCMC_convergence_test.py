#!/usr/bin/env python3
"""
Systematic MCMC convergence study for ``ACSF_hoppings_sk``.

Sweeps ACSF size ``(M, W)`` (hence ``n_params``) and the number of emcee
walkers ``n_walkers = mult × n_params`` with ``mult ∈ [2, 10]``.  For each
``(M, W, n_walkers)`` a chain of length ``burnin + N_ref`` is run; the first
``burnin`` steps (default 500) are discarded.  At checkpoints ``N ≤ N_ref``
on the post-burn-in chain:

**Parameter criterion** (one value per coefficient *i*)::

    z_i(N) = |μ_i(N) − μ_i(N_ref)| / σ_i(N_ref)

**Hopping predictive criterion** (one value per training hopping *j*)::

    z_j(N) = |⟨t_j⟩(N) − ⟨t_j⟩(N_ref)| / σ_j(N_ref)

where means / stds are over the flat walker ensemble after *N* (or *N_ref*)
steps.  Convergence is declared when the **mean** of ``z`` over *i* (or *j*)
Convergence is declared from a **local log–log** fit around the discrete
threshold crossing of mean ``z(N)``; ``N_conv`` is where that fit crosses
``--tol`` (default ``0.1``).

Training hoppings are randomly subsampled to ``--max-hop-eval`` (default 500)
pairs; the linear SK model is **refit in memory** on that subset to set
``C0`` / temperature (the refit is not written to ``best_fit_params/``).

Plots (under ``MCMC_convergence/``)
-----------------------------------
* Per run: ``z(N)`` vs steps — transparent gray for each parameter / hopping
  observation, bold black for the mean over that set.
* Summary: steps-to-tolerance vs ``n_params``.

Examples
--------
::

    cd uncertainty_quantification
    python MCMC_convergence_test.py --quick
    python MCMC_convergence_test.py -M 6 10 -W 0 3 --walker-mult 2 4 6 8 10 \\
        --n-ref 2000 --beta 0.01
    python MCMC_convergence_test.py --replay MCMC_convergence/study_*.npz
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import emcee
import matplotlib
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.chdir(HERE)

from blg_model_builder.EMCEE_generate_ensemble import (  # noqa: E402
    MCMC_DEFAULTS,
    _clamp_cpu_thread_env_for_lammps_emcee,
    _log_prob_worker,
    _ravel_observation_blocks,
    _set_worker_globals,
    get_C0,
    log_probability,
)
from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs  # noqa: E402
from blg_model_builder.model_fit import fit_acsf_linear_hopping  # noqa: E402

CSFONT = {"fontname": "sans-serif", "size": 18}
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": 12,
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)

DEFAULT_OUTPUT_DIR = HERE / "MCMC_convergence"
DEFAULT_TOL = 0.1
DEFAULT_BETA = 1.0
DEFAULT_N_REF = 8000
DEFAULT_BURNIN = 500
DEFAULT_M = [6, 8, 10]
DEFAULT_W = [0, 3, 6]
DEFAULT_WALKER_MULT = [2] #, 4, 6, 8]
DEFAULT_MAX_HOP_EVAL = 500
DEFAULT_MAX_PLOT_CURVES = 200
MODEL_BASE = "ACSF_hoppings_sk"


# ---------------------------------------------------------------------------
# Model / data helpers
# ---------------------------------------------------------------------------

def n_params_acsf_sk(M: int, W: int) -> int:
    """SK ACSF coefficient count: ``2 * (M + M·W)``."""
    return int(2 * (int(M) + int(M) * int(W)))


def model_name_for(M: int, W: int) -> str:
    return f"{MODEL_BASE}_M_{int(M)}_W_{int(W)}"


def stack_hopping_descriptors(x_hopping: Sequence) -> np.ndarray:
    """Stack list-of-structure descriptor blocks → ``(n_obs, n_feat)``."""
    blocks = []
    for blk in x_hopping:
        a = np.asarray(blk, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2 or a.shape[0] == 0:
            continue
        blocks.append(a)
    if not blocks:
        raise ValueError("No hopping descriptor blocks found.")
    return np.concatenate(blocks, axis=0)


def subsample_hopping_training(
    x_hopping: Sequence,
    y_hopping: Sequence,
    n_max: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack and randomly keep at most ``n_max`` hopping rows.

    Returns
    -------
    A, y : ``(n_keep, n_feat)``, ``(n_keep,)``
    """
    A = stack_hopping_descriptors(x_hopping)
    y = _ravel_observation_blocks(y_hopping)
    if A.shape[0] != int(y.size):
        raise ValueError(
            f"Hopping X/y length mismatch: {A.shape[0]} descriptors vs {y.size} targets"
        )
    n_max = int(n_max)
    if n_max <= 0 or A.shape[0] <= n_max:
        return A, y
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(A.shape[0], size=n_max, replace=False))
    return A[idx], y[idx]


def refit_hopping_on_subset(
    A: np.ndarray,
    y: np.ndarray,
    *,
    param_bound: float = 1e4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    In-memory ACSF-SK linear refit on a hopping subset (not written to disk).

    Returns
    -------
    params, bounds
        ``params`` shape ``(n_feat,)``; ``bounds`` shape ``(n_feat, 2)``.
    """
    w, _yp = fit_acsf_linear_hopping([A], [y])
    params = np.asarray(w, dtype=float).ravel()
    n_par = int(params.size)
    bounds = np.array([[-param_bound, param_bound]] * n_par, dtype=float)
    return params, bounds


def _subsample_pool_initializer(calc, xtrain, ytrain, T, weights, params, bounds):
    """Worker init that uses the *already subsampled* training tables."""
    _clamp_cpu_thread_env_for_lammps_emcee()
    _set_worker_globals(calc, xtrain, ytrain, T, weights, params, bounds)


def checkpoint_grid(n_ref: int, n_checkpoints: int) -> np.ndarray:
    """Integer step counts in ``[2, n_ref]``, denser early, always includes ``n_ref``."""
    n_ref = int(n_ref)
    n_checkpoints = max(2, int(n_checkpoints))
    if n_ref < 2:
        raise ValueError("n_ref must be >= 2")
    raw = np.unique(
        np.round(np.geomspace(2, n_ref, num=n_checkpoints)).astype(int)
    )
    if raw[-1] != n_ref:
        raw = np.append(raw, n_ref)
    return raw[raw >= 2]


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------

def run_emcee_chain(
    *,
    xdata_train: dict,
    ydata_train: dict,
    Temperature: float,
    calc: dict,
    weights: dict,
    params: dict,
    bounds: dict,
    nwalkers: int,
    n_steps: int,
    step_size: float,
    seed: int,
    pool=None,
) -> Tuple[np.ndarray, float]:
    """
    Run emcee and return ``(chain, mean_acceptance)``.

    ``chain`` has shape ``(n_steps, n_walkers, n_dim)``.
    """
    key = "hopping"
    theta0 = np.asarray(params[key], dtype=float).ravel()
    ndim = int(theta0.size)
    nwalkers = int(nwalkers)
    if nwalkers < 2 * ndim:
        raise ValueError(
            f"emcee requires nwalkers >= 2*ndim; got nwalkers={nwalkers}, ndim={ndim}"
        )

    rng = np.random.default_rng(int(seed))
    scale = MCMC_DEFAULTS["walker_init_scale"] * np.maximum(np.abs(theta0), 1e-8)
    p0 = theta0[np.newaxis, :] + scale[np.newaxis, :] * rng.normal(
        size=(nwalkers, ndim)
    )

    _clamp_cpu_thread_env_for_lammps_emcee()
    move = emcee.moves.StretchMove(a=float(step_size))

    if pool is not None:
        log_fn = _log_prob_worker
        log_args: tuple = ()
    else:
        log_fn = log_probability
        log_args = (xdata_train, ydata_train, Temperature, calc, weights, params, bounds)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_fn,
        args=log_args,
        moves=move,
        pool=pool,
    )
    sampler.run_mcmc(p0, int(n_steps), progress=False)
    chain = np.asarray(sampler.get_chain(), dtype=float)
    return chain, float(np.mean(sampler.acceptance_fraction))


def flat_prefix(chain: np.ndarray, n_steps: int) -> np.ndarray:
    """Flatten walkers for the first ``n_steps``: ``(n_steps * n_walkers, n_dim)``."""
    n_steps = int(n_steps)
    return chain[:n_steps].reshape(-1, chain.shape[-1])


def thin_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    n = int(arr.shape[0])
    max_rows = int(max_rows)
    if n <= max_rows or max_rows <= 0:
        return arr
    idx = np.linspace(0, n - 1, max_rows, dtype=int)
    return arr[idx]


# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------

def param_z_scores(
    samples_N: np.ndarray,
    mu_ref: np.ndarray,
    sigma_ref: np.ndarray,
) -> np.ndarray:
    """``|μ(N) − μ_ref| / σ_ref`` per parameter."""
    mu = np.mean(samples_N, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(mu - mu_ref) / sigma_ref
    z[~np.isfinite(sigma_ref) | (sigma_ref <= 0)] = np.nan
    return z


def hopping_ensemble_stats(
    samples: np.ndarray,
    A: np.ndarray,
    *,
    max_eval: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble mean and std of hoppings ``t = A @ θ``.

    ``max_eval <= 0`` keeps every flat sample (needed for the reference
    ``σ_ref`` from the full post-burn-in chain).

    Returns
    -------
    mu, sigma : ``(n_obs,)``
    """
    S = thin_rows(samples, max_eval) if int(max_eval) > 0 else np.asarray(samples, dtype=float)
    # predictions: (n_ens, n_obs)
    pred = S @ A.T
    mu = np.mean(pred, axis=0)
    if pred.shape[0] < 2:
        sigma = np.full(pred.shape[1], np.nan, dtype=float)
    else:
        sigma = np.std(pred, axis=0, ddof=1)
    return mu, sigma


def hopping_z_scores(
    samples_N: np.ndarray,
    A: np.ndarray,
    mu_ref: np.ndarray,
    sigma_ref: np.ndarray,
    *,
    max_eval: int,
) -> np.ndarray:
    """``|⟨t⟩(N) − ⟨t⟩_ref| / σ_ref`` per hopping; ``σ_ref`` from full MCMC."""
    mu_N, _ = hopping_ensemble_stats(samples_N, A, max_eval=max_eval)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(mu_N - mu_ref) / sigma_ref
    z[~np.isfinite(sigma_ref) | (sigma_ref <= 0)] = np.nan
    return z


def n_conv_from_log_fit(
    steps: np.ndarray,
    mean_z: np.ndarray,
    tol: float,
    *,
    half_window: int = 0,
) -> Tuple[float, Optional[Tuple[float, float]], Optional[np.ndarray]]:
    """
    Estimate ``N`` where mean ``z`` crosses ``tol`` via a local log–log fit.

    Locates the discrete crossing (last point with ``z > tol``, first with
    ``z ≤ tol`` on ``N < N_ref``), then fits ``log(z) = a + b·log(N)`` only on
    those bracketing points (plus ``half_window`` neighbors on each side if
    requested).

    Returns
    -------
    N_cross, (a, b) or None, fit_step_indices or None
    """
    steps = np.asarray(steps, dtype=float)
    mean_z = np.asarray(mean_z, dtype=float)
    tol = float(tol)
    if steps.size < 2:
        return float("nan"), None, None

    n_ref = float(steps[-1])
    valid = (
        (steps < n_ref)
        & np.isfinite(steps)
        & np.isfinite(mean_z)
        & (mean_z > 0.0)
    )
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return float("nan"), None, None

    z_v = mean_z[idx]
    # Need a point above tol and a later point at/below tol.
    above = z_v > tol
    below = z_v <= tol
    if not np.any(above) or not np.any(below):
        return float("nan"), None, None

    # First index in the valid subsequence that is at/below tol after having
    # been above at least once.
    i_below_local = None
    seen_above = False
    for j, (is_above, is_below) in enumerate(zip(above, below)):
        if is_above:
            seen_above = True
        if seen_above and is_below:
            i_below_local = j
            break
    if i_below_local is None or i_below_local == 0:
        return float("nan"), None, None

    i_above_local = i_below_local - 1
    # Map back to full-array indices; expand by half_window within valid set.
    j0 = max(0, i_above_local - int(half_window))
    j1 = min(idx.size - 1, i_below_local + int(half_window))
    fit_idx = idx[j0 : j1 + 1]
    if fit_idx.size < 2:
        return float("nan"), None, None

    x = np.log(steps[fit_idx])
    y = np.log(mean_z[fit_idx])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.std(x) == 0.0:
        return float("nan"), None, None

    b, a = np.polyfit(x, y, 1)  # y = b*x + a
    if not np.isfinite(a) or not np.isfinite(b) or b >= 0.0:
        return float("nan"), (float(a), float(b)), fit_idx

    n_cross = float(np.exp((np.log(tol) - a) / b))
    if not np.isfinite(n_cross) or n_cross <= 0.0:
        return float("nan"), (float(a), float(b)), fit_idx
    return n_cross, (float(a), float(b)), fit_idx


def steps_to_tol(steps: np.ndarray, mean_z: np.ndarray, tol: float) -> float:
    """``N`` where the local log–log fit to ``mean_z(N)`` crosses ``tol``."""
    n_cross, _, _ = n_conv_from_log_fit(steps, mean_z, tol)
    return n_cross


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _downsample_for_plot(
    steps: np.ndarray,
    curves: np.ndarray,
    max_curves: int,
) -> np.ndarray:
    """Subsample columns of ``curves`` (n_steps, n_items) for gray overlays."""
    n = int(curves.shape[1])
    max_curves = int(max_curves)
    if n <= max_curves or max_curves <= 0:
        return curves
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, size=max_curves, replace=False))
    return curves[:, idx]


def _fit_tuple(fit_arr) -> Optional[Tuple[float, float]]:
    if fit_arr is None:
        return None
    arr = np.asarray(fit_arr, dtype=float).ravel()
    if arr.size < 2 or not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None
    return float(arr[0]), float(arr[1])


def plot_z_vs_steps(
    steps: np.ndarray,
    z: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    tol: float,
    dpi: int = 150,
    param_alpha: float = 0.12,
    max_curves: int = DEFAULT_MAX_PLOT_CURVES,
    n_conv: Optional[float] = None,
    log_fit: Optional[Tuple[float, float]] = None,
    fit_steps: Optional[np.ndarray] = None,
) -> None:
    """Gray per-item ``z`` curves + bold black mean (+ optional local log–log fit)."""
    steps = np.asarray(steps, dtype=float)
    z = np.asarray(z, dtype=float)
    mean_z = np.nanmean(z, axis=1)
    z_plot = _downsample_for_plot(steps, z, max_curves)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(steps, z_plot, color="0.45", alpha=param_alpha, lw=0.8, zorder=1)
    ax.plot(steps, mean_z, color="black", lw=2.4, zorder=2, label="mean")
    if log_fit is not None:
        a, b = log_fit
        if fit_steps is not None and np.asarray(fit_steps).size >= 2:
            x_fit = np.asarray(fit_steps, dtype=float)
        else:
            fit_mask = (steps < steps[-1]) & np.isfinite(mean_z) & (mean_z > 0.0)
            x_fit = steps[fit_mask]
        if x_fit.size >= 2:
            # Dense polyline over the local fit window only.
            x_line = np.geomspace(float(x_fit.min()), float(x_fit.max()), 50)
            y_fit = np.exp(a + b * np.log(x_line))
            ax.plot(
                x_line,
                y_fit,
                color="C0",
                ls="--",
                lw=1.8,
                zorder=3,
                label="local log–log fit",
            )
            ax.plot(
                x_fit,
                np.exp(a + b * np.log(x_fit)),
                "o",
                color="C0",
                ms=6,
                zorder=4,
                label="fit points",
            )
    ax.axhline(float(tol), color="C3", ls="--", lw=1.2, label=rf"tol={tol:g}")
    if n_conv is not None and np.isfinite(n_conv) and n_conv > 0:
        ax.axvline(
            float(n_conv),
            color="C2",
            ls=":",
            lw=1.4,
            label=rf"$N_{{\mathrm{{conv}}}}={n_conv:.4g}$",
        )
    ax.set_xlabel(r"$N_{\mathrm{steps}}$", fontdict=CSFONT)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.set_title(title, fontdict=CSFONT)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(float(steps[0]), float(steps[-1]))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_nconv_vs_nparams(
    records: Sequence[dict],
    *,
    key: str,
    tol: float,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """
    Steps-to-tolerance vs ``n_params``, with an ``a n^p + b`` overlay.

    ``key`` is ``'n_conv_param'`` or ``'n_conv_hop'``.  If several runs share
    the same ``n_params`` (e.g. different walker multipliers), their
    ``N_conv`` values are averaged.
    """
    by_n: Dict[int, List[float]] = {}
    for rec in records:
        y = rec.get(key)
        if y is None or not np.isfinite(float(y)):
            continue
        n_p = int(rec.get("n_params", n_params_acsf_sk(int(rec["M"]), int(rec["W"]))))
        by_n.setdefault(n_p, []).append(float(y))

    if not by_n:
        return

    xs = np.asarray(sorted(by_n), dtype=float)
    ys = np.asarray([float(np.mean(by_n[int(n)])) for n in xs], dtype=float)
    y_sigma = 200.0  # ±N_conv step uncertainty used in the fit

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.errorbar(
        xs,
        ys,
        yerr=y_sigma,
        fmt="o-",
        color="black",
        lw=2.2,
        ms=8,
        capsize=3,
        label=rf"data ($\pm {y_sigma:g}$)",
        zorder=2,
    )

    # Fit N_conv = a * n_params^p + b  (a, p, b free; need >= 3 distinct n)
    if xs.size >= 3 and np.all(xs > 0) and np.all(np.isfinite(ys)):
        def model(n: np.ndarray, a: float, p: float, b: float) -> np.ndarray:
            return a * np.power(n, p) + b

        a0, b0 = np.polyfit(np.sqrt(xs), ys, 1)
        p0 = 0.5
        try:
            (a, p, b), _ = curve_fit(
                model,
                xs,
                ys,
                sigma=np.full(xs.shape, y_sigma),
                absolute_sigma=True,
                p0=(float(a0), p0, float(b0)),
                bounds=([-np.inf, 0.1, -np.inf], [np.inf, 0.9, np.inf]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            a = p = b = np.nan
        if np.isfinite(a) and np.isfinite(p) and np.isfinite(b):
            x_line = np.linspace(float(xs.min()), float(xs.max()), 200)
            y_line = model(x_line, a, p, b)
            ax.plot(
                x_line,
                y_line,
                color="C0",
                ls="--",
                lw=2.0,
                zorder=3,
                label=rf"$a n^{{p}}+b$  ($a={a:.3g}$, $p={p:.3g}$, $b={b:.3g}$)",
            )

    ax.set_xlabel(r"$n_{\mathrm{params}}$", fontdict=CSFONT)
    ax.set_ylabel(rf"$N_{{\mathrm{{steps}}}}$ to mean $z < {tol:g}$", fontdict=CSFONT)
    ax.set_title(key.replace("_", " "), fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_dir_for(
    out_dir: Path,
    model_name: str,
    nwalkers: int,
    walker_mult: int,
) -> Path:
    tag = f"{model_name}_walkers{nwalkers}_mult{walker_mult}"
    return Path(out_dir) / tag


def run_result_path(run_dir: Path) -> Path:
    return Path(run_dir) / "run_result.npz"


def nwalkers_for(n_params: int, walker_mult: int) -> int:
    nwalkers = int(walker_mult) * int(n_params)
    if nwalkers % 2 == 1:
        nwalkers += 1
    return nwalkers


def save_run_result(run_dir: Path, rec: dict) -> None:
    """Persist one run so later invocations can skip MCMC."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    for key, val in rec.items():
        if key in ("run_dir", "model_name"):
            payload[key] = np.array(str(val))
        else:
            payload[key] = np.asarray(val)
    np.savez_compressed(run_result_path(run_dir), **payload)


def load_run_result(run_dir: Path) -> Optional[dict]:
    path = run_result_path(run_dir)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)
    rec: dict = {}
    for key in z.files:
        val = z[key]
        if key in ("run_dir", "model_name"):
            rec[key] = str(val)
        elif np.ndim(val) == 0:
            rec[key] = val.item() if hasattr(val, "item") else val
        else:
            rec[key] = np.asarray(val)
    rec["run_dir"] = str(run_dir)
    return rec


def run_matches_request(
    rec: dict,
    *,
    n_ref: int,
    burnin: int,
    beta: float,
    tol: float,
    max_hop_eval: int,
) -> bool:
    """True if a saved run was produced with the same study settings."""
    try:
        return (
            int(rec["n_ref"]) == int(n_ref)
            and int(rec["burnin"]) == int(burnin)
            and float(rec["beta"]) == float(beta)
            and float(rec["tol"]) == float(tol)
            and int(rec.get("max_hop_eval", -1)) == int(max_hop_eval)
        )
    except (KeyError, TypeError, ValueError):
        return False


def existing_run_record(
    out_dir: Path,
    *,
    M: int,
    W: int,
    walker_mult: int,
    n_ref: int,
    burnin: int,
    beta: float,
    tol: float,
    max_hop_eval: int,
) -> Optional[dict]:
    """Return a saved run record if present and compatible, else None."""
    model_name = model_name_for(M, W)
    n_params = n_params_acsf_sk(M, W)
    nwalkers = nwalkers_for(n_params, walker_mult)
    run_dir = run_dir_for(out_dir, model_name, nwalkers, walker_mult)
    rec = load_run_result(run_dir)
    if rec is None:
        return None
    if not run_matches_request(
        rec,
        n_ref=n_ref,
        burnin=burnin,
        beta=beta,
        tol=tol,
        max_hop_eval=max_hop_eval,
    ):
        print(
            f"  existing {run_dir.name} has different settings "
            f"(n_ref={rec.get('n_ref')}, burnin={rec.get('burnin')}, "
            f"beta={rec.get('beta')}, tol={rec.get('tol')}, "
            f"max_hop_eval={rec.get('max_hop_eval')}); re-running",
            flush=True,
        )
        return None
    return rec


# ---------------------------------------------------------------------------
# One (M, W, n_walkers) study
# ---------------------------------------------------------------------------

def analyze_chain(
    chain: np.ndarray,
    A: np.ndarray,
    checkpoints: np.ndarray,
    *,
    tol: float,
    max_hop_eval: int,
) -> dict:
    """Compute parameter / hopping ``z(N)`` grids and steps-to-tol."""
    n_ref = int(chain.shape[0])
    samples_ref = flat_prefix(chain, n_ref)
    mu_ref = np.mean(samples_ref, axis=0)
    sigma_ref = np.std(samples_ref, axis=0, ddof=1)

    mu_t_ref, sigma_t_ref = hopping_ensemble_stats(
        samples_ref, A, max_eval=0,  # full post-burn-in ensemble for σ_ref
    )

    z_param = []
    z_hop = []
    for N in checkpoints:
        samples_N = flat_prefix(chain, int(N))
        z_param.append(param_z_scores(samples_N, mu_ref, sigma_ref))
        z_hop.append(
            hopping_z_scores(
                samples_N, A, mu_t_ref, sigma_t_ref, max_eval=max_hop_eval,
            )
        )

    z_param_arr = np.vstack(z_param)
    z_hop_arr = np.vstack(z_hop)
    mean_z_param = np.nanmean(z_param_arr, axis=1)
    mean_z_hop = np.nanmean(z_hop_arr, axis=1)

    n_conv_param, fit_param, fit_idx_param = n_conv_from_log_fit(
        checkpoints, mean_z_param, tol,
    )
    n_conv_hop, fit_hop, fit_idx_hop = n_conv_from_log_fit(
        checkpoints, mean_z_hop, tol,
    )

    return {
        "checkpoints": np.asarray(checkpoints, dtype=int),
        "z_param": z_param_arr,
        "z_hop": z_hop_arr,
        "mean_z_param": mean_z_param,
        "mean_z_hop": mean_z_hop,
        "n_conv_param": n_conv_param,
        "n_conv_hop": n_conv_hop,
        "log_fit_param": (
            np.asarray(fit_param, dtype=float)
            if fit_param is not None
            else np.array([np.nan, np.nan])
        ),
        "log_fit_hop": (
            np.asarray(fit_hop, dtype=float)
            if fit_hop is not None
            else np.array([np.nan, np.nan])
        ),
        "fit_steps_param": (
            np.asarray(checkpoints, dtype=float)[fit_idx_param]
            if fit_idx_param is not None
            else np.array([], dtype=float)
        ),
        "fit_steps_hop": (
            np.asarray(checkpoints, dtype=float)[fit_idx_hop]
            if fit_idx_hop is not None
            else np.array([], dtype=float)
        ),
        "mu_ref": mu_ref,
        "sigma_ref": sigma_ref,
        "mu_t_ref": mu_t_ref,
        "sigma_t_ref": sigma_t_ref,
    }


def run_one_setting(
    *,
    M: int,
    W: int,
    walker_mult: int,
    n_ref: int,
    checkpoints: np.ndarray,
    beta: float,
    step_size: float,
    seed: int,
    tol: float,
    max_hop_eval: int,
    out_dir: Path,
    burnin: int = DEFAULT_BURNIN,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    dpi: int = 150,
    force: bool = False,
) -> dict:
    model_name = model_name_for(M, W)
    print(f"\n{'=' * 60}\n {model_name}  walker_mult={walker_mult}\n{'=' * 60}", flush=True)

    if not force:
        existing = existing_run_record(
            out_dir,
            M=M,
            W=W,
            walker_mult=walker_mult,
            n_ref=n_ref,
            burnin=burnin,
            beta=beta,
            tol=tol,
            max_hop_eval=max_hop_eval,
        )
        if existing is not None:
            print(
                f"  skip: found {existing['run_dir']} "
                f"(n_ref={existing.get('n_ref')}, burnin={existing.get('burnin')}, "
                f"n_obs={existing.get('n_obs')})",
                flush=True,
            )
            return existing

    calc, xdata_train_full, xdata_test, xdata, ydata_train_full, ydata_test, ydata, \
        ypred_bestfit, params_full, bounds_full = get_MCMC_inputs(
            model_name, supercells=1, M=int(M), W=int(W),
        )
    if "hopping" not in calc:
        raise RuntimeError(f"{model_name}: no hopping calculator")

    A, y_sub = subsample_hopping_training(
        xdata_train_full["hopping"],
        ydata_train_full["hopping"],
        int(max_hop_eval),
        seed=seed,
    )
    params_hop, bounds_hop = refit_hopping_on_subset(A, y_sub)
    params = {"hopping": params_hop}
    bounds = {"hopping": bounds_hop}
    xdata_train = {"hopping": [A]}
    ydata_train = {"hopping": [y_sub]}

    n_params = int(params_hop.size)
    expected = n_params_acsf_sk(M, W)
    if n_params != expected:
        print(
            f"  warning: n_params={n_params} != 2*(M+M*W)={expected} "
            "(using actual parameter vector length)",
            flush=True,
        )
    if A.shape[1] != n_params:
        raise ValueError(
            f"Descriptor width {A.shape[1]} != n_params {n_params} for {model_name}"
        )

    nwalkers = nwalkers_for(n_params, walker_mult)

    yvar = float(np.var(y_sub))
    w0 = {
        "energy": 0.0,
        "forces": 0.0,
        "hopping": 1.0 / max(y_sub.size, 1) / max(yvar, 1e-30),
    }
    C0 = get_C0(params, xdata_train, ydata_train, calc, w0)
    Temperature = float(beta) * float(C0) / max(n_params, 1)
    burnin = max(0, int(burnin))
    n_total = burnin + int(n_ref)
    if n_ref < 2:
        raise ValueError(f"n_ref must be >= 2 after burn-in, got {n_ref}")

    print(
        f"  n_params={n_params}  nwalkers={nwalkers}  "
        f"burnin={burnin}  N_ref={n_ref}  (total steps={n_total})  "
        f"β={beta:g}  T={Temperature:.3e}  "
        f"n_obs={A.shape[0]} (subsample of train, refit in-memory)",
        flush=True,
    )

    pool = None
    try:
        if parallel:
            n_w = int(n_workers or os.cpu_count() or 1)
            print(f"  parallel pool: {n_w} workers", flush=True)
            pool = multiprocessing.Pool(
                processes=n_w,
                initializer=_subsample_pool_initializer,
                initargs=(
                    calc,
                    xdata_train,
                    ydata_train,
                    Temperature,
                    w0,
                    params,
                    bounds,
                ),
            )

        t0 = time()
        chain_full, acc = run_emcee_chain(
            xdata_train=xdata_train,
            ydata_train=ydata_train,
            Temperature=Temperature,
            calc=calc,
            weights=w0,
            params=params,
            bounds=bounds,
            nwalkers=nwalkers,
            n_steps=n_total,
            step_size=step_size,
            seed=seed,
            pool=pool,
        )
        print(
            f"  MCMC done in {time() - t0:.1f}s  acceptance={acc:.4f}",
            flush=True,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = chain_full[burnin:]
    if chain.shape[0] != int(n_ref):
        raise RuntimeError(
            f"Post-burn-in chain length {chain.shape[0]} != n_ref={n_ref}"
        )
    if burnin > 0:
        print(f"  discarded burnin={burnin} steps → chain shape {chain.shape}", flush=True)

    metrics = analyze_chain(
        chain, A, checkpoints, tol=tol, max_hop_eval=max_hop_eval,
    )

    run_dir = run_dir_for(out_dir, model_name, nwalkers, walker_mult)
    run_dir.mkdir(parents=True, exist_ok=True)

    plot_z_vs_steps(
        metrics["checkpoints"],
        metrics["z_param"],
        title=rf"{model_name}, $n_{{\mathrm{{w}}}}={nwalkers}$ — parameters",
        ylabel=r"$|\mu_i(N)-\mu_i(N_{\mathrm{ref}})|/\sigma_i(N_{\mathrm{ref}})$",
        out_path=run_dir / "param_z_vs_steps.png",
        tol=tol,
        dpi=dpi,
        n_conv=float(metrics["n_conv_param"]),
        log_fit=_fit_tuple(metrics["log_fit_param"]),
        fit_steps=metrics.get("fit_steps_param"),
    )
    plot_z_vs_steps(
        metrics["checkpoints"],
        metrics["z_hop"],
        title=rf"{model_name}, $n_{{\mathrm{{w}}}}={nwalkers}$ — hoppings",
        ylabel=(
            r"$|\langle t\rangle(N)-\langle t\rangle_{\mathrm{ref}}|"
            r"/\sigma_{\mathrm{ref}}$"
        ),
        out_path=run_dir / "hopping_z_vs_steps.png",
        tol=tol,
        dpi=dpi,
        n_conv=float(metrics["n_conv_hop"]),
        log_fit=_fit_tuple(metrics["log_fit_hop"]),
        fit_steps=metrics.get("fit_steps_hop"),
    )

    print(
        f"  N_conv(param)={metrics['n_conv_param']}  "
        f"N_conv(hop)={metrics['n_conv_hop']}  "
        f"(tol={tol:g})",
        flush=True,
    )

    rec = {
        "M": int(M),
        "W": int(W),
        "model_name": model_name,
        "n_params": n_params,
        "walker_mult": int(walker_mult),
        "nwalkers": nwalkers,
        "n_ref": int(n_ref),
        "burnin": int(burnin),
        "beta": float(beta),
        "Temperature": float(Temperature),
        "acceptance": float(acc),
        "tol": float(tol),
        "max_hop_eval": int(max_hop_eval),
        "n_obs": int(A.shape[0]),
        "checkpoints": metrics["checkpoints"],
        "z_param": metrics["z_param"],
        "z_hop": metrics["z_hop"],
        "mean_z_param": metrics["mean_z_param"],
        "mean_z_hop": metrics["mean_z_hop"],
        "n_conv_param": metrics["n_conv_param"],
        "n_conv_hop": metrics["n_conv_hop"],
        "log_fit_param": metrics["log_fit_param"],
        "log_fit_hop": metrics["log_fit_hop"],
        "fit_steps_param": metrics["fit_steps_param"],
        "fit_steps_hop": metrics["fit_steps_hop"],
        "run_dir": str(run_dir),
    }
    save_run_result(run_dir, rec)
    return rec


# ---------------------------------------------------------------------------
# Persistence / replay
# ---------------------------------------------------------------------------

def save_study(path: Path, records: Sequence[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"meta_json": json.dumps(meta)}
    # ragged: store each run under run_{k}_*
    payload["n_runs"] = np.int64(len(records))
    for k, rec in enumerate(records):
        prefix = f"run{k}_"
        for key, val in rec.items():
            if key == "run_dir":
                payload[prefix + "run_dir"] = np.array(str(val))
                continue
            if key == "model_name":
                payload[prefix + "model_name"] = np.array(str(val))
                continue
            arr = np.asarray(val)
            payload[prefix + key] = arr
    np.savez_compressed(path, **payload)
    print(f"Saved study → {path}", flush=True)


def load_study(path: Path) -> Tuple[List[dict], dict]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"][()]))
    n_runs = int(z["n_runs"])
    records: List[dict] = []
    for k in range(n_runs):
        prefix = f"run{k}_"
        rec: dict = {}
        for key in z.files:
            if not key.startswith(prefix):
                continue
            short = key[len(prefix) :]
            val = z[key]
            if short in ("model_name", "run_dir"):
                rec[short] = str(val)
            elif np.ndim(val) == 0:
                rec[short] = val.item() if hasattr(val, "item") else val
            else:
                rec[short] = np.asarray(val)
        records.append(rec)
    return records, meta


def replot_from_records(
    records: Sequence[dict],
    *,
    out_dir: Path,
    tol: float,
    dpi: int = 150,
) -> None:
    for rec in records:
        run_dir = Path(rec.get("run_dir", out_dir / rec["model_name"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        steps = np.asarray(rec["checkpoints"])
        # Recompute N_conv from stored mean curves so old runs get the log fit.
        mean_z_p = (
            np.asarray(rec["mean_z_param"])
            if "mean_z_param" in rec
            else np.nanmean(np.asarray(rec["z_param"]), axis=1)
        )
        mean_z_h = (
            np.asarray(rec["mean_z_hop"])
            if "mean_z_hop" in rec
            else np.nanmean(np.asarray(rec["z_hop"]), axis=1)
        )
        n_conv_p, fit_p, fit_idx_p = n_conv_from_log_fit(steps, mean_z_p, tol)
        n_conv_h, fit_h, fit_idx_h = n_conv_from_log_fit(steps, mean_z_h, tol)
        rec["n_conv_param"] = n_conv_p
        rec["n_conv_hop"] = n_conv_h
        fit_steps_p = steps[fit_idx_p] if fit_idx_p is not None else None
        fit_steps_h = steps[fit_idx_h] if fit_idx_h is not None else None

        plot_z_vs_steps(
            steps,
            np.asarray(rec["z_param"]),
            title=(
                rf"{rec['model_name']}, "
                rf"$n_{{\mathrm{{w}}}}={int(rec['nwalkers'])}$ — parameters"
            ),
            ylabel=r"$|\mu_i(N)-\mu_i(N_{\mathrm{ref}})|/\sigma_i(N_{\mathrm{ref}})$",
            out_path=run_dir / "param_z_vs_steps.png",
            tol=tol,
            dpi=dpi,
            n_conv=n_conv_p,
            log_fit=fit_p,
            fit_steps=fit_steps_p,
        )
        plot_z_vs_steps(
            steps,
            np.asarray(rec["z_hop"]),
            title=(
                rf"{rec['model_name']}, "
                rf"$n_{{\mathrm{{w}}}}={int(rec['nwalkers'])}$ — hoppings"
            ),
            ylabel=(
                r"$|\langle t\rangle(N)-\langle t\rangle_{\mathrm{ref}}|"
                r"/\sigma_{\mathrm{ref}}$"
            ),
            out_path=run_dir / "hopping_z_vs_steps.png",
            tol=tol,
            dpi=dpi,
            n_conv=n_conv_h,
            log_fit=fit_h,
            fit_steps=fit_steps_h,
        )

    plot_nconv_vs_nparams(
        records,
        key="n_conv_param",
        tol=tol,
        out_path=out_dir / "nconv_param_vs_nparams.png",
        dpi=dpi,
    )
    plot_nconv_vs_nparams(
        records,
        key="n_conv_hop",
        tol=tol,
        out_path=out_dir / "nconv_hop_vs_nparams.png",
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACSF_hoppings_sk MCMC convergence study (M/W × walkers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-M",
        nargs="+",
        type=int,
        default=None,
        help=f"ACSF radial counts (default: {DEFAULT_M}).",
    )
    p.add_argument(
        "-W",
        nargs="+",
        type=int,
        default=None,
        help=f"ACSF angular counts (default: {DEFAULT_W}).",
    )
    p.add_argument(
        "--walker-mult",
        nargs="+",
        type=int,
        default=None,
        help=f"n_walkers / n_params multipliers (default: {DEFAULT_WALKER_MULT}).",
    )
    p.add_argument(
        "--n-ref",
        type=int,
        default=DEFAULT_N_REF,
        help=f"Post-burn-in reference MCMC steps (default: {DEFAULT_N_REF}).",
    )
    p.add_argument(
        "--burnin",
        type=int,
        default=DEFAULT_BURNIN,
        help=f"Discard this many initial MCMC steps (default: {DEFAULT_BURNIN}).",
    )
    p.add_argument(
        "--n-checkpoints",
        type=int,
        default=40,
        help="Number of geometric checkpoints between 2 and n-ref.",
    )
    p.add_argument(
        "-B",
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"Temperature weight β (default: {DEFAULT_BETA}).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"Convergence tolerance on mean z (default: {DEFAULT_TOL}).",
    )
    p.add_argument(
        "--step-size",
        type=float,
        default=MCMC_DEFAULTS["step_size"],
        help="emcee StretchMove scale a.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-hop-eval",
        type=int,
        default=DEFAULT_MAX_HOP_EVAL,
        help=(
            "Max training hopping pairs for MCMC / predictive z "
            f"(subsample + in-memory refit; default: {DEFAULT_MAX_HOP_EVAL})."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Figure / npz output directory.",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Multiprocessing pool for walker likelihoods.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker processes for --parallel (default: cpu count).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny grid for smoke tests: M=6 W=0, mult=2 4, burnin=50, n_ref=200.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if run_result.npz already exists for a setting.",
    )
    p.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Replot from a saved study_*.npz (skip MCMC).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replay is not None:
        records, meta = load_study(Path(args.replay))
        tol = float(meta.get("tol", args.tol))
        replot_from_records(records, out_dir=out_dir, tol=tol, dpi=args.dpi)
        print("Replay done.", flush=True)
        return

    if args.quick:
        M_list = [6]
        W_list = [0]
        walker_mults = [2, 4]
        n_ref = min(200, int(args.n_ref))
        burnin = min(50, int(args.burnin))
    else:
        M_list = list(args.M) if args.M is not None else list(DEFAULT_M)
        W_list = list(args.W) if args.W is not None else list(DEFAULT_W)
        walker_mults = (
            list(args.walker_mult)
            if args.walker_mult is not None
            else list(DEFAULT_WALKER_MULT)
        )
        n_ref = int(args.n_ref)
        burnin = int(args.burnin)

    walker_mults = sorted({int(m) for m in walker_mults if int(m) >= 2})
    if not walker_mults:
        raise SystemExit("--walker-mult must include values >= 2")
    if max(walker_mults) > 10:
        print(
            "  note: walker multipliers > 10 are allowed but outside the "
            "requested 2×–10× study window.",
            flush=True,
        )

    checkpoints = checkpoint_grid(n_ref, int(args.n_checkpoints))
    print(
        f"Study: M={M_list} W={W_list} walker_mult={walker_mults} "
        f"burnin={burnin} N_ref={n_ref} checkpoints={len(checkpoints)} "
        f"tol={args.tol:g}",
        flush=True,
    )

    # Spawn method required when workers rebuild LAMMPS / large state.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    records: List[dict] = []
    run_id = 0
    for M in M_list:
        for W in W_list:
            for mult in walker_mults:
                rec = run_one_setting(
                    M=int(M),
                    W=int(W),
                    walker_mult=int(mult),
                    n_ref=n_ref,
                    checkpoints=checkpoints,
                    beta=float(args.beta),
                    step_size=float(args.step_size),
                    seed=int(args.seed) + run_id,
                    tol=float(args.tol),
                    max_hop_eval=int(args.max_hop_eval),
                    out_dir=out_dir,
                    burnin=burnin,
                    parallel=bool(args.parallel),
                    n_workers=args.n_workers,
                    dpi=int(args.dpi),
                    force=bool(args.force),
                )
                records.append(rec)
                run_id += 1

    meta = {
        "M": M_list,
        "W": W_list,
        "walker_mult": walker_mults,
        "n_ref": n_ref,
        "burnin": burnin,
        "tol": float(args.tol),
        "beta": float(args.beta),
        "max_hop_eval": int(args.max_hop_eval),
        "model": MODEL_BASE,
    }
    npz_path = out_dir / f"study_beta{args.beta:g}_nref{n_ref}.npz"
    save_study(npz_path, records, meta)
    replot_from_records(
        records, out_dir=out_dir, tol=float(args.tol), dpi=int(args.dpi),
    )
    print(f"Done.  {len(records)} run(s) → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
