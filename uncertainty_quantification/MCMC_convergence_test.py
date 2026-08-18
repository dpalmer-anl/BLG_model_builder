#!/usr/bin/env python3
"""
MCMC convergence diagnostics for emcee ensemble sampling.

Estimates reasonable ``n_samples`` (MCMC steps, ``MCMC_DEFAULTS["n_samples"]``)
and ``chain_samples`` (thinned posterior size) by tracking:

1. **Parameter variance vs MCMC steps** — mean variance across parameters and a
   few individual parameter traces.
2. **Single-walker parameter traces** — parameter values vs MCMC step for one
   emcee walker (one chain).
3. **Predictive variance vs chain_samples** — mean observation variance of
   ensemble predictions after the same thinning used in
   :mod:`blg_model_builder.EMCEE_generate_ensemble`.
4. **Mean predictive MAE vs chain_samples** — MAE between test ``y_true`` and
   the ensemble-mean prediction vs thinned posterior size.

Works for any model supported by ``get_MCMC_inputs`` / ``EMCEE_generate_ensemble``.

Examples
--------
::

    cd uncertainty_quantification
    python MCMC_convergence_test.py -m ACSF_hoppings_sk -M 12 -W 1 --parallel

    python MCMC_convergence_test.py -m POD_energy --POD-index 0 \\
        --n-samples 400 --max-chain-samples 2000 --beta 0.01

``--parallel`` uses workers for MCMC only; predictions run on the main process
by default (``--parallel-eval`` for a separate prediction pool). By default,
``--max-eval-chain`` matches ``--max-chain-samples`` (how many thinned states
are evaluated for predictive variance / MAE). Set ``--max-eval-chain 0`` to
evaluate every flat-chain state (can exhaust memory on long runs).

    # Replot from a saved run (skip MCMC + prediction eval)
    python MCMC_convergence_test.py --replay MCMC_convergence/.../convergence.npz
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import sys
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence, Tuple

import emcee
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.chdir(HERE)

from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_model_name_arg,
    collect_workflow_hyperparams,
    expand_ensemble_model_name,
)
from blg_model_builder.EMCEE_generate_ensemble import (
    MCMC_DEFAULTS,
    _eval_ensemble_worker,
    _log_prob_worker,
    _pool_initializer,
    _ravel_observation_blocks,
    _set_worker_globals,
    get_C0,
    log_probability,
    worker,
)
from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs

DEFAULT_OUTPUT_DIR = HERE / "MCMC_convergence"
DEFAULT_N_SAMPLES = max(MCMC_DEFAULTS["n_samples"], 400)
DEFAULT_MAX_CHAIN_SAMPLES = MCMC_DEFAULTS["chain_samples"]
DEFAULT_STEP_STRIDE = 5
DEFAULT_N_PARAM_TRACES = 5
DEFAULT_N_CHAIN_GRID = 100


def thin_chain_to_target(samples: np.ndarray, target: int) -> np.ndarray:
    """Same stride thinning as ``get_MCMC_ensemble``."""
    samples = np.asarray(samples, dtype=float)
    n = int(samples.shape[0])
    target = int(target)
    if n <= target:
        return samples
    stride = max(1, int(n / target))
    return samples[::stride][:target]


def subsample_indices(n_total: int, target: int) -> np.ndarray:
    """Indices for ``thin_chain_to_target`` without materializing the chain."""
    n_total = int(n_total)
    target = int(target)
    if n_total <= target:
        return np.arange(n_total, dtype=int)
    stride = max(1, int(n_total / target))
    return np.arange(0, n_total, stride, dtype=int)[:target]


def flat_chain_to_sample_dict(
    flat_chain: np.ndarray,
    params: Dict[str, np.ndarray],
    model_keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    sample_dict: Dict[str, np.ndarray] = {}
    theta_ind = 0
    for key in model_keys:
        n = len(params[key])
        sample_dict[key] = np.asarray(flat_chain[:, theta_ind : theta_ind + n], dtype=float)
        theta_ind += n
    return sample_dict


def parameter_labels(params: Dict[str, np.ndarray], model_keys: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for key in model_keys:
        for i in range(len(params[key])):
            labels.append(f"{key}[{i}]")
    return labels


def compute_parameter_variance_vs_steps(
    chain: np.ndarray,
    *,
    burn_in: int = 0,
    step_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameter variance after each MCMC step count.

    Parameters
    ----------
    chain
        ``(n_steps, n_walkers, n_dim)`` from ``sampler.get_chain()``.

    Returns
    -------
    steps, mean_var, param_var
        ``param_var`` has shape ``(n_step_points, n_dim)``.
    """
    chain = np.asarray(chain, dtype=float)
    n_steps, _n_walkers, n_dim = chain.shape
    burn_in = max(0, int(burn_in))
    step_stride = max(1, int(step_stride))

    steps: List[int] = []
    mean_vars: List[float] = []
    param_vars: List[np.ndarray] = []

    for k in range(burn_in + 1, n_steps + 1, step_stride):
        block = chain[burn_in:k].reshape(-1, n_dim)
        if block.shape[0] < 2:
            continue
        v = np.var(block, axis=0, ddof=1)
        steps.append(k)
        mean_vars.append(float(np.mean(v)))
        param_vars.append(v)

    if not steps:
        empty = np.array([], dtype=int)
        return empty, np.array([], dtype=float), np.empty((0, n_dim), dtype=float)

    return (
        np.asarray(steps, dtype=int),
        np.asarray(mean_vars, dtype=float),
        np.vstack(param_vars),
    )


def select_parameter_indices(n_dim: int, n_traces: int) -> np.ndarray:
    """Evenly spaced parameter indices for trace / variance overlays."""
    n_dim = int(n_dim)
    if n_dim <= 0:
        return np.array([], dtype=int)
    n_traces = max(1, min(int(n_traces), n_dim))
    if n_traces == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n_dim - 1, n_traces, dtype=int)


def walker_parameter_traces(
    chain: np.ndarray,
    param_indices: Sequence[int],
    *,
    walker: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameter values along one emcee walker vs MCMC step.

    Parameters
    ----------
    chain
        ``(n_steps, n_walkers, n_dim)`` from ``sampler.get_chain()``.

    Returns
    -------
    steps, traces
        ``steps`` is ``1 … n_steps``; ``traces`` has shape ``(n_steps, n_params)``.
    """
    chain = np.asarray(chain, dtype=float)
    n_steps, n_walkers, _n_dim = chain.shape
    walker = int(walker) % int(n_walkers)
    idx = [int(i) for i in param_indices]
    steps = np.arange(1, n_steps + 1, dtype=int)
    if not idx:
        return steps, np.empty((n_steps, 0), dtype=float)
    traces = chain[:, walker, :][:, idx]
    return steps, np.asarray(traces, dtype=float)


def best_fit_vector(params: Dict[str, np.ndarray], model_keys: Sequence[str]) -> np.ndarray:
    parts = [np.asarray(params[key], dtype=float).ravel() for key in model_keys]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def ypred_dict_to_matrix(ypred_dict: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    """Concatenate observable blocks into ``(n_samples, n_obs)``."""
    parts: List[np.ndarray] = []
    for key in keys:
        if key == "forces" or key not in ypred_dict:
            continue
        yp = np.asarray(ypred_dict[key], dtype=float)
        if yp.ndim == 1:
            yp = yp[:, np.newaxis]
        parts.append(yp)
    if not parts:
        return np.empty((0, 0), dtype=float)
    widths = [p.shape[1] for p in parts]
    if len(set(widths)) == 1 and len(parts) > 1:
        # Same row count, different keys — concatenate columns.
        n_rows = parts[0].shape[0]
        if all(p.shape[0] == n_rows for p in parts):
            return np.hstack(parts)
    # Fallback: ravel each row across keys.
    rows = []
    n_rows = min(p.shape[0] for p in parts)
    for i in range(n_rows):
        rows.append(
            np.concatenate([np.ravel(parts[j][i]) for j in range(len(parts))])
        )
    return np.vstack(rows)


def ytrue_vector(y_data: dict, key: str) -> np.ndarray:
    """Flatten test targets for one observable block to 1-D."""
    if key not in y_data:
        return np.array([], dtype=float)
    return _ravel_observation_blocks(y_data[key])


def ytrue_dict_to_vector(y_data: dict, keys: Sequence[str]) -> np.ndarray:
    """Concatenate flattened test targets across observable keys."""
    parts = [
        ytrue_vector(y_data, key)
        for key in keys
        if key != "forces" and key in y_data
    ]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _align_ypred_ytrue(
    ypred_full: np.ndarray,
    ytrue: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ypred_full = np.asarray(ypred_full, dtype=float)
    ytrue = np.ravel(np.asarray(ytrue, dtype=float))
    if ypred_full.ndim == 1:
        ypred_full = ypred_full[:, np.newaxis]
    if ypred_full.size == 0 or ytrue.size == 0:
        return None, None
    n = min(int(ypred_full.shape[1]), int(ytrue.size))
    return ypred_full[:, :n], ytrue[:n]


def make_chain_samples_grid(
    n_total: int,
    max_chain_samples: int,
    *,
    n_grid: int = DEFAULT_N_CHAIN_GRID,
    chain_grid: Optional[Sequence[int]] = None,
    min_size: int = 1,
) -> np.ndarray:
    n_total = int(n_total)
    max_chain_samples = int(max_chain_samples)
    min_size = max(1, int(min_size))

    if chain_grid is None:
        hi = min(max_chain_samples, n_total)
        lo = max(min_size, 1)
        if hi <= lo:
            grid = np.array([hi], dtype=int)
        else:
            grid = np.unique(np.round(np.linspace(lo, hi, int(n_grid))).astype(int))
    else:
        grid = np.unique(np.asarray(chain_grid, dtype=int))
        grid = grid[(grid >= min_size) & (grid <= n_total)]
    return grid


def compute_ypred_variance_vs_chain_samples(
    ypred_full: np.ndarray,
    *,
    max_chain_samples: int,
    n_grid: int = DEFAULT_N_CHAIN_GRID,
    chain_grid: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean observation variance of predictions vs thinned ensemble size.

    Uses the same thinning rule as production MCMC. ``ypred_full`` must be
    ordered consistently with ``sampler.get_chain(flat=True)``.
    """
    ypred_full = np.asarray(ypred_full, dtype=float)
    n_total = int(ypred_full.shape[0])
    grid = make_chain_samples_grid(
        n_total,
        max_chain_samples,
        n_grid=n_grid,
        chain_grid=chain_grid,
        min_size=1,
    )

    mean_vars: List[float] = []
    for m in grid:
        idx = subsample_indices(n_total, int(m))
        block = ypred_full[idx]
        if block.ndim == 1:
            block = block[:, np.newaxis]
        if block.shape[0] < 2:
            mean_vars.append(np.nan)
            continue
        obs_var = np.var(block, axis=0, ddof=1)
        mean_vars.append(float(np.nanmean(obs_var)))

    return grid, np.asarray(mean_vars, dtype=float)


def compute_mean_ypred_mae_vs_chain_samples(
    ypred_full: np.ndarray,
    ytrue: np.ndarray,
    *,
    max_chain_samples: int,
    n_grid: int = DEFAULT_N_CHAIN_GRID,
    chain_grid: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean absolute error between test targets and ensemble-mean predictions.

    For each ``chain_samples`` size ``m``, thin to ``m`` posterior draws,
    form ``mean(y_pred)`` over those draws, and return
    ``mean(|y_true - mean(y_pred)|)`` across observations.
    """
    ypred_aligned, ytrue_aligned = _align_ypred_ytrue(ypred_full, ytrue)
    if ypred_aligned is None or ytrue_aligned is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    n_total = int(ypred_aligned.shape[0])
    grid = make_chain_samples_grid(
        n_total,
        max_chain_samples,
        n_grid=n_grid,
        chain_grid=chain_grid,
        min_size=1,
    )

    maes: List[float] = []
    for m in grid:
        idx = subsample_indices(n_total, int(m))
        block = ypred_aligned[idx]
        y_mean = np.mean(block, axis=0)
        maes.append(float(np.nanmean(np.abs(ytrue_aligned - y_mean))))

    return grid, np.asarray(maes, dtype=float)


def run_emcee_sampler(
    x_train,
    y_train,
    temperature: float,
    calc,
    weights,
    params,
    bounds,
    *,
    n_samples: int,
    step_size: float,
    pool=None,
):
    """Run emcee and return ``(sampler, model_keys)`` without thinning."""
    model_keys = list(calc.keys())
    nwalkers = 0
    ndim: Dict[str, int] = {}
    for key in model_keys:
        nw = int(2 * len(params[key]))
        ndim[key] = len(params[key])
        nwalkers += nw
    nwalkers = max(nwalkers, MCMC_DEFAULTS["n_walkers_min"])

    theta_walkers = None
    for it, key in enumerate(model_keys):
        scale = MCMC_DEFAULTS["walker_init_scale"] * np.abs(params[key])
        new_block = np.random.normal(
            loc=params[key], scale=scale, size=(nwalkers, ndim[key])
        )
        if it == 0:
            theta_walkers = new_block
        else:
            theta_walkers = np.append(theta_walkers, new_block, axis=1)

    move = emcee.moves.StretchMove(a=float(step_size))
    if pool is not None:
        log_fn = _log_prob_worker
        log_args = ()
    else:
        log_fn = log_probability
        log_args = (x_train, y_train, temperature, calc, weights, params, bounds)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        int(theta_walkers.shape[1]),
        log_fn,
        args=log_args,
        moves=move,
        pool=pool,
    )
    sampler.run_mcmc(theta_walkers, int(n_samples), progress=True)
    return sampler, model_keys, nwalkers


def resolve_max_eval_chain(
    max_eval_arg: Optional[int],
    max_chain_samples: int,
    n_flat: int,
) -> int:
    """States to evaluate: defaults to ``max_chain_samples``; 0 means all."""
    if max_eval_arg is None:
        cap = int(max_chain_samples)
    elif int(max_eval_arg) <= 0:
        return int(n_flat)
    else:
        cap = int(max_eval_arg)
    return min(cap, int(n_flat))


def _stack_predictions(preds: Sequence) -> np.ndarray:
    """Stack per-sample predictions to ``(n_expected, n_obs)`` without dropping rows."""
    if not preds:
        return np.empty((0, 0), dtype=float)
    first = preds[0]
    if isinstance(first, tuple) and len(first) == 2:
        energies = np.asarray([r[0] for r in preds], dtype=float)
        if energies.ndim == 1:
            energies = energies[:, np.newaxis]
        return energies
    if isinstance(first, list) and len(first) > 0:
        rows = []
        for r in preds:
            if not isinstance(r, list) or len(r) != len(first):
                rows.append(np.full(
                    int(np.concatenate([np.asarray(first[k], dtype=float).ravel() for k in range(len(first))]).size),
                    np.nan,
                ))
                continue
            rows.append(
                np.concatenate([np.asarray(r[k], dtype=float).ravel() for k in range(len(r))])
            )
        if not rows:
            return np.empty((0, 0), dtype=float)
        lens = np.array([row.size for row in rows])
        L = int(np.min(lens)) if lens.size else 0
        if L > 0 and np.any(lens != lens[0]):
            rows = [row[:L] for row in rows]
        return np.vstack(rows)
    squeezed = [np.squeeze(np.asarray(r)) for r in preds]
    if squeezed and np.ndim(squeezed[0]) == 0:
        return np.column_stack([[s] for s in squeezed])
    return np.vstack(squeezed)


def _predict_block(
    sample_dict: Dict[str, np.ndarray],
    calc,
    x_eval,
    model_keys: Sequence[str],
    *,
    pool=None,
    dataset_id: str = "test",
) -> Dict[str, np.ndarray]:
    """Evaluate one parameter block; preserve one row per input sample."""
    use_tetb = ("hopping" in calc) and ("energy" in calc)
    n_rows = int(next(iter(sample_dict.values())).shape[0])
    out: Dict[str, np.ndarray] = {}

    for key in model_keys:
        if key == "forces":
            continue
        if use_tetb and key == "energy":
            theta = np.hstack((sample_dict["hopping"], sample_dict["energy"]))
        else:
            theta = sample_dict[key]

        if pool is not None:
            tasks = [(dataset_id, key, theta[i, :]) for i in range(n_rows)]
            preds = pool.map(_eval_ensemble_worker, tasks)
        else:
            preds = [
                worker((theta[i, :], calc[key], x_eval[key]))
                for i in range(n_rows)
            ]
        block = _stack_predictions(preds)
        if block.shape[0] != n_rows:
            raise RuntimeError(
                f"prediction row mismatch for {key}: got {block.shape[0]}, expected {n_rows}"
            )
        out[key] = block
    return out


def _shutdown_pool(pool) -> None:
    if pool is not None:
        pool.close()
        pool.join()


def thin_flat_chain_for_eval(
    flat_chain: np.ndarray,
    max_eval_chain: int,
) -> Tuple[np.ndarray, int]:
    """Evenly subsample the flat chain before prediction evaluation."""
    flat_chain = np.asarray(flat_chain, dtype=float)
    n_orig = int(flat_chain.shape[0])
    max_eval_chain = int(max_eval_chain)
    if n_orig <= max_eval_chain:
        return flat_chain, n_orig
    idx = subsample_indices(n_orig, max_eval_chain)
    return flat_chain[idx], n_orig


def evaluate_flat_chain_predictions(
    flat_chain: np.ndarray,
    calc,
    x_eval,
    y_eval,
    params,
    model_keys: Sequence[str],
    *,
    pool=None,
    chunk_size: int = 50,
    dataset_id: str = "test",
) -> Dict[str, np.ndarray]:
    """Evaluate predictions for every row of ``flat_chain`` (1:1 row alignment)."""
    del y_eval  # signature kept for callers; workers use preloaded test data
    n_total = int(flat_chain.shape[0])
    out: Dict[str, np.ndarray] = {}
    row_offset = 0

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        block = flat_chain[start:end]
        sample_dict = flat_chain_to_sample_dict(block, params, model_keys)
        ypred_block = _predict_block(
            sample_dict, calc, x_eval, model_keys,
            pool=pool, dataset_id=dataset_id,
        )
        for key, block_arr in ypred_block.items():
            block_arr = np.asarray(block_arr, dtype=float)
            if block_arr.ndim == 1:
                block_arr = block_arr[:, np.newaxis]
            if key not in out:
                out[key] = np.empty((n_total, block_arr.shape[1]), dtype=float)
            out[key][row_offset : row_offset + block_arr.shape[0]] = block_arr
        row_offset += end - start
        if end == n_total or end % max(chunk_size * 20, 500) == 0:
            print(f"  [predictions] {end}/{n_total} chain states evaluated", flush=True)

    return out


def plot_parameter_variance(
    steps: np.ndarray,
    mean_var: np.ndarray,
    param_var: np.ndarray,
    labels: Sequence[str],
    *,
    n_traces: int,
    model_name: str,
    beta: float,
    out_path: Path,
    reference_n_samples: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(steps, mean_var, color="C0", lw=2.0, label="mean param variance")

    n_dim = param_var.shape[1]
    if n_dim > 0:
        trace_idx = select_parameter_indices(n_dim, n_traces)
        for j, idx in enumerate(trace_idx):
            ax.plot(
                steps,
                param_var[:, idx],
                lw=1.2,
                alpha=0.85,
                label=labels[idx] if idx < len(labels) else f"param {idx}",
            )

    if reference_n_samples is not None:
        ax.axvline(
            float(reference_n_samples),
            color="0.4",
            ls="--",
            lw=1.0,
            label=f"default n_samples={reference_n_samples}",
        )

    ax.set_xlabel("MCMC steps (n_samples)")
    ax.set_ylabel("posterior variance")
    ax.set_title(rf"{model_name}  $\beta={beta:g}$ — parameter variance vs steps")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_walker_parameter_traces(
    steps: np.ndarray,
    traces: np.ndarray,
    labels: Sequence[str],
    *,
    walker_index: int,
    model_name: str,
    beta: float,
    out_path: Path,
    best_fit: Optional[np.ndarray] = None,
    reference_n_samples: Optional[int] = None,
) -> None:
    """Parameter value vs MCMC step for one walker (one chain)."""
    steps = np.asarray(steps, dtype=float)
    traces = np.asarray(traces, dtype=float)
    n_params = int(traces.shape[1])
    if n_params == 0:
        return

    fig_h = max(2.4, 2.0 * n_params)
    fig, axes = plt.subplots(n_params, 1, figsize=(7.5, fig_h), squeeze=False)

    for i, ax in enumerate(axes.ravel()):
        ax.plot(steps, traces[:, i], lw=0.9, color="C0")
        if best_fit is not None and i < len(best_fit):
            ax.axhline(
                float(best_fit[i]),
                color="C3",
                ls="--",
                lw=1.0,
                label="best fit",
            )
        if reference_n_samples is not None:
            ax.axvline(
                float(reference_n_samples),
                color="0.45",
                ls=":",
                lw=0.9,
            )
        ylab = labels[i] if i < len(labels) else f"param {i}"
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(True, alpha=0.25)
        if best_fit is not None and i == 0:
            ax.legend(fontsize=7, loc="best")

    axes.ravel()[-1].set_xlabel("MCMC step (n_samples)")
    fig.suptitle(
        rf"{model_name}  $\beta={beta:g}$ — walker {walker_index} parameter traces",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ypred_variance(
    chain_sizes: np.ndarray,
    mean_obs_var: np.ndarray,
    *,
    model_name: str,
    beta: float,
    observable_label: str,
    out_path: Path,
    reference_chain_samples: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(chain_sizes, mean_obs_var, "o-", color="C1", lw=1.8, ms=4)

    if reference_chain_samples is not None and mean_obs_var.size:
        ref_idx = np.argmin(np.abs(chain_sizes - reference_chain_samples))
        ax.axvline(
            float(chain_sizes[ref_idx]),
            color="0.4",
            ls="--",
            lw=1.0,
            label=f"default chain_samples≈{reference_chain_samples}",
        )

    ax.set_xlabel("chain_samples (thinned posterior size)")
    ax.set_ylabel("mean Var(y_pred) across observations")
    ax.set_title(
        rf"{model_name}  $\beta={beta:g}$ — {observable_label} predictive variance"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_ypred_mae(
    chain_sizes: np.ndarray,
    mae: np.ndarray,
    *,
    model_name: str,
    beta: float,
    observable_label: str,
    out_path: Path,
    reference_chain_samples: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(chain_sizes, mae, "o-", color="C2", lw=1.8, ms=4)

    if reference_chain_samples is not None and mae.size:
        ref_idx = np.argmin(np.abs(chain_sizes - reference_chain_samples))
        ax.axvline(
            float(chain_sizes[ref_idx]),
            color="0.4",
            ls="--",
            lw=1.0,
            label=f"default chain_samples≈{reference_chain_samples}",
        )

    ax.set_xlabel("chain_samples (thinned posterior size)")
    ax.set_ylabel(r"MAE$(y_{\mathrm{true}},\ \langle y_{\mathrm{pred}}\rangle)$")
    ax.set_title(
        rf"{model_name}  $\beta={beta:g}$ — {observable_label} ensemble-mean MAE"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_convergence_npz(
    out_path: Path,
    *,
    meta: dict,
    steps: np.ndarray,
    mean_param_var: np.ndarray,
    param_var: np.ndarray,
    chain_sizes: np.ndarray,
    mean_ypred_var: np.ndarray,
    ypred_var_by_key: Dict[str, np.ndarray],
    mean_ypred_mae: np.ndarray,
    ypred_mae_by_key: Dict[str, np.ndarray],
    flat_chain: Optional[np.ndarray] = None,
    walker_trace_steps: Optional[np.ndarray] = None,
    walker_trace_values: Optional[np.ndarray] = None,
) -> None:
    payload = {
        "meta_json": np.array([json.dumps(meta)]),
        "steps": steps,
        "mean_param_var": mean_param_var,
        "param_var": param_var,
        "chain_sizes": chain_sizes,
        "mean_ypred_var": mean_ypred_var,
        "mean_ypred_mae": mean_ypred_mae,
    }
    for key, arr in ypred_var_by_key.items():
        payload[f"ypred_var_{key}"] = arr
    for key, arr in ypred_mae_by_key.items():
        payload[f"ypred_mae_{key}"] = arr
    if flat_chain is not None:
        payload["flat_chain"] = flat_chain
    if walker_trace_steps is not None:
        payload["walker_trace_steps"] = walker_trace_steps
    if walker_trace_values is not None:
        payload["walker_trace_values"] = walker_trace_values
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def load_convergence_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta_json"][0]))
    out = {
        "meta": meta,
        "steps": np.asarray(data["steps"]),
        "mean_param_var": np.asarray(data["mean_param_var"]),
        "param_var": np.asarray(data["param_var"]),
        "chain_sizes": np.asarray(data["chain_sizes"]),
        "mean_ypred_var": np.asarray(data["mean_ypred_var"]),
        "mean_ypred_mae": np.asarray(data["mean_ypred_mae"])
        if "mean_ypred_mae" in data.files
        else np.array([], dtype=float),
    }
    out["ypred_var_by_key"] = {}
    out["ypred_mae_by_key"] = {}
    for k in data.files:
        if k.startswith("ypred_var_"):
            out["ypred_var_by_key"][k.replace("ypred_var_", "", 1)] = np.asarray(data[k])
        elif k.startswith("ypred_mae_"):
            out["ypred_mae_by_key"][k.replace("ypred_mae_", "", 1)] = np.asarray(data[k])
    if "walker_trace_steps" in data.files:
        out["walker_trace_steps"] = np.asarray(data["walker_trace_steps"])
        out["walker_trace_values"] = np.asarray(data["walker_trace_values"])
    if "flat_chain" in data.files:
        out["flat_chain"] = np.asarray(data["flat_chain"], dtype=float)
    return out


def recompute_chain_sample_curves(
    flat_chain: np.ndarray,
    calc,
    x_test,
    y_test,
    params,
    model_keys: Sequence[str],
    *,
    max_chain_samples: int,
    n_grid: int = DEFAULT_N_CHAIN_GRID,
    chain_grid: Optional[Sequence[int]] = None,
    pool=None,
    predict_chunk: int = 50,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Re-evaluate predictions and rebuild variance / MAE vs chain_samples curves."""
    ypred_dict = evaluate_flat_chain_predictions(
        flat_chain,
        calc,
        x_test,
        y_test,
        params,
        model_keys,
        pool=pool,
        chunk_size=predict_chunk,
    )

    chain_sizes = np.array([], dtype=int)
    ypred_var_by_key: Dict[str, np.ndarray] = {}
    ypred_mae_by_key: Dict[str, np.ndarray] = {}
    mean_ypred_var = np.array([], dtype=float)
    mean_ypred_mae = np.array([], dtype=float)

    grid_kw = {"n_grid": n_grid, "chain_grid": chain_grid}
    for key in model_keys:
        if key not in ypred_dict:
            continue
        if chain_sizes.size == 0:
            chain_sizes = make_chain_samples_grid(
                ypred_dict[key].shape[0],
                max_chain_samples,
                min_size=1,
                **grid_kw,
            )
        _, var_curve = compute_ypred_variance_vs_chain_samples(
            ypred_dict[key],
            max_chain_samples=max_chain_samples,
            chain_grid=chain_sizes if chain_grid is None else chain_grid,
            n_grid=n_grid,
        )
        ypred_var_by_key[key] = var_curve
        _, mae_curve = compute_mean_ypred_mae_vs_chain_samples(
            ypred_dict[key],
            ytrue_vector(y_test, key),
            max_chain_samples=max_chain_samples,
            chain_grid=chain_sizes if chain_grid is None else chain_grid,
            n_grid=n_grid,
        )
        ypred_mae_by_key[key] = mae_curve

    nrow_set = {ypred_dict[k].shape[0] for k in ypred_var_by_key}
    if len(nrow_set) == 1 and ypred_var_by_key:
        ypred_combined = ypred_dict_to_matrix(ypred_dict, model_keys)
        ytrue_combined = ytrue_dict_to_vector(y_test, model_keys)
        if ypred_combined.size:
            _, mean_ypred_var = compute_ypred_variance_vs_chain_samples(
                ypred_combined,
                max_chain_samples=max_chain_samples,
                chain_grid=chain_sizes if chain_grid is None else chain_grid,
                n_grid=n_grid,
            )
            _, mean_ypred_mae = compute_mean_ypred_mae_vs_chain_samples(
                ypred_combined,
                ytrue_combined,
                max_chain_samples=max_chain_samples,
                chain_grid=chain_sizes if chain_grid is None else chain_grid,
                n_grid=n_grid,
            )

    return chain_sizes, ypred_var_by_key, mean_ypred_var, ypred_mae_by_key, mean_ypred_mae


def _supercells_for_model(model_name: str) -> int:
    if model_name in ("DRIP", "Tersoff+DRIP"):
        return 2
    return 1


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="MCMC convergence diagnostics (n_samples and chain_samples).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_model_name_arg(parser, default="MK")
    parser.add_argument("-B", "--beta", type=float, default=1.0)
    parser.add_argument("-M", "--M", type=int, default=10)
    parser.add_argument("-W", "--W", type=int, default=6)
    parser.add_argument("--POD-index", type=int, default=None, dest="pod_index")
    parser.add_argument("--allegro-checkpoint", type=str, default=None, dest="allegro_checkpoint")
    parser.add_argument("--allegro-r-max", type=float, default=5.0, dest="allegro_r_max")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"MCMC steps to run (default {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--max-chain-samples",
        type=int,
        default=DEFAULT_MAX_CHAIN_SAMPLES,
        help=f"Upper target for chain_samples grid (default {DEFAULT_MAX_CHAIN_SAMPLES}).",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=DEFAULT_STEP_STRIDE,
        help="Stride for parameter-variance step grid.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help="Discard this many initial MCMC steps before variance accumulation.",
    )
    parser.add_argument(
        "--n-param-traces",
        type=int,
        default=DEFAULT_N_PARAM_TRACES,
        help="Number of parameters to show on variance and single-walker trace plots.",
    )
    parser.add_argument(
        "--trace-walker",
        type=int,
        default=0,
        help="Walker index for single-chain parameter trace plot (default 0).",
    )
    parser.add_argument(
        "--chain-grid",
        type=int,
        nargs="*",
        default=None,
        help="Explicit chain_samples values for predictive-variance / MAE plots.",
    )
    parser.add_argument(
        "--n-chain-grid",
        type=int,
        default=DEFAULT_N_CHAIN_GRID,
        help=(
            f"Number of equally spaced chain_samples points for MAE / variance "
            f"curves (default {DEFAULT_N_CHAIN_GRID})."
        ),
    )
    parser.add_argument(
        "--predict-chunk",
        type=int,
        default=50,
        help="Chunk size for batched prediction evaluation on the flat chain.",
    )
    parser.add_argument(
        "--max-eval-chain",
        type=int,
        default=None,
        help=(
            "Max flat-chain states to evaluate for predictive variance / MAE "
            "(evenly thinned from the full chain). Default: same as --max-chain-samples. "
            "Set 0 to evaluate every state (can exhaust memory on long runs)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for NPZ + figures.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Load saved convergence NPZ and regenerate plots only.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use a worker pool for MCMC only (emcee). Predictions run serially unless --parallel-eval is set.",
    )
    parser.add_argument(
        "--parallel-eval",
        action="store_true",
        dest="parallel_eval",
        help="Use a fresh worker pool for flat-chain prediction evaluation (after MCMC pool is shut down).",
    )
    parser.add_argument("--n-workers", type=int, default=None, dest="n_workers")
    parser.add_argument("--step-size", type=float, default=MCMC_DEFAULTS["step_size"])
    parser.add_argument("--no-save-chain", action="store_true", help="Omit flat_chain from NPZ.")
    add_hyperparam_args(parser)
    args, unknown = parser.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, unknown)

    if args.replay is not None:
        saved = load_convergence_npz(args.replay)
        meta = saved["meta"]
        model_name = meta["model_name"]
        beta = float(meta["beta"])
        labels = meta.get("param_labels", [])
        out_dir = args.replay.parent

        chain_sizes = saved["chain_sizes"]
        mean_ypred_var = saved["mean_ypred_var"]
        mean_ypred_mae = saved.get("mean_ypred_mae", np.array([], dtype=float))
        ypred_var_by_key = dict(saved["ypred_var_by_key"])
        ypred_mae_by_key = dict(saved.get("ypred_mae_by_key", {}))

        if "flat_chain" in saved:
            print(
                f"Recomputing predictive curves from flat_chain "
                f"({saved['flat_chain'].shape[0]} states, "
                f"n_chain_grid={args.n_chain_grid}) …",
                flush=True,
            )
            sc = _supercells_for_model(model_name)
            calc, x_train, x_test, _xf, y_train, y_test, _y, _yp, params, bounds = (
                get_MCMC_inputs(model_name, supercells=sc)
            )
            model_keys = list(calc.keys())
            _t0 = time()
            (
                chain_sizes,
                ypred_var_by_key,
                mean_ypred_var,
                ypred_mae_by_key,
                mean_ypred_mae,
            ) = recompute_chain_sample_curves(
                saved["flat_chain"],
                calc,
                x_test,
                y_test,
                params,
                model_keys,
                max_chain_samples=int(
                    meta.get("max_chain_samples", DEFAULT_MAX_CHAIN_SAMPLES)
                ),
                n_grid=args.n_chain_grid,
                chain_grid=args.chain_grid,
                predict_chunk=args.predict_chunk,
            )
            print(
                f"[timing] replay predictive curves: {time() - _t0:.1f}s  "
                f"({chain_sizes.size} chain_samples points)",
                flush=True,
            )

        plot_parameter_variance(
            saved["steps"],
            saved["mean_param_var"],
            saved["param_var"],
            labels,
            n_traces=int(meta.get("n_param_traces", DEFAULT_N_PARAM_TRACES)),
            model_name=model_name,
            beta=beta,
            out_path=out_dir / "param_variance_vs_n_samples.png",
            reference_n_samples=MCMC_DEFAULTS["n_samples"],
        )
        if "walker_trace_steps" in saved and "walker_trace_values" in saved:
            trace_idx = meta.get("walker_trace_param_idx", [])
            trace_labels = [
                labels[int(i)] if int(i) < len(labels) else f"param {i}"
                for i in trace_idx
            ]
            bf = meta.get("walker_trace_best_fit")
            plot_walker_parameter_traces(
                saved["walker_trace_steps"],
                saved["walker_trace_values"],
                trace_labels,
                walker_index=int(meta.get("trace_walker", 0)),
                model_name=model_name,
                beta=beta,
                out_path=out_dir / "param_trace_one_walker.png",
                best_fit=np.asarray(bf, dtype=float) if bf is not None else None,
                reference_n_samples=MCMC_DEFAULTS["n_samples"],
            )
        for key, arr in ypred_var_by_key.items():
            plot_ypred_variance(
                chain_sizes,
                arr,
                model_name=model_name,
                beta=beta,
                observable_label=key,
                out_path=out_dir / f"ypred_variance_vs_chain_samples_{key}.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        if mean_ypred_var.size:
            plot_ypred_variance(
                chain_sizes,
                mean_ypred_var,
                model_name=model_name,
                beta=beta,
                observable_label="all observables",
                out_path=out_dir / "ypred_variance_vs_chain_samples.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        for key, arr in ypred_mae_by_key.items():
            plot_mean_ypred_mae(
                chain_sizes,
                arr,
                model_name=model_name,
                beta=beta,
                observable_label=key,
                out_path=out_dir / f"mean_ypred_mae_vs_chain_samples_{key}.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        if mean_ypred_mae.size:
            plot_mean_ypred_mae(
                chain_sizes,
                mean_ypred_mae,
                model_name=model_name,
                beta=beta,
                observable_label="all observables",
                out_path=out_dir / "mean_ypred_mae_vs_chain_samples.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        print(f"Replotted from {args.replay}", flush=True)
        return

    from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

    def _get_mcmc_kw():
        kw = {"M": args.M, "W": args.W}
        if args.pod_index is not None:
            pod_hp, pod_cutoff, pod_hash = pod_hyperparams_for_index(args.pod_index)
            kw["pod_hyperparams"] = pod_hp
            kw["pod_cutoff"] = pod_cutoff
            kw["pod_hash"] = pod_hash
        if args.allegro_checkpoint is not None:
            kw["allegro_checkpoint"] = args.allegro_checkpoint
        if args.allegro_r_max is not None:
            kw["allegro_r_max"] = args.allegro_r_max
        kw.update(cli_hyperparams)
        return kw

    _mcmc_kw = _get_mcmc_kw()
    model_name = expand_ensemble_model_name(args.model_name, args, _mcmc_kw)
    sc = _supercells_for_model(model_name)

    print(f"[convergence] model={model_name}  beta={args.beta}", flush=True)
    _t0 = time()
    calc, x_train, x_test, x_full, y_train, y_test, y_full, ypred_bf, params, bounds = (
        get_MCMC_inputs(model_name, supercells=sc, **_mcmc_kw)
    )
    print(f"[timing] get_MCMC_inputs: {time() - _t0:.1f}s", flush=True)

    weights = {"energy": 0.0, "forces": 0.0, "hopping": 0.0}
    for key in y_train:
        if isinstance(y_train[key], list):
            yvar = np.var(_ravel_observation_blocks(y_train[key]))
            n_pts = sum(np.asarray(block).size for block in y_train[key])
            weights[key] += 1.0 / n_pts / yvar
        else:
            yvar = float(np.var(y_train[key]))
            weights[key] = 1.0 / len(y_train[key]) / yvar
    weights["forces"] = 0.0

    C0 = get_C0(params, x_train, y_train, calc, weights)
    n_params = sum(len(params[k]) for k in calc)
    temperature = float(args.beta) * C0 / n_params
    print(f"C0={C0:.6g}  T={temperature:.6g}  n_params={n_params}", flush=True)

    pool = None
    if args.parallel:
        n_workers = args.n_workers or os.cpu_count()
        print(f"Multiprocessing pool (MCMC): {n_workers} workers", flush=True)
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_pool_initializer,
            initargs=(model_name, sc, _mcmc_kw, temperature, weights, params, bounds),
        )

    eval_pool = None
    try:
        _t0 = time()
        sampler, model_keys, nwalkers = run_emcee_sampler(
            x_train,
            y_train,
            temperature,
            calc,
            weights,
            params,
            bounds,
            n_samples=args.n_samples,
            step_size=args.step_size,
            pool=pool,
        )
        print(
            f"[timing] MCMC: {time() - _t0:.1f}s  "
            f"acceptance={np.mean(sampler.acceptance_fraction):.4f}",
            flush=True,
        )
        acceptance_fraction = float(np.mean(sampler.acceptance_fraction))

        # Free worker processes (each holds a full LAMMPS stack) before prediction eval.
        _shutdown_pool(pool)
        pool = None
        gc.collect()

        chain = sampler.get_chain()
        steps, mean_param_var, param_var = compute_parameter_variance_vs_steps(
            chain,
            burn_in=args.burn_in,
            step_stride=args.step_stride,
        )
        labels = parameter_labels(params, model_keys)

        n_dim = int(chain.shape[2])
        trace_param_idx = select_parameter_indices(n_dim, args.n_param_traces)
        trace_steps, trace_values = walker_parameter_traces(
            chain,
            trace_param_idx,
            walker=args.trace_walker,
        )
        trace_labels = [labels[int(i)] for i in trace_param_idx]
        trace_best_fit = best_fit_vector(params, model_keys)[trace_param_idx]

        flat_chain = sampler.get_chain(flat=True)
        del sampler, chain
        gc.collect()

        max_eval = resolve_max_eval_chain(
            args.max_eval_chain,
            args.max_chain_samples,
            int(flat_chain.shape[0]),
        )
        flat_chain, n_flat_orig = thin_flat_chain_for_eval(flat_chain, max_eval)
        if flat_chain.shape[0] < n_flat_orig:
            print(
                f"[predictions] evaluating {flat_chain.shape[0]} thinned states "
                f"(from {n_flat_orig} flat-chain rows)",
                flush=True,
            )

        if args.parallel_eval:
            n_workers = args.n_workers or os.cpu_count()
            print(f"Multiprocessing pool (predictions): {n_workers} workers", flush=True)
            eval_pool = multiprocessing.Pool(
                processes=n_workers,
                initializer=_pool_initializer,
                initargs=(model_name, sc, _mcmc_kw, temperature, weights, params, bounds),
            )
        else:
            _set_worker_globals(
                calc, x_train, y_train, temperature, weights, params, bounds,
                xdata_test=x_test, xdata_full=x_full,
            )

        _t0 = time()
        ypred_dict = evaluate_flat_chain_predictions(
            flat_chain,
            calc,
            x_test,
            y_test,
            params,
            model_keys,
            pool=eval_pool,
            chunk_size=args.predict_chunk,
        )
        print(f"[timing] flat-chain predictions: {time() - _t0:.1f}s", flush=True)

        ypred_var_by_key: Dict[str, np.ndarray] = {}
        ypred_mae_by_key: Dict[str, np.ndarray] = {}
        chain_sizes = np.array([], dtype=int)
        mean_ypred_var = np.array([], dtype=float)
        mean_ypred_mae = np.array([], dtype=float)
        chain_grid_arg = args.chain_grid
        for key in model_keys:
            if key not in ypred_dict:
                continue
            if chain_sizes.size == 0:
                chain_sizes = make_chain_samples_grid(
                    ypred_dict[key].shape[0],
                    args.max_chain_samples,
                    chain_grid=chain_grid_arg,
                    n_grid=args.n_chain_grid,
                    min_size=1,
                )
            grid_kw = {
                "chain_grid": chain_sizes if chain_grid_arg is None else chain_grid_arg,
                "n_grid": args.n_chain_grid,
            }
            _, var_curve = compute_ypred_variance_vs_chain_samples(
                ypred_dict[key],
                max_chain_samples=args.max_chain_samples,
                **grid_kw,
            )
            ypred_var_by_key[key] = var_curve
            _, mae_curve = compute_mean_ypred_mae_vs_chain_samples(
                ypred_dict[key],
                ytrue_vector(y_test, key),
                max_chain_samples=args.max_chain_samples,
                **grid_kw,
            )
            ypred_mae_by_key[key] = mae_curve

        nrow_set = {ypred_dict[k].shape[0] for k in ypred_var_by_key}
        if len(nrow_set) == 1 and ypred_var_by_key:
            ypred_combined = ypred_dict_to_matrix(ypred_dict, model_keys)
            ytrue_combined = ytrue_dict_to_vector(y_test, model_keys)
            if ypred_combined.size:
                grid_kw = {
                    "chain_grid": chain_sizes if chain_grid_arg is None else chain_grid_arg,
                    "n_grid": args.n_chain_grid,
                }
                _, mean_ypred_var = compute_ypred_variance_vs_chain_samples(
                    ypred_combined,
                    max_chain_samples=args.max_chain_samples,
                    **grid_kw,
                )
                _, mean_ypred_mae = compute_mean_ypred_mae_vs_chain_samples(
                    ypred_combined,
                    ytrue_combined,
                    max_chain_samples=args.max_chain_samples,
                    **grid_kw,
                )

        out_dir = Path(args.output_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"beta{args.beta:g}".replace(".", "p")
        npz_path = out_dir / f"convergence_{tag}.npz"

        meta = {
            "model_name": model_name,
            "beta": float(args.beta),
            "temperature": float(temperature),
            "n_samples": int(args.n_samples),
            "n_walkers": int(nwalkers),
            "burn_in": int(args.burn_in),
            "max_chain_samples": int(args.max_chain_samples),
            "max_eval_chain": int(max_eval),
            "flat_chain_orig": int(n_flat_orig),
            "acceptance_fraction": acceptance_fraction,
            "param_labels": labels,
            "n_param_traces": int(args.n_param_traces),
            "trace_walker": int(args.trace_walker),
            "walker_trace_param_idx": [int(i) for i in trace_param_idx],
            "walker_trace_best_fit": [float(x) for x in trace_best_fit],
        }

        save_convergence_npz(
            npz_path,
            meta=meta,
            steps=steps,
            mean_param_var=mean_param_var,
            param_var=param_var,
            chain_sizes=chain_sizes,
            mean_ypred_var=mean_ypred_var,
            ypred_var_by_key=ypred_var_by_key,
            mean_ypred_mae=mean_ypred_mae,
            ypred_mae_by_key=ypred_mae_by_key,
            flat_chain=None if args.no_save_chain else flat_chain,
            walker_trace_steps=trace_steps,
            walker_trace_values=trace_values,
        )

        plot_parameter_variance(
            steps,
            mean_param_var,
            param_var,
            labels,
            n_traces=args.n_param_traces,
            model_name=model_name,
            beta=float(args.beta),
            out_path=out_dir / "param_variance_vs_n_samples.png",
            reference_n_samples=MCMC_DEFAULTS["n_samples"],
        )
        plot_walker_parameter_traces(
            trace_steps,
            trace_values,
            trace_labels,
            walker_index=int(args.trace_walker),
            model_name=model_name,
            beta=float(args.beta),
            out_path=out_dir / "param_trace_one_walker.png",
            best_fit=trace_best_fit,
            reference_n_samples=MCMC_DEFAULTS["n_samples"],
        )
        for key, var_curve in ypred_var_by_key.items():
            plot_ypred_variance(
                chain_sizes,
                var_curve,
                model_name=model_name,
                beta=float(args.beta),
                observable_label=key,
                out_path=out_dir / f"ypred_variance_vs_chain_samples_{key}.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        if mean_ypred_var.size:
            plot_ypred_variance(
                chain_sizes,
                mean_ypred_var,
                model_name=model_name,
                beta=float(args.beta),
                observable_label="all observables",
                out_path=out_dir / "ypred_variance_vs_chain_samples.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        for key, mae_curve in ypred_mae_by_key.items():
            plot_mean_ypred_mae(
                chain_sizes,
                mae_curve,
                model_name=model_name,
                beta=float(args.beta),
                observable_label=key,
                out_path=out_dir / f"mean_ypred_mae_vs_chain_samples_{key}.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )
        if mean_ypred_mae.size:
            plot_mean_ypred_mae(
                chain_sizes,
                mean_ypred_mae,
                model_name=model_name,
                beta=float(args.beta),
                observable_label="all observables",
                out_path=out_dir / "mean_ypred_mae_vs_chain_samples.png",
                reference_chain_samples=MCMC_DEFAULTS["chain_samples"],
            )

        print(f"Wrote {npz_path}", flush=True)
        print(f"Figures in {out_dir}", flush=True)
    finally:
        _shutdown_pool(eval_pool)
        _shutdown_pool(pool)


if __name__ == "__main__":
    main()
