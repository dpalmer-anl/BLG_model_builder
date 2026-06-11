"""
MCMC ensemble generation with emcee.

Total cost C(θ) = Σ_types w_n Σ_i (y_in - f_n(x_in, θ))² (energy residuals
shifted to the minimum-energy reference; forces only if the model returns
them). Scalar temperature T = T_weight * C0 * (2 / n_params) with C0 = C(θ*)
at the best fit. Log-likelihood contribution from the data: -C(θ) / T (linear
in C, not C/T²).

Parallelization
---------------
Two back-ends are supported via the ``--parallel`` / ``--mpi`` flags:

Multiprocessing (single node, recommended default)
    python EMCEE_generate_ensemble.py -m Tersoff+DRIP --parallel
    python EMCEE_generate_ensemble.py -m Tersoff+DRIP --parallel --n-workers 8

MPI (multi-node HPC cluster via schwimmbad)
    mpiexec -n 8 python EMCEE_generate_ensemble.py -m Tersoff+DRIP --mpi

In both cases every worker process calls ``get_MCMC_inputs`` once in its
initializer/startup, creating a fully independent LAMMPS instance.
``OMP_NUM_THREADS`` and common BLAS thread variables are forced to 1 in each
process so that LAMMPS/NumPy do not spawn additional threads that would
over-subscribe the CPU (especially under Slurm’s node-wide ``OMP`` exports).
"""

import multiprocessing
import os
import sys

# Slurm and other schedulers often export OMP_NUM_THREADS to the full node count.
# emcee serial mode evaluates *one* log_probability (one LAMMPS run) at a time, so
# letting OpenMP/BLAS use dozens of threads per call oversubscribes the CPU and
# can make each evaluation orders of magnitude slower than OMP_NUM_THREADS=1.
# Force low thread counts before importing NumPy / LAMMPS.
def _clamp_cpu_thread_env_for_lammps_emcee() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


_clamp_cpu_thread_env_for_lammps_emcee()

import numpy as np
import emcee
import argparse
import pickle
import pandas as pd
from blg_model_builder.model_fit import get_prediction
from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs, build_tetb_pod_hyperparams_from_data_kw
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_model_name_arg,
    collect_workflow_hyperparams,
    expand_ensemble_model_name,
)
import matplotlib.pyplot as plt
from time import time

MCMC_DEFAULTS = {
    "n_walkers_min": 100,
    "n_samples": 100,
    "walker_init_scale": 1e-6,
    "test_size": 0.2,
    "relative_weight": 0.5,
    "step_size": 0.1,
}


def _ravel_observation_blocks(y):
    """Flatten training targets to 1-D: list, ndarray, or object ndarray of blocks."""
    if isinstance(y, list):
        parts = []
        for block in y:
            parts.append(np.asarray(block, dtype=float).ravel())
        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)

    y_arr = np.asarray(y)
    if y_arr.dtype == object:
        parts = []
        for block in y_arr.ravel():
            parts.append(np.asarray(block, dtype=float).ravel())
        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)

    return np.asarray(y_arr, dtype=float).ravel()


def _mae_best_fit(y_true, y_pred):
    """Mean absolute error for ndarray or list-of-per-structure blocks."""
    yt = _ravel_observation_blocks(y_true)
    yp = _ravel_observation_blocks(y_pred)
    if yt.size != yp.size:
        raise ValueError(
            f"MAE length mismatch: y_true has {yt.size} values, y_pred has {yp.size}."
        )
    return float(np.mean(np.abs(yt - yp)))


# ── per-process worker state (set once by pool initializer or MPI startup) ───
# LAMMPS objects are not picklable, so we cannot pass calc between processes.
# Instead, each worker builds its own LAMMPS instance and stores it here.
_W_CALC = None
_W_XDATA = None        # training xdata
_W_XDATA_TEST = None   # test xdata       (used by evaluate_ensemble)
_W_XDATA_FULL = None   # full xdata       (used by evaluate_ensemble)
_W_YDATA = None
_W_T = None
_W_WEIGHTS = None
_W_THETA_BF = None
_W_BOUNDS = None

# ── diagnostics: track log_probability call count and elapsed time ────────────
_DIAG = {"lp_calls": 0, "lp_time": 0.0, "lp_last_report": 0}
_DIAG_REPORT_INTERVAL = 200  # print a summary every N log_probability calls

def worker(args):
    theta, model,x = args
    return get_prediction(model, x, theta)

def evaluate_ensemble(ensemble_samples, x, y, model, pool=None, dataset_id="full"):
    """Evaluate ensemble samples; return ypred_samples (energy, forces when available) and clean_ensemble_samples.

    Parameters
    ----------
    pool : multiprocessing.Pool or schwimmbad.MPIPool, optional
        If provided, sample evaluations are distributed across workers via
        ``pool.map(_eval_ensemble_worker, tasks)``.  Workers must have all
        three xdata variants pre-loaded in their globals (done by
        ``_set_worker_globals`` / ``_pool_initializer``).
        The pool must still be open (i.e. called before ``pool.close()``).
    dataset_id : {"full", "test", "train"}
        Which pre-loaded xdata the workers should evaluate on.
        Ignored when ``pool`` is None (``x`` is used directly).
    """
    ypred_samples = {}
    clean_ensemble_samples = {}
    # Joint TETB+POD evaluation: energy calculator expects [TB weights | POD coeffs], optional +shift.
    use_TETB = ("hopping" in model) and ("energy" in model)
    for key in ensemble_samples:
        if use_TETB and key == "energy":
            theta = np.hstack((ensemble_samples["hopping"], ensemble_samples["energy"]))
        else:
            theta = ensemble_samples[key]

        if pool is not None:
            tasks = [(dataset_id, key, theta[n, :]) for n in range(theta.shape[0])]
            n_tasks = len(tasks)
            # Chunk the map into batches so progress is printed regularly.
            # Each chunk hits all workers once; total chunks ≈ 10.
            chunk = max(1, n_tasks // 10)
            ypred_samples_list = []
            _t_eval = time()
            for start in range(0, n_tasks, chunk):
                batch = tasks[start : start + chunk]
                ypred_samples_list.extend(pool.map(_eval_ensemble_worker, batch))
                done = len(ypred_samples_list)
                elapsed = time() - _t_eval
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_tasks - done) / rate if rate > 0 else float("nan")
                print(
                    f"  [evaluate_ensemble] {key} ({dataset_id}): "
                    f"{done}/{n_tasks} samples  "
                    f"({elapsed:.1f}s elapsed, ETA {eta:.0f}s)",
                    flush=True,
                )
        else:
            ypred_samples_list = [worker((theta[n, :], model[key], x[key]))
                                  for n in range(theta.shape[0])]

        # Handle (energy, forces) tuple, list-of-config arrays (e.g. ACSF hoppings), vs ndarray stack
        first = ypred_samples_list[0]
        if isinstance(first, tuple) and len(first) == 2:
            # get_prediction(..., list xdata) with forces returns (energies, forces_list)
            energies_arr = np.array([r[0] for r in ypred_samples_list])
            forces_list = [r[1] for r in ypred_samples_list]
            ypreds = np.squeeze(energies_arr) if energies_arr.ndim > 1 else energies_arr
            if ypreds.ndim == 1:
                ypreds = ypreds[:, np.newaxis]
            nan_ind = np.isnan(ypreds).any(axis=1) if ypreds.ndim > 1 else np.isnan(ypreds)
            ypreds = ypreds[~nan_ind]
            forces_clean = [f for (i, f) in enumerate(forces_list) if not nan_ind[i]]
            ypred_samples[key] = ypreds
            ypred_samples["forces"] = forces_clean
            clean_ensemble_samples[key] = ensemble_samples[key][~nan_ind]
        elif isinstance(first, list) and len(first) > 0:
            # One entry per structure; each is ndarray (n_pairs_k,) — same layout as list ydata
            rows = []
            for r in ypred_samples_list:
                if not isinstance(r, list) or len(r) != len(first):
                    rows.append(np.array([np.nan], dtype=float))
                    continue
                try:
                    flat = np.concatenate(
                        [
                            np.asarray(r[k], dtype=float).ravel()
                            for k in range(len(r))
                        ]
                    )
                except (ValueError, TypeError):
                    flat = np.array([np.nan], dtype=float)
                rows.append(flat)
            lens = np.array([row.size for row in rows])
            L = int(np.min(lens)) if lens.size else 0
            if L > 0 and lens.size and np.any(lens != lens[0]):
                rows = [row[:L] for row in rows]
            ypreds = np.vstack(rows) if rows else np.empty((0, 0), dtype=float)
            nan_ind = ~np.isfinite(ypreds).all(axis=1) if ypreds.ndim == 2 else ~np.isfinite(ypreds)
            ypreds = ypreds[~nan_ind]
            ypred_samples[key] = ypreds
            clean_ensemble_samples[key] = ensemble_samples[key][~nan_ind]
        else:
            squeezed = [np.squeeze(np.asarray(r)) for r in ypred_samples_list]
            if squeezed and np.ndim(squeezed[0]) == 0:
                ypreds = np.column_stack([[s] for s in squeezed])
            else:
                ypreds = np.vstack(squeezed)
            nan_ind = np.isnan(ypreds).any(axis=1)
            ypreds = ypreds[~nan_ind]
            ypred_samples[key] = ypreds
            clean_ensemble_samples[key] = ensemble_samples[key][~nan_ind]
        print("shape of cleaned " + key + " ensemble = ", np.shape(clean_ensemble_samples[key]))
    return ypred_samples, clean_ensemble_samples


def _test_indices_in_full(x_full, x_test):
    """Map each test entry to its index in the full dataset (aligned train/test split).

    ``train_test_split`` in DataLoader keeps the same Python objects for list-backed
    data (e.g. lists of ``ase.Atoms``), so identity mapping works. For ndarray-backed
    data, rows are matched numerically.
    """
    if x_full is None or x_test is None:
        return None
    if isinstance(x_full, list) and isinstance(x_test, list):
        if len(x_test) > len(x_full):
            return None
        idx_map = {id(a): i for i, a in enumerate(x_full)}
        try:
            return [idx_map[id(a)] for a in x_test]
        except KeyError:
            return None
    xf = np.asarray(x_full, dtype=float)
    xt = np.asarray(x_test, dtype=float)
    if xf.ndim == 1:
        xf = xf.reshape(-1, 1)
    if xt.ndim == 1:
        xt = xt.reshape(-1, 1)
    if xt.shape[1] != xf.shape[1]:
        return None
    idx = []
    for r in range(len(xt)):
        dist = np.max(np.abs(xf - xt[r]), axis=1)
        m = np.where(dist < 1e-9)[0]
        if len(m) != 1:
            return None
        idx.append(int(m[0]))
    return idx


def slice_ypred_test_from_full(ypred_samples, xdata, xdata_test, ensemble_keys):
    """Build test-set predictions by column-slicing full predictions.

    This is valid when each row of ``ypred_samples[key]`` has one column per full
    dataset entry in the same order as ``xdata[key]`` (e.g. batch LAMMPS energy).
    It is **invalid** for layouts where predictions are flattened across structures
    (e.g. some hopping pipelines); those return ``None`` so callers can evaluate
    on ``dataset_id='test'`` separately.

    Returns
    -------
    dict or None
        Test predictions mirroring ``ypred_samples``, or ``None`` if slicing is
        unsafe or indices cannot be recovered.
    """
    test_ix = {}
    for key in ensemble_keys:
        if key == "forces":
            continue
        if key not in xdata_test or key not in xdata:
            continue
        ix = _test_indices_in_full(xdata[key], xdata_test[key])
        if ix is None:
            return None
        test_ix[key] = ix

    out = {}
    for key in ensemble_keys:
        if key == "forces":
            continue
        if key not in ypred_samples or key not in test_ix:
            return None
        yp = np.asarray(ypred_samples[key])
        ix = test_ix[key]
        n_full = len(xdata[key])
        if yp.ndim != 2 or yp.shape[1] != n_full:
            return None
        out[key] = yp[:, ix]

    if "forces" in ypred_samples and "energy" in test_ix:
        ix = test_ix["energy"]
        out["forces"] = [
            [sample_f[j] for j in ix] for sample_f in ypred_samples["forces"]
        ]

    return out


def _set_worker_globals(calc, xdata_train, ydata_train, T, weights,
                        theta_best_fit, bounds,
                        xdata_test=None, xdata_full=None):
    """Write worker state into this process's module globals."""
    global _W_CALC, _W_XDATA, _W_XDATA_TEST, _W_XDATA_FULL
    global _W_YDATA, _W_T, _W_WEIGHTS, _W_THETA_BF, _W_BOUNDS
    _W_CALC = calc
    _W_XDATA = xdata_train
    _W_XDATA_TEST = xdata_test
    _W_XDATA_FULL = xdata_full
    _W_YDATA = ydata_train
    _W_T = T
    _W_WEIGHTS = weights
    _W_THETA_BF = theta_best_fit
    _W_BOUNDS = bounds


def _pool_initializer(model_name, supercells, get_mcmc_kw, T, weights,
                      theta_best_fit, bounds):
    """Called once per worker process by multiprocessing.Pool.

    Creates a fully independent LAMMPS instance inside the worker and stores
    it in module-level globals so ``_log_prob_worker`` can access it without
    any pickling.
    """
    _clamp_cpu_thread_env_for_lammps_emcee()

    # Re-import inside the worker to avoid any shared state from fork.
    from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs as _get

    _t0 = time()
    cal, xtrain, xtest, xfull, ytrain, _yt, _y, _yp, _p, _b = _get(
        model_name, supercells=supercells, **get_mcmc_kw
    )
    print(f"[timing worker pid={os.getpid()}] get_MCMC_inputs (worker init): "
          f"{time() - _t0:.2f}s", flush=True)
    _set_worker_globals(cal, xtrain, ytrain, T, weights, theta_best_fit, bounds,
                        xdata_test=xtest, xdata_full=xfull)


def _log_prob_worker(theta):
    """Entry point called by emcee on every pool worker.

    Reads all required state from module globals (set by the pool initializer
    or the MPI rank-startup block) so that nothing needs to be pickled per call.
    """
    return log_probability(
        theta,
        _W_XDATA, _W_YDATA, _W_T,
        _W_CALC, _W_WEIGHTS, _W_THETA_BF, _W_BOUNDS,
    )


def _eval_ensemble_worker(args):
    """Entry point for parallel evaluate_ensemble.

    Args
    ----
    args : (dataset_id, key, theta_n)
        dataset_id : "train" | "test" | "full"  — selects pre-loaded xdata
        key        : model key (e.g. "energy")
        theta_n    : 1-D parameter vector for this sample

    Workers pre-load all three xdata variants in _set_worker_globals so
    nothing large needs to be pickled and sent per call.
    """
    dataset_id, key, theta_n = args
    if dataset_id == "test":
        x_key = _W_XDATA_TEST[key]
    elif dataset_id == "full":
        x_key = _W_XDATA_FULL[key]
    else:
        x_key = _W_XDATA[key]
    return get_prediction(_W_CALC[key], x_key, theta_n)


# --- residuals / SSE ----------------------------------------------------------

def get_residual_error_hopping(ydata, ypred):
    if isinstance(ydata, list):
        return [
            np.asarray(ydata[i], dtype=float) - np.asarray(ypred[i], dtype=float)
            for i in range(len(ydata))
        ]
    return np.asarray(ydata, dtype=float) - np.asarray(ypred, dtype=float)


def get_residual_error_energy(ydata, ypred):
    shift_ind = int(np.argmin(ydata))
    ypred_shift = 0 #float(ypred[shift_ind])
    ydata_shift = 0 #float(ydata[shift_ind])
    ypred_scaled = np.nan_to_num(np.asarray(ypred, dtype=float) - ypred_shift)
    ydata_scaled = np.nan_to_num(np.asarray(ydata, dtype=float) - ydata_shift)
    return ydata_scaled - ypred_scaled


def _prediction_has_nonfinite(pred):
    if pred is None:
        return False
    if isinstance(pred, (list, tuple)):
        return any(
            p is None or not np.isfinite(np.asarray(p, dtype=float)).all()
            for p in pred
        )
    a = np.asarray(pred)
    if a.dtype == object:
        return any(not np.isfinite(np.asarray(a[i], dtype=float)).all() for i in range(len(a)))
    return not np.isfinite(a.astype(float, copy=False)).all()


def _split_energy_forces(ypred):
    if isinstance(ypred, tuple) and len(ypred) == 2:
        return ypred[0], ypred[1]
    return ypred, None


def _flatten_force_pairs(ydata_forces, ypred_forces):
    """Flatten list/tuple/object array of per-config force arrays; check lengths."""
    ylist = list(ydata_forces)
    if isinstance(ypred_forces, np.ndarray) and ypred_forces.dtype == object:
        plist = [ypred_forces[i] for i in range(len(ypred_forces))]
    elif isinstance(ypred_forces, (list, tuple)):
        plist = list(ypred_forces)
    else:
        plist = [ypred_forces]
    if len(plist) != len(ylist):
        return None, None
    try:
        yflat = np.concatenate([np.ravel(np.asarray(f, dtype=float)) for f in ylist])
        pflat = np.concatenate([np.ravel(np.asarray(f, dtype=float)) for f in plist])
    except (ValueError, TypeError):
        return None, None
    if yflat.size != pflat.size:
        return None, None
    return yflat, pflat


def get_residual_error(ydata, ypred, key, ydata_forces=None, ypred_forces=None):
    if key == "energy":
        r_e = get_residual_error_energy(ydata, ypred)
        if ydata_forces is not None and ypred_forces is not None:
            yflat, pflat = _flatten_force_pairs(ydata_forces, ypred_forces)
            if yflat is None:
                return r_e, None
            return r_e, yflat - pflat
        return r_e
    if key in ("hopping", "hoppings"):
        return get_residual_error_hopping(ydata, ypred)


def _sse_hopping(ydata, ypred):
    if isinstance(ydata, list):
        if not isinstance(ypred, list) or len(ypred) != len(ydata):
            return np.inf
        sse = 0.0
        for i in range(len(ydata)):
            yi = np.asarray(ydata[i], dtype=float)
            pi = np.asarray(ypred[i], dtype=float)
            if yi.shape != pi.shape:
                return np.inf
            ri = yi - pi
            sse += float(np.nansum(ri ** 2))
        return sse
    yd = np.asarray(ydata, dtype=float)
    pr = np.asarray(ypred, dtype=float)
    if yd.shape != pr.shape:
        return np.inf
    return float(np.nansum((yd - pr) ** 2))


def _sse_energy(ydata, ypred):
    r = get_residual_error_energy(np.asarray(ydata, dtype=float), np.asarray(ypred, dtype=float))
    return float(np.nansum(np.asarray(r, dtype=float) ** 2))


def _component_weight(weights, key, fallback_key=None):
    if isinstance(weights, dict):
        if key in weights:
            return float(weights[key])
        if fallback_key is not None and fallback_key in weights:
            return float(weights[fallback_key])
        return 1.0
    return float(weights)


def weighted_cost(key, partial_theta, x_key, y, model_block, weights):
    """Return w * SSE contribution for one model block (energy+forces together when applicable)."""
    ydict = y if isinstance(y, dict) else {"energy": y}
    y_obs = ydict[key] if key in ydict else y

    ypred = get_prediction(model_block, x_key, partial_theta)
    ypred_e, ypred_f = _split_energy_forces(ypred)

    if ypred_e is None or _prediction_has_nonfinite(ypred_e):
        return np.inf

    w_key = _component_weight(weights, key)

    y_f = ydict.get("forces") if isinstance(y, dict) else None

    if y_f is not None and ypred_f is not None:
        if _prediction_has_nonfinite(ypred_f):
            return np.inf
        yflat, pflat = _flatten_force_pairs(y_f, ypred_f)
        if yflat is None:
            return np.inf
        sse_e = _sse_energy(y_obs, ypred_e)
        sse_f = float(np.nansum((yflat - pflat) ** 2))
        w_f = _component_weight(weights, "forces", fallback_key=key)
        return w_key * sse_e + w_f * sse_f

    if key in ("hopping", "hoppings"):
        sse = _sse_hopping(y_obs, ypred_e)
        return np.inf if not np.isfinite(sse) else w_key * sse

    sse = _sse_energy(y_obs, ypred_e)
    return np.inf if not np.isfinite(sse) else w_key * sse


def get_C0(theta, x, y, model, weights):
    """Scalar C(θ) at ``theta`` (dict per model key), same definition as in log_probability."""
    C = 0.0
    for key in model:
        c = weighted_cost(key, theta[key], x[key], y, model[key], weights)
        if not np.isfinite(c):
            return np.inf
        C += c
    return C


# --- prior & full log-prob ----------------------------------------------------

def logprior_uniform(x: np.ndarray, bounds: np.ndarray) -> float:
    l_bounds, u_bounds = bounds
    if all(np.less(x, u_bounds)) and all(np.greater(x, l_bounds)):
        return 0.0
    return -np.inf


def log_probability(theta, x, y, T, model, weights,
                    theta_best_fit, bounds):
    _lp_t0 = time()
    log_prior = 0.0
    cost = 0.0
    theta_ind = 0
    use_TETB = ("hopping" in model and "energy" in model)
    for key in model:
        if use_TETB and key == "energy":
            partial_theta = theta.copy()
            low_bound = np.append(bounds["hopping"][:, 0].copy(),
                                  bounds["energy"][:, 0].copy())
            up_bound = np.append(bounds["hopping"][:, 1].copy(),
                                 bounds["energy"][:, 1].copy())
        else:
            partial_theta = theta[theta_ind:theta_ind + len(theta_best_fit[key])]
            low_bound = bounds[key][:, 0]
            up_bound = bounds[key][:, 1]
        theta_ind += len(theta_best_fit[key])
        lpu = logprior_uniform(partial_theta, (low_bound, up_bound))
        if lpu == -np.inf:
            _DIAG["lp_calls"] += 1
            return -np.inf
        log_prior += lpu
        c = weighted_cost(key, partial_theta, x[key], y, model[key], weights)
        if not np.isfinite(c):
            _DIAG["lp_calls"] += 1
            return -np.inf
        cost += c

    _DIAG["lp_calls"] += 1
    _DIAG["lp_time"] += time() - _lp_t0
    n = _DIAG["lp_calls"]
    if n - _DIAG["lp_last_report"] >= _DIAG_REPORT_INTERVAL:
        _DIAG["lp_last_report"] = n
        mean_ms = _DIAG["lp_time"] / n * 1000
        print(f"[timing] log_probability: {n} calls completed, "
              f"mean {mean_ms:.1f} ms/call, "
              f"cumulative {_DIAG['lp_time']:.1f}s", flush=True)

    return log_prior - 0.5 * cost / T


# --- sampler -----------------------------------------------------------------

def get_MCMC_ensemble(x, y, T, model, weights, theta_best_fit,
                      bounds, N_samples=None, step_size=None, pool=None):
    """Generate MCMC ensemble using emcee.

    Parameters are dicts keyed by ``"energy"`` and/or ``"hopping"``.

    Parameters
    ----------
    pool : multiprocessing.Pool or schwimmbad.MPIPool, optional
        If provided, emcee distributes walker evaluations across the pool.
        Each worker must have already been initialized (via the pool initializer
        or the MPI rank-startup block) so ``_log_prob_worker`` can run without
        any inter-process data transfer.

    Returns
    -------
    sample_dict : dict of np.ndarray
    acceptance_fraction : float
    """
    if N_samples is None:
        N_samples = MCMC_DEFAULTS["n_samples"]
    if step_size is None:
        step_size = MCMC_DEFAULTS["step_size"]

    nwalkers = 0
    ndim = {}
    for key in model:
        nw = int(2 * len(theta_best_fit[key]))
        nd = len(theta_best_fit[key])
        nwalkers += nw
        ndim[key] = nd

    nwalkers = max(nwalkers, MCMC_DEFAULTS["n_walkers_min"])
    print("nwalkers =", nwalkers)

    if pool is None:
        # Serial emcee: one walker likelihood at a time. Do *not* tie OMP to
        # nwalkers or os.cpu_count() — that spawns hundreds of threads inside a
        # single LAMMPS evaluate_batch and destroys throughput on clusters.
        _clamp_cpu_thread_env_for_lammps_emcee()
        print(
            "Serial MCMC: OMP/BLAS thread env clamped to 1 per process "
            f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r})",
            flush=True,
        )
    else:
        # Parallel: each worker process handles one walker evaluation at a time.
        _clamp_cpu_thread_env_for_lammps_emcee()

    theta_walkers = None
    for it, key in enumerate(model):
        scale = MCMC_DEFAULTS["walker_init_scale"] * np.abs(theta_best_fit[key])
        new_block = np.random.normal(
            loc=theta_best_fit[key], scale=scale, size=(nwalkers, ndim[key])
        )
        if it == 0:
            theta_walkers = new_block
            print("starting walkers")
        else:
            theta_walkers = np.append(theta_walkers, new_block, axis=1)

    move = emcee.moves.StretchMove(a=step_size)
    nsteps = N_samples
    print("running", nsteps)

    if pool is not None:
        # Workers evaluate log_probability using their own LAMMPS instances
        # stored in module globals.  No args are passed per-call.
        log_fn = _log_prob_worker
        log_args = ()
    else:
        log_fn = log_probability
        log_args = (x, y, T, model, weights, theta_best_fit, bounds)

    sampler = emcee.EnsembleSampler(
        nwalkers, np.shape(theta_walkers)[1], log_fn,
        args=log_args,
        moves=move,
        pool=pool,
    )

    _t_run = time()
    sampler.run_mcmc(theta_walkers, nsteps, progress=True)
    _dt_run = time() - _t_run
    print(f"[timing] sampler.run_mcmc: {_dt_run:.2f}s total  |  "
          f"{_dt_run / nsteps:.2f}s/step  |  "
          f"~{_dt_run / nsteps / nwalkers * 1000:.1f}ms per walker eval",
          flush=True)

    acceptance_fraction = sampler.acceptance_fraction
    print("Mean acceptance fraction: {:.8f}".format(np.mean(acceptance_fraction)))
    samples = sampler.get_chain(flat=True)
    if len(samples) > 1000:
        samples = samples[::int(len(samples) / 1000)]

    print("Shape of ensemble =", np.shape(samples))
    sample_dict = {}
    theta_ind = 0
    for key in model:
        sample_dict[key] = samples[:, theta_ind:theta_ind + len(theta_best_fit[key])]
        theta_ind += len(theta_best_fit[key])
    return sample_dict, np.mean(acceptance_fraction)


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    # Required for multiprocessing on Windows / macOS ("spawn" default).
    # Safe to call on Linux too (overrides "fork" default for safety with LAMMPS).
    multiprocessing.set_start_method("spawn", force=True)

    

    parser = argparse.ArgumentParser(
        description="Run emcee MCMC ensemble sampling for BLG potentials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
Serial (default):
  python EMCEE_generate_ensemble.py -m Tersoff+DRIP

Multiprocessing on all CPU cores:
  python EMCEE_generate_ensemble.py -m Tersoff+DRIP --parallel

Multiprocessing with 8 workers:
  python EMCEE_generate_ensemble.py -m Tersoff+DRIP --parallel --n-workers 8

MPI across 8 ranks (requires schwimmbad):
  mpiexec -n 8 python EMCEE_generate_ensemble.py -m Tersoff+DRIP --mpi

TETB+POD (ensemble directory name includes TB M/W derived from ``-M`` / ``-W``
and optionally ``--POD-index`` (selecting a passing hyperparameter-search POD
descriptor), e.g. ensemble directories under
``ensembles/TETB_POD_tb_M_10_W_6_pod_M_12_W_6_POD_index_0_<hash>/``:
  python EMCEE_generate_ensemble.py -m TETB_POD -B 0.0001 -M 10 -W 6
""",
    )
    add_model_name_arg(parser, default="MK")
    parser.add_argument("-B", "--beta", type=str, default="1")
    parser.add_argument(
        "-M", "--M", type=int, default=10,
        help=(
            "Default radial count. For ACSF_hoppings and the ACSF hopping block in "
            "TETB_POD, this sets number of radial basis functions. Appended to model name cache tag "
            "when applicable."
        ),
    )
    parser.add_argument("-W", "--W", type=int, default=6)
    parser.add_argument(
        "--POD-index",
        type=int,
        default=None,
        dest="pod_index",
        help=(
            "Select a POD potential by index into use_pod_models_hash.txt "
            "(0-based). Each line is a hyperparameter-search hash; rows are "
            "looked up in pod_hyperparam_search_results.csv."
        ),
    )
    parser.add_argument(
        "--allegro-checkpoint",
        type=str,
        default=None,
        dest="allegro_checkpoint",
        help=(
            "Path to a trained Allegro .ckpt file (for -m Allegro_energy). "
            "Defaults to initial_allegro_tests/allegro_blg_output/best-v2.ckpt "
            "(small model, ~1760 params)."
        ),
    )
    parser.add_argument(
        "--allegro-r-max",
        type=float,
        default=5.0,
        dest="allegro_r_max",
        help="Neighbor cutoff (Å) used when building the Allegro calculator.",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Distribute walker evaluations across CPU cores using multiprocessing.Pool.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        dest="n_workers",
        help="Number of worker processes for --parallel (default: all CPU cores).",
    )
    parser.add_argument(
        "--mpi", action="store_true",
        help="Distribute walker evaluations across MPI ranks (requires schwimmbad). "
             "Launch with: mpiexec -n N python EMCEE_generate_ensemble.py --mpi",
    )
    parser.add_argument(
        "--eval-full", action="store_true", dest="eval_full",
        help=(
            "Evaluate the ensemble on the full dataset (train + test) in addition to "
            "the test set. By default only the test set is evaluated, which is faster "
            "for batch-backed calculators that run LAMMPS on all configs regardless."
        ),
    )
    add_hyperparam_args(parser)
    args, _unknown = parser.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, _unknown)
    if cli_hyperparams:
        print(f"[EMCEE] CLI hyperparameters: {cli_hyperparams}", flush=True)

    from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

    def _pod_hyperparams_from_search_index(pod_index: int):
        """Return (pod_hyperparams, pod_cutoff, hash) for ``use_pod_models_hash.txt`` index."""
        return pod_hyperparams_for_index(pod_index)

    def _get_mcmc_kw():
        """Keyword dict for ``get_MCMC_inputs`` (shared by main process and pool workers)."""
        kw = {"M": args.M, "W": args.W}
        if args.pod_index is not None:
            pod_hp, pod_cutoff, pod_hash = _pod_hyperparams_from_search_index(args.pod_index)
            kw["pod_hyperparams"] = pod_hp
            kw["pod_cutoff"] = pod_cutoff
            kw["pod_hash"] = pod_hash  # for tagging/logging in callers (ignored by get_MCMC_inputs)
        if args.allegro_checkpoint is not None:
            kw["allegro_checkpoint"] = args.allegro_checkpoint
        if args.allegro_r_max is not None:
            kw["allegro_r_max"] = args.allegro_r_max
        # Generic CLI hyperparameters override everything above so users can
        # tune any model-specific knob (e.g. --two_body_radial 2) without a
        # dedicated flag.
        kw.update(cli_hyperparams)
        return kw

    if args.parallel and args.mpi:
        parser.error("--parallel and --mpi are mutually exclusive.")

    _mcmc_kw = _get_mcmc_kw()
    model_name = expand_ensemble_model_name(args.model_name, args, _mcmc_kw)

    Temperature_weight = float(args.beta)

    if model_name == "DRIP" or model_name == "Tersoff+DRIP":
        sc = 2
    else:
        sc = 1

    # ── For MPI: all ranks initialize their own LAMMPS instance here, then
    # non-master ranks enter pool.wait().  For multiprocessing: only the main
    # process runs get_MCMC_inputs; workers run it inside _pool_initializer.
    print("[timing] starting get_MCMC_inputs ...", flush=True)
    _t0 = time()
    if args.pod_index is not None and args.model_name in ("POD_energy", "TETB_POD"):
        print(
            f"[EMCEE] POD_index={int(args.pod_index)} "
            f"(hash={_mcmc_kw.get('pod_hash', 'unknown')}, "
            f"rcut={_mcmc_kw.get('pod_cutoff', 'unknown')})",
            flush=True,
        )
    calc, xdata_train, xdata_test, xdata, ydata_train, ydata_test, ydata, \
        ypred_bestfit, params, bounds = \
        get_MCMC_inputs(model_name, supercells=sc, **_mcmc_kw)
    print(f"[timing] get_MCMC_inputs: {time() - _t0:.2f}s", flush=True)

    for key in calc:
        mae = _mae_best_fit(ydata_train[key], ypred_bestfit[key])
        print(key, " MAE from best fit =", mae)

    if not os.path.exists("ensembles/" + model_name + "/"):
        os.makedirs("ensembles/" + model_name + "/", exist_ok=True)

    w0 = {"energy": 0.0, "forces": 0.0, "hopping": 0.0}
    for key in ydata_train:
        if isinstance(ydata_train[key], list):
            yvar = np.var(_ravel_observation_blocks(ydata_train[key]))
            num_data_points = 0
            for i in range(len(ydata_train[key])):
                num_data_points += ydata_train[key][i].size
            w0[key] += 1 / num_data_points / yvar
        else:
            yvar = np.var(ydata_train[key])
            w0[key] = 1 / len(ydata_train[key]) / yvar
    w0["forces"] = 0.0
    print("weights =", w0)

    _t0 = time()
    C0 = get_C0(params, xdata_train, ydata_train, calc, w0)
    print(f"[timing] get_C0 (best-fit cost eval): {time() - _t0:.2f}s", flush=True)
    n_params = sum(len(params[k]) for k in calc)
    # in Frederiksen et al. T0 = C0 * 2 / n_params, C = \sum_i (y_i - y_pred_i)^2 / 2
    # in this code, T0 = C0 / n_params, C = \sum_i (y_i - y_pred_i)^2
    Temperature = Temperature_weight * C0 * (1 / n_params) 
    print("C0 (weighted SSE at best fit):", C0)
    print("T = T_weight * C0 / n_params:", Temperature)

    step_size = 0.1
    print("running MCMC for", model_name)
    start = time()

    pool = None
    _t0 = time()
    if args.mpi:
        try:
            from schwimmbad import MPIPool
        except ImportError:
            sys.exit(
                "ERROR: schwimmbad is not installed. "
                "Install it with: pip install schwimmbad"
            )
        # Store state in this rank's globals so _log_prob_worker and
        # _eval_ensemble_worker can access data without pickling per call.
        _set_worker_globals(calc, xdata_train, ydata_train, Temperature,
                            w0, params, bounds,
                            xdata_test=xdata_test, xdata_full=xdata)
        mpi_pool = MPIPool()
        if not mpi_pool.is_master():
            mpi_pool.wait()
            sys.exit(0)
        pool = mpi_pool
        print(f"[timing] MPI pool ready: {time() - _t0:.2f}s", flush=True)

    elif args.parallel:
        n_workers = args.n_workers or os.cpu_count()
        print(f"Multiprocessing pool: {n_workers} workers (spawning + worker init) ...",
              flush=True)
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_pool_initializer,
            initargs=(model_name, sc, _mcmc_kw,
                      Temperature, w0, params, bounds),
        )
        print(f"[timing] multiprocessing pool ready ({n_workers} workers): "
              f"{time() - _t0:.2f}s", flush=True)

    _t0 = time()
    try:
        ensemble_samples, acceptance_fraction = get_MCMC_ensemble(
            xdata_train, ydata_train, Temperature, calc, w0,
            params, bounds, step_size=step_size,
            pool=pool,
        )
        print(f"[timing] get_MCMC_ensemble: {time() - _t0:.2f}s", flush=True)

        # ── Checkpoint: save ensemble parameters immediately so Slurm timeout
        # cannot lose them.  Predictions are added (overwriting) once ready.
        filename = f"ensembles/{model_name}/{model_name}_ensemble_T_{Temperature_weight}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        _checkpoint = {
            "ensemble": ensemble_samples,
            "ypred_samples": None,
            "ypred_samples_test": None,
            "ydata": ydata,
            "ydata_test": ydata_test,
            "xdata": xdata,
            "xdata_test": xdata_test,
        }
        print(f"[checkpoint] saving ensemble (no predictions yet) → {filename}", flush=True)
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "wb") as _f:
            pickle.dump(_checkpoint, _f)
        print("[checkpoint] done", flush=True)

        # evaluate_ensemble runs before pool.close() so workers are still
        # available to process tasks via pool.map(_eval_ensemble_worker, ...).
        print("getting ypred samples")
        _t0_eval = time()
        ens_keys = list(ensemble_samples.keys())

        if args.eval_full:
            # Evaluate on the full dataset. For batch-backed energy calculators
            # (``_make_batch_evaluator``), each sample already runs LAMMPS on *all*
            # configs; a separate ``dataset_id='test'`` pass repeats that full cost
            # without saving work — easily ~2× runtime. Test-set predictions are
            # obtained by slicing columns when layouts match (see
            # ``slice_ypred_test_from_full``).
            ypred_samples, _ = evaluate_ensemble(
                ensemble_samples, xdata, ydata, calc,
                pool=pool, dataset_id="full",
            )
            ypred_samples_test = slice_ypred_test_from_full(
                ypred_samples, xdata, xdata_test, ens_keys,
            )
            if ypred_samples_test is None:
                print(
                    "[evaluate_ensemble] could not derive test predictions from full "
                    "(e.g. non-columnar hopping layout); running test set separately.",
                    flush=True,
                )
                ypred_samples_test, _ = evaluate_ensemble(
                    ensemble_samples, xdata_test, ydata_test, calc,
                    pool=pool, dataset_id="test",
                )
            else:
                print(
                    "[evaluate_ensemble] test predictions = slice of full (no second pass).",
                    flush=True,
                )
        else:
            # Default: evaluate only on the test set (faster).
            print(
                "[evaluate_ensemble] evaluating test set only "
                "(use --eval-full to also evaluate the full dataset).",
                flush=True,
            )
            ypred_samples = None
            ypred_samples_test, _ = evaluate_ensemble(
                ensemble_samples, xdata_test, ydata_test, calc,
                pool=pool, dataset_id="test",
            )

        print(f"[timing] evaluate_ensemble: {time() - _t0_eval:.2f}s",
              flush=True)
    finally:
        if pool is not None:
            pool.close()
            if hasattr(pool, "join"):  # multiprocessing.Pool only; MPIPool has no join()
                pool.join()
    print(f"[timing] MCMC + evaluate_ensemble + pool teardown: {time() - _t0:.2f}s",
          flush=True)
    if ypred_samples is not None:
        for key in calc:
            print(key, " shape", np.shape(ypred_samples[key]))

    end = time()
    print("total time =", end - start)

    ensemble_dict = {
        "ensemble": ensemble_samples,
        "ypred_samples": ypred_samples,
        "ypred_samples_test": ypred_samples_test,
        "ydata": ydata,
        "ydata_test": ydata_test,
        "xdata": xdata,
        "xdata_test": xdata_test,
    }
    print(f"saving model (with predictions) → {filename}")
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "wb") as file:
        pickle.dump(ensemble_dict, file)

    """mean_ypred = np.mean(ypred_samples["energy"], axis=0)
    std_ypred = np.std(ypred_samples["energy"], axis=0)
    parity = np.linspace(0, np.max(ydata["energy"] - np.min(ydata["energy"])), 100)
    plt.errorbar(
        ydata["energy"] - np.min(ydata["energy"]),
        mean_ypred - np.min(ydata["energy"]),
        yerr=std_ypred,
        fmt="o", label="evaluated ensemble",
    )
    plt.plot(parity, parity, label="parity")
    plt.legend()
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.savefig(f"figures/ensemble_comparison_{model_name}.png")
    plt.clf()"""


if __name__ == "__main__":
    main()
