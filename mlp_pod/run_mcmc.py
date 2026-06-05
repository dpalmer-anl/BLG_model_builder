"""
run_mcmc.py — MCMC uncertainty quantification for the MLP-POD model.

Uses emcee EnsembleSampler with StretchMove, following the same conventions
as EMCEE_generate_ensemble.py in the main uncertainty_quantification module.

Temperature formula (T0):
    T = T_WEIGHT * C0 * (2 / n_params)
where C0 = SSE at the best-fit parameters (same as EMCEE_generate_ensemble.py).

All log_probability evaluations use pure numpy MLP forward passes on the
pre-cached descriptors — NO LAMMPS at MCMC time.

Run from the mlp_pod directory:
    python run_mcmc.py
"""

import os
import sys
import time
import pickle
import numpy as np
import emcee

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
from mlp_model import MLP


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    cache_path = os.path.join(_HERE, cfg.DESCRIPTOR_CACHE)
    ckpt_path  = os.path.join(_HERE, cfg.MODEL_CHECKPOINT)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Descriptor cache not found: {cache_path}\n"
            "Run  python fit_model_mlp.py  first."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}\n"
            "Run  python fit_model_mlp.py  first."
        )

    import torch
    cache = np.load(cache_path, allow_pickle=True)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    descriptors      = cache["descriptors"]      # (n_struct, n_desc)
    energies         = cache["energies"]         # (n_struct,)
    train_idx        = ckpt["train_idx"]
    test_idx         = ckpt["test_idx"]
    flat_params      = ckpt["flat_params"]
    n_atoms          = int(ckpt["n_atoms"])
    n_desc           = int(ckpt["n_desc"])
    E_mean_per_atom  = float(ckpt.get("E_mean_per_atom", 0.0))
    D_mean = ckpt.get("D_mean", None)
    D_std  = ckpt.get("D_std",  None)

    # Per-structure atom counts — needed to reconstruct the training shift
    if "n_atoms_per_struct" in cache:
        na_all = cache["n_atoms_per_struct"].astype(float)
    else:
        na_all = np.full(len(energies), float(n_atoms))

    return (
        descriptors, energies, na_all, train_idx, test_idx,
        flat_params, n_atoms, n_desc, E_mean_per_atom, D_mean, D_std, ckpt,
    )


def rebuild_mlp(ckpt, n_desc: int) -> MLP:
    import torch
    mlp = MLP(
        input_dim  = n_desc,
        hidden_dim = int(ckpt["hidden_dim"]),
        n_layers   = int(ckpt["n_layers"]),
        activation = str(ckpt["activation"]),
    )
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.eval()
    return mlp


# ── Log-probability ────────────────────────────────────────────────────────────

# Module-level globals used by the log_probability function.
# Avoids pickling issues when emcee runs in a Pool.
_G_MLP       = None   # MLP instance
_G_D_TRAIN   = None   # (n_train, n_desc)
_G_E_TRAIN   = None   # (n_train,)
_G_T         = None   # float — MCMC temperature
_G_BOUNDS_LO = None   # (n_params,)
_G_BOUNDS_HI = None   # (n_params,)


def _set_globals(mlp, D_train, E_train, T, bounds_lo, bounds_hi):
    global _G_MLP, _G_D_TRAIN, _G_E_TRAIN, _G_T, _G_BOUNDS_LO, _G_BOUNDS_HI
    _G_MLP       = mlp
    _G_D_TRAIN   = D_train
    _G_E_TRAIN   = E_train
    _G_T         = T
    _G_BOUNDS_LO = bounds_lo
    _G_BOUNDS_HI = bounds_hi


def log_probability(theta: np.ndarray) -> float:
    """Log-posterior for the MLP parameters.

    log P(θ | data) ∝ -0.5 * SSE(θ) / T   [+ uniform prior]

    SSE = Σ_i (E_pred_i − E_DFT_i)²

    Uses a pure-numpy batch forward pass — no torch overhead.
    """
    # Uniform prior
    if np.any(theta < _G_BOUNDS_LO) or np.any(theta > _G_BOUNDS_HI):
        return -np.inf

    E_pred = _G_MLP.forward_numpy_batch_with_params(_G_D_TRAIN, theta)
    if not np.all(np.isfinite(E_pred)):
        return -np.inf

    sse = float(np.sum((E_pred - _G_E_TRAIN) ** 2))
    return -0.5 * sse / _G_T


# ── MCMC ──────────────────────────────────────────────────────────────────────

def run_mcmc(
    mlp: MLP,
    D_train: np.ndarray,
    E_train: np.ndarray,
    theta_best: np.ndarray,
) -> dict:
    n_params = len(theta_best)
    n_desc   = D_train.shape[1]

    # Compute C0 = SSE at best-fit parameters
    E_pred_bf = mlp.forward_numpy_batch(D_train)
    C0 = float(np.sum((E_pred_bf - E_train) ** 2))
    if C0 < 1e-30:
        C0 = 1e-30

    # T0 = T_WEIGHT * C0 * (2 / n_params)  — same formula as EMCEE_generate_ensemble.py
    T = cfg.T_WEIGHT * C0 * (2.0 / n_params)
    print(f"C0 = {C0:.6e}   T (T0) = {T:.6e}   n_params = {n_params}")

    # Bounds: ±10 * |θ_best|  (with floor to avoid zero-width bounds)
    abs_theta = np.abs(theta_best)
    width     = np.where(abs_theta > 1e-8, abs_theta, 1e-3)
    bounds_lo = theta_best - 10.0 * width
    bounds_hi = theta_best + 10.0 * width

    # Set module-level globals for log_probability
    _set_globals(mlp, D_train, E_train, T, bounds_lo, bounds_hi)

    # Walker count: emcee requires n_walkers >= 2 * n_dim (must be even)
    n_walkers = cfg.N_WALKERS
    if n_walkers is None:
        n_walkers = 2 * n_params
    n_walkers = max(n_walkers, 2 * n_params)
    if n_walkers % 2 != 0:
        n_walkers += 1
    print(f"n_walkers = {n_walkers}, n_steps = {cfg.N_STEPS}")

    # Initialise walkers around best-fit (tight Gaussian ball)
    init_scale = 1e-6 * np.abs(theta_best)
    init_scale = np.where(init_scale < 1e-12, 1e-12, init_scale)
    rng = np.random.default_rng(cfg.RANDOM_SEED + 99)
    theta_walkers = theta_best[None, :] + rng.standard_normal(
        (n_walkers, n_params)
    ) * init_scale[None, :]

    # Run sampler
    move    = emcee.moves.StretchMove(a=cfg.STEP_SIZE)
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, moves=move)

    t0 = time.time()
    sampler.run_mcmc(theta_walkers, cfg.N_STEPS, progress=True)
    dt = time.time() - t0
    print(f"\nMCMC complete: {dt:.1f}s total | {dt/cfg.N_STEPS:.2f}s/step")

    acc_frac = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance fraction: {acc_frac:.4f}")

    # Flatten chain and subsample to ≤ 1000 draws (same as EMCEE_generate_ensemble.py)
    samples   = sampler.get_chain(flat=True)          # (n_walkers * n_steps, n_params)
    log_probs = sampler.get_log_prob(flat=True)        # (n_walkers * n_steps,)
    if len(samples) > 1000:
        step = max(1, len(samples) // 1000)
        samples   = samples[::step]
        log_probs = log_probs[::step]

    print(f"Ensemble size after subsampling: {len(samples)}")

    return {
        "samples":             samples,
        "log_probs":           log_probs,
        "acceptance_fraction": acc_frac,
        "T":                   T,
        "C0":                  C0,
        "n_params":            n_params,
        "n_desc":              n_desc,
        "hidden_dim":          cfg.MLP_HIDDEN_DIM,
        "n_layers":            cfg.MLP_N_LAYERS,
        "activation":          cfg.MLP_ACTIVATION,
        "best_fit_params":     theta_best,
        "bounds_lo":           bounds_lo,
        "bounds_hi":           bounds_hi,
        "pod_hash":            cfg.POD_HASH,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    (descriptors, energies, na_all, train_idx, test_idx,
     flat_params, n_atoms, n_desc, E_mean_per_atom, D_mean, D_std, ckpt) = load_data()

    D_train = descriptors[train_idx]
    D_test  = descriptors[test_idx]
    E_train = energies[train_idx]
    E_test  = energies[test_idx]

    # Apply the same descriptor scaling used during training
    if D_mean is not None and D_std is not None:
        D_std_safe  = np.where(D_std < 1e-10, 1.0, D_std)
        D_train_sc  = (D_train - D_mean) / D_std_safe
        D_test_sc   = (D_test  - D_mean) / D_std_safe
    else:
        D_train_sc, D_test_sc = D_train, D_test

    # MCMC samples the MLP on the same shifted residuals used during training
    na_train = na_all[train_idx]
    na_test  = na_all[test_idx]
    E_train_fit = E_train - E_mean_per_atom * na_train
    E_test_fit  = E_test  - E_mean_per_atom * na_test

    mlp = rebuild_mlp(ckpt, n_desc)
    theta_best = np.asarray(flat_params, dtype=np.float64)

    print(f"n_train={len(train_idx)}  n_test={len(test_idx)}  "
          f"n_desc={n_desc}  n_params={len(theta_best)}")
    print(f"E_mean_per_atom = {E_mean_per_atom:.6f} eV/atom")

    result = run_mcmc(mlp, D_train_sc, E_train_fit, theta_best)

    # Attach split info and normalization constants for downstream plotting
    result["train_idx"]       = train_idx
    result["test_idx"]        = test_idx
    result["n_atoms"]         = n_atoms
    result["E_mean_per_atom"] = E_mean_per_atom
    result["D_mean"]          = D_mean
    result["D_std"]           = D_std

    os.makedirs(os.path.join(_HERE, cfg.ENSEMBLE_SAVE_DIR), exist_ok=True)
    ens_path = os.path.join(_HERE, cfg.ENSEMBLE_FILE)
    with open(ens_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nEnsemble saved to {ens_path}")

    # Quick sanity: ensemble energy spread (in shifted space, then per atom)
    sse_samples = []
    for k in range(min(50, len(result["samples"]))):
        theta_k = result["samples"][k]
        E_k_fit = mlp.forward_numpy_batch_with_params(D_test_sc, theta_k)
        mae_k   = float(np.mean(np.abs(E_k_fit - E_test_fit) / na_test))
        sse_samples.append(mae_k)
    print(f"Test MAE range over 50 ensemble draws: "
          f"{np.min(sse_samples)*1000:.2f} – {np.max(sse_samples)*1000:.2f} meV/atom")


if __name__ == "__main__":
    main()
