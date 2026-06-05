"""
fit_model_mlp.py — Train the MLP-POD model on a subset of the BLG dataset.

Workflow
--------
1. Load xyz data, subsample DATA_FRACTION structures
2. Pre-compute (or load cached) global POD descriptors + Jacobians via LAMMPS
3. Train/test split
4. Train MLP with Adam + ReduceLROnPlateau
5. Save model checkpoint + loss history
6. Report train/test MAE

Run from the mlp_pod directory:
    python fit_model_mlp.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
from pod_descriptor import PODDescriptorCalculator
from mlp_model import MLP, DTYPE

import ase.io


# ── Data loading ──────────────────────────────────────────────────────────────

def load_structures(data_file: str, fraction: float, seed: int):
    """Load xyz file and subsample a deterministic fraction of structures."""
    print(f"Loading structures from {data_file} ...")
    atoms_all = ase.io.read(data_file, index=":")
    n_total   = len(atoms_all)
    print(f"  {n_total} structures found.")

    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n_total * fraction)))
    idx = np.sort(rng.choice(n_total, n_keep, replace=False))
    atoms = [atoms_all[i] for i in idx]
    print(f"  Using {len(atoms)} structures (DATA_FRACTION={fraction}).")
    return atoms


def extract_dft_labels(atoms_list):
    """Return per-structure energies (eV) and forces (eV/Å) from ASE calculators."""
    energies = np.array([a.get_potential_energy() for a in atoms_list], dtype=np.float64)
    forces   = [a.get_forces() for a in atoms_list]
    return energies, forces


# ── Descriptor pre-computation (cached) ──────────────────────────────────────

def _jacobians_path(cache_path: str) -> str:
    """Companion file for ragged per-structure Jacobians (list of 2-D arrays)."""
    base, _ = os.path.splitext(cache_path)
    return base + "_jacobians.npy"


def compute_or_load_descriptors(atoms_list, energies, forces, cache_path: str):
    """Load descriptors from cache, or compute them and save.

    The descriptor cache (``cache_path``) stores a fixed-shape ``.npz`` with
    descriptors, energies, and per-structure atom counts.  Jacobians — which
    are ragged (different atom counts per structure) — are stored alongside in
    a separate ``.npy`` file loaded with ``allow_pickle=True``.
    """
    jac_path = _jacobians_path(cache_path)

    if os.path.exists(cache_path) and os.path.exists(jac_path):
        print(f"Loading descriptor cache from {cache_path} ...")
        data      = np.load(cache_path, allow_pickle=True)
        jacobians = list(np.load(jac_path, allow_pickle=True))
        descriptors = data["descriptors"]
        print(f"  Cache hit: descriptors {descriptors.shape}, "
              f"{len(jacobians)} Jacobians")
        return descriptors, jacobians

    print("Descriptor cache not found — computing via LAMMPS ...")
    t0 = time.time()
    with PODDescriptorCalculator(
        hyperparams=cfg.POD_HYPERPARAMS,
        elements=cfg.POD_ELEMENTS,
        cutoff=cfg.POD_CUTOFF,
    ) as calc:
        descriptors, jacobians = calc.compute_descriptors_and_jacobians(
            atoms_list, verbose=True
        )

    elapsed = time.time() - t0
    print(f"  Computation finished in {elapsed:.1f}s.")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    n_atoms_per_struct = np.array([len(a) for a in atoms_list], dtype=np.int64)

    # Save descriptors + energies in a fixed-shape npz
    np.savez(
        cache_path,
        descriptors=descriptors,
        energies=energies,
        forces=np.array(forces, dtype=object),
        n_atoms_per_struct=n_atoms_per_struct,
        n_desc=np.int64(descriptors.shape[1]),
        pod_hash=cfg.POD_HASH,
    )

    # Save ragged Jacobians as an object array in a separate file
    jac_arr = np.empty(len(jacobians), dtype=object)
    for s, j in enumerate(jacobians):
        jac_arr[s] = j
    np.save(jac_path, jac_arr, allow_pickle=True)

    print(f"  Saved cache to {cache_path} and {jac_path}")
    return descriptors, jacobians


# ── Train/test split ──────────────────────────────────────────────────────────

def make_split(n_struct: int, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed + 1)
    idx = np.arange(n_struct)
    rng.shuffle(idx)
    n_test  = max(1, int(round(n_struct * test_fraction)))
    n_train = n_struct - n_test
    train_idx = np.sort(idx[:n_train])
    test_idx  = np.sort(idx[n_train:])
    return train_idx, test_idx


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp(
    mlp: MLP,
    D_train: np.ndarray,
    E_train: np.ndarray,    # already shifted to near-zero (residuals from per-atom mean)
    D_test:  np.ndarray,
    E_test:  np.ndarray,    # same shift applied
    n_atoms: int,           # modal atom count — only used for per-atom MAE display
    J_train = None,
    F_train = None,
) -> dict:
    """Train the MLP on variance-normalised energy residuals.

    Parameters
    ----------
    D_train  : (n_train, n_desc)
    E_train  : (n_train,)  — shifted energies (E_total - E_mean_per_atom * na)
    D_test   : (n_test,  n_desc)
    E_test   : (n_test,)   — same shift as E_train
    n_atoms  : int          — modal atom count (display only)
    """
    n_train    = len(D_train)
    use_forces = cfg.COMPUTE_FORCES and J_train is not None and F_train is not None

    D_t = torch.tensor(D_train, dtype=DTYPE)
    E_t = torch.tensor(E_train, dtype=DTYPE)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=50, factor=0.5, min_lr=1e-7
    )
    mse_fn = nn.MSELoss()

    # Normalise loss by variance of training energies (user suggestion).
    # After shifting by per-atom mean, E_train has small variance = the signal
    # the MLP needs to learn. Dividing loss by this variance makes the loss
    # dimensionless and well-conditioned regardless of the energy scale.
    E_var = float(np.var(E_train))
    if E_var < 1e-30:
        E_var = 1.0
    w_e = 1.0 / E_var
    print(f"  Energy residual std = {np.std(E_train)*1e3:.4f} meV  "
          f"  var = {E_var:.4e} eV²  →  w_e = {w_e:.4e}")

    # Force normalisation weight
    if use_forces:
        F_flat = np.concatenate([f.ravel() for f in F_train])
        F_scale = float(np.linalg.norm(F_flat))
        if F_scale < 1e-12:
            F_scale = 1.0
        w_f = cfg.FORCE_WEIGHT / F_scale

    train_losses, test_maes = [], []

    print(f"\nTraining MLP: n_train={n_train}, n_desc={D_train.shape[1]}, "
          f"n_params={mlp.n_params}, use_forces={use_forces}")
    print(f"{'Epoch':>6}  {'Loss':>12}  {'Train MAE (eV/at)':>18}  "
          f"{'Test MAE (eV/at)':>17}  {'LR':>10}")

    for epoch in range(cfg.NUM_EPOCHS):
        mlp.train()
        optimizer.zero_grad()

        E_pred = torch.stack([mlp(D_t[i]) for i in range(n_train)])
        loss = w_e * mse_fn(E_pred, E_t)

        if use_forces:
            for i in range(n_train):
                D_i = torch.tensor(D_train[i], dtype=DTYPE, requires_grad=True)
                E_i = mlp(D_i)
                dE_dD = torch.autograd.grad(E_i, D_i, create_graph=True)[0]
                # F = -J^T @ dE_dD; J_train[i] shape: (n_desc, n_atoms_i*3)
                J_i    = torch.tensor(J_train[i], dtype=DTYPE)
                n_at_i = F_train[i].shape[0]
                F_pred = -(J_i.T @ dE_dD).reshape(n_at_i, 3)
                F_ref  = torch.tensor(F_train[i], dtype=DTYPE)
                loss   = loss + w_f * mse_fn(F_pred, F_ref)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=10.0)
        optimizer.step()

        train_loss_val = loss.item()
        train_losses.append(train_loss_val)

        mlp.eval()
        with torch.no_grad():
            E_pred_np  = np.array([mlp(D_t[i]).item() for i in range(n_train)])
            train_mae  = float(np.mean(np.abs(E_pred_np - E_train))) / n_atoms

            D_test_t   = torch.tensor(D_test, dtype=DTYPE)
            E_pred_test= np.array([mlp(D_test_t[i]).item() for i in range(len(D_test))])
            test_mae   = float(np.mean(np.abs(E_pred_test - E_test))) / n_atoms
        test_maes.append(test_mae)

        scheduler.step(train_loss_val)
        lr_now = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"{epoch+1:>6}  {train_loss_val:>12.6f}  {train_mae:>18.6f}  "
                  f"{test_mae:>17.6f}  {lr_now:>10.2e}")

    return {
        "train_losses": train_losses,
        "test_maes":    test_maes,
        "best_params":  mlp.get_flat_params(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(cfg.CACHE_DIR,         exist_ok=True)
    os.makedirs(cfg.ENSEMBLE_SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR,       exist_ok=True)

    # 1. Load data
    data_file = os.path.join(_HERE, cfg.DATA_FILE)
    atoms_list = load_structures(data_file, cfg.DATA_FRACTION, cfg.RANDOM_SEED)
    energies, forces = extract_dft_labels(atoms_list)

    # Most common atom count (used only for per-atom MAE reporting)
    from collections import Counter
    n_atoms_counts = Counter(len(a) for a in atoms_list)
    n_atoms = n_atoms_counts.most_common(1)[0][0]
    print(f"  Atom counts: {dict(n_atoms_counts)} — using n_atoms={n_atoms} for MAE")

    # 2. Pre-compute / load descriptors
    cache_path = os.path.join(_HERE, cfg.DESCRIPTOR_CACHE)
    descriptors, jacobians = compute_or_load_descriptors(
        atoms_list, energies, forces, cache_path
    )
    n_desc = descriptors.shape[1]
    print(f"\nn_desc={n_desc}, n_atoms={n_atoms}, n_struct={len(atoms_list)}")

    # 3. Train/test split
    train_idx, test_idx = make_split(len(atoms_list), cfg.TEST_FRACTION, cfg.RANDOM_SEED)
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test")

    D_train, D_test = descriptors[train_idx], descriptors[test_idx]
    E_train, E_test = energies[train_idx],    energies[test_idx]
    J_train = [jacobians[i] for i in train_idx] if cfg.COMPUTE_FORCES else None
    F_train = [forces[i]   for i in train_idx] if cfg.COMPUTE_FORCES else None

    # Per-structure atom counts for proper energy normalisation
    cache_data = np.load(os.path.join(_HERE, cfg.DESCRIPTOR_CACHE), allow_pickle=True)
    if "n_atoms_per_struct" in cache_data:
        na_all   = cache_data["n_atoms_per_struct"].astype(float)
        na_train = na_all[train_idx]
        na_test  = na_all[test_idx]
    else:
        na_train = np.full(len(train_idx), float(n_atoms))
        na_test  = np.full(len(test_idx),  float(n_atoms))

    # Subtract per-atom mean energy so both 4-atom and 16-atom structures have
    # near-zero residuals. The MLP fits E_total - E_mean_per_atom * n_atoms.
    E_mean_per_atom = float(np.mean(E_train / na_train))
    E_train_fit = E_train - E_mean_per_atom * na_train
    E_test_fit  = E_test  - E_mean_per_atom * na_test
    print(f"  E_mean_per_atom = {E_mean_per_atom:.6f} eV/atom")
    print(f"  Residual std (train) = {np.std(E_train_fit)*1e3:.4f} meV")

    # Standardize descriptors by training-set mean and std (Z-score per component).
    # Without this, descriptor components with large absolute values dominate the
    # gradient and the MLP cannot converge to meV accuracy.
    D_mean = D_train.mean(axis=0)                          # (n_desc,)
    D_std  = D_train.std(axis=0)                           # (n_desc,)
    D_std  = np.where(D_std < 1e-10, 1.0, D_std)          # avoid div-by-zero
    D_train_sc = (D_train - D_mean) / D_std
    D_test_sc  = (D_test  - D_mean) / D_std
    print(f"  Descriptor mean range: [{D_mean.min():.3f}, {D_mean.max():.3f}]")
    print(f"  Descriptor std  range: [{D_std.min():.3e},  {D_std.max():.3e}]")

    # 4. Build MLP  (E0 = 0; per-atom mean handled by E_mean_per_atom above)
    mlp = MLP(
        input_dim  = n_desc,
        hidden_dim = cfg.MLP_HIDDEN_DIM,
        n_layers   = cfg.MLP_N_LAYERS,
        activation = cfg.MLP_ACTIVATION,
    )
    print(f"MLP: {n_desc} → [{cfg.MLP_HIDDEN_DIM}]×{cfg.MLP_N_LAYERS} → 1  "
          f"| n_params={mlp.n_params}")

    # 5. Train on scaled descriptors + shifted residuals, variance-normalised loss
    t0 = time.time()
    history = train_mlp(
        mlp, D_train_sc, E_train_fit, D_test_sc, E_test_fit, n_atoms,
        J_train, F_train,
    )
    print(f"\nTraining complete in {time.time() - t0:.1f}s")

    # Final MAE — add back per-atom mean for proper total-energy comparison
    mlp.eval()
    with torch.no_grad():
        E_pred_train_fit = np.array([mlp(torch.tensor(D_train_sc[i], dtype=DTYPE)).item()
                                     for i in range(len(D_train_sc))])
        E_pred_test_fit  = np.array([mlp(torch.tensor(D_test_sc[i],  dtype=DTYPE)).item()
                                     for i in range(len(D_test_sc))])
    E_pred_train = E_pred_train_fit + E_mean_per_atom * na_train
    E_pred_test  = E_pred_test_fit  + E_mean_per_atom * na_test
    mae_train = float(np.mean(np.abs(E_pred_train - E_train) / na_train))
    mae_test  = float(np.mean(np.abs(E_pred_test  - E_test)  / na_test))
    print(f"\nFinal MAE — train: {mae_train*1000:.3f} meV/atom  "
          f"| test: {mae_test*1000:.3f} meV/atom")

    # 6. Save checkpoint
    ckpt_path = os.path.join(_HERE, cfg.MODEL_CHECKPOINT)
    torch.save(
        {
            "model_state_dict":  mlp.state_dict(),
            "flat_params":       mlp.get_flat_params(),
            "train_losses":      history["train_losses"],
            "test_maes":         history["test_maes"],
            "train_idx":         train_idx,
            "test_idx":          test_idx,
            "n_desc":            n_desc,
            "n_atoms":           n_atoms,
            "hidden_dim":        cfg.MLP_HIDDEN_DIM,
            "n_layers":          cfg.MLP_N_LAYERS,
            "activation":        cfg.MLP_ACTIVATION,
            "pod_hash":          cfg.POD_HASH,
            "E_mean_per_atom":   E_mean_per_atom,
            "D_mean":            D_mean,
            "D_std":             D_std,
            "mae_train_eV_atom": mae_train,
            "mae_test_eV_atom":  mae_test,
        },
        ckpt_path,
    )
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
