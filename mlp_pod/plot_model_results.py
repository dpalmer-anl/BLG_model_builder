"""
plot_model_results.py — Visualise MLP-POD fitting results.

Figures produced (saved to config.FIGURES_DIR)
-----------------------------------------------
  fit_scatter.png   — E_pred vs E_DFT (per atom, eV), train + test sets
  fit_residuals.png — residual histogram for train + test
  loss_curve.png    — training loss and test MAE vs epoch (log scale)

Run from the mlp_pod directory:
    python plot_model_results.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
from mlp_model import MLP, DTYPE

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


# ── Load checkpoint and cache ─────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


def load_cache(cache_path: str) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    result = {
        "descriptors": data["descriptors"],
        "energies":    data["energies"],
        "n_desc":      int(data["n_desc"]),
    }
    if "n_atoms_per_struct" in data:
        result["n_atoms_per_struct"] = data["n_atoms_per_struct"].astype(int)
    return result


def rebuild_mlp(ckpt: dict) -> MLP:
    mlp = MLP(
        input_dim  = int(ckpt["n_desc"]),
        hidden_dim = int(ckpt["hidden_dim"]),
        n_layers   = int(ckpt["n_layers"]),
        activation = str(ckpt["activation"]),
    )
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.eval()
    return mlp


# ── Predictions ───────────────────────────────────────────────────────────────

def get_predictions(mlp: MLP, D: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        D_t = torch.tensor(D, dtype=DTYPE)
        return np.array([mlp(D_t[i]).item() for i in range(len(D))])


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scatter(E_train, E_pred_train, E_test, E_pred_test,
                 na_train, na_test, out_path: str) -> None:
    """Energy parity plot (per-atom eV).

    na_train / na_test : int or array-like — atom count(s) per structure.
    """
    na_tr = np.asarray(na_train, dtype=float)
    na_te = np.asarray(na_test,  dtype=float)

    e_tr  = E_train      / na_tr
    ep_tr = E_pred_train / na_tr
    e_te  = E_test       / na_te
    ep_te = E_pred_test  / na_te

    mae_tr = float(np.mean(np.abs(ep_tr - e_tr)))
    mae_te = float(np.mean(np.abs(ep_te - e_te)))

    all_e = np.concatenate([e_tr, e_te, ep_tr, ep_te])
    lo, hi = all_e.min() - 0.005, all_e.max() + 0.005

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)
    ax.scatter(e_tr, ep_tr, s=20, alpha=0.7, label=f"Train (MAE={mae_tr*1e3:.2f} meV/at)")
    ax.scatter(e_te, ep_te, s=30, marker="^", alpha=0.9,
               label=f"Test  (MAE={mae_te*1e3:.2f} meV/at)")
    ax.set_xlabel("E$_\\mathrm{DFT}$ (eV/atom)")
    ax.set_ylabel("E$_\\mathrm{MLP}$ (eV/atom)")
    ax.set_title("MLP-POD Energy Parity")
    ax.legend(fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_residuals(E_train, E_pred_train, E_test, E_pred_test,
                   na_train, na_test, out_path: str) -> None:
    """Residual histogram for train + test."""
    res_tr = (E_pred_train - E_train) / np.asarray(na_train, float) * 1e3  # meV/atom
    res_te = (E_pred_test  - E_test)  / np.asarray(na_test,  float) * 1e3

    bins = np.linspace(
        min(res_tr.min(), res_te.min()) - 1,
        max(res_tr.max(), res_te.max()) + 1,
        40,
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(res_tr, bins=bins, alpha=0.6, label="Train")
    ax.hist(res_te, bins=bins, alpha=0.6, label="Test")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("E$_\\mathrm{MLP}$ − E$_\\mathrm{DFT}$ (meV/atom)")
    ax.set_ylabel("Count")
    ax.set_title("Energy Residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_loss_curve(train_losses, test_maes, out_path: str) -> None:
    """Training loss and test MAE on a log-scale y-axis.

    test_maes are already in eV/atom (divided by n_atoms during training).
    """
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    color1 = "steelblue"
    ax1.semilogy(epochs, train_losses, color=color1, lw=1.2, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    if test_maes:
        ax2 = ax1.twinx()
        color2 = "darkorange"
        test_maes_meV = np.array(test_maes) * 1e3   # already eV/atom → meV/atom
        ax2.semilogy(epochs, test_maes_meV, color=color2, lw=1.2,
                     ls="--", label="Test MAE")
        ax2.set_ylabel("Test MAE (meV/atom, log)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    ax1.set_title("Training Curve")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ckpt_path  = os.path.join(_HERE, cfg.MODEL_CHECKPOINT)
    cache_path = os.path.join(_HERE, cfg.DESCRIPTOR_CACHE)

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run  python fit_model_mlp.py  first.")
        return
    if not os.path.exists(cache_path):
        print(f"Descriptor cache not found: {cache_path}")
        print("Run  python fit_model_mlp.py  first.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = load_checkpoint(ckpt_path)
    cache = load_cache(cache_path)

    mlp     = rebuild_mlp(ckpt)
    n_atoms = int(ckpt["n_atoms"])   # most-common count, used as fallback

    D_all = cache["descriptors"]    # (n_struct, n_desc)
    E_all = cache["energies"]       # (n_struct,)

    # Per-structure atom counts (stored in cache since last fit run).
    # Fall back to the scalar stored in the checkpoint for older caches.
    if "n_atoms_per_struct" in cache:
        na_all = cache["n_atoms_per_struct"]
    else:
        na_all = np.full(len(E_all), n_atoms, dtype=int)

    train_idx = ckpt["train_idx"]
    test_idx  = ckpt["test_idx"]

    D_train, E_train = D_all[train_idx], E_all[train_idx]
    D_test,  E_test  = D_all[test_idx],  E_all[test_idx]
    na_train = na_all[train_idx]
    na_test  = na_all[test_idx]

    # Apply the same descriptor scaling used during training
    D_mean = ckpt.get("D_mean", np.zeros(D_train.shape[1]))
    D_std  = ckpt.get("D_std",  np.ones(D_train.shape[1]))
    D_train_sc = (D_train - D_mean) / D_std
    D_test_sc  = (D_test  - D_mean) / D_std

    # MLP was trained on (E - E_mean_per_atom * na); add back to get total energy
    E_mean_per_atom = float(ckpt.get("E_mean_per_atom", 0.0))
    E_pred_train = get_predictions(mlp, D_train_sc) + E_mean_per_atom * na_train
    E_pred_test  = get_predictions(mlp, D_test_sc)  + E_mean_per_atom * na_test

    mae_tr = float(np.mean(np.abs((E_pred_train - E_train) / na_train)))
    mae_te = float(np.mean(np.abs((E_pred_test  - E_test)  / na_test)))
    print(f"\nMAE — train: {mae_tr*1000:.3f} meV/atom  "
          f"| test: {mae_te*1000:.3f} meV/atom")

    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    figs = cfg.FIGURES_DIR

    plot_scatter(
        E_train, E_pred_train, E_test, E_pred_test,
        na_train, na_test,
        os.path.join(_HERE, figs, "fit_scatter.png"),
    )
    plot_residuals(
        E_train, E_pred_train, E_test, E_pred_test,
        na_train, na_test,
        os.path.join(_HERE, figs, "fit_residuals.png"),
    )
    plot_loss_curve(
        ckpt["train_losses"], ckpt["test_maes"],
        os.path.join(_HERE, figs, "loss_curve.png"),
    )
    print("\nDone — all figures saved.")


if __name__ == "__main__":
    main()
