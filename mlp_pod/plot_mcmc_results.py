"""
plot_mcmc_results.py — Visualise MLP-POD MCMC uncertainty quantification.

Figures produced (saved to config.FIGURES_DIR)
-----------------------------------------------
  mcmc_energy_ensemble.png
      Scatter plot: E_pred ± 2σ vs E_DFT (per atom, eV) for train and test sets.
      Each dot shows the ensemble mean; error bars show ±2σ over ensemble draws.

  mcmc_weight_uncertainty.png
      Grid of imshow panels showing the standard deviation of each MLP weight
      matrix and bias vector across the ensemble samples.
      Layout (left → right):  W₁ | b₁ → W₂ | b₂ → … → Wₒᵤₜ | bₒᵤₜ
      Output-layer weights shown as transposed (N,1) column vector.
      Arrows between layers. Single shared colorbar on the right.

Run from the mlp_pod directory:
    python plot_mcmc_results.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
from mlp_model import MLP

plt.rcParams.update({"font.size": 18, "figure.dpi": 130})


# ── Load data ──────────────────────────────────────────────────────────────────

def load_ensemble(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cache(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    result = {"descriptors": data["descriptors"], "energies": data["energies"]}
    if "n_atoms_per_struct" in data:
        result["n_atoms_per_struct"] = data["n_atoms_per_struct"].astype(int)
    return result


def rebuild_mlp(ens: dict) -> MLP:
    return MLP(
        input_dim  = int(ens["n_desc"]),
        hidden_dim = int(ens["hidden_dim"]),
        n_layers   = int(ens["n_layers"]),
        activation = str(ens["activation"]),
    )


# ── Ensemble predictions ──────────────────────────────────────────────────────

def ensemble_predictions(mlp: MLP, samples: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Evaluate all ensemble draws on D.

    Returns
    -------
    E_samples : (n_samples, n_struct)
    """
    n_samples = len(samples)
    n_struct  = len(D)
    E_samples = np.zeros((n_samples, n_struct), dtype=np.float64)
    for k, theta in enumerate(samples):
        E_samples[k] = mlp.forward_numpy_batch_with_params(D, theta)
    return E_samples


# ── Unpack MLP weights from flat parameter array ──────────────────────────────

def unpack_weights(mlp: MLP, flat_params_matrix: np.ndarray):
    """Extract per-layer weight matrices and bias vectors from the sample matrix.

    Parameters
    ----------
    flat_params_matrix : (n_samples, n_params)

    Returns
    -------
    layers : list of dicts  {"W": (n_samples, out, in), "b": (n_samples, out), "label": str}
    """
    shapes = [(name, p.shape, p.numel()) for name, p in mlp.named_parameters()]
    n_samples = flat_params_matrix.shape[0]

    layers = []
    offset = 0
    layer_idx = 0
    i = 0
    while i < len(shapes):
        name_w, shape_w, n_w = shapes[i]
        name_b, shape_b, n_b = shapes[i + 1]

        W_samples = flat_params_matrix[:, offset:offset + n_w].reshape(
            n_samples, *shape_w
        )
        offset += n_w
        b_samples = flat_params_matrix[:, offset:offset + n_b].reshape(
            n_samples, *shape_b
        )
        offset += n_b

        is_output = (i == len(shapes) - 2)
        label = f"Output layer" if is_output else f"Layer {layer_idx + 1}"
        layers.append({"W": W_samples, "b": b_samples, "label": label})
        layer_idx += 1
        i += 2

    return layers


# ── Figure 1: energy ensemble scatter ─────────────────────────────────────────

def plot_energy_ensemble(
    E_samples_train, E_train, E_samples_test, E_test,
    na_train, na_test, out_path: str,
) -> None:
    """Scatter: ensemble mean ± 2σ vs DFT energy (per atom).

    na_train / na_test : int or array-like — atom count(s) per structure.
    """
    na_tr = np.asarray(na_train, dtype=float)
    na_te = np.asarray(na_test,  dtype=float)

    mu_tr = E_samples_train.mean(axis=0) / na_tr
    sd_tr = E_samples_train.std(axis=0)  / na_tr
    mu_te = E_samples_test.mean(axis=0)  / na_te
    sd_te = E_samples_test.std(axis=0)   / na_te

    e_tr  = E_train / na_tr
    e_te  = E_test  / na_te

    all_e = np.concatenate([e_tr, e_te, mu_tr, mu_te])
    lo, hi = all_e.min() - 0.005, all_e.max() + 0.005

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)

    ax.errorbar(
        e_tr, mu_tr, yerr=2 * sd_tr,
        fmt="o", ms=5, alpha=0.6, capsize=2, lw=0.8,
        label=f"Train (n={len(e_tr)})",
    )
    ax.errorbar(
        e_te, mu_te, yerr=2 * sd_te,
        fmt="^", ms=6, alpha=0.9, capsize=3, lw=1.0,
        label=f"Test  (n={len(e_te)})",
    )

    mae_tr = float(np.mean(np.abs(mu_tr - e_tr)))
    mae_te = float(np.mean(np.abs(mu_te - e_te)))
    ax.set_title(
        f"MCMC Ensemble Energy Predictions\n"
        f"MAE train={mae_tr*1e3:.2f} meV/at, test={mae_te*1e3:.2f} meV/at"
    )
    ax.set_xlabel("E$_\\mathrm{DFT}$ (eV/atom)")
    ax.set_ylabel("E$_\\mathrm{MLP}$ ensemble mean ± 2σ (eV/atom)")
    ax.legend(fontsize=18)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 2: layerwise weight uncertainty ────────────────────────────────────

def plot_weight_uncertainty(layers: list, out_path: str) -> None:
    """Imshow grid of std(W) and std(b) for each MLP layer, left to right.

    Layout (left → right):
      [W₁ σ] [b₁ σ] → [W₂ σ] [b₂ σ] → … → [Wₒᵤₜ σ] [bₒᵤₜ σ]  [colorbar]

    - Arrows between each bias panel and the next weight panel.
    - Output-layer weight transposed to (in_dim, 1) column vector.
    - Single shared colorbar on the far right.
    """
    n_layers = len(layers)

    # ── Build panel descriptors ────────────────────────────────────────────
    # Each entry: (kind, image, title, xlabel, ylabel)
    panels = []
    for li, layer in enumerate(layers):
        is_output = (li == n_layers - 1)
        std_W = layer["W"].std(axis=0)   # (out_dim, in_dim)
        std_b = layer["b"].std(axis=0)   # (out_dim,)
        label = layer["label"]

        if is_output:
            W_img = std_W.T              # (in_dim, 1) — column vector
        else:
            W_img = std_W               # (out_dim, in_dim)

        b_img = std_b.reshape(-1, 1)    # (out_dim, 1)

        panels.append(("W", W_img, f"{label}\n$\\sigma$(W)", "", ""))
        panels.append(("b", b_img, f"$\\sigma$(b)",           "", ""))

    # ── Build column layout explicitly ────────────────────────────────────
    # Columns: [panel0, arrow?, panel1, arrow?, ..., panelN, colorbar]
    col_widths  = []   # relative width of each GridSpec column
    panel_cols  = []   # GridSpec column index for each panel
    arrow_cols  = []   # GridSpec column indices that hold arrows

    # Arrow spacer width: wide enough to avoid crowding at 18 pt font
    ARROW_W = 3

    for i, (kind, img, *_) in enumerate(panels):
        panel_cols.append(len(col_widths))
        col_widths.append(max(1, img.shape[1]))   # width ∝ image columns
        # Insert a spacer column after each bias panel except the last
        if kind == "b" and i < len(panels) - 1:
            arrow_cols.append(len(col_widths))
            col_widths.append(ARROW_W)

    # Colorbar column at the far right
    cb_col  = len(col_widths)
    cb_width = max(4, max(col_widths) // 6)
    col_widths.append(cb_width)

    # ── Figure size ───────────────────────────────────────────────────────
    # Scale up per-column width to give labels and tick text breathing room
    scale    = 0.42
    fig_w    = max(12, sum(col_widths) * scale + 2.0)
    max_rows = max(p[1].shape[0] for p in panels)
    fig_h    = max(5.0, max_rows * 0.28 + 3.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    n_gs_cols = len(col_widths)
    gs = gridspec.GridSpec(
        1, n_gs_cols,
        width_ratios=col_widths,
        wspace=0.20,
        left=0.04, right=0.96, top=0.82, bottom=0.14,
    )

    # ── Shared colour scale ────────────────────────────────────────────────
    all_vals = np.concatenate([p[1].ravel() for p in panels])
    pos_vals = all_vals[all_vals > 0]
    vmax = float(np.percentile(pos_vals, 98)) if pos_vals.size > 0 else 1e-8
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = "viridis"

    # ── Draw panels ───────────────────────────────────────────────────────
    last_im = None
    for i, (kind, img, title, xlabel, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[0, panel_cols[i]])
        im = ax.imshow(img, aspect="auto", origin="upper", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=18, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(labelsize=17)
        last_im = im

    # ── Draw arrows ───────────────────────────────────────────────────────
    for ac in arrow_cols:
        ax_arr = fig.add_subplot(gs[0, ac])
        ax_arr.set_xlim(0, 1)
        ax_arr.set_ylim(0, 1)
        ax_arr.set_axis_off()
        ax_arr.annotate(
            "", xy=(0.75, 0.5), xytext=(0.25, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0,
                            mutation_scale=20),
        )

    # ── Single colorbar in the dedicated rightmost column ─────────────────
    ax_cb = fig.add_subplot(gs[0, cb_col])
    ax_cb.set_axis_off()
    # Render the figure so axes positions are finalised
    fig.canvas.draw()
    pos = ax_cb.get_position()
    cbar_ax = fig.add_axes([pos.x0 + pos.width * 0.1, pos.y0,
                             pos.width * 0.35, pos.height])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("σ (std over ensemble)", fontsize=18)
    cb.ax.tick_params(labelsize=17)

    fig.suptitle(
        "MCMC Ensemble: MLP Weight & Bias Uncertainty (std over samples)",
        fontsize=18,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ens_path   = os.path.join(_HERE, cfg.ENSEMBLE_FILE)
    cache_path = os.path.join(_HERE, cfg.DESCRIPTOR_CACHE)

    if not os.path.exists(ens_path):
        print(f"Ensemble file not found: {ens_path}")
        print("Run  python run_mcmc.py  first.")
        return
    if not os.path.exists(cache_path):
        print(f"Descriptor cache not found: {cache_path}")
        print("Run  python fit_model_mlp.py  first.")
        return

    print(f"Loading ensemble: {ens_path}")
    ens   = load_ensemble(ens_path)
    cache = load_cache(cache_path)

    samples         = ens["samples"]         # (n_samples, n_params)
    n_atoms         = int(ens["n_atoms"])    # most-common, fallback only
    E_mean_per_atom = float(ens.get("E_mean_per_atom", 0.0))
    train_idx       = ens["train_idx"]
    test_idx        = ens["test_idx"]

    D_all = cache["descriptors"]
    E_all = cache["energies"]

    # Per-structure atom counts — use cache if available, else fallback scalar
    if "n_atoms_per_struct" in cache:
        na_all = cache["n_atoms_per_struct"]
    else:
        na_all = np.full(len(E_all), n_atoms, dtype=int)

    D_train, E_train = D_all[train_idx], E_all[train_idx]
    D_test,  E_test  = D_all[test_idx],  E_all[test_idx]
    na_train = na_all[train_idx]
    na_test  = na_all[test_idx]

    # Apply the same descriptor scaling used during training
    D_mean = ens.get("D_mean", None)
    D_std  = ens.get("D_std",  None)
    if D_mean is not None and D_std is not None:
        D_std_safe  = np.where(np.asarray(D_std) < 1e-10, 1.0, np.asarray(D_std))
        D_train_sc  = (D_train - D_mean) / D_std_safe
        D_test_sc   = (D_test  - D_mean) / D_std_safe
    else:
        D_train_sc, D_test_sc = D_train, D_test

    mlp = rebuild_mlp(ens)

    print(f"Evaluating {len(samples)} ensemble draws on train ({len(D_train)}) "
          f"and test ({len(D_test)}) sets ...")
    # MLP outputs shifted residuals; add back per-atom mean to get total energies
    E_samp_train = ensemble_predictions(mlp, samples, D_train_sc) + E_mean_per_atom * na_train
    E_samp_test  = ensemble_predictions(mlp, samples, D_test_sc)  + E_mean_per_atom * na_test

    os.makedirs(os.path.join(_HERE, cfg.FIGURES_DIR), exist_ok=True)
    figs = os.path.join(_HERE, cfg.FIGURES_DIR)

    # ── Figure 1: energy ensemble ───────────────────────────────────────
    plot_energy_ensemble(
        E_samp_train, E_train, E_samp_test, E_test,
        na_train, na_test,
        os.path.join(figs, "mcmc_energy_ensemble.png"),
    )

    # ── Figure 2: layerwise weight uncertainty ───────────────────────────
    print("Unpacking weight matrices ...")
    layers = unpack_weights(mlp, samples)
    for layer in layers:
        W, b = layer["W"], layer["b"]
        print(f"  {layer['label']}: W {W.shape[1:]}  b {b.shape[1:]}")

    plot_weight_uncertainty(
        layers,
        os.path.join(figs, "mcmc_weight_uncertainty.png"),
    )

    # ── Summary statistics ───────────────────────────────────────────────
    mu_train = E_samp_train.mean(axis=0) / na_train
    mu_test  = E_samp_test.mean(axis=0)  / na_test
    mae_tr = float(np.mean(np.abs(mu_train - E_train / na_train)))
    mae_te = float(np.mean(np.abs(mu_test  - E_test  / na_test)))

    sigma_train = E_samp_train.std(axis=0) / na_train
    sigma_test  = E_samp_test.std(axis=0)  / na_test

    print(f"\nEnsemble summary:")
    print(f"  Acceptance fraction : {ens['acceptance_fraction']:.4f}")
    print(f"  Temperature T       : {ens['T']:.4e}")
    print(f"  Ensemble size       : {len(samples)}")
    print(f"  Mean ±2σ (train)    : {sigma_train.mean()*1e3*2:.3f} meV/atom")
    print(f"  Mean ±2σ (test)     : {sigma_test.mean()*1e3*2:.3f} meV/atom")
    print(f"  Ensemble MAE train  : {mae_tr*1000:.3f} meV/atom")
    print(f"  Ensemble MAE test   : {mae_te*1000:.3f} meV/atom")
    print("\nDone — all figures saved.")


if __name__ == "__main__":
    main()
