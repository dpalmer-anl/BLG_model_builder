"""
Parity plot: POD predicted energy vs DFT for train and test splits.

Uses the same interlayer rVV10 dataset and ``DataLoader.train_test_split``
(``TEST_SIZE=0.2``, ``np.random.seed(42)``) as ``run_MCMC.py`` /
``get_MCMC_inputs``.

Energies are total energy per atom (eV/atom).  Training and test points are
colored differently.

Run from ``uncertainty_quantification``::

    python visualizations/plot_parity.py --models POD_energy --POD-index 27
    python visualizations/plot_parity.py --models 'POD_energy_POD_index_27*'
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from blg_model_builder.DataLoader import TEST_SIZE, load_data_for_model
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent

_vis_dir = str(HERE)
if _vis_dir not in sys.path:
    sys.path.insert(0, _vis_dir)
_uq_dir = str(UQ_DIR)
if _uq_dir not in sys.path:
    sys.path.insert(0, _uq_dir)

from pod_plot_runtime import (  # noqa: E402
    build_model_calculator as _build_model_calculator,
    chdir_for_dataloader as _chdir_for_dataloader,
    energy_per_atom_total,
    evaluate_model_energies_on_atoms,
    is_pod_family_model as _is_pod_family_model,
    model_plot_label as _model_plot_label,
    pod_family_load_name as _pod_family_load_name,
    resolve_model_names as _resolve_model_names,
)

CSFONT = {"fontname": "sans-serif", "size": 15}
DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"

_COLOR_TRAIN = "#1f77b4"
_COLOR_TEST = "#d62728"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name.strip())


def _energies_per_atom(atoms_list: list, e_tot: np.ndarray) -> np.ndarray:
    e_tot = np.asarray(e_tot, dtype=float).ravel()
    return np.asarray(
        [energy_per_atom_total(atoms, e) for atoms, e in zip(atoms_list, e_tot)],
        dtype=float,
    )


def _load_mcmc_train_test(load_name: str):
    """
    Load interlayer energy data and apply the MCMC train/test split.

    Re-seeds NumPy to 42 immediately before the split so the indices match
    ``DataLoader`` / ``run_MCMC`` regardless of prior RNG use in this process.
    """
    np.random.seed(42)
    return load_data_for_model(
        load_name,
        supercells=1,
        level_of_theory="rVV10",
    )


def plot_pod_energy_parity(
    *,
    e_dft_train: np.ndarray,
    e_pred_train: np.ndarray,
    e_dft_test: np.ndarray,
    e_pred_test: np.ndarray,
    model_label: str,
    out_path: Path,
    dpi: int = 180,
) -> None:
    """Scatter ``E_pred`` vs ``E_dft`` (eV/atom) for train and test."""
    mae_train = float(np.mean(np.abs(e_pred_train - e_dft_train)))
    mae_test = float(np.mean(np.abs(e_pred_test - e_dft_test)))
    label_train = f"train (MAE={1e3 * mae_train:.2f} meV/atom)"
    label_test = f"test (MAE={1e3 * mae_test:.2f} meV/atom)"

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        e_dft_train,
        e_pred_train,
        s=28,
        alpha=0.75,
        color=_COLOR_TRAIN,
        edgecolors="none",
        label=label_train,
        zorder=2,
    )
    ax.scatter(
        e_dft_test,
        e_pred_test,
        s=28,
        alpha=0.75,
        color=_COLOR_TEST,
        edgecolors="none",
        label=label_test,
        zorder=3,
    )

    all_e = np.concatenate(
        [e_dft_train, e_pred_train, e_dft_test, e_pred_test],
    )
    lo = float(np.nanmin(all_e))
    hi = float(np.nanmax(all_e))
    pad = 0.02 * (hi - lo) if hi > lo else 0.01
    lims = np.array([lo - pad, hi + pad])
    ax.plot(lims, lims, "k-", linewidth=1.0, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$E_{\mathrm{dft}}$ (eV/atom)", **CSFONT)
    ax.set_ylabel(r"$E_{\mathrm{pred}}$ (eV/atom)", **CSFONT)
    ax.set_title(model_label, fontdict=CSFONT)
    ax.legend(
        handles=[
            mlines.Line2D(
                [], [], color=_COLOR_TRAIN, marker="o", linestyle="None",
                markersize=8, label=label_train,
            ),
            mlines.Line2D(
                [], [], color=_COLOR_TEST, marker="o", linestyle="None",
                markersize=8, label=label_test,
            ),
        ],
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Parity plot of POD predicted energies vs DFT on the MCMC "
            "train/test split (eV/atom)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p, required=True)
    p.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    p.add_argument("-M", type=int, default=10, help="ACSF/POD M (bare model names only).")
    p.add_argument("-W", type=int, default=6, help="ACSF/POD W (bare model names only).")
    p.add_argument(
        "--POD-index",
        type=int,
        default=None,
        dest="pod_index",
        help="POD hyperparameter-search index (with bare POD_energy / PODD3_energy).",
    )
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--dpi", type=int, default=180)
    add_hyperparam_args(p)
    return p.parse_known_args()


def main() -> None:
    args, unknown = _parse_args()
    cli_hyperparams = collect_workflow_hyperparams(args, unknown)
    if args.pod_index is None and "POD_index" in cli_hyperparams:
        args.pod_index = int(cli_hyperparams["POD_index"])
    if cli_hyperparams:
        print(f"CLI hyperparameters: {cli_hyperparams}", flush=True)

    os.chdir(UQ_DIR)
    models = _resolve_model_names(
        args.models, args, cli_hyperparams, args.ensemble_dir,
    )
    if not models:
        raise SystemExit("No models resolved from --models.")
    print(f"Models: {models}", flush=True)

    _chdir_for_dataloader()
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir

    for model_name in models:
        if not _is_pod_family_model(model_name):
            print(
                f"  Warning: skipping non-POD model {model_name!r}.",
                file=sys.stderr,
            )
            continue

        load_name = _pod_family_load_name(model_name)
        model_label = _model_plot_label(model_name, load_name)
        print(f"\n--- Model: {model_name} ---", flush=True)
        print(
            f"  Loading rVV10 + MCMC split "
            f"(TEST_SIZE={TEST_SIZE}, seed=42) as {load_name!r} …",
            flush=True,
        )
        (
            xdata_train,
            xdata_test,
            _xdata,
            ydata_train,
            ydata_test,
            _ydata,
        ) = _load_mcmc_train_test(load_name)

        atoms_train = list(xdata_train["energy"])
        atoms_test = list(xdata_test["energy"])
        e_dft_train = _energies_per_atom(atoms_train, ydata_train["energy"])
        e_dft_test = _energies_per_atom(atoms_test, ydata_test["energy"])
        print(
            f"  n_train={len(atoms_train)}  n_test={len(atoms_test)}",
            flush=True,
        )

        calc_obj, close_calc = _build_model_calculator(
            model_name, cli_hyperparams or None,
        )
        print(f"  Calculator: {model_label}", flush=True)
        try:
            print("  Evaluating POD energies on train …", flush=True)
            e_pred_train = _energies_per_atom(
                atoms_train,
                evaluate_model_energies_on_atoms(calc_obj, atoms_train),
            )
            print("  Evaluating POD energies on test …", flush=True)
            e_pred_test = _energies_per_atom(
                atoms_test,
                evaluate_model_energies_on_atoms(calc_obj, atoms_test),
            )
        finally:
            close_calc()

        mae_train = float(np.mean(np.abs(e_pred_train - e_dft_train)))
        mae_test = float(np.mean(np.abs(e_pred_test - e_dft_test)))
        print(
            f"  MAE (eV/atom): train={mae_train:.6g}  test={mae_test:.6g}",
            flush=True,
        )

        out_path = figures_dir / f"{_safe_filename(model_name)}_parity_train_test.png"
        plot_pod_energy_parity(
            e_dft_train=e_dft_train,
            e_pred_train=e_pred_train,
            e_dft_test=e_dft_test,
            e_pred_test=e_pred_test,
            model_label=model_label,
            out_path=out_path,
            dpi=args.dpi,
        )
        print(f"  Wrote {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
