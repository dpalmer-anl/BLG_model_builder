#!/usr/bin/env python3
"""
plot_propagation_uncertainty_comparison.py
==========================================
Compare TBLG band-structure uncertainty from three propagation modes:

1. ``bands/propagation_mean_relax`` — mean relaxed geometry, full TB ensemble
   (no relaxation uncertainty, TB model uncertainty).
2. ``bands/propagation_mean_tb`` — relaxed-structure ensemble, mean TB parameters
   (relaxation uncertainty, no TB model uncertainty).
3. ``bands/propagation`` — full structure × TB ensemble
   (relaxation uncertainty, TB model uncertainty).

For each matching (model, T, TB model, twist angle) directory, loads all
``sampleNNNN.npz`` files, computes per-band mean ± std, and writes:

* ``propagation_uncertainty_comparison.png`` — 3-panel band-structure comparison
* ``propagation_uncertainty_bandwidth_histogram.png`` — overlaid histograms of
  occupied flat-band width (2 lowest of 4 bands nearest E_F, all k-points)

Examples
--------
::

    python visualizations/plot_propagation_uncertainty_comparison.py
    python visualizations/plot_propagation_uncertainty_comparison.py \\
        --twist-angle 0.99 --ylim -0.1 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_tblg_bands import (  # noqa: E402
    CSFONT,
    DEFAULT_DPI,
    DEFAULT_YLIM,
    compute_sample_band_metrics,
    discover_npz_files,
    parse_theta_from_dir,
    stack_ensemble,
)

DEFAULT_PROPAGATION_DIR = UQ_DIR / "bands" / "propagation"
DEFAULT_MEAN_RELAX_DIR = UQ_DIR / "bands" / "propagation_mean_relax"
DEFAULT_MEAN_TB_DIR = UQ_DIR / "bands" / "propagation_mean_tb"

PANEL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("mean_relax", "(no relaxation uncertainty, TB model uncertainty)"),
    ("mean_tb", "(relaxation uncertainty, no TB model uncertainty)"),
    ("full", "(relaxation uncertainty, TB model uncertainty)"),
)
SUBPLOT_TITLE_FONT = {"fontname": CSFONT["fontname"], "size": 13}
HISTOGRAM_COLORS = ("tab:blue", "tab:orange", "tab:green")
HISTOGRAM_ALPHA = 0.45
HISTOGRAM_BINS = 20
HISTOGRAM_LEGEND_LABELS = (
    "no relaxation uncertainty,\nTB model uncertainty",
    "relaxation uncertainty,\nno TB model uncertainty",
    "relaxation uncertainty,\nTB model uncertainty",
)


def _wrap_panel_title(title: str) -> str:
    """Break long panel titles at commas so they fit above each subplot."""
    return title.replace(", ", ",\n")


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return UQ_DIR / path


def discover_theta_leaf_dirs(root: Path) -> List[Path]:
    """Return sorted ``.../theta<angle>deg`` leaf directories under *root*."""
    if not root.is_dir():
        return []
    leaves = [
        p
        for p in root.rglob("theta*deg")
        if p.is_dir() and parse_theta_from_dir(p) is not None
    ]
    return sorted(leaves, key=lambda p: (str(p.parent), parse_theta_from_dir(p) or 0.0))


def relative_theta_path(leaf_dir: Path, root: Path) -> Path:
    """Path of *leaf_dir* relative to *root* (e.g. model/T0.1/.../theta0.99deg)."""
    return leaf_dir.relative_to(root)


def plot_bands_on_ax(
    ax: plt.Axes,
    data: dict,
    *,
    title: str,
    ylim: Tuple[float, float],
    band_color: str = "steelblue",
    mean_lw: float = 0.8,
    mean_alpha: float = 0.85,
    fill_alpha: float = 0.3,
    show_xlabel: bool = True,
) -> None:
    """Draw ensemble mean ± std band structure on *ax*."""
    evals_stack = data["evals_stack"]
    k_dist = data["k_dist"]
    k_node = data["k_node"]
    sym_labels = data["sym_labels"]

    n_samples, _n_kpts, n_bands = evals_stack.shape
    mean_bands = np.mean(evals_stack, axis=0)
    std_bands = np.std(evals_stack, axis=0, ddof=1 if n_samples > 1 else 0)

    emin, emax = ylim
    in_window = np.any(
        (mean_bands - std_bands < emax) & (mean_bands + std_bands > emin),
        axis=0,
    )

    for b in range(n_bands):
        if not in_window[b]:
            continue
        mu = mean_bands[:, b]
        sig = std_bands[:, b]
        ax.plot(k_dist, mu, color=band_color, linewidth=mean_lw, alpha=mean_alpha, zorder=2)
        ax.fill_between(
            k_dist,
            mu - sig,
            mu + sig,
            color=band_color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=1,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9, zorder=3)
    for xv in k_node:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.6, zorder=3)

    ax.set_xlim(float(k_dist[0]), float(k_dist[-1]))
    ax.set_ylim(emin, emax)
    ax.set_xticks(k_node)
    ax.set_xticklabels(sym_labels, fontdict=CSFONT)
    if show_xlabel:
        ax.set_xlabel("k-path", fontdict=CSFONT)
    ax.set_ylabel("Energy (eV)", fontdict=CSFONT)
    ax.set_title(_wrap_panel_title(title), fontdict=SUBPLOT_TITLE_FONT, pad=10)


def occupied_bandwidth_per_sample(data: dict, theta_deg: float) -> np.ndarray:
    """Return occupied flat-band width (eV) for each ensemble sample."""
    evals_stack = data["evals_stack"]
    kvec = data.get("kvec")
    k_dist = data.get("k_dist")
    k_node = data.get("k_node")

    widths: List[float] = []
    for s in range(evals_stack.shape[0]):
        metrics = compute_sample_band_metrics(
            evals_stack[s],
            kvec=kvec,
            k_dist=k_dist,
            k_node=k_node,
            theta_deg=theta_deg,
        )
        width = metrics["occupied_bandwidth"]
        if np.isfinite(width):
            widths.append(float(width))
    return np.asarray(widths, dtype=float)


def plot_bandwidth_histogram(
    datasets: Sequence[Tuple[str, dict]],
    *,
    suptitle: str,
    out_path: Path,
    theta_deg: float,
    dpi: int = DEFAULT_DPI,
    n_bins: int = HISTOGRAM_BINS,
    hist_alpha: float = HISTOGRAM_ALPHA,
) -> None:
    """Overlaid histograms of occupied flat-band width for the three modes."""
    widths_by_mode: List[np.ndarray] = []
    for _subtitle, data in datasets:
        widths = occupied_bandwidth_per_sample(data, theta_deg)
        if widths.size == 0:
            raise ValueError("no finite occupied flat-band widths in ensemble")
        widths_by_mode.append(widths)

    all_widths = np.concatenate(widths_by_mode)
    w_min = float(np.min(all_widths))
    w_max = float(np.max(all_widths))
    if w_max <= w_min:
        w_max = w_min + 1e-6
    bins = np.linspace(w_min, w_max, int(n_bins) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for widths, color, label in zip(
        widths_by_mode,
        HISTOGRAM_COLORS,
        HISTOGRAM_LEGEND_LABELS,
    ):
        ax.hist(
            widths,
            bins=bins,
            color=color,
            alpha=hist_alpha,
            edgecolor=color,
            linewidth=0.8,
            label=label,
            density=True,
        )

    ax.set_xlabel("Occupied flat-band width (eV)", fontdict=CSFONT)
    ax.set_ylabel("Count", fontdict=CSFONT)
    ax.legend(loc="best", fontsize=11)
    fig.suptitle(
        suptitle,
        fontname=CSFONT["fontname"],
        fontsize=CSFONT["size"],
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_ensemble_from_dir(leaf_dir: Path) -> Optional[dict]:
    """Load and stack all ``.npz`` samples in a theta leaf directory."""
    npz_files = discover_npz_files(leaf_dir)
    if not npz_files:
        return None
    return stack_ensemble(npz_files)


def plot_uncertainty_comparison(
    datasets: Sequence[Tuple[str, dict]],
    *,
    suptitle: str,
    out_path: Path,
    ylim: Tuple[float, float] = DEFAULT_YLIM,
    dpi: int = DEFAULT_DPI,
    band_color: str = "steelblue",
    fill_alpha: float = 0.3,
) -> None:
    """Create a 1×3 subplot figure comparing the three propagation modes."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)

    for ax, (subtitle, data) in zip(axes, datasets):
        plot_bands_on_ax(
            ax,
            data,
            title=subtitle,
            ylim=ylim,
            band_color=band_color,
            fill_alpha=fill_alpha,
            show_xlabel=True,
        )

    fig.suptitle(
        suptitle,
        fontname=CSFONT["fontname"],
        fontsize=CSFONT["size"],
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.35, top=0.78)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--propagation-dir",
        type=Path,
        default=DEFAULT_PROPAGATION_DIR,
        help="Full structure × TB ensemble (default: bands/propagation).",
    )
    p.add_argument(
        "--mean-relax-dir",
        type=Path,
        default=DEFAULT_MEAN_RELAX_DIR,
        help="Mean structure × TB ensemble (default: bands/propagation_mean_relax).",
    )
    p.add_argument(
        "--mean-tb-dir",
        type=Path,
        default=DEFAULT_MEAN_TB_DIR,
        help="Structure ensemble × mean TB (default: bands/propagation_mean_tb).",
    )
    p.add_argument(
        "--twist-angle",
        type=float,
        default=None,
        help="Only plot this twist angle (degrees); default: all available.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root; figures mirror the propagation tree beneath it.",
    )
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=list(DEFAULT_YLIM),
        help=f"Energy window in eV (default: {DEFAULT_YLIM[0]} {DEFAULT_YLIM[1]}).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Figure DPI (default: {DEFAULT_DPI}).",
    )
    p.add_argument(
        "--band-color",
        default="steelblue",
        help="Colour for mean bands and uncertainty fill.",
    )
    p.add_argument(
        "--fill-alpha",
        type=float,
        default=0.3,
        help="Opacity of ±std fill regions.",
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=HISTOGRAM_BINS,
        help=f"Number of histogram bins (default: {HISTOGRAM_BINS}).",
    )
    p.add_argument(
        "--hist-alpha",
        type=float,
        default=HISTOGRAM_ALPHA,
        help=f"Histogram fill opacity (default: {HISTOGRAM_ALPHA}).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(UQ_DIR)

    propagation_root = _resolve_path(args.propagation_dir)
    mean_relax_root = _resolve_path(args.mean_relax_dir)
    mean_tb_root = _resolve_path(args.mean_tb_dir)
    output_root = (
        _resolve_path(args.output_dir) if args.output_dir is not None else None
    )
    ylim = (float(args.ylim[0]), float(args.ylim[1]))

    leaves = discover_theta_leaf_dirs(propagation_root)
    if args.twist_angle is not None:
        leaves = [
            p
            for p in leaves
            if parse_theta_from_dir(p) is not None
            and abs(parse_theta_from_dir(p) - float(args.twist_angle)) < 1e-6
        ]

    if not leaves:
        raise SystemExit(
            f"No theta*deg directories found under {propagation_root}"
            + (f" for θ={args.twist_angle:g}°" if args.twist_angle is not None else "")
        )

    print(
        f"Scanning {len(leaves)} twist-angle ensemble(s) under {propagation_root}",
        flush=True,
    )

    n_written = 0
    n_skipped = 0

    for leaf in leaves:
        rel = relative_theta_path(leaf, propagation_root)
        theta = parse_theta_from_dir(leaf)
        theta_str = f"{theta:g}" if theta is not None else "unknown"

        dir_map = {
            "mean_relax": mean_relax_root / rel,
            "mean_tb": mean_tb_root / rel,
            "full": leaf,
        }

        missing = [key for key, d in dir_map.items() if not d.is_dir()]
        if missing:
            print(
                f"  skip θ={theta_str}° ({rel}): missing {', '.join(missing)}",
                file=sys.stderr,
                flush=True,
            )
            n_skipped += 1
            continue

        loaded: List[Tuple[str, dict]] = []
        load_failed = False
        for key, subtitle in PANEL_SPECS:
            data = load_ensemble_from_dir(dir_map[key])
            if data is None:
                print(
                    f"  skip θ={theta_str}° ({rel}): no samples in {dir_map[key]}",
                    file=sys.stderr,
                    flush=True,
                )
                load_failed = True
                break
            loaded.append((subtitle, data))

        if load_failed:
            n_skipped += 1
            continue

        suptitle = rf"$\theta = {theta_str}^\circ$"
        if output_root is not None:
            out_dir = output_root / rel
        else:
            out_dir = leaf
        bands_out = out_dir / "propagation_uncertainty_comparison.png"
        hist_out = out_dir / "propagation_uncertainty_bandwidth_histogram.png"

        plot_uncertainty_comparison(
            loaded,
            suptitle=suptitle,
            out_path=bands_out,
            ylim=ylim,
            dpi=args.dpi,
            band_color=args.band_color,
            fill_alpha=args.fill_alpha,
        )
        plot_bandwidth_histogram(
            loaded,
            suptitle=suptitle,
            out_path=hist_out,
            theta_deg=float(theta) if theta is not None else float("nan"),
            dpi=args.dpi,
            n_bins=args.hist_bins,
            hist_alpha=args.hist_alpha,
        )
        n_written += 1
        print(
            f"  θ={theta_str}°  "
            f"n=({loaded[0][1]['n_loaded']}, {loaded[1][1]['n_loaded']}, {loaded[2][1]['n_loaded']})  "
            f"→ {bands_out}\n"
            f"       → {hist_out}",
            flush=True,
        )

    print(f"\nDone: {n_written} comparison set(s) written, {n_skipped} skipped.", flush=True)


if __name__ == "__main__":
    main()
