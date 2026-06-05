#!/usr/bin/env python3
"""
plot_tblg_bands.py
==================
Plot ensemble-average TBLG band structures with uncertainty bands.

Scans ``bands/propagation/`` (or ``--bands-dir``) for ``.npz`` files written
by :mod:`run_uq_propagation_bands`.  Files sharing the same (model, T, TB
model, TB T, twist angle) prefix are grouped as one ensemble; their
eigenvalues are stacked and the **per-band mean ± std** is plotted using
``plt.fill_between(..., alpha=0.3)``.

Plotting conventions match ``tests/test_acsf_band_structure.py``:

* k-path: K → Γ → M → K (high-symmetry points from saved ``k_node`` /
  ``sym_labels`` arrays).
* Vertical dashed lines at high-symmetry points.
* Dashed red horizontal line at E = 0 (Fermi level, already subtracted).
* Energy window defaults to ``--ylim -0.5 0.5`` eV (±0.5 eV around the Fermi level).

Output
------
One PNG per ensemble group, saved alongside the ``.npz`` files:

    ``<group_prefix>_mean_bands.png``

Examples
--------
::

    python visualizations/plot_tblg_bands.py
    python visualizations/plot_tblg_bands.py --bands-dir bands/propagation \\
        --ylim -3.0 3.0 --dpi 200
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
DEFAULT_BANDS_DIR = UQ_DIR / "bands" / "propagation"
DEFAULT_YLIM = (-0.5, 0.5)   # eV — ±0.5 eV around the Fermi level (E = 0)
DEFAULT_DPI = 150

# Regex to strip the trailing _sample<NNNN>.npz from a filename to get the
# group prefix.
_RE_SAMPLE_SUFFIX = re.compile(r"_sample\d+\.npz$", re.I)


# ---------------------------------------------------------------------------
# Discovery and grouping
# ---------------------------------------------------------------------------

def discover_npz_files(bands_dir: Path) -> List[Path]:
    """Return all .npz files under *bands_dir*, sorted."""
    if not bands_dir.is_dir():
        return []
    return sorted(bands_dir.rglob("*.npz"))


def group_npz_files(npz_files: List[Path]) -> Dict[Tuple[Path, str], List[Path]]:
    """Group .npz files by (parent_directory, group_prefix).

    The group prefix is the filename with ``_sample<NNNN>.npz`` stripped.
    Files that do not match the sample-suffix pattern are ignored.
    """
    groups: Dict[Tuple[Path, str], List[Path]] = {}
    for f in npz_files:
        m = _RE_SAMPLE_SUFFIX.search(f.name)
        if m is None:
            continue
        prefix = f.name[: m.start()]
        key = (f.parent, prefix)
        groups.setdefault(key, []).append(f)
    return groups


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_npz_safe(path: Path) -> Optional[dict]:
    """Load a band-structure .npz file; return None on failure."""
    try:
        d = np.load(str(path), allow_pickle=False)
        return {k: d[k] for k in d.files}
    except Exception as exc:
        print(f"  skip {path.name}: {exc}", file=sys.stderr)
        return None


def stack_ensemble(npz_files: List[Path]) -> Optional[dict]:
    """Load and stack eigenvalues from all files in one ensemble group.

    Returns a dict with:

    ``evals_stack`` : ndarray (n_samples, n_kpts, n_bands)
    ``k_dist``      : ndarray (n_kpts,)
    ``k_node``      : ndarray (n_nodes,)
    ``sym_labels``  : list[str]
    ``twist_angle`` : float
    ``n_loaded``    : int
    ``n_failed``    : int

    Returns None if no samples could be loaded.
    """
    first_shape: Optional[Tuple[int, int]] = None
    k_dist: Optional[np.ndarray] = None
    k_node: Optional[np.ndarray] = None
    sym_labels: Optional[List[str]] = None
    twist_angle: float = float("nan")

    evals_list: List[np.ndarray] = []
    n_failed = 0

    for f in sorted(npz_files):
        d = load_npz_safe(f)
        if d is None:
            n_failed += 1
            continue

        ev = np.asarray(d["evals"], dtype=float)  # (n_kpts, n_bands)
        if ev.ndim != 2:
            print(f"  skip {f.name}: unexpected evals shape {ev.shape}", file=sys.stderr)
            n_failed += 1
            continue

        shape = ev.shape
        if first_shape is None:
            first_shape = shape
            k_dist = np.asarray(d["k_dist"], dtype=float)
            k_node = np.asarray(d["k_node"], dtype=float)
            # sym_labels saved as array of strings
            try:
                sym_labels = [str(s) for s in d["sym_labels"]]
            except KeyError:
                sym_labels = ["K", "\u0393", "M", "K"]
            try:
                twist_angle = float(d["twist_angle"])
            except (KeyError, TypeError):
                pass
        elif shape != first_shape:
            print(
                f"  skip {f.name}: shape {shape} != expected {first_shape}",
                file=sys.stderr,
            )
            n_failed += 1
            continue

        evals_list.append(ev)

    if not evals_list:
        return None

    return {
        "evals_stack": np.stack(evals_list, axis=0),  # (n_samples, n_kpts, n_bands)
        "k_dist": k_dist,
        "k_node": k_node,
        "sym_labels": sym_labels,
        "twist_angle": twist_angle,
        "n_loaded": len(evals_list),
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mean_bands(
    data: dict,
    title: str,
    out_path: Path,
    ylim: Tuple[float, float] = DEFAULT_YLIM,
    dpi: int = DEFAULT_DPI,
    band_color: str = "steelblue",
    mean_lw: float = 0.8,
    mean_alpha: float = 0.85,
    fill_alpha: float = 0.3,
    scatter: bool = False,
) -> None:
    """Plot ensemble-mean ± std band structure and save to *out_path*.

    Parameters
    ----------
    data : dict
        Output of :func:`stack_ensemble`.
    title : str
        Figure title.
    out_path : Path
        Output PNG path.
    ylim : (ymin, ymax)
        Energy window in eV.
    band_color : str
        Colour for mean bands and fill.
    mean_lw : float
        Line width for mean bands.
    mean_alpha : float
        Opacity of mean-band lines.
    fill_alpha : float
        Opacity of ±std fill regions.
    scatter : bool
        If True, use scatter instead of line for mean bands.
    """
    evals_stack = data["evals_stack"]  # (n_samples, n_kpts, n_bands)
    k_dist = data["k_dist"]            # (n_kpts,)
    k_node = data["k_node"]            # (n_nodes,)
    sym_labels = data["sym_labels"]

    n_samples, n_kpts, n_bands = evals_stack.shape

    # Per-band statistics over the sample axis
    mean_bands = np.mean(evals_stack, axis=0)   # (n_kpts, n_bands)
    std_bands = np.std(evals_stack, axis=0, ddof=1 if n_samples > 1 else 0)

    # Only render bands that have any mean value within the energy window
    # (± one std) to avoid cluttering with remote bands.
    emin, emax = ylim
    in_window = np.any(
        (mean_bands - std_bands < emax) & (mean_bands + std_bands > emin),
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    for b in range(n_bands):
        if not in_window[b]:
            continue
        mu = mean_bands[:, b]
        sig = std_bands[:, b]

        if scatter:
            ax.scatter(k_dist, mu, s=1.0, color=band_color,
                       alpha=mean_alpha, linewidths=0, zorder=2)
        else:
            ax.plot(k_dist, mu, color=band_color,
                    linewidth=mean_lw, alpha=mean_alpha, zorder=2)

        ax.fill_between(
            k_dist,
            mu - sig,
            mu + sig,
            color=band_color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=1,
        )

    # Fermi level reference line
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9,
               label="$E = 0$", zorder=3)

    # High-symmetry point markers
    for xv in k_node:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.6, zorder=3)

    ax.set_xlim(float(k_dist[0]), float(k_dist[-1]))
    ax.set_ylim(emin, emax)
    ax.set_xticks(k_node)
    ax.set_xticklabels(sym_labels, fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=band_color, linewidth=1.5,
               alpha=mean_alpha, label=f"Mean ({n_samples} samples)"),
        Patch(facecolor=band_color, alpha=fill_alpha, label="±1 std"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.0,
               label="$E_F$"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot ensemble-mean TBLG band structures with ±std fill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--bands-dir",
        type=Path,
        default=DEFAULT_BANDS_DIR,
        help=f"Directory containing band-structure .npz files "
             f"(default: {DEFAULT_BANDS_DIR}).",
    )
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=list(DEFAULT_YLIM),
        metavar=("EMIN", "EMAX"),
        help=f"Energy window in eV (default: {DEFAULT_YLIM[0]} {DEFAULT_YLIM[1]}).",
    )
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument(
        "--scatter",
        action="store_true",
        help="Use scatter markers instead of lines for mean bands.",
    )
    p.add_argument(
        "--band-color",
        default="steelblue",
        help="Colour for mean-band lines and fill (default: steelblue).",
    )
    p.add_argument(
        "--fill-alpha",
        type=float,
        default=0.3,
        help="Opacity of the ±std fill regions (default: 0.3).",
    )
    args = p.parse_args()

    os.chdir(UQ_DIR)

    bands_dir = Path(args.bands_dir)
    if not bands_dir.is_absolute():
        bands_dir = UQ_DIR / bands_dir

    npz_files = discover_npz_files(bands_dir)
    if not npz_files:
        p.error(f"No .npz files found under {bands_dir}")

    groups = group_npz_files(npz_files)
    if not groups:
        p.error(
            f"Found {len(npz_files)} .npz file(s) under {bands_dir} but none "
            "matched the expected _sample<NNNN>.npz naming pattern."
        )

    print(
        f"Found {len(groups)} ensemble group(s) "
        f"({len(npz_files)} total .npz file(s)).",
        flush=True,
    )

    ylim = tuple(args.ylim)

    for (parent_dir, prefix), members in sorted(groups.items()):
        print(f"\n  {prefix}  ({len(members)} sample(s))", flush=True)

        data = stack_ensemble(members)
        if data is None:
            print(f"    no valid samples — skipping.", file=sys.stderr)
            continue

        n_loaded = data["n_loaded"]
        n_failed = data["n_failed"]
        theta = data["twist_angle"]
        n_bands = data["evals_stack"].shape[2]

        print(
            f"    loaded {n_loaded}/{n_loaded + n_failed}  "
            f"shape=({n_loaded}, {data['evals_stack'].shape[1]}, {n_bands})  "
            f"θ={theta:g}°",
            flush=True,
        )

        theta_str = f"{theta:g}" if np.isfinite(theta) else "unknown"
        title = (
            f"{prefix}\n"
            rf"$\theta = {theta_str}^\circ$  "
            f"($n = {n_loaded}$ samples)"
        )

        out_path = parent_dir / f"{prefix}_mean_bands.png"
        plot_mean_bands(
            data,
            title=title,
            out_path=out_path,
            ylim=ylim,
            dpi=args.dpi,
            scatter=args.scatter,
            band_color=args.band_color,
            fill_alpha=args.fill_alpha,
        )
        print(f"    Wrote {out_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
