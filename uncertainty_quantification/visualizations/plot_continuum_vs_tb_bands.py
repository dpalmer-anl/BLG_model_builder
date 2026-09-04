#!/usr/bin/env python3
"""
plot_continuum_vs_tb_bands.py
==============================
Plot a comparison of the ensemble tight-binding (TB) band structure
(mean +- std, from ``bands/propagation/.../thetaXdeg/sampleNNNN.npz``,
written by ``run_uq_propagation_bands.py``) against the fitted continuum-model
band structure (mean +- std, from
``continuum_fits/propagation/.../thetaXdeg/sampleNNNN.npz``, written by
``run_uq_propagation_continuum_model.py``).  All bands stored in those files
are plotted (TB: the saved Fermi window, typically 25 below + 25 above E_F;
continuum: the full plane-wave spectrum).  When present, the continuum bands
evaluated at the Nam & Koshino paper parameters are overlaid as well.

Only samples with *both* a TB bands file and a continuum-model fit file are
included in the ensemble average (the pairing is by identical sample index,
exactly as enforced when the continuum fits were produced).

Example
-------
::

    python visualizations/plot_continuum_vs_tb_bands.py \\
        --bands-dir "../bands/propagation/POD_energy_POD_index_15_8bb97b2162397248/T0.1/ACSF_hoppings_sk_M_9_W_6_tbT0.25/theta0.83deg"

    # Explicit continuum-fit directory (default: swap "bands/propagation" for
    # "continuum_fits/propagation" in --bands-dir):
    python visualizations/plot_continuum_vs_tb_bands.py \\
        --bands-dir .../bands/propagation/.../theta0.83deg \\
        --continuum-dir .../continuum_fits/propagation/.../theta0.83deg \\
        --output twist0.83_tb_vs_continuum_bands.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np

_SAMPLE_RE = re.compile(r"sample(\d{4})\.npz$")


def _sample_index(path: str) -> int:
    m = _SAMPLE_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"cannot parse sample index from {path!r}")
    return int(m.group(1))


def _discover_samples(directory: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "sample*.npz"))):
        try:
            out[_sample_index(path)] = path
        except ValueError:
            continue
    return out


def _stack_mean_std(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def load_ensemble(
    bands_dir: str, continuum_dir: str,
) -> Tuple:
    """Return ``(k_dist, k_node, sym_labels, tb_mean, tb_std, cont_mean,
    cont_std, paper_mean, paper_std, common)``.  ``paper_*`` are ``None``
    if no fit file stored ``continuum_evals_paper``."""
    bands_samples = _discover_samples(bands_dir)
    cont_samples = _discover_samples(continuum_dir)
    common = sorted(set(bands_samples) & set(cont_samples))
    if not common:
        raise FileNotFoundError(
            f"No sample present in both {bands_dir!r} and {continuum_dir!r}."
        )

    tb_stack: List[np.ndarray] = []
    cont_stack: List[np.ndarray] = []
    paper_stack: List[np.ndarray] = []
    k_dist = k_node = sym_labels = None
    for idx in common:
        bdata = np.load(bands_samples[idx])
        cdata = np.load(cont_samples[idx])

        tb_full = np.asarray(bdata["evals"], dtype=float)
        if "continuum_evals_full" in cdata:
            cont_full = np.asarray(cdata["continuum_evals_full"], dtype=float)
        else:
            cont_full = np.asarray(cdata["continuum_evals"], dtype=float)
        if tb_full.shape[0] != cont_full.shape[0]:
            continue

        tb_stack.append(tb_full)
        cont_stack.append(cont_full)
        if "continuum_evals_paper" in cdata:
            paper_stack.append(np.asarray(cdata["continuum_evals_paper"], dtype=float))
        if k_dist is None:
            k_dist = np.asarray(bdata["k_dist"], dtype=float)
            k_node = np.asarray(bdata["k_node"], dtype=float)
            sym_labels = [str(s) for s in np.asarray(bdata["sym_labels"]).tolist()]

    if not tb_stack:
        raise FileNotFoundError(
            f"No compatible (matching k-path) sample pairs in "
            f"{bands_dir!r} / {continuum_dir!r}."
        )

    tb_mean, tb_std = _stack_mean_std(tb_stack)
    cont_mean, cont_std = _stack_mean_std(cont_stack)
    if paper_stack and all(a.shape == paper_stack[0].shape for a in paper_stack):
        paper_mean, paper_std = _stack_mean_std(paper_stack)
    else:
        paper_mean = paper_std = None
    return (
        k_dist, k_node, sym_labels, tb_mean, tb_std, cont_mean, cont_std,
        paper_mean, paper_std, common,
    )


def plot_comparison(
    output_path: str,
    k_dist: np.ndarray,
    k_node: np.ndarray,
    sym_labels: List[str],
    tb_mean: np.ndarray,
    tb_std: np.ndarray,
    cont_mean: np.ndarray,
    cont_std: np.ndarray,
    n_samples: int,
    *,
    paper_mean: Optional[np.ndarray] = None,
    paper_std: Optional[np.ndarray] = None,
    title: str = "",
    dpi: int = 150,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    def _draw(mean, std, color, ls, label_mean, label_std):
        for b in range(mean.shape[1]):
            m = mean[:, b] * 1e3
            s = std[:, b] * 1e3
            ax.plot(k_dist, m, color=color, lw=1.0, ls=ls,
                    label=label_mean if b == 0 else None)
            ax.fill_between(k_dist, m - s, m + s, color=color, alpha=0.18, lw=0,
                             label=label_std if b == 0 else None)

    _draw(tb_mean, tb_std, "tab:blue", "-", "tight-binding (mean)", "tight-binding (std)")
    if paper_mean is not None and paper_std is not None:
        _draw(paper_mean, paper_std, "tab:green", ":",
              "continuum paper params (mean)", "continuum paper params (std)")
    _draw(cont_mean, cont_std, "tab:red", "--",
          "continuum fit (mean)", "continuum fit (std)")

    ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    for kn in k_node:
        ax.axvline(kn, color="gray", lw=0.5, alpha=0.5)
    ax.set_xticks(k_node)
    ax.set_xticklabels(sym_labels)
    ax.set_ylabel("E (meV)")
    ax.set_ylim(-200.0, 200.0)
    ax.set_title(f"{title}  (n={n_samples} ensemble samples)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {output_path}", flush=True)


def _default_continuum_dir(bands_dir: str) -> str:
    norm = bands_dir.replace("\\", "/").rstrip("/")
    if "bands/propagation" in norm:
        return norm.replace("bands/propagation", "continuum_fits/propagation")
    return os.path.join(os.path.dirname(bands_dir), "continuum_fits_propagation")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bands-dir", required=True,
        help="Leaf directory of TB band .npz files "
             "(bands/propagation/<model>/T<T>/<tb>_tbT<tbT>/theta<angle>deg).",
    )
    p.add_argument(
        "--continuum-dir", default=None,
        help="Leaf directory of continuum-model fit .npz files "
             "(default: --bands-dir with 'bands/propagation' -> "
             "'continuum_fits/propagation').",
    )
    p.add_argument("--output", default=None,
                    help="Output PNG path (default: <continuum-dir>/tb_vs_continuum_bands.png).")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    bands_dir = args.bands_dir
    continuum_dir = args.continuum_dir or _default_continuum_dir(bands_dir)

    (k_dist, k_node, sym_labels, tb_mean, tb_std,
     cont_mean, cont_std, paper_mean, paper_std, common) = load_ensemble(
        bands_dir, continuum_dir,
    )

    output = args.output or os.path.join(continuum_dir, "tb_vs_continuum_bands.png")
    title = os.path.basename(os.path.normpath(continuum_dir))
    plot_comparison(
        output, k_dist, k_node, sym_labels, tb_mean, tb_std, cont_mean, cont_std,
        len(common), paper_mean=paper_mean, paper_std=paper_std,
        title=title, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
