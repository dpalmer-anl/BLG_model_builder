"""
Plot ensemble mean hopping and per-bond standard deviation vs. bond distance
for an ACSF_hoppings MCMC ensemble saved by EMCEE_generate_ensemble.

Distances come from ``xdata["hopping_dist"]`` (same row order as flattened
``ypred_samples["hopping"]`` from ``evaluate_ensemble``).
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def _ensemble_predictions_to_matrix(yp):
    """(n_walkers, n_bonds) from evaluate_ensemble output (list or stacked ndarray)."""
    if isinstance(yp, list):
        rows = []
        for r in yp:
            if isinstance(r, list):
                flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in r])
            else:
                flat = np.asarray(r, dtype=float).ravel()
            rows.append(flat)
        lens = [row.size for row in rows]
        L = min(lens) if lens else 0
        if L and max(lens) != min(lens):
            rows = [row[:L] for row in rows]
        return np.vstack(rows) if rows else np.empty((0, 0), dtype=float)

    yp = np.asarray(yp)
    if yp.dtype == object:
        rows = []
        for r in yp:
            flat = np.asarray(r, dtype=float).ravel()
            rows.append(flat)
        lens = [row.size for row in rows]
        L = min(lens) if lens else 0
        if L and max(lens) != min(lens):
            rows = [row[:L] for row in rows]
        return np.vstack(rows) if rows else np.empty((0, 0), dtype=float)

    yp = np.asarray(yp, dtype=float)
    if yp.ndim == 1:
        return yp[:, np.newaxis]
    return yp


def _bond_distances_from_xdata(xdata: dict) -> np.ndarray:
    if "hopping_dist" not in xdata:
        raise KeyError('xdata must contain "hopping_dist" (list per structure, Å).')
    parts = [np.asarray(d, dtype=float).ravel() for d in xdata["hopping_dist"]]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def load_ensemble_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def default_ensemble_path(
    m: int,
    w: int,
    temperature: float,
    *,
    sk: bool = False,
) -> str:
    file_str = f"ACSF_hoppings_sk_M_{m}_W_{w}" if sk else f"ACSF_hoppings_M_{m}_W_{w}"
    return os.path.join(
        "ensembles",
        f"{file_str}",
        f"{file_str}_ensemble_T_{temperature}.pkl",
    )


def _pick_ypred_xdata(ensemble_dict: dict) -> tuple[dict, dict]:
    """Prefer test-set predictions + distances (EMCEE default eval)."""
    yp_test = ensemble_dict.get("ypred_samples_test")
    if isinstance(yp_test, dict) and yp_test:
        xd_test = ensemble_dict.get("xdata_test") or ensemble_dict.get("xdata")
        if isinstance(xd_test, dict) and xd_test:
            return yp_test, xd_test
    yp = ensemble_dict.get("ypred_samples")
    xd = ensemble_dict.get("xdata")
    if isinstance(yp, dict) and yp and isinstance(xd, dict) and xd:
        return yp, xd
    raise KeyError(
        "ensemble dict needs ypred_samples_test+xdata_test "
        "or ypred_samples+xdata with hopping predictions."
    )


def plot_acsf_hopping_ensemble_vs_distance(
    ensemble_dict: dict,
    *,
    title: str | None = None,
    outfile: str | None = None,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ypred_samples, xdata = _pick_ypred_xdata(ensemble_dict)

    ypred = ypred_samples.get("hopping")
    if ypred is None:
        raise KeyError('ypred_samples must contain key "hopping".')

    Y = _ensemble_predictions_to_matrix(ypred)
    if Y.size == 0:
        raise ValueError("Empty hopping predictions in ensemble.")

    dist = _bond_distances_from_xdata(xdata)
    n_bonds = Y.shape[1]
    if dist.size != n_bonds:
        L = int(min(dist.size, n_bonds))
        if L == 0:
            raise ValueError("No overlapping bond columns between distances and predictions.")
        dist = dist[:L]
        Y = Y[:, :L]

    mean_h = np.mean(Y, axis=0)
    std_h = np.std(Y, axis=0, ddof=0)

    ok = np.isfinite(dist) & np.isfinite(mean_h) & np.isfinite(std_h)
    dist = dist[ok]
    mean_h = mean_h[ok]
    std_h = std_h[ok]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        dist,
        mean_h,
        yerr=std_h,
        fmt="o",
        markersize=4,
        alpha=0.65,
        capsize=2,
        elinewidth=0.8,
        markeredgecolor="navy",
        markerfacecolor="steelblue",
        ecolor="gray",
        ls="none",
    )
    ax.set_xlabel(r"Distance ($\AA$)")
    ax.set_ylabel(r"Hopping (eV)")
    #ax.set_title(title or "ACSF hopping ensemble vs distance")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if outfile:
        out_dir = os.path.dirname(os.path.abspath(outfile))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(outfile, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dist, mean_h, std_h


def main() -> None:
    p = argparse.ArgumentParser(
        description="Scatter mean hopping ± std (ensemble) vs distance."
    )
    p.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Path to ensemble .pkl (from EMCEE_generate_ensemble)",
    )
    p.add_argument("--m", type=int, default=5, help="ACSF M (used with --w, --t if no --pkl)")
    p.add_argument("--w", type=int, default=1, help="ACSF W")
    p.add_argument(
        "--sk",
        action="store_true",
        help="Use ACSF_hoppings_sk_* ensemble path (with --m/--w/--t).",
    )
    p.add_argument(
        "--t",
        type=float,
        default=1.0,
        dest="temperature",
        help='Temperature_weight $\\beta$ in filename ..._ensemble_T_<beta>.pkl',
    )
    p.add_argument("--output", "-o", type=str, default=None, help="Figure path (png/pdf)")
    p.add_argument("--no-show", action="store_true", help="Save only, do not open a window")
    args = p.parse_args()

    path = args.pkl
    if path is None:
        path = default_ensemble_path(
            args.m, args.w, args.temperature, sk=args.sk,
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    d = load_ensemble_pkl(path)
    title = os.path.basename(path)
    plot_acsf_hopping_ensemble_vs_distance(
        d,
        title=title,
        outfile=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
