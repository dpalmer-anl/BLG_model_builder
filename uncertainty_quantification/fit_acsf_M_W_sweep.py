"""
Sweep ACSF linear hopping fits over radial count M and angular count W.

For each W in 1..6, fits least-squares weights for several M values and records
test-set mean absolute error (MAE). The train/test split is kept identical
across sweeps by reseeding NumPy before each load (matches DataLoader split).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from blg_model_builder.DataLoader import load_data_for_model
from blg_model_builder.tb_models import get_acsf_hoppings
from blg_model_builder.model_fit import fit_acsf_linear_hopping, get_prediction


def mean_absolute_error_lists(y_pred, y_true) -> float:
    """MAE over all pairs; ``y_*`` are lists of 1-D arrays (one per structure)."""
    total_abs = 0.0
    total_n = 0
    for p, t in zip(y_pred, y_true):
        p = np.asarray(p, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        n = min(p.size, t.size)
        if n == 0:
            continue
        total_abs += float(np.sum(np.abs(p[:n] - t[:n])))
        total_n += n
    return total_abs / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser(description="ACSF M vs test MAE for fixed W (1–6).")
    parser.add_argument("--supercells", type=int, default=1, help="Supercell arg for loader.")
    parser.add_argument("--m-min", type=int, default=3, help="Smallest M (radial Chebyshev count).")
    parser.add_argument("--m-max", type=int, default=18, help="Largest M (inclusive).")
    parser.add_argument("--w-min", type=int, default=1, help="Smallest W (default 1).")
    parser.add_argument("--w-max", type=int, default=6, help="Largest W (default 6, inclusive).")
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Tikhonov ridge for lstsq (e.g. 1e-8 if ill-conditioned).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/acsf_M_sweep_mae.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    m_values = list(range(args.m_min, args.m_max + 1))
    w_values = list(range(args.w_min, args.w_max + 1))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8.5, 5.5))
    for W in w_values:
        maes = []
        for M in m_values:
            np.random.seed(42)
            xdata_train, xdata_test, _, ydata_train, ydata_test, _ = load_data_for_model(
                "ACSF_hoppings",
                supercells=args.supercells,
                M=M,
                W=W,
            )

            w_vec, _ = fit_acsf_linear_hopping(
                xdata_train["hopping"],
                ydata_train["hopping"],
                ridge=args.ridge,
            )
            y_test_pred = get_prediction(get_acsf_hoppings, xdata_test["hopping"], w_vec)
            mae = mean_absolute_error_lists(y_test_pred, ydata_test["hopping"])
            maes.append(mae)
            print(f"W={W} M={M}  n_feat={M + M * W}  test_MAE={mae:.6g}")

        plt.plot(m_values, maes, marker="o", markersize=4, linewidth=1.5, label=f"W = {W}")

    plt.xlabel("M (number of radial Chebyshev basis functions)")
    plt.ylabel("Test mean absolute error (|t_pred − t_ref|)")
    plt.title("ACSF linear model: test MAE vs M for fixed W")
    plt.legend(title="Angular terms")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
