#!/usr/bin/env python3
"""Plot Allegro (ns, nt) frontier for n_params < BUDGET."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _count_allegro_params import build_model, n_trainable

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "allegro_param_budget_frontier.png"
OUT_JSON = HERE / "allegro_param_budget_frontier.json"
BUDGET = 2000
# even scalar features only; tensor features any positive int
NS_VALUES = list(range(2, 17, 2))
NT_MAX_SEARCH = 200


def max_nt_under_budget(ns: int, *, mlp_width: int | None) -> tuple[int | None, int | None]:
    """Largest nt with n_params < BUDGET; return (nt, n_params) or (None, None)."""
    last_nt = None
    last_n = None
    for nt in range(1, NT_MAX_SEARCH + 1):
        kw = dict(num_scalar_features=ns, num_tensor_features=nt)
        if mlp_width is not None:
            kw["mlp_width"] = mlp_width
            kw["readout_width"] = mlp_width
        n = n_trainable(build_model(**kw))
        if n >= BUDGET:
            return last_nt, last_n
        last_nt, last_n = nt, n
    return last_nt, last_n


def frontier(mlp_width: int | None) -> list[dict]:
    rows = []
    for ns in NS_VALUES:
        nt, n = max_nt_under_budget(ns, mlp_width=mlp_width)
        rows.append(
            {
                "num_scalar_features": ns,
                "max_num_tensor_features": nt,
                "n_params_at_max": n,
            }
        )
        print(f"  ns={ns:2d}  max_nt={nt}  n={n}", flush=True)
    return rows


def main() -> None:
    print(f"Frontier (budget={BUDGET}): mlp_width = num_scalar_features", flush=True)
    f_track = frontier(mlp_width=None)
    print(f"Frontier (budget={BUDGET}): mlp_width = readout = 4", flush=True)
    f_fixed = frontier(mlp_width=4)

    payload = {
        "budget": BUDGET,
        "mlp_tracks_ns": f_track,
        "mlp_width_fixed_4": f_fixed,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_JSON}", flush=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    def xy(rows: list[dict], *, drop_to_zero: bool = True) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for r in rows:
            nt = r["max_num_tensor_features"]
            if nt is not None:
                xs.append(r["num_scalar_features"])
                ys.append(nt)
            elif drop_to_zero and xs:
                # First infeasible ns: mark boundary at nt=0
                xs.append(r["num_scalar_features"])
                ys.append(0)
                break
        return np.asarray(xs, float), np.asarray(ys, float)

    x1, y1 = xy(f_track)
    x2, y2 = xy(f_fixed)

    if x1.size:
        ax.fill_between(
            x1,
            0,
            y1,
            color="C0",
            alpha=0.15,
            label=rf"$n_{{\mathrm{{params}}}} < {BUDGET}$ (widths $= n_s$)",
            zorder=1,
        )
        ax.plot(
            x1,
            y1,
            "o-",
            color="C0",
            lw=2.2,
            ms=7,
            zorder=3,
            label=r"frontier (MLP width $= n_s$)",
        )
    if x2.size:
        ax.plot(
            x2,
            y2,
            "s--",
            color="C3",
            lw=2.0,
            ms=7,
            zorder=3,
            label=r"frontier (MLP width $= 4$)",
        )

    # Mark current trained config
    ax.plot(4, 8, "*", color="black", ms=16, zorder=4, label="current (4, 8)")

    ax.set_xlabel(r"num_scalar_features ($n_s$)", fontsize=12)
    ax.set_ylabel(r"num_tensor_features ($n_t$)", fontsize=12)
    ax.set_title(
        rf"Allegro size frontier: $n_{{\mathrm{{params}}}} < {BUDGET}$"
        "\n"
        r"($l_{\max}=1$, 1 layer, 1 species, 8 Bessels)",
        fontsize=12,
    )
    ax.set_xlim(1.5, 16.5)
    ymax = max(
        [*(y1.tolist() if x1.size else [0]), *(y2.tolist() if x2.size else [0]), 10]
    )
    ax.set_ylim(0, ymax * 1.08 + 1)
    ax.set_xticks(NS_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
