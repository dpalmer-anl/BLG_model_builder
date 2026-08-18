#!/usr/bin/env python3
"""
Mean ± std ACSF_sk hoppings vs distance on the mean POD15 relaxed TBLG
structure at θ = 1.05°.

One figure per TB model (M=10,W=5 and M=12,W=6), saved under figures/.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from blg_model_builder.tb_descriptors import get_acsf_sk_hopping_descriptors

UQ = Path(__file__).resolve().parents[1]
TRAJ_DIR = (
    UQ
    / "trajectories/relaxation"
    / "POD_energy_POD_index_15_8bb97b2162397248"
    / "T0.1/theta1.05deg"
)
FIGURES = UQ / "figures"
R_CUT = 6.0
PLOT_N_BONDS = 8000  # scatter subsample for readability
RNG = np.random.default_rng(0)

# Same temperature across models so the comparison is architecture, not β.
MODELS = [
    {
        "label": "ACSF_hoppings_sk_M_10_W_0",
        "M": 10,
        "W": 0,
        "pkl": (
            UQ
            / "ensembles/ACSF_hoppings_sk_M_10_W_0"
            / "ACSF_hoppings_sk_M_10_W_0_ensemble_T_0.25.pkl"
        ),
        "tbT": 0.25,
        "outfile": (
            "POD15_T0.1_theta1.05_mean_struct_"
            "ACSF_hoppings_sk_M_10_W_0_tbT0.25_hopping_vs_distance.png"
        ),
    },
    {
        "label": "ACSF_hoppings_sk_M_10_W_5",
        "M": 10,
        "W": 5,
        "pkl": (
            UQ
            / "ensembles/ACSF_hoppings_sk_M_10_W_5"
            / "ACSF_hoppings_sk_M_10_W_5_ensemble_T_0.25.pkl"
        ),
        "tbT": 0.25,
        "outfile": (
            "POD15_T0.1_theta1.05_mean_struct_"
            "ACSF_hoppings_sk_M_10_W_5_tbT0.25_hopping_vs_distance.png"
        ),
    },
    {
        "label": "ACSF_hoppings_sk_M_12_W_6",
        "M": 12,
        "W": 6,
        "pkl": (
            UQ
            / "ensembles/ACSF_hoppings_sk_M_12_W_6"
            / "ACSF_hoppings_sk_M_12_W_6_ensemble_T_0.25.pkl"
        ),
        "tbT": 0.25,
        "outfile": (
            "POD15_T0.1_theta1.05_mean_struct_"
            "ACSF_hoppings_sk_M_12_W_6_tbT0.25_hopping_vs_distance.png"
        ),
    },
]


def last_relaxed_frame(traj_path: Path) -> Atoms:
    frames = read(str(traj_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    for fr in reversed(frames):
        if str(fr.info.get("frame", "")).lower() == "relaxed":
            return fr
    return frames[-1]


def mean_relaxed_structure(traj_dir: Path) -> Atoms:
    """Atom-wise MIC average of relaxed frames (xy wrapped to reference)."""
    trajs = sorted(p for p in traj_dir.glob("*.traj") if "_FAIL" not in p.name)
    if not trajs:
        raise FileNotFoundError(f"No .traj files in {traj_dir}")

    print(f"Averaging {len(trajs)} relaxed structures from {traj_dir.name} …", flush=True)
    ref = last_relaxed_frame(trajs[0])
    cell = np.asarray(ref.cell, dtype=float)
    inv = np.linalg.inv(cell)
    ref_pos = np.asarray(ref.positions, dtype=float)
    acc = np.zeros_like(ref_pos)
    n = 0
    for f in trajs:
        a = last_relaxed_frame(f)
        pos = np.asarray(a.positions, dtype=float)
        if pos.shape != ref_pos.shape:
            print(f"  skip {f.name}: shape {pos.shape}", flush=True)
            continue
        d = pos - ref_pos
        frac = d @ inv
        frac[:, :2] -= np.round(frac[:, :2])
        acc += ref_pos + frac @ cell
        n += 1
    if n == 0:
        raise RuntimeError("No structures averaged.")
    out = ref.copy()
    out.set_positions(acc / n)
    print(f"  mean structure: N={len(out)}  from {n} samples", flush=True)
    return out


def hopping_mean_std(descriptors: np.ndarray, params: np.ndarray):
    """Linear ACSF: mean = D·⟨p⟩, std from parameter covariance."""
    D = np.asarray(descriptors, dtype=float)
    P = np.asarray(params, dtype=float)
    if D.ndim != 2 or P.ndim != 2 or D.shape[1] != P.shape[1]:
        raise ValueError(f"Bad shapes D={D.shape} P={P.shape}")
    mean_p = P.mean(axis=0)
    mean_h = D @ mean_p
    # Unbiased sample covariance of parameters.
    cov = np.cov(P, rowvar=False, ddof=1)
    # var_i = d_i^T Cov d_i
    var_h = np.sum((D @ cov) * D, axis=1)
    std_h = np.sqrt(np.maximum(var_h, 0.0))
    return mean_h, std_h


def bond_distances(pair_v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(pair_v, dtype=float), axis=1)


def plot_hopping_vs_distance(
    dist: np.ndarray,
    mean_h: np.ndarray,
    std_h: np.ndarray,
    *,
    outfile: Path,
    title: str | None = None,
) -> None:
    ok = np.isfinite(dist) & np.isfinite(mean_h) & np.isfinite(std_h)
    dist = dist[ok]
    mean_h = mean_h[ok]
    std_h = std_h[ok]

    if dist.size > PLOT_N_BONDS:
        ix = RNG.choice(dist.size, size=PLOT_N_BONDS, replace=False)
        dist, mean_h, std_h = dist[ix], mean_h[ix], std_h[ix]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        dist,
        mean_h,
        yerr=std_h,
        fmt="o",
        markersize=3,
        alpha=0.55,
        capsize=1.5,
        elinewidth=0.6,
        markeredgecolor="navy",
        markerfacecolor="steelblue",
        ecolor="gray",
        ls="none",
    )
    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Hopping (eV)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Wrote {outfile}", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Substring filter on model label (e.g. M_10_W_0).",
    )
    args = p.parse_args()
    models = MODELS
    if args.only:
        models = [c for c in MODELS if args.only in c["label"]]
        if not models:
            raise SystemExit(f"No MODELS match --only {args.only!r}")

    cache = FIGURES / "POD15_T0.1_theta1.05_mean_relaxed.xyz"
    if cache.is_file():
        print(f"Loading cached mean structure {cache.name}", flush=True)
        atoms = read(str(cache))
    else:
        atoms = mean_relaxed_structure(TRAJ_DIR)
        FIGURES.mkdir(parents=True, exist_ok=True)
        atoms.write(str(cache))
        print(f"  cached {cache}", flush=True)

    for cfg in models:
        print(
            f"\n{cfg['label']}  T={cfg['tbT']}  "
            f"pkl={cfg['pkl'].name}",
            flush=True,
        )
        if not cfg["pkl"].is_file():
            raise FileNotFoundError(cfg["pkl"])

        with open(cfg["pkl"], "rb") as f:
            ens = pickle.load(f)
        params = np.asarray(ens["ensemble"]["hopping"], dtype=float)
        print(f"  ensemble params: {params.shape}", flush=True)

        print(
            f"  building ACSF_sk descriptors M={cfg['M']} W={cfg['W']} …",
            flush=True,
        )
        descriptors, (_i, _j, pair_v) = get_acsf_sk_hopping_descriptors(
            atoms, M=cfg["M"], W=cfg["W"], r_cut=R_CUT,
        )
        descriptors = np.asarray(descriptors, dtype=float)
        dist = bond_distances(pair_v)
        print(
            f"  bonds={descriptors.shape[0]}  feat={descriptors.shape[1]}",
            flush=True,
        )

        mean_h, std_h = hopping_mean_std(descriptors, params)
        print(
            f"  |t| median={np.median(np.abs(mean_h)):.4f}  "
            f"σ median={np.median(std_h):.4f}  "
            f"σ q95={np.quantile(std_h, 0.95):.4f}  "
            f"σ max={std_h.max():.4f}",
            flush=True,
        )

        out = FIGURES / cfg["outfile"]
        plot_hopping_vs_distance(
            dist,
            mean_h,
            std_h,
            outfile=out,
            title=None,
        )


if __name__ == "__main__":
    main()
