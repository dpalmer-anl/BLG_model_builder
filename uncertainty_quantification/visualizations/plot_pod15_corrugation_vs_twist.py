#!/usr/bin/env python3
"""Plot corrugation (max z − min z) vs twist for POD_index 15 best-fit relaxes."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

UQ = Path(__file__).resolve().parents[1]
TAG = (
    "POD_energy_POD_index_15_8bb97b2162397248_rcut7_trainfrac1_"
    "strained_bilayer_graphene_rVV10"
)
TRAJ_DIR = UQ / "trajectories" / "relaxation_best_fit" / TAG
OUT = UQ / "figures" / "POD_energy_POD_index_15_8bb97b2162397248_corrugation.png"


def corrugation(atoms) -> float:
    z = np.asarray(atoms.get_positions(wrap=False), dtype=float)[:, 2]
    return float(np.max(z) - np.min(z))


def main() -> None:
    rows: list[tuple[float, float]] = []
    for path in sorted(TRAJ_DIR.glob("theta*deg_bestfit*.traj")):
        if "FAIL" in path.name:
            continue
        m = re.search(r"theta([0-9.]+)deg", path.name)
        if not m:
            continue
        # Prefer endpoints companion if both exist; else the .traj itself.
        if path.name.endswith("_endpoints.traj"):
            continue
        endpoints = path.with_name(path.name.replace(".traj", "_endpoints.traj"))
        use = endpoints if endpoints.is_file() else path
        atoms = read(str(use), index=-1)
        th = float(m.group(1))
        c = corrugation(atoms)
        rows.append((th, c))
        print(f"θ={th:g}°  file={use.name}  corrugation={c:.6f} Å", flush=True)

    # Also pick up angles that only have *_endpoints.traj
    for path in sorted(TRAJ_DIR.glob("theta*deg_bestfit_endpoints.traj")):
        m = re.search(r"theta([0-9.]+)deg", path.name)
        if not m:
            continue
        th = float(m.group(1))
        if any(abs(th - t) < 1e-9 for t, _ in rows):
            continue
        atoms = read(str(path), index=-1)
        c = corrugation(atoms)
        rows.append((th, c))
        print(f"θ={th:g}°  file={path.name}  corrugation={c:.6f} Å", flush=True)

    if not rows:
        raise SystemExit(f"No trajectories found under {TRAJ_DIR}")

    rows.sort(key=lambda x: x[0])
    thetas = np.array([r[0] for r in rows], dtype=float)
    corr = np.array([r[1] for r in rows], dtype=float)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(thetas, corr, "o-", color="C0")
    ax.set_xlabel(r"Initial twist angle $\theta$ (°)")
    ax.set_ylabel(r"corrugation $\max(z)-\min(z)$ (Å)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
