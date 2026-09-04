#!/usr/bin/env python3
"""
Analyze TBLG / monolayer-hBN stacking registry and equilibrium lattice constant.

Uses best-fit ``POD_energy_POD_index_15_8bb97b2162397248`` for C–C (POD),
BN ExTeP for BN intralayer, and BNCH ILP (C–C zeroed) for B–C / N–C interlayer.

Workflow
--------
1. Build TBLG at θ = 9.43° (flatgraphene) + hBN substrate.
2. Scan hBN stacking over the graphene primitive cell → contour plot.
3. At the best stacking, scan isotropic in-plane lattice constant → curve plot.
4. Optionally write the optimum into ``tblg_hbn_geometry.py``.

Run from ``uncertainty_quantification``::

    python analyze_tblg_hbn_stacking_lattice.py --write-constants
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from blg_model_builder.potentials import (  # noqa: E402
    PODExtepILPLammpsCalculator,
    ncoeff_from_params,
)
from blg_model_builder.pod_model_selection import pod_hyperparams_for_index  # noqa: E402

from tblg_hbn_geometry import (  # noqa: E402
    DEFAULT_HBN_TWIST_DEG,
    HBN_GRAPHENE_SEP,
    TBLG_LAYER_SEP,
    build_tblg_on_hbn,
    scale_inplane_lattice,
    translate_hbn_stacking,
)

DEFAULT_POD_INDEX = 15
DEFAULT_POD_HASH = "8bb97b2162397248"
DEFAULT_N_STACK = 21
DEFAULT_LAT_SCAN = np.linspace(2.40, 2.55, 31)
CSFONT = {"fontname": "sans-serif", "size": 16}


def load_pod15_calculator(
    *,
    pod_index: int = DEFAULT_POD_INDEX,
    pod_hash: str = DEFAULT_POD_HASH,
) -> PODExtepILPLammpsCalculator:
    """Best-fit POD index-15 calculator wrapped with ExTeP + ILP."""
    hyperparams, rcut, resolved_hash = pod_hyperparams_for_index(int(pod_index))
    if str(resolved_hash) != str(pod_hash):
        print(
            f"  Warning: CSV hash {resolved_hash} != expected {pod_hash}",
            file=sys.stderr,
            flush=True,
        )
    ncoeffs = int(ncoeff_from_params(hyperparams))
    npz = (
        HERE
        / "best_fit_params"
        / f"POD_energy_{ncoeffs}_reg1e-12_{resolved_hash}_best_fit_params.npz"
    )
    if not npz.is_file():
        raise FileNotFoundError(f"Best-fit POD params not found: {npz}")
    params = np.asarray(np.load(npz)["params"], dtype=float)
    print(
        f"Loaded POD best-fit: {npz.name}  "
        f"(ncoeff={params.size}, rcut={rcut:g} Å)",
        flush=True,
    )
    return PODExtepILPLammpsCalculator(
        hyperparams, params, cutoff=float(rcut),
    )


def energy_per_atom(calc, atoms) -> float:
    e = float(calc.get_potential_energy(atoms))
    return e / len(atoms)


def scan_stacking(
    calc,
    base_atoms,
    lat_con: float,
    n_grid: int = DEFAULT_N_STACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float]:
    """Return ``sx, sy, E[ns,ns], (sx_opt, sy_opt), E_opt`` (eV/atom)."""
    sx = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    sy = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    E = np.full((len(sy), len(sx)), np.nan, dtype=float)
    print(f"Stacking scan: {n_grid}×{n_grid} = {n_grid * n_grid} evaluations …", flush=True)
    for iy, y in enumerate(sy):
        for ix, x in enumerate(sx):
            atoms = translate_hbn_stacking(base_atoms, (x, y), lat_con)
            try:
                E[iy, ix] = energy_per_atom(calc, atoms)
            except Exception as exc:
                print(f"  fail sx={x:.3f} sy={y:.3f}: {exc}", file=sys.stderr)
        print(f"  row sy={y:.3f} done", flush=True)
    iy, ix = np.unravel_index(int(np.nanargmin(E)), E.shape)
    opt = (float(sx[ix]), float(sy[iy]))
    return sx, sy, E, opt, float(E[iy, ix])


def scan_lattice(
    calc,
    atoms_at_ref,
    lat_con_ref: float,
    lat_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``lat, E, lat_opt, E_opt``."""
    E = np.full(len(lat_grid), np.nan, dtype=float)
    print(f"Lattice scan: {len(lat_grid)} points around {lat_con_ref:g} Å …", flush=True)
    for i, a in enumerate(lat_grid):
        atoms = scale_inplane_lattice(atoms_at_ref, float(a), lat_con_ref)
        try:
            E[i] = energy_per_atom(calc, atoms)
        except Exception as exc:
            print(f"  fail a={a:.4f}: {exc}", file=sys.stderr)
        print(f"  a={a:.4f} Å  E={E[i]:.6f} eV/atom", flush=True)
    i_opt = int(np.nanargmin(E))
    return lat_grid, E, float(lat_grid[i_opt]), float(E[i_opt])


def plot_stacking_contour(
    sx: np.ndarray,
    sy: np.ndarray,
    E: np.ndarray,
    opt: tuple[float, float],
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    # Extend grid for contour (periodic; endpoint duplicate of first column/row).
    sx_p = np.append(sx, sx[0] + 1.0)
    sy_p = np.append(sy, sy[0] + 1.0)
    E_p = np.pad(E, ((0, 1), (0, 1)), mode="wrap")
    SX, SY = np.meshgrid(sx_p, sy_p)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    emin = float(np.nanmin(E))
    levels = np.linspace(emin, float(np.nanpercentile(E, 95)), 24)
    cf = ax.contourf(SX, SY, E_p, levels=levels, cmap="viridis")
    ax.contour(SX, SY, E_p, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    ax.plot(opt[0], opt[1], "r*", markersize=16, label=rf"min $({opt[0]:.3f},{opt[1]:.3f})$")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Energy (eV/atom)", fontdict=CSFONT)
    ax.set_xlabel(r"$s_x$ (graphene $a_1$ fraction)", fontdict=CSFONT)
    ax.set_ylabel(r"$s_y$ (graphene $a_2$ fraction)", fontdict=CSFONT)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_lattice_curve(
    lat: np.ndarray,
    E: np.ndarray,
    lat_opt: float,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(lat, E, "o-", color="C0", lw=1.8, markersize=5)
    ax.axvline(lat_opt, color="C3", ls="--", lw=1.5, label=rf"$a={lat_opt:.4f}$ Å")
    ax.set_xlabel(r"In-plane lattice constant $a$ (Å)", fontdict=CSFONT)
    ax.set_ylabel("Energy (eV/atom)", fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def write_constants(stacking: tuple[float, float], lat_con: float) -> None:
    path = HERE / "tblg_hbn_geometry.py"
    text = path.read_text(encoding="utf-8")
    text2 = re.sub(
        r"HBN_STACKING_FRAC: Tuple\[float, float\] = \([^)]*\)",
        f"HBN_STACKING_FRAC: Tuple[float, float] = ({stacking[0]:.8f}, {stacking[1]:.8f})",
        text,
        count=1,
    )
    text2 = re.sub(
        r"HBN_TBLG_LAT_CON: float = [0-9.]+",
        f"HBN_TBLG_LAT_CON: float = {lat_con:.8f}",
        text2,
        count=1,
    )
    path.write_text(text2, encoding="utf-8")
    print(f"Updated constants in {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--twist-angle", type=float, default=DEFAULT_HBN_TWIST_DEG)
    p.add_argument("--n-stack", type=int, default=DEFAULT_N_STACK)
    p.add_argument("--lat-min", type=float, default=2.40)
    p.add_argument("--lat-max", type=float, default=2.55)
    p.add_argument("--n-lat", type=int, default=31)
    p.add_argument("--seed-lat-con", type=float, default=2.4694)
    p.add_argument("--figures-dir", type=Path, default=HERE / "figures")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--write-constants",
        action="store_true",
        help="Write optimum stacking / lattice into tblg_hbn_geometry.py",
    )
    args = p.parse_args()

    os.chdir(HERE)
    calc = load_pod15_calculator()

    # Start stacking scan at graphene lattice constant.
    seed_a = float(args.seed_lat_con)
    print(
        f"\nBuilding TBLG/hBN stack θ={args.twist_angle:g}°  a={seed_a:g} Å …",
        flush=True,
    )
    base = build_tblg_on_hbn(
        args.twist_angle,
        lat_con=seed_a,
        stacking_frac=(0.0, 0.0),
        tblg_sep=TBLG_LAYER_SEP,
        hbn_sep=HBN_GRAPHENE_SEP,
    )
    print(
        f"  natoms={len(base)}  species={sorted(set(base.get_chemical_symbols()))}  "
        f"mol-ids={sorted(set(base.get_array('mol-id').tolist()))}",
        flush=True,
    )

    sx, sy, E_stack, stacking_opt, e_stack = scan_stacking(
        calc, base, seed_a, n_grid=args.n_stack,
    )
    print(
        f"\nBest stacking (graphene primitive fractions): "
        f"sx={stacking_opt[0]:.6f}, sy={stacking_opt[1]:.6f}  "
        f"E={e_stack:.6f} eV/atom",
        flush=True,
    )

    figs = Path(args.figures_dir)
    plot_stacking_contour(
        sx, sy, E_stack, stacking_opt,
        figs / "tblg_hbn_stacking_energy_contour.png",
        dpi=args.dpi,
    )

    # Lattice scan at best stacking.
    best_stack_atoms = translate_hbn_stacking(base, stacking_opt, seed_a)
    lat_grid = np.linspace(args.lat_min, args.lat_max, args.n_lat)
    lat, E_lat, lat_opt, e_lat = scan_lattice(
        calc, best_stack_atoms, seed_a, lat_grid,
    )
    print(
        f"\nBest lattice constant: a={lat_opt:.6f} Å  E={e_lat:.6f} eV/atom",
        flush=True,
    )
    plot_lattice_curve(
        lat, E_lat, lat_opt,
        figs / "tblg_hbn_energy_vs_lattice_constant.png",
        dpi=args.dpi,
    )

    # Optional refine: re-scan stacking at lat_opt (coarse → already good).
    print("\n=== Summary ===", flush=True)
    print(f"  twist angle:          {args.twist_angle:g}°", flush=True)
    print(
        f"  best stacking (sx,sy): ({stacking_opt[0]:.6f}, {stacking_opt[1]:.6f})",
        flush=True,
    )
    print(f"  lattice constant:     {lat_opt:.6f} Å", flush=True)
    print(f"  E(stacking min):      {e_stack:.6f} eV/atom", flush=True)
    print(f"  E(lattice min):       {e_lat:.6f} eV/atom", flush=True)

    if args.write_constants:
        write_constants(stacking_opt, lat_opt)

    if hasattr(calc, "close"):
        calc.close()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
