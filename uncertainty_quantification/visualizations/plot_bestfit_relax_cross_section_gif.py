#!/usr/bin/env python3
"""
Animate top-layer moiré cross sections from best-fit LAMMPS relaxation dumps.

Reads ``theta*deg_bestfit.dump`` files (LAMMPS ``custom`` dump with BOX BOUNDS),
subsamples every ``--stride`` dump frame, and writes a GIF of
``2·(z_top − ⟨z⟩)`` along the cell diagonal — same construction as
:func:`plot_tblg_cross_section_ensemble.get_struct_cross_sect` /
:func:`plot_ensemble_cross_section`.

Axis limits are fixed for the whole animation (and shared across all dumps
processed in one run unless ``--per-dump-ylim`` is set).

Example
-------
::

    python visualizations/plot_bestfit_relax_cross_section_gif.py \\
        --angles 1.08 1.12 --stride 100
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse ensemble cross-section helpers (same stack path + stacking ticks).
from plot_tblg_cross_section_ensemble import (  # noqa: E402
    CROSS_SECTION_SEP_YLIM,
    CSFONT,
    DEFAULT_NPOINTS,
    get_struct_cross_sect,
)

DEFAULT_GIF_YLIM = (CROSS_SECTION_SEP_YLIM[0], 3.65)

UQ_DIR = HERE.parent
DEFAULT_DUMP_DIR = (
    UQ_DIR
    / "trajectories"
    / "relaxation_best_fit"
    / "POD_energy_POD_index_15_8bb97b2162397248_rcut7_trainfrac1_strained_bilayer_graphene_rVV10"
)
DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
_RE_THETA = re.compile(r"theta(?P<th>[0-9.]+)deg_bestfit\.dump$", re.I)


def _box_to_cell(box_lines: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(cell_3x3, origin_xyz)`` from LAMMPS ``BOX BOUNDS`` lines.

    Dump files report *bound* extents for triclinic cells, not the lo/hi used to
    build the parallelepiped. Invert with the standard LAMMPS relations before
    forming ``a,b,c`` (see LAMMPS ``dump custom`` / ``read_dump`` docs)::

        xlo = xlo_bound - MIN(0, xy, xz, xy+xz)
        xhi = xhi_bound - MAX(0, xy, xz, xy+xz)
        ...
        a = (xhi-xlo, 0, 0),  b = (xy, yhi-ylo, 0),  c = (xz, yz, zhi-zlo)
    """
    vals = [list(map(float, ln.split())) for ln in box_lines]
    xlo_b, xhi_b = vals[0][0], vals[0][1]
    xy = vals[0][2] if len(vals[0]) > 2 else 0.0
    ylo_b, yhi_b = vals[1][0], vals[1][1]
    xz = vals[1][2] if len(vals[1]) > 2 else 0.0
    zlo_b, zhi_b = vals[2][0], vals[2][1]
    yz = vals[2][2] if len(vals[2]) > 2 else 0.0

    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    zlo, zhi = zlo_b, zhi_b

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=float,
    )
    origin = np.array([xlo, ylo, zlo], dtype=float)
    return cell, origin


def iter_dump_frames(
    dump_path: Path,
    *,
    stride: int = 100,
) -> Iterator[Tuple[int, int, Atoms]]:
    """
    Stream LAMMPS dump frames; yield ``(frame_index, timestep, atoms)`` for
    ``frame_index % stride == 0``.
    """
    stride = max(1, int(stride))
    frame_index = 0
    with dump_path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            ts_line = fh.readline()
            if not ts_line:
                break
            timestep = int(float(ts_line.strip()))

            line = fh.readline()  # ITEM: NUMBER OF ATOMS
            n_line = fh.readline()
            if not n_line:
                break
            natoms = int(n_line.strip())

            box_hdr = fh.readline()  # ITEM: BOX BOUNDS ...
            box_lines = [fh.readline() for _ in range(3)]
            if not all(box_lines):
                break
            cell, origin = _box_to_cell(box_lines)

            atoms_hdr = fh.readline()  # ITEM: ATOMS ...
            if not atoms_hdr:
                break
            cols = atoms_hdr.split()[2:]  # after "ITEM: ATOMS"
            try:
                ix = cols.index("x")
                iy = cols.index("y")
                iz = cols.index("z")
            except ValueError as exc:
                raise RuntimeError(
                    f"{dump_path}: dump missing x/y/z columns: {cols}"
                ) from exc
            id_idx = cols.index("id") if "id" in cols else None
            type_idx = cols.index("type") if "type" in cols else None

            keep = frame_index % stride == 0
            if keep:
                ids: List[int] = []
                types: List[int] = []
                pos = np.empty((natoms, 3), dtype=float)
                for i in range(natoms):
                    parts = fh.readline().split()
                    if not parts:
                        raise RuntimeError(
                            f"{dump_path}: truncated at frame {frame_index}"
                        )
                    if id_idx is not None:
                        ids.append(int(float(parts[id_idx])))
                    if type_idx is not None:
                        types.append(int(float(parts[type_idx])))
                    pos[i, 0] = float(parts[ix]) - origin[0]
                    pos[i, 1] = float(parts[iy]) - origin[1]
                    pos[i, 2] = float(parts[iz]) - origin[2]
                # Sort by id so atom order is stable across frames.
                if ids:
                    order = np.argsort(np.asarray(ids, dtype=int))
                    pos = pos[order]
                    if types:
                        types_arr = np.asarray(types, dtype=int)[order]
                    else:
                        types_arr = np.ones(natoms, dtype=int)
                else:
                    types_arr = (
                        np.asarray(types, dtype=int)
                        if types
                        else np.ones(natoms, dtype=int)
                    )
                atoms = Atoms(
                    symbols=["C"] * natoms,
                    positions=pos,
                    cell=cell,
                    pbc=True,
                )
                # Retain type array for diagnostics; geometry only needs C.
                atoms.set_array("type", types_arr)
                yield frame_index, timestep, atoms
            else:
                for _ in range(natoms):
                    if not fh.readline():
                        raise RuntimeError(
                            f"{dump_path}: truncated while skipping frame {frame_index}"
                        )
            frame_index += 1


def collect_cross_sections(
    dump_path: Path,
    *,
    stride: int,
    npoints: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return ``(frame_indices, path_len, sep_curves[n_frames,npoints], theta)``.

    ``sep_curves`` are ``2·(z_top − ⟨z⟩)`` (Å), matching the ensemble plot.
    """
    m = _RE_THETA.search(dump_path.name)
    theta = float(m.group("th")) if m else float("nan")

    frame_ids: List[int] = []
    curves: List[np.ndarray] = []
    path_ref: Optional[np.ndarray] = None

    t0 = time.perf_counter()
    n_seen = 0
    for frame_index, _ts, atoms in iter_dump_frames(dump_path, stride=stride):
        n_seen += 1
        path, layer_z = get_struct_cross_sect(
            atoms, npoints=npoints, relative=True,
        )
        if path_ref is None:
            path_ref = path
        curves.append(2.0 * np.asarray(layer_z, dtype=float))
        frame_ids.append(int(frame_index))
        if n_seen % 10 == 0:
            print(
                f"  {dump_path.name}: kept {n_seen} frames "
                f"(last dump index {frame_index}) "
                f"[{time.perf_counter() - t0:.1f}s]",
                flush=True,
            )

    if path_ref is None or not curves:
        raise RuntimeError(f"no frames read from {dump_path}")

    print(
        f"  {dump_path.name}: done — {len(curves)} frames "
        f"(stride={stride}) in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return (
        np.asarray(frame_ids, dtype=int),
        np.asarray(path_ref, dtype=float),
        np.vstack(curves),
        theta,
    )


def write_cross_section_gif(
    *,
    frame_ids: np.ndarray,
    path: np.ndarray,
    seps: np.ndarray,
    theta: float,
    out_path: Path,
    ylim: Tuple[float, float],
    fps: float,
    dpi: int,
    tag: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    (line,) = ax.plot(path, seps[0], color="C0", lw=2.0)
    title = ax.set_title(
        rf"$\theta = {theta:g}^\circ$  dump step {int(frame_ids[0])}",
        fontdict=CSFONT,
    )
    # X = distance along (0,0) → L1+L2; fixed for the whole animation.
    ax.set_xlim(0.0, float(path[-1]))
    ax.set_xlabel(r"$|(0,0)\rightarrow L_1+L_2|$ (Å)", fontdict=CSFONT)
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_ylabel("interlayer separation (Å)", fontdict=CSFONT)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    def _update(i: int):
        line.set_ydata(seps[i])
        title.set_text(
            rf"$\theta = {theta:g}^\circ$  dump step {int(frame_ids[i])}"
        )
        return (line, title)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=1000.0 / max(fps, 1e-6),
        blit=False,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(frame_ids)} frames, tag={tag})", flush=True)
    return out_path


def discover_dumps(dump_dir: Path, angles: Optional[Sequence[float]]) -> List[Path]:
    dumps = sorted(dump_dir.glob("theta*deg_bestfit.dump"))
    if angles is None:
        return dumps
    wanted = {float(a) for a in angles}
    out: List[Path] = []
    for p in dumps:
        m = _RE_THETA.search(p.name)
        if m and float(m.group("th")) in wanted:
            out.append(p)
    missing = wanted - {
        float(_RE_THETA.search(p.name).group("th"))  # type: ignore[union-attr]
        for p in out
    }
    if missing:
        raise FileNotFoundError(
            f"no dump for angle(s) {sorted(missing)} under {dump_dir}"
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump-dir",
        type=Path,
        default=DEFAULT_DUMP_DIR,
        help="Directory containing theta*deg_bestfit.dump files",
    )
    p.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[1.08, 1.12],
        help="Twist angles (deg) to animate (default: 1.08 1.12)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Keep every N-th dump frame (default 100)",
    )
    p.add_argument("--npoints", type=int, default=DEFAULT_NPOINTS)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=list(DEFAULT_GIF_YLIM),
        metavar=("YMIN", "YMAX"),
        help=(
            "Fixed y-limits for interlayer separation (Å). "
            f"Default {DEFAULT_GIF_YLIM}."
        ),
    )
    p.add_argument(
        "--auto-ylim",
        action="store_true",
        help=(
            "Ignore --ylim and use a constant range spanning min/max of all "
            "kept frames across the selected dumps (plus 2%% padding)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for output GIFs (default: figures/)",
    )
    args = p.parse_args(argv)

    dump_dir = args.dump_dir
    if not dump_dir.is_dir():
        print(f"dump dir not found: {dump_dir}", file=sys.stderr)
        return 1

    dumps = discover_dumps(dump_dir, args.angles)
    if not dumps:
        print(f"no dumps matched under {dump_dir}", file=sys.stderr)
        return 1

    tag = dump_dir.name
    print(
        f"Scanning {len(dumps)} dump(s) under {dump_dir}\n"
        f"  stride={args.stride}  ylim={'auto' if args.auto_ylim else tuple(args.ylim)}",
        flush=True,
    )

    datasets = []
    for dump_path in dumps:
        print(f"\n=== {dump_path.name} ===", flush=True)
        datasets.append(
            collect_cross_sections(
                dump_path,
                stride=int(args.stride),
                npoints=int(args.npoints),
            )
        )

    if args.auto_ylim:
        lo = min(float(np.nanmin(seps)) for _f, _p, seps, _th in datasets)
        hi = max(float(np.nanmax(seps)) for _f, _p, seps, _th in datasets)
        pad = 0.02 * max(hi - lo, 1e-6)
        ylim = (lo - pad, hi + pad)
        print(f"\nauto-ylim → ({ylim[0]:.4f}, {ylim[1]:.4f}) Å", flush=True)
    else:
        ylim = (float(args.ylim[0]), float(args.ylim[1]))

    out_dir = args.out_dir
    written: List[Path] = []
    for frame_ids, path, seps, theta in datasets:
        out_path = (
            out_dir
            / f"{tag}_theta{theta:g}deg_toplayer_cross_section_stride{args.stride}.gif"
        )
        written.append(
            write_cross_section_gif(
                frame_ids=frame_ids,
                path=path,
                seps=seps,
                theta=theta,
                out_path=out_path,
                ylim=ylim,
                fps=float(args.fps),
                dpi=int(args.dpi),
                tag=tag,
            )
        )

    print("\nDone:", flush=True)
    for w in written:
        print(f"  {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
