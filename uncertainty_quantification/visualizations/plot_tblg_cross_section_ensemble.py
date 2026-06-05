#!/usr/bin/env python3
"""
Mean top-layer cross section ± uncertainty for TBLG relaxation ensembles.

Scans ``trajectories/relaxation/<model>/T<temperature>/theta<angle>deg/`` for
``*_sample*.traj`` files produced by :mod:`run_uq_propagation_relaxation`, builds
the in-plane cross section along the cell diagonal (same method as
``layer_sep_uq_plotter.get_struct_cross_sect``), and saves

``twist_angle_<θ>_<model>_cross_section.png``

and a two-panel top-layer separation field from two ensemble samples:

``twist_angle_<θ>_<model>_toplayer_sepfield.png``

(both in the same ensemble directory).

Examples
--------
::

    python visualizations/plot_tblg_cross_section_ensemble.py
    python visualizations/plot_tblg_cross_section_ensemble.py \\
        --trajectories-dir trajectories/relaxation --sep-filter
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
DEFAULT_TRAJ_ROOT = UQ_DIR / "trajectories" / "relaxation"
DEFAULT_NPOINTS = 80

_RE_THETA_DIR = re.compile(r"^theta(.+)deg$", re.I)
_RE_T_DIR = re.compile(r"^T(.+)$", re.I)
_RE_SAMPLE_TRAJ = re.compile(r"_sample\d+\.traj$", re.I)
_RE_SAMPLE_IDX = re.compile(r"_sample(\d+)\.traj$", re.I)
DEFAULT_SEPFIELD_SAMPLES = 2


@dataclass(frozen=True)
class EnsembleGroup:
    model_name: str
    temperature_label: str
    twist_angle: float
    directory: Path
    trajectory_paths: Tuple[Path, ...]


def max_layer_separation(atoms) -> float:
    """2·max|z − ⟨z⟩| (Å), same convention as relaxation propagation."""
    z = np.asarray(atoms.get_positions(wrap=False), dtype=float)[:, 2]
    dz = np.abs(z - float(np.mean(z)))
    return 2.0 * float(np.max(dz))


def get_struct_cross_sect(
    atoms,
    npoints: int = DEFAULT_NPOINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-layer height along the in-plane cell diagonal.

    Matches ``layer_sep_uq_plotter.get_struct_cross_sect``: interpolate top-layer
    *z* on a path from the origin to ``a₁ + a₂`` in the xy plane; return
    ``(path_arc_length, z_top − ⟨z⟩)``.
    """
    atoms = atoms.copy()
    cell = np.abs(np.asarray(atoms.get_cell(), dtype=float))
    atoms.set_cell(cell)
    atoms.wrap()
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    xy_lim = cell[0, :] + cell[1, :]
    mesh = np.linspace(0.0, 1.0, int(npoints))
    path_xy = xy_lim[:2, np.newaxis] @ mesh[:, np.newaxis].T
    path_len = mesh * float(np.linalg.norm(xy_lim[:2]))

    mean_z = float(np.mean(pos[:, 2]))
    top_pos = pos[pos[:, 2] > mean_z]
    if top_pos.shape[0] < 3:
        return path_len, np.full(int(npoints), np.nan)

    interp = LinearNDInterpolator(
        list(zip(top_pos[:, 0], top_pos[:, 1])),
        top_pos[:, 2],
    )
    zpath_top = np.asarray(interp(path_xy.T), dtype=float)
    return path_len, zpath_top - mean_z


def read_relaxed_frame(
    traj_path: Path,
    *,
    min_displacement: float = 0.01,
) -> "ase.Atoms":
    """
    Final relaxed structure from an ASE trajectory.

    LAMMPS relaxations store frame 0 = initial, frame 1 = relaxed.  We never use
    frame 0 unless it is the only frame (that case is rejected).  Prefer a frame
    tagged with ``info['frame'] == 'relaxed'`` when present.
    """
    from ase import Atoms
    from ase.io.trajectory import Trajectory

    with Trajectory(str(traj_path), "r") as traj:
        n = len(traj)
        if n == 0:
            raise ValueError(f"empty trajectory: {traj_path}")

        for i in range(n - 1, -1, -1):
            fr = traj[i]
            if str(fr.info.get("frame", "")).lower() == "relaxed":
                return fr.copy()

        if n == 1:
            raise ValueError(
                f"{traj_path.name}: only the initial frame is stored "
                "(missing relaxed frame — re-run run_uq_propagation_relaxation.py)"
            )

        initial = traj[0]
        final = traj[-1]
        r0 = np.asarray(initial.get_positions(wrap=False), dtype=float)
        r1 = np.asarray(final.get_positions(wrap=False), dtype=float)
        if r0.shape != r1.shape:
            raise ValueError(f"{traj_path.name}: frame shape mismatch")
        rmsd = float(np.sqrt(np.mean((r1 - r0) ** 2)))
        if rmsd < float(min_displacement):
            raise ValueError(
                f"{traj_path.name}: last frame RMSD from initial = {rmsd:.4e} Å "
                f"(< {min_displacement:g} Å); not treating as relaxed"
            )
        out: Atoms = final.copy()
        out.info = dict(out.info)
        out.info["frame"] = "relaxed"
        return out


def _sample_index_from_path(traj_path: Path) -> Optional[int]:
    m = _RE_SAMPLE_IDX.search(traj_path.name)
    return int(m.group(1)) if m else None


def toplayer_separation_field(atoms) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per top-layer atom: xy position and ``2·(zᵢ − ⟨z⟩)`` (Å).

    Same definition as ``test_relaxation._plot_tblg_toplayer_interlayer_separation_field``.
    """
    atoms = atoms.copy()
    cell = np.abs(np.asarray(atoms.get_cell(), dtype=float))
    atoms.set_cell(cell)
    atoms.wrap()
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    z_mean = float(np.mean(pos[:, 2]))

    if atoms.has("mol-id"):
        mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
        u = np.unique(mol)
        if u.size == 2:
            z0 = float(np.mean(pos[mol == int(u[0]), 2]))
            z1 = float(np.mean(pos[mol == int(u[1]), 2]))
            top = int(u[0]) if z0 > z1 else int(u[1])
            mask = mol == top
        else:
            mask = pos[:, 2] > z_mean
    else:
        mask = pos[:, 2] > z_mean

    xy = pos[mask, :2]
    sep = 2.0 * (pos[mask, 2] - z_mean)
    return xy, sep


def _select_sample_trajectories(
    paths: Sequence[Path],
    n_samples: int,
    seed: int,
) -> List[Path]:
    paths = list(paths)
    if len(paths) <= n_samples:
        return paths
    rng = np.random.default_rng(seed)
    pick = sorted(int(i) for i in rng.choice(len(paths), size=n_samples, replace=False))
    return [paths[i] for i in pick]


def discover_ensemble_groups(traj_root: Path) -> List[EnsembleGroup]:
    """Find ``(model, T, θ)`` directories that contain sample trajectories."""
    if not traj_root.is_dir():
        return []

    groups: List[EnsembleGroup] = []
    for model_dir in sorted(traj_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for t_dir in sorted(model_dir.iterdir()):
            if not t_dir.is_dir():
                continue
            m_t = _RE_T_DIR.match(t_dir.name)
            if not m_t:
                continue
            t_label = m_t.group(1)
            for theta_dir in sorted(t_dir.iterdir()):
                if not theta_dir.is_dir():
                    continue
                m_th = _RE_THETA_DIR.match(theta_dir.name)
                if not m_th:
                    continue
                try:
                    twist = float(m_th.group(1))
                except ValueError:
                    continue
                trajs = sorted(
                    p for p in theta_dir.glob("*.traj")
                    if _RE_SAMPLE_TRAJ.search(p.name)
                )
                if not trajs:
                    continue
                groups.append(
                    EnsembleGroup(
                        model_name=model_name,
                        temperature_label=t_label,
                        twist_angle=twist,
                        directory=theta_dir,
                        trajectory_paths=tuple(trajs),
                    )
                )
    return groups


def _filter_structures(
    atoms_list: List,
    *,
    fmax_max: Optional[float],
    sep_min: Optional[float],
    sep_max: Optional[float],
) -> List:
    kept: List = []
    for atoms in atoms_list:
        if fmax_max is not None and atoms.calc is not None:
            try:
                f = np.asarray(atoms.get_forces(), dtype=float)
                if float(np.max(np.linalg.norm(f, axis=1))) > fmax_max:
                    continue
            except Exception:
                pass
        if sep_min is not None or sep_max is not None:
            sep = max_layer_separation(atoms)
            if sep_min is not None and sep < sep_min:
                continue
            if sep_max is not None and sep > sep_max:
                continue
        kept.append(atoms)
    return kept


def plot_ensemble_cross_section(
    group: EnsembleGroup,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: Optional[float] = None,
    sep_min: Optional[float] = None,
    sep_max: Optional[float] = None,
    min_displacement: float = 0.01,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Plot ensemble mean ± std of ``2·(z_top − ⟨z⟩)`` along the cross-section path.

    Returns the path to the saved PNG, or ``None`` if no valid structures remain.
    """
    atoms_list: List = []
    for traj_path in group.trajectory_paths:
        try:
            atoms_list.append(
                read_relaxed_frame(traj_path, min_displacement=min_displacement)
            )
        except Exception as exc:
            print(f"  skip {traj_path.name}: {exc}", file=sys.stderr)

    n_read = len(atoms_list)
    atoms_list = _filter_structures(
        atoms_list,
        fmax_max=fmax_max,
        sep_min=sep_min,
        sep_max=sep_max,
    )
    if not atoms_list:
        print(
            f"  no structures left after filtering for "
            f"{group.model_name} T={group.temperature_label} θ={group.twist_angle:g}°",
            file=sys.stderr,
        )
        return None

    layer_z_ensemble = np.full((npoints, len(atoms_list)), np.nan, dtype=float)
    path_ref: Optional[np.ndarray] = None

    for j, atoms in enumerate(atoms_list):
        path, layer_z = get_struct_cross_sect(atoms, npoints=npoints)
        if path_ref is None:
            path_ref = path
        layer_z_ensemble[:, j] = layer_z

    if path_ref is None:
        return None

    # Match layer_sep_uq_plotter: plot 2×(z_top − ⟨z⟩); mean uses factor 2, std does not.
    layer_z_mean = 2.0 * np.nanmean(layer_z_ensemble, axis=1)
    layer_z_std = np.nanstd(layer_z_ensemble, axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(path_ref, layer_z_mean, label="ensemble mean", color="C0")
    ax.fill_between(
        path_ref,
        layer_z_mean - layer_z_std,
        layer_z_mean + layer_z_std,
        label="ensemble std",
        alpha=0.3,
        color="C0",
    )
    ax.set_xlabel(r"path length along $\mathbf{a}_1 + \mathbf{a}_2$ (Å)")
    ax.set_ylabel(r"$2 \times$ top-layer height rel. $\langle z \rangle$ (Å)")
    ax.set_title(
        rf"{group.model_name}  $T={group.temperature_label}$  "
        rf"$\theta={group.twist_angle:g}^\circ$  ($n={len(atoms_list)}/{n_read}$)"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_name = (
        f"twist_angle_{group.twist_angle:g}_"
        f"{group.model_name}_cross_section.png"
    )
    out_path = group.directory / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_toplayer_sepfield_samples(
    group: EnsembleGroup,
    *,
    n_samples: int = DEFAULT_SEPFIELD_SAMPLES,
    seed: int = 0,
    fmax_max: Optional[float] = None,
    sep_min: Optional[float] = None,
    sep_max: Optional[float] = None,
    min_displacement: float = 0.01,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Scatter of top-layer ``2·(zᵢ − ⟨z⟩)`` at each atom xy for ``n_samples`` trajectories.

    One figure with ``n_samples`` panels (default 2), each with a colorbar.
    """
    chosen = _select_sample_trajectories(
        group.trajectory_paths, n_samples, seed,
    )
    if not chosen:
        return None

    panels: List[Tuple[Path, np.ndarray, np.ndarray]] = []
    for traj_path in chosen:
        try:
            atoms = read_relaxed_frame(traj_path)
        except Exception as exc:
            print(f"  skip {traj_path.name}: {exc}", file=sys.stderr)
            continue
        filtered = _filter_structures(
            [atoms],
            fmax_max=fmax_max,
            sep_min=sep_min,
            sep_max=sep_max,
        )
        if not filtered:
            continue
        xy, sep = toplayer_separation_field(filtered[0])
        panels.append((traj_path, xy, sep))

    if not panels:
        print(
            f"  no samples for top-layer sepfield: "
            f"{group.model_name} T={group.temperature_label} θ={group.twist_angle:g}°",
            file=sys.stderr,
        )
        return None

    n_pan = len(panels)
    fig, axes = plt.subplots(1, n_pan, figsize=(6.5 * n_pan, 5.6), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (traj_path, xy, sep) in zip(axes_flat, panels):
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=sep,
            s=20,
            cmap="viridis",
            linewidths=0.0,
        )
        cb = fig.colorbar(sc, ax=ax) #, fraction=0.046, pad=0.04
        cb.set_label(r"$2\,(z_i - \langle z \rangle)$ (Å)")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        si = _sample_index_from_path(traj_path)
        label = f"sample {si}" if si is not None else traj_path.stem
        ax.set_title(label)

    fig.suptitle(
        rf"{group.model_name}  $T={group.temperature_label}$  "
        rf"$\theta={group.twist_angle:g}^\circ$ — top-layer separation",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    out_name = (
        f"twist_angle_{group.twist_angle:g}_"
        f"{group.model_name}_toplayer_sepfield.png"
    )
    out_path = group.directory / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot mean TBLG cross section ± std from relaxation trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trajectories-dir",
        type=Path,
        default=DEFAULT_TRAJ_ROOT,
        help="Root directory (default: trajectories/relaxation).",
    )
    p.add_argument("--npoints", type=int, default=DEFAULT_NPOINTS)
    p.add_argument(
        "--fmax-max",
        type=float,
        default=None,
        help="Drop samples with max|F| above this (eV/Å); needs calculator on traj.",
    )
    p.add_argument(
        "--sep-filter",
        action="store_true",
        help="Keep only samples with max layer sep in [3.4, 4.0] Å "
        "(same as layer_sep_uq_plotter).",
    )
    p.add_argument("--sep-min", type=float, default=3.4)
    p.add_argument("--sep-max", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--sepfield-samples",
        type=int,
        default=DEFAULT_SEPFIELD_SAMPLES,
        help="Number of ensemble trajectories to show in the top-layer sepfield figure.",
    )
    p.add_argument(
        "--sepfield-seed",
        type=int,
        default=0,
        help="RNG seed for choosing which trajectories to plot.",
    )
    p.add_argument(
        "--no-sepfield",
        action="store_true",
        help="Skip the top-layer separation field figure.",
    )
    p.add_argument(
        "--no-cross-section",
        action="store_true",
        help="Skip the mean cross-section figure.",
    )
    p.add_argument(
        "--min-relax-displacement",
        type=float,
        default=0.01,
        help="Minimum RMSD (Å) between first and last traj frame to accept as relaxed.",
    )
    args = p.parse_args()

    os.chdir(UQ_DIR)
    traj_root = Path(args.trajectories_dir)
    if not traj_root.is_absolute():
        traj_root = UQ_DIR / traj_root

    sep_min = args.sep_min if args.sep_filter else None
    sep_max = args.sep_max if args.sep_filter else None

    groups = discover_ensemble_groups(traj_root)
    if not groups:
        p.error(f"No TBLG relaxation ensembles found under {traj_root}")

    print(f"Found {len(groups)} ensemble(s) under {traj_root}", flush=True)
    for group in groups:
        print(
            f"  {group.model_name}  T={group.temperature_label}  "
            f"θ={group.twist_angle:g}°  n_traj={len(group.trajectory_paths)}",
            flush=True,
        )
        if not args.no_cross_section:
            out = plot_ensemble_cross_section(
                group,
                npoints=args.npoints,
                fmax_max=args.fmax_max,
                sep_min=sep_min,
                sep_max=sep_max,
                min_displacement=args.min_relax_displacement,
                dpi=args.dpi,
            )
            if out is not None:
                print(f"    Wrote {out}", flush=True)

        if not args.no_sepfield:
            out_sf = plot_toplayer_sepfield_samples(
                group,
                n_samples=args.sepfield_samples,
                seed=args.sepfield_seed,
                fmax_max=args.fmax_max,
                sep_min=sep_min,
                sep_max=sep_max,
                min_displacement=args.min_relax_displacement,
                dpi=args.dpi,
            )
            if out_sf is not None:
                print(f"    Wrote {out_sf}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
