#!/usr/bin/env python3
"""
Mean top-layer cross section ± uncertainty for TBLG relaxation ensembles.

Scans trajectory directories of the form
``trajectories/relaxation/<model>/T<temperature>/theta<angle>deg/`` for
``*_sample*.traj`` files produced by :mod:`run_uq_propagation_relaxation`.
``--trajectories-dir`` may be any level of that tree (or a glob such as
``.../T0.65/theta*``); matching ``theta*deg`` folders are discovered automatically.
Builds the in-plane cross section along the cell diagonal (same method as
``layer_sep_uq_plotter.get_struct_cross_sect``), and saves

``twist_angle_<θ>_<model>_cross_section.png``

``twist_angle_<θ>_<model>_toplayer_z_cross_section.png``

``twist_angle_<θ>_<model>_intralayer_displacement_cross_section.png``

and a two-panel top-layer separation field from two ensemble samples:

``twist_angle_<θ>_<model>_toplayer_sepfield.png``

(both in the same ensemble directory).

Samples are included only when the final frame has saved forces with
``np.max(atoms.get_forces()) ≤ 1e-4`` eV/Å (override with ``--fmax-max``).
Trajectories without forces are omitted with a warning.

Examples
--------
::

    python visualizations/plot_tblg_cross_section_ensemble.py
    python visualizations/plot_tblg_cross_section_ensemble.py \\
        --trajectories-dir trajectories/relaxation

    # One temperature: all twist angles (glob; quote if the shell would expand it)::
    python visualizations/plot_tblg_cross_section_ensemble.py \\
        --trajectories-dir \\
        trajectories/relaxation/POD_energy_POD_index_27_*/T0.65/theta*
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

CSFONT = {"fontname": "sans-serif", "size": 20}
LEGEND_FONTSIZE = 10
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": LEGEND_FONTSIZE,
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
DEFAULT_TRAJ_ROOT = UQ_DIR / "trajectories" / "relaxation"
DEFAULT_NPOINTS = 100
# Force-tolerance gate for ensemble membership (eV/Å).
DEFAULT_FMAX_MAX = 1e-4
# Fixed y-range for interlayer-separation cross-section plots (Å).
CROSS_SECTION_SEP_YLIM = (3.37, 3.60)

# Moiré path stacking high-symmetry points (fraction of ‖a₁ + a₂‖).
# Same fractions as tegtb_production/plot_cross_section.py.
STACKING_LABELS = ("AA", "AB", "SP", "BA", "AA")
STACKING_FRACTIONS = np.array([0.0, 0.33, 0.5, 0.66, 1.0], dtype=float)

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


def get_struct_cross_sect(
    atoms,
    npoints: int = DEFAULT_NPOINTS,
    *,
    relative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-layer height along the in-plane cell diagonal.

    Matches ``layer_sep_uq_plotter.get_struct_cross_sect``: interpolate top-layer
    *z* on a path from the origin ``(0, 0)`` to ``L1 + L2`` in the xy plane,
    where ``L1, L2`` are the first two cell vectors.  The returned x-coordinate
    is arc length ``linspace(0, ‖L1+L2‖_xy, npoints)``.

    With ``relative=True`` (default), return ``(path_arc_length, z_top − ⟨z⟩)``.
    With ``relative=False``, return ``(path_arc_length, z_top)`` (raw Cartesian z).
    """
    atoms = atoms.copy()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)

    l1 = np.asarray(cell[0, :2], dtype=float)
    l2 = np.asarray(cell[1, :2], dtype=float)
    end = l1 + l2
    mesh = np.linspace(0.0, 1.0, int(npoints))
    path_xy = mesh[:, np.newaxis] * end[np.newaxis, :]
    path_len = mesh * float(np.linalg.norm(end))

    mean_z = float(np.mean(pos[:, 2]))
    top_pos = pos[pos[:, 2] > mean_z]
    if top_pos.shape[0] < 3:
        return path_len, np.full(int(npoints), np.nan)

    interp = LinearNDInterpolator(
        list(zip(top_pos[:, 0], top_pos[:, 1])),
        top_pos[:, 2],
    )
    zpath_top = np.asarray(interp(path_xy), dtype=float)
    if relative:
        return path_len, zpath_top - mean_z
    return path_len, zpath_top


def get_intralayer_displacement_cross_sect(
    init_atoms,
    atoms,
    npoints: int = DEFAULT_NPOINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-layer in-plane displacement magnitude along the cell diagonal.

    Matches ``tegtb_production.plot_cross_section.plot_intralayer_cross_sect``:
    interpolate ``‖Δr_xy‖`` for top-layer atoms on the path from the origin to
    ``L1 + L2`` in the xy plane.

    Displacements use the ASE minimum-image convention (in-plane PBC) so atoms
    that cross a periodic boundary are not assigned a spurious jump of ~|cell|.
    Positions for interpolation are wrapped into the relaxed cell.
    """
    from ase.geometry import find_mic

    atoms = atoms.copy()
    init_atoms = init_atoms.copy()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    atoms.set_cell(cell, scale_atoms=False)
    # Keep init in the same cell metric without scaling atomic positions.
    init_atoms.set_cell(cell, scale_atoms=False)

    pos_raw = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    init_raw = np.asarray(init_atoms.get_positions(wrap=False), dtype=float)

    l1 = np.asarray(cell[0, :2], dtype=float)
    l2 = np.asarray(cell[1, :2], dtype=float)
    end = l1 + l2
    mesh = np.linspace(0.0, 1.0, int(npoints))
    path_xy = mesh[:, np.newaxis] * end[np.newaxis, :]
    path_len = mesh * float(np.linalg.norm(end))

    if pos_raw.shape != init_raw.shape:
        return path_len, np.full(int(npoints), np.nan)

    # MIC on Cartesian displacement (True PBC), not wrap-then-subtract.
    mic_disp, _mic_d = find_mic(pos_raw - init_raw, cell, pbc=[True, True, False])
    dist_all = np.linalg.norm(np.asarray(mic_disp, dtype=float)[:, :2], axis=1)

    atoms.wrap()
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mean_z = float(np.mean(pos[:, 2]))
    top_mask = pos[:, 2] > mean_z
    top_pos = pos[top_mask]
    if top_pos.shape[0] < 3:
        return path_len, np.full(int(npoints), np.nan)

    dist = dist_all[top_mask]

    interp = LinearNDInterpolator(
        list(zip(top_pos[:, 0], top_pos[:, 1])),
        dist,
    )
    disp_path = np.asarray(interp(path_xy), dtype=float)
    return path_len, disp_path


def read_relaxed_frame(traj_path: Path) -> "ase.Atoms":
    """
    Final relaxed structure from an ASE trajectory via :func:`ase.io.read`.

    Prefer a frame tagged with ``info['frame'] == 'relaxed'`` when present;
    otherwise use the last frame.  Reject empty trajectories and single-frame
    files that are not tagged as relaxed (missing relaxed endpoint).

    The returned ``Atoms`` keeps any attached calculator / stored forces so
    ``atoms.get_forces()`` works for ensemble gating.
    """
    import ase.io

    frames = ase.io.read(str(traj_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise ValueError(f"empty trajectory: {traj_path}")

    for fr in reversed(frames):
        if str(fr.info.get("frame", "")).lower() == "relaxed":
            return fr

    if len(frames) == 1:
        raise ValueError(
            f"{traj_path.name}: only the initial frame is stored "
            "(missing relaxed frame — re-run run_uq_propagation_relaxation.py)"
        )
    out = frames[-1]
    out.info = dict(out.info)
    out.info["frame"] = "relaxed"
    return out


def read_initial_and_relaxed_frames(
    traj_path: Path,
) -> Tuple["ase.Atoms", "ase.Atoms"]:
    """Initial (frame 0) and relaxed structure from an ASE trajectory."""
    import ase.io

    frames = ase.io.read(str(traj_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise ValueError(f"empty trajectory: {traj_path}")
    initial = frames[0]
    relaxed = read_relaxed_frame(traj_path)
    return initial, relaxed


def _passes_fmax_gate(
    atoms,
    traj_path: Path,
    *,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> bool:
    """True if ``np.max(atoms.get_forces()) ≤ fmax_max``; warn if forces missing."""
    try:
        forces = atoms.get_forces()
    except Exception:
        print(
            f"  warning: {traj_path.name}: no forces saved; "
            "leaving out of ensemble",
            file=sys.stderr,
            flush=True,
        )
        return False
    if forces is None:
        print(
            f"  warning: {traj_path.name}: no forces saved; "
            "leaving out of ensemble",
            file=sys.stderr,
            flush=True,
        )
        return False
    forces = np.asarray(forces, dtype=float)
    if forces.size == 0 or not np.all(np.isfinite(forces)):
        print(
            f"  warning: {traj_path.name}: no forces saved; "
            "leaving out of ensemble",
            file=sys.stderr,
            flush=True,
        )
        return False
    if float(np.max(forces)) > float(fmax_max):
        return False
    return True


def load_ensemble_relaxed_frames(
    trajectory_paths: Sequence[Path],
    *,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> List:
    """Load final frames that pass the fmax gate (only ensemble selection rule)."""
    kept: List = []
    for traj_path in trajectory_paths:
        try:
            atoms = read_relaxed_frame(traj_path)
        except Exception as exc:
            print(f"  skip {traj_path.name}: {exc}", file=sys.stderr)
            continue
        if _passes_fmax_gate(atoms, traj_path, fmax_max=fmax_max):
            kept.append(atoms)
    return kept


def load_ensemble_initial_relaxed_pairs(
    trajectory_paths: Sequence[Path],
    *,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> List[Tuple]:
    """Load ``(initial, relaxed)`` pairs that pass the fmax gate on ``relaxed``."""
    kept: List[Tuple] = []
    for traj_path in trajectory_paths:
        try:
            initial, relaxed = read_initial_and_relaxed_frames(traj_path)
        except Exception as exc:
            print(f"  skip {traj_path.name}: {exc}", file=sys.stderr)
            continue
        if _passes_fmax_gate(relaxed, traj_path, fmax_max=fmax_max):
            kept.append((initial, relaxed))
    return kept


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


def _set_moire_stacking_xticks(ax, path_length: float) -> None:
    """Place AA/AB/SP/BA/AA labels at fractions of the moiré diagonal path."""
    path_length = float(path_length)
    xtick_mesh = STACKING_FRACTIONS * path_length
    ax.set_xlim(0.0, path_length)
    ax.set_xticks(xtick_mesh)
    ax.set_xticklabels(list(STACKING_LABELS), fontdict=CSFONT)


def _sample_trajs_in(theta_dir: Path) -> List[Path]:
    return sorted(
        p for p in theta_dir.glob("*.traj")
        if _RE_SAMPLE_TRAJ.search(p.name)
    )


def _ensemble_group_from_theta_dir(theta_dir: Path) -> Optional[EnsembleGroup]:
    """Build an :class:`EnsembleGroup` from a ``theta<angle>deg`` directory."""
    theta_dir = theta_dir.resolve()
    if not theta_dir.is_dir():
        return None
    m_th = _RE_THETA_DIR.match(theta_dir.name)
    if not m_th:
        return None
    try:
        twist = float(m_th.group(1))
    except ValueError:
        return None
    trajs = _sample_trajs_in(theta_dir)
    if not trajs:
        return None

    t_dir = theta_dir.parent
    m_t = _RE_T_DIR.match(t_dir.name)
    if m_t:
        t_label = m_t.group(1)
        model_name = t_dir.parent.name
    else:
        t_label = t_dir.name
        model_name = t_dir.parent.name if t_dir.parent != t_dir else t_dir.name

    return EnsembleGroup(
        model_name=model_name,
        temperature_label=t_label,
        twist_angle=twist,
        directory=theta_dir,
        trajectory_paths=tuple(trajs),
    )


def _theta_dirs_under(root: Path) -> List[Path]:
    """Yield ``theta*deg`` dirs: ``root`` itself, its children, or nested."""
    if not root.is_dir():
        return []
    if _RE_THETA_DIR.match(root.name):
        return [root]

    found: List[Path] = []
    if _RE_T_DIR.match(root.name):
        for child in sorted(root.iterdir()):
            if child.is_dir() and _RE_THETA_DIR.match(child.name):
                found.append(child)
        return found

    for p in sorted(root.rglob("theta*deg")):
        if p.is_dir() and _RE_THETA_DIR.match(p.name):
            found.append(p)
    return found


def expand_trajectory_path_patterns(
    patterns: Sequence[str],
    *,
    base_dir: Path,
) -> List[Path]:
    """
    Resolve ``--trajectories-dir`` values, expanding shell-style globs.

    Accepts roots at any level (``relaxation/``, ``<model>/``, ``T*/``, or
    ``theta*deg/``), including patterns like ``.../T0.65/theta*``.
    """
    expanded: List[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        raw = Path(os.path.expanduser(str(pat)))
        candidate = raw if raw.is_absolute() else (base_dir / raw)
        matches = [Path(m) for m in sorted(glob.glob(str(candidate)))]
        if not matches and candidate.exists():
            matches = [candidate]
        for match in matches:
            key = str(match.resolve()) if match.exists() else str(match)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(match)
    return expanded


def discover_ensemble_groups(
    traj_roots: Union[Path, Sequence[Path]],
) -> List[EnsembleGroup]:
    """Find ``(model, T, θ)`` directories that contain sample trajectories."""
    if isinstance(traj_roots, Path):
        roots: Sequence[Path] = [traj_roots]
    else:
        roots = list(traj_roots)

    groups: List[EnsembleGroup] = []
    seen_dirs: set[str] = set()
    for root in roots:
        for theta_dir in _theta_dirs_under(root):
            key = str(theta_dir.resolve())
            if key in seen_dirs:
                continue
            group = _ensemble_group_from_theta_dir(theta_dir)
            if group is None:
                continue
            seen_dirs.add(key)
            groups.append(group)

    groups.sort(key=lambda g: (g.model_name, g.temperature_label, g.twist_angle))
    return groups



def plot_ensemble_cross_section(
    group: EnsembleGroup,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Plot ensemble mean ± std of ``2·(z_top − ⟨z⟩)`` along the cross-section path.

    Returns the path to the saved PNG, or ``None`` if no valid structures remain.
    """
    atoms_list = load_ensemble_relaxed_frames(
        group.trajectory_paths, fmax_max=fmax_max,
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
    _set_moire_stacking_xticks(ax, float(path_ref[-1]))
    ax.set_ylim(*CROSS_SECTION_SEP_YLIM)
    ax.set_ylabel("interlayer separation (Å)", fontdict=CSFONT)
    ax.set_title(rf"$\theta = {group.twist_angle:g}^\circ$", fontdict=CSFONT)
    ax.legend(
        loc="best",
        prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE},
    )
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


def plot_ensemble_toplayer_z_cross_section(
    group: EnsembleGroup,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Plot ensemble mean ± std of raw top-layer *z* along the cross-section path.

    Same layout as :func:`plot_ensemble_cross_section`, but the y-axis is the
    Cartesian top-layer height (Å) rather than interlayer separation.

    Returns the path to the saved PNG, or ``None`` if no valid structures remain.
    """
    atoms_list = load_ensemble_relaxed_frames(
        group.trajectory_paths, fmax_max=fmax_max,
    )
    if not atoms_list:
        print(
            f"  no structures left after filtering for toplayer-z "
            f"{group.model_name} T={group.temperature_label} θ={group.twist_angle:g}°",
            file=sys.stderr,
        )
        return None

    z_ensemble = np.full((npoints, len(atoms_list)), np.nan, dtype=float)
    path_ref: Optional[np.ndarray] = None

    for j, atoms in enumerate(atoms_list):
        path, z_top = get_struct_cross_sect(
            atoms, npoints=npoints, relative=False,
        )
        if path_ref is None:
            path_ref = path
        z_ensemble[:, j] = z_top

    if path_ref is None:
        return None

    z_mean = np.nanmean(z_ensemble, axis=1)
    z_std = np.nanstd(z_ensemble, axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(path_ref, z_mean, label="ensemble mean", color="C0")
    ax.fill_between(
        path_ref,
        z_mean - z_std,
        z_mean + z_std,
        label="ensemble std",
        alpha=0.3,
        color="C0",
    )
    _set_moire_stacking_xticks(ax, float(path_ref[-1]))
    ax.set_ylabel("top-layer z (Å)", fontdict=CSFONT)
    ax.set_title(rf"$\theta = {group.twist_angle:g}^\circ$", fontdict=CSFONT)
    ax.legend(
        loc="best",
        prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE},
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_name = (
        f"twist_angle_{group.twist_angle:g}_"
        f"{group.model_name}_toplayer_z_cross_section.png"
    )
    out_path = group.directory / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ensemble_intralayer_displacement_cross_section(
    group: EnsembleGroup,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Plot ensemble mean ± std of top-layer in-plane displacement along the path.

    Returns the path to the saved PNG, or ``None`` if no valid structures remain.
    """
    kept_pairs = load_ensemble_initial_relaxed_pairs(
        group.trajectory_paths, fmax_max=fmax_max,
    )
    if not kept_pairs:
        print(
            f"  no structures left after filtering for intralayer displacement "
            f"{group.model_name} T={group.temperature_label} θ={group.twist_angle:g}°",
            file=sys.stderr,
        )
        return None

    disp_ensemble = np.full((npoints, len(kept_pairs)), np.nan, dtype=float)
    path_ref: Optional[np.ndarray] = None

    for j, (initial, relaxed) in enumerate(kept_pairs):
        path, disp = get_intralayer_displacement_cross_sect(
            initial, relaxed, npoints=npoints,
        )
        if path_ref is None:
            path_ref = path
        disp_ensemble[:, j] = disp

    if path_ref is None:
        return None

    disp_mean = np.nanmean(disp_ensemble, axis=1)
    disp_std = np.nanstd(disp_ensemble, axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(path_ref, disp_mean, label="ensemble mean", color="C0")
    ax.fill_between(
        path_ref,
        disp_mean - disp_std,
        disp_mean + disp_std,
        label="ensemble std",
        alpha=0.3,
        color="C0",
    )
    _set_moire_stacking_xticks(ax, float(path_ref[-1]))
    ax.set_ylabel("intralayer disp. mag. (Å)", fontdict=CSFONT)
    ax.set_title(rf"$\theta = {group.twist_angle:g}^\circ$", fontdict=CSFONT)
    ax.legend(
        loc="best",
        prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE},
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_name = (
        f"twist_angle_{group.twist_angle:g}_"
        f"{group.model_name}_intralayer_displacement_cross_section.png"
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
    fmax_max: float = DEFAULT_FMAX_MAX,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Scatter of top-layer ``2·(zᵢ − ⟨z⟩)`` at each atom xy for ``n_samples`` trajectories.

    One figure with ``n_samples`` panels (default 2), each with a colorbar.
    Only trajectories that pass the fmax gate are eligible.
    """
    passing_paths: List[Path] = []
    for traj_path in group.trajectory_paths:
        try:
            atoms = read_relaxed_frame(traj_path)
        except Exception as exc:
            print(f"  skip {traj_path.name}: {exc}", file=sys.stderr)
            continue
        if _passes_fmax_gate(atoms, traj_path, fmax_max=fmax_max):
            passing_paths.append(traj_path)

    chosen = _select_sample_trajectories(passing_paths, n_samples, seed)
    if not chosen:
        print(
            f"  no samples for top-layer sepfield: "
            f"{group.model_name} T={group.temperature_label} θ={group.twist_angle:g}°",
            file=sys.stderr,
        )
        return None

    panels: List[Tuple[Path, np.ndarray, np.ndarray]] = []
    for traj_path in chosen:
        atoms = read_relaxed_frame(traj_path)
        xy, sep = toplayer_separation_field(atoms)
        panels.append((traj_path, xy, sep))

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
        cb = fig.colorbar(sc, ax=ax)  # , fraction=0.046, pad=0.04
        cb.set_label(r"$2\,(z_i - \langle z \rangle)$ (Å)", fontdict=CSFONT)
        ax.set_xlabel("x (Å)", fontdict=CSFONT)
        ax.set_ylabel("y (Å)", fontdict=CSFONT)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        si = _sample_index_from_path(traj_path)
        label = f"sample {si}" if si is not None else traj_path.stem
        ax.set_title(label, fontdict=CSFONT)

    fig.suptitle(
        rf"$\theta = {group.twist_angle:g}^\circ$ — top-layer separation",
        fontsize=CSFONT["size"],
        fontname=CSFONT["fontname"],
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
        nargs="+",
        default=[str(DEFAULT_TRAJ_ROOT)],
        metavar="PATH",
        help="Trajectory root(s) or glob(s). May be the relaxation root, a "
        "model dir, a T* dir, theta*deg dirs, or a pattern such as "
        ".../T0.65/theta* (shell-expanded paths also work). "
        f"Default: {DEFAULT_TRAJ_ROOT}.",
    )
    p.add_argument("--npoints", type=int, default=DEFAULT_NPOINTS)
    p.add_argument(
        "--fmax-max",
        type=float,
        default=DEFAULT_FMAX_MAX,
            help=(
            "Only ensemble criterion: drop samples with "
            f"np.max(forces) above this (eV/Å). Default {DEFAULT_FMAX_MAX:g}. "
            "Trajectories without saved forces are omitted with a warning."
        ),
    )
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
        help="Skip the mean interlayer-separation cross-section figure.",
    )
    p.add_argument(
        "--no-toplayer-z-cross-section",
        action="store_true",
        help="Skip the mean raw top-layer z cross-section figure.",
    )
    p.add_argument(
        "--no-intralayer-displacement",
        action="store_true",
        help="Skip the intralayer displacement cross-section figure.",
    )
    args = p.parse_args()

    os.chdir(UQ_DIR)
    roots = expand_trajectory_path_patterns(args.trajectories_dir, base_dir=UQ_DIR)
    if not roots:
        p.error(
            "No paths matched --trajectories-dir "
            f"{args.trajectories_dir!r} (cwd={UQ_DIR})"
        )

    groups = discover_ensemble_groups(roots)
    if not groups:
        p.error(
            "No TBLG relaxation ensembles found under "
            + ", ".join(str(r) for r in roots)
        )

    print(
        f"Found {len(groups)} ensemble(s) from {len(roots)} path(s) "
        f"(fmax≤{args.fmax_max:g} eV/Å)",
        flush=True,
    )
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
                dpi=args.dpi,
            )
            if out is not None:
                print(f"    Wrote {out}", flush=True)

        if not args.no_toplayer_z_cross_section:
            out_z = plot_ensemble_toplayer_z_cross_section(
                group,
                npoints=args.npoints,
                fmax_max=args.fmax_max,
                dpi=args.dpi,
            )
            if out_z is not None:
                print(f"    Wrote {out_z}", flush=True)

        if not args.no_intralayer_displacement:
            out_id = plot_ensemble_intralayer_displacement_cross_section(
                group,
                npoints=args.npoints,
                fmax_max=args.fmax_max,
                dpi=args.dpi,
            )
            if out_id is not None:
                print(f"    Wrote {out_id}", flush=True)

        if not args.no_sepfield:
            out_sf = plot_toplayer_sepfield_samples(
                group,
                n_samples=args.sepfield_samples,
                seed=args.sepfield_seed,
                fmax_max=args.fmax_max,
                dpi=args.dpi,
            )
            if out_sf is not None:
                print(f"    Wrote {out_sf}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
