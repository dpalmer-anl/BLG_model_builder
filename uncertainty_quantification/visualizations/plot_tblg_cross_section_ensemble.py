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

``twist_angle_<θ>_<model>_toplayer_mean_interlayer_separation.png`` — ensemble-mean
top-layer ``2·(zᵢ − ⟨z⟩)`` in the xy plane

``twist_angle_<θ>_<model>_toplayer_mean_intralayer_displacement.png`` — ensemble-mean
top-layer in-plane MIC displacement magnitude in the xy plane

``twist_angle_<θ>_<model>_toplayer_position_uncertainty.png``

``twist_angle_<θ>_<model>_toplayer_displacement_magnitude.png``

``twist_angle_<θ>_<model>_toplayer_local_energy.png`` — ensemble-mean
top-layer local (per-atom) energy in the xy plane

``twist_angle_<θ>_<model>_toplayer_local_energy_uncertainty.png`` — ensemble
std of that local energy at each top-layer atom

``twist_angle_<θ>_<model>_unc_vs_disp.png`` — position uncertainty vs
displacement magnitude (one scatter per twist angle)

``unc_vs_disp_all_twist.png`` — overlay of those scatters, colored by θ
(written in the temperature directory)

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
import gc
import glob
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic, wrap_positions
from ase.io.trajectory import Trajectory
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
# Fixed y-range for intralayer-displacement cross-section plots (Å).
CROSS_SECTION_DISP_YLIM = (0.0, 0.17)
# Annotate domain-wall width on the displacement profile at this twist (deg).
DW_WIDTH_ANNOTATION_THETA_DEG = 0.83

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


@dataclass
class GroupPlotData:
    """Interpolated / atom-wise fields for one (model, T, θ) ensemble."""
    group: EnsembleGroup
    traj_paths: Tuple[Path, ...]
    path: np.ndarray
    z_rel_mean: np.ndarray
    z_rel_std: np.ndarray
    z_abs_mean: np.ndarray
    z_abs_std: np.ndarray
    disp_mean: np.ndarray
    disp_std: np.ndarray
    xy: np.ndarray
    unc: np.ndarray
    mag: np.ndarray
    e_loc_mean: np.ndarray
    e_loc_std: np.ndarray
    sep_mean: np.ndarray
    disp_xy_mean: np.ndarray


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
    path_len, z_abs, mean_z = _top_layer_z_on_diagonal(atoms, npoints=npoints)
    if relative:
        return path_len, z_abs - mean_z
    return path_len, z_abs


def _top_layer_z_on_diagonal(
    atoms,
    npoints: int = DEFAULT_NPOINTS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Interpolate top-layer *z* on the cell diagonal. Returns path, z_abs, ⟨z⟩."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pos = wrap_positions(
        np.asarray(atoms.get_positions(wrap=False), dtype=float),
        cell,
        pbc=True,
    )
    return _top_layer_z_on_diagonal_np(cell, pos, npoints=npoints)


def _top_layer_z_on_diagonal_np(
    cell: np.ndarray,
    pos: np.ndarray,
    npoints: int = DEFAULT_NPOINTS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    cell = np.asarray(cell, dtype=float)
    pos = wrap_positions(np.asarray(pos, dtype=float), cell, pbc=True)

    l1 = np.asarray(cell[0, :2], dtype=float)
    l2 = np.asarray(cell[1, :2], dtype=float)
    end = l1 + l2
    mesh = np.linspace(0.0, 1.0, int(npoints))
    path_xy = mesh[:, np.newaxis] * end[np.newaxis, :]
    path_len = mesh * float(np.linalg.norm(end))
    mean_z = float(np.mean(pos[:, 2]))
    top_pos = pos[pos[:, 2] > mean_z]
    if top_pos.shape[0] < 3:
        return path_len, np.full(int(npoints), np.nan), mean_z

    interp = LinearNDInterpolator(
        list(zip(top_pos[:, 0], top_pos[:, 1])),
        top_pos[:, 2],
    )
    zpath_top = np.asarray(interp(path_xy), dtype=float)
    return path_len, zpath_top, mean_z


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
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pos_raw = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    init_raw = np.asarray(init_atoms.get_positions(wrap=False), dtype=float)
    return _intralayer_displacement_cross_sect_np(
        cell, init_raw, pos_raw, npoints=npoints,
    )


def _intralayer_displacement_cross_sect_np(
    cell: np.ndarray,
    init_raw: np.ndarray,
    pos_raw: np.ndarray,
    npoints: int = DEFAULT_NPOINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    cell = np.asarray(cell, dtype=float)
    pos_raw = np.asarray(pos_raw, dtype=float)
    init_raw = np.asarray(init_raw, dtype=float)

    l1 = np.asarray(cell[0, :2], dtype=float)
    l2 = np.asarray(cell[1, :2], dtype=float)
    end = l1 + l2
    mesh = np.linspace(0.0, 1.0, int(npoints))
    path_xy = mesh[:, np.newaxis] * end[np.newaxis, :]
    path_len = mesh * float(np.linalg.norm(end))

    if pos_raw.shape != init_raw.shape:
        return path_len, np.full(int(npoints), np.nan)

    mic_disp, _mic_d = find_mic(pos_raw - init_raw, cell, pbc=[True, True, False])
    dist_all = np.linalg.norm(np.asarray(mic_disp, dtype=float)[:, :2], axis=1)

    pos = wrap_positions(pos_raw, cell, pbc=True)
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


def _traj_nframes(traj_path: Path) -> int:
    with Trajectory(str(traj_path), "r") as traj:
        return len(traj)


def _frame_forces(atoms) -> Optional[np.ndarray]:
    """Forces from ``get_forces()``, calculator results, or the ``forces`` array."""
    forces = None
    try:
        forces = atoms.get_forces()
    except Exception:
        forces = None
    if forces is None:
        calc = getattr(atoms, "calc", None)
        results = getattr(calc, "results", None) if calc is not None else None
        if isinstance(results, dict):
            forces = results.get("forces")
    if forces is None and hasattr(atoms, "arrays"):
        forces = atoms.arrays.get("forces")
    if forces is None:
        return None
    forces = np.asarray(forces, dtype=float)
    if forces.size == 0:
        return None
    return forces


def _frame_local_energies(atoms) -> Optional[np.ndarray]:
    """Per-atom energies from ``get_potential_energies()``, SPC, or ``local_energy``."""
    local = None
    try:
        local = atoms.get_potential_energies()
    except Exception:
        local = None
    if local is None:
        calc = getattr(atoms, "calc", None)
        results = getattr(calc, "results", None) if calc is not None else None
        if isinstance(results, dict):
            local = results.get("energies")
    if local is None and hasattr(atoms, "arrays"):
        local = atoms.arrays.get("local_energy")
    if local is None:
        return None
    local = np.asarray(local, dtype=float).ravel()
    if local.size == 0 or not np.all(np.isfinite(local)):
        return None
    return local


def read_relaxed_frame(traj_path: Path) -> "ase.Atoms":
    """
    Final relaxed structure from an ASE trajectory.

    Uses ``ase.io.read(..., index=-1)`` so the SinglePointCalculator (forces)
    is restored.  Intermediate optimizer steps are not loaded.
    """
    n = _traj_nframes(traj_path)
    if n == 0:
        raise ValueError(f"empty trajectory: {traj_path}")
    last = ase.io.read(str(traj_path), index=-1)
    if n == 1 and str(last.info.get("frame", "")).lower() != "relaxed":
        raise ValueError(
            f"{traj_path.name}: only the initial frame is stored "
            "(missing relaxed frame — re-run run_uq_propagation_relaxation.py)"
        )
    last.info = dict(last.info)
    last.info["frame"] = "relaxed"
    return last


def read_initial_and_relaxed_frames(
    traj_path: Path,
) -> Tuple["ase.Atoms", "ase.Atoms"]:
    """Initial (index 0) and last frame via ``ase.io.read`` — not the full path."""
    n = _traj_nframes(traj_path)
    if n == 0:
        raise ValueError(f"empty trajectory: {traj_path}")
    initial = ase.io.read(str(traj_path), index=0)
    if n == 1:
        if str(initial.info.get("frame", "")).lower() != "relaxed":
            raise ValueError(
                f"{traj_path.name}: only the initial frame is stored "
                "(missing relaxed frame — re-run run_uq_propagation_relaxation.py)"
            )
        return initial, initial
    relaxed = ase.io.read(str(traj_path), index=-1)
    relaxed.info = dict(relaxed.info)
    relaxed.info["frame"] = "relaxed"
    return initial, relaxed


def _read_traj_numpy(
    traj_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(cell, pos_init, pos_rel, forces_rel, local_e)`` without keeping Atoms."""
    initial, relaxed = read_initial_and_relaxed_frames(traj_path)
    cell = np.asarray(relaxed.get_cell(), dtype=float)
    pos_init = np.asarray(initial.get_positions(wrap=False), dtype=float)
    pos_rel = np.asarray(relaxed.get_positions(wrap=False), dtype=float)
    forces = _frame_forces(relaxed)
    local_e = _frame_local_energies(relaxed)
    del initial, relaxed
    return cell, pos_init, pos_rel, forces, local_e


def _passes_fmax_gate(
    atoms_or_forces,
    traj_path: Path,
    *,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> bool:
    """True if ``np.max(forces) ≤ fmax_max``; warn if forces missing."""
    if isinstance(atoms_or_forces, np.ndarray) or atoms_or_forces is None:
        forces = None if atoms_or_forces is None else np.asarray(atoms_or_forces, dtype=float)
        if forces is not None and forces.size == 0:
            forces = None
    else:
        forces = _frame_forces(atoms_or_forces)
    if forces is None:
        print(
            f"  warning: {traj_path.name}: no forces saved; "
            "leaving out of ensemble",
            file=sys.stderr,
            flush=True,
        )
        return False
    if not np.all(np.isfinite(forces)):
        print(
            f"  warning: {traj_path.name}: non-finite forces; "
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
    cell = np.abs(np.asarray(atoms.get_cell(), dtype=float))
    pos = wrap_positions(
        np.asarray(atoms.get_positions(wrap=False), dtype=float),
        cell,
        pbc=True,
    )
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


def _inner_two_disp_peak_centers(
    path_len: np.ndarray,
    displacement: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Path positions of the inner two displacement maxima.

    Same AA/AB/SP/BA/AA intervals as the domain-wall width in
    ``plot_tblg_structure_v_twist_angle``: four peaks, width from the
    middle pair (indices 1 and 2).  Uses the sampled maximum of ``displacement``
    in each interval (the ensemble-mean curve).
    """
    path_len = np.asarray(path_len, dtype=float).ravel()
    displacement = np.asarray(displacement, dtype=float).ravel()
    m = np.isfinite(path_len) & np.isfinite(displacement)
    path_len, displacement = path_len[m], displacement[m]
    if path_len.size < 5:
        return None
    total_len = float(path_len[-1])
    if not np.isfinite(total_len) or total_len <= 0.0:
        return None
    xs: List[float] = []
    for f_lo, f_hi in zip(STACKING_FRACTIONS[:-1], STACKING_FRACTIONS[1:]):
        x_lo, x_hi = float(f_lo) * total_len, float(f_hi) * total_len
        section = (path_len >= x_lo) & (path_len <= x_hi)
        idx = np.flatnonzero(section)
        if idx.size == 0:
            return None
        peak_idx = int(idx[np.argmax(displacement[idx])])
        xs.append(float(path_len[peak_idx]))
    if len(xs) < 4:
        return None
    x0, x1 = xs[1], xs[2]
    return (min(x0, x1), max(x0, x1))


def _annotate_dw_width_bracket(
    ax,
    path_len: np.ndarray,
    disp_mean: np.ndarray,
    disp_std: np.ndarray,
) -> None:
    """Square bracket above the inner two mean maxima, just over mean+std."""
    peaks = _inner_two_disp_peak_centers(path_len, disp_mean)
    if peaks is None:
        return
    x0, x1 = peaks
    y_hi = np.interp(
        np.array([x0, x1], dtype=float),
        np.asarray(path_len, dtype=float),
        np.asarray(disp_mean, dtype=float) + np.asarray(disp_std, dtype=float),
    )
    if not np.all(np.isfinite(y_hi)):
        return
    y_top = float(np.max(y_hi))
    y_range = float(CROSS_SECTION_DISP_YLIM[1] - CROSS_SECTION_DISP_YLIM[0])
    tick = 0.012 * y_range
    gap = 0.018 * y_range
    y_bar = y_top + gap
    color = "k"
    ax.plot([x0, x0], [y_bar - tick, y_bar], color=color, lw=1.4, zorder=5)
    ax.plot([x1, x1], [y_bar - tick, y_bar], color=color, lw=1.4, zorder=5)
    ax.plot([x0, x1], [y_bar, y_bar], color=color, lw=1.4, zorder=5)
    ax.text(
        0.5 * (x0 + x1),
        y_bar + 0.006 * y_range,
        "DW wall width",
        ha="center",
        va="bottom",
        color=color,
        fontdict=CSFONT,
        zorder=6,
    )


class _UncMagAccumulator:
    """Online RMS-vs-first-sample and mean displacement; O(n_atoms) memory."""

    def __init__(self) -> None:
        self.n = 0
        self.cell: Optional[np.ndarray] = None
        self.pos0: Optional[np.ndarray] = None
        self.mean_d: Optional[np.ndarray] = None
        self.m2_d: Optional[np.ndarray] = None
        self.mean_mag: Optional[np.ndarray] = None
        self.mean_xy: Optional[np.ndarray] = None
        self.mean_sep: Optional[np.ndarray] = None
        self.mean_disp_xy: Optional[np.ndarray] = None
        self.n_e = 0
        self.mean_e: Optional[np.ndarray] = None
        self.m2_e: Optional[np.ndarray] = None

    def update(
        self,
        pos_init: np.ndarray,
        pos_rel: np.ndarray,
        cell: np.ndarray,
        local_e: Optional[np.ndarray] = None,
    ) -> None:
        pos_rel = np.asarray(pos_rel, dtype=float)
        pos_init = np.asarray(pos_init, dtype=float)
        cell = np.asarray(cell, dtype=float)
        if self.pos0 is None:
            self.cell = cell
            self.pos0 = wrap_positions(pos_rel, cell, pbc=True)
            n_atoms = int(self.pos0.shape[0])
            self.mean_d = np.zeros((n_atoms, 3), dtype=float)
            self.m2_d = np.zeros((n_atoms, 3), dtype=float)
            self.mean_mag = np.zeros(n_atoms, dtype=float)
            self.mean_xy = np.zeros((n_atoms, 2), dtype=float)
            self.mean_sep = np.zeros(n_atoms, dtype=float)
            self.mean_disp_xy = np.zeros(n_atoms, dtype=float)
        pos0 = self.pos0
        if pos_rel.shape != pos0.shape or pos_init.shape != pos0.shape:
            return
        mic_ref, _ = find_mic(pos_rel - pos0, self.cell, pbc=[True, True, True])
        mic_ref = np.asarray(mic_ref, dtype=float)
        self.n += 1
        delta = mic_ref - self.mean_d
        self.mean_d = self.mean_d + delta / self.n
        self.m2_d = self.m2_d + delta * (mic_ref - self.mean_d)
        mic_init, _ = find_mic(pos_rel - pos_init, self.cell, pbc=[True, True, True])
        mag_j = np.linalg.norm(np.asarray(mic_init, dtype=float), axis=1)
        self.mean_mag = self.mean_mag + (mag_j - self.mean_mag) / self.n
        pos_w = wrap_positions(pos_rel, self.cell, pbc=True)
        xy_j = pos_w[:, :2]
        self.mean_xy = self.mean_xy + (xy_j - self.mean_xy) / self.n
        z_mean = float(np.mean(pos_w[:, 2]))
        sep_j = 2.0 * (pos_w[:, 2] - z_mean)
        self.mean_sep = self.mean_sep + (sep_j - self.mean_sep) / self.n
        mic_xy, _ = find_mic(pos_rel - pos_init, self.cell, pbc=[True, True, False])
        disp_xy_j = np.linalg.norm(np.asarray(mic_xy, dtype=float)[:, :2], axis=1)
        self.mean_disp_xy = self.mean_disp_xy + (disp_xy_j - self.mean_disp_xy) / self.n
        if local_e is None:
            return
        e = np.asarray(local_e, dtype=float).ravel()
        if e.shape[0] != int(pos0.shape[0]):
            return
        if self.mean_e is None:
            self.mean_e = np.zeros(e.shape[0], dtype=float)
            self.m2_e = np.zeros(e.shape[0], dtype=float)
        self.n_e += 1
        delta_e = e - self.mean_e
        self.mean_e = self.mean_e + delta_e / self.n_e
        self.m2_e = self.m2_e + delta_e * (e - self.mean_e)

    def result(self) -> Tuple[np.ndarray, ...]:
        empty = (
            np.zeros((0, 2)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
        )
        if self.pos0 is None or self.n == 0:
            return empty
        n_atoms = int(self.pos0.shape[0])
        if self.n > 1:
            std = np.sqrt(self.m2_d / (self.n - 1))
        else:
            std = np.zeros((n_atoms, 3), dtype=float)
        unc = np.sqrt(np.sum(np.square(std), axis=1))
        mean_z = float(np.mean(self.pos0[:, 2]))
        top = self.pos0[:, 2] > mean_z
        if self.mean_e is None or self.n_e == 0:
            e_mean = np.full(int(np.count_nonzero(top)), np.nan)
            e_std = np.full(int(np.count_nonzero(top)), np.nan)
        else:
            if self.n_e > 1:
                e_std_all = np.sqrt(self.m2_e / (self.n_e - 1))
            else:
                e_std_all = np.zeros(n_atoms, dtype=float)
            e_mean = self.mean_e[top]
            e_std = e_std_all[top]
        sep_mean = self.mean_sep[top] if self.mean_sep is not None else e_mean
        disp_xy_mean = self.mean_disp_xy[top] if self.mean_disp_xy is not None else e_mean
        return (
            self.mean_xy[top],
            unc[top],
            self.mean_mag[top],
            e_mean,
            e_std,
            sep_mean,
            disp_xy_mean,
        )


def toplayer_unc_and_mag(
    pairs: Sequence[Tuple],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Top-layer mean xy, RMS position uncertainty, and mean 3D displacement mag.

    Uncertainty is the RMS of MIC displacements of relaxed coordinates relative
    to the first sample.  Magnitude is the ensemble mean of
    ``‖r_relaxed − r_initial‖`` (MIC, xyz).
    """
    acc = _UncMagAccumulator()
    for initial, relaxed in pairs:
        cell = np.asarray(relaxed.get_cell(), dtype=float)
        acc.update(
            np.asarray(initial.get_positions(wrap=False), dtype=float),
            np.asarray(relaxed.get_positions(wrap=False), dtype=float),
            cell,
            local_e=_frame_local_energies(relaxed),
        )
    xy, unc, mag, _e_mean, _e_std, _sep, _dxy = acc.result()
    return xy, unc, mag


def toplayer_position_uncertainty(
    atoms_list: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-layer mean xy (Å) and RMS position uncertainty (Å)."""
    pairs = [(at, at) for at in atoms_list]
    xy, unc, _mag = toplayer_unc_and_mag(pairs)
    return xy, unc


def toplayer_displacement_magnitude(
    pairs: Sequence[Tuple],
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-layer mean xy (Å) and ensemble-mean 3D displacement magnitude (Å)."""
    xy, _unc, mag = toplayer_unc_and_mag(pairs)
    return xy, mag


def _global_vmin_vmax(values: Sequence[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    chunks = [v[np.isfinite(v)] for v in values if np.any(np.isfinite(v))]
    if not chunks:
        return None, None
    all_v = np.concatenate(chunks)
    vmin = float(np.min(all_v))
    vmax = float(np.max(all_v))
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def _plot_toplayer_xy_scalar(
    group: EnsembleGroup,
    xy: np.ndarray,
    values: np.ndarray,
    *,
    cbar_label: str,
    out_suffix: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
    dpi: int = 150,
) -> Optional[Path]:
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float)
    if xy.shape[0] == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=20,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
    )
    if show_colorbar:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label, fontdict=CSFONT)
    ax.set_xlabel("x (Å)", fontdict=CSFONT)
    ax.set_ylabel("y (Å)", fontdict=CSFONT)
    ax.set_title(rf"$\theta = {group.twist_angle:g}^\circ$", fontdict=CSFONT)
    plt.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_name = (
        f"twist_angle_{group.twist_angle:g}_"
        f"{group.model_name}_{out_suffix}.png"
    )
    out_path = group.directory / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


def _nanmean_std(values: Sequence[np.ndarray], npoints: int) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.column_stack(list(values)) if values else np.empty((npoints, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stacked, axis=1) if stacked.size else np.full(npoints, np.nan)
            std = np.nanstd(stacked, axis=1) if stacked.size else np.full(npoints, np.nan)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 12:
        return True
    return "Cannot allocate memory" in str(exc)


def build_group_plot_data(
    group: EnsembleGroup,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> Optional[GroupPlotData]:
    """Load first/last frames only; stream statistics without keeping Atoms."""
    traj_ok: List[Path] = []
    z_rel_cols: List[np.ndarray] = []
    z_abs_cols: List[np.ndarray] = []
    disp_cols: List[np.ndarray] = []
    path_ref: Optional[np.ndarray] = None
    acc = _UncMagAccumulator()

    for i, traj_path in enumerate(group.trajectory_paths):
        try:
            cell, pos_init, pos_rel, forces, local_e = _read_traj_numpy(traj_path)
        except Exception as exc:
            print(f"  skip {traj_path.name}: {exc}", file=sys.stderr)
            if _is_memory_error(exc):
                gc.collect()
            continue
        if not _passes_fmax_gate(forces, traj_path, fmax_max=fmax_max):
            continue
        path, z_abs_j, mean_z = _top_layer_z_on_diagonal_np(cell, pos_rel, npoints=npoints)
        if path_ref is None:
            path_ref = path
        z_abs_cols.append(z_abs_j)
        z_rel_cols.append(z_abs_j - mean_z)
        _p, disp_j = _intralayer_displacement_cross_sect_np(
            cell, pos_init, pos_rel, npoints=npoints,
        )
        disp_cols.append(disp_j)
        acc.update(pos_init, pos_rel, cell, local_e=local_e)
        traj_ok.append(traj_path)
        del cell, pos_init, pos_rel, forces, local_e
        if (i + 1) % 50 == 0:
            gc.collect()

    if not traj_ok or path_ref is None:
        print(
            f"  no structures left after filtering for "
            f"{group.model_name} T={group.temperature_label} θ={group.twist_angle:g}°",
            file=sys.stderr,
        )
        return None

    z_rel_mean, z_rel_std = _nanmean_std(z_rel_cols, npoints)
    z_abs_mean, z_abs_std = _nanmean_std(z_abs_cols, npoints)
    disp_mean, disp_std = _nanmean_std(disp_cols, npoints)
    xy, unc, mag, e_loc_mean, e_loc_std, sep_mean, disp_xy_mean = acc.result()
    del z_rel_cols, z_abs_cols, disp_cols, acc
    gc.collect()
    return GroupPlotData(
        group=group,
        traj_paths=tuple(traj_ok),
        path=path_ref,
        z_rel_mean=z_rel_mean,
        z_rel_std=z_rel_std,
        z_abs_mean=z_abs_mean,
        z_abs_std=z_abs_std,
        disp_mean=disp_mean,
        disp_std=disp_std,
        xy=xy,
        unc=unc,
        mag=mag,
        e_loc_mean=e_loc_mean,
        e_loc_std=e_loc_std,
        sep_mean=sep_mean,
        disp_xy_mean=disp_xy_mean,
    )



def _plot_path_mean_std(
    group: EnsembleGroup,
    path: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    ylabel: str,
    out_suffix: str,
    ylim: Optional[Tuple[float, float]] = None,
    dw_bracket: bool = False,
    dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    with np.errstate(all="ignore"):
        ax.plot(path, mean, label="ensemble mean", color="C0")
        ax.fill_between(
            path, mean - std, mean + std,
            label="ensemble std", alpha=0.3, color="C0",
        )
    _set_moire_stacking_xticks(ax, float(path[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.set_title(rf"$\theta = {group.twist_angle:g}^\circ$", fontdict=CSFONT)
    if dw_bracket:
        _annotate_dw_width_bracket(ax, path, mean, std)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = (
        group.directory
        / f"twist_angle_{group.twist_angle:g}_{group.model_name}_{out_suffix}.png"
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ensemble_cross_section(
    data: GroupPlotData,
    *,
    dpi: int = 150,
) -> Path:
    """Ensemble mean ± std of ``2·(z_top − ⟨z⟩)`` along the cross-section path."""
    return _plot_path_mean_std(
        data.group, data.path, 2.0 * data.z_rel_mean, 2.0 * data.z_rel_std,
        ylabel="interlayer separation (Å)",
        out_suffix="cross_section",
        ylim=CROSS_SECTION_SEP_YLIM,
        dpi=dpi,
    )


def plot_ensemble_toplayer_z_cross_section(
    data: GroupPlotData,
    *,
    dpi: int = 150,
) -> Path:
    """Ensemble mean ± std of raw top-layer *z* along the cross-section path."""
    return _plot_path_mean_std(
        data.group, data.path, data.z_abs_mean, data.z_abs_std,
        ylabel="top-layer z (Å)",
        out_suffix="toplayer_z_cross_section",
        dpi=dpi,
    )


def plot_ensemble_intralayer_displacement_cross_section(
    data: GroupPlotData,
    *,
    dpi: int = 150,
) -> Path:
    """Ensemble mean ± std of top-layer in-plane displacement along the path."""
    return _plot_path_mean_std(
        data.group, data.path, data.disp_mean, data.disp_std,
        ylabel="intralayer disp. mag. (Å)",
        out_suffix="intralayer_displacement_cross_section",
        ylim=CROSS_SECTION_DISP_YLIM,
        dw_bracket=abs(float(data.group.twist_angle) - DW_WIDTH_ANNOTATION_THETA_DEG) < 1e-6,
        dpi=dpi,
    )


def plot_toplayer_position_uncertainty(
    data: GroupPlotData,
    *,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
) -> Optional[Path]:
    """Scatter of top-layer RMS position uncertainty in the xy plane (Å)."""
    return _plot_toplayer_xy_scalar(
        data.group,
        data.xy,
        data.unc,
        cbar_label="position uncertainty (Å)",
        out_suffix="toplayer_position_uncertainty",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        dpi=dpi,
    )


def plot_toplayer_displacement_magnitude(
    data: GroupPlotData,
    *,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
) -> Optional[Path]:
    """Scatter of top-layer ensemble-mean 3D displacement magnitude (Å)."""
    return _plot_toplayer_xy_scalar(
        data.group,
        data.xy,
        data.mag,
        cbar_label="displacement mag. (Å)",
        out_suffix="toplayer_displacement_magnitude",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        dpi=dpi,
    )


def plot_toplayer_mean_interlayer_separation(
    data: GroupPlotData,
    *,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
) -> Optional[Path]:
    """Scatter of top-layer ensemble-mean interlayer separation in the xy plane (Å)."""
    return _plot_toplayer_xy_scalar(
        data.group,
        data.xy,
        data.sep_mean,
        cbar_label=r"$2\,(z_i - \langle z \rangle)$ (Å)",
        out_suffix="toplayer_mean_interlayer_separation",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        dpi=dpi,
    )


def plot_toplayer_mean_intralayer_displacement(
    data: GroupPlotData,
    *,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
) -> Optional[Path]:
    """Scatter of top-layer ensemble-mean in-plane displacement magnitude (Å)."""
    return _plot_toplayer_xy_scalar(
        data.group,
        data.xy,
        data.disp_xy_mean,
        cbar_label="intralayer disp. mag. (Å)",
        out_suffix="toplayer_mean_intralayer_displacement",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        dpi=dpi,
    )


def plot_toplayer_local_energy(
    data: GroupPlotData,
    *,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
) -> Optional[Path]:
    """Scatter of top-layer ensemble-mean local energy in the xy plane (eV)."""
    return _plot_toplayer_xy_scalar(
        data.group,
        data.xy,
        data.e_loc_mean,
        cbar_label="mean local energy (eV)",
        out_suffix="toplayer_local_energy",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        dpi=dpi,
    )


def plot_toplayer_local_energy_uncertainty(
    data: GroupPlotData,
    *,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
) -> Optional[Path]:
    """Scatter of top-layer local-energy ensemble std in the xy plane (eV)."""
    return _plot_toplayer_xy_scalar(
        data.group,
        data.xy,
        data.e_loc_std,
        cbar_label="local energy uncertainty (eV)",
        out_suffix="toplayer_local_energy_uncertainty",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
        dpi=dpi,
    )


def plot_unc_vs_disp(
    data: GroupPlotData,
    *,
    dpi: int = 150,
) -> Optional[Path]:
    """Scatter of position uncertainty vs displacement magnitude for one θ."""
    mag = np.asarray(data.mag, dtype=float)
    unc = np.asarray(data.unc, dtype=float)
    m = np.isfinite(mag) & np.isfinite(unc)
    if not np.any(m):
        return None
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.scatter(mag[m], unc[m], s=16, color="C0", linewidths=0.0, alpha=0.7)
    ax.set_xlabel("displacement mag. (Å)", fontdict=CSFONT)
    ax.set_ylabel("position uncertainty (Å)", fontdict=CSFONT)
    ax.set_title(rf"$\theta = {data.group.twist_angle:g}^\circ$", fontdict=CSFONT)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = (
        data.group.directory
        / (
            f"twist_angle_{data.group.twist_angle:g}_"
            f"{data.group.model_name}_unc_vs_disp.png"
        )
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_unc_vs_disp_all_twist(
    datasets: Sequence[GroupPlotData],
    out_path: Path,
    *,
    dpi: int = 150,
) -> Optional[Path]:
    """Overlay unc vs disp scatters for all twist angles, colored by θ."""
    if not datasets:
        return None
    thetas = np.array([d.group.twist_angle for d in datasets], dtype=float)
    tmin, tmax = float(np.min(thetas)), float(np.max(thetas))
    if tmin == tmax:
        tmax = tmin + 1e-12
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=tmin, vmax=tmax)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    sc = None
    for data in datasets:
        mag = np.asarray(data.mag, dtype=float)
        unc = np.asarray(data.unc, dtype=float)
        m = np.isfinite(mag) & np.isfinite(unc)
        if not np.any(m):
            continue
        color = cmap(norm(float(data.group.twist_angle)))
        sc = ax.scatter(
            mag[m], unc[m], s=12, c=[color], linewidths=0.0, alpha=0.65,
        )
    if sc is None:
        plt.close(fig)
        return None
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(r"twist angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_xlabel("displacement mag. (Å)", fontdict=CSFONT)
    ax.set_ylabel("position uncertainty (Å)", fontdict=CSFONT)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_toplayer_sepfield_samples(
    data: GroupPlotData,
    *,
    n_samples: int = DEFAULT_SEPFIELD_SAMPLES,
    seed: int = 0,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Scatter of top-layer ``2·(zᵢ − ⟨z⟩)`` at each atom xy for ``n_samples`` trajectories.
    """
    n = len(data.traj_paths)
    if n == 0:
        return None
    idxs = _select_sample_trajectories(list(range(n)), n_samples, seed)
    group = data.group

    n_pan = len(idxs)
    fig, axes = plt.subplots(1, n_pan, figsize=(6.5 * n_pan, 5.6), squeeze=False)
    axes_flat = axes.ravel()

    for ax, i in zip(axes_flat, idxs):
        traj_path = data.traj_paths[i]
        relaxed = read_relaxed_frame(traj_path)
        xy, sep = toplayer_separation_field(relaxed)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=sep,
            s=20,
            cmap="viridis",
            linewidths=0.0,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$2\,(z_i - \langle z \rangle)$ (Å)", fontdict=CSFONT)
        ax.set_xlabel("x (Å)", fontdict=CSFONT)
        ax.set_ylabel("y (Å)", fontdict=CSFONT)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        si = _sample_index_from_path(traj_path)
        label = f"sample {si}" if si is not None else traj_path.stem
        ax.set_title(label, fontdict=CSFONT)
        del relaxed

    fig.suptitle(
        rf"$\theta = {group.twist_angle:g}^\circ$ — top-layer separation",
        fontsize=CSFONT["size"],
        fontname=CSFONT["fontname"],
        y=1.02,
    )
    fig.tight_layout()
    out_path = (
        group.directory
        / f"twist_angle_{group.twist_angle:g}_{group.model_name}_toplayer_sepfield.png"
    )
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
    p.add_argument(
        "--no-mean-interlayer-separation",
        action="store_true",
        help="Skip the top-layer ensemble-mean interlayer-separation xy map.",
    )
    p.add_argument(
        "--no-mean-intralayer-displacement",
        action="store_true",
        help="Skip the top-layer ensemble-mean in-plane displacement xy map.",
    )
    p.add_argument(
        "--no-position-uncertainty",
        action="store_true",
        help="Skip the top-layer position-uncertainty scatter figure.",
    )
    p.add_argument(
        "--no-displacement-magnitude",
        action="store_true",
        help="Skip the top-layer 3D displacement-magnitude scatter figure.",
    )
    p.add_argument(
        "--no-local-energy",
        action="store_true",
        help="Skip the top-layer mean local-energy scatter figure.",
    )
    p.add_argument(
        "--no-local-energy-uncertainty",
        action="store_true",
        help="Skip the top-layer local-energy uncertainty scatter figure.",
    )
    p.add_argument(
        "--no-unc-vs-disp",
        action="store_true",
        help="Skip position-uncertainty vs displacement-magnitude scatters.",
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

    caches: List[GroupPlotData] = []
    for group in groups:
        print(
            f"  {group.model_name}  T={group.temperature_label}  "
            f"θ={group.twist_angle:g}°  n_traj={len(group.trajectory_paths)}",
            flush=True,
        )
        data = build_group_plot_data(
            group, npoints=args.npoints, fmax_max=args.fmax_max,
        )
        if data is None:
            continue
        caches.append(data)

        if not args.no_cross_section:
            print(f"    Wrote {plot_ensemble_cross_section(data, dpi=args.dpi)}", flush=True)
        if not args.no_toplayer_z_cross_section:
            print(
                f"    Wrote {plot_ensemble_toplayer_z_cross_section(data, dpi=args.dpi)}",
                flush=True,
            )
        if not args.no_intralayer_displacement:
            print(
                f"    Wrote {plot_ensemble_intralayer_displacement_cross_section(data, dpi=args.dpi)}",
                flush=True,
            )
        if not args.no_mean_interlayer_separation:
            out_ms = plot_toplayer_mean_interlayer_separation(
                data, dpi=args.dpi, show_colorbar=True,
            )
            if out_ms is not None:
                print(f"    Wrote {out_ms}", flush=True)
        if not args.no_mean_intralayer_displacement:
            out_md = plot_toplayer_mean_intralayer_displacement(
                data, dpi=args.dpi, show_colorbar=True,
            )
            if out_md is not None:
                print(f"    Wrote {out_md}", flush=True)
        if not args.no_sepfield:
            out_sf = plot_toplayer_sepfield_samples(
                data,
                n_samples=args.sepfield_samples,
                seed=args.sepfield_seed,
                dpi=args.dpi,
            )
            if out_sf is not None:
                print(f"    Wrote {out_sf}", flush=True)
        if not args.no_position_uncertainty:
            out_pu = plot_toplayer_position_uncertainty(
                data, dpi=args.dpi, show_colorbar=True,
            )
            if out_pu is not None:
                print(f"    Wrote {out_pu}", flush=True)
        if not args.no_displacement_magnitude:
            out_dm = plot_toplayer_displacement_magnitude(
                data, dpi=args.dpi, show_colorbar=True,
            )
            if out_dm is not None:
                print(f"    Wrote {out_dm}", flush=True)
        if not args.no_local_energy:
            out_el = plot_toplayer_local_energy(
                data, dpi=args.dpi, show_colorbar=True,
            )
            if out_el is not None:
                print(f"    Wrote {out_el}", flush=True)
        if not args.no_local_energy_uncertainty:
            out_eu = plot_toplayer_local_energy_uncertainty(
                data, dpi=args.dpi, show_colorbar=True,
            )
            if out_eu is not None:
                print(f"    Wrote {out_eu}", flush=True)
        if not args.no_unc_vs_disp:
            out_uv = plot_unc_vs_disp(data, dpi=args.dpi)
            if out_uv is not None:
                print(f"    Wrote {out_uv}", flush=True)
        gc.collect()

    if not args.no_unc_vs_disp:
        by_mt: Dict[Tuple[str, str], List[GroupPlotData]] = {}
        for data in caches:
            key = (data.group.model_name, data.group.temperature_label)
            by_mt.setdefault(key, []).append(data)
        for (model_name, t_label), items in by_mt.items():
            out_dir = items[0].group.directory.parent
            out_ov = plot_unc_vs_disp_all_twist(
                items,
                out_dir / f"{model_name}_T{t_label}_unc_vs_disp_all_twist.png",
                dpi=args.dpi,
            )
            if out_ov is not None:
                print(f"    Wrote {out_ov}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
