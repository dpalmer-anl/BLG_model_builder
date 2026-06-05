#!/usr/bin/env python3
"""
plot_tblg_structure_v_twist_angle.py
=====================================
Structural statistics of relaxed TBLG ensembles as a function of twist angle.

For each (model, temperature) pair found under ``trajectories/relaxation/``:

**Figure 1 — layer_sep_vs_twist_angle.png**
    Mean ± std (over ensemble) of the local layer separation at AA-stacking
    and AB-stacking atoms, plotted against initial twist angle θ.

**Figure 2 — local_twist_vs_twist_angle.png**
    Mean ± std (over ensemble) of the local twist angle at AA-stacking sites
    in the *relaxed* structure, plotted against initial twist angle θ.

Both figures are saved in ``trajectories/relaxation/<model_name>/T<label>/``.

Stacking identification
-----------------------
Atoms are classified from the **relaxed frame** (frame 1 of each ``.traj``
file) using the local interlayer separation:

* **AA** site: the top-layer atom with the **largest** interlayer separation
  ``2*(z_i − mean(z))``.  In relaxed TBLG the layers buckle outward at AA
  sites (atoms directly above each other repel), so these atoms sit highest.
* **AB** site: the top-layer atom with the **smallest** interlayer separation.
  AB/SP hollow sites are pulled inward and sit lowest.

Layer separation
----------------
``sep_i = 2 * (z_relaxed[i] − mean(z_relaxed))``

The sign is positive for top-layer atoms.  The value for the single AA (or
AB) representative atom is used directly; the ensemble mean ± std is plotted.

Local twist angle
-----------------
For each AA top-layer atom *i*:
1. Find its 3 nearest top-layer neighbours ``j_k`` (within ``--nn-cut``, default 1.65 Å)
   using the unrelaxed positions.
2. Compute the in-plane displacement vectors in the **unrelaxed** frame:
   ``r_k_0 = pos_j_k_0 − pos_i_0``.
3. Compute the same vectors in the **relaxed** frame:
   ``r_k_1 = pos_j_k_1 − pos_i_1``.
4. The rotation of neighbour *k* around atom *i* is
   ``Δφ_k = atan2(cross(r_k_0, r_k_1), dot(r_k_0, r_k_1))``
   (positive = counter-clockwise).
5. ``local_twist_i = θ_initial + mean_k(Δφ_k)``  (degrees)

The ensemble mean ± std of the per-AA-site local twist angles is plotted.

Examples
--------
::

    python visualizations/plot_tblg_structure_v_twist_angle.py
    python visualizations/plot_tblg_structure_v_twist_angle.py \\
        --trajectories-dir trajectories/relaxation
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
DEFAULT_TRAJ_ROOT = UQ_DIR / "trajectories" / "relaxation"

_RE_THETA_DIR = re.compile(r"^theta(.+)deg$", re.I)
_RE_T_DIR = re.compile(r"^T(.+)$", re.I)
_RE_SAMPLE_TRAJ = re.compile(r"_sample\d+\.traj$", re.I)

DEFAULT_NN_CUT: float = 1.65        # Å — in-plane NN cut-off for twist angle


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AngleStats:
    """Per-twist-angle statistics collected from one (model, T) ensemble."""
    theta: float                    # initial twist angle (degrees)

    # Layer separation statistics (eV across ensemble)
    aa_sep_mean: float = np.nan
    aa_sep_std: float = np.nan
    aa_n: int = 0                   # samples with ≥1 AA atom

    ab_sep_mean: float = np.nan
    ab_sep_std: float = np.nan
    ab_n: int = 0

    # Local twist angle statistics
    lt_mean: float = np.nan
    lt_std: float = np.nan
    lt_n: int = 0                   # samples with ≥1 AA site with NNs


# ---------------------------------------------------------------------------
# PBC-aware 2-D geometry helpers
# ---------------------------------------------------------------------------

def _frac_to_cart_2d(frac: np.ndarray, cell_2d: np.ndarray) -> np.ndarray:
    """Convert (N, 2) fractional → Cartesian using the 2×2 cell matrix."""
    return frac @ cell_2d


def _min_image_2d(
    diff_xy: np.ndarray,
    cell_2d: np.ndarray,
    cell_2d_inv: np.ndarray,
) -> np.ndarray:
    """Apply minimum-image convention in 2-D to an (N, 2) difference array."""
    frac = diff_xy @ cell_2d_inv
    frac -= np.round(frac)
    return frac @ cell_2d


def _build_cell_2d(atoms) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cell_2d, cell_2d_inv) from an ASE Atoms object."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    cell_2d = cell[:2, :2].copy()
    cell_2d_inv = np.linalg.inv(cell_2d)
    return cell_2d, cell_2d_inv


# ---------------------------------------------------------------------------
# Layer split
# ---------------------------------------------------------------------------

def _split_layers(pos: np.ndarray):
    """Return boolean masks for top and bottom layers."""
    z_mean = float(np.mean(pos[:, 2]))
    top_mask = pos[:, 2] > z_mean
    bot_mask = pos[:, 2] < z_mean
    return top_mask, bot_mask, z_mean


# ---------------------------------------------------------------------------
# Stacking classification (from relaxed frame)
# ---------------------------------------------------------------------------

def identify_stacking_atoms(relaxed_atoms) -> Tuple[int, int]:
    """Return the AA and AB representative atom indices from the relaxed frame.

    AA site: the top-layer atom with the **largest** interlayer separation
    ``2*(z_i − mean(z))``.  In relaxed TBLG the layers buckle outward at AA
    sites, so these atoms sit highest.

    AB site: the top-layer atom with the **smallest** interlayer separation.
    AB/SP hollow sites are pulled inward and sit lowest.

    Parameters
    ----------
    relaxed_atoms : ase.Atoms
        Relaxed TBLG structure (frame 1 of the trajectory).

    Returns
    -------
    aa_idx : int  — global atom index of the AA-site representative.
    ab_idx : int  — global atom index of the AB-site representative.
    """
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    top_mask, _, z_mean = _split_layers(pos)
    top_idx = np.where(top_mask)[0]
    sep = 2.0 * (pos[top_idx, 2] - z_mean)
    aa_idx = int(top_idx[np.argmax(sep)])
    ab_idx = int(top_idx[np.argmin(sep)])
    return aa_idx, ab_idx


# ---------------------------------------------------------------------------
# Layer separation
# ---------------------------------------------------------------------------

def layer_sep_for_indices(relaxed_atoms, atom_idx: np.ndarray) -> np.ndarray:
    """Compute ``2*(z[atom_idx] - mean(z))`` (Å) for *atom_idx* atoms.

    Parameters
    ----------
    relaxed_atoms : ase.Atoms
    atom_idx : ndarray — atom indices (from the same atoms object).

    Returns
    -------
    sep : ndarray, shape (len(atom_idx),) — signed layer separation per atom.
    """
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    z_mean = float(np.mean(pos[:, 2]))
    return 2.0 * (pos[atom_idx, 2] - z_mean)


# ---------------------------------------------------------------------------
# Local twist angle
# ---------------------------------------------------------------------------

def _inplane_nn_indices(
    atoms,
    center_idx: int,
    layer_mask: np.ndarray,
    nn_cut: float,
) -> np.ndarray:
    """Global indices of in-plane NNs of *center_idx* within *nn_cut* Å (MIC).

    Only atoms flagged by *layer_mask* are considered.
    """
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    cell_2d, cell_2d_inv = _build_cell_2d(atoms)

    p0 = pos[center_idx, :2]
    candidate_idx = np.where(layer_mask)[0]
    candidate_idx = candidate_idx[candidate_idx != center_idx]

    diffs = pos[candidate_idx, :2] - p0
    mic = _min_image_2d(diffs, cell_2d, cell_2d_inv)
    dists = np.linalg.norm(mic, axis=1)
    return candidate_idx[dists < nn_cut]


def local_twist_at_aa_sites(
    initial_atoms,
    relaxed_atoms,
    aa_top_idx: np.ndarray,
    theta_initial_deg: float,
    nn_cut: float = DEFAULT_NN_CUT,
) -> np.ndarray:
    """Compute the local twist angle (degrees) at each AA-stacking site.

    For each AA top-layer atom *i*:
    1. Find its 3 nearest top-layer neighbours in the *unrelaxed* frame.
    2. For each neighbour *j_k*, compute the in-plane rotation of the bond
       vector ``pos_jk − pos_i`` between the unrelaxed and relaxed frames.
    3. The local twist angle is ``θ_initial + mean_k(Δφ_k)``.

    Parameters
    ----------
    initial_atoms, relaxed_atoms : ase.Atoms
        Initial and relaxed frames (must have the same atom ordering).
    aa_top_idx : ndarray
        Global indices of AA top-layer atoms (from identify_stacking_atoms).
    theta_initial_deg : float
        Nominal twist angle of this structure.
    nn_cut : float
        In-plane NN search radius (Å).

    Returns
    -------
    local_twists : ndarray, shape (n_sites,) — one value per AA site.
        Sites with fewer than 2 NNs are skipped (not included in the output).
    """
    pos0 = np.asarray(initial_atoms.get_positions(wrap=False), dtype=float)
    pos1 = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    cell_2d_0, cell_2d_inv_0 = _build_cell_2d(initial_atoms)
    cell_2d_1, cell_2d_inv_1 = _build_cell_2d(relaxed_atoms)

    top_mask0 = pos0[:, 2] > float(np.mean(pos0[:, 2]))
    top_mask1 = pos1[:, 2] > float(np.mean(pos1[:, 2]))

    local_twists: List[float] = []

    for i in aa_top_idx:
        # NNs identified from the unrelaxed frame
        nn_idx = _inplane_nn_indices(initial_atoms, int(i), top_mask0, nn_cut)
        if len(nn_idx) < 2:
            continue

        delta_phis: List[float] = []
        for j in nn_idx:
            # Bond vector in unrelaxed frame (MIC)
            d0 = pos0[j, :2] - pos0[i, :2]
            d0 = _min_image_2d(d0[np.newaxis, :], cell_2d_0, cell_2d_inv_0)[0]

            # Bond vector in relaxed frame (MIC)
            d1 = pos1[j, :2] - pos1[i, :2]
            d1 = _min_image_2d(d1[np.newaxis, :], cell_2d_1, cell_2d_inv_1)[0]

            # Rotation of bond vector between frames
            cross_z = float(d0[0] * d1[1] - d0[1] * d1[0])
            dot_val = float(np.dot(d0, d1))
            delta_phis.append(np.degrees(np.arctan2(cross_z, dot_val)))

        if delta_phis:
            local_twists.append(theta_initial_deg + float(np.mean(delta_phis)))

    return np.asarray(local_twists, dtype=float)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def read_both_frames(traj_path: Path):
    """Return (initial_atoms, relaxed_atoms) from a 2-frame trajectory.

    Raises ValueError if the trajectory has fewer than 2 frames or either
    frame looks invalid.
    """
    with Trajectory(str(traj_path), "r") as traj:
        n = len(traj)
        if n < 2:
            raise ValueError(
                f"{traj_path.name}: need ≥2 frames (initial + relaxed), "
                f"found {n}."
            )
        initial = traj[0].copy()
        relaxed = traj[1].copy()
    return initial, relaxed


def process_sample(
    traj_path: Path,
    theta_deg: float,
    nn_cut: float,
) -> Optional[Dict]:
    """Extract stacking statistics from one trajectory file.

    Returns a dict with keys ``aa_sep, ab_sep, local_twist``, or None on
    failure.  Stacking sites are identified from the relaxed frame as the
    top-layer atoms with the largest (AA) and smallest (AB) interlayer
    separation.
    """
    try:
        initial, relaxed = read_both_frames(traj_path)
    except Exception as exc:
        print(f"    skip {traj_path.name}: {exc}", file=sys.stderr)
        return None

    try:
        aa_idx, ab_idx = identify_stacking_atoms(relaxed)
    except Exception as exc:
        print(f"    stacking id failed {traj_path.name}: {exc}", file=sys.stderr)
        return None

    # Layer separations — single representative atom per site type
    aa_sep_arr = layer_sep_for_indices(relaxed, np.array([aa_idx]))
    ab_sep_arr = layer_sep_for_indices(relaxed, np.array([ab_idx]))

    result: Dict = {
        "aa_sep": float(aa_sep_arr[0]),
        "ab_sep": float(ab_sep_arr[0]),
    }

    # Local twist angle (uses AA atom identified from relaxed frame)
    try:
        ltwists = local_twist_at_aa_sites(
            initial, relaxed, np.array([aa_idx]), theta_deg, nn_cut=nn_cut
        )
        result["local_twist"] = float(np.mean(ltwists)) if len(ltwists) > 0 else np.nan
    except Exception as exc:
        print(f"    twist calc failed {traj_path.name}: {exc}", file=sys.stderr)
        result["local_twist"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Ensemble-level aggregation
# ---------------------------------------------------------------------------

def compute_angle_stats(
    traj_paths: List[Path],
    theta_deg: float,
    nn_cut: float,
) -> AngleStats:
    """Collect per-sample results and compute ensemble mean ± std."""
    aa_seps, ab_seps, local_twists = [], [], []

    for tp in traj_paths:
        res = process_sample(tp, theta_deg, nn_cut)
        if res is None:
            continue
        if np.isfinite(res.get("aa_sep", np.nan)):
            aa_seps.append(res["aa_sep"])
        if np.isfinite(res.get("ab_sep", np.nan)):
            ab_seps.append(res["ab_sep"])
        if np.isfinite(res.get("local_twist", np.nan)):
            local_twists.append(res["local_twist"])

    stats = AngleStats(theta=theta_deg)

    if aa_seps:
        stats.aa_sep_mean = float(np.mean(aa_seps))
        stats.aa_sep_std = float(np.std(aa_seps, ddof=1) if len(aa_seps) > 1 else 0.0)
        stats.aa_n = len(aa_seps)

    if ab_seps:
        stats.ab_sep_mean = float(np.mean(ab_seps))
        stats.ab_sep_std = float(np.std(ab_seps, ddof=1) if len(ab_seps) > 1 else 0.0)
        stats.ab_n = len(ab_seps)

    if local_twists:
        stats.lt_mean = float(np.mean(local_twists))
        stats.lt_std = float(np.std(local_twists, ddof=1) if len(local_twists) > 1 else 0.0)
        stats.lt_n = len(local_twists)

    return stats


# ---------------------------------------------------------------------------
# Trajectory discovery (same regex as cross_section_ensemble)
# ---------------------------------------------------------------------------

@dataclass
class EnsembleGroup:
    model_name: str
    temperature_label: str
    twist_angle: float
    directory: Path
    trajectory_paths: Tuple[Path, ...]


def discover_ensemble_groups(traj_root: Path) -> List[EnsembleGroup]:
    if not traj_root.is_dir():
        return []
    groups: List[EnsembleGroup] = []
    for model_dir in sorted(traj_root.iterdir()):
        if not model_dir.is_dir():
            continue
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
                if trajs:
                    groups.append(EnsembleGroup(
                        model_name=model_dir.name,
                        temperature_label=t_label,
                        twist_angle=twist,
                        directory=theta_dir,
                        trajectory_paths=tuple(trajs),
                    ))
    return groups


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_layer_sep_vs_twist(
    stats_list: List[AngleStats],
    model_name: str,
    t_label: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Layer separation at AA and AB stacking vs twist angle.

    AA and AB are shown on separate subplots.  The ensemble mean is drawn as
    a solid line with markers; the ±std band is shown with fill_between.
    """
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list])

    aa_mean = np.array([s.aa_sep_mean for s in stats_list])
    aa_std  = np.array([s.aa_sep_std  for s in stats_list])
    ab_mean = np.array([s.ab_sep_mean for s in stats_list])
    ab_std  = np.array([s.ab_sep_std  for s in stats_list])

    ylabel = r"Layer separation $2(z_i - \langle z \rangle)$ (Å)"
    xlabel = r"Initial twist angle $\theta$ (°)"

    fig, (ax_aa, ax_ab) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    # --- AA subplot ---
    valid_aa = np.isfinite(aa_mean)
    if valid_aa.any():
        t_aa = thetas[valid_aa]
        m_aa = aa_mean[valid_aa]
        s_aa = aa_std[valid_aa]
        ax_aa.plot(t_aa, m_aa, "o-", color="C0", label="AA stacking")
        ax_aa.fill_between(t_aa, m_aa - s_aa, m_aa + s_aa,
                           color="C0", alpha=0.3)
    ax_aa.set_xlabel(xlabel, fontsize=12)
    ax_aa.set_ylabel(ylabel, fontsize=12)
    ax_aa.set_title("AA stacking", fontsize=11)
    ax_aa.legend(fontsize=10)
    ax_aa.grid(True, alpha=0.3)

    # --- AB subplot ---
    valid_ab = np.isfinite(ab_mean)
    if valid_ab.any():
        t_ab = thetas[valid_ab]
        m_ab = ab_mean[valid_ab]
        s_ab = ab_std[valid_ab]
        ax_ab.plot(t_ab, m_ab, "s-", color="C1", label="AB stacking")
        ax_ab.fill_between(t_ab, m_ab - s_ab, m_ab + s_ab,
                           color="C1", alpha=0.3)
    ax_ab.set_xlabel(xlabel, fontsize=12)
    ax_ab.set_ylabel(ylabel, fontsize=12)
    ax_ab.set_title("AB stacking", fontsize=11)
    ax_ab.legend(fontsize=10)
    ax_ab.grid(True, alpha=0.3)

    fig.suptitle(
        rf"{model_name}  $T = {t_label}$  — local layer separation",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_local_twist_vs_theta(
    stats_list: List[AngleStats],
    model_name: str,
    t_label: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Local twist angle at AA sites vs initial twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list])
    lt_mean = np.array([s.lt_mean for s in stats_list])
    lt_std = np.array([s.lt_std for s in stats_list])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    valid = np.isfinite(lt_mean)
    if valid.any():
        t_v = thetas[valid]
        m_v = lt_mean[valid]
        s_v = lt_std[valid]
        ax.plot(t_v, m_v, "o-", color="C2", label="local twist (AA sites)")
        ax.fill_between(t_v, m_v - s_v, m_v + s_v, color="C2", alpha=0.3)
        # Reference line y = x
        theta_range = np.array([t_v.min(), t_v.max()])
        ax.plot(theta_range, theta_range, "k--", linewidth=0.8, alpha=0.6,
                label=r"$\theta_\mathrm{local} = \theta_\mathrm{initial}$")

    ax.set_xlabel(r"Initial twist angle $\theta$ (°)", fontsize=12)
    ax.set_ylabel(r"Local twist angle at AA site (°)", fontsize=12)
    ax.set_title(
        rf"{model_name}  $T = {t_label}$" + "\nLocal twist angle at AA stacking",
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Structural statistics of relaxed TBLG ensembles vs twist angle."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trajectories-dir",
        type=Path,
        default=DEFAULT_TRAJ_ROOT,
        help=f"Root directory for relaxation trajectories (default: {DEFAULT_TRAJ_ROOT}).",
    )
    p.add_argument(
        "--nn-cut",
        type=float,
        default=DEFAULT_NN_CUT,
        help=(
            f"In-plane NN search radius (Å) for local twist angle calculation "
            f"(default: {DEFAULT_NN_CUT})."
        ),
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--no-layer-sep",
        action="store_true",
        help="Skip layer separation figure.",
    )
    p.add_argument(
        "--no-local-twist",
        action="store_true",
        help="Skip local twist angle figure.",
    )
    args = p.parse_args()

    os.chdir(UQ_DIR)
    traj_root = Path(args.trajectories_dir)
    if not traj_root.is_absolute():
        traj_root = UQ_DIR / traj_root

    groups = discover_ensemble_groups(traj_root)
    if not groups:
        p.error(f"No TBLG relaxation ensembles found under {traj_root}")

    print(
        f"Found {len(groups)} ensemble group(s) across "
        f"{len({g.twist_angle for g in groups})} twist angle(s).",
        flush=True,
    )

    # Group by (model_name, temperature_label)
    by_model_t: Dict[Tuple[str, str], List[EnsembleGroup]] = {}
    for g in groups:
        key = (g.model_name, g.temperature_label)
        by_model_t.setdefault(key, []).append(g)

    for (model_name, t_label), angle_groups in sorted(by_model_t.items()):
        print(
            f"\n{'='*60}\n Model: {model_name}  T={t_label}  "
            f"({len(angle_groups)} twist angle(s))\n{'='*60}",
            flush=True,
        )

        stats_list: List[AngleStats] = []

        for grp in sorted(angle_groups, key=lambda g: g.twist_angle):
            theta = grp.twist_angle
            n_traj = len(grp.trajectory_paths)
            print(
                f"  θ={theta:g}°  {n_traj} trajectory file(s) …",
                flush=True,
            )
            st = compute_angle_stats(
                list(grp.trajectory_paths),
                theta,
                nn_cut=args.nn_cut,
            )
            stats_list.append(st)
            print(
                f"    AA sep={st.aa_sep_mean:.3f}±{st.aa_sep_std:.3f} Å (n={st.aa_n})  "
                f"AB sep={st.ab_sep_mean:.3f}±{st.ab_sep_std:.3f} Å (n={st.ab_n})  "
                f"local θ={st.lt_mean:.3f}±{st.lt_std:.3f}° (n={st.lt_n})",
                flush=True,
            )

        if not stats_list:
            continue

        # Save figures in trajectories/relaxation/<model>/<T>/
        out_dir = traj_root / model_name / f"T{t_label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_layer_sep:
            plot_layer_sep_vs_twist(
                stats_list,
                model_name=model_name,
                t_label=t_label,
                out_path=out_dir / "layer_sep_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_local_twist:
            plot_local_twist_vs_theta(
                stats_list,
                model_name=model_name,
                t_label=t_label,
                out_path=out_dir / "local_twist_vs_twist_angle.png",
                dpi=args.dpi,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
