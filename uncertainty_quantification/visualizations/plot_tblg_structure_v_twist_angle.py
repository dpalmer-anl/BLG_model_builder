#!/usr/bin/env python3
"""
plot_tblg_structure_v_twist_angle.py
=====================================
Structural statistics of relaxed TBLG ensembles as a function of twist angle.

For each (model, temperature) pair found under ``trajectories/relaxation/``,
one figure is written per extracted quantity (no plot titles):

* ``aa_layer_sep_vs_twist_angle.png`` — AA-site layer separation
* ``ab_layer_sep_vs_twist_angle.png`` — AB-site layer separation
* ``mean_layer_sep_vs_twist_angle.png`` — mean top-layer interlayer
  separation over the moiré cell
* ``mean_z_vs_twist_angle.png`` — mean Cartesian *z* over all atoms
* ``corrugation_amplitude_vs_twist_angle.png`` — AA − AB top-layer
  separation (corrugation amplitude)
* ``dw_width_vs_twist_angle.png`` — domain wall width from the
  intralayer-displacement cross section
* ``max_intralayer_disp_vs_twist_angle.png`` — largest fitted in-plane
  displacement peak on the moiré cross section
* ``local_twist_vs_twist_angle.png`` — local twist angle at AA stacking
* ``cell_vector_lengths_vs_twist_angle.png`` — lengths of cell vectors
  ``|a₁|`` and ``|a₂|`` (first sample only; cell is fixed across the ensemble)
* ``elastic_inplane_A_mode1_vs_twist_angle.png`` — top-layer elastic-plate
  in-plane coefficient ``A`` (mode 1)
* ``elastic_outplane_D_mode1_vs_twist_angle.png`` — top-layer elastic-plate
  out-of-plane coefficient ``D`` (mode 1)

All figures are saved in ``trajectories/relaxation/<model_name>/T<label>/``.

Ensemble selection
------------------
Samples are included only when the relaxed frame has saved forces with
``np.max(atoms.get_forces()) ≤ 1e-4`` eV/Å (override with ``--fmax-max``).
Trajectories without forces are omitted with a warning.

Mean *z*
--------
``mean_z = mean_i(z_i)`` over **all** atoms (top and bottom layers) in the
relaxed frame.  Ensemble mean ± std vs twist angle.

Corrugation amplitude
---------------------
``corrugation = sep_AA − sep_AB`` using the same top-layer AA/AB
representatives as the layer-separation figures
(``sep = 2*(z − mean(z))``).  Ensemble mean ± std vs twist angle.

Domain wall width
-----------------
For each sample, the top-layer intralayer displacement magnitude is interpolated
along the moiré diagonal (same path as
``plot_tblg_cross_section_ensemble.get_intralayer_displacement_cross_sect``).
The known AA/AB/SP/BA/AA positions split that path into four intervals.  One
Gaussian peak is fit locally in each interval; the domain wall width is the
path-length difference between the two middle fitted peak centers.

Max intralayer displacement
---------------------------
The maximum of the four locally fitted displacement-peak heights (ensemble
mean ± std vs twist).

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

Mean interlayer separation (moiré cell)
---------------------------------------
For every top-layer atom (``z > mean(z)``)::

    sep_i = 2 * |z_i − mean(z)|

then ``mean_layer_sep = mean_i(sep_i)`` over the top layer (moiré-cell average).
Ensemble mean ± std vs twist angle.

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

Elastic plate Fourier coefficients (mode 1, top layer)
-----------------------------------------------------
Displacements ``u = r_relaxed − r_initial`` on top-layer atoms
(``z_initial > mean(z_initial)``) are projected onto the continuum elastic
plate Fourier basis (see ``elastic_plate_basis.py``, ported from
``Elastic_basis_Dan``).  Reported vs twist:

* in-plane coefficient ``A`` of **mode 1**
* out-of-plane coefficient ``D`` of **mode 1**

Ensemble mean ± std.

Examples
--------
::

    python visualizations/plot_tblg_structure_v_twist_angle.py
    python visualizations/plot_tblg_structure_v_twist_angle.py \\
        --trajectories-dir trajectories/relaxation

    # One temperature: all twist angles (glob)::
    python visualizations/plot_tblg_structure_v_twist_angle.py \\
        --trajectories-dir \\
        trajectories/relaxation/POD_energy_POD_index_27_*/T0.65/theta*
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

CSFONT = {"fontname": "sans-serif", "size": 20}
LEGEND_FONTSIZE = 12
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

_vis_dir = str(HERE)
if _vis_dir not in sys.path:
    sys.path.insert(0, _vis_dir)

from plot_tblg_cross_section_ensemble import (  # noqa: E402
    DEFAULT_FMAX_MAX,
    DEFAULT_NPOINTS,
    EnsembleGroup,
    STACKING_FRACTIONS,
    _passes_fmax_gate,
    discover_ensemble_groups,
    expand_trajectory_path_patterns,
    get_intralayer_displacement_cross_sect,
)
from elastic_plate_basis import top_layer_mode1_A_D  # noqa: E402

DEFAULT_NN_CUT: float = 1.65        # Å — in-plane NN cut-off for twist angle
DISP_PEAK_FIT_HALF_WIDTH: float = 8.0  # Å around each section maximum


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AngleStats:
    """Per-twist-angle statistics collected from one (model, T) ensemble."""
    theta: float                    # initial twist angle (degrees)

    # Layer separation statistics (Å across ensemble)
    aa_sep_mean: float = np.nan
    aa_sep_std: float = np.nan
    aa_n: int = 0                   # samples with ≥1 AA atom

    ab_sep_mean: float = np.nan
    ab_sep_std: float = np.nan
    ab_n: int = 0

    # Mean top-layer interlayer separation over the moiré cell (Å)
    mean_sep_mean: float = np.nan
    mean_sep_std: float = np.nan
    mean_sep_n: int = 0

    # Mean Cartesian z over all atoms (Å)
    mean_z_mean: float = np.nan
    mean_z_std: float = np.nan
    mean_z_n: int = 0

    # Corrugation amplitude = AA − AB layer separation (Å)
    corr_mean: float = np.nan
    corr_std: float = np.nan
    corr_n: int = 0

    # Domain wall width (Å) from middle two of four fitted peak centers
    dw_mean: float = np.nan
    dw_std: float = np.nan
    dw_n: int = 0

    # Largest of four fitted cross-section displacement peaks (Å)
    max_disp_mean: float = np.nan
    max_disp_std: float = np.nan
    max_disp_n: int = 0

    # Local twist angle statistics
    lt_mean: float = np.nan
    lt_std: float = np.nan
    lt_n: int = 0                   # samples with ≥1 AA site with NNs

    # Elastic plate Fourier coeffs (mode 1, top layer), Å
    A1_mean: float = np.nan
    A1_std: float = np.nan
    A1_n: int = 0
    D1_mean: float = np.nan
    D1_std: float = np.nan
    D1_n: int = 0

    # Atom count of the TBLG cell (from relaxed frame; same for all samples)
    n_atoms: int = 0


# ---------------------------------------------------------------------------
# PBC-aware 2-D geometry helpers
# ---------------------------------------------------------------------------

def _frac_to_cart_2d(frac: np.ndarray, cell_2d: np.ndarray) -> np.ndarray:
    """Convert (N, 2) fractional → Cartesian using the 2×2 cell matrix."""
    return frac @ cell_2d


def _mic_vectors(
    diffs: np.ndarray,
    cell: np.ndarray,
    *,
    pbc=(True, True, False),
) -> np.ndarray:
    """Minimum-image Cartesian vectors for an ``(N, 3)`` difference array."""
    from ase.geometry import find_mic

    diffs = np.asarray(diffs, dtype=float)
    if diffs.ndim == 1:
        diffs = diffs.reshape(1, -1)
    mic, _ = find_mic(diffs, np.asarray(cell, dtype=float), pbc=list(pbc))
    return np.asarray(mic, dtype=float)


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


def mean_z_all_atoms(relaxed_atoms) -> float:
    """Mean Cartesian *z* (Å) over all atoms in the relaxed structure."""
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    return float(np.mean(pos[:, 2]))


def mean_top_layer_interlayer_sep(relaxed_atoms) -> float:
    """
    Mean interlayer separation (Å) over top-layer atoms in the moiré cell.

    For each top-layer atom (``z > mean(z)``)::

        sep_i = 2 * |z_i − mean(z)|

    Returns the mean of ``sep_i`` over the top layer.
    """
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    top_mask, _, z_mean = _split_layers(pos)
    if int(np.count_nonzero(top_mask)) < 1:
        return float("nan")
    sep = 2.0 * np.abs(pos[top_mask, 2] - z_mean)
    return float(np.mean(sep))


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
    cell = np.asarray(atoms.get_cell(), dtype=float)

    candidate_idx = np.where(layer_mask)[0]
    candidate_idx = candidate_idx[candidate_idx != center_idx]
    if candidate_idx.size == 0:
        return candidate_idx

    diffs = pos[candidate_idx] - pos[center_idx]
    mic = _mic_vectors(diffs, cell, pbc=(True, True, False))
    dists = np.linalg.norm(mic[:, :2], axis=1)
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
       vector ``pos_jk − pos_i`` between the unrelaxed and relaxed frames
       (minimum-image convention / PBC).
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
    cell0 = np.asarray(initial_atoms.get_cell(), dtype=float)
    cell1 = np.asarray(relaxed_atoms.get_cell(), dtype=float)

    top_mask0 = pos0[:, 2] > float(np.mean(pos0[:, 2]))

    local_twists: List[float] = []

    for i in aa_top_idx:
        # NNs identified from the unrelaxed frame
        nn_idx = _inplane_nn_indices(initial_atoms, int(i), top_mask0, nn_cut)
        if len(nn_idx) < 2:
            continue

        delta_phis: List[float] = []
        for j in nn_idx:
            # Bond vectors with full-cell MIC (handles triclinic PBC correctly)
            d0 = _mic_vectors(pos0[j] - pos0[i], cell0, pbc=(True, True, False))[0, :2]
            d1 = _mic_vectors(pos1[j] - pos1[i], cell1, pbc=(True, True, False))[0, :2]

            # Rotation of bond vector between frames
            cross_z = float(d0[0] * d1[1] - d0[1] * d1[0])
            dot_val = float(np.dot(d0, d1))
            delta_phis.append(np.degrees(np.arctan2(cross_z, dot_val)))

        if delta_phis:
            local_twists.append(theta_initial_deg + float(np.mean(delta_phis)))

    return np.asarray(local_twists, dtype=float)


# ---------------------------------------------------------------------------
# Domain wall width / max intralayer displacement
# ---------------------------------------------------------------------------

def fit_intralayer_displacement_peaks(
    path_len: np.ndarray,
    displacement: np.ndarray,
    *,
    stacking_fractions: np.ndarray = STACKING_FRACTIONS,
    fit_half_width: float = DISP_PEAK_FIT_HALF_WIDTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit one displacement-magnitude peak between each pair of stackings.

    The moiré diagonal has stackings at path fractions
    ``[0, 0.33, 0.5, 0.66, 1]`` (AA, AB, SP, BA, AA), hence four known
    intervals containing one displacement peak each.  In every interval, find
    the sampled maximum and fit ``baseline + amplitude * Gaussian`` to points
    within ``fit_half_width`` Å of it.

    Returns
    -------
    centers, heights
        Fitted peak centers (Å along the path) and fitted peak heights (Å).
        A failed fit falls back to that interval's sampled maximum.
    """
    path_len = np.asarray(path_len, dtype=float).ravel()
    displacement = np.asarray(displacement, dtype=float).ravel()
    m = np.isfinite(path_len) & np.isfinite(displacement)
    path_len, displacement = path_len[m], displacement[m]
    fractions = np.asarray(stacking_fractions, dtype=float).ravel()
    n_peaks = fractions.size - 1
    centers = np.full(n_peaks, np.nan, dtype=float)
    heights = np.full(n_peaks, np.nan, dtype=float)
    if path_len.size < 5 or n_peaks < 1:
        return centers, heights

    total_len = float(path_len[-1])
    dx = float(np.median(np.diff(path_len)))

    def gaussian(
        x: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float,
    ) -> np.ndarray:
        return baseline + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    for i, (f_lo, f_hi) in enumerate(zip(fractions[:-1], fractions[1:])):
        x_lo, x_hi = float(f_lo * total_len), float(f_hi * total_len)
        section = (path_len >= x_lo) & (path_len <= x_hi)
        section_idx = np.flatnonzero(section)
        if section_idx.size == 0:
            continue

        peak_idx = int(section_idx[np.argmax(displacement[section_idx])])
        peak_x = float(path_len[peak_idx])
        peak_y = float(displacement[peak_idx])
        fit_mask = section & (np.abs(path_len - peak_x) <= float(fit_half_width))
        x_fit, y_fit = path_len[fit_mask], displacement[fit_mask]

        # Always retain a deterministic section-maximum fallback.
        centers[i], heights[i] = peak_x, peak_y
        if x_fit.size < 5:
            continue

        baseline0 = max(0.0, float(np.min(y_fit)))
        amplitude0 = max(peak_y - baseline0, np.finfo(float).eps)
        interval_width = max(x_hi - x_lo, dx)
        try:
            popt, _ = curve_fit(
                gaussian,
                x_fit,
                y_fit,
                p0=(baseline0, amplitude0, peak_x, min(4.0, interval_width / 4.0)),
                bounds=(
                    (0.0, 0.0, x_lo, max(dx / 2.0, 1e-3)),
                    (
                        max(2.0 * peak_y, 1e-6),
                        max(2.0 * peak_y, 1e-6),
                        x_hi,
                        interval_width,
                    ),
                ),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        centers[i] = float(popt[2])
        heights[i] = float(popt[0] + popt[1])

    return centers, heights


def displacement_metrics_from_peak_fits(
    path_len: np.ndarray,
    displacement: np.ndarray,
) -> Tuple[float, float]:
    """Return ``(domain-wall width, max displacement)`` from four peak fits."""
    centers, heights = fit_intralayer_displacement_peaks(path_len, displacement)
    if centers.size < 4 or not np.all(np.isfinite(centers)):
        dw_width = float("nan")
    else:
        dw_width = float(abs(centers[2] - centers[1]))
    max_disp = (
        float(np.max(heights[np.isfinite(heights)]))
        if np.any(np.isfinite(heights))
        else float("nan")
    )
    return dw_width, max_disp


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def read_both_frames(traj_path: Path):
    """Return (initial_atoms, relaxed_atoms) via :func:`ase.io.read`.

    Raises ValueError if the trajectory has fewer than 2 frames.
    Keeps attached calculators so ``atoms.get_forces()`` can be used for gating.
    """
    import ase.io

    frames = ase.io.read(str(traj_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) < 2:
        raise ValueError(
            f"{traj_path.name}: need ≥2 frames (initial + relaxed), "
            f"found {len(frames)}."
        )
    return frames[0], frames[-1]


def process_sample(
    traj_path: Path,
    theta_deg: float,
    nn_cut: float,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> Optional[Dict]:
    """Extract stacking statistics from one trajectory file.

    Returns a dict with keys ``aa_sep, ab_sep, mean_sep, mean_z, corrugation,
    dw_width, max_intralayer_disp, local_twist, A_mode1, D_mode1``, or None on
    failure.  Stacking sites are identified from the relaxed frame as the
    top-layer atoms with the largest (AA) and smallest (AB) interlayer
    separation.

    Inclusion rule (same as ``plot_tblg_cross_section_ensemble``): require
    saved forces on the relaxed frame with
    ``np.max(atoms.get_forces()) ≤ fmax_max``.
    """
    try:
        initial, relaxed = read_both_frames(traj_path)
    except Exception as exc:
        print(f"    skip {traj_path.name}: {exc}", file=sys.stderr)
        return None

    if not _passes_fmax_gate(relaxed, traj_path, fmax_max=fmax_max):
        return None

    try:
        aa_idx, ab_idx = identify_stacking_atoms(relaxed)
    except Exception as exc:
        print(f"    stacking id failed {traj_path.name}: {exc}", file=sys.stderr)
        return None

    # Layer separations — single representative atom per site type
    aa_sep_arr = layer_sep_for_indices(relaxed, np.array([aa_idx]))
    ab_sep_arr = layer_sep_for_indices(relaxed, np.array([ab_idx]))
    aa_sep = float(aa_sep_arr[0])
    ab_sep = float(ab_sep_arr[0])

    result: Dict = {
        "aa_sep": aa_sep,
        "ab_sep": ab_sep,
        "mean_sep": mean_top_layer_interlayer_sep(relaxed),
        "mean_z": mean_z_all_atoms(relaxed),
        "corrugation": aa_sep - ab_sep,
        "n_atoms": int(len(relaxed)),
        "A_mode1": float("nan"),
        "D_mode1": float("nan"),
    }

    # Fit one peak in each known stacking interval and derive both metrics from
    # the same smooth peak model.
    try:
        path_len, disp = get_intralayer_displacement_cross_sect(
            initial, relaxed, npoints=npoints,
        )
        dw_width, max_disp = displacement_metrics_from_peak_fits(path_len, disp)
        result["dw_width"] = dw_width
        result["max_intralayer_disp"] = max_disp
    except Exception as exc:
        print(f"    displacement peak fits failed {traj_path.name}: {exc}", file=sys.stderr)
        result["dw_width"] = float("nan")
        result["max_intralayer_disp"] = float("nan")

    # Local twist angle (uses AA atom identified from relaxed frame)
    try:
        ltwists = local_twist_at_aa_sites(
            initial, relaxed, np.array([aa_idx]), theta_deg, nn_cut=nn_cut
        )
        result["local_twist"] = float(np.mean(ltwists)) if len(ltwists) > 0 else np.nan
    except Exception as exc:
        print(f"    twist calc failed {traj_path.name}: {exc}", file=sys.stderr)
        result["local_twist"] = np.nan

    # Elastic plate Fourier coeffs (mode 1): in-plane A, out-of-plane D
    try:
        A1, D1 = top_layer_mode1_A_D(initial, relaxed)
        result["A_mode1"] = A1
        result["D_mode1"] = D1
    except Exception as exc:
        print(f"    elastic basis failed {traj_path.name}: {exc}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Ensemble-level aggregation
# ---------------------------------------------------------------------------

def compute_angle_stats(
    traj_paths: List[Path],
    theta_deg: float,
    nn_cut: float,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> AngleStats:
    """Collect per-sample results and compute ensemble mean ± std."""
    aa_seps, ab_seps, mean_seps, mean_zs, corrs, dw_widths, max_disps, local_twists = (
        [], [], [], [], [], [], [], [],
    )
    A1_vals: List[float] = []
    D1_vals: List[float] = []
    n_atoms = 0

    for tp in traj_paths:
        res = process_sample(
            tp, theta_deg, nn_cut, npoints=npoints, fmax_max=fmax_max,
        )
        if res is None:
            continue
        if n_atoms <= 0 and int(res.get("n_atoms", 0)) > 0:
            n_atoms = int(res["n_atoms"])
        if np.isfinite(res.get("aa_sep", np.nan)):
            aa_seps.append(res["aa_sep"])
        if np.isfinite(res.get("ab_sep", np.nan)):
            ab_seps.append(res["ab_sep"])
        if np.isfinite(res.get("mean_sep", np.nan)):
            mean_seps.append(res["mean_sep"])
        if np.isfinite(res.get("mean_z", np.nan)):
            mean_zs.append(res["mean_z"])
        if np.isfinite(res.get("corrugation", np.nan)):
            corrs.append(res["corrugation"])
        if np.isfinite(res.get("dw_width", np.nan)):
            dw_widths.append(res["dw_width"])
        if np.isfinite(res.get("max_intralayer_disp", np.nan)):
            max_disps.append(res["max_intralayer_disp"])
        if np.isfinite(res.get("local_twist", np.nan)):
            local_twists.append(res["local_twist"])
        if np.isfinite(res.get("A_mode1", np.nan)):
            A1_vals.append(float(res["A_mode1"]))
        if np.isfinite(res.get("D_mode1", np.nan)):
            D1_vals.append(float(res["D_mode1"]))

    stats = AngleStats(theta=theta_deg, n_atoms=int(n_atoms))

    if aa_seps:
        stats.aa_sep_mean = float(np.mean(aa_seps))
        stats.aa_sep_std = float(np.std(aa_seps, ddof=1) if len(aa_seps) > 1 else 0.0)
        stats.aa_n = len(aa_seps)

    if ab_seps:
        stats.ab_sep_mean = float(np.mean(ab_seps))
        stats.ab_sep_std = float(np.std(ab_seps, ddof=1) if len(ab_seps) > 1 else 0.0)
        stats.ab_n = len(ab_seps)

    if mean_seps:
        stats.mean_sep_mean = float(np.mean(mean_seps))
        stats.mean_sep_std = float(
            np.std(mean_seps, ddof=1) if len(mean_seps) > 1 else 0.0
        )
        stats.mean_sep_n = len(mean_seps)

    if mean_zs:
        stats.mean_z_mean = float(np.mean(mean_zs))
        stats.mean_z_std = float(np.std(mean_zs, ddof=1) if len(mean_zs) > 1 else 0.0)
        stats.mean_z_n = len(mean_zs)

    if corrs:
        stats.corr_mean = float(np.mean(corrs))
        stats.corr_std = float(np.std(corrs, ddof=1) if len(corrs) > 1 else 0.0)
        stats.corr_n = len(corrs)

    if dw_widths:
        stats.dw_mean = float(np.mean(dw_widths))
        stats.dw_std = float(np.std(dw_widths, ddof=1) if len(dw_widths) > 1 else 0.0)
        stats.dw_n = len(dw_widths)

    if max_disps:
        stats.max_disp_mean = float(np.mean(max_disps))
        stats.max_disp_std = float(
            np.std(max_disps, ddof=1) if len(max_disps) > 1 else 0.0
        )
        stats.max_disp_n = len(max_disps)

    if local_twists:
        stats.lt_mean = float(np.mean(local_twists))
        stats.lt_std = float(np.std(local_twists, ddof=1) if len(local_twists) > 1 else 0.0)
        stats.lt_n = len(local_twists)

    if A1_vals:
        stats.A1_mean = float(np.mean(A1_vals))
        stats.A1_std = float(np.std(A1_vals, ddof=1) if len(A1_vals) > 1 else 0.0)
        stats.A1_n = len(A1_vals)

    if D1_vals:
        stats.D1_mean = float(np.mean(D1_vals))
        stats.D1_std = float(np.std(D1_vals, ddof=1) if len(D1_vals) > 1 else 0.0)
        stats.D1_n = len(D1_vals)

    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_mean_std_vs_twist(
    thetas: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    ylabel: str,
    out_path: Path,
    color: str = "C0",
    marker: str = "o",
    reference_y_equals_x: bool = False,
    dpi: int = 150,
) -> None:
    """Single mean ± std curve vs twist angle; no title."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xlabel = r"Initial twist angle $\theta$ (°)"

    valid = np.isfinite(mean)
    if valid.any():
        t_v = thetas[valid]
        m_v = mean[valid]
        s_v = std[valid]
        ax.plot(t_v, m_v, f"{marker}-", color=color, label="ensemble mean")
        ax.fill_between(
            t_v, m_v - s_v, m_v + s_v,
            color=color, alpha=0.3, label="ensemble std",
        )
        if reference_y_equals_x:
            theta_range = np.array([float(t_v.min()), float(t_v.max())])
            ax.plot(
                theta_range, theta_range, "k--", linewidth=0.8, alpha=0.6,
                label=r"$\theta_\mathrm{local} = \theta_\mathrm{initial}$",
            )

    ax.set_xlabel(xlabel, fontdict=CSFONT)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.legend(
        loc="best",
        prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE},
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_aa_layer_sep_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """AA-site layer separation vs twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.aa_sep_mean for s in stats_list], dtype=float)
    std = np.array([s.aa_sep_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel="AA layer separation (Å)",
        out_path=out_path,
        color="C0",
        marker="o",
        dpi=dpi,
    )


def plot_ab_layer_sep_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """AB-site layer separation vs twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.ab_sep_mean for s in stats_list], dtype=float)
    std = np.array([s.ab_sep_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel="AB layer separation (Å)",
        out_path=out_path,
        color="C1",
        marker="s",
        dpi=dpi,
    )


def plot_mean_layer_sep_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Mean top-layer interlayer separation over the moiré cell vs twist."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.mean_sep_mean for s in stats_list], dtype=float)
    std = np.array([s.mean_sep_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel="mean interlayer separation (Å)",
        out_path=out_path,
        color="C5",
        marker="P",
        dpi=dpi,
    )


def plot_mean_z_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Mean Cartesian *z* (all atoms) vs twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.mean_z_mean for s in stats_list], dtype=float)
    std = np.array([s.mean_z_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel=r"mean $z$ (Å)",
        out_path=out_path,
        color="C4",
        marker="o",
        dpi=dpi,
    )


def plot_corrugation_amplitude_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Corrugation amplitude (AA − AB layer sep) vs twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.corr_mean for s in stats_list], dtype=float)
    std = np.array([s.corr_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel="corrugation amplitude (Å)",
        out_path=out_path,
        color="C5",
        marker="^",
        dpi=dpi,
    )


def plot_dw_width_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Domain wall width vs twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.dw_mean for s in stats_list], dtype=float)
    std = np.array([s.dw_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel="domain wall width (Å)",
        out_path=out_path,
        color="C3",
        marker="D",
        dpi=dpi,
    )


def plot_max_intralayer_disp_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Largest fitted displacement peak vs twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.max_disp_mean for s in stats_list], dtype=float)
    std = np.array([s.max_disp_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel=r"max intralayer disp. mag. (Å)",
        out_path=out_path,
        color="C6",
        marker="v",
        dpi=dpi,
    )


def plot_local_twist_vs_theta(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Local twist angle at AA sites vs initial twist angle (mean ± std)."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.lt_mean for s in stats_list], dtype=float)
    std = np.array([s.lt_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel="twist angle at AA stacking (°)",
        out_path=out_path,
        color="C2",
        marker="o",
        reference_y_equals_x=True,
        dpi=dpi,
    )


def plot_elastic_inplane_A_mode1_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Top-layer elastic-plate in-plane coefficient ``A`` (mode 1) vs twist."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.A1_mean for s in stats_list], dtype=float)
    std = np.array([s.A1_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel=r"in-plane $A$ (mode 1) (Å)",
        out_path=out_path,
        color="C8",
        marker="o",
        dpi=dpi,
    )


def plot_elastic_outplane_D_mode1_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Top-layer elastic-plate out-of-plane coefficient ``D`` (mode 1) vs twist."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    mean = np.array([s.D1_mean for s in stats_list], dtype=float)
    std = np.array([s.D1_std for s in stats_list], dtype=float)
    _plot_mean_std_vs_twist(
        thetas, mean, std,
        ylabel=r"out-of-plane $D$ (mode 1) (Å)",
        out_path=out_path,
        color="C9",
        marker="s",
        dpi=dpi,
    )


def cell_vector_lengths(atoms) -> Tuple[float, float]:
    """Return ``(|a₁|, |a₂|)`` from the first two lattice vectors (Å)."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    return float(np.linalg.norm(cell[0])), float(np.linalg.norm(cell[1]))


def collect_cell_lengths_first_sample(
    angle_groups: List[EnsembleGroup],
    *,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lattice-vector lengths from the **first** trajectory at each twist angle
    that passes the fmax gate.

    The cell is static across ensemble samples, so only one sample is needed.
    Uses the initial frame of that trajectory.

    Returns
    -------
    thetas, a1_len, a2_len : 1-D arrays (sorted by twist angle)
    """
    rows: List[Tuple[float, float, float]] = []
    for grp in angle_groups:
        paths = list(grp.trajectory_paths)
        if not paths:
            continue
        chosen: Optional[Path] = None
        for traj_path in paths:
            try:
                _initial, relaxed = read_both_frames(traj_path)
            except Exception as exc:
                print(f"    skip {traj_path.name}: {exc}", file=sys.stderr)
                continue
            if _passes_fmax_gate(relaxed, traj_path, fmax_max=fmax_max):
                chosen = traj_path
                break
        if chosen is None:
            print(
                f"    cell lengths: no sample with forces ≤ {fmax_max:g} eV/Å "
                f"at θ={grp.twist_angle:g}°",
                file=sys.stderr,
            )
            continue
        try:
            import ase.io

            frames = ase.io.read(str(chosen), index=":")
            if not isinstance(frames, list):
                frames = [frames]
            if len(frames) < 1:
                raise ValueError(f"{chosen.name}: empty trajectory")
            atoms = frames[0]
            a1, a2 = cell_vector_lengths(atoms)
        except Exception as exc:
            print(
                f"    cell lengths failed θ={grp.twist_angle:g}° "
                f"({chosen.name}): {exc}",
                file=sys.stderr,
            )
            continue
        rows.append((float(grp.twist_angle), a1, a2))

    if not rows:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    rows.sort(key=lambda r: r[0])
    thetas = np.array([r[0] for r in rows], dtype=float)
    a1_len = np.array([r[1] for r in rows], dtype=float)
    a2_len = np.array([r[2] for r in rows], dtype=float)
    return thetas, a1_len, a2_len


def plot_cell_vector_lengths_vs_twist(
    angle_groups: List[EnsembleGroup],
    out_path: Path,
    dpi: int = 150,
    *,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> None:
    """Plot ``|a₁|`` and ``|a₂|`` vs twist angle (first gated sample per angle)."""
    thetas, a1_len, a2_len = collect_cell_lengths_first_sample(
        angle_groups, fmax_max=fmax_max,
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xlabel = r"Initial twist angle $\theta$ (°)"

    if thetas.size:
        ax.plot(
            thetas, a1_len, "o-", color="C0",
            label=r"$|\mathbf{a}_1|$",
        )
        ax.plot(
            thetas, a2_len, "s--", color="C1",
            label=r"$|\mathbf{a}_2|$",
        )

    ax.set_xlabel(xlabel, fontdict=CSFONT)
    ax.set_ylabel("lattice vector length (Å)", fontdict=CSFONT)
    ax.legend(
        loc="best",
        prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE},
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        nargs="+",
        default=[str(DEFAULT_TRAJ_ROOT)],
        metavar="PATH",
        help="Trajectory root(s) or glob(s). May be the relaxation root, a "
        "model dir, a T* dir, theta*deg dirs, or a pattern such as "
        ".../T0.65/theta* (shell-expanded paths also work). "
        f"Default: {DEFAULT_TRAJ_ROOT}.",
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
        "--no-layer-sep",
        action="store_true",
        help="Skip AA/AB layer separation figures.",
    )
    p.add_argument(
        "--no-mean-layer-sep",
        action="store_true",
        help="Skip mean top-layer interlayer separation vs twist angle figure.",
    )
    p.add_argument(
        "--no-mean-z",
        action="store_true",
        help="Skip mean-z vs twist angle figure.",
    )
    p.add_argument(
        "--no-corrugation",
        action="store_true",
        help="Skip corrugation amplitude (AA − AB) figure.",
    )
    p.add_argument(
        "--no-dw-width",
        action="store_true",
        help="Skip domain wall width figure.",
    )
    p.add_argument(
        "--no-max-intralayer-disp",
        action="store_true",
        help="Skip max intralayer displacement magnitude figure.",
    )
    p.add_argument(
        "--no-local-twist",
        action="store_true",
        help="Skip local twist angle figure.",
    )
    p.add_argument(
        "--no-cell-lengths",
        action="store_true",
        help="Skip cell vector length (|a1|, |a2|) vs twist angle figure.",
    )
    p.add_argument(
        "--no-elastic-basis",
        action="store_true",
        help=(
            "Skip elastic-plate Fourier coefficient figures "
            "(top-layer mode-1 A and D)."
        ),
    )
    p.add_argument(
        "--npoints",
        type=int,
        default=DEFAULT_NPOINTS,
        help=(
            "Number of points along the moiré diagonal for the intralayer "
            f"displacement cross section / DW width (default: {DEFAULT_NPOINTS})."
        ),
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
        f"Found {len(groups)} ensemble group(s) from {len(roots)} path(s) across "
        f"{len({g.twist_angle for g in groups})} twist angle(s) "
        f"(fmax≤{args.fmax_max:g} eV/Å).",
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
                npoints=args.npoints,
                fmax_max=args.fmax_max,
            )
            stats_list.append(st)
            print(
                f"    n_atoms={st.n_atoms}  "
                f"AA sep={st.aa_sep_mean:.3f}±{st.aa_sep_std:.3f} Å (n={st.aa_n})  "
                f"AB sep={st.ab_sep_mean:.3f}±{st.ab_sep_std:.3f} Å (n={st.ab_n})  "
                f"⟨sep⟩={st.mean_sep_mean:.3f}±{st.mean_sep_std:.3f} Å "
                f"(n={st.mean_sep_n})  "
                f"⟨z⟩={st.mean_z_mean:.3f}±{st.mean_z_std:.3f} Å (n={st.mean_z_n})  "
                f"corr={st.corr_mean:.3f}±{st.corr_std:.3f} Å (n={st.corr_n})  "
                f"DW={st.dw_mean:.3f}±{st.dw_std:.3f} Å (n={st.dw_n})  "
                f"max‖Δr‖={st.max_disp_mean:.3f}±{st.max_disp_std:.3f} Å "
                f"(n={st.max_disp_n})  "
                f"local θ={st.lt_mean:.3f}±{st.lt_std:.3f}° (n={st.lt_n})  "
                f"A₁={st.A1_mean:.4g}±{st.A1_std:.4g} (n={st.A1_n})  "
                f"D₁={st.D1_mean:.4g}±{st.D1_std:.4g} (n={st.D1_n})",
                flush=True,
            )

        if not stats_list:
            continue

        # Save figures next to the theta dirs: .../<model>/T<label>/
        out_dir = angle_groups[0].directory.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_layer_sep:
            plot_aa_layer_sep_vs_twist(
                stats_list,
                out_path=out_dir / "aa_layer_sep_vs_twist_angle.png",
                dpi=args.dpi,
            )
            plot_ab_layer_sep_vs_twist(
                stats_list,
                out_path=out_dir / "ab_layer_sep_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_mean_layer_sep:
            plot_mean_layer_sep_vs_twist(
                stats_list,
                out_path=out_dir / "mean_layer_sep_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_mean_z:
            plot_mean_z_vs_twist(
                stats_list,
                out_path=out_dir / "mean_z_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_corrugation:
            plot_corrugation_amplitude_vs_twist(
                stats_list,
                out_path=out_dir / "corrugation_amplitude_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_dw_width:
            plot_dw_width_vs_twist(
                stats_list,
                out_path=out_dir / "dw_width_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_max_intralayer_disp:
            plot_max_intralayer_disp_vs_twist(
                stats_list,
                out_path=out_dir / "max_intralayer_disp_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_local_twist:
            plot_local_twist_vs_theta(
                stats_list,
                out_path=out_dir / "local_twist_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_cell_lengths:
            plot_cell_vector_lengths_vs_twist(
                angle_groups,
                out_path=out_dir / "cell_vector_lengths_vs_twist_angle.png",
                dpi=args.dpi,
                fmax_max=args.fmax_max,
            )

        if not args.no_elastic_basis:
            plot_elastic_inplane_A_mode1_vs_twist(
                stats_list,
                out_path=out_dir / "elastic_inplane_A_mode1_vs_twist_angle.png",
                dpi=args.dpi,
            )
            plot_elastic_outplane_D_mode1_vs_twist(
                stats_list,
                out_path=out_dir / "elastic_outplane_D_mode1_vs_twist_angle.png",
                dpi=args.dpi,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
