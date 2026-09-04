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
  separation over the moiré cell (ARPES experiment overlaid when
  ``data/arpes_extracted_mean_layer_sep_critical_role_lattice_relaxations.csv``
  is present)
* ``mean_z_vs_twist_angle.png`` — mean Cartesian *z* over all atoms
* ``corrugation_amplitude_vs_twist_angle.png`` — AA − AB top-layer
  separation (corrugation amplitude)
* ``dw_width_vs_twist_angle.png`` — domain wall width from the
  intralayer-displacement cross section
* ``max_intralayer_disp_vs_twist_angle.png`` — largest fitted in-plane
  displacement peak on the moiré cross section
* ``local_twist_vs_twist_angle.png`` — local twist angle at AA stacking
* ``rel_uncertainty_vs_twist_angle.png`` — relative uncertainty vs twist
  for AA/AB layer separation (``σ/|μ − d_eq|`` with primitive AA/AB
  bilayer equilibria), local twist at AA (``σ/|μ − θ_initial|``),
  domain wall width, and max intralayer displacement (same colors as
  the individual mean±std figures)
* ``cell_vector_lengths_vs_twist_angle.png`` — lengths of cell vectors
  ``|a₁|`` and ``|a₂|`` (first sample only; cell is fixed across the ensemble)
* ``elastic_basis_A123_vs_twist_angle.png`` — in-plane sin coefficients
  ``A₁, A₂, A₃`` (TEM diffraction PLD overlaid on ``A₁`` when xlsx present)
* ``elastic_basis_A456_vs_twist_angle.png`` — in-plane cos coefficients
  ``A₄, A₅, A₆``
* ``elastic_basis_A789_vs_twist_angle.png`` — out-of-plane sin coefficients
  ``A₇, A₈, A₉``
* ``elastic_basis_A101112_vs_twist_angle.png`` — out-of-plane cos coefficients
  ``A₁₀, A₁₁, A₁₂``

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

Elastic plate Fourier coefficients (modes 1–3, top layer)
---------------------------------------------------------
Displacements ``u = r_relaxed − r_initial`` on top-layer atoms
(``z_initial > mean(z_initial)``) are projected onto the continuum elastic
plate Fourier basis (see ``elastic_plate_basis.py``, ported from
``Elastic_basis_Dan``).  Twelve coefficients ``A₁…A₁₂`` are reported vs twist
(ensemble mean ± std), grouped into four figures:

* ``A₁, A₂, A₃`` — in-plane sin (modes 1–3)
* ``A₄, A₅, A₆`` — in-plane cos (modes 1–3)
* ``A₇, A₈, A₉`` — out-of-plane sin (modes 1–3)
* ``A₁₀, A₁₁, A₁₂`` — out-of-plane cos (modes 1–3)

TEM diffraction PLD data are overlaid on the ``A₁, A₂, A₃`` figure only.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
REPO_ROOT = UQ_DIR.parent
DEFAULT_TRAJ_ROOT = UQ_DIR / "trajectories" / "relaxation"
DEFAULT_TEM_PLD_XLSX = (
    REPO_ROOT / "data" / "pld_amp_data_Torsional_periodic_lattice_distortions.xlsx"
)
DEFAULT_ARPES_MEAN_SEP_CSV = (
    REPO_ROOT
    / "data"
    / "arpes_extracted_mean_layer_sep_critical_role_lattice_relaxations.csv"
)
PM_TO_ANG = 0.01  # 1 pm = 0.01 Å

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
    read_initial_and_relaxed_frames,
)
from elastic_plate_basis import top_layer_elastic_coeffs  # noqa: E402

N_ELASTIC_COEFFS = 12
_ELASTIC_GROUP_SPECS: Tuple[Tuple[str, Tuple[int, ...], str], ...] = (
    ("elastic_basis_A123_vs_twist_angle", (1, 2, 3), r"$A_i$ (Å)"),
    ("elastic_basis_A456_vs_twist_angle", (4, 5, 6), r"$A_i$ (Å)"),
    ("elastic_basis_A789_vs_twist_angle", (7, 8, 9), r"$A_i$ (Å)"),
    ("elastic_basis_A101112_vs_twist_angle", (10, 11, 12), r"$A_i$ (Å)"),
)

DEFAULT_NN_CUT: float = 1.65        # Å — in-plane NN cut-off for twist angle
DISP_PEAK_FIT_HALF_WIDTH: float = 8.0  # Å around each section maximum

# Primitive untwisted bilayer equilibrium interlayer separations (Å).
# From ``bilayer_graphene_elastic_constants_structures.xyz`` ``d0`` labels
# used by ``plot_elastic_moduli.py`` / ``calc_elastic_constants.py``.
AA_EQ_LAYER_SEP: float = 3.577
AB_EQ_LAYER_SEP: float = 3.3909


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

    # Elastic plate Fourier coeffs A1..A12 (top layer), Å
    elastic_mean: np.ndarray = field(
        default_factory=lambda: np.full(N_ELASTIC_COEFFS, np.nan),
    )
    elastic_std: np.ndarray = field(
        default_factory=lambda: np.full(N_ELASTIC_COEFFS, np.nan),
    )
    elastic_n: int = 0

    # Cell vector lengths from the first gated sample (Å; cell is fixed)
    a1_len: float = np.nan
    a2_len: float = np.nan

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
# Layer split / stacking / separations (single pass over positions)
# ---------------------------------------------------------------------------

def _split_layers(pos: np.ndarray):
    """Return boolean masks for top and bottom layers."""
    z_mean = float(np.mean(pos[:, 2]))
    top_mask = pos[:, 2] > z_mean
    bot_mask = pos[:, 2] < z_mean
    return top_mask, bot_mask, z_mean


def identify_stacking_atoms(relaxed_atoms) -> Tuple[int, int]:
    """Return the AA and AB representative atom indices from the relaxed frame.

    AA site: the top-layer atom with the **largest** interlayer separation
    ``2*(z_i − mean(z))``.  In relaxed TBLG the layers buckle outward at AA
    sites, so these atoms sit highest.

    AB site: the top-layer atom with the **smallest** interlayer separation.
    AB/SP hollow sites are pulled inward and sit lowest.
    """
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    top_mask, _, z_mean = _split_layers(pos)
    top_idx = np.where(top_mask)[0]
    sep = 2.0 * (pos[top_idx, 2] - z_mean)
    aa_idx = int(top_idx[np.argmax(sep)])
    ab_idx = int(top_idx[np.argmin(sep)])
    return aa_idx, ab_idx


def relaxed_layer_metrics(relaxed_atoms) -> Dict:
    """AA/AB sep, mean sep, mean z, corrugation, indices, and atom count."""
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    top_mask, _, z_mean = _split_layers(pos)
    top_idx = np.where(top_mask)[0]
    if top_idx.size < 1:
        raise ValueError("no top-layer atoms found")
    sep_top = 2.0 * (pos[top_idx, 2] - z_mean)
    aa_local = int(np.argmax(sep_top))
    ab_local = int(np.argmin(sep_top))
    aa_sep = float(sep_top[aa_local])
    ab_sep = float(sep_top[ab_local])
    return {
        "aa_idx": int(top_idx[aa_local]),
        "ab_idx": int(top_idx[ab_local]),
        "aa_sep": aa_sep,
        "ab_sep": ab_sep,
        "mean_sep": float(np.mean(np.abs(sep_top))),
        "mean_z": z_mean,
        "corrugation": aa_sep - ab_sep,
        "n_atoms": int(pos.shape[0]),
    }


def layer_sep_for_indices(relaxed_atoms, atom_idx: np.ndarray) -> np.ndarray:
    """Compute ``2*(z[atom_idx] - mean(z))`` (Å) for *atom_idx* atoms."""
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    z_mean = float(np.mean(pos[:, 2]))
    return 2.0 * (pos[atom_idx, 2] - z_mean)


def mean_z_all_atoms(relaxed_atoms) -> float:
    """Mean Cartesian *z* (Å) over all atoms in the relaxed structure."""
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    return float(np.mean(pos[:, 2]))


def mean_top_layer_interlayer_sep(relaxed_atoms) -> float:
    """Mean interlayer separation (Å) over top-layer atoms in the moiré cell."""
    pos = np.asarray(relaxed_atoms.get_positions(wrap=False), dtype=float)
    top_mask, _, z_mean = _split_layers(pos)
    if int(np.count_nonzero(top_mask)) < 1:
        return float("nan")
    sep = 2.0 * np.abs(pos[top_mask, 2] - z_mean)
    return float(np.mean(sep))


def cell_vector_lengths(atoms) -> Tuple[float, float]:
    """Return ``(|a₁|, |a₂|)`` from the first two lattice vectors (Å)."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    return float(np.linalg.norm(cell[0])), float(np.linalg.norm(cell[1]))


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
    """Return ``(initial_atoms, relaxed_atoms)`` without loading intermediate frames."""
    return read_initial_and_relaxed_frames(traj_path)


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
    dw_width, max_intralayer_disp, local_twist, A_mode1, D_mode1, a1_len,
    a2_len, n_atoms``, or None on failure.

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
        layer = relaxed_layer_metrics(relaxed)
    except Exception as exc:
        print(f"    stacking id failed {traj_path.name}: {exc}", file=sys.stderr)
        return None

    a1_len, a2_len = cell_vector_lengths(initial)
    result: Dict = {
        "aa_sep": layer["aa_sep"],
        "ab_sep": layer["ab_sep"],
        "mean_sep": layer["mean_sep"],
        "mean_z": layer["mean_z"],
        "corrugation": layer["corrugation"],
        "n_atoms": layer["n_atoms"],
        "a1_len": a1_len,
        "a2_len": a2_len,
        **{f"elastic_A{i}": float("nan") for i in range(1, N_ELASTIC_COEFFS + 1)},
        "dw_width": float("nan"),
        "max_intralayer_disp": float("nan"),
        "local_twist": float("nan"),
    }

    try:
        path_len, disp = get_intralayer_displacement_cross_sect(
            initial, relaxed, npoints=npoints,
        )
        dw_width, max_disp = displacement_metrics_from_peak_fits(path_len, disp)
        result["dw_width"] = dw_width
        result["max_intralayer_disp"] = max_disp
    except Exception as exc:
        print(f"    displacement peak fits failed {traj_path.name}: {exc}", file=sys.stderr)

    try:
        ltwists = local_twist_at_aa_sites(
            initial, relaxed, np.array([layer["aa_idx"]]), theta_deg, nn_cut=nn_cut,
        )
        if len(ltwists) > 0:
            result["local_twist"] = float(np.mean(ltwists))
    except Exception as exc:
        print(f"    twist calc failed {traj_path.name}: {exc}", file=sys.stderr)

    try:
        coeffs = top_layer_elastic_coeffs(initial, relaxed, num_mode=3)
        for i, val in enumerate(coeffs, start=1):
            result[f"elastic_A{i}"] = float(val)
    except Exception as exc:
        print(f"    elastic basis failed {traj_path.name}: {exc}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Ensemble-level aggregation
# ---------------------------------------------------------------------------

def _values_within_n_std(values: Sequence[float], n_std: float = 5.0) -> List[float]:
    """Drop entries more than ``n_std`` sample stds from the mean."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return [float(x) for x in arr]
    mu = float(np.mean(arr))
    sig = float(np.std(arr, ddof=1))
    if not np.isfinite(sig) or sig <= 0.0:
        return [float(x) for x in arr]
    keep = np.abs(arr - mu) <= n_std * sig
    return [float(x) for x in arr[keep]]


def _mean_std_n(values: Sequence[float]) -> Tuple[float, float, int]:
    """Return ``(mean, std, n)``; NaNs if empty. ``std=0`` when ``n==1``."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return mean, std, n


_SAMPLE_METRIC_KEYS: Tuple[str, ...] = (
    "aa_sep",
    "ab_sep",
    "mean_sep",
    "mean_z",
    "corrugation",
    "dw_width",
    "max_intralayer_disp",
    "local_twist",
    *(f"elastic_A{i}" for i in range(1, N_ELASTIC_COEFFS + 1)),
)

# AngleStats field prefixes for each sample-metric key
_STATS_FIELD_PREFIX: Dict[str, str] = {
    "aa_sep": "aa_sep",
    "ab_sep": "ab_sep",
    "mean_sep": "mean_sep",
    "mean_z": "mean_z",
    "corrugation": "corr",
    "dw_width": "dw",
    "max_intralayer_disp": "max_disp",
    "local_twist": "lt",
}

_OUTLIER_CLIP_KEYS = frozenset({"dw_width", "max_intralayer_disp", "local_twist"})


def compute_angle_stats(
    traj_paths: List[Path],
    theta_deg: float,
    nn_cut: float,
    *,
    npoints: int = DEFAULT_NPOINTS,
    fmax_max: float = DEFAULT_FMAX_MAX,
) -> AngleStats:
    """Collect per-sample results and compute ensemble mean ± std.

    Each trajectory is read once (initial + final frames only).
    """
    buckets: Dict[str, List[float]] = {k: [] for k in _SAMPLE_METRIC_KEYS}
    n_atoms = 0
    a1_len = float("nan")
    a2_len = float("nan")

    for tp in traj_paths:
        res = process_sample(
            tp, theta_deg, nn_cut, npoints=npoints, fmax_max=fmax_max,
        )
        if res is None:
            continue
        if n_atoms <= 0 and int(res.get("n_atoms", 0)) > 0:
            n_atoms = int(res["n_atoms"])
        if not np.isfinite(a1_len):
            a1_len = float(res.get("a1_len", np.nan))
            a2_len = float(res.get("a2_len", np.nan))
        for key in _SAMPLE_METRIC_KEYS:
            val = res.get(key, np.nan)
            if np.isfinite(val):
                buckets[key].append(float(val))

    for key in _OUTLIER_CLIP_KEYS:
        buckets[key] = _values_within_n_std(buckets[key])

    stats = AngleStats(
        theta=theta_deg,
        n_atoms=int(n_atoms),
        a1_len=a1_len,
        a2_len=a2_len,
    )
    for key, prefix in _STATS_FIELD_PREFIX.items():
        mean, std, n = _mean_std_n(buckets[key])
        setattr(stats, f"{prefix}_mean", mean)
        setattr(stats, f"{prefix}_std", std)
        setattr(stats, f"{prefix}_n", n)

    elastic_mean = np.full(N_ELASTIC_COEFFS, np.nan, dtype=float)
    elastic_std = np.full(N_ELASTIC_COEFFS, np.nan, dtype=float)
    elastic_n = 0
    for i in range(1, N_ELASTIC_COEFFS + 1):
        mean, std, n = _mean_std_n(buckets[f"elastic_A{i}"])
        elastic_mean[i - 1] = mean
        elastic_std[i - 1] = std
        elastic_n = max(elastic_n, n)
    stats.elastic_mean = elastic_mean
    stats.elastic_std = elastic_std
    stats.elastic_n = elastic_n
    return stats


# ---------------------------------------------------------------------------
# TEM diffraction PLD (experimental A₁)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemPldA1Point:
    """TEM diffraction PLD in-plane mode-1 amplitude at one twist angle."""
    theta_deg: float
    A1_angstrom: float
    yerr_angstrom: float


def load_tem_pld_A1_data(path: Path) -> List[TemPldA1Point]:
    """
    Load TEM PLD ``A₁`` vs twist from ``pld_amp_data_*.xlsx``.

    Columns: twist angle (degrees), PLD amplitude A1 (picometers),
    errorbar (picometers).  Values are converted to Å for plotting alongside
    the elastic-basis ``A`` coefficient.
    """
    if not path.is_file():
        print(f"  TEM PLD xlsx not found: {path}", file=sys.stderr)
        return []

    try:
        import pandas as pd
    except ImportError:
        print("  TEM PLD: pandas required to read xlsx", file=sys.stderr)
        return []

    try:
        df = pd.read_excel(path)
    except ImportError as exc:
        print(
            f"  TEM PLD: could not read {path.name} ({exc}); "
            "install openpyxl to load xlsx.",
            file=sys.stderr,
        )
        return []
    except Exception as exc:
        print(f"  TEM PLD: failed to read {path}: {exc}", file=sys.stderr)
        return []

    # Normalise column names (strip whitespace).
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    theta_col = "twist angle (degrees)"
    amp_col = "PLD amplitude, A1 (picometers)"
    err_col = "errorbar (picometers)"

    points: List[TemPldA1Point] = []
    for _, row in df.iterrows():
        try:
            theta = float(row[theta_col])
            amp_pm = float(row[amp_col])
            err_pm = float(row[err_col])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  skip TEM PLD row: {exc}", file=sys.stderr)
            continue
        if not np.isfinite(err_pm) or err_pm < 0.0:
            print(f"  skip TEM PLD θ={theta:g}°: invalid errorbar", file=sys.stderr)
            continue
        points.append(
            TemPldA1Point(
                theta_deg=theta,
                A1_angstrom=amp_pm * PM_TO_ANG,
                yerr_angstrom=err_pm * PM_TO_ANG,
            )
        )
    return sorted(points, key=lambda p: p.theta_deg)


def overlay_tem_pld_A1(
    ax: plt.Axes,
    tem_points: Sequence[TemPldA1Point],
) -> None:
    """Overlay TEM diffraction PLD ``A₁`` with error bars on *ax*."""
    if not tem_points:
        return
    thetas = np.array([p.theta_deg for p in tem_points], dtype=float)
    y = np.array([p.A1_angstrom for p in tem_points], dtype=float)
    yerr = np.array([p.yerr_angstrom for p in tem_points], dtype=float)
    ax.errorbar(
        thetas,
        y,
        yerr=yerr,
        fmt="D-",
        color="C2",
        ms=7,
        lw=1.8,
        capsize=4,
        label="TEM diffraction",
        zorder=4,
    )


# ---------------------------------------------------------------------------
# ARPES mean interlayer separation (experiment)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArpesMeanLayerSepPoint:
    """ARPES-extracted mean interlayer separation at one twist angle."""
    theta_deg: float
    sep_angstrom: float
    yerr_lower: float
    yerr_upper: float


def load_arpes_mean_layer_sep_data(path: Path) -> List[ArpesMeanLayerSepPoint]:
    """
    Load ARPES mean layer separation vs twist from CSV.

    Columns: twist angle, mean layer separation (angstroms),
    errorbar max (angstroms), errorbar min (angstroms).
    """
    import csv

    if not path.is_file():
        print(f"  ARPES mean layer sep CSV not found: {path}", file=sys.stderr)
        return []

    points: List[ArpesMeanLayerSepPoint] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                theta = float(str(row["twist angle"]).strip())
                sep = float(str(row["mean layer separation (angstroms)"]).strip())
                hi = float(str(row["errorbar max (angstroms)"]).strip())
                lo = float(str(row["errorbar min (angstroms)"]).strip())
            except (KeyError, TypeError, ValueError) as exc:
                print(f"  skip ARPES mean sep row: {exc}", file=sys.stderr)
                continue
            yerr_lo = sep - lo
            yerr_hi = hi - sep
            if not all(np.isfinite([theta, sep, yerr_lo, yerr_hi])):
                print(f"  skip ARPES θ={theta:g}°: non-finite values", file=sys.stderr)
                continue
            if yerr_lo < 0.0 or yerr_hi < 0.0:
                print(f"  skip ARPES θ={theta:g}°: invalid error bars", file=sys.stderr)
                continue
            points.append(
                ArpesMeanLayerSepPoint(
                    theta_deg=theta,
                    sep_angstrom=sep,
                    yerr_lower=float(yerr_lo),
                    yerr_upper=float(yerr_hi),
                )
            )
    return sorted(points, key=lambda p: p.theta_deg)


def overlay_arpes_mean_layer_sep(
    ax: plt.Axes,
    arpes_points: Sequence[ArpesMeanLayerSepPoint],
) -> None:
    """Overlay ARPES mean interlayer separation with asymmetric error bars."""
    if not arpes_points:
        return
    thetas = np.array([p.theta_deg for p in arpes_points], dtype=float)
    y = np.array([p.sep_angstrom for p in arpes_points], dtype=float)
    yerr = np.array(
        [
            [p.yerr_lower for p in arpes_points],
            [p.yerr_upper for p in arpes_points],
        ],
        dtype=float,
    )
    ax.errorbar(
        thetas,
        y,
        yerr=yerr,
        fmt="^-",
        color="C2",
        ms=7,
        lw=1.8,
        capsize=4,
        label="experiment",
        zorder=4,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# (filename_stem, AngleStats mean attr, std attr, ylabel, color, marker, y=x ref)
_MEAN_STD_PLOT_SPECS: Tuple[Tuple[str, str, str, str, str, str, bool], ...] = (
    ("aa_layer_sep_vs_twist_angle", "aa_sep_mean", "aa_sep_std",
     "AA layer separation (Å)", "C0", "o", False),
    ("ab_layer_sep_vs_twist_angle", "ab_sep_mean", "ab_sep_std",
     "AB layer separation (Å)", "C1", "s", False),
    ("mean_layer_sep_vs_twist_angle", "mean_sep_mean", "mean_sep_std",
     "mean interlayer separation (Å)", "C5", "P", False),
    ("mean_z_vs_twist_angle", "mean_z_mean", "mean_z_std",
     r"mean $z$ (Å)", "C4", "o", False),
    ("corrugation_amplitude_vs_twist_angle", "corr_mean", "corr_std",
     "corrugation amplitude (Å)", "C5", "^", False),
    ("dw_width_vs_twist_angle", "dw_mean", "dw_std",
     "domain wall width (Å)", "C3", "D", False),
    ("max_intralayer_disp_vs_twist_angle", "max_disp_mean", "max_disp_std",
     r"max intralayer disp. mag. (Å)", "C6", "v", False),
    ("local_twist_vs_twist_angle", "lt_mean", "lt_std",
     "twist angle at AA stacking (°)", "C2", "o", True),
)


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
    arpes_mean_sep_points: Optional[Sequence[ArpesMeanLayerSepPoint]] = None,
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

    if arpes_mean_sep_points:
        overlay_arpes_mean_layer_sep(ax, arpes_mean_sep_points)
    if arpes_mean_sep_points:
        ax.legend(prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE})

    ax.set_xlabel(xlabel, fontdict=CSFONT)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def write_mean_std_figures(
    stats_list: List[AngleStats],
    out_dir: Path,
    *,
    dpi: int = 150,
    enabled: Optional[Dict[str, bool]] = None,
    arpes_mean_sep_points: Optional[Sequence[ArpesMeanLayerSepPoint]] = None,
) -> None:
    """Write all standard mean±std vs-twist figures (skips disabled stems)."""
    enabled = enabled or {}
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    for stem, mean_attr, std_attr, ylabel, color, marker, yeqx in _MEAN_STD_PLOT_SPECS:
        if not enabled.get(stem, True):
            continue
        mean = np.array([getattr(s, mean_attr) for s in stats_list], dtype=float)
        std = np.array([getattr(s, std_attr) for s in stats_list], dtype=float)
        _plot_mean_std_vs_twist(
            thetas, mean, std,
            ylabel=ylabel,
            out_path=out_dir / f"{stem}.png",
            color=color,
            marker=marker,
            reference_y_equals_x=yeqx,
            dpi=dpi,
            arpes_mean_sep_points=(
                arpes_mean_sep_points
                if stem == "mean_layer_sep_vs_twist_angle"
                else None
            ),
        )


def _plot_elastic_group_vs_twist(
    stats_list: List[AngleStats],
    coeff_indices: Sequence[int],
    *,
    ylabel: str,
    out_path: Path,
    dpi: int = 150,
    tem_pld_points: Optional[Sequence[TemPldA1Point]] = None,
) -> None:
    """Plot mean ± std for a group of elastic coefficients ``A_i`` vs twist."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    colors = ("C0", "C1", "C8")
    markers = ("o", "s", "^")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xlabel = r"Initial twist angle $\theta$ (°)"
    has_curve = False

    for idx, color, marker in zip(coeff_indices, colors, markers):
        mean = np.array(
            [s.elastic_mean[idx - 1] for s in stats_list], dtype=float,
        )
        std = np.array(
            [s.elastic_std[idx - 1] for s in stats_list], dtype=float,
        )
        valid = np.isfinite(mean)
        if not valid.any():
            continue
        has_curve = True
        t_v = thetas[valid]
        m_v = mean[valid]
        s_v = std[valid]
        ax.plot(
            t_v,
            m_v,
            f"{marker}-",
            color=color,
            label=rf"$A_{{{idx}}}$",
        )
        ax.fill_between(
            t_v,
            m_v - s_v,
            m_v + s_v,
            color=color,
            alpha=0.25,
        )

    if tem_pld_points and 1 in coeff_indices:
        overlay_tem_pld_A1(ax, tem_pld_points)

    if has_curve or tem_pld_points:
        ax.legend(prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE})

    ax.set_xlabel(xlabel, fontdict=CSFONT)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def write_elastic_basis_group_figures(
    stats_list: List[AngleStats],
    out_dir: Path,
    *,
    dpi: int = 150,
    tem_pld_points: Optional[Sequence[TemPldA1Point]] = None,
) -> None:
    """Write the four elastic-basis coefficient-group figures."""
    for stem, coeff_indices, ylabel in _ELASTIC_GROUP_SPECS:
        _plot_elastic_group_vs_twist(
            stats_list,
            coeff_indices,
            ylabel=ylabel,
            out_path=out_dir / f"{stem}.png",
            dpi=dpi,
            tem_pld_points=(
                tem_pld_points if coeff_indices == (1, 2, 3) else None
            ),
        )


def _rel_uncertainty(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    offset: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Relative uncertainty ``σ / |μ − offset|``; NaN where the denom is ~0."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    denom = mean - np.asarray(offset, dtype=float)
    out = np.full_like(mean, np.nan, dtype=float)
    ok = np.isfinite(mean) & np.isfinite(std) & np.isfinite(denom) & (np.abs(denom) > 1e-15)
    out[ok] = std[ok] / np.abs(denom[ok])
    return out


def plot_rel_uncertainty_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
    *,
    aa_eq_sep: float = AA_EQ_LAYER_SEP,
    ab_eq_sep: float = AB_EQ_LAYER_SEP,
) -> None:
    """Relative uncertainty vs twist for the main structural metrics.

    AA/AB layer separations use ``σ / |μ − d_eq|`` with the primitive
    untwisted bilayer equilibria ``aa_eq_sep`` / ``ab_eq_sep``.  Local
    twist at AA uses ``σ / |μ − θ_initial|``.  Other quantities use
    ``σ / |μ|``.

    Colors and markers match the individual mean±std figures:
    AA sep (C0), AB sep (C1), local twist (C2), DW width (C3),
    max intralayer disp (C6).
    """
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)

    # offset: scalar equilibrium / zero, or None → use θ_initial per point
    series = [
        ("AA layer separation", "aa_sep_mean", "aa_sep_std", "C0", "o", float(aa_eq_sep)),
        ("AB layer separation", "ab_sep_mean", "ab_sep_std", "C1", "s", float(ab_eq_sep)),
        ("twist angle at AA stacking", "lt_mean", "lt_std", "C2", "o", None),
        ("domain wall width", "dw_mean", "dw_std", "C3", "D", 0.0),
        ("max intralayer disp. mag.", "max_disp_mean", "max_disp_std", "C6", "v", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xlabel = r"Initial twist angle $\theta$ (°)"
    for label, mean_attr, std_attr, color, marker, offset in series:
        mean = np.array([getattr(s, mean_attr) for s in stats_list], dtype=float)
        std = np.array([getattr(s, std_attr) for s in stats_list], dtype=float)
        off = thetas if offset is None else offset
        rel = _rel_uncertainty(mean, std, offset=off)
        valid = np.isfinite(rel)
        if not valid.any():
            continue
        ax.plot(
            thetas[valid],
            rel[valid],
            f"{marker}-",
            color=color,
            label=label,
        )

    ax.set_xlabel(xlabel, fontdict=CSFONT)
    ax.set_ylabel(r"rel. uncertainty", fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_cell_vector_lengths_vs_twist(
    stats_list: List[AngleStats],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Plot ``|a₁|`` and ``|a₂|`` vs twist from lengths captured during stats."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list], dtype=float)
    a1_len = np.array([s.a1_len for s in stats_list], dtype=float)
    a2_len = np.array([s.a2_len for s in stats_list], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xlabel = r"Initial twist angle $\theta$ (°)"
    valid = np.isfinite(a1_len) & np.isfinite(a2_len)
    if valid.any():
        ax.plot(
            thetas[valid], a1_len[valid], "o-", color="C0",
            label=r"$|\mathbf{a}_1|$",
        )
        ax.plot(
            thetas[valid], a2_len[valid], "s--", color="C1",
            label=r"$|\mathbf{a}_2|$",
        )

    ax.set_xlabel(xlabel, fontdict=CSFONT)
    ax.set_ylabel("lattice vector length (Å)", fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
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
    p.add_argument(
        "--skip-theta",
        type=float,
        nargs="*",
        default=[1.05, 1.08],
        metavar="DEG",
        help="Twist angles (degrees) to omit from all figures. "
        "Default: 1.05 1.08. Pass with no values to skip none.",
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
        "--no-rel-uncertainty",
        action="store_true",
        help=(
            "Skip relative-uncertainty overlay vs twist angle "
            "(AA/AB sep, DW width, local twist, max intralayer disp)."
        ),
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
            "(top-layer A₁…A₁₂ grouped into four plots)."
        ),
    )
    p.add_argument(
        "--tem-pld-xlsx",
        type=Path,
        default=DEFAULT_TEM_PLD_XLSX,
        help=(
            "TEM diffraction PLD data for overlay on the "
            f"A₁,A₂,A₃ elastic-basis figure (default: {DEFAULT_TEM_PLD_XLSX.name})."
        ),
    )
    p.add_argument(
        "--no-tem-pld",
        action="store_true",
        help="Do not overlay TEM diffraction PLD data on the A₁,A₂,A₃ elastic-basis figure.",
    )
    p.add_argument(
        "--arpes-mean-sep-csv",
        type=Path,
        default=DEFAULT_ARPES_MEAN_SEP_CSV,
        help=(
            "ARPES mean interlayer separation vs twist for overlay on "
            f"mean_layer_sep figure (default: {DEFAULT_ARPES_MEAN_SEP_CSV.name})."
        ),
    )
    p.add_argument(
        "--no-arpes-mean-sep",
        action="store_true",
        help="Do not overlay ARPES mean layer separation on the mean sep figure.",
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

    skip_theta = [float(t) for t in (args.skip_theta or [])]

    tem_pld_path = Path(args.tem_pld_xlsx)
    if not tem_pld_path.is_absolute():
        tem_pld_path = REPO_ROOT / tem_pld_path
    tem_pld_points: Optional[List[TemPldA1Point]] = None
    if not args.no_tem_pld:
        tem_pld_points = load_tem_pld_A1_data(tem_pld_path)
        if tem_pld_points:
            print(
                f"Loaded {len(tem_pld_points)} TEM PLD A₁ point(s) from {tem_pld_path}",
                flush=True,
            )

    arpes_sep_path = Path(args.arpes_mean_sep_csv)
    if not arpes_sep_path.is_absolute():
        arpes_sep_path = REPO_ROOT / arpes_sep_path
    arpes_mean_sep_points: Optional[List[ArpesMeanLayerSepPoint]] = None
    if not args.no_arpes_mean_sep:
        arpes_mean_sep_points = load_arpes_mean_layer_sep_data(arpes_sep_path)
        if arpes_mean_sep_points:
            print(
                f"Loaded {len(arpes_mean_sep_points)} ARPES mean layer sep point(s) "
                f"from {arpes_sep_path}",
                flush=True,
            )

    def _skip_this_theta(theta: float) -> bool:
        return any(abs(float(theta) - s) < 1e-6 for s in skip_theta)

    n_before = len(groups)
    skipped = sorted({g.twist_angle for g in groups if _skip_this_theta(g.twist_angle)})
    groups = [g for g in groups if not _skip_this_theta(g.twist_angle)]
    if skipped:
        print(
            f"Skipping {n_before - len(groups)} ensemble group(s) at θ="
            + ", ".join(f"{t:g}°" for t in skipped)
            + ".",
            flush=True,
        )

    if not groups:
        p.error(
            "No TBLG relaxation ensembles remain after --skip-theta "
            f"{skip_theta!r}."
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
                f"A₁={st.elastic_mean[0]:.4g}±{st.elastic_std[0]:.4g}  "
                f"A₁₀={st.elastic_mean[9]:.4g}±{st.elastic_std[9]:.4g} "
                f"(n={st.elastic_n})",
                flush=True,
            )

        if not stats_list:
            continue

        # Save figures next to the theta dirs: .../<model>/T<label>/
        out_dir = angle_groups[0].directory.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        enabled = {
            "aa_layer_sep_vs_twist_angle": not args.no_layer_sep,
            "ab_layer_sep_vs_twist_angle": not args.no_layer_sep,
            "mean_layer_sep_vs_twist_angle": not args.no_mean_layer_sep,
            "mean_z_vs_twist_angle": not args.no_mean_z,
            "corrugation_amplitude_vs_twist_angle": not args.no_corrugation,
            "dw_width_vs_twist_angle": not args.no_dw_width,
            "max_intralayer_disp_vs_twist_angle": not args.no_max_intralayer_disp,
            "local_twist_vs_twist_angle": not args.no_local_twist,
        }
        write_mean_std_figures(
            stats_list, out_dir, dpi=args.dpi, enabled=enabled,
            arpes_mean_sep_points=arpes_mean_sep_points,
        )
        if not args.no_elastic_basis:
            write_elastic_basis_group_figures(
                stats_list,
                out_dir,
                dpi=args.dpi,
                tem_pld_points=tem_pld_points,
            )

        if not args.no_rel_uncertainty:
            plot_rel_uncertainty_vs_twist(
                stats_list,
                out_path=out_dir / "rel_uncertainty_vs_twist_angle.png",
                dpi=args.dpi,
            )

        if not args.no_cell_lengths:
            plot_cell_vector_lengths_vs_twist(
                stats_list,
                out_path=out_dir / "cell_vector_lengths_vs_twist_angle.png",
                dpi=args.dpi,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
