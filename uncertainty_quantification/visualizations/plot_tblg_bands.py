#!/usr/bin/env python3
"""
plot_tblg_bands.py
==================
Plot ensemble-average TBLG band structures with uncertainty bands.

Scans ``bands/propagation/`` (or ``--bands-dir``) for ``.npz`` files written
by :mod:`run_uq_propagation_bands`.  Files sharing the same (model, T, TB
model, TB T, twist angle) prefix are grouped as one ensemble; their
eigenvalues are stacked and the **per-band mean ± std** is plotted using
``plt.fill_between(..., alpha=0.3)``.

Plotting conventions match ``tests/test_acsf_band_structure.py``:

* k-path: K → Γ → M → K (high-symmetry points from saved ``k_node`` /
  ``sym_labels`` arrays).
* Vertical dashed lines at high-symmetry points.
* Dashed red horizontal line at E = 0 (Fermi level, already subtracted).
* Energy window defaults to ``--ylim -0.5 0.5`` eV (±0.5 eV around the Fermi level).

Output
------
One PNG per ensemble group, saved alongside the ``.npz`` files:

    ``mean_bands.png``

When multiple twist angles are present under the same
``<model>/T<label>/<tb_model>_tbT<label>/`` directory, summary figures are
written there:

* ``fermi_velocity_vs_twist_angle.png`` — mean ± std of |v_F| at K for the
  lowest band above E_F, averaged over the three nearest k-points in k-space
  (plotted as $|v_F| / 10^6$ m/s).
* ``fermi_velocity_vs_twist_angle_samples.png`` — same mean ± std with 10
  individual ensemble-sample curves overlaid.
* ``flat_band_width_vs_twist_angle.png`` — mean ± std of the occupied
  flat-band energy spread (the two lowest flat bands), with ARPES data overlaid when
  ``data/arpes_data_tblg_band_width_Evolution_of_the_flat_band.csv`` is present.
* ``flat_band_width_vs_twist_angle_samples.png`` — same occupied-band mean ± std with 10
  individual ensemble-sample curves overlaid (ARPES overlay included).
* ``band_gaps_vs_twist_angle.png`` — mean ± std of valence and conduction
  band gaps (flat bands vs adjacent dispersive bands, all k-points).
* ``flat_to_dispersive_gap_presence.png`` — percent of band structures with /
  without an open gap between the upper flat band and the next (lower)
  dispersive band vs twist angle (gap < 1e-3 eV counts as closed).
* ``flat_to_dispersive_gap_examples.png`` — side-by-side example band
  structures at θ = 0.99° flagged as having / not having that gap.
* ``magic_angle_histogram.png`` — histogram of per-sample magic angles (θ at
  min $|v_F|$ on complete ensemble curves); one bin per displayed twist angle,
  centered on the nominal θ; line plot of counts vs twist angle overlaid.

Flat bands are the four bands with energy closest to E = 0 at K (used to
label band indices).  Occupied flat-band width is max(flat eigenvalues) − min(flat
eigenvalues) over the two lowest of those four bands and every k-point.  The valence-band gap
is min(flat) − max(lower dispersive band); the conduction-band gap is
min(upper dispersive band) − max(flat), where the dispersive bands are the
single band indices immediately below/above the flat-band block (reported as
|·|).  Fermi
velocity is computed from the lowest unoccupied band at K: for each of the
three k-points nearest to K in reduced k-space, |v_F| = (1/ℏ)|ΔE/Δk| with
Δk converted to Å⁻¹ via an approximate moiré reciprocal scale from the twist
angle; the three values are then averaged.

Examples
--------
::

    python visualizations/plot_tblg_bands.py
    python visualizations/plot_tblg_bands.py --bands-dir bands/propagation \\
        --ylim -3.0 3.0 --dpi 200
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
REPO_ROOT = UQ_DIR.parent
DEFAULT_BANDS_DIR = UQ_DIR / "bands" / "propagation"
DEFAULT_ARPES_BANDWIDTH_CSV = (
    REPO_ROOT / "data" / "arpes_data_tblg_band_width_Evolution_of_the_flat_band.csv"
)
DEFAULT_YLIM = (-0.1, 0.1)   # eV — ±0.5 eV around the Fermi level (E = 0)
GAP_EXAMPLE_YLIM = (-0.15, 0.15)  # eV — wider window for gap vs no-gap examples
DEFAULT_DPI = 150
N_FLAT_BANDS = 4
N_OCCUPIED_FLAT_BANDS = 2
K_POINT_INDEX = 0
N_FERMI_KEEP = 25          # bands below/above nocc in run_uq_propagation_bands.py
N_FERMI_BANDS = 2 * N_FERMI_KEEP  # 50
N_NEAREST_K = 3            # k-neighbours used for Fermi-velocity finite difference
GRAPHENE_LATTICE_A = 2.46   # Å — used for approximate moiré |b|
# m/s per (eV / Å⁻¹): e / (ℏ · 10⁻¹⁰)
FERMI_VEL_SCALE = 1.519267447e5
# Display scale: plot |v_F| / FERMI_VEL_DISPLAY_SCALE (values in units of 10⁶ m/s)
FERMI_VEL_DISPLAY_SCALE = 1e6
FERMI_VEL_YLABEL = r"Fermi velocity $|v_F|$ ($\times 10^6$ m/s)"
DEFAULT_N_FERMI_SAMPLE_CURVES = 10
DEFAULT_N_BANDWIDTH_SAMPLE_CURVES = 10
# Gap between upper flat band and next dispersive band is "open" above this (eV).
GAP_OPEN_THRESHOLD_EV = 1e-4
TWIST_PLOT_MIN_DEG = 0.83
TWIST_PLOT_MAX_DEG = 1.47

# Regex to strip a trailing sample index from a filename to get the group
# prefix.  Accepts both ``…_sample<NNNN>.npz`` and bare ``sample<NNNN>.npz``.
_RE_SAMPLE_SUFFIX = re.compile(r"(?:^|_)sample\d+\.npz$", re.I)
_RE_SAMPLE_INDEX = re.compile(r"(?:^|_)sample(\d+)\.npz$", re.I)
_RE_THETA_DIR = re.compile(r"^theta(.+)deg$", re.I)
_RE_T_DIR = re.compile(r"^T(.+)$", re.I)


# ---------------------------------------------------------------------------
# Discovery and grouping
# ---------------------------------------------------------------------------

def parse_sample_index(path: Path) -> Optional[int]:
    """Parse ensemble sample index from ``*_sample<NNNN>.npz`` or ``sample<NNNN>.npz``."""
    m = _RE_SAMPLE_INDEX.search(path.name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def scale_fermi_velocity(v: np.ndarray | float) -> np.ndarray | float:
    """Scale |v_F| for plotting (divide by 10⁶ m/s)."""
    return np.asarray(v, dtype=float) / FERMI_VEL_DISPLAY_SCALE


def format_fermi_velocity(v: float) -> str:
    """Format |v_F| for log output (units of 10⁶ m/s)."""
    if not np.isfinite(v):
        return "nan"
    return f"{float(scale_fermi_velocity(v)):.4f}"


def discover_npz_files(bands_dir: Path) -> List[Path]:
    """Return all .npz files under *bands_dir*, sorted."""
    if not bands_dir.is_dir():
        return []
    return sorted(bands_dir.rglob("*.npz"))


def group_npz_files(npz_files: List[Path]) -> Dict[Tuple[Path, str], List[Path]]:
    """Group .npz files by (parent_directory, group_prefix).

    Every ``*.npz`` is included.  If the name ends with ``_sample<NNNN>`` or
    is ``sample<NNNN>.npz``, that suffix is stripped for the group prefix
    (bare ``sampleNNNN.npz`` files in one directory share one ensemble, keyed by
    the parent directory name).  Otherwise the file stem is used as the prefix.
    """
    groups: Dict[Tuple[Path, str], List[Path]] = {}
    for f in npz_files:
        m = _RE_SAMPLE_SUFFIX.search(f.name)
        if m is not None:
            prefix = f.name[: m.start()]
            if not prefix:
                prefix = f.parent.name
        else:
            prefix = f.stem
        key = (f.parent, prefix)
        groups.setdefault(key, []).append(f)
    return groups


@dataclass(frozen=True)
class ArpesBandwidthPoint:
    """ARPES flat-band width at one twist angle (energies in eV)."""
    theta_deg: float
    width_ev: float
    yerr_lower_ev: float
    yerr_upper_ev: float


def load_arpes_bandwidth_csv(path: Path) -> List[ArpesBandwidthPoint]:
    """
    Load ARPES flat-band widths from ``arpes_data_tblg_band_width_*.csv``.

    The min / max error-bar columns are **absolute** whisker positions (meV).
    Asymmetric uncertainties are ``mean − min`` and ``max − mean``; the total
    span is ``max − min``.
    """
    if not path.is_file():
        print(f"  ARPES bandwidth CSV not found: {path}", file=sys.stderr)
        return []

    mev_to_ev = 1.0e-3
    points: List[ArpesBandwidthPoint] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        rows = [{k.strip(): v for k, v in row.items()} for row in reader]

    for row in rows:
        try:
            theta = float(row["twist angle"])
            mean_mev = float(row["mean band width (meV)"])
            min_mev = float(str(row["min band width error bar"]).strip())
            max_mev = float(str(row["max band width errorbar"]).strip())
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  skip ARPES row {row!r}: {exc}", file=sys.stderr)
            continue

        mean_ev = mean_mev * mev_to_ev
        min_ev = min_mev * mev_to_ev
        max_ev = max_mev * mev_to_ev
        if max_ev < min_ev:
            print(
                f"  skip ARPES θ={theta:g}°: max error bar < min ({max_mev} < {min_mev} meV)",
                file=sys.stderr,
            )
            continue

        points.append(
            ArpesBandwidthPoint(
                theta_deg=theta,
                width_ev=mean_ev,
                yerr_lower_ev=mean_ev - min_ev,
                yerr_upper_ev=max_ev - mean_ev,
            )
        )

    return sorted(points, key=lambda p: p.theta_deg)


def overlay_arpes_bandwidth(
    ax: plt.Axes,
    arpes_points: Sequence[ArpesBandwidthPoint],
) -> None:
    """Overlay ARPES flat-band width vs twist angle on *ax*."""
    if not arpes_points:
        return

    thetas = np.array([p.theta_deg for p in arpes_points], dtype=float)
    y = np.array([p.width_ev for p in arpes_points], dtype=float)
    yerr = np.array(
        [
            [p.yerr_lower_ev for p in arpes_points],
            [p.yerr_upper_ev for p in arpes_points],
        ],
        dtype=float,
    )
    ax.errorbar(
        thetas,
        y,
        yerr=yerr,
        fmt="D-",
        color="C2",
        ms=7,
        lw=1.8,
        capsize=4,
        label="ARPES",
        zorder=4,
    )


@dataclass
class BandConfigGroup:
    """All twist-angle ensembles for one (model, T, TB) configuration."""
    model_name: str
    temperature_label: str
    tb_label: str
    config_dir: Path


@dataclass
class TwistBandMetrics:
    """Ensemble mean ± std of flat-band observables at one twist angle."""
    theta: float
    v_f_mean: float = float("nan")
    v_f_std: float = float("nan")
    v_f_n: int = 0
    occupied_bandwidth_mean: float = float("nan")
    occupied_bandwidth_std: float = float("nan")
    occupied_bandwidth_n: int = 0
    gap_below_mean: float = float("nan")
    gap_below_std: float = float("nan")
    gap_below_n: int = 0
    gap_above_mean: float = float("nan")
    gap_above_std: float = float("nan")
    gap_above_n: int = 0


def discover_band_config_groups(bands_dir: Path) -> List[BandConfigGroup]:
    """Return (model, T, TB) groups that contain at least one theta* directory."""
    if not bands_dir.is_dir():
        return []

    groups: List[BandConfigGroup] = []
    for model_dir in sorted(bands_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for t_dir in sorted(model_dir.iterdir()):
            if not t_dir.is_dir():
                continue
            m_t = _RE_T_DIR.match(t_dir.name)
            if not m_t:
                continue
            t_label = m_t.group(1)
            for tb_dir in sorted(t_dir.iterdir()):
                if not tb_dir.is_dir():
                    continue
                has_theta = any(
                    d.is_dir() and _RE_THETA_DIR.match(d.name)
                    for d in tb_dir.iterdir()
                )
                if has_theta:
                    groups.append(BandConfigGroup(
                        model_name=model_dir.name,
                        temperature_label=t_label,
                        tb_label=tb_dir.name,
                        config_dir=tb_dir,
                    ))
    return groups


def parse_theta_from_dir(path: Path) -> Optional[float]:
    """Parse twist angle (degrees) from a ``theta<angle>deg`` directory name."""
    m = _RE_THETA_DIR.match(path.name)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_npz_safe(path: Path) -> Optional[dict]:
    """Load a band-structure .npz file; return None on failure."""
    try:
        d = np.load(str(path), allow_pickle=False)
        return {k: d[k] for k in d.files}
    except Exception as exc:
        print(f"  skip {path.name}: {exc}", file=sys.stderr)
        return None


def extract_fermi_band_window(
    evals: np.ndarray,
    n_atoms: Optional[int] = None,
    n_keep: int = N_FERMI_KEEP,
) -> np.ndarray:
    """Return the Fermi-centred band window, shape ``(n_kpts, ≤2*n_keep)``.

    Matches the truncation in :mod:`run_uq_propagation_bands` — keep *n_keep*
    bands below and *n_keep* bands above ``nocc = N // 2``.  Files that were
    already saved with only this window are returned unchanged.  Older files
    that contain the full spectrum are sliced before stacking.
    """
    n_kpts, n_total = evals.shape
    target = 2 * n_keep

    if n_total <= target:
        return evals

    nocc = n_atoms // 2 if n_atoms is not None else n_total // 2
    band_lo = max(nocc - n_keep, 0)
    band_hi = min(nocc + n_keep, n_total)
    return evals[:, band_lo:band_hi]


def stack_ensemble(npz_files: List[Path]) -> Optional[dict]:
    """Load and stack eigenvalues from all files in one ensemble group.

    Returns a dict with:

    ``evals_stack`` : ndarray (n_samples, n_kpts, n_bands)
    ``k_dist``      : ndarray (n_kpts,)
    ``k_node``      : ndarray (n_nodes,)
    ``kvec``        : ndarray (n_kpts, 3) reduced coordinates
    ``sym_labels``  : list[str]
    ``twist_angle`` : float
    ``n_loaded``    : int
    ``n_failed``    : int

    Returns None if no samples could be loaded.
    """
    first_n_kpts: Optional[int] = None
    k_dist: Optional[np.ndarray] = None
    k_node: Optional[np.ndarray] = None
    kvec: Optional[np.ndarray] = None
    sym_labels: Optional[List[str]] = None
    twist_angle: float = float("nan")

    evals_list: List[np.ndarray] = []
    n_failed = 0
    n_resampled = 0

    for f in sorted(npz_files):
        d = load_npz_safe(f)
        if d is None:
            n_failed += 1
            continue

        ev = np.asarray(d["evals"], dtype=float)  # (n_kpts, n_bands)
        if ev.ndim != 2:
            print(f"  skip {f.name}: unexpected evals shape {ev.shape}", file=sys.stderr)
            n_failed += 1
            continue

        n_atoms: Optional[int] = None
        try:
            n_atoms = int(d["n_atoms"])
        except (KeyError, TypeError, ValueError):
            pass

        raw_shape = ev.shape
        ev = extract_fermi_band_window(ev, n_atoms=n_atoms)
        if ev.shape != raw_shape:
            n_resampled += 1

        n_kpts, n_bands = ev.shape
        if first_n_kpts is None:
            first_n_kpts = n_kpts
            k_dist = np.asarray(d["k_dist"], dtype=float)
            k_node = np.asarray(d["k_node"], dtype=float)
            try:
                kvec = np.asarray(d["kvec"], dtype=float)
            except KeyError:
                kvec = None
            try:
                sym_labels = [str(s) for s in d["sym_labels"]]
            except KeyError:
                sym_labels = ["K", "\u0393", "M", "K"]
            try:
                twist_angle = float(d["twist_angle"])
            except (KeyError, TypeError):
                pass
        else:
            if n_kpts != first_n_kpts:
                print(
                    f"  skip {f.name}: n_kpts {n_kpts} != expected {first_n_kpts}",
                    file=sys.stderr,
                )
                n_failed += 1
                continue

        evals_list.append(ev)

    if not evals_list:
        return None

    if n_resampled:
        print(
            f"    extracted Fermi-centred window ({N_FERMI_BANDS} bands) from "
            f"{n_resampled} file(s) with wider spectra",
            flush=True,
        )

    return {
        "evals_stack": np.stack(evals_list, axis=0),  # (n_samples, n_kpts, n_bands)
        "k_dist": k_dist,
        "k_node": k_node,
        "kvec": kvec,
        "sym_labels": sym_labels,
        "twist_angle": twist_angle,
        "n_loaded": len(evals_list),
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mean_bands(
    data: dict,
    title: str,
    out_path: Path,
    ylim: Tuple[float, float] = DEFAULT_YLIM,
    dpi: int = DEFAULT_DPI,
    band_color: str = "steelblue",
    mean_lw: float = 0.8,
    mean_alpha: float = 0.85,
    fill_alpha: float = 0.3,
    scatter: bool = False,
) -> None:
    """Plot ensemble-mean ± std band structure and save to *out_path*.

    Parameters
    ----------
    data : dict
        Output of :func:`stack_ensemble`.
    title : str
        Figure title.
    out_path : Path
        Output PNG path.
    ylim : (ymin, ymax)
        Energy window in eV.
    band_color : str
        Colour for mean bands and fill.
    mean_lw : float
        Line width for mean bands.
    mean_alpha : float
        Opacity of mean-band lines.
    fill_alpha : float
        Opacity of ±std fill regions.
    scatter : bool
        If True, use scatter instead of line for mean bands.
    """
    evals_stack = data["evals_stack"]  # (n_samples, n_kpts, n_bands)
    k_dist = data["k_dist"]            # (n_kpts,)
    k_node = data["k_node"]            # (n_nodes,)
    sym_labels = data["sym_labels"]

    n_samples, n_kpts, n_bands = evals_stack.shape

    # Per-band statistics over the sample axis
    mean_bands = np.mean(evals_stack, axis=0)   # (n_kpts, n_bands)
    std_bands = np.std(evals_stack, axis=0, ddof=1 if n_samples > 1 else 0)

    # Only render bands that have any mean value within the energy window
    # (± one std) to avoid cluttering with remote bands.
    emin, emax = ylim
    in_window = np.any(
        (mean_bands - std_bands < emax) & (mean_bands + std_bands > emin),
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    for b in range(n_bands):
        if not in_window[b]:
            continue
        mu = mean_bands[:, b]
        sig = std_bands[:, b]

        if scatter:
            ax.scatter(k_dist, mu, s=1.0, color=band_color,
                       alpha=mean_alpha, linewidths=0, zorder=2)
        else:
            ax.plot(k_dist, mu, color=band_color,
                    linewidth=mean_lw, alpha=mean_alpha, zorder=2)

        ax.fill_between(
            k_dist,
            mu - sig,
            mu + sig,
            color=band_color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=1,
        )

    # Fermi level reference line
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9,
               label="$E = 0$", zorder=3)

    # High-symmetry point markers
    for xv in k_node:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.6, zorder=3)

    ax.set_xlim(float(k_dist[0]), float(k_dist[-1]))
    ax.set_ylim(emin, emax)
    ax.set_xticks(k_node)
    ax.set_xticklabels(sym_labels, fontdict=CSFONT)
    ax.set_ylabel("Energy (eV)", fontdict=CSFONT)
    ax.set_title(title, fontdict=CSFONT)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Flat-band observables vs twist angle
# ---------------------------------------------------------------------------

def moire_reciprocal_scale(theta_deg: float, a: float = GRAPHENE_LATTICE_A) -> float:
    """Approximate moiré reciprocal scale |b| (Å⁻¹) from twist angle."""
    theta = np.radians(theta_deg)
    sin_half = np.sin(theta / 2.0)
    if sin_half <= 0.0:
        return float("nan")
    L_moire = a / (2.0 * sin_half)
    return 2.0 * np.pi / L_moire


def _delta_k_cart_angstrom(
    kvec: np.ndarray,
    i0: int,
    i1: int,
    theta_deg: float,
) -> float:
    """Approximate |Δk| (Å⁻¹) between two reduced-coord k-points."""
    dk_frac = np.asarray(kvec[i1, :2], dtype=float) - np.asarray(kvec[i0, :2], dtype=float)
    return float(np.linalg.norm(dk_frac) * moire_reciprocal_scale(theta_deg))


def find_k_point_index(
    k_dist: Optional[np.ndarray],
    k_node: Optional[np.ndarray],
) -> int:
    """Return the k-path index of the first K high-symmetry point."""
    if k_dist is not None and k_node is not None and len(k_node) > 0:
        return int(np.argmin(np.abs(k_dist - k_node[0])))
    return K_POINT_INDEX


def band_index_above_fermi(e_at_k: np.ndarray) -> Optional[int]:
    """Index of the lowest band strictly above E = 0 at K (first unoccupied)."""
    above = np.where(e_at_k > 0.0)[0]
    if len(above) > 0:
        return int(above[0])
    at_or_below = np.where(e_at_k <= 0.0)[0]
    if len(at_or_below) == 0:
        return None
    candidate = int(at_or_below[-1] + 1)
    if candidate >= len(e_at_k):
        return None
    return candidate


def compute_fermi_velocity(
    evals: np.ndarray,
    kvec: np.ndarray,
    k_dist: Optional[np.ndarray],
    k_node: Optional[np.ndarray],
    theta_deg: float,
    n_nearest: int = N_NEAREST_K,
) -> float:
    """|v_F| at K from the band above E_F, averaged over nearest k-neighbours.

    For each of the *n_nearest* k-points closest to K in reduced k-space,
    compute |v_F| = (1/ℏ)|ΔE/Δk| using the energy of the lowest unoccupied
    band at K.  Δk is converted from reduced coordinates to Å⁻¹ using an
    approximate moiré reciprocal scale derived from the twist angle.
    """
    if not np.isfinite(theta_deg):
        return float("nan")

    k_index = find_k_point_index(k_dist, k_node)
    band = band_index_above_fermi(evals[k_index])
    if band is None:
        return float("nan")

    n_kpts = evals.shape[0]
    if n_kpts <= 1:
        return float("nan")

    k0 = kvec[k_index, :2]
    dists = np.full(n_kpts, np.inf, dtype=float)
    for i in range(n_kpts):
        if i == k_index:
            continue
        dists[i] = float(np.linalg.norm(kvec[i, :2] - k0))
    nearest = np.argsort(dists)[:n_nearest]

    e_k = float(evals[k_index, band])
    vels: List[float] = []
    for j in nearest:
        j = int(j)
        dk = _delta_k_cart_angstrom(kvec, k_index, j, theta_deg)
        if dk <= 0.0:
            continue
        dE = abs(float(evals[j, band] - e_k))
        vels.append(dE / dk * FERMI_VEL_SCALE)

    if not vels:
        return float("nan")
    return float(np.mean(vels))


def identify_flat_band_indices(
    evals: np.ndarray,
    k_index: int = K_POINT_INDEX,
    n_flat: int = N_FLAT_BANDS,
) -> np.ndarray:
    """Return sorted band indices of the *n_flat* bands nearest E = 0 at *k_index*."""
    e_at_k = evals[k_index, :]
    nearest = np.argsort(np.abs(e_at_k))[:n_flat]
    return np.sort(nearest)


def compute_sample_band_metrics(
    evals: np.ndarray,
    kvec: Optional[np.ndarray],
    k_dist: Optional[np.ndarray] = None,
    k_node: Optional[np.ndarray] = None,
    k_index: int = K_POINT_INDEX,
    theta_deg: float = float("nan"),
) -> Dict[str, float]:
    """Occupied flat-band width, valence/conduction gaps, and Fermi velocity."""
    k_index = find_k_point_index(k_dist, k_node) if k_dist is not None else k_index
    fb = identify_flat_band_indices(evals, k_index=k_index)
    n_bands = evals.shape[1]

    flat_evals = evals[:, fb]
    occupied_flat_evals = flat_evals[:, :N_OCCUPIED_FLAT_BANDS]
    occupied_bandwidth = float(
        np.max(occupied_flat_evals) - np.min(occupied_flat_evals)
    )
    flat_min = float(np.min(flat_evals))
    flat_max = float(np.max(flat_evals))

    gap_below = float("nan")
    gap_above = float("nan")
    if fb[0] > 0:
        lower_dispersive = evals[:, fb[0] - 1]
        gap_below = abs(float(flat_min - np.max(lower_dispersive)))
    if fb[-1] < n_bands - 1:
        upper_dispersive = evals[:, fb[-1] + 1]
        gap_above = abs(float(np.min(upper_dispersive) - flat_max))

    v_f = float("nan")
    if kvec is not None:
        v_f = compute_fermi_velocity(
            evals, kvec, k_dist, k_node, theta_deg=theta_deg,
        )

    return {
        "v_f": v_f,
        "occupied_bandwidth": occupied_bandwidth,
        "gap_below": gap_below,
        "gap_above": gap_above,
    }


def _aggregate_metric(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, std, n) for a list of finite samples."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std, int(arr.size)


def compute_twist_band_metrics(
    data: dict,
    theta_deg: float,
) -> TwistBandMetrics:
    """Aggregate flat-band observables over all samples in one ensemble."""
    evals_stack = data["evals_stack"]
    kvec = data.get("kvec")
    k_dist = data.get("k_dist")
    k_node = data.get("k_node")
    v_f, occupied_bw, gap_lo, gap_hi = [], [], [], []

    for s in range(evals_stack.shape[0]):
        metrics = compute_sample_band_metrics(
            evals_stack[s],
            kvec=kvec,
            k_dist=k_dist,
            k_node=k_node,
            theta_deg=theta_deg,
        )
        if np.isfinite(metrics["v_f"]):
            v_f.append(metrics["v_f"])
        if np.isfinite(metrics["occupied_bandwidth"]):
            occupied_bw.append(metrics["occupied_bandwidth"])
        if np.isfinite(metrics["gap_below"]):
            gap_lo.append(metrics["gap_below"])
        if np.isfinite(metrics["gap_above"]):
            gap_hi.append(metrics["gap_above"])

    stats = TwistBandMetrics(theta=theta_deg)
    stats.v_f_mean, stats.v_f_std, stats.v_f_n = _aggregate_metric(v_f)
    (
        stats.occupied_bandwidth_mean,
        stats.occupied_bandwidth_std,
        stats.occupied_bandwidth_n,
    ) = _aggregate_metric(occupied_bw)
    stats.gap_below_mean, stats.gap_below_std, stats.gap_below_n = _aggregate_metric(gap_lo)
    stats.gap_above_mean, stats.gap_above_std, stats.gap_above_n = _aggregate_metric(gap_hi)
    return stats


def collect_per_sample_observables(
    config: BandConfigGroup,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build |v_F|(θ) and occupied flat-band width(θ) for each ensemble sample index.

    Returns ``(thetas, v_f_samples, occupied_bandwidth_samples)`` where sample arrays
    have shape ``(n_samples, n_theta)``.
    """
    by_sample: Dict[int, Dict[float, Dict[str, float]]] = {}
    thetas: List[float] = []

    theta_dirs = sorted(
        (
            d for d in config.config_dir.iterdir()
            if d.is_dir() and _RE_THETA_DIR.match(d.name)
        ),
        key=lambda d: parse_theta_from_dir(d) or 0.0,
    )

    for theta_dir in theta_dirs:
        theta = parse_theta_from_dir(theta_dir)
        if theta is None:
            continue
        th = float(theta)
        thetas.append(th)

        for f in sorted(theta_dir.glob("*.npz")):
            sid = parse_sample_index(f)
            if sid is None:
                continue
            d = load_npz_safe(f)
            if d is None:
                continue

            ev = np.asarray(d["evals"], dtype=float)
            if ev.ndim != 2:
                continue

            n_atoms: Optional[int] = None
            try:
                n_atoms = int(d["n_atoms"])
            except (KeyError, TypeError, ValueError):
                pass
            ev = extract_fermi_band_window(ev, n_atoms=n_atoms)

            k_dist = np.asarray(d["k_dist"], dtype=float)
            k_node = np.asarray(d["k_node"], dtype=float)
            try:
                kvec = np.asarray(d["kvec"], dtype=float)
            except KeyError:
                kvec = None

            metrics = compute_sample_band_metrics(
                ev,
                kvec=kvec,
                k_dist=k_dist,
                k_node=k_node,
                theta_deg=th,
            )
            entry: Dict[str, float] = {}
            for key in ("v_f", "occupied_bandwidth"):
                val = float(metrics[key])
                if np.isfinite(val):
                    entry[key] = val
            if entry:
                by_sample.setdefault(sid, {})[th] = entry

    theta_arr = np.asarray(sorted(set(thetas)), dtype=float)
    if theta_arr.size == 0 or not by_sample:
        empty = np.zeros((0, 0), dtype=float)
        return theta_arr, empty, empty

    sample_ids = sorted(by_sample)
    v_f_samples = np.full((len(sample_ids), theta_arr.size), np.nan, dtype=float)
    occupied_bandwidth_samples = np.full(
        (len(sample_ids), theta_arr.size), np.nan, dtype=float
    )
    for i, sid in enumerate(sample_ids):
        for j, th in enumerate(theta_arr):
            obs = by_sample[sid].get(float(th), {})
            if "v_f" in obs:
                v_f_samples[i, j] = obs["v_f"]
            if "occupied_bandwidth" in obs:
                occupied_bandwidth_samples[i, j] = obs["occupied_bandwidth"]

    return theta_arr, v_f_samples, occupied_bandwidth_samples


def collect_per_sample_fermi_velocities(
    config: BandConfigGroup,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(thetas, v_f_samples)`` — see :func:`collect_per_sample_observables`."""
    thetas, v_f_samples, _ = collect_per_sample_observables(config)
    return thetas, v_f_samples


def magic_angle_from_curve(thetas: np.ndarray, v_f: np.ndarray) -> float:
    """Twist angle at minimum |v_F| on one sample's θ scan."""
    thetas = np.asarray(thetas, dtype=float).ravel()
    v_f = np.asarray(v_f, dtype=float).ravel()
    m = np.isfinite(thetas) & np.isfinite(v_f)
    if not np.any(m):
        return float("nan")
    t = thetas[m]
    v = v_f[m]
    return float(t[int(np.argmin(v))])


def magic_angles_per_sample(thetas: np.ndarray, v_f_samples: np.ndarray) -> np.ndarray:
    """Magic angle for samples having finite |v_F| at every twist angle."""
    v_f_samples = np.asarray(v_f_samples, dtype=float)
    if v_f_samples.ndim != 2 or v_f_samples.shape[0] == 0:
        return np.array([], dtype=float)
    complete = np.all(np.isfinite(v_f_samples), axis=1)
    return np.array(
        [
            magic_angle_from_curve(thetas, v_f_samples[i, :])
            for i in np.flatnonzero(complete)
        ],
        dtype=float,
    )


def complete_sample_mask(v_f_samples: np.ndarray) -> np.ndarray:
    """Return rows with finite Fermi velocities at every supplied twist angle."""
    values = np.asarray(v_f_samples, dtype=float)
    if values.ndim != 2:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(values), axis=1)


def equal_width_bar_width(thetas: np.ndarray) -> float:
    """Uniform bar width for equal-width bars centered on each twist angle."""
    thetas = np.sort(np.asarray(thetas, dtype=float).ravel())
    if thetas.size == 0:
        return 1.0
    if thetas.size == 1:
        return 0.1 * max(abs(float(thetas[0])), 1.0)
    return float(np.min(np.diff(thetas)))


def counts_per_twist_angle(magic: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Count magic angles in equal-width bins centered on each twist angle."""
    thetas = np.sort(np.asarray(thetas, dtype=float).ravel())
    magic = np.asarray(magic, dtype=float).ravel()
    magic = magic[np.isfinite(magic)]
    if thetas.size == 0:
        return np.array([], dtype=float)
    width = equal_width_bar_width(thetas)
    half = 0.5 * width
    counts = np.zeros(thetas.size, dtype=float)
    for i, t in enumerate(thetas):
        # Inclusive on the left, exclusive on the right except for the last bin.
        lo, hi = float(t) - half, float(t) + half
        if i == thetas.size - 1:
            counts[i] = float(np.sum((magic >= lo) & (magic <= hi)))
        else:
            counts[i] = float(np.sum((magic >= lo) & (magic < hi)))
    return counts


def plot_magic_angle_histogram(
    config: BandConfigGroup,
    thetas: np.ndarray,
    v_f_samples: np.ndarray,
    out_path: Path,
    dpi: int = DEFAULT_DPI,
    display_thetas: Optional[np.ndarray] = None,
) -> None:
    """Histogram of complete-sample magic angles over the displayed twist range."""
    thetas = np.sort(np.asarray(thetas, dtype=float).ravel())
    magic = magic_angles_per_sample(thetas, v_f_samples)
    magic = magic[np.isfinite(magic)]
    plot_thetas = (
        thetas
        if display_thetas is None
        else np.sort(np.asarray(display_thetas, dtype=float).ravel())
    )
    if plot_thetas.size == 0 or magic.size == 0:
        return

    width = equal_width_bar_width(plot_thetas)
    counts = counts_per_twist_angle(magic, plot_thetas)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(
        plot_thetas,
        counts,
        width=width,
        align="center",
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        label="ensemble samples",
    )
    ax.plot(
        plot_thetas,
        counts,
        "ko-",
        linewidth=2.0,
        markersize=6,
        label="counts vs twist angle",
    )

    ax.set_xlabel(r"Magic angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_ylabel("Count", fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def magic_angle_ensemble_stats(
    thetas: np.ndarray,
    v_f_samples: np.ndarray,
) -> Tuple[float, float, int]:
    """Mean and std of magic angle over complete ensemble samples."""
    angles = magic_angles_per_sample(thetas, v_f_samples)
    angles = angles[np.isfinite(angles)]
    return _aggregate_metric(angles.tolist())


def _plot_mean_std_vs_twist(
    stats_list: List[TwistBandMetrics],
    *,
    mean_attr: str,
    std_attr: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    color: str = "C0",
    label: str = "ensemble mean",
    y_scale: float = 1.0,
    arpes_points: Optional[Sequence[ArpesBandwidthPoint]] = None,
) -> None:
    """Generic mean ± std line plot vs twist angle."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list])
    mean = np.array([getattr(s, mean_attr) for s in stats_list], dtype=float) / y_scale
    std = np.array([getattr(s, std_attr) for s in stats_list], dtype=float) / y_scale

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    valid = np.isfinite(mean)
    if valid.any():
        t_v = thetas[valid]
        m_v = mean[valid]
        s_v = std[valid]
        ax.plot(t_v, m_v, "o-", color=color, label=label)
        ax.fill_between(t_v, m_v - s_v, m_v + s_v, color=color, alpha=0.3)

    if arpes_points:
        overlay_arpes_bandwidth(ax, arpes_points)
        ax.legend(prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE})

    ax.set_xlabel(r"Twist angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    #print(f"  Wrote {out_path}", flush=True)


def plot_fermi_velocity_vs_twist(
    stats_list: List[TwistBandMetrics],
    model_name: str,
    t_label: str,
    tb_label: str,
    out_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Fermi velocity (lowest band above E_F at K) vs twist angle."""
    _plot_mean_std_vs_twist(
        stats_list,
        mean_attr="v_f_mean",
        std_attr="v_f_std",
        ylabel=FERMI_VEL_YLABEL,
        title=(
            rf"{model_name}  $T = {t_label}$  {tb_label}"
            + "\nFermi velocity at K (band above $E_F$, 3 nearest $k$-points)"
        ),
        out_path=out_path,
        dpi=dpi,
        color="C0",
        label=r"$|v_F|$ (mean over 3 nearest $k$)",
        y_scale=FERMI_VEL_DISPLAY_SCALE,
    )


def plot_vs_twist_with_samples(
    stats_list: List[TwistBandMetrics],
    thetas: np.ndarray,
    sample_values: np.ndarray,
    *,
    mean_attr: str,
    std_attr: str,
    ylabel: str,
    title: str,
    subtitle: str,
    out_path: Path,
    color: str = "C0",
    n_curves: int = DEFAULT_N_FERMI_SAMPLE_CURVES,
    dpi: int = DEFAULT_DPI,
    y_scale: float = 1.0,
    sample_transform=None,
    arpes_points: Optional[Sequence[ArpesBandwidthPoint]] = None,
) -> None:
    """Mean ± std vs twist angle with individual ensemble-sample curves overlaid."""
    if sample_transform is None:
        sample_transform = lambda y: np.asarray(y, dtype=float) / y_scale

    stats_list = sorted(stats_list, key=lambda s: s.theta)
    theta_stats = np.array([s.theta for s in stats_list], dtype=float)
    mean = sample_transform(np.array([getattr(s, mean_attr) for s in stats_list], dtype=float))
    std = sample_transform(np.array([getattr(s, std_attr) for s in stats_list], dtype=float))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    sample_values = np.asarray(sample_values, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    n_shown = 0
    if sample_values.ndim == 2 and sample_values.shape[0] > 0 and thetas.size > 0:
        valid_rows = [
            i for i in range(sample_values.shape[0])
            if np.any(np.isfinite(sample_values[i]))
        ]
        if valid_rows:
            if len(valid_rows) <= n_curves:
                show_rows = valid_rows
            else:
                pick = np.linspace(0, len(valid_rows) - 1, n_curves, dtype=int)
                show_rows = [valid_rows[int(j)] for j in pick]
            n_shown = len(show_rows)
            for row in show_rows:
                y = sample_transform(sample_values[row])
                m = np.isfinite(thetas) & np.isfinite(y)
                if m.any():
                    ax.plot(thetas[m], y[m], "-", color="0.65", lw=1.0, alpha=0.75, zorder=1)

    valid = np.isfinite(mean)
    if valid.any():
        t_v = theta_stats[valid]
        m_v = mean[valid]
        s_v = std[valid]
        ax.plot(t_v, m_v, "o-", color=color, lw=2.0, label="ensemble mean", zorder=3)
        ax.fill_between(
            t_v, m_v - s_v, m_v + s_v, color=color, alpha=0.3,
            label="ensemble std", zorder=2,
        )

    if arpes_points:
        overlay_arpes_bandwidth(ax, arpes_points)
        ax.legend(prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE})

    ax.set_xlabel(r"Twist angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_ylabel(ylabel, fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_fermi_velocity_vs_twist_with_samples(
    stats_list: List[TwistBandMetrics],
    thetas: np.ndarray,
    v_f_samples: np.ndarray,
    model_name: str,
    t_label: str,
    tb_label: str,
    out_path: Path,
    *,
    n_curves: int = DEFAULT_N_FERMI_SAMPLE_CURVES,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Mean ± std |v_F|(θ) with individual ensemble-sample curves overlaid."""
    plot_vs_twist_with_samples(
        stats_list,
        thetas,
        v_f_samples,
        mean_attr="v_f_mean",
        std_attr="v_f_std",
        ylabel=FERMI_VEL_YLABEL,
        title=rf"{model_name}  $T = {t_label}$  {tb_label}",
        subtitle="Fermi velocity at K",
        out_path=out_path,
        color="C0",
        n_curves=n_curves,
        dpi=dpi,
        y_scale=FERMI_VEL_DISPLAY_SCALE,
        sample_transform=scale_fermi_velocity,
    )


def plot_flat_band_width_vs_twist_with_samples(
    stats_list: List[TwistBandMetrics],
    thetas: np.ndarray,
    bandwidth_samples: np.ndarray,
    model_name: str,
    t_label: str,
    tb_label: str,
    out_path: Path,
    *,
    n_curves: int = DEFAULT_N_BANDWIDTH_SAMPLE_CURVES,
    dpi: int = DEFAULT_DPI,
    arpes_points: Optional[Sequence[ArpesBandwidthPoint]] = None,
) -> None:
    """Mean ± std occupied flat-band width(θ) with sample curves overlaid."""
    plot_vs_twist_with_samples(
        stats_list,
        thetas,
        bandwidth_samples,
        mean_attr="occupied_bandwidth_mean",
        std_attr="occupied_bandwidth_std",
        ylabel="Occupied flat-band width (eV)",
        title=rf"{model_name}  $T = {t_label}$  {tb_label}",
        subtitle="Occupied flat-band width (2 lowest of 4 bands nearest $E_F$)",
        out_path=out_path,
        color="C1",
        n_curves=n_curves,
        dpi=dpi,
        arpes_points=arpes_points,
    )


def plot_flat_band_width_vs_twist(
    stats_list: List[TwistBandMetrics],
    model_name: str,
    t_label: str,
    tb_label: str,
    out_path: Path,
    dpi: int = DEFAULT_DPI,
    arpes_points: Optional[Sequence[ArpesBandwidthPoint]] = None,
) -> None:
    """Occupied flat-band energy width vs twist angle, compared with ARPES."""
    _plot_mean_std_vs_twist(
        stats_list,
        mean_attr="occupied_bandwidth_mean",
        std_attr="occupied_bandwidth_std",
        ylabel="Occupied flat-band width (eV)",
        title=(
            rf"{model_name}  $T = {t_label}$  {tb_label}"
            + "\nOccupied flat-band width (all $k$, 2 lowest of 4 bands)"
        ),
        out_path=out_path,
        dpi=dpi,
        color="C1",
        label="occupied flat-band width",
        arpes_points=arpes_points,
    )


def plot_band_gaps_vs_twist(
    stats_list: List[TwistBandMetrics],
    model_name: str,
    t_label: str,
    tb_label: str,
    out_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Valence and conduction band gaps vs twist angle."""
    stats_list = sorted(stats_list, key=lambda s: s.theta)
    thetas = np.array([s.theta for s in stats_list])
    gap_lo_mean = np.array([s.gap_below_mean for s in stats_list])
    gap_lo_std = np.array([s.gap_below_std for s in stats_list])
    gap_hi_mean = np.array([s.gap_above_mean for s in stats_list])
    gap_hi_std = np.array([s.gap_above_std for s in stats_list])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    valid_lo = np.isfinite(gap_lo_mean)
    if valid_lo.any():
        t_lo = thetas[valid_lo]
        m_lo = gap_lo_mean[valid_lo]
        s_lo = gap_lo_std[valid_lo]
        ax.plot(t_lo, m_lo, "o-", color="C2", label="valence band gap")
        ax.fill_between(t_lo, m_lo - s_lo, m_lo + s_lo, color="C2", alpha=0.3)

    valid_hi = np.isfinite(gap_hi_mean)
    if valid_hi.any():
        t_hi = thetas[valid_hi]
        m_hi = gap_hi_mean[valid_hi]
        s_hi = gap_hi_std[valid_hi]
        ax.plot(t_hi, m_hi, "s-", color="C3", label="conduction band gap")
        ax.fill_between(t_hi, m_hi - s_hi, m_hi + s_hi, color="C3", alpha=0.3)

    ax.set_xlabel(r"Twist angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_ylabel("Band gap (eV)", fontdict=CSFONT)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    #print(f"  Wrote {out_path}", flush=True)


def collect_gap_above_by_twist(
    config: BandConfigGroup,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Collect per-sample conduction gaps grouped by twist angle.

    Returns ``(thetas, gaps_per_theta)`` where ``gaps_per_theta[i]`` is the
    1-D array of finite ``gap_above`` values at ``thetas[i]``.
    """
    by_theta: Dict[float, List[float]] = {}

    theta_dirs = sorted(
        (
            d for d in config.config_dir.iterdir()
            if d.is_dir() and _RE_THETA_DIR.match(d.name)
        ),
        key=lambda d: parse_theta_from_dir(d) or 0.0,
    )

    for theta_dir in theta_dirs:
        theta = parse_theta_from_dir(theta_dir)
        if theta is None:
            continue
        th = float(theta)
        gaps: List[float] = []
        for f in sorted(theta_dir.glob("*.npz")):
            d = load_npz_safe(f)
            if d is None:
                continue
            ev = np.asarray(d["evals"], dtype=float)
            if ev.ndim != 2:
                continue

            n_atoms: Optional[int] = None
            try:
                n_atoms = int(d["n_atoms"])
            except (KeyError, TypeError, ValueError):
                pass
            ev = extract_fermi_band_window(ev, n_atoms=n_atoms)

            k_dist = np.asarray(d["k_dist"], dtype=float)
            k_node = np.asarray(d["k_node"], dtype=float)
            try:
                kvec = np.asarray(d["kvec"], dtype=float)
            except KeyError:
                kvec = None

            metrics = compute_sample_band_metrics(
                ev,
                kvec=kvec,
                k_dist=k_dist,
                k_node=k_node,
                theta_deg=th,
            )
            g = float(metrics["gap_above"])
            if np.isfinite(g):
                gaps.append(g)

        if gaps:
            by_theta[th] = gaps

    thetas = np.asarray(sorted(by_theta), dtype=float)
    gaps_per_theta = [np.asarray(by_theta[float(t)], dtype=float) for t in thetas]
    return thetas, gaps_per_theta


def plot_flat_to_dispersive_gap_presence(
    thetas: np.ndarray,
    gaps_per_theta: Sequence[np.ndarray],
    out_path: Path,
    *,
    threshold_ev: float = GAP_OPEN_THRESHOLD_EV,
    dpi: int = DEFAULT_DPI,
) -> None:
    """
    Percent of band structures with / without an open gap between the upper
    flat band and the next dispersive band, plotted vs twist angle.

    Gaps smaller than *threshold_ev* (default 1e-3 eV) count as closed.
    """
    thetas = np.asarray(thetas, dtype=float).ravel()
    if thetas.size == 0 or len(gaps_per_theta) != thetas.size:
        print(f"  skip gap-presence plot: no data ({out_path.name})", flush=True)
        return

    pct_yes = np.full(thetas.size, np.nan, dtype=float)
    pct_no = np.full(thetas.size, np.nan, dtype=float)
    n_tot = np.zeros(thetas.size, dtype=int)

    for i, gaps in enumerate(gaps_per_theta):
        g = np.asarray(gaps, dtype=float).ravel()
        g = g[np.isfinite(g)]
        if g.size == 0:
            continue
        has_gap = g >= float(threshold_ev)
        n = int(g.size)
        n_tot[i] = n
        pct_yes[i] = 100.0 * float(np.sum(has_gap)) / n
        pct_no[i] = 100.0 * float(np.sum(~has_gap)) / n

    valid = np.isfinite(pct_yes) & np.isfinite(pct_no)
    if not np.any(valid):
        print(f"  skip gap-presence plot: no finite gaps ({out_path.name})", flush=True)
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        thetas[valid], pct_yes[valid], "o-", color="C0",
        label="has gap",
    )
    ax.plot(
        thetas[valid], pct_no[valid], "s-", color="C1",
        label="no gap",
    )
    ax.set_xlabel(r"Twist angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_ylabel("percent (%)", fontdict=CSFONT)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(
        f"  Flat→dispersive gap presence vs θ "
        f"(threshold={threshold_ev:g} eV):",
        flush=True,
    )
    for t, py, pn, n in zip(thetas[valid], pct_yes[valid], pct_no[valid], n_tot[valid]):
        print(
            f"    θ={float(t):g}°  has gap={py:.1f}%  no gap={pn:.1f}%  (n={int(n)})",
            flush=True,
        )
    print(f"  Wrote {out_path}", flush=True)


@dataclass(frozen=True)
class GapBandExample:
    """One band-structure sample used as a gap / no-gap illustration."""
    path: Path
    theta: float
    gap: float
    evals: np.ndarray
    k_dist: np.ndarray
    k_node: np.ndarray
    sym_labels: Tuple[str, ...]


def _load_band_example_from_npz(path: Path, theta: float) -> Optional[Tuple[GapBandExample, float]]:
    """Load one npz; return ``(example, gap_above)`` or None on failure."""
    d = load_npz_safe(path)
    if d is None:
        return None
    ev = np.asarray(d["evals"], dtype=float)
    if ev.ndim != 2:
        return None

    n_atoms: Optional[int] = None
    try:
        n_atoms = int(d["n_atoms"])
    except (KeyError, TypeError, ValueError):
        pass
    ev = extract_fermi_band_window(ev, n_atoms=n_atoms)

    k_dist = np.asarray(d["k_dist"], dtype=float)
    k_node = np.asarray(d["k_node"], dtype=float)
    try:
        kvec = np.asarray(d["kvec"], dtype=float)
    except KeyError:
        kvec = None

    metrics = compute_sample_band_metrics(
        ev,
        kvec=kvec,
        k_dist=k_dist,
        k_node=k_node,
        theta_deg=float(theta),
    )
    g = float(metrics["gap_above"])
    if not np.isfinite(g):
        return None

    try:
        sym_raw = d["sym_labels"]
        sym_labels = tuple(str(s) for s in np.asarray(sym_raw).tolist())
    except KeyError:
        sym_labels = tuple(f"P{i}" for i in range(len(k_node)))

    return GapBandExample(
        path=path,
        theta=float(theta),
        gap=g,
        evals=ev,
        k_dist=k_dist,
        k_node=k_node,
        sym_labels=sym_labels,
    ), g


def find_gap_band_examples(
    config: BandConfigGroup,
    *,
    threshold_ev: float = GAP_OPEN_THRESHOLD_EV,
    theta_target_deg: float = 0.99,
    theta_atol: float = 1e-3,
) -> Tuple[Optional[GapBandExample], Optional[GapBandExample]]:
    """
    Pick clearest examples with / without an open flat→dispersive gap at
    ``theta_target_deg`` (default 0.99°).

    * has gap: largest ``gap_above`` among samples with gap ≥ threshold
    * no gap: smallest ``gap_above`` among samples with gap < threshold
    """
    best_open: Optional[GapBandExample] = None
    best_closed: Optional[GapBandExample] = None

    thr = float(threshold_ev)
    found_theta_dir = False
    for theta_dir in sorted(config.config_dir.iterdir()):
        if not theta_dir.is_dir():
            continue
        if not _RE_THETA_DIR.match(theta_dir.name):
            continue
        theta = parse_theta_from_dir(theta_dir)
        if theta is None or not np.isclose(float(theta), theta_target_deg, atol=theta_atol):
            continue
        found_theta_dir = True
        for f in sorted(theta_dir.glob("*.npz")):
            loaded = _load_band_example_from_npz(f, float(theta))
            if loaded is None:
                continue
            ex, g = loaded
            if g >= thr:
                if best_open is None or g > best_open.gap:
                    best_open = ex
            else:
                if best_closed is None or g < best_closed.gap:
                    best_closed = ex

    if not found_theta_dir:
        print(
            f"  no θ={theta_target_deg:g}° band directory under {config.config_dir}",
            flush=True,
        )
    return best_open, best_closed


def _plot_single_band_structure(
    ax,
    evals: np.ndarray,
    k_dist: np.ndarray,
    k_node: np.ndarray,
    sym_labels: Sequence[str],
    *,
    ylim: Tuple[float, float],
    band_color: str = "steelblue",
    title: str = "",
) -> None:
    """Draw one sample's bands on *ax* (no ensemble fill)."""
    emin, emax = ylim
    n_bands = evals.shape[1]
    in_window = np.any((evals < emax) & (evals > emin), axis=0)

    for b in range(n_bands):
        if not in_window[b]:
            continue
        ax.plot(
            k_dist, evals[:, b], color=band_color,
            linewidth=0.9, alpha=0.9, zorder=2,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9, zorder=3)
    for xv in k_node:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.6, zorder=3)

    ax.set_xlim(float(k_dist[0]), float(k_dist[-1]))
    ax.set_ylim(emin, emax)
    ax.set_xticks(k_node)
    ax.set_xticklabels(list(sym_labels), fontdict=CSFONT)
    ax.set_ylabel("Energy (eV)", fontdict=CSFONT)
    if title:
        ax.set_title(title, fontdict=CSFONT)


def plot_gap_presence_example_bands(
    config: BandConfigGroup,
    out_path: Path,
    *,
    threshold_ev: float = GAP_OPEN_THRESHOLD_EV,
    theta_target_deg: float = 0.99,
    ylim: Tuple[float, float] = GAP_EXAMPLE_YLIM,
    dpi: int = DEFAULT_DPI,
) -> None:
    """
    Side-by-side example band structures at ``theta_target_deg`` (default 0.99°):
    one with an open flat→dispersive gap and one without (gap < *threshold_ev*).
    """
    has_ex, no_ex = find_gap_band_examples(
        config,
        threshold_ev=threshold_ev,
        theta_target_deg=theta_target_deg,
    )
    if has_ex is None and no_ex is None:
        print(
            f"  skip gap-example bands: no finite-gap samples at "
            f"θ={theta_target_deg:g}° ({out_path.name})",
            flush=True,
        )
        return
    if has_ex is None:
        print(
            f"  skip gap-example bands: no sample at θ={theta_target_deg:g}° "
            f"with gap ≥ {threshold_ev:g} eV",
            flush=True,
        )
        return
    if no_ex is None:
        print(
            f"  skip gap-example bands: no sample at θ={theta_target_deg:g}° "
            f"with gap < {threshold_ev:g} eV",
            flush=True,
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    _plot_single_band_structure(
        axes[0],
        has_ex.evals,
        has_ex.k_dist,
        has_ex.k_node,
        has_ex.sym_labels,
        ylim=ylim,
        band_color="C0",
        title=(
            f"has gap\n"
            rf"$\theta={has_ex.theta:g}^\circ$, $\Delta={has_ex.gap:.4f}$ eV"
        ),
    )
    _plot_single_band_structure(
        axes[1],
        no_ex.evals,
        no_ex.k_dist,
        no_ex.k_node,
        no_ex.sym_labels,
        ylim=ylim,
        band_color="C1",
        title=(
            f"no gap\n"
            rf"$\theta={no_ex.theta:g}^\circ$, $\Delta={no_ex.gap:.4e}$ eV"
        ),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  Gap examples: has gap ← {has_ex.path.name} "
        f"(θ={has_ex.theta:g}°, Δ={has_ex.gap:.4f} eV); "
        f"no gap ← {no_ex.path.name} "
        f"(θ={no_ex.theta:g}°, Δ={no_ex.gap:.4e} eV)",
        flush=True,
    )
    print(f"  Wrote {out_path}", flush=True)


def plot_twist_angle_summaries(
    config: BandConfigGroup,
    stats_list: List[TwistBandMetrics],
    dpi: int,
    *,
    thetas: Optional[np.ndarray] = None,
    v_f_samples: Optional[np.ndarray] = None,
    bandwidth_samples: Optional[np.ndarray] = None,
    n_fermi_sample_curves: int = DEFAULT_N_FERMI_SAMPLE_CURVES,
    n_bandwidth_sample_curves: int = DEFAULT_N_BANDWIDTH_SAMPLE_CURVES,
    ylim: Tuple[float, float] = DEFAULT_YLIM,
    arpes_points: Optional[Sequence[ArpesBandwidthPoint]] = None,
) -> None:
    """Write all flat-band summary figures for one (model, T, TB) configuration."""
    if not stats_list:
        return

    stats_for_other_plots = [
        stats for stats in stats_list
        if TWIST_PLOT_MIN_DEG <= stats.theta <= TWIST_PLOT_MAX_DEG
    ]
    filtered_thetas: Optional[np.ndarray] = None
    filtered_v_f_samples: Optional[np.ndarray] = None
    if thetas is not None:
        all_thetas = np.asarray(thetas, dtype=float)
        theta_mask = (
            (all_thetas >= TWIST_PLOT_MIN_DEG)
            & (all_thetas <= TWIST_PLOT_MAX_DEG)
        )
        filtered_thetas = all_thetas[theta_mask]
        if v_f_samples is not None:
            values = np.asarray(v_f_samples, dtype=float)
            if values.ndim == 2 and values.shape[1] == all_thetas.size:
                filtered_v_f_samples = values[:, theta_mask]
            else:
                filtered_v_f_samples = values
    common = dict(
        model_name=config.model_name,
        t_label=config.temperature_label,
        tb_label=config.tb_label,
        dpi=dpi,
    )
    out_dir = config.config_dir

    plot_fermi_velocity_vs_twist(
        stats_for_other_plots,
        out_path=out_dir / "fermi_velocity_vs_twist_angle.png",
        **common,
    )

    if thetas is not None and v_f_samples is not None and v_f_samples.size > 0:
        plot_fermi_velocity_vs_twist_with_samples(
            stats_for_other_plots,
            filtered_thetas,
            filtered_v_f_samples,
            out_path=out_dir / "fermi_velocity_vs_twist_angle_samples.png",
            n_curves=n_fermi_sample_curves,
            **common,
        )

        n_complete = int(np.sum(complete_sample_mask(v_f_samples)))
        print(
            f"  Samples present at all twist angles with finite |v_F|: "
            f"{n_complete}/{v_f_samples.shape[0]}",
            flush=True,
        )
        magic_mean, magic_std, magic_n = magic_angle_ensemble_stats(thetas, v_f_samples)
        if np.isfinite(magic_mean):
            if np.isfinite(magic_std) and magic_n > 1:
                print(
                    f"  Magic angle (min $|v_F|$): "
                    f"θ = {magic_mean:.4f} ± {magic_std:.4f}°  (n={magic_n})",
                    flush=True,
                )
            else:
                print(
                    f"  Magic angle (min $|v_F|$): θ = {magic_mean:.4f}°  (n={magic_n})",
                    flush=True,
                )
        else:
            print("  Magic angle (min $|v_F|$): nan", flush=True)

        plot_magic_angle_histogram(
            config,
            thetas,
            v_f_samples,
            out_path=out_dir / "magic_angle_histogram.png",
            dpi=dpi,
            display_thetas=filtered_thetas,
        )

    # Keep the occupied-band/ARPES comparison on the full available angle range.
    plot_flat_band_width_vs_twist(
        stats_list,
        out_path=out_dir / "flat_band_width_vs_twist_angle.png",
        arpes_points=arpes_points,
        **common,
    )

    if thetas is not None and bandwidth_samples is not None and bandwidth_samples.size > 0:
        plot_flat_band_width_vs_twist_with_samples(
            stats_list,
            thetas,
            bandwidth_samples,
            out_path=out_dir / "flat_band_width_vs_twist_angle_samples.png",
            n_curves=n_bandwidth_sample_curves,
            arpes_points=arpes_points,
            **common,
        )

    plot_band_gaps_vs_twist(
        stats_for_other_plots,
        out_path=out_dir / "band_gaps_vs_twist_angle.png",
        **common,
    )

    gap_thetas, gaps_per_theta = collect_gap_above_by_twist(config)
    gap_mask = (
        (gap_thetas >= TWIST_PLOT_MIN_DEG)
        & (gap_thetas <= TWIST_PLOT_MAX_DEG)
    )
    plot_flat_to_dispersive_gap_presence(
        gap_thetas[gap_mask],
        [g for g, keep in zip(gaps_per_theta, gap_mask) if keep],
        out_path=out_dir / "flat_to_dispersive_gap_presence.png",
        dpi=dpi,
    )
    plot_gap_presence_example_bands(
        config,
        out_path=out_dir / "flat_to_dispersive_gap_examples.png",
        ylim=GAP_EXAMPLE_YLIM,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot ensemble-mean TBLG band structures with ±std fill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--bands-dir",
        type=Path,
        default=DEFAULT_BANDS_DIR,
        help=f"Directory containing band-structure .npz files "
             f"(default: {DEFAULT_BANDS_DIR}).",
    )
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=list(DEFAULT_YLIM),
        metavar=("EMIN", "EMAX"),
        help=f"Energy window in eV (default: {DEFAULT_YLIM[0]} {DEFAULT_YLIM[1]}).",
    )
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument(
        "--scatter",
        action="store_true",
        help="Use scatter markers instead of lines for mean bands.",
    )
    p.add_argument(
        "--band-color",
        default="steelblue",
        help="Colour for mean-band lines and fill (default: steelblue).",
    )
    p.add_argument(
        "--fill-alpha",
        type=float,
        default=0.3,
        help="Opacity of the ±std fill regions (default: 0.3).",
    )
    p.add_argument(
        "--no-mean-bands",
        action="store_true",
        help="Skip per-ensemble mean band-structure PNGs.",
    )
    p.add_argument(
        "--no-vs-twist",
        action="store_true",
        help="Skip flat-band observables vs twist-angle summary PNGs.",
    )
    p.add_argument(
        "--n-fermi-sample-curves",
        type=int,
        default=DEFAULT_N_FERMI_SAMPLE_CURVES,
        help=(
            "Number of individual ensemble |v_F|(θ) curves to overlay on "
            f"fermi_velocity_vs_twist_angle_samples.png (default: "
            f"{DEFAULT_N_FERMI_SAMPLE_CURVES})."
        ),
    )
    p.add_argument(
        "--arpes-bandwidth-csv",
        type=Path,
        default=DEFAULT_ARPES_BANDWIDTH_CSV,
        help=(
            "ARPES flat-band width vs twist angle CSV for bandwidth summary plots "
            f"(default: {DEFAULT_ARPES_BANDWIDTH_CSV.name})."
        ),
    )
    p.add_argument(
        "--no-arpes-bandwidth",
        action="store_true",
        help="Do not overlay ARPES data on flat-band width vs twist-angle plots.",
    )
    args = p.parse_args()

    os.chdir(UQ_DIR)

    bands_dir = Path(args.bands_dir)
    if not bands_dir.is_absolute():
        bands_dir = UQ_DIR / bands_dir

    npz_files = discover_npz_files(bands_dir)
    if not npz_files:
        p.error(f"No .npz files found under {bands_dir}")

    groups = group_npz_files(npz_files)
    if not groups:
        p.error(f"Found {len(npz_files)} .npz file(s) under {bands_dir} but none could be grouped.")

    print(
        f"Found {len(groups)} ensemble group(s) "
        f"({len(npz_files)} total .npz file(s)).",
        flush=True,
    )

    ylim = tuple(args.ylim)

    arpes_bandwidth_path = Path(args.arpes_bandwidth_csv)
    if not arpes_bandwidth_path.is_absolute():
        arpes_bandwidth_path = REPO_ROOT / arpes_bandwidth_path
    arpes_points: Optional[List[ArpesBandwidthPoint]] = None
    if not args.no_arpes_bandwidth:
        arpes_points = load_arpes_bandwidth_csv(arpes_bandwidth_path)
        if arpes_points:
            print(
                f"Loaded {len(arpes_points)} ARPES flat-band width point(s) "
                f"from {arpes_bandwidth_path}",
                flush=True,
            )

    if not args.no_mean_bands:
        for (parent_dir, prefix), members in sorted(groups.items()):
            print(f"\n  {prefix}  ({len(members)} sample(s))", flush=True)

            data = stack_ensemble(members)
            if data is None:
                print(f"    no valid samples — skipping.", file=sys.stderr)
                continue

            n_loaded = data["n_loaded"]
            n_failed = data["n_failed"]
            theta = data["twist_angle"]
            n_bands = data["evals_stack"].shape[2]

            print(
                f"    loaded {n_loaded}/{n_loaded + n_failed}  "
                f"shape=({n_loaded}, {data['evals_stack'].shape[1]}, {n_bands})  "
                f"θ={theta:g}°",
                flush=True,
            )

            theta_str = f"{theta:g}" if np.isfinite(theta) else "unknown"
            title = rf"$\theta = {theta_str}^\circ$"

            out_path = parent_dir / "mean_bands.png"
            plot_mean_bands(
                data,
                title=title,
                out_path=out_path,
                ylim=ylim,
                dpi=args.dpi,
                scatter=args.scatter,
                band_color=args.band_color,
                fill_alpha=args.fill_alpha,
            )
            print(f"    Wrote {out_path}", flush=True)

    if not args.no_vs_twist:
        config_groups = discover_band_config_groups(bands_dir)
        if not config_groups:
            print(
                "\nNo (model/T/TB) configuration directories with theta* "
                "subdirectories found — skipping vs-twist summaries.",
                flush=True,
            )
        else:
            print(
                f"\nGenerating flat-band observables vs twist angle for "
                f"{len(config_groups)} configuration(s) …",
                flush=True,
            )
            for config in config_groups:
                print(
                    f"\n{'=' * 60}\n"
                    f" Config: {config.model_name}  T={config.temperature_label}  "
                    f"{config.tb_label}\n{'=' * 60}",
                    flush=True,
                )
                stats_list: List[TwistBandMetrics] = []

                theta_dirs = sorted(
                    (
                        d for d in config.config_dir.iterdir()
                        if d.is_dir() and _RE_THETA_DIR.match(d.name)
                    ),
                    key=lambda d: parse_theta_from_dir(d) or 0.0,
                )
                for theta_dir in theta_dirs:
                    theta = parse_theta_from_dir(theta_dir)
                    if theta is None:
                        continue

                    npz_in_dir = sorted(theta_dir.glob("*.npz"))
                    dir_groups = group_npz_files(npz_in_dir)
                    if not dir_groups:
                        print(
                            f"  θ={theta:g}°  no sample .npz files — skipping.",
                            file=sys.stderr,
                        )
                        continue

                    # All prefixes in one theta directory share the same twist angle.
                    members = next(iter(dir_groups.values()))
                    data = stack_ensemble(members)
                    if data is None:
                        print(
                            f"  θ={theta:g}°  no valid samples — skipping.",
                            file=sys.stderr,
                        )
                        continue

                    st = compute_twist_band_metrics(data, theta_deg=theta)
                    stats_list.append(st)
                    print(
                        f"  θ={theta:g}°  n={data['n_loaded']}  "
                        f"|v_F|={format_fermi_velocity(st.v_f_mean)}"
                        f"±{format_fermi_velocity(st.v_f_std) if np.isfinite(st.v_f_std) else 'nan'}"
                        f" (×10⁶ m/s)  "
                        f"ΔE_occ={st.occupied_bandwidth_mean:.4f}"
                        f"±{st.occupied_bandwidth_std:.4f} eV  "
                        f"gap_val={st.gap_below_mean:.4f}±{st.gap_below_std:.4f} eV  "
                        f"gap_cond={st.gap_above_mean:.4f}±{st.gap_above_std:.4f} eV",
                        flush=True,
                    )

                thetas, v_f_samples, bandwidth_samples = collect_per_sample_observables(
                    config,
                )
                plot_twist_angle_summaries(
                    config,
                    stats_list,
                    dpi=args.dpi,
                    thetas=thetas,
                    v_f_samples=v_f_samples,
                    bandwidth_samples=bandwidth_samples,
                    n_fermi_sample_curves=args.n_fermi_sample_curves,
                    n_bandwidth_sample_curves=args.n_fermi_sample_curves,
                    ylim=ylim,
                    arpes_points=arpes_points,
                )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
