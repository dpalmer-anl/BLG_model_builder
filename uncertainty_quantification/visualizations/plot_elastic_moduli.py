#!/usr/bin/env python3
"""
Bilayer-graphene elastic stiffness constants: DFT vs POD ensemble UQ.

Workflow (matches ``calc_elastic_constants.py``)
-------------------------------------------------
1. Match DFT result structures to reference labels ``(stacking, mode, delta)``
   by geometry fingerprinting (same ``fingerprint`` / ``cluster_reference_frames``
   / matching logic as ``calc_elastic_constants.py``).
2. For each stacking and mode, fit total energies along the 1-D strain path
   ``delta`` with a quadratic polynomial relative to ``delta = 0``.
3. Convert the fitted curvature ``b2`` for each mode into Voigt stiffnesses
   ``C11, C12, C13, C33, C44`` (units eV/Å³) using the slab volume
   ``V = A d0``, then report GPa via ``1 eV/Å³ = 160.21766208 GPa``.

Model ensemble UQ
------------------
For each successful MCMC ensemble draw, the model's total energy is evaluated
on every DFT elastic-constant structure (batched), the same quadratic fits are
applied per mode, and the resulting elastic constants are recorded. The mean
and std over ensemble draws give the model's elastic constants with
uncertainty, printed alongside the DFT values for comparison.

Plots show, per stacking, one subplot per strain mode with the ensemble
mean ± std energy curve overlaid on the DFT energies (no quadratic fit line).
A single shared legend is drawn in the unused (bottom-right) subplot slot.

A separate figure shows histograms of each elastic constant over the
ensemble, with the DFT value marked as a solid vertical line.

Output:
  ``figures/<model>_<stacking>_elastic_constants.png``
  ``figures/<model>_<stacking>_elastic_constants_histogram.png``

Examples
--------
::

    python visualizations/plot_elastic_moduli.py --models POD_energy
    python visualizations/plot_elastic_moduli.py \\
        --models 'POD_energy_POD_index*' --n-samples 200

With no ``--models``, uses the ``POD_energy_POD_index*`` ensemble with the
lowest NLL on the saved calibration grid.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

CSFONT = {"fontname": "sans-serif", "size": 20}
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": 15,
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)

# Side effect: ``Atoms.relax_structure`` patch (Allegro / LAMMPS helpers).
from blg_model_builder.potentials import PODASECalculator  # noqa: F401

from blg_model_builder.ensemble_io import (
    DEFAULT_CALIBRATION_METRICS_DIR,
    expand_model_patterns,
    load_ensemble_pickle,
    load_metrics_npz,
    metrics_npz_path,
    resolve_ensemble_pickle,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
REPO_ROOT = UQ_DIR.parent

_uq_dir = str(UQ_DIR)
if _uq_dir not in sys.path:
    sys.path.insert(0, _uq_dir)

from uq_model_runtime import (  # noqa: E402
    apply_uq_parameters,
    build_uq_calculator,
    is_uq_energy_model,
)

DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"
DEFAULT_REF_XYZ = (
    REPO_ROOT.parent
    / "DD-TETB"
    / "generate_data"
    / "Carbon_training_data"
    / "bilayer_graphene_elastic_constants_structures.xyz"
)
DEFAULT_DFT_XYZ = REPO_ROOT / "data" / "blg_elastic_constants_structures.xyz"

DEFAULT_N_SAMPLES = 500
DEFAULT_ENSEMBLE_SHUFFLE_SEED = 0

EV_A3_TO_GPA = 160.21766208
MODES = ("A", "B", "C", "D", "E")
CONSTANT_NAMES = ("C11", "C12", "C13", "C33", "C44")
CONSTANT_TITLE = {
    "C11": r"$C_{11}$",
    "C12": r"$C_{12}$",
    "C13": r"$C_{13}$",
    "C33": r"$C_{33}$",
    "C44": r"$C_{44}$",
}

# Strain paths from DD-TETB/generate_data/gen_elastic_constant_structures.py
# (``apply_mode``): A: e1, B: e1=e2, C: e3, D: e1&e3, E: e4 (yz shear).
MODE_TITLE = {
    "A": r"$C_{11}$  ($\epsilon_{xx}=\delta$)",                    # e1
    "B": r"$C_{12}$  ($\epsilon_{xx}=\epsilon_{yy}=\delta$)",      # e1 = e2
    "C": r"$C_{33}$  ($\epsilon_{zz}=\delta$)",                    # e3
    "D": r"$C_{13}$  ($\epsilon_{xx}=\epsilon_{zz}=\delta$)",      # e1 & e3
    "E": r"$C_{44}$  ($\epsilon_{yz}=\delta$)",                    # e4 (interlayer shear)
}

# Literature AB bilayer graphene DFT stiffnesses (GPa) for sanity check.
REFERENCE_GPA = {
    "C11": 1080.0,
    "C12": 162.0,
    "C13": -4.63,
    "C33": 33.13,
    "C44": 3.32,
}


# ----------------------------------------------------------------------
# Geometry matching (same as calc_elastic_constants.py)
# ----------------------------------------------------------------------
def fingerprint(atoms) -> np.ndarray:
    """
    A permutation- and order-invariant geometric fingerprint of a strained
    bilayer frame: in-plane lattice vector lengths + angle, interlayer
    separation, and relative interlayer lateral (shear) offset.
    """
    cell = atoms.cell[:]
    a1, a2 = cell[0], cell[1]
    a1_len = np.linalg.norm(a1)
    a2_len = np.linalg.norm(a2)
    cos_ang = np.dot(a1, a2) / (a1_len * a2_len)
    ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    z = atoms.positions[:, 2]
    zmid = 0.5 * (z.min() + z.max())
    bottom = atoms[z < zmid]
    top = atoms[z >= zmid]
    interlayer_sep = top.positions[:, 2].mean() - bottom.positions[:, 2].mean()
    shear_y = top.positions[:, 1].mean() - bottom.positions[:, 1].mean()

    return np.array([a1_len, a2_len, ang, interlayer_sep, shear_y])


def cluster_reference_frames(ref_frames, tol: float = 1e-4) -> list:
    """
    Group reference frames that are geometrically identical (this happens at
    delta=0, where all 5 modes reduce to the same structure). Returns a list
    of ``[representative_fingerprint, [list of (stacking, mode, delta, d0)]]``.
    """
    fps = np.array([fingerprint(a) for a in ref_frames])
    scale = fps.std(axis=0)
    scale[scale < 1e-8] = 1.0

    clusters: list = []
    for a, fp in zip(ref_frames, fps):
        label = (a.info["stacking"], a.info["mode"], a.info["delta"], a.info["d0"])
        placed = False
        for c in clusters:
            if np.linalg.norm((fp - c[0]) / scale) < tol:
                c[1].append(label)
                placed = True
                break
        if not placed:
            clusters.append([fp, [label]])
    return clusters


def match_records(result_frames, clusters, max_dist: float = 1e-3) -> List[dict]:
    """
    For each DFT result frame, find its nearest reference cluster and emit
    ONE record ``{stacking, mode, delta, d0, frame_index}`` per label in that
    cluster (the ``delta=0`` frame is shared across all 5 modes). Energies are
    looked up separately by ``frame_index`` so the same matching can be reused
    for DFT and for any model's per-sample energies.
    """
    rep_fps = np.array([c[0] for c in clusters])
    scale = rep_fps.std(axis=0)
    scale[scale < 1e-8] = 1.0

    records: List[dict] = []
    for i, a in enumerate(result_frames):
        fp = fingerprint(a)
        dist = np.linalg.norm((rep_fps - fp) / scale, axis=1)
        idx = int(np.argmin(dist))
        if dist[idx] > max_dist:
            print(
                f"  WARNING: closest match has normalized distance {dist[idx]:.2e} "
                f"(fingerprint={fp}) -- check this structure manually."
            )
        for (stacking, mode, delta, d0) in clusters[idx][1]:
            records.append(
                dict(stacking=stacking, mode=mode, delta=delta, d0=d0, frame_index=i)
            )
    return records


def load_and_match(ref_xyz: Path, dft_xyz: Path) -> Tuple[list, list, List[dict]]:
    """Load reference labels + DFT result frames and match by geometry."""
    ref_frames = read(str(ref_xyz), index=":")
    result_frames = read(str(dft_xyz), index=":")
    print(
        f"loaded {len(ref_frames)} reference structures, "
        f"{len(result_frames)} DFT result structures",
        flush=True,
    )

    clusters = cluster_reference_frames(ref_frames)
    print(
        f"reference set collapses to {len(clusters)} distinct geometries "
        f"({len(ref_frames)} labels -- the delta=0 points are shared across the 5 modes)",
        flush=True,
    )

    records = match_records(result_frames, clusters)

    ref_keys = sorted({(s, m, round(d, 6)) for c in clusters for (s, m, d, _) in c[1]})
    matched_keys = [(r["stacking"], r["mode"], round(r["delta"], 6)) for r in records]
    counts = Counter(matched_keys)
    missing = [k for k in ref_keys if counts.get(k, 0) == 0]
    duplicated = [k for k, c in counts.items() if c > 1]
    if missing:
        print(f"  WARNING: {len(missing)} reference points have no matching DFT result: {missing}")
    if duplicated:
        print(
            f"  NOTE: {len(duplicated)} points matched more than once -- expected only for "
            f"delta=0 (shared across all 5 modes): {duplicated}"
        )
    print(flush=True)
    return ref_frames, result_frames, records


# ----------------------------------------------------------------------
# Elastic constant calculation (same formulas as calc_elastic_constants.py)
# ----------------------------------------------------------------------
def slab_volume(area: float, d0: float) -> float:
    """Slab volume V = A d0 (in-plane area × interlayer spacing, Å³)."""
    return float(area) * float(d0)


def quadratic_fit_delta_energy(
    deltas: np.ndarray,
    energies: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit ``ΔE = b2 δ² + b1 δ + b0`` relative to the point nearest ``δ=0``."""
    deltas = np.asarray(deltas, dtype=float)
    energies = np.asarray(energies, dtype=float)
    i0 = int(np.argmin(np.abs(deltas)))
    e_ref = float(energies[i0])
    de = energies - e_ref
    b2, b1, b0 = np.polyfit(deltas, de, 2)
    fit = np.polyval([b2, b1, b0], deltas)
    rms = float(np.sqrt(np.mean((fit - de) ** 2)))
    return float(b2), float(b1), float(b0 + e_ref), rms


def elastic_constants_from_modes(
    b2A: float, b2B: float, b2C: float, b2D: float, b2E: float, volume: float,
) -> Dict[str, float]:
    """Map mode curvatures ``b2`` to Voigt stiffnesses (eV/Å³)."""
    v = float(volume)
    if v <= 0.0:
        return {}
    c11 = b2A / v
    c12 = b2B / (2.0 * v) - c11
    c33 = 2.0 * b2C / v
    c13 = (b2A - b2D) / v
    c44 = 2.0 * abs(b2E) / (3.0 * v)
    return {"C11": c11, "C12": c12, "C13": c13, "C33": c33, "C44": c44}


class StackGeometry:
    """Per-stacking mode -> (deltas, frame_indices) plus slab volume."""

    def __init__(self, volume: float) -> None:
        self.volume = volume
        self.mode_deltas: Dict[str, np.ndarray] = {}
        self.mode_frame_indices: Dict[str, List[int]] = {}


def build_stack_geometry(
    ref_frames,
    records: Sequence[dict],
) -> Dict[str, StackGeometry]:
    """Group matched records by ``(stacking, mode)`` and compute slab volumes."""
    by_group: Dict[Tuple[str, str], List[dict]] = {}
    for r in records:
        by_group.setdefault((r["stacking"], r["mode"]), []).append(r)

    stacks = sorted({r["stacking"] for r in records})
    out: Dict[str, StackGeometry] = {}
    for stack in stacks:
        ref0 = [a for a in ref_frames if a.info["stacking"] == stack and a.info["delta"] == 0.0][0]
        area = float(np.linalg.norm(np.cross(ref0.cell[0], ref0.cell[1])))
        d0 = float(ref0.info["d0"])
        geom = StackGeometry(volume=slab_volume(area, d0))
        for mode in MODES:
            pts = sorted(by_group.get((stack, mode), []), key=lambda r: r["delta"])
            if len(pts) < 3:
                print(f"  WARNING: only {len(pts)} points for {stack}/{mode}, skipping")
                continue
            geom.mode_deltas[mode] = np.array([r["delta"] for r in pts], dtype=float)
            geom.mode_frame_indices[mode] = [r["frame_index"] for r in pts]
        out[stack] = geom
    return out


def elastic_constants_for_energies(
    geom: StackGeometry,
    energies_by_frame: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Elastic constants for one stacking from a single energy vector (all frames)."""
    if not all(m in geom.mode_deltas for m in MODES):
        return None
    b2 = {}
    for mode in MODES:
        deltas = geom.mode_deltas[mode]
        idx = geom.mode_frame_indices[mode]
        energies = energies_by_frame[idx]
        b2[mode], _b1, _b0, _rms = quadratic_fit_delta_energy(deltas, energies)
    return elastic_constants_from_modes(
        b2["A"], b2["B"], b2["C"], b2["D"], b2["E"], geom.volume,
    )


# ----------------------------------------------------------------------
# Model ensemble evaluation
# ----------------------------------------------------------------------
def _shuffle_ensemble(ensemble: np.ndarray, seed: int) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 2:
        raise ValueError(f"Expected 2-D ensemble array, got shape {ensemble.shape}")
    order = np.random.default_rng(seed).permutation(ensemble.shape[0])
    return ensemble[order]


def _is_lammps_error(exc: BaseException) -> bool:
    cur: Optional[BaseException] = exc
    seen: set = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if "lammps" in type(cur).__module__.lower():
            return True
        msg = str(cur).lower()
        if "lammps" in msg or "lmp_" in msg:
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "lammps" in msg or "relax_structure" in msg:
            return True
    return False


def evaluate_ensemble_energies(
    calc_obj,
    ensemble_shuffled: np.ndarray,
    n_samples: int,
    atoms_list: Sequence,
    *,
    set_params_fn=None,
) -> Tuple[np.ndarray, int]:
    """
    Evaluate total energies on ``atoms_list`` for successive ensemble draws.

    Returns ``(energy_ensemble, n_success)`` where ``energy_ensemble`` has
    shape ``(len(atoms_list), n_success)``.
    """
    n_frames = len(atoms_list)
    energy_ensemble = np.full((n_frames, n_samples), np.nan, dtype=float)
    n_success = 0

    calc_obj.prepare_batch(list(atoms_list))

    for theta in ensemble_shuffled:
        if n_success >= n_samples:
            break
        try:
            apply_uq_parameters(calc_obj, theta, set_params_fn)
            energies, _ = calc_obj.evaluate_batch()
            energies = np.asarray(energies, dtype=float).ravel()
            if energies.size != n_frames or not np.all(np.isfinite(energies)):
                continue
            energy_ensemble[:, n_success] = energies
            n_success += 1
        except Exception as exc:
            if _is_lammps_error(exc):
                print(f"    skip ensemble member (LAMMPS): {exc}", file=sys.stderr)
            else:
                print(f"    skip ensemble member: {exc}", file=sys.stderr)
            continue

    return energy_ensemble[:, :n_success], n_success


def select_pod_model_lowest_nll(
    ensemble_dir: str,
    calibration_metrics_dir: str,
    *,
    calibration_technique: str = "mcmc",
    calibration_target: str = "energy",
) -> str:
    """Return the ``POD_energy_POD_index*`` folder with minimum calibration NLL."""
    candidates = expand_model_patterns(["POD_energy_POD_index*"], ensemble_dir)
    best_name: Optional[str] = None
    best_nll = float("inf")
    for model_name in candidates:
        path = metrics_npz_path(
            calibration_metrics_dir, model_name, calibration_technique, calibration_target,
        )
        if not os.path.isfile(path):
            continue
        nll_arr = np.asarray(load_metrics_npz(path)["nll"], dtype=float)
        nll_min = float(np.nanmin(nll_arr)) if nll_arr.size else float("nan")
        if np.isfinite(nll_min) and nll_min < best_nll:
            best_nll = nll_min
            best_name = model_name
    if best_name is None:
        raise ValueError(
            "No POD_energy_POD_index* model with finite calibration NLL found under "
            f"{ensemble_dir!r} / {calibration_metrics_dir!r}."
        )
    print(f"Auto-selected POD model {best_name!r} (lowest NLL = {best_nll:.6g})", flush=True)
    return best_name


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def print_dft_elastic_constants(stacking: str, constants: Dict[str, float]) -> None:
    print(f"\n{stacking} DFT elastic constants (Voigt, eV/Å³ → GPa):")
    for name in CONSTANT_NAMES:
        val = constants.get(name, float("nan"))
        gpa = val * EV_A3_TO_GPA
        line = f"  {name} = {gpa:10.3f} GPa   ({val:.6f} eV/Å³)"
        if name in REFERENCE_GPA and np.isfinite(gpa):
            ref = REFERENCE_GPA[name]
            line += f"    [lit. {ref:.2f} GPa, Δ={gpa - ref:+.1f}]"
        print(line)


def print_model_elastic_constants(
    model_name: str,
    stacking: str,
    dft_constants: Dict[str, float],
    model_mean_std: Dict[str, Tuple[float, float]],
) -> None:
    print(f"\n{stacking} {model_name} elastic constants (mean ± std over ensemble, GPa):")
    for name in CONSTANT_NAMES:
        mu, sig = model_mean_std.get(name, (float("nan"), float("nan")))
        mu_gpa, sig_gpa = mu * EV_A3_TO_GPA, sig * EV_A3_TO_GPA
        dft_gpa = dft_constants.get(name, float("nan")) * EV_A3_TO_GPA
        line = f"  {name} = {mu_gpa:10.3f} ± {sig_gpa:.3f} GPa"
        if np.isfinite(dft_gpa):
            line += f"    [DFT {dft_gpa:.3f} GPa, Δ={mu_gpa - dft_gpa:+.3f}]"
        print(line)


def collect_per_sample_constants(
    geom: StackGeometry,
    energy_ensemble: np.ndarray,
    n_ok: int,
) -> List[Dict[str, float]]:
    """Elastic constants (eV/Å³) for each successful ensemble draw."""
    out: List[Dict[str, float]] = []
    for j in range(n_ok):
        constants = elastic_constants_for_energies(geom, energy_ensemble[:, j])
        if constants is not None:
            out.append(constants)
    return out


def mean_std_constants(
    per_sample_constants: Sequence[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Mean and std (eV/Å³) of each elastic constant over ensemble draws."""
    model_mean_std: Dict[str, Tuple[float, float]] = {}
    for name in CONSTANT_NAMES:
        vals = np.array([c[name] for c in per_sample_constants], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            model_mean_std[name] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            model_mean_std[name] = (float("nan"), float("nan"))
    return model_mean_std


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_stack_elastic_moduli(
    model_name: str,
    stacking: str,
    geom: StackGeometry,
    dft_energies_by_frame: np.ndarray,
    model_energy_ensemble_by_frame: np.ndarray,
    *,
    figures_dir: Path,
    dpi: int = 150,
) -> Path:
    """
    Per-mode subplots of ensemble mean ± std energy vs strain overlaid on DFT,
    with a single shared legend in the unused bottom-right subplot slot.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes_flat = axes.ravel()

    legend_handles = None
    legend_labels = None

    for ax_idx, mode in enumerate(MODES):
        ax = axes_flat[ax_idx]
        if mode not in geom.mode_deltas:
            ax.set_title(f"mode {mode} (missing)", fontdict=CSFONT)
            ax.axis("off")
            continue

        deltas = geom.mode_deltas[mode]
        idx = geom.mode_frame_indices[mode]
        i0 = int(np.argmin(np.abs(deltas)))

        dft_e = dft_energies_by_frame[idx]
        dft_rel = dft_e - dft_e[i0]

        model_e = model_energy_ensemble_by_frame[idx, :]  # (npts, n_success)
        model_rel = model_e - model_e[i0:i0 + 1, :]
        mean_rel = np.mean(model_rel, axis=1)
        std_rel = np.std(model_rel, axis=1)

        (line_mean,) = ax.plot(deltas, mean_rel, "-", color="C0", lw=2.0, zorder=2, label="ensemble mean")
        band = ax.fill_between(
            deltas, mean_rel - std_rel, mean_rel + std_rel,
            alpha=0.3, color="C0", zorder=1, label="ensemble std",
        )
        (line_dft,) = ax.plot(deltas, dft_rel, "o-", color="C2", ms=7, zorder=3, label="DFT")

        if legend_handles is None:
            legend_handles = [line_mean, band, line_dft]
            legend_labels = ["ensemble mean", "ensemble std", "DFT"]

        ax.set_xlabel(r"strain $\delta$", fontdict=CSFONT)
        ax.set_ylabel(r"$\Delta E$ (eV)", fontdict=CSFONT)
        ax.set_title(MODE_TITLE[mode], fontdict=CSFONT)
        ax.grid(True, alpha=0.3)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    if legend_handles is not None:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            prop={"family": CSFONT["fontname"], "size": 15},
            frameon=True,
        )

    fig.suptitle(f"{stacking} stacking — {model_name}", fontsize=CSFONT["size"], fontname=CSFONT["fontname"])
    fig.tight_layout()
    out = figures_dir / f"{model_name}_{stacking}_elastic_constants.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


def plot_elastic_constants_histograms(
    model_name: str,
    stacking: str,
    per_sample_constants: Sequence[Dict[str, float]],
    dft_constants: Dict[str, float],
    *,
    figures_dir: Path,
    dpi: int = 150,
    n_bins: int = 30,
) -> Path:
    """Histogram of each elastic constant (GPa) with DFT value as a vertical line."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes_flat = axes.ravel()

    legend_handles = None
    legend_labels = None

    for ax_idx, name in enumerate(CONSTANT_NAMES):
        ax = axes_flat[ax_idx]
        vals_gpa = np.array([c[name] for c in per_sample_constants], dtype=float) * EV_A3_TO_GPA
        vals_gpa = vals_gpa[np.isfinite(vals_gpa)]
        if vals_gpa.size == 0:
            ax.set_title(f"{CONSTANT_TITLE[name]} (no data)", fontdict=CSFONT)
            ax.axis("off")
            continue

        _n_hist, _edges, patches = ax.hist(
            vals_gpa,
            bins=n_bins,
            color="C0",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            label="ensemble",
        )

        dft_ev_a3 = dft_constants.get(name, float("nan"))
        dft_gpa = float(dft_ev_a3) * EV_A3_TO_GPA
        dft_line = None
        if np.isfinite(dft_gpa):
            dft_line = ax.axvline(
                dft_gpa,
                color="C2",
                lw=2.5,
                linestyle="-",
                zorder=5,
                label="DFT",
            )

        if legend_handles is None:
            legend_handles = [patches[0]]
            legend_labels = ["ensemble"]
            if dft_line is not None:
                legend_handles.append(dft_line)
                legend_labels.append("DFT")

        ax.set_xlabel("GPa", fontdict=CSFONT)
        ax.set_ylabel("count", fontdict=CSFONT)
        ax.set_title(CONSTANT_TITLE[name], fontdict=CSFONT)
        ax.grid(True, alpha=0.3, axis="y")

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    if legend_handles is not None:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            prop={"family": CSFONT["fontname"], "size": 15},
            frameon=True,
        )

    fig.suptitle(
        f"{stacking} stacking — {model_name}",
        fontsize=CSFONT["size"],
        fontname=CSFONT["fontname"],
    )
    fig.tight_layout()
    out = figures_dir / f"{model_name}_{stacking}_elastic_constants_histogram.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot bilayer graphene elastic stiffness constants: DFT vs POD ensemble UQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p, required=False)
    p.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    p.add_argument(
        "--temperature", type=float, default=None,
        help="MCMC temperature weight T for ensemble pickle (nearest match).",
    )
    p.add_argument(
        "--calibration-metrics-dir", default=DEFAULT_CALIBRATION_METRICS_DIR,
        help="Directory with calibration_*.npz from plot_bayes_factor.py --calculate.",
    )
    p.add_argument(
        "--calibration-target", default="energy",
        help="Target key in calibration npz (default: energy).",
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help="Target number of successful ensemble evaluations.",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_ENSEMBLE_SHUFFLE_SEED,
        help="RNG seed for ensemble shuffle.",
    )
    p.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help="Output directory for PNG figures (default: uncertainty_quantification/figures).",
    )
    p.add_argument(
        "--ref-xyz", type=Path, default=DEFAULT_REF_XYZ,
        help="Reference (stacking, mode, delta)-labeled structures.",
    )
    p.add_argument(
        "--dft-xyz", type=Path, default=DEFAULT_DFT_XYZ,
        help="DFT single-point elastic-constant structures (default: data/blg_elastic_constants_structures.xyz).",
    )
    p.add_argument("--dpi", type=int, default=150)
    add_hyperparam_args(p)
    args, unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, unknown)

    os.chdir(UQ_DIR)

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    ref_xyz = Path(args.ref_xyz)
    dft_xyz = Path(args.dft_xyz)
    if not dft_xyz.is_absolute():
        dft_xyz = REPO_ROOT / dft_xyz if (REPO_ROOT / dft_xyz).exists() else UQ_DIR / dft_xyz

    print(f"Loading reference labels from {ref_xyz}", flush=True)
    print(f"Loading DFT structures from {dft_xyz}", flush=True)
    ref_frames, result_frames, records = load_and_match(ref_xyz, dft_xyz)

    dft_energies_by_frame = np.array(
        [a.get_potential_energy() for a in result_frames], dtype=float,
    )

    stack_geoms = build_stack_geometry(ref_frames, records)

    dft_constants: Dict[str, Dict[str, float]] = {}
    for stacking, geom in stack_geoms.items():
        constants = elastic_constants_for_energies(geom, dft_energies_by_frame)
        if constants is None:
            print(f"\n{stacking}: incomplete mode set — skipping DFT constants")
            continue
        dft_constants[stacking] = constants
        print_dft_elastic_constants(stacking, constants)

    if args.models:
        models = expand_model_patterns(args.models, args.ensemble_dir)
        if not models:
            p.error("No models matched --models patterns.")
    else:
        try:
            models = [
                select_pod_model_lowest_nll(
                    args.ensemble_dir, args.calibration_metrics_dir,
                    calibration_target=args.calibration_target,
                )
            ]
        except ValueError as exc:
            p.error(str(exc))

    print(f"\nModels: {models}", flush=True)

    for model_name in models:
        if not is_uq_energy_model(model_name):
            print(f"  skip {model_name!r}: unsupported UQ model", file=sys.stderr)
            continue

        print(f"\n--- Model: {model_name} ---", flush=True)
        pkl_path, t_used = resolve_ensemble_pickle(
            model_name, args.ensemble_dir, args.temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_target=args.calibration_target,
        )
        print(f"  Ensemble pickle: {pkl_path}  (T={t_used:g})", flush=True)

        ens_dict = load_ensemble_pickle(pkl_path)
        ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)
        ensemble_shuffled = _shuffle_ensemble(ensemble, args.seed)
        print(
            f"  Shuffled ensemble (seed={args.seed}): {ensemble.shape[0]} members; "
            f"target {args.n_samples} successful",
            flush=True,
        )

        calc_obj, set_params_fn, load_name = build_uq_calculator(
            model_name, extra_kw=cli_hyperparams or None,
        )
        print(f"  Calculator: {load_name}", flush=True)

        energy_ensemble, n_ok = evaluate_ensemble_energies(
            calc_obj, ensemble_shuffled, args.n_samples, result_frames,
            set_params_fn=set_params_fn,
        )
        print(f"  {n_ok} successful ensemble member(s)", flush=True)

        if hasattr(calc_obj, "close"):
            calc_obj.close()

        if n_ok == 0:
            print("  no successful ensemble members; skipping model", file=sys.stderr)
            continue

        for stacking, geom in stack_geoms.items():
            per_sample_constants = collect_per_sample_constants(geom, energy_ensemble, n_ok)
            if not per_sample_constants:
                print(f"  {stacking}: incomplete mode set — skipping model constants")
                continue

            model_mean_std = mean_std_constants(per_sample_constants)

            print_model_elastic_constants(
                model_name, stacking, dft_constants.get(stacking, {}), model_mean_std,
            )

            plot_stack_elastic_moduli(
                model_name, stacking, geom,
                dft_energies_by_frame, energy_ensemble,
                figures_dir=figures_dir, dpi=args.dpi,
            )
            plot_elastic_constants_histograms(
                model_name, stacking, per_sample_constants,
                dft_constants.get(stacking, {}),
                figures_dir=figures_dir, dpi=args.dpi,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
