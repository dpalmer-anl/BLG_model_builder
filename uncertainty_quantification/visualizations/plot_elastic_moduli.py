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
   In-plane and out-of-plane Poisson ratios follow from the same stiffnesses::

       ν₁₂ = (C₁₂ C₃₃ − C₁₃²) / (C₁₁ C₃₃ − C₁₃²)
       ν₁₃ = C₁₃ (C₁₁ − C₁₂) / (C₁₁ C₃₃ − C₁₃²)

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

A separate figure shows histograms of each elastic constant and Poisson
ratio over the ensemble, with the DFT value marked as a solid vertical line
(bilayer AA/AB).  Graphite uses a **z-periodic** Bernal unit cell
(``c = 2 d``, no vacuum).  The equilibrium interlayer separation is found
first with the **best-fit** POD parameters (quadratic ``E(d)`` scan), then
elastic-constant structures are built at that ``d_eq`` and evaluated with
the MCMC ensemble.  Graphite histograms include translucent bars for
experimental bulk-graphite ranges (no DFT overlay yet).

Output:
  ``figures/<model>_<stacking>_elastic_constants.png``
  ``figures/<model>_<stacking>_elastic_constants_histogram.png``
  ``figures/<model>_graphite_elastic_constants.png``
  ``figures/<model>_graphite_elastic_constants_histogram.png``
  ``figures/POD_hyperparam_elastic_moduli_<stacking>.png`` (with ``--plot-hyperparam-sweep``)

Examples
--------
::

    python visualizations/plot_elastic_moduli.py --models POD_energy
    python visualizations/plot_elastic_moduli.py \\
        --models 'POD_energy_POD_index*' --n-samples 200
    python visualizations/plot_elastic_moduli.py --plot-hyperparam-sweep \\
        --hyperparam-stacking AB --hyperparam-n-samples 30
    python visualizations/plot_elastic_moduli.py --print-beefvdw-elastic

With no ``--models``, uses the ``POD_energy_POD_index*`` ensemble with the
lowest NLL on the saved calibration grid.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from blg_model_builder.pod_model_selection import (  # noqa: E402
    load_pod_search_results,
    pod_energy_ensemble_names_from_csv,
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
    mcmc_kw_for_model,
)

from blg_model_builder.geom_tools import get_bilayer_atoms  # noqa: E402
from blg_model_builder.potentials import (  # noqa: E402
    PODD3LammpsCalculator,
    PODLammpsCalculator,
    ncoeff_from_params,
)
from blg_model_builder.strain_data import LAT_CON  # noqa: E402

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
DEFAULT_BEEFVDW_XYZ = (
    REPO_ROOT / "data" / "blg_elastic_constants_structures_beefvdw.xyz"
)

DEFAULT_N_SAMPLES = 500
DEFAULT_HYPERPARAM_N_SAMPLES = 50
DEFAULT_ENSEMBLE_SHUFFLE_SEED = 0
DEFAULT_HYPERPARAM_STACKING = "AB"

# POD hyperparameter grid (matches ``pod_hyperparameter_search.GRID``).
_POD_TWO_BODY_RADIAL_GRID = list(range(6, 14, 1))
_POD_THREE_BODY_RADIAL_GRID = (4, 6, 8, 10)
_POD_THREE_BODY_ANGULAR_GRID = (4, 6, 8)

EV_A3_TO_GPA = 160.21766208
MODES = ("A", "B", "C", "D", "E")
CONSTANT_NAMES = ("C11", "C12", "C13", "C33", "C44")
POISSON_NAMES = ("nu12", "nu13")
HIST_NAMES = CONSTANT_NAMES + POISSON_NAMES
CONSTANT_TITLE = {
    "C11": r"$C_{11}$",
    "C12": r"$C_{12}$",
    "C13": r"$C_{13}$",
    "C33": r"$C_{33}$",
    "C44": r"$C_{44}$",
    "nu12": r"$\nu_{12}$",
    "nu13": r"$\nu_{13}$",
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

# Graphite: z-periodic Bernal unit cell (``get_bilayer_atoms`` with ``c = 2 d``).
GRAPHITE_STACKING = "graphite"
GRAPHITE_D0_LITERATURE = 3.35583  # Å — DD-TETB graphite training default
GRAPHITE_D_SCAN = np.linspace(3.0, 4.0, 41)
GRAPHITE_DELTAS = (-0.01, -0.005, 0.0, 0.005, 0.01)

# Experimental bulk-graphite stiffness ranges (GPa) for histogram overlays.
GRAPHITE_EXPERIMENTAL_GPA: Dict[str, Tuple[float, float]] = {
    "C11": (1060.0, 1109.0),
    "C12": (139.0, 190.0),
    "C13": (0.0, 15.0),
    "C33": (36.5, 38.7),
    "C44": (0.27, 4.95),
}


@dataclass(frozen=True)
class HyperparamElasticRecord:
    model_name: str
    pod_index: int
    two_body_radial: int
    three_body_radial: int
    three_body_angular: int
    rcut: float
    mean_std_gpa: Dict[str, Tuple[float, float]]


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
    valid = np.isfinite(deltas) & np.isfinite(energies)
    deltas = deltas[valid]
    energies = energies[valid]
    if deltas.size < 3 or np.unique(deltas).size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
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
    out = {"C11": c11, "C12": c12, "C13": c13, "C33": c33, "C44": c44}
    out.update(poisson_ratios_from_constants(out))
    return out


def poisson_ratios_from_constants(constants: Dict[str, float]) -> Dict[str, float]:
    """In-plane ``ν₁₂`` and out-of-plane ``ν₁₃`` from hexagonal Voigt Cᵢⱼ."""
    c11 = float(constants["C11"])
    c12 = float(constants["C12"])
    c13 = float(constants["C13"])
    c33 = float(constants["C33"])
    den = c11 * c33 - c13 * c13
    if not np.isfinite(den) or abs(den) < 1e-30:
        return {"nu12": float("nan"), "nu13": float("nan")}
    nu12 = (c12 * c33 - c13 * c13) / den
    nu13 = (c13 * (c11 - c12)) / den
    return {"nu12": float(nu12), "nu13": float(nu13)}


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
# Graphite (z-periodic Bernal unit cell)
# ----------------------------------------------------------------------
def _is_pod_family_model(model_name: str) -> bool:
    return model_name.startswith("POD_energy") or model_name.startswith("PODD3_energy")


def _pod_family_load_name(model_name: str) -> str:
    if model_name.startswith("PODD3_energy"):
        return "PODD3_energy"
    if model_name.startswith("POD_energy"):
        return "POD_energy"
    raise ValueError(f"Not a POD-family model: {model_name!r}")


def _pod_best_fit_npz(model_name: str, extra_kw: Optional[Dict]) -> Path:
    load_name = _pod_family_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), **(extra_kw or {})}
    pod_hp = data_kw.get("pod_hyperparams")
    if not isinstance(pod_hp, dict) or not pod_hp:
        raise ValueError(
            f"Could not resolve POD hyperparameters for {model_name!r}."
        )
    hyperparams = dict(pod_hp)
    hyperparams.setdefault("species", ["C"])
    ncoeffs = ncoeff_from_params(hyperparams)
    regularization = float(data_kw.get("regularization", 1e-12))
    include_intra = bool(data_kw.get("include_intralayer", False))
    pod_hash = str(data_kw.get("pod_hash", "")).strip()
    reg_tag = f"reg{regularization:.0e}"
    intra_tag = "_intra" if include_intra else ""
    hash_tag = f"_{pod_hash}" if pod_hash else ""
    path = (
        UQ_DIR
        / "best_fit_params"
        / f"{load_name}_{ncoeffs}_{reg_tag}{intra_tag}{hash_tag}_best_fit_params.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"{load_name} best-fit parameters not found: {path}")
    return path


def build_pod_best_fit_calculator(
    model_name: str,
    extra_kw: Optional[Dict] = None,
):
    """POD / PODD3 calculator with cached best-fit coefficients."""
    load_name = _pod_family_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), **(extra_kw or {})}
    pod_hp = data_kw.get("pod_hyperparams")
    if not isinstance(pod_hp, dict) or not pod_hp:
        raise ValueError(
            f"Could not resolve POD hyperparameters for {model_name!r}."
        )
    hyperparams = dict(pod_hp)
    hyperparams.setdefault("species", ["C"])
    rcut = float(data_kw.get("pod_cutoff", data_kw.get("rcut", 6.0)))
    params = np.asarray(
        np.load(_pod_best_fit_npz(model_name, extra_kw))["params"], dtype=float,
    )
    if load_name == "PODD3_energy":
        return PODD3LammpsCalculator(
            hyperparams,
            params,
            elements=["C"],
            cutoff=rcut,
            d3_damping=str(data_kw.get("d3_damping", "zerom")),
            d3_functional=str(data_kw.get("d3_functional", "pbe")),
            d3_cutoff=float(data_kw.get("d3_cutoff", 30.0)),
            d3_cn_cutoff=float(data_kw.get("d3_cn_cutoff", 20.0)),
        )
    return PODLammpsCalculator(
        hyperparams, params, elements=["C"], cutoff=rcut,
    )


def evaluate_calc_energies(calc_obj, atoms_list: Sequence) -> np.ndarray:
    """Total energies (eV) on a list of structures."""
    if not atoms_list:
        return np.array([], dtype=float)
    if hasattr(calc_obj, "prepare_batch"):
        calc_obj.prepare_batch(list(atoms_list))
        energies, _ = calc_obj.evaluate_batch()
        return np.asarray(energies, dtype=float).ravel()
    return np.asarray(
        [float(calc_obj.get_potential_energy(a)) for a in atoms_list],
        dtype=float,
    )


def equilibrium_layer_sep_quadratic(
    sep: np.ndarray,
    energy: np.ndarray,
    n_local: int = 5,
) -> float:
    """Equilibrium interlayer separation from a quadratic ``E(d)`` fit."""
    sep = np.asarray(sep, dtype=float).ravel()
    energy = np.asarray(energy, dtype=float).ravel()
    m = np.isfinite(sep) & np.isfinite(energy)
    sep, energy = sep[m], energy[m]
    if sep.size < 3:
        return float(sep[np.argmin(energy)]) if sep.size else float("nan")
    order = np.argsort(energy)
    n_use = min(int(n_local), sep.size)
    ix = order[:n_use]
    xs = sep[ix]
    ys = energy[ix]
    j = np.argsort(xs)
    xs, ys = xs[j], ys[j]
    try:
        xc = xs - np.mean(xs)
        a2, a1, _a0 = np.polyfit(xc, ys, 2, rcond=None)
        d_min = float(np.mean(xs) - 0.5 * a1 / a2)
    except np.linalg.LinAlgError:
        return float(xs[np.argmin(ys)])
    if a2 <= 0:
        return float(xs[np.argmin(ys)])
    if d_min < xs.min() or d_min > xs.max():
        return float(xs[np.argmin(ys)])
    return d_min


def make_graphite_unit_cell(d: float, a: float = LAT_CON, sc: int = 1):
    """Bernal graphite primitive cell, periodic in *z* with ``c = 2 d``."""
    c = 2.0 * float(d)
    return get_bilayer_atoms(float(d), 0.0, a=float(a), c=c, sc=int(sc))


def _graphite_layer_masks(atoms) -> Tuple[np.ndarray, np.ndarray]:
    z = atoms.positions[:, 2]
    zmid = 0.5 * (z.min() + z.max())
    bottom = z < zmid
    top = ~bottom
    return bottom, top


def apply_graphite_strain_mode(atoms0, mode: str, delta: float, d0: float):
    """
    Apply one elastic strain mode to a z-periodic graphite unit cell.

    In-plane modes scale ``a₁/a₂``.  Out-of-plane mode C scales the ``c`` axis.
    Mode D combines in-plane and ``c`` scaling; mode E applies interlayer shear
    via opposite layer shifts in ``y`` (same path as bilayer training data).
    """
    atoms = atoms0.copy()
    bottom, top = _graphite_layer_masks(atoms)

    if mode in ("A", "B", "D"):
        e1 = float(delta)
        e2 = float(delta) if mode == "B" else 0.0
        F = np.diag([1.0 + e1, 1.0 + e2, 1.0])
        atoms.set_cell(atoms.cell[:] @ F.T, scale_atoms=True)

    if mode in ("C", "D"):
        cell = atoms.cell.copy()
        cell[2, :] *= 1.0 + float(delta)
        atoms.set_cell(cell, scale_atoms=True)

    if mode == "E":
        dy = 0.5 * float(delta) * float(d0)
        pos = atoms.get_positions()
        pos[bottom, 1] -= dy
        pos[top, 1] += dy
        atoms.set_positions(pos)

    return atoms


def build_graphite_elastic_frames(
    d0: float,
    a: float = LAT_CON,
) -> List:
    """Strained graphite frames at equilibrium interlayer separation ``d0``."""
    atoms0 = make_graphite_unit_cell(d0, a=a)
    frames: List = []
    for mode in MODES:
        for delta in GRAPHITE_DELTAS:
            deformed = apply_graphite_strain_mode(atoms0, mode, delta, d0)
            deformed.info["stacking"] = GRAPHITE_STACKING
            deformed.info["mode"] = mode
            deformed.info["delta"] = float(delta)
            deformed.info["d0"] = float(d0)
            frames.append(deformed)
    return frames


def build_graphite_stack_geometry(frames: Sequence) -> StackGeometry:
    """Mode groupings and unit-cell volume ``V = A c`` for graphite frames."""
    by_mode: Dict[str, List[Tuple[float, int]]] = {}
    for i, atoms in enumerate(frames):
        by_mode.setdefault(atoms.info["mode"], []).append(
            (float(atoms.info["delta"]), i),
        )

    ref0 = next(a for a in frames if abs(float(a.info["delta"])) < 1e-15)
    area = float(np.linalg.norm(np.cross(ref0.cell[0], ref0.cell[1])))
    c_len = float(np.linalg.norm(ref0.cell[2]))
    geom = StackGeometry(volume=area * c_len)

    for mode in MODES:
        pts = sorted(by_mode.get(mode, []), key=lambda t: t[0])
        if len(pts) < 3:
            print(f"  WARNING: only {len(pts)} graphite points for mode {mode}, skipping")
            continue
        geom.mode_deltas[mode] = np.array([p[0] for p in pts], dtype=float)
        geom.mode_frame_indices[mode] = [p[1] for p in pts]
    return geom


def find_graphite_equilibrium_layer_sep(
    model_name: str,
    extra_kw: Optional[Dict] = None,
    *,
    d_scan: np.ndarray = GRAPHITE_D_SCAN,
    a: float = LAT_CON,
) -> float:
    """
    Equilibrium graphite interlayer separation from best-fit POD ``E(d)``.

    Uses a z-periodic Bernal unit cell at each separation in ``d_scan``.
    """
    if not _is_pod_family_model(model_name):
        raise ValueError(
            f"Graphite d_eq requires a POD-family model with best-fit cache; "
            f"got {model_name!r}."
        )
    calc = build_pod_best_fit_calculator(model_name, extra_kw)
    try:
        d_scan = np.asarray(d_scan, dtype=float).ravel()
        atoms_list = [make_graphite_unit_cell(d, a=a) for d in d_scan]
        energies = evaluate_calc_energies(calc, atoms_list)
        d_eq = equilibrium_layer_sep_quadratic(d_scan, energies)
        if not np.isfinite(d_eq):
            raise ValueError("quadratic fit returned non-finite d_eq")
        return float(d_eq)
    finally:
        if hasattr(calc, "close"):
            calc.close()


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
def print_reference_elastic_constants(
    stacking: str,
    constants: Dict[str, float],
    *,
    source_label: str = "DFT",
    compare_to_reference: bool = False,
) -> None:
    print(f"\n{stacking} {source_label} elastic constants (Voigt, eV/Å³ → GPa):")
    for name in CONSTANT_NAMES:
        val = constants.get(name, float("nan"))
        gpa = val * EV_A3_TO_GPA
        line = f"  {name} = {gpa:10.3f} GPa   ({val:.6f} eV/Å³)"
        if compare_to_reference and name in REFERENCE_GPA and np.isfinite(gpa):
            ref = REFERENCE_GPA[name]
            line += f"    [lit. {ref:.2f} GPa, Δ={gpa - ref:+.1f}]"
        print(line)
    print(f"{stacking} {source_label} Poisson ratios (dimensionless):")
    for name in POISSON_NAMES:
        val = constants.get(name, float("nan"))
        print(f"  {name} = {val:10.5f}")


def print_dft_elastic_constants(stacking: str, constants: Dict[str, float]) -> None:
    print_reference_elastic_constants(
        stacking,
        constants,
        source_label="DFT",
        compare_to_reference=True,
    )


def compute_elastic_constants_from_result_frames(
    ref_frames: Sequence,
    records: Sequence[dict],
    result_frames: Optional[Sequence] = None,
    *,
    energies_by_frame: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract Voigt stiffnesses from matched reference labels and frame energies."""
    if energies_by_frame is None:
        if result_frames is None:
            raise ValueError("provide either result_frames or energies_by_frame")
        energies_by_frame = np.array(
            [a.get_potential_energy() for a in result_frames], dtype=float,
        )
    else:
        energies_by_frame = np.asarray(energies_by_frame, dtype=float)
    stack_geoms = build_stack_geometry(ref_frames, records)
    constants_by_stack: Dict[str, Dict[str, float]] = {}
    for stacking, geom in stack_geoms.items():
        constants = elastic_constants_for_energies(geom, energies_by_frame)
        if constants is None:
            print(
                f"\n{stacking}: incomplete mode set — skipping elastic constants",
                flush=True,
            )
            continue
        constants_by_stack[stacking] = constants
    return constants_by_stack


def align_result_energies_to_reference_frames(
    reference_frames: Sequence,
    result_frames: Sequence,
    *,
    max_dist: float = 0.25,
) -> np.ndarray:
    """
    Map *result_frames* energies onto *reference_frames* indices by geometry.

    Uses greedy one-to-one assignment on the Cartesian fingerprint distance,
    preferring equal frame indices when they are already close.  This lets
    alternate result sets (e.g. BEEF-vdW) reuse labels established from a
    complete reference/DFT match even when the alternate file has fewer frames
    or slightly relaxed geometries.
    """
    ref_fps = [fingerprint(a) for a in reference_frames]
    res_fps = [fingerprint(a) for a in result_frames]
    used_result: set[int] = set()
    energies = np.full(len(reference_frames), np.nan, dtype=float)

    for i, rfp in enumerate(ref_fps):
        best_j: Optional[int] = None
        best_dist = float("inf")

        if i < len(result_frames) and i not in used_result:
            dist_same = float(np.linalg.norm(np.asarray(rfp) - np.asarray(res_fps[i])))
            if dist_same < best_dist:
                best_dist = dist_same
                best_j = i

        for j, sfp in enumerate(res_fps):
            if j in used_result:
                continue
            dist = float(np.linalg.norm(np.asarray(rfp) - np.asarray(sfp)))
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is None or best_dist > max_dist:
            continue

        energies[i] = float(result_frames[best_j].get_potential_energy())
        used_result.add(best_j)

    return energies


def print_beefvdw_elastic_constants(
    ref_frames: Sequence,
    dft_frames: Sequence,
    records: Sequence[dict],
    beefvdw_xyz: Path,
) -> Dict[str, Dict[str, float]]:
    """Print elastic constants from BEEF-vdW energies aligned to DFT labels."""
    print(f"\nLoading BEEF-vdW structures from {beefvdw_xyz}", flush=True)
    beefvdw_frames = read(str(beefvdw_xyz), index=":")
    print(
        f"  aligned {len(beefvdw_frames)} BEEF-vdW frame(s) to "
        f"{len(dft_frames)} DFT-labelled frame(s)",
        flush=True,
    )
    beefvdw_energies = align_result_energies_to_reference_frames(
        dft_frames,
        beefvdw_frames,
    )
    n_matched = int(np.sum(np.isfinite(beefvdw_energies)))
    print(
        f"  matched {n_matched}/{len(dft_frames)} DFT-labelled frame(s) to "
        f"BEEF-vdW energies",
        flush=True,
    )
    n_missing = len(dft_frames) - n_matched
    if n_missing:
        print(
            f"  WARNING: {n_missing} DFT-labelled frame(s) have no BEEF-vdW match",
            flush=True,
        )

    constants_by_stack = compute_elastic_constants_from_result_frames(
        ref_frames,
        records,
        energies_by_frame=beefvdw_energies,
    )
    if not constants_by_stack:
        print("  No complete BEEF-vdW elastic-constant sets could be extracted.", flush=True)
    for stacking, constants in constants_by_stack.items():
        print_reference_elastic_constants(
            stacking,
            constants,
            source_label="BEEF-vdW",
        )
    return constants_by_stack


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
    print(f"{stacking} {model_name} Poisson ratios (mean ± std over ensemble):")
    for name in POISSON_NAMES:
        mu, sig = model_mean_std.get(name, (float("nan"), float("nan")))
        dft_val = dft_constants.get(name, float("nan"))
        line = f"  {name} = {mu:10.5f} ± {sig:.5f}"
        if np.isfinite(dft_val):
            line += f"    [DFT {dft_val:.5f}, Δ={mu - dft_val:+.5f}]"
        print(line)


def print_graphite_elastic_constants(
    model_name: str,
    model_mean_std: Dict[str, Tuple[float, float]],
    *,
    d_eq: float,
    experimental_gpa: Dict[str, Tuple[float, float]] = GRAPHITE_EXPERIMENTAL_GPA,
) -> None:
    """Print graphite POD ensemble constants vs experimental GPa ranges."""
    print(
        f"\n{GRAPHITE_STACKING} {model_name} "
        f"(z-periodic, d_eq = {d_eq:.4f} Å from best-fit POD)",
        flush=True,
    )
    print(f"{GRAPHITE_STACKING} elastic constants (mean ± std over ensemble, GPa):")
    for name in CONSTANT_NAMES:
        mu, sig = model_mean_std.get(name, (float("nan"), float("nan")))
        mu_gpa, sig_gpa = mu * EV_A3_TO_GPA, sig * EV_A3_TO_GPA
        line = f"  {name} = {mu_gpa:10.3f} ± {sig_gpa:.3f} GPa"
        exp_range = experimental_gpa.get(name)
        if exp_range is not None:
            lo, hi = exp_range
            line += f"    [exp. {lo:.1f}–{hi:.1f} GPa]"
        print(line)
    print(f"{GRAPHITE_STACKING} {model_name} Poisson ratios (mean ± std over ensemble):")
    for name in POISSON_NAMES:
        mu, sig = model_mean_std.get(name, (float("nan"), float("nan")))
        print(f"  {name} = {mu:10.5f} ± {sig:.5f}")


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
    """Mean and std of each elastic constant (eV/Å³) and Poisson ratio."""
    model_mean_std: Dict[str, Tuple[float, float]] = {}
    for name in HIST_NAMES:
        vals = np.array([c[name] for c in per_sample_constants], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            model_mean_std[name] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            model_mean_std[name] = (float("nan"), float("nan"))
    return model_mean_std


def _pod_valid_two_body_radial_values(
    three_body_radial: int,
    three_body_angular: int,
) -> List[int]:
    out: List[int] = []
    for n2 in _POD_TWO_BODY_RADIAL_GRID:
        if three_body_radial > n2:
            continue
        if three_body_angular > three_body_radial:
            continue
        out.append(int(n2))
    return out


def _pod_valid_hyperparameter_groups() -> List[Tuple[int, int]]:
    groups: List[Tuple[int, int]] = []
    for n3r in _POD_THREE_BODY_RADIAL_GRID:
        for n3a in _POD_THREE_BODY_ANGULAR_GRID:
            if _pod_valid_two_body_radial_values(n3r, n3a):
                groups.append((int(n3r), int(n3a)))
    return groups


def _pod_index_from_model_name(model_name: str) -> Optional[int]:
    m = re.match(r"^POD_energy_POD_index_(\d+)_", model_name, re.I)
    return int(m.group(1)) if m else None


def _hyperparam_row_for_model(model_name: str, search_df) -> Optional[Dict[str, Any]]:
    pod_idx = _pod_index_from_model_name(model_name)
    if pod_idx is None or pod_idx < 0 or pod_idx >= len(search_df):
        return None
    row = search_df.iloc[int(pod_idx)]
    return {
        "pod_index": int(pod_idx),
        "two_body_radial": int(row["two_body_radial"]),
        "three_body_radial": int(row["three_body_radial"]),
        "three_body_angular": int(row["three_body_angular"]),
        "rcut": float(row["rcut"]),
        "hash": str(row["hash"]),
    }


def _elastic_mean_std_gpa(
    mean_std: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for name in CONSTANT_NAMES:
        mu, sig = mean_std.get(name, (float("nan"), float("nan")))
        out[name] = (float(mu) * EV_A3_TO_GPA, float(sig) * EV_A3_TO_GPA)
    return out


def evaluate_model_elastic_mean_std(
    model_name: str,
    *,
    stack_geoms: Dict[str, StackGeometry],
    result_frames: Sequence,
    stacking: str,
    ensemble_dir: str,
    calibration_metrics_dir: str,
    calibration_target: str,
    temperature: Optional[float],
    n_samples: int,
    seed: int,
    cli_hyperparams: Optional[dict],
) -> Optional[Dict[str, Tuple[float, float]]]:
    """Return mean ± std elastic constants (eV/Å³) for one stacking."""
    if stacking not in stack_geoms:
        return None
    if not is_uq_energy_model(model_name):
        return None

    pkl_path, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_target=calibration_target,
    )
    ens_dict = load_ensemble_pickle(pkl_path)
    ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)
    ensemble_shuffled = _shuffle_ensemble(ensemble, seed)

    calc_obj, set_params_fn, _load_name = build_uq_calculator(
        model_name, extra_kw=cli_hyperparams or None,
    )
    energy_ensemble, n_ok = evaluate_ensemble_energies(
        calc_obj,
        ensemble_shuffled,
        n_samples,
        result_frames,
        set_params_fn=set_params_fn,
    )
    if hasattr(calc_obj, "close"):
        calc_obj.close()
    if n_ok == 0:
        print(f"    no successful ensemble members (T={t_used:g})", file=sys.stderr)
        return None

    geom = stack_geoms[stacking]
    per_sample_constants = collect_per_sample_constants(geom, energy_ensemble, n_ok)
    if not per_sample_constants:
        print(f"    incomplete mode set for {stacking}", file=sys.stderr)
        return None
    return mean_std_constants(per_sample_constants)


def collect_hyperparam_elastic_records(
    model_names: Sequence[str],
    search_df,
    *,
    stack_geoms: Dict[str, StackGeometry],
    result_frames: Sequence,
    stacking: str,
    ensemble_dir: str,
    calibration_metrics_dir: str,
    calibration_target: str,
    temperature: Optional[float],
    n_samples: int,
    seed: int,
    cli_hyperparams: Optional[dict],
) -> List[HyperparamElasticRecord]:
    records: List[HyperparamElasticRecord] = []
    for i, model_name in enumerate(model_names, start=1):
        hp = _hyperparam_row_for_model(model_name, search_df)
        if hp is None:
            print(f"  [{i}/{len(model_names)}] skip {model_name}: no CSV row", flush=True)
            continue
        print(
            f"  [{i}/{len(model_names)}] {model_name}  "
            f"(n2={hp['two_body_radial']}, n3r={hp['three_body_radial']}, "
            f"l3a={hp['three_body_angular']}, rcut={hp['rcut']:.0f})",
            flush=True,
        )
        mean_std = evaluate_model_elastic_mean_std(
            model_name,
            stack_geoms=stack_geoms,
            result_frames=result_frames,
            stacking=stacking,
            ensemble_dir=ensemble_dir,
            calibration_metrics_dir=calibration_metrics_dir,
            calibration_target=calibration_target,
            temperature=temperature,
            n_samples=n_samples,
            seed=seed,
            cli_hyperparams=cli_hyperparams,
        )
        if mean_std is None:
            continue
        records.append(
            HyperparamElasticRecord(
                model_name=model_name,
                pod_index=int(hp["pod_index"]),
                two_body_radial=int(hp["two_body_radial"]),
                three_body_radial=int(hp["three_body_radial"]),
                three_body_angular=int(hp["three_body_angular"]),
                rcut=float(hp["rcut"]),
                mean_std_gpa=_elastic_mean_std_gpa(mean_std),
            )
        )
    return records


def plot_elastic_moduli_hyperparam_sweep(
    records: Sequence[HyperparamElasticRecord],
    *,
    stacking: str,
    figures_dir: Path,
    dft_constants: Optional[Dict[str, float]] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Plot ensemble mean ± std of each elastic constant vs 2-body radial count.

    Each curve is a fixed ``(three_body_radial, three_body_angular)`` pair from
    ``pod_hyperparam_search.csv``.
    """
    if not records:
        print("  No hyperparameter elastic records; skipping sweep figure.", flush=True)
        return None

    groups: Dict[Tuple[int, int], List[HyperparamElasticRecord]] = {}
    for rec in records:
        key = (rec.three_body_radial, rec.three_body_angular)
        groups.setdefault(key, []).append(rec)

    dft_constants = dft_constants or {}
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5))
    axes_flat = axes.ravel()
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C3", "C4", "C5"])
    sorted_groups = sorted(groups)

    for ax_idx, name in enumerate(CONSTANT_NAMES):
        ax = axes_flat[ax_idx]
        dft_raw = float(dft_constants.get(name, float("nan")))
        if np.isfinite(dft_raw):
            ax.axhline(
                dft_raw * EV_A3_TO_GPA,
                color="C2",
                lw=2.0,
                ls="--",
                zorder=1,
            )

        for gi, (n3r, n3a) in enumerate(sorted_groups):
            color = prop_cycle[gi % len(prop_cycle)]
            pts = sorted(groups[(n3r, n3a)], key=lambda r: r.two_body_radial)
            x = np.array([p.two_body_radial for p in pts], dtype=float)
            y = np.array([p.mean_std_gpa[name][0] for p in pts], dtype=float)
            yerr = np.array([p.mean_std_gpa[name][1] for p in pts], dtype=float)
            expected_x = _pod_valid_two_body_radial_values(n3r, n3a)
            missing_x = sorted(set(expected_x) - set(int(v) for v in x))
            if missing_x:
                print(
                    f"  hyperparam sweep {name}: group (n3r={n3r}, l3a={n3a}) "
                    f"missing 2-body radial value(s) {missing_x}",
                    flush=True,
                )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                lw=1.8,
                markersize=6,
                capsize=3,
                color=color,
                zorder=3,
            )

        ax.set_title(CONSTANT_TITLE[name], fontdict=CSFONT)
        ax.set_xlabel(r"$n_{\mathrm{rad}}$ (2-body radial basis functions)", fontdict=CSFONT)
        ax.set_ylabel("GPa", fontdict=CSFONT)
        ax.set_xticks(list(_POD_TWO_BODY_RADIAL_GRID))
        ax.set_xlim(min(_POD_TWO_BODY_RADIAL_GRID) - 0.5, max(_POD_TWO_BODY_RADIAL_GRID) + 0.5)
        ax.grid(True, alpha=0.3)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    full_handles = [
        plt.Line2D([0], [0], color=prop_cycle[gi % len(prop_cycle)], marker="o", lw=1.8, markersize=6)
        for gi, (n3r, n3a) in enumerate(sorted_groups)
    ]
    full_labels = [
        rf"$N_{{\mathrm{{3b}}}}={int(n3r) * int(n3a)}$"
        for (n3r, n3a) in sorted_groups
    ]
    if any(np.isfinite(float(dft_constants.get(n, float("nan")))) for n in CONSTANT_NAMES):
        full_handles.append(plt.Line2D([0], [0], color="C2", lw=2.0, ls="--"))
        full_labels.append("DFT")
    legend_ax.legend(
        full_handles,
        full_labels,
        loc="center left",
        prop={"family": CSFONT["fontname"], "size": 14},
        frameon=True,
    )

    fig.suptitle(
        f"POD hyperparameter sweep — {stacking} elastic moduli (ensemble mean ± std)",
        fontsize=CSFONT["size"],
        fontname=CSFONT["fontname"],
    )
    fig.tight_layout()
    out = figures_dir / f"POD_hyperparam_elastic_moduli_{stacking}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  wrote {out}  ({len(records)} models, {len(groups)} 3-body group(s))",
        flush=True,
    )
    return out


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_stack_elastic_moduli(
    model_name: str,
    stacking: str,
    geom: StackGeometry,
    model_energy_ensemble_by_frame: np.ndarray,
    *,
    dft_energies_by_frame: Optional[np.ndarray] = None,
    figures_dir: Path,
    dpi: int = 150,
) -> Path:
    """
    Per-mode subplots of ensemble mean ± std energy vs strain, optionally
    overlaid on DFT, with a single shared legend in the unused bottom-right
    subplot slot.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes_flat = axes.ravel()

    legend_handles = None
    legend_labels = None
    show_dft = dft_energies_by_frame is not None

    for ax_idx, mode in enumerate(MODES):
        ax = axes_flat[ax_idx]
        if mode not in geom.mode_deltas:
            ax.set_title(f"mode {mode} (missing)", fontdict=CSFONT)
            ax.axis("off")
            continue

        deltas = geom.mode_deltas[mode]
        idx = geom.mode_frame_indices[mode]
        i0 = int(np.argmin(np.abs(deltas)))

        model_e = model_energy_ensemble_by_frame[idx, :]  # (npts, n_success)
        model_rel = model_e - model_e[i0:i0 + 1, :]
        mean_rel = np.mean(model_rel, axis=1)
        std_rel = np.std(model_rel, axis=1)

        (line_mean,) = ax.plot(deltas, mean_rel, "-", color="C0", lw=2.0, zorder=2, label="ensemble mean")
        band = ax.fill_between(
            deltas, mean_rel - std_rel, mean_rel + std_rel,
            alpha=0.3, color="C0", zorder=1, label="ensemble std",
        )

        line_dft = None
        if show_dft:
            dft_e = dft_energies_by_frame[idx]
            dft_rel = dft_e - dft_e[i0]
            (line_dft,) = ax.plot(deltas, dft_rel, "o-", color="C2", ms=7, zorder=3, label="DFT")

        if legend_handles is None:
            legend_handles = [line_mean, band]
            legend_labels = ["ensemble mean", "ensemble std"]
            if line_dft is not None:
                legend_handles.append(line_dft)
                legend_labels.append("DFT")

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

    fig.suptitle(f"{stacking} stacking", fontsize=CSFONT["size"], fontname=CSFONT["fontname"])
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
    *,
    dft_constants: Optional[Dict[str, float]] = None,
    experimental_gpa_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    figures_dir: Path,
    dpi: int = 150,
    n_bins: int = 30,
) -> Path:
    """Histogram of each Cᵢⱼ (GPa) and Poisson ratio with optional DFT / experiment."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.0))
    axes_flat = axes.ravel()

    dft_constants = dft_constants or {}
    experimental_gpa_ranges = experimental_gpa_ranges or {}

    legend_handles = None
    legend_labels = None

    for ax_idx, name in enumerate(HIST_NAMES):
        ax = axes_flat[ax_idx]
        raw = np.array([c[name] for c in per_sample_constants], dtype=float)
        is_poisson = name in POISSON_NAMES
        vals = raw if is_poisson else raw * EV_A3_TO_GPA
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.set_title(f"{CONSTANT_TITLE[name]} (no data)", fontdict=CSFONT)
            ax.axis("off")
            continue

        exp_range = experimental_gpa_ranges.get(name)
        exp_patch = None
        if exp_range is not None and not is_poisson:
            lo, hi = exp_range
            exp_patch = ax.axvspan(
                lo, hi, alpha=0.35, color="C3", zorder=0, label="experiment",
            )

        _n_hist, _edges, patches = ax.hist(
            vals,
            bins=n_bins,
            color="C0",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            label="ensemble",
            zorder=2,
        )

        dft_raw = float(dft_constants.get(name, float("nan")))
        dft_mark = dft_raw if is_poisson else dft_raw * EV_A3_TO_GPA
        dft_line = None
        if np.isfinite(dft_mark):
            dft_line = ax.axvline(
                dft_mark,
                color="C2",
                lw=2.5,
                linestyle="-",
                zorder=5,
                label="DFT",
            )

        if legend_handles is None:
            legend_handles = []
            legend_labels = []
            if exp_patch is not None:
                legend_handles.append(exp_patch)
                legend_labels.append("experiment")
            legend_handles.append(patches[0])
            legend_labels.append("ensemble")
            if dft_line is not None:
                legend_handles.append(dft_line)
                legend_labels.append("DFT")

        ax.set_xlabel("" if is_poisson else "GPa", fontdict=CSFONT)
        ax.set_ylabel("count", fontdict=CSFONT)
        ax.set_title(CONSTANT_TITLE[name], fontdict=CSFONT)
        ax.grid(True, alpha=0.3, axis="y")

    for ax in axes_flat[len(HIST_NAMES):-1]:
        ax.axis("off")

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
        f"{stacking} stacking",
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
    p.add_argument(
        "--print-beefvdw-elastic",
        action="store_true",
        help=(
            "Also compute and print elastic constants from "
            "data/blg_elastic_constants_structures_beefvdw.xyz (no figure overlay)."
        ),
    )
    p.add_argument(
        "--beefvdw-xyz",
        type=Path,
        default=DEFAULT_BEEFVDW_XYZ,
        help=(
            "BEEF-vdW elastic-constant structures for --print-beefvdw-elastic "
            "(default: data/blg_elastic_constants_structures_beefvdw.xyz)."
        ),
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--plot-hyperparam-sweep",
        action="store_true",
        help=(
            "Evaluate elastic moduli for every POD ensemble listed in "
            "pod_hyperparam_search.csv and plot mean ± std vs 2-body radial count."
        ),
    )
    p.add_argument(
        "--hyperparam-stacking",
        default=DEFAULT_HYPERPARAM_STACKING,
        help="Stacking label for the hyperparameter sweep figure (default: AB).",
    )
    p.add_argument(
        "--hyperparam-n-samples",
        type=int,
        default=DEFAULT_HYPERPARAM_N_SAMPLES,
        help="Successful ensemble draws per model for --plot-hyperparam-sweep.",
    )
    p.add_argument(
        "--hyperparam-csv",
        type=Path,
        default=None,
        help="POD hyperparameter search CSV (default: pod_hyperparam_search/pod_hyperparam_search.csv).",
    )
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
    beefvdw_xyz = Path(args.beefvdw_xyz)
    if not beefvdw_xyz.is_absolute():
        beefvdw_xyz = (
            REPO_ROOT / beefvdw_xyz
            if (REPO_ROOT / beefvdw_xyz).exists()
            else UQ_DIR / beefvdw_xyz
        )

    print(f"Loading reference labels from {ref_xyz}", flush=True)
    print(f"Loading DFT structures from {dft_xyz}", flush=True)
    ref_frames, result_frames, records = load_and_match(ref_xyz, dft_xyz)

    dft_constants = compute_elastic_constants_from_result_frames(
        ref_frames,
        records,
        result_frames,
    )
    for stacking, constants in dft_constants.items():
        print_dft_elastic_constants(stacking, constants)

    if args.print_beefvdw_elastic:
        if not beefvdw_xyz.is_file():
            p.error(f"BEEF-vdW elastic-constant file not found: {beefvdw_xyz}")
        print_beefvdw_elastic_constants(
            ref_frames,
            result_frames,
            records,
            beefvdw_xyz,
        )

    stack_geoms = build_stack_geometry(ref_frames, records)
    dft_energies_by_frame = np.array(
        [a.get_potential_energy() for a in result_frames], dtype=float,
    )

    if args.models:
        models = expand_model_patterns(args.models, args.ensemble_dir)
        if not models:
            p.error("No models matched --models patterns.")
    elif args.plot_hyperparam_sweep:
        models = []
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

    if models:
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

        # Graphite: z-periodic cell; d_eq from best-fit POD, then ensemble UQ.
        graphite_d_eq = float("nan")
        graphite_frames: List = []
        graphite_geom: Optional[StackGeometry] = None
        try:
            graphite_d_eq = find_graphite_equilibrium_layer_sep(
                model_name, cli_hyperparams or None,
            )
            print(
                f"  {GRAPHITE_STACKING} equilibrium layer separation "
                f"(best-fit POD): {graphite_d_eq:.4f} Å",
                flush=True,
            )
            graphite_frames = build_graphite_elastic_frames(graphite_d_eq)
            graphite_geom = build_graphite_stack_geometry(graphite_frames)
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"  {GRAPHITE_STACKING}: could not determine d_eq ({exc}); skipping graphite",
                file=sys.stderr,
            )

        eval_frames = list(result_frames)
        n_bilayer_frames = len(eval_frames)
        if graphite_frames:
            eval_frames.extend(graphite_frames)

        energy_ensemble, n_ok = evaluate_ensemble_energies(
            calc_obj, ensemble_shuffled, args.n_samples, eval_frames,
            set_params_fn=set_params_fn,
        )
        print(f"  {n_ok} successful ensemble member(s)", flush=True)

        if hasattr(calc_obj, "close"):
            calc_obj.close()

        if n_ok == 0:
            print("  no successful ensemble members; skipping model", file=sys.stderr)
            continue

        bilayer_energy = energy_ensemble[:n_bilayer_frames, :]
        graphite_energy = (
            energy_ensemble[n_bilayer_frames:, :]
            if graphite_frames
            else None
        )

        for stacking, geom in stack_geoms.items():
            per_sample_constants = collect_per_sample_constants(geom, bilayer_energy, n_ok)
            if not per_sample_constants:
                print(f"  {stacking}: incomplete mode set — skipping model constants")
                continue

            model_mean_std = mean_std_constants(per_sample_constants)

            print_model_elastic_constants(
                model_name, stacking, dft_constants.get(stacking, {}), model_mean_std,
            )

            plot_stack_elastic_moduli(
                model_name, stacking, geom,
                bilayer_energy,
                dft_energies_by_frame=dft_energies_by_frame,
                figures_dir=figures_dir, dpi=args.dpi,
            )
            plot_elastic_constants_histograms(
                model_name, stacking, per_sample_constants,
                dft_constants=dft_constants.get(stacking, {}),
                figures_dir=figures_dir, dpi=args.dpi,
            )

        if graphite_geom is not None and graphite_energy is not None:
            per_sample_graphite = collect_per_sample_constants(
                graphite_geom, graphite_energy, n_ok,
            )
            if per_sample_graphite:
                graphite_mean_std = mean_std_constants(per_sample_graphite)
                print_graphite_elastic_constants(
                    model_name, graphite_mean_std, d_eq=graphite_d_eq,
                )
                plot_stack_elastic_moduli(
                    model_name, GRAPHITE_STACKING, graphite_geom,
                    graphite_energy,
                    dft_energies_by_frame=None,
                    figures_dir=figures_dir, dpi=args.dpi,
                )
                plot_elastic_constants_histograms(
                    model_name, GRAPHITE_STACKING, per_sample_graphite,
                    experimental_gpa_ranges=GRAPHITE_EXPERIMENTAL_GPA,
                    figures_dir=figures_dir, dpi=args.dpi,
                )
            else:
                print(
                    f"  {GRAPHITE_STACKING}: incomplete mode set — skipping",
                    file=sys.stderr,
                )

    if args.plot_hyperparam_sweep:
        hyper_csv = args.hyperparam_csv
        if hyper_csv is not None and not hyper_csv.is_absolute():
            hyper_csv = UQ_DIR / "pod_hyperparam_search" / hyper_csv
        search_df = load_pod_search_results(hyper_csv)
        hyper_models = pod_energy_ensemble_names_from_csv(
            args.ensemble_dir,
            csv_path=hyper_csv,
            passing_only=False,
            require_fit_cache=False,
        )
        print(
            f"\n=== Hyperparameter elastic-moduli sweep "
            f"({len(hyper_models)} models, stacking={args.hyperparam_stacking}) ===",
            flush=True,
        )
        hp_records = collect_hyperparam_elastic_records(
            hyper_models,
            search_df,
            stack_geoms=stack_geoms,
            result_frames=result_frames,
            stacking=str(args.hyperparam_stacking),
            ensemble_dir=args.ensemble_dir,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_target=args.calibration_target,
            temperature=args.temperature,
            n_samples=int(args.hyperparam_n_samples),
            seed=int(args.seed),
            cli_hyperparams=cli_hyperparams or None,
        )
        plot_elastic_moduli_hyperparam_sweep(
            hp_records,
            stacking=str(args.hyperparam_stacking),
            figures_dir=figures_dir,
            dft_constants=dft_constants.get(str(args.hyperparam_stacking)),
            dpi=args.dpi,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
