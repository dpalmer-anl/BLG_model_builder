"""
Potential-energy surfaces for unstrained bilayer graphene (AB and AA).

**DFT sources** (``rVV10`` by default; ``MBD`` with ``--include-mbd``) come from
:func:`load_energy_data`.
Unstrained frames require both in-plane cell parameters to match the modal
equilibrium lattice constant ``a_ref`` within ``_STRAIN_TOL_ANG``.
For rVV10, frames with interlayer separation ``> 4.5`` Å are also pulled from
``data/TrainingData`` / ``data/TestData`` (the main xyz currently stops near
4.5 Å; those splits still have larger separations such as 6 and 7 Å).

Stacking from max per-atom min-distance between top and bottom layers (MIC xy):

- **AA**  : max_min < ``_STACK_AA_MAX_XY`` (≈ 0.01 Å)
- **AB**  : max_min > ``_STACK_AB_MIN_XY`` (≈ 1.2 Å)
- **SP**  : excluded

**QMC** data come from ``data/qmc.csv`` (stacking, ``d``, energy in eV/atom).

Models (``--models``) are evaluated on the **same** unstrained rVV10 ``Atoms``
objects as the rVV10 curves.

Two figures are written under ``--figures-dir``:

**Morse-/POD-shifted binding** (``…_pes_morse_shifted_3_to_8A.png``) — for DFT/QMC,
fit **separate** Morse potentials to AB and AA, shift by each stacking's
``E(∞)``, and plot for ``d ∈ [3, 8]`` Å.  For POD-family models, evaluate the
potential on a dense interlayer grid ``d ∈ [3, 8]`` (fixed in-plane cell /
stacking from an rVV10 template) and shift by ``E(d=8 Å)`` instead of a Morse
curve.  MBD is omitted unless ``--include-mbd`` is set.

**AB-equilibrium–shifted binding** (``…_pes_ab_eq_shifted_3_to_5A.png``) — same
units, ``d ∈ [3, 5]`` Å, but each data source is shifted so the AB well bottom is
zero: subtract ``E_AB(l_0)`` from both AB and AA (``l_0`` = AB equilibrium
separation for that source).

Morse form (eV/atom; DFT/QMC only)::

    E(d) = D_e [e^{-2a(d-d_e)} - 2 e^{-a(d-d_e)}] + E_∞

so ``E(∞) = E_∞`` and the shifted curves approach 0 at large separation.

Run from ``uncertainty_quantification``::

    python visualizations/plot_bilayer_graphene_pes.py
    python visualizations/plot_bilayer_graphene_pes.py --models POD_energy --POD-index 0
    python visualizations/plot_bilayer_graphene_pes.py --models 'POD_energy_POD_index*'
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import ase.io

from blg_model_builder.DataLoader import load_energy_data
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
    expand_ensemble_model_name,
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
REPO_ROOT = UQ_DIR.parent

_uq_dir = str(UQ_DIR)
if _uq_dir not in sys.path:
    sys.path.insert(0, _uq_dir)

from uq_model_runtime import (  # noqa: E402
    build_uq_calculator,
    is_uq_energy_model,
    mcmc_kw_for_model,
)
from blg_model_builder.potentials import (  # noqa: E402
    PODD3LammpsCalculator,
    PODLammpsCalculator,
    ncoeff_from_params,
)
from run_uq_propagation_relaxation import expand_models_for_relaxation  # noqa: E402

# ── stacking thresholds ────────────────────────────────────────────────────
# These are the *per-top-atom* minimum xy distances to any bottom atom (MIC).
# AA:  all top atoms sit directly above bottom atoms → max_min ≈ 0
# AB:  one top atom sits directly above a bottom atom, other above void → max_min ≈ bond
# SP:  intermediate configuration → max_min ≈ half bond (~0.7 Å) – excluded from PES
_STACK_AA_MAX_XY = 0.01    # Å  – strict: excludes GSFE sliding frames
_STACK_AB_MIN_XY = 1.2     # Å  – min threshold for AB identification

# In-plane strain tolerance: |cell param - reference| must be less than this
_STRAIN_TOL_ANG = 0.005    # Å  – separates consecutive strain-grid points (~0.01*a)

_COLOR_RVV10 = "#1f77b4"
_COLOR_MBD = "#9467bd"
_COLOR_QMC = "#d95f02"
_COLOR_MODEL_CYCLE = (
    "#2ca02c",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_MARKER_AB = "o"
_MARKER_AA = "s"

DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"
STACKINGS = ("AB", "AA")
DFT_LEVELS_DEFAULT = ("rVV10",)
POD_FAMILY_PREFIXES = ("POD_energy", "PODD3_energy")

# Morse-/POD-shifted plot window (Å)
XLIM_MORSE = (3.0, 8.0)
# AB-equilibrium–shifted plot window (Å)
XLIM_AB_EQ = (3.0, 5.0)

# Main ``strained_bilayer_graphene_rVV10.xyz`` tops out near 4.5 Å; Training/Test
# splits still contain unstrained frames at larger separations (e.g. 6 and 7 Å).
_LARGE_D_MIN_ANG = 4.5
_RVV10_LARGE_D_XYZ = (
    REPO_ROOT / "data" / "TrainingData" / "strained_bilayer_graphene_rVV10.xyz",
    REPO_ROOT / "data" / "TestData" / "strained_bilayer_graphene_rVV10.xyz",
)


# ── data structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UnstrainedFrames:
    """Unstrained structures for one stacking type, with total energies (eV)."""
    atoms: list
    dft_energies: np.ndarray


@dataclass(frozen=True)
class PesSeries:
    """Interlayer PES for one method/source (energies in eV/atom, unshifted)."""
    label: str
    color: str
    ab_d: np.ndarray
    ab_e: np.ndarray
    aa_d: np.ndarray
    aa_e: np.ndarray
    # Dense evaluated curve (POD-family); if set, Morse panel uses this vs Morse fit
    ab_curve_d: np.ndarray | None = None
    ab_curve_e: np.ndarray | None = None
    aa_curve_d: np.ndarray | None = None
    aa_curve_e: np.ndarray | None = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return REPO_ROOT


def _chdir_for_dataloader() -> None:
    root = _repo_root()
    uq = root / "uncertainty_quantification"
    os.chdir(str(uq if uq.is_dir() else root))


def _resolve_model_names(
    model_patterns: list[str],
    args,
    cli_hyperparams: dict,
    ensemble_dir: str,
) -> list[str]:
    """Expand ``--models`` patterns (same rules as relaxation propagation)."""
    expanded = expand_models_for_relaxation(model_patterns, ensemble_dir)
    if not expanded:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for pattern in expanded:
        name = expand_ensemble_model_name(pattern, args, cli_hyperparams)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _model_plot_label(model_name: str, load_name: str) -> str:
    if model_name.startswith(load_name):
        return model_name.split("_POD_index_")[0] if "_POD_index_" in model_name else model_name
    return load_name


def _is_pod_family_model(model_name: str) -> bool:
    return model_name.startswith(POD_FAMILY_PREFIXES)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name.strip())


# ── lattice reference ─────────────────────────────────────────────────────────

def reference_lattice_from_mode(atoms_list: list) -> tuple[float, float]:
    """Return (a_ref, b_ref) as the modal cell[0,0] and sqrt(3)/2 * a_ref."""
    a_vals = [round(float(a.cell[0, 0]), 3) for a in atoms_list]
    cnt = collections.Counter(a_vals)
    a_ref = float(cnt.most_common(1)[0][0])
    b_ref = float(np.sqrt(3) / 2 * a_ref)
    return a_ref, b_ref


def is_unstrained(atoms, a_ref: float, b_ref: float) -> bool:
    """True when both in-plane cell parameters match the reference lattice within tolerance."""
    lx = float(atoms.cell[0, 0])
    ly = float(atoms.cell[1, 1])
    return abs(lx - a_ref) < _STRAIN_TOL_ANG and abs(ly - b_ref) < _STRAIN_TOL_ANG


# ── stacking detection ────────────────────────────────────────────────────────

def _max_min_xy_top_to_bottom(atoms) -> float:
    """Max over top-layer atoms of (min xy distance to any bottom-layer atom, MIC)."""
    pos  = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell,      dtype=float)
    z    = pos[:, 2]
    zmid = float(np.median(z))
    itop = np.where(z >= zmid)[0]
    ibot = np.where(z <  zmid)[0]
    if itop.size == 0 or ibot.size == 0:
        return 0.0
    inv    = np.linalg.inv(cell.T)
    fcoords = (inv @ pos.T).T
    t_cell  = cell[:2, :2]
    mins: list[float] = []
    for i in itop:
        fi = fcoords[i, :2]
        fj = fcoords[ibot, :2]
        df = fj - fi[None, :]
        df -= np.round(df)
        dxy = df @ t_cell
        mins.append(float(np.min(np.linalg.norm(dxy, axis=1))))
    return float(np.max(mins))


def bilayer_stacking_is_aa(atoms) -> bool:
    return _max_min_xy_top_to_bottom(atoms) < _STACK_AA_MAX_XY


def bilayer_stacking_is_ab(atoms) -> bool:
    return _max_min_xy_top_to_bottom(atoms) > _STACK_AB_MIN_XY


# ── geometry ──────────────────────────────────────────────────────────────────

def interlayer_separation(atoms) -> float:
    z    = atoms.positions[:, 2]
    zmid = float(np.median(z))
    bot  = z < zmid
    top  = ~bot
    return float(np.mean(z[top]) - np.mean(z[bot]))


def set_interlayer_separation(atoms, d: float):
    """
    Return a copy with mean top−bottom layer separation set to ``d`` (Å).

    In-plane positions and stacking are preserved; only the top-layer ``z``
    coordinates are rigidly shifted.
    """
    out = atoms.copy()
    z = out.positions[:, 2]
    zmid = float(np.median(z))
    bot = z < zmid
    top = ~bot
    if not np.any(bot) or not np.any(top):
        raise ValueError("Could not identify bilayer top/bottom layers.")
    z_bot = float(np.mean(z[bot]))
    z_top = float(np.mean(z[top]))
    out.positions[top, 2] += float(d) - (z_top - z_bot)
    return out


def _pick_template_atoms(frames: UnstrainedFrames, *, target_d: float = 3.4):
    """Pick an unstrained frame whose interlayer separation is closest to ``target_d``."""
    if not frames.atoms:
        raise ValueError("No atoms available for interlayer template.")
    ds = np.asarray([interlayer_separation(a) for a in frames.atoms], dtype=float)
    i = int(np.argmin(np.abs(ds - float(target_d))))
    return frames.atoms[i].copy()


# ── energy helpers ────────────────────────────────────────────────────────────

def energy_per_atom_total(atoms, energy_total: float) -> float:
    return float(energy_total) / len(atoms)


# ── Morse potential ───────────────────────────────────────────────────────────

def morse_potential(
    d: np.ndarray,
    de_well: float,
    a: float,
    d_eq: float,
    e_inf: float,
) -> np.ndarray:
    """
    Morse interlayer potential (eV/atom).

    ``E(d) = D_e [exp(-2a(d-d_e)) - 2 exp(-a(d-d_e))] + E_∞``
    with ``E(∞) = E_∞`` and ``E(d_e) = E_∞ - D_e``.
    """
    d = np.asarray(d, dtype=float)
    x = d - float(d_eq)
    return (
        float(de_well) * (np.exp(-2.0 * a * x) - 2.0 * np.exp(-a * x))
        + float(e_inf)
    )


def _morse_a_from_attractive_point(
    d: float,
    e: float,
    *,
    d_eq: float,
    e_inf: float,
    de_well: float,
) -> float | None:
    """
    Estimate Morse ``a`` from one point on the attractive branch
    (``d > d_eq``, ``E_∞ - D_e < E < E_∞``).
    """
    x = float(d) - float(d_eq)
    if x <= 1e-6 or de_well <= 0.0:
        return None
    # E - E_∞ = D_e (u^2 - 2u), u = exp(-a x) ∈ (0, 1)
    r = (float(e) - float(e_inf)) / float(de_well)
    if not (-1.0 < r < 0.0):
        return None
    disc = 1.0 + r
    if disc <= 0.0:
        return None
    u = 1.0 - np.sqrt(disc)
    if not (0.0 < u < 1.0):
        return None
    return float(-np.log(u) / x)


def fit_morse_curve(
    d: np.ndarray,
    e: np.ndarray,
    *,
    d_min_fit: float | None = 2.8,
) -> tuple[np.ndarray, dict[str, float]] | tuple[None, None]:
    """
    Fit Morse parameters to ``(d, E/atom)`` data for one stacking.

    Initial guesses are anchored to well-defined data features:

    - ``d_e`` ← interlayer separation at the energy minimum
    - ``E_∞`` ← energy at the largest available separation (≈ dissociated limit)
    - ``D_e`` ← ``E_∞ - E_min``
    - ``a`` ← estimate(s) from attractive-branch points, plus a few fallbacks

    By default, points with ``d < d_min_fit`` (hard wall) are omitted from the
    least-squares fit so the short-range repulsion does not pull ``d_e`` /
    ``E_∞``.  The anchors themselves still use the full ``(d, e)`` arrays.

    Returns ``(params, info)`` with ``params = (D_e, a, d_e, E_∞)``, or
    ``(None, None)`` if the fit fails / not enough points.
    """
    d_all = np.asarray(d, dtype=float).ravel()
    e_all = np.asarray(e, dtype=float).ravel()
    m = np.isfinite(d_all) & np.isfinite(e_all)
    d_all, e_all = d_all[m], e_all[m]
    if d_all.size < 4:
        return None, None
    order = np.argsort(d_all)
    d_all, e_all = d_all[order], e_all[order]

    # Anchors from full curve
    i_min = int(np.argmin(e_all))
    d_eq0 = float(d_all[i_min])
    e_min = float(e_all[i_min])
    # E(d_max) ≈ E_∞ (use the furthest point; average last two if available)
    if d_all.size >= 2 and abs(d_all[-1] - d_all[-2]) < 0.5:
        e_inf0 = float(0.5 * (e_all[-1] + e_all[-2]))
    else:
        e_inf0 = float(e_all[-1])
    de0 = max(e_inf0 - e_min, 1e-6)

    # Fit window: keep the well and asymptote; drop deep repulsive wall
    if d_min_fit is None:
        d_fit, e_fit = d_all, e_all
    else:
        keep = d_all >= float(d_min_fit)
        d_fit, e_fit = d_all[keep], e_all[keep]
        if d_fit.size < 4:
            d_fit, e_fit = d_all, e_all

    # Seed a from attractive-branch points (d > d_eq)
    a_seeds: list[float] = []
    for di, ei in zip(d_fit, e_fit):
        if di <= d_eq0 + 0.05:
            continue
        a_est = _morse_a_from_attractive_point(
            float(di), float(ei), d_eq=d_eq0, e_inf=e_inf0, de_well=de0,
        )
        if a_est is not None and 0.1 < a_est < 10.0:
            a_seeds.append(a_est)
    if a_seeds:
        a_seeds = [float(np.median(a_seeds))] + a_seeds[:5]
    a_seeds.extend([1.0, 1.2, 1.5, 2.0])
    # unique while preserving order
    seen_a: set[float] = set()
    a0_list: list[float] = []
    for a in a_seeds:
        key = round(a, 4)
        if key not in seen_a:
            seen_a.add(key)
            a0_list.append(float(a))

    # Tight bounds around data anchors
    d_lo = max(float(d_fit.min()) * 0.8, d_eq0 - 0.35)
    d_hi = min(float(d_fit.max()) * 1.2, d_eq0 + 0.35)
    if d_hi <= d_lo:
        d_lo, d_hi = d_eq0 - 0.2, d_eq0 + 0.2
    # E_∞ near E(d_max); allow a little slack (not fully dissociated at finite d)
    e_slack = max(0.05 * de0, 5e-4)
    e_inf_lo = e_inf0 - e_slack
    e_inf_hi = e_inf0 + max(e_slack, 2e-3)
    de_lo = max(1e-8, 0.25 * de0)
    de_hi = max(de0 * 4.0, de0 + 1e-3)

    bounds = (
        (de_lo, 0.05, d_lo, e_inf_lo),
        (de_hi, 12.0, d_hi, e_inf_hi),
    )

    best: tuple[np.ndarray, float] | None = None
    last_err: Exception | None = None
    for a0 in a0_list:
        p0 = (de0, a0, d_eq0, e_inf0)
        try:
            popt, _pcov = curve_fit(
                morse_potential,
                d_fit,
                e_fit,
                p0=p0,
                bounds=bounds,
                maxfev=50_000,
            )
        except Exception as exc:
            last_err = exc
            continue
        resid = e_fit - morse_potential(d_fit, *popt)
        rss = float(np.dot(resid, resid))
        if best is None or rss < best[1]:
            best = (np.asarray(popt, dtype=float), rss)

    if best is None:
        if last_err is not None:
            print(
                f"  Morse fit failed: {type(last_err).__name__}: {last_err}",
                file=sys.stderr,
            )
        return None, None

    popt, rss = best
    de_well, a, d_eq, e_inf = (float(x) for x in popt)
    info = {
        "De": de_well,
        "a": a,
        "de": d_eq,
        "E_inf": e_inf,
        "E_min_data": e_min,
        "d_eq_data": d_eq0,
        "E_inf_data": e_inf0,
        "rmse_fit": float(np.sqrt(rss / max(e_fit.size, 1))),
        "n_fit": int(e_fit.size),
    }
    return popt, info


# Back-compat alias
fit_morse_to_ab = fit_morse_curve


# ── curve construction ────────────────────────────────────────────────────────

def _merge_same_d(d_list: list, e_list: list) -> tuple[np.ndarray, np.ndarray]:
    if not d_list:
        return np.array([]), np.array([])
    buckets: dict[float, list[float]] = {}
    for di, ei in zip(d_list, e_list):
        key = round(float(di), 2)
        buckets.setdefault(key, []).append(float(ei))
    keys  = sorted(buckets)
    d_out = np.asarray(keys, dtype=float)
    e_out = np.asarray([min(buckets[k]) for k in keys], dtype=float)
    return d_out, e_out


# ── data selection ────────────────────────────────────────────────────────────

def select_unstrained_by_stacking(
    atoms_list: list,
    energies: np.ndarray,
    a_ref: float,
    b_ref: float,
) -> dict[str, UnstrainedFrames]:
    """
    Filter a DFT interlayer dataset to unstrained AB and AA frames.

    Unstrained: both cell[0,0] ≈ a_ref and cell[1,1] ≈ b_ref within _STRAIN_TOL_ANG.
    AA: max_min_xy < _STACK_AA_MAX_XY.
    AB: max_min_xy > _STACK_AB_MIN_XY (excludes saddle-point SP stacking).
    """
    ab_atoms: list = []
    ab_e: list[float] = []
    aa_atoms: list = []
    aa_e: list[float] = []

    for atoms, e_tot in zip(atoms_list, energies):
        if not is_unstrained(atoms, a_ref, b_ref):
            continue
        if bilayer_stacking_is_aa(atoms):
            aa_atoms.append(atoms)
            aa_e.append(float(e_tot))
        elif bilayer_stacking_is_ab(atoms):
            ab_atoms.append(atoms)
            ab_e.append(float(e_tot))

    return {
        "AB": UnstrainedFrames(ab_atoms, np.asarray(ab_e, dtype=float)),
        "AA": UnstrainedFrames(aa_atoms, np.asarray(aa_e, dtype=float)),
    }


# Keep old name as alias for any external imports.
select_unstrained_rvv10_by_stacking = select_unstrained_by_stacking


def _atoms_total_energy(atoms) -> float:
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if "free_energy" in atoms.info:
        return float(atoms.info["free_energy"])
    if atoms.calc is not None:
        return float(atoms.get_potential_energy())
    raise ValueError("Atoms object has no energy in info or calculator.")


def _geometry_fingerprint(atoms) -> tuple:
    return (
        round(float(atoms.cell[0, 0]), 5),
        round(float(atoms.cell[1, 1]), 5),
        round(interlayer_separation(atoms), 3),
        round(_max_min_xy_top_to_bottom(atoms), 3),
        len(atoms),
    )


def _ensure_mol_id(atoms):
    if "mol-id" in getattr(atoms, "arrays", {}):
        return atoms
    out = atoms.copy()
    z = out.positions[:, 2]
    mean_z = float(np.mean(z))
    mol_id = np.ones(len(out), dtype=np.int8)
    mol_id[z > mean_z] = 2
    out.set_array("mol-id", mol_id)
    return out


def append_large_d_rvv10_frames(
    atoms_list: list,
    energies: np.ndarray,
) -> tuple[list, np.ndarray]:
    """
    Append rVV10 frames with interlayer separation ``> 4.5`` Å from the
    Training/TestData xyz files when they are missing from ``atoms_list``.

    The main ``data/strained_bilayer_graphene_rVV10.xyz`` set used by
    :func:`load_energy_data` currently maxes out near 4.5 Å for unstrained
    cells; the train/test splits still include larger separations (6 and 7 Å).
    """
    seen = {_geometry_fingerprint(a) for a in atoms_list}
    extra_atoms: list = []
    extra_e: list[float] = []
    for path in _RVV10_LARGE_D_XYZ:
        if not path.is_file():
            continue
        for atoms in ase.io.read(str(path), format="extxyz", index=":"):
            d = interlayer_separation(atoms)
            if d <= _LARGE_D_MIN_ANG + 1e-6:
                continue
            fp = _geometry_fingerprint(atoms)
            if fp in seen:
                continue
            seen.add(fp)
            e_tot = _atoms_total_energy(atoms)
            atoms = _ensure_mol_id(atoms)
            extra_atoms.append(atoms)
            extra_e.append(e_tot)
    if not extra_atoms:
        return atoms_list, np.asarray(energies, dtype=float)
    d_added = sorted({round(interlayer_separation(a), 2) for a in extra_atoms})
    print(
        f"  Appended {len(extra_atoms)} large-d rVV10 frames "
        f"(d > {_LARGE_D_MIN_ANG:g} Å) from Training/TestData; "
        f"unique d={d_added}",
        flush=True,
    )
    atoms_out = list(atoms_list) + extra_atoms
    energies_out = np.concatenate(
        [np.asarray(energies, dtype=float), np.asarray(extra_e, dtype=float)]
    )
    return atoms_out, energies_out


# ── PES curve builders ────────────────────────────────────────────────────────

def frames_to_pes_per_atom(
    frames: UnstrainedFrames,
) -> tuple[np.ndarray, np.ndarray]:
    """Merged ``(d, E/atom)`` from total energies (no reference shift)."""
    if not frames.atoms:
        return np.array([]), np.array([])
    d_list = [interlayer_separation(a) for a in frames.atoms]
    e_list = [
        energy_per_atom_total(a, e)
        for a, e in zip(frames.atoms, frames.dft_energies)
    ]
    return _merge_same_d(d_list, e_list)


def model_totals_to_pes_per_atom(
    frames: UnstrainedFrames,
    model_energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames.atoms:
        return np.array([]), np.array([])
    model_energies = np.asarray(model_energies, dtype=float).ravel()
    if model_energies.size != len(frames.atoms):
        raise ValueError("model_energies length must match frames.atoms")
    d_list: list[float] = []
    e_list: list[float] = []
    for atoms, e_tot in zip(frames.atoms, model_energies):
        if not np.isfinite(e_tot):
            raise RuntimeError("Model returned non-finite energy.")
        d_list.append(interlayer_separation(atoms))
        e_list.append(energy_per_atom_total(atoms, e_tot))
    return _merge_same_d(d_list, e_list)


def load_qmc_interlayer_csv() -> pd.DataFrame:
    path = _repo_root() / "data" / "qmc.csv"
    df = pd.read_csv(path)
    return df.rename(columns={c: c.strip() for c in df.columns})


def qmc_pes_per_atom(stacking: str) -> tuple[np.ndarray, np.ndarray]:
    df = load_qmc_interlayer_csv()
    sub = df[df["stacking"].str.upper() == stacking.upper()]
    if sub.empty:
        return np.array([]), np.array([])
    d_u, e_u = _merge_same_d(
        sub["d"].values.tolist(),
        sub["energy"].values.astype(float).tolist(),
    )
    order = np.argsort(d_u)
    return d_u[order], e_u[order]


def qmc_pes_series() -> PesSeries:
    ab_d, ab_e = qmc_pes_per_atom("AB")
    aa_d, aa_e = qmc_pes_per_atom("AA")
    return PesSeries(
        label="QMC",
        color=_COLOR_QMC,
        ab_d=ab_d,
        ab_e=ab_e,
        aa_d=aa_d,
        aa_e=aa_e,
    )


def evaluate_model_energies_on_atoms(calc_obj, atoms_list: list) -> np.ndarray:
    """Evaluate total energies (eV) on *atoms_list*."""
    if not atoms_list:
        return np.array([], dtype=float)
    if hasattr(calc_obj, "prepare_batch"):
        calc_obj.prepare_batch(atoms_list)
        energies, _ = calc_obj.evaluate_batch()
        return np.asarray(energies, dtype=float).ravel()
    return np.asarray(
        [float(calc_obj.get_potential_energy(a)) for a in atoms_list],
        dtype=float,
    )


def evaluate_model_pes_vs_separation(
    calc_obj,
    template_atoms,
    d_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate energy/atom (eV) on copies of ``template_atoms`` with interlayer
    separations ``d_grid`` (Å).
    """
    d_grid = np.asarray(d_grid, dtype=float).ravel()
    atoms_list = [set_interlayer_separation(template_atoms, float(d)) for d in d_grid]
    e_tot = evaluate_model_energies_on_atoms(calc_obj, atoms_list)
    n = len(template_atoms)
    return np.asarray(e_tot, dtype=float) / float(n)


# ── POD / PODD3 calculator (best-fit, fast path) ─────────────────────────────

def _pod_family_load_name(model_name: str) -> str:
    if model_name.startswith("PODD3_energy"):
        return "PODD3_energy"
    if model_name.startswith("POD_energy"):
        return "POD_energy"
    raise ValueError(f"Not a POD-family model: {model_name!r}")


def _pod_best_fit_npz(model_name: str, extra_kw: dict | None) -> Path:
    load_name = _pod_family_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), **(extra_kw or {})}
    pod_hp = data_kw.get("pod_hyperparams")
    if not isinstance(pod_hp, dict) or not pod_hp:
        raise ValueError(
            f"Could not resolve POD hyperparameters for {model_name!r}. "
            "Pass --POD-index or use a folder name with an embedded hash."
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


def _build_pod_family_calculator(
    model_name: str, extra_kw: dict | None,
) -> PODLammpsCalculator | PODD3LammpsCalculator:
    """Build POD or PODD3 calculator from cached best-fit coefficients."""
    load_name = _pod_family_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), **(extra_kw or {})}
    pod_hp = data_kw.get("pod_hyperparams")
    if not isinstance(pod_hp, dict) or not pod_hp:
        raise ValueError(
            f"Could not resolve POD hyperparameters for {model_name!r}. "
            "Pass --POD-index or use a folder name with an embedded hash."
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


def _build_model_calculator(model_name: str, extra_kw: dict | None):
    """Return (calculator, close_fn). POD/PODD3 use the fast best-fit path."""
    if _is_pod_family_model(model_name):
        calc = _build_pod_family_calculator(model_name, extra_kw)
        return calc, getattr(calc, "close", lambda: None)
    calc_obj, _set_params_fn, _load_name = build_uq_calculator(
        model_name, extra_kw=extra_kw,
    )
    return calc_obj, getattr(calc_obj, "close", lambda: None)


# ── plotting ──────────────────────────────────────────────────────────────────

def _series_stacking_arrays(
    series: PesSeries, stacking: str,
) -> tuple[np.ndarray, np.ndarray]:
    if stacking == "AA":
        return series.aa_d, series.aa_e
    return series.ab_d, series.ab_e


def _series_stacking_curves(
    series: PesSeries, stacking: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if stacking == "AA":
        return series.aa_curve_d, series.aa_curve_e
    return series.ab_curve_d, series.ab_curve_e


def plot_morse_shifted_panels(
    series_list: list[PesSeries],
    *,
    out_path: Path,
    dpi: int = 180,
) -> None:
    """
    Binding curves for ``d ∈ XLIM_MORSE``, shifted to approach 0 at large ``d``.

    DFT/QMC: fit a Morse curve per stacking and shift by that fit's ``E_∞``.
    POD-family series with a dense evaluated curve: plot that curve shifted by
    ``E(d_max)`` (no Morse line).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    d_morse = np.linspace(XLIM_MORSE[0], XLIM_MORSE[1], 400)

    for series in series_list:
        for ax, stacking in zip(axes, STACKINGS):
            d, e = _series_stacking_arrays(series, stacking)
            d_curve, e_curve = _series_stacking_curves(series, stacking)
            marker = _MARKER_AA if stacking == "AA" else _MARKER_AB

            if (
                d_curve is not None
                and e_curve is not None
                and np.asarray(d_curve).size
                and np.asarray(e_curve).size
            ):
                d_curve = np.asarray(d_curve, dtype=float)
                e_curve = np.asarray(e_curve, dtype=float)
                e_inf = float(e_curve[-1])
                print(
                    f"  POD curve {series.label} {stacking}: "
                    f"E(d={d_curve[-1]:.2f} Å)={e_inf:.6g} eV/atom "
                    f"({d_curve.size} pts)",
                    flush=True,
                )
                if d.size:
                    m = (d >= XLIM_MORSE[0] - 0.05) & (d <= XLIM_MORSE[1] + 0.05)
                    d_p, e_p = d[m], e[m] - e_inf
                    if d_p.size:
                        ax.plot(
                            d_p,
                            e_p,
                            linestyle="none",
                            marker=marker,
                            markersize=7,
                            color=series.color,
                        )
                ax.plot(
                    d_curve,
                    e_curve - e_inf,
                    linestyle="-",
                    linewidth=2.0,
                    color=series.color,
                    label=f"{series.label} ({stacking})",
                )
                continue

            params, info = fit_morse_curve(d, e)
            if params is None or info is None:
                print(
                    f"  Skipping Morse ({stacking}) for {series.label} "
                    f"(fit failed or too few points)",
                    flush=True,
                )
                continue
            e_inf = float(info["E_inf"])
            print(
                f"  Morse {series.label} {stacking}: "
                f"D_e={info['De']:.6g}  a={info['a']:.4g}  "
                f"d_e={info['de']:.4f} Å  E_∞={e_inf:.6g} eV/atom"
                + (
                    f"  RMSE={info['rmse_fit']:.3g} (n={info['n_fit']})"
                    if "rmse_fit" in info
                    else ""
                ),
                flush=True,
            )
            if d.size:
                m = (d >= XLIM_MORSE[0] - 0.05) & (d <= XLIM_MORSE[1] + 0.05)
                d_p, e_p = d[m], e[m] - e_inf
                if d_p.size:
                    ax.plot(
                        d_p,
                        e_p,
                        linestyle="none",
                        marker=marker,
                        markersize=7,
                        color=series.color,
                        label=f"{series.label} ({stacking})",
                    )
            e_morse = morse_potential(d_morse, *params) - e_inf
            ax.plot(
                d_morse,
                e_morse,
                linestyle="-",
                linewidth=2.0,
                color=series.color,
                label=f"{series.label} Morse ({stacking})",
            )

    for ax, stacking in zip(axes, STACKINGS):
        ax.set_xlabel("Interlayer separation (Å)")
        ax.set_ylabel(r"$E - E_\infty$ (eV/atom)")
        ax.set_title(f"{stacking} stacking (shifted)")
        ax.set_xlim(*XLIM_MORSE)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)

    fig.suptitle(
        r"DFT/QMC: Morse fits; POD: evaluated $E(d)$; each shifted by $E_\infty$",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def ab_equilibrium_energy(series: PesSeries) -> tuple[float, float] | None:
    """
    Return ``(l_0, E_AB(l_0))`` for one data source (eV/atom).

    Preference order:
    1. Dense AB POD curve (argmin)
    2. AB Morse fit (``d_e``, ``E(d_e)``)
    3. Discrete AB scatter points (argmin)
    """
    d_curve, e_curve = series.ab_curve_d, series.ab_curve_e
    if (
        d_curve is not None
        and e_curve is not None
        and np.asarray(d_curve).size
        and np.asarray(e_curve).size
    ):
        d_c = np.asarray(d_curve, dtype=float).ravel()
        e_c = np.asarray(e_curve, dtype=float).ravel()
        m = np.isfinite(d_c) & np.isfinite(e_c)
        d_c, e_c = d_c[m], e_c[m]
        if d_c.size:
            i = int(np.argmin(e_c))
            return float(d_c[i]), float(e_c[i])

    d_ab, e_ab = np.asarray(series.ab_d, dtype=float), np.asarray(series.ab_e, dtype=float)
    params, info = fit_morse_curve(d_ab, e_ab)
    if params is not None and info is not None:
        l0 = float(info["de"])
        e0 = float(morse_potential(np.asarray([l0]), *params)[0])
        return l0, e0

    m = np.isfinite(d_ab) & np.isfinite(e_ab)
    d_ab, e_ab = d_ab[m], e_ab[m]
    if not d_ab.size:
        return None
    i = int(np.argmin(e_ab))
    return float(d_ab[i]), float(e_ab[i])


def plot_ab_eq_shifted_panels(
    series_list: list[PesSeries],
    *,
    out_path: Path,
    dpi: int = 180,
) -> None:
    """
    Binding curves for ``d ∈ XLIM_AB_EQ``, shifted so AB equilibrium is zero.

    For each data source, find AB equilibrium separation ``l_0`` and energy
    ``E_AB(l_0)``, then subtract that energy from both AB and AA.  Units and
    layout match :func:`plot_morse_shifted_panels` (x-range is ``XLIM_AB_EQ``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    d_morse = np.linspace(XLIM_AB_EQ[0], XLIM_AB_EQ[1], 400)

    for series in series_list:
        eq = ab_equilibrium_energy(series)
        if eq is None:
            print(
                f"  Skipping AB-eq shift for {series.label} (no AB data)",
                flush=True,
            )
            continue
        l0, e_ab_l0 = eq
        print(
            f"  AB eq {series.label}: l_0={l0:.4f} Å  "
            f"E_AB(l_0)={e_ab_l0:.6g} eV/atom",
            flush=True,
        )

        for ax, stacking in zip(axes, STACKINGS):
            d, e = _series_stacking_arrays(series, stacking)
            d_curve, e_curve = _series_stacking_curves(series, stacking)
            marker = _MARKER_AA if stacking == "AA" else _MARKER_AB

            if (
                d_curve is not None
                and e_curve is not None
                and np.asarray(d_curve).size
                and np.asarray(e_curve).size
            ):
                d_curve = np.asarray(d_curve, dtype=float)
                e_curve = np.asarray(e_curve, dtype=float)
                m_c = (
                    (d_curve >= XLIM_AB_EQ[0] - 0.05)
                    & (d_curve <= XLIM_AB_EQ[1] + 0.05)
                )
                d_curve, e_curve = d_curve[m_c], e_curve[m_c]
                if d.size:
                    m = (d >= XLIM_AB_EQ[0] - 0.05) & (d <= XLIM_AB_EQ[1] + 0.05)
                    d_p, e_p = d[m], e[m] - e_ab_l0
                    if d_p.size:
                        ax.plot(
                            d_p,
                            e_p,
                            linestyle="none",
                            marker=marker,
                            markersize=7,
                            color=series.color,
                        )
                if d_curve.size:
                    ax.plot(
                        d_curve,
                        e_curve - e_ab_l0,
                        linestyle="-",
                        linewidth=2.0,
                        color=series.color,
                        label=f"{series.label} ({stacking})",
                    )
                continue

            params, info = fit_morse_curve(d, e)
            if d.size:
                m = (d >= XLIM_AB_EQ[0] - 0.05) & (d <= XLIM_AB_EQ[1] + 0.05)
                d_p, e_p = d[m], e[m] - e_ab_l0
                if d_p.size:
                    ax.plot(
                        d_p,
                        e_p,
                        linestyle="none",
                        marker=marker,
                        markersize=7,
                        color=series.color,
                        label=f"{series.label} ({stacking})",
                    )
            if params is not None and info is not None:
                e_morse = morse_potential(d_morse, *params) - e_ab_l0
                ax.plot(
                    d_morse,
                    e_morse,
                    linestyle="-",
                    linewidth=2.0,
                    color=series.color,
                    label=f"{series.label} Morse ({stacking})",
                )
            elif not d.size:
                print(
                    f"  Skipping {stacking} for {series.label} "
                    f"(no data / Morse fit)",
                    flush=True,
                )

    for ax, stacking in zip(axes, STACKINGS):
        ax.set_xlabel("Interlayer separation (Å)")
        ax.set_ylabel(r"$E - E_{\mathrm{AB}}(l_0)$ (eV/atom)")
        ax.set_title(f"{stacking} stacking (shifted)")
        ax.set_xlim(*XLIM_AB_EQ)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)

    fig.suptitle(
        r"Each source shifted by its AB equilibrium energy $E_{\mathrm{AB}}(l_0)$",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot bilayer graphene PES vs interlayer separation "
            "(Morse-/POD-shifted and AB-equilibrium–shifted binding, 3–8 Å)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p, required=False)
    p.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    p.add_argument("-M", type=int, default=10, help="ACSF/POD M (bare model names only).")
    p.add_argument("-W", type=int, default=6, help="ACSF/POD W (bare model names only).")
    p.add_argument(
        "--POD-index",
        type=int,
        default=None,
        dest="pod_index",
        help="POD hyperparameter-search index (with bare POD_energy / PODD3_energy).",
    )
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--include-mbd",
        action="store_true",
        help="Include the MBD DFT dataset (omitted by default).",
    )
    add_hyperparam_args(p)
    return p.parse_known_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args, unknown = _parse_args()
    cli_hyperparams = collect_workflow_hyperparams(args, unknown)
    if args.pod_index is None and "POD_index" in cli_hyperparams:
        args.pod_index = int(cli_hyperparams["POD_index"])
    if cli_hyperparams:
        print(f"CLI hyperparameters: {cli_hyperparams}", flush=True)

    os.chdir(UQ_DIR)
    model_patterns = args.models or []
    models = (
        _resolve_model_names(model_patterns, args, cli_hyperparams, args.ensemble_dir)
        if model_patterns
        else []
    )
    if models:
        print(f"Models: {models}", flush=True)
    else:
        print("No --models specified: plotting DFT sources + QMC only.", flush=True)

    _chdir_for_dataloader()

    series_list: list[PesSeries] = []

    # DFT sources (MBD off by default)
    dft_levels = list(DFT_LEVELS_DEFAULT)
    if args.include_mbd:
        dft_levels.append("MBD")
    dft_colors = {"rVV10": _COLOR_RVV10, "MBD": _COLOR_MBD}
    rvv10_unstrained: dict[str, UnstrainedFrames] | None = None
    for lot in dft_levels:
        print(f"\nLoading {lot} …", flush=True)
        try:
            atoms_list, energies, _ = load_energy_data(
                "interlayer", supercells=1, level_of_theory=lot,
            )
        except Exception as exc:
            print(f"  Skipping {lot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        energies = np.asarray(energies, dtype=float)
        if lot == "rVV10":
            atoms_list, energies = append_large_d_rvv10_frames(
                atoms_list, energies,
            )
        a_ref, b_ref = reference_lattice_from_mode(atoms_list)
        print(
            f"  {lot}: a_ref={a_ref:.5f} Å, b_ref={b_ref:.5f} Å, "
            f"n_total={len(atoms_list)}",
            flush=True,
        )
        unstrained = select_unstrained_by_stacking(
            atoms_list, energies, a_ref, b_ref,
        )
        for stk in STACKINGS:
            fr = unstrained[stk]
            if fr.atoms:
                d_vals = sorted(
                    {round(interlayer_separation(a), 2) for a in fr.atoms}
                )
                print(
                    f"    Unstrained {stk}: {len(fr.atoms)} frames, "
                    f"d ∈ [{min(d_vals):g}, {max(d_vals):g}] "
                    f"({len(d_vals)} unique)",
                    flush=True,
                )
            else:
                print(f"    Unstrained {stk}: 0 frames", flush=True)
        if lot == "rVV10":
            rvv10_unstrained = unstrained
        ab_d, ab_e = frames_to_pes_per_atom(unstrained["AB"])
        aa_d, aa_e = frames_to_pes_per_atom(unstrained["AA"])
        if ab_d.size or aa_d.size:
            series_list.append(
                PesSeries(
                    label=lot,
                    color=dft_colors[lot],
                    ab_d=ab_d,
                    ab_e=ab_e,
                    aa_d=aa_d,
                    aa_e=aa_e,
                )
            )

    # QMC
    print("\nLoading QMC …", flush=True)
    series_list.append(qmc_pes_series())
    q = series_list[-1]
    print(
        f"  QMC AB: n={q.ab_d.size}  AA: n={q.aa_d.size}",
        flush=True,
    )

    # Models evaluated on rVV10 unstrained geometries
    if models and rvv10_unstrained is None:
        print(
            "  Warning: no rVV10 frames available; skipping model evaluations.",
            file=sys.stderr,
        )
        models = []

    for i_model, model_name in enumerate(models):
        print(f"\n--- Model: {model_name} ---", flush=True)
        if not is_uq_energy_model(model_name) and not _is_pod_family_model(model_name):
            print(
                f"  Warning: unsupported model; skipping {model_name!r}.",
                file=sys.stderr,
            )
            continue

        model_label = _model_plot_label(
            model_name,
            _pod_family_load_name(model_name)
            if _is_pod_family_model(model_name)
            else model_name,
        )
        color = _COLOR_MODEL_CYCLE[i_model % len(_COLOR_MODEL_CYCLE)]
        calc_obj, close_calc = _build_model_calculator(
            model_name, cli_hyperparams or None,
        )
        print(f"  Calculator: {model_label}", flush=True)
        try:
            assert rvv10_unstrained is not None
            model_by_stacking: dict[str, np.ndarray] = {}
            for stacking in STACKINGS:
                n = len(rvv10_unstrained[stacking].atoms)
                print(
                    f"  Evaluating on {n} unstrained {stacking} frames …",
                    flush=True,
                )
                model_by_stacking[stacking] = evaluate_model_energies_on_atoms(
                    calc_obj, rvv10_unstrained[stacking].atoms,
                )
            ab_d, ab_e = model_totals_to_pes_per_atom(
                rvv10_unstrained["AB"], model_by_stacking["AB"],
            )
            aa_d, aa_e = model_totals_to_pes_per_atom(
                rvv10_unstrained["AA"], model_by_stacking["AA"],
            )
            ab_curve_d = ab_curve_e = aa_curve_d = aa_curve_e = None
            if _is_pod_family_model(model_name):
                d_grid = np.linspace(XLIM_MORSE[0], XLIM_MORSE[1], 81)
                print(
                    f"  Evaluating POD curve on d ∈ "
                    f"[{XLIM_MORSE[0]:.1f}, {XLIM_MORSE[1]:.1f}] Å "
                    f"({d_grid.size} pts) …",
                    flush=True,
                )
                ab_curve_d = d_grid.copy()
                aa_curve_d = d_grid.copy()
                ab_curve_e = evaluate_model_pes_vs_separation(
                    calc_obj,
                    _pick_template_atoms(rvv10_unstrained["AB"]),
                    d_grid,
                )
                aa_curve_e = evaluate_model_pes_vs_separation(
                    calc_obj,
                    _pick_template_atoms(rvv10_unstrained["AA"]),
                    d_grid,
                )
            series_list.append(
                PesSeries(
                    label=model_label,
                    color=color,
                    ab_d=ab_d,
                    ab_e=ab_e,
                    aa_d=aa_d,
                    aa_e=aa_e,
                    ab_curve_d=ab_curve_d,
                    ab_curve_e=ab_curve_e,
                    aa_curve_d=aa_curve_d,
                    aa_curve_e=aa_curve_e,
                )
            )
        finally:
            close_calc()

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    tag = "dft_qmc"
    if models:
        tag = _safe_filename(models[0]) if len(models) == 1 else "multi_model"

    print("\nWriting figures …", flush=True)
    plot_morse_shifted_panels(
        series_list,
        out_path=figures_dir / f"{tag}_bilayer_graphene_pes_morse_shifted_3_to_8A.png",
        dpi=args.dpi,
    )
    plot_ab_eq_shifted_panels(
        series_list,
        out_path=figures_dir / f"{tag}_bilayer_graphene_pes_ab_eq_shifted_3_to_5A.png",
        dpi=args.dpi,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
