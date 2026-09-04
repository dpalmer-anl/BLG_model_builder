#!/usr/bin/env python3
"""
Bilayer-graphene PES for unstrained AB and AA stacking.

1. Extract unstrained AB/AA rVV10 structures (unique *d* in the plot window).
2. Measure interlayer separation.
3. Evaluate a POD ensemble on those structures.
4. Plot Morse-shifted and AB-eq–shifted binding curves, plus a *d*_eq histogram.

Run from ``uncertainty_quantification``::

    python visualizations/plot_bilayer_graphene_pes.py \\
        --models POD_energy_POD_index_15_8bb97b2162397248
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from blg_model_builder.DataLoader import load_energy_data
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
    expand_ensemble_model_name,
)
from blg_model_builder.ensemble_io import (
    DEFAULT_CALIBRATION_METRICS_DIR,
    load_ensemble_pickle,
    resolve_ensemble_pickle,
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
REPO_ROOT = UQ_DIR.parent
if str(UQ_DIR) not in sys.path:
    sys.path.insert(0, str(UQ_DIR))

from uq_model_runtime import (  # noqa: E402
    apply_uq_parameters,
    build_uq_calculator,
    is_uq_energy_model,
)
from run_uq_propagation_relaxation import expand_models_for_relaxation  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

_STACK_AA_MAX_XY = 0.01
_STACK_AB_MIN_XY = 1.2
_STRAIN_TOL_ANG = 0.005
_LARGE_D_MIN_ANG = 4.5
_RVV10_LARGE_D_XYZ = (
    REPO_ROOT / "data" / "TrainingData" / "strained_bilayer_graphene_rVV10.xyz",
    REPO_ROOT / "data" / "TestData" / "strained_bilayer_graphene_rVV10.xyz",
)

XLIM_MORSE = (3.0, 8.0)
XLIM_AB_EQ = (3.0, 4.0)
MODEL_EVAL_D_WINDOW = (2.8, XLIM_MORSE[1])
N_QUAD_EQ_POINTS = 5

_COLOR_RVV10 = "#1f77b4"
_COLOR_POD = "#2ca02c"
_MARKER_AB = "o"
_MARKER_AA = "s"
STACKINGS = ("AB", "AA")

DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"
DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 0
DEFAULT_HIST_BINS = 30

CSFONT = {"fontname": "sans-serif", "size": 20}
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)


@dataclass(frozen=True)
class PlotFrames:
    """One structure per unique interlayer separation, with DFT total energies."""
    atoms: list
    dft_energies: np.ndarray


@dataclass(frozen=True)
class PesSeries:
    label: str
    color: str
    ab_d: np.ndarray
    ab_e: np.ndarray
    aa_d: np.ndarray
    aa_e: np.ndarray
    ab_e_std: np.ndarray | None = None
    aa_e_std: np.ndarray | None = None


# ── structure extraction ──────────────────────────────────────────────────────

def interlayer_separation(atoms) -> float:
    """Mean top−bottom layer separation (Å)."""
    z = atoms.positions[:, 2]
    zmid = float(np.median(z))
    bot = z < zmid
    top = ~bot
    return float(np.mean(z[top]) - np.mean(z[bot]))


def _max_min_xy_top_to_bottom(atoms) -> float:
    pos = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    z = pos[:, 2]
    zmid = float(np.median(z))
    itop = np.where(z >= zmid)[0]
    ibot = np.where(z < zmid)[0]
    if itop.size == 0 or ibot.size == 0:
        return 0.0
    inv = np.linalg.inv(cell.T)
    fcoords = (inv @ pos.T).T
    t_cell = cell[:2, :2]
    mins: list[float] = []
    for i in itop:
        fi = fcoords[i, :2]
        fj = fcoords[ibot, :2]
        df = fj - fi[None, :]
        df -= np.round(df)
        dxy = df @ t_cell
        mins.append(float(np.min(np.linalg.norm(dxy, axis=1))))
    return float(np.max(mins))


def _is_unstrained(atoms, a_ref: float, b_ref: float) -> bool:
    lx = float(atoms.cell[0, 0])
    ly = float(atoms.cell[1, 1])
    return abs(lx - a_ref) < _STRAIN_TOL_ANG and abs(ly - b_ref) < _STRAIN_TOL_ANG


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


def _append_large_d_rvv10(atoms_list: list, energies: np.ndarray) -> tuple[list, np.ndarray]:
    """Add rVV10 frames with *d* > 4.5 Å from Training/TestData if missing."""
    seen = {_geometry_fingerprint(a) for a in atoms_list}
    extra_atoms: list = []
    extra_e: list[float] = []
    for path in _RVV10_LARGE_D_XYZ:
        if not path.is_file():
            continue
        for atoms in ase.io.read(str(path), format="extxyz", index=":"):
            if interlayer_separation(atoms) <= _LARGE_D_MIN_ANG + 1e-6:
                continue
            fp = _geometry_fingerprint(atoms)
            if fp in seen:
                continue
            seen.add(fp)
            extra_atoms.append(atoms)
            extra_e.append(_atoms_total_energy(atoms))
    if not extra_atoms:
        return atoms_list, np.asarray(energies, dtype=float)
    print(
        f"  Appended {len(extra_atoms)} large-d rVV10 frames "
        f"(d > {_LARGE_D_MIN_ANG:g} Å)",
        flush=True,
    )
    return (
        list(atoms_list) + extra_atoms,
        np.concatenate(
            [np.asarray(energies, dtype=float), np.asarray(extra_e, dtype=float)]
        ),
    )


def load_unstrained_ab_aa_frames() -> dict[str, PlotFrames]:
    """
    Load rVV10 interlayer data and return unstrained AB/AA frames.

    One frame per unique *d* (0.02 Å) inside ``MODEL_EVAL_D_WINDOW``, preferring
    the lowest DFT energy when duplicates exist.
    """
    atoms_list, energies, _ = load_energy_data(
        "interlayer", supercells=1, level_of_theory="rVV10",
    )
    energies = np.asarray(energies, dtype=float)
    atoms_list, energies = _append_large_d_rvv10(atoms_list, energies)

    a_vals = [round(float(a.cell[0, 0]), 3) for a in atoms_list]
    a_ref = float(collections.Counter(a_vals).most_common(1)[0][0])
    b_ref = float(np.sqrt(3) / 2 * a_ref)
    print(f"  rVV10: a_ref={a_ref:.5f} Å, n_total={len(atoms_list)}", flush=True)

    buckets: dict[str, dict[float, tuple[object, float]]] = {"AB": {}, "AA": {}}
    for atoms, e_tot in zip(atoms_list, energies):
        if not _is_unstrained(atoms, a_ref, b_ref):
            continue
        d = interlayer_separation(atoms)
        if not (MODEL_EVAL_D_WINDOW[0] - 1e-9 <= d <= MODEL_EVAL_D_WINDOW[1] + 1e-9):
            continue
        key = round(float(d), 2)
        mm = _max_min_xy_top_to_bottom(atoms)
        if mm < _STACK_AA_MAX_XY:
            stacking = "AA"
        elif mm > _STACK_AB_MIN_XY:
            stacking = "AB"
        else:
            continue
        e_tot = float(e_tot)
        if key not in buckets[stacking] or e_tot < buckets[stacking][key][1]:
            buckets[stacking][key] = (atoms, e_tot)

    out: dict[str, PlotFrames] = {}
    for stacking in STACKINGS:
        keys = sorted(buckets[stacking])
        frames = PlotFrames(
            atoms=[buckets[stacking][k][0] for k in keys],
            dft_energies=np.asarray(
                [buckets[stacking][k][1] for k in keys], dtype=float,
            ),
        )
        out[stacking] = frames
        if frames.atoms:
            print(
                f"    {stacking}: {len(frames.atoms)} unique-d frames, "
                f"d ∈ [{keys[0]:g}, {keys[-1]:g}]",
                flush=True,
            )
        else:
            print(f"    {stacking}: 0 frames", flush=True)
    return out


def frames_to_pes(frames: PlotFrames) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``(d, E/atom)`` for one stacking."""
    if not frames.atoms:
        return np.array([]), np.array([])
    d = np.asarray([interlayer_separation(a) for a in frames.atoms], dtype=float)
    e = np.asarray(
        [float(et) / len(a) for a, et in zip(frames.atoms, frames.dft_energies)],
        dtype=float,
    )
    order = np.argsort(d)
    return d[order], e[order]


# ── layer-separation estimator ────────────────────────────────────────────────

def quadratic_equilibrium_separation(
    d: np.ndarray,
    e: np.ndarray,
    *,
    n_fit: int = N_QUAD_EQ_POINTS,
) -> float:
    """Fit a quadratic to the ``n_fit`` lowest-energy points; return vertex *d*."""
    d_arr = np.asarray(d, dtype=float).ravel()
    e_arr = np.asarray(e, dtype=float).ravel()
    m = np.isfinite(d_arr) & np.isfinite(e_arr)
    d_arr, e_arr = d_arr[m], e_arr[m]
    if d_arr.size == 0:
        return float("nan")
    if d_arr.size < 3:
        return float(d_arr[int(np.argmin(e_arr))])

    n = min(int(n_fit), int(d_arr.size))
    idx = np.argpartition(e_arr, n - 1)[:n]
    d_f = d_arr[idx]
    e_f = e_arr[idx]
    order = np.argsort(d_f)
    d_f, e_f = d_f[order], e_f[order]
    if np.unique(np.round(d_f, 8)).size < 3:
        return float(d_arr[int(np.argmin(e_arr))])

    a, b, _c = np.polyfit(d_f, e_f, 2)
    if not np.isfinite(a) or a <= 0.0:
        return float(d_arr[int(np.argmin(e_arr))])
    d_min = float(-b / (2.0 * a))
    if d_min < float(d_f[0]) or d_min > float(d_f[-1]):
        return float(d_arr[int(np.argmin(e_arr))])
    return d_min


# ── POD ensemble evaluation ───────────────────────────────────────────────────

def _evaluate_energies(calc_obj, atoms_list: list) -> np.ndarray:
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


def evaluate_pod_ensemble_on_frames(
    model_name: str,
    ab_frames: PlotFrames,
    aa_frames: PlotFrames,
    *,
    ensemble_dir: str,
    temperature: float | None,
    calibration_metrics_dir: str,
    calibration_target: str,
    n_samples: int,
    seed: int,
    extra_kw: dict | None = None,
) -> tuple[PesSeries, np.ndarray, np.ndarray]:
    """
    Evaluate the POD energy ensemble on the plotted AB/AA structures.

    Returns ``(pes_series, ab_eq_samples, aa_eq_samples)``.
    """
    pkl_path, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_target=calibration_target,
    )
    print(f"  Ensemble: {pkl_path}  (T={t_used:g})", flush=True)
    ens_dict = load_ensemble_pickle(pkl_path)
    ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)
    order = np.random.default_rng(seed).permutation(ensemble.shape[0])
    ensemble = ensemble[order]

    ab_atoms = list(ab_frames.atoms)
    aa_atoms = list(aa_frames.atoms)
    ab_d = np.asarray([interlayer_separation(a) for a in ab_atoms], dtype=float)
    aa_d = np.asarray([interlayer_separation(a) for a in aa_atoms], dtype=float)
    ab_nat = np.asarray([len(a) for a in ab_atoms], dtype=float)
    aa_nat = np.asarray([len(a) for a in aa_atoms], dtype=float)

    calc, set_params_fn, load_name = build_uq_calculator(
        model_name, extra_kw=extra_kw or None,
    )
    ab_rows: list[np.ndarray] = []
    aa_rows: list[np.ndarray] = []
    ab_eq: list[float] = []
    aa_eq: list[float] = []
    try:
        for theta in ensemble:
            if len(ab_rows) >= n_samples:
                break
            try:
                apply_uq_parameters(calc, theta, set_params_fn)
                e_ab = _evaluate_energies(calc, ab_atoms)
                e_aa = _evaluate_energies(calc, aa_atoms)
                if (
                    e_ab.size != len(ab_atoms)
                    or e_aa.size != len(aa_atoms)
                    or not np.all(np.isfinite(e_ab))
                    or not np.all(np.isfinite(e_aa))
                ):
                    continue
                e_ab = np.asarray(e_ab, dtype=float)
                e_aa = np.asarray(e_aa, dtype=float)
                ab_rows.append(e_ab)
                aa_rows.append(e_aa)
                ab_eq.append(quadratic_equilibrium_separation(ab_d, e_ab / ab_nat))
                aa_eq.append(quadratic_equilibrium_separation(aa_d, e_aa / aa_nat))
            except Exception as exc:
                print(f"    skip ensemble member: {exc}", file=sys.stderr)
                continue
    finally:
        if hasattr(calc, "close"):
            calc.close()

    n_ok = len(ab_rows)
    if n_ok == 0:
        raise RuntimeError(f"No successful ensemble members for {model_name}")

    ab_stack = np.vstack(ab_rows)
    aa_stack = np.vstack(aa_rows)
    ab_mean = np.mean(ab_stack, axis=0)
    aa_mean = np.mean(aa_stack, axis=0)
    ab_std = np.std(ab_stack, axis=0, ddof=1) if n_ok > 1 else np.zeros_like(ab_mean)
    aa_std = np.std(aa_stack, axis=0, ddof=1) if n_ok > 1 else np.zeros_like(aa_mean)

    label = load_name.split("_POD_index_")[0] if "_POD_index_" in model_name else load_name
    if model_name.startswith(load_name):
        label = model_name.split("_POD_index_")[0]

    print(
        f"  {n_ok} ensemble member(s) on "
        f"{len(ab_atoms)} AB + {len(aa_atoms)} AA structures",
        flush=True,
    )
    series = PesSeries(
        label=label,
        color=_COLOR_POD,
        ab_d=ab_d,
        ab_e=ab_mean / ab_nat,
        aa_d=aa_d,
        aa_e=aa_mean / aa_nat,
        ab_e_std=ab_std / ab_nat,
        aa_e_std=aa_std / aa_nat,
    )
    return (
        series,
        np.asarray(ab_eq, dtype=float),
        np.asarray(aa_eq, dtype=float),
    )


# ── Morse fit (for Morse-shifted plot only) ───────────────────────────────────

def _morse(d, de_well, a, d_eq, e_inf):
    x = np.asarray(d, dtype=float) - float(d_eq)
    return float(de_well) * (np.exp(-2.0 * a * x) - 2.0 * np.exp(-a * x)) + float(e_inf)


def fit_morse(d: np.ndarray, e: np.ndarray, *, d_min_fit: float = 2.8):
    """Return ``(params, e_inf)`` with ``params=(D_e, a, d_e, E_∞)``, or ``None``."""
    d = np.asarray(d, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    m = np.isfinite(d) & np.isfinite(e)
    d, e = d[m], e[m]
    if d.size < 4:
        return None
    order = np.argsort(d)
    d, e = d[order], e[order]
    i_min = int(np.argmin(e))
    d_eq0 = float(d[i_min])
    e_min = float(e[i_min])
    e_inf0 = float(0.5 * (e[-1] + e[-2])) if d.size >= 2 else float(e[-1])
    de0 = max(e_inf0 - e_min, 1e-6)
    keep = d >= float(d_min_fit)
    d_fit, e_fit = d[keep], e[keep]
    if d_fit.size < 4:
        d_fit, e_fit = d, e
    bounds = (
        (max(1e-8, 0.25 * de0), 0.05, max(d_eq0 - 0.35, d_fit.min() * 0.8), e_inf0 - 5e-4),
        (max(de0 * 4.0, de0 + 1e-3), 12.0, d_eq0 + 0.35, e_inf0 + 2e-3),
    )
    best = None
    for a0 in (1.0, 1.2, 1.5, 2.0):
        try:
            popt, _ = curve_fit(
                _morse, d_fit, e_fit, p0=(de0, a0, d_eq0, e_inf0),
                bounds=bounds, maxfev=50_000,
            )
        except Exception:
            continue
        rss = float(np.sum((_morse(d_fit, *popt) - e_fit) ** 2))
        if best is None or rss < best[1]:
            best = (np.asarray(popt, dtype=float), rss)
    if best is None:
        return None
    return best[0]


# ── plotting ──────────────────────────────────────────────────────────────────

def _stacking_xy(series: PesSeries, stacking: str):
    if stacking == "AA":
        return series.aa_d, series.aa_e, series.aa_e_std, _MARKER_AA
    return series.ab_d, series.ab_e, series.ab_e_std, _MARKER_AB


def _cubic_spline(d: np.ndarray, e: np.ndarray, n_fine: int = 400):
    d = np.asarray(d, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    m = np.isfinite(d) & np.isfinite(e)
    d, e = d[m], e[m]
    if d.size < 2:
        return None, None
    order = np.argsort(d)
    d, e = d[order], e[order]
    d_fine = np.linspace(d[0], d[-1], max(n_fine, d.size))
    if d.size < 4:
        return d_fine, np.interp(d_fine, d, e)
    return d_fine, CubicSpline(d, e, bc_type="natural")(d_fine)


def plot_morse_shifted(series_list: list[PesSeries], out_path: Path, dpi: int = 180) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    d_line = np.linspace(XLIM_MORSE[0], XLIM_MORSE[1], 400)
    for series in series_list:
        for stacking in STACKINGS:
            d, e, e_std, marker = _stacking_xy(series, stacking)
            params = fit_morse(d, e)
            if params is None:
                print(f"  skip Morse {series.label} {stacking}", flush=True)
                continue
            e_inf = float(params[3])
            m = (d >= XLIM_MORSE[0] - 0.05) & (d <= XLIM_MORSE[1] + 0.05)
            d_p, e_p = d[m], e[m] - e_inf
            if d_p.size:
                order = np.argsort(d_p)
                d_p, e_p = d_p[order], e_p[order]
                if e_std is not None and np.asarray(e_std).size == d.size:
                    std_p = np.asarray(e_std, dtype=float)[m][order]
                    if np.any(std_p > 0):
                        ax.fill_between(
                            d_p, e_p - std_p, e_p + std_p,
                            color=series.color, alpha=0.28, lw=0, zorder=1,
                        )
                ax.plot(
                    d_p, e_p, ls="none", marker=marker, ms=7,
                    color=series.color, zorder=3,
                )
            ax.plot(
                d_line, _morse(d_line, *params) - e_inf,
                "-", lw=2.0, color=series.color,
                label=f"{series.label} ({stacking})", zorder=4,
            )
    ax.set_xlabel("Interlayer separation (Å)")
    ax.set_ylabel(r"$E - E_\infty$ (eV/atom)")
    ax.set_xlim(*XLIM_MORSE)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_ab_eq_shifted(series_list: list[PesSeries], out_path: Path, dpi: int = 180) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for series in series_list:
        params = fit_morse(series.ab_d, series.ab_e)
        if params is not None:
            l0 = float(params[2])
            e_ab_l0 = float(_morse(np.array([l0]), *params)[0])
        elif series.ab_d.size:
            i = int(np.argmin(series.ab_e))
            l0, e_ab_l0 = float(series.ab_d[i]), float(series.ab_e[i])
        else:
            print(f"  skip AB-eq {series.label}", flush=True)
            continue
        print(f"  AB eq {series.label}: l_0={l0:.4f} Å", flush=True)

        for stacking in STACKINGS:
            d, e, e_std, marker = _stacking_xy(series, stacking)
            m = (d >= XLIM_AB_EQ[0] - 0.05) & (d <= XLIM_AB_EQ[1] + 0.05)
            d_p, e_p = d[m], e[m] - e_ab_l0
            if not d_p.size:
                continue
            order = np.argsort(d_p)
            d_p, e_p = d_p[order], e_p[order]
            if e_std is not None and np.asarray(e_std).size == d.size:
                std_p = np.asarray(e_std, dtype=float)[m][order]
                if np.any(std_p > 0):
                    ax.fill_between(
                        d_p, e_p - std_p, e_p + std_p,
                        color=series.color, alpha=0.28, lw=0, zorder=1,
                    )
            ax.plot(
                d_p, e_p, ls="none", marker=marker, ms=7,
                color=series.color, zorder=3,
            )
            d_s, e_s = _cubic_spline(d_p, e_p)
            if d_s is not None:
                ax.plot(
                    d_s, e_s, "-", lw=2.0, color=series.color,
                    label=f"{series.label} ({stacking})", zorder=4,
                )
    ax.set_xlabel("Interlayer separation (Å)")
    ax.set_ylabel(r"$E - E_{\mathrm{AB}}(l_0)$ (eV/atom)")
    ax.set_xlim(*XLIM_AB_EQ)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_eq_sep_histograms(
    *,
    ab_samples: np.ndarray,
    aa_samples: np.ndarray,
    dft_ab: float,
    dft_aa: float,
    out_path: Path,
    dpi: int = 180,
    n_bins: int = DEFAULT_HIST_BINS,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    legend_handles = legend_labels = None
    for ax, stacking, samples, dft_val in zip(
        axes, STACKINGS, (ab_samples, aa_samples), (dft_ab, dft_aa),
    ):
        vals = np.asarray(samples, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.axis("off")
            continue
        _n, _e, patches = ax.hist(
            vals, bins=n_bins, color="C0", alpha=0.75,
            edgecolor="white", lw=0.5, zorder=2,
        )
        dft_line = None
        if np.isfinite(dft_val):
            dft_line = ax.axvline(dft_val, color="C2", lw=2.5, zorder=5)
        if legend_handles is None:
            legend_handles = [patches[0]]
            legend_labels = ["ensemble"]
            if dft_line is not None:
                legend_handles.append(dft_line)
                legend_labels.append("DFT")
        ax.set_xlabel("Interlayer separation (Å)", fontdict=CSFONT)
        ax.set_ylabel("count", fontdict=CSFONT)
        ax.grid(True, alpha=0.3, axis="y")
    if legend_handles is not None:
        axes[-1].legend(
            legend_handles, legend_labels, loc="best",
            prop={"family": CSFONT["fontname"], "size": 15},
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


# ── CLI / main ────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name.strip())


def _resolve_models(patterns: list[str], args, cli_hp: dict, ensemble_dir: str) -> list[str]:
    expanded = expand_models_for_relaxation(patterns, ensemble_dir)
    seen: set[str] = set()
    out: list[str] = []
    for pattern in expanded:
        name = expand_ensemble_model_name(pattern, args, cli_hp)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot unstrained AB/AA bilayer graphene PES (DFT + POD ensemble).",
    )
    add_energy_models_arg(p, required=False)
    p.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    p.add_argument("-M", type=int, default=10)
    p.add_argument("-W", type=int, default=6)
    p.add_argument("--POD-index", type=int, default=None, dest="pod_index")
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--calibration-metrics-dir", default=DEFAULT_CALIBRATION_METRICS_DIR)
    p.add_argument("--calibration-target", default="energy")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    add_hyperparam_args(p)
    return p.parse_known_args()


def main() -> None:
    args, unknown = _parse_args()
    cli_hp = collect_workflow_hyperparams(args, unknown)
    if args.pod_index is None and "POD_index" in cli_hp:
        args.pod_index = int(cli_hp["POD_index"])

    os.chdir(UQ_DIR)
    models = (
        _resolve_models(args.models, args, cli_hp, args.ensemble_dir)
        if args.models
        else []
    )
    if models:
        print(f"Models: {models}", flush=True)

    print("\nLoading unstrained AB/AA structures …", flush=True)
    frames = load_unstrained_ab_aa_frames()
    ab_d, ab_e = frames_to_pes(frames["AB"])
    aa_d, aa_e = frames_to_pes(frames["AA"])
    series_list = [
        PesSeries(
            label="rVV10", color=_COLOR_RVV10,
            ab_d=ab_d, ab_e=ab_e, aa_d=aa_d, aa_e=aa_e,
        )
    ]
    dft_ab_eq = quadratic_equilibrium_separation(ab_d, ab_e)
    dft_aa_eq = quadratic_equilibrium_separation(aa_d, aa_e)
    print(
        f"\nrVV10 DFT d_eq (quadratic, Å):  AB={dft_ab_eq:.4f}  AA={dft_aa_eq:.4f}",
        flush=True,
    )

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tag = "dft"

    for model_name in models:
        print(f"\n--- Model: {model_name} ---", flush=True)
        if not is_uq_energy_model(model_name):
            print(f"  Skipping non-UQ model {model_name!r}", file=sys.stderr)
            continue
        try:
            pod_series, ab_eq, aa_eq = evaluate_pod_ensemble_on_frames(
                model_name,
                frames["AB"],
                frames["AA"],
                ensemble_dir=args.ensemble_dir,
                temperature=args.temperature,
                calibration_metrics_dir=args.calibration_metrics_dir,
                calibration_target=args.calibration_target,
                n_samples=args.n_samples,
                seed=args.seed,
                extra_kw=cli_hp or None,
            )
        except Exception as exc:
            print(f"  Ensemble failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        series_list.append(pod_series)
        tag = _safe_filename(model_name)

        ab_mu, ab_sig = float(np.mean(ab_eq)), float(np.std(ab_eq, ddof=1))
        aa_mu, aa_sig = float(np.mean(aa_eq)), float(np.std(aa_eq, ddof=1))
        print(
            f"\n{pod_series.label} d_eq (quadratic, mean±std Å):\n"
            f"  AB: {ab_mu:.4f} ± {ab_sig:.4f}  (DFT {dft_ab_eq:.4f})\n"
            f"  AA: {aa_mu:.4f} ± {aa_sig:.4f}  (DFT {dft_aa_eq:.4f})",
            flush=True,
        )
        plot_eq_sep_histograms(
            ab_samples=ab_eq,
            aa_samples=aa_eq,
            dft_ab=dft_ab_eq,
            dft_aa=dft_aa_eq,
            out_path=figures_dir / f"{tag}_bilayer_graphene_interlayer_separation_histogram.png",
            dpi=args.dpi,
            n_bins=args.hist_bins,
        )

    print("\nWriting PES figures …", flush=True)
    plot_morse_shifted(
        series_list,
        figures_dir / f"{tag}_bilayer_graphene_pes_morse_shifted_3_to_8A.png",
        dpi=args.dpi,
    )
    plot_ab_eq_shifted(
        series_list,
        figures_dir / f"{tag}_bilayer_graphene_pes_ab_eq_shifted_3_to_4A.png",
        dpi=args.dpi,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
