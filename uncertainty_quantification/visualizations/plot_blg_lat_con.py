#!/usr/bin/env python3
"""
Total energy vs in-plane lattice constant for bilayer graphene.

DFT reference: single-point energies in ``data/blg_lat_con_structures.xyz``.
Model ensemble: total energies on the same structures for each MCMC draw.
Plots show ensemble mean ± std with DFT overlay, one figure per stacking.
At the end, prints the equilibrium lattice constant from a quadratic ``E(a)``
fit for DFT and ensemble (mean ± std).

Output: ``figures/<model>_<stacking>_energy_vs_lat_con.png``

Examples
--------
::

    python visualizations/plot_blg_lat_con.py --models POD_energy
    python visualizations/plot_blg_lat_con.py \\
        --models 'POD_energy_POD_index*' --n-samples 50 --temperature 0.5

With no ``--models``, uses the ``POD_energy_POD_index*`` ensemble with the lowest
NLL on the saved calibration grid.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

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
from blg_model_builder.strain_data import identify_stacking, layers_have_uniform_z

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
DEFAULT_DFT_XYZ = REPO_ROOT / "data" / "blg_lat_con_structures.xyz"
DEFAULT_ENSEMBLE_DIR = "ensembles"
STACKINGS = ("AB", "AA")

DEFAULT_N_SAMPLES = 500
DEFAULT_ENSEMBLE_SHUFFLE_SEED = 0


@dataclass(frozen=True)
class LatConFrame:
    atoms: Atoms
    stacking: str
    lat_con: float
    dft_energy: float


@dataclass(frozen=True)
class LatConPoint:
    stacking: str
    lat_con: float
    energy: float


def _shuffle_ensemble(ensemble: np.ndarray, seed: int) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 2:
        raise ValueError(f"Expected 2-D ensemble array, got shape {ensemble.shape}")
    order = np.random.default_rng(seed).permutation(ensemble.shape[0])
    return ensemble[order]


def _is_lammps_error(exc: BaseException) -> bool:
    cur: Optional[BaseException] = exc
    seen: set[int] = set()
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


def lattice_constant(atoms: Atoms) -> float:
    """In-plane lattice constant *a* (Å) from the first Bravais vector."""
    return float(np.linalg.norm(np.asarray(atoms.get_cell()[0], dtype=float)))


def load_lat_con_frames(xyz_path: Path) -> List[LatConFrame]:
    """Load DFT single-point lattice-constant scan structures from extxyz."""
    if not xyz_path.is_file():
        raise FileNotFoundError(f"DFT reference not found: {xyz_path}")

    frames: List[LatConFrame] = []
    n_skip_z = 0
    for atoms in ase.io.read(str(xyz_path), index=":"):
        if not layers_have_uniform_z(atoms):
            n_skip_z += 1
            continue
        stack = identify_stacking(atoms)
        if stack not in STACKINGS:
            continue
        frames.append(
            LatConFrame(
                atoms=atoms,
                stacking=stack,
                lat_con=lattice_constant(atoms),
                dft_energy=float(atoms.get_potential_energy()),
            )
        )

    if n_skip_z:
        print(f"  skipped {n_skip_z} frame(s) with non-uniform layer z", flush=True)
    return frames


def _sorted_indices_for_stacking(
    frames: Sequence[LatConFrame],
    stacking: str,
) -> Tuple[np.ndarray, List[int]]:
    """Return sorted lattice constants and frame indices for one stacking."""
    sub = [(i, f) for i, f in enumerate(frames) if f.stacking == stacking]
    if not sub:
        return np.array([], dtype=float), []
    order = np.argsort([f.lat_con for _, f in sub])
    indices = [sub[j][0] for j in order]
    lat_cons = np.array([frames[i].lat_con for i in indices], dtype=float)
    return lat_cons, indices


def dft_energy_curve(
    frames: Sequence[LatConFrame],
    stacking: str,
) -> Tuple[np.ndarray, np.ndarray, List[LatConPoint]]:
    """Return ``(lat_cons, energies, points)`` for DFT data."""
    lat_cons, indices = _sorted_indices_for_stacking(frames, stacking)
    if not indices:
        return lat_cons, np.array([], dtype=float), []
    energies = np.array([frames[i].dft_energy for i in indices], dtype=float)
    points = [
        LatConPoint(stacking=stacking, lat_con=float(a), energy=float(e))
        for a, e in zip(lat_cons, energies)
    ]
    return lat_cons, energies, points


def evaluate_ensemble_vs_lat_con(
    calc_obj,
    ensemble_shuffled: np.ndarray,
    n_samples: int,
    stacking: str,
    frames: Sequence[LatConFrame],
    *,
    set_params_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Evaluate total energies on all lattice-constant structures for ensemble draws.

    Returns ``(lat_cons, mean_energy, std_energy, energy_ensemble, n_success)``
    where ``energy_ensemble`` has shape ``(n_configs, n_success)``.
    """
    lat_cons, indices = _sorted_indices_for_stacking(frames, stacking)
    n_configs = len(indices)
    if n_configs == 0:
        return lat_cons, np.array([]), np.array([]), np.zeros((0, 0)), 0

    atoms_list = [frames[i].atoms for i in indices]
    energy_ensemble = np.full((n_configs, n_samples), np.nan, dtype=float)
    n_success = 0

    calc_obj.prepare_batch(atoms_list)

    for theta in ensemble_shuffled:
        if n_success >= n_samples:
            break
        try:
            apply_uq_parameters(calc_obj, theta, set_params_fn)
            energies, _ = calc_obj.evaluate_batch()
            energies = np.asarray(energies, dtype=float).ravel()
            if energies.size != n_configs or not np.all(np.isfinite(energies)):
                continue
            energy_ensemble[:, n_success] = energies
            n_success += 1
        except Exception as exc:
            if _is_lammps_error(exc):
                print(f"    skip ensemble member (LAMMPS): {exc}", file=sys.stderr)
            else:
                print(f"    skip ensemble member: {exc}", file=sys.stderr)
            continue

    if n_success:
        mean_energy = np.nanmean(energy_ensemble[:, :n_success], axis=1)
        std_energy = np.nanstd(energy_ensemble[:, :n_success], axis=1)
        ens_out = energy_ensemble[:, :n_success].copy()
    else:
        mean_energy = np.full(n_configs, np.nan)
        std_energy = np.full(n_configs, np.nan)
        ens_out = np.zeros((n_configs, 0), dtype=float)
    return lat_cons, mean_energy, std_energy, ens_out, n_success


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
            calibration_metrics_dir,
            model_name,
            calibration_technique,
            calibration_target,
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
    print(
        f"Auto-selected POD model {best_name!r} (lowest NLL = {best_nll:.6g})",
        flush=True,
    )
    return best_name


def plot_energy_vs_lat_con(
    lat_cons: np.ndarray,
    mean_energy: np.ndarray,
    std_energy: np.ndarray,
    dft_points: Sequence[LatConPoint],
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Save total energy vs lattice constant with DFT overlay."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    ax.plot(lat_cons, mean_energy, label="ensemble mean", color="C0", zorder=2)
    ax.fill_between(
        lat_cons,
        mean_energy - std_energy,
        mean_energy + std_energy,
        alpha=0.3,
        color="C0",
        label="ensemble std",
        zorder=1,
    )

    if dft_points:
        dft_x = np.asarray([p.lat_con for p in dft_points], dtype=float)
        dft_y = np.asarray([p.energy for p in dft_points], dtype=float)
        ax.plot(
            dft_x,
            dft_y,
            "o-",
            color="C2",
            label="DFT",
            zorder=3,
            markersize=6,
        )

    ax.set_xlabel(r"lattice constant $a$ (Å)", fontdict=CSFONT)
    ax.set_ylabel("total energy (eV)", fontdict=CSFONT)
    ax.legend(prop={"family": CSFONT["fontname"], "size": 15})
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_equilibrium_lat_con_summary(
    dft_a_eq: Dict[str, float],
    model_a_eq: Dict[str, Dict[str, Tuple[float, float]]],
) -> None:
    """Print equilibrium *a* from quadratic ``E(a)`` fits."""
    print(
        "\n=== Equilibrium lattice constant (quadratic E(a) minimum) ===",
        flush=True,
    )
    for stacking in STACKINGS:
        a_eq = float(dft_a_eq.get(stacking, float("nan")))
        if np.isfinite(a_eq):
            print(f"  DFT {stacking}:  a = {a_eq:.4f} Å", flush=True)
        else:
            print(f"  DFT {stacking}:  a = nan", flush=True)

    if not model_a_eq:
        return

    for model_name in sorted(model_a_eq):
        print(f"  {model_name}:", flush=True)
        for stacking in STACKINGS:
            mu, sig = model_a_eq[model_name].get(stacking, (float("nan"), float("nan")))
            if np.isfinite(mu) and np.isfinite(sig):
                print(f"    {stacking}:  a = {mu:.4f} ± {sig:.4f} Å", flush=True)
            elif np.isfinite(mu):
                print(f"    {stacking}:  a = {mu:.4f} Å", flush=True)
            else:
                print(f"    {stacking}:  a = nan", flush=True)


def fit_quadratic_energy_vs_lat_con(
    lat_cons: np.ndarray,
    energies: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Quadratic least-squares fit ``E(a)``; return ``(coeffs, a_eq, e_eq)``.

    ``a_eq`` is the parabolic minimum; falls back to the lowest-energy grid
    point if the fit is ill-conditioned or non-convex.
    """
    lat_cons = np.asarray(lat_cons, dtype=float).ravel()
    energies = np.asarray(energies, dtype=float).ravel()
    m = np.isfinite(lat_cons) & np.isfinite(energies)
    lat_cons, energies = lat_cons[m], energies[m]
    if lat_cons.size == 0:
        return np.array([np.nan, np.nan, np.nan]), float("nan"), float("nan")
    if lat_cons.size < 3:
        j = int(np.argmin(energies))
        return np.array([0.0, 0.0, float(energies[j])]), float(lat_cons[j]), float(energies[j])

    order = np.argsort(lat_cons)
    xs, ys = lat_cons[order], energies[order]

    try:
        coeffs = np.polyfit(xs, ys, 2)
    except np.linalg.LinAlgError:
        j = int(np.argmin(ys))
        return np.array([0.0, 0.0, float(ys[j])]), float(xs[j]), float(ys[j])

    a2, a1 = float(coeffs[0]), float(coeffs[1])
    if abs(a2) < 1e-18 or a2 <= 0.0:
        j = int(np.argmin(ys))
        a_eq = float(xs[j])
    else:
        a_eq = float(-0.5 * a1 / a2)
        if a_eq < xs.min() or a_eq > xs.max():
            j = int(np.argmin(ys))
            a_eq = float(xs[j])

    e_eq = float(np.polyval(coeffs, a_eq))
    return coeffs, a_eq, e_eq


def equilibrium_lat_con(
    lat_cons: np.ndarray,
    energies: np.ndarray,
) -> float:
    """Equilibrium lattice constant from a quadratic ``E(a)`` fit."""
    _coeffs, a_eq, _e_eq = fit_quadratic_energy_vs_lat_con(lat_cons, energies)
    return a_eq


def equilibrium_lat_con_ensemble_stats(
    lat_cons: np.ndarray,
    energy_ensemble: np.ndarray,
) -> Tuple[float, float]:
    """Mean and std of equilibrium *a* from quadratic fits over ensemble members."""
    energy_ensemble = np.asarray(energy_ensemble, dtype=float)
    if energy_ensemble.ndim != 2 or energy_ensemble.shape[1] == 0:
        return float("nan"), float("nan")
    a_samples = np.array(
        [
            equilibrium_lat_con(lat_cons, energy_ensemble[:, j])
            for j in range(energy_ensemble.shape[1])
        ],
        dtype=float,
    )
    ok = np.isfinite(a_samples)
    if not np.any(ok):
        return float("nan"), float("nan")
    return float(np.mean(a_samples[ok])), float(np.std(a_samples[ok]))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot bilayer graphene total energy vs lattice constant with ensemble UQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p, required=False)
    p.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="MCMC temperature weight T for ensemble pickle (nearest match).",
    )
    p.add_argument(
        "--calibration-metrics-dir",
        default=DEFAULT_CALIBRATION_METRICS_DIR,
        help="Directory with calibration_*.npz from plot_bayes_factor.py --calculate.",
    )
    p.add_argument(
        "--calibration-target",
        default="energy",
        help="Target key in calibration npz (default: energy).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Target number of successful ensemble evaluations per stacking.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_ENSEMBLE_SHUFFLE_SEED,
        help="RNG seed for ensemble shuffle.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for PNG figures (default: uncertainty_quantification/figures).",
    )
    p.add_argument(
        "--dft-xyz",
        type=Path,
        default=DEFAULT_DFT_XYZ,
        help=(
            "DFT single-point lattice-constant scan structures "
            "(default: data/blg_lat_con_structures.xyz)."
        ),
    )
    p.add_argument("--dpi", type=int, default=150)
    add_hyperparam_args(p)
    args, unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, unknown)

    os.chdir(UQ_DIR)

    dft_xyz = Path(args.dft_xyz)
    if not dft_xyz.is_absolute():
        dft_xyz = REPO_ROOT / dft_xyz if (REPO_ROOT / dft_xyz).exists() else UQ_DIR / dft_xyz

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DFT structures from {dft_xyz}", flush=True)
    lat_con_frames = load_lat_con_frames(dft_xyz)
    if not lat_con_frames:
        p.error(f"No AB/AA lattice-constant frames found in {dft_xyz}")

    dft_by_stack: Dict[str, List[LatConPoint]] = {}
    dft_a_eq: Dict[str, float] = {}
    for stacking in STACKINGS:
        lat_cons, dft_energies, points = dft_energy_curve(lat_con_frames, stacking)
        dft_by_stack[stacking] = points
        dft_a_eq[stacking] = equilibrium_lat_con(lat_cons, dft_energies)
        if points:
            a_vals = ", ".join(f"{p.lat_con:.3f}" for p in points)
            print(f"  DFT {stacking}: {len(points)} point(s)  a (Å) = {a_vals}", flush=True)

    if args.models:
        models = expand_model_patterns(args.models, args.ensemble_dir)
        if not models:
            p.error("No models matched --models patterns.")
    else:
        try:
            models = [
                select_pod_model_lowest_nll(
                    args.ensemble_dir,
                    args.calibration_metrics_dir,
                    calibration_target=args.calibration_target,
                )
            ]
        except ValueError as exc:
            p.error(str(exc))

    print(f"Models: {models}", flush=True)

    model_a_eq: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for model_name in models:
        if not is_uq_energy_model(model_name):
            print(
                f"  skip {model_name!r}: unsupported UQ model",
                file=sys.stderr,
            )
            continue

        print(f"\n--- Model: {model_name} ---", flush=True)
        pkl_path, t_used = resolve_ensemble_pickle(
            model_name,
            args.ensemble_dir,
            args.temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_target=args.calibration_target,
        )
        print(f"  Ensemble pickle: {pkl_path}  (T={t_used:g})", flush=True)

        ens_dict = load_ensemble_pickle(pkl_path)
        ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)
        ensemble_shuffled = _shuffle_ensemble(ensemble, args.seed)
        print(
            f"  Shuffled ensemble (seed={args.seed}): {ensemble.shape[0]} members; "
            f"target {args.n_samples} successful per stacking",
            flush=True,
        )

        calc_obj, set_params_fn, load_name = build_uq_calculator(
            model_name, extra_kw=cli_hyperparams or None,
        )
        print(f"  Calculator: {load_name}", flush=True)

        model_a_eq[model_name] = {}

        for stacking in STACKINGS:
            if not dft_by_stack.get(stacking):
                continue

            print(f"  Stacking {stacking} …", flush=True)
            lat_cons, mean_energy, std_energy, energy_ensemble, n_ok = evaluate_ensemble_vs_lat_con(
                calc_obj,
                ensemble_shuffled,
                args.n_samples,
                stacking,
                lat_con_frames,
                set_params_fn=set_params_fn,
            )
            if n_ok == 0:
                print(f"    no successful ensemble members for {stacking}", file=sys.stderr)
                continue

            a_mean, a_std = equilibrium_lat_con_ensemble_stats(lat_cons, energy_ensemble)
            model_a_eq[model_name][stacking] = (a_mean, a_std)

            out_name = f"{model_name}_{stacking}_energy_vs_lat_con.png"
            out_path = figures_dir / out_name
            plot_energy_vs_lat_con(
                lat_cons,
                mean_energy,
                std_energy,
                dft_by_stack.get(stacking, []),
                out_path,
                dpi=args.dpi,
            )
            print(f"    Wrote {out_path}  (n={n_ok})", flush=True)

        if hasattr(calc_obj, "close"):
            calc_obj.close()

    print_equilibrium_lat_con_summary(dft_a_eq, model_a_eq)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
