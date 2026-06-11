#!/usr/bin/env python3
"""
Relax twisted bilayer graphene (TBLG) with subsampled MCMC ensemble parameters.

For each selected LAMMPS model (``--models`` with optional glob wildcards,
same as :mod:`plot_bayes_factor`). Supported ensembles:

* ``POD_energy`` / ``POD_energy_POD_index_*``
* ``TETB_POD`` / ``TETB_POD_*_POD_index_*``
* ``Tersoff+DRIP``
* ``Allegro_energy`` / ``Allegro_energy_ckpt_*`` (Python ASE FIRE)

* Build a commensurate TBLG supercell with ``flatgraphene`` at ``--twist-angle``.
* Shuffle the MCMC ensemble and relax until ``--n-samples`` trajectories are saved
  (LAMMPS errors are skipped; stop when the ensemble is exhausted).
* Relax each sample with the model's LAMMPS calculator (POD, TETB-POD, Tersoff+KC, Tersoff+DRIP).
* Print max / min layer separation after relaxation (see :func:`layer_separation_metrics`).
* Save a trajectory (initial + relaxed frames) when LAMMPS relaxation completes.
  Samples are skipped (no ``.traj`` written) only if LAMMPS raises an error.
  Non-converged relaxations are still saved.

Examples
--------
::

    python run_uq_propagation_relaxation.py --models 'POD_energy_POD_index*' \\
        --temperature 0.00112884 --twist-angle 9.43 --n-samples 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Tuple

import numpy as np

from blg_model_builder.ensemble_io import (
    DEFAULT_CALIBRATION_METRICS_DIR,
    expand_model_patterns,
    load_ensemble_pickle,
    resolve_ensemble_pickle,
)

from run_uq_propagation_elasticity import (
    DEFAULT_ENSEMBLE_SHUFFLE_SEED,
    DEFAULT_N_SAMPLES as DEFAULT_ELASTICITY_N_SAMPLES,
    _is_lammps_error,
    _shuffle_ensemble,
)
from uq_model_runtime import (
    apply_uq_parameters,
    build_uq_calculator,
    is_uq_energy_model,
    is_uq_python_model,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_N_SAMPLES = DEFAULT_ELASTICITY_N_SAMPLES
DEFAULT_TWIST_ANGLE = 9.43
DEFAULT_LAT_CON = 2.46
DEFAULT_INITIAL_SEP = 3.35
DEFAULT_OUTPUT_DIR = "trajectories/relaxation"

# TBLG relaxation defaults (pod_hyperparameter_search / test_relaxation pod_energy case)
DEFAULT_RELAX_ETOL = 1e-10
DEFAULT_RELAX_FTOL = 1e-10
DEFAULT_RELAX_MAXITER = 1_000
DEFAULT_RELAX_MAXEVAL = 8_000


# ---------------------------------------------------------------------------
# TBLG geometry (flatgraphene)
# ---------------------------------------------------------------------------


def _ensure_mol_id_from_z(atoms) -> None:
    """Assign ``mol-id`` ∈ {1, 2} by z-coordinate if not already set."""
    if atoms.has("mol-id"):
        return
    z = atoms.positions[:, 2]
    mid = float(np.median(z))
    mol = np.where(z < mid, np.int8(1), np.int8(2))
    z1 = float(np.mean(z[mol == 1]))
    z2 = float(np.mean(z[mol == 2]))
    if z1 > z2:
        mol = np.where(mol == 1, np.int8(2), np.int8(1))
    atoms.set_array("mol-id", mol)


def build_tblg_atoms(
    theta_deg: float,
    *,
    lat_con: float = DEFAULT_LAT_CON,
    sep: float = DEFAULT_INITIAL_SEP,
    h_vac: float = 20.0,
):
    """Build a commensurate TBLG supercell with ``flatgraphene``."""
    try:
        import flatgraphene as fg
    except ImportError as exc:
        raise ImportError(
            "flatgraphene is required for TBLG structure generation. "
            "Install it in the active environment."
        ) from exc

    p, q, _ = fg.twist.find_p_q(float(theta_deg), a_tol=0.01)
    atoms = fg.twist.make_graphene(
        cell_type="hex",
        n_layer=2,
        p=p,
        q=q,
        lat_con=float(lat_con),
        sym=["C", "C"],
        mass=[12.01, 12.01],
        sep=float(sep),
        h_vac=float(h_vac),
    )
    _ensure_mol_id_from_z(atoms)
    return atoms


def layer_separation_metrics(atoms) -> Tuple[float, float]:
    """
    Max and min layer separation from Cartesian *z* (Å).

    max_sep = 2 * max(|z - mean(z)|)
    min_sep = 2 * min(|z - mean(z)|)
    """
    z = np.asarray(atoms.get_positions(wrap=False), dtype=float)[:, 2]
    dz = np.abs(z - float(np.mean(z)))
    max_sep = 2.0 * float(np.max(dz))
    min_sep = 2.0 * float(np.min(dz))
    return max_sep, min_sep


def _safe_filename_part(s: str) -> str:
    return re.sub(r"[^\w.\-+]+", "_", str(s))


def _trajectory_path(
    output_dir: str,
    model_name: str,
    temperature_label: str,
    twist_angle: float,
    sample_index: int,
) -> str:
    base = (
        f"{_safe_filename_part(model_name)}"
        f"_T{_safe_filename_part(temperature_label)}"
        f"_theta{twist_angle:g}deg"
        f"_sample{sample_index:04d}.traj"
    )
    return os.path.join(output_dir, base)


def _remove_traj(traj_path: str) -> None:
    """Delete a partial trajectory file after a failed relaxation."""
    try:
        if os.path.isfile(traj_path):
            os.remove(traj_path)
    except OSError:
        pass


def _max_force(atoms) -> float:
    f = np.asarray(atoms.get_forces(), dtype=float)
    return float(np.max(np.linalg.norm(f, axis=1)))


def _write_success_trajectory(traj_path: str, initial, relaxed) -> None:
    """Write initial + relaxed frames after LAMMPS relaxation returns."""
    from ase.io.trajectory import Trajectory

    initial = initial.copy()
    initial.calc = None
    initial.info = dict(initial.info)
    initial.info["frame"] = "initial"
    _ensure_mol_id_from_z(initial)

    relaxed = relaxed.copy()
    relaxed.calc = None
    relaxed.info = dict(relaxed.info)
    relaxed.info["frame"] = "relaxed"
    _ensure_mol_id_from_z(relaxed)

    with Trajectory(traj_path, mode="w") as traj:
        traj.write(initial)
        traj.write(relaxed)


def relax_tblg_sample(
    atoms_template,
    calc,
    theta: np.ndarray,
    traj_path: str,
    *,
    relax_backend: str,
    etol: float,
    ftol: float,
    maxiter: int,
    maxeval: int,
    set_params_fn=None,
):
    """
    Relax one ensemble draw and write ``traj_path``.

    Raises if LAMMPS (or the ASE optimizer driving LAMMPS) fails.  Does not
    reject non-converged force norms — those structures are still saved.
    """
    atoms = atoms_template.copy()
    apply_uq_parameters(calc, theta, set_params_fn)
    atoms.calc = calc
    _ensure_mol_id_from_z(atoms)

    initial = atoms.copy()
    initial.calc = calc

    backend = str(relax_backend).strip().lower()
    if backend == "lammps":
        relaxed = calc.relax_structure(
            atoms,
            relax_backend="lammps",
            etol=etol,
            ftol=ftol,
            maxiter=maxiter,
            maxeval=maxeval,
        )
        relaxed.calc = calc
        _ensure_mol_id_from_z(relaxed)
        _write_success_trajectory(traj_path, initial, relaxed)
        return relaxed

    if backend == "ase":
        from ase.optimize import FIRE

        dyn = FIRE(atoms, logfile=None)
        dyn.run(fmax=ftol, steps=maxiter)
        atoms.calc = calc
        _ensure_mol_id_from_z(atoms)
        _write_success_trajectory(traj_path, initial, atoms)
        return atoms

    raise ValueError(f"Unknown relax_backend {relax_backend!r}; use 'lammps' or 'ase'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="TBLG relaxation UQ propagation from MCMC LAMMPS ensembles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p)
    p.add_argument("--ensemble-dir", default="ensembles")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="MCMC temperature weight T for ensemble pickle (nearest match). "
        "Default: T that minimizes miscalibration_area in --calibration-metrics-dir.",
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
        help="Target number of successful relaxations (trajectories saved).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_ENSEMBLE_SHUFFLE_SEED,
        help="RNG seed for ensemble shuffle (must match run_uq_propagation_elasticity.py).",
    )
    p.add_argument(
        "--twist-angle",
        type=float,
        default=DEFAULT_TWIST_ANGLE,
        help="Twist angle in degrees (flatgraphene commensurate cell).",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for relaxation trajectory files.",
    )
    p.add_argument("--lat-con", type=float, default=DEFAULT_LAT_CON)
    p.add_argument("--initial-sep", type=float, default=DEFAULT_INITIAL_SEP)
    p.add_argument(
        "--relax-backend",
        choices=("lammps", "ase"),
        default="lammps",
        help="lammps: FIRE minimize (trajectory = initial + final). "
        "ase: ASE FIRE with full trajectory.",
    )
    p.add_argument("--relax-etol", type=float, default=DEFAULT_RELAX_ETOL)
    p.add_argument("--relax-ftol", type=float, default=DEFAULT_RELAX_FTOL)
    p.add_argument("--relax-maxiter", type=int, default=DEFAULT_RELAX_MAXITER)
    p.add_argument("--relax-maxeval", type=int, default=DEFAULT_RELAX_MAXEVAL)
    add_hyperparam_args(p)
    args, _unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, _unknown)
    if cli_hyperparams:
        print(f"  CLI hyperparameters: {cli_hyperparams}", flush=True)

    os.chdir(HERE)
    models = expand_model_patterns(args.models, args.ensemble_dir)
    if not models:
        p.error("No models matched --models patterns.")
    print(f"Models: {models}", flush=True)

    print(
        f"\nBuilding TBLG at θ={args.twist_angle:g}° "
        f"(lat_con={args.lat_con:g} Å, sep={args.initial_sep:g} Å) …",
        flush=True,
    )
    tblg_template = build_tblg_atoms(
        args.twist_angle,
        lat_con=args.lat_con,
        sep=args.initial_sep,
    )
    print(f"  n_atoms={len(tblg_template)}", flush=True)

    for model_name in models:
        print(f"\n--- Model: {model_name} ---", flush=True)
        if not is_uq_energy_model(model_name):
            print(
                f"  Warning: unsupported model (need UQ energy model); "
                f"skipping {model_name!r}.",
                file=sys.stderr,
            )
            continue

        relax_backend = args.relax_backend
        if is_uq_python_model(model_name) and relax_backend == "lammps":
            relax_backend = "ase"
            print("  Using relax_backend=ase for Python Allegro model.", flush=True)

        pkl_path, t_used = resolve_ensemble_pickle(
            model_name,
            args.ensemble_dir,
            args.temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_target=args.calibration_target,
        )
        t_label = f"{t_used:g}"
        print(f"  Ensemble pickle: {pkl_path}  (T={t_label})", flush=True)

        ens_dict = load_ensemble_pickle(pkl_path)
        ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)
        ensemble_shuffled = _shuffle_ensemble(ensemble, args.seed)
        print(
            f"  Shuffled ensemble (seed={args.seed}): {ensemble.shape[0]} members; "
            f"target {args.n_samples} successful relaxations",
            flush=True,
        )

        calc_obj, set_params_fn, _load_name = build_uq_calculator(
            model_name, extra_kw=cli_hyperparams or None,
        )
        print(f"  Calculator: {_load_name}", flush=True)
        calc_obj.prepare_batch([tblg_template])

        out_dir = os.path.join(
            args.output_dir,
            _safe_filename_part(model_name),
            f"T{t_label}",
            f"theta{args.twist_angle:g}deg",
        )
        os.makedirs(out_dir, exist_ok=True)

        n_saved = 0
        n_skipped = 0
        n_tried = 0
        for theta in ensemble_shuffled:
            if n_saved >= args.n_samples:
                break
            n_tried += 1
            traj_path = _trajectory_path(
                out_dir, model_name, t_label, args.twist_angle, n_saved,
            )
            print(
                f"  attempt {n_tried} → target sample {n_saved + 1}/{args.n_samples} …",
                flush=True,
            )
            try:
                relaxed = relax_tblg_sample(
                    tblg_template,
                    calc_obj,
                    theta,
                    traj_path,
                    relax_backend=relax_backend,
                    etol=args.relax_etol,
                    ftol=args.relax_ftol,
                    maxiter=args.relax_maxiter,
                    maxeval=args.relax_maxeval,
                    set_params_fn=set_params_fn,
                )
            except Exception as exc:
                _remove_traj(traj_path)
                if not _is_lammps_error(exc):
                    raise
                n_skipped += 1
                print(
                    f"    SKIPPED (LAMMPS error, no traj saved): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue

            n_saved += 1
            print(f"    saved {traj_path}", flush=True)
            max_sep, min_sep = layer_separation_metrics(relaxed)
            fmax = _max_force(relaxed)
            print(
                f"    max layer separation = {max_sep:.6f} Å  "
                f"(2·max|z−⟨z⟩|)",
                flush=True,
            )
            print(
                f"    min layer separation = {min_sep:.6f} Å  "
                f"(2·min|z−⟨z⟩|)",
                flush=True,
            )
            print(f"    max |F| = {fmax:.4e} eV/Å", flush=True)

        print(
            f"  Finished {model_name}: {n_saved}/{args.n_samples} saved, "
            f"{n_skipped} skipped, {n_tried} tried "
            f"(pool size {ensemble_shuffled.shape[0]})",
            flush=True,
        )
        calc_obj.close()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
