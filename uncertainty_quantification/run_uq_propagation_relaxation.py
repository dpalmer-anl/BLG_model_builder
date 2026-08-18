#!/usr/bin/env python3
"""
Relax twisted bilayer graphene (TBLG) with subsampled MCMC ensemble parameters.

For each selected LAMMPS model (``--models`` with optional glob wildcards,
same as :mod:`plot_bayes_factor`). If ``--models`` is omitted, the passing POD
row in ``pod_hyperparam_search.csv`` with the lowest
calibration NLL is chosen automatically. Supported ensembles:

* ``POD_energy`` / ``POD_energy_POD_index_*``
* ``TETB_POD`` / ``TETB_POD_*_POD_index_*``
* ``Tersoff+DRIP``
* ``Allegro_energy`` / ``Allegro_energy_ckpt_*`` (Python ASE LBFGS)

* Build a commensurate TBLG supercell with ``flatgraphene`` at ``--twist-angle``.
* Prefer a pre-screened parameter subset under ``ensembles/propagation/`` when
  present for ``(model, T, twist)``. Otherwise shuffle the MCMC ensemble.
* With ``--build-propagation-subset``, screen the shuffled ensemble using this
  script's default relax settings until ``--n-samples`` draws satisfy
  ``3 ≤ min/max layer sep ≤ 4`` Å and ``max|F| < 1e-4`` eV/Å, then write the
  subset ``.npz`` and exit.
* Shuffle / screen until ``--n-samples`` trajectories are saved
  (successful and failed relaxations both count when not using a subset; stop
  when the ensemble is exhausted).
* Relax each sample with the model's calculator (default LAMMPS CG via
  ``--relax-backend lammps``; override with ``--relax-min-style fire``.
  ``--relax-backend ase`` uses ASE
  :class:`~ase.optimize.LBFGS` with a small maxstep).
* Print max / min layer separation after relaxation (see :func:`layer_separation_metrics`).
* Save a trajectory (initial + relaxed frames) when relaxation completes.
  During ASE LBFGS, the same sample ``.traj`` is overwritten every 1000 steps
  as a checkpoint; the final write replaces it. The relaxed frame stores total
  energy, forces, and per-atom (local) energies
  via a :class:`~ase.calculators.singlepoint.SinglePointCalculator` (and a
  ``local_energy`` array when pe/atom is available).
  If relaxation raises an error, still write a trajectory whose filename ends in
  ``_FAIL.traj``, with frames labeled as failed. Non-converged (but non-crashing)
  relaxations are saved as ordinary (non-FAIL) trajectories.
* By default (``--skip-existing``), only **pending** sample indices are assigned
  to ranks (round-robin).  Pending means no OK/FAIL trajectory, **or** the
  existing final frame does not meet ``--relax-ftol`` (missing forces count as
  not met).  Unconverged trajectories are resumed from their last frame;
  converged ones are not re-assigned.

Examples
--------
::

    python run_uq_propagation_relaxation.py --models 'POD_energy_POD_index*' \\
        --temperature 0.00112884 --twist-angle 9.43 --n-samples 10

    # Parallel over ensemble samples (unique index per task; covers all n_samples)::
    mpirun -np 32 python run_uq_propagation_relaxation.py \\
        --models POD_energy_POD_index_27_b17b22afb666496b --twist-angle 9.43

    # Same under Slurm (``srun`` with --ntasks=32); uses SLURM_PROCID if MPI
    # world size is 1 so tasks do not all redo every sample.

``POD_energy_POD_index*`` expands from passing rows in
``pod_hyperparam_search/pod_hyperparam_search.csv`` whose
associated fit cache and ensemble folder exist.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from typing import Any, Tuple

import numpy as np

from blg_model_builder.ensemble_io import (
    DEFAULT_CALIBRATION_METRICS_DIR,
    expand_model_patterns,
    load_ensemble_pickle,
    load_metrics_npz,
    metrics_npz_path,
    resolve_ensemble_pickle,
)

from run_uq_propagation_elasticity import (
    DEFAULT_ENSEMBLE_SHUFFLE_SEED,
    DEFAULT_N_SAMPLES as DEFAULT_ELASTICITY_N_SAMPLES,
    _shuffle_ensemble,
)
from uq_model_runtime import (
    apply_uq_parameters,
    build_uq_calculator,
    is_uq_energy_model,
    is_uq_python_model,
)
from blg_model_builder.pod_model_selection import (
    pod_energy_ensemble_names_from_csv,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
)

from blg_model_builder.strain_data import LAT_CON

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_N_SAMPLES = 500
DEFAULT_TWIST_ANGLE = 9.43
DEFAULT_LAT_CON = LAT_CON
DEFAULT_INITIAL_SEP = 3.35
DEFAULT_OUTPUT_DIR = "trajectories/relaxation"
DEFAULT_PROPAGATION_DIR = os.path.join("ensembles", "propagation")


def _mpi_state() -> tuple[Any, int, int]:
    """Return ``(comm_or_None, rank, size)`` for parallel sample assignment.

    Preference order:
    1. ``mpi4py`` world with ``size > 1`` (``mpirun`` / PMI-aware ``srun``)
    2. Slurm env ``SLURM_PROCID`` / ``SLURM_NTASKS`` when ``srun`` launches
       multiple independent Python processes that do **not** share an MPI world
       (otherwise every task would see rank=0,size=1 and redo all samples)
    3. Serial ``(None, 0, 1)``
    """
    # 1) Real MPI multi-rank world
    try:
        from mpi4py import MPI  # noqa: PLC0415

        comm = MPI.COMM_WORLD
        size = int(comm.Get_size())
        rank = int(comm.Get_rank())
        if size > 1:
            return comm, rank, size
    except Exception:
        comm = None
    else:
        # mpi4py present but size==1 — may still be one of many Slurm tasks
        pass

    # 2) Slurm multi-task without a working multi-rank MPI communicator
    ntasks = os.environ.get("SLURM_NTASKS") or os.environ.get("SLURM_NPROCS")
    procid = os.environ.get("SLURM_PROCID")
    if ntasks is not None and procid is not None:
        size = int(ntasks)
        rank = int(procid)
        if size > 1:
            if not (0 <= rank < size):
                raise RuntimeError(
                    f"Invalid Slurm task id: SLURM_PROCID={rank}, SLURM_NTASKS={size}"
                )
            return None, rank, size

    return None, 0, 1


def _partition_sample_indices(n_samples: int, rank: int, size: int) -> list[int]:
    """Round-robin unique sample indices for this rank.

    Guarantees for ``0 <= rank < size``:
    * no two ranks share an index
    * the union over ranks is exactly ``range(n_samples)``
    """
    n_samples = int(n_samples)
    size = int(size)
    rank = int(rank)
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")
    if size < 1:
        raise ValueError(f"parallel size must be >= 1, got {size}")
    if not (0 <= rank < size):
        raise ValueError(f"rank must satisfy 0 <= rank < size, got rank={rank}, size={size}")
    return list(range(rank, n_samples, size))


def _partition_index_list(indices: list[int], rank: int, size: int) -> list[int]:
    """Round-robin partition of an arbitrary index list (e.g. pending samples)."""
    size = int(size)
    rank = int(rank)
    if size < 1:
        raise ValueError(f"parallel size must be >= 1, got {size}")
    if not (0 <= rank < size):
        raise ValueError(f"rank must satisfy 0 <= rank < size, got rank={rank}, size={size}")
    return list(indices[rank::size])


def _assert_partition_covers_all(n_samples: int, size: int) -> None:
    """Raise if round-robin partitions would miss or duplicate any sample index."""
    covered: list[int] = []
    for r in range(size):
        covered.extend(_partition_sample_indices(n_samples, r, size))
    covered_sorted = sorted(covered)
    expected = list(range(n_samples))
    if covered_sorted != expected:
        raise RuntimeError(
            f"Sample partition bug: size={size}, n_samples={n_samples}, "
            f"covered={covered_sorted[:20]}{'...' if len(covered_sorted) > 20 else ''}"
        )


def _assert_list_partition_covers(indices: list[int], size: int) -> None:
    """Raise if round-robin over *indices* would miss or duplicate any entry."""
    covered: list[int] = []
    for r in range(size):
        covered.extend(_partition_index_list(indices, r, size))
    if sorted(covered) != sorted(indices):
        raise RuntimeError(
            f"Pending-index partition bug: size={size}, n_pending={len(indices)}"
        )


def _mpi_print(rank: int, *args, **kwargs) -> None:
    """Print from every rank with a rank tag when size > 1; kwargs as for print."""
    size = kwargs.pop("_size", 1)
    if size > 1:
        print(f"[rank {rank}]", *args, **kwargs)
    else:
        print(*args, **kwargs)


# Acceptance criteria for pre-screened propagation subsets.
PROPAGATION_SEP_MIN_A = 3.0
PROPAGATION_SEP_MAX_A = 4.0
PROPAGATION_FMAX_MAX = 1e-4  # eV/Å; matches DEFAULT_RELAX_FTOL below

# TBLG relaxation defaults (ASE LBFGS / LAMMPS CG)
DEFAULT_RELAX_ETOL = 0
DEFAULT_RELAX_FTOL = 1e-4
DEFAULT_RELAX_MAXITER = 20_000
DEFAULT_RELAX_MAXEVAL = 20_000
DEFAULT_RELAX_BACKEND = "lammps"
DEFAULT_RELAX_MIN_STYLE = "cg"


def _pattern_has_pod_wildcard(pattern: str) -> bool:
    return (
        "POD_energy" in pattern
        and any(ch in pattern for ch in "*?[]")
    )


def expand_models_for_relaxation(
    model_patterns: list[str],
    ensemble_dir: str,
) -> list[str]:
    """
    Expand ``--models`` patterns for relaxation propagation.

    POD wildcard patterns (e.g. ``POD_energy_POD_index*``) resolve from passing
    rows of the tightened search CSV and their associated fit caches.
    """
    pod_wildcards = [p for p in model_patterns if _pattern_has_pod_wildcard(p)]
    other_patterns = [p for p in model_patterns if p not in pod_wildcards]

    models: list[str] = []
    if other_patterns:
        models.extend(expand_model_patterns(other_patterns, ensemble_dir))
    if pod_wildcards:
        csv_models = pod_energy_ensemble_names_from_csv(ensemble_dir=ensemble_dir)
        models.extend(
            name
            for name in csv_models
            if any(fnmatch.fnmatch(name, pattern) for pattern in pod_wildcards)
        )

    seen: set[str] = set()
    out: list[str] = []
    for name in models:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def select_pod_model_lowest_nll(
    ensemble_dir: str,
    calibration_metrics_dir: str,
    *,
    calibration_technique: str = "mcmc",
    calibration_target: str = "energy",
) -> str:
    """
    Select the passing tightened-CSV POD model with the lowest calibration NLL.
    """
    candidates = pod_energy_ensemble_names_from_csv(ensemble_dir=ensemble_dir)
    if not candidates:
        raise ValueError(
            "No POD_energy ensemble folders found for passing rows in "
            "pod_hyperparam_search.csv with associated fit "
            f"caches under {ensemble_dir!r}."
        )
    best_name: str | None = None
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
            "No passing POD model from pod_hyperparam_search.csv "
            "has a finite "
            f"calibration NLL under {calibration_metrics_dir!r}."
        )
    print(
        f"Auto-selected POD model {best_name!r} "
        f"(lowest NLL = {best_nll:.6g} among "
        f"{len(candidates)} passing CSV models)",
        flush=True,
    )
    return best_name


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


def _remove_atoms_outside_cell(atoms_object) -> None:
    """
    Delete atoms whose Cartesian positions move under ``Atoms.wrap()``.

    Some older ``flatgraphene`` installs use a broken scaled-position loop that
    leaves duplicate images in the supercell (inflating atom counts ~3×).  This
    matches the wrap-based implementation used in the ``blg_uq`` environment.
    """
    import copy

    original_positions = atoms_object.get_positions()
    temp = copy.deepcopy(atoms_object)
    temp.wrap()
    wrapped_positions = temp.get_positions()
    out_of_bounds = [
        i
        for i in range(original_positions.shape[0])
        if not np.allclose(original_positions[i], wrapped_positions[i])
    ]
    if out_of_bounds:
        del atoms_object[out_of_bounds]


def _patch_flatgraphene_cell_cleanup(fg) -> None:
    """Ensure ``fg.twist.remove_atoms_outside_cell`` uses the wrap-based cleanup."""
    fg.twist.remove_atoms_outside_cell = _remove_atoms_outside_cell


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

    _patch_flatgraphene_cell_cleanup(fg)
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
    *,
    failed: bool = False,
) -> str:
    base = (
        f"{_safe_filename_part(model_name)}"
        f"_T{_safe_filename_part(temperature_label)}"
        f"_theta{twist_angle:g}deg"
        f"_sample{sample_index:04d}"
    )
    if failed:
        base = f"{base}_FAIL"
    return os.path.join(output_dir, f"{base}.traj")


def _existing_relaxation_trajectory(traj_path: str, fail_traj_path: str) -> str | None:
    """Return the path to an existing OK or FAIL trajectory, or ``None``."""
    if os.path.isfile(traj_path):
        return traj_path
    if os.path.isfile(fail_traj_path):
        return fail_traj_path
    return None


def relaxation_output_dir(
    output_dir: str,
    model_name: str,
    temperature_label: str,
    twist_angle: float,
) -> str:
    """Directory that holds traj files for one (model, T, twist) job."""
    return os.path.join(
        output_dir,
        _safe_filename_part(model_name),
        f"T{_safe_filename_part(temperature_label)}",
        f"theta{float(twist_angle):g}deg",
    )


def _final_fmax_from_traj(traj_path: str) -> float | None:
    """Return max per-atom force on the last frame, or ``None`` if unavailable."""
    try:
        from ase.io import read  # noqa: PLC0415

        atoms = read(str(traj_path), index=-1)
    except Exception:
        return None
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        forces = atoms.arrays.get("forces")
        if forces is None:
            return None
        forces = np.asarray(forces, dtype=float)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        return None
    return float(np.max(np.linalg.norm(forces, axis=1)))


def trajectory_meets_relax_ftol(
    traj_path: str,
    ftol: float = DEFAULT_RELAX_FTOL,
) -> bool:
    """True if the last frame has forces and ``max_i ||F_i|| ≤ ftol``."""
    fmax = _final_fmax_from_traj(traj_path)
    return fmax is not None and float(fmax) <= float(ftol)


def pending_relaxation_sample_indices(
    output_dir: str,
    model_name: str,
    temperature_label: str,
    twist_angle: float,
    n_samples: int,
    *,
    ftol: float = DEFAULT_RELAX_FTOL,
) -> list[int]:
    """
    Sample indices in ``0 .. n_samples-1`` that still need a relaxation.

    An index is **done** (not pending) only when an OK or FAIL trajectory exists
    **and** its final frame meets ``ftol``.  Trajectories that exist but are
    above ``ftol`` (or lack forces) are treated as pending so relaxation can
    resume from that last frame.
    """
    out_dir = relaxation_output_dir(
        output_dir, model_name, temperature_label, twist_angle,
    )
    pending: list[int] = []
    for sample_index in range(int(n_samples)):
        traj_path = _trajectory_path(
            out_dir, model_name, temperature_label, twist_angle, sample_index,
            failed=False,
        )
        fail_traj_path = _trajectory_path(
            out_dir, model_name, temperature_label, twist_angle, sample_index,
            failed=True,
        )
        existing = _existing_relaxation_trajectory(traj_path, fail_traj_path)
        if existing is None:
            pending.append(int(sample_index))
            continue
        if trajectory_meets_relax_ftol(existing, ftol=ftol):
            continue
        pending.append(int(sample_index))
    return pending


def _max_force(atoms) -> float:
    f = np.asarray(atoms.get_forces(), dtype=float)
    return float(np.max(np.linalg.norm(f, axis=1)))


def _local_energies_via_lammps(calc, atoms) -> np.ndarray | None:
    """Per-atom energies (eV) from a live LAMMPS instance via ``pe/atom``, if possible."""
    lmp = getattr(calc, "_lmp", None)
    if lmp is None:
        return None
    n_atoms = len(atoms)
    compute_id = "peatom_uq_relax"
    try:
        try:
            lmp.command(f"uncompute {compute_id}")
        except Exception:
            pass
        lmp.command(f"compute {compute_id} all pe/atom")
        lmp.command("run 0")
        pe: np.ndarray | None = None
        if hasattr(lmp, "numpy") and hasattr(lmp.numpy, "extract_compute"):
            # LMP_STYLE_ATOM=1, LMP_TYPE_VECTOR=1
            pe = np.asarray(
                lmp.numpy.extract_compute(compute_id, 1, 1), dtype=float,
            ).ravel()
        else:
            try:
                from lammps import LMP_STYLE_ATOM, LMP_TYPE_VECTOR
            except ImportError:
                LMP_STYLE_ATOM, LMP_TYPE_VECTOR = 1, 1
            ptr = lmp.extract_compute(compute_id, LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            pe = np.asarray([float(ptr[i]) for i in range(n_atoms)], dtype=float)
        if pe is None or pe.size != n_atoms or not np.all(np.isfinite(pe)):
            return None
        return pe
    except Exception:
        return None
    finally:
        try:
            lmp.command(f"uncompute {compute_id}")
        except Exception:
            pass


def _evaluate_energy_forces_local(
    atoms,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """
    Evaluate total energy, forces, and local (per-atom) energies on *atoms*.

    Returns ``(energy, forces, local_energies)``; any entry may be ``None`` if
    unavailable.  Local energies prefer ``Atoms.get_potential_energies()``, then
    LAMMPS ``compute pe/atom`` when a live ``_lmp`` handle exists.
    """
    if atoms.calc is None:
        return None, None, None

    energy: float | None = None
    forces: np.ndarray | None = None
    local: np.ndarray | None = None
    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        energy = None
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            forces = None
    except Exception:
        forces = None

    try:
        local = np.asarray(atoms.get_potential_energies(), dtype=float).ravel()
        if local.shape != (len(atoms),) or not np.all(np.isfinite(local)):
            local = None
    except Exception:
        local = None
    if local is None:
        local = _local_energies_via_lammps(atoms.calc, atoms)

    return energy, forces, local


def _attach_singlepoint_results(
    atoms,
    *,
    energy: float | None,
    forces: np.ndarray | None,
    local_energies: np.ndarray | None = None,
) -> None:
    """Attach a SinglePointCalculator so traj I/O keeps energy / forces / local E."""
    from ase.calculators.singlepoint import SinglePointCalculator

    kwargs: dict = {}
    if energy is not None and np.isfinite(energy):
        kwargs["energy"] = float(energy)
        atoms.info["energy"] = float(energy)
        atoms.info["total_energy"] = float(energy)
    if forces is not None:
        kwargs["forces"] = np.asarray(forces, dtype=float)
    if local_energies is not None:
        loc = np.asarray(local_energies, dtype=float).ravel()
        if loc.shape == (len(atoms),) and np.all(np.isfinite(loc)):
            kwargs["energies"] = loc
            atoms.set_array("local_energy", loc)

    if kwargs:
        atoms.calc = SinglePointCalculator(atoms, **kwargs)
    else:
        atoms.calc = None


def _write_success_trajectory(traj_path: str, initial, relaxed) -> None:
    """Write initial + relaxed frames after relaxation returns.

    The relaxed frame includes total energy, forces, and local (per-atom)
    energies when the calculator can provide them.
    """
    from ase.io.trajectory import Trajectory

    energy, forces, local_e = _evaluate_energy_forces_local(relaxed)

    initial = initial.copy()
    initial.calc = None
    initial.info = dict(initial.info)
    initial.info["frame"] = "initial"
    initial.info["relaxation_status"] = "OK"
    _ensure_mol_id_from_z(initial)

    out = relaxed.copy()
    out.info = dict(out.info)
    out.info["frame"] = "relaxed"
    out.info["relaxation_status"] = "OK"
    _ensure_mol_id_from_z(out)
    _attach_singlepoint_results(
        out, energy=energy, forces=forces, local_energies=local_e,
    )

    with Trajectory(traj_path, mode="w") as traj:
        traj.write(initial)
        traj.write(out)


def _write_progress_trajectory(traj_path: str, initial, current) -> None:
    """Overwrite ``traj_path`` with initial + mid-relaxation frames (checkpoint)."""
    from ase.io.trajectory import Trajectory

    energy, forces, local_e = _evaluate_energy_forces_local(current)

    initial_out = initial.copy()
    initial_out.calc = None
    initial_out.info = dict(initial_out.info)
    initial_out.info["frame"] = "initial"
    initial_out.info["relaxation_status"] = "IN_PROGRESS"
    _ensure_mol_id_from_z(initial_out)

    out = current.copy()
    out.info = dict(out.info)
    out.info["frame"] = "checkpoint"
    out.info["relaxation_status"] = "IN_PROGRESS"
    _ensure_mol_id_from_z(out)
    _attach_singlepoint_results(
        out, energy=energy, forces=forces, local_energies=local_e,
    )

    with Trajectory(traj_path, mode="w") as traj:
        traj.write(initial_out)
        traj.write(out)


# Interval (ASE optimizer steps) between overwriting the sample ``.traj`` file.
ASE_RELAX_TRAJ_CHECKPOINT_INTERVAL = 1000


# ---------------------------------------------------------------------------
# Pre-screened propagation ensemble subsets
# ---------------------------------------------------------------------------


def propagation_subset_path(
    propagation_dir: str,
    model_name: str,
    temperature_label: str,
    twist_angle: float,
) -> str:
    """Path for a TBLG-relaxation-validated parameter subset ``.npz``."""
    base = (
        f"{_safe_filename_part(model_name)}"
        f"_T{_safe_filename_part(temperature_label)}"
        f"_theta{float(twist_angle):g}deg"
        f"_relaxation_subset"
    )
    return os.path.join(propagation_dir, f"{base}.npz")


def passes_propagation_criteria(
    atoms,
    *,
    sep_lo: float = PROPAGATION_SEP_MIN_A,
    sep_hi: float = PROPAGATION_SEP_MAX_A,
    fmax_max: float = PROPAGATION_FMAX_MAX,
) -> Tuple[bool, float, float, float]:
    """
    Return ``(ok, max_sep, min_sep, fmax)`` for a relaxed structure.

    Requires ``sep_lo ≤ min_sep ≤ max_sep ≤ sep_hi`` and ``fmax < fmax_max``.
    """
    max_sep, min_sep = layer_separation_metrics(atoms)
    fmax = _max_force(atoms)
    ok = (
        np.isfinite(max_sep)
        and np.isfinite(min_sep)
        and np.isfinite(fmax)
        and sep_lo <= min_sep <= sep_hi
        and sep_lo <= max_sep <= sep_hi
        and fmax < fmax_max
    )
    return bool(ok), float(max_sep), float(min_sep), float(fmax)


def save_propagation_subset(
    path: str,
    *,
    params: np.ndarray,
    source_indices: np.ndarray,
    model_name: str,
    temperature: float,
    twist_angle: float,
    max_sep: np.ndarray,
    min_sep: np.ndarray,
    fmax: np.ndarray,
    seed: int,
    source_ensemble_pkl: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        params=np.asarray(params, dtype=float),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        max_sep=np.asarray(max_sep, dtype=float),
        min_sep=np.asarray(min_sep, dtype=float),
        fmax=np.asarray(fmax, dtype=float),
        model_name=np.asarray(model_name),
        temperature=np.asarray(float(temperature)),
        twist_angle=np.asarray(float(twist_angle)),
        seed=np.asarray(int(seed)),
        source_ensemble_pkl=np.asarray(source_ensemble_pkl),
        sep_lo=np.asarray(PROPAGATION_SEP_MIN_A),
        sep_hi=np.asarray(PROPAGATION_SEP_MAX_A),
        fmax_max=np.asarray(PROPAGATION_FMAX_MAX),
        relax_backend=np.asarray(DEFAULT_RELAX_BACKEND),
        relax_etol=np.asarray(DEFAULT_RELAX_ETOL),
        relax_ftol=np.asarray(DEFAULT_RELAX_FTOL),
        relax_maxiter=np.asarray(DEFAULT_RELAX_MAXITER),
        relax_maxeval=np.asarray(DEFAULT_RELAX_MAXEVAL),
    )


def load_propagation_subset(path: str) -> np.ndarray:
    """Load screened parameter array shaped ``(n_samples, n_params)``."""
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["params"], dtype=float)


def _save_propagation_partial(
    path: str,
    *,
    params: list[np.ndarray],
    source_indices: list[int],
    max_sep: list[float],
    min_sep: list[float],
    fmax: list[float],
    n_tried: int,
    n_fail: int,
    n_reject: int,
) -> None:
    """Checkpoint accepted propagation samples so long screening runs resume."""
    if not params:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        params=np.vstack(params),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        max_sep=np.asarray(max_sep, dtype=float),
        min_sep=np.asarray(min_sep, dtype=float),
        fmax=np.asarray(fmax, dtype=float),
        n_tried=np.asarray(int(n_tried)),
        n_fail=np.asarray(int(n_fail)),
        n_reject=np.asarray(int(n_reject)),
    )


def _load_propagation_partial(path: str):
    """Return saved partial state, or ``None`` when no checkpoint exists."""
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        n_fail = int(np.asarray(z["n_fail"]).item()) if "n_fail" in z.files else 0
        n_reject = (
            int(np.asarray(z["n_reject"]).item()) if "n_reject" in z.files else 0
        )
        return {
            "params": [np.asarray(row, dtype=float).ravel() for row in z["params"]],
            "source_indices": [int(x) for x in np.asarray(z["source_indices"]).ravel()],
            "max_sep": [float(x) for x in np.asarray(z["max_sep"]).ravel()],
            "min_sep": [float(x) for x in np.asarray(z["min_sep"]).ravel()],
            "fmax": [float(x) for x in np.asarray(z["fmax"]).ravel()],
            "n_tried": int(np.asarray(z["n_tried"]).item()),
            "n_fail": n_fail,
            "n_reject": n_reject,
        }


def build_propagation_subset(
    *,
    model_name: str,
    atoms_template,
    ensemble: np.ndarray,
    temperature: float,
    twist_angle: float,
    n_samples: int,
    seed: int,
    source_ensemble_pkl: str,
    out_path: str,
    calc,
    set_params_fn=None,
    relax_backend: str = DEFAULT_RELAX_BACKEND,
    etol: float = DEFAULT_RELAX_ETOL,
    ftol: float = DEFAULT_RELAX_FTOL,
    maxiter: int = DEFAULT_RELAX_MAXITER,
    maxeval: int = DEFAULT_RELAX_MAXEVAL,
    min_style: str = DEFAULT_RELAX_MIN_STYLE,
) -> str:
    """
    Screen shuffled ensemble draws until ``n_samples`` pass the separation /
    force criteria; write ``out_path`` and return it.

    Uses the same ``relax_tblg_sample`` settings as production relaxation runs.
    """
    ensemble_shuffled = _shuffle_ensemble(np.asarray(ensemble, dtype=float), seed)
    partial_path = f"{out_path}.partial.npz"
    partial = _load_propagation_partial(partial_path)
    if partial is None:
        accepted_params: list[np.ndarray] = []
        accepted_idx: list[int] = []
        accepted_max_sep: list[float] = []
        accepted_min_sep: list[float] = []
        accepted_fmax: list[float] = []
        n_tried = 0
        n_fail = 0
        n_reject = 0
    else:
        accepted_params = partial["params"]
        accepted_idx = partial["source_indices"]
        accepted_max_sep = partial["max_sep"]
        accepted_min_sep = partial["min_sep"]
        accepted_fmax = partial["fmax"]
        n_tried = partial["n_tried"]
        n_fail = partial["n_fail"]
        n_reject = partial["n_reject"]
        print(
            f"  Resuming from {partial_path}: accepted={len(accepted_params)}, "
            f"tried={n_tried}, fail={n_fail}, reject={n_reject}",
            flush=True,
        )
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)) or ".", "_tmp_screen")
    os.makedirs(tmp_dir, exist_ok=True)

    print(
        f"  Screening for {n_samples} propagation samples "
        f"(sep ∈ [{PROPAGATION_SEP_MIN_A:g}, {PROPAGATION_SEP_MAX_A:g}] Å, "
        f"fmax < {PROPAGATION_FMAX_MAX:g} eV/Å)\n"
        f"  Relax settings: backend={relax_backend} min_style={min_style} "
        f"etol={etol:g} ftol={ftol:g} maxiter={maxiter} maxeval={maxeval}",
        flush=True,
    )

    for src_i in range(n_tried, ensemble_shuffled.shape[0]):
        if len(accepted_params) >= n_samples:
            break
        theta = ensemble_shuffled[src_i]
        n_tried += 1
        traj_path = os.path.join(tmp_dir, f"screen_{n_tried:06d}.traj")
        fail_path = os.path.join(tmp_dir, f"screen_{n_tried:06d}_FAIL.traj")
        print(
            f"  try {n_tried} (accepted {len(accepted_params)}/{n_samples}) …",
            flush=True,
        )
        try:
            relaxed = relax_tblg_sample(
                atoms_template,
                calc,
                theta,
                traj_path,
                relax_backend=relax_backend,
                etol=etol,
                ftol=ftol,
                maxiter=maxiter,
                maxeval=maxeval,
                set_params_fn=set_params_fn,
                fail_traj_path=fail_path,
                min_style=min_style,
            )
        except Exception as exc:
            n_fail += 1
            print(f"    reject FAIL ({type(exc).__name__}: {exc})", flush=True)
            for p in (traj_path, fail_path):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            continue

        ok, max_sep, min_sep, fmax = passes_propagation_criteria(relaxed)
        for p in (traj_path, fail_path):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if not ok:
            n_reject += 1
            print(
                f"    reject criteria min_sep={min_sep:.4f} max_sep={max_sep:.4f} "
                f"fmax={fmax:.3e}",
                flush=True,
            )
            continue

        accepted_params.append(np.asarray(theta, dtype=float).ravel().copy())
        accepted_idx.append(int(src_i))
        accepted_max_sep.append(max_sep)
        accepted_min_sep.append(min_sep)
        accepted_fmax.append(fmax)
        _save_propagation_partial(
            partial_path,
            params=accepted_params,
            source_indices=accepted_idx,
            max_sep=accepted_max_sep,
            min_sep=accepted_min_sep,
            fmax=accepted_fmax,
            n_tried=n_tried,
            n_fail=n_fail,
            n_reject=n_reject,
        )
        print(
            f"    ACCEPT #{len(accepted_params)} "
            f"min_sep={min_sep:.4f} max_sep={max_sep:.4f} fmax={fmax:.3e}",
            flush=True,
        )

    if len(accepted_params) < n_samples:
        raise RuntimeError(
            f"Only found {len(accepted_params)}/{n_samples} accepted samples "
            f"after trying {n_tried} draws "
            f"(fail={n_fail}, criteria_reject={n_reject})."
        )

    save_propagation_subset(
        out_path,
        params=np.vstack(accepted_params),
        source_indices=np.asarray(accepted_idx, dtype=np.int64),
        model_name=model_name,
        temperature=temperature,
        twist_angle=twist_angle,
        max_sep=np.asarray(accepted_max_sep, dtype=float),
        min_sep=np.asarray(accepted_min_sep, dtype=float),
        fmax=np.asarray(accepted_fmax, dtype=float),
        seed=seed,
        source_ensemble_pkl=source_ensemble_pkl,
    )
    print(
        f"  Wrote {out_path}  ({len(accepted_params)} samples; "
        f"tried={n_tried}, fail={n_fail}, reject={n_reject})",
        flush=True,
    )
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    if os.path.isfile(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass
    return out_path


def _write_fail_trajectory(
    traj_path: str,
    initial,
    *,
    failed_atoms=None,
    error_message: str = "",
) -> None:
    """Write a failed-relaxation trajectory (filename should end with ``_FAIL.traj``)."""
    from ase.io.trajectory import Trajectory

    initial = initial.copy()
    initial.calc = None
    initial.info = dict(initial.info)
    initial.info["frame"] = "initial"
    initial.info["relaxation_status"] = "FAIL"
    if error_message:
        initial.info["relaxation_error"] = str(error_message)[:2000]
    _ensure_mol_id_from_z(initial)

    with Trajectory(traj_path, mode="w") as traj:
        traj.write(initial)
        if failed_atoms is not None:
            failed = failed_atoms.copy()
            failed.calc = None
            failed.info = dict(failed.info)
            failed.info["frame"] = "failed"
            failed.info["relaxation_status"] = "FAIL"
            if error_message:
                failed.info["relaxation_error"] = str(error_message)[:2000]
            _ensure_mol_id_from_z(failed)
            traj.write(failed)


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
    fail_traj_path: str | None = None,
    write_trajectory: bool = True,
    log_path: str | None = None,
    dump_path: str | None = None,
    full_trajectory: bool = False,
    dump_every: int = 100,
    min_style: str = DEFAULT_RELAX_MIN_STYLE,
):
    """
    Relax one ensemble draw and write ``traj_path``.

    On LAMMPS / optimizer failure, write ``fail_traj_path`` (or ``traj_path`` with
    a ``_FAIL`` suffix if unset) containing the initial frame (and any available
    partial structure), labeled ``relaxation_status=FAIL``, then re-raise.
    Non-converged force norms that do not raise are still saved as successes.

    Parameters
    ----------
    write_trajectory
        If False, skip writing success/fail ``.traj`` files (use on non-root
        MPI ranks during a multi-rank LAMMPS minimize).
    log_path
        Optional optimizer / LAMMPS log file path.
    dump_path
        Optional LAMMPS ``dump`` file path (full minimize trajectory).  Ignored
        for the ASE backend.
    full_trajectory
        If True with ASE: pass ``trajectory=traj_path`` and ``logfile=log_path``
        to LBFGS (every step).  If True with LAMMPS: requires ``dump_path`` /
        ``log_path`` (or derives them from ``traj_path``).  Skips the
        two-frame overwrite of ``traj_path`` so the full history is kept.
    dump_every
        LAMMPS dump interval (force evaluations) when ``dump_path`` is used.
    min_style
        LAMMPS ``min_style`` (``fire`` or ``cg``); ignored for ASE.
    """
    atoms = atoms_template.copy()
    apply_uq_parameters(calc, theta, set_params_fn)
    atoms.calc = calc
    _ensure_mol_id_from_z(atoms)

    initial = atoms.copy()
    initial.calc = calc

    fail_path = fail_traj_path
    if fail_path is None:
        root, ext = os.path.splitext(traj_path)
        fail_path = f"{root}_FAIL{ext}"

    backend = str(relax_backend).strip().lower()
    root_no_ext, _ext = os.path.splitext(traj_path)
    if full_trajectory:
        if log_path is None:
            log_path = f"{root_no_ext}.log"
        if backend == "lammps" and dump_path is None:
            dump_path = f"{root_no_ext}.dump"

    try:
        if backend == "lammps":
            # dump/log must be set on every MPI rank (same paths) when using
            # COMM_WORLD; only the ASE companion / FAIL traj are root-only.
            relaxed = calc.relax_structure(
                atoms,
                relax_backend="lammps",
                etol=etol,
                ftol=ftol,
                maxiter=maxiter,
                maxeval=maxeval,
                dump_path=dump_path,
                log_path=log_path,
                dump_every=dump_every,
                min_style=min_style,
            )
            relaxed.calc = calc
            _ensure_mol_id_from_z(relaxed)
            # Keep LAMMPS dump as the full trajectory; write ASE 2-frame summary
            # only when not requesting a full dump history.
            if write_trajectory and not full_trajectory:
                _write_success_trajectory(traj_path, initial, relaxed)
            elif write_trajectory and full_trajectory:
                # Compact ASE companion for post-processing (initial + final).
                summary = f"{root_no_ext}_endpoints.traj"
                _write_success_trajectory(summary, initial, relaxed)
            return relaxed

        if backend == "ase":
            from ase.optimize import LBFGS
            from blg_model_builder.potentials import _ASE_FIRE_MAXSTEP

            dyn_kw: dict = {
                "logfile": log_path if (write_trajectory and log_path) else None,
                "maxstep": _ASE_FIRE_MAXSTEP,
            }
            if write_trajectory and full_trajectory:
                dyn_kw["trajectory"] = traj_path
            # Keep a small maxstep for POD TBLG stability (same cap as prior FIRE).
            dyn = LBFGS(atoms, **dyn_kw)
            if write_trajectory and not full_trajectory:
                # Overwrite the sample traj every N steps so long runs leave a
                # resumable checkpoint; final write below replaces this file.
                def _checkpoint_traj() -> None:
                    _write_progress_trajectory(traj_path, initial, atoms)

                dyn.attach(
                    _checkpoint_traj,
                    interval=int(ASE_RELAX_TRAJ_CHECKPOINT_INTERVAL),
                )
            dyn.run(fmax=ftol, steps=maxiter)
            atoms.calc = calc
            _ensure_mol_id_from_z(atoms)
            if write_trajectory and not full_trajectory:
                _write_success_trajectory(traj_path, initial, atoms)
            return atoms

        raise ValueError(f"Unknown relax_backend {relax_backend!r}; use 'lammps' or 'ase'.")
    except Exception as exc:
        failed_atoms = None
        try:
            # Prefer the atoms object mid-relaxation if positions are available.
            failed_atoms = atoms.copy()
            failed_atoms.calc = None
        except Exception:
            failed_atoms = None
        if write_trajectory:
            _write_fail_trajectory(
                fail_path,
                initial,
                failed_atoms=failed_atoms,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        # Attach path for the caller so it can log the FAIL traj.
        setattr(exc, "fail_traj_path", fail_path)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Relax ensemble samples; under ``mpirun -np N`` samples are split by rank."""
    comm, rank, size = _mpi_state()
    is_root = rank == 0

    p = argparse.ArgumentParser(
        description="TBLG relaxation UQ propagation from MCMC LAMMPS ensembles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p, required=False)
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
        help="Target number of unique ensemble samples to relax "
        "(successful and FAIL both count). Under mpirun/srun, pending sample "
        "indices (those without an existing OK/FAIL trajectory when "
        "--skip-existing) are partitioned round-robin across tasks.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_ENSEMBLE_SHUFFLE_SEED,
        help="RNG seed for ensemble shuffle (must match run_uq_propagation_elasticity.py).",
    )
    p.add_argument(
        "--twist-angle",
        "--theta",
        type=float,
        default=DEFAULT_TWIST_ANGLE,
        dest="twist_angle",
        help="Twist angle in degrees (flatgraphene commensurate cell). "
        "``--theta`` is an alias.",
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
        default=DEFAULT_RELAX_BACKEND,
        help="Geometry optimizer backend (default: lammps). "
        "lammps: LAMMPS minimize (see --relax-min-style); "
        "ase: ASE LBFGS with small maxstep.",
    )
    p.add_argument(
        "--relax-min-style",
        choices=("cg", "fire"),
        default=DEFAULT_RELAX_MIN_STYLE,
        help=(
            "LAMMPS min_style when --relax-backend lammps "
            f"(default: {DEFAULT_RELAX_MIN_STYLE}). Ignored for ASE."
        ),
    )
    p.add_argument("--relax-etol", type=float, default=DEFAULT_RELAX_ETOL)
    p.add_argument("--relax-ftol", type=float, default=DEFAULT_RELAX_FTOL)
    p.add_argument("--relax-maxiter", type=int, default=DEFAULT_RELAX_MAXITER)
    p.add_argument("--relax-maxeval", type=int, default=DEFAULT_RELAX_MAXEVAL)
    p.add_argument(
        "--propagation-dir",
        default=DEFAULT_PROPAGATION_DIR,
        help="Directory for pre-screened relaxation parameter subsets "
        "(default: ensembles/propagation).",
    )
    p.add_argument(
        "--build-propagation-subset",
        action="store_true",
        help="Screen the shuffled ensemble with this script's relax settings "
        "until --n-samples pass sep∈[3,4] Å and fmax<1e-4 eV/Å; write the "
        "subset under --propagation-dir and exit (rank 0 only under MPI).",
    )
    p.add_argument(
        "--ignore-propagation-subset",
        action="store_true",
        help="Do not load a pre-screened subset even if one exists.",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before assigning work, drop sample indices whose OK/FAIL "
        "trajectory already exists under --output-dir AND meets --relax-ftol "
        "(default: true).  Existing trajectories above ftol (or without "
        "forces) stay pending and are resumed from their last frame. "
        "Use --no-skip-existing to assign/rerun all 0..n_samples-1 from the "
        "pristine TBLG template.",
    )
    add_hyperparam_args(p)
    args, _unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(
        args, _unknown, extra_reserved=("theta", "twist_angle"),
    )
    if cli_hyperparams and is_root:
        print(f"  CLI hyperparameters: {cli_hyperparams}", flush=True)

    os.chdir(HERE)
    if args.models:
        models = expand_models_for_relaxation(args.models, args.ensemble_dir)
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
    if is_root:
        print(f"Models: {models}", flush=True)
        if args.skip_existing:
            print(
                "Skip-existing: enabled (skip only OK/FAIL trajs that meet "
                f"relax_ftol={args.relax_ftol:g}; otherwise resume from last frame)",
                flush=True,
            )
        if size > 1:
            backend = "mpi4py" if comm is not None else "SLURM_PROCID"
            print(
                f"Sample-parallel: {size} tasks via {backend} "
                f"(each unique ensemble index assigned to exactly one task)",
                flush=True,
            )

    if is_root:
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
    if is_root:
        print(f"  n_atoms={len(tblg_template)}", flush=True)

    for model_name in models:
        if is_root:
            print(f"\n--- Model: {model_name} ---", flush=True)
        if not is_uq_energy_model(model_name):
            if is_root:
                print(
                    f"  Warning: unsupported model (need UQ energy model); "
                    f"skipping {model_name!r}.",
                    file=sys.stderr,
                )
            continue

        relax_backend = args.relax_backend
        if is_uq_python_model(model_name) and relax_backend == "lammps":
            relax_backend = "ase"
            if is_root:
                print("  Using relax_backend=ase for Python Allegro model.", flush=True)

        pkl_path, t_used = resolve_ensemble_pickle(
            model_name,
            args.ensemble_dir,
            args.temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_target=args.calibration_target,
        )
        t_label = f"{t_used:g}"
        if is_root:
            print(f"  Ensemble pickle: {pkl_path}  (T={t_label})", flush=True)

        ens_dict = load_ensemble_pickle(pkl_path)
        ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)

        # Each MPI rank owns its own LAMMPS world (COMM_SELF in lammps_interface).
        calc_obj, set_params_fn, _load_name = build_uq_calculator(
            model_name, extra_kw=cli_hyperparams or None,
        )
        if is_root:
            print(f"  Calculator: {_load_name}", flush=True)
        calc_obj.prepare_batch([tblg_template])

        subset_path = propagation_subset_path(
            args.propagation_dir, model_name, t_label, args.twist_angle,
        )

        if args.build_propagation_subset:
            if is_root:
                build_propagation_subset(
                    model_name=model_name,
                    atoms_template=tblg_template,
                    ensemble=ensemble,
                    temperature=float(t_used),
                    twist_angle=float(args.twist_angle),
                    n_samples=int(args.n_samples),
                    seed=int(args.seed),
                    source_ensemble_pkl=pkl_path,
                    out_path=subset_path,
                    calc=calc_obj,
                    set_params_fn=set_params_fn,
                    relax_backend=relax_backend,
                    etol=args.relax_etol,
                    ftol=args.relax_ftol,
                    maxiter=args.relax_maxiter,
                    maxeval=args.relax_maxeval,
                    min_style=args.relax_min_style,
                )
            calc_obj.close()
            if comm is not None:
                comm.Barrier()
            continue

        if (
            not args.ignore_propagation_subset
            and os.path.isfile(subset_path)
        ):
            ensemble_pool = load_propagation_subset(subset_path)
            if is_root:
                print(
                    f"  Using pre-screened propagation subset: {subset_path}  "
                    f"({ensemble_pool.shape[0]} samples)",
                    flush=True,
                )
        else:
            ensemble_pool = _shuffle_ensemble(ensemble, args.seed)
            if is_root:
                print(
                    f"  Shuffled ensemble (seed={args.seed}): "
                    f"{ensemble_pool.shape[0]} members",
                    flush=True,
                )

        n_target = min(int(args.n_samples), int(ensemble_pool.shape[0]))
        # First n_target rows are the work list; each global index maps to one
        # unique ensemble draw. With --skip-existing, only indices that are still
        # pending (missing traj, or traj above relax_ftol) are eligible; those
        # pending indices are then partitioned round-robin across ranks.
        work = np.asarray(ensemble_pool[:n_target], dtype=float)
        if work.shape[0] != n_target:
            raise RuntimeError(
                f"Ensemble work list length {work.shape[0]} != n_target {n_target}"
            )

        out_dir = relaxation_output_dir(
            args.output_dir, model_name, t_label, args.twist_angle,
        )
        if args.skip_existing:
            pending_indices = pending_relaxation_sample_indices(
                args.output_dir,
                model_name,
                t_label,
                args.twist_angle,
                n_target,
                ftol=float(args.relax_ftol),
            )
            n_done = n_target - len(pending_indices)
            if is_root:
                print(
                    f"  converged trajectories (fmax≤{args.relax_ftol:g}): "
                    f"{n_done}/{n_target}  pending: {len(pending_indices)}",
                    flush=True,
                )
        else:
            pending_indices = list(range(n_target))

        if not pending_indices:
            if is_root:
                print(
                    f"  Nothing to do for {model_name} at θ={args.twist_angle:g}° "
                    f"(all {n_target} trajectories meet relax_ftol="
                    f"{args.relax_ftol:g}).",
                    flush=True,
                )
            calc_obj.close()
            if comm is not None:
                comm.Barrier()
            continue

        _assert_list_partition_covers(pending_indices, size)
        my_indices = _partition_index_list(pending_indices, rank, size)
        if is_root:
            print(
                f"  target {n_target} unique ensemble samples "
                f"({len(pending_indices)} pending assigned across "
                f"{size} parallel task(s))",
                flush=True,
            )
        _mpi_print(
            rank,
            f"assigned {len(my_indices)}/{len(pending_indices)} pending sample "
            f"indices (first={my_indices[0] if my_indices else 'none'}, "
            f"last={my_indices[-1] if my_indices else 'none'})",
            flush=True,
            _size=size,
        )
        # Cross-check: no two ranks claim the same pending index (MPI only).
        if comm is not None and size > 1:
            from mpi4py import MPI  # noqa: PLC0415

            counts = np.zeros(n_target, dtype=np.int32)
            for sample_index in my_indices:
                counts[sample_index] = 1
            totals = np.zeros(n_target, dtype=np.int32)
            comm.Allreduce(counts, totals, op=MPI.SUM)
            expected = np.zeros(n_target, dtype=np.int32)
            for sample_index in pending_indices:
                expected[sample_index] = 1
            if is_root and not np.array_equal(totals, expected):
                bad = np.where(totals != expected)[0]
                raise RuntimeError(
                    f"Sample-index collision/gap across MPI ranks at indices "
                    f"{bad[:20].tolist()} (totals min={totals.min()}, max={totals.max()})"
                )

        if is_root:
            os.makedirs(out_dir, exist_ok=True)
        if comm is not None:
            comm.Barrier()
        else:
            os.makedirs(out_dir, exist_ok=True)

        n_ok = 0
        n_fail = 0
        n_skip = 0
        for sample_index in my_indices:
            # Unique ensemble row for this global sample index.
            theta = work[sample_index]
            traj_path = _trajectory_path(
                out_dir, model_name, t_label, args.twist_angle, sample_index,
                failed=False,
            )
            fail_traj_path = _trajectory_path(
                out_dir, model_name, t_label, args.twist_angle, sample_index,
                failed=True,
            )
            atoms_start = tblg_template
            resume_from: str | None = None
            # Defense in depth: another task may have finished this index
            # between assignment and now (or NFS lag on a prior run).
            if args.skip_existing:
                existing = _existing_relaxation_trajectory(traj_path, fail_traj_path)
                if existing is not None:
                    if trajectory_meets_relax_ftol(existing, ftol=float(args.relax_ftol)):
                        n_skip += 1
                        _mpi_print(
                            rank,
                            f"sample {sample_index + 1}/{n_target}: skip converged "
                            f"{os.path.basename(existing)}",
                            flush=True,
                            _size=size,
                        )
                        continue
                    # Unconverged / no forces: resume from last frame.
                    try:
                        from ase.io import read as _ase_read  # noqa: PLC0415

                        atoms_start = _ase_read(str(existing), index=-1).copy()
                        atoms_start.calc = None
                        resume_from = existing
                    except Exception as exc:
                        _mpi_print(
                            rank,
                            f"sample {sample_index + 1}/{n_target}: could not read "
                            f"{os.path.basename(existing)} ({exc}); "
                            f"restarting from pristine TBLG",
                            flush=True,
                            _size=size,
                        )
                        atoms_start = tblg_template
            _mpi_print(
                rank,
                f"sample {sample_index + 1}/{n_target}"
                + (
                    f" (resume {os.path.basename(resume_from)})"
                    if resume_from
                    else ""
                )
                + " …",
                flush=True,
                _size=size,
            )
            try:
                relaxed = relax_tblg_sample(
                    atoms_start,
                    calc_obj,
                    theta,
                    traj_path,
                    relax_backend=relax_backend,
                    etol=args.relax_etol,
                    ftol=args.relax_ftol,
                    maxiter=args.relax_maxiter,
                    maxeval=args.relax_maxeval,
                    set_params_fn=set_params_fn,
                    fail_traj_path=fail_traj_path,
                    min_style=args.relax_min_style,
                )
            except Exception as exc:
                saved_fail = getattr(exc, "fail_traj_path", fail_traj_path)
                n_fail += 1
                print(
                    f"    FAIL ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
                _mpi_print(rank, f"saved {saved_fail}", flush=True, _size=size)
                continue

            n_ok += 1
            _mpi_print(rank, f"saved {traj_path}", flush=True, _size=size)
            max_sep, min_sep = layer_separation_metrics(relaxed)
            fmax = _max_force(relaxed)
            _mpi_print(
                rank,
                f"max layer separation = {max_sep:.6f} Å  (2·max|z−⟨z⟩|)",
                flush=True,
                _size=size,
            )
            _mpi_print(
                rank,
                f"min layer separation = {min_sep:.6f} Å  (2·min|z−⟨z⟩|)",
                flush=True,
                _size=size,
            )
            _mpi_print(
                rank,
                f"max |F| = {fmax:.4e} eV/Å",
                flush=True,
                _size=size,
            )

        calc_obj.close()

        if comm is not None:
            from mpi4py import MPI  # noqa: PLC0415

            n_ok_tot = int(comm.allreduce(n_ok, op=MPI.SUM))
            n_fail_tot = int(comm.allreduce(n_fail, op=MPI.SUM))
            n_skip_tot = int(comm.allreduce(n_skip, op=MPI.SUM))
            n_saved_tot = n_ok_tot + n_fail_tot
            if is_root:
                skip_part = f", {n_skip_tot} skipped" if n_skip_tot else ""
                print(
                    f"  Finished {model_name}: {n_saved_tot}/{n_target} saved "
                    f"({n_ok_tot} OK, {n_fail_tot} FAIL{skip_part}) "
                    f"(pool size {ensemble_pool.shape[0]}, ranks={size})",
                    flush=True,
                )
        else:
            # Slurm multi-task without MPI: no global reduce; each task reports
            # its own unique slice (union over tasks = all n_target samples).
            n_local = n_ok + n_fail
            skip_part = f", {n_skip} skipped" if n_skip else ""
            _mpi_print(
                rank,
                f"Finished {model_name}: local {n_local}/{len(my_indices)} "
                f"assigned samples saved ({n_ok} OK, {n_fail} FAIL{skip_part}); "
                f"global target {n_target} unique samples across {size} tasks",
                flush=True,
                _size=size,
            )

    if comm is not None:
        comm.Barrier()
    if is_root:
        print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
