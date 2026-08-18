#!/usr/bin/env python3
"""
Use ``POD_index`` from a hyperparameter-search CSV, load or refit coefficients,
then relax TBLG at selected twist angles and plot AA/AB layer separations.

Best-fit quick_pod jobs use the LAMMPS backend under ``mpirun -np N`` so each
minimize domain-decomposes force/energy evaluation across MPI ranks.  Keep the
same ``--relax-ftol`` as ASE ``fmax`` (default ``1e-3``): LAMMPS is configured
with ``min_modify norm max`` so the stopping criterion matches ASE.

Examples
--------
::

    # Serial (single process LAMMPS or ASE)
    python visualizations/plot_pod_best_aa_ab_sep_vs_twist.py \\
        --pod-index 27 --rcut 6 --train-frac 1.0 --twist-angles 1.2 \\
        --relax-backend lammps --relax-ftol 1e-3

    # Parallel force/energy evals during minimize
    mpirun -np 16 python visualizations/plot_pod_best_aa_ab_sep_vs_twist.py \\
        --pod-index 8 --rcut 7 --train-frac 1.0 --twist-angles 0.99 \\
        --relax-backend lammps --relax-ftol 1e-3

Output
------
``figures/<tag>_aa_ab_layer_sep_vs_twist_angle.png``
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
if str(UQ_DIR) not in sys.path:
    sys.path.insert(0, str(UQ_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from blg_model_builder.DataLoader import load_data_for_model, train_test_split  # noqa: E402
from blg_model_builder.model_fit import fit_pod  # noqa: E402
from blg_model_builder.pod_model_selection import (  # noqa: E402
    load_pod_hyperparam_search_fit,
    load_pod_search_results,
    pod_energy_best_fit_cache_path,
    pod_hyperparams_from_row,
)
from blg_model_builder.potentials import (  # noqa: E402
    PODLammpsCalculator,
    ncoeff_from_params,
    pod_hyperparams_to_str,
)
import ase.io  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from plot_tblg_structure_v_twist_angle import (  # noqa: E402
    identify_stacking_atoms,
    layer_sep_for_indices,
)
from run_uq_propagation_relaxation import (  # noqa: E402
    _attach_singlepoint_results,
    _evaluate_energy_forces_local,
    _mpi_state,
    _write_success_trajectory,
    build_tblg_atoms,
    relax_tblg_sample,
)

DEFAULT_TWIST_ANGLES = (1.2, 0.99)
DEFAULT_RCUT = 5.0
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_POD_INDEX = 27
DEFAULT_DATA_FILE = "strained_bilayer_graphene_rVV10.xyz"
FIGURES_DIR = UQ_DIR / "figures"
BEST_FIT_DIR = UQ_DIR / "best_fit_params"
DATA_DIR = UQ_DIR.parent / "data"
# When run from uncertainty_quantification/, data/ is UQ_DIR.parent/data;
# also accept UQ_DIR/../data via package root.
if not DATA_DIR.is_dir():
    DATA_DIR = UQ_DIR / "data"


def _lammps_exec_works(path: str) -> bool:
    """True if ``path`` exists and can resolve shared libs enough to start."""
    import shutil
    import subprocess

    if not path or not os.path.isfile(path):
        return False
    if not os.access(path, os.X_OK) and shutil.which(path) is None:
        # Still try if it is a path we can execute via shell
        pass
    try:
        proc = subprocess.run(
            [path, "-h"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    # Broken liblammps typically exits non-zero with "error while loading shared libraries"
    err = (proc.stderr or "") + (proc.stdout or "")
    if "error while loading shared libraries" in err:
        return False
    # ``lmp -h`` may return 0 or non-zero depending on build; accept if no loader error
    return "liblammps" not in err.lower() or proc.returncode == 0 or "LAMMPS" in err


def _resolve_lammps_exec() -> str:
    """Pick a runnable LAMMPS binary.

    Order: ``$LAMMPS_EXECUTABLE`` (if it works), known local build, then
    ``lmp`` on PATH (if it works).  Raises if none can load ``liblammps``.
    """
    import shutil

    candidates: list[str] = []
    env = os.environ.get("LAMMPS_EXECUTABLE", "").strip()
    if env:
        candidates.append(env)

    wsl_default = "/mnt/c/Users/Daniel/Documents/research/lammps/build/lmp"
    candidates.append(wsl_default)

    which = shutil.which("lmp")
    if which:
        candidates.append(which)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    tried: list[str] = []
    for path in unique:
        tried.append(path)
        if _lammps_exec_works(path):
            return path

    raise FileNotFoundError(
        "No working LAMMPS executable found. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\nSet LAMMPS_EXECUTABLE to a build that can load liblammps.so.0 "
        "(on this machine: /mnt/c/Users/Daniel/Documents/research/lammps/build/lmp)."
    )


def select_pod_row_from_csv(
    pod_index: int = DEFAULT_POD_INDEX,
    *,
    csv_path: Path | str | None = None,
) -> pd.Series:
    """Return the hyperparameter-search row for ``POD_index``."""
    resolved: Path | None = None
    if csv_path is not None:
        raw = Path(str(csv_path)).expanduser()
        candidates = [
            raw,
            Path.cwd() / raw,
            UQ_DIR / raw,
            UQ_DIR / "pod_hyperparam_search" / raw.name,
            UQ_DIR / "pod_hyperparam_search" / raw,
        ]
        for path in candidates:
            if path.is_file():
                resolved = path.resolve()
                break
        if resolved is None:
            raise FileNotFoundError(
                f"POD search CSV not found: {csv_path!r}. Tried:\n  - "
                + "\n  - ".join(str(p) for p in candidates)
            )
    df = load_pod_search_results(resolved)
    idx = int(pod_index)
    if idx < 0 or idx >= len(df):
        raise IndexError(
            f"POD_index={idx} out of range for {len(df)} rows in search CSV"
            + (f" ({resolved})" if resolved is not None else "")
        )
    row = df.iloc[idx].copy()
    row["pod_index"] = idx
    if resolved is not None:
        row["search_csv"] = str(resolved)
    return row


def resolve_data_file(data_file: str | Path) -> Path:
    """Resolve a training extxyz path (absolute, cwd-relative, or under ``data/``)."""
    raw = Path(str(data_file)).expanduser()
    candidates = [
        raw,
        Path.cwd() / raw,
        UQ_DIR / raw,
        DATA_DIR / raw.name,
        DATA_DIR / raw,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Training data file not found: {data_file!r}. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
    )


def data_file_tag(data_path: Path) -> str:
    """Short tag for cache / figure names from the data filename stem."""
    return data_path.stem


def run_tag(row: pd.Series, rcut: float, train_frac: float, data_tag: str) -> str:
    return (
        f"POD_energy_POD_index_{int(row['pod_index'])}_{row['hash']}"
        f"_rcut{rcut:g}_trainfrac{train_frac:g}_{data_tag}"
    )


def fit_cache_path(
    row: pd.Series,
    rcut: float,
    train_frac: float,
    ncoeffs: int,
    data_tag: str,
) -> Path:
    reg = float(row.get("regularization", 1e-12))
    return (
        BEST_FIT_DIR
        / (
            f"POD_energy_{ncoeffs}_reg{reg:.0e}_{row['hash']}"
            f"_rcut{rcut:g}_trainfrac{train_frac:g}_{data_tag}_best_fit_params.npz"
        )
    )


def _write_fit_cache(
    cache: Path,
    *,
    params: np.ndarray,
    row: pd.Series,
    pod_hp: dict,
    pod_hash: str,
    rcut: float,
    train_frac: float,
    data_path: Path,
    n_train_used: int | None = None,
    n_train_full: int | None = None,
) -> None:
    BEST_FIT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "rcut": np.asarray(float(rcut)),
        "train_frac": np.asarray(float(train_frac)),
        "pod_hash": np.asarray(str(pod_hash)),
        "pod_index": np.asarray(int(row["pod_index"])),
        "data_file": np.asarray(str(data_path)),
        **{k: v for k, v in pod_hp.items() if k != "species"},
    }
    if n_train_used is not None:
        payload["n_train_used"] = np.asarray(int(n_train_used))
    if n_train_full is not None:
        payload["n_train_full"] = np.asarray(int(n_train_full))
    np.savez(cache, **payload)


def _load_params_from_npz(path: Path) -> np.ndarray | None:
    """Load POD coefficients from an ``.npz`` cache (``pod_params`` or ``params``)."""
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    if "pod_params" in data.files:
        return np.asarray(data["pod_params"], dtype=float).ravel()
    if "params" in data.files:
        return np.asarray(data["params"], dtype=float).ravel()
    return None


def _uq_search_cache_candidates(pod_hash: str) -> list[Path]:
    """Search-cache paths under this repo's ``uncertainty_quantification/`` tree."""
    h = str(pod_hash).strip()
    cache_root = UQ_DIR / "pod_hyperparam_search" / "pod_hyperparam_search_cache"
    paths = [
        cache_root / "POD_energy_MBD" / f"{h}.npz",
        cache_root / "POD_energy" / f"{h}.npz",
        cache_root / f"{h}.npz",
    ]
    if cache_root.is_dir():
        for sub in sorted(cache_root.glob(f"*/{h}.npz")):
            if sub not in paths:
                paths.append(sub)
    return paths


def _load_existing_pod_params(
    row: pd.Series,
    pod_hp: dict,
    pod_hash: str,
) -> tuple[np.ndarray | None, str | None]:
    """Load coefficients from hyperparam-search or standard best-fit caches."""
    # Prefer this repo's UQ tree (independent of CWD / installed-package layout).
    tried: list[str] = []
    for path in _uq_search_cache_candidates(pod_hash):
        tried.append(str(path))
        params = _load_params_from_npz(path)
        if params is not None:
            return params, str(path)

    search_params = load_pod_hyperparam_search_fit(pod_hash)
    if search_params is not None:
        return np.asarray(search_params, dtype=float).ravel(), "pod_hyperparam_search_cache"

    reg = float(row.get("regularization", 1e-12))
    include_intra = bool(row.get("include_intralayer", False))
    best_fit = pod_energy_best_fit_cache_path(
        pod_hp,
        regularization=reg,
        include_intralayer=include_intra,
        pod_hash=pod_hash,
        best_fit_dir=UQ_DIR,
    )
    tried.append(str(best_fit))
    if best_fit.is_file():
        params = _load_params_from_npz(best_fit)
        if params is not None:
            return params, str(best_fit)

    print(
        "  [cache] no existing POD coefficients found for "
        f"hash={pod_hash!r}. Tried:\n    - " + "\n    - ".join(tried),
        flush=True,
    )
    return None, None


def _assign_mol_id(atoms) -> None:
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    mean_z = float(np.mean(z))
    top_ind = np.where(z > mean_z)[0]
    mol_id = np.ones(len(atoms), dtype=np.int8)
    mol_id[top_ind] = 2
    atoms.set_array("mol-id", mol_id)


def _energy_forces_from_atoms(atoms) -> tuple[float, np.ndarray]:
    """Read energy/forces from calc, else extxyz ``info`` / ``arrays``."""
    if atoms.calc is not None:
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            return energy, forces
        except Exception:
            pass
    if "energy" not in atoms.info:
        raise RuntimeError(
            "Atoms has no calculator and no info['energy'] "
            "(extxyz energy column missing?)."
        )
    energy = float(atoms.info["energy"])
    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=float)
    else:
        raise RuntimeError(
            "Atoms has no calculator and no arrays['forces'] "
            "(extxyz forces column missing?)."
        )
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"forces shape {forces.shape}, expected ({len(atoms)}, 3)")
    return energy, forces


def load_training_atoms_from_extxyz(data_path: Path) -> list:
    """Load extxyz bilayer structures and return the train split (same 80/20 as DataLoader)."""
    frames = ase.io.read(str(data_path), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    atoms_list = []
    energies = []
    forces = []
    for atoms in frames:
        energy, force = _energy_forces_from_atoms(atoms)
        # ``Atoms.copy()`` drops the calculator — reattach SinglePoint results.
        a = atoms.copy()
        a.calc = SinglePointCalculator(a, energy=energy, forces=force)
        _assign_mol_id(a)
        atoms_list.append(a)
        energies.append(energy)
        forces.append(force)
    xdata = {"energy": atoms_list}
    ydata = {"energy": np.asarray(energies, dtype=float), "forces": forces}
    x_tr, _x_te, _y_tr, _y_te = train_test_split(xdata, ydata)
    return list(x_tr["energy"])


def half_training_atoms(train_atoms: list, train_frac: float, seed: int = 42) -> list:
    n = len(train_atoms)
    n_keep = max(1, int(round(n * float(train_frac))))
    if n_keep >= n:
        return list(train_atoms)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=n_keep, replace=False))
    return [train_atoms[i] for i in idx]


def fit_or_load_params(
    row: pd.Series,
    *,
    rcut: float,
    train_frac: float,
    data_path: Path,
    force_refit: bool = False,
) -> tuple[np.ndarray, dict, Path]:
    pod_hp, _csv_rcut, pod_hash = pod_hyperparams_from_row(row)
    pod_hp = dict(pod_hp)
    pod_hp.setdefault("species", ["C"])
    ncoeffs = int(ncoeff_from_params(pod_hp))
    data_tag = data_file_tag(data_path)
    cache = fit_cache_path(row, rcut, train_frac, ncoeffs, data_tag)

    if cache.is_file() and not force_refit:
        params = np.asarray(np.load(cache)["params"], dtype=float).ravel()
        print(f"Loaded fit cache: {cache.name}  (n_params={params.size})", flush=True)
        return params, pod_hp, cache

    if not force_refit:
        params, source = _load_existing_pod_params(row, pod_hp, pod_hash)
        if params is not None:
            print(
                f"Loaded POD coefficients from {source} "
                f"(n_params={params.size}, no refit)",
                flush=True,
            )
            _write_fit_cache(
                cache,
                params=params,
                row=row,
                pod_hp=pod_hp,
                pod_hash=pod_hash,
                rcut=float(rcut),
                train_frac=float(train_frac),
                data_path=data_path,
            )
            print(f"  wrote tag cache {cache.name}", flush=True)
            return params, pod_hp, cache
        raise FileNotFoundError(
            f"No cached POD coefficients for hash={pod_hash!r} "
            f"(POD_index={int(row['pod_index'])}). "
            "Expected a file under "
            "pod_hyperparam_search/pod_hyperparam_search_cache/ "
            "(e.g. POD_energy_MBD/<hash>.npz). "
            "Sync the cache to the cluster, or pass --force-refit to fit with LAMMPS."
        )

    print(
        f"Fitting POD: rcut={rcut:g} Å, train_frac={train_frac:g}, "
        f"ncoeffs={ncoeffs}, hash={pod_hash}\n"
        f"  data={data_path}",
        flush=True,
    )
    # Prefer explicit extxyz; fall back to default POD_energy loader for rVV10 name.
    if data_path.is_file():
        train_atoms = load_training_atoms_from_extxyz(data_path)
    else:
        x_tr, _x_te, _x, _y_tr, _y_te, _y = load_data_for_model(
            "POD_energy", supercells=1,
        )
        train_atoms = list(x_tr["energy"])
    fit_atoms = half_training_atoms(train_atoms, train_frac)
    print(
        f"  full train={len(train_atoms)}  using={len(fit_atoms)} "
        f"({100.0 * len(fit_atoms) / max(len(train_atoms), 1):.1f}%)",
        flush=True,
    )

    hyperparams_str = pod_hyperparams_to_str(pod_hp, float(rcut), ["C"])
    lmp = _resolve_lammps_exec()
    print(f"  LAMMPS executable: {lmp}", flush=True)
    params = np.asarray(
        fit_pod(
            hyperparams_str,
            fit_atoms,
            lammps_exec=lmp,
            regularization=float(row.get("regularization", 1e-12)),
            weight_energy=float(row.get("weight_energy", 1000.0)),
            weight_force=float(row.get("weight_force", 1.0)),
        ),
        dtype=float,
    ).ravel()

    _write_fit_cache(
        cache,
        params=params,
        row=row,
        pod_hp=pod_hp,
        pod_hash=pod_hash,
        rcut=float(rcut),
        train_frac=float(train_frac),
        data_path=data_path,
        n_train_used=len(fit_atoms),
        n_train_full=len(train_atoms),
    )
    print(f"  wrote {cache}", flush=True)
    return params, pod_hp, cache


def aa_ab_sep_from_atoms(atoms) -> tuple[float, float]:
    aa_idx, ab_idx = identify_stacking_atoms(atoms)
    aa = float(layer_sep_for_indices(atoms, np.array([aa_idx]))[0])
    ab = float(layer_sep_for_indices(atoms, np.array([ab_idx]))[0])
    return aa, ab


def _finalize_relaxed_with_forces(
    relaxed, calc, traj_path: str, *, write_trajectory: bool = True,
):
    """Attach energy/forces to the relaxed frame and optionally rewrite the trajectory.

    When ``write_trajectory`` is False (e.g. a full LAMMPS dump / ASE step
    history already exists), only evaluate forces and return ``fmax``.
    """
    from ase.io import read

    out = relaxed.copy()
    out.calc = calc
    energy, forces, local_e = _evaluate_energy_forces_local(out)
    if forces is None:
        try:
            forces = np.asarray(out.get_forces(), dtype=float)
        except Exception as exc:
            raise RuntimeError(
                f"Could not evaluate forces after relaxation ({traj_path})"
            ) from exc
    if forces.shape != (len(out), 3) or not np.all(np.isfinite(forces)):
        raise RuntimeError(
            f"Invalid forces after relaxation ({traj_path}): shape={forces.shape}"
        )
    _attach_singlepoint_results(
        out, energy=energy, forces=forces, local_energies=local_e,
    )
    if write_trajectory:
        initial = read(traj_path, index=0)
        _write_success_trajectory(traj_path, initial, out)
    fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    return out, fmax


def results_cache_path(tag: str) -> Path:
    return FIGURES_DIR / f"{tag}_aa_ab_layer_sep_vs_twist_angle.npz"


def plot_aa_ab(
    thetas: np.ndarray,
    aa: np.ndarray,
    ab: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    order = np.argsort(thetas)
    t = thetas[order]
    ax.plot(t, aa[order], "o-", color="C0", label="AA")
    ax.plot(t, ab[order], "s-", color="C1", label="AB")
    ax.set_xlabel(r"Initial twist angle $\theta$ (°)", fontdict=CSFONT)
    ax.set_ylabel("layer separation (Å)", fontdict=CSFONT)
    ax.legend(
        loc="best",
        prop={"family": CSFONT["fontname"], "size": LEGEND_FONTSIZE},
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--twist-angles",
        type=float,
        nargs="*",
        default=None,
        help=f"Twist angles in degrees (default: {list(DEFAULT_TWIST_ANGLES)})",
    )
    p.add_argument("--rcut", type=float, default=DEFAULT_RCUT)
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument(
        "--pod-index",
        type=int,
        default=DEFAULT_POD_INDEX,
        help=f"POD_index row in search CSV (default: {DEFAULT_POD_INDEX})",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "POD hyperparameter-search CSV (default: pod_hyperparam_search.csv). "
            "Example: pod_hyperparam_search/pod_hyperparam_search_MBD.csv"
        ),
    )
    p.add_argument(
        "--data-file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help=(
            "Training extxyz under data/ (or absolute path). Examples: "
            "strained_bilayer_graphene_rVV10.xyz, "
            "bilayer_graphene_MBD.xyz, "
            "strained_bilayer_graphene_dftd3.xyz "
            f"(default: {DEFAULT_DATA_FILE})."
        ),
    )
    p.add_argument(
        "--force-refit",
        action="store_true",
        help="Refit POD with LAMMPS even when cached coefficients exist.",
    )
    p.add_argument(
        "--relax-backend",
        default="lammps",
        choices=("lammps", "ase"),
        help=(
            "Relaxation backend (default: lammps). Use lammps under "
            "``mpirun -np N`` so energy/force evals are "
            "domain-decomposed across MPI ranks. ASE is serial-only."
        ),
    )
    p.add_argument(
        "--relax-etol",
        type=float,
        default=0.0,
        help=(
            "Relative energy tolerance for LAMMPS minimize "
            "(0 = ignore etol so only ftol stops the run; unused by ASE FIRE)."
        ),
    )
    p.add_argument(
        "--relax-ftol",
        type=float,
        default=1e-3,
        help=(
            "Force stopping tolerance (eV/Å). For ASE this is fmax "
            "(max per-atom ||F||). For LAMMPS this script uses the same "
            "numeric value with min_modify norm max so the criterion matches ASE."
        ),
    )
    p.add_argument("--relax-maxiter", type=int, default=2_000)
    p.add_argument(
        "--relax-maxeval",
        type=int,
        default=None,
        help=(
            "LAMMPS maximize force-eval budget. Default: 2 * --relax-maxiter "
            "(avoids the old 800_000 default that paired with every-step dumps "
            "to produce multi-GB trajectories)."
        ),
    )
    p.add_argument(
        "--relax-dump-every",
        type=int,
        default=100,
        help=(
            "LAMMPS dump interval (force evaluations) for the minimize "
            "trajectory. Default 100; use 1 only for very short runs."
        ),
    )
    p.add_argument(
        "--relax-min-style",
        default="fire",
        choices=("fire", "cg"),
        help=(
            "LAMMPS min_style for --relax-backend lammps "
            "(default: fire). CG skips FIRE timestep caps."
        ),
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--reuse-cache", action="store_true")
    args = p.parse_args()
    if args.relax_maxeval is None:
        args.relax_maxeval = int(2 * int(args.relax_maxiter))

    comm, rank, size = _mpi_state()
    is_root = rank == 0
    backend = str(args.relax_backend).strip().lower()

    if size > 1 and backend != "lammps":
        raise SystemExit(
            f"Multi-rank run (size={size}) requires --relax-backend lammps "
            "(ASE FIRE cannot share one minimize across MPI ranks)."
        )
    if size > 1 and comm is None:
        raise SystemExit(
            "Parallel LAMMPS minimize needs a real mpi4py MPI world "
            "(COMM_WORLD size > 1). Slurm reported "
            f"ntasks={size} but mpi4py sees a singleton (PMIx/PMI mismatch).\n"
            "Launch with mpirun after loading OpenMPI:\n"
            "  module load openmpi\n"
            "  module load anaconda/2023-Mar/3\n"
            "  source activate blg_uq\n"
            "  mpirun -np N python visualizations/plot_pod_best_aa_ab_sep_vs_twist.py ..."
        )

    os.chdir(UQ_DIR)
    if is_root:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Rank 0 loads/fits; all ranks receive params + hyperparams for LAMMPS WORLD.
    params = None
    pod_hp = None
    tag = None
    thetas_sorted: list[float] = []
    cached: dict[float, tuple[float, float]] = {}
    pod_index = int(args.pod_index)
    pod_hash = ""
    setup_err: str | None = None

    if is_root:
        try:
            row = select_pod_row_from_csv(args.pod_index, csv_path=args.csv)
            data_path = resolve_data_file(args.data_file)
            data_tag = data_file_tag(data_path)
            tag = run_tag(row, args.rcut, args.train_frac, data_tag)
            csv_note = row.get("search_csv", "pod_hyperparam_search.csv")
            print(
                f"CSV selection: POD_index={int(row['pod_index'])} hash={row['hash']}\n"
                f"  csv={csv_note}\n"
                f"  ncoeff={int(row['ncoeff'])}  csv_rcut={row['rcut']}  "
                f"test RMSE/atom={float(row['test_energy_rmse_per_atom_eV']):.6g} eV\n"
                f"  refit rcut={args.rcut:g} Å  train_frac={args.train_frac:g}\n"
                f"  data_file={data_path}\n"
                f"  tag={tag}\n"
                f"  relax_backend={backend}  MPI ranks={size}  "
                f"relax_ftol={args.relax_ftol:g} (ASE fmax ≡ LAMMPS ftol w/ norm max)\n"
                f"  relax_maxiter={int(args.relax_maxiter)}  "
                f"relax_maxeval={int(args.relax_maxeval)}  "
                f"dump_every={int(args.relax_dump_every)}",
                flush=True,
            )

            params, pod_hp, _fit_cache = fit_or_load_params(
                row,
                rcut=float(args.rcut),
                train_frac=float(args.train_frac),
                data_path=data_path,
                force_refit=bool(args.force_refit),
            )

            thetas = (
                list(DEFAULT_TWIST_ANGLES)
                if args.twist_angles is None
                else [float(t) for t in args.twist_angles]
            )
            thetas_sorted = sorted(thetas, reverse=True)

            npz = results_cache_path(tag)
            if args.reuse_cache and npz.is_file():
                data = np.load(npz)
                for t, aa, ab in zip(data["theta"], data["aa_sep"], data["ab_sep"]):
                    if np.isfinite(aa) and np.isfinite(ab):
                        cached[float(t)] = (float(aa), float(ab))
                print(
                    f"Loaded {len(cached)} cached angle(s) from {npz.name}",
                    flush=True,
                )

            pod_index = int(row["pod_index"])
            pod_hash = str(row["hash"])
        except Exception as exc:
            setup_err = f"{type(exc).__name__}: {exc}"

    if size > 1:
        setup_err = comm.bcast(setup_err, root=0)
    if setup_err:
        raise SystemExit(f"Setup failed on rank 0: {setup_err}")

    if size > 1:
        shared = None
        if is_root:
            shared = {
                "params": np.asarray(params, dtype=float),
                "pod_hp": dict(pod_hp),
                "tag": tag,
                "thetas_sorted": list(thetas_sorted),
                "cached": dict(cached),
                "pod_index": pod_index,
                "pod_hash": pod_hash,
            }
        shared = comm.bcast(shared, root=0)
        if not is_root:
            params = shared["params"]
            pod_hp = shared["pod_hp"]
            tag = shared["tag"]
            thetas_sorted = shared["thetas_sorted"]
            cached = shared["cached"]
            pod_index = shared["pod_index"]
            pod_hash = shared["pod_hash"]

    if is_root:
        print(
            f"Building PODLammpsCalculator (n_params={params.size}, "
            f"rcut={args.rcut:g}) …",
            flush=True,
        )
    calc = PODLammpsCalculator(
        pod_hp, params, elements=["C"], cutoff=float(args.rcut),
    )
    if size > 1:
        from mpi4py import MPI  # noqa: PLC0415

        calc.set_lammps_comm(MPI.COMM_WORLD)
        if is_root:
            print(
                f"LAMMPS using MPI.COMM_WORLD (size={size}) for parallel minimize.",
                flush=True,
            )
    if is_root:
        print("Calculator ready.", flush=True)

    results: dict[float, tuple[float, float]] = dict(cached)
    fmax_by_theta: dict[float, float] = {}
    traj_root = UQ_DIR / "trajectories" / "relaxation_best_fit" / tag
    if is_root:
        traj_root.mkdir(parents=True, exist_ok=True)
    if size > 1:
        comm.Barrier()

    for theta in thetas_sorted:
        if theta in results:
            if is_root:
                print(
                    f"θ={theta:g}°: reuse AA={results[theta][0]:.4f} "
                    f"AB={results[theta][1]:.4f}",
                    flush=True,
                )
            continue

        if is_root:
            print(f"\n=== θ={theta:g}° building cell ===", flush=True)
            atoms0 = build_tblg_atoms(theta)
            print(f"  n_atoms={len(atoms0)}", flush=True)
        else:
            atoms0 = None
        if size > 1:
            atoms0 = comm.bcast(atoms0, root=0)

        traj_suffix = "bestfit" if str(args.relax_min_style) == "fire" else f"bestfit_{args.relax_min_style}"
        traj_path = str(traj_root / f"theta{theta:g}deg_{traj_suffix}.traj")
        fail_path = str(traj_root / f"theta{theta:g}deg_{traj_suffix}_FAIL.traj")
        log_path = str(traj_root / f"theta{theta:g}deg_{traj_suffix}.log")
        dump_path = str(traj_root / f"theta{theta:g}deg_{traj_suffix}.dump")
        if is_root:
            print(
                f"  full traj/log → {Path(traj_path).name} / "
                f"{Path(log_path).name}"
                + (f" / {Path(dump_path).name}" if backend == "lammps" else "")
                + (f"  min_style={args.relax_min_style}" if backend == "lammps" else ""),
                flush=True,
            )

        # All MPI ranks must pass the same dump/log paths for LAMMPS WORLD.
        relaxed = relax_tblg_sample(
            atoms0,
            calc,
            params,
            traj_path,
            relax_backend=backend,
            etol=args.relax_etol,
            ftol=args.relax_ftol,
            maxiter=args.relax_maxiter,
            maxeval=args.relax_maxeval,
            set_params_fn=None,
            fail_traj_path=fail_path,
            write_trajectory=is_root,
            log_path=log_path,
            dump_path=dump_path if backend == "lammps" else None,
            full_trajectory=True,
            dump_every=int(args.relax_dump_every),
            min_style=str(args.relax_min_style),
        )
        # All ranks must join post-relax force eval when sharing COMM_WORLD.
        # Do not rewrite ASE traj / wipe LAMMPS dump: full history is already on disk.
        relaxed, fmax = _finalize_relaxed_with_forces(
            relaxed, calc, traj_path, write_trajectory=False,
        )
        if is_root:
            aa, ab = aa_ab_sep_from_atoms(relaxed)
            results[theta] = (aa, ab)
            fmax_by_theta[theta] = fmax
            print(
                f"  AA={aa:.4f} Å  AB={ab:.4f} Å  max|F|={fmax:.4g} eV/Å",
                flush=True,
            )

            t_arr = np.array(sorted(results), dtype=float)
            aa_arr = np.array([results[t][0] for t in t_arr], dtype=float)
            ab_arr = np.array([results[t][1] for t in t_arr], dtype=float)
            fmax_arr = np.array(
                [fmax_by_theta.get(t, np.nan) for t in t_arr], dtype=float,
            )
            npz = results_cache_path(tag)
            np.savez(
                npz,
                theta=t_arr,
                aa_sep=aa_arr,
                ab_sep=ab_arr,
                fmax=fmax_arr,
                tag=np.asarray(tag),
                rcut=np.asarray(float(args.rcut)),
                train_frac=np.asarray(float(args.train_frac)),
                pod_index=np.asarray(int(pod_index)),
                pod_hash=np.asarray(str(pod_hash)),
            )
        if size > 1:
            comm.Barrier()

    # Close LAMMPS on *all* ranks before any root-only plotting so MPI
    # finalization does not hang with an open WORLD instance on some ranks.
    close_fn = getattr(calc, "close", None)
    if callable(close_fn):
        close_fn()
    if size > 1:
        comm.Barrier()

    if is_root:
        t_arr = np.array(sorted(results), dtype=float)
        aa_arr = np.array([results[t][0] for t in t_arr], dtype=float)
        ab_arr = np.array([results[t][1] for t in t_arr], dtype=float)
        out_png = FIGURES_DIR / f"{tag}_aa_ab_layer_sep_vs_twist_angle.png"
        plot_aa_ab(t_arr, aa_arr, ab_arr, out_png, dpi=args.dpi)
        print(
            f"\nDone. tag={tag}  angles={list(t_arr)}  figure={out_png}",
            flush=True,
        )


if __name__ == "__main__":
    main()
