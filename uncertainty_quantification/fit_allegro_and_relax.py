#!/usr/bin/env python3
"""
fit_allegro_and_relax.py
========================
Fit an Allegro potential (cutoff 6 Å) to
``data/strained_bilayer_graphene_rVV10.xyz`` using the **same train/test
split** as POD MCMC (``DataLoader.train_test_split``: ``TEST_SIZE=0.2``,
``np.random.seed(42)``), save the trained model, then relax TBLG cells at
selected twist angles with that model.

A validation subset is carved from the POD **train** set only (test is
untouched and matches ``run_MCMC.py`` / ``model_fit.py``).

Training is **energy-only** (no forces or stress in the loss or the written
split files).

Run from ``uncertainty_quantification`` in the ``allegro_env`` conda env::

    conda activate allegro_env
    python fit_allegro_and_relax.py
    python fit_allegro_and_relax.py --epochs 100 --device cuda
    python fit_allegro_and_relax.py --skip-fit --model-dir allegro_blg_rcut6
    python fit_allegro_and_relax.py --force-fit   # retrain even if ckpt exists
    python fit_allegro_and_relax.py --skip-relax   # fit only

If ``allegro_blg_rcut6`` (or ``--model-dir``) already contains a checkpoint
(``best.ckpt`` / ``best-*.ckpt`` / ``last.ckpt``), training is skipped and that
model is used for TBLG relaxations unless ``--force-fit`` is set.

Twist angles default to::

    0.93, 0.99, 1.05, 1.08, 1.12, 1.16, 1.2

TBLG cells match ``run_uq_propagation_relaxation.py``: ``find_p_q(..., a_tol=0.01)``,
hex bilayer, and wrap-based cell cleanup (some ``flatgraphene`` installs otherwise
inflate atom counts ~3×).
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from strain_data import LAT_CON  # noqa: E402

DEFAULT_LAT_CON = float(LAT_CON)
DEFAULT_INITIAL_SEP = 3.35
DEFAULT_XYZ = REPO_ROOT / "data" / "strained_bilayer_graphene_rVV10.xyz"
DEFAULT_OUT = HERE / "allegro_blg_rcut6"
DEFAULT_RELAX_DIR = DEFAULT_OUT / "relaxations"
DEFAULT_R_MAX = 6.0
DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.1  # fraction of POD train used as Allegro val
DEFAULT_TWIST_ANGLES = (0.99, 1.05, 1.08, 1.12, 1.16, 1.2)
DEFAULT_FMAX = 0.001
DEFAULT_MAXSTEPS = 2000
DEFAULT_EPOCHS = 100


# ---------------------------------------------------------------------------
# TBLG geometry (flatgraphene)
# ---------------------------------------------------------------------------

def _remove_atoms_outside_cell(atoms_object) -> None:
    """
    Delete atoms whose Cartesian positions move under ``Atoms.wrap()``.

    Some older ``flatgraphene`` installs use a broken scaled-position loop that
    leaves duplicate images in the supercell (inflating atom counts ~3×).
    """
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
    """Ensure ``fg.twist.remove_atoms_outside_cell`` uses wrap-based cleanup."""
    fg.twist.remove_atoms_outside_cell = _remove_atoms_outside_cell


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
            "flatgraphene is required for TBLG structure generation."
        ) from exc

    _patch_flatgraphene_cell_cleanup(fg)
    p, q, theta_comp = fg.twist.find_p_q(float(theta_deg), a_tol=0.01)
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
    atoms.info["tblg_p"] = int(p)
    atoms.info["tblg_q"] = int(q)
    atoms.info["tblg_theta_comp"] = float(theta_comp)
    return atoms


def layer_separation_metrics(atoms) -> tuple[float, float]:
    """Max and min layer separation from Cartesian *z* (Å)."""
    z = np.asarray(atoms.get_positions(wrap=False), dtype=float)[:, 2]
    dz = np.abs(z - float(np.mean(z)))
    return 2.0 * float(np.max(dz)), 2.0 * float(np.min(dz))


# ---------------------------------------------------------------------------
# Env / imports
# ---------------------------------------------------------------------------

def _ensure_allegro() -> None:
    missing = []
    for pkg in ("nequip", "allegro", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SystemExit(
            f"Required packages not found: {missing}\n"
            "Use conda env allegro_env (nequip + allegro + torch)."
        )


# ---------------------------------------------------------------------------
# POD-matching train / test split
# ---------------------------------------------------------------------------

def pod_train_test_indices(
    n_total: int,
    *,
    test_size: float | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Indices matching ``blg_model_builder.DataLoader.train_test_split``.

    Uses the global NumPy RNG after ``np.random.seed(seed)`` and
    ``n_test = int(n_total * TEST_SIZE)``, same as POD energy MCMC.
    """
    from blg_model_builder.DataLoader import TEST_SIZE

    if test_size is None:
        test_size = float(TEST_SIZE)
    np.random.seed(int(seed))
    n_test = int(n_total * float(test_size))
    test_idx = np.random.choice(n_total, size=n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def carve_val_from_train(
    train_idx: np.ndarray,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split POD train indices into Allegro fit-train and validation."""
    train_idx = np.asarray(train_idx, dtype=int).ravel()
    n_val = max(1, int(round(float(val_fraction) * train_idx.size)))
    n_val = min(n_val, max(train_idx.size - 1, 1))
    rng = np.random.RandomState(int(seed))
    pick = rng.choice(train_idx.size, size=n_val, replace=False)
    val_idx = np.sort(train_idx[pick])
    fit_idx = np.setdiff1d(train_idx, val_idx)
    return fit_idx, val_idx


def _energy_only_atoms(atoms):
    """Copy an Atoms object keeping only total energy (drop forces / stress)."""
    from ase.calculators.singlepoint import SinglePointCalculator

    out = atoms.copy()
    energy = None
    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        if "energy" in atoms.info:
            energy = float(atoms.info["energy"])
    if energy is None:
        raise ValueError("Frame has no energy; cannot build energy-only training data.")

    # Drop stress / force-like keys that cause inconsistent batching in NequIP.
    for key in list(out.info.keys()):
        kl = str(key).lower()
        if "stress" in kl or "force" in kl:
            del out.info[key]
    for key in list(out.arrays.keys()):
        kl = str(key).lower()
        if "force" in kl or "stress" in kl:
            del out.arrays[key]

    out.calc = SinglePointCalculator(out, energy=energy)
    return out


def write_split_xyz(
    frames: list,
    indices: np.ndarray,
    path: Path,
) -> None:
    import ase.io

    path.parent.mkdir(parents=True, exist_ok=True)
    subset = [
        _energy_only_atoms(frames[int(i)])
        for i in np.asarray(indices, dtype=int).ravel()
    ]
    ase.io.write(str(path), subset, format="extxyz")
    print(f"  Wrote {path}  ({len(subset)} frames, energy-only)", flush=True)


def prepare_pod_splits(
    xyz_path: Path,
    split_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> dict:
    """Load xyz, apply POD split, write train/val/test extxyz + indices.npz."""
    import ase.io
    from blg_model_builder.DataLoader import TEST_SIZE

    print(f"\nLoading {xyz_path} …", flush=True)
    frames = ase.io.read(str(xyz_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    n_total = len(frames)
    train_idx, test_idx = pod_train_test_indices(n_total, seed=seed)
    fit_idx, val_idx = carve_val_from_train(
        train_idx, val_fraction=val_fraction, seed=1,
    )
    print(
        f"  POD split (TEST_SIZE={TEST_SIZE}, seed={seed}): "
        f"n_total={n_total}  n_train={train_idx.size}  n_test={test_idx.size}",
        flush=True,
    )
    print(
        f"  Allegro carve: n_fit={fit_idx.size}  n_val={val_idx.size}  "
        f"(val from POD train only)",
        flush=True,
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.xyz"
    val_path = split_dir / "val.xyz"
    test_path = split_dir / "test.xyz"
    write_split_xyz(frames, fit_idx, train_path)
    write_split_xyz(frames, val_idx, val_path)
    write_split_xyz(frames, test_idx, test_path)

    indices_path = split_dir / "pod_split_indices.npz"
    np.savez(
        indices_path,
        train_idx=train_idx,
        test_idx=test_idx,
        fit_idx=fit_idx,
        val_idx=val_idx,
        seed=int(seed),
        test_size=float(TEST_SIZE),
        val_fraction=float(val_fraction),
        n_total=int(n_total),
        xyz=str(xyz_path.resolve()),
    )
    print(f"  Wrote {indices_path}", flush=True)
    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "indices_path": indices_path,
        "n_total": n_total,
        "n_fit": int(fit_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# Allegro config + training
# ---------------------------------------------------------------------------

def write_allegro_config(
    config_path: Path,
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    epochs: int,
    r_max: float,
    seed: int,
    batch_size: int = 4,
    lr: float = 0.001,
    l_max: int = 1,
    num_layers: int = 1,
    num_scalar_features: int = 8,
    num_tensor_features: int = 8,
    num_bessels: int = 8,
    polynomial_cutoff_p: int = 6,
) -> None:
    """Write nequip 0.18 / allegro 0.8 config with explicit train/val/test files.

    Energy-only fit: no forces/stress in the loss; dataset stats use
    ``EnergyOnlyDataStatisticsManager``.
    """
    cfg = textwrap.dedent(f"""\
        # Allegro config — strained bilayer graphene (rVV10)
        # Auto-generated by fit_allegro_and_relax.py
        # Train/test match DataLoader.train_test_split (POD MCMC); val ⊂ train.
        # Energy-only training (no forces / stress).

        run: [train, test]

        seed: {seed}
        cutoff_radius: {r_max}
        model_type_names: [C]

        data:
          _target_: nequip.data.datamodule.ASEDataModule
          seed: ${{seed}}
          train_file_path: {train_path.resolve().as_posix()}
          val_file_path: {val_path.resolve().as_posix()}
          test_file_path: {test_path.resolve().as_posix()}
          exclude_keys: [forces, force, stress]
          transforms:
            - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
              model_type_names: ${{model_type_names}}
              chemical_species_to_atom_type_map:
                C: C
            - _target_: nequip.data.transforms.NeighborListTransform
              r_max: ${{cutoff_radius}}
          train_dataloader:
            _target_: torch.utils.data.DataLoader
            batch_size: {batch_size}
          val_dataloader:
            _target_: torch.utils.data.DataLoader
            batch_size: 8
          test_dataloader: ${{data.val_dataloader}}
          stats_manager:
            _target_: nequip.data.EnergyOnlyDataStatisticsManager
            type_names: ${{model_type_names}}

        trainer:
          _target_: lightning.Trainer
          max_epochs: {epochs}
          check_val_every_n_epoch: 1
          log_every_n_steps: 10
          logger:
            _target_: lightning.pytorch.loggers.CSVLogger
            save_dir: ${{hydra:runtime.output_dir}}
            name: ""
            version: 0
          callbacks:
            - _target_: lightning.pytorch.callbacks.ModelCheckpoint
              monitor: val0_epoch/weighted_sum
              mode: min
              dirpath: ${{hydra:runtime.output_dir}}
              filename: best
              save_last: true
            - _target_: nequip.train.callbacks.TestTimeXYZFileWriter
              out_file: ${{hydra:runtime.output_dir}}/test_predictions
              output_fields_from_original_dataset: [total_energy]
              chemical_symbols: [C]
            - _target_: nequip.train.callbacks.LossCoefficientMonitor
              interval: epoch
              frequency: 10

        num_scalar_features: {num_scalar_features}

        training_module:
          _target_: nequip.train.EMALightningModule
          loss:
            _target_: nequip.train.EnergyOnlyLoss
            per_atom_energy: true
          val_metrics:
            _target_: nequip.train.EnergyOnlyMetrics
            coeffs:
              per_atom_energy_mae: 1.0
              total_energy_mae: null
              per_atom_energy_rmse: null
              total_energy_rmse: null
              per_atom_energy_maxabserr: null
              total_energy_maxabserr: null
          test_metrics: ${{training_module.val_metrics}}
          optimizer:
            _target_: torch.optim.Adam
            lr: {lr}

          model:
            _target_: allegro.model.AllegroModel
            seed: {seed}
            model_dtype: float32
            type_names: ${{model_type_names}}
            r_max: ${{cutoff_radius}}
            radial_chemical_embed:
              _target_: allegro.nn.TwoBodyBesselScalarEmbed
              num_bessels: {num_bessels}
              bessel_trainable: false
              polynomial_cutoff_p: {polynomial_cutoff_p}
            l_max: {l_max}
            parity: true
            num_layers: {num_layers}
            num_scalar_features: ${{num_scalar_features}}
            num_tensor_features: {num_tensor_features}
            allegro_mlp_hidden_layers_width: {num_scalar_features}
            readout_mlp_hidden_layers_width: {num_scalar_features}
            avg_num_neighbors: ${{training_data_stats:per_type_num_neighbors_mean}}
            per_type_energy_shifts: ${{training_data_stats:per_atom_energy_mean}}
            per_type_energy_scales: ${{training_data_stats:total_energy_std}}
    """)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(cfg)
    print(f"Config written to {config_path}", flush=True)


def run_training(output_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "nequip.scripts.train",
        f"--config-path={output_dir.as_posix()}",
        "--config-name=config",
        f"hydra.run.dir={output_dir.as_posix()}",
    ]
    print(f"\nRunning: {' '.join(cmd)}\n", flush=True)
    subprocess.check_call(cmd)


def resolve_best_checkpoint(model_dir: Path) -> Path:
    ckpt = find_existing_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No Allegro checkpoint found under {model_dir}")
    return ckpt


def find_existing_checkpoint(model_dir: Path) -> Path | None:
    """Return a trained checkpoint path if one exists, else ``None``."""
    if not model_dir.is_dir():
        return None
    for name in ("best.ckpt", "best-v1.ckpt", "best-v2.ckpt"):
        p = model_dir / name
        if p.is_file():
            return p
    candidates = sorted(model_dir.glob("best*.ckpt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    last = model_dir / "last.ckpt"
    if last.is_file():
        return last
    return None


def save_uq_best_fit_params(
    ckpt: Path,
    *,
    r_max: float,
    device: str,
    out_dir: Path,
) -> Path:
    """Write ``best_fit_params/Allegro_energy_ckpt_<tag>_best_fit_params.npz``."""
    from blg_model_builder.allegro_interface import (
        AllegroCalculator,
        allegro_bounds_from_params,
        checkpoint_tag,
    )

    calc = AllegroCalculator(ckpt, r_max=r_max, device=device)
    params = np.asarray(calc.get_parameters(), dtype=float)
    bounds = allegro_bounds_from_params(params)
    tag = checkpoint_tag(ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Allegro_energy_ckpt_{tag}_best_fit_params.npz"
    np.savez(
        out_path,
        params=params,
        bounds=bounds,
        allegro_checkpoint=str(ckpt.resolve()),
        allegro_ckpt_tag=tag,
        allegro_r_max=float(r_max),
    )
    print(f"Saved UQ best-fit params → {out_path}", flush=True)
    print(f"  Model name for UQ: Allegro_energy_ckpt_{tag}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# TBLG relaxation
# ---------------------------------------------------------------------------

def load_allegro_calculator(model_dir: Path, r_max: float, device: str):
    """Load trained Allegro checkpoint as ASE NequIPCalculator."""
    import torch
    from nequip.utils.global_state import set_global_state

    set_global_state()
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
    )
    from nequip.integrations.ase import NequIPCalculator
    from nequip.train import EMALightningModule

    ckpt = resolve_best_checkpoint(model_dir)
    print(f"Loading checkpoint: {ckpt}", flush=True)
    module = EMALightningModule.load_from_checkpoint(
        str(ckpt), map_location=device,
    )
    module.eval()
    inner_model = module.model["sole_model"]
    inner_model.eval()

    class _GradModel(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self._m = m

        def forward(self, data):
            with torch.enable_grad():
                return self._m(data)

    wrapped = _GradModel(inner_model)
    wrapped.eval()
    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=["C"],
            chemical_species_to_atom_type_map={"C": "C"},
        ),
        NeighborListTransform(r_max=r_max),
    ]
    calc = NequIPCalculator(model=wrapped, device=device, transforms=transforms)
    print(f"NequIPCalculator ready (device={device}, r_max={r_max} Å)", flush=True)
    return calc, ckpt


def relax_tblg(
    calc,
    *,
    twist_angle: float,
    lat_con: float,
    initial_sep: float,
    fmax: float,
    maxsteps: int,
    out_dir: Path,
) -> Path:
    """Relax one TBLG cell built via ``build_tblg_atoms``."""
    from ase.optimize import FIRE
    from ase.io.trajectory import Trajectory

    print(
        f"\n=== Relax TBLG θ={twist_angle:g}°  "
        f"(a={lat_con:g} Å, sep0={initial_sep:g} Å) ===",
        flush=True,
    )
    atoms = build_tblg_atoms(
        twist_angle, lat_con=lat_con, sep=initial_sep,
    )
    print(
        f"  find_p_q(a_tol=0.01): p={atoms.info['tblg_p']} "
        f"q={atoms.info['tblg_q']} "
        f"θ_comp={atoms.info['tblg_theta_comp']:.4f}°  "
        f"n_atoms={len(atoms)}",
        flush=True,
    )
    atoms.calc = calc

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"allegro_theta{twist_angle:g}deg"
    traj_path = out_dir / f"{tag}.traj"
    log_path = out_dir / f"{tag}.log"
    dyn = FIRE(atoms, logfile=str(log_path))
    traj_writer = Trajectory(str(traj_path), "w", atoms)
    dyn.attach(traj_writer.write, interval=10)

    print(
        f"  FIRE  fmax={fmax} eV/Å  maxsteps={maxsteps} …",
        flush=True,
    )
    converged = dyn.run(fmax=fmax, steps=maxsteps)
    fmax_final = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    max_sep, min_sep = layer_separation_metrics(atoms)
    if converged:
        print(
            f"  Converged after {dyn.nsteps} steps  "
            f"|F|max={fmax_final:.4e}  max_sep={max_sep:.4f}  "
            f"min_sep={min_sep:.4f} Å",
            flush=True,
        )
    else:
        print(
            f"  WARNING: not converged after {maxsteps} steps  "
            f"|F|max={fmax_final:.4e}  max_sep={max_sep:.4f}  "
            f"min_sep={min_sep:.4f} Å",
            flush=True,
        )
    traj_writer.write()
    traj_writer.close()
    print(f"  Wrote {traj_path}", flush=True)
    return traj_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit Allegro (rcut=6 Å) with POD MCMC train/test split, then "
            "relax TBLG at selected twist angles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--xyz", type=Path, default=DEFAULT_XYZ,
                   help=f"Training extxyz (default: {DEFAULT_XYZ})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                   help=f"Model / split output directory (default: {DEFAULT_OUT})")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Checkpoint directory for relax / --skip-fit "
                        "(default: --output-dir).")
    p.add_argument("--relax-dir", type=Path, default=None,
                   help=f"TBLG trajectory directory (default: <output-dir>/relaxations)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--r-max", type=float, default=DEFAULT_R_MAX,
                   help=f"Cutoff radius in Å (default: {DEFAULT_R_MAX})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"POD split seed (default: {DEFAULT_SEED})")
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                   help="Fraction of POD train used as Allegro val "
                        f"(default: {DEFAULT_VAL_FRACTION})")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", default="cpu",
                   help="Torch device for training artifacts / relax (cpu|cuda)")
    p.add_argument(
        "--twist-angles",
        type=float,
        nargs="+",
        default=None,
        help=f"TBLG twist angles in degrees (default: {list(DEFAULT_TWIST_ANGLES)})",
    )
    p.add_argument(
        "--twist-angle",
        type=float,
        action="append",
        default=None,
        dest="twist_angle_list",
        help="Single TBLG twist angle in degrees (repeatable; alias of --twist-angles).",
    )
    p.add_argument("--lat-con", type=float, default=DEFAULT_LAT_CON,
                   help=f"TBLG lattice constant (default: {DEFAULT_LAT_CON})")
    p.add_argument("--initial-sep", type=float, default=DEFAULT_INITIAL_SEP)
    p.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    p.add_argument("--maxsteps", type=int, default=DEFAULT_MAXSTEPS)
    p.add_argument("--skip-fit", action="store_true",
                   help="Skip training; use existing checkpoint in --model-dir.")
    p.add_argument("--force-fit", action="store_true",
                   help="Retrain even if a checkpoint already exists.")
    p.add_argument("--skip-relax", action="store_true",
                   help="Skip TBLG relaxations.")
    p.add_argument(
        "--best-fit-dir",
        type=Path,
        default=HERE / "best_fit_params",
        help="Where to write Allegro_energy_ckpt_*_best_fit_params.npz",
    )
    return p.parse_args()


def _resolve_twist_angles(args: argparse.Namespace) -> list[float]:
    """Merge ``--twist-angles`` / ``--twist-angle``; default if neither given."""
    angles: list[float] = []
    if args.twist_angles:
        angles.extend(float(t) for t in args.twist_angles)
    if args.twist_angle_list:
        angles.extend(float(t) for t in args.twist_angle_list)
    if not angles:
        angles = list(DEFAULT_TWIST_ANGLES)
    return angles


def main() -> None:
    args = _parse_args()
    args.twist_angles = _resolve_twist_angles(args)
    os.chdir(HERE)

    xyz_path = args.xyz.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = (args.model_dir or args.output_dir).resolve()
    relax_dir = (
        args.relax_dir.resolve()
        if args.relax_dir is not None
        else (output_dir / "relaxations")
    )

    existing_ckpt = find_existing_checkpoint(model_dir)
    do_fit = True
    if args.skip_fit:
        do_fit = False
    elif existing_ckpt is not None and not args.force_fit:
        do_fit = False
        print(
            f"Found existing Allegro checkpoint: {existing_ckpt}\n"
            f"  Skipping fit (pass --force-fit to retrain).",
            flush=True,
        )

    if do_fit:
        if not xyz_path.is_file():
            raise SystemExit(f"Dataset not found: {xyz_path}")
        _ensure_allegro()
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = prepare_pod_splits(
            xyz_path,
            output_dir / "splits",
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
        config_path = output_dir / "config.yaml"
        write_allegro_config(
            config_path,
            train_path=splits["train_path"],
            val_path=splits["val_path"],
            test_path=splits["test_path"],
            epochs=args.epochs,
            r_max=args.r_max,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        run_training(output_dir)
        model_dir = output_dir
        ckpt = resolve_best_checkpoint(model_dir)
        try:
            save_uq_best_fit_params(
                ckpt,
                r_max=args.r_max,
                device=args.device,
                out_dir=args.best_fit_dir.resolve(),
            )
        except Exception as exc:
            print(
                f"WARNING: could not write UQ best_fit_params ({type(exc).__name__}: {exc}). "
                "Checkpoint is still in the output directory.",
                file=sys.stderr,
            )
        print(f"\nTraining complete. Checkpoint: {ckpt}", flush=True)
    else:
        if find_existing_checkpoint(model_dir) is None:
            raise SystemExit(
                f"No Allegro checkpoint found under {model_dir}. "
                "Train first or pass a --model-dir that contains best.ckpt / last.ckpt."
            )
        print(f"Using fitted model in {model_dir}", flush=True)

    if args.skip_relax:
        print("Skipping TBLG relaxations (--skip-relax).", flush=True)
        print("Done.", flush=True)
        return

    _ensure_allegro()
    calc, ckpt = load_allegro_calculator(
        model_dir, r_max=args.r_max, device=args.device,
    )
    print(f"Relaxing with {ckpt}", flush=True)
    for theta in args.twist_angles:
        relax_tblg(
            calc,
            twist_angle=float(theta),
            lat_con=args.lat_con,
            initial_sep=args.initial_sep,
            fmax=args.fmax,
            maxsteps=args.maxsteps,
            out_dir=relax_dir,
        )
    print(f"\nAll relaxations written under {relax_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
