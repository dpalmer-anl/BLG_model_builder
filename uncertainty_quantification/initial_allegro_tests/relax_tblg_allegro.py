#!/usr/bin/env python3
"""
relax_tblg_allegro.py
=====================
Relax a TBLG structure at a chosen twist angle using a trained Allegro model,
then plot the interlayer separation of the top layer.

Usage
-----
# relax at 2.88 degrees using the default output directory
python relax_tblg_allegro.py

# specify twist angle and trained model directory
python relax_tblg_allegro.py --twist-angle 2.88 --model-dir allegro_blg_output

Requirements
------------
conda activate allegro_env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "allegro_blg_output"
DEFAULT_TWIST_ANGLE = 2.88
DEFAULT_LAT_CON = 2.46
DEFAULT_INITIAL_SEP = 3.35
DEFAULT_R_MAX = 6.0
DEFAULT_FMAX = 0.001   # eV/Å  convergence criterion
DEFAULT_MAXSTEPS = 2000


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_allegro_calculator(model_dir: Path, r_max: float, device: str):
    """Load a trained Allegro checkpoint and return an ASE NequIPCalculator."""
    import torch
    from nequip.utils.global_state import set_global_state
    set_global_state()

    from nequip.train import EMALightningModule
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
    )
    from nequip.integrations.ase import NequIPCalculator

    # Prefer the small default checkpoint (best-v2.ckpt, ~1760 params).
    ckpt = model_dir / "best-v2.ckpt"
    if not ckpt.exists():
        ckpt = model_dir / "best.ckpt"
    if not ckpt.exists():
        candidates = sorted(model_dir.glob("*.ckpt"))
        if not candidates:
            raise FileNotFoundError(
                f"No .ckpt file found in {model_dir}. "
                "Train the model first with fit_allegro.py."
            )
        ckpt = candidates[-1]
    print(f"Loading checkpoint: {ckpt}", flush=True)

    module = EMALightningModule.load_from_checkpoint(
        str(ckpt), map_location=device
    )
    module.eval()

    # Extract the inner graph model (stored as model["sole_model"])
    inner_model = module.model["sole_model"]
    inner_model.eval()

    # Wrap to keep torch.enable_grad active during forward (needed for forces)
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
    return calc


# ---------------------------------------------------------------------------
# TBLG structure (reuses logic from run_uq_propagation_relaxation.py)
# ---------------------------------------------------------------------------

def _build_tblg(theta_deg: float, lat_con: float, sep: float, h_vac: float = 20.0):
    try:
        import flatgraphene as fg
    except ImportError as exc:
        raise ImportError(
            "flatgraphene is required. Install it in the active environment."
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
    return atoms


def _top_layer_mask(atoms) -> np.ndarray:
    """Boolean mask selecting the top graphene layer (higher z)."""
    z = atoms.positions[:, 2]
    return z > float(np.median(z))


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def relax(atoms, calc, fmax: float, maxsteps: int, traj_path: Path | None = None):
    """Run FIRE relaxation; return the relaxed atoms object."""
    from ase.optimize import FIRE

    atoms.calc = calc
    if traj_path is not None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
    logfile_arg = str(traj_path.with_suffix(".log")) if traj_path else "-"
    dyn = FIRE(atoms, logfile=logfile_arg)

    if traj_path is not None:
        from ase.io.trajectory import Trajectory
        traj_writer = Trajectory(str(traj_path), "w", atoms)
        dyn.attach(traj_writer.write, interval=10)

    print(
        f"Starting FIRE relaxation  (fmax={fmax} eV/Å, maxsteps={maxsteps}) …",
        flush=True,
    )
    converged = dyn.run(fmax=fmax, steps=maxsteps)
    fmax_final = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    if converged:
        print(f"Converged after {dyn.nsteps} steps  (|F|max = {fmax_final:.4e} eV/Å)",
              flush=True)
    else:
        print(
            f"WARNING: not converged after {maxsteps} steps  "
            f"(|F|max = {fmax_final:.4e} eV/Å)",
            flush=True,
        )
    return atoms


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_interlayer_separation(atoms, theta_deg: float, out_path: Path) -> None:
    """
    Plot interlayer separation of the top layer atoms across the supercell.

    Interlayer separation is defined per-atom as:
        delta_z = 2 * (z_i - mean(z_top))
    where mean(z_top) is the mean z of all top-layer atoms.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean_z = float(np.mean(atoms.positions[:, 2]))
    mask = _top_layer_mask(atoms)
    pos = atoms.positions[mask]          # (N_top, 3)
    xy = pos[:, :2]
    z  = pos[:, 2]

    layer_sep = 2.0 * (z - float(mean_z))   # signed, Å

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=layer_sep,
        #cmap="RdBu_r",
        s=60,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Interlayer separation  2(z − ⟨z⟩)  (Å)", fontsize=11)
    ax.set_xlabel("x  (Å)", fontsize=11)
    ax.set_ylabel("y  (Å)", fontsize=11)
    ax.set_title(
        f"Allegro-relaxed TBLG  θ = {theta_deg}°\n"
        f"top layer  |  max = {layer_sep.max():.3f} Å  min = {layer_sep.min():.3f} Å",
        fontsize=11,
    )
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved interlayer separation plot to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Relax a TBLG structure with a trained Allegro model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--twist-angle", type=float, default=DEFAULT_TWIST_ANGLE,
                   help=f"Twist angle in degrees (default: {DEFAULT_TWIST_ANGLE})")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help=f"Directory containing best.ckpt (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--r-max", type=float, default=DEFAULT_R_MAX,
                   help=f"Cutoff radius used during training (default: {DEFAULT_R_MAX} Å)")
    p.add_argument("--lat-con", type=float, default=DEFAULT_LAT_CON,
                   help=f"Graphene lattice constant (default: {DEFAULT_LAT_CON} Å)")
    p.add_argument("--initial-sep", type=float, default=DEFAULT_INITIAL_SEP,
                   help=f"Initial interlayer separation (default: {DEFAULT_INITIAL_SEP} Å)")
    p.add_argument("--fmax", type=float, default=DEFAULT_FMAX,
                   help=f"Force convergence criterion (default: {DEFAULT_FMAX} eV/Å)")
    p.add_argument("--maxsteps", type=int, default=DEFAULT_MAXSTEPS,
                   help=f"Max FIRE steps (default: {DEFAULT_MAXSTEPS})")
    p.add_argument("--device", default="cpu",
                   help="Torch device: cpu or cuda (default: cpu)")
    p.add_argument("--output-dir", type=Path, default=HERE / "relaxations_allegro",
                   help="Where to save trajectory and figures.")
    args = p.parse_args()

    # ── build structure ──
    print(
        f"\nBuilding TBLG at θ = {args.twist_angle}°  "
        f"(lat_con = {args.lat_con} Å, sep = {args.initial_sep} Å) …",
        flush=True,
    )
    atoms = _build_tblg(args.twist_angle, args.lat_con, args.initial_sep)
    n_atoms = len(atoms)
    n_top = int(_top_layer_mask(atoms).sum())
    print(f"  n_atoms = {n_atoms}  (top layer: {n_top})", flush=True)

    # ── load calculator ──
    calc = _load_allegro_calculator(
        args.model_dir.resolve(), r_max=args.r_max, device=args.device
    )

    # ── relax ──
    tag = f"allegro_theta{args.twist_angle:g}deg"
    traj_path = args.output_dir / f"{tag}.traj"
    relaxed = relax(atoms, calc, fmax=args.fmax, maxsteps=args.maxsteps,
                    traj_path=traj_path)

    # ── summary ──
    mask = _top_layer_mask(relaxed)
    z_top = relaxed.positions[mask, 2]
    layer_sep = 2.0 * (z_top - float(np.mean(z_top)))
    print(f"\nTop-layer interlayer separation (2·(z − ⟨z⟩)):", flush=True)
    print(f"  max = {layer_sep.max():.4f} Å", flush=True)
    print(f"  min = {layer_sep.min():.4f} Å", flush=True)
    print(f"  mean = {layer_sep.mean():.4f} Å  (≈ 0 by construction)", flush=True)

    # ── plot ──
    plot_path = args.output_dir / f"{tag}_interlayer_sep.png"
    plot_interlayer_separation(relaxed, args.twist_angle, plot_path)


if __name__ == "__main__":
    main()
