"""
Generalized stacking-fault energy (GSFE) at **zero in-plane strain** and a
fixed interlayer separation.

DFT frames come from :func:`load_energy_data` (rVV10).  Zero-strain selection
uses the project equilibrium lattice ``LAT_CON`` (≈ 2.469 Å) via
:func:`blg_model_builder.strain_data.cell_strains` with
``STRAIN_RANGE``, and keeps only frames with flat layers
(:func:`layers_have_uniform_z`) so buckled / MD-like geometries are excluded.

Stacking (AB / SP / AA) uses :func:`blg_model_builder.strain_data.identify_stacking`.
Only frames whose interlayer separation is within ``--sep-tol`` of
``--layer-sep`` are kept.

POD / PODD3 models (``--models``) are built the same way as in
``plot_bilayer_graphene_pes.py`` (best-fit LAMMPS calculator from cached
coefficients / ``POD_index``).  For models, the GSFE curve is **evaluated**
on a dense disregistry grid via :func:`get_bilayer_atoms` at ``LAT_CON``.
DFT rVV10 is shown only as AB / SP / AA scatter points (no spline).

Output: ``figures/<tag>_gsfe_zero_strain_d*.png``

Run from ``uncertainty_quantification``::

    python visualizations/plot_gsfe.py
    python visualizations/plot_gsfe.py --models POD_energy --POD-index 27
    python visualizations/plot_gsfe.py --models 'POD_energy_POD_index_27*' --layer-sep 3.43
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from blg_model_builder.DataLoader import load_energy_data
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
)
from blg_model_builder.geom_tools import get_bilayer_atoms
from blg_model_builder.strain_data import (
    LAT_CON,
    STRAIN_RANGE,
    identify_stacking,
    interlayer_sep,
    layers_have_uniform_z,
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
REPO_ROOT = UQ_DIR.parent

_vis_dir = str(HERE)
if _vis_dir not in sys.path:
    sys.path.insert(0, _vis_dir)
_uq_dir = str(UQ_DIR)
if _uq_dir not in sys.path:
    sys.path.insert(0, _uq_dir)

from plot_bilayer_graphene_pes import (  # noqa: E402
    _build_model_calculator,
    _chdir_for_dataloader,
    _is_pod_family_model,
    _model_plot_label,
    _pod_family_load_name,
    _resolve_model_names,
    energy_per_atom_total,
    evaluate_model_energies_on_atoms,
)
from uq_model_runtime import is_uq_energy_model  # noqa: E402

CSFONT = {"fontname": "sans-serif", "size": 15}
DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"
DEFAULT_LAYER_SEP = 3.43
DEFAULT_SEP_TOL = 0.02
DEFAULT_N_DISREG = 101
DEFAULT_LAT_CON = float(LAT_CON)  # 2.4694 Å
STACKINGS_GSFE = ("AB", "SP", "AA")
# Disregistry milestones for the GSFE path (AB → SP → BA → AA → AB)
_DISREG_AB = 0.0
_DISREG_SP = 1.0 / 6.0
_DISREG_BA = 1.0 / 3.0
_DISREG_AA = 2.0 / 3.0
_DISREG_AB2 = 1.0

_COLOR_DFT = "black"
_COLOR_MODEL_CYCLE = (
    "#2ca02c",
    "#d62728",
    "#1f77b4",
    "#9467bd",
    "#8c564b",
    "#e377c2",
)


def build_disregistry_path_atoms(
    *,
    layer_sep: float,
    a: float,
    s_array: np.ndarray,
    c: float = 40.0,
) -> list:
    """
    Build zero-strain bilayer cells along the GSFE disregistry path.

    ``s`` follows :func:`get_bilayer_atoms` /
    :func:`get_basis`: ``0`` = AB, ``1/6`` = SP, ``2/3`` = AA, ``1`` = AB.
    """
    return [
        get_bilayer_atoms(float(layer_sep), float(s), a=float(a), c=float(c), sc=1)
        for s in np.asarray(s_array, dtype=float).ravel()
    ]


def evaluate_gsfe_along_disregistry(
    calc_obj,
    *,
    layer_sep: float,
    a: float,
    s_array: np.ndarray,
) -> np.ndarray:
    """
    Evaluate energy/atom (eV) along ``s_array``, shifted so ``E(s=0)=0``.
    """
    atoms_list = build_disregistry_path_atoms(
        layer_sep=layer_sep, a=a, s_array=s_array,
    )
    e_tot = evaluate_model_energies_on_atoms(calc_obj, atoms_list)
    e_per = np.asarray(
        [energy_per_atom_total(atoms, e) for atoms, e in zip(atoms_list, e_tot)],
        dtype=float,
    )
    return e_per - float(e_per[0])


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name.strip())


def is_zero_strain(
    atoms,
    *,
    lat_con: float = DEFAULT_LAT_CON,
    strain_range: float = STRAIN_RANGE,
) -> bool:
    """True when in-plane strains vs ``lat_con`` are within ``strain_range``."""
    cell = atoms.get_cell()
    dx = (float(cell[0, 0]) - float(lat_con)) / float(lat_con)
    dy = (float(np.linalg.norm(cell[1, :2])) - float(lat_con)) / float(lat_con)
    return abs(dx) <= float(strain_range) and abs(dy) <= float(strain_range)


def select_unstrained_gsfe_frames(
    atoms_list: list,
    energies: np.ndarray,
    *,
    layer_sep: float,
    sep_tol: float,
    lat_con: float = DEFAULT_LAT_CON,
    strain_range: float = STRAIN_RANGE,
) -> dict[str, tuple[object, float]]:
    """
    Pick one zero-strain, flat-layer frame per stacking (AB, SP, AA).

    Filters:
    - flat layers (:func:`layers_have_uniform_z`)
    - ``|ε| ≤ strain_range`` vs ``lat_con`` (same convention as ``cell_strains``)
    - ``|d − layer_sep| ≤ sep_tol``

    Preference order: closest interlayer sep, then closest to ``lat_con``,
    then lowest DFT energy.
    """
    candidates: dict[str, list[tuple[object, float, float, float]]] = {
        s: [] for s in STACKINGS_GSFE
    }
    for atoms, e_tot in zip(atoms_list, energies):
        if not layers_have_uniform_z(atoms):
            continue
        if not is_zero_strain(atoms, lat_con=lat_con, strain_range=strain_range):
            continue
        d = float(interlayer_sep(atoms))
        if abs(d - float(layer_sep)) > float(sep_tol):
            continue
        stack = identify_stacking(atoms)
        if stack not in candidates:
            continue
        a_cell = float(atoms.get_cell()[0, 0])
        candidates[stack].append((atoms, float(e_tot), d, a_cell))

    out: dict[str, tuple[object, float]] = {}
    for stack, frames in candidates.items():
        if not frames:
            continue
        frames_sorted = sorted(
            frames,
            key=lambda t: (
                abs(t[2] - float(layer_sep)),
                abs(t[3] - float(lat_con)),
                t[1],
            ),
        )
        atoms, e_tot, _d, _a = frames_sorted[0]
        out[stack] = (atoms, e_tot)
    return out


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot bilayer graphene GSFE at zero strain and fixed layer separation "
            "(DFT rVV10 + optional POD models)."
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
    p.add_argument(
        "--layer-sep",
        type=float,
        default=DEFAULT_LAYER_SEP,
        help=f"Target interlayer separation in Å (default: {DEFAULT_LAYER_SEP}).",
    )
    p.add_argument(
        "--sep-tol",
        type=float,
        default=DEFAULT_SEP_TOL,
        help=f"Allowed |d − layer-sep| in Å (default: {DEFAULT_SEP_TOL}).",
    )
    p.add_argument(
        "--lat-con",
        type=float,
        default=DEFAULT_LAT_CON,
        help=(
            "Equilibrium in-plane lattice constant a (Å) for zero-strain "
            f"selection and POD evaluation (default: {DEFAULT_LAT_CON:g})."
        ),
    )
    p.add_argument(
        "--strain-range",
        type=float,
        default=STRAIN_RANGE,
        help=(
            "Max |ε| vs --lat-con for zero-strain DFT frames "
            f"(default: {STRAIN_RANGE:g})."
        ),
    )
    p.add_argument(
        "--n-disreg",
        type=int,
        default=DEFAULT_N_DISREG,
        help=(
            "Number of disregistry samples along [0, 1] for POD evaluation "
            f"(default: {DEFAULT_N_DISREG})."
        ),
    )
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--dpi", type=int, default=180)
    add_hyperparam_args(p)
    return p.parse_known_args()


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
        print("No --models specified: plotting DFT rVV10 only.", flush=True)

    _chdir_for_dataloader()
    print("\nLoading rVV10 …", flush=True)
    atoms_list, energies, _ = load_energy_data(
        "interlayer", supercells=1, level_of_theory="rVV10",
    )
    energies = np.asarray(energies, dtype=float)
    a_ref = float(args.lat_con)
    print(
        f"  lat_con={a_ref:.5f} Å  strain_range={args.strain_range:g}  "
        f"n_total={len(atoms_list)}",
        flush=True,
    )

    frames = select_unstrained_gsfe_frames(
        atoms_list,
        energies,
        layer_sep=args.layer_sep,
        sep_tol=args.sep_tol,
        lat_con=a_ref,
        strain_range=args.strain_range,
    )
    missing = [s for s in STACKINGS_GSFE if s not in frames]
    if missing:
        raise RuntimeError(
            f"Missing zero-strain flat GSFE frames at d≈{args.layer_sep:g} Å "
            f"(lat_con={a_ref:g}, |ε|≤{args.strain_range:g}) "
            f"for stacking(s): {missing}. "
            f"Found: {sorted(frames)}. "
            f"Try --layer-sep 3.43 (AB/SP/AA available) or a larger --sep-tol."
        )
    for stack in STACKINGS_GSFE:
        atoms, e_tot = frames[stack]
        d = float(interlayer_sep(atoms))
        cell = atoms.get_cell()
        dx = (float(cell[0, 0]) - a_ref) / a_ref
        dy = (float(np.linalg.norm(cell[1, :2])) - a_ref) / a_ref
        print(
            f"  {stack}: d={d:.4f} Å  a={float(cell[0, 0]):.4f} Å  "
            f"ε=({dx:.4f},{dy:.4f})  "
            f"E/atom={energy_per_atom_total(atoms, e_tot):.6g} eV"
            f"  (n_atoms={len(atoms)})",
            flush=True,
        )

    # DFT energies per atom, shifted so AB = 0
    dft_e = {
        s: energy_per_atom_total(frames[s][0], frames[s][1]) for s in STACKINGS_GSFE
    }
    e_ab_dft = dft_e["AB"]
    dft_gsfe = {s: dft_e[s] - e_ab_dft for s in STACKINGS_GSFE}
    print(
        f"  GSFE (eV/atom, AB=0): "
        f"AB={dft_gsfe['AB']:.6g}  SP={dft_gsfe['SP']:.6g}  "
        f"AA={dft_gsfe['AA']:.6g}",
        flush=True,
    )

    s_array = np.linspace(0.0, 1.0, max(int(args.n_disreg), 5))
    scatter_s = np.array([_DISREG_AB, _DISREG_SP, _DISREG_AA, _DISREG_AB2])
    scatter_dft = np.array(
        [dft_gsfe["AB"], dft_gsfe["SP"], dft_gsfe["AA"], dft_gsfe["AB"]],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        scatter_s, scatter_dft,
        marker="x", color=_COLOR_DFT, s=70, zorder=3, label="rVV10",
    )

    legend_handles = [
        mlines.Line2D(
            [], [], color=_COLOR_DFT, marker="x", linestyle="None",
            markersize=8, label="rVV10",
        ),
    ]

    for i_model, model_name in enumerate(models):
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
        print(f"\n--- Model: {model_name} ---", flush=True)
        calc_obj, close_calc = _build_model_calculator(
            model_name, cli_hyperparams or None,
        )
        print(f"  Calculator: {model_label}", flush=True)
        try:
            print(
                f"  Evaluating POD GSFE on {s_array.size} disregistry points "
                f"(a={a_ref:.5f} Å, d={args.layer_sep:g} Å) …",
                flush=True,
            )
            gamma = evaluate_gsfe_along_disregistry(
                calc_obj,
                layer_sep=args.layer_sep,
                a=a_ref,
                s_array=s_array,
            )
            # Milestone markers: nearest grid points to AB / SP / AA
            milestone_s = np.array([_DISREG_AB, _DISREG_SP, _DISREG_AA, _DISREG_AB2])
            milestone_e = np.array(
                [gamma[int(np.argmin(np.abs(s_array - s)))] for s in milestone_s]
            )
            print(
                f"  GSFE (eV/atom, AB=0): "
                f"AB={milestone_e[0]:.6g}  SP={milestone_e[1]:.6g}  "
                f"AA={milestone_e[2]:.6g}",
                flush=True,
            )
            ax.plot(s_array, gamma, color=color, linewidth=2.0, zorder=1)
            ax.scatter(
                milestone_s, milestone_e,
                marker="o", color=color, s=50, zorder=2,
            )
            legend_handles.append(
                mlines.Line2D(
                    [], [], color=color, marker="o", linestyle="-",
                    markersize=8, label=model_label,
                )
            )
        finally:
            close_calc()

    ax.set_xlabel("Disregistry", **CSFONT)
    ax.set_ylabel("GSFE (eV/atom)", **CSFONT)
    ax.set_title(
        rf"Zero strain ($a={a_ref:g}$ Å), $d = {args.layer_sep:g}$ Å",
        fontdict=CSFONT,
    )
    for x in (_DISREG_AB, _DISREG_SP, _DISREG_BA, _DISREG_AA, _DISREG_AB2):
        ax.axvline(x=x, color="black", linestyle="dotted", linewidth=1, alpha=0.5)
    ax.set_xticks(
        [_DISREG_AB, _DISREG_SP, _DISREG_BA, _DISREG_AA, _DISREG_AB2],
        ["AB", "SP", "BA", "AA", "AB"],
    )
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
    ax.legend(handles=legend_handles, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tag = "dft"
    if models:
        tag = _safe_filename(models[0]) if len(models) == 1 else "multi_model"
    out_path = figures_dir / (
        f"{tag}_gsfe_zero_strain_d{args.layer_sep:g}.png"
    )
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Wrote {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
