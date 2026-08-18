#!/usr/bin/env python3
"""
UMAP of training vs tBLG descriptors for POD energy and ACSF_sk hoppings.

Training descriptors are taken from the same datasets used by
``get_MCMC_inputs`` / ``load_data_for_model``.  The tBLG structure at
``--twist-angle`` is either:

* ``flat`` — commensurate unrelaxed cell from ``build_tblg_atoms``
* ``relaxed`` — atom-wise MIC mean of POD_index ensemble trajectories

POD descriptors are **per atom** (``compute pod/atom``), shape
``(n_atoms, n_desc)``.  Training points are stacked over all training
structures (``Σ n_atoms``).  ACSF_sk descriptors remain per bond.

Example
-------
::

    python visualizations/plot_descriptor_umap.py \\
        --pod-index 15 --M 10 --W 5 --structure flat
    python visualizations/plot_descriptor_umap.py \\
        --pod-index 15 --M 10 --W 5 --structure relaxed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from sklearn.preprocessing import RobustScaler
from umap import UMAP

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
sys.path.insert(0, str(UQ_DIR))

from run_uq_propagation_relaxation import build_tblg_atoms  # noqa: E402

from blg_model_builder.DataLoader import load_data_for_model
from blg_model_builder.pod_model_selection import pod_hyperparams_for_index
from blg_model_builder.potentials import ncoeff_from_params
from blg_model_builder.tb_descriptors import get_acsf_sk_hopping_descriptors


def _PODLammpsCalculator():
    """Lazy import to avoid the potentials ↔ lammps_interface cycle."""
    from blg_model_builder.lammps_interface import PODLammpsCalculator

    return PODLammpsCalculator

CSFONT = {"fontname": "sans-serif", "size": 16}
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": 12,
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)

DEFAULT_TWIST = 1.05
DEFAULT_POD_T = 0.1
DEFAULT_R_CUT_TB = 6.0
DEFAULT_MAX_TRAIN = 20000
DEFAULT_MAX_TBLG = 20000
TRAIN_COLOR = "steelblue"
TBLG_COLOR = "darkorange"
RNG = np.random.default_rng(0)


def last_relaxed_frame(traj_path: Path) -> Atoms:
    frames = read(str(traj_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    for fr in reversed(frames):
        if str(fr.info.get("frame", "")).lower() == "relaxed":
            return fr
    return frames[-1]


def mean_relaxed_structure(traj_dir: Path) -> Atoms:
    """Atom-wise MIC average of relaxed frames (xy wrapped to a reference)."""
    trajs = sorted(p for p in traj_dir.glob("*.traj") if "_FAIL" not in p.name)
    if not trajs:
        raise FileNotFoundError(f"No .traj files in {traj_dir}")

    print(
        f"Averaging {len(trajs)} relaxed structures from {traj_dir} …",
        flush=True,
    )
    ref = last_relaxed_frame(trajs[0])
    cell = np.asarray(ref.cell, dtype=float)
    inv = np.linalg.inv(cell)
    ref_pos = np.asarray(ref.positions, dtype=float)
    acc = np.zeros_like(ref_pos)
    n = 0
    for f in trajs:
        a = last_relaxed_frame(f)
        pos = np.asarray(a.positions, dtype=float)
        if pos.shape != ref_pos.shape:
            print(f"  skip {f.name}: shape {pos.shape}", flush=True)
            continue
        d = pos - ref_pos
        frac = d @ inv
        frac[:, :2] -= np.round(frac[:, :2])
        acc += ref_pos + frac @ cell
        n += 1
    if n == 0:
        raise RuntimeError("No structures averaged.")
    out = ref.copy()
    out.set_positions(acc / float(n))
    print(f"  mean structure: N={len(out)} from {n} samples", flush=True)
    return out


def resolve_tblg_atoms(
    *,
    structure: str,
    twist_angle: float,
    pod_index: int,
    pod_hash: str,
    pod_temperature: float,
    cache_dir: Path,
) -> Atoms:
    if structure == "flat":
        print(
            f"Building flat (unrelaxed) TBLG at θ={twist_angle:g}° …",
            flush=True,
        )
        return build_tblg_atoms(float(twist_angle))

    if structure != "relaxed":
        raise ValueError(f"Unknown structure mode {structure!r}")

    model_dir = (
        UQ_DIR
        / "trajectories"
        / "relaxation"
        / f"POD_energy_POD_index_{int(pod_index)}_{pod_hash}"
        / f"T{float(pod_temperature):g}"
        / f"theta{float(twist_angle):g}deg"
    )
    cache = (
        cache_dir
        / f"POD_energy_POD_index_{int(pod_index)}_{pod_hash}"
        / f"T{float(pod_temperature):g}"
        / f"theta{float(twist_angle):g}_mean_relaxed.xyz"
    )
    if cache.is_file():
        print(f"Loading cached mean relaxed structure {cache}", flush=True)
        return read(str(cache))

    # Reuse the earlier figures/ cache if present.
    legacy = (
        UQ_DIR
        / "figures"
        / f"POD15_T{float(pod_temperature):g}_theta{float(twist_angle):g}_mean_relaxed.xyz"
    )
    if legacy.is_file() and int(pod_index) == 15:
        print(f"Loading legacy mean relaxed structure {legacy}", flush=True)
        return read(str(legacy))

    atoms = mean_relaxed_structure(model_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    atoms.write(str(cache))
    print(f"  cached {cache}", flush=True)
    return atoms


def stack_descriptor_blocks(blocks: Sequence) -> np.ndarray:
    rows = [np.asarray(b, dtype=float) for b in blocks if np.asarray(b).size]
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def subsample_rows(X: np.ndarray, n_max: int, *, rng=RNG) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")
    if X.shape[0] <= n_max or n_max <= 0:
        return X
    ix = rng.choice(X.shape[0], size=int(n_max), replace=False)
    return X[ix]


def _finite_rows(X: np.ndarray, *, label: str, clip: float = 1e6) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")
    keep = np.all(np.isfinite(X), axis=1)
    n_drop = int(np.count_nonzero(~keep))
    if n_drop:
        print(f"  {label}: dropped {n_drop}/{X.shape[0]} non-finite descriptor rows", flush=True)
    X = np.clip(X[keep], -float(clip), float(clip))
    return X


def fit_umap_projection(
    X_train: np.ndarray,
    X_tblg: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    X_train = _finite_rows(X_train, label="train")
    X_tblg = _finite_rows(X_tblg, label="tBLG")
    if X_train.shape[0] == 0 or X_tblg.shape[0] == 0:
        raise RuntimeError("No finite descriptor rows left for UMAP.")

    X_all = np.vstack([X_train, X_tblg])
    scaler = RobustScaler()
    Z = scaler.fit_transform(X_all)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    n_neighbors = int(min(n_neighbors, max(2, Z.shape[0] - 1)))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        init="random",
        random_state=int(random_state),
    )
    emb = reducer.fit_transform(Z)
    n_tr = X_train.shape[0]
    return emb[:n_tr], emb[n_tr:]


def load_acsf_train_descriptors(M: int, W: int, r_cut: float) -> np.ndarray:
    print(
        f"Loading ACSF_hoppings_sk training descriptors (M={M}, W={W}) …",
        flush=True,
    )
    x_train, _x_test, _x, _y_train, _y_test, _y = load_data_for_model(
        "ACSF_hoppings_sk",
        M=int(M),
        W=int(W),
        r_cut=float(r_cut),
    )
    X = stack_descriptor_blocks(x_train["hopping"])
    print(f"  train hopping descriptors: {X.shape}", flush=True)
    return X


def acsf_tblg_descriptors(
    atoms: Atoms,
    *,
    M: int,
    W: int,
    r_cut: float,
) -> np.ndarray:
    print(
        f"Computing ACSF_sk descriptors on TBLG (N={len(atoms)}, M={M}, W={W}) …",
        flush=True,
    )
    dsc, _ = get_acsf_sk_hopping_descriptors(
        atoms, M=int(M), W=int(W), r_cut=float(r_cut),
    )
    X = np.asarray(dsc, dtype=float)
    print(f"  TBLG hopping descriptors: {X.shape}", flush=True)
    return X


def load_pod_train_and_calc(
    pod_index: int,
    *,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, object, str, float]:
    pod_hp, rcut, pod_hash = pod_hyperparams_for_index(
        int(pod_index), require_fit_cache=False,
    )
    ncoeff = int(ncoeff_from_params(pod_hp))
    print(
        f"POD_index={pod_index} hash={pod_hash} rcut={rcut:g} ncoeff={ncoeff}",
        flush=True,
    )

    calc = _PODLammpsCalculator()(
        pod_hp,
        np.zeros(ncoeff, dtype=float),
        elements=["C"],
        cutoff=float(rcut),
    )

    cache_path = None
    if cache_dir is not None:
        cache_path = (
            Path(cache_dir)
            / f"POD_energy_POD_index_{int(pod_index)}_{pod_hash}"
            / "train_pod_atom_descriptors.npz"
        )
        if cache_path.is_file():
            print(f"Loading cached train POD atom descriptors {cache_path}", flush=True)
            X_train = np.asarray(np.load(cache_path)["descriptors"], dtype=float)
            if X_train.ndim != 2 or X_train.shape[1] < 1:
                print(
                    f"  ignoring cache with shape {X_train.shape}",
                    flush=True,
                )
            else:
                print(f"  train POD atom descriptors: {X_train.shape}", flush=True)
                return X_train, calc, pod_hash, float(rcut)

    print("Loading POD_energy training structures …", flush=True)
    x_train, _x_test, _x, _y_train, _y_test, _y = load_data_for_model(
        "POD_energy",
        supercells=1,
    )
    train_atoms = list(x_train["energy"])
    n_train_atoms = int(sum(len(a) for a in train_atoms))
    print(
        f"  train structures: {len(train_atoms)}  "
        f"atoms: {n_train_atoms}",
        flush=True,
    )

    print("Computing per-atom POD descriptors on training set …", flush=True)
    X_train = np.asarray(
        calc.compute_pod_atom_descriptors_batch(train_atoms, verbose=True),
        dtype=float,
    )
    print(f"  train POD atom descriptors: {X_train.shape}", flush=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, descriptors=X_train)
        print(f"  cached {cache_path}", flush=True)
    return X_train, calc, pod_hash, float(rcut)


def pod_tblg_descriptors(calc, atoms: Atoms) -> np.ndarray:
    print(f"Computing per-atom POD descriptors on TBLG (N={len(atoms)}) …", flush=True)
    X = np.array(calc.compute_pod_atom_descriptors(atoms), dtype=np.float64, copy=True)
    print(f"  TBLG POD atom descriptors: {X.shape}", flush=True)
    return X


def plot_umap(
    emb_train: np.ndarray,
    emb_tblg: np.ndarray,
    *,
    out_path: Path,
    title: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(
        emb_train[:, 0],
        emb_train[:, 1],
        s=12,
        alpha=0.55,
        c=TRAIN_COLOR,
        label=f"training (n={emb_train.shape[0]})",
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        emb_tblg[:, 0],
        emb_tblg[:, 1],
        s=28 if emb_tblg.shape[0] > 1 else 80,
        alpha=0.85,
        c=TBLG_COLOR,
        label=f"tBLG (n={emb_tblg.shape[0]})",
        edgecolors="k",
        linewidths=0.4,
        zorder=2,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "UMAP projections of POD / ACSF_sk training descriptors vs tBLG."
        ),
    )
    p.add_argument("--pod-index", type=int, default=15, help="POD CSV row index.")
    p.add_argument("--M", type=int, default=10, help="ACSF_sk radial basis count.")
    p.add_argument("--W", type=int, default=5, help="ACSF_sk angular exponent count.")
    p.add_argument(
        "--structure",
        choices=("flat", "relaxed"),
        required=True,
        help="tBLG geometry: unrelaxed flat cell, or mean POD relaxed structure.",
    )
    p.add_argument(
        "--twist-angle",
        type=float,
        default=DEFAULT_TWIST,
        help=f"Twist angle in degrees (default: {DEFAULT_TWIST:g}).",
    )
    p.add_argument(
        "--pod-temperature",
        type=float,
        default=DEFAULT_POD_T,
        help=(
            "POD ensemble temperature label for relaxed trajectories "
            f"(default: {DEFAULT_POD_T:g})."
        ),
    )
    p.add_argument(
        "--targets",
        choices=("both", "pod", "acsf"),
        default="both",
        help="Which descriptor families to plot (default: both).",
    )
    p.add_argument(
        "--tb-rcut",
        type=float,
        default=DEFAULT_R_CUT_TB,
        help=f"ACSF hopping cutoff in Å (default: {DEFAULT_R_CUT_TB:g}).",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=DEFAULT_MAX_TRAIN,
        help=f"Max training descriptor rows for UMAP (default: {DEFAULT_MAX_TRAIN}).",
    )
    p.add_argument(
        "--max-tblg",
        type=int,
        default=DEFAULT_MAX_TBLG,
        help=f"Max tBLG descriptor rows for UMAP (default: {DEFAULT_MAX_TBLG}).",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=UQ_DIR / "figures" / "umap",
        help="Root output directory (default: figures/umap).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(UQ_DIR)

    figures_root = Path(args.figures_dir)
    if not figures_root.is_absolute():
        figures_root = UQ_DIR / figures_root

    _pod_hp, _rcut0, pod_hash = pod_hyperparams_for_index(
        int(args.pod_index), require_fit_cache=False,
    )

    tblg = resolve_tblg_atoms(
        structure=args.structure,
        twist_angle=float(args.twist_angle),
        pod_index=int(args.pod_index),
        pod_hash=str(pod_hash),
        pod_temperature=float(args.pod_temperature),
        cache_dir=figures_root,
    )
    struct_tag = args.structure
    theta_tag = f"theta{float(args.twist_angle):g}"

    if args.targets in ("both", "pod"):
        X_train, calc, pod_hash_run, _ = load_pod_train_and_calc(
            int(args.pod_index), cache_dir=figures_root,
        )
        if pod_hash_run != pod_hash:
            print(
                f"Warning: POD hash mismatch {pod_hash_run} vs {pod_hash}",
                flush=True,
            )
        X_tblg = pod_tblg_descriptors(calc, tblg)
        try:
            calc.close()
        except Exception:
            pass
        X_train_s = subsample_rows(X_train, int(args.max_train))
        X_tblg_s = subsample_rows(X_tblg, int(args.max_tblg))
        emb_tr, emb_tb = fit_umap_projection(X_train_s, X_tblg_s)
        out_dir = (
            figures_root
            / f"POD_energy_POD_index_{int(args.pod_index)}_{pod_hash}"
        )
        out_path = out_dir / f"umap_{struct_tag}_{theta_tag}.png"
        plot_umap(
            emb_tr,
            emb_tb,
            out_path=out_path,
            title=(
                f"POD_index {int(args.pod_index)} descriptors\n"
                f"tBLG {struct_tag}, $\\theta={float(args.twist_angle):g}^\\circ$"
            ),
            dpi=int(args.dpi),
        )

    if args.targets in ("both", "acsf"):
        X_train = load_acsf_train_descriptors(
            int(args.M), int(args.W), float(args.tb_rcut),
        )
        X_tblg = acsf_tblg_descriptors(
            tblg, M=int(args.M), W=int(args.W), r_cut=float(args.tb_rcut),
        )
        X_train_s = subsample_rows(X_train, int(args.max_train))
        X_tblg_s = subsample_rows(X_tblg, int(args.max_tblg))
        print(
            f"UMAP subsample: train {X_train_s.shape[0]}, "
            f"tBLG {X_tblg_s.shape[0]}",
            flush=True,
        )
        emb_tr, emb_tb = fit_umap_projection(X_train_s, X_tblg_s)
        out_dir = (
            figures_root
            / f"ACSF_hoppings_sk_M_{int(args.M)}_W_{int(args.W)}"
        )
        out_path = out_dir / f"umap_{struct_tag}_{theta_tag}.png"
        plot_umap(
            emb_tr,
            emb_tb,
            out_path=out_path,
            title=(
                f"ACSF_hoppings_sk M={int(args.M)}, W={int(args.W)}\n"
                f"tBLG {struct_tag}, $\\theta={float(args.twist_angle):g}^\\circ$"
            ),
            dpi=int(args.dpi),
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
