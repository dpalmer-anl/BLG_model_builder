#!/usr/bin/env python3
"""
UMAP / PCA / Mahalanobis plots of training vs tBLG descriptors (POD, ACSF_sk).

Training descriptors come from the same datasets as ``get_MCMC_inputs``.
The tBLG structure at ``--twist-angle`` is either ``flat`` (unrelaxed) or
``relaxed`` (mean POD ensemble trajectories).

Reference graphite and diamond structures (100 by default) are overlaid on
UMAP scatter plots only; PCA and Mahalanobis plots use training + tBLG.

POD descriptors are **per atom**; ACSF_sk remain per bond.

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
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read
from sklearn.decomposition import PCA
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
DEFAULT_N_REF_FRAMES = 100
DEFAULT_STRAIN_MAX = 0.01
DEFAULT_GRAPHITE_D0 = 3.35
DEFAULT_GRAPHITE_A = 2.4694
DEFAULT_DIAMOND_A0 = 3.567
DEFAULT_MAHALANOBIS_RIDGE = 1e-6
TRAIN_COLOR = "steelblue"
TBLG_COLOR = "darkorange"
GRAPHITE_COLOR = "forestgreen"
DIAMOND_COLOR = "crimson"
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


def reference_strain_combinations(
    n_frames: int = DEFAULT_N_REF_FRAMES,
    strain_max: float = DEFAULT_STRAIN_MAX,
) -> list[tuple[float, float, float]]:
    """
    ``(e1, e2, e3)`` normal strains along the three cell lattice vectors.

    Builds an ``n1 × n2 × n3`` grid with ``n1 * n2 * n3 == n_frames`` (default
    100 = 10 × 10 × 1: in-plane ``a₁`` / ``a₂`` strains, ``a₃`` unstrained).
    """
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    # Prefer a 2-D in-plane grid when n is a perfect square (e.g. 100 = 10×10).
    n_side = int(round(np.sqrt(n_frames)))
    if n_side * n_side == n_frames:
        n1 = n2 = n_side
        n3 = 1
    else:
        # General factorization n1 × n2 × n3 ≈ n_frames (e.g. 10×5×2).
        n1 = int(round(n_frames ** (1.0 / 3.0)))
        n1 = max(2, n1)
        n2 = int(round(np.sqrt(n_frames / n1)))
        n2 = max(2, n2)
        n3 = max(1, int(round(n_frames / (n1 * n2))))
        while n1 * n2 * n3 < n_frames:
            n2 += 1
        while n1 * n2 * n3 > n_frames and n3 > 1:
            n3 -= 1

    e1_vals = np.linspace(-float(strain_max), float(strain_max), n1)
    e2_vals = np.linspace(-float(strain_max), float(strain_max), n2)
    if n3 <= 1:
        e3_vals = np.array([0.0])
    else:
        e3_vals = np.linspace(
            -0.5 * float(strain_max), 0.5 * float(strain_max), n3,
        )

    combos: list[tuple[float, float, float]] = []
    for e1 in e1_vals:
        for e2 in e2_vals:
            for e3 in e3_vals:
                combos.append((float(e1), float(e2), float(e3)))
                if len(combos) >= n_frames:
                    return combos[:n_frames]
    return combos[:n_frames]


def apply_cell_vector_strain(
    atoms: Atoms,
    e1: float,
    e2: float,
    e3: float,
) -> Atoms:
    """Apply normal strains ``e1, e2, e3`` along the three cell lattice vectors."""
    out = atoms.copy()
    F = np.diag([1.0 + float(e1), 1.0 + float(e2), 1.0 + float(e3)])
    out.set_cell(out.cell[:] @ F.T, scale_atoms=True)
    return out


def build_graphite_reference_frames(
    *,
    d0: float = DEFAULT_GRAPHITE_D0,
    a: float = DEFAULT_GRAPHITE_A,
    n_frames: int = DEFAULT_N_REF_FRAMES,
    strain_max: float = DEFAULT_STRAIN_MAX,
) -> list[Atoms]:
    """Bernal graphite unit cell with strains along ``a₁``, ``a₂``, ``c``."""
    from blg_model_builder.geom_tools import get_bilayer_atoms

    combos = reference_strain_combinations(n_frames, strain_max)
    frames: list[Atoms] = []
    for e1, e2, e3 in combos:
        atoms = get_bilayer_atoms(float(d0), 0.0, a=float(a), c=2.0 * float(d0), sc=1)
        atoms = apply_cell_vector_strain(atoms, e1, e2, e3)
        atoms.info["reference"] = "graphite"
        atoms.info["strain"] = (float(e1), float(e2), float(e3))
        frames.append(atoms)
    print(
        f"Built {len(frames)} graphite reference frames "
        f"(d0={d0:g} Å, |e|≤{strain_max:g}, grid along cell vectors)",
        flush=True,
    )
    return frames


def build_diamond_reference_frames(
    *,
    a0: float = DEFAULT_DIAMOND_A0,
    n_frames: int = DEFAULT_N_REF_FRAMES,
    strain_max: float = DEFAULT_STRAIN_MAX,
) -> list[Atoms]:
    """Diamond cubic unit cell with strains along ``a₁``, ``a₂``, ``a₃``."""
    combos = reference_strain_combinations(n_frames, strain_max)
    frames: list[Atoms] = []
    for e1, e2, e3 in combos:
        atoms = bulk("C", crystalstructure="diamond", a=float(a0), cubic=True)
        atoms = apply_cell_vector_strain(atoms, e1, e2, e3)
        atoms.info["reference"] = "diamond"
        atoms.info["strain"] = (float(e1), float(e2), float(e3))
        frames.append(atoms)
    print(
        f"Built {len(frames)} diamond reference frames "
        f"(a0={a0:g} Å, |e|≤{strain_max:g}, grid along cell vectors)",
        flush=True,
    )
    return frames


def pod_atoms_list_descriptors(calc, atoms_list: Sequence[Atoms], *, label: str) -> np.ndarray:
    print(f"Computing per-atom POD descriptors on {label} …", flush=True)
    chunks = [
        np.asarray(calc.compute_pod_atom_descriptors(a), dtype=np.float64)
        for a in atoms_list
    ]
    X = np.vstack(chunks)
    print(f"  {label} POD atom descriptors: {X.shape}", flush=True)
    return X


def acsf_atoms_list_descriptors(
    atoms_list: Sequence[Atoms],
    *,
    M: int,
    W: int,
    r_cut: float,
    label: str,
) -> np.ndarray:
    print(f"Computing ACSF_sk descriptors on {label} …", flush=True)
    chunks = []
    for atoms in atoms_list:
        dsc, _ = get_acsf_sk_hopping_descriptors(
            atoms, M=int(M), W=int(W), r_cut=float(r_cut),
        )
        chunks.append(np.asarray(dsc, dtype=float))
    X = np.vstack(chunks)
    print(f"  {label} ACSF_sk descriptors: {X.shape}", flush=True)
    return X


def _scale_blocks(
    X_train: np.ndarray,
    blocks: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], RobustScaler]:
    """Robust-scale training and query blocks using training fit statistics."""
    X_train = _finite_rows(X_train, label="train")
    names = list(blocks.keys())
    X_other = [_finite_rows(blocks[k], label=k) for k in names]
    scaler = RobustScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_other = {
        name: scaler.transform(X)
        for name, X in zip(names, X_other)
    }
    Z_train = np.nan_to_num(Z_train, nan=0.0, posinf=0.0, neginf=0.0)
    for name in Z_other:
        Z_other[name] = np.nan_to_num(
            Z_other[name], nan=0.0, posinf=0.0, neginf=0.0,
        )
    return Z_train, Z_other, scaler


def fit_pca_projection(
    X_train: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    *,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """PCA fit on training; transform training and overlay blocks."""
    del random_state  # sklearn PCA is deterministic for fixed data.
    Z_train, Z_other, _ = _scale_blocks(X_train, blocks)
    if Z_train.shape[0] < 2:
        raise RuntimeError("Need at least two training rows for PCA.")
    pca = PCA(n_components=2)
    emb_train = pca.fit_transform(Z_train)
    emb_other = {name: pca.transform(Z) for name, Z in Z_other.items()}
    print(
        f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}",
        flush=True,
    )
    return emb_train, emb_other


def fit_umap_projection_blocks(
    X_train: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """UMAP fit on training + overlays jointly (same as legacy train+tBLG)."""
    Z_train, Z_other, _ = _scale_blocks(X_train, blocks)
    if Z_train.shape[0] == 0:
        raise RuntimeError("No finite training rows left for UMAP.")
    names = list(Z_other.keys())
    Z_stack = [Z_train] + [Z_other[k] for k in names]
    Z_all = np.vstack(Z_stack)
    n_neighbors = int(min(n_neighbors, max(2, Z_all.shape[0] - 1)))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        init="random",
        random_state=int(random_state),
    )
    emb_all = reducer.fit_transform(Z_all)
    n_tr = Z_train.shape[0]
    emb_train = emb_all[:n_tr]
    emb_other: Dict[str, np.ndarray] = {}
    start = n_tr
    for name in names:
        n = Z_other[name].shape[0]
        emb_other[name] = emb_all[start : start + n]
        start += n
    return emb_train, emb_other


def fit_umap_projection(
    X_train: np.ndarray,
    X_tblg: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper: train + single tBLG block."""
    emb_tr, emb_map = fit_umap_projection_blocks(
        X_train,
        {"tBLG": X_tblg},
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return emb_tr, emb_map["tBLG"]


def fit_mahalanobis_model(
    X_train: np.ndarray,
    *,
    ridge: float = DEFAULT_MAHALANOBIS_RIDGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, cov_inv)`` from training descriptor rows."""
    X = _finite_rows(X_train, label="train (Mahalanobis fit)")
    if X.shape[0] < 2:
        raise RuntimeError("Need at least two training rows for Mahalanobis.")
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov += float(ridge) * np.eye(cov.shape[0], dtype=float)
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv


def mahalanobis_distances(
    X: np.ndarray,
    mu: np.ndarray,
    cov_inv: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Per-row Mahalanobis distance to the training distribution."""
    X = _finite_rows(X, label=label)
    if X.shape[0] == 0:
        return np.empty(0, dtype=float)
    d = X - mu
    return np.sqrt(np.clip(np.sum(d @ cov_inv * d, axis=1), 0.0, None))


def plot_descriptor_projection(
    emb_train: np.ndarray,
    emb_overlays: Mapping[str, np.ndarray],
    *,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    dpi: int = 150,
) -> None:
    """Scatter training + labelled overlay groups in a 2-D embedding."""
    overlay_style = {
        "tBLG": dict(color=TBLG_COLOR, s=28, alpha=0.85, edgecolors="k", linewidths=0.4, zorder=3),
        "graphite": dict(color=GRAPHITE_COLOR, s=36, alpha=0.9, edgecolors="k", linewidths=0.35, zorder=4),
        "diamond": dict(color=DIAMOND_COLOR, s=36, alpha=0.9, edgecolors="k", linewidths=0.35, zorder=5),
    }
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(
        emb_train[:, 0],
        emb_train[:, 1],
        s=12,
        alpha=0.55,
        c=TRAIN_COLOR,
        label="training",
        edgecolors="none",
        zorder=1,
    )
    for name, emb in emb_overlays.items():
        if emb.shape[0] == 0:
            continue
        style = overlay_style.get(
            name,
            dict(color="gray", s=24, alpha=0.85, edgecolors="k", linewidths=0.3, zorder=2),
        )
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            label=name,
            **style,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_umap(
    emb_train: np.ndarray,
    emb_tblg: np.ndarray,
    *,
    out_path: Path,
    dpi: int = 150,
    emb_graphite: Optional[np.ndarray] = None,
    emb_diamond: Optional[np.ndarray] = None,
) -> None:
    overlays: Dict[str, np.ndarray] = {"tBLG": emb_tblg}
    if emb_graphite is not None and emb_graphite.size:
        overlays["graphite"] = emb_graphite
    if emb_diamond is not None and emb_diamond.size:
        overlays["diamond"] = emb_diamond
    plot_descriptor_projection(
        emb_train,
        overlays,
        out_path=out_path,
        xlabel="UMAP-1",
        ylabel="UMAP-2",
        dpi=dpi,
    )


def plot_pca(
    emb_train: np.ndarray,
    emb_overlays: Mapping[str, np.ndarray],
    *,
    out_path: Path,
    dpi: int = 150,
) -> None:
    plot_descriptor_projection(
        emb_train,
        emb_overlays,
        out_path=out_path,
        xlabel="PC-1",
        ylabel="PC-2",
        dpi=dpi,
    )


def _hist_on_axis(
    ax,
    distances: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    x_lo: float,
    x_hi: float,
    bins: int,
    colors: Mapping[str, str],
) -> None:
    edges = np.linspace(float(x_lo), float(x_hi), int(bins) + 1)
    for name in names:
        if name not in distances:
            continue
        d = np.asarray(distances[name], dtype=float).ravel()
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        ax.hist(
            d,
            bins=edges,
            histtype="stepfilled",
            density=True,
            alpha=0.35,
            color=colors.get(name, "gray"),
            edgecolor=colors.get(name, "gray"),
            linewidth=0.8,
            label=name,
        )


def _mark_broken_xaxis(ax_left, ax_right, *, d: float = 0.012) -> None:
    """Diagonal tick marks indicating a broken x-axis between two panels."""
    kw = dict(color="k", clip_on=False, linewidth=1.0, transform=ax_left.transAxes)
    ax_left.plot((1.0 - d, 1.0 + d), (-d, +d), **kw)
    ax_left.plot((1.0 - d, 1.0 + d), (1.0 - d, 1.0 + d), **kw)
    kw["transform"] = ax_right.transAxes
    ax_right.plot((-d, +d), (-d, +d), **kw)
    ax_right.plot((-d, +d), (1.0 - d, 1.0 + d), **kw)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False)


def _mahalanobis_broken_axis_ranges(
    distances: Mapping[str, np.ndarray],
    *,
    main_names: Sequence[str] = ("training", "tBLG", "graphite"),
    main_pct: float = 99.5,
    diamond_pct: float = 99.5,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Return ``((x_lo_main, x_hi_main), (x_lo_diamond, x_hi_diamond))`` when a
    broken axis helps; otherwise ``None``.
    """
    d_diamond = np.asarray(distances.get("diamond", []), dtype=float).ravel()
    d_diamond = d_diamond[np.isfinite(d_diamond)]
    if d_diamond.size == 0:
        return None

    main_chunks = []
    for name in main_names:
        if name not in distances:
            continue
        d = np.asarray(distances[name], dtype=float).ravel()
        d = d[np.isfinite(d)]
        if d.size:
            main_chunks.append(d)
    if not main_chunks:
        return None

    main_all = np.concatenate(main_chunks)
    hi_main = float(np.percentile(main_all, main_pct))
    lo_diamond = float(np.min(d_diamond))
    if lo_diamond <= hi_main * 1.02:
        return None

    pad_main = max(0.02 * hi_main, 1e-6)
    pad_diamond = max(0.02 * float(np.ptp(d_diamond)), 1e-6)
    main_range = (0.0, hi_main + pad_main)
    diamond_range = (
        max(0.0, lo_diamond - pad_diamond),
        float(np.percentile(d_diamond, diamond_pct)) + pad_diamond,
    )
    return main_range, diamond_range


def plot_mahalanobis_histogram(
    distances: Mapping[str, np.ndarray],
    *,
    out_path: Path,
    dpi: int = 150,
    bins: int = 40,
    diamond_bins: int = 24,
) -> None:
    """Overlaid density histograms of Mahalanobis distance (each integrates to 1)."""
    colors = {
        "training": TRAIN_COLOR,
        "tBLG": TBLG_COLOR,
        "graphite": GRAPHITE_COLOR,
        "diamond": DIAMOND_COLOR,
    }
    main_names = ("training", "tBLG", "graphite")
    broken = _mahalanobis_broken_axis_ranges(distances)

    if broken is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        all_finite = np.concatenate(
            [np.asarray(d, dtype=float).ravel() for d in distances.values() if np.size(d)],
        )
        all_finite = all_finite[np.isfinite(all_finite)]
        if all_finite.size == 0:
            raise RuntimeError("No finite Mahalanobis distances to plot.")
        hi = float(np.percentile(all_finite, 99.5))
        _hist_on_axis(
            ax,
            distances,
            distances.keys(),
            x_lo=0.0,
            x_hi=hi,
            bins=bins,
            colors=colors,
        )
        ax.set_xlim(0.0, hi)
        ax.set_xlabel("Mahalanobis distance")
        ax.set_ylabel("Probability density")
        ax.legend(frameon=False, loc="best", fontsize=11)
        ax.grid(True, alpha=0.25)
    else:
        (x_lo_main, x_hi_main), (x_lo_diamond, x_hi_diamond) = broken
        fig, (ax_main, ax_diamond) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(8.5, 5.5),
            gridspec_kw={"width_ratios": [3.2, 1.0], "wspace": 0.08},
        )
        _hist_on_axis(
            ax_main,
            distances,
            main_names,
            x_lo=x_lo_main,
            x_hi=x_hi_main,
            bins=bins,
            colors=colors,
        )
        _hist_on_axis(
            ax_diamond,
            distances,
            ("diamond",),
            x_lo=x_lo_diamond,
            x_hi=x_hi_diamond,
            bins=diamond_bins,
            colors=colors,
        )
        ax_main.set_xlim(x_lo_main, x_hi_main)
        ax_diamond.set_xlim(x_lo_diamond, x_hi_diamond)
        _mark_broken_xaxis(ax_main, ax_diamond)
        ax_main.set_ylabel("Probability density")
        h_main, l_main = ax_main.get_legend_handles_labels()
        h_dia, l_dia = ax_diamond.get_legend_handles_labels()
        ax_main.legend(
            h_main + h_dia,
            l_main + l_dia,
            frameon=False,
            loc="best",
            fontsize=11,
        )
        ax_main.grid(True, alpha=0.25)
        ax_diamond.grid(True, alpha=0.25)
        fig.supxlabel("Mahalanobis distance", fontsize=CSFONT["size"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def run_descriptor_plots(
    *,
    family_label: str,
    out_dir: Path,
    struct_tag: str,
    theta_tag: str,
    X_train: np.ndarray,
    X_tblg: np.ndarray,
    X_graphite: np.ndarray,
    X_diamond: np.ndarray,
    max_train: int,
    max_tblg: int,
    dpi: int,
) -> None:
    """UMAP, PCA, and Mahalanobis plots for one descriptor family."""
    X_train_s = subsample_rows(X_train, int(max_train))
    X_tblg_s = subsample_rows(X_tblg, int(max_tblg))
    overlays_umap = {
        "tBLG": X_tblg_s,
        "graphite": X_graphite,
        "diamond": X_diamond,
    }
    overlays_tblg_only = {"tBLG": X_tblg_s}
    print(
        f"{family_label} subsample: train {X_train_s.shape[0]}, "
        f"tBLG {X_tblg_s.shape[0]}, graphite {X_graphite.shape[0]}, "
        f"diamond {X_diamond.shape[0]}",
        flush=True,
    )

    emb_tr_umap, emb_ov_umap = fit_umap_projection_blocks(X_train_s, overlays_umap)
    plot_umap(
        emb_tr_umap,
        emb_ov_umap["tBLG"],
        emb_graphite=emb_ov_umap.get("graphite"),
        emb_diamond=emb_ov_umap.get("diamond"),
        out_path=out_dir / f"umap_{struct_tag}_{theta_tag}.png",
        dpi=int(dpi),
    )

    emb_tr_pca, emb_ov_pca = fit_pca_projection(X_train_s, overlays_tblg_only)
    plot_pca(
        emb_tr_pca,
        emb_ov_pca,
        out_path=out_dir / f"pca_{struct_tag}_{theta_tag}.png",
        dpi=int(dpi),
    )

    mu, cov_inv = fit_mahalanobis_model(X_train)
    dist_train = mahalanobis_distances(
        subsample_rows(X_train, min(int(max_train), X_train.shape[0])),
        mu,
        cov_inv,
        label="training (Mahalanobis)",
    )
    dist_tblg = mahalanobis_distances(X_tblg, mu, cov_inv, label="tBLG (Mahalanobis)")
    plot_mahalanobis_histogram(
        {
            "training": dist_train,
            "tBLG": dist_tblg,
        },
        out_path=out_dir / f"mahalanobis_{struct_tag}_{theta_tag}.png",
        dpi=int(dpi),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "UMAP / PCA / Mahalanobis plots of POD and ACSF_sk training "
            "descriptors vs tBLG (graphite/diamond references on UMAP only)."
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
    p.add_argument(
        "--n-ref-frames",
        type=int,
        default=DEFAULT_N_REF_FRAMES,
        help=(
            "Number of graphite / diamond reference structures "
            f"(default: {DEFAULT_N_REF_FRAMES}). Strains combine along cell "
            "lattice vectors on an n1×n2×n3 grid."
        ),
    )
    p.add_argument(
        "--ref-strain-max",
        type=float,
        default=DEFAULT_STRAIN_MAX,
        help=f"Max normal strain magnitude per lattice vector (default: {DEFAULT_STRAIN_MAX}).",
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

    graphite_frames = build_graphite_reference_frames(
        n_frames=int(args.n_ref_frames),
        strain_max=float(args.ref_strain_max),
    )
    diamond_frames = build_diamond_reference_frames(
        n_frames=int(args.n_ref_frames),
        strain_max=float(args.ref_strain_max),
    )

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
        X_graphite = pod_atoms_list_descriptors(
            calc, graphite_frames, label="graphite references",
        )
        X_diamond = pod_atoms_list_descriptors(
            calc, diamond_frames, label="diamond references",
        )
        try:
            calc.close()
        except Exception:
            pass
        out_dir = (
            figures_root
            / f"POD_energy_POD_index_{int(args.pod_index)}_{pod_hash}"
        )
        run_descriptor_plots(
            family_label=f"POD_index {int(args.pod_index)}",
            out_dir=out_dir,
            struct_tag=struct_tag,
            theta_tag=theta_tag,
            X_train=X_train,
            X_tblg=X_tblg,
            X_graphite=X_graphite,
            X_diamond=X_diamond,
            max_train=int(args.max_train),
            max_tblg=int(args.max_tblg),
            dpi=int(args.dpi),
        )

    if args.targets in ("both", "acsf"):
        X_train = load_acsf_train_descriptors(
            int(args.M), int(args.W), float(args.tb_rcut),
        )
        X_tblg = acsf_tblg_descriptors(
            tblg, M=int(args.M), W=int(args.W), r_cut=float(args.tb_rcut),
        )
        X_graphite = acsf_atoms_list_descriptors(
            graphite_frames,
            M=int(args.M),
            W=int(args.W),
            r_cut=float(args.tb_rcut),
            label="graphite references",
        )
        X_diamond = acsf_atoms_list_descriptors(
            diamond_frames,
            M=int(args.M),
            W=int(args.W),
            r_cut=float(args.tb_rcut),
            label="diamond references",
        )
        out_dir = (
            figures_root
            / f"ACSF_hoppings_sk_M_{int(args.M)}_W_{int(args.W)}"
        )
        run_descriptor_plots(
            family_label=(
                f"ACSF_hoppings_sk M={int(args.M)}, W={int(args.W)}"
            ),
            out_dir=out_dir,
            struct_tag=struct_tag,
            theta_tag=theta_tag,
            X_train=X_train,
            X_tblg=X_tblg,
            X_graphite=X_graphite,
            X_diamond=X_diamond,
            max_train=int(args.max_train),
            max_tblg=int(args.max_tblg),
            dpi=int(args.dpi),
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
