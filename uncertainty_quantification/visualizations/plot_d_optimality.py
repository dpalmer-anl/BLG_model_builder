#!/usr/bin/env python3
"""
D-optimality / leverage of TBLG environments vs POD and ACSF hyperparameters.

For a linear model ``E = c · B`` the leverage of a new descriptor ``b`` is

    γ(b) = bᵀ (Aᵀ A)⁻¹ b

where the rows of ``A`` are the training descriptor vectors (one per atomic
environment used in the fit) and ``b`` is a TBLG environment descriptor.

Layout matches ``plot_bayes_factor.py`` NLL-vs-two-body figures: x-axis is the
two-body radial count (POD ``n_rad`` / ACSF ``M``); each curve is a fixed
three-body setting (POD ``N_3b = n3r × n3a``; ACSF ``W``).

Example
-------
::

    python visualizations/plot_d_optimality.py --structure flat
    python visualizations/plot_d_optimality.py --structure relaxed --targets acsf
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
sys.path.insert(0, str(UQ_DIR))
sys.path.insert(0, str(HERE))

from plot_descriptor_umap import resolve_tblg_atoms  # noqa: E402

from blg_model_builder.DataLoader import load_data_for_model
from blg_model_builder.pod_model_selection import (
    load_pod_search_results,
    pod_hyperparams_from_row,
    pod_hyperparams_for_index,
)
from blg_model_builder.potentials import ncoeff_from_params
from blg_model_builder.tb_descriptors import get_acsf_sk_hopping_descriptors

CSFONT = {"fontname": "sans-serif", "size": 20}
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": CSFONT["size"],
        "xtick.labelsize": CSFONT["size"],
        "ytick.labelsize": CSFONT["size"],
    }
)

DEFAULT_TWIST = 1.05
DEFAULT_POD_T = 0.1
DEFAULT_R_CUT_TB = 6.0
DEFAULT_POD_INDEX_GEOM = 15
POD_TWO_BODY_RADIAL_GRID = list(range(6, 14, 1))
ACSF_M_GRID = (6, 7, 8, 9, 10, 11, 12, 14, 15)
ACSF_W_GRID = (0, 1, 2, 3, 4, 5, 6)
SCORE_FIELDS = (
    "family",
    "key",
    "rcut",
    "two_body_radial",
    "three_body_radial",
    "three_body_angular",
    "M",
    "W",
    "n_train",
    "n_tblg",
    "n_desc",
    "gamma_mean",
    "gamma_max",
    "gamma_p95",
    "ridge",
)


def _PODLammpsCalculator():
    from blg_model_builder.lammps_interface import PODLammpsCalculator

    return PODLammpsCalculator


def stack_descriptor_blocks(blocks: Sequence) -> np.ndarray:
    rows = [np.asarray(b, dtype=np.float64) for b in blocks if np.asarray(b).size]
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(rows)


def _to_numpy(X) -> np.ndarray:
    if hasattr(X, "get"):
        X = X.get()
    return np.asarray(X, dtype=np.float64)


def finite_rows(X: np.ndarray, *, label: str) -> np.ndarray:
    X = _to_numpy(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")
    keep = np.all(np.isfinite(X), axis=1)
    n_drop = int(np.count_nonzero(~keep))
    if n_drop:
        print(f"  {label}: dropped {n_drop}/{X.shape[0]} non-finite rows", flush=True)
    return X[keep]


def leverage_scores(
    A: np.ndarray,
    B: np.ndarray,
    *,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Return ``γ(b) = bᵀ (Aᵀ A)⁻¹ b`` for each row of ``B``."""
    A = finite_rows(A, label="train A")
    B = finite_rows(B, label="tBLG b")
    if A.size == 0 or B.size == 0:
        raise RuntimeError("Empty design matrix or TBLG descriptors.")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A ncols {A.shape[1]} != b ncols {B.shape[1]}")

    n_col = int(A.shape[1])
    ata = A.T @ A
    tr = float(np.trace(ata))
    lam = float(ridge) + 1e-10 * (tr / max(n_col, 1))
    ata = ata + lam * np.eye(n_col, dtype=np.float64)
    try:
        x = np.linalg.solve(ata, B.T)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(ata, B.T, rcond=None)[0]
    gamma = np.einsum("nd,dn->n", B, x)
    gamma = np.maximum(np.asarray(gamma, dtype=np.float64), 0.0)
    return gamma, lam


def summarize_gamma(gamma: np.ndarray) -> Dict[str, float]:
    g = np.asarray(gamma, dtype=np.float64)
    return {
        "gamma_mean": float(np.mean(g)),
        "gamma_max": float(np.max(g)),
        "gamma_p95": float(np.percentile(g, 95)),
    }


def slice_acsf_sk(
    D: np.ndarray,
    *,
    M: int,
    W: int,
    M_full: int,
    W_full: int,
) -> np.ndarray:
    """Slice an SK ACSF matrix computed at ``(M_full, W_full)`` down to ``(M, W)``."""
    D = _to_numpy(D)
    nfeat_full = int(M_full + M_full * W_full)
    if D.shape[1] != 2 * nfeat_full:
        raise ValueError(
            f"SK ACSF expected {2 * nfeat_full} columns, got {D.shape[1]}"
        )

    def _slice_half(H: np.ndarray) -> np.ndarray:
        two = H[:, :M_full]
        parts = [two[:, :M]]
        if W > 0:
            if W_full < 1:
                raise ValueError("Cannot slice W>0 from a two-body-only matrix.")
            three = H[:, M_full:].reshape(H.shape[0], M_full, W_full)
            parts.append(three[:, :M, :W].reshape(H.shape[0], M * W))
        return np.concatenate(parts, axis=1)

    return np.concatenate(
        [_slice_half(D[:, :nfeat_full]), _slice_half(D[:, nfeat_full:])],
        axis=1,
    )


def load_score_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            out[str(rec["key"])] = rec
    return out


def write_score_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCORE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for rec in rows:
            writer.writerow({k: rec.get(k, "") for k in SCORE_FIELDS})


def _plot_curves(
    groups: Dict[Any, List[Tuple[float, float]]],
    *,
    xlabel: str,
    ylabel: str,
    xticks: Sequence[int],
    legend_label,
    out_path: Path,
    dpi: int,
    log_y: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.5))
    for key in sorted(groups):
        pts = sorted(groups[key], key=lambda t: t[0])
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        ax.plot(x, y, "o-", lw=1.8, markersize=6, label=legend_label(key))
    if xticks:
        ax.set_xticks(list(xticks))
        ax.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    y_all = np.concatenate(
        [np.array([p[1] for p in pts], dtype=float) for pts in groups.values()]
    ) if groups else np.array([], dtype=float)
    if log_y and np.any(y_all > 0):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_pod_dopt(
    records: Sequence[Dict[str, Any]],
    figures_dir: Path,
    *,
    structure: str,
    twist_angle: float,
    metric_key: str,
    ylabel: str,
    dpi: int,
) -> None:
    by_rcut: Dict[Optional[float], List[Dict[str, Any]]] = {}
    for rec in records:
        rcut_raw = rec.get("rcut", "")
        try:
            rcut = float(rcut_raw) if str(rcut_raw).strip() != "" else None
        except (TypeError, ValueError):
            rcut = None
        by_rcut.setdefault(rcut, []).append(rec)

    theta_tag = f"theta{float(twist_angle):g}"
    for rcut, recs in sorted(by_rcut.items(), key=lambda kv: (kv[0] is None, kv[0] or 0.0)):
        groups: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for rec in recs:
            key = (int(rec["three_body_radial"]), int(rec["three_body_angular"]))
            groups.setdefault(key, []).append(
                (float(rec["two_body_radial"]), float(rec[metric_key]))
            )
        family = "POD_energy" if rcut is None else f"POD_energy_rcut{rcut:g}"
        out = (
            figures_dir
            / f"{family}_{metric_key}_vs_two_body_radial_{structure}_{theta_tag}.png"
        )
        _plot_curves(
            groups,
            xlabel=r"$n_{\mathrm{rad}}$ (2-body radial basis functions)",
            ylabel=ylabel,
            xticks=POD_TWO_BODY_RADIAL_GRID,
            legend_label=lambda k: rf"$N_{{\mathrm{{3b}}}}={int(k[0]) * int(k[1])}$",
            out_path=out,
            dpi=dpi,
            log_y=True,
        )


def plot_acsf_dopt(
    records: Sequence[Dict[str, Any]],
    figures_dir: Path,
    *,
    structure: str,
    twist_angle: float,
    metric_key: str,
    ylabel: str,
    dpi: int,
) -> None:
    groups: Dict[int, List[Tuple[float, float]]] = {}
    for rec in records:
        groups.setdefault(int(rec["W"]), []).append(
            (float(rec["M"]), float(rec[metric_key]))
        )
    theta_tag = f"theta{float(twist_angle):g}"
    out = (
        figures_dir
        / f"ACSF_hoppings_sk_{metric_key}_vs_M_{structure}_{theta_tag}.png"
    )
    _plot_curves(
        groups,
        xlabel=r"$M$ (radial basis functions)",
        ylabel=ylabel,
        xticks=list(ACSF_M_GRID),
        legend_label=lambda w: rf"$W={int(w)}$",
        out_path=out,
        dpi=dpi,
        log_y=True,
    )


def load_pod_train_atoms() -> List[Atoms]:
    print("Loading POD_energy training structures …", flush=True)
    x_train, _x_test, _x, _y_train, _y_test, _y = load_data_for_model(
        "POD_energy", supercells=1,
    )
    train_atoms = list(x_train["energy"])
    n_atoms = int(sum(len(a) for a in train_atoms))
    print(
        f"  train structures: {len(train_atoms)}  atoms: {n_atoms}",
        flush=True,
    )
    return train_atoms


def pod_atom_descriptors(
    pod_hp: dict,
    rcut: float,
    pod_hash: str,
    atoms_list: Sequence[Atoms],
    cache_path: Path,
    *,
    label: str,
) -> np.ndarray:
    if cache_path.is_file():
        print(f"  loading cached {label} {cache_path.name}", flush=True)
        return np.asarray(np.load(cache_path)["descriptors"], dtype=np.float64)

    if label == "train":
        umap_hits = sorted(
            (UQ_DIR / "figures" / "umap").glob(
                f"POD_energy_POD_index_*_{pod_hash}/train_pod_atom_descriptors.npz"
            )
        )
        if umap_hits:
            print(f"  reusing UMAP train cache {umap_hits[0]}", flush=True)
            X = np.asarray(np.load(umap_hits[0])["descriptors"], dtype=np.float64)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, descriptors=X)
            return X

    ncoeff = int(ncoeff_from_params(pod_hp))
    calc = _PODLammpsCalculator()(
        pod_hp,
        np.zeros(ncoeff, dtype=float),
        elements=["C"],
        cutoff=float(rcut),
    )
    try:
        if len(atoms_list) == 1:
            print(
                f"  compute pod/atom {label} hash={pod_hash} "
                f"N={len(atoms_list[0])} ncoeff={ncoeff}",
                flush=True,
            )
            X = np.array(
                calc.compute_pod_atom_descriptors(atoms_list[0]),
                dtype=np.float64,
                copy=True,
            )
        else:
            print(
                f"  compute pod/atom {label} hash={pod_hash} "
                f"n_struct={len(atoms_list)} ncoeff={ncoeff}",
                flush=True,
            )
            X = np.array(
                calc.compute_pod_atom_descriptors_batch(atoms_list, verbose=True),
                dtype=np.float64,
                copy=True,
            )
    finally:
        try:
            calc.close()
        except Exception:
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, descriptors=X)
    print(f"  cached {cache_path}  shape={X.shape}", flush=True)
    return X


def collect_pod_scores(
    *,
    tblg: Atoms,
    cache_dir: Path,
    scores_path: Path,
    structure: str,
    twist_angle: float,
    recompute: bool,
) -> List[Dict[str, Any]]:
    df = load_pod_search_results()
    existing = {} if recompute else load_score_rows(scores_path)
    train_atoms: Optional[List[Atoms]] = None
    rows: List[Dict[str, Any]] = []

    for idx, rec in df.iterrows():
        pod_hp, rcut, pod_hash = pod_hyperparams_from_row(rec)
        key = f"pod_{pod_hash}"
        if key in existing and existing[key].get("gamma_mean"):
            rows.append(existing[key])
            print(f"  skip cached score {key}", flush=True)
            continue

        if train_atoms is None:
            train_atoms = load_pod_train_atoms()

        print(
            f"POD_index={int(idx)} hash={pod_hash}  "
            f"n2={pod_hp['twobody_number_radial_basis_functions']} "
            f"n3r={pod_hp['threebody_number_radial_basis_functions']} "
            f"n3a={pod_hp['threebody_angular_degree']} rcut={rcut:g}",
            flush=True,
        )
        A = pod_atom_descriptors(
            pod_hp,
            rcut,
            pod_hash,
            train_atoms,
            cache_dir / f"pod_train_{pod_hash}.npz",
            label="train",
        )
        B = pod_atom_descriptors(
            pod_hp,
            rcut,
            pod_hash,
            [tblg],
            cache_dir / f"pod_tblg_{pod_hash}_{structure}_theta{twist_angle:g}.npz",
            label="tBLG",
        )
        ridge = float(rec.get("regularization", 1e-12))
        gamma, lam = leverage_scores(A, B, ridge=ridge)
        stats = summarize_gamma(gamma)
        row = {
            "family": "POD_energy",
            "key": key,
            "rcut": rcut,
            "two_body_radial": int(pod_hp["twobody_number_radial_basis_functions"]),
            "three_body_radial": int(pod_hp["threebody_number_radial_basis_functions"]),
            "three_body_angular": int(pod_hp["threebody_angular_degree"]),
            "M": "",
            "W": "",
            "n_train": int(A.shape[0]),
            "n_tblg": int(B.shape[0]),
            "n_desc": int(A.shape[1]),
            "ridge": lam,
            **stats,
        }
        rows.append(row)
        write_score_rows(scores_path, rows)
        print(
            f"  γ mean={stats['gamma_mean']:.4e}  "
            f"max={stats['gamma_max']:.4e}  p95={stats['gamma_p95']:.4e}",
            flush=True,
        )
    return rows


def collect_acsf_scores(
    *,
    tblg: Atoms,
    cache_dir: Path,
    scores_path: Path,
    structure: str,
    twist_angle: float,
    r_cut: float,
    recompute: bool,
) -> List[Dict[str, Any]]:
    M_full = int(max(ACSF_M_GRID))
    W_full = int(max(ACSF_W_GRID))
    existing = {} if recompute else load_score_rows(scores_path)
    train_cache = cache_dir / f"acsf_sk_train_M{M_full}_W{W_full}_rcut{r_cut:g}.npz"
    tblg_cache = (
        cache_dir
        / f"acsf_sk_tblg_M{M_full}_W{W_full}_rcut{r_cut:g}_{structure}_theta{twist_angle:g}.npz"
    )

    if train_cache.is_file():
        print(f"Loading cached ACSF_sk train descriptors {train_cache}", flush=True)
        A_full = np.asarray(np.load(train_cache)["descriptors"], dtype=np.float64)
    else:
        print(
            f"Computing ACSF_sk training descriptors (M={M_full}, W={W_full}) …",
            flush=True,
        )
        x_train, _x_test, _x, _y_train, _y_test, _y = load_data_for_model(
            "ACSF_hoppings_sk", M=M_full, W=W_full, r_cut=float(r_cut),
        )
        A_full = stack_descriptor_blocks(x_train["hopping"])
        train_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_cache, descriptors=A_full)
        print(f"  train ACSF_sk: {A_full.shape}  cached {train_cache}", flush=True)

    if tblg_cache.is_file():
        print(f"Loading cached ACSF_sk TBLG descriptors {tblg_cache}", flush=True)
        B_full = np.asarray(np.load(tblg_cache)["descriptors"], dtype=np.float64)
    else:
        print(
            f"Computing ACSF_sk TBLG descriptors "
            f"(N={len(tblg)}, M={M_full}, W={W_full}) …",
            flush=True,
        )
        dsc, _ = get_acsf_sk_hopping_descriptors(
            tblg, M=M_full, W=W_full, r_cut=float(r_cut),
        )
        B_full = _to_numpy(dsc)
        tblg_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tblg_cache, descriptors=B_full)
        print(f"  TBLG ACSF_sk: {B_full.shape}  cached {tblg_cache}", flush=True)

    rows: List[Dict[str, Any]] = []
    for M in ACSF_M_GRID:
        for W in ACSF_W_GRID:
            key = f"acsf_sk_M{int(M)}_W{int(W)}"
            if key in existing and existing[key].get("gamma_mean"):
                rows.append(existing[key])
                continue
            A = slice_acsf_sk(A_full, M=int(M), W=int(W), M_full=M_full, W_full=W_full)
            B = slice_acsf_sk(B_full, M=int(M), W=int(W), M_full=M_full, W_full=W_full)
            gamma, lam = leverage_scores(A, B, ridge=0.0)
            stats = summarize_gamma(gamma)
            row = {
                "family": "ACSF_hoppings_sk",
                "key": key,
                "rcut": r_cut,
                "two_body_radial": int(M),
                "three_body_radial": "",
                "three_body_angular": int(W),
                "M": int(M),
                "W": int(W),
                "n_train": int(A.shape[0]),
                "n_tblg": int(B.shape[0]),
                "n_desc": int(A.shape[1]),
                "ridge": lam,
                **stats,
            }
            rows.append(row)
            print(
                f"  ACSF_sk M={M} W={W}  ncols={A.shape[1]}  "
                f"γ mean={stats['gamma_mean']:.4e}  max={stats['gamma_max']:.4e}",
                flush=True,
            )
    write_score_rows(scores_path, rows)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--structure",
        choices=("flat", "relaxed"),
        required=True,
        help="tBLG geometry used for the new descriptor vectors b.",
    )
    p.add_argument("--twist-angle", type=float, default=DEFAULT_TWIST)
    p.add_argument("--pod-index", type=int, default=DEFAULT_POD_INDEX_GEOM,
                   help="POD_index whose relaxed ensemble supplies the mean tBLG geometry.")
    p.add_argument("--pod-temperature", type=float, default=DEFAULT_POD_T)
    p.add_argument(
        "--targets",
        choices=("both", "pod", "acsf"),
        default="both",
    )
    p.add_argument("--tb-rcut", type=float, default=DEFAULT_R_CUT_TB)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--recompute", action="store_true",
                   help="Ignore cached γ scores (descriptor caches are still reused).")
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=UQ_DIR / "figures" / "d_optimality",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(UQ_DIR)

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    cache_dir = figures_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _pod_hp, _rcut0, pod_hash = pod_hyperparams_for_index(
        int(args.pod_index), require_fit_cache=False,
    )
    tblg = resolve_tblg_atoms(
        structure=args.structure,
        twist_angle=float(args.twist_angle),
        pod_index=int(args.pod_index),
        pod_hash=str(pod_hash),
        pod_temperature=float(args.pod_temperature),
        cache_dir=UQ_DIR / "figures" / "umap",
    )
    print(f"tBLG {args.structure} θ={float(args.twist_angle):g}°  N={len(tblg)}", flush=True)

    metric_specs = (
        ("gamma_mean", r"$\langle\gamma\rangle$"),
        ("gamma_max", r"$\gamma_{\max}$"),
        ("gamma_p95", r"$\gamma_{95}$"),
    )
    theta_tag = f"theta{float(args.twist_angle):g}"

    if args.targets in ("both", "acsf"):
        acsf_scores = figures_dir / f"acsf_sk_scores_{args.structure}_{theta_tag}.csv"
        acsf_rows = collect_acsf_scores(
            tblg=tblg,
            cache_dir=cache_dir,
            scores_path=acsf_scores,
            structure=args.structure,
            twist_angle=float(args.twist_angle),
            r_cut=float(args.tb_rcut),
            recompute=bool(args.recompute),
        )
        for key, ylabel in metric_specs:
            plot_acsf_dopt(
                acsf_rows,
                figures_dir,
                structure=args.structure,
                twist_angle=float(args.twist_angle),
                metric_key=key,
                ylabel=ylabel,
                dpi=int(args.dpi),
            )

    if args.targets in ("both", "pod"):
        pod_scores = figures_dir / f"pod_scores_{args.structure}_{theta_tag}.csv"
        pod_rows = collect_pod_scores(
            tblg=tblg,
            cache_dir=cache_dir,
            scores_path=pod_scores,
            structure=args.structure,
            twist_angle=float(args.twist_angle),
            recompute=bool(args.recompute),
        )
        for key, ylabel in metric_specs:
            plot_pod_dopt(
                pod_rows,
                figures_dir,
                structure=args.structure,
                twist_angle=float(args.twist_angle),
                metric_key=key,
                ylabel=ylabel,
                dpi=int(args.dpi),
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
