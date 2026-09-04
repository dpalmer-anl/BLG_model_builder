#!/usr/bin/env python3
"""
Compare ARPES, DFT, and ACSF_hoppings_sk ensemble bands for AB bilayer graphene.

The k-path runs from Γ to K in the first Brillouin zone.  CSV x-coordinates are
**negative distance from K** in Å⁻¹ (x → 0 at K).  Energies are E − E_F (eV).

The AB equilibrium geometry for TB is chosen once by matching AB structures in
``data/hoppings/*.hdf5`` (rebuilt as standard Bernal bilayers) to the DFT bands,
then cached as ``data/arpes_dft_tb_comparison_structure.xyz``.

Examples
--------
::

    python visualizations/plot_arpes_dft_model_comparison.py
    python visualizations/plot_arpes_dft_model_comparison.py \\
        --tb-model ACSF_hoppings_sk_M_9_W_6 --tb-temperature 0.25 \\
        --refit-structure
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from ase import Atoms
from ase.io import read, write

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

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent
REPO_ROOT = UQ_DIR.parent

_uq_dir = str(UQ_DIR)
if _uq_dir not in sys.path:
    sys.path.insert(0, _uq_dir)

from blg_model_builder.ensemble_io import (  # noqa: E402
    DEFAULT_CALIBRATION_METRICS_DIR,
    load_ensemble_pickle,
    resolve_ensemble_pickle,
)
from blg_model_builder.geom_tools import get_bilayer_atoms  # noqa: E402
from blg_model_builder.strain_data import identify_stacking  # noqa: E402
from blg_model_builder.tb_descriptors import get_acsf_sk_hopping_descriptors  # noqa: E402
from blg_model_builder.tb_models import get_acsf_hoppings_sk, get_recip_cell  # noqa: E402

DEFAULT_ARPES_CSV = REPO_ROOT / "data" / "arpes_data_ab_blg.csv"
DEFAULT_DFT_CSV = REPO_ROOT / "data" / "dft_bands_data_ab_blg.csv"
DEFAULT_HOPPINGS_DIR = REPO_ROOT / "data" / "hoppings"
DEFAULT_STRUCTURE_XYZ = REPO_ROOT / "data" / "arpes_dft_tb_comparison_structure.xyz"
DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"
DEFAULT_BEST_FIT_DIR = UQ_DIR / "best_fit_params"

DEFAULT_TB_MODEL = "ACSF_hoppings_sk_M_9_W_6"
DEFAULT_TB_TEMPERATURE = 0.25
DEFAULT_TB_RCUT = 6.0
DEFAULT_N_SAMPLES = 200
DEFAULT_ENSEMBLE_SEED = 0
DEFAULT_DFT_MATCH_TOL_EV = 0.1
DEFAULT_VACUUM_C = 20.0

K_FRAC = np.array([1.0 / 3.0, 2.0 / 3.0, 0.0])
GAMMA_FRAC = np.array([0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# CSV I/O
# ----------------------------------------------------------------------
def load_band_csv(path: Path) -> List[np.ndarray]:
    """
    Parse ``(x, E)`` CSV into separate bands.

    Bands are split when *x* jumps backward (a new curve starting near Γ).
    """
    raw = np.loadtxt(str(path), delimiter=",", ndmin=2)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"Expected N×2 CSV at {path}")
    bands: List[np.ndarray] = []
    current: List[List[float]] = []
    prev_x: Optional[float] = None
    for x, e in raw:
        if prev_x is not None and float(x) < float(prev_x) - 0.05:
            if current:
                bands.append(np.asarray(current, dtype=float))
            current = []
        current.append([float(x), float(e)])
        prev_x = float(x)
    if current:
        bands.append(np.asarray(current, dtype=float))
    return bands


def union_x_grid(bands: Sequence[np.ndarray]) -> np.ndarray:
    xs = np.unique(np.concatenate([b[:, 0] for b in bands]))
    return np.sort(xs)


# ----------------------------------------------------------------------
# Hoppings structures
# ----------------------------------------------------------------------
def load_hoppings_structure(hdf5_path: Path) -> Atoms:
    """Build an ASE ``Atoms`` from one ``data/hoppings`` HDF5 file."""
    with h5py.File(hdf5_path, "r") as hdf:
        lattice_vectors = np.asarray(hdf["lattice_vectors"][:], dtype=float)
        atomic_basis = np.asarray(hdf["atomic_basis"][:], dtype=float)
    atoms = Atoms(
        symbols=["C"] * len(atomic_basis),
        positions=atomic_basis,
        cell=lattice_vectors,
        pbc=[True, True, False],
    )
    z = atomic_basis[:, 2]
    mean_z = float(np.mean(z))
    mol_id = np.ones(len(atoms), dtype=np.int64)
    mol_id[z > mean_z] = 2
    atoms.set_array("mol-id", mol_id)
    atoms.info["source_hdf5"] = hdf5_path.name
    return atoms


def list_hoppings_structures(hoppings_dir: Path) -> List[Tuple[Path, Atoms]]:
    paths = sorted(hoppings_dir.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 structures in {hoppings_dir}")
    return [(p, load_hoppings_structure(p)) for p in paths]


def interlayer_separation(atoms: Atoms) -> float:
    z = atoms.positions[:, 2]
    zmid = float(np.median(z))
    bot = z < zmid
    top = ~bot
    return float(np.mean(z[top]) - np.mean(z[bot]))


def inplane_lattice_a(atoms: Atoms) -> float:
    return float(np.linalg.norm(atoms.cell[0, :2]))


def rebuild_ab_bilayer(atoms_ref: Atoms, *, vacuum_c: float = DEFAULT_VACUUM_C) -> Atoms:
    """
    Standard Bernal AB bilayer with ``(a, d)`` taken from a hoppings HDF5 frame.

    Raw HDF5 coordinates are not used directly because the ACSF SK TB model
    expects the ``get_bilayer_atoms`` geometry.
    """
    a = inplane_lattice_a(atoms_ref)
    d = interlayer_separation(atoms_ref)
    out = get_bilayer_atoms(d, 0.0, a=a, c=float(vacuum_c), sc=1)
    out.info.update(atoms_ref.info)
    out.info["bilayer_a"] = a
    out.info["bilayer_d"] = d
    return out


def list_ab_candidate_bilayers(
    hoppings_dir: Path,
    *,
    vacuum_c: float = DEFAULT_VACUUM_C,
) -> List[Tuple[str, Atoms]]:
    out: List[Tuple[str, Atoms]] = []
    for path, atoms in list_hoppings_structures(hoppings_dir):
        if identify_stacking(atoms) != "AB":
            continue
        rebuilt = rebuild_ab_bilayer(atoms, vacuum_c=vacuum_c)
        out.append((path.name, rebuilt))
    if not out:
        raise RuntimeError(f"No AB stacking structures found in {hoppings_dir}")
    return out


# ----------------------------------------------------------------------
# k-path: Γ → K, x = −|k − K| (Å⁻¹)
# ----------------------------------------------------------------------
def k_cart_from_x_distance(
    x_from_k: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Map negative distance-from-K coordinates to Cartesian k-points."""
    B = get_recip_cell(np.asarray(cell, dtype=float).T)
    k_cart = K_FRAC @ B
    g_cart = GAMMA_FRAC @ B
    direction = g_cart - k_cart
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        raise ValueError("Degenerate Γ–K direction.")
    unit = direction / norm
    dist = -np.asarray(x_from_k, dtype=float)
    return k_cart + dist[:, None] * unit[None, :]


# ----------------------------------------------------------------------
# TB band structure (ACSF SK, CPU)
# ----------------------------------------------------------------------
def build_sk_bands(
    atoms: Atoms,
    params: np.ndarray,
    kvec_cart: np.ndarray,
    M: int,
    W: int,
    r_cut: float,
    *,
    x_from_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """All eigenvalues (eV), with E_F = 0 at the k-point closest to K."""
    descriptors, (pair_i, pair_j, pair_v) = get_acsf_sk_hopping_descriptors(
        atoms, M=M, W=W, r_cut=r_cut,
    )
    hoppings = get_acsf_hoppings_sk(descriptors, params)
    N = len(atoms)
    nocc = N // 2
    evals_list: List[np.ndarray] = []

    for ik, k_cart in enumerate(kvec_cart):
        phases = np.exp(1j * (pair_v @ k_cart))
        hop_vals = hoppings * phases
        H = scipy.sparse.coo_matrix(
            (hop_vals, (pair_i, pair_j)),
            shape=(N, N),
            dtype=np.complex128,
        ).tocsr()
        H = H + H.conj().T
        evals_list.append(np.linalg.eigh(H.toarray())[0])

    evals = np.asarray(evals_list, dtype=float)
    if x_from_k is not None:
        i_k = int(np.argmin(np.abs(np.asarray(x_from_k, dtype=float))))
    else:
        i_k = 0
    fermi = float((evals[i_k, nocc] + evals[i_k, nocc - 1]) / 2.0)
    return evals - fermi


def assign_tb_indices_to_dft_bands(
    tb_evals: np.ndarray,
    dft_bands: Sequence[np.ndarray],
    x_grid: np.ndarray,
) -> List[int]:
    """
    For each DFT band, pick the TB eigenvalue index with minimum RMSE on ``x_grid``.
    """
    n_tb = tb_evals.shape[1]
    used: set[int] = set()
    indices: List[int] = []
    for dft_band in dft_bands[:2]:
        target = np.interp(x_grid, dft_band[:, 0], dft_band[:, 1])
        best_idx = -1
        best_rmse = float("inf")
        for j in range(n_tb):
            if j in used:
                continue
            rmse = float(np.sqrt(np.mean((tb_evals[:, j] - target) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = j
        if best_idx < 0:
            raise RuntimeError("Could not assign a unique TB band index.")
        used.add(best_idx)
        indices.append(best_idx)
    return indices


def extract_assigned_bands(
    tb_evals: np.ndarray,
    tb_indices: Sequence[int],
) -> np.ndarray:
    return tb_evals[:, list(tb_indices)]


def rmse_tb_vs_dft(
    atoms: Atoms,
    params: np.ndarray,
    dft_bands: Sequence[np.ndarray],
    M: int,
    W: int,
    r_cut: float,
    tb_indices: Optional[Sequence[int]] = None,
) -> Tuple[float, List[int]]:
    """RMSE between assigned TB bands and DFT (two valence bands near E_F)."""
    x_grid = union_x_grid(dft_bands)
    kvec = k_cart_from_x_distance(x_grid, atoms.cell.array)
    tb_all = build_sk_bands(
        atoms, params, kvec, M, W, r_cut, x_from_k=x_grid,
    )
    if tb_indices is None:
        tb_indices = assign_tb_indices_to_dft_bands(tb_all, dft_bands, x_grid)
    tb_bands = extract_assigned_bands(tb_all, tb_indices)

    errs: List[float] = []
    for ib, dft_band in enumerate(dft_bands[:2]):
        x_d = dft_band[:, 0]
        e_d = dft_band[:, 1]
        e_tb = np.interp(x_d, x_grid, tb_bands[:, ib])
        errs.append(float(np.sqrt(np.mean((e_tb - e_d) ** 2))))
    return float(np.mean(errs)), list(tb_indices)


def select_best_structure(
    candidates: Sequence[Tuple[str, Atoms]],
    dft_bands: Sequence[np.ndarray],
    params: np.ndarray,
    M: int,
    W: int,
    r_cut: float,
) -> Tuple[Atoms, float, str, List[int]]:
    """Pick the AB bilayer whose TB bands best match DFT."""
    best_atoms: Optional[Atoms] = None
    best_rmse = float("inf")
    best_name = ""
    best_indices: List[int] = []
    for name, atoms in candidates:
        try:
            score, tb_idx = rmse_tb_vs_dft(
                atoms, params, dft_bands, M, W, r_cut,
            )
        except Exception as exc:
            print(f"  skip {name}: {exc}", flush=True)
            continue
        if score < best_rmse:
            best_rmse = score
            best_atoms = atoms
            best_name = name
            best_indices = tb_idx
    if best_atoms is None:
        raise RuntimeError("No candidate structure produced finite TB/DFT comparison.")
    return best_atoms, best_rmse, best_name, best_indices


def load_ensemble_mean_params(model_name: str, ensemble_dir: str, temperature: float,
                              calibration_metrics_dir: str) -> Tuple[np.ndarray, float]:
    pkl_path, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_target="hopping",
    )
    ens_dict = load_ensemble_pickle(pkl_path)
    ensemble = np.asarray(ens_dict["ensemble"]["hopping"], dtype=float)
    return ensemble.mean(axis=0), float(t_used)


def ensure_comparison_structure(
    *,
    structure_xyz: Path,
    hoppings_dir: Path,
    dft_bands: Sequence[np.ndarray],
    model_name: str,
    ensemble_dir: str,
    tb_temperature: float,
    calibration_metrics_dir: str,
    M: int,
    W: int,
    r_cut: float,
    refit: bool,
    vacuum_c: float,
) -> Tuple[Atoms, float, str, List[int]]:
    """Load cached structure or match against ``data/hoppings`` AB candidates."""
    params, _t = load_ensemble_mean_params(
        model_name, ensemble_dir, tb_temperature, calibration_metrics_dir,
    )

    if structure_xyz.is_file() and not refit:
        atoms = read(str(structure_xyz))
        src = str(atoms.info.get("source_hdf5", "cached"))
        tb_idx = atoms.info.get("tb_band_indices")
        if tb_idx is not None:
            tb_indices = [int(i) for i in np.asarray(tb_idx).ravel().tolist()]
        else:
            _, tb_indices = rmse_tb_vs_dft(
                atoms, params, dft_bands, M, W, r_cut,
            )
        rmse = rmse_tb_vs_dft(
            atoms, params, dft_bands, M, W, r_cut, tb_indices=tb_indices,
        )[0]
        print(f"  Loaded cached structure {structure_xyz}  (source {src})", flush=True)
        return atoms, rmse, src, tb_indices

    print(f"  Matching AB structures from {hoppings_dir} …", flush=True)
    candidates = list_ab_candidate_bilayers(hoppings_dir, vacuum_c=vacuum_c)
    atoms, rmse, src, tb_indices = select_best_structure(
        candidates, dft_bands, params, M, W, r_cut,
    )
    atoms.info["source_hdf5"] = src
    atoms.info["dft_match_rmse_eV"] = rmse
    atoms.info["tb_band_indices"] = np.asarray(tb_indices, dtype=int)
    structure_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(str(structure_xyz), atoms, format="extxyz")
    print(
        f"  Wrote {structure_xyz}  (best {src}, RMSE={rmse:.4f} eV, "
        f"TB idx={tb_indices}, d={interlayer_separation(atoms):.4f} Å, "
        f"a={inplane_lattice_a(atoms):.4f} Å)",
        flush=True,
    )
    return atoms, rmse, src, tb_indices


# ----------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------
def _shuffle_ensemble(ensemble: np.ndarray, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(ensemble.shape[0])
    return ensemble[order]


def load_tb_ensemble(
    model_name: str,
    ensemble_dir: str,
    temperature: float,
    calibration_metrics_dir: str,
) -> Tuple[np.ndarray, float]:
    pkl_path, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_target="hopping",
    )
    ens_dict = load_ensemble_pickle(pkl_path)
    ensemble = np.asarray(ens_dict["ensemble"]["hopping"], dtype=float)
    return ensemble, float(t_used)


def ensemble_band_statistics(
    atoms: Atoms,
    ensemble: np.ndarray,
    x_grid: np.ndarray,
    tb_indices: Sequence[int],
    M: int,
    W: int,
    r_cut: float,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean ± std of DFT-matched TB bands over shuffled ensemble draws."""
    kvec = k_cart_from_x_distance(x_grid, atoms.cell.array)
    shuffled = _shuffle_ensemble(ensemble, seed)
    samples: List[np.ndarray] = []
    for theta in shuffled:
        if len(samples) >= n_samples:
            break
        try:
            tb_all = build_sk_bands(
                atoms, theta, kvec, M, W, r_cut, x_from_k=x_grid,
            )
            samples.append(extract_assigned_bands(tb_all, tb_indices))
        except Exception:
            continue
    if not samples:
        raise RuntimeError("No successful ensemble TB band calculations.")
    stacked = np.stack(samples, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ----------------------------------------------------------------------
# Plotting / validation
# ----------------------------------------------------------------------
def plot_comparison(
    *,
    arpes_bands: Sequence[np.ndarray],
    dft_bands: Sequence[np.ndarray],
    tb_mean: np.ndarray,
    tb_std: np.ndarray,
    x_grid: np.ndarray,
    out_path: Path,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    tb_color = "C0"

    for band in arpes_bands:
        ax.scatter(
            band[:, 0], band[:, 1],
            s=18, color="C3", alpha=0.75, linewidths=0,
        )
    for band in dft_bands[:2]:
        ax.scatter(
            band[:, 0], band[:, 1],
            s=22, color="C2", alpha=0.9, linewidths=0,
        )

    std_band = None
    mean_line = None
    for ib in range(tb_mean.shape[1]):
        m = tb_mean[:, ib]
        s = tb_std[:, ib]
        (line,) = ax.plot(
            x_grid, m, "-", color=tb_color, lw=2.0, zorder=3,
        )
        if ib == 0:
            mean_line = line
        band_fill = ax.fill_between(
            x_grid, m - s, m + s,
            color=tb_color, alpha=0.25, zorder=2,
        )
        if ib == 0:
            std_band = band_fill

    ax.axhline(0.0, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.axvline(0.0, color="k", ls=":", lw=1.0, alpha=0.5)
    ax.set_xlabel(r"distance from $K$ (Å$^{-1}$)", fontdict=CSFONT)
    ax.set_ylabel(r"$E - E_F$ (eV)", fontdict=CSFONT)
    ax.grid(True, alpha=0.25)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label="ARPES"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", markersize=8, label="DFT"),
    ]
    if mean_line is not None:
        mean_line.set_label("TB mean")
        legend_handles.append(mean_line)
    if std_band is not None:
        legend_handles.append(
            Patch(facecolor=tb_color, alpha=0.25, edgecolor="none", label=r"TB $\pm 1\sigma$"),
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        prop={"family": CSFONT["fontname"], "size": 15},
        frameon=True,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def mae_on_reference_x(
    model_bands: np.ndarray,
    x_model: np.ndarray,
    ref_bands: Sequence[np.ndarray],
) -> float:
    """Mean absolute error after interpolating ``model_bands`` onto each ref band's x."""
    errs: List[float] = []
    n_pair = min(model_bands.shape[1], len(ref_bands))
    for ib in range(n_pair):
        x_r = ref_bands[ib][:, 0]
        e_r = ref_bands[ib][:, 1]
        e_model = np.interp(x_r, x_model, model_bands[:, ib])
        errs.append(float(np.mean(np.abs(e_model - e_r))))
    return float(np.mean(errs)) if errs else float("nan")


def mae_dft_vs_arpes(
    dft_bands: Sequence[np.ndarray],
    arpes_bands: Sequence[np.ndarray],
) -> float:
    """MAE between DFT and ARPES with DFT linearly interpolated onto ARPES x."""
    errs: List[float] = []
    n_pair = min(len(dft_bands), len(arpes_bands))
    for ib in range(n_pair):
        x_a = arpes_bands[ib][:, 0]
        e_a = arpes_bands[ib][:, 1]
        e_d = np.interp(x_a, dft_bands[ib][:, 0], dft_bands[ib][:, 1])
        errs.append(float(np.mean(np.abs(e_d - e_a))))
    return float(np.mean(errs)) if errs else float("nan")


def max_mean_dft_error(
    tb_mean: np.ndarray,
    dft_bands: Sequence[np.ndarray],
    x_grid: np.ndarray,
) -> float:
    errs: List[float] = []
    for ib, dft_band in enumerate(dft_bands[:2]):
        x_d = dft_band[:, 0]
        e_d = dft_band[:, 1]
        e_tb = np.interp(x_d, x_grid, tb_mean[:, ib])
        errs.append(float(np.max(np.abs(e_tb - e_d))))
    return float(max(errs))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_tb_model(name: str) -> Tuple[int, int, str]:
    m = re.search(r"(?i)acsf[_\-]hoppings[_\-]sk[_\-]m[_\-](\d+)[_\-]w[_\-](\d+)", name)
    if m is None:
        raise ValueError(f"Expected ACSF_hoppings_sk_M_<M>_W_<W>, got {name!r}")
    M, W = int(m.group(1)), int(m.group(2))
    canonical = f"ACSF_hoppings_sk_M_{M}_W_{W}"
    return M, W, canonical


def main() -> None:
    p = argparse.ArgumentParser(
        description="ARPES / DFT / ACSF_hoppings_sk band comparison (AB BLG, Γ–K path).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arpes-csv", type=Path, default=DEFAULT_ARPES_CSV)
    p.add_argument("--dft-csv", type=Path, default=DEFAULT_DFT_CSV)
    p.add_argument("--hoppings-dir", type=Path, default=DEFAULT_HOPPINGS_DIR)
    p.add_argument("--structure-xyz", type=Path, default=DEFAULT_STRUCTURE_XYZ)
    p.add_argument("--refit-structure", action="store_true")
    p.add_argument("--tb-model", default=DEFAULT_TB_MODEL)
    p.add_argument("--tb-temperature", type=float, default=DEFAULT_TB_TEMPERATURE)
    p.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    p.add_argument(
        "--calibration-metrics-dir", default=DEFAULT_CALIBRATION_METRICS_DIR,
    )
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_ENSEMBLE_SEED)
    p.add_argument("--tb-rcut", type=float, default=DEFAULT_TB_RCUT)
    p.add_argument("--vacuum-c", type=float, default=DEFAULT_VACUUM_C)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--dft-match-tol", type=float, default=DEFAULT_DFT_MATCH_TOL_EV,
        help="Warn if max |ensemble mean − DFT| exceeds this (eV).",
    )
    args = p.parse_args()

    os.chdir(UQ_DIR)

    M, W, canonical_tb = _parse_tb_model(args.tb_model)
    model_label = f"{canonical_tb}_tbT{args.tb_temperature:g}"

    arpes_bands = load_band_csv(Path(args.arpes_csv))
    dft_bands = load_band_csv(Path(args.dft_csv))
    x_grid = union_x_grid(dft_bands)
    print(
        f"ARPES: {len(arpes_bands)} band(s); DFT: {len(dft_bands)} band(s); "
        f"{len(x_grid)} k-points",
        flush=True,
    )

    atoms, match_rmse, src, tb_indices = ensure_comparison_structure(
        structure_xyz=Path(args.structure_xyz),
        hoppings_dir=Path(args.hoppings_dir),
        dft_bands=dft_bands,
        model_name=canonical_tb,
        ensemble_dir=args.ensemble_dir,
        tb_temperature=args.tb_temperature,
        calibration_metrics_dir=args.calibration_metrics_dir,
        M=M,
        W=W,
        r_cut=args.tb_rcut,
        refit=args.refit_structure,
        vacuum_c=args.vacuum_c,
    )
    print(
        f"  Structure: d={interlayer_separation(atoms):.4f} Å, "
        f"a={inplane_lattice_a(atoms):.4f} Å, TB band indices={tb_indices}",
        flush=True,
    )

    ensemble, t_used = load_tb_ensemble(
        canonical_tb,
        args.ensemble_dir,
        args.tb_temperature,
        args.calibration_metrics_dir,
    )
    print(
        f"  TB ensemble: {canonical_tb}  T={t_used:g}  "
        f"({ensemble.shape[0]} walkers, target {args.n_samples} samples)",
        flush=True,
    )

    tb_mean, tb_std = ensemble_band_statistics(
        atoms, ensemble, x_grid, tb_indices, M, W, args.tb_rcut,
        args.n_samples, args.seed,
    )
    max_err = max_mean_dft_error(tb_mean, dft_bands, x_grid)
    if max_err > args.dft_match_tol:
        print(
            f"  WARNING: ensemble mean differs from DFT by > {args.dft_match_tol:g} eV "
            f"(max |Δ| = {max_err:.4f} eV)",
            file=sys.stderr,
        )

    mae_tb_dft = mae_on_reference_x(tb_mean, x_grid, dft_bands)
    mae_tb_arpes = mae_on_reference_x(tb_mean, x_grid, arpes_bands)
    mae_dft_arpes = mae_dft_vs_arpes(dft_bands, arpes_bands)

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir
    out_path = figures_dir / f"arpes_dft_{model_label}_comparison.png"
    plot_comparison(
        arpes_bands=arpes_bands,
        dft_bands=dft_bands,
        tb_mean=tb_mean,
        tb_std=tb_std,
        x_grid=x_grid,
        out_path=out_path,
        dpi=args.dpi,
    )

    print(f"\nMAE (eV):", flush=True)
    print(f"  TB mean vs DFT:   {mae_tb_dft:.4f}", flush=True)
    print(f"  TB mean vs ARPES: {mae_tb_arpes:.4f}", flush=True)
    print(f"  DFT vs ARPES:     {mae_dft_arpes:.4f}  (DFT interpolated onto ARPES x)", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
