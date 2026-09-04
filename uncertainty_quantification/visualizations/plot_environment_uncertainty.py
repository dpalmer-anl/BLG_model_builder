#!/usr/bin/env python3
"""
Scatter plots of ensemble uncertainty vs structure deformation categories.

For POD energy (rVV10 bilayer training set) and ACSF_hoppings_sk hopping models,
the first figure has four panels:

1. in-plane strain magnitude vs stacking
2. disorder vs stacking
3. interlayer separation vs stacking
4. interlayer separation vs in-plane strain magnitude

A second figure plots uncertainty vs each deformation dimension (stacking, strain
magnitude, layer separation, disorder) with a linear fit and annotated
``y = a + b x`` coefficients plus ``χ² = Σ (y_i - y_fit,i)²``.

Point color is the ensemble uncertainty:

* **POD** — for each draw, total energy divided by ``N_atoms``; then ``std`` over
  draws (eV/atom)
* **Hopping** — mean bond-wise standard deviation of predicted hoppings (eV)

POD primitive strain cells (4 atoms, flat layers) are separated from disorder
supercells (MD / random perturbations, ``len(atoms) > 4`` or buckled layers).
Hopping structures are classified as systematic (grid strains/shifts) vs random
(deformations off the ``{-1%, 0, +1%}`` strain grid and/or buckled layers).

Examples
--------
::

    cd uncertainty_quantification
    python visualizations/plot_environment_uncertainty.py
    python visualizations/plot_environment_uncertainty.py \\
        --pod-model POD_energy_POD_index_15_8bb97b2162397248 \\
        --hopping-model ACSF_hoppings_sk_M_9_W_6 \\
        --pod-temperature 0.1 --hopping-temperature 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from matplotlib.colors import Normalize

CSFONT = {"fontname": "sans-serif", "size": 18}
plt.rcParams.update(
    {
        "font.family": CSFONT["fontname"],
        "font.size": CSFONT["size"],
        "axes.labelsize": CSFONT["size"],
        "axes.titlesize": CSFONT["size"],
        "legend.fontsize": 14,
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

from blg_model_builder.DataLoader import (  # noqa: E402
    _ydata_hopping_aligned_to_acsf_order,
    load_hopping_data,
)
from blg_model_builder.ensemble_io import (  # noqa: E402
    DEFAULT_CALIBRATION_METRICS_DIR,
    expand_model_patterns,
    load_ensemble_pickle,
    load_metrics_npz,
    metrics_npz_path,
    resolve_ensemble_pickle,
)
from blg_model_builder.strain_data import (  # noqa: E402
    LAT_CON,
    identify_stacking,
    interlayer_sep,
    layers_have_uniform_z,
)
from blg_model_builder.tb_descriptors import get_acsf_sk_hopping_descriptors  # noqa: E402

DEFAULT_FIGURES_DIR = UQ_DIR / "figures"
DEFAULT_ENSEMBLE_DIR = "ensembles"
DEFAULT_POD_MODEL = "POD_energy_POD_index_15_8bb97b2162397248"
DEFAULT_HOPPING_MODEL = "ACSF_hoppings_sk_M_9_W_6"
DEFAULT_POD_TEMPERATURE = 0.1
DEFAULT_HOPPING_TEMPERATURE = 0.25

STACKING_LABELS = ("AA", "AB", "SP")
STACKING_X = {"AA": 0.0, "AB": 1.0, "SP": 2.0}
STRAIN_GRID = (-0.01, 0.0, 0.01)
STRAIN_GRID_TOL = 0.002
DISORDER_FLOOR = 1e-4

# (attribute key, x-axis label, panel title) for 1-D uncertainty plots.
DIMENSION_PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("stacking_code", "stacking", "stacking"),
    ("strain_mag", r"in-plane strain magnitude $|\varepsilon|$", "strain magnitude"),
    ("layer_sep", r"interlayer separation (Å)", "layer separation"),
    ("disorder", "disorder magnitude", "disorder"),
)


@dataclass(frozen=True)
class LinearFitResult:
    slope: float
    intercept: float
    chi2: float
    n_points: int


@dataclass(frozen=True)
class StructureRecord:
    atoms: Atoms
    stacking: str
    stacking_code: float
    stacking_x: float
    strain_mag: float
    layer_sep: float
    disorder: float
    uncertainty: float
    n_atoms: int
    category: str  # "strain" | "disorder"


def _pick_ypred_xdata(ensemble_dict: dict) -> tuple[dict, dict]:
    yp_test = ensemble_dict.get("ypred_samples_test")
    if isinstance(yp_test, dict) and yp_test:
        xd_test = ensemble_dict.get("xdata_test") or ensemble_dict.get("xdata")
        if isinstance(xd_test, dict) and xd_test:
            return yp_test, xd_test
    yp = ensemble_dict.get("ypred_samples")
    xd = ensemble_dict.get("xdata")
    if isinstance(yp, dict) and yp and isinstance(xd, dict) and xd:
        return yp, xd
    raise KeyError(
        "ensemble dict needs ypred_samples_test+xdata_test "
        "or ypred_samples+xdata."
    )


def _energy_atom_counts(xdata_energy: Sequence[Atoms], n_configs: int) -> np.ndarray:
    nat = np.ones(max(int(n_configs), 0), dtype=float)
    for i in range(min(int(n_configs), len(xdata_energy))):
        try:
            nat[i] = float(len(xdata_energy[i]))
        except Exception:
            nat[i] = 1.0
    return nat


def layer_corrugation(atoms: Atoms) -> float:
    pos = atoms.get_positions()
    z = pos[:, 2]
    idx = np.argsort(z)
    half = len(z) // 2
    if half == 0:
        return 0.0
    return float(max(np.ptp(z[idx[:half]]), np.ptp(z[idx[half:]])))


def inplane_strain_components(atoms: Atoms, *, a_ref: float) -> Tuple[float, float, float]:
    cell = atoms.cell.array
    eps_x = (float(np.linalg.norm(cell[0, :2])) - a_ref) / a_ref
    eps_y = (float(np.linalg.norm(cell[1, :2])) - a_ref) / a_ref
    mag = float(np.hypot(eps_x, eps_y))
    return eps_x, eps_y, mag


def _snap_to_strain_grid(value: float) -> float:
    return float(min(STRAIN_GRID, key=lambda g: abs(value - g)))


def strain_off_grid_magnitude(eps_x: float, eps_y: float) -> float:
    gx = _snap_to_strain_grid(eps_x)
    gy = _snap_to_strain_grid(eps_y)
    return float(np.hypot(eps_x - gx, eps_y - gy))


def is_hopping_systematic(atoms: Atoms, *, a_ref: float) -> bool:
    if not layers_have_uniform_z(atoms):
        return False
    _, _, _ = inplane_strain_components(atoms, a_ref=a_ref)
    eps_x, eps_y, _ = inplane_strain_components(atoms, a_ref=a_ref)
    off = strain_off_grid_magnitude(eps_x, eps_y)
    return off <= STRAIN_GRID_TOL and layer_corrugation(atoms) <= DISORDER_FLOOR


def is_pod_strain_cell(atoms: Atoms) -> bool:
    return len(atoms) == 4 and layers_have_uniform_z(atoms)


def hopping_disorder(atoms: Atoms, *, a_ref: float) -> float:
    eps_x, eps_y, _ = inplane_strain_components(atoms, a_ref=a_ref)
    off = strain_off_grid_magnitude(eps_x, eps_y)
    corr = layer_corrugation(atoms)
    return float(np.hypot(off, corr))


def pod_disorder(atoms: Atoms) -> float:
    if is_pod_strain_cell(atoms):
        return 0.0
    return layer_corrugation(atoms)


def hopping_a_ref(atoms_list: Sequence[Atoms]) -> float:
    vals = []
    for atoms in atoms_list:
        if layers_have_uniform_z(atoms):
            vals.append(float(np.linalg.norm(atoms.cell.array[0, :2])))
    if not vals:
        return float(np.median([np.linalg.norm(a.cell.array[0, :2]) for a in atoms_list]))
    return float(np.median(vals))


def _stacking_x_with_jitter(stacking: str, rng: np.random.Generator) -> float:
    base = STACKING_X.get(stacking, 1.0)
    return base + float(rng.uniform(-0.12, 0.12))


def _parse_acsf_mw(model_name: str) -> Tuple[int, int]:
    import re

    m = re.search(r"[_\-]M[_\-](\d+)[_\-]W[_\-](\d+)", model_name, re.I)
    if not m:
        raise ValueError(f"Could not parse M/W from hopping model name {model_name!r}.")
    return int(m.group(1)), int(m.group(2))


def _hopping_reference_fingerprint(
    atoms: Atoms,
    hopping_row: dict,
    k: int,
    *,
    m: int,
    w: int,
    r_cut: float,
) -> np.ndarray:
    dsc_sk, (pair_i, pair_j, pair_v) = get_acsf_sk_hopping_descriptors(
        atoms, M=m, W=w, r_cut=r_cut, use_envelope=True,
    )
    y_row, mask = _ydata_hopping_aligned_to_acsf_order(
        atoms,
        pair_i,
        pair_j,
        pair_v,
        hopping_row["i"][k],
        hopping_row["j"][k],
        hopping_row["di"][k],
        hopping_row["dj"][k],
        hopping_row["hopping"][k],
    )
    return np.asarray(y_row[mask], dtype=float).ravel()


def map_hopping_atoms_to_xdata_indices(
    ydata_hopping: Sequence,
    *,
    m: int,
    w: int,
    r_cut: float = 6.0,
) -> List[int]:
    """Map ``load_hopping_data()['atoms'][k]`` to pickle ``xdata['hopping']`` index."""
    hopping_row = load_hopping_data(hopping_type="all")
    refs = [np.asarray(y, dtype=float).ravel() for y in ydata_hopping]
    out: List[int] = []
    for k, atoms in enumerate(hopping_row["atoms"]):
        fp = _hopping_reference_fingerprint(
            atoms, hopping_row, k, m=m, w=w, r_cut=r_cut,
        )
        matches = [
            i
            for i, ref in enumerate(refs)
            if ref.size == fp.size and np.allclose(ref, fp, rtol=0.0, atol=1e-7)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Hopping structure loader index {k} matched {len(matches)} pickle rows "
                f"(expected 1)."
            )
        out.append(int(matches[0]))
    return out


def hopping_std_per_structure(ensemble_params: np.ndarray, xdata_hopping: Sequence) -> np.ndarray:
    params = np.asarray(ensemble_params, dtype=float)
    stds = np.full(len(xdata_hopping), np.nan, dtype=float)
    for i, xi in enumerate(xdata_hopping):
        Xi = np.asarray(xi, dtype=float)
        if Xi.size == 0:
            continue
        preds = Xi @ params.T
        stds[i] = float(np.mean(np.std(preds, axis=1, ddof=0)))
    return stds


def pod_std_per_atom(
    ypred_energy: np.ndarray,
    xdata_energy: Sequence[Atoms],
) -> np.ndarray:
    """Per-config ensemble std of total energy per atom (eV/atom).

    For each structure ``i`` and ensemble draw ``s``,

        (E/N)_s,i = E_s,i / N_i ,

    then ``σ_i = std_s((E/N)_s,i)``.  ``ypred_energy`` is expected as
    ``(n_samples, n_configs)`` from ``evaluate_ensemble``; rows are transposed
    only when configs are stored on axis 0 instead.
    """
    yp = np.asarray(ypred_energy, dtype=float)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)
    n_configs = len(xdata_energy)
    if yp.ndim == 2 and yp.shape[1] != n_configs and yp.shape[0] == n_configs:
        yp = yp.T
    if yp.ndim != 2:
        raise ValueError(f"Expected 2-D energy predictions, got shape {yp.shape}.")
    n_use = int(min(n_configs, yp.shape[1]))
    yp = yp[:, :n_use]
    nat = _energy_atom_counts(xdata_energy, n_use)
    energy_per_atom = yp / nat[np.newaxis, :]
    return np.std(energy_per_atom, axis=0, ddof=0)


def build_structure_records(
    atoms_list: Sequence[Atoms],
    uncertainties: np.ndarray,
    *,
    a_ref: float,
    pod_mode: bool,
) -> List[StructureRecord]:
    rng = np.random.default_rng(0)
    records: List[StructureRecord] = []
    for atoms, unc in zip(atoms_list, uncertainties):
        if not np.isfinite(unc):
            continue
        stack = identify_stacking(atoms)
        if pod_mode:
            if is_pod_strain_cell(atoms):
                dx, dy = _pod_strain_mag(atoms)
                strain_mag = float(np.hypot(dx, dy))
            else:
                strain_mag = 0.0
            disorder = pod_disorder(atoms)
            category = "strain" if is_pod_strain_cell(atoms) else "disorder"
        else:
            _, _, strain_mag = inplane_strain_components(atoms, a_ref=a_ref)
            disorder = hopping_disorder(atoms, a_ref=a_ref)
            category = "strain" if is_hopping_systematic(atoms, a_ref=a_ref) else "disorder"
        records.append(
            StructureRecord(
                atoms=atoms,
                stacking=stack,
                stacking_code=float(STACKING_X.get(stack, 1.0)),
                stacking_x=_stacking_x_with_jitter(stack, rng),
                strain_mag=strain_mag,
                layer_sep=float(interlayer_sep(atoms)),
                disorder=disorder,
                uncertainty=float(unc),
                n_atoms=len(atoms),
                category=category,
            )
        )
    return records


def _pod_strain_mag(atoms: Atoms) -> Tuple[float, float]:
    cell = atoms.cell.array
    dx = (cell[0, 0] - LAT_CON) / LAT_CON
    dy = (np.linalg.norm(cell[1, :2]) - LAT_CON) / LAT_CON
    return float(dx), float(dy)


def _auto_pod_model(ensemble_dir: str, calibration_metrics_dir: str) -> str:
    candidates = expand_model_patterns(["POD_energy_POD_index*"], ensemble_dir)
    best_name: Optional[str] = None
    best_nll = float("inf")
    for model_name in candidates:
        path = metrics_npz_path(
            calibration_metrics_dir, model_name, "mcmc", "energy",
        )
        if not os.path.isfile(path):
            continue
        nll_arr = np.asarray(load_metrics_npz(path)["nll"], dtype=float)
        nll_min = float(np.nanmin(nll_arr)) if nll_arr.size else float("nan")
        if np.isfinite(nll_min) and nll_min < best_nll:
            best_nll = nll_min
            best_name = model_name
    return best_name or DEFAULT_POD_MODEL


def _scatter_panel(
    ax: plt.Axes,
    records: Sequence[StructureRecord],
    *,
    x_attr: str,
    y_attr: str,
    title: str,
    cbar_label: str,
    cmap,
    norm: Normalize,
    filter_fn,
) -> None:
    pts = [r for r in records if filter_fn(r)]
    if not pts:
        ax.set_title(title, fontsize=CSFONT["size"], fontname=CSFONT["fontname"])
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return None

    x = np.asarray([getattr(r, x_attr) for r in pts], dtype=float)
    y = np.asarray([getattr(r, y_attr) for r in pts], dtype=float)
    c = np.asarray([r.uncertainty for r in pts], dtype=float)

    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        norm=norm,
        s=55,
        edgecolors="0.15",
        linewidths=0.4,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=CSFONT["size"], fontname=CSFONT["fontname"])
    ax.set_xlabel(_axis_label(x_attr), fontdict=CSFONT)
    ax.set_ylabel(_axis_label(y_attr), fontdict=CSFONT)
    ax.grid(True, alpha=0.25)
    if x_attr == "stacking_x":
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_xticklabels(list(STACKING_LABELS))
        ax.set_xlim(-0.35, 2.35)
    return sc


def _axis_label(attr: str) -> str:
    if attr in ("stacking_x", "stacking_code"):
        return "stacking"
    if attr == "strain_mag":
        return r"in-plane strain magnitude $|\varepsilon|$"
    if attr == "layer_sep":
        return r"interlayer separation (Å)"
    if attr == "disorder":
        return "disorder magnitude"
    return attr


def _dimension_value(record: StructureRecord, attr: str) -> float:
    return float(getattr(record, attr))


def fit_linear_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[LinearFitResult]:
    """Least-squares line ``y = intercept + slope * x`` and unweighted ``χ²``."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    chi2 = float(np.sum((y - y_fit) ** 2))
    return LinearFitResult(
        slope=float(slope),
        intercept=float(intercept),
        chi2=chi2,
        n_points=int(x.size),
    )


def _fit_annotation_text(fit: LinearFitResult) -> str:
    return (
        rf"$y = {fit.intercept:.4g} + ({fit.slope:.4g})\,x$" + "\n"
        + rf"$\chi^2 = {fit.chi2:.4g}$"
    )


def _uncertainty_ylabel(dataset_label: str) -> str:
    if "POD" in dataset_label:
        return r"$\sigma(E)/N_\mathrm{atoms}$ (eV/atom)"
    return r"mean bond $\sigma(t)$ (eV)"


def plot_uncertainty_vs_dimension_figure(
    records: Sequence[StructureRecord],
    *,
    dataset_label: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Uncertainty vs each deformation dimension with linear fits."""
    if not records:
        print(f"  No records for {dataset_label}; skipping 1-D figure.", flush=True)
        return

    y_label = _uncertainty_ylabel(dataset_label)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)

    print(f"\n  Linear fits: {dataset_label}", flush=True)
    for ax, (x_attr, x_label, title) in zip(axes.ravel(), DIMENSION_PANELS):
        x = np.asarray([_dimension_value(r, x_attr) for r in records], dtype=float)
        y = np.asarray([r.uncertainty for r in records], dtype=float)

        ax.scatter(
            x,
            y,
            s=45,
            color="steelblue",
            edgecolors="0.15",
            linewidths=0.4,
            alpha=0.85,
            zorder=3,
        )

        fit = fit_linear_uncertainty(x, y)
        if fit is not None:
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            y_line = fit.intercept + fit.slope * x_line
            ax.plot(x_line, y_line, color="C3", lw=2.0, zorder=2)
            ax.text(
                0.04,
                0.96,
                _fit_annotation_text(fit),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=13,
                fontname=CSFONT["fontname"],
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
            )
            print(
                f"    {title}:  y = {fit.intercept:.6g} + ({fit.slope:.6g}) x,  "
                f"chi^2 = {fit.chi2:.6g}  (N = {fit.n_points})",
                flush=True,
            )
        else:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"uncertainty vs {title}", fontsize=CSFONT["size"], fontname=CSFONT["fontname"])
        ax.set_xlabel(x_label, fontdict=CSFONT)
        ax.set_ylabel(y_label, fontdict=CSFONT)
        ax.grid(True, alpha=0.25)
        if x_attr == "stacking_code":
            ax.set_xticks([0.0, 1.0, 2.0])
            ax.set_xticklabels(list(STACKING_LABELS))
            ax.set_xlim(-0.35, 2.35)

    fig.suptitle(
        f"{dataset_label} — uncertainty vs dimension",
        fontsize=CSFONT["size"],
        fontname=CSFONT["fontname"],
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def plot_environment_uncertainty_figure(
    records: Sequence[StructureRecord],
    *,
    dataset_label: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    if not records:
        print(f"  No records for {dataset_label}; skipping figure.", flush=True)
        return

    unc = np.asarray([r.uncertainty for r in records], dtype=float)
    vmin, vmax = float(np.min(unc)), float(np.max(unc))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["viridis"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    panels = [
        (axes[0, 0], "stacking_x", "strain_mag", "strain vs stacking", lambda r: r.category == "strain"),
        (axes[0, 1], "stacking_x", "disorder", "disorder vs stacking", lambda r: r.category == "disorder"),
        (axes[1, 0], "stacking_x", "layer_sep", "layer separation vs stacking", lambda r: True),
        (axes[1, 1], "strain_mag", "layer_sep", "layer separation vs strain", lambda r: r.category == "strain"),
    ]

    last_sc = None
    cbar_label = _uncertainty_ylabel(dataset_label)
    for ax, x_attr, y_attr, title, filt in panels:
        sc = _scatter_panel(
            ax,
            records,
            x_attr=x_attr,
            y_attr=y_attr,
            title=title,
            cbar_label=cbar_label,
            cmap=cmap,
            norm=norm,
            filter_fn=filt,
        )
        if sc is not None:
            last_sc = sc

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), pad=0.02, fraction=0.046)
        cbar.set_label(cbar_label, fontdict=CSFONT)

    fig.suptitle(dataset_label, fontsize=CSFONT["size"], fontname=CSFONT["fontname"], y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}", flush=True)


def load_pod_records(
    model_name: str,
    *,
    ensemble_dir: str,
    calibration_metrics_dir: str,
    temperature: Optional[float],
) -> List[StructureRecord]:
    pkl_path, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_target="energy",
    )
    print(f"POD ensemble: {pkl_path}  (T={t_used:g})", flush=True)
    ens_dict = load_ensemble_pickle(pkl_path)
    ypred, xdata = _pick_ypred_xdata(ens_dict)
    if "energy" not in ypred or "energy" not in xdata:
        raise KeyError("POD pickle missing energy predictions/xdata.")

    atoms_list = list(xdata["energy"])
    unc = pod_std_per_atom(ypred["energy"], atoms_list)
    print(
        f"  POD test configs: {len(atoms_list)} "
        f"({sum(is_pod_strain_cell(a) for a in atoms_list)} strain, "
        f"{sum(not is_pod_strain_cell(a) for a in atoms_list)} disorder)",
        flush=True,
    )
    return build_structure_records(
        atoms_list, unc, a_ref=LAT_CON, pod_mode=True,
    )


def load_hopping_records(
    model_name: str,
    *,
    ensemble_dir: str,
    calibration_metrics_dir: str,
    temperature: Optional[float],
    r_cut: float,
) -> List[StructureRecord]:
    pkl_path, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature,
        calibration_metrics_dir=calibration_metrics_dir,
        calibration_target="hopping",
    )
    print(f"Hopping ensemble: {pkl_path}  (T={t_used:g})", flush=True)
    ens_dict = load_ensemble_pickle(pkl_path)
    xdata_full = ens_dict["xdata"]
    ydata_full = ens_dict["ydata"]
    if "hopping" not in xdata_full or "hopping" not in ydata_full:
        raise KeyError("Hopping pickle missing full xdata/ydata.")

    m, w = _parse_acsf_mw(model_name)
    index_map = map_hopping_atoms_to_xdata_indices(
        ydata_full["hopping"], m=m, w=w, r_cut=r_cut,
    )
    hopping_row = load_hopping_data(hopping_type="all")
    atoms_list = hopping_row["atoms"]
    a_ref = hopping_a_ref(atoms_list)

    params = np.asarray(ens_dict["ensemble"]["hopping"], dtype=float)
    unc_full = hopping_std_per_structure(params, xdata_full["hopping"])
    unc = np.asarray([unc_full[i] for i in index_map], dtype=float)

    n_sys = sum(is_hopping_systematic(a, a_ref=a_ref) for a in atoms_list)
    print(
        f"  Hopping configs: {len(atoms_list)} ({n_sys} systematic, "
        f"{len(atoms_list) - n_sys} random/disordered)",
        flush=True,
    )
    return build_structure_records(
        atoms_list, unc, a_ref=a_ref, pod_mode=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ensemble uncertainty vs deformation categories.",
    )
    parser.add_argument(
        "--target",
        choices=("pod", "hopping", "both"),
        default="both",
        help="Which dataset/model to plot (default: both).",
    )
    parser.add_argument(
        "--pod-model",
        default=None,
        help=f"POD energy ensemble folder (default: lowest-NLL POD_energy_POD_index* "
        f"or {DEFAULT_POD_MODEL}).",
    )
    parser.add_argument(
        "--hopping-model",
        default=DEFAULT_HOPPING_MODEL,
        help="ACSF_hoppings_sk ensemble folder.",
    )
    parser.add_argument(
        "--pod-temperature",
        type=float,
        default=None,
        help=f"POD ensemble temperature (default: min miscalibration or {DEFAULT_POD_TEMPERATURE}).",
    )
    parser.add_argument(
        "--hopping-temperature",
        type=float,
        default=DEFAULT_HOPPING_TEMPERATURE,
        help="Hopping ensemble temperature.",
    )
    parser.add_argument("--ensemble-dir", default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument(
        "--calibration-metrics-dir",
        default=DEFAULT_CALIBRATION_METRICS_DIR,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for figures.",
    )
    parser.add_argument("--r-cut", type=float, default=6.0, help="ACSF cutoff (Å).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    os.chdir(UQ_DIR)
    figures_dir = args.figures_dir
    if not figures_dir.is_absolute():
        figures_dir = UQ_DIR / figures_dir

    if args.target in ("pod", "both"):
        pod_model = args.pod_model or _auto_pod_model(
            args.ensemble_dir, args.calibration_metrics_dir,
        )
        pod_records = load_pod_records(
            pod_model,
            ensemble_dir=args.ensemble_dir,
            calibration_metrics_dir=args.calibration_metrics_dir,
            temperature=args.pod_temperature,
        )
        plot_environment_uncertainty_figure(
            pod_records,
            dataset_label=f"POD energy uncertainty ({pod_model})",
            out_path=figures_dir / f"{pod_model}_environment_uncertainty.png",
            dpi=args.dpi,
        )
        plot_uncertainty_vs_dimension_figure(
            pod_records,
            dataset_label=f"POD energy uncertainty ({pod_model})",
            out_path=figures_dir / f"{pod_model}_uncertainty_vs_dimension.png",
            dpi=args.dpi,
        )

    if args.target in ("hopping", "both"):
        hop_records = load_hopping_records(
            args.hopping_model,
            ensemble_dir=args.ensemble_dir,
            calibration_metrics_dir=args.calibration_metrics_dir,
            temperature=args.hopping_temperature,
            r_cut=args.r_cut,
        )
        plot_environment_uncertainty_figure(
            hop_records,
            dataset_label=f"Hopping uncertainty ({args.hopping_model})",
            out_path=figures_dir / f"{args.hopping_model}_environment_uncertainty.png",
            dpi=args.dpi,
        )
        plot_uncertainty_vs_dimension_figure(
            hop_records,
            dataset_label=f"Hopping uncertainty ({args.hopping_model})",
            out_path=figures_dir / f"{args.hopping_model}_uncertainty_vs_dimension.png",
            dpi=args.dpi,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
