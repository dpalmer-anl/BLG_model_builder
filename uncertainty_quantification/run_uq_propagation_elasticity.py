#!/usr/bin/env python3
"""
Elastic constants and Poisson ratios from MCMC ensemble samples and DFT reference.

For each selected model (``--models`` with optional glob wildcards, same as
:mod:`plot_bayes_factor`). Supported LAMMPS ensembles:

* ``POD_energy`` / ``POD_energy_POD_index_*``
* ``TETB_POD`` / ``TETB_POD_*_POD_index_*``
* ``Tersoff+DRIP``
* ``Tersoff+Kolmogorov_Crespi``

* Shuffle the MCMC ensemble and run until ``--n-samples`` successful evaluations
  (failed LAMMPS runs are skipped; stop early if the ensemble is exhausted).
* Evaluate energies on a bilayer strain / separation grid (AB and AA).
* Compute Poisson ratios via polynomial fits evaluated at zero strain.
* Compute Young's moduli from quadratic energy–strain fits (2D stress
  :math:`\\sigma = \\partial e / \\partial\\varepsilon`, :math:`Y = \\partial\\sigma/\\partial\\varepsilon`).
* Plot stress–strain curves and coupling paths (in-plane strain vs separation,
  in-plane strain vs in-plane strain) with ensemble mean ± std.

DFT Poisson ratios use discrete rVV10 structures: at each in-plane strain,
group configs by separation, fit a quadratic through the five lowest-energy
points near the minimum to get equilibrium ``l``, then
``ν = -(l - l₀) / ε`` (polynomial derivative at zero strain).

Examples
--------
::

    python run_uq_propagation_elasticity.py --models 'POD_energy_POD_index*' \\
        --temperature 0.5 --n-samples 100 --include-dft

    python run_uq_propagation_elasticity.py --models Tersoff+DRIP --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

# Local imports (same package)
from blg_model_builder.ensemble_io import (
    DEFAULT_CALIBRATION_METRICS_DIR,
    expand_model_patterns,
    load_ensemble_pickle,
    resolve_ensemble_pickle,
)
from blg_model_builder.strain_data import LAT_CON, STRAIN_RANGE, load_strained_data

from uq_model_runtime import (
    apply_uq_parameters,
    build_uq_lammps_calculator,
    is_uq_lammps_model,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args
from blg_model_builder.cli_model_names import (
    add_energy_models_arg,
    collect_workflow_hyperparams,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STACKINGS_ELASTIC = ("AB", "AA")
DEFAULT_N_SAMPLES = 100
DEFAULT_ENSEMBLE_SHUFFLE_SEED = 0  # shared with run_uq_propagation_relaxation.py
DEFAULT_STRAIN_RANGE = STRAIN_RANGE
DEFAULT_POLY_DEGREE = 2

# Geometry (match tests/test_poisson_ratio.py)
A0 = LAT_CON
C_VACUUM = 20.0
AB_D_GRID = (3.35, 3.5)
AA_D_GRID = (3.5, 3.8)
INPLANE_D_REF = {"AB": 3.43, "AA": 3.65}

# DFT discrete Poisson extraction (strained_bilayer_graphene_rVV10.xyz grid).
# Strain grid matches gen_training_data_structures.py:
#   strain_array = linspace(-0.02, 0.02, 5); cell[0,:] *= 1+dx; cell[1,:] *= 1+dy
DFT_STRAIN_DECIMALS = 4
DFT_CROSS_STRAIN_ATOL = 0.003   # other in-plane strain held ≈ 0
DFT_SEP_BAND_ATOL = 0.02        # Å, fixed-sep slice for in-plane ν_xy
DFT_N_LOCAL_QUAD = 5            # points in quadratic fit near E minimum
DFT_STRAIN_GRID = np.linspace(-0.02, 0.02, 5)  # generator strain_array
DFT_LAYER_SEP_GRID = np.array(
    [3.0, 3.1, 3.2, 3.3, 3.35, 3.43, 3.5, 3.65, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0],
)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def _apply_biaxial_inplane_strain(base: Atoms, eps1: float, eps2: float) -> Atoms:
    a = base.copy()
    s = a.get_scaled_positions()
    c = np.asarray(a.get_cell(), dtype=float)
    c[0] *= 1.0 + float(eps1)
    c[1] *= 1.0 + float(eps2)
    a.set_cell(c, scale_atoms=False)
    a.set_scaled_positions(s)
    return a


def _bilayer_builder(stacking: str) -> Callable[[float, float, float], Atoms]:
    from blg_model_builder.geom_tools import get_aa_bilayer_atoms, get_bilayer_atoms

    def build(d: float, eps1: float = 0.0, eps2: float = 0.0) -> Atoms:
        if stacking == "AB":
            b = get_bilayer_atoms(d, 0.0, a=float(A0), c=C_VACUUM, sc=1, zshift="CM")
        else:
            b = get_aa_bilayer_atoms(d, a=float(A0), c=C_VACUUM, sc=1, zshift="CM")
        return _apply_biaxial_inplane_strain(b, eps1, eps2)

    return build


def _hex_cell_area(a0: float = A0) -> float:
    """In-plane area of the primitive hex cell (Å²)."""
    return float(a0 * a0 * np.sqrt(3.0) / 2.0)


# ---------------------------------------------------------------------------
# Polynomial fits at zero strain
# ---------------------------------------------------------------------------


def _poly_derivative_at_zero(x: np.ndarray, y: np.ndarray, degree: int = DEFAULT_POLY_DEGREE) -> float:
    """Return dy/dx at x=0 from a polynomial least-squares fit."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < degree + 1:
        return float("nan")
    p = np.polyfit(x, y, degree)
    return float(np.polyval(np.polyder(p), 0.0))


def poisson_nu_xz_poly(dx_g: np.ndarray, sep_g: np.ndarray, E_xz: np.ndarray, sep0: float, degree: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """Like :func:`plot_poisson_ratio.poisson_nu_xz` but ν = −dε_z/dε_x|₀ from a polynomial."""
    from blg_model_builder.strain_data import parabolic_min as _parabolic_min

    sep_star = np.array([
        _parabolic_min(sep_g, E_xz[i, :], int(np.argmin(E_xz[i, :])))
        for i in range(len(dx_g))
    ])
    eps_z = (sep_star - sep0) / sep0
    nu = -_poly_derivative_at_zero(dx_g, eps_z, degree)
    return float(nu), sep_star, eps_z


def poisson_nu_yz_poly(dy_g: np.ndarray, sep_g: np.ndarray, E_yz: np.ndarray, sep0: float, degree: int) -> Tuple[float, np.ndarray, np.ndarray]:
    from blg_model_builder.strain_data import parabolic_min as _parabolic_min

    sep_star = np.array([
        _parabolic_min(sep_g, E_yz[j, :], int(np.argmin(E_yz[j, :])))
        for j in range(len(dy_g))
    ])
    eps_z = (sep_star - sep0) / sep0
    nu = -_poly_derivative_at_zero(dy_g, eps_z, degree)
    return float(nu), sep_star, eps_z


def poisson_nu_xy_poly(dx_g: np.ndarray, dy_g: np.ndarray, E_xy: np.ndarray, degree: int) -> Tuple[float, np.ndarray]:
    from blg_model_builder.strain_data import parabolic_min as _parabolic_min

    dy_star = np.array([
        _parabolic_min(dy_g, E_xy[i, :], int(np.argmin(E_xy[i, :])))
        for i in range(len(dx_g))
    ])
    nu = -_poly_derivative_at_zero(dx_g, dy_star, degree)
    return float(nu), dy_star


def young_modulus_from_strain_energy(strains: np.ndarray, e_per_atom: np.ndarray, degree: int = 2) -> Tuple[float, np.ndarray]:
    """
    Fit E/atom(ε) with a polynomial; return Y = d²(E/atom)/dε² at ε=0 and
    σ = d(E/atom)/dε (both in eV/atom).

    To convert Y to the 2D Young's modulus in N/m per graphene layer::

        n_layers = 2          # BLG
        n_atoms  = 4          # primitive bilayer cell
        A_cell   = a0**2 * sqrt(3)/2   # Å²  (a0 ≈ 2.46 Å → A_cell ≈ 5.24 Å²)
        Y_2D_per_layer [N/m] = Y [eV/atom] * (n_atoms / n_layers) / A_cell * 16.0218
    """
    s = np.asarray(strains, dtype=float).ravel()
    e = np.asarray(e_per_atom, dtype=float).ravel()
    m = np.isfinite(s) & np.isfinite(e)
    s, e = s[m], e[m]
    strains_out = np.asarray(strains, dtype=float).ravel()
    if s.size < degree + 1:
        return float("nan"), np.full_like(strains_out, np.nan, dtype=float)
    p = np.polyfit(s, e, degree)
    Y = float(2.0 * p[0]) if degree >= 2 else float("nan")
    dp = np.polyder(p)
    stress = np.polyval(dp, strains_out)
    return Y, stress


# ---------------------------------------------------------------------------
# Model / ensemble I/O
# ---------------------------------------------------------------------------


def _shuffle_ensemble(ensemble: np.ndarray, seed: int) -> np.ndarray:
    """Return a full copy of the ensemble in random order."""
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 2:
        raise ValueError(f"Expected 2-D ensemble array, got shape {ensemble.shape}")
    n = ensemble.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    return ensemble[order]


def _is_lammps_error(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) is a LAMMPS failure."""
    cur: Optional[BaseException] = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if "lammps" in type(cur).__module__.lower():
            return True
        msg = str(cur).lower()
        if "lammps" in msg or "lmp_" in msg:
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "lammps" in msg or "relax_structure" in msg:
            return True
    return False


# ---------------------------------------------------------------------------
# Energy grid for one ensemble member
# ---------------------------------------------------------------------------


def _build_elasticity_atom_list(
    stacking: str,
    strain_range: float,
    n_strain: int,
    n_sep: int,
) -> Tuple[List[Atoms], Dict[str, Any]]:
    """All unique structures for xz / yz / xy slices and uniaxial stress curves."""
    builder = _bilayer_builder(stacking)
    sep_range = AB_D_GRID if stacking == "AB" else AA_D_GRID
    dx_g = np.linspace(-strain_range, strain_range, n_strain)
    dy_g = dx_g.copy()
    sep_g = np.linspace(sep_range[0], sep_range[1], n_sep)

    atoms_list: List[Atoms] = []
    meta_xz: List[Tuple[int, int, int]] = []
    meta_yz: List[Tuple[int, int, int]] = []
    meta_xy: List[Tuple[int, int, int]] = []
    meta_sx: List[int] = []
    meta_sy: List[int] = []
    meta_sz: List[int] = []

    def _add(atoms: Atoms) -> int:
        atoms_list.append(atoms)
        return len(atoms_list) - 1

    for i, dx in enumerate(dx_g):
        for j, sep in enumerate(sep_g):
            meta_xz.append((i, j, _add(builder(sep, dx, 0.0))))
    for j, dy in enumerate(dy_g):
        for k, sep in enumerate(sep_g):
            meta_yz.append((j, k, _add(builder(sep, 0.0, dy))))
    # xy slice at sep0 placeholder (filled after sep0 known — use mid sep for grid index)
    sep_mid = float(0.5 * (sep_range[0] + sep_range[1]))
    for i, dx in enumerate(dx_g):
        for j, dy in enumerate(dy_g):
            meta_xy.append((i, j, _add(builder(sep_mid, dx, dy))))
    for i, eps in enumerate(dx_g):
        meta_sx.append(_add(builder(sep_mid, eps, 0.0)))
    for j, eps in enumerate(dy_g):
        meta_sy.append(_add(builder(sep_mid, 0.0, eps)))
    for k, sep in enumerate(sep_g):
        meta_sz.append(_add(builder(sep, 0.0, 0.0)))

    meta = {
        "dx_g": dx_g,
        "dy_g": dy_g,
        "sep_g": sep_g,
        "sep_range": sep_range,
        "meta_xz": meta_xz,
        "meta_yz": meta_yz,
        "meta_xy": meta_xy,
        "meta_sx": meta_sx,
        "meta_sy": meta_sy,
        "meta_sz": meta_sz,
        "n_strain": n_strain,
        "n_sep": n_sep,
        "sep_mid": sep_mid,
    }
    return atoms_list, meta


def _energies_per_atom(
    calc_obj,
    theta: np.ndarray,
    n_atoms: int,
    set_params_fn: Optional[Callable[[np.ndarray], None]] = None,
) -> np.ndarray:
    apply_uq_parameters(calc_obj, theta, set_params_fn)
    energies, _ = calc_obj.evaluate_batch()
    e = np.asarray(energies, dtype=float).ravel()
    return e / float(n_atoms)


def _analyze_member_from_energies(
    stacking: str,
    e_pa: np.ndarray,
    meta: Dict[str, Any],
    poly_degree: int,
) -> Dict[str, Any]:
    n_strain = meta["n_strain"]
    n_sep = meta["n_sep"]
    dx_g = meta["dx_g"]
    dy_g = meta["dy_g"]
    sep_g = meta["sep_g"]
    sep_range = meta["sep_range"]

    E_xz = np.full((n_strain, n_sep), np.nan)
    for i, j, aidx in meta["meta_xz"]:
        E_xz[i, j] = e_pa[aidx]
    E_yz = np.full((n_strain, n_sep), np.nan)
    for j, k, aidx in meta["meta_yz"]:
        E_yz[j, k] = e_pa[aidx]
    E_xy = np.full((n_strain, n_strain), np.nan)
    for i, j, aidx in meta["meta_xy"]:
        E_xy[i, j] = e_pa[aidx]

    # sep0 from 1-D scan at dx=dy=0
    e_1d = np.array([e_pa[aidx] for aidx in meta["meta_sz"]])
    sep0 = float(sep_g[int(np.nanargmin(e_1d))])

    # Re-evaluate xy at sep0 would be better; use sep_mid slice for nu_xy
    nu_xz, sep_xz, eps_z_xz = poisson_nu_xz_poly(dx_g, sep_g, E_xz, sep0, poly_degree)
    nu_yz, sep_yz, eps_z_yz = poisson_nu_yz_poly(dy_g, sep_g, E_yz, sep0, poly_degree)
    nu_xy, dy_xy = poisson_nu_xy_poly(dx_g, dy_g, E_xy, poly_degree)
    dx_xy = np.asarray(dx_g[: len(dy_xy)], dtype=float)

    area = _hex_cell_area()
    strains = dx_g

    def _uniaxial(meta_idx: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
        e2d = np.array([e_pa[i] for i in meta_idx], dtype=float)
        Y, stress = young_modulus_from_strain_energy(strains, e2d, poly_degree)
        return Y, strains, stress

    Y_x, sx, sig_x = _uniaxial(meta["meta_sx"])
    Y_y, sy, sig_y = _uniaxial(meta["meta_sy"])
    # z: strain = (sep - sep0) / sep0
    eps_z = (sep_g - sep0) / sep0
    e2d_z = np.array([e_pa[i] for i in meta["meta_sz"]], dtype=float)
    Y_z, sig_z = young_modulus_from_strain_energy(eps_z, e2d_z, poly_degree)

    return {
        "stacking": stacking,
        "sep0": sep0,
        "nu_xz": nu_xz,
        "nu_yz": nu_yz,
        "nu_xy": nu_xy,
        "Y_x": Y_x,
        "Y_y": Y_y,
        "Y_z": Y_z,
        "dx_g": dx_g,
        "dy_g": dy_g,
        "sep_g": sep_g,
        "sep_xz": sep_xz,
        "sep_yz": sep_yz,
        "dx_xy": dx_xy,
        "dy_xy": dy_xy,
        "eps_z_xz": eps_z_xz,
        "eps_z_yz": eps_z_yz,
        "E_xz": E_xz,
        "E_yz": E_yz,
        "E_xy": E_xy,
        "strains": strains,
        "strains_x": strains,
        "strains_y": dy_g,
        "stress_x": sig_x,
        "stress_y": sig_y,
        "stress_z": sig_z,
        "eps_z": eps_z,
        "area": area,
    }


def analyze_ensemble_stacking(
    calc_obj,
    ensemble_shuffled: np.ndarray,
    n_target: int,
    stacking: str,
    strain_range: float,
    n_strain: int,
    n_sep: int,
    poly_degree: int,
    set_params_fn: Optional[Callable[[np.ndarray], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate up to ``n_target`` successful members from a shuffled ensemble.

    Skips LAMMPS failures and non-finite energies; stops when ``n_target``
    successes are collected or the shuffled list is exhausted.
    """
    atoms_list, meta = _build_elasticity_atom_list(stacking, strain_range, n_strain, n_sep)
    n_atoms = len(atoms_list[0])
    calc_obj.prepare_batch(list(atoms_list))
    results: List[Dict[str, Any]] = []
    n_skipped = 0
    n_tried = 0

    for theta in ensemble_shuffled:
        if len(results) >= n_target:
            break
        n_tried += 1
        try:
            e_pa = _energies_per_atom(calc_obj, theta, n_atoms, set_params_fn)
            if not np.all(np.isfinite(e_pa)):
                n_skipped += 1
                print(
                    f"    [{stacking}] skip ensemble member {n_tried}: "
                    f"non-finite energies",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            results.append(
                _analyze_member_from_energies(stacking, e_pa, meta, poly_degree)
            )
        except Exception as exc:
            if not _is_lammps_error(exc):
                raise
            n_skipped += 1
            print(
                f"    [{stacking}] skip ensemble member {n_tried}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if len(results) == 1 or len(results) % 10 == 0:
            print(
                f"    [{stacking}] {len(results)}/{n_target} successful "
                f"(tried {n_tried})",
                flush=True,
            )

    print(
        f"    [{stacking}] done: {len(results)} successful, {n_skipped} skipped, "
        f"{n_tried} tried (pool size {ensemble_shuffled.shape[0]})",
        flush=True,
    )
    return results


def _aggregate_scalar(samples: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float]:
    vals = np.array([r[key] for r in samples if np.isfinite(r.get(key, np.nan))], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# DFT Poisson ratios from discrete structures (quadratic fit near E minimum)
# ---------------------------------------------------------------------------


def _match_strain(values: np.ndarray, target: float, decimals: int = DFT_STRAIN_DECIMALS) -> np.ndarray:
    return np.isclose(values, float(target), atol=10.0 ** (-decimals))


def _strain_near(values: np.ndarray, target: float = 0.0) -> np.ndarray:
    """True where recovered strain is ≈ ``target`` (hold other in-plane strain fixed)."""
    return np.abs(np.asarray(values, dtype=float) - float(target)) <= DFT_CROSS_STRAIN_ATOL


def _unique_strain_values(values: np.ndarray, decimals: int = DFT_STRAIN_DECIMALS) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    rounded = np.round(values, decimals)
    return np.sort(np.unique(rounded))


def _l0_from_zero_strain_slice(
    data: np.ndarray,
    n_local: int = DFT_N_LOCAL_QUAD,
) -> float:
    """Equilibrium sep at ε_x = ε_y = 0 from all separations in the dataset."""
    z = _match_strain(data[:, 0], 0.0) & _match_strain(data[:, 1], 0.0)
    if not np.any(z):
        return float("nan")
    return _equilibrium_sep_quadratic(data[z, 2], data[z, 3], n_local=n_local)


def _equilibrium_sep_quadratic(
    sep: np.ndarray,
    energy: np.ndarray,
    n_local: int = DFT_N_LOCAL_QUAD,
) -> float:
    """
    Equilibrium interlayer separation from E(sep).

    Uses the ``n_local`` points with lowest energy, fits E = a·sep² + b·sep + c,
    and returns the separation at the parabolic minimum (clamped to the fitted range).
    """
    sep = np.asarray(sep, dtype=float).ravel()
    energy = np.asarray(energy, dtype=float).ravel()
    m = np.isfinite(sep) & np.isfinite(energy)
    sep, energy = sep[m], energy[m]
    if sep.size < 3:
        return float(sep[np.argmin(energy)]) if sep.size else float("nan")
    order = np.argsort(energy)
    n_use = min(int(n_local), sep.size)
    ix = order[:n_use]
    xs = sep[ix]
    ys = energy[ix]
    j = np.argsort(xs)
    xs, ys = xs[j], ys[j]
    try:
        xc = xs - np.mean(xs)
        a2, a1, a0 = np.polyfit(xc, ys, 2, rcond=None)
        d_min = -0.5 * a1 / a2
        d_min = float(np.mean(xs) + d_min)
    except np.linalg.LinAlgError:
        return float(xs[np.argmin(ys)])
    if a2 <= 0:
        return float(xs[np.argmin(ys)])
    if d_min < xs.min() or d_min > xs.max():
        return float(xs[np.argmin(ys)])
    return float(d_min)


def _equilibrium_dy_quadratic(
    dy: np.ndarray,
    energy: np.ndarray,
    n_local: int = DFT_N_LOCAL_QUAD,
) -> float:
    """Equilibrium in-plane strain ε_y* from E(ε_y) at fixed ε_x and sep."""
    return _equilibrium_sep_quadratic(dy, energy, n_local=n_local)


def _poisson_nu_at_zero_from_l(
    strains: np.ndarray,
    l_eq: np.ndarray,
    l0: float,
    poly_degree: int,
) -> Tuple[float, np.ndarray]:
    """
    Poisson ratio at ε = 0 from equilibrium separations l(ε).

    Per the DFT procedure, ν(ε) = -(l(ε) - l₀) / ε.  The reported scalar is
    dν/dε|₀ from a polynomial fit to l(ε), i.e. ν = -dl/dε|₀ in the small-strain
    limit (equivalent to the limit of -(l-l₀)/ε).
    """
    eps = np.asarray(strains, dtype=float).ravel()
    l = np.asarray(l_eq, dtype=float).ravel()
    m = np.isfinite(eps) & np.isfinite(l)
    eps, l = eps[m], l[m]
    nu_raw = np.full(eps.size, np.nan, dtype=float)
    nz = np.abs(eps) > 1e-12
    nu_raw[nz] = -(l[nz] - l0) / eps[nz]
    if eps.size < 2:
        return float("nan"), nu_raw
    deg = min(poly_degree, int(eps.size) - 1)
    p = np.polyfit(eps, l, deg)
    nu = -float(np.polyval(np.polyder(p), 0.0))
    return nu, nu_raw


def _poisson_nu_xy_at_zero(
    dx_g: np.ndarray,
    dy_star: np.ndarray,
    poly_degree: int,
) -> float:
    """In-plane ν_xy = -dε_y*/dε_x|₀ from equilibrium ε_y*(ε_x) at fixed sep ≈ sep₀."""
    dx = np.asarray(dx_g, dtype=float).ravel()
    dy = np.asarray(dy_star, dtype=float).ravel()
    m = np.isfinite(dx) & np.isfinite(dy)
    dx, dy = dx[m], dy[m]
    if dx.size < 2:
        return float("nan")
    deg = min(poly_degree, int(dx.size) - 1)
    p = np.polyfit(dx, dy, deg)
    return -float(np.polyval(np.polyder(p), 0.0))


def _out_of_plane_l_path(
    data: np.ndarray,
    fixed_strain_col: int,
    free_strain_col: int,
    fixed_value: float = 0.0,
    n_local: int = DFT_N_LOCAL_QUAD,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Out-of-plane path: hold strain along ``free_strain_col`` ≈ 0, scan
    ``fixed_strain_col``; at each strain, fit l from E(sep) over all separations.

    Returns (strain_values, l_eq, l0).
    """
    # Hold transverse in-plane strain on the generator grid (exact ε = 0).
    cross_ok = _match_strain(data[:, free_strain_col], fixed_value)
    sub = data[cross_ok]
    if sub.size == 0:
        return np.array([]), np.array([]), float("nan")

    l0 = _l0_from_zero_strain_slice(data, n_local=n_local)

    l_list: List[float] = []
    s_list: List[float] = []
    for s in DFT_STRAIN_GRID:
        if np.abs(s) <= DFT_CROSS_STRAIN_ATOL:
            continue
        block = sub[_match_strain(sub[:, fixed_strain_col], float(s))]
        if block.shape[0] < 3:
            continue
        l_eq = _equilibrium_sep_quadratic(block[:, 2], block[:, 3], n_local=n_local)
        s_list.append(float(s))
        l_list.append(l_eq)

    return np.asarray(s_list), np.asarray(l_list), l0


def _inplane_dy_star_path(
    data: np.ndarray,
    sep0: float,
    n_local: int = DFT_N_LOCAL_QUAD,
) -> Tuple[np.ndarray, np.ndarray]:
    """At sep ≈ sep₀, for each ε_x find ε_y* minimizing E(ε_y) via local quadratic fit."""
    sep_snap = float(DFT_LAYER_SEP_GRID[np.argmin(np.abs(DFT_LAYER_SEP_GRID - sep0))])
    sub = data[np.isclose(data[:, 2], sep_snap, atol=0.04)]
    if sub.size == 0:
        return np.array([]), np.array([])

    dy_list: List[float] = []
    dx_list: List[float] = []
    for dx in DFT_STRAIN_GRID:
        block = sub[_match_strain(sub[:, 0], float(dx))]
        if block.shape[0] < 3:
            continue
        dy_star = _equilibrium_dy_quadratic(block[:, 1], block[:, 3], n_local=n_local)
        dx_list.append(float(dx))
        dy_list.append(dy_star)
    return np.asarray(dx_list), np.asarray(dy_list)


def _youngs_from_equilibrium_energies(
    strains: np.ndarray,
    energies: np.ndarray,
    poly_degree: int,
) -> Tuple[float, np.ndarray]:
    """Young's modulus and stress curve from E(ε) on the equilibrium path."""
    return young_modulus_from_strain_energy(strains, energies, poly_degree)


def analyze_dft_stacking_discrete(
    stacking: str,
    data: np.ndarray,
    poly_degree: int,
    n_local: int = DFT_N_LOCAL_QUAD,
) -> Dict[str, Any]:
    """
    DFT Poisson ratios from discrete rVV10 structures.

    Out-of-plane (ε_x loading, ε_y = 0): for each ε_x, group structures with
    that strain and varying sep; quadratic fit on the five lowest-E points →
    l(ε_x).  Similarly for ε_y loading.

    In-plane: at sep ≈ sep₀, for each ε_x group varying ε_y, quadratic fit →
    ε_y*(ε_x); ν_xy = -dε_y*/dε_x|₀.

    ν = -(l - l₀) / ε (small-strain limit: ν = -dl/dε|₀ from polynomial fit).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("data must be (N, 4): [dx, dy, sep, E_per_atom]")

    # --- Out-of-plane: strain along a₁ (ε_x), hold ε_y ≈ 0 ---
    dx_g, sep_xz, l0 = _out_of_plane_l_path(
        data, fixed_strain_col=0, free_strain_col=1, fixed_value=0.0, n_local=n_local,
    )
    nu_xz, nu_raw_xz = _poisson_nu_at_zero_from_l(dx_g, sep_xz, l0, poly_degree)
    eps_z_xz = (sep_xz - l0) / l0 if np.isfinite(l0) and l0 > 0 else np.full_like(sep_xz, np.nan)

    # --- Out-of-plane: strain along a₂ (ε_y), hold ε_x ≈ 0 ---
    dy_g, sep_yz, _ = _out_of_plane_l_path(
        data, fixed_strain_col=1, free_strain_col=0, fixed_value=0.0, n_local=n_local,
    )
    nu_yz, nu_raw_yz = _poisson_nu_at_zero_from_l(dy_g, sep_yz, l0, poly_degree)
    eps_z_yz = (sep_yz - l0) / l0 if np.isfinite(l0) and l0 > 0 else np.full_like(sep_yz, np.nan)

    # --- In-plane: ε_y*(ε_x) at sep ≈ sep₀ ---
    dx_xy, dy_xy = _inplane_dy_star_path(data, l0, n_local=n_local)
    nu_xy = _poisson_nu_xy_at_zero(dx_xy, dy_xy, poly_degree)

    # --- Young's moduli from equilibrium-path energies (discrete) ---
    def _e_eq_along(fixed_col: int, free_col: int, strain_vals: np.ndarray) -> np.ndarray:
        out = []
        free = data[:, free_col]
        for s in strain_vals:
            cross = _strain_near(free, 0.0)
            block = data[cross & _match_strain(data[:, fixed_col], float(s))]
            if block.shape[0] < 3:
                out.append(np.nan)
                continue
            i_min = int(np.argmin(block[:, 3]))
            out.append(float(block[i_min, 3]))
        return np.asarray(out, dtype=float)

    e_x = _e_eq_along(0, 1, dx_g) if dx_g.size else np.array([])
    e_y = _e_eq_along(1, 0, dy_g) if dy_g.size else np.array([])
    Y_x, sig_x = _youngs_from_equilibrium_energies(dx_g, e_x, poly_degree) if dx_g.size else (float("nan"), np.array([]))
    Y_y, sig_y = _youngs_from_equilibrium_energies(dy_g, e_y, poly_degree) if dy_g.size else (float("nan"), np.array([]))

    zero_mask = _strain_near(data[:, 0], 0.0) & _strain_near(data[:, 1], 0.0)
    sub_z = data[zero_mask]
    sep_g = _unique_strain_values(sub_z[:, 2]) if sub_z.size else np.array([])
    e_z = []
    for sep in sep_g:
        blk = sub_z[np.isclose(sub_z[:, 2], sep, atol=DFT_SEP_BAND_ATOL)]
        if blk.size:
            e_z.append(float(np.min(blk[:, 3])))
        else:
            e_z.append(np.nan)
    sep_g = np.asarray(sep_g, dtype=float)
    e_z = np.asarray(e_z, dtype=float)
    eps_z = (sep_g - l0) / l0 if np.isfinite(l0) and l0 > 0 else np.full_like(sep_g, np.nan)
    Y_z, sig_z = _youngs_from_equilibrium_energies(eps_z, e_z, poly_degree) if sep_g.size else (float("nan"), np.array([]))

    print(
        f"  [DFT discrete {stacking}]  N={data.shape[0]} configs,  "
        f"l₀={l0:.4f} Å,  ν_xz={nu_xz:.4f},  ν_yz={nu_yz:.4f},  ν_xy={nu_xy:.4f}  "
        f"(quadratic fit on {n_local} lowest-E points per strain)",
        flush=True,
    )

    return {
        "stacking": stacking,
        "sep0": l0,
        "nu_xz": nu_xz,
        "nu_yz": nu_yz,
        "nu_xy": nu_xy,
        "Y_x": Y_x,
        "Y_y": Y_y,
        "Y_z": Y_z,
        "dx_g": dx_g,
        "dy_g": dy_g,
        "sep_g": sep_g,
        "sep_xz": sep_xz,
        "sep_yz": sep_yz,
        "dx_xy": dx_xy,
        "dy_xy": dy_xy,
        "eps_z_xz": eps_z_xz,
        "eps_z_yz": eps_z_yz,
        "nu_raw_xz": nu_raw_xz,
        "nu_raw_yz": nu_raw_yz,
        "strains": dx_g,
        "strains_x": dx_g,
        "strains_y": dy_g,
        "stress_x": sig_x,
        "stress_y": sig_y,
        "stress_z": sig_z,
        "eps_z": eps_z,
        "area": _hex_cell_area(),
        "source": "DFT_discrete",
    }


# ---------------------------------------------------------------------------
# Printing and plots
# ---------------------------------------------------------------------------


def print_elastic_summary(label: str, res: Dict[str, Any]) -> None:
    nu_note = (
        "ν from discrete DFT: quadratic on 5 lowest-E points, ν=-(l-l₀)/ε → dl/dε|₀"
        if res.get("source") == "DFT_discrete"
        else "ν (poly @ ε=0)"
    )
    # Y is d²(E/atom)/dε² → eV/atom.  Conversion: Y [N/m per layer] = Y * (n_atoms/n_layers) / A_cell * 16.0218
    # For BLG primitive cell: n_atoms=4, n_layers=2, A_cell = A0²√3/2 ≈ 5.24 Å²
    _n_atoms, _n_layers = 4, 2
    _A_cell = float(A0 ** 2 * np.sqrt(3.0) / 2.0)
    _ev_per_atom_to_Nm = (_n_atoms / _n_layers) / _A_cell * 16.0218

    def _fmt_Y(key: str) -> str:
        v = res.get(key, float("nan"))
        if not np.isfinite(float(v)):
            return f"{key}=nan"
        v = float(v)
        return f"{key}={v:.4f} eV/atom ({v * _ev_per_atom_to_Nm:.1f} N/m per layer)"

    print(
        f"\n{label} ({res['stacking']})  sep0={res['sep0']:.4f} Å\n"
        f"  {nu_note}:  ν_xz={res['nu_xz']:.4f}  ν_yz={res['nu_yz']:.4f}  ν_xy={res['nu_xy']:.4f}\n"
        f"  Young's modulus:  {_fmt_Y('Y_x')}  {_fmt_Y('Y_y')}  {_fmt_Y('Y_z')}",
        flush=True,
    )


def print_ensemble_summary(model_name: str, stacking: str, samples: List[Dict[str, Any]]) -> None:
    _n_atoms, _n_layers = 4, 2
    _A_cell = float(A0 ** 2 * np.sqrt(3.0) / 2.0)
    _ev_per_atom_to_Nm = (_n_atoms / _n_layers) / _A_cell * 16.0218

    print(f"\n=== Ensemble summary: {model_name} / {stacking} (n={len(samples)}) ===")
    for key in ("nu_xz", "nu_yz", "nu_xy"):
        mu, sd = _aggregate_scalar(samples, key)
        print(f"  {key:8s}:  mean={mu:+.5f}  std={sd:.5f}  (dimensionless)")
    for key in ("Y_x", "Y_y", "Y_z"):
        mu, sd = _aggregate_scalar(samples, key)
        mu_Nm = mu * _ev_per_atom_to_Nm
        sd_Nm = sd * _ev_per_atom_to_Nm
        print(
            f"  {key:8s}:  mean={mu:+.5f} eV/atom ({mu_Nm:.1f} N/m per layer)  "
            f"std={sd:.5f} eV/atom ({sd_Nm:.1f} N/m per layer)"
        )


def _plot_stress_strain_ensemble(
    model_name: str,
    stacking: str,
    samples: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    directions = [
        ("x", "strains_x", "stress_x", r"$\varepsilon_x$"),
        ("y", "strains_y", "stress_y", r"$\varepsilon_y$"),
        ("z", "eps_z", "stress_z", r"$\varepsilon_z$"),
    ]
    for ax, (tag, sk, stk, xlab) in zip(axes, directions):
        curves_xy: List[Tuple[np.ndarray, np.ndarray]] = []
        for r in samples:
            sk_use = sk if sk in r else ("strains" if tag == "x" else "dy_g" if tag == "y" else sk)
            x = np.asarray(r.get(sk_use, []), dtype=float).ravel() * 100.0
            y = np.asarray(r.get(stk, []), dtype=float).ravel()
            if x.size == 0 or y.size == 0:
                continue
            n = int(min(x.size, y.size))
            curves_xy.append((x[:n], y[:n]))
        if not curves_xy:
            ax.set_title(f"stress–strain ({tag}) — no data")
            ax.grid(True, alpha=0.3)
            continue
        x_ref = curves_xy[0][0]
        ys = [c[1] for c in curves_xy]
        n_min = min(len(y) for y in ys)
        x_ref = x_ref[:n_min]
        C = np.vstack([y[:n_min] for y in ys])
        mu = np.nanmean(C, axis=0)
        sd = np.nanstd(C, axis=0)
        mfin = np.isfinite(mu)
        if not np.any(mfin):
            ax.set_title(f"stress–strain ({tag}) — no finite data")
            ax.grid(True, alpha=0.3)
            continue
        ax.plot(x_ref[mfin], mu[mfin], "C0-", lw=2, label="mean")
        ax.fill_between(
            x_ref[mfin], (mu - sd)[mfin], (mu + sd)[mfin],
            color="C0", alpha=0.25, label=r"$\pm 1\sigma$",
        )
        ax.set_xlabel(xlab + " (%)")
        ax.set_ylabel(r"$\sigma$ (eV/atom)")
        ax.set_title(f"stress–strain ({tag})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{model_name} — {stacking} ensemble stress–strain", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_{stacking}_stress_strain.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}", flush=True)


def _aligned_curve_stats(
    samples: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair (x, y) per sample, align lengths, return (x_ref, mean(y), std(y))."""
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in samples:
        x = np.asarray(r.get(x_key, []), dtype=float).ravel() * x_scale
        y = np.asarray(r.get(y_key, []), dtype=float).ravel() * y_scale
        if x.size == 0 or y.size == 0:
            continue
        n = int(min(x.size, y.size))
        pairs.append((x[:n], y[:n]))
    if not pairs:
        return np.array([]), np.array([]), np.array([])
    n_min = min(p[0].size for p in pairs)
    x_ref = pairs[0][0][:n_min]
    ys = np.vstack([p[1][:n_min] for p in pairs])
    return x_ref, np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)


def _plot_coupling_panel(
    ax,
    samples: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    xlab: str,
    ylab: str,
    title: str,
    x_scale: float = 100.0,
    y_scale: float = 1.0,
) -> None:
    x_ref, mu, sd = _aligned_curve_stats(samples, x_key, y_key, x_scale, y_scale)
    if x_ref.size == 0 or not np.any(np.isfinite(mu)):
        ax.set_title(f"{title} — no data")
        ax.grid(True, alpha=0.3)
        return
    m = np.isfinite(mu)
    ax.plot(x_ref[m], mu[m], "o-", ms=3, lw=1.5)
    if np.any(np.isfinite(sd)):
        ax.fill_between(
            x_ref[m], (mu - sd)[m], (mu + sd)[m], alpha=0.25, label=r"$\pm 1\sigma$",
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_coupling_ensemble(
    model_name: str,
    stacking: str,
    samples: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    _plot_coupling_panel(
        axes[0], samples, "dx_g", "sep_xz",
        r"$\varepsilon_x$ (%)", "sep* (Å)", r"$\nu_{xz}$ path",
    )
    _plot_coupling_panel(
        axes[1], samples, "dy_g", "sep_yz",
        r"$\varepsilon_y$ (%)", "sep* (Å)", r"$\nu_{yz}$ path",
    )
    _plot_coupling_panel(
        axes[2], samples, "dx_xy", "dy_xy",
        r"$\varepsilon_x$ (%)", r"$\varepsilon_y^*$ (%)",
        r"$\nu_{xy}$ at sep$\approx$sep$_0$",
        x_scale=100.0,
        y_scale=100.0,
    )

    fig.suptitle(f"{model_name} — {stacking} in-plane / out-of-plane coupling", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_{stacking}_coupling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}", flush=True)


def _plot_dft_vs_ensemble_nu(
    model_name: str,
    dft_res: Dict[str, Dict[str, Any]],
    ens_res: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
) -> None:
    keys = ("nu_xz", "nu_yz", "nu_xy")
    fig, ax = plt.subplots(figsize=(7, 4))
    xlabels = []
    xpos = 0
    width = 0.35
    for stacking in STACKINGS_ELASTIC:
        if stacking not in dft_res or stacking not in ens_res:
            continue
        d = dft_res[stacking]
        for ki, key in enumerate(keys):
            mu, sd = _aggregate_scalar(ens_res[stacking], key)
            ax.bar(xpos - width / 2, d[key], width, color="C2", label="DFT" if xpos == 0 and ki == 0 else "")
            ax.bar(xpos + width / 2, mu, width, yerr=sd, color="C0", capsize=3,
                   label="ensemble" if xpos == 0 and ki == 0 else "")
            xlabels.append(f"{stacking}\n{key}")
            xpos += 1
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(r"Poisson ratio (@ $\varepsilon=0$)")
    ax.set_title(f"{model_name} vs DFT — Poisson ratios")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_poisson_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Elasticity / Poisson UQ propagation from MCMC ensembles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_energy_models_arg(p)
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
        help="Target number of successful ensemble evaluations per stacking.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_ENSEMBLE_SHUFFLE_SEED,
        help="RNG seed for ensemble shuffle (must match run_uq_propagation_relaxation.py).",
    )
    p.add_argument("--n-grid", type=int, default=30, help="Grid points per strain axis.")
    p.add_argument("--n-sep", type=int, default=80, help="Grid points along separation axis.")
    p.add_argument("--strain-range", type=float, default=DEFAULT_STRAIN_RANGE)
    p.add_argument("--poly-degree", type=int, default=DEFAULT_POLY_DEGREE,
                   help="Polynomial degree for ν and stress fits at zero strain.")
    p.add_argument("--figures-dir", default="figures/elasticity")
    p.add_argument("--include-dft", action="store_true", help="Also run DFT (rVV10) reference.")
    p.add_argument("--no-plots", action="store_true")
    add_hyperparam_args(p)
    args, _unknown = p.parse_known_args()
    cli_hyperparams = collect_workflow_hyperparams(args, _unknown)
    if cli_hyperparams:
        print(f"  CLI hyperparameters: {cli_hyperparams}", flush=True)

    os.chdir(HERE)
    models = expand_model_patterns(args.models, args.ensemble_dir)
    if not models:
        p.error("No models matched --models patterns.")
    print(f"Models: {models}", flush=True)

    dft_results: Dict[str, Dict[str, Any]] = {}
    if args.include_dft:
        print("\n--- DFT (rVV10) reference ---", flush=True)
        strained = load_strained_data()
        for stacking in STACKINGS_ELASTIC:
            if stacking not in strained:
                print(f"  skip DFT {stacking}: no data", file=sys.stderr)
                continue
            dft_results[stacking] = analyze_dft_stacking_discrete(
                stacking,
                strained[stacking],
                args.poly_degree,
                n_local=DFT_N_LOCAL_QUAD,
            )
            print_elastic_summary("DFT", dft_results[stacking])
            if not args.no_plots:
                dft_out = os.path.join(args.figures_dir, "DFT_rVV10")
                os.makedirs(dft_out, exist_ok=True)
                _plot_stress_strain_ensemble(
                    "DFT_rVV10", stacking, [dft_results[stacking]], dft_out,
                )
                _plot_coupling_ensemble(
                    "DFT_rVV10", stacking, [dft_results[stacking]], dft_out,
                )
        for stacking in sorted(dft_results):
            res = dft_results[stacking]
            if not res:
                continue
            parts = [
                f"{key}={float(res[key]):.4f}"
                for key in ("nu_xz", "nu_yz", "nu_xy")
                if key in res and np.isfinite(float(res[key]))
            ]
            sep0 = float(res.get("sep0", float("nan")))
            src = res.get("source", "reference")
            print(
                f"  {src} ({stacking})  sep0={sep0:.4f} Å"
                + ("  " + "  ".join(parts) if parts else ""),
                flush=True,
            )

    for model_name in models:
        print(f"\n--- Model: {model_name} ---", flush=True)
        if not is_uq_lammps_model(model_name):
            print(
                f"  Warning: unsupported model (need LAMMPS UQ model); "
                f"skipping {model_name!r}.",
                file=sys.stderr,
            )
            continue

        pkl_path, t_used = resolve_ensemble_pickle(
            model_name,
            args.ensemble_dir,
            args.temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_target=args.calibration_target,
        )
        print(f"  Ensemble pickle: {pkl_path}  (T={t_used:g})", flush=True)
        ens_dict = load_ensemble_pickle(pkl_path)
        ensemble = np.asarray(ens_dict["ensemble"]["energy"], dtype=float)
        ensemble_shuffled = _shuffle_ensemble(ensemble, args.seed)
        print(
            f"  Shuffled ensemble (seed={args.seed}): {ensemble.shape[0]} members; "
            f"target {args.n_samples} successful per stacking",
            flush=True,
        )

        calc_obj, set_params_fn, _load_name = build_uq_lammps_calculator(
            model_name, extra_kw=cli_hyperparams or None,
        )
        print(f"  LAMMPS calculator: {_load_name}", flush=True)

        out_dir = os.path.join(args.figures_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        ens_by_stacking: Dict[str, List[Dict[str, Any]]] = {}

        for stacking in STACKINGS_ELASTIC:
            print(f"  Stacking {stacking} …", flush=True)
            samples = analyze_ensemble_stacking(
                calc_obj,
                ensemble_shuffled,
                args.n_samples,
                stacking,
                args.strain_range,
                args.n_grid,
                args.n_sep,
                args.poly_degree,
                set_params_fn=set_params_fn,
            )
            ens_by_stacking[stacking] = samples
            print_ensemble_summary(model_name, stacking, samples)
            if not args.no_plots:
                _plot_stress_strain_ensemble(model_name, stacking, samples, out_dir)
                _plot_coupling_ensemble(model_name, stacking, samples, out_dir)

        summary_path = os.path.join(out_dir, f"{model_name}_elasticity_summary.json")
        summary = {
            "model_name": model_name,
            "temperature": t_used,
            "pkl_path": pkl_path,
            "n_samples_target": int(args.n_samples),
            "n_samples_success": {
                st: len(ens_by_stacking[st]) for st in ens_by_stacking
            },
            "strain_range": args.strain_range,
            "poly_degree": args.poly_degree,
            "per_stacking": {
                st: {
                    key: {"mean": _aggregate_scalar(ens_by_stacking[st], key)[0],
                          "std": _aggregate_scalar(ens_by_stacking[st], key)[1]}
                    for key in ("nu_xz", "nu_yz", "nu_xy", "Y_x", "Y_y", "Y_z")
                }
                for st in ens_by_stacking
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Wrote {summary_path}", flush=True)

        if args.include_dft and dft_results and not args.no_plots:
            _plot_dft_vs_ensemble_nu(model_name, dft_results, ens_by_stacking, out_dir)

        if hasattr(calc_obj, "close"):
            calc_obj.close()


if __name__ == "__main__":
    main()
