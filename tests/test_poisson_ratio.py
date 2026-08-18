"""
In-plane and out-of-plane (interlayer) Poisson-style response for AB and AA
bilayer graphene, for every calculator in ``MODEL_KEYS``.

* **Out-of-plane** (two, straining lattice vector **a1** or **a2** in turn): for each
  in-plane engineering strain, scan interlayer distance ``d``, fit a parabola to
  :math:`E(d)`, and read the equilibrium :math:`d_\\text{eq}(\\epsilon)`.
  Out-of-plane Poisson ratio uses the equilibrium interlayer separation
  :math:`l(\\epsilon)` from the :math:`E(d)` parabolic minimum and
  :math:`l(0)` at zero strain. With the fixed scale ``LAT_CON`` (``POISSON_NU_DENOM_A``,
  same as ``POISSON_A0``, equilibrium :math:`a = 2.4694` Å), we use
  :math:`\\nu(\\epsilon) = -(l(\\epsilon)-l(0))/(a\\,\\epsilon)`.
  The **reported** scalar ``nu`` is :math:`\\nu(\\epsilon_\\text{ref})` at
  ``POISSON_NU_REFERENCE_STRAIN`` (default ``+1\\%``), with :math:`l(\\epsilon_\\text{ref})`
  from the grid or linear interpolation. At :math:`\\epsilon=0` the ratio is undefined;
  that entry is shown as N/A in the per-strain table.
* **In-plane**: total energy on a 5×5 grid of in-plane strains :math:`(\\epsilon_1, \\epsilon_2)`;
  fit :math:`E = E_0 + \\tfrac12 C_{11}\\epsilon_1^2 + C_{12}\\epsilon_1\\epsilon_2
  + \\tfrac12 C_{22}\\epsilon_2^2` and set :math:`\\nu_\\text{in} = C_{12} / C_{11}` (symmetric hex, :math:`C_{11} \\approx C_{22}`).

All bilayer cells are built with :mod:`blg_model_builder.geom_tools` (unstrained
reference, then in-plane cell strains via fixed reduced coordinates).

**Hyperparameters** and ``MODEL_KEYS`` are imported from :mod:`test_relaxation` so
they stay in sync; edit the globals in ``tests/test_relaxation.py``.

Run
---
``pytest tests/test_poisson_ratio.py -v -s``  (import ``test_relaxation``; needs LAMMPS
Python module for ``tersoff_kc`` / ``POD_energy``; ``tetb_pod`` is expensive).

``pytest -m "not slow"`` skips the heavy work if you add other slow tests; this file
is marked slow by default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
from ase import Atoms

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Import calculators so ``Atoms`` gets ``relax_structure`` patches (same as other tests).
from blg_model_builder.potentials import (  # noqa: F401 — side effect: patch ``Atoms.relax_structure``
    PODASECalculator,
    TETB_PODASECalculator,
    TersoffKolmogorovCrespiASECalculator,
)

from blg_model_builder.geom_tools import get_aa_bilayer_atoms, get_bilayer_atoms
from blg_model_builder.strain_data import LAT_CON

# ---------------------------------------------------------------------------
# Hyperparameters — single source of truth in ``test_relaxation.py``
# (listed here for visibility; change values in that file.)
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import test_relaxation as tr  # noqa: E402

RELAXATION_TETB_POD_TB_M = tr.RELAXATION_TETB_POD_TB_M
RELAXATION_TETB_POD_TB_W = tr.RELAXATION_TETB_POD_TB_W
RELAXATION_TETB_POD_POD_M = tr.RELAXATION_TETB_POD_POD_M
RELAXATION_TETB_POD_POD_W = tr.RELAXATION_TETB_POD_POD_W
POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE = tr.POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE
POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE = tr.POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE
MODEL_KEYS: Tuple[str, ...] = tr.MODEL_KEYS

# Geometry: reference in-plane constant (Å), c-axis box (Å; fixed in strain tests)
POISSON_A0 = LAT_CON
POISSON_C_VACUUM = 20.0
# Scale in ``ν(ε) = -(l(ε)−l(0)) / (POISSON_NU_DENOM_A * ε)``; same as in-plane ``a``.
POISSON_NU_DENOM_A = float(POISSON_A0)
# Engineering strain at which the scalar ``ν`` is evaluated (fraction, not percent).
POISSON_NU_REFERENCE_STRAIN = 0.01
# Fixed interlayer for in-plane 5×5 energy grid
POISSON_INPLANE_D_REF = 3.43
# d scans (Å)
POISSON_AB_D_GRID = np.linspace(3.35, 3.5, 15)
POISSON_AA_D_GRID = np.linspace(3.5, 3.8, 15)
# In-plane engineering strains
POISSON_STRAINS = np.asarray([-0.01, -0.005, 0.0, 0.005, 0.01], dtype=float)
relaxation_case_or_skip = tr.relaxation_case_or_skip

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _artifacts_dir() -> Path:
    d = _TESTS_DIR / "_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Geometry: ``geom_tools`` + independent strains on a1, a2
# ---------------------------------------------------------------------------


def _apply_biaxial_inplane_strain(base: Atoms, eps1: float, eps2: float) -> Atoms:
    """
    Deform the first two Bravais rows by ``(1+eps1)``, ``(1+eps2)``; keep c fixed.
    Fractional (reduced) positions unchanged = homogeneous in-plane engineering strain.
    """
    a = base.copy()
    s = a.get_scaled_positions()
    c = np.asarray(a.get_cell(), dtype=float)
    c0, c1, c2 = c[0], c[1], c[2]
    c0n = c0 * (1.0 + float(eps1))
    c1n = c1 * (1.0 + float(eps2))
    a.set_cell(np.vstack([c0n, c1n, c2]), scale_atoms=False)
    a.set_scaled_positions(s)
    return a


def _bilayer_ab(
    d: float,
    eps1: float = 0.0,
    eps2: float = 0.0,
    a0: float = POISSON_A0,
) -> Atoms:
    b = get_bilayer_atoms(
        d, 0.0, a=float(a0), c=POISSON_C_VACUUM, sc=1, zshift="CM",
    )
    return _apply_biaxial_inplane_strain(b, eps1, eps2)


def _bilayer_aa(
    d: float,
    eps1: float = 0.0,
    eps2: float = 0.0,
    a0: float = POISSON_A0,
) -> Atoms:
    b = get_aa_bilayer_atoms(
        d, a=float(a0), c=POISSON_C_VACUUM, sc=1, zshift="CM",
    )
    return _apply_biaxial_inplane_strain(b, eps1, eps2)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _eq_d_from_quadratic(
    d_grid: np.ndarray, energies: np.ndarray,
) -> float:
    """Equilibrium d from a quadratic E(d); fallback to argmin on grid if ill-conditioned."""
    d = np.asarray(d_grid, dtype=float)
    e = np.asarray(energies, dtype=float)
    ok = np.isfinite(e) & np.isfinite(d)
    d, e = d[ok], e[ok]
    if d.size < 3:
        return float(d[e.argmin()]) if d.size else float("nan")
    p2, p1, p0 = np.polyfit(d, e, 2)
    if abs(p2) < 1e-18 or p2 <= 0.0:
        j = int(np.argmin(e))
        return float(d[j])
    dmin = -0.5 * p1 / p2
    d_lo, d_hi = float(d.min()), float(d.max())
    if dmin < d_lo or dmin > d_hi:
        j = int(np.argmin(e))
        return float(d[j])
    return float(dmin)


def _out_of_plane_poisson_from_d_eq(
    eps: np.ndarray,
    d_eq: np.ndarray,
    denom_a: float = POISSON_NU_DENOM_A,
    eps_ref: float = POISSON_NU_REFERENCE_STRAIN,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-plane ``ν(ε) = -(l(ε)−l(0)) / (denom_a * ε)``; ``l`` = equilibrium d.

    ``l(0)`` from data at ``ε=0`` (or linear interpolation). Scalar ``nu`` = ``ν(ε_ref)``.

    Returns
    -------
    nu, l0, np.array([l_ref, eps_ref]), d_filtered, nu_raw
        ``nu_raw[i] = ν(e_[i])``; NaN where ``e_[i]`` = 0.
    """
    m = np.isfinite(eps) & np.isfinite(d_eq)
    e_ = np.asarray(eps, dtype=float)[m]
    d_ = np.asarray(d_eq, dtype=float)[m]
    den = float(denom_a)
    if e_.size < 1 or den < 1e-12:
        return float("nan"), float("nan"), np.array([]), d_, np.array([])

    z = np.abs(e_) < 1e-14
    if np.any(z):
        l0 = float(np.mean(d_[z]))
    else:
        l0 = float(np.interp(0.0, e_, d_))

    nu_raw = np.full(d_.size, np.nan, dtype=float)
    nz = np.abs(e_) > 1e-20
    nu_raw[nz] = -(d_[nz] - l0) / (den * e_[nz])

    er = float(eps_ref)
    if abs(er) < 1e-20:
        return float("nan"), float(l0), np.array([float("nan"), er]), d_, nu_raw
    hit = np.where(np.isclose(e_, er, atol=1e-12, rtol=0.0))[0]
    if hit.size:
        l_ref = float(d_[hit[0]])
    else:
        order = np.argsort(e_)
        e_s, d_s = e_[order], d_[order]
        l_ref = float(np.interp(er, e_s, d_s))

    nu = -(l_ref - l0) / (den * er)
    return float(nu), float(l0), np.array([l_ref, er]), d_, nu_raw


def _fit_inplane_nu_from_energies(
    eps1_g: np.ndarray, eps2_g: np.ndarray, e_grid: np.ndarray,
) -> float:
    """
    Fit E = c0 + c11 e1^2 + c12 e1 e2 + c22 e2^2 to the 5×5 energies;
    return nu = c12 / (2 c11) with c11 the coefficient of e1^2 (so C11 = 2 c11).
    """
    e1 = np.asarray(eps1_g, dtype=float).ravel()
    e2 = np.asarray(eps2_g, dtype=float).ravel()
    e_ = np.asarray(e_grid, dtype=float).ravel()
    o = np.ones_like(e1, dtype=float)
    A = np.column_stack(
        [o, e1 * e1, e1 * e2, e2 * e2],
    )
    coef, *_ = np.linalg.lstsq(A, e_, rcond=None)
    c0, a11, a12, a22 = coef[0], coef[1], coef[2], coef[3]
    c11, c12, c22 = 2.0 * a11, a12, 2.0 * a22
    _ = c0, c22  # reference energy (unused)
    if abs(c11) < 1e-20:
        return float("nan")
    return float(c12 / c11)


# ---------------------------------------------------------------------------
# Energy sweeps
# ---------------------------------------------------------------------------


def _energy_vs_d(
    d_grid: np.ndarray,
    builder: Callable[[float, float, float], Atoms],
    extra_eps1: float,
    extra_eps2: float,
    case,
) -> np.ndarray:
    """E(d) for fixed (extra_eps1, extra_eps2) in builder(d, e1, e2)."""
    d_grid = np.asarray(d_grid, dtype=float)
    out = np.empty(d_grid.size, dtype=float)
    t = builder(float(d_grid[0]), float(extra_eps1), float(extra_eps2))
    calc0 = case.calc_factory(t)
    t.calc = calc0
    e0 = float(t.get_potential_energy())
    out[0] = e0
    for i in range(1, d_grid.size):
        t = builder(float(d_grid[i]), float(extra_eps1), float(extra_eps2))
        t.calc = calc0
        out[i] = float(t.get_potential_energy())
    return out


def _energy_inplane_grid(
    builder: Callable[[float, float, float], Atoms],
    d_fixed: float,
    strains: np.ndarray,
    case,
) -> np.ndarray:
    s = np.asarray(strains, dtype=float)
    n = s.size
    e2d = np.empty((n, n), dtype=float)
    t0 = builder(float(d_fixed), float(s[0]), float(s[0]))
    calc0 = case.calc_factory(t0)
    for i, e1 in enumerate(s):
        for j, e2 in enumerate(s):
            t = builder(float(d_fixed), float(e1), float(e2))
            t.calc = calc0
            e2d[i, j] = float(t.get_potential_energy())
    return e2d


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------


@pytest.fixture
def _require_lammps_py(request):
    """Skip if the current parametrized model needs LAMMPS but it is unavailable.

    Models not in ``tr._MODELS_NEEDING_LAMMPS`` skip this check.
    """
    callspec = getattr(request.node, "callspec", None)
    model_key = callspec.params.get("model_key") if callspec else None
    if model_key is not None and model_key not in tr._MODELS_NEEDING_LAMMPS:
        return  # pure-Python model — no LAMMPS required
    try:
        import lammps  # noqa: F401
    except Exception as exc:
        pytest.skip(f"LAMMPS Python module not available: {exc}")


def _print_and_assert_poisson(name: str, nu: float) -> None:
    print(
        f"  [{name}] Poisson-like coefficient = {nu:.5f} "
        f"({'finite' if np.isfinite(nu) else 'non-finite'})",
        flush=True,
    )
    if np.isfinite(nu) and (nu < -5.0 or nu > 5.0):
        # sanity only — surface numerics, do not fail loose classical fits
        print(f"  (warning) [{name}] value is outside a typical [-5, 5] window.", flush=True)
    if not np.isfinite(nu):
        raise AssertionError(f"Non-finite Poisson value for {name!r}.")


@pytest.mark.slow
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_poisson_ratios_bilayer(model_key: str, _require_lammps_py) -> None:
    case = relaxation_case_or_skip(model_key)
    strains = POISSON_STRAINS
    s_ab1 = "AB, strain a₁ (ε₂=0)"
    s_ab2 = "AB, strain a₂ (ε₁=0)"
    s_aa1 = "AA, strain a₁ (ε₂=0)"
    s_aa2 = "AA, strain a₂ (ε₁=0)"
    s_ip_ab = "AB, in-plane (5×5 strain grid)"
    s_ip_aa = "AA, in-plane (5×5 strain grid)"

    d_ab = np.asarray(POISSON_AB_D_GRID, dtype=float)
    d_aa = np.asarray(POISSON_AA_D_GRID, dtype=float)

    # --- OOP: d_eq(ε) for a1 / a2 for AB
    d_eq_ab1 = np.empty(strains.size, dtype=float)
    d_eq_ab2 = np.empty(strains.size, dtype=float)
    for k, e in enumerate(strains):
        eab1 = _energy_vs_d(
            d_ab, _bilayer_ab, e, 0.0, case,
        )
        d_eq_ab1[k] = _eq_d_from_quadratic(d_ab, eab1)
    for k, e in enumerate(strains):
        eab2 = _energy_vs_d(
            d_ab, _bilayer_ab, 0.0, e, case,
        )
        d_eq_ab2[k] = _eq_d_from_quadratic(d_ab, eab2)

    nu_za1_ab, l0_za1_ab, _, _, nu_raw_ab1 = _out_of_plane_poisson_from_d_eq(
        strains, d_eq_ab1,
    )
    nu_za2_ab, l0_za2_ab, _, _, nu_raw_ab2 = _out_of_plane_poisson_from_d_eq(
        strains, d_eq_ab2,
    )

    # --- OOP: AA
    d_eq_aa1 = np.empty(strains.size, dtype=float)
    d_eq_aa2 = np.empty(strains.size, dtype=float)
    for k, e in enumerate(strains):
        d_eq_aa1[k] = _eq_d_from_quadratic(
            d_aa, _energy_vs_d(d_aa, _bilayer_aa, e, 0.0, case),
        )
    for k, e in enumerate(strains):
        d_eq_aa2[k] = _eq_d_from_quadratic(
            d_aa, _energy_vs_d(d_aa, _bilayer_aa, 0.0, e, case),
        )

    nu_za1_aa, l0_za1_aa, _, _, nu_raw_aa1 = _out_of_plane_poisson_from_d_eq(
        strains, d_eq_aa1,
    )
    nu_za2_aa, l0_za2_aa, _, _, nu_raw_aa2 = _out_of_plane_poisson_from_d_eq(
        strains, d_eq_aa2,
    )

    # --- In-plane 5×5, fixed d
    e1m, e2m = np.meshgrid(strains, strains, indexing="ij")
    e2d_ab = _energy_inplane_grid(
        _bilayer_ab, POISSON_INPLANE_D_REF, strains, case,
    )
    e2d_aa = _energy_inplane_grid(
        _bilayer_aa, POISSON_INPLANE_D_REF, strains, case,
    )
    # Remove reference energy E(0,0) to emphasize curvature for C_ij fit.
    mid = 2
    e2d_ab = e2d_ab - e2d_ab[mid, mid]
    e2d_aa = e2d_aa - e2d_aa[mid, mid]
    nu_in_ab = _fit_inplane_nu_from_energies(e1m, e2m, e2d_ab)
    nu_in_aa = _fit_inplane_nu_from_energies(e1m, e2m, e2d_aa)

    def _fmt_nu_raw(eps_arr: np.ndarray, raw: np.ndarray) -> str:
        """Format per-strain ν(ε) = -(l−l0)/(a·ε); N/A at ε=0."""
        parts: list[str] = []
        for e, r in zip(
            np.asarray(eps_arr, dtype=float), np.asarray(raw, dtype=float),
        ):
            if not np.isfinite(r):
                parts.append(f"ε={100.0 * float(e):+.2f}% → N/A (ε=0)")
            else:
                parts.append(
                    f"ε={100.0 * float(e):+.2f}% → "
                    f"−(l−l0)/({POISSON_NU_DENOM_A:.2f}·ε)={float(r):+.5f}",
                )
        return "  |  ".join(parts)

    print(
        f"\n=== Poisson response — {case.energy_display_name!r} ({model_key!r}) ===\n"
        f"  Ref: a0={POISSON_A0} Å,  ν(ε) = −(l(ε)−l(0)) / ({POISSON_NU_DENOM_A:.2f}·ε),  "
        f"c={POISSON_C_VACUUM} Å, d(in-plane grid)={POISSON_INPLANE_D_REF} Å\n"
        f"  Strains (engineering): {np.array2string(strains, precision=4, floatmode='fixed')}\n"
        f"  OOP reported ν = ν(ε_ref) with ε_ref={100.0 * POISSON_NU_REFERENCE_STRAIN:+.2f}%\n"
        f"  {s_ab1}:  l(0)={l0_za1_ab:.5f} Å,  ν={nu_za1_ab:.5f}\n"
        f"           {_fmt_nu_raw(strains, nu_raw_ab1)}\n"
        f"  {s_ab2}:  l(0)={l0_za2_ab:.5f} Å,  ν={nu_za2_ab:.5f}\n"
        f"           {_fmt_nu_raw(strains, nu_raw_ab2)}\n"
        f"  {s_aa1}:  l(0)={l0_za1_aa:.5f} Å,  ν={nu_za1_aa:.5f}\n"
        f"           {_fmt_nu_raw(strains, nu_raw_aa1)}\n"
        f"  {s_aa2}:  l(0)={l0_za2_aa:.5f} Å,  ν={nu_za2_aa:.5f}\n"
        f"           {_fmt_nu_raw(strains, nu_raw_aa2)}\n"
        f"  In-plane: ν = C₁₂/C₁₁ (5×5 E grid, d={POISSON_INPLANE_D_REF} Å): "
        f"{s_ip_ab} → {nu_in_ab:.5f}  |  {s_ip_aa} → {nu_in_aa:.5f}\n"
        f"  AB: ν_out(a₁)={nu_za1_ab:.5f}  |  ν_out(a₂)={nu_za2_ab:.5f}  |  ν_in={nu_in_ab:.5f}\n"
        f"  AA: ν_out(a₁)={nu_za1_aa:.5f}  |  ν_out(a₂)={nu_za2_aa:.5f}  |  ν_in={nu_in_aa:.5f}\n",
        flush=True,
    )

    _print_and_assert_poisson(f"{model_key} {s_ab1} OOP", nu_za1_ab)
    _print_and_assert_poisson(f"{model_key} {s_ab2} OOP", nu_za2_ab)
    _print_and_assert_poisson(f"{model_key} {s_aa1} OOP", nu_za1_aa)
    _print_and_assert_poisson(f"{model_key} {s_aa2} OOP", nu_za2_aa)
    _print_and_assert_poisson(f"{model_key} {s_ip_ab} in", nu_in_ab)
    _print_and_assert_poisson(f"{model_key} {s_ip_aa} in", nu_in_aa)

    if plt is not None:
        out = _artifacts_dir() / f"poisson_d_eq_{model_key}.png"
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), dpi=120)
        panels: list[tuple[str, np.ndarray, str]] = [
            ("AB, strain a₁ (ε₂=0)", d_eq_ab1, "AB"),
            ("AB, strain a₂ (ε₁=0)", d_eq_ab2, "AB"),
            ("AA, strain a₁ (ε₂=0)", d_eq_aa1, "AA"),
            ("AA, strain a₂ (ε₁=0)", d_eq_aa2, "AA"),
        ]
        for ax, (title, d_curve, st) in zip(np.ravel(axes), panels):
            ax.plot(
                100.0 * strains, d_curve, "o-", color="#1a3d6b", label="d_eq(ε) from E(d) quadratic min",
            )
            ax.set_xlabel("in-plane engineering strain in strained direction (%)")
            ax.set_ylabel("equilibrium d (Å)")
            ax.set_title(f"{case.energy_display_name} — {title} ({st})")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="best", fontsize=7)
        fig.suptitle(
            f"Equilibrium interlayer separation d vs strain (parabolic E(d) on d-grids) — {model_key}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f"  [plot] saved: {out}", flush=True)
    else:
        print("  (matplotlib not available, skipping figure)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(
        __import__("pytest").main([__file__, "-v", "--tb=short", "-s", *sys.argv[1:]]),
    )
