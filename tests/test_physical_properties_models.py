"""
Physical-property verification suite for BLG models.

Implements the checklist in `tests/physical_properties_models.md`:
  - Force/energy consistency (handled in dedicated tests per model)
  - Relaxation stability and reproducibility under perturbations
  - Energy/atom convergence with supercell size
  - AB/AA equilibrium interlayer separations and rigid-scan vs relaxed consistency
  - No in-plane buckling (per-layer z spread should be small)

This file focuses on the *physical* requirements (not low-level interface smoke tests).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest

def _ensure_importable_package() -> None:
    """Prefer installed package; fallback to repo `src/` only if needed."""
    try:
        import blg_model_builder  # noqa: F401
        return
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import blg_model_builder  # noqa: F401
        return
    except Exception:
        import types

        pkg = types.ModuleType("blg_model_builder")
        pkg.__path__ = [str(src)]  # type: ignore[attr-defined]
        sys.modules["blg_model_builder"] = pkg


_ensure_importable_package()

# Ensure this `tests/` directory is importable as a plain module path so we can
# import the shared harness without relying on `tests` being a package.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Require the LAMMPS Python module (DO NOT skip: fail loudly).
try:
    import lammps  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The LAMMPS Python module is required for physical-property tests. "
        "Build LAMMPS with -DWITH_PYTHON=yes and run `make install-python`. "
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc

from blg_model_builder.potentials import (
    PODASECalculator,
    TersoffDRIPASECalculator,
    TersoffKolmogorovCrespiASECalculator,
)

from physical_properties_harness import (
    POD_BLG_HYPERPARAMS,
    artifacts_dir,
    assert_no_buckling,
    compute_layer_metrics,
    layer_tags_from_mol_id,
    load_pod_best_fit_energy_params,
    make_blg_aa,
    make_blg_ab,
    make_calc_tersoff_drip_best_fit_estimate,
    make_calc_tersoff_kc_best_fit_estimate,
    perturb_positions,
    relax_atoms,
    rigid_scan_interlayer_minimum,
    save_scan_csv,
    save_scan_plot,
)


_A0 = 2.46
_A_TOL = 0.04

_AB_BAND = (3.4 - 0.1, 3.4 + 0.1)
_AA_BAND = (3.6 - 0.1, 3.6 + 0.1)


def _d_grid():
    # Keep this relatively coarse; each point is an expensive LAMMPS evaluation.
    return np.linspace(3.0, 5.0, 21)


def _assert_in_band(val: float, band, msg: str):
    lo, hi = float(band[0]), float(band[1])
    assert lo <= float(val) <= hi, f"{msg}: {val:.4f} not in [{lo:.4f}, {hi:.4f}]"


def _assert_relax_reproducible(atoms, *, max_d_diff: float = 0.03, seed: int = 0):
    """
    Relax a structure, then perturb and relax again. Check separations agree.
    """
    rel1 = relax_atoms(atoms, backend="ase", ftol=1e-5, maxiter=500)
    m1 = compute_layer_metrics(rel1)

    atoms2 = perturb_positions(atoms, sigma=0.01, seed=seed)
    atoms2.calc = atoms.calc
    rel2 = relax_atoms(atoms2, backend="ase", ftol=1e-5, maxiter=500)
    m2 = compute_layer_metrics(rel2)

    assert abs(m1.separation - m2.separation) <= max_d_diff, (
        f"Relaxation not reproducible: d1={m1.separation:.4f} d2={m2.separation:.4f}"
    )
    return rel1, rel2


@pytest.mark.parametrize("stacking", ["AB", "AA"])
def test_tersoff_kc_equilibrium_separation_and_scan_consistency(stacking):
    te, kc = make_calc_tersoff_kc_best_fit_estimate()
    # KC/DRIP layer-normal construction is not reliable on the primitive 4-atom
    # cell with small PBC; use a larger graphene supercell.
    sc = 7

    def atoms_builder(d):
        return make_blg_ab(d, a=_A0, sc=sc) if stacking == "AB" else make_blg_aa(d, a=_A0, sc=sc)

    def calc_builder(atoms):
        return TersoffKolmogorovCrespiASECalculator(
            te.tolist(),
            kc.tolist(),
            kc_cutoff=14.0,
            layer_tags=layer_tags_from_mol_id(atoms),
        )

    # Rigid scan at a=2.46 Å
    d_min_scan, _ = rigid_scan_interlayer_minimum(atoms_builder, calc_builder, d_grid=_d_grid())
    # Recompute energies for artifact saving (rigid_scan already computed, but doesn't return grid).
    dgrid = _d_grid()
    e_scan = np.empty_like(dgrid, dtype=float)
    for i, d in enumerate(dgrid):
        at = atoms_builder(float(d))
        at.calc = calc_builder(at)
        e_scan[i] = float(at.get_potential_energy())
    stem = f"tersoff_kc_{stacking}_rigid_scan"
    csv_path = save_scan_csv(dgrid, e_scan, stem=stem)
    png_path = save_scan_plot(dgrid, e_scan, stem=stem, title=f"Tersoff+KC {stacking} rigid scan")
    print(f"[Tersoff+KC {stacking}] rigid scan: d_min={d_min_scan:.4f} Å, saved {csv_path}"
          + (f", {png_path}" if png_path else " (no matplotlib; no png)"))

    # Relax starting near the scan minimum
    atoms0 = atoms_builder(d_min_scan + 0.2)
    atoms0.calc = calc_builder(atoms0)
    relaxed, _ = _assert_relax_reproducible(atoms0, seed=1)
    m = compute_layer_metrics(relaxed)
    print(f"[Tersoff+KC {stacking}] relaxed: d={m.separation:.4f} Å, "
          f"buckling std(z) layer0={m.buckling_layer0:.4f} Å layer1={m.buckling_layer1:.4f} Å")
    assert_no_buckling(m, max_std=0.02)

    if stacking == "AB":
        _assert_in_band(m.separation, _AB_BAND, "Tersoff+KC AB relaxed separation")
    else:
        _assert_in_band(m.separation, _AA_BAND, "Tersoff+KC AA relaxed separation")

    assert abs(m.separation - d_min_scan) <= 0.10, (
        f"Rigid-scan min d={d_min_scan:.4f} differs from relaxed d={m.separation:.4f}"
    )


@pytest.mark.parametrize("stacking", ["AB", "AA"])
def test_tersoff_drip_equilibrium_separation_and_scan_consistency(stacking):
    te, drip = make_calc_tersoff_drip_best_fit_estimate()
    # DRIP normals also behave best on a larger periodic sheet.
    sc = 7

    def atoms_builder(d):
        return make_blg_ab(d, a=_A0, sc=sc) if stacking == "AB" else make_blg_aa(d, a=_A0, sc=sc)

    def calc_builder(atoms):
        # DRIP normal construction is sensitive to ncut; use a value derived from
        # graphene's same-layer nearest-neighbour distance (~1.42 Å).
        ncut = 2.2
        return TersoffDRIPASECalculator(
            te.tolist(),
            drip.tolist(),
            ncut=ncut,
            layer_tags=layer_tags_from_mol_id(atoms),
        )

    d_min_scan, _ = rigid_scan_interlayer_minimum(atoms_builder, calc_builder, d_grid=_d_grid())
    dgrid = _d_grid()
    e_scan = np.empty_like(dgrid, dtype=float)
    for i, d in enumerate(dgrid):
        at = atoms_builder(float(d))
        at.calc = calc_builder(at)
        e_scan[i] = float(at.get_potential_energy())
    stem = f"tersoff_drip_{stacking}_rigid_scan"
    csv_path = save_scan_csv(dgrid, e_scan, stem=stem)
    png_path = save_scan_plot(dgrid, e_scan, stem=stem, title=f"Tersoff+DRIP {stacking} rigid scan")
    print(f"[Tersoff+DRIP {stacking}] rigid scan: d_min={d_min_scan:.4f} Å, saved {csv_path}"
          + (f", {png_path}" if png_path else " (no matplotlib; no png)"))

    atoms0 = atoms_builder(d_min_scan + 0.2)
    atoms0.calc = calc_builder(atoms0)
    relaxed, _ = _assert_relax_reproducible(atoms0, seed=2)
    m = compute_layer_metrics(relaxed)
    print(f"[Tersoff+DRIP {stacking}] relaxed: d={m.separation:.4f} Å, "
          f"buckling std(z) layer0={m.buckling_layer0:.4f} Å layer1={m.buckling_layer1:.4f} Å")
    assert_no_buckling(m, max_std=0.02)

    if stacking == "AB":
        _assert_in_band(m.separation, _AB_BAND, "Tersoff+DRIP AB relaxed separation")
    else:
        _assert_in_band(m.separation, _AA_BAND, "Tersoff+DRIP AA relaxed separation")

    assert abs(m.separation - d_min_scan) <= 0.12, (
        f"Rigid-scan min d={d_min_scan:.4f} differs from relaxed d={m.separation:.4f}"
    )


@pytest.mark.parametrize("stacking", ["AB", "AA"])
def test_pod_equilibrium_separation_and_scan_consistency(stacking):
    # POD coefficients from best_fit_params/; auto-fit is attempted when missing.
    try:
        params = load_pod_best_fit_energy_params(POD_BLG_HYPERPARAMS, require_file=True)
    except FileNotFoundError as exc:
        pytest.skip(
            f"POD best-fit coefficients unavailable and auto-fit failed: {exc}"
        )

    def atoms_builder(d):
        return make_blg_ab(d, a=_A0, sc=1) if stacking == "AB" else make_blg_aa(d, a=_A0, sc=1)

    def calc_builder(_atoms):
        return PODASECalculator(POD_BLG_HYPERPARAMS, params, elements=["C"], cutoff=6.0)

    d_min_scan, _ = rigid_scan_interlayer_minimum(atoms_builder, calc_builder, d_grid=_d_grid())
    dgrid = _d_grid()
    e_scan = np.empty_like(dgrid, dtype=float)
    for i, d in enumerate(dgrid):
        at = atoms_builder(float(d))
        at.calc = calc_builder(at)
        e_scan[i] = float(at.get_potential_energy())
    stem = f"pod_{stacking}_rigid_scan"
    csv_path = save_scan_csv(dgrid, e_scan, stem=stem)
    png_path = save_scan_plot(dgrid, e_scan, stem=stem, title=f"POD {stacking} rigid scan")
    print(f"[POD {stacking}] rigid scan: d_min={d_min_scan:.4f} Å, saved {csv_path}"
          + (f", {png_path}" if png_path else " (no matplotlib; no png)"))

    atoms0 = atoms_builder(d_min_scan + 0.2)
    atoms0.calc = calc_builder(atoms0)
    relaxed, _ = _assert_relax_reproducible(atoms0, seed=3)
    m = compute_layer_metrics(relaxed)
    print(f"[POD {stacking}] relaxed: d={m.separation:.4f} Å, "
          f"buckling std(z) layer0={m.buckling_layer0:.4f} Å layer1={m.buckling_layer1:.4f} Å")
    assert_no_buckling(m, max_std=0.02)

    if stacking == "AB":
        _assert_in_band(m.separation, _AB_BAND, "POD AB relaxed separation")
    else:
        _assert_in_band(m.separation, _AA_BAND, "POD AA relaxed separation")

    assert abs(m.separation - d_min_scan) <= 0.12, (
        f"Rigid-scan min d={d_min_scan:.4f} differs from relaxed d={m.separation:.4f}"
    )


@pytest.mark.parametrize("stacking", ["AB", "AA"])
@pytest.mark.parametrize("model", ["tersoff_kc", "tersoff_drip"])
def test_energy_per_atom_converges_with_supercell_size(model, stacking):
    """
    Energy per atom should be roughly constant as we increase supercell size.
    This is a sanity check for neighbor/PBC handling and hybrid overlay behavior.
    """
    if model == "tersoff_kc":
        te, kc = make_calc_tersoff_kc_best_fit_estimate()

        def mk_calc(atoms):
            return TersoffKolmogorovCrespiASECalculator(
                te.tolist(),
                kc.tolist(),
                kc_cutoff=14.0,
                layer_tags=layer_tags_from_mol_id(atoms),
            )

    else:
        te, drip = make_calc_tersoff_drip_best_fit_estimate()

        def mk_calc(atoms):
            ncut = 2.2
            return TersoffDRIPASECalculator(
                te.tolist(),
                drip.tolist(),
                ncut=ncut,
                layer_tags=layer_tags_from_mol_id(atoms),
            )

    # Use a near-equilibrium d to avoid large relaxation work in this test.
    d0 = 3.4 if stacking == "AB" else 3.6

    # For KC/DRIP, avoid tiny PBC cells that can destabilize normal construction.
    sc_list = (5, 6, 7) if model in ("tersoff_kc", "tersoff_drip") else (1, 2, 3)

    epa = []
    for sc in sc_list:
        atoms = (make_blg_ab(d0, a=_A0, sc=sc) if stacking == "AB" else make_blg_aa(d0, a=_A0, sc=sc))
        atoms.calc = mk_calc(atoms)
        e = float(atoms.get_potential_energy())
        epa.append(e / len(atoms))

    print(f"[{model} {stacking}] supercell E/N: " + ", ".join(
        f"sc={sc_list[i]} -> {epa[i]:.8f} eV/atom" for i in range(len(sc_list))
    ))

    # Compare to the largest cell (sc=3) reference.
    ref = float(epa[-1])
    tol = 5e-3  # eV/atom (5 meV/atom) – tune if needed per model.
    for i, val in enumerate(epa[:-1]):
        sc = sc_list[i]
        assert abs(float(val) - ref) <= tol, (
            f"{model} {stacking}: E/N(sc={sc})={val:.6f} differs from sc={sc_list[-1]} {ref:.6f} by > {tol}"
        )

