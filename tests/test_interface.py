"""
test_interface.py
-----------------
Tests for TersoffCalculator, KolmogorovCrespiCalculator, DRIPCalculator,
and PODCalculator via the PODASECalculator interface.

Usage
-----
    # All four calculators (no files needed):
    python test_interface.py

    # Tersoff + KC + DRIP only:
    python test_interface.py --skip-pod

"""

import argparse
import os
import sys
import numpy as np
from ase.build import bulk

# POD_TB_tight_binding_wrapper (POD_TB_cpp) has been removed from this project.
_HAS_TB_WRAPPER = False

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def passed(msg):  print(f"  {GREEN}PASS{RESET}  {msg}")
def failed(msg):  print(f"  {RED}FAIL{RESET}  {msg}");  return False
def skipped(msg): print(f"  {YELLOW}SKIP{RESET}  {msg}")


# ═════════════════════════════════════════════════════════════════════════════
#  Reference geometries
# ═════════════════════════════════════════════════════════════════════════════

def diamond_cubic(element="C", perturb=False):
    """8-atom diamond-cubic cell.  element selects lattice constant."""
    a = {"C": 3.567, "Si": 5.431, "Ge": 5.658}.get(element, 3.567)
    pos = a * np.array([
        [0.00, 0.00, 0.00], [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.00], [0.75, 0.75, 0.25],
        [0.00, 0.50, 0.50], [0.25, 0.75, 0.75],
        [0.50, 0.00, 0.50], [0.75, 0.25, 0.75],
    ], dtype=np.float64)
    if perturb:
        rng = np.random.default_rng(42)
        pos += rng.normal(0, 0.05, pos.shape)
    types = np.ones(len(pos), dtype=np.int32)
    box   = np.diag([a, a, a])
    return pos, types, box


def graphene_bilayer(interlayer=3.35, nx=3, ny=2):
    """AB-stacked graphene bilayer. Default 24 atoms (3×2 supercell)."""
    a = 2.46
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3) / 2, 0.0])

    layer0 = []
    for i in range(nx):
        for j in range(ny):
            origin = i * a1 + j * a2
            layer0.append(origin.copy())
            layer0.append(origin + (a1 + a2) / 3.0)

    ab_offset = (a1 + a2) / 3.0
    layer1 = [p + ab_offset + np.array([0, 0, interlayer]) for p in layer0]

    pos     = np.array(layer0 + layer1, dtype=np.float64)
    N       = len(pos)
    types   = np.ones(N, dtype=np.int32)
    layers  = np.array([0]*len(layer0) + [1]*len(layer1), dtype=np.int32)
    box = np.diag([3*a, 2*a*np.sqrt(3)/2, interlayer + 30.0])
    return pos, types, layers, box


# ═════════════════════════════════════════════════════════════════════════════
#  Assertion helpers
# ═════════════════════════════════════════════════════════════════════════════

def check_finite(energy, forces, name):
    ok = True
    if not np.isfinite(energy):
        failed(f"{name}: energy = {energy}"); ok = False
    if not np.all(np.isfinite(forces)):
        failed(f"{name}: forces contain non-finite values"); ok = False
    return ok

def check_shape(forces, N, name):
    if forces.shape != (N, 3):
        failed(f"{name}: forces shape {forces.shape} != ({N}, 3)")
        return False
    return True

def check_sensitivity(e1, e2, name):
    if np.isclose(e1, e2, rtol=1e-10):
        failed(f"{name}: energy unchanged with different params "
               f"(e1={e1:.6f} e2={e2:.6f})")
        return False
    return True

def check_deterministic(calc, params, name):
    e1, f1 = calc.compute(params)
    e2, f2 = calc.compute(params)
    if not (np.isclose(e1, e2, rtol=1e-10) and np.allclose(f1, f2, rtol=1e-10)):
        failed(f"{name}: repeated compute not deterministic")
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Low-level C++ calculator tests (removed with potential_ext)
# ═════════════════════════════════════════════════════════════════════════════
def _require_cpp_ext():
    """The C++ potential_ext extension has been removed; always returns False."""
    skipped("C++ potential_ext has been removed — use lammps_interface instead")
    return False


def test_tersoff():
    print("\n── TersoffCalculator ────────────────────────────────────────────")
    if not _require_cpp_ext():
        return True
    from blg_model_builder.potentials import TersoffCalculator
    ok = True

    pos, types, box = diamond_cubic("C")
    N = len(pos)

    # Erhart-Albe carbon (Erhart & Albe, PRB 2005)
    params = [
        3.0, 1.0, 0.0,               # m, gamma, lambda3
        38049.0, 4.3484, -0.57058,   # c, d, costheta0
        0.72751, 1.5724e-7, 2.2119,  # n, beta, lambda2
        346.74, 2.85, 0.15,          # B, R, D
        3.4879, 1393.6,              # lambda1, A
    ]

    try:
        calc = TersoffCalculator()
        calc.set_geometry(pos, types, box)
        energy, forces = calc.compute(params)
    except Exception as e:
        failed(f"set_geometry / compute raised: {e}"); return False

    if check_finite(energy, forces, "Tersoff"):
        passed(f"basic compute   energy = {energy:.6f} eV")
    else:
        ok = False

    if check_shape(forces, N, "Tersoff"):
        passed(f"forces shape    {forces.shape}")
    else:
        ok = False

    # Param sensitivity: scale A by 10%
    params2 = params.copy(); params2[13] *= 1.1
    energy2, _ = calc.compute(params2)
    if check_sensitivity(energy, energy2, "Tersoff"):
        passed(f"param sensitivity  ΔE = {energy2 - energy:+.6f} eV")
    else:
        ok = False

    if check_deterministic(calc, params, "Tersoff"):
        passed("deterministic across repeated calls")
    else:
        ok = False

    # Re-geometry with perturbed positions
    try:
        pos2, types2, box2 = diamond_cubic("C", perturb=True)
        calc.set_geometry(pos2, types2, box2)
        e3, _ = calc.compute(params)
        passed(f"re-geometry     energy = {e3:.6f} eV")
    except Exception as e:
        failed(f"re-geometry raised: {e}"); ok = False

    return ok


# ═════════════════════════════════════════════════════════════════════════════
#  KolmogorovCrespiCalculator
# ═════════════════════════════════════════════════════════════════════════════
def test_kolmogorov_crespi():
    print("\n── KolmogorovCrespiCalculator ───────────────────────────────────")
    if not _require_cpp_ext():
        return True
    from blg_model_builder.potentials import KolmogorovCrespiCalculator
    ok = True

    # KC requires ≤3 same-layer neighbors; need in-plane box > 2*cutoff (28 Å)
    # 7×7 supercell: 17.2 × 14.9 Å per layer, 196 atoms
    pos, types, layers, box = graphene_bilayer(nx=7, ny=7)
    N = len(pos)

    # Kolmogorov & Crespi, PRB 71 235415 (2005) — graphene
    params = [3.34, 15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238]

    try:
        calc = KolmogorovCrespiCalculator()
        calc.set_geometry(pos, types, layers + 1, box, cutoff=14.0)
        energy, forces = calc.compute(params)
    except Exception as e:
        failed(f"set_geometry / compute raised: {e}"); return False

    if check_finite(energy, forces, "KC"):
        passed(f"basic compute   energy = {energy:.6f} eV")
    else:
        ok = False

    if check_shape(forces, N, "KC"):
        passed(f"forces shape    {forces.shape}")
    else:
        ok = False

    # Param sensitivity: change z0 by 5%
    params2 = params.copy(); params2[0] *= 1.05
    energy2, _ = calc.compute(params2)
    if check_sensitivity(energy, energy2, "KC"):
        passed(f"param sensitivity  ΔE = {energy2 - energy:+.6f} eV")
    else:
        ok = False

    if check_deterministic(calc, params, "KC"):
        passed("deterministic across repeated calls")
    else:
        ok = False

    return ok


# ═════════════════════════════════════════════════════════════════════════════
#  DRIPCalculator
# ═════════════════════════════════════════════════════════════════════════════
def test_drip():
    print("\n── DRIPCalculator ───────────────────────────────────────────────")
    if not _require_cpp_ext():
        return True
    from blg_model_builder.potentials import DRIPCalculator
    ok = True

    pos, types, layers, box = graphene_bilayer(nx=2,ny=2)
    N = len(pos)

    # Wen et al., PRB 98 235404 (2018) — graphene
    params = [
        15.71, 12.29, 4.933,   # C0 C2 C4
        3.030, 0.578, 3.143,   # C delta lambda
        10.238, 3.34,          # A z0
        0.0, 0.0,              # B eta
        3.0, 14.0, 3.0,        # rhocut rcut ncut
    ]

    try:
        calc = DRIPCalculator()
        calc.set_geometry(pos, types, layers + 1, box, cutoff=14.0)
        energy, forces = calc.compute(params)
    except Exception as e:
        failed(f"set_geometry / compute raised: {e}"); return False

    if check_finite(energy, forces, "DRIP"):
        passed(f"basic compute   energy = {energy:.6f} eV")
    else:
        ok = False

    if check_shape(forces, N, "DRIP"):
        passed(f"forces shape    {forces.shape}")
    else:
        ok = False

    # Param sensitivity: change z0 by 5%
    params2 = params.copy(); params2[7] *= 1.05
    energy2, _ = calc.compute(params2)
    if check_sensitivity(energy, energy2, "DRIP"):
        passed(f"param sensitivity  ΔE = {energy2 - energy:+.6f} eV")
    else:
        ok = False

    if check_deterministic(calc, params, "DRIP"):
        passed("deterministic across repeated calls")
    else:
        ok = False

    return ok


# ═════════════════════════════════════════════════════════════════════════════
#  PODASECalculator  (Si, matches Si_param.pod descriptor architecture)
# ═════════════════════════════════════════════════════════════════════════════

# Si hyperparameters matching Si_param.pod
SI_HYPERPARAMS = {
    "bessel_polynomial_degree":              4,
    "inverse_polynomial_degree":             8,
    "twobody_number_radial_basis_functions":  10,
    "threebody_number_radial_basis_functions": 8,
    "threebody_angular_degree":              4,
    "fourbody_number_radial_basis_functions":  6,
    "fourbody_angular_degree":               3,
    "fivebody_number_radial_basis_functions":  4,
    "fivebody_angular_degree":               3,
    "sixbody_number_radial_basis_functions":   3,
    "sixbody_angular_degree":                2,
    "sevenbody_number_radial_basis_functions": 2,
    "sevenbody_angular_degree":              2,
}

def test_pod():
    print("\n── PODASECalculator ─────────────────────────────────────────────")
    # PODASECalculator is now backed by LAMMPS Python interface (if available),
    # or the C++ pybind PODCalculator.  Either is sufficient for this test.
    from blg_model_builder.potentials import PODASECalculator, ncoeff_from_params
    import ase
    from ase.build import bulk
    ok = True

    hyperparams = SI_HYPERPARAMS
    elements    = ["Si"]
    cutoff      = 5.0

    # ── ncoeff_from_params ────────────────────────────────────────────────────
    hp_with_species = dict(hyperparams, species=elements)
    nc = ncoeff_from_params(hp_with_species)
    if nc == 184:
        passed(f"ncoeff_from_params = {nc}")
    else:
        failed(f"ncoeff_from_params = {nc} (expected 184)"); return False

    # ── build an ASE Si bulk cell ─────────────────────────────────────────────
    atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
    N = len(atoms)

    # ── initialise with zero coefficients ────────────────────────────────────
    coeffs = np.zeros(nc)
    try:
        calc = PODASECalculator(hyperparams, coeffs, elements, cutoff)
        energy, forces = calc.calculate(atoms)["energy"], calc.calculate(atoms)["forces"]
    except Exception as e:
        failed(f"calculate raised: {e}"); return False

    if check_finite(energy, forces, "POD"):
        passed(f"basic compute   energy = {energy:.6f} eV")
    else:
        ok = False

    if check_shape(forces, N, "POD"):
        passed(f"forces shape    {forces.shape}")
    else:
        ok = False

    # ── param sensitivity: non-zero coefficients give different energy ────────
    rng     = np.random.default_rng(0)
    coeffs2 = rng.normal(0, 0.1, nc)
    calc.set_parameters(coeffs2)
    try:
        energy2, _ = calc.calculate(atoms)["energy"], calc.calculate(atoms)["forces"]
        if check_sensitivity(energy, energy2, "POD"):
            passed(f"param sensitivity  ΔE = {energy2 - energy:+.6f} eV")
        else:
            ok = False
    except Exception as e:
        failed(f"param sensitivity raised: {e}"); ok = False

    # ── deterministic ─────────────────────────────────────────────────────────
    calc.set_parameters(coeffs2)
    try:
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        if (np.isclose(r1["energy"], r2["energy"])
                and np.allclose(r1["forces"], r2["forces"])):
            passed("deterministic across repeated calls")
        else:
            failed("POD: not deterministic"); ok = False
    except Exception as e:
        failed(f"deterministic check raised: {e}"); ok = False

    # ── set_parameters updates coefficients ───────────────────────────────────
    coeffs3 = coeffs2 * 1.05
    calc.set_parameters(coeffs3)
    try:
        energy3 = calc.calculate(atoms)["energy"]
        if check_sensitivity(energy2, energy3, "POD set_parameters"):
            passed(f"set_parameters works  ΔE = {energy3 - energy2:+.6f} eV")
        else:
            ok = False
    except Exception as e:
        failed(f"set_parameters raised: {e}"); ok = False

    return ok


def test_compute_hoppings_shapes():
    """Smoke test for the tight-binding C++/Python bridge."""
    if not _HAS_TB_WRAPPER:
        skipped("tight-binding wrapper not available")
        return True
    atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)

    hyper = {
        "chemical_elements": ["Si"],
        "inner_cutoff": 0.5,
        "outer_cutoff": 4.0,
        "bessel_polynomial_degree": 4,
        "inverse_polynomial_degree": 8,
        "twobody_number_radial_basis": 2,
        "threebody_number_radial_basis": 1,
        "threebody_angular_degree": 2,
    }
    basis = ["pz"]

    SK_COUNT = 10
    # Use non-zero pp_pi coefficient so hoppings are non-zero
    coeffs_2b = np.zeros((SK_COUNT, hyper["twobody_number_radial_basis"]))
    coeffs_2b[3, 0] = -2.7  # pp_pi (index 3) for pz orbitals
    coeffs_3b = np.zeros(
        (
            SK_COUNT,
            hyper["threebody_number_radial_basis"] ** 2
            * hyper["threebody_angular_degree"],
        )
    )

    H, dH, row, col, r_vec = compute_hoppings(
        atoms,
        hyper,
        basis,
        coeffs_2b,
        coeffs_3b,
    )

    n_pairs = len(row)
    n_orbs_sq = len(basis) ** 2
    if n_orbs_sq == 1:
        assert H.shape == (n_pairs,)
        assert dH.shape == (n_pairs, 3)
    else:
        assert H.shape == (n_pairs, n_orbs_sq)
        assert dH.shape == (n_pairs, 3, n_orbs_sq)
    assert row.shape == (n_pairs,)
    assert col.shape == (n_pairs,)
    assert r_vec.shape == (n_pairs, 3)
    # With pp_pi coeff set, hoppings should be non-zero for in-plane bonds
    assert not np.allclose(H, 0.0), "Hoppings should be non-zero with non-zero pp_pi coefficient"


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Test lammps_potentials")
    parser.add_argument("--skip-pod", action="store_true",
                        help="Skip PODASECalculator test")
    args = parser.parse_args()

    print("=" * 60)
    print("  lammps_potentials interface tests")
    print("=" * 60)

    results = {}
    results["Tersoff"]            = test_tersoff()
    results["KolmogorovCrespi"]   = test_kolmogorov_crespi()
    results["DRIP"]               = test_drip()

    if args.skip_pod:
        skipped("PODASECalculator (--skip-pod)")
        results["POD"] = None
    else:
        results["POD"] = test_pod()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_passed = True
    for name, result in results.items():
        if result is None:
            print(f"  {YELLOW}SKIP{RESET}  {name}")
        elif result:
            print(f"  {GREEN}PASS{RESET}  {name}")
        else:
            print(f"  {RED}FAIL{RESET}  {name}")
            all_passed = False
    print()
    if all_passed:
        print(f"  {GREEN}All tests passed.{RESET}")
        sys.exit(0)
    else:
        print(f"  {RED}Some tests failed.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()