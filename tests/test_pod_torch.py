"""
test_pod_torch.py
-----------------
DEPRECATED: PODTorchCalculator and precompute_pod_descriptors depended on the
C++ potential_ext extension (descriptor Jacobians via compute_descriptors()),
which has been removed.

These tests are skipped entirely.  For POD energy / force evaluation in MCMC,
use PODLammpsCalculator (backed by the LAMMPS Python module) with
prepare_batch / evaluate_batch.
"""

import pytest
pytest.skip(
    "PODTorchCalculator and precompute_pod_descriptors have been removed "
    "(depended on C++ potential_ext); use PODLammpsCalculator instead.",
    allow_module_level=True,
)

import os
import sys
import traceback
import numpy as np

from blg_model_builder_v2.potentials import (
    PODASECalculator,
    ncoeff_from_params,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_POD_COEFF_PATH = os.path.join(_HERE, "C_coefficients.pod")

_HYPERPARAMS = {
    "species": ["C"],
    "bessel_polynomial_degree": 4,
    "inverse_polynomial_degree": 8,
    "twobody_number_radial_basis_functions": 10,
    "threebody_number_radial_basis_functions": 8,
    "threebody_angular_degree": 4,
    "fourbody_number_radial_basis_functions": 6,
    "fourbody_angular_degree": 3,
    "fivebody_number_radial_basis_functions": 4,
    "fivebody_angular_degree": 3,
    "sixbody_number_radial_basis_functions": 3,
    "sixbody_angular_degree": 2,
    "sevenbody_number_radial_basis_functions": 2,
    "sevenbody_angular_degree": 2,
}
_RCUT     = 5.0
_ELEMENTS = ["C"]

_ATOL_E = 1e-10   # eV
_ATOL_F = 1e-9    # eV/Å

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_params():
    coeffs   = np.loadtxt(_POD_COEFF_PATH, skiprows=1, dtype=np.float64)
    expected = ncoeff_from_params(_HYPERPARAMS)
    assert len(coeffs) == expected, (
        f"C_coefficients.pod has {len(coeffs)} coefficients but hyperparams "
        f"expect {expected}. Check that _HYPERPARAMS matches the .pod file."
    )
    return coeffs


def _graphene_atoms(a: float = 2.46, vacuum: float = 20.0):
    """2-atom graphene unit cell, first lattice vector along x."""
    import ase
    cell = np.array([
        [a,           0.0,                   0.0],
        [-a / 2.0,    a * np.sqrt(3) / 2.0,  0.0],
        [0.0,         0.0,                   vacuum],
    ])
    positions = np.array([
        [0.0,      0.0,                     vacuum / 2],
        [a / 2.0,  a / (2.0 * np.sqrt(3)), vacuum / 2],
    ])
    return ase.Atoms("C2", positions=positions, cell=cell, pbc=[True, True, False])


def _rotated_graphene_atoms(angle_deg: float = 30.0):
    """Graphene rotated so the first cell vector is NOT along x."""
    atoms = _graphene_atoms()
    atoms.rotate(angle_deg, "z", rotate_cell=True)
    return atoms


# ---------------------------------------------------------------------------
# Minimal test runner
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []   # (name, passed, message)

def _run(name: str, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception:
        msg = traceback.format_exc()
        _results.append((name, False, msg))
        print(f"  FAIL  {name}")
        # Print the last line of the traceback (the assertion message) for quick reading
        last = [l for l in msg.strip().splitlines() if l.strip()][-1]
        print(f"        {last}")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_energy_aligned_cell():
    """Energy must match for a standard axis-aligned graphene cell."""
    params = _load_params()
    atoms  = _graphene_atoms()

    e_ase        = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_potential_energy(atoms)
    e_torch, _   = PODTorchCalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).forward(atoms)

    assert np.isclose(e_ase, float(e_torch), atol=_ATOL_E), (
        f"ASE={e_ase:.12f}  Torch={float(e_torch):.12f}"
    )


def test_energy_rotated_cell():
    """Energy must match for a 30°-rotated graphene cell (energy is frame-independent)."""
    params = _load_params()
    atoms  = _rotated_graphene_atoms(30.0)

    e_ase      = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_potential_energy(atoms)
    e_torch, _ = PODTorchCalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).forward(atoms)

    assert np.isclose(e_ase, float(e_torch), atol=_ATOL_E), (
        f"ASE={e_ase:.12f}  Torch={float(e_torch):.12f}"
    )


def test_forces_aligned_cell():
    """Forces must match for an axis-aligned graphene cell."""
    params = _load_params()
    atoms  = _graphene_atoms()

    f_ase      = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_forces(atoms)
    _, f_torch = PODTorchCalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).forward(atoms)
    f_torch_np = f_torch.detach().numpy()

    max_err = np.abs(f_ase - f_torch_np).max()
    assert np.allclose(f_ase, f_torch_np, atol=_ATOL_F), (
        f"max |diff| = {max_err:.2e} eV/Å"
    )


def test_forces_rotated_cell():
    """Forces must match for a 30°-rotated cell.

    Regression test for the LAMMPS→ASE force-frame bug: compute_descriptors
    returns derivatives in the LAMMPS lower-triangular frame; without the
    rotation back to the ASE frame the forces would be wrong here.
    """
    params = _load_params()
    atoms  = _rotated_graphene_atoms(30.0)

    f_ase      = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_forces(atoms)
    _, f_torch = PODTorchCalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).forward(atoms)
    f_torch_np = f_torch.detach().numpy()

    max_err = np.abs(f_ase - f_torch_np).max()
    assert np.allclose(f_ase, f_torch_np, atol=_ATOL_F), (
        f"max |diff| = {max_err:.2e} eV/Å\n"
        f"ASE:\n{f_ase}\nTorch:\n{f_torch_np}"
    )


def test_precomputed_energy_aligned():
    """dot(params, precomputed_desc) must equal LAMMPS energy (aligned cell)."""
    params = _load_params()
    atoms  = _graphene_atoms()

    e_ase = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_potential_energy(atoms)

    desc_list, _ = precompute_pod_descriptors([atoms], _HYPERPARAMS, _ELEMENTS, _RCUT, params=params)
    e_pre = float(np.dot(params, desc_list[0]))

    assert np.isclose(e_ase, e_pre, atol=_ATOL_E), (
        f"ASE={e_ase:.12f}  precomputed={e_pre:.12f}"
    )


def test_precomputed_energy_rotated():
    """dot(params, precomputed_desc) must equal LAMMPS energy (rotated cell)."""
    params = _load_params()
    atoms  = _rotated_graphene_atoms(30.0)

    e_ase = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_potential_energy(atoms)

    desc_list, _ = precompute_pod_descriptors([atoms], _HYPERPARAMS, _ELEMENTS, _RCUT, params=params)
    e_pre = float(np.dot(params, desc_list[0]))

    assert np.isclose(e_ase, e_pre, atol=_ATOL_E), (
        f"ASE={e_ase:.12f}  precomputed={e_pre:.12f}"
    )


def test_precomputed_forces_aligned():
    """Precomputed-path forces must match LAMMPS (aligned cell)."""
    params = _load_params()
    atoms  = _graphene_atoms()

    f_ase = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_forces(atoms)

    _, deriv_list = precompute_pod_descriptors([atoms], _HYPERPARAMS, _ELEMENTS, _RCUT, params=params)
    f_pre = -np.einsum("m,mia->ia", params, deriv_list[0])

    max_err = np.abs(f_ase - f_pre).max()
    assert np.allclose(f_ase, f_pre, atol=_ATOL_F), (
        f"max |diff| = {max_err:.2e} eV/Å"
    )


def test_precomputed_forces_rotated():
    """Precomputed-path forces must match LAMMPS (rotated cell).

    Verifies that precompute_pod_descriptors correctly rotates derivatives
    from the LAMMPS frame to the ASE frame.
    """
    params = _load_params()
    atoms  = _rotated_graphene_atoms(30.0)

    f_ase = PODASECalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT).get_forces(atoms)

    _, deriv_list = precompute_pod_descriptors([atoms], _HYPERPARAMS, _ELEMENTS, _RCUT, params=params)
    f_pre = -np.einsum("m,mia->ia", params, deriv_list[0])

    max_err = np.abs(f_ase - f_pre).max()
    assert np.allclose(f_ase, f_pre, atol=_ATOL_F), (
        f"max |diff| = {max_err:.2e} eV/Å\n"
        f"ASE:\n{f_ase}\nPrecomputed:\n{f_pre}"
    )


def test_set_parameters_updates_energy():
    """set_parameters must change the computed energy."""
    params = _load_params()
    atoms  = _graphene_atoms()

    calc = PODTorchCalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT)
    e1, _ = calc.forward(atoms)

    rng = np.random.default_rng(0)
    perturbed = params + rng.normal(scale=0.01, size=len(params))
    calc.set_parameters(perturbed)
    e2, _ = calc.forward(atoms)

    assert not np.isclose(float(e1), float(e2)), (
        "set_parameters had no effect on the computed energy."
    )


def test_atoms_vs_precomputed_consistency():
    """Atoms path and precomputed path must agree on both energy and forces."""
    params = _load_params()
    atoms  = _rotated_graphene_atoms(45.0)

    calc = PODTorchCalculator(_HYPERPARAMS, params.tolist(), _ELEMENTS, _RCUT)

    e_atoms, f_atoms = calc.forward(atoms)

    desc_list, deriv_list = precompute_pod_descriptors(
        [atoms], _HYPERPARAMS, _ELEMENTS, _RCUT, params=params
    )
    e_pre, f_pre = calc.forward((desc_list[0], deriv_list[0]))

    assert np.isclose(float(e_atoms), float(e_pre), atol=_ATOL_E), (
        f"Energy: atoms={float(e_atoms):.12f}  precomputed={float(e_pre):.12f}"
    )
    assert np.allclose(f_atoms.detach().numpy(), f_pre.detach().numpy(), atol=_ATOL_F), (
        "Forces differ between atoms path and precomputed path."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_energy_aligned_cell,
        test_energy_rotated_cell,
        test_forces_aligned_cell,
        test_forces_rotated_cell,
        test_precomputed_energy_aligned,
        test_precomputed_energy_rotated,
        test_precomputed_forces_aligned,
        test_precomputed_forces_rotated,
        test_set_parameters_updates_energy,
        test_atoms_vs_precomputed_consistency,
    ]

    print(f"\nRunning {len(tests)} tests\n" + "-" * 50)
    for fn in tests:
        _run(fn.__name__, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print("-" * 50)
    print(f"{passed} passed, {failed} failed\n")

    if failed:
        print("Failures\n" + "=" * 50)
        for name, ok, msg in _results:
            if not ok:
                print(f"\n--- {name} ---")
                print(msg)
        sys.exit(1)
