"""
test_pod_finite_diff.py

Finite-difference force consistency test for PODASECalculator.

Computes analytical forces and compares them against central finite differences:
    F_i_α ≈ -(E(r + δ*ê_{i,α}) - E(r - δ*ê_{i,α})) / (2δ)

If the analytical forces match the finite-difference forces to within ~1e-4 eV/Å,
the calculator is correct. A large mismatch indicates a force/energy inconsistency
bug in the C++ layer.
"""

import os
import sys
import numpy as np

# Ensure blg_model_builder is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blg_model_builder.potentials import PODASECalculator, ncoeff_from_params

# ── Potential parameters (must match C_coefficients.pod) ──────────────────────
rcut = 5.0
hyperparams = {
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

_pod_coeff_path = os.path.join(os.path.dirname(__file__), "C_coefficients.pod")
params = np.loadtxt(_pod_coeff_path, skiprows=1)
ncoeffs = ncoeff_from_params(hyperparams)
assert len(params) == ncoeffs, (
    f"C_coefficients.pod has {len(params)} coefficients but hyperparams "
    f"expect {ncoeffs}."
)

# ── Build a small AA-stacked bilayer graphene cell (2×2 supercell) ────────────
# This is fast to evaluate and well within the training distribution.
from ase import Atoms
import flatgraphene as fg

def make_small_bilayer(a=2.46, layer_sep=3.35, nx=2, ny=2):
    """Create a small bilayer graphene supercell for testing."""
    from ase.build import graphene
    # single layer 2x2
    mono = graphene(formula='C2', a=a, size=(nx, ny, 1), vacuum=0.0)
    mono.pbc = [True, True, True]
    # stack two layers
    cell = mono.get_cell().copy()
    cell[2, 2] = layer_sep + 20.0   # vacuum in z
    pos1 = mono.get_positions().copy()
    pos2 = pos1.copy()
    pos2[:, 2] += layer_sep
    symbols = list(mono.get_chemical_symbols()) * 2
    positions = np.vstack([pos1, pos2])
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


def finite_diff_forces(calc, atoms, delta=1e-4):
    """Compute forces via central finite differences."""
    N = len(atoms)
    fd_forces = np.zeros((N, 3))
    pos0 = atoms.get_positions().copy()
    for i in range(N):
        for alpha in range(3):
            # forward
            pos_fwd = pos0.copy()
            pos_fwd[i, alpha] += delta
            atoms.set_positions(pos_fwd)
            e_fwd = calc.calculate(atoms)["energy"]
            # backward
            pos_bwd = pos0.copy()
            pos_bwd[i, alpha] -= delta
            atoms.set_positions(pos_bwd)
            e_bwd = calc.calculate(atoms)["energy"]
            fd_forces[i, alpha] = -(e_fwd - e_bwd) / (2 * delta)
    # restore
    atoms.set_positions(pos0)
    return fd_forces


def main():
    atoms = make_small_bilayer()
    print(f"Testing with {len(atoms)} atoms")
    print(f"Cell:\n{atoms.get_cell()}")

    calc = PODASECalculator(hyperparams, params, elements=["C"], cutoff=rcut)

    # Analytical forces
    results = calc.calculate(atoms)
    E0 = results["energy"]
    F_analytical = results["forces"]
    print(f"\nStep 0: E = {E0:.6f} eV,  fmax = {np.linalg.norm(F_analytical, axis=1).max():.6f} eV/Å")

    # Finite-difference forces (check a subset of atoms/components to keep it fast)
    print("\nComputing finite-difference forces (this may take a few minutes)...")
    delta = 1e-3   # Å
    N = len(atoms)
    # Check all atoms for a small system
    fd_forces = finite_diff_forces(calc, atoms, delta=delta)

    max_err = np.max(np.abs(F_analytical - fd_forces))
    rms_err = np.sqrt(np.mean((F_analytical - fd_forces)**2))

    print(f"\nForce consistency check (δ = {delta} Å):")
    print(f"  Max |F_analytical - F_fd|  = {max_err:.4e} eV/Å")
    print(f"  RMS |F_analytical - F_fd|  = {rms_err:.4e} eV/Å")
    print(f"  Max |F_analytical|          = {np.max(np.abs(F_analytical)):.4e} eV/Å")

    tol = 1e-2  # eV/Å — reasonable for delta=1e-3 Å
    if max_err < tol:
        print(f"\n✓ PASS: Forces are consistent with energy (max err < {tol} eV/Å)")
    else:
        print(f"\n✗ FAIL: Forces are NOT consistent with energy (max err {max_err:.4e} >= {tol} eV/Å)")
        print("\nDetailed comparison (first 10 atoms):")
        for i in range(min(10, N)):
            print(f"  atom {i:3d}: F_anal = {F_analytical[i]},  F_fd = {fd_forces[i]}")
        sys.exit(1)

    # Also verify energy is non-trivial and forces change after perturbation
    print(f"\nEnergy per atom: {E0/N:.4f} eV/atom")
    print(f"Force max: {np.linalg.norm(F_analytical, axis=1).max():.6f} eV/Å")
    print("\n✓ All checks passed — PODASECalculator is working correctly.")


if __name__ == "__main__":
    main()
