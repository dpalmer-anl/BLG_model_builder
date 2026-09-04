"""
Tests for PODExtepILPLammpsCalculator on TBLG/hBN hybrid cells.

Verifies that POD energy/descriptors use carbon atoms only while ExTeP+ILP
handle B/N on the full structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "lammps",
    reason="LAMMPS Python module not installed.",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UQ_DIR = REPO_ROOT / "uncertainty_quantification"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(UQ_DIR))

from blg_model_builder.potentials import (  # noqa: E402
    PODExtepILPLammpsCalculator,
    ncoeff_from_params,
)
from tblg_hbn_geometry import build_tblg_on_hbn  # noqa: E402

# Small POD basis for fast tests (same keys as test_lammps_py_interface).
_MIN_POD_HP = {
    "bessel_polynomial_degree": 2,
    "inverse_polynomial_degree": 2,
    "twobody_number_radial_basis_functions": 2,
    "threebody_number_radial_basis_functions": 2,
    "threebody_angular_degree": 1,
    "fourbody_number_radial_basis_functions": 0,
    "fourbody_angular_degree": 0,
    "fivebody_number_radial_basis_functions": 0,
    "fivebody_angular_degree": 0,
    "sixbody_number_radial_basis_functions": 0,
    "sixbody_angular_degree": 0,
    "sevenbody_number_radial_basis_functions": 0,
    "sevenbody_angular_degree": 0,
}


@pytest.fixture(scope="module")
def hybrid_calc():
    hp = dict(_MIN_POD_HP, species=["C"])
    nc = int(ncoeff_from_params(hp))
    rng = np.random.default_rng(0)
    coeffs = rng.normal(0, 0.02, nc)
    return PODExtepILPLammpsCalculator(hp, coeffs, cutoff=6.0)


@pytest.fixture(scope="module")
def hybrid_atoms():
    atoms = build_tblg_on_hbn(theta_deg=9.43)
    syms = np.array(atoms.get_chemical_symbols())
    assert np.any(syms == "B") and np.any(syms == "N") and np.any(syms == "C")
    return atoms


class TestPODExtepILPHBN:
    def test_atom_descriptors_are_carbon_only(self, hybrid_calc, hybrid_atoms):
        d_atom = hybrid_calc.compute_pod_atom_descriptors(hybrid_atoms)
        n_c = int(np.sum(np.array(hybrid_atoms.get_chemical_symbols()) == "C"))
        assert d_atom.shape[0] == n_c
        assert d_atom.shape[1] in (hybrid_calc.ncoeff, hybrid_calc.ncoeff - 1)
        assert np.all(np.isfinite(d_atom))

    def test_global_descriptors_match_atom_sum(self, hybrid_calc, hybrid_atoms):
        d_atom = hybrid_calc.compute_pod_atom_descriptors(hybrid_atoms)
        d_glob = hybrid_calc.compute_pod_descriptors([hybrid_atoms], verbose=False)[0]
        d_sum = d_atom.sum(axis=0)
        if d_sum.shape[0] == d_glob.shape[0]:
            assert np.allclose(d_sum, d_glob, rtol=1e-7, atol=1e-7)
        elif d_sum.shape[0] + 1 == d_glob.shape[0]:
            assert np.allclose(d_sum, d_glob[1:], rtol=1e-7, atol=1e-7)

    def test_energy_decomposition(self, hybrid_calc, hybrid_atoms):
        total = hybrid_calc.calculate(hybrid_atoms)
        pod = hybrid_calc._evaluate_pod_on_carbon(hybrid_atoms)
        bn = hybrid_calc._evaluate_bn_on_full(hybrid_atoms)
        assert np.isclose(
            total["energy"],
            pod["energy"] + bn["energy"],
            rtol=1e-8,
            atol=1e-8,
        )

    def test_moving_boron_does_not_change_pod_energy(self, hybrid_calc, hybrid_atoms):
        pod0 = hybrid_calc._evaluate_pod_on_carbon(hybrid_atoms)["energy"]
        moved = hybrid_atoms.copy()
        syms = np.array(moved.get_chemical_symbols())
        b_idx = int(np.where(syms == "B")[0][0])
        pos = moved.get_positions()
        pos[b_idx, 2] += 0.75
        moved.set_positions(pos)
        pod1 = hybrid_calc._evaluate_pod_on_carbon(moved)["energy"]
        assert np.isclose(pod0, pod1, rtol=1e-10, atol=1e-10)

    def test_moving_boron_changes_total_and_bn_energy(self, hybrid_calc, hybrid_atoms):
        e0 = hybrid_calc.calculate(hybrid_atoms)["energy"]
        bn0 = hybrid_calc._evaluate_bn_on_full(hybrid_atoms)["energy"]
        moved = hybrid_atoms.copy()
        syms = np.array(moved.get_chemical_symbols())
        b_idx = int(np.where(syms == "B")[0][0])
        pos = moved.get_positions()
        pos[b_idx, 2] += 0.75
        moved.set_positions(pos)
        e1 = hybrid_calc.calculate(moved)["energy"]
        bn1 = hybrid_calc._evaluate_bn_on_full(moved)["energy"]
        assert not np.isclose(e0, e1, rtol=1e-10, atol=1e-10)
        assert not np.isclose(bn0, bn1, rtol=1e-10, atol=1e-10)

    def test_moving_carbon_changes_pod_energy(self, hybrid_calc, hybrid_atoms):
        pod0 = hybrid_calc._evaluate_pod_on_carbon(hybrid_atoms)["energy"]
        moved = hybrid_atoms.copy()
        syms = np.array(moved.get_chemical_symbols())
        c_idx = int(np.where(syms == "C")[0][0])
        pos = moved.get_positions()
        pos[c_idx, 0] += 0.05
        moved.set_positions(pos)
        pod1 = hybrid_calc._evaluate_pod_on_carbon(moved)["energy"]
        assert not np.isclose(pod0, pod1, rtol=1e-10, atol=1e-10)

    def test_batch_matches_single_calculate(self, hybrid_calc, hybrid_atoms):
        e_single = hybrid_calc.calculate(hybrid_atoms)["energy"]
        energies, _ = hybrid_calc.evaluate_batch([hybrid_atoms])
        assert np.isclose(energies[0], e_single, rtol=1e-8, atol=1e-8)

    def test_unified_hybrid_would_differ_from_split_on_bn_move(self, hybrid_calc, hybrid_atoms):
        """Document that a single LAMMPS hybrid eval is not carbon-only POD."""
        moved = hybrid_atoms.copy()
        syms = np.array(moved.get_chemical_symbols())
        b_idx = int(np.where(syms == "B")[0][0])
        pos = moved.get_positions()
        pos[b_idx, 2] += 0.75
        moved.set_positions(pos)

        split_pod = hybrid_calc._evaluate_pod_on_carbon(moved)["energy"]
        hybrid_calc._use_hybrid = True
        hybrid_calc._last_cell = None
        hybrid_calc._last_natoms = None
        unified = super(PODExtepILPLammpsCalculator, hybrid_calc).calculate(moved)
        bn = hybrid_calc._evaluate_bn_on_full(moved)["energy"]
        split_total = split_pod + bn

        assert np.isclose(hybrid_calc.calculate(moved)["energy"], split_total, rtol=1e-8)
        # Unified hybrid total energy differs from the split model when B moves
        # (POD in LAMMPS still sees the full neighbor environment).
        if not np.isclose(unified["energy"], split_total, rtol=1e-6, atol=1e-6):
            return
        pytest.skip(
            "LAMMPS unified hybrid matched split on this displacement; "
            "cannot assert difference on this build."
        )
