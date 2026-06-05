"""
test_calculators.py
-------------------
Tests for the ASE calculator wrappers (TersoffASECalculator,
KolmogorovCrespiASECalculator, DRIPASECalculator, PODASECalculator).

All calculator classes are now backed by the LAMMPS Python module
(``lammps_interface.py``).  Tests are skipped when the LAMMPS Python
module is not installed.

Run with: pytest tests/test_calculators.py -v
"""

import pytest
import numpy as np
from ase.build import bulk

# Re-use geometry helpers from test_interface
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interface import (
    diamond_cubic,
    graphene_bilayer,
    check_finite,
    check_shape,
)

# All calculators require the LAMMPS Python module.
try:
    import lammps as _lammps_module  # noqa: F401
    _LAMMPS_PY = True
except ImportError:
    _LAMMPS_PY = False

if not _LAMMPS_PY:
    pytest.skip(
        "LAMMPS Python module not available — install via `make install-python`.",
        allow_module_level=True,
    )


def _tersoff_params():
    """Erhart-Albe carbon parameters."""
    return [
        3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058,
        0.72751, 1.5724e-7, 2.2119, 346.74, 2.85, 0.15,
        3.4879, 1393.6,
    ]


def _kc_params():
    """Kolmogorov & Crespi graphene parameters."""
    return [3.34, 15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238]


def _drip_params():
    """Wen et al. DRIP graphene parameters."""
    return [
        15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238, 3.34, 0.0, 0.0,
        3.0, 14.0, 3.0,
    ]


# ── ASE Calculator tests (LAMMPS Python module) ───────────────────────────────


class TestTersoffASECalculator:
    """TersoffASECalculator backed by LAMMPS Python module"""

    def test_basic_compute(self):
        from blg_model_builder.potentials import TersoffASECalculator
        atoms = bulk("C", crystalstructure="diamond", a=3.567, cubic=True)
        calc = TersoffASECalculator(_tersoff_params())
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))
        assert forces.shape == (len(atoms), 3)


class TestKolmogorovCrespiASECalculator:
    """KolmogorovCrespiASECalculator backed by LAMMPS Python module"""

    def test_basic_compute(self):
        from blg_model_builder.potentials import KolmogorovCrespiASECalculator
        atoms, layers = _graphene_bilayer_ase(nx=7, ny=7)
        N = len(atoms)
        calc = KolmogorovCrespiASECalculator(_kc_params(), layer_tags=layers.tolist())
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))
        assert forces.shape == (N, 3)


class TestDRIPASECalculator:
    """DRIPASECalculator backed by LAMMPS Python module"""

    def test_basic_compute(self):
        from blg_model_builder.potentials import DRIPASECalculator
        atoms, layers = _graphene_bilayer_ase(nx=3, ny=3)
        N = len(atoms)
        params = _drip_params()[:8]  # eight fitted params; B, eta fixed in class
        calc = DRIPASECalculator(params, layer_tags=layers.tolist())
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))
        assert forces.shape == (N, 3)


class TestPODASECalculator:
    """PODASECalculator backed by LAMMPS Python module"""

    @pytest.fixture
    def si_hyperparams(self):
        return {
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

    def test_basic_compute(self, si_hyperparams):
        from blg_model_builder.potentials import PODASECalculator, ncoeff_from_params
        hp = dict(si_hyperparams, species=["Si"])
        nc = ncoeff_from_params(hp)
        coeffs = np.zeros(nc)
        atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
        calc = PODASECalculator(si_hyperparams, coeffs, ["Si"], 5.0)
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))
        assert forces.shape == (len(atoms), 3)


def _graphene_bilayer_ase(nx=3, ny=3, interlayer=3.35):
    """Build ASE Atoms for AB-stacked graphene bilayer with correct cell."""
    from ase import Atoms
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
    pos = np.array(layer0 + layer1, dtype=np.float64)
    cell = np.array([nx * a1, ny * a2, [0, 0, interlayer + 15.0]])
    layers = np.array([0] * len(layer0) + [1] * len(layer1), dtype=np.int32)
    return Atoms(
        symbols=["C"] * len(pos),
        positions=pos,
        cell=cell,
        pbc=[True, True, False],
    ), layers
