"""
test_lammps_py_interface.py
---------------------------
Tests for all calculators in ``blg_model_builder.lammps_interface``.

These tests require the LAMMPS Python module (``from lammps import lammps``)
to be importable.  Build LAMMPS and run ``make install-python`` (or the CMake
equivalent) to make it available, then activate the conda environment.

Run with::

    pytest tests/test_lammps_py_interface.py -v

"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if the lammps Python module is not installed.
lammps = pytest.importorskip(
    "lammps",
    reason="LAMMPS Python module not installed.  Run 'make install-python'.",
)

# Also skip if the lammps_interface module itself fails to import.
lammps_interface = pytest.importorskip(
    "blg_model_builder.lammps_interface",
    reason="blg_model_builder.lammps_interface not importable.",
)

from blg_model_builder.lammps_interface import (
    TersoffLammpsCalculator,
    KolmogorovCrespiLammpsCalculator,
    DRIPLammpsCalculator,
    TersoffKCLammpsCalculator,
    TersoffDRIPLammpsCalculator,
    PODLammpsCalculator,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Reference geometries (ASE Atoms)
# ═══════════════════════════════════════════════════════════════════════════════

def _diamond_cubic_atoms(element: str = "C", perturb: bool = False):
    """8-atom diamond-cubic ASE Atoms."""
    from ase import Atoms
    a = {"C": 3.567, "Si": 5.431, "Ge": 5.658}.get(element, 3.567)
    frac = np.array([
        [0.00, 0.00, 0.00], [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.00], [0.75, 0.75, 0.25],
        [0.00, 0.50, 0.50], [0.25, 0.75, 0.75],
        [0.50, 0.00, 0.50], [0.75, 0.25, 0.75],
    ], dtype=np.float64)
    pos = frac * a
    if perturb:
        rng = np.random.default_rng(42)
        pos += rng.normal(0, 0.03, pos.shape)
    cell = np.diag([a, a, a])
    return Atoms(symbols=[element] * 8, positions=pos, cell=cell, pbc=True)


def _graphene_bilayer_atoms(nx: int = 3, ny: int = 3, interlayer: float = 3.35):
    """AB-stacked graphene bilayer ASE Atoms with mol-id array for layers."""
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
    cell = np.array([nx * a1, ny * a2, [0, 0, interlayer + 20.0]])
    layers = np.array([0] * len(layer0) + [1] * len(layer1), dtype=np.int32)
    atoms = Atoms(
        symbols=["C"] * len(pos),
        positions=pos,
        cell=cell,
        pbc=[True, True, False],
    )
    atoms.new_array("mol-id", (layers + 1).astype(int))
    return atoms, layers


# ═══════════════════════════════════════════════════════════════════════════════
#  Reference parameter sets
# ═══════════════════════════════════════════════════════════════════════════════

def _tersoff_params():
    """Erhart-Albe carbon Tersoff parameters."""
    return [
        3.0, 1.0, 0.0,
        38049.0, 4.3484, -0.57058,
        0.72751, 1.5724e-7, 2.2119,
        346.74, 2.85, 0.15,
        3.4879, 1393.6,
    ]


def _kc_params():
    """Kolmogorov & Crespi (2005) graphene parameters."""
    return [3.34, 15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238]


def _drip_params():
    """Wen et al. (2018) DRIP graphene parameters (8 physical values)."""
    return [15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238, 3.34]


_SI_HYPERPARAMS = {
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Assertion helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _assert_valid(energy, forces, N, label: str = ""):
    assert np.isfinite(energy), f"{label}: energy not finite ({energy})"
    assert np.all(np.isfinite(forces)), f"{label}: forces contain non-finite values"
    assert forces.shape == (N, 3), f"{label}: forces shape {forces.shape} != ({N}, 3)"


# ═══════════════════════════════════════════════════════════════════════════════
#  TersoffLammpsCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestTersoffLammpsCalculator:

    def test_basic_compute(self):
        atoms = _diamond_cubic_atoms("C")
        calc = TersoffLammpsCalculator(_tersoff_params())
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        _assert_valid(energy, forces, len(atoms), "Tersoff")

    def test_param_sensitivity(self):
        atoms = _diamond_cubic_atoms("C")
        params1 = _tersoff_params()
        calc = TersoffLammpsCalculator(params1)
        e1 = calc.get_potential_energy(atoms)
        params2 = list(params1)
        params2[-1] *= 1.1  # scale A
        calc.set_parameters(params2)
        e2 = calc.get_potential_energy(atoms)
        assert not np.isclose(e1, e2, rtol=1e-10), \
            f"Tersoff: energy unchanged after param change (e1={e1}, e2={e2})"

    def test_deterministic(self):
        atoms = _diamond_cubic_atoms("C")
        calc = TersoffLammpsCalculator(_tersoff_params())
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        assert np.isclose(r1["energy"], r2["energy"], rtol=1e-10)
        assert np.allclose(r1["forces"], r2["forces"], rtol=1e-10)

    def test_position_injection(self):
        """Second calculate with perturbed positions should re-evaluate (inject path)."""
        atoms1 = _diamond_cubic_atoms("C")
        atoms2 = _diamond_cubic_atoms("C", perturb=True)
        calc = TersoffLammpsCalculator(_tersoff_params())
        e1 = calc.get_potential_energy(atoms1)
        e2 = calc.get_potential_energy(atoms2)
        assert not np.isclose(e1, e2, rtol=1e-6), \
            "Tersoff: energy unchanged after position change"

    def test_batch_consistent_with_single(self):
        atoms_list = [_diamond_cubic_atoms("C"),
                      _diamond_cubic_atoms("C", perturb=True)]
        params = _tersoff_params()
        calc = TersoffLammpsCalculator(params)
        e_single = [calc.get_potential_energy(a) for a in atoms_list]

        calc2 = TersoffLammpsCalculator(params)
        calc2.prepare_batch(atoms_list)
        energies, forces_list = calc2.evaluate_batch(atoms_list)
        for i, (e_b, e_s) in enumerate(zip(energies, e_single)):
            assert np.isclose(e_b, e_s, rtol=1e-8), \
                f"Tersoff batch[{i}]: {e_b} vs single {e_s}"

    def test_relax_lowers_energy(self):
        atoms = _diamond_cubic_atoms("C", perturb=True)
        calc = TersoffLammpsCalculator(_tersoff_params())
        e_before = calc.get_potential_energy(atoms)
        relaxed = calc.relax_structure(atoms, relax_backend="lammps",
                                       ftol=1e-3, maxiter=200)
        e_after = calc.get_potential_energy(relaxed)
        assert e_after <= e_before + 1e-6, \
            f"Tersoff relax: energy increased ({e_before:.6f} → {e_after:.6f})"


# ═══════════════════════════════════════════════════════════════════════════════
#  KolmogorovCrespiLammpsCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestKolmogorovCrespiLammpsCalculator:

    def test_basic_compute(self):
        # KC needs a supercell large enough for the cutoff (~14 Å)
        atoms, layers = _graphene_bilayer_atoms(nx=7, ny=7)
        calc = KolmogorovCrespiLammpsCalculator(_kc_params(), cutoff=14.0)
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        _assert_valid(energy, forces, len(atoms), "KC")

    def test_param_sensitivity(self):
        atoms, _ = _graphene_bilayer_atoms(nx=7, ny=7)
        params1 = _kc_params()
        calc = KolmogorovCrespiLammpsCalculator(params1, cutoff=14.0)
        e1 = calc.get_potential_energy(atoms)
        params2 = list(params1)
        params2[0] *= 1.05  # change z0
        calc.set_parameters(params2)
        e2 = calc.get_potential_energy(atoms)
        assert not np.isclose(e1, e2, rtol=1e-10), \
            f"KC: energy unchanged after param change"

    def test_deterministic(self):
        atoms, _ = _graphene_bilayer_atoms(nx=7, ny=7)
        calc = KolmogorovCrespiLammpsCalculator(_kc_params(), cutoff=14.0)
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        assert np.isclose(r1["energy"], r2["energy"], rtol=1e-10)
        assert np.allclose(r1["forces"], r2["forces"], rtol=1e-10)

    def test_batch_consistent_with_single(self):
        a1, _ = _graphene_bilayer_atoms(nx=7, ny=7)
        a2, _ = _graphene_bilayer_atoms(nx=7, ny=7, interlayer=3.5)
        atoms_list = [a1, a2]
        params = _kc_params()
        calc = KolmogorovCrespiLammpsCalculator(params, cutoff=14.0)
        e_single = [calc.get_potential_energy(a) for a in atoms_list]

        calc2 = KolmogorovCrespiLammpsCalculator(params, cutoff=14.0)
        calc2.prepare_batch(atoms_list)
        energies, _ = calc2.evaluate_batch(atoms_list)
        for i, (e_b, e_s) in enumerate(zip(energies, e_single)):
            assert np.isclose(e_b, e_s, rtol=1e-8), \
                f"KC batch[{i}]: {e_b} vs single {e_s}"


# ═══════════════════════════════════════════════════════════════════════════════
#  DRIPLammpsCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestDRIPLammpsCalculator:

    def test_basic_compute(self):
        atoms, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        calc = DRIPLammpsCalculator(_drip_params(), cutoff=14.0)
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        _assert_valid(energy, forces, len(atoms), "DRIP")

    def test_param_sensitivity(self):
        atoms, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        params1 = _drip_params()
        calc = DRIPLammpsCalculator(params1, cutoff=14.0)
        e1 = calc.get_potential_energy(atoms)
        params2 = list(params1)
        params2[7] *= 1.05  # change z0
        calc.set_parameters(params2)
        e2 = calc.get_potential_energy(atoms)
        assert not np.isclose(e1, e2, rtol=1e-10), \
            "DRIP: energy unchanged after param change"

    def test_deterministic(self):
        atoms, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        calc = DRIPLammpsCalculator(_drip_params(), cutoff=14.0)
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        assert np.isclose(r1["energy"], r2["energy"], rtol=1e-10)
        assert np.allclose(r1["forces"], r2["forces"], rtol=1e-10)

    def test_batch_consistent_with_single(self):
        a1, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        a2, _ = _graphene_bilayer_atoms(nx=3, ny=3, interlayer=3.5)
        atoms_list = [a1, a2]
        params = _drip_params()
        calc = DRIPLammpsCalculator(params, cutoff=14.0)
        e_single = [calc.get_potential_energy(a) for a in atoms_list]

        calc2 = DRIPLammpsCalculator(params, cutoff=14.0)
        calc2.prepare_batch(atoms_list)
        energies, _ = calc2.evaluate_batch(atoms_list)
        for i, (e_b, e_s) in enumerate(zip(energies, e_single)):
            assert np.isclose(e_b, e_s, rtol=1e-8), \
                f"DRIP batch[{i}]: {e_b} vs single {e_s}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TersoffKCLammpsCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestTersoffKCLammpsCalculator:

    def test_basic_compute(self):
        atoms, _ = _graphene_bilayer_atoms(nx=7, ny=7)
        calc = TersoffKCLammpsCalculator(
            _tersoff_params(), _kc_params(), kc_cutoff=14.0
        )
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        _assert_valid(energy, forces, len(atoms), "Tersoff+KC")

    def test_param_sensitivity(self):
        atoms, _ = _graphene_bilayer_atoms(nx=7, ny=7)
        te = _tersoff_params()
        kc = _kc_params()
        calc = TersoffKCLammpsCalculator(te, kc, kc_cutoff=14.0)
        e1 = calc.get_potential_energy(atoms)
        kc2 = list(kc)
        kc2[0] *= 1.05
        calc.set_parameters(te, kc2)
        e2 = calc.get_potential_energy(atoms)
        assert not np.isclose(e1, e2, rtol=1e-10), \
            "Tersoff+KC: energy unchanged after KC param change"

    def test_deterministic(self):
        atoms, _ = _graphene_bilayer_atoms(nx=7, ny=7)
        calc = TersoffKCLammpsCalculator(_tersoff_params(), _kc_params(), kc_cutoff=14.0)
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        assert np.isclose(r1["energy"], r2["energy"], rtol=1e-10)
        assert np.allclose(r1["forces"], r2["forces"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
#  TersoffDRIPLammpsCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestTersoffDRIPLammpsCalculator:

    def test_basic_compute(self):
        atoms, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        calc = TersoffDRIPLammpsCalculator(
            _tersoff_params(), _drip_params(), cutoff=14.0
        )
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        _assert_valid(energy, forces, len(atoms), "Tersoff+DRIP")

    def test_param_sensitivity(self):
        atoms, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        te = _tersoff_params()
        drip = _drip_params()
        calc = TersoffDRIPLammpsCalculator(te, drip, cutoff=14.0)
        e1 = calc.get_potential_energy(atoms)
        drip2 = list(drip)
        drip2[7] *= 1.05  # z0
        calc.set_parameters(te, drip2)
        e2 = calc.get_potential_energy(atoms)
        assert not np.isclose(e1, e2, rtol=1e-10), \
            "Tersoff+DRIP: energy unchanged after DRIP param change"

    def test_deterministic(self):
        atoms, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        calc = TersoffDRIPLammpsCalculator(
            _tersoff_params(), _drip_params(), cutoff=14.0
        )
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        assert np.isclose(r1["energy"], r2["energy"], rtol=1e-10)
        assert np.allclose(r1["forces"], r2["forces"], rtol=1e-10)

    def test_batch_consistent_with_single(self):
        a1, _ = _graphene_bilayer_atoms(nx=3, ny=3)
        a2, _ = _graphene_bilayer_atoms(nx=3, ny=3, interlayer=3.5)
        atoms_list = [a1, a2]
        te = _tersoff_params()
        drip = _drip_params()
        calc = TersoffDRIPLammpsCalculator(te, drip, cutoff=14.0)
        e_single = [calc.get_potential_energy(a) for a in atoms_list]

        calc2 = TersoffDRIPLammpsCalculator(te, drip, cutoff=14.0)
        calc2.prepare_batch(atoms_list)
        energies, _ = calc2.evaluate_batch(atoms_list)
        for i, (e_b, e_s) in enumerate(zip(energies, e_single)):
            assert np.isclose(e_b, e_s, rtol=1e-8), \
                f"Tersoff+DRIP batch[{i}]: {e_b} vs single {e_s}"


# ═══════════════════════════════════════════════════════════════════════════════
#  PODLammpsCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestPODLammpsCalculator:

    @pytest.fixture
    def si_setup(self):
        from ase.build import bulk
        from blg_model_builder.potentials import ncoeff_from_params
        hp = dict(_SI_HYPERPARAMS, species=["Si"])
        nc = ncoeff_from_params(hp)
        coeffs = np.zeros(nc)
        atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
        return atoms, _SI_HYPERPARAMS, coeffs, nc

    def test_basic_compute(self, si_setup):
        atoms, hp, coeffs, nc = si_setup
        calc = PODLammpsCalculator(hp, coeffs, elements=["Si"], cutoff=5.0)
        energy = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        _assert_valid(energy, forces, len(atoms), "POD")

    def test_param_sensitivity(self, si_setup):
        atoms, hp, coeffs, nc = si_setup
        calc = PODLammpsCalculator(hp, coeffs, elements=["Si"], cutoff=5.0)
        e1 = calc.get_potential_energy(atoms)
        rng = np.random.default_rng(0)
        coeffs2 = rng.normal(0, 0.1, nc)
        calc.set_parameters(coeffs2)
        e2 = calc.get_potential_energy(atoms)
        assert not np.isclose(e1, e2, rtol=1e-10), \
            "POD: energy unchanged after param change"

    def test_deterministic(self, si_setup):
        atoms, hp, coeffs, nc = si_setup
        rng = np.random.default_rng(1)
        c = rng.normal(0, 0.05, nc)
        calc = PODLammpsCalculator(hp, c, elements=["Si"], cutoff=5.0)
        r1 = calc.calculate(atoms)
        r2 = calc.calculate(atoms)
        assert np.isclose(r1["energy"], r2["energy"], rtol=1e-10)
        assert np.allclose(r1["forces"], r2["forces"], rtol=1e-10)

    def test_ncoeff_property(self, si_setup):
        _, hp, coeffs, nc = si_setup
        calc = PODLammpsCalculator(hp, coeffs, elements=["Si"], cutoff=5.0)
        assert calc.ncoeff == nc

    def test_batch_consistent_with_single(self, si_setup):
        from ase.build import bulk
        atoms1 = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
        atoms2 = bulk("Si", crystalstructure="diamond", a=5.5, cubic=True)
        atoms_list = [atoms1, atoms2]
        _, hp, _, nc = si_setup
        rng = np.random.default_rng(2)
        coeffs = rng.normal(0, 0.05, nc)

        calc = PODLammpsCalculator(hp, coeffs, elements=["Si"], cutoff=5.0)
        e_single = [calc.get_potential_energy(a) for a in atoms_list]

        calc2 = PODLammpsCalculator(hp, coeffs, elements=["Si"], cutoff=5.0)
        calc2.prepare_batch(atoms_list)
        energies, _ = calc2.evaluate_batch(atoms_list)
        for i, (e_b, e_s) in enumerate(zip(energies, e_single)):
            assert np.isclose(e_b, e_s, rtol=1e-8), \
                f"POD batch[{i}]: {e_b} vs single {e_s}"

    def test_energy_from_descriptors_matches_lammps(self, si_setup):
        from ase.build import bulk
        atoms1 = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
        atoms2 = bulk("Si", crystalstructure="diamond", a=5.5, cubic=True)
        atoms_list = [atoms1, atoms2]
        _, hp, _, nc = si_setup
        rng = np.random.default_rng(3)
        coeffs = rng.normal(0, 0.05, nc)

        calc = PODLammpsCalculator(hp, coeffs, elements=["Si"], cutoff=5.0)
        calc.prepare_batch(atoms_list)
        descriptors = calc.compute_pod_descriptors(atoms_list, verbose=False)
        e_desc = calc.energy_from_descriptors(coeffs, descriptors)

        calc.set_parameters(coeffs)
        e_lammps, _ = calc.evaluate_batch(atoms_list)
        assert np.allclose(e_desc, e_lammps, rtol=1e-8), \
            f"descriptor energy {e_desc} vs LAMMPS {e_lammps}"

    def test_pod_atom_descriptors_sum_to_global(self, si_setup):
        """``compute pod/atom`` matches globally summed descriptors in one shot."""
        from ase.build import bulk

        atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
        _, hp, _, nc = si_setup
        calc = PODLammpsCalculator(
            hp, np.zeros(nc), elements=["Si"], cutoff=5.0,
        )
        D_atom = calc.compute_pod_atom_descriptors(atoms)
        assert D_atom.ndim == 2
        assert D_atom.shape[0] == len(atoms)
        assert np.all(np.isfinite(D_atom))
        D_sum = D_atom.sum(axis=0)
        D_glob = calc.compute_pod_descriptors([atoms], verbose=False)[0]
        if D_sum.shape[0] == D_glob.shape[0]:
            assert np.allclose(D_sum, D_glob, rtol=1e-8, atol=1e-8)
        elif D_sum.shape[0] + 1 == D_glob.shape[0]:
            # Coeff file leading constant/shift term is absent from compute pod/atom.
            match = np.allclose(D_sum, D_glob[1:], rtol=1e-8, atol=1e-8)
            assert match, (
                f"atom-sum {D_sum.shape} does not match global {D_glob.shape}"
            )
        else:
            raise AssertionError(
                f"atom-sum ncols={D_sum.size} vs global ncols={D_glob.size}"
            )
        D_atom2 = calc.compute_pod_atom_descriptors(atoms)
        assert np.allclose(D_atom, D_atom2, rtol=1e-10, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
#  potentials.py aliasing test
# ═══════════════════════════════════════════════════════════════════════════════

class TestPotentialsAliasing:
    """Verify that potentials.py exposes the LAMMPS Python classes under the
    legacy *ASECalculator names when lammps is available."""

    def test_tersoff_alias(self):
        from blg_model_builder.potentials import (
            TersoffASECalculator,
            _LAMMPS_PY_INTERFACE_AVAILABLE,
        )
        if not _LAMMPS_PY_INTERFACE_AVAILABLE:
            pytest.skip("LAMMPS Python interface not available")
        assert TersoffASECalculator is TersoffLammpsCalculator

    def test_kc_alias(self):
        from blg_model_builder.potentials import (
            KolmogorovCrespiASECalculator,
            _LAMMPS_PY_INTERFACE_AVAILABLE,
        )
        if not _LAMMPS_PY_INTERFACE_AVAILABLE:
            pytest.skip("LAMMPS Python interface not available")
        assert KolmogorovCrespiASECalculator is KolmogorovCrespiLammpsCalculator

    def test_drip_alias(self):
        from blg_model_builder.potentials import (
            DRIPASECalculator,
            _LAMMPS_PY_INTERFACE_AVAILABLE,
        )
        if not _LAMMPS_PY_INTERFACE_AVAILABLE:
            pytest.skip("LAMMPS Python interface not available")
        assert DRIPASECalculator is DRIPLammpsCalculator

    def test_pod_alias(self):
        from blg_model_builder.potentials import (
            PODASECalculator,
            _LAMMPS_PY_INTERFACE_AVAILABLE,
        )
        if not _LAMMPS_PY_INTERFACE_AVAILABLE:
            pytest.skip("LAMMPS Python interface not available")
        assert PODASECalculator is PODLammpsCalculator


# ═══════════════════════════════════════════════════════════════════════════════
#  evaluate_batch with params argument
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateBatchWithParams:
    """evaluate_batch(params=...) should update parameters and re-evaluate."""

    def test_tersoff_batch_params_update(self):
        atoms_list = [_diamond_cubic_atoms("C"),
                      _diamond_cubic_atoms("C", perturb=True)]
        params1 = _tersoff_params()
        params2 = list(params1)
        params2[-1] *= 1.1

        calc = TersoffLammpsCalculator(params1)
        calc.prepare_batch(atoms_list)
        e1, _ = calc.evaluate_batch(atoms_list)
        e2, _ = calc.evaluate_batch(atoms_list, params=params2)
        assert not np.allclose(e1, e2, rtol=1e-8), \
            "evaluate_batch with params: energies unchanged after param update"
