"""
test_TETB.py
------------
Tests for the tight-binding core of TETB_PODLammpsCalculator.

All tests call the module-level TB helpers in lammps_interface.py directly,
so they do **not** require LAMMPS to be installed.  The ACSF hopping model is
configured with M=15, W=6; the POD coefficients are set to zero so the total
energy is purely TB.

Tests
-----
TestTBEnergy
    Verify that the density-matrix band energy
        E_band = (1/n_kp) Σ_k Re[Tr(H_k · DM_k)]
    is numerically identical to the eigenvalue sum
        E_eig  = (2/n_kp) Σ_k Σ_{n=1..N/2} ε_{n,k}
    (factor of 2 from spin degeneracy; N//2 occupied bands at half-filling).

TestHellmannFeynmanForces
    Verify that the analytic band forces from _compute_band_forces (including
    three-body k-leg contributions) match finite-difference estimates of the
    band energy along the x and y directions on a 2-D square lattice with
    periodic boundary conditions.

TestHoppingGradient
    Verify the hopping element gradients returned by
    _acsf_hopping_gradient_from_pairs via finite differences:
      - J-leg gradient: ∂t_p/∂r_{p,α}  (perturb atom pair_j[p])
      - K-leg gradient: ∂t_p/∂r_{q,α}  (perturb atom pair_j[q] for each
        k-leg triplet (p, q))

Run with:
    cd tests/
    pytest test_TETB.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms


# ---------------------------------------------------------------------------
# Package import setup (mirrors test_acsf_band_structure.py)
# ---------------------------------------------------------------------------

def _ensure_importable_package() -> None:
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

# Pre-warm potentials.py BEFORE any import of lammps_interface.py.
#
# Why this matters
# ----------------
# lammps_interface.py imports utility helpers from blg_model_builder.potentials
# (line ~53).  potentials.py in turn does
#   ``from blg_model_builder.lammps_interface import LammpsCalculatorBase, ...``
# at line 486.  If lammps_interface is the *first* module loaded, Python
# registers it as partially-initialized in sys.modules, then potentials.py
# tries to read LammpsCalculatorBase from it before that class has been
# defined → ImportError (circular import).
#
# Loading potentials.py FIRST breaks the cycle:
#   1. potentials.py defines ase_to_lammps etc. (lines 1-485).
#   2. potentials.py line 486 → starts loading lammps_interface.py.
#   3. lammps_interface.py imports ase_to_lammps from the *partially*-initialized
#      potentials module — those symbols already exist → succeeds.
#   4. lammps_interface.py finishes → LammpsCalculatorBase etc. are defined.
#   5. potentials.py line 486 import completes → both modules fully initialized.
#
import blg_model_builder.potentials  # noqa: F401, E402  ← must stay at module level


# ── ACSF hyperparameters used in all tests ─────────────────────────────────
M = 15
W = 6
R_CUT = 4.5   # Å — captures ~2 nearest-neighbour shells in the test lattice

N_FEATURES = M + M * W   # 15 + 90 = 105

# Reproducible random TB parameters (small amplitude → well-behaved H)
_RNG = np.random.default_rng(seed=2025)
TB_PARAMS = _RNG.normal(0.0, 0.1, size=N_FEATURES)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def _square_lattice_2d(nx: int = 3, ny: int = 3, a: float = 2.5,
                        perturb: float = 0.05) -> Atoms:
    """Build an (nx × ny) 2-D square lattice with a small random perturbation.

    The structure is 2-D periodic in x and y with a large vacuum layer in z.
    The perturbation ensures the system is not at a high-symmetry configuration
    where cancellation could mask force errors.
    """
    rng = np.random.default_rng(seed=7)
    positions = []
    for i in range(nx):
        for j in range(ny):
            # Start at (i+0.5)*a so no atom sits at x=0 (periodic boundary).
            # Without this offset, atom 0 ends up at x ≈ 1e-4 Å: a -delta
            # perturbation crosses the boundary and changes the neighbour list.
            x = (i + 0.5) * a + rng.normal(0.0, perturb)
            y = (j + 0.5) * a + rng.normal(0.0, perturb)
            positions.append([x, y, 0.0])
    positions = np.array(positions, dtype=float)
    cell = [[nx * a, 0, 0], [0, ny * a, 0], [0, 0, 20.0]]
    return Atoms(
        symbols=["C"] * (nx * ny),
        positions=positions,
        cell=cell,
        pbc=[True, True, False],
    )


def _gamma_kpoints() -> np.ndarray:
    """Single Γ-point (0, 0, 0) in Cartesian Å⁻¹."""
    return np.zeros((1, 3), dtype=np.float64)


def _kpoint_mesh_2d(atoms: Atoms, n: int = 3) -> np.ndarray:
    """Uniform Monkhorst-Pack n×n mesh in the xy plane (Cartesian Å⁻¹).

    k_uniform_mesh requires a *tuple* for mesh_size because it does
    ``mesh_size + (3,)`` internally — list + tuple raises TypeError.
    """
    from blg_model_builder.tb_models import k_uniform_mesh, get_recip_cell

    k_reduced = k_uniform_mesh((n, n, 1))                  # tuple, not list
    cell = np.array(atoms.get_cell(), dtype=float)
    recip = get_recip_cell(cell.T)                         # (3, 3) Å⁻¹
    return k_reduced @ recip                               # (n_kp, 3) Cartesian


# ---------------------------------------------------------------------------
# Core TB helper: band energy for a given atoms + k-points
# ---------------------------------------------------------------------------

def _tb_band_energy(atoms: Atoms, tb_params: np.ndarray,
                    kpoints: np.ndarray) -> float:
    """Compute TB band energy E = (1/n_kp) Σ_k Tr(H_k · DM_k).

    Uses the same internal helpers as TETB_PODLammpsCalculator._compute_tb.
    """
    from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
    from blg_model_builder.lammps_interface import (
        _build_hamiltonians_kpoints,
        _solve_density_matrix,
    )

    N = len(atoms)
    n_kp = len(kpoints)

    descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
        atoms, M=M, W=W, r_cut=R_CUT,
    )
    t_ij = descriptors @ tb_params

    H_all = _build_hamiltonians_kpoints(pair_i, pair_j, pair_v, t_ij, kpoints, N)
    DM_all = np.array([_solve_density_matrix(H_all[ki]) for ki in range(n_kp)])

    return float(np.real(np.einsum("kij,kji->", H_all, DM_all)) / n_kp)


# ════════════════════════════════════════════════════════════════════════════
#  Test 1 — Band energy: Tr(H·DM) == 2 × Σ occupied eigenvalues
# ════════════════════════════════════════════════════════════════════════════

class TestTBEnergy:
    """Density-matrix E_band equals the eigenvalue sum at half-filling."""

    @pytest.fixture
    def atoms(self):
        return _square_lattice_2d(nx=3, ny=3)

    @pytest.fixture(params=["gamma", "mesh"])
    def kpoints(self, request, atoms):
        if request.param == "gamma":
            return _gamma_kpoints()
        return _kpoint_mesh_2d(atoms, n=3)

    def test_dm_energy_matches_eigenvalue_sum(self, atoms, kpoints):
        """E_band from Tr(H·DM) must equal 2 × Σ_{k,occ} ε_{n,k} / n_kp."""
        from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
        from blg_model_builder.lammps_interface import (
            _build_hamiltonians_kpoints,
            _solve_density_matrix,
        )

        N = len(atoms)
        nocc = N // 2
        n_kp = len(kpoints)

        descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=R_CUT,
        )
        t_ij = descriptors @ TB_PARAMS

        H_all = _build_hamiltonians_kpoints(
            pair_i, pair_j, pair_v, t_ij, kpoints, N,
        )

        # ── Density-matrix E_band (what _compute_tb returns) ─────────────────
        DM_all = np.array([_solve_density_matrix(H_all[ki]) for ki in range(n_kp)])
        E_dm = float(np.real(np.einsum("kij,kji->", H_all, DM_all)) / n_kp)

        # ── Eigenvalue sum: 2 × Σ_{k,n<nocc} ε_{n,k} / n_kp ─────────────────
        # Factor of 2 for spin degeneracy; N//2 occupied bands at half-filling.
        E_eig = 0.0
        for ki in range(n_kp):
            evals = np.linalg.eigvalsh(H_all[ki])   # real, ascending order
            E_eig += 2.0 * float(np.sum(evals[:nocc]))
        E_eig /= n_kp

        assert np.isfinite(E_dm), "Density-matrix E_band is not finite"
        assert np.isfinite(E_eig), "Eigenvalue E_band is not finite"

        np.testing.assert_allclose(
            E_dm, E_eig, rtol=1e-10, atol=1e-10,
            err_msg=(
                f"DM-based E_band = {E_dm:.12g} eV,  "
                f"eigenvalue sum   = {E_eig:.12g} eV  "
                f"(kpoints shape {kpoints.shape})"
            ),
        )


# ════════════════════════════════════════════════════════════════════════════
#  Test 2 — Hellmann-Feynman forces vs finite differences (x and y only)
# ════════════════════════════════════════════════════════════════════════════

class TestHellmannFeynmanForces:
    """_compute_band_forces must match -(dE_band/dr) via finite differences.

    Uses full TB_PARAMS including three-body terms.  The analytic forces
    include both the j-leg contribution (∂t_p/∂r_p) and the k-leg
    contribution (∂t_p/∂r_q for each 3-body triplet), so the comparison
    against finite differences should hold to near floating-point precision.
    """

    @pytest.fixture
    def atoms(self):
        return _square_lattice_2d(nx=3, ny=3, perturb=0.08)

    @pytest.fixture
    def kpoints(self, atoms):
        return _kpoint_mesh_2d(atoms, n=3)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _analytic_forces(self, atoms: Atoms, kpoints: np.ndarray) -> np.ndarray:
        """Hellmann-Feynman band forces (N, 3) via the internal helpers used
        inside TETB_PODLammpsCalculator._compute_tb."""
        from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
        from blg_model_builder.lammps_interface import (
            _build_hamiltonians_kpoints,
            _solve_density_matrix,
            _acsf_hopping_gradient_from_pairs,
            _compute_band_forces,
        )

        N = len(atoms)
        n_kp = len(kpoints)

        descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=R_CUT,
        )
        t_ij = descriptors @ TB_PARAMS

        H_all = _build_hamiltonians_kpoints(
            pair_i, pair_j, pair_v, t_ij, kpoints, N,
        )
        DM_all = np.array([_solve_density_matrix(H_all[ki]) for ki in range(n_kp)])

        grad_t, kleg_t_p, kleg_t_q, kleg_grad = _acsf_hopping_gradient_from_pairs(
            M, W, R_CUT, TB_PARAMS, pair_i, pair_j, pair_v, N,
        )
        return _compute_band_forces(
            pair_i, pair_j, pair_v, t_ij, grad_t, DM_all, kpoints, N,
            kleg_t_p=kleg_t_p, kleg_t_q=kleg_t_q, grad_kleg=kleg_grad,
        )

    # ── tests ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("alpha", [0, 1], ids=["x", "y"])
    def test_hellmann_feynman_vs_fd(self, atoms, kpoints, alpha):
        """Analytic force component *alpha* matches -(dE/dr_alpha) for every atom."""
        delta = 1e-4    # Å — small enough for 2nd-order accuracy
        N = len(atoms)
        F_analytic = self._analytic_forces(atoms, kpoints)

        pos0 = atoms.get_positions().copy()
        F_fd = np.zeros((N, 3), dtype=float)

        for i in range(N):
            pos_p = pos0.copy();  pos_p[i, alpha] += delta
            atoms_p = atoms.copy();  atoms_p.set_positions(pos_p)

            pos_m = pos0.copy();  pos_m[i, alpha] -= delta
            atoms_m = atoms.copy();  atoms_m.set_positions(pos_m)

            E_p = _tb_band_energy(atoms_p, TB_PARAMS, kpoints)
            E_m = _tb_band_energy(atoms_m, TB_PARAMS, kpoints)

            F_fd[i, alpha] = -(E_p - E_m) / (2.0 * delta)

        max_err = float(np.max(np.abs(F_analytic[:, alpha] - F_fd[:, alpha])))
        rms_err = float(np.sqrt(np.mean((F_analytic[:, alpha] - F_fd[:, alpha]) ** 2)))
        direction = ["x", "y"][alpha]

        assert max_err < 5e-4, (
            f"HF forces ({direction}):  "
            f"max |F_analytic - F_fd| = {max_err:.3e} eV/Å,  "
            f"rms = {rms_err:.3e} eV/Å"
        )

    def test_force_conservation(self, atoms, kpoints):
        """Newton's third law: sum of analytic band forces must vanish."""
        F = self._analytic_forces(atoms, kpoints)
        net = np.sum(F, axis=0)
        np.testing.assert_allclose(
            net, np.zeros(3), atol=1e-10,
            err_msg=f"Net band force is not zero: {net}",
        )


# ════════════════════════════════════════════════════════════════════════════
#  Test 3 — Hopping gradient: j-leg and k-leg via finite differences
# ════════════════════════════════════════════════════════════════════════════

class TestHoppingGradient:
    """Verify _acsf_hopping_gradient_from_pairs via element-wise FD.

    J-leg test
    ----------
    For each half-list bond p = (i, j), perturbing atom j by ±δ in direction
    α changes t_p only through the j-leg (since r_p = R_j - R_i changes while
    all other bond vectors from centre i are unaffected).  The FD derivative
    should equal grad_t[p, α].

    K-leg test
    ----------
    For each triplet n = (p, q) where q = (i, k) is a k-leg bond of p,
    perturbing atom k by ±δ changes t_p through the k-leg 3-body term
    (since r_q = R_k - R_i changes).  The FD derivative of t_p should equal
    kleg_grad[n, α].
    """

    delta = 1e-5   # Å — central-difference step

    @pytest.fixture
    def grad_data(self):
        """Compute gradient data once and share across parametrized tests."""
        from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
        from blg_model_builder.lammps_interface import _acsf_hopping_gradient_from_pairs

        atoms = _square_lattice_2d(nx=3, ny=3, perturb=0.05)
        N = len(atoms)

        descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=R_CUT,
        )
        grad_t, kleg_t_p, kleg_t_q, kleg_grad = _acsf_hopping_gradient_from_pairs(
            M, W, R_CUT, TB_PARAMS, pair_i, pair_j, pair_v, N,
        )
        return atoms, pair_i, pair_j, pair_v, grad_t, kleg_t_p, kleg_t_q, kleg_grad

    # ── helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _hoppings(atoms: Atoms, pair_i_ref, pair_j_ref) -> np.ndarray:
        """Compute t_ij for `atoms`, asserting bond topology is unchanged."""
        from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
        desc, (pi, pj, _) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=R_CUT,
        )
        assert np.array_equal(pi, pair_i_ref) and np.array_equal(pj, pair_j_ref), (
            "Bond topology changed after perturbation — delta is too large "
            "or the structure is too close to a cutoff boundary."
        )
        return desc @ TB_PARAMS

    # ── j-leg gradient test ───────────────────────────────────────────────────

    @pytest.mark.parametrize("alpha", [0, 1], ids=["x", "y"])
    def test_jleg_gradient(self, grad_data, alpha):
        """∂t_p/∂r_{p,α} matches FD when atom pair_j[p] is perturbed."""
        atoms, pair_i, pair_j, _, grad_t, *_ = grad_data
        delta = self.delta
        pos0 = atoms.get_positions().copy()

        max_err = 0.0
        for p in range(len(pair_i)):
            j_atom = int(pair_j[p])

            pos_p = pos0.copy();  pos_p[j_atom, alpha] += delta
            pos_m = pos0.copy();  pos_m[j_atom, alpha] -= delta

            at_p = atoms.copy();  at_p.set_positions(pos_p)
            at_m = atoms.copy();  at_m.set_positions(pos_m)

            t_plus  = self._hoppings(at_p, pair_i, pair_j)
            t_minus = self._hoppings(at_m, pair_i, pair_j)

            fd = (t_plus[p] - t_minus[p]) / (2.0 * delta)
            err = abs(fd - grad_t[p, alpha])
            max_err = max(max_err, err)

        direction = ["x", "y"][alpha]
        assert max_err < 1e-6, (
            f"J-leg gradient ({direction}): max |FD - analytic| = {max_err:.3e} eV/Å"
        )

    # ── k-leg gradient test ───────────────────────────────────────────────────

    @pytest.mark.parametrize("alpha", [0, 1], ids=["x", "y"])
    def test_kleg_gradient(self, grad_data, alpha):
        """∂t_p/∂r_{q,α} matches FD when k-leg atom pair_j[q] is perturbed."""
        atoms, pair_i, pair_j, _, _, kleg_t_p, kleg_t_q, kleg_grad = grad_data

        if len(kleg_t_p) == 0:
            pytest.skip("No triplets found in this structure")

        delta = self.delta
        pos0 = atoms.get_positions().copy()

        max_err = 0.0
        for n in range(len(kleg_t_p)):
            p = int(kleg_t_p[n])
            q = int(kleg_t_q[n])
            k_atom = int(pair_j[q])   # the k-leg neighbour atom

            pos_p = pos0.copy();  pos_p[k_atom, alpha] += delta
            pos_m = pos0.copy();  pos_m[k_atom, alpha] -= delta

            at_p = atoms.copy();  at_p.set_positions(pos_p)
            at_m = atoms.copy();  at_m.set_positions(pos_m)

            t_plus  = self._hoppings(at_p, pair_i, pair_j)
            t_minus = self._hoppings(at_m, pair_i, pair_j)

            # FD derivative of t_p when k_atom moves — captures only the
            # k-leg contribution since r_q = R_k - R_i changes while r_p is fixed.
            fd = (t_plus[p] - t_minus[p]) / (2.0 * delta)
            err = abs(fd - kleg_grad[n, alpha])
            max_err = max(max_err, err)

        direction = ["x", "y"][alpha]
        assert max_err < 1e-6, (
            f"K-leg gradient ({direction}): "
            f"max |FD - analytic| = {max_err:.3e} eV/Å "
            f"over {len(kleg_t_p)} triplets"
        )
