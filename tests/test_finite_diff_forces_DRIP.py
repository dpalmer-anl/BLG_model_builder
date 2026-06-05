"""
Finite-difference force consistency test for TersoffDRIPASECalculator.

Matches the philosophy of `tests/test_finite_diff_forces_KC.py`:
compare analytic forces from the hybrid calculator against central finite
differences of the total energy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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

# Allow importing the shared harness when tests/ is not a package.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Require the LAMMPS Python module
import pytest
pytest.importorskip("lammps")

from blg_model_builder.geom_tools import get_bilayer_atoms
from blg_model_builder.potentials import TersoffDRIPASECalculator

from physical_properties_harness import (
    layer_tags_from_mol_id,
    make_calc_tersoff_drip_best_fit_estimate,
)


def _central_fd_forces(calc, atoms, *, delta: float, components=(2,)):
    """
    Central finite-difference forces for selected Cartesian components.
    components: iterable of 0=x,1=y,2=z.
    Returns fd forces array (N, 3) with only requested components filled.
    """
    n = len(atoms)
    fd = np.zeros((n, 3), dtype=float)
    pos0 = atoms.get_positions().copy()
    for i in range(n):
        for a in components:
            pos_f = pos0.copy()
            pos_f[i, a] += delta
            atoms.set_positions(pos_f)
            e_f = calc.get_potential_energy(atoms)

            pos_b = pos0.copy()
            pos_b[i, a] -= delta
            atoms.set_positions(pos_b)
            e_b = calc.get_potential_energy(atoms)

            fd[i, a] = -(e_f - e_b) / (2.0 * delta)
    atoms.set_positions(pos0)
    return fd


def test_tersoff_drip_fz_matches_finite_difference():
    """
    Check Fz only on the primitive 4-atom AB bilayer to keep runtime small.
    """
    atoms = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    te, drip = make_calc_tersoff_drip_best_fit_estimate()
    calc = TersoffDRIPASECalculator(
        te.tolist(),
        drip.tolist(),
        layer_tags=layer_tags_from_mol_id(atoms),
    )
    atoms.calc = calc

    delta = 5e-4
    res = calc.calculate(atoms)
    f = np.asarray(res["forces"], dtype=float)
    fd = _central_fd_forces(calc, atoms, delta=delta, components=(2,))

    # Tolerances: DRIP normals can be numerically sensitive; start moderately loose.
    np.testing.assert_allclose(
        f[:, 2],
        fd[:, 2],
        atol=0.10,
        rtol=0.03,
        err_msg=f"LAMMPS vs FD F_z (δ={delta} Å) for Tersoff+DRIP",
    )

