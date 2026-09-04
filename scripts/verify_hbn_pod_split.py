#!/usr/bin/env python3
"""Standalone verification for carbon-only POD on TBLG/hBN hybrid cells."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
UQ_DIR = REPO_ROOT / "uncertainty_quantification"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(UQ_DIR))

from blg_model_builder.potentials import (  # noqa: E402
    PODExtepILPLammpsCalculator,
    ncoeff_from_params,
)
from tblg_hbn_geometry import build_tblg_on_hbn  # noqa: E402

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


def main() -> int:
    hp = dict(_MIN_POD_HP, species=["C"])
    nc = int(ncoeff_from_params(hp))
    rng = np.random.default_rng(0)
    coeffs = rng.normal(0, 0.02, nc)
    calc = PODExtepILPLammpsCalculator(hp, coeffs, cutoff=6.0)
    atoms = build_tblg_on_hbn(theta_deg=9.43)
    syms = np.array(atoms.get_chemical_symbols())
    n_c = int(np.sum(syms == "C"))
    assert np.any(syms == "B") and np.any(syms == "N")

    d_atom = calc.compute_pod_atom_descriptors(atoms)
    assert d_atom.shape[0] == n_c, f"expected {n_c} C rows, got {d_atom.shape[0]}"
    print(f"OK atom descriptors: {d_atom.shape}")

    total = calc.calculate(atoms)
    pod = calc._evaluate_pod_on_carbon(atoms)
    bn = calc._evaluate_bn_on_full(atoms)
    assert np.isclose(total["energy"], pod["energy"] + bn["energy"], rtol=1e-8)
    print(f"OK energy split: total={total['energy']:.6f} pod={pod['energy']:.6f} bn={bn['energy']:.6f}")

    pod0 = pod["energy"]
    moved = atoms.copy()
    b_idx = int(np.where(syms == "B")[0][0])
    pos = moved.get_positions()
    pos[b_idx, 2] += 0.75
    moved.set_positions(pos)
    pod1 = calc._evaluate_pod_on_carbon(moved)["energy"]
    assert np.isclose(pod0, pod1, rtol=1e-10, atol=1e-10), "B move changed POD energy"
    print("OK moving B does not change POD-on-carbon energy")

    e1 = calc.calculate(moved)["energy"]
    assert not np.isclose(total["energy"], e1, rtol=1e-10), "B move did not change total"
    print("OK moving B changes total hybrid energy")

    energies, _ = calc.evaluate_batch([atoms])
    assert np.isclose(energies[0], total["energy"], rtol=1e-8)
    print("OK evaluate_batch matches calculate")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
