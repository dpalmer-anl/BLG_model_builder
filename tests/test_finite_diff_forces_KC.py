"""
test_finite_diff_forces_KC.py

Compare LAMMPS analytic forces from TersoffKolmogorovCrespiASECalculator against
central finite-difference forces from single-point energies.

* ``test_tersoff_kc_fz_matches_finite_difference``: :math:`F_z` only on the
  default primitive bilayer cell (minimal).
* ``test_tersoff_kc_fxy_matches_finite_difference_expanded_cell``: :math:`F_x`
  and :math:`F_y` with the cell vectors scaled by 3× and the structure shifted
  so the COM sits at the body-center of the larger cell. This keeps in-plane
  finite-difference steps from moving atoms across periodic images, which can
  break KC layer-normal construction.

Geometry: AB bilayer at 3.5 Å from ``get_bilayer_atoms(3.5, 0.0, sc=1)``.

Run: pytest tests/test_finite_diff_forces_KC.py -v
     python tests/test_finite_diff_forces_KC.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_tests_dir = Path(__file__).resolve().parent
_pkg_src = _tests_dir.parent / "src"
for _p in (_tests_dir, _pkg_src):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from blg_model_builder.geom_tools import get_bilayer_atoms
from blg_model_builder.potentials import TersoffKolmogorovCrespiASECalculator

# Same layout as tests/test_relaxation.py (do not import that module — it runs on import)
_BEST_FIT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "uncertainty_quantification", "best_fit_params")
)


def _load_tersoff_kolmogorov_crespi_params():
    """
    Prefer separate *estimate* files; fall back to combined fit if present.

    Layout for combined ``Tersoff+Kolmogorov_Crespi_best_fit_params.npz`` matches
    ``get_MCMC_inputs.py``: first 10 values = KC, remaining 14 = Tersoff.
    """
    kc_est = os.path.join(_BEST_FIT_DIR, "Kolmogorov_Crespi_best_fit_params_estimate.npz")
    te_est = os.path.join(_BEST_FIT_DIR, "Tersoff_best_fit_params_estimate.npz")
    combined = os.path.join(_BEST_FIT_DIR, "Tersoff+Kolmogorov_Crespi_best_fit_params.npz")

    if os.path.isfile(kc_est) and os.path.isfile(te_est):
        kc_p = np.load(kc_est)["params"]
        kc_p = np.array(
            [
                3.416084,
                20.021583,
                10.9055107,
                4.2756354,
                1.0010836e-2,
                0.8447122,
                2.9360584,
                14.3132588,
            ]
        )
        te_p = np.load(te_est)["params"]
        return np.asarray(te_p, dtype=float), np.asarray(kc_p, dtype=float)

    if os.path.isfile(combined):
        p = np.load(combined)["params"]
        if len(p) >= 24:
            kc_p, te_p = p[:10], p[10:24]
        elif len(p) >= 22:
            kc_p, te_p = p[:8], p[8:22]
        else:
            raise ValueError(
                f"Combined Tersoff+KC params length {len(p)}; need >= 22 (8+14 or 10+14)."
            )
        return np.asarray(te_p, dtype=float), np.asarray(kc_p, dtype=float)

    raise FileNotFoundError(
        f"Expected ({kc_est!s} and {te_est!s}) or {combined!s} under best_fit_params."
    )


def _layer_tags_from_mol_id(atoms):
    mol = atoms.get_array("mol-id")
    return np.searchsorted(np.unique(mol), mol).astype(int).tolist()


def _expand_cell_and_center_atoms(atoms, scale: float = 3.0):
    """
    Scale every Bravais vector by ``scale`` (larger periodic box) and translate
    atoms so the Cartesian center of mass coincides with the parallelepiped
    body-center ``(a+b+c)/2`` in the new cell. In-plane distances are unchanged;
    avoids KC normals failing when FD steps wrap atoms across small-cell PBCs.
    """
    cell = np.asarray(atoms.get_cell(), dtype=float)
    new_cell = scale * cell
    pos = atoms.get_positions()
    com = pos.mean(axis=0)
    cell_center = 0.5 * np.sum(new_cell, axis=0)
    atoms.set_cell(new_cell)
    atoms.set_positions(pos + (cell_center - com))


def _central_finite_diff_fxy_energy_only(calc, atoms, delta: float) -> np.ndarray:
    """In-plane :math:`F_x`, :math:`F_y` via central differences; restores positions."""
    n = len(atoms)
    fd_xy = np.zeros((n, 2))
    pos0 = atoms.get_positions().copy()
    for i in range(n):
        for j, alpha in enumerate((0, 1)):
            pos_fwd = pos0.copy()
            pos_fwd[i, alpha] += delta
            atoms.set_positions(pos_fwd)
            e_fwd = calc.get_potential_energy(atoms)

            pos_bwd = pos0.copy()
            pos_bwd[i, alpha] -= delta
            atoms.set_positions(pos_bwd)
            e_bwd = calc.get_potential_energy(atoms)

            fd_xy[i, j] = -(e_fwd - e_bwd) / (2.0 * delta)
    atoms.set_positions(pos0)
    return fd_xy


def _central_finite_diff_fz_energy_only(calc, atoms, delta: float) -> np.ndarray:
    """z-component forces only via central differences; restores positions after."""
    n = len(atoms)
    fd_z = np.zeros(n)
    pos0 = atoms.get_positions().copy()
    z_axis = 2
    for i in range(n):
        pos_fwd = pos0.copy()
        pos_fwd[i, z_axis] += delta
        atoms.set_positions(pos_fwd)
        e_fwd = calc.get_potential_energy(atoms)

        pos_bwd = pos0.copy()
        pos_bwd[i, z_axis] -= delta
        atoms.set_positions(pos_bwd)
        e_bwd = calc.get_potential_energy(atoms)

        fd_z[i] = -(e_fwd - e_bwd) / (2.0 * delta)
    atoms.set_positions(pos0)
    return fd_z


# z-only check; slightly tighter absolute scale than full 3D when |Fz| is moderate
_ATOL_FZ = 0.07
_RTOL_FZ = 0.02
_ATOL_FXY = 0.08
_RTOL_FXY = 0.02


def test_tersoff_kc_fz_matches_finite_difference():
    """LAMMPS F_z vs finite differences on a 4-atom AB bilayer at d = 3.5 Å."""
    atoms = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    te_p, kc_p = _load_tersoff_kolmogorov_crespi_params()
    calc = TersoffKolmogorovCrespiASECalculator(
        te_p.tolist(),
        kc_p.tolist(),
        layer_tags=_layer_tags_from_mol_id(atoms),
    )
    atoms.calc = calc

    delta = 5e-4
    res = calc.calculate(atoms)
    fz_lammps = np.asarray(res["forces"], dtype=float)[:, 2]
    fz_fd = _central_finite_diff_fz_energy_only(calc, atoms, delta=delta)

    np.testing.assert_allclose(
        fz_lammps,
        fz_fd,
        atol=_ATOL_FZ,
        rtol=_RTOL_FZ,
        err_msg=f"LAMMPS vs FD F_z (δ = {delta} Å)",
    )


def test_tersoff_kc_fxy_matches_finite_difference_expanded_cell():
    """
    LAMMPS F_x, F_y vs finite differences on a 4-atom AB bilayer at d = 3.5 Å,
    with the cell enlarged 3× and the cluster centered so FD steps stay away
    from periodic image boundaries (KC layer normals).
    """
    atoms = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    _expand_cell_and_center_atoms(atoms, scale=3.0)

    te_p, kc_p = _load_tersoff_kolmogorov_crespi_params()
    calc = TersoffKolmogorovCrespiASECalculator(
        te_p.tolist(),
        kc_p.tolist(),
        layer_tags=_layer_tags_from_mol_id(atoms),
    )
    atoms.calc = calc

    delta = 5e-4
    res = calc.calculate(atoms)
    f = np.asarray(res["forces"], dtype=float)
    fxy_lammps = f[:, :2]
    fxy_fd = _central_finite_diff_fxy_energy_only(calc, atoms, delta=delta)

    np.testing.assert_allclose(
        fxy_lammps,
        fxy_fd,
        atol=_ATOL_FXY,
        rtol=_RTOL_FXY,
        err_msg=f"LAMMPS vs FD F_x,F_y (δ = {delta} Å, 3× cell, centered)",
    )


def main():
    """CLI: run F_z test on primitive cell, then F_x,F_y on 3× centered cell."""
    atoms = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    te_p, kc_p = _load_tersoff_kolmogorov_crespi_params()
    calc = TersoffKolmogorovCrespiASECalculator(
        te_p.tolist(), kc_p.tolist(), layer_tags=_layer_tags_from_mol_id(atoms)
    )
    atoms.calc = calc

    delta = 5e-4
    res = calc.calculate(atoms)
    fz_lammps = np.asarray(res["forces"], dtype=float)[:, 2]
    e0 = res["energy"]
    fz_fd = _central_finite_diff_fz_energy_only(calc, atoms, delta=delta)

    diff = fz_lammps - fz_fd
    print("fz lammps = ", fz_lammps)
    print("fz fd = ", fz_fd)
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    print(f"Atoms: {len(atoms)}, E = {e0:.8f} eV, δ = {delta} Å (F_z only)")
    print(f"max |Fz_lammps - Fz_fd| = {max_abs:.6e} eV/Å")
    print(f"RMS |diff|              = {rms:.6e} eV/Å")

    try:
        np.testing.assert_allclose(
            fz_lammps, fz_fd, atol=_ATOL_FZ, rtol=_RTOL_FZ
        )
    except AssertionError:
        print("FAIL: F_z inconsistent with finite-difference energies.")
        return 1
    print("PASS: LAMMPS F_z consistent with finite-difference energies.")

    atoms2 = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    _expand_cell_and_center_atoms(atoms2, scale=3.0)
    calc2 = TersoffKolmogorovCrespiASECalculator(
        te_p.tolist(), kc_p.tolist(), layer_tags=_layer_tags_from_mol_id(atoms2)
    )
    atoms2.calc = calc2
    res2 = calc2.calculate(atoms2)
    fxy_l = np.asarray(res2["forces"], dtype=float)[:, :2]
    fxy_d = _central_finite_diff_fxy_energy_only(calc2, atoms2, delta=delta)
    print("fxy lammps = ", np.round(fxy_l,3))
    print("fxy fd = ", np.round(fxy_d,3))
    dxy = fxy_l - fxy_d
    print(
        f"\nF_x,F_y (3× centered cell): max|diff|={np.max(np.abs(dxy)):.6e} eV/Å"
    )
    try:
        np.testing.assert_allclose(
            fxy_l, fxy_d, atol=_ATOL_FXY, rtol=_RTOL_FXY
        )
    except AssertionError:
        print("FAIL: F_x,F_y inconsistent (expanded cell).")
        return 1
    print("PASS: F_x,F_y OK on expanded centered cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
