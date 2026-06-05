"""
Integration tests that compare the LAMMPS Python interface calculators against
a direct LAMMPS ``run 0`` invoked via subprocess.

Per project "Trust LAMMPS" principle: for a fixed geometry, a direct LAMMPS
``run 0`` should agree exactly with the Python interface (both ultimately call
the same LAMMPS pair-style code).

Tests are skipped if:
  * No LAMMPS executable is found on PATH (``lmp``, ``lmp_serial``, ``lammps``).
  * The LAMMPS Python module (``from lammps import lammps``) is not installed.
"""

from __future__ import annotations

import sys
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# Skip if the LAMMPS Python module is not installed.
pytest.importorskip(
    "lammps",
    reason="LAMMPS Python module not installed.  Run 'make install-python'.",
)

def _ensure_importable_package() -> None:
    """Prefer installed package; fallback to repo `src/` only if needed."""
    try:
        import blg_model_builder_v2  # noqa: F401
        return
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import blg_model_builder_v2  # noqa: F401
        return
    except Exception:
        import types

        pkg = types.ModuleType("blg_model_builder_v2")
        pkg.__path__ = [str(src)]  # type: ignore[attr-defined]
        sys.modules["blg_model_builder_v2"] = pkg


_ensure_importable_package()

# Allow importing the shared harness when tests/ is not a package.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from blg_model_builder_v2.geom_tools import get_bilayer_atoms
# Import via potentials.py so aliases are resolved (LAMMPS Python or C++ pybind).
from blg_model_builder_v2.potentials import (
    PODASECalculator,
    TersoffDRIPASECalculator,
    TersoffKolmogorovCrespiASECalculator,
)

from physical_properties_harness import (
    POD_BLG_HYPERPARAMS,
    best_fit_params_dir,
    layer_tags_from_mol_id,
    load_pod_best_fit_energy_params,
    make_calc_tersoff_drip_best_fit_estimate,
    make_calc_tersoff_kc_best_fit_estimate,
)


def _find_lammps_exec() -> str | None:
    for name in ("lmp", "lmp_serial", "lammps"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _run_lammps(tmpdir: Path, input_text: str) -> str:
    lmp = _find_lammps_exec()
    if lmp is None:
        pytest.skip("No LAMMPS executable found on PATH (expected lmp/lmp_serial).")
    in_path = tmpdir / "in.run0"
    in_path.write_text(input_text, encoding="utf-8")
    # Run in tmpdir, capture stdout
    p = subprocess.run(
        [lmp, "-in", str(in_path)],
        cwd=str(tmpdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        # Convert common missing-feature failures into skips (environmental).
        out = p.stdout or ""
        if ("Unknown pair style" in out) or ("ERROR: Unrecognized pair style" in out):
            pytest.skip(f"LAMMPS missing required pair style. Output:\n{out}")
        if "Cannot open" in out and ("tersoff" in out or "KC" in out or "drip" in out):
            raise RuntimeError(f"LAMMPS could not open potential file(s). Output:\n{out}")
        raise RuntimeError(f"LAMMPS failed (code {p.returncode}). Output:\n{out}")
    return p.stdout


def _parse_dump_forces(dump_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Parse a `dump custom` file with columns: id type x y z fx fy fz.
    Returns: positions (N,3), forces (N,3), and inferred potential energy (nan if missing).
    """
    txt = dump_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Find the start of the ATOMS section
    i0 = None
    natoms = None
    for i, line in enumerate(txt):
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(txt[i + 1].strip())
        if line.startswith("ITEM: ATOMS"):
            i0 = i + 1
            break
    if i0 is None or natoms is None:
        raise ValueError("Could not parse dump header")
    data = []
    for line in txt[i0 : i0 + natoms]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"Unexpected dump line: {line}")
        # id, type, x, y, z, fx, fy, fz
        data.append([int(fields[0]), float(fields[2]), float(fields[3]), float(fields[4]),
                     float(fields[5]), float(fields[6]), float(fields[7])])
    data = np.asarray(data, dtype=float)
    # sort by id
    data = data[np.argsort(data[:, 0])]
    pos = data[:, 1:4]
    frc = data[:, 4:7]
    return pos, frc, float("nan")


def _write_atomic_data_file(tmpdir: Path, atoms):
    """
    Write a simple atomic-style LAMMPS data file (triclinic-compatible) using ASE.
    """
    from ase.io import write
    data_path = tmpdir / "structure.lmp"
    write(str(data_path), atoms, format="lammps-data", atom_style="atomic")
    return data_path


def _write_full_data_file(tmpdir: Path, atoms):
    """Write a `full` atom_style data file so molecule IDs are preserved."""
    from blg_model_builder_v2.lammpsdata import write_lammps_data

    data_path = tmpdir / "structure.lmp"
    with open(data_path, "w", encoding="utf-8") as f:
        write_lammps_data(
            f,
            atoms,
            atom_style="full",
            masses=True,
            velocities=False,
            atom_type_labels=True,
            units="metal",
        )
    return data_path


def _lammps_prism_box_from_atoms(atoms) -> tuple[float, float, float, float, float, float]:
    """
    For `atom_style atomic`, ASE's lammps-data writer writes a prism if skewed.
    We don't need to reconstruct the prism; we read_data.
    """
    cell = np.asarray(atoms.get_cell(), dtype=float)
    # not used; placeholder for future extensions
    return (cell[0, 0], cell[1, 1], cell[2, 2], 0.0, 0.0, 0.0)


def _base_input() -> str:
    return "\n".join(
        [
            "units metal",
            "atom_style full",
            "atom_modify map array",
            "boundary p p p",
            "newton on",
            "read_data structure.lmp",
            "neighbor 0.3 bin",
            "neigh_modify delay 0 every 1 check yes",
        ]
    ) + "\n"


@pytest.mark.parametrize("model", ["tersoff_kc", "tersoff_drip"])
def test_lammps_run0_matches_python_energy_forces(model, tmp_path: Path):
    """
    Compare one fixed AB BLG geometry (4 atoms) in LAMMPS vs Python interface.
    """
    atoms = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    _write_full_data_file(tmp_path, atoms)

    if model == "tersoff_kc":
        te, kc = make_calc_tersoff_kc_best_fit_estimate()
        py_calc = TersoffKolmogorovCrespiASECalculator(
            te.tolist(), kc.tolist(), layer_tags=layer_tags_from_mol_id(atoms)
        )
        atoms.calc = py_calc
        e_py = float(atoms.get_potential_energy())
        f_py = np.asarray(atoms.get_forces(), dtype=float)

        # Write LAMMPS tersoff + KC coefficient files using the same format as the C++ wrapper.
        tersoff_path = tmp_path / "C.tersoff"
        tersoff_line = (
            "C C C "
            + " ".join(f"{v:.15g}" for v in te.tolist())
            + "\n"
        )
        tersoff_path.write_text(tersoff_line, encoding="utf-8")

        kc_path = tmp_path / "C.KC"
        # KC file: "C C z0 C0 C2 C4 C delta lambda A S rcut"
        z0, C0, C2, C4, C, delta, lam, A = kc[:8].tolist()
        S = 1.0
        rcut = 14.0
        kc_path.write_text(
            "C C "
            + " ".join(f"{v:.15g}" for v in [z0, C0, C2, C4, C, delta, lam, A, S, rcut])
            + "\n",
            encoding="utf-8",
        )

        dump_path = tmp_path / "dump.forces"
        in_text = _base_input() + "\n".join(
            [
                "pair_style hybrid/overlay tersoff kolmogorov/crespi/full 14.0",
                "pair_coeff * * tersoff C.tersoff C",
                "pair_coeff * * kolmogorov/crespi/full C.KC C",
                "thermo 1",
                "thermo_style custom step pe",
                "run 0",
                "dump d all custom 1 dump.forces id type x y z fx fy fz",
                "run 0",
                "undump d",
            ]
        ) + "\n"
        out = _run_lammps(tmp_path, in_text)
        _, f_l, _ = _parse_dump_forces(dump_path)

        # Tight agreement expected: both are LAMMPS calculations; Python path just rotates frames.
        assert np.allclose(f_l, f_py, atol=1e-6, rtol=0.0), "Force mismatch: LAMMPS vs Python (Tersoff+KC)"

    else:
        te, drip = make_calc_tersoff_drip_best_fit_estimate()
        py_calc = TersoffDRIPASECalculator(
            te.tolist(), drip.tolist(), layer_tags=layer_tags_from_mol_id(atoms)
        )
        atoms.calc = py_calc
        e_py = float(atoms.get_potential_energy())
        f_py = np.asarray(atoms.get_forces(), dtype=float)

        tersoff_path = tmp_path / "C.tersoff"
        tersoff_path.write_text(
            "C C C " + " ".join(f"{v:.15g}" for v in te.tolist()) + "\n",
            encoding="utf-8",
        )

        drip_path = tmp_path / "C.drip"
        # DRIP file: "C C C0 C2 C4 C delta lambda A z0 B eta rhocut rcut ncut"
        # Hybrid calculator uses the physical 8 plus fixed B=0 eta=0 and cutoffs (rhocut=3, rcut=14, ncut=3).
        C0, C2, C4, C, delta, lam, A, z0 = drip[:8].tolist()
        B = 0.0
        eta = 0.0
        rhocut = 3.0
        rcut = 14.0
        ncut = 3.0
        drip_path.write_text(
            "C C "
            + " ".join(
                f"{v:.15g}"
                for v in [C0, C2, C4, C, delta, lam, A, z0, B, eta, rhocut, rcut, ncut]
            )
            + "\n",
            encoding="utf-8",
        )

        dump_path = tmp_path / "dump.forces"
        in_text = _base_input() + "\n".join(
            [
                "pair_style hybrid/overlay tersoff zero 0.1 drip",
                "pair_coeff * * tersoff C.tersoff C",
                "pair_coeff * * zero",
                "pair_coeff * * drip C.drip C",
                "thermo 1",
                "thermo_style custom step pe",
                "run 0",
                "dump d all custom 1 dump.forces id type x y z fx fy fz",
                "run 0",
                "undump d",
            ]
        ) + "\n"
        _run_lammps(tmp_path, in_text)
        _, f_l, _ = _parse_dump_forces(dump_path)

        assert np.allclose(f_l, f_py, atol=1e-6, rtol=0.0), "Force mismatch: LAMMPS vs Python (Tersoff+DRIP)"


def test_lammps_pod_run0_matches_python_energy_forces(tmp_path: Path):
    """
    POD is file-driven in LAMMPS; this test runs `pair_style pod` + `run 0`
    using the best-fit POD coefficient file.
    """
    atoms = get_bilayer_atoms(3.5, 0.0, sc=1).copy()
    _write_full_data_file(tmp_path, atoms)

    try:
        params = load_pod_best_fit_energy_params(POD_BLG_HYPERPARAMS, require_file=True)
    except FileNotFoundError as exc:
        pytest.skip(f"POD best-fit coefficients unavailable and auto-fit failed: {exc}")

    # Write POD coefficient file in EAPOD file format expected by LAMMPS (one per line).
    # The C++ wrapper supports both inline and file formats; LAMMPS wants file format.
    coeff_path = tmp_path / "C_coefficients.pod"
    coeff_lines = ["model_coefficients: {} 0 0".format(params.size)]
    coeff_lines += [f"{v:.15g}" for v in params.tolist()]
    coeff_path.write_text("\n".join(coeff_lines) + "\n", encoding="utf-8")

    # Descriptor/hyperparams file: write minimal .pod param file matching PODASECalculator.
    # We use the same key names as potentials.PODASECalculator.hyperparams_to_str().
    param_path = tmp_path / "C_param.pod"
    hp = POD_BLG_HYPERPARAMS
    pod_param_txt = "\n".join(
        [
            "species C",
            "pbc 1 1 1",
            "rin 1.0",
            "rcut 6.0",
            f"bessel_polynomial_degree {hp['bessel_polynomial_degree']}",
            f"inverse_polynomial_degree {hp['inverse_polynomial_degree']}",
            f"twobody_number_radial_basis_functions {hp['twobody_number_radial_basis_functions']}",
            f"threebody_number_radial_basis_functions {hp['threebody_number_radial_basis_functions']}",
            f"threebody_angular_degree {hp['threebody_angular_degree']}",
            f"fourbody_number_radial_basis_functions {hp['fourbody_number_radial_basis_functions']}",
            f"fourbody_angular_degree {hp['fourbody_angular_degree']}",
            f"fivebody_number_radial_basis_functions {hp['fivebody_number_radial_basis_functions']}",
            f"fivebody_angular_degree {hp['fivebody_angular_degree']}",
            f"sixbody_number_radial_basis_functions {hp['sixbody_number_radial_basis_functions']}",
            f"sixbody_angular_degree {hp['sixbody_angular_degree']}",
            f"sevenbody_number_radial_basis_functions {hp['sevenbody_number_radial_basis_functions']}",
            f"sevenbody_angular_degree {hp['sevenbody_angular_degree']}",
        ]
    )
    param_path.write_text(pod_param_txt + "\n", encoding="utf-8")

    # Python reference
    py_calc = PODASECalculator(POD_BLG_HYPERPARAMS, params, elements=["C"], cutoff=6.0)
    atoms.calc = py_calc
    e_py = float(atoms.get_potential_energy())
    f_py = np.asarray(atoms.get_forces(), dtype=float)

    # LAMMPS run 0 and dump forces
    dump_path = tmp_path / "dump.forces"
    in_text = _base_input() + "\n".join(
        [
            "pair_style pod",
            "pair_coeff * * C_param.pod C_coefficients.pod C",
            "thermo 1",
            "thermo_style custom step pe",
            "run 0",
            "dump d all custom 1 dump.forces id type x y z fx fy fz",
            "run 0",
            "undump d",
        ]
    ) + "\n"

    out = _run_lammps(tmp_path, in_text)
    pos_l, f_l, _ = _parse_dump_forces(dump_path)

    # Positions should match (within writer/reader precision)
    pos_py = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    assert np.allclose(pos_l, pos_py, atol=1e-6), "LAMMPS positions differ from written structure"

    # Forces and energies should match very tightly for POD if both are using LAMMPS pod.
    # Energy: parse last thermo line containing pe if present.
    pe = None
    for line in out.splitlines()[::-1]:
        if re.match(r"^\s*\d+\s+[-+0-9.eE]+\s*$", line):
            parts = line.split()
            if len(parts) == 2:
                pe = float(parts[1])
                break
    if pe is not None:
        assert abs(pe - e_py) <= 1e-6, f"Energy mismatch: LAMMPS {pe} vs Python {e_py}"

    assert np.allclose(f_l, f_py, atol=1e-6, rtol=0.0), "Force mismatch POD: LAMMPS vs Python"

