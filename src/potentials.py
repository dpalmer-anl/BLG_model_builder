"""
potentials.py — high-level Python interface for LAMMPS-backed interatomic potentials.

All interatomic potentials (Tersoff, Kolmogorov-Crespi Full, DRIP, POD) are now
called through the official LAMMPS Python module (``from lammps import lammps``).
The former C++ pybind11 ``potential_ext`` extension has been removed.

Public API
----------
Geometry conversion utilities (used internally and by lammps_interface.py):
  - ase_to_lammps(atoms)             Prism + convert → LAMMPS metal frame
  - lammps_to_ase_forces(f, atoms)   inverse Prism; LAMMPS metal → ASE frame
  - lammps_positions_to_ase(x, atoms, cell)   map relaxed coords back to ASE
  - lammps_molecule_ids_from_atoms   mol-id column for full/molecular atom_style

POD descriptor helpers:
  - ncoeff_from_params(params)       compute ncoeff from a hyperparameter dict
  - init_pod_coefficients(params)    zero-initialised coefficient vector
  - pod_hyperparams_to_str(...)      build .pod descriptor file content
  - parse_pod_param_file(path)       read a .pod file into a dict

Calculator classes (backed by LAMMPS Python module):
  - TersoffLammpsCalculator
  - KolmogorovCrespiLammpsCalculator
  - DRIPLammpsCalculator
  - TersoffKCLammpsCalculator
  - TersoffDRIPLammpsCalculator
  - PODLammpsCalculator
  - TETB_PODLammpsCalculator / TETB_PODASECalculator

Backward-compatible aliases (same names as the old C++ wrapper classes):
  - TersoffASECalculator            → TersoffLammpsCalculator
  - KolmogorovCrespiASECalculator   → KolmogorovCrespiLammpsCalculator
  - DRIPASECalculator               → DRIPLammpsCalculator
  - TersoffKolmogorovCrespiASECalculator → TersoffKCLammpsCalculator
  - TersoffDRIPASECalculator        → TersoffDRIPLammpsCalculator
  - PODASECalculator                → PODLammpsCalculator

Relaxation helper (patched onto ase.Atoms):
  - Atoms.relax_structure()         delegates to ``calc.relax_structure()``
"""

from __future__ import annotations

import sys
import numpy as np
from typing import Any, Dict, List, Optional

from ase.calculators.lammps import Prism, convert


# ═════════════════════════════════════════════════════════════════════════════
#  POD ncoeff computation
# ═════════════════════════════════════════════════════════════════════════════

_DABF3 = list(range(13))
_DABF4 = [0,1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,6]
_NB    = [1, 2, 4, 7, 11, 16, 23]


def _crossindices_count(dabf1, nabf1, nrbf1, nebf1,
                        dabf2, nabf2, nrbf2, nebf2,
                        dabf12, nrbf12) -> int:
    """Python port of EAPOD::crossindices (count-only overload)."""
    n = 0
    for i1 in range(nebf1):
        for j1 in range(nrbf1):
            for k1 in range(nabf1):
                m1 = k1 + j1 * nabf1
                a1 = dabf1[k1]
                for i2 in range(nebf2):
                    for j2 in range(nrbf2):
                        for k2 in range(nabf2):
                            m2 = k2 + j2 * nabf2
                            a2 = dabf2[k2]
                            if (m2 >= m1 and i2 >= i1
                                    and a1 + a2 <= dabf12
                                    and j1 + j2 < nrbf12):
                                n += 1
    return n


def ncoeff_from_params(params: Dict) -> int:
    """
    Compute the total number of POD coefficients from a hyperparameter dict.

    The dict keys match keywords in a .pod descriptor file:

        species                                 list[str]  (required)
        onebody                                 int        default 1
        twobody_number_radial_basis_functions   int        default 0
        threebody_number_radial_basis_functions int        default 0
        threebody_angular_degree                int        default 4
        fourbody_number_radial_basis_functions  int        default 0
        fourbody_angular_degree                 int        default 3
        fivebody_number_radial_basis_functions  int        default 0
        fivebody_angular_degree                 int        default 3
        sixbody_number_radial_basis_functions   int        default 0
        sixbody_angular_degree                  int        default 2
        sevenbody_number_radial_basis_functions int        default 0
        sevenbody_angular_degree                int        default 2
    """
    sp = params.get("species", ["C"])
    if isinstance(sp, str):
        sp = sp.split()
    Ne = len(sp)
    if Ne == 0:
        raise ValueError("params must contain 'species' with at least one element")

    onebody = int(params.get("onebody", 1))
    nrbf2   = int(params.get("twobody_number_radial_basis_functions",   0))
    nrbf3   = int(params.get("threebody_number_radial_basis_functions", 0))
    P3      = int(params.get("threebody_angular_degree",                4))
    nrbf4   = int(params.get("fourbody_number_radial_basis_functions",  0))
    P4      = int(params.get("fourbody_angular_degree",                 3))
    nrbf33  = int(params.get("fivebody_number_radial_basis_functions",  0))
    P33     = int(params.get("fivebody_angular_degree",                 3))
    nrbf34  = int(params.get("sixbody_number_radial_basis_functions",   0))
    P34     = int(params.get("sixbody_angular_degree",                  2))
    nrbf44  = int(params.get("sevenbody_number_radial_basis_functions", 0))
    P44     = int(params.get("sevenbody_angular_degree",                2))

    nabf3  = P3  + 1
    nabf4  = _NB[P4]
    nabf33 = P33 + 1
    nabf34 = P34 + 1
    nabf44 = _NB[P44]

    nebf3 = Ne * (Ne + 1) // 2
    nebf4 = Ne * (Ne + 1) * (Ne + 2) // 6

    nl1 = onebody * Ne
    nl2 = nrbf2 * Ne
    nl3 = nabf3 * nrbf3 * nebf3
    nl4 = nabf4 * nrbf4 * nebf4

    dabf3 = _DABF3[:nabf3]
    dabf4 = _DABF4[:nabf4]

    nld33 = nld34 = nld44 = 0
    if nrbf33 > 0:
        nld33 = _crossindices_count(
            dabf3, nabf3, nrbf3, nebf3,
            dabf3, nabf3, nrbf3, nebf3, P33, nrbf33)
    if nrbf34 > 0:
        nld34 = _crossindices_count(
            dabf3, nabf3, nrbf3, nebf3,
            dabf4, nabf4, nrbf4, nebf4, P34, nrbf34)
    if nrbf44 > 0:
        nld44 = _crossindices_count(
            dabf4, nabf4, nrbf4, nebf4,
            dabf4, nabf4, nrbf4, nebf4, P44, nrbf44)

    Mdesc = nl2 + nl3 + nl4 + nld33 + nld34 + nld44
    nCoeffPerElement = nl1 + Mdesc
    return nCoeffPerElement * Ne


def init_pod_coefficients(params: Dict, fill: float = 0.0) -> np.ndarray:
    """Return a zero-initialised coefficient vector of the correct length."""
    return np.full(ncoeff_from_params(params), fill, dtype=np.float64)


def pod_hyperparams_to_str(hyperparams: Dict, cutoff: float, elements: List[str]) -> str:
    """Build POD descriptor file string from hyperparameters."""
    hp = hyperparams
    lines = [
        "species " + " ".join(elements),
        "pbc 1 1 1",
        "rin 1.0",
        f"rcut {cutoff}",
        f"bessel_polynomial_degree {hp['bessel_polynomial_degree']}",
        f"inverse_polynomial_degree {hp['inverse_polynomial_degree']}",
        f"twobody_number_radial_basis_functions "
            f"{hp['twobody_number_radial_basis_functions']}",
        f"threebody_number_radial_basis_functions "
            f"{hp['threebody_number_radial_basis_functions']}",
        f"threebody_angular_degree {hp['threebody_angular_degree']}",
        f"fourbody_number_radial_basis_functions "
            f"{hp['fourbody_number_radial_basis_functions']}",
        f"fourbody_angular_degree {hp['fourbody_angular_degree']}",
        f"fivebody_number_radial_basis_functions "
            f"{hp['fivebody_number_radial_basis_functions']}",
        f"fivebody_angular_degree {hp['fivebody_angular_degree']}",
        f"sixbody_number_radial_basis_functions "
            f"{hp['sixbody_number_radial_basis_functions']}",
        f"sixbody_angular_degree {hp['sixbody_angular_degree']}",
        f"sevenbody_number_radial_basis_functions "
            f"{hp['sevenbody_number_radial_basis_functions']}",
        f"sevenbody_angular_degree {hp['sevenbody_angular_degree']}",
    ]
    return "\n".join(lines) + "\n"


def parse_pod_param_file(path: str) -> Dict:
    """Parse a .pod descriptor file into a dict."""
    result: Dict = {}
    with open(path) as f:
        for raw in f:
            line = raw.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key  = parts[0]
            vals = parts[1:]
            result[key] = vals[0] if len(vals) == 1 else vals
    if 'species' in result:
        sp = result['species']
        result['species'] = sp if isinstance(sp, list) else [sp]
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  ASE ↔ LAMMPS geometry conversion
# ═════════════════════════════════════════════════════════════════════════════

_LAMMPS_UNITS = "metal"


def _prism_for_atoms(atoms, reduce_cell: bool = False) -> Prism:
    """
    Same cell handling as :func:`ase.io.lammpsdata.write_lammps_data`:
    ASE :class:`~ase.calculators.lammps.Prism` (QR decomposition + optional skew flips).
    """
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    pbc  = tuple(bool(x) for x in atoms.get_pbc())
    try:
        return Prism(cell, pbc=pbc, reduce_cell=reduce_cell)
    except TypeError:
        return Prism(cell, pbc=pbc)


def lammps_molecule_ids_from_atoms(
    atoms,
    layer_tags: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Per-atom LAMMPS molecule IDs for ``full`` / molecular ``atom_style``.

    Priority:
    1. ``atoms.get_array('mol-id')`` if present.
    2. ``layer_tags`` (0-based) → stored as ``tag + 1``.
    3. Inferred from z midplane → ``1 + (z > z_mid)``.
    """
    n = len(atoms)
    if layer_tags is not None:
        out = np.asarray(layer_tags, dtype=np.int32).ravel()
        if out.shape[0] != n:
            raise ValueError("layer_tags must have one entry per atom")
        return out + 1
    if atoms.has("mol-id"):
        mid = np.asarray(atoms.get_array("mol-id"), dtype=np.int32).ravel()
        if mid.shape[0] != n:
            raise ValueError('atoms array "mol-id" must have length len(atoms)')
        return mid
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    z_mid = 0.5 * (pos[:, 2].min() + pos[:, 2].max())
    layers = (pos[:, 2] > z_mid).astype(np.int32)
    return layers + 1


def ase_to_lammps(atoms, element_order: Optional[List[str]] = None):
    """
    Convert an ``ase.Atoms`` object to arrays in the LAMMPS metal frame.

    Returns
    -------
    pos_lammps : ndarray (N, 3)   positions in Å (LAMMPS Prism frame)
    types      : ndarray (N,)     1-based atom type indices
    lammps_cell: ndarray (6,)     [xhi, yhi, zhi, xy, xz, yz] in Å
    element_order : list[str]     element symbols in type-index order
    """
    prism = _prism_for_atoms(atoms, reduce_cell=False)
    lammps_cell = np.asarray(prism.lammps_cell, dtype=np.float64)
    pos_ase = np.asarray(atoms.get_positions(wrap=False), dtype=np.float64)
    pos_lammps = prism.vector_to_lammps(pos_ase, wrap=False)
    pos_lammps = convert(pos_lammps, "distance", "ASE", _LAMMPS_UNITS)

    symbols = atoms.get_chemical_symbols()
    if element_order is None:
        seen: List[str] = []
        for s in symbols:
            if s not in seen:
                seen.append(s)
        element_order = seen

    type_map = {s: i + 1 for i, s in enumerate(element_order)}
    types = np.array([type_map[s] for s in symbols], dtype=np.int32)

    return pos_lammps.astype(np.float64), types, lammps_cell, element_order


def lammps_to_ase_forces(forces_lammps: np.ndarray, atoms) -> np.ndarray:
    """Map LAMMPS-metal forces to the original ASE Cartesian frame (inverse Prism)."""
    prism = _prism_for_atoms(atoms)
    f = convert(
        np.asarray(forces_lammps, dtype=np.float64),
        "force",
        _LAMMPS_UNITS,
        "ASE",
    )
    return np.asarray(prism.vector_to_ase(f, wrap=False), dtype=np.float64)


def lammps_positions_to_ase(
    pos_lammps: np.ndarray, atoms, lammps_cell: np.ndarray
) -> np.ndarray:
    """
    Map relaxed LAMMPS coordinates to ASE Cartesian Å, inverting :func:`ase_to_lammps`.

    ``lammps_cell`` is unused; kept for call-site compatibility.
    """
    prism = _prism_for_atoms(atoms)
    r = convert(
        np.asarray(pos_lammps, dtype=np.float64),
        "distance",
        _LAMMPS_UNITS,
        "ASE",
    )
    return np.asarray(prism.vector_to_ase(r, wrap=False), dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
#  FIRE relaxation helpers
# ═════════════════════════════════════════════════════════════════════════════

_FIRE_RELAX_DEFAULT_TIMESTEP = 0.00025
_FIRE_RELAX_DEFAULT_ETOL = 1e-4
_FIRE_RELAX_DEFAULT_FTOL = 1e-5
_FIRE_RELAX_DEFAULT_MAXITER = 1000
_FIRE_RELAX_DEFAULT_MAXEVAL = 10000

_ASE_FIRE_DT = 0.1
_ASE_FIRE_MAXSTEP = 0.1
_ASE_FIRE_DTMAX = 1.0
_ASE_FIRE_DOWNHILL_CHECK = False


def _geometry_changed(atoms, last_pos, last_cell) -> bool:
    """Return True if atom positions or cell have changed since last call."""
    if last_pos is None or last_cell is None:
        return True
    pos = atoms.get_positions()
    if pos.shape != np.asarray(last_pos).shape:
        return True
    if not np.allclose(atoms.get_cell(), last_cell, atol=1e-10):
        return True
    if not np.allclose(pos, last_pos, atol=1e-10):
        return True
    return False


def _annotate_relax_energies(calc, out, energy_fire: float) -> None:
    """Store FIRE endpoint energy and a single-point re-evaluation on ``out``."""
    out.info["lammps_fire_relax_energy"] = float(energy_fire)
    e_sp = calc.get_potential_energy(out)
    out.info["lammps_pe_singlepoint"] = float(e_sp)
    out.info["lammps_fire_vs_singlepoint_de"] = float(abs(e_sp - float(energy_fire)))


def _normalize_relax_backend(relax_backend: str) -> str:
    b = relax_backend.strip().lower()
    if b in ("ase", "lammps"):
        return b
    raise ValueError(
        f"relax_backend must be 'ase' or 'lammps', got {relax_backend!r}"
    )


def _relaxation_diagnostics_message(calc, atoms) -> str:
    """Human-readable snapshot for failed relaxations."""
    lines: List[str] = []
    if atoms is not None:
        try:
            pos = atoms.get_positions()
            n = len(atoms)
            z = pos[:, 2]
            lines.append(
                f"Structure: N={n} atoms, z span [{float(z.min()):.4f}, {float(z.max()):.4f}] Å"
            )
            if n >= 2 and n <= 500:
                d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                iu = np.triu_indices(n, k=1)
                dmin = float(d[iu].min())
                lines.append(
                    f"Min pairwise distance (raw Cartesian, no MIC): {dmin:.4f} Å"
                )
        except Exception as ex:
            lines.append(f"(Could not summarize positions: {ex})")

    res = getattr(calc, "results", None)
    if isinstance(res, dict) and res.get("energy") is not None:
        lines.append(f"Cached calculator.results energy: {float(res['energy']):.10f} eV")
        if res.get("forces") is not None:
            f = np.asarray(res["forces"], dtype=float)
            fn = np.linalg.norm(f, axis=1)
            lines.append(
                f"Cached forces: fmax={float(fn.max()):.6e} eV/Å, "
                f"mean|F|={float(fn.mean()):.6e} eV/Å"
            )
    else:
        lines.append("No cached calculator.results.")

    if atoms is not None:
        lines.append("Fresh calculate() at current Atoms (may repeat the same error):")
        try:
            r2 = calc.calculate(atoms)
            e2 = r2.get("energy")
            f2 = np.asarray(r2.get("forces", [[]]))
            lines.append(f"  energy={e2:.10f} eV")
            if f2.size:
                fn2 = np.linalg.norm(f2, axis=1)
                lines.append(
                    f"  fmax={float(fn2.max()):.6e} eV/Å, "
                    f"mean|F|={float(fn2.mean()):.6e} eV/Å"
                )
        except Exception as ex2:
            lines.append(f"  (re-evaluation also failed: {ex2})")

    return "\n".join(lines)


def _reraise_relax_failed(label: str, calc, atoms, exc: Exception) -> None:
    diagnostics = _relaxation_diagnostics_message(calc, atoms)
    msg = (
        f"Relaxation failed in {label}\n"
        f"Original error ({type(exc).__name__}): {exc}\n\n"
        f"--- diagnostics ---\n{diagnostics}"
    )
    print(msg, file=sys.stderr)
    raise RuntimeError(msg) from exc


def _relax_structure_ase(
    calc,
    out,
    *,
    fmax: float,
    steps: int,
    fire_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Relax ``out`` with ASE :class:`~ase.optimize.FIRE` using ``calc``.

    Uses conservative defaults (small dt, maxstep) for stability on classical
    interlayer models; override via ``fire_kwargs``.
    """
    from ase.optimize import FIRE  # noqa: PLC0415

    kw: Dict[str, Any] = dict(
        logfile="log",
        dt=_ASE_FIRE_DT,
        maxstep=_ASE_FIRE_MAXSTEP,
        dtmax=_ASE_FIRE_DTMAX,
        downhill_check=_ASE_FIRE_DOWNHILL_CHECK,
    )
    if fire_kwargs:
        kw.update(fire_kwargs)

    out.calc = calc
    dyn = FIRE(out, **kw)
    try:
        dyn.run(fmax=fmax, steps=steps)
    except Exception as exc:
        _reraise_relax_failed(
            f"{type(calc).__name__}.relax_structure (ASE FIRE)",
            calc,
            out,
            exc,
        )
    return float(calc.get_potential_energy(out))


# ═════════════════════════════════════════════════════════════════════════════
#  LAMMPS Python module interface
#
#  All interatomic potential calculators are now backed by the official LAMMPS
#  Python module (``from lammps import lammps``).  The former C++ pybind11
#  ``potential_ext`` extension has been removed.
#
#  Install: build LAMMPS with ``make install-python``  (or cmake equivalent).
# ═════════════════════════════════════════════════════════════════════════════

from blg_model_builder_v2.lammps_interface import (  # noqa: E402
    LammpsCalculatorBase,
    TersoffLammpsCalculator,
    KolmogorovCrespiLammpsCalculator,
    DRIPLammpsCalculator,
    TersoffKCLammpsCalculator,
    TersoffDRIPLammpsCalculator,
    PODLammpsCalculator,
    TETB_PODLammpsCalculator,
    _write_tersoff_file,
    _write_kc_file,
    _write_drip_file,
    _write_pod_param_file,
    _write_pod_coeff_file,
)

# Backward-compatible aliases — existing call-sites (get_MCMC_inputs.py,
# model_fit.py, tests) continue to work without modification.
TersoffASECalculator                 = TersoffLammpsCalculator
KolmogorovCrespiASECalculator        = KolmogorovCrespiLammpsCalculator
DRIPASECalculator                    = DRIPLammpsCalculator
TersoffKolmogorovCrespiASECalculator = TersoffKCLammpsCalculator
TersoffDRIPASECalculator             = TersoffDRIPLammpsCalculator
PODASECalculator                     = PODLammpsCalculator
TETB_PODASECalculator                = TETB_PODLammpsCalculator

_LAMMPS_PY_INTERFACE_AVAILABLE = True  # always True; _ext path has been removed


# ═════════════════════════════════════════════════════════════════════════════
#  Atoms.relax_structure() convenience method
# ═════════════════════════════════════════════════════════════════════════════

def _atoms_relax_structure(
    self,
    timestep: float = _FIRE_RELAX_DEFAULT_TIMESTEP,
    etol: float = _FIRE_RELAX_DEFAULT_ETOL,
    ftol: float = _FIRE_RELAX_DEFAULT_FTOL,
    maxiter: int = _FIRE_RELAX_DEFAULT_MAXITER,
    maxeval: int = _FIRE_RELAX_DEFAULT_MAXEVAL,
    *,
    relax_backend: str = "ase",
    relax_fire_kwargs: Optional[Dict[str, Any]] = None,
):
    if self.calc is None:
        raise ValueError("Atoms.calc must be set before relax_structure()")
    relax_fn = getattr(self.calc, "relax_structure", None)
    if relax_fn is None:
        raise TypeError(
            f"Calculator {type(self.calc).__name__} does not implement relax_structure()"
        )
    return relax_fn(
        self,
        timestep=timestep,
        etol=etol,
        ftol=ftol,
        maxiter=maxiter,
        maxeval=maxeval,
        relax_backend=relax_backend,
        relax_fire_kwargs=relax_fire_kwargs,
    )


def _patch_ase_atoms_relax_structure() -> None:
    from ase import Atoms  # noqa: PLC0415

    if getattr(Atoms, "_blg_relax_structure_patched", False):
        return
    Atoms.relax_structure = _atoms_relax_structure  # type: ignore[method-assign]
    Atoms._blg_relax_structure_patched = True  # type: ignore[attr-defined]


_patch_ase_atoms_relax_structure()
