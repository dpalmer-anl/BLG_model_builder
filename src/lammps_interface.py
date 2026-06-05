"""
lammps_interface.py — ASE-compatible calculators using the LAMMPS Python module.

Provides drop-in replacements for the C++ pybind11-backed *ASECalculator classes
using the official ``lammps`` Python module (``from lammps import lammps``).

All LAMMPS output (screen + log) is suppressed via ``cmdargs=["-screen","none","-log","none"]``.

Classes
-------
TersoffLammpsCalculator
KolmogorovCrespiLammpsCalculator
DRIPLammpsCalculator
TersoffKCLammpsCalculator        (hybrid/overlay Tersoff + KC Full)
TersoffDRIPLammpsCalculator      (hybrid/overlay Tersoff + DRIP)
PODLammpsCalculator

Batch evaluation
----------------
Every calculator exposes::

    calc.prepare_batch(atoms_list)            # write data files once
    energies, forces_list = calc.evaluate_batch(atoms_list, params)

Call ``prepare_batch`` once at the start of an MCMC run.  The data files are
reused for every ``evaluate_batch`` call; only the potential-parameter files are
rewritten when ``set_parameters`` is called.

Relaxation
----------
    relaxed = calc.relax_structure(atoms, relax_backend='lammps')  # LAMMPS FIRE
    relaxed = calc.relax_structure(atoms, relax_backend='ase')     # ASE FIRE

Notes on LAMMPS Python module installation
------------------------------------------
Build LAMMPS with ``make install-python`` (or ``cmake --install`` + ``pip install``).
The Python wrapper must be importable as ``from lammps import lammps``.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from time import time

import numpy as np

# Geometry conversion utilities (Prism transform, force frame conversion, etc.)
from blg_model_builder.potentials import (
    ase_to_lammps,
    lammps_molecule_ids_from_atoms,
    lammps_positions_to_ase,
    lammps_to_ase_forces,
    _FIRE_RELAX_DEFAULT_ETOL,
    _FIRE_RELAX_DEFAULT_FTOL,
    _FIRE_RELAX_DEFAULT_MAXITER,
    _FIRE_RELAX_DEFAULT_MAXEVAL,
    _annotate_relax_energies,
    _normalize_relax_backend,
    _relax_structure_ase,
    _reraise_relax_failed,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAMMPS instance factory
# ═══════════════════════════════════════════════════════════════════════════════

def _make_lammps_instance():
    """Create a LAMMPS instance with all output suppressed.

    When running under MPI (e.g. via schwimmbad MPIPool), LAMMPS must be
    initialized with ``MPI.COMM_SELF`` so that each rank owns an independent,
    single-process LAMMPS world.  Without this, every LAMMPS call issues MPI
    collectives over MPI_COMM_WORLD, which deadlocks when the master rank is
    blocked in pool.map() waiting for worker results while the worker rank's
    LAMMPS waits for the master to join the collective.
    """
    from lammps import lammps  # noqa: PLC0415
    try:
        from mpi4py import MPI  # noqa: PLC0415
        return lammps(
            comm=MPI.COMM_SELF,
            cmdargs=["-screen", "none", "-log", "none"],
        )
    except (ImportError, TypeError):
        # mpi4py unavailable (serial/OpenMP-only build) or this LAMMPS version
        # does not accept the comm kwarg — fall back to the default communicator.
        return lammps(cmdargs=["-screen", "none", "-log", "none"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameter file writers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_tersoff_file(path: str, params, elements: List[str]) -> None:
    """Write a Tersoff potential parameter file.

    File format (one line per triplet)::

        elem1 elem2 elem3  m gamma lambda3 c d costheta0 n beta lambda2 B R D lambda1 A

    Parameters
    ----------
    params : sequence of 14 floats
    elements : list of str, e.g. ["C"]
    """
    e = elements[0]
    param_str = " ".join(f"{v:.15g}" for v in params)
    Path(path).write_text(
        "# Tersoff potential parameters (auto-generated)\n\n"
        f"{e} {e} {e} {param_str}\n"
    )


def _write_kc_file(
    path: str, params, elements: List[str], cutoff: float = 14.0
) -> None:
    """Write a Kolmogorov-Crespi Full potential parameter file.

    File format::

        elem1 elem2  z0 C0 C2 C4 C delta lambda A S rcut

    Parameters
    ----------
    params : sequence of 8 floats [z0, C0, C2, C4, C, delta, lambda, A]
             or 9 with taper scale S appended.
    cutoff : float  outer cutoff in Å (appended to the line)
    """
    p = list(params)
    if len(p) == 8:
        p.append(1.0)   # default taper scale S
    elif len(p) != 9:
        raise ValueError(f"KC params: expected 8 or 9 values, got {len(p)}")
    e = elements[0]
    vals_str = " ".join(f"{v:.15g}" for v in p + [cutoff])
    Path(path).write_text(
        "# KC Full potential parameters (auto-generated)\n\n"
        f"{e} {e} {vals_str}\n"
    )


def _write_drip_file(
    path: str,
    params,
    elements: List[str],
    cutoff: float = 14.0,
    rhocut: float = 3.0,
    ncut: float = 3.0,
) -> None:
    """Write a DRIP potential parameter file.

    File format::

        elem1 elem2  C0 C2 C4 C delta lambda A z0 B eta rhocut rcut ncut

    Parameters
    ----------
    params : sequence of 8 floats [C0, C2, C4, C, delta, lambda, A, z0]
             or 10 with B, eta appended (B and eta are ignored and fixed at 0).
    """
    p = list(params)
    if len(p) >= 10:
        p = p[:8]
    elif len(p) != 8:
        raise ValueError(f"DRIP params: expected 8 (or 10 legacy), got {len(p)}")
    C0, C2, C4, C, delta, lam, A, z0 = p
    B, eta = 0.0, 0.0
    e = elements[0]
    vals = [C0, C2, C4, C, delta, lam, A, z0, B, eta, rhocut, cutoff, ncut]
    vals_str = " ".join(f"{v:.15g}" for v in vals)
    Path(path).write_text(
        "# DRIP potential parameters (auto-generated)\n\n"
        f"{e} {e} {vals_str}\n"
    )


def _write_pod_param_file(
    path: str,
    hyperparams: Dict,
    elements: List[str],
    cutoff: float,
) -> None:
    """Write a POD descriptor/hyperparameter file (.pod format)."""
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
    Path(path).write_text("\n".join(lines) + "\n")


def _write_pod_coeff_file(path: str, coeffs) -> None:
    """Write a POD coefficient file (EAPOD format expected by LAMMPS)."""
    arr = np.asarray(coeffs, dtype=np.float64)
    lines = [f"model_coefficients: {arr.size} 0 0"]
    lines += [f"{v:.15g}" for v in arr.tolist()]
    Path(path).write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Structure file writer
# ═══════════════════════════════════════════════════════════════════════════════

def _write_lammps_structure(
    path: str,
    atoms,
    atom_style: str = "atomic",
    layer_tags: Optional[List[int]] = None,
) -> None:
    """Write an ``ase.Atoms`` object to a LAMMPS data file.

    For ``atom_style='full'`` (required by KC and DRIP), molecule IDs are
    computed via :func:`lammps_molecule_ids_from_atoms` and stored in a copy
    of the atoms object before writing.
    """
    from blg_model_builder.lammpsdata import write_lammps_data  # noqa: PLC0415

    if atom_style == "full":
        mol_ids = lammps_molecule_ids_from_atoms(atoms, layer_tags)
        a = atoms.copy()
        a.set_array("mol-id", mol_ids.astype(int))
    else:
        a = atoms

    with open(path, "w") as fh:
        write_lammps_data(
            fh,
            a,
            atom_style=atom_style,
            masses=True,
            velocities=False,
            atom_type_labels=False,
            units="metal",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAMMPS array helpers (gather / scatter)
# ═══════════════════════════════════════════════════════════════════════════════

def _gather_forces(lmp, N: int) -> np.ndarray:
    """Gather forces from LAMMPS (LAMMPS metal frame) → (N, 3) ndarray."""
    f_c = lmp.gather_atoms("f", 1, 3)
    return np.frombuffer(f_c, dtype=np.float64).reshape(N, 3).copy()


def _gather_positions(lmp, N: int) -> np.ndarray:
    """Gather atom positions from LAMMPS (LAMMPS frame) → (N, 3) ndarray."""
    x_c = lmp.gather_atoms("x", 1, 3)
    return np.frombuffer(x_c, dtype=np.float64).reshape(N, 3).copy()


def _scatter_positions(lmp, pos_lammps: np.ndarray) -> None:
    """Scatter positions (LAMMPS frame, (N,3)) into a running LAMMPS instance."""
    x_flat = np.ascontiguousarray(pos_lammps, dtype=np.float64).flatten()
    x_c = (ctypes.c_double * len(x_flat))(*x_flat)
    lmp.scatter_atoms("x", 1, 3, x_c)


# ═══════════════════════════════════════════════════════════════════════════════
#  Base calculator
# ═══════════════════════════════════════════════════════════════════════════════

class LammpsCalculatorBase:
    """Base class for LAMMPS Python-module backed ASE calculators.

    Subclasses must implement:

    * ``_atom_style()``              → ``"atomic"`` or ``"full"``
    * ``_write_potential_files(tmp_dir)``  write potential parameter files to *tmp_dir*
    * ``_pair_style_commands(tmp_dir)``    return list of LAMMPS commands
                                           (``pair_style``, ``pair_coeff``, …)
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self) -> None:
        self._tmpdir: str = tempfile.mkdtemp(prefix="lammps_calc_")
        self._lmp = None
        self._last_cell: Optional[np.ndarray] = None
        self._last_natoms: Optional[int] = None
        self._batch_files: Optional[List[str]] = None
        self._batch_atoms: Optional[list] = None
        self.results: Dict = {}
        # Energy shift applied on top of the LAMMPS potential energy.
        # Not written to any parameter file; used only in cost-function
        # evaluation during fitting and MCMC.  Subclasses extract this from
        # the last element of their param array.
        self._shift: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        self.close()
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    def close(self) -> None:
        """Finalize and release the internal LAMMPS instance."""
        if self._lmp is not None:
            try:
                self._lmp.close()
            except Exception:
                pass
            self._lmp = None

    def _get_lmp(self):
        if self._lmp is None:
            self._lmp = _make_lammps_instance()
        return self._lmp

    # ── Subclass interface ─────────────────────────────────────────────────────

    def _atom_style(self) -> str:
        return "atomic"

    def _write_potential_files(self, tmp_dir: str) -> None:
        raise NotImplementedError

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        raise NotImplementedError

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _boundary_str(self, atoms) -> str:
        return " ".join("p" if pb else "s" for pb in atoms.get_pbc())

    def _needs_full_setup(self, atoms) -> bool:
        """True when cell or atom count changed since last full setup."""
        if self._last_cell is None or self._last_natoms is None:
            return True
        if self._last_natoms != len(atoms):
            return True
        return not np.allclose(atoms.get_cell(), self._last_cell, atol=1e-10)

    # ── LAMMPS setup ───────────────────────────────────────────────────────────

    def _setup_lammps(self, atoms) -> None:
        """Full LAMMPS setup: clear, read structure file, configure potential."""
        lmp = self._get_lmp()
        struct_path = os.path.join(self._tmpdir, "structure.lmp")
        _write_lammps_structure(
            struct_path, atoms,
            atom_style=self._atom_style(),
            layer_tags=getattr(self, "_layer_tags", None),
        )
        lmp.command("clear")
        lmp.command("units metal")
        lmp.command(f"atom_style {self._atom_style()}")
        lmp.command("atom_modify map array")
        lmp.command(f"boundary {self._boundary_str(atoms)}")
        lmp.command("newton on")
        lmp.command(f"read_data {struct_path}")
        lmp.command("neighbor 0.3 bin")
        lmp.command("neigh_modify delay 0 every 1 check yes")
        for cmd in self._pair_style_commands(self._tmpdir):
            lmp.command(cmd)
        self._last_cell = np.array(atoms.get_cell())
        self._last_natoms = len(atoms)

    def _inject_positions(self, atoms) -> None:
        """Inject new positions into LAMMPS without full re-setup.

        Valid only when the cell and atom count are unchanged (e.g. during ASE
        FIRE relaxation).  Uses ``scatter_atoms`` in the LAMMPS metal frame.
        """
        pos_lammps, _, _, _ = ase_to_lammps(atoms)
        _scatter_positions(self._get_lmp(), pos_lammps)

    def _run_and_collect(self, atoms) -> Dict:
        """Execute ``run 0`` and return ``{"energy": float, "forces": ndarray}``."""
        lmp = self._get_lmp()
        lmp.command("run 0")
        pe = float(lmp.get_thermo("pe")) + self._shift
        N = lmp.get_natoms()
        forces_lammps = _gather_forces(lmp, N)
        forces = lammps_to_ase_forces(forces_lammps, atoms)
        return {"energy": pe, "forces": forces}

    # ── Public ASE-compatible API ──────────────────────────────────────────────

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=None,
    ) -> Dict:
        """Compute energy and forces for *atoms*.

        Returns
        -------
        dict with keys ``"energy"`` (float, eV) and ``"forces"`` (ndarray, eV/Å).
        """
        if atoms is None:
            raise ValueError("atoms must be provided")
        if self._needs_full_setup(atoms):
            self._setup_lammps(atoms)
        else:
            self._inject_positions(atoms)
        self.results = self._run_and_collect(atoms)
        return self.results

    def get_potential_energy(self, atoms=None, force_consistent=None) -> float:
        return self.calculate(atoms)["energy"]

    def get_forces(self, atoms=None) -> np.ndarray:
        return self.calculate(atoms)["forces"]

    # ── Batch evaluation ───────────────────────────────────────────────────────

    def prepare_batch(
        self,
        atoms_list,
        batch_dir: Optional[str] = None,
    ) -> None:
        """Pre-write LAMMPS data files for all training/evaluation configurations.

        Call **once** at the start of an MCMC run or potential fitting.  The
        data files on disk remain valid as long as the atomic structures do not
        change; only the potential-parameter files (Tersoff, KC, DRIP, POD
        coefficients) are rewritten on each ``set_parameters`` call.

        Parameters
        ----------
        atoms_list : sequence of ase.Atoms
        batch_dir : str, optional
            Directory to write data files.  Defaults to a ``batch/`` sub-folder
            of the calculator's temporary directory.
        """
        if batch_dir is None:
            batch_dir = os.path.join(self._tmpdir, "batch")
        os.makedirs(batch_dir, exist_ok=True)
        layer_tags = getattr(self, "_layer_tags", None)
        atom_style = self._atom_style()
        files: List[str] = []
        for i, atoms in enumerate(atoms_list):
            p = os.path.join(batch_dir, f"config_{i:05d}.lmp")
            _write_lammps_structure(p, atoms, atom_style=atom_style,
                                    layer_tags=layer_tags)
            files.append(p)
        self._batch_files = files
        self._batch_atoms = list(atoms_list)

    def evaluate_batch(
        self,
        atoms_list=None,
        params=None,
    ):
        """Evaluate energies and forces for all configurations in the batch.

        Parameters
        ----------
        atoms_list : sequence of ase.Atoms, optional
            If *None*, the list stored by the most recent :meth:`prepare_batch`
            call is used.
        params : optional
            If provided, calls ``self.set_parameters(params)`` first.

        Returns
        -------
        energies : np.ndarray, shape (N,)
            Total potential energies in eV.
        forces_list : list[np.ndarray]
            Per-configuration forces in the ASE frame, each shape (natoms, 3)
            in eV/Å.
        """
        if params is not None:
            self.set_parameters(params)

        if atoms_list is None:
            atoms_list = self._batch_atoms
        if atoms_list is None:
            raise RuntimeError(
                "prepare_batch() must be called before evaluate_batch() "
                "when atoms_list is not supplied."
            )
        if self._batch_files is None:
            self.prepare_batch(atoms_list)

        lmp = self._get_lmp()
        atom_style = self._atom_style()
        n = len(atoms_list)
        energies = np.zeros(n, dtype=np.float64)
        forces_list: List[Optional[np.ndarray]] = [None] * n
        for i, (atoms, data_file) in enumerate(zip(atoms_list, self._batch_files)):          
            try:
                lmp.command("clear")
                lmp.command("units metal")
                lmp.command(f"atom_style {atom_style}")
                lmp.command("atom_modify map array")
                lmp.command(f"boundary {self._boundary_str(atoms)}")
                lmp.command("newton on")
                lmp.command(f"read_data {data_file}")
                lmp.command("neighbor 0.3 bin")
                lmp.command("neigh_modify delay 0 every 1 check yes")
                for cmd in self._pair_style_commands(self._tmpdir):
                    lmp.command(cmd)
                lmp.command("run 0")
                energies[i] = lmp.get_thermo("pe") + self._shift
                N = lmp.get_natoms()
                forces_lammps = _gather_forces(lmp, N)
                forces_list[i] = lammps_to_ase_forces(forces_lammps, atoms)
                
            except Exception:
                # Unphysical parameters can cause LAMMPS to fail on individual
                # configs during fitting / differential evolution.  Return NaN so
                # the loss function can gracefully return a large penalty value.
                energies[i] = np.nan
                forces_list[i] = np.full((len(atoms), 3), np.nan)
                # Reset the LAMMPS instance so subsequent configs can still run.
                self._lmp = None
                lmp = self._get_lmp()

        # Invalidate single-structure cache so next calculate() does full setup.
        self._last_cell = None
        self._last_natoms = None
        
        return energies, forces_list

    # ── Relaxation ────────────────────────────────────────────────────────────

    def relax_structure(
        self,
        atoms,
        etol: float = _FIRE_RELAX_DEFAULT_ETOL,
        ftol: float = _FIRE_RELAX_DEFAULT_FTOL,
        maxiter: int = _FIRE_RELAX_DEFAULT_MAXITER,
        maxeval: int = _FIRE_RELAX_DEFAULT_MAXEVAL,
        relax_backend: str = "lammps",
        relax_fire_kwargs: Optional[Dict[str, Any]] = None,
        # legacy keyword accepted for API compatibility with *ASECalculator classes
        timestep: float = None,
    ):
        """Relax the atomic geometry and return a new ``Atoms`` at the minimum.

        Parameters
        ----------
        relax_backend : ``'lammps'`` (default) or ``'ase'``
            ``'lammps'``: LAMMPS ``min_style fire`` minimisation.
            ``'ase'``:    ASE :class:`~ase.optimize.FIRE` (calls ``calculate``
                          iteratively; uses position injection for speed).
        etol, ftol, maxiter, maxeval
            Passed to LAMMPS ``minimize etol ftol maxiter maxeval``, or
            ``fmax=ftol, steps=maxiter`` for the ASE FIRE backend.
        relax_fire_kwargs : dict, optional
            Extra keyword arguments forwarded to :class:`~ase.optimize.FIRE`
            (only used when ``relax_backend='ase'``).
        """
        backend = _normalize_relax_backend(relax_backend)

        if backend == "ase":
            out = atoms.copy()
            try:
                e = _relax_structure_ase(
                    self, out,
                    fmax=ftol,
                    steps=maxiter,
                    fire_kwargs=relax_fire_kwargs,
                )
            except Exception as exc:
                _reraise_relax_failed(
                    f"{type(self).__name__}.relax_structure (ASE FIRE)",
                    self, out, exc,
                )
            out.calc = self
            self._last_cell = None
            self._last_natoms = None
            _annotate_relax_energies(self, out, e)
            out.info["relax_backend"] = "ase"
            return out

        # LAMMPS FIRE relaxation
        out = atoms.copy()
        try:
            self._setup_lammps(out)
            lmp = self._get_lmp()
            lmp.command("min_style fire")
            lmp.command(f"minimize {etol} {ftol} {maxiter} {maxeval}")
            # A `run 0` after minimize ensures energy + forces are available.
            lmp.command("run 0")
            energy = float(lmp.get_thermo("pe"))
            N = lmp.get_natoms()
            pos_lammps = _gather_positions(lmp, N)
            pos_ase = lammps_positions_to_ase(
                pos_lammps, out, np.asarray(out.get_cell())
            )
        except Exception as exc:
            _reraise_relax_failed(
                f"{type(self).__name__}.relax_structure (LAMMPS FIRE)",
                self, out, exc,
            )

        out.set_positions(pos_ase)
        out.calc = self
        self._last_cell = None
        self._last_natoms = None
        _annotate_relax_energies(self, out, energy)
        out.info["relax_backend"] = "lammps"
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Concrete calculator classes
# ═══════════════════════════════════════════════════════════════════════════════

class TersoffLammpsCalculator(LammpsCalculatorBase):
    """Tersoff potential via the LAMMPS Python module.

    Parameters
    ----------
    params : list[float]
        14 Tersoff parameters followed by an optional energy-shift scalar:
        [m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A, shift?]
        The shift (15th element if present) is added to every LAMMPS energy
        but is **not** written to the potential file.
    elements : list[str], optional
        Element symbols (default ``["C"]``).
    cutoff : float, optional
        Accepted for API compatibility; ignored (cutoff is determined by R+D
        in the parameter set).
    ntypes : int, optional
        Accepted for API compatibility; ignored (derived from *elements*).
    """

    N_FITTED_PARAMS = 14

    def __init__(
        self,
        params,
        elements: Optional[List[str]] = None,
        cutoff: float = None,   # accepted for compat, unused
        ntypes: int = None,     # accepted for compat, unused
    ) -> None:
        super().__init__()
        self._elements = list(elements or ["C"])
        p = list(params)
        if len(p) == self.N_FITTED_PARAMS + 1:
            self._shift = float(p[-1])
            p = p[:self.N_FITTED_PARAMS]
        self._params = p
        self._write_potential_files(self._tmpdir)

    # ── Subclass interface ─────────────────────────────────────────────────────

    def _atom_style(self) -> str:
        return "atomic"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_tersoff_file(
            os.path.join(tmp_dir, "potential.tersoff"),
            self._params, self._elements,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        f = os.path.join(tmp_dir, "potential.tersoff")
        elems = " ".join(self._elements)
        return [
            "pair_style tersoff",
            f"pair_coeff * * {f} {elems}",
        ]

    # ── Parameter management ───────────────────────────────────────────────────

    def set_parameters(self, params) -> None:
        p = list(params)
        if len(p) == self.N_FITTED_PARAMS + 1:
            self._shift = float(p[-1])
            p = p[:self.N_FITTED_PARAMS]
        self._params = p
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_parameters(self) -> List[float]:
        return list(self._params) + [self._shift]


class KolmogorovCrespiLammpsCalculator(LammpsCalculatorBase):
    """Kolmogorov-Crespi Full (KC) interlayer potential via the LAMMPS Python module.

    Requires ``atom_style full`` in LAMMPS; molecule IDs (1 = bottom layer,
    2 = top layer) are written to the data file and used by KC to distinguish
    intra- from inter-layer pairs.

    Parameters
    ----------
    params : list[float]
        8 KC parameters followed by an optional energy-shift scalar:
        [z0, C0, C2, C4, C, delta, lambda, A, shift?]
        Optionally 9 with taper scale S appended instead (if 9 elements and
        no separate shift kwarg is used — the 9th element is treated as shift,
        not taper scale; pass exactly 8 values to use the default taper scale).
        The shift (9th element if present) is added to every LAMMPS energy but
        is **not** written to the potential file.
    cutoff : float
        Interlayer cutoff in Å (default 14.0).
    elements : list[str], optional
    layer_tags : list[int], optional
        0-based layer index per atom.  If *None*, layers are inferred from the
        z midplane.
    """

    N_FITTED_PARAMS = 8

    def __init__(
        self,
        params,
        cutoff: float = 14.0,
        elements: Optional[List[str]] = None,
        layer_tags: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._elements = list(elements or ["C"])
        self._cutoff = float(cutoff)
        self._layer_tags = layer_tags
        p = list(params)
        if len(p) == self.N_FITTED_PARAMS + 1:
            self._shift = float(p[-1])
            p = p[:self.N_FITTED_PARAMS]
        self._params = p
        self._write_potential_files(self._tmpdir)

    def _atom_style(self) -> str:
        return "full"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_kc_file(
            os.path.join(tmp_dir, "potential.KC"),
            self._params, self._elements, self._cutoff,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        f = os.path.join(tmp_dir, "potential.KC")
        elems = " ".join(self._elements)
        return [
            f"pair_style kolmogorov/crespi/full {self._cutoff} 1",
            f"pair_coeff * * {f} {elems}",
        ]

    def set_parameters(self, params) -> None:
        p = list(params)
        if len(p) == self.N_FITTED_PARAMS + 1:
            self._shift = float(p[-1])
            p = p[:self.N_FITTED_PARAMS]
        self._params = p
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_parameters(self) -> List[float]:
        return list(self._params) + [self._shift]

    def set_cutoff(self, cutoff: float) -> None:
        self._cutoff = float(cutoff)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_cutoff(self) -> float:
        return self._cutoff


class DRIPLammpsCalculator(LammpsCalculatorBase):
    """DRIP interlayer potential via the LAMMPS Python module.

    Requires ``atom_style full``; molecule IDs identify layers.

    Parameters
    ----------
    params : list[float]
        8 physical DRIP parameters followed by an optional energy-shift scalar:
        [C0, C2, C4, C, delta, lambda, A, z0, shift?]
        B and eta are fixed at 0 (consistent with :class:`DRIPASECalculator`).
        Legacy length-10 vectors ``[…, z0, B, eta]`` are also accepted; only
        the first 8 values are used.
        The shift (9th element if present) is added to every LAMMPS energy but
        is **not** written to the potential file.
    cutoff : float  (default 14.0)
    layer_tags : list[int], optional
    rhocut : float  (default 3.0)
    ncut : float    (default 3.0)
    elements : list[str], optional
    """

    N_FITTED_PARAMS = 8
    _B_FIXED = 0.0
    _ETA_FIXED = 0.0

    def __init__(
        self,
        params,
        cutoff: float = 14.0,
        layer_tags: Optional[List[int]] = None,
        rhocut: float = 3.0,
        ncut: float = 3.0,
        elements: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._elements = list(elements or ["C"])
        self._cutoff = float(cutoff)
        self._rhocut = float(rhocut)
        self._ncut = float(ncut)
        self._layer_tags = layer_tags
        p = list(params)
        p, shift = self._extract_shift(p)
        self._shift = shift
        self._params = self._coerce_physical_params(p)
        self._write_potential_files(self._tmpdir)

    @staticmethod
    def _extract_shift(params) -> tuple:
        """Split trailing shift from physical params.

        Returns ``(physical_params, shift)`` where shift defaults to 0.0 if not
        present.  Accepted lengths: 8 (no shift), 9 (8+shift), 10 (legacy B/eta,
        no shift), 11 (legacy + shift).
        """
        p = list(params)
        n = len(p)
        if n in (9, 11):          # 8+shift or 10-legacy+shift
            return p[:-1], float(p[-1])
        return p, 0.0             # 8 or 10-legacy: no shift

    @staticmethod
    def _coerce_physical_params(params) -> List[float]:
        """Accept 8 fitted values or legacy length-10 with trailing B, eta."""
        p = list(params)
        if len(p) == 10:
            return p[:8]
        if len(p) == 8:
            return p
        raise ValueError(
            f"DRIPLammpsCalculator expects 8 physical parameters, got {len(p)}."
        )

    def _atom_style(self) -> str:
        return "full"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_drip_file(
            os.path.join(tmp_dir, "potential.drip"),
            self._params, self._elements,
            cutoff=self._cutoff, rhocut=self._rhocut, ncut=self._ncut,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        f = os.path.join(tmp_dir, "potential.drip")
        elems = " ".join(self._elements)
        return [
            "pair_style drip",
            f"pair_coeff * * {f} {elems}",
        ]

    def set_parameters(self, params) -> None:
        p, shift = self._extract_shift(list(params))
        self._shift = shift
        self._params = self._coerce_physical_params(p)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_parameters(self) -> List[float]:
        return list(self._params) + [self._shift]

    def set_cutoff(self, cutoff: float) -> None:
        self._cutoff = float(cutoff)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_cutoff(self) -> float:
        return self._cutoff


class TersoffKCLammpsCalculator(LammpsCalculatorBase):
    """Hybrid Tersoff + KC Full bilayer potential (``hybrid/overlay``).

    Equivalent to the C++ ``TersoffKolmogorovCrespiCalculator`` but backed by
    the LAMMPS Python module::

        pair_style hybrid/overlay tersoff kolmogorov/crespi/full {cutoff} 1

    Parameters
    ----------
    tersoff_params : list[float]   14 Tersoff parameters
    kc_params : list[float]        8 KC parameters
    kc_cutoff : float              KC cutoff in Å (default 20.0)
    elements : list[str], optional
    layer_tags : list[int], optional
    shift : float, optional
        Energy shift added to every LAMMPS energy (default 0.0).  Not written
        to any parameter file; only used in cost-function evaluation.
    """

    N_TERSOFF_PARAMS = 14
    N_KC_PARAMS = 8

    def __init__(
        self,
        tersoff_params,
        kc_params=None,
        kc_cutoff: float = 20.0,
        elements: Optional[List[str]] = None,
        shift: float = 0.0,
        # legacy alias (TersoffKolmogorovCrespiASECalculator used this name)
        kolmogorov_crespi_params=None,
    ) -> None:
        super().__init__()
        self._elements = list(elements or ["C"])
        self._tersoff_params = list(tersoff_params)
        if kc_params is None:
            kc_params = kolmogorov_crespi_params
        if kc_params is None:
            raise ValueError(
                "TersoffKCLammpsCalculator: kc_params must be provided."
            )
        self._kc_params = list(kc_params)
        self._kc_cutoff = float(kc_cutoff)
        self._shift = float(shift)
        self._write_potential_files(self._tmpdir)

    def _atom_style(self) -> str:
        return "full"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_tersoff_file(
            os.path.join(tmp_dir, "tersoff.tersoff"),
            self._tersoff_params, self._elements,
        )
        _write_kc_file(
            os.path.join(tmp_dir, "kc.KC"),
            self._kc_params, self._elements, self._kc_cutoff,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        tf = os.path.join(tmp_dir, "tersoff.tersoff")
        kf = os.path.join(tmp_dir, "kc.KC")
        elems = " ".join(self._elements)
        return [
            f"pair_style hybrid/overlay tersoff "
            f"kolmogorov/crespi/full {self._kc_cutoff} 1",
            f"pair_coeff * * tersoff {tf} {elems}",
            f"pair_coeff * * kolmogorov/crespi/full {kf} {elems}",
        ]

    def set_parameters(self, tersoff_params, kc_params=None,
                       kolmogorov_crespi_params=None, shift: float = None) -> None:
        if kc_params is None:
            kc_params = kolmogorov_crespi_params
        self._tersoff_params = list(tersoff_params)
        self._kc_params = list(kc_params)
        if shift is not None:
            self._shift = float(shift)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_parameters(self):
        return list(self._tersoff_params), list(self._kc_params), self._shift


class TersoffDRIPLammpsCalculator(LammpsCalculatorBase):
    """Hybrid Tersoff + DRIP bilayer potential (``hybrid/overlay``).

    Equivalent to the C++ ``TersoffDRIPCalculator`` but backed by the LAMMPS
    Python module::

        pair_style hybrid/overlay tersoff zero 0.1 drip

    Parameters
    ----------
    tersoff_params : list[float]   14 Tersoff parameters
    drip_params : list[float]      8 physical DRIP parameters
    cutoff : float                 DRIP cutoff in Å (default 14.0)
    layer_tags : list[int], optional
    rhocut : float (default 3.0)
    ncut : float (default 3.0)
    elements : list[str], optional
    shift : float, optional
        Energy shift added to every LAMMPS energy (default 0.0).  Not written
        to any parameter file; only used in cost-function evaluation.
    """

    N_FITTED_PARAMS = DRIPLammpsCalculator.N_FITTED_PARAMS  # 8 — DRIP split index
    N_TERSOFF_PARAMS = 14

    def __init__(
        self,
        tersoff_params,
        drip_params,
        cutoff: float = 14.0,
        layer_tags: Optional[List[int]] = None,
        rhocut: float = 3.0,
        ncut: float = 3.0,
        elements: Optional[List[str]] = None,
        shift: float = 0.0,
        # legacy alias
        drip_cutoff: float = None,
    ) -> None:
        super().__init__()
        self._elements = list(elements or ["C"])
        self._tersoff_params = list(tersoff_params)
        self._drip_params = DRIPLammpsCalculator._coerce_physical_params(drip_params)
        self._cutoff = float(drip_cutoff if drip_cutoff is not None else cutoff)
        self._rhocut = float(rhocut)
        self._ncut = float(ncut)
        self._layer_tags = layer_tags
        self._shift = float(shift)
        self._write_potential_files(self._tmpdir)

    def _atom_style(self) -> str:
        return "full"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_tersoff_file(
            os.path.join(tmp_dir, "tersoff.tersoff"),
            self._tersoff_params, self._elements,
        )
        _write_drip_file(
            os.path.join(tmp_dir, "drip.drip"),
            self._drip_params, self._elements,
            cutoff=self._cutoff, rhocut=self._rhocut, ncut=self._ncut,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        tf = os.path.join(tmp_dir, "tersoff.tersoff")
        df = os.path.join(tmp_dir, "drip.drip")
        elems = " ".join(self._elements)
        return [
            "pair_style hybrid/overlay tersoff zero 0.1 drip",
            f"pair_coeff * * tersoff {tf} {elems}",
            "pair_coeff * * zero",
            f"pair_coeff * * drip {df} {elems}",
        ]

    def set_parameters(self, tersoff_params, drip_params, shift: float = None) -> None:
        self._tersoff_params = list(tersoff_params)
        self._drip_params = DRIPLammpsCalculator._coerce_physical_params(drip_params)
        if shift is not None:
            self._shift = float(shift)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_parameters(self):
        return list(self._tersoff_params), list(self._drip_params), self._shift


class PODLammpsCalculator(LammpsCalculatorBase):
    """POD machine-learning potential via the LAMMPS Python module.

    Parameters
    ----------
    hyperparams : dict
        POD descriptor hyperparameters (same keys as a ``.pod`` descriptor file).
    params : array-like
        Linear POD coefficients, length = ``ncoeff_from_params(hyperparams)``.
    elements : list[str], optional
    cutoff : float  (default 5.0)
    """

    def __init__(
        self,
        hyperparams: Dict,
        params,
        elements: Optional[List[str]] = None,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self._elements      = list(elements or ["C"])
        self._hyperparams   = dict(hyperparams)
        self._params        = np.asarray(params, dtype=np.float64)
        self._cutoff        = float(cutoff)
        self._write_potential_files(self._tmpdir)

    def _atom_style(self) -> str:
        return "atomic"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_pod_param_file(
            os.path.join(tmp_dir, "pod_param.pod"),
            self._hyperparams, self._elements, self._cutoff,
        )
        _write_pod_coeff_file(
            os.path.join(tmp_dir, "pod_coeff.pod"),
            self._params,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        pf    = os.path.join(tmp_dir, "pod_param.pod")
        cf    = os.path.join(tmp_dir, "pod_coeff.pod")
        elems = " ".join(self._elements)
        return [
            "pair_style pod",
            f"pair_coeff * * {pf} {cf} {elems}",
        ]

    def set_parameters(self, params) -> None:
        self._params = np.asarray(params, dtype=np.float64)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_parameters(self) -> np.ndarray:
        return self._params.copy()

    def set_cutoff(self, cutoff: float) -> None:
        self._cutoff = float(cutoff)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def get_cutoff(self) -> float:
        return self._cutoff

    def hyperparams_to_str(self) -> str:
        """Return the POD descriptor file content as a string."""
        hp = self._hyperparams
        lines = [
            "species " + " ".join(self._elements),
            "pbc 1 1 1",
            "rin 1.0",
            f"rcut {self._cutoff}",
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

    @property
    def ncoeff(self) -> int:
        return len(self._params)


# ═══════════════════════════════════════════════════════════════════════════════
#  TETB-POD helpers — tight-binding math
# ═══════════════════════════════════════════════════════════════════════════════

def _chebyshev_and_grad(x: np.ndarray, M: int):
    """Chebyshev polynomials T_1…T_M and their derivatives dT_m/dx.

    Uses the three-term recurrence:
        T_1 = x,          dT_1 = 1
        T_2 = 2x²-1,      dT_2 = 4x
        T_m = 2x T_{m-1} - T_{m-2}
       dT_m = 2 T_{m-1} + 2x dT_{m-1} - dT_{m-2}

    Parameters
    ----------
    x : ndarray, shape (N,)
    M : int

    Returns
    -------
    T  : ndarray, shape (N, M)
    dT : ndarray, shape (N, M)  — dT_m/dx
    """
    N = len(x)
    T  = np.empty((N, M), dtype=np.float64)
    dT = np.empty((N, M), dtype=np.float64)
    T[:, 0]  = x
    dT[:, 0] = 1.0
    if M > 1:
        T[:, 1]  = 2.0 * x * x - 1.0
        dT[:, 1] = 4.0 * x
    for m in range(2, M):
        T[:, m]  = 2.0 * x * T[:, m - 1] - T[:, m - 2]
        dT[:, m] = 2.0 * T[:, m - 1] + 2.0 * x * dT[:, m - 1] - dT[:, m - 2]
    return T, dT


def _acsf_hopping_gradient_from_pairs(
    M: int,
    W: int,
    r_cut: float,
    tb_params: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_v: np.ndarray,
    N: int,
    r_inner_cut: float = 1.0,
):
    """Analytic gradient of ACSF hopping amplitudes.

    **J-leg gradient** (w.r.t. the bond's own displacement r_p = R_j - R_i):

        grad_t[p, α] = ∂t_p / ∂r_{p,α}

    **K-leg gradient** (w.r.t. each other-neighbour bond r_q = R_k - R_i
    that appears in the 3-body angular sum of bond p):

        kleg_grad[n, α] = ∂t_{kleg_t_p[n]} / ∂r_{kleg_t_q[n], α}

    The k-leg gradient has the same Part-2 scalar as the j-leg but uses the
    k-leg angle derivative d cos(θ)/d r_q = (r̂_p − cos·r̂_q) / r_q.

    Parameters
    ----------
    M, W, r_cut, tb_params, pair_i, pair_j, pair_v, N
        As returned by ``get_acsf_hopping_descriptors``.
    r_inner_cut : float  inner cutoff (default 1.0 Å)

    Returns
    -------
    grad_t     : ndarray (n_pairs, 3)    — j-leg gradient ∂t_p/∂r_p
    kleg_t_p   : ndarray (n_triplets,)  — j-leg bond indices
    kleg_t_q   : ndarray (n_triplets,)  — k-leg bond indices
    kleg_grad  : ndarray (n_triplets, 3) — ∂t_{kleg_t_p}/∂r_{kleg_t_q}
    """
    from blg_model_builder.tb_descriptors import _build_triplet_indices

    pair_r = np.linalg.norm(pair_v, axis=1)           # (n_pairs,)
    n_pairs = len(pair_r)

    # ── Chebyshev basis and its r-derivative ────────────────────────────────
    scale = r_cut - r_inner_cut
    x     = (2.0 * pair_r - (r_inner_cut + r_cut)) / scale  # (n_pairs,)
    T, dTdx = _chebyshev_and_grad(x, M)                     # (n_pairs, M) each

    dx_dr  = 2.0 / scale                                    # scalar
    fc     = 0.5 * (np.cos(np.pi * pair_r / r_cut) + 1.0)  # (n_pairs,)
    dfc_dr = -0.5 * np.pi / r_cut * np.sin(np.pi * pair_r / r_cut)

    # dD_2b[p,m]/dr = dT_m/dx·dx/dr·fc + T_m·dfc/dr  (scalar per bond, per basis fn)
    dD2b_dr = dTdx * (dx_dr * fc[:, np.newaxis]) + T * dfc_dr[:, np.newaxis]
    # cheb[p,m] = T_m(x_ij)·f_c(r_ij)
    cheb = T * fc[:, np.newaxis]                             # (n_pairs, M)

    # ── Two-body contribution to grad_t ─────────────────────────────────────
    w_2b      = tb_params[:M]                                # (M,)
    dt_2b_dr  = dD2b_dr @ w_2b                              # (n_pairs,) scalar
    r_hat     = pair_v / pair_r[:, np.newaxis]               # (n_pairs, 3)
    grad_t_2b = dt_2b_dr[:, np.newaxis] * r_hat             # (n_pairs, 3)

    # ── Three-body contribution ──────────────────────────────────────────────
    grad_t_3b = np.zeros((n_pairs, 3), dtype=np.float64)
    w_3b = tb_params[M:].reshape(M, W)                      # (M, W)

    t_p, t_q = _build_triplet_indices(pair_i, N)            # (n_triplets,) each

    # Default empty k-leg arrays (returned even when no triplets exist)
    kleg_grad = np.zeros((len(t_p), 3), dtype=np.float64)

    if len(t_p) > 0:
        v_p = pair_v[t_p]                                   # (n_triplets, 3) r_ij
        v_q = pair_v[t_q]                                   # (n_triplets, 3) r_ik
        r_p = pair_r[t_p]                                   # (n_triplets,)
        r_q = pair_r[t_q]

        cos_theta = np.einsum("nd,nd->n", v_p, v_q) / (r_p * r_q)
        np.clip(cos_theta, -1.0, 1.0, out=cos_theta)

        # cos^{wi+1}(θ) for wi = 0..W-1
        cos_pw = np.empty((len(t_p), W), dtype=np.float64)
        cos_pw[:, 0] = cos_theta
        for wi in range(1, W):
            cos_pw[:, wi] = cos_pw[:, wi - 1] * cos_theta

        # ── Angle derivatives ─────────────────────────────────────────────
        # J-leg: d cos(θ) / d r_ij = (r̂_ik - cos·r̂_ij) / |r_ij|
        d_cos_drij = (
            v_q / r_q[:, np.newaxis]
            - cos_theta[:, np.newaxis] * v_p / r_p[:, np.newaxis]
        ) / r_p[:, np.newaxis]

        # K-leg: d cos(θ) / d r_ik = (r̂_ij - cos·r̂_ik) / |r_ik|
        d_cos_driq = (
            v_p / r_p[:, np.newaxis]
            - cos_theta[:, np.newaxis] * v_q / r_q[:, np.newaxis]
        ) / r_q[:, np.newaxis]

        cheb_q      = cheb[t_q]                             # (n_triplets, M)
        dcheb_p_dr  = dD2b_dr[t_p]                         # (n_triplets, M) j-leg radial deriv
        dcheb_q_dr  = dD2b_dr[t_q]                         # (n_triplets, M) k-leg radial deriv
        cheb_p      = cheb[t_p]                             # (n_triplets, M)
        r_hat_p     = r_hat[t_p]                            # (n_triplets, 3)
        r_hat_q     = r_hat[t_q]                            # (n_triplets, 3)

        # Shared angular weight: cos_weighted[n,m] = Σ_w w_3b[m,w]·cos^{w+1}(θ_n)
        cos_weighted = cos_pw @ w_3b.T                      # (n_triplets, M)

        # ── J-leg Part 1: radial ∂T_m(x_ij)/∂r_ij × angular sum ─────────
        angular_factor_jleg = cheb_q * cos_weighted         # (n_triplets, M)
        part1_jleg_scalar   = np.einsum("nm,nm->n", dcheb_p_dr, angular_factor_jleg)
        np.add.at(grad_t_3b, t_p, part1_jleg_scalar[:, np.newaxis] * r_hat_p)

        # ── Shared angular-derivative scalar for Part 2 ───────────────────
        # d(cos^{wi+1})/d(cos) = (wi+1)·cos^{wi}(θ)
        dcos_pw_dcos = np.empty((len(t_p), W), dtype=np.float64)
        dcos_pw_dcos[:, 0] = 1.0
        for wi in range(1, W):
            dcos_pw_dcos[:, wi] = (wi + 1) * cos_pw[:, wi - 1]

        dcos_weighted     = dcos_pw_dcos @ w_3b.T           # (n_triplets, M)
        part2_weight      = cheb_p * cheb_q * dcos_weighted # (n_triplets, M)
        part2_scalar_dcos = np.sum(part2_weight, axis=1)    # (n_triplets,)

        # ── J-leg Part 2: angle change through r_ij ───────────────────────
        np.add.at(grad_t_3b, t_p,
                  part2_scalar_dcos[:, np.newaxis] * d_cos_drij)

        # ── K-leg gradient: ∂t_p / ∂r_q ──────────────────────────────────
        # Part 1: radial ∂T_m(x_ik)/∂r_ik × (j-leg Chebyshev × angular sum)
        angular_factor_kleg  = cheb_p * cos_weighted        # (n_triplets, M)
        part1_kleg_scalar    = np.einsum("nm,nm->n", dcheb_q_dr, angular_factor_kleg)
        kleg_grad = (
            part1_kleg_scalar[:, np.newaxis] * r_hat_q      # Part 1
            + part2_scalar_dcos[:, np.newaxis] * d_cos_driq  # Part 2 (same scalar, k-leg dir)
        )

    return grad_t_2b + grad_t_3b, t_p, t_q, kleg_grad


def _build_hamiltonians_kpoints(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_v: np.ndarray,
    t_ij: np.ndarray,
    kpoints: np.ndarray,
    N: int,
) -> np.ndarray:
    """Build Bloch Hamiltonians H(k) for every k-point.

    Convention: NeighborList(bothways=False) gives a *half-list* — each bond
    (i,j) is stored once.  The full Hermitian Hamiltonian is built as::

        H_k[pair_i, pair_j] += t_ij · exp(ik·r_ij)
        H_k += H_k†   (no factor of ½)

    Parameters
    ----------
    pair_i, pair_j : (n_pairs,) int arrays
    pair_v : (n_pairs, 3) displacement vectors r_ij in Å
    t_ij   : (n_pairs,) real hopping amplitudes
    kpoints : (n_kp, 3) Cartesian k-vectors in Å⁻¹ (including 2π factor)
    N : int  number of atoms

    Returns
    -------
    H_all : ndarray, shape (n_kp, N, N), complex128
    """
    n_kp   = len(kpoints)
    phases = np.exp(1j * (kpoints @ pair_v.T))   # (n_kp, n_pairs)
    H_all  = np.zeros((n_kp, N, N), dtype=np.complex128)
    for ki in range(n_kp):
        np.add.at(H_all[ki], (pair_i, pair_j), t_ij * phases[ki])
    H_all += H_all.conj().transpose(0, 2, 1)     # add Hermitian conjugate
    return H_all


def _solve_density_matrix(H_k: np.ndarray, method: str = "diagonalization") -> np.ndarray:
    """Compute the density matrix for a single k-point Hamiltonian.

    At half-filling (one pz electron per atom ⟹ N/2 occupied bands) the
    spin-summed density matrix is::

        DM = 2 · V_occ @ V_occ†

    where V_occ are the N/2 lowest eigenvectors of H_k.

    Parameters
    ----------
    H_k    : ndarray (N, N), complex128 — Hermitian Hamiltonian
    method : {"diagonalization", "sparse_diagonalization"}

    Returns
    -------
    DM : ndarray (N, N), complex128
    """
    N    = H_k.shape[0]
    nocc = N // 2

    if method == "diagonalization":
        _, eigvecs = np.linalg.eigh(H_k)           # columns = eigenvectors, sorted
        DM = 2.0 * eigvecs[:, :nocc] @ eigvecs[:, :nocc].conj().T
        return DM

    elif method == "sparse_diagonalization":
        import scipy.sparse
        import scipy.sparse.linalg
        H_sp  = scipy.sparse.csr_matrix(H_k)
        k_req = min(nocc + 4, N - 2)
        vals, vecs = scipy.sparse.linalg.eigsh(H_sp, k=k_req, sigma=0.0)
        order     = np.argsort(vals)
        vecs_occ  = vecs[:, order[:nocc]]
        DM = 2.0 * vecs_occ @ vecs_occ.conj().T
        return DM

    else:
        raise ValueError(
            f"_solve_density_matrix: unknown method {method!r}. "
            "Choose 'diagonalization' or 'sparse_diagonalization'."
        )


def _compute_band_forces(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_v: np.ndarray,
    t_ij: np.ndarray,
    grad_t: np.ndarray,
    DM_all: np.ndarray,
    kpoints: np.ndarray,
    N: int,
    kleg_t_p: np.ndarray | None = None,
    kleg_t_q: np.ndarray | None = None,
    grad_kleg: np.ndarray | None = None,
) -> np.ndarray:
    """Hellman–Feynman band forces from the tight-binding density matrix.

    **J-leg contribution** (each half-list bond p = (i, j)):

        F_i_α += +(2/n_kp) Σ_k Re[(∂t_p/∂r_{p,α} + i·k_α·t_p) e^{ik·r_p} DM_k[j,i]]
        F_j_α += -(same)

    **K-leg contribution** (each triplet (p, q) where q=(i,k) is a k-leg of p):

        K_p = (2/n_kp) Σ_k Re[e^{ik·r_p} DM_k[j_p, i_p]]   (hopping-only kernel)
        F_k_α  += -K_p · (∂t_p/∂r_{q,α})
        F_i_α  += +K_p · (∂t_p/∂r_{q,α})   (Newton's 3rd)

    Parameters
    ----------
    pair_i, pair_j : (n_pairs,) int arrays
    pair_v  : (n_pairs, 3) displacement vectors r_ij in Å
    t_ij    : (n_pairs,) real hopping amplitudes
    grad_t  : (n_pairs, 3) — ∂t_{ij}/∂r_{ij}  (j-leg gradient)
    DM_all  : (n_kp, N, N) complex128 density matrices
    kpoints : (n_kp, 3) Cartesian k-vectors in Å⁻¹ (with 2π)
    N : int
    kleg_t_p  : (n_triplets,) j-leg bond indices (from _acsf_hopping_gradient_from_pairs)
    kleg_t_q  : (n_triplets,) k-leg bond indices
    grad_kleg : (n_triplets, 3) — ∂t_{kleg_t_p}/∂r_{kleg_t_q}

    Returns
    -------
    F_band : ndarray (N, 3), eV/Å
    """
    n_kp   = len(kpoints)
    phases = np.exp(1j * (kpoints @ pair_v.T))   # (n_kp, n_pairs)

    DM_nm = DM_all[:, pair_j, pair_i]             # (n_kp, n_pairs) complex

    F_band = np.zeros((N, 3), dtype=np.float64)
    for alpha in range(3):
        k_alpha = kpoints[:, alpha]               # (n_kp,)
        kernel = (
            grad_t[np.newaxis, :, alpha]
            + 1j * k_alpha[:, np.newaxis] * t_ij[np.newaxis, :]
        ) * phases * DM_nm
        bond_force = 2.0 * np.real(np.sum(kernel, axis=0)) / n_kp  # (n_pairs,)

        # F[pair_i] = +bond_force, F[pair_j] = -bond_force
        np.add.at(F_band[:, alpha], pair_i, +bond_force)
        np.add.at(F_band[:, alpha], pair_j, -bond_force)

    # ── K-leg 3-body contributions ────────────────────────────────────────
    if kleg_t_p is not None and len(kleg_t_p) > 0:
        # Hopping-only kernel: K_p = (2/n_kp) Σ_k Re[e^{ik·r_p} DM_k[j_p, i_p]]
        # The phase depends on r_p (j-leg bond), not r_q — no i·k·t term here.
        phases_p  = np.exp(1j * (kpoints @ pair_v[kleg_t_p].T))  # (n_kp, n_triplets)
        DM_nm_p   = DM_all[:, pair_j[kleg_t_p], pair_i[kleg_t_p]]  # (n_kp, n_triplets)
        kleg_kern = 2.0 * np.real(np.sum(phases_p * DM_nm_p, axis=0)) / n_kp  # (n_triplets,)

        for alpha in range(3):
            kleg_f = kleg_kern * grad_kleg[:, alpha]
            # k-neighbour (pair_j[t_q]) receives -K_p·∂t_p/∂r_q
            np.add.at(F_band[:, alpha], pair_j[kleg_t_q], -kleg_f)
            # centre atom (pair_i[t_p] = pair_i[t_q]) receives +K_p·∂t_p/∂r_q
            np.add.at(F_band[:, alpha], pair_i[kleg_t_p], +kleg_f)

    return F_band


# ═══════════════════════════════════════════════════════════════════════════════
#  TETB-POD calculator
# ═══════════════════════════════════════════════════════════════════════════════


def tetb_auto_select_kmesh(atoms) -> Tuple[int, int, int]:
    """Return a Monkhorst–Pack mesh size ``(Kx, Ky, 1)`` from the ASE cell.

    Same empirical rule as ``BLG_model_builder.TETB_model_builder.TETB_model.
    auto_select_kpoints`` (converged k-density for a 1×1 bilayer graphene cell,
    scaled to larger in-plane supercells via 2.46 Å reference lengths).
    """
    cell = atoms.get_cell()
    cell_length_1 = 2.46
    cell_length_2 = 2.46
    ncellsx = float(np.ceil(np.round(np.linalg.norm(cell[0, :]) / cell_length_1)))
    ncellsy = float(np.ceil(np.round(np.linalg.norm(cell[1, :]) / cell_length_2)))
    if ncellsx <= 0.0:
        ncellsx = 1.0
    if ncellsy <= 0.0:
        ncellsy = 1.0
    Kx = int(np.round(25 * 1.0 / ncellsx))
    Ky = int(np.round(25 * 1.0 / ncellsy))
    # ``int(round(...))`` can be 0 for very large supercells; mesh sizes must be ≥1.
    Kx = max(1, Kx)
    Ky = max(1, Ky)
    return (Kx, Ky, 1)


class TETB_PODLammpsCalculator(LammpsCalculatorBase):
    """Total-energy tight-binding + POD residual + Ewald Coulomb calculator.

    Energy decomposition::

        E_total = E_band + E_residual(POD) + E_ewald(Mulliken charges)

    where:

    * **E_band** — tight-binding band energy using a linear ACSF hopping model
      solved over a k-point mesh via full diagonalization (or sparse/PEXSI).
    * **E_residual** — POD machine-learning potential (short-range corrective).
    * **E_ewald** — Coulomb energy of Mulliken charges via LAMMPS ``coul/long``
      + ``kspace_style pppm``.

    The Mulliken charge on atom i is::

        q_i = Z_i - (1/n_kp) Σ_k DM_k[i,i]

    where Z_i = ``valence_charge`` (default 1.0 for pz carbon) and DM_k is the
    spin-summed density matrix (trace = N_electrons).

    The POD + Coulomb part is evaluated by a single LAMMPS call using::

        pair_style hybrid/overlay pod coul/long {ewald_cutoff}
        pair_coeff * * pod ...
        pair_coeff * * coul/long
        kspace_style pppm {pppm_accuracy}

    Parameters
    ----------
    tb_params : array-like, shape (M + M*W,)
        Linear ACSF hopping weights.
    pod_params : array-like
        POD linear coefficients.
    tb_hyperparams : dict
        Keys: ``M``, ``W``, ``r_cut`` (also accepts ``acsf_M``, ``acsf_W``).
    pod_hyperparams : dict
        POD descriptor hyperparameter dict (keys as in ``_write_pod_param_file``).
    pod_cutoff : float, optional
        POD real-space cutoff in Å (default 5.0).
    elements : list[str], optional
        Element symbols (default ``["C"]``).
    kpoints : array-like, optional
        **Default:** ``None`` — automatic Monkhorst–Pack mesh from the current
        ASE cell (legacy ``TETB_model`` ``auto_select_kpoints`` rule).

        **Overrides:**

        * ``[nx, ny, nz]`` — uniform MP mesh (positive integers; float values
          that are whole numbers, e.g. from JSON, are accepted). Converted to
          Cartesian **k**-vectors using the cell at each ``calculate`` call.
        * ``(n_kp, 3)`` array — explicit Cartesian **k**-vectors (Å⁻¹, including
          the usual ``2π`` reciprocal convention). Use at least two dimensions
          so a single **k**-vector is not mistaken for an MP triple (e.g.
          ``np.reshape([kx, ky, kz], (1, 3))``).
        * Anything else that is not a length-3 MP triple is reshaped to
          ``(n_kp, 3)`` and treated as Cartesian.
        * ``[1, 1, 1]`` — minimal mesh (single **k**-point at the zone centre).
    tb_solver_method : {"diagonalization", "sparse_diagonalization"}
        Tight-binding solver.  "diagonalization" uses ``numpy.linalg.eigh``
        (correct for complex H at general k-points).
    ewald_cutoff : float, optional  (default 12.0 Å)
    pppm_accuracy : float, optional  (default 1e-4)
    valence_charge : float, optional
        Neutral-atom reference charge Z_i (default 1.0 for pz carbon).
    shift : float, optional
        Constant energy shift added to every evaluation (default 0.0).
    """

    def __init__(
        self,
        tb_params,
        pod_params,
        tb_hyperparams: Dict,
        pod_hyperparams: Dict,
        pod_cutoff: float = 5.0,
        elements: Optional[List[str]] = None,
        kpoints=None,  # default: auto MP mesh from cell; pass mesh or (n_k,3) k to override
        tb_solver_method: str = "diagonalization",
        ewald_cutoff: float = 12.0,
        pppm_accuracy: float = 1e-4,
        valence_charge: float = 1.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self._elements = list(elements or ["C"])

        # TB hyperparameters — accept both {M,W,r_cut} and {acsf_M,acsf_W,...}
        hp = tb_hyperparams
        self._tb_M    = int(hp.get("M",     hp.get("acsf_M",   10)))
        self._tb_W    = int(hp.get("W",     hp.get("acsf_W",    3)))
        self._tb_rcut = float(hp.get("r_cut", hp.get("rcut", hp.get("acsf_r_cut", 6.0))))
        self._tb_params = np.asarray(tb_params, dtype=np.float64)

        # POD
        self._pod_hyperparams = dict(pod_hyperparams)
        self._pod_params      = np.asarray(pod_params, dtype=np.float64)
        self._pod_cutoff      = float(pod_cutoff)

        # k-points specification (resolved per call when mesh-size list)
        self._kpoints_spec = kpoints

        # Solver & Ewald
        self._tb_solver_method = tb_solver_method
        self._ewald_cutoff     = float(ewald_cutoff)
        self._pppm_accuracy    = float(pppm_accuracy)
        self._valence_charge   = float(valence_charge)
        self._shift            = float(shift)

        # Log ``(Kx, Ky, 1)`` when ``kpoints=None`` auto-mesh changes (see ``_resolve_kpoints``).
        self._last_logged_kmesh: Optional[Tuple[int, int, int]] = None

        self._write_potential_files(self._tmpdir)

    # ── Subclass interface ─────────────────────────────────────────────────────

    def _atom_style(self) -> str:
        return "charge"

    def _write_potential_files(self, tmp_dir: str) -> None:
        _write_pod_param_file(
            os.path.join(tmp_dir, "pod_param.pod"),
            self._pod_hyperparams, self._elements, self._pod_cutoff,
        )
        _write_pod_coeff_file(
            os.path.join(tmp_dir, "pod_coeff.pod"),
            self._pod_params,
        )

    def _pair_style_commands(self, tmp_dir: str) -> List[str]:
        pf    = os.path.join(tmp_dir, "pod_param.pod")
        cf    = os.path.join(tmp_dir, "pod_coeff.pod")
        elems = " ".join(self._elements)
        return [
            f"pair_style hybrid/overlay pod coul/long {self._ewald_cutoff}",
            f"pair_coeff * * pod {pf} {cf} {elems}",
            "pair_coeff * * coul/long",
            f"kspace_style pppm {self._pppm_accuracy}",
        ]

    # ── k-point resolution ─────────────────────────────────────────────────────

    def _resolve_kpoints(self, atoms) -> np.ndarray:
        """Return Cartesian k-vectors (Å⁻¹, with 2π) for this atoms object."""
        from blg_model_builder.tb_models import k_uniform_mesh, get_recip_cell

        spec = self._kpoints_spec
        if spec is None:
            kmesh = tetb_auto_select_kmesh(atoms)
            if kmesh != self._last_logged_kmesh:
                print("auto selected kmesh = ", kmesh)
                self._last_logged_kmesh = kmesh
            k_reduced = k_uniform_mesh(list(kmesh))
            cell = np.array(atoms.get_cell())
            recip = get_recip_cell(cell.T)
            return k_reduced @ recip

        spec_arr = np.asarray(spec)
        # Monkhorst–Pack mesh [nx, ny, nz]: length-3, finite, ≥1, integer-valued
        # (accept e.g. float64 from NumPy / JSON without mis-reading as one Cartesian k).
        if spec_arr.ndim == 1 and spec_arr.shape[0] == 3:
            v = spec_arr.astype(np.float64, copy=False)
            if (
                np.all(np.isfinite(v))
                and np.all(v >= 1.0)
                and np.allclose(v, np.rint(v), rtol=0.0, atol=1e-8)
            ):
                mesh = np.rint(v).astype(np.int64)
                k_reduced = k_uniform_mesh(mesh.tolist())
                cell = np.array(atoms.get_cell())
                recip = get_recip_cell(cell.T)
                return k_reduced @ recip

        # Pre-computed Cartesian k-vectors (n_kp, 3)
        return np.asarray(spec, dtype=np.float64).reshape(-1, 3)

    # ── Tight-binding core ─────────────────────────────────────────────────────

    def _compute_tb(self, atoms, kpoints: np.ndarray):
        """Compute E_band, F_band, and Mulliken charges q_i.

        Parameters
        ----------
        atoms : ase.Atoms
        kpoints : (n_kp, 3) Cartesian k-vectors

        Returns
        -------
        E_band : float (eV)
        F_band : ndarray (N, 3) (eV/Å)
        q_i    : ndarray (N,) Mulliken charges
        """
        from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors

        M, W, rcut = self._tb_M, self._tb_W, self._tb_rcut
        N          = len(atoms)
        n_kp       = len(kpoints)

        # Descriptors and pair topology from a single neighbour-list pass
        descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
            atoms, M=M, W=W, r_cut=rcut,
        )
        t_ij = descriptors @ self._tb_params   # (n_pairs,) real hoppings

        # H(k) for all k-points: (n_kp, N, N) complex128
        H_all = _build_hamiltonians_kpoints(pair_i, pair_j, pair_v, t_ij, kpoints, N)

        # Density matrices: (n_kp, N, N) complex128
        DM_all = np.empty((n_kp, N, N), dtype=np.complex128)
        for ki in range(n_kp):
            DM_all[ki] = _solve_density_matrix(H_all[ki], self._tb_solver_method)

        # Band energy: E_band = (1/n_kp) Σ_k Re[Tr(H_k · DM_k)]
        E_band = float(
            np.real(np.einsum("kij,kji->", H_all, DM_all)) / n_kp
        )

        # Mulliken charges: q_i = Z_i - (1/n_kp) Σ_k DM_k[i,i]
        DM_avg_diag = np.real(
            np.mean(np.diagonal(DM_all, axis1=1, axis2=2), axis=0)
        )                                                      # (N,)
        q_i = self._valence_charge - DM_avg_diag

        # Analytic hopping gradients: j-leg (per bond) and k-leg (per triplet)
        grad_t, kleg_t_p, kleg_t_q, kleg_grad = _acsf_hopping_gradient_from_pairs(
            M, W, rcut, self._tb_params, pair_i, pair_j, pair_v, N,
        )

        # Hellman–Feynman band forces (j-leg + k-leg 3-body contributions)
        F_band = _compute_band_forces(
            pair_i, pair_j, pair_v, t_ij, grad_t, DM_all, kpoints, N,
            kleg_t_p=kleg_t_p, kleg_t_q=kleg_t_q, grad_kleg=kleg_grad,
        )

        return E_band, F_band, q_i

    # ── Public ASE-compatible API ──────────────────────────────────────────────

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=None,
    ) -> Dict:
        """Compute total TETB-POD energy and forces.

        Workflow
        --------
        1. Build ACSF descriptors and H(k); solve for density matrix.
        2. Compute E_band, F_band, Mulliken charges q_i.
        3. Set q_i on a copy of atoms; write LAMMPS data with atom_style charge.
        4. Run LAMMPS (POD + coul/long + pppm) to get E_residual + E_ewald.
        5. Sum parts: E_total = E_band + E_lammps + shift.
        """
        if atoms is None:
            raise ValueError("atoms must be provided")

        kpoints = self._resolve_kpoints(atoms)

        # ── 1–2: tight-binding ───────────────────────────────────────────────
        E_band, F_band, q_i = self._compute_tb(atoms, kpoints)

        # ── 3: inject Mulliken charges into atom copy for LAMMPS data file ───
        atoms_charged = atoms.copy()
        atoms_charged.set_initial_charges(q_i)

        # ── 4: LAMMPS run (POD + coul/long) ─────────────────────────────────
        # Always force a full setup: charges change at every step.
        self._last_cell   = None
        self._last_natoms = None
        self._setup_lammps(atoms_charged)

        # Scatter Mulliken charges explicitly (guards against LAMMPS
        # rounding the charges in the data file at read time).
        lmp    = self._get_lmp()
        n_at   = len(atoms)
        q_arr  = np.ascontiguousarray(q_i, dtype=np.float64)
        q_c    = (ctypes.c_double * n_at)(*q_arr)
        lmp.scatter_atoms("q", 1, 1, q_c)

        lmp.command("run 0")
        E_lammps       = float(lmp.get_thermo("pe"))
        forces_lammps  = _gather_forces(lmp, n_at)
        F_lammps       = lammps_to_ase_forces(forces_lammps, atoms)

        # ── 5: sum contributions ─────────────────────────────────────────────
        E_total = E_band + E_lammps + self._shift
        F_total = F_band + F_lammps

        self.results = {"energy": E_total, "forces": F_total}
        return self.results

    def get_potential_energy(self, atoms=None, force_consistent=None) -> float:
        return self.calculate(atoms)["energy"]

    def get_forces(self, atoms=None) -> np.ndarray:
        return self.calculate(atoms)["forces"]

    # ── Batch evaluation ───────────────────────────────────────────────────────

    def evaluate_batch(self, atoms_list=None, params=None):
        """Evaluate TETB-POD energies and forces for a list of structures.

        Overrides the base class to run the full TB + LAMMPS workflow per
        configuration rather than the pure-LAMMPS fast path.

        Parameters
        ----------
        atoms_list : sequence of ase.Atoms, optional
            Defaults to the list stored by ``prepare_batch``.
        params : array-like, optional
            If provided, calls ``self.set_parameters(params)`` first.

        Returns
        -------
        energies   : ndarray (N,) in eV
        forces_list : list[ndarray (natoms, 3)] in eV/Å
        """
        if params is not None:
            self.set_parameters(params)

        if atoms_list is None:
            atoms_list = self._batch_atoms
        if atoms_list is None:
            raise RuntimeError(
                "prepare_batch() must be called before evaluate_batch() "
                "when atoms_list is not supplied."
            )

        n          = len(atoms_list)
        energies   = np.zeros(n, dtype=np.float64)
        forces_list: List[Optional[np.ndarray]] = [None] * n

        for i, atoms in enumerate(atoms_list):
            try:
                res          = self.calculate(atoms)
                energies[i]  = res["energy"]
                forces_list[i] = res["forces"]
            except Exception:
                energies[i]    = np.nan
                forces_list[i] = np.full((len(atoms), 3), np.nan)
                self._lmp         = None
                self._last_cell   = None
                self._last_natoms = None

        self._last_cell   = None
        self._last_natoms = None
        return energies, forces_list

    # ── Parameter management ───────────────────────────────────────────────────

    def set_parameters(self, params) -> None:
        """Set POD coefficients (and optionally energy shift).

        Parameter vector layout: ``[pod_coeffs..., shift?]``.
        The TB parameters are held fixed; use ``set_tb_parameters`` to update them.

        Parameters
        ----------
        params : array-like, length n_pod or n_pod+1
        """
        p = list(np.asarray(params, dtype=np.float64))
        n_pod = len(self._pod_params)
        if len(p) == n_pod + 1:
            self._shift = float(p[-1])
            p = p[:n_pod]
        elif len(p) != n_pod:
            raise ValueError(
                f"TETB_PODLammpsCalculator.set_parameters: expected {n_pod} "
                f"(or {n_pod + 1} with shift) parameters, got {len(p)}."
            )
        self._pod_params = np.asarray(p, dtype=np.float64)
        self._write_potential_files(self._tmpdir)
        self._last_cell = None

    def set_tb_parameters(self, tb_params) -> None:
        """Update the ACSF linear hopping weights (TB parameters)."""
        self._tb_params = np.asarray(tb_params, dtype=np.float64)

    def get_parameters(self) -> np.ndarray:
        """Return ``[pod_coeffs..., shift]`` as a flat array."""
        return np.append(self._pod_params, self._shift)

    def get_tb_parameters(self) -> np.ndarray:
        """Return the ACSF TB weights."""
        return self._tb_params.copy()

    @property
    def n_pod_params(self) -> int:
        return len(self._pod_params)

