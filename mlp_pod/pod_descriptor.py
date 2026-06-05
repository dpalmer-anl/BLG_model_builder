"""
pod_descriptor.py — POD descriptor and Jacobian extraction via LAMMPS.

Architecture
------------
This module implements the *global MLP* approach:

    E_total = NN(D_1, D_2, ..., D_nd)

where D_α = Σ_i D_{i,α} is the globally summed (over all atoms) POD
descriptor α.  This is consistent with how LAMMPS internally accumulates
descriptors (see ``mlpod.cpp:linear_descriptors``).

Force computation via the chain rule (mirrors ``mlpod.cpp:calculate_energyforce``):

    F_{n,m} = -Σ_α  (∂E / ∂D_α)  ×  (∂D_α / ∂r_{n,m})

where:
  - ∂E/∂D_α  is obtained from PyTorch autograd through the MLP
  - ∂D_α/∂r_{n,m}  is the LAMMPS analytical Jacobian (NOT finite differences)

Jacobian extraction
-------------------
``pair_style pod`` computes:  force[n,m] = -Σ_α coeff[α] × (∂D_α/∂r_{n,m})

Setting coeff = e_k (unit vector, 1 at position k, 0 elsewhere) gives:
    force_k[n,m] = -(∂D_k/∂r_{n,m})   →   J[k, nm] = -force_k.ravel()

Calling ``evaluate_batch`` once for each unit-vector k = 0..n_desc-1
builds the full Jacobian matrix analytically in n_desc batch passes,
each processing all structures simultaneously.

Design mirrors ``blg_model_builder_v2.lammps_interface``.
"""

import sys
import os
import numpy as np
from typing import List, Optional, Tuple

# Allow running from within the mlp_pod directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from blg_model_builder_v2.potentials import ncoeff_from_params


class PODDescriptorCalculator:
    """Compute global POD descriptors and analytical Jacobians via LAMMPS.

    Uses ``PODLammpsCalculator.prepare_batch`` / ``evaluate_batch`` to extract:

    1. Global descriptors  D  ∈ R^n_desc  per structure
       (energy when coeff = unit vector e_k  →  energy = D_k)

    2. Descriptor Jacobian  J  ∈ R^(n_desc × n_atoms*3)  per structure
       (negative forces when coeff = e_k  →  J[k,:] = -forces.ravel())

    Parameters
    ----------
    hyperparams : dict
        POD descriptor hyperparameters (same keys as a ``.pod`` file).
    elements : list[str]
    cutoff : float
    """

    def __init__(
        self,
        hyperparams: dict,
        elements: Optional[List[str]] = None,
        cutoff: float = 6.0,
    ) -> None:
        self._hyperparams = dict(hyperparams)
        self._elements    = list(elements or ["C"])
        self._cutoff      = float(cutoff)

        hp_with_species = dict(hyperparams)
        if "species" not in hp_with_species:
            hp_with_species["species"] = self._elements

        self._n_desc: int = ncoeff_from_params(hp_with_species)

        # Deferred import: avoids the circular import cycle between
        # blg_model_builder_v2.lammps_interface and blg_model_builder_v2.potentials
        # (same pattern as pod_hyperparameter_search.py in the parent package).
        from blg_model_builder_v2.lammps_interface import PODLammpsCalculator  # noqa: PLC0415

        # Internal PODLammpsCalculator: coefficients will be overwritten per call
        self._calc = PODLammpsCalculator(
            hyperparams=self._hyperparams,
            params=np.zeros(self._n_desc, dtype=np.float64),
            elements=self._elements,
            cutoff=self._cutoff,
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_desc(self) -> int:
        """Number of POD descriptor components (ncoeff)."""
        return self._n_desc

    # ── Public API ───────────────────────────────────────────────────────────

    def compute_descriptors_and_jacobians(
        self,
        atoms_list,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, list]:
        """Compute global descriptors and analytical Jacobians for all structures.

        Parameters
        ----------
        atoms_list : list of ase.Atoms
        verbose : bool
            Print progress bar to stdout.

        Returns
        -------
        descriptors : np.ndarray, shape (n_struct, n_desc)
            Global descriptor  D_α = Σ_i D_{i,α}  for each structure.
        jacobians : list of np.ndarray, each shape (n_desc, n_atoms_s * 3)
            Analytical Jacobian  J[s][α, nm] = ∂D_α / ∂r_{nm}  for structure s.
            Each element is independently sized because structures may have
            different numbers of atoms.  Obtained without finite differences via
            the unit-vector trick.
        """
        n_struct = len(atoms_list)
        n_desc   = self._n_desc

        descriptors  = np.zeros((n_struct, n_desc), dtype=np.float64)
        # Lazily allocated: jacobians[s] is (n_desc, n_atoms_s * 3).
        # We can't pre-allocate a uniform 3-D array because structures may have
        # different atom counts (mixed supercell sizes in the dataset).
        jacobians: list = [None] * n_struct

        if verbose:
            n_atoms_counts = sorted({len(a) for a in atoms_list})
            print(
                f"[PODDescriptorCalculator] Computing descriptors + Jacobians "
                f"for {n_struct} structures, n_desc={n_desc}."
            )
            print(
                f"  Atom counts in dataset: {n_atoms_counts}"
            )
            print(
                f"  Method: {n_desc} evaluate_batch calls with unit-coefficient vectors "
                f"(analytical LAMMPS derivatives)."
            )

        # Prepare batch data files once — this is the expensive I/O step
        self._calc.prepare_batch(atoms_list)

        for k in range(n_desc):
            if verbose and (k % max(1, n_desc // 10) == 0 or k == n_desc - 1):
                print(f"  descriptor {k+1}/{n_desc} ...", flush=True)

            # Unit-vector coefficient: c = e_k
            c = np.zeros(n_desc, dtype=np.float64)
            c[k] = 1.0
            self._calc.set_parameters(c)

            energies, forces_list = self._calc.evaluate_batch()

            # energy[s] = Σ_α c[α] * D_α[s] = D_k[s]   (global descriptor k)
            descriptors[:, k] = energies

            # force[s, n, m] = -∂D_k[s]/∂r_{n,m}
            # → J[s][k, nm] = -forces[s].ravel()
            for s, forces in enumerate(forces_list):
                if forces is None or np.any(np.isnan(forces)):
                    continue
                n_dof = forces.size   # n_atoms_s * 3
                if jacobians[s] is None:
                    jacobians[s] = np.zeros((n_desc, n_dof), dtype=np.float64)
                jacobians[s][k, :] = -forces.ravel()

        if verbose:
            print("[PODDescriptorCalculator] Done.")

        return descriptors, jacobians

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the internal LAMMPS instance."""
        if self._calc is not None:
            self._calc.close()
            self._calc = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
