"""Continuum LJ 9–3 hBN substrate parameters and geometry helpers.

Models graphene on a bulk hBN continuum half-space with
``z_eq = 3.35 Å``, ``E_bind = 0.0415 eV/atom``, and hBN number density
``q = 0.077495 Å^-3``.  See ``_fit_lj_continuum.py`` and
:class:`PODLJContinuumSubstrateLammpsCalculator`.
"""
from __future__ import annotations

import numpy as np

# Fitted continuum LJ parameters (σ from force=0 at z_eq; ε from E_bind;
# q is the hBN continuum density from example_lj_continuum_substrate.in).
# Do not reuse that file's ε or σ (those used a pairwise-style σ formula).
LJ_SUBSTRATE_Z_EQ = 3.35  # Å
LJ_SUBSTRATE_E_BIND = 0.0415  # eV / C atom
LJ_SUBSTRATE_EPS = 0.0040806808  # eV
LJ_SUBSTRATE_SIGMA = 3.90272672  # Å
LJ_SUBSTRATE_Q = 0.077495  # Å^-3 (hBN bulk)
LJ_SUBSTRATE_SH = 0.0  # Å


def place_atoms_on_lj_substrate(
    atoms,
    *,
    z_eq: float = LJ_SUBSTRATE_Z_EQ,
    substrate_z: float = LJ_SUBSTRATE_SH,
):
    """Shift ``atoms`` so the bottom-layer mean *z* sits at ``substrate_z + z_eq``.

    Returns the same atoms object (modified in place) for chaining.
    """
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float).copy()
    z = pos[:, 2]
    zmid = float(np.median(z))
    bottom = z < zmid
    if not np.any(bottom):
        bottom = z <= zmid
    z_bottom = float(np.mean(z[bottom]))
    shift = float(substrate_z + z_eq) - z_bottom
    pos[:, 2] += shift
    atoms.set_positions(pos)
    return atoms


def continuum_lj_energy(
    z: float,
    *,
    eps: float = LJ_SUBSTRATE_EPS,
    sigma: float = LJ_SUBSTRATE_SIGMA,
    density: float = LJ_SUBSTRATE_Q,
    substrate_z: float = LJ_SUBSTRATE_SH,
) -> float:
    """Continuum LJ 9–3 energy (eV) for one atom at height ``z``."""
    dz = float(z) - float(substrate_z)
    r = dz / float(sigma)
    bracket = (1.0 / 15.0) * r ** (-9) - 0.5 * r ** (-3)
    return (4.0 * np.pi / 3.0) * float(eps) * (float(sigma) ** 3) * float(density) * bracket
