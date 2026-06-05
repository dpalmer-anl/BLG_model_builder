"""
Strained bilayer graphene data loading and geometry helpers (non-plotting).
"""
from __future__ import annotations

import os
from typing import Dict

import ase.io
import numpy as np

# Default paths relative to uncertainty_quantification/ when scripts chdir there.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAINED_XYZ = os.path.join(_PKG_ROOT, "data", "strained_bilayer_graphene_rVV10.xyz")

LAT_CON = 2.46
STACKINGS = ["AB", "AA", "SP"]
STRAIN_RANGE = 0.005
LAYER_Z_FLAT_TOL = 1e-6


def identify_stacking(atoms):
    """Identify AB / AA / SP stacking from fractional centre-of-mass shift."""
    pos = atoms.get_scaled_positions()
    n = len(pos)
    idx = np.argsort(pos[:, 2])
    bot_xy = pos[idx[: n // 2], :2]
    top_xy = pos[idx[n // 2 :], :2]
    shift = np.mean(top_xy, axis=0) - np.mean(bot_xy, axis=0)
    shift = (shift + 0.5) % 1.0 - 0.5
    sx, sy = np.abs(shift)
    if sx < 0.12 and sy < 0.12:
        return "AA"
    if max(sx, sy) > 0.40:
        return "SP"
    return "AB"


def interlayer_sep(atoms):
    pos = atoms.get_positions()
    n = len(pos)
    idx = np.argsort(pos[:, 2])
    return np.mean(pos[idx[n // 2 :], 2]) - np.mean(pos[idx[: n // 2], 2])


def layers_have_uniform_z(atoms, tol: float = LAYER_Z_FLAT_TOL) -> bool:
    pos = atoms.get_positions()
    n = len(pos)
    if n < 4 or n % 2 != 0:
        return False
    z = pos[:, 2]
    idx = np.argsort(z)
    half = n // 2
    for layer_ix in (idx[:half], idx[half:]):
        if float(np.ptp(z[layer_ix])) > float(tol):
            return False
    return True


def cell_strains(atoms):
    cell = atoms.get_cell()
    dx = (cell[0, 0] - LAT_CON) / LAT_CON
    dy = (np.linalg.norm(cell[1, :2]) - LAT_CON) / LAT_CON
    return dx, dy


def load_strained_data(
    z_flat_tol: float = LAYER_Z_FLAT_TOL,
    xyz_path: str | None = None,
) -> Dict[str, np.ndarray]:
    """Returns dict: stacking -> ndarray (N, 4) = [dx, dy, sep, E_per_atom]."""
    path = xyz_path or STRAINED_XYZ
    frames = ase.io.read(path, index=":")
    records = {s: [] for s in STACKINGS}
    n_skip_md = 0

    for atoms in frames:
        if not layers_have_uniform_z(atoms, tol=z_flat_tol):
            n_skip_md += 1
            continue
        stack = identify_stacking(atoms)
        if stack not in STACKINGS:
            continue
        dx, dy = cell_strains(atoms)
        sep = interlayer_sep(atoms)
        E = atoms.get_potential_energy() / len(atoms)
        records[stack].append((dx, dy, sep, E))

    if n_skip_md:
        print(
            f"  load_strained_data: skipped {n_skip_md} MD-like frame(s) "
            f"(layer z spread > {z_flat_tol:g} Å)",
            flush=True,
        )

    return {s: np.array(v) for s, v in records.items() if v}


def parabolic_min(grid, E_1d, k):
    """Sub-grid minimum via parabolic interpolation through grid[k-1:k+2]."""
    if k == 0 or k == len(grid) - 1:
        return grid[k]
    h = grid[1] - grid[0]
    E0, E1, E2 = E_1d[k - 1], E_1d[k], E_1d[k + 1]
    denom = E0 - 2 * E1 + E2
    if denom <= 0:
        return grid[k]
    return grid[k] - 0.5 * h * (E2 - E0) / denom
