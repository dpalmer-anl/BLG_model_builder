#!/usr/bin/env python3
"""
TBLG on monolayer hBN: geometry helpers and stored equilibrium parameters.

Stacking registry and in-plane lattice constant were determined once with the
best-fit ``POD_energy_POD_index_15_8bb97b2162397248`` potential (C–C via POD,
BN intralayer via ExTeP, B–C / N–C interlayer via BNCH ILP with C–C zeroed).
See ``uncertainty_quantification/analyze_tblg_hbn_stacking_lattice.py``.

Stored results (θ = 9.43°)
-------------------------
* Best hBN registry (fractional graphene primitive cell):
  ``HBN_STACKING_FRAC = (0.57142857, 0.00000000)``
  (sx = 4/7 along a1; sy = 0 along a2)
* Equilibrium in-plane lattice constant: ``HBN_TBLG_LAT_CON = 2.48`` Å
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Stored equilibrium parameters (filled after analysis; keep in sync with the
# printed report from analyze_tblg_hbn_stacking_lattice.py)
# ---------------------------------------------------------------------------

# Fractional shift of the hBN layer in the graphene primitive (a1, a2) basis.
# Placeholder until the analysis script is run; the analyzer overwrites these
# constants in this file via --write-constants.
HBN_STACKING_FRAC: Tuple[float, float] = (0.57142857, 0.00000000)

# Isotropic in-plane lattice constant (Å) minimizing E for the TBLG/hBN stack.
HBN_TBLG_LAT_CON: float = 2.48000000

# Interlayer separations used when assembling the stack (Å).
HBN_GRAPHENE_SEP: float = 3.35
TBLG_LAYER_SEP: float = 3.35

DEFAULT_HBN_TWIST_DEG: float = 9.43


def assign_bn_checkerboard(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return chemical symbols ``B``/``N`` on a honeycomb by BFS 2-coloring."""
    from ase.geometry import find_mic

    n = len(positions)
    symbols = np.array(["X"] * n, dtype=object)
    if n == 0:
        return symbols

    # Build adjacency within ~1.6 Å (BN / graphene bond).
    neigh: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d, _ = find_mic(positions[j] - positions[i], cell, pbc=True)
            if float(np.linalg.norm(d)) < 1.75:
                neigh[i].append(j)
                neigh[j].append(i)

    # BFS from atom 0 → B; neighbors → N.
    queue = [0]
    symbols[0] = "B"
    seen = {0}
    while queue:
        i = queue.pop(0)
        for j in neigh[i]:
            if j in seen:
                continue
            symbols[j] = "N" if symbols[i] == "B" else "B"
            seen.add(j)
            queue.append(j)

    if np.any(symbols == "X"):
        # Disconnected fragments: assign remaining by nearest colored atom.
        for i in range(n):
            if symbols[i] != "X":
                continue
            best_j, best_d = None, np.inf
            for j in range(n):
                if symbols[j] == "X":
                    continue
                d, _ = find_mic(positions[i] - positions[j], cell, pbc=True)
                nd = float(np.linalg.norm(d))
                if nd < best_d:
                    best_d, best_j = nd, j
            symbols[i] = "N" if symbols[best_j] == "B" else "B"
    return symbols


def graphene_primitive_vectors(lat_con: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return graphene primitive lattice vectors a1, a2 (Å) for hex cell."""
    a = float(lat_con)
    a1 = np.array([a, 0.0, 0.0], dtype=float)
    a2 = np.array([0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], dtype=float)
    return a1, a2


def ensure_three_layer_mol_ids(atoms) -> None:
    """Assign ``mol-id`` ∈ {1,2,3} by sorted unique z-layer clusters."""
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    # K-means-ish: sort unique layers by z median of 3 bins via z-order.
    order = np.argsort(z)
    n = len(z)
    # Use gap clustering on sorted z.
    zs = z[order]
    gaps = np.diff(zs)
    # Pick the two largest gaps → three layers.
    if n < 3:
        atoms.set_array("mol-id", np.ones(n, dtype=np.int8))
        return
    gap_idx = np.argsort(gaps)[-2:]
    cuts = sorted(int(i) + 1 for i in gap_idx)
    mol = np.zeros(n, dtype=np.int8)
    bounds = [0] + cuts + [n]
    layer = 1
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mol[order[lo:hi]] = layer
        layer += 1
    # Ensure layer 1 = lowest z.
    z_mean = {k: float(np.mean(z[mol == k])) for k in (1, 2, 3) if np.any(mol == k)}
    ranked = sorted(z_mean, key=z_mean.get)
    remap = {old: new for new, old in enumerate(ranked, start=1)}
    mol = np.array([remap[int(m)] for m in mol], dtype=np.int8)
    atoms.set_array("mol-id", mol)


def build_hbn_monolayer_from_graphene_layer(
    graphene_atoms,
    *,
    target_z: float,
) -> "ase.Atoms":
    """
    Build an hBN monolayer copying the in-plane lattice of a graphene sheet.

    Uses BFS checkerboard coloring of the honeycomb for B/N assignment.
    """
    from ase import Atoms

    pos = np.asarray(graphene_atoms.get_positions(), dtype=float).copy()
    cell = np.asarray(graphene_atoms.get_cell(), dtype=float)
    symbols = assign_bn_checkerboard(pos, cell)
    # Set mean z to target_z.
    pos[:, 2] = float(target_z) + (pos[:, 2] - float(np.mean(pos[:, 2])))
    masses = np.array([10.811 if s == "B" else 14.007 for s in symbols], dtype=float)
    return Atoms(
        symbols=list(symbols),
        positions=pos,
        cell=cell,
        pbc=True,
        masses=masses,
    )


def build_tblg_on_hbn(
    theta_deg: float = DEFAULT_HBN_TWIST_DEG,
    *,
    lat_con: float = HBN_TBLG_LAT_CON,
    stacking_frac: Sequence[float] = HBN_STACKING_FRAC,
    tblg_sep: float = TBLG_LAYER_SEP,
    hbn_sep: float = HBN_GRAPHENE_SEP,
    h_vac: float = 20.0,
):
    """
    Build twisted bilayer graphene on a monolayer hBN substrate.

    Layer order (low → high z): hBN (mol-id 1), graphene (2), graphene (3).
    ``stacking_frac`` translates the hBN layer in the graphene primitive cell.
    """
    from ase import Atoms

    # Local import avoids circular dependency at module load.
    from run_uq_propagation_relaxation import (  # noqa: PLC0415
        _ensure_mol_id_from_z,
        build_tblg_atoms,
    )

    tblg = build_tblg_atoms(
        float(theta_deg),
        lat_con=float(lat_con),
        sep=float(tblg_sep),
        h_vac=float(h_vac),
    )
    _ensure_mol_id_from_z(tblg)
    mol = tblg.get_array("mol-id")
    pos = tblg.get_positions()
    z1 = float(np.mean(pos[mol == 1, 2]))
    z2 = float(np.mean(pos[mol == 2, 2]))

    # Graphene template for hBN: copy bottom graphene layer (mol-id 1).
    bottom = tblg[mol == 1]
    hbn_z = z1 - float(hbn_sep)
    hbn = build_hbn_monolayer_from_graphene_layer(bottom, target_z=hbn_z)

    # Apply stacking translation in graphene primitive basis.
    a1, a2 = graphene_primitive_vectors(lat_con)
    sx, sy = float(stacking_frac[0]), float(stacking_frac[1])
    shift = sx * a1 + sy * a2
    hbn_pos = hbn.get_positions()
    hbn_pos[:, 0] += shift[0]
    hbn_pos[:, 1] += shift[1]
    hbn.set_positions(hbn_pos)

    # Raise vacuum: shift all z so hBN is above a vacuum pad.
    z_min = float(np.min(hbn.get_positions()[:, 2]))
    z_max = float(np.max(pos[:, 2]))
    cell = np.asarray(tblg.get_cell(), dtype=float).copy()
    cell[2, 2] = (z_max - z_min) + float(h_vac)

    combined = tblg + hbn
    combined.set_cell(cell)
    combined.set_pbc(True)
    ensure_three_layer_mol_ids(combined)
    # Re-wrap into cell.
    combined.wrap()
    return combined


def translate_hbn_stacking(atoms, stacking_frac: Sequence[float], lat_con: float):
    """Return a copy with hBN (lowest mol-id) shifted by ``stacking_frac``."""
    out = atoms.copy()
    mol = out.get_array("mol-id")
    hbn_id = int(np.min(mol))
    a1, a2 = graphene_primitive_vectors(lat_con)
    # Remove any previous fractional shift by working from absolute positions
    # relative to a reference — caller should start from stacking (0,0) atoms.
    shift = float(stacking_frac[0]) * a1 + float(stacking_frac[1]) * a2
    pos = out.get_positions()
    mask = mol == hbn_id
    pos[mask, 0] += shift[0]
    pos[mask, 1] += shift[1]
    out.set_positions(pos)
    out.wrap()
    return out


def scale_inplane_lattice(atoms, lat_con_new: float, lat_con_old: float):
    """Isotropically scale in-plane cell and xy positions; keep z fixed."""
    out = atoms.copy()
    scale = float(lat_con_new) / float(lat_con_old)
    cell = np.asarray(out.get_cell(), dtype=float).copy()
    cell[0, :] *= scale
    cell[1, :] *= scale
    pos = out.get_positions()
    pos[:, 0] *= scale
    pos[:, 1] *= scale
    out.set_cell(cell)
    out.set_positions(pos)
    return out
