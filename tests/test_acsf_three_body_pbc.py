"""Three-body ACSF descriptors must be intensive under PBC supercell copies."""
from __future__ import annotations

import numpy as np
import pytest
from ase.build import make_supercell

from blg_model_builder.geom_tools import get_bilayer_atoms
from blg_model_builder.tb_descriptors import get_acsf_sk_hopping_descriptors


def _group_mean_descriptors(atoms, M: int, W: int, r_cut: float = 6.0):
    dsc, (_pi, _pj, pv) = get_acsf_sk_hopping_descriptors(
        atoms, M=M, W=W, r_cut=r_cut,
    )
    D = np.asarray(dsc, dtype=np.float64)
    pv = np.asarray(pv, dtype=np.float64)
    r = np.linalg.norm(pv, axis=1)
    dz = np.abs(pv[:, 2])
    key = np.round(np.stack([r, dz], axis=1), 3)
    out = {}
    for k, row in zip(map(tuple, key), D):
        out.setdefault(k, []).append(row)
    return {k: np.mean(v, axis=0) for k, v in out.items()}


@pytest.mark.parametrize("W", [0, 1, 6])
def test_acsf_sk_three_body_matches_large_supercell(W):
    """4-atom AB cell vs 6×6 copy: same (r, |dz|) bonds must have the same ACSF.

    A cutoff of 6 Å sees periodic images of the partner atom in the primitive
    cell.  Excluding k-legs by atom index (instead of Cartesian image) makes
    W>0 descriptors cell-size dependent.
    """
    prim = get_bilayer_atoms(3.35, 0.0, sc=1)
    large = make_supercell(prim, np.diag([6, 6, 1]))
    gp = _group_mean_descriptors(prim, M=8, W=W)
    gl = _group_mean_descriptors(large, M=8, W=W)
    common = set(gp) & set(gl)
    assert common, "no shared bond types"
    diffs = [float(np.max(np.abs(gp[k] - gl[k]))) for k in common]
    assert max(diffs) < 1e-8, (
        f"W={W}: max |Δdesc|={max(diffs):.3e} between primitive and 6×6 supercell"
    )
