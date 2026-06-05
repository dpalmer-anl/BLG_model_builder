"""
Potential energy surface (PES) sanity check: POD_energy and TETB+POD vs DFT.

Uses the same ``RELAXATION_*`` ``M`` / ``W`` knobs and loader/fit helpers as
``test_relaxation.py`` (edit those globals there).  ``test_blg_pes_models_vs_dft``
only runs **model** entries listed in ``DATASOURCES`` (same expansion as
``test_interlayer_pes_unstrained_vs_strained_ab_aa``).  Each selected model gets
a figure under ``tests/_artifacts/`` vs filtered rVV10 DFT for **AB** and
**unstrained AA** stackings.
AA means **no in-plane registry shift**: each top C shares the same ``(x,y)``
as a bottom C (identified in the DFT frames and built with
:func:`blg_model_builder_v2.geom_tools.get_aa_bilayer_atoms`).

``test_blg_pes_models_vs_dft`` also writes a **parity** scatter (DFT vs model
energy / atom) using **every** configuration returned by the interlayer rVV10
loader (no flat-layer or stacking prefilter for this figure).  Points are split
into **strained configs** (blue) vs **md configs** (orange) by within-layer
``z`` spread (:data:`PARITY_STRAINED_MD_ZTOL`).

Logged **DFT** equilibrium spacings use a **quadratic** least-squares fit to
samples near the lowest-energy DFT point (by :func:`_dft_equilibrium_d_quadratic_fit`);
model curves still use the discrete grid minimum.

Edit ``DATASOURCES`` (same token scheme as ``tests/test_compare_qmc.py``) for
``test_interlayer_pes_unstrained_vs_strained_ab_aa``.  In-plane strain follows
``DD-TETB/generate_data/gen_training_data_structures.py``: scale
``cell[0,:] *= (1+dx)``, ``cell[1,:] *= (1+dy)``, ``set_cell(..., scale_atoms=True)``.
``STRAIN_FRAC`` is ``dx`` on the **first** Bravais row; ``STRAIN_DY`` is ``dy``
on the second (default ``0``).  DFT strained points are those whose recovered
``(dx, dy)`` (see :func:`_recover_generator_strains_dx_dy`) match within
``STRAIN_MATCH_TOL`` and satisfy ``dx >= dy`` as in the generator loops.

Run (from repo root, after ``pip install -e .``)::

    pytest tests/test_PES.py -v -s
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from blg_model_builder_v2.geom_tools import get_aa_bilayer_atoms, get_bilayer_atoms

# Hyperparameters and loaders: single source of truth in test_relaxation.py
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import test_relaxation as tr  # noqa: E402

# ---------------------------------------------------------------------------
# User: which sources appear on the strain-comparison figures (same tokens as
# ``test_compare_qmc.DATASOURCES``).  ``"rVV10"`` / ``"MBD"`` load filtered DFT;
# ``"models"`` expands to every ``test_relaxation.MODEL_KEYS`` entry; or name a
# single model key (e.g. ``"pod_energy"``, ``"tetb_pod"``).
# ---------------------------------------------------------------------------
DATASOURCES: Tuple[str, ...] = (
    "rVV10",
    "pod_energy",
    #"tetb_pod",
)

# Interlayer separation range (Å) for model grids and plot x-limits.
PLOT_D_MIN: float = 3.0
PLOT_D_MAX: float = 4.0

# In-plane engineering strains ``(dx, dy)`` on Bravais rows 0 and 1 (same as
# ``gen_training_data_structures.py`` bilayer strain block).  ``STRAIN_FRAC`` ≡ ``dx``.
STRAIN_FRAC: float = 0.02
STRAIN_DY: float = 0.0
# Half of the generator's default ``linspace(-0.02, 0.02, 5)`` step is 0.01; use
# slightly tighter matching for noisy DFT cells.
STRAIN_MATCH_TOL: float = 0.006
STRAIN_AXIS_ALIGN_COS_MIN: float = 0.985
# z spread within each mol-id layer: ≤ this → "strained" (rigid scan); larger → "md"
PARITY_STRAINED_MD_ZTOL: float = 1e-5


def _filter_dft_ab_near_246_atoms_energies(
    atoms_list: List,
    energies: np.ndarray,
    *,
    a0: float = 2.46,
    a_tol: float = 0.02,
    ab_xy_min: float = 0.22,
    ab_xy_max: float = 1.55,
) -> Tuple[List, np.ndarray, int]:
    """Same filter as :func:`_filter_dft_ab_near_246`, but return kept ``Atoms`` + energies."""
    kept_atoms: list = []
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if abs(L1 - a0) > a_tol or abs(L2 - a0) > a_tol:
            n_skip += 1
            continue
        if not _is_unstrained_hexagonal(atoms):
            n_skip += 1
            continue
        d_sep, ab_score = _layer_separation_and_ab_score(atoms)
        if not np.isfinite(d_sep) or not (ab_xy_min <= ab_score <= ab_xy_max):
            n_skip += 1
            continue
        kept_atoms.append(atoms)
        d_list.append(float(d_sep))
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    atoms_ord = [kept_atoms[i] for i in order]
    e_ord = np.asarray([e_list[i] for i in order], dtype=float)
    return atoms_ord, e_ord, n_skip


def _parity_strained_vs_md_label(atoms, *, ztol: float = PARITY_STRAINED_MD_ZTOL) -> str:
    """``strained`` if every ``mol-id`` layer is coplanar within ``ztol``; else ``md``."""
    return "strained" if _has_flat_layers(atoms, ztol=ztol) else "md"


def _expand_datasources() -> list[str]:
    """Return canonical source ids in plot order (no duplicates)."""
    out: list[str] = []
    seen: set[str] = set()
    allowed_special = {"qmc", "rvv10", "mbd", "models"}
    model_set = set(tr.MODEL_KEYS)
    for raw in DATASOURCES:
        key = str(raw).strip()
        if not key:
            continue
        low = key.lower()
        if low == "models":
            for m in tr.MODEL_KEYS:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
            continue
        if low == "qmc":
            canon = "qmc"
        elif low == "rvv10":
            canon = "rVV10"
        elif low == "mbd":
            canon = "MBD"
        elif key in model_set:
            canon = key
        else:
            raise ValueError(
                f"Unknown DATASOURCES entry {raw!r}. "
                f"Expected one of {sorted(allowed_special)} (any case), "
                f"'models', or a key from MODEL_KEYS={tr.MODEL_KEYS!r}.",
            )
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


# ---------------------------------------------------------------------------
# Geometry filters (DFT training set)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    import os

    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def _hex_cell_metrics(atoms) -> Tuple[float, float, float]:
    """Return (|a1|, |a2|, cos(angle between a1,a2)) in the xy plane."""
    a1 = np.asarray(atoms.cell[0, :2], dtype=float)
    a2 = np.asarray(atoms.cell[1, :2], dtype=float)
    L1 = float(np.linalg.norm(a1))
    L2 = float(np.linalg.norm(a2))
    if L1 <= 0 or L2 <= 0:
        return L1, L2, np.nan
    cos_g = float(np.dot(a1, a2) / (L1 * L2))
    return L1, L2, cos_g


def _layer_separation_and_ab_score(atoms) -> Tuple[float, float]:
    """Interlayer distance (Å) and a Bernal-vs-AA score.

    ``score = min_top min_bottom d_xy`` (Å).  Perfect AA has a top atom directly
    above a bottom atom → ~0.  Primitive AB often has ~0.6–1.0 Å here (not 1.42).
    """
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    layers = np.searchsorted(np.unique(mol), mol).astype(int)
    if np.unique(layers).size != 2:
        return float("nan"), float("nan")
    z0 = pos[layers == 0, 2]
    z1 = pos[layers == 1, 2]
    d = float(np.mean(z1) - np.mean(z0))
    p0 = pos[layers == 0][:, :2]
    p1 = pos[layers == 1][:, :2]
    mins = []
    for r in p1:
        dxy = np.linalg.norm(p0 - r, axis=1)
        mins.append(float(np.min(dxy)))
    return d, float(np.min(mins)) if mins else float("nan")


def _is_unstrained_hexagonal(atoms, *, cos60_tol: float = 0.035, length_tol: float = 0.04) -> bool:
    L1, L2, cos_g = _hex_cell_metrics(atoms)
    if not np.isfinite(cos_g):
        return False
    return bool(abs(L1 - L2) < length_tol and abs(cos_g - 0.5) < cos60_tol)


def _has_flat_layers(atoms, *, ztol: float = 1e-6) -> bool:
    """True only if every ``mol-id`` layer has a single shared ``z`` (rigid scan).

    MD / relaxation frames show buckling (finite z-spread within a layer); those
    are dropped when ``ztol`` is tiny (same convention as ``test_compare_qmc``).
    """
    try:
        mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    except Exception:
        return False
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    for layer_id in np.unique(mol):
        z = pos[mol == layer_id, 2]
        if z.size > 1 and (float(z.max()) - float(z.min())) > ztol:
            return False
    return True


def _prefilter_flat_layers(atoms_list: List, energies: np.ndarray) -> Tuple[List, np.ndarray]:
    """Keep only configurations with co-planar atoms within each layer."""
    e = np.asarray(energies, dtype=float).ravel()
    mask = np.asarray([_has_flat_layers(a) for a in atoms_list], dtype=bool)
    kept_a = [a for a, ok in zip(atoms_list, mask) if ok]
    return kept_a, e[mask]


def load_interlayer_dft_flat_masked(level: str) -> Tuple[List, np.ndarray]:
    """Load interlayer DFT and drop MD / buckled frames (same pipeline as ``test_compare_qmc``).

    Mirrors the first part of ``test_compare_qmc._load_dft_filtered``: chdir to
    ``uncertainty_quantification/``, ``load_energy_data(...)``, then keep only
    structures where every ``mol-id`` layer is flat (:func:`_has_flat_layers`).
    """
    uq = _repo_root() / "uncertainty_quantification"
    if not uq.is_dir():
        raise FileNotFoundError(f"Expected uncertainty_quantification at {uq}")
    from blg_model_builder_v2.DataLoader import load_energy_data

    with _working_directory(uq):
        atoms_list, energies, _f = load_energy_data(
            "interlayer", supercells=1, level_of_theory=level,
        )
    energies = np.asarray(energies, dtype=float).ravel()
    flat_mask = np.asarray([_has_flat_layers(a) for a in atoms_list], dtype=bool)
    atoms_kept = [a for a, ok in zip(atoms_list, flat_mask) if ok]
    return atoms_kept, energies[flat_mask]


def _filter_dft_ab_near_246(
    atoms_list: List,
    energies: np.ndarray,
    *,
    a0: float = 2.46,
    a_tol: float = 0.02,
    ab_xy_min: float = 0.22,
    ab_xy_max: float = 1.55,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (d, E_total, n_skipped) for primitive AB-like near-equilibrium in-plane cell."""
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if abs(L1 - a0) > a_tol or abs(L2 - a0) > a_tol:
            n_skip += 1
            continue
        if not _is_unstrained_hexagonal(atoms):
            n_skip += 1
            continue
        d_sep, ab_score = _layer_separation_and_ab_score(atoms)
        if not np.isfinite(d_sep) or not (ab_xy_min <= ab_score <= ab_xy_max):
            n_skip += 1
            continue
        d_list.append(d_sep)
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    return np.asarray(d_list, dtype=float)[order], np.asarray(e_list, dtype=float)[order], n_skip


def _reference_hex_inplane_rows(*, a: float = 2.46) -> np.ndarray:
    """Unstrained primitive in-plane Bravais rows (``geom_tools.get_lattice_vectors`` xy)."""
    return np.array(
        [[a, 0.0], [0.5 * a, 0.5 * (3.0 ** 0.5) * a]],
        dtype=float,
    )


def _recover_generator_strains_dx_dy(
    atoms,
    *,
    a: float = 2.46,
    align_cos_min: float = STRAIN_AXIS_ALIGN_COS_MIN,
) -> Tuple[float, float]:
    """Recover ``(dx, dy)`` from ``cell[0:2,0:2]`` vs unstrained hex rows (generator convention)."""
    rmat = _reference_hex_inplane_rows(a=a)
    mmat = np.asarray(atoms.cell[:2, :2], dtype=float)
    dx_out: list[float] = []
    for i in range(2):
        ri = rmat[i]
        mi = mmat[i]
        nri = float(np.linalg.norm(ri))
        nmi = float(np.linalg.norm(mi))
        if nri <= 0.0 or nmi <= 0.0:
            return float("nan"), float("nan")
        c_align = float(np.dot(mi, ri) / (nmi * nri))
        if c_align < float(align_cos_min):
            return float("nan"), float("nan")
        dx_out.append(nmi / nri - 1.0)
    return float(dx_out[0]), float(dx_out[1])


def _strains_match_generator_targets(
    dx: float,
    dy: float,
    *,
    target_dx: float,
    target_dy: float,
    tol: float = STRAIN_MATCH_TOL,
) -> bool:
    """Same selection idea as ``gen_training_data_structures`` bilayer loops: ``dx >= dy``."""
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return False
    if dx < dy - 1e-12:
        return False
    return bool(abs(dx - target_dx) < tol and abs(dy - target_dy) < tol)


def _filter_dft_ab_generator_strain(
    atoms_list: List,
    energies: np.ndarray,
    *,
    target_dx: float = STRAIN_FRAC,
    target_dy: float = STRAIN_DY,
    a0: float = 2.46,
    a_tol: float = 0.02,
    match_tol: float = STRAIN_MATCH_TOL,
    ab_xy_min: float = 0.22,
    ab_xy_max: float = 1.55,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """AB-like flat scans whose recovered ``(dx, dy)`` match the generator strain."""
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if (
            abs(L1 - a0) <= a_tol
            and abs(L2 - a0) <= a_tol
            and _is_unstrained_hexagonal(atoms)
        ):
            n_skip += 1
            continue
        rdx, rdy = _recover_generator_strains_dx_dy(atoms, a=a0)
        if not _strains_match_generator_targets(
            rdx, rdy, target_dx=target_dx, target_dy=target_dy, tol=match_tol,
        ):
            n_skip += 1
            continue
        d_sep, ab_score = _layer_separation_and_ab_score(atoms)
        if not np.isfinite(d_sep) or not (ab_xy_min <= ab_score <= ab_xy_max):
            n_skip += 1
            continue
        d_list.append(d_sep)
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    return np.asarray(d_list, dtype=float)[order], np.asarray(e_list, dtype=float)[order], n_skip


def _filter_dft_ab_generator_strain_atoms_energies(
    atoms_list: List,
    energies: np.ndarray,
    *,
    target_dx: float = STRAIN_FRAC,
    target_dy: float = STRAIN_DY,
    a0: float = 2.46,
    a_tol: float = 0.02,
    match_tol: float = STRAIN_MATCH_TOL,
    ab_xy_min: float = 0.22,
    ab_xy_max: float = 1.55,
) -> Tuple[List, np.ndarray, int]:
    """Same as :func:`_filter_dft_ab_generator_strain` but return ``Atoms`` + energies (sorted by ``d``)."""
    kept_atoms: list = []
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if (
            abs(L1 - a0) <= a_tol
            and abs(L2 - a0) <= a_tol
            and _is_unstrained_hexagonal(atoms)
        ):
            n_skip += 1
            continue
        rdx, rdy = _recover_generator_strains_dx_dy(atoms, a=a0)
        if not _strains_match_generator_targets(
            rdx, rdy, target_dx=target_dx, target_dy=target_dy, tol=match_tol,
        ):
            n_skip += 1
            continue
        d_sep, ab_score = _layer_separation_and_ab_score(atoms)
        if not np.isfinite(d_sep) or not (ab_xy_min <= ab_score <= ab_xy_max):
            n_skip += 1
            continue
        kept_atoms.append(atoms)
        d_list.append(float(d_sep))
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    atoms_ord = [kept_atoms[i] for i in order]
    e_ord = np.asarray([e_list[i] for i in order], dtype=float)
    return atoms_ord, e_ord, n_skip


def _filter_dft_aa_generator_strain_atoms_energies(
    atoms_list: List,
    energies: np.ndarray,
    *,
    target_dx: float = STRAIN_FRAC,
    target_dy: float = STRAIN_DY,
    a0: float = 2.46,
    a_tol: float = 0.02,
    match_tol: float = STRAIN_MATCH_TOL,
    xy_tol: float = 0.04,
) -> Tuple[List, np.ndarray, int]:
    """Same as :func:`_filter_dft_aa_generator_strain` but return ``Atoms`` + energies (sorted by ``d``)."""
    kept_atoms: list = []
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if (
            abs(L1 - a0) <= a_tol
            and abs(L2 - a0) <= a_tol
            and _is_unstrained_hexagonal(atoms)
        ):
            n_skip += 1
            continue
        rdx, rdy = _recover_generator_strains_dx_dy(atoms, a=a0)
        if not _strains_match_generator_targets(
            rdx, rdy, target_dx=target_dx, target_dy=target_dy, tol=match_tol,
        ):
            n_skip += 1
            continue
        if not _dft_layers_xy_aligned_aa(atoms, xy_tol=xy_tol):
            n_skip += 1
            continue
        d_sep = _mean_interlayer_separation(atoms)
        if not np.isfinite(d_sep):
            n_skip += 1
            continue
        kept_atoms.append(atoms)
        d_list.append(float(d_sep))
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    atoms_ord = [kept_atoms[i] for i in order]
    e_ord = np.asarray([e_list[i] for i in order], dtype=float)
    return atoms_ord, e_ord, n_skip


def _generator_strained_dft_atoms_energies_parity(
    atoms_list: List,
    energies: np.ndarray,
) -> Tuple[List, np.ndarray, List[str], int, int]:
    """In-plane generator-strained DFT configs: **AB** then **AA** (each sorted by ``d``).

    Returns ``(atoms, energies, stacking_labels, n_skip_ab, n_skip_aa)``.
    """
    ab_a, ab_e, sab = _filter_dft_ab_generator_strain_atoms_energies(
        atoms_list, energies,
    )
    aa_a, aa_e, saa = _filter_dft_aa_generator_strain_atoms_energies(
        atoms_list, energies,
    )
    atoms_out = list(ab_a) + list(aa_a)
    e_out = np.concatenate(
        [np.asarray(ab_e, dtype=float), np.asarray(aa_e, dtype=float)],
    )
    labels = ["AB"] * len(ab_a) + ["AA"] * len(aa_a)
    return atoms_out, e_out, labels, sab, saa


def _dft_layers_xy_aligned_aa(atoms, *, xy_tol: float = 0.04) -> bool:
    """True if primitive bilayer is AA: same ``(x,y)`` registry (no in-plane shift).

    Each bottom C pairs with a distinct top C at ``‖Δxy‖ < xy_tol`` (both
    layer bijections are tried).
    """
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    layers = np.searchsorted(np.unique(mol), mol).astype(int)
    if np.unique(layers).size != 2:
        return False
    p0 = pos[layers == 0][:, :2]
    p1 = pos[layers == 1][:, :2]
    if p0.shape != (2, 2) or p1.shape != (2, 2):
        return False
    match01 = max(
        float(np.linalg.norm(p0[0] - p1[0])),
        float(np.linalg.norm(p0[1] - p1[1])),
    )
    match10 = max(
        float(np.linalg.norm(p0[0] - p1[1])),
        float(np.linalg.norm(p0[1] - p1[0])),
    )
    return bool(min(match01, match10) < xy_tol)


def _mean_interlayer_separation(atoms) -> float:
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    layers = np.searchsorted(np.unique(mol), mol).astype(int)
    z0 = pos[layers == 0, 2]
    z1 = pos[layers == 1, 2]
    return float(np.mean(z1) - np.mean(z0))


def _filter_dft_aa_unstrained_near_246(
    atoms_list: List,
    energies: np.ndarray,
    *,
    a0: float = 2.46,
    a_tol: float = 0.02,
    xy_tol: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (d, E_total, n_skipped): unstrained primitive AA (same xy across layers)."""
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if abs(L1 - a0) > a_tol or abs(L2 - a0) > a_tol:
            n_skip += 1
            continue
        if not _is_unstrained_hexagonal(atoms):
            n_skip += 1
            continue
        if not _dft_layers_xy_aligned_aa(atoms, xy_tol=xy_tol):
            n_skip += 1
            continue
        d_sep = _mean_interlayer_separation(atoms)
        if not np.isfinite(d_sep):
            n_skip += 1
            continue
        d_list.append(d_sep)
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    return np.asarray(d_list, dtype=float)[order], np.asarray(e_list, dtype=float)[order], n_skip


def _filter_dft_aa_generator_strain(
    atoms_list: List,
    energies: np.ndarray,
    *,
    target_dx: float = STRAIN_FRAC,
    target_dy: float = STRAIN_DY,
    a0: float = 2.46,
    a_tol: float = 0.02,
    match_tol: float = STRAIN_MATCH_TOL,
    xy_tol: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """AA registry with recovered ``(dx, dy)`` matching the generator strain."""
    d_list: list[float] = []
    e_list: list[float] = []
    n_skip = 0
    for atoms, e in zip(atoms_list, np.asarray(energies, dtype=float).ravel()):
        if len(atoms) != 4:
            n_skip += 1
            continue
        L1, L2, _ = _hex_cell_metrics(atoms)
        if (
            abs(L1 - a0) <= a_tol
            and abs(L2 - a0) <= a_tol
            and _is_unstrained_hexagonal(atoms)
        ):
            n_skip += 1
            continue
        rdx, rdy = _recover_generator_strains_dx_dy(atoms, a=a0)
        if not _strains_match_generator_targets(
            rdx, rdy, target_dx=target_dx, target_dy=target_dy, tol=match_tol,
        ):
            n_skip += 1
            continue
        if not _dft_layers_xy_aligned_aa(atoms, xy_tol=xy_tol):
            n_skip += 1
            continue
        d_sep = _mean_interlayer_separation(atoms)
        if not np.isfinite(d_sep):
            n_skip += 1
            continue
        d_list.append(d_sep)
        e_list.append(float(e))
    order = np.argsort(np.asarray(d_list, dtype=float))
    return np.asarray(d_list, dtype=float)[order], np.asarray(e_list, dtype=float)[order], n_skip


def _bilayer_apply_generator_inplane_strain(
    atoms,
    dx: float,
    dy: float,
) -> Any:
    """Apply row-wise in-plane strains like ``gen_training_data_structures.py``."""
    out = atoms.copy()
    cell = np.asarray(out.cell, dtype=float).copy()
    cell[0, :] *= 1.0 + float(dx)
    cell[1, :] *= 1.0 + float(dy)
    out.set_cell(cell, scale_atoms=True)
    return out


def _artifacts_dir() -> Path:
    d = Path(__file__).resolve().parent / "_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_dft_interlayer_rvv10() -> Tuple[List, np.ndarray, np.ndarray]:
    """Load primitive interlayer rVV10 set; cwd must be uncertainty_quantification/."""
    from blg_model_builder_v2.DataLoader import load_energy_data

    return load_energy_data("interlayer", supercells=1, level_of_theory="rVV10")


def _plot_pes(
    title: str,
    d_model_ab: np.ndarray,
    e_model_ab_pa: np.ndarray,
    d_dft_ab: np.ndarray,
    e_dft_ab_pa: np.ndarray,
    out_path: Path,
    *,
    d_model_aa: Optional[np.ndarray] = None,
    e_model_aa_pa: Optional[np.ndarray] = None,
    d_dft_aa: Optional[np.ndarray] = None,
    e_dft_aa_pa: Optional[np.ndarray] = None,
    model_source_label: str = "model",
    dft_source_label: str = "rVV10 DFT",
    d_plot_window: Optional[Tuple[float, float]] = None,
) -> None:
    assert plt is not None
    print(f"[test_PES] PES equilibrium estimates ({out_path.name}):", flush=True)
    dm_ab = np.asarray(d_model_ab, dtype=float).ravel()
    em_ab = np.asarray(e_model_ab_pa, dtype=float).ravel()
    dd_ab = np.asarray(d_dft_ab, dtype=float).ravel()
    ed_ab = np.asarray(e_dft_ab_pa, dtype=float).ravel()
    strain_u = _strain_state_label(unstrained=True)

    for stacking, d_a, e_a, src, use_dft_quadratic in (
        ("AB", dm_ab, em_ab, model_source_label, False),
        ("AB", dd_ab, ed_ab, dft_source_label, True),
    ):
        if use_dft_quadratic:
            d_mn, e_mn, method = _dft_equilibrium_d_quadratic_fit(d_a, e_a)
        else:
            d_mn, e_mn = _d_and_e_at_discrete_minimum(d_a, e_a)
            method = "discrete_minimum"
        if d_mn is not None and e_mn is not None:
            _print_pes_minimum_line(
                datasource=src,
                stacking=stacking,
                strain=strain_u,
                d_eq=d_mn,
                e_eq=e_mn,
                equilibrium_method=method,
                d_window=d_plot_window,
            )

    dm_aa = (
        np.asarray(d_model_aa, dtype=float).ravel()
        if d_model_aa is not None
        else np.array([], dtype=float)
    )
    em_aa = (
        np.asarray(e_model_aa_pa, dtype=float).ravel()
        if e_model_aa_pa is not None
        else np.array([], dtype=float)
    )
    if dm_aa.size and em_aa.size == dm_aa.size:
        d_mn, e_mn = _d_and_e_at_discrete_minimum(dm_aa, em_aa)
        if d_mn is not None and e_mn is not None:
            _print_pes_minimum_line(
                datasource=model_source_label,
                stacking="AA",
                strain=strain_u,
                d_eq=d_mn,
                e_eq=e_mn,
                equilibrium_method="discrete_minimum",
                d_window=d_plot_window,
            )

    if d_dft_aa is not None and e_dft_aa_pa is not None:
        dd_aa = np.asarray(d_dft_aa, dtype=float).ravel()
        ed_aa = np.asarray(e_dft_aa_pa, dtype=float).ravel()
        if dd_aa.size and ed_aa.size == dd_aa.size:
            d_mn, e_mn, method = _dft_equilibrium_d_quadratic_fit(dd_aa, ed_aa)
            if d_mn is not None and e_mn is not None:
                _print_pes_minimum_line(
                    datasource=dft_source_label,
                    stacking="AA",
                    strain=strain_u,
                    d_eq=d_mn,
                    e_eq=e_mn,
                    equilibrium_method=method,
                    d_window=d_plot_window,
                )

    min_dft_energy = float(np.min(ed_ab))
    e_dft_ab_pa = ed_ab - min_dft_energy
    e_dft_aa_pa_plot: Optional[np.ndarray] = None
    if e_dft_aa_pa is not None:
        e_dft_aa_pa_plot = np.asarray(e_dft_aa_pa, dtype=float).ravel() - min_dft_energy
    min_model_energy = float(np.min(em_ab))
    e_model_ab_pa = em_ab - min_model_energy
    e_model_aa_pa = np.asarray(e_model_aa_pa, dtype=float).ravel() - min_model_energy
    fig, ax = plt.subplots(figsize=(7.5, 4.75))
    ax.plot(
        dm_ab,
        e_model_ab_pa,
        "-",
        lw=2.0,
        color="C0",
        label="model AB (per atom)",
    )
    ax.scatter(
        dd_ab,
        e_dft_ab_pa,
        s=38,
        c="k",
        zorder=5,
        label="DFT AB (per atom)",
    )
    if dm_aa.size and em_aa.size == dm_aa.size:
        ax.plot(
            dm_aa,
            e_model_aa_pa,
            "--",
            lw=2.0,
            color="C1",
            label="model AA — (per atom)",
        )
    if d_dft_aa is not None and e_dft_aa_pa is not None:
        dd_aa_sc = np.asarray(d_dft_aa, dtype=float).ravel()
        if (
            dd_aa_sc.size
            and e_dft_aa_pa_plot is not None
            and e_dft_aa_pa_plot.size == dd_aa_sc.size
        ):
            ax.scatter(
                dd_aa_sc,
                e_dft_aa_pa_plot,
                s=38,
                c="C3",
                marker="^",
                zorder=5,
                edgecolors="darkred",
                linewidths=0.4,
                label="DFT AA — (per atom)",
            )
    ax.set_xlabel(r"Interlayer separation $d$ (Å)")
    ax.set_ylabel(r"Total energy / atom (eV)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_dft_vs_model_parity_strained_md(
    *,
    model_label: str,
    model_key: str,
    atoms_list: List,
    e_dft_total: np.ndarray,
    e_model_total: np.ndarray,
    out_path: Path,
) -> None:
    """Scatter DFT vs model **total** energy per atom in two groups by layer flatness.

    **Strained configs** (blue): every ``mol-id`` layer is coplanar in ``z``
    within :data:`PARITY_STRAINED_MD_ZTOL`.  **md configs** (orange): any layer
    has a larger within-layer ``z`` spread.
    """
    assert plt is not None
    n = len(atoms_list)
    if n == 0:
        return
    n_at_arr = np.array([max(int(len(a)), 1) for a in atoms_list], dtype=float)
    e_d_pa = np.asarray(e_dft_total, dtype=float).ravel() / n_at_arr
    e_m_pa = np.asarray(e_model_total, dtype=float).ravel() / n_at_arr
    if e_m_pa.size != e_d_pa.size:
        raise ValueError("parity: DFT and model energy length mismatch")

    kinds = [_parity_strained_vs_md_label(a) for a in atoms_list]
    strain_m = np.array([k == "strained" for k in kinds], dtype=bool)
    md_m = ~strain_m

    fig, ax = plt.subplots(figsize=(6.25, 6.25))
    finite = np.isfinite(e_d_pa) & np.isfinite(e_m_pa)
    if np.any(finite):
        lo = float(np.min(np.minimum(e_d_pa[finite], e_m_pa[finite])))
        hi = float(np.max(np.maximum(e_d_pa[finite], e_m_pa[finite])))
        span = max(hi - lo, 1e-6)
        pad = 0.02 * span
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            "k--",
            lw=1.0,
            alpha=0.75,
            label="y = x",
        )

    ztol = PARITY_STRAINED_MD_ZTOL
    if np.any(strain_m & finite):
        ax.scatter(
            e_d_pa[strain_m & finite],
            e_m_pa[strain_m & finite],
            s=42,
            c="C0",
            edgecolors="navy",
            linewidths=0.45,
            label=f"strained configs (same $z$ per layer, Δ ≤ {ztol:g} Å)",
            zorder=4,
        )
    if np.any(md_m & finite):
        ax.scatter(
            e_d_pa[md_m & finite],
            e_m_pa[md_m & finite],
            s=42,
            c="C1",
            edgecolors="darkred",
            linewidths=0.45,
            label="md configs",
            zorder=4,
        )

    ax.set_xlabel(r"DFT energy / atom (eV)")
    ax.set_ylabel(f"{model_label} energy / atom (eV)")
    ax.set_title(
        f"Parity: DFT vs {model_label} [{model_key}]\n"
        r"(full interlayer rVV10 dataset)",
    )
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(True, alpha=0.35)
    if np.any(finite):
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _evaluate_unstrained_bilayer_pes_curves(
    model_key: str,
    d_grid: np.ndarray,
    *,
    a0: float = 2.46,
    n_at: int = 4,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """AB / AA total energy per atom on ``d_grid`` for one ``MODEL_KEYS`` entry."""
    case = tr.build_relaxation_case(model_key)
    e_ab = np.empty_like(d_grid, dtype=float)
    e_aa = np.empty_like(d_grid, dtype=float)
    calc_pod: Any = None
    calc_tp: Any = None
    try:
        for i, d in enumerate(d_grid):
            atoms_ab = get_bilayer_atoms(float(d), 0.0, a=float(a0), sc=1)
            atoms_aa = get_aa_bilayer_atoms(float(d), a=float(a0), sc=1)
            if model_key == "pod_energy":
                if calc_pod is None:
                    calc_pod = case.calc_factory(atoms_ab)
                atoms_ab.calc = calc_pod
                atoms_aa.calc = calc_pod
            elif model_key == "tetb_pod":
                if calc_tp is None:
                    calc_tp = case.calc_factory(atoms_ab)
                atoms_ab.calc = calc_tp
                atoms_aa.calc = calc_tp
            else:
                atoms_ab.calc = case.calc_factory(atoms_ab)
                atoms_aa.calc = case.calc_factory(atoms_aa)
            e_ab[i] = float(atoms_ab.get_potential_energy())
            e_aa[i] = float(atoms_aa.get_potential_energy())
    finally:
        if calc_pod is not None and hasattr(calc_pod, "close"):
            calc_pod.close()
        if calc_tp is not None and hasattr(calc_tp, "close"):
            calc_tp.close()
    return case.energy_display_name, e_ab / float(n_at), e_aa / float(n_at)


def _d_and_e_at_discrete_minimum(
    d: np.ndarray,
    e: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """``(d, e)`` at the lowest *e* among finite aligned samples; ``(None, None)`` if unusable."""
    d = np.asarray(d, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    if d.size == 0 or e.size == 0 or d.size != e.size:
        return None, None
    m = np.isfinite(d) & np.isfinite(e)
    if not np.any(m):
        return None, None
    d, e = d[m], e[m]
    j = int(np.argmin(e))
    return float(d[j]), float(e[j])


def _dft_equilibrium_d_quadratic_fit(
    d: np.ndarray,
    e: np.ndarray,
    *,
    n_each_side: int = 2,
    min_fit_points: int = 4,
    vertex_margin_A: float = 0.75,
) -> Tuple[Optional[float], Optional[float], str]:
    """Equilibrium spacing for sparse DFT samples: quadratic fit near discrete minimum, vertex.

    Points are sorted by ``d``; a window around the lowest-energy sample is fit with
    ``deg=2``. If the fit is convex (``a_2 > 0``) and the vertex lies near that window,
    returns ``(d_vertex, E(d_vertex), method_tag)``. Otherwise falls back to the discrete
    minimum on the full set.
    """
    d = np.asarray(d, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    if d.size != e.size or d.size < 3:
        return None, None, "insufficient_points"
    m = np.isfinite(d) & np.isfinite(e)
    if not np.any(m):
        return None, None, "insufficient_points"
    d, e = d[m], e[m]

    def _discrete_tag(tag: str) -> Tuple[Optional[float], Optional[float], str]:
        d0, e0 = _d_and_e_at_discrete_minimum(d, e)
        if d0 is None:
            return None, None, "insufficient_points"
        return d0, e0, f"discrete_minimum_fallback({tag})"

    order = np.argsort(d, kind="mergesort")
    d_s = d[order]
    e_s = e[order]
    n = int(d_s.size)
    j = int(np.argmin(e_s))
    w = max(1, int(n_each_side))
    i0 = max(0, j - w)
    i1 = min(n - 1, j + w)
    target = max(3, min(min_fit_points, n))
    while i1 - i0 + 1 < target and (i0 > 0 or i1 < n - 1):
        if i0 > 0:
            i0 -= 1
        if i1 - i0 + 1 >= target:
            break
        if i1 < n - 1:
            i1 += 1

    d_w = d_s[i0 : i1 + 1]
    e_w = e_s[i0 : i1 + 1]
    if d_w.size < 3:
        return _discrete_tag("fit_window_lt_3")

    coef = np.polyfit(d_w, e_w, 2)
    a2, a1 = float(coef[0]), float(coef[1])
    if abs(a2) < 1e-20 or a2 <= 0.0:
        return _discrete_tag("nonconvex_or_singular_quadratic")

    d_vertex = -0.5 * a1 / a2
    span_lo, span_hi = float(d_w[0]), float(d_w[-1])
    mg = float(vertex_margin_A)
    if d_vertex < span_lo - mg or d_vertex > span_hi + mg:
        return _discrete_tag("vertex_outside_margin")

    e_vertex = float(np.polyval(coef, d_vertex))
    return d_vertex, e_vertex, f"quadratic_vertex(n_fit={d_w.size})"


def _strain_state_label(*, unstrained: bool) -> str:
    if unstrained:
        return "unstrained"
    return (
        f"strained (dx={STRAIN_FRAC * 100:+.1f}%, dy={STRAIN_DY * 100:+.1f}%)"
    )


def _print_pes_minimum_line(
    *,
    datasource: str,
    stacking: str,
    strain: str,
    d_eq: float,
    e_eq: float,
    equilibrium_method: str,
    d_window: Optional[Tuple[float, float]] = None,
) -> None:
    win = ""
    if d_window is not None:
        lo, hi = d_window
        win = f" | d_plot_window=[{lo:.4g},{hi:.4g}] Å"
    print(
        "[test_PES] PES equilibrium | "
        f"datasource={datasource} | stacking={stacking} | strain={strain} | "
        f"d_eq={d_eq:.4f} Å | E(d_eq)/atom={e_eq:.6f} eV | method={equilibrium_method}"
        f"{win}",
        flush=True,
    )


def _clip_d_e(
    d: np.ndarray,
    e: np.ndarray,
    lo: float,
    hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    if d.size == 0:
        return d, e
    m = (d >= lo) & (d <= hi)
    return d[m], e[m]


def _dft_strain_curve_bundle(
    level: str,
    *,
    lo: float,
    hi: float,
    n_at: int = 4,
) -> dict[str, np.ndarray]:
    """Per-atom energies vs ``d`` for unstrained / strained AB and AA.

    **Unstrained** curves use the same filters as ``test_compare_qmc._load_dft_filtered``
    (flat layers, then :func:`_filter_dft_ab_near_246` /
    :func:`_filter_dft_aa_unstrained_near_246`).  **Strained** curves use the same
    flat list, then AB / AA registry tests with :func:`_recover_generator_strains_dx_dy`
    matched to ``(STRAIN_FRAC, STRAIN_DY)`` (generator ``dx >= dy`` rule).
    """
    atoms_f, e_f = load_interlayer_dft_flat_masked(level)
    d_ab_u, e_ab_u, _ = _filter_dft_ab_near_246(atoms_f, e_f, a0=2.46)
    d_aa_u, e_aa_u, _ = _filter_dft_aa_unstrained_near_246(atoms_f, e_f, a0=2.46)
    d_ab_s, e_ab_s, _ = _filter_dft_ab_generator_strain(
        atoms_f, e_f, target_dx=STRAIN_FRAC, target_dy=STRAIN_DY,
    )
    d_aa_s, e_aa_s, _ = _filter_dft_aa_generator_strain(
        atoms_f, e_f, target_dx=STRAIN_FRAC, target_dy=STRAIN_DY,
    )
    out: dict[str, np.ndarray] = {}
    for key, d_raw, e_raw in (
        ("ab_u", d_ab_u, e_ab_u),
        ("ab_s", d_ab_s, e_ab_s),
        ("aa_u", d_aa_u, e_aa_u),
        ("aa_s", d_aa_s, e_aa_s),
    ):
        d_c, e_c = _clip_d_e(d_raw, e_raw, lo, hi)
        out[f"d_{key}"] = d_c
        out[f"e_{key}_pa"] = e_c / float(n_at)
    return out


def _model_strain_curve_bundle(
    model_key: str,
    d_grid: np.ndarray,
    *,
    a0: float = 2.46,
    strain_dx: float = STRAIN_FRAC,
    strain_dy: float = STRAIN_DY,
    n_at: int = 4,
) -> Tuple[str, dict[str, np.ndarray]]:
    """``e_*_pa`` on ``d_grid``; strained branch uses generator ``(dx, dy)`` on rows 0/1."""
    case = tr.build_relaxation_case(model_key)
    e_ab_u = np.empty_like(d_grid, dtype=float)
    e_ab_s = np.empty_like(d_grid, dtype=float)
    e_aa_u = np.empty_like(d_grid, dtype=float)
    e_aa_s = np.empty_like(d_grid, dtype=float)
    calc_pod: Any = None
    calc_tp: Any = None
    try:
        for i, d in enumerate(d_grid):
            ab_u = get_bilayer_atoms(float(d), 0.0, a=float(a0), sc=1)
            ab_s = _bilayer_apply_generator_inplane_strain(ab_u, strain_dx, strain_dy)
            aa_u = get_aa_bilayer_atoms(float(d), a=float(a0), sc=1)
            aa_s = _bilayer_apply_generator_inplane_strain(aa_u, strain_dx, strain_dy)
            if model_key == "pod_energy":
                if calc_pod is None:
                    calc_pod = case.calc_factory(ab_u)
                for at in (ab_u, ab_s, aa_u, aa_s):
                    at.calc = calc_pod
            elif model_key == "tetb_pod":
                if calc_tp is None:
                    calc_tp = case.calc_factory(ab_u)
                for at in (ab_u, ab_s, aa_u, aa_s):
                    at.calc = calc_tp
            else:
                ab_u.calc = case.calc_factory(ab_u)
                ab_s.calc = case.calc_factory(ab_s)
                aa_u.calc = case.calc_factory(aa_u)
                aa_s.calc = case.calc_factory(aa_s)
            e_ab_u[i] = float(ab_u.get_potential_energy())
            e_ab_s[i] = float(ab_s.get_potential_energy())
            e_aa_u[i] = float(aa_u.get_potential_energy())
            e_aa_s[i] = float(aa_s.get_potential_energy())
    finally:
        if calc_pod is not None and hasattr(calc_pod, "close"):
            calc_pod.close()
        if calc_tp is not None and hasattr(calc_tp, "close"):
            calc_tp.close()
    bundle = {
        "d_grid": np.asarray(d_grid, dtype=float),
        "e_ab_u_pa": e_ab_u / float(n_at),
        "e_ab_s_pa": e_ab_s / float(n_at),
        "e_aa_u_pa": e_aa_u / float(n_at),
        "e_aa_s_pa": e_aa_s / float(n_at),
    }
    return case.energy_display_name, bundle


def _plot_strain_comparison_figure(
    *,
    stacking: str,
    lo: float,
    hi: float,
    series: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """One stacking (``"AB"`` or ``"AA"``): unstrained solid, strained dashed per source."""
    assert plt is not None
    assert stacking in ("AB", "AA")
    suf_u, suf_s = ("ab_u", "ab_s") if stacking == "AB" else ("aa_u", "aa_s")
    all_e: list[float] = []
    for s in series:
        for key in (f"e_{suf_u}_pa", f"e_{suf_s}_pa"):
            arr = np.asarray(s.get(key, []), dtype=float).ravel()
            if arr.size:
                all_e.extend(float(x) for x in arr)
    ref = float(np.min(all_e)) if all_e else 0.0

    print(
        f"[test_PES] PES equilibrium estimates ({out_path.name}, stacking={stacking}):",
        flush=True,
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    n_series = max(1, len(series))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(8, n_series)))

    d_win = (float(lo), float(hi))
    strain_u = _strain_state_label(unstrained=True)
    strain_s = _strain_state_label(unstrained=False)

    for i, s in enumerate(series):
        c = colors[i % len(colors)]
        name = str(s["name"])
        d_u = np.asarray(s.get(f"d_{suf_u}", []), dtype=float).ravel()
        e_u_raw = np.asarray(s.get(f"e_{suf_u}_pa", []), dtype=float).ravel()
        e_u = e_u_raw - ref
        d_s = np.asarray(s.get(f"d_{suf_s}", []), dtype=float).ravel()
        e_s_raw = np.asarray(s.get(f"e_{suf_s}_pa", []), dtype=float).ravel()
        e_s = e_s_raw - ref

        is_dft = str(name).endswith(" DFT")
        if is_dft:
            d_mn, e_mn, method = _dft_equilibrium_d_quadratic_fit(d_u, e_u_raw)
        else:
            d_mn, e_mn = _d_and_e_at_discrete_minimum(d_u, e_u_raw)
            method = "discrete_minimum"
        if d_mn is not None and e_mn is not None:
            _print_pes_minimum_line(
                datasource=name,
                stacking=stacking,
                strain=strain_u,
                d_eq=d_mn,
                e_eq=e_mn,
                equilibrium_method=method,
                d_window=d_win,
            )
        if is_dft:
            d_mn, e_mn, method = _dft_equilibrium_d_quadratic_fit(d_s, e_s_raw)
        else:
            d_mn, e_mn = _d_and_e_at_discrete_minimum(d_s, e_s_raw)
            method = "discrete_minimum"
        if d_mn is not None and e_mn is not None:
            _print_pes_minimum_line(
                datasource=name,
                stacking=stacking,
                strain=strain_s,
                d_eq=d_mn,
                e_eq=e_mn,
                equilibrium_method=method,
                d_window=d_win,
            )

        if d_u.size and e_u.size == d_u.size:
            ax.plot(
                d_u,
                e_u,
                "-",
                color=c,
                lw=2.0,
                label=f"{name} unstrained",
            )
            ax.scatter(d_u, e_u, s=28, color=c, zorder=4, marker="o")
        if d_s.size and e_s.size == d_s.size:
            ax.plot(
                d_s,
                e_s,
                "--",
                color=c,
                lw=2.0,
                label=(
                    f"{name} strained "
                    f"(dx={STRAIN_FRAC * 100:+.1f}%, dy={STRAIN_DY * 100:+.1f}%)"
                ),
            )
            ax.scatter(d_s, e_s, s=28, color=c, zorder=4, marker="s", facecolors="none", linewidths=1.2)

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"Interlayer separation $d$ (Å)")
    ax.set_ylabel(r"Energy / atom $-$ min (curve set) (eV)")
    ax.set_title(
        f"Bilayer {stacking}: unstrained vs generator in-plane strain "
        f"(dx={STRAIN_FRAC * 100:+.1f}%, dy={STRAIN_DY * 100:+.1f}%; flat-layer DFT)",
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_blg_pes_models_vs_dft():
    """Write ``PES_<model>_vs_DFT.png`` and ``PES_<model>_parity_DFT_vs_model_strained_md.png``."""
    try:
        expanded = _expand_datasources()
    except ValueError as exc:
        pytest.fail(str(exc))
    model_set = set(tr.MODEL_KEYS)
    want_models = [s for s in expanded if s in model_set]
    if not want_models:
        pytest.skip(
            "DATASOURCES has no model keys (e.g. add 'pod_energy', 'tetb_pod', or 'models').",
        )
    if any(m in tr._MODELS_NEEDING_LAMMPS for m in want_models):
        try:
            import lammps  # noqa: F401
        except Exception as exc:
            pytest.skip(f"LAMMPS Python module not available for model energies: {exc}")
    tr._ensure_get_mcmc_inputs_importable()

    if "tetb_pod" in want_models:
        from get_MCMC_inputs import build_tetb_pod_hyperparams_from_data_kw

        _, _pod_hp_tetb, _ = build_tetb_pod_hyperparams_from_data_kw(
            {
                "tb_M": tr.RELAXATION_TETB_POD_TB_M,
                "tb_W": tr.RELAXATION_TETB_POD_TB_W,
                "pod_M": tr.RELAXATION_TETB_POD_POD_M,
                "pod_W": tr.RELAXATION_TETB_POD_POD_W,
            },
        )
        assert int(_pod_hp_tetb["bessel_polynomial_degree"]) == tr.POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE
        assert int(_pod_hp_tetb["inverse_polynomial_degree"]) == tr.POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE

    uq = _repo_root() / "uncertainty_quantification"
    if not uq.is_dir():
        pytest.skip(f"Missing uncertainty_quantification at {uq}")

    with _working_directory(uq):
        atoms_list, energies, _forces = _load_dft_interlayer_rvv10()

    # Entire loaded set for DFT–model parity (no flat-layer / strain / stacking filter).
    atoms_parity = list(atoms_list)
    e_dft_parity = np.asarray(energies, dtype=float).ravel().copy()

    atoms_list, energies = _prefilter_flat_layers(atoms_list, energies)

    d_dft_ab, e_dft_ab, n_skip_ab = _filter_dft_ab_near_246(atoms_list, energies, a0=2.46)
    if d_dft_ab.size < 3:
        pytest.skip(
            f"Too few DFT points after AB / a≈2.46 / unstrained filter "
            f"(got {d_dft_ab.size}, skipped {n_skip_ab}).",
        )

    d_dft_aa, e_dft_aa, n_skip_aa = _filter_dft_aa_unstrained_near_246(
        atoms_list, energies, a0=2.46,
    )

    n_at = 4
    e_dft_ab_pa = e_dft_ab / n_at
    e_dft_aa_pa = e_dft_aa / n_at if d_dft_aa.size else np.array([], dtype=float)

    d_lo = 3.0
    d_hi = 4.5
    d_model = np.linspace(d_lo, d_hi, max(40, int((d_hi - d_lo) / 0.02)))

    art = _artifacts_dir()
    written: list[str] = []
    failed: list[str] = []
    for model_key in want_models:
        try:
            label, e_ab_pa, e_aa_pa = _evaluate_unstrained_bilayer_pes_curves(
                model_key, d_model, a0=2.46, n_at=n_at,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            failed.append(f"{model_key}: {exc}")
            continue
        out = art / f"PES_{model_key}_vs_DFT.png"
        _plot_pes(
            f"{label} vs DFT — AB & AA, a≈2.46 Å",
            d_model,
            e_ab_pa,
            d_dft_ab,
            e_dft_ab_pa,
            out,
            d_model_aa=d_model,
            e_model_aa_pa=e_aa_pa,
            d_dft_aa=d_dft_aa if d_dft_aa.size else None,
            e_dft_aa_pa=e_dft_aa_pa if d_dft_aa.size else None,
            model_source_label=f"{label} [{model_key}]",
            dft_source_label="rVV10 DFT",
            d_plot_window=(d_lo, d_hi),
        )
        written.append(str(out))

        if atoms_parity:
            case = tr.build_relaxation_case(model_key)
            e_mod = np.full(len(atoms_parity), np.nan, dtype=float)
            calc_pod: Any = None
            calc_tp: Any = None
            n_parity_fail = 0
            try:
                for i, at0 in enumerate(atoms_parity):
                    at = at0.copy()
                    if model_key == "pod_energy":
                        if calc_pod is None:
                            calc_pod = case.calc_factory(at)
                        at.calc = calc_pod
                    elif model_key == "tetb_pod":
                        if calc_tp is None:
                            calc_tp = case.calc_factory(at)
                        at.calc = calc_tp
                    else:
                        at.calc = case.calc_factory(at)
                    try:
                        e_mod[i] = float(at.get_potential_energy())
                    except Exception:
                        n_parity_fail += 1
            finally:
                if calc_pod is not None and hasattr(calc_pod, "close"):
                    calc_pod.close()
                if calc_tp is not None and hasattr(calc_tp, "close"):
                    calc_tp.close()

            out_p = art / f"PES_{model_key}_parity_DFT_vs_model_strained_md.png"
            _plot_dft_vs_model_parity_strained_md(
                model_label=label,
                model_key=model_key,
                atoms_list=atoms_parity,
                e_dft_total=e_dft_parity,
                e_model_total=e_mod,
                out_path=out_p,
            )
            written.append(str(out_p))
            if n_parity_fail:
                print(
                    f"[test_PES] Parity model energies failed for {n_parity_fail}/"
                    f"{len(atoms_parity)} configs ({model_key}).",
                    flush=True,
                )
        else:
            print(
                "[test_PES] Skipping parity plot: empty DFT interlayer list.",
                flush=True,
            )

    if not written:
        pytest.skip(
            "No model PES figures produced for DATASOURCES="
            f"{list(DATASOURCES)!r}; failures: {'; '.join(failed)}",
        )
    if failed:
        print(f"[test_PES] Skipped models: {'; '.join(failed)}", flush=True)
    print(
        f"[test_PES] Wrote {written} "
        f"(models={want_models!r}; DFT AB: {d_dft_ab.size} pts, skipped {n_skip_ab}; "
        f"DFT AA: {d_dft_aa.size} pts, skipped {n_skip_aa}; "
        f"parity DFT configs: {len(atoms_parity)}).",
        flush=True,
    )


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_interlayer_pes_unstrained_vs_strained_ab_aa():
    """Unstrained vs strained ``a_1`` PES for AB and AA (two figures under ``_artifacts``)."""
    try:
        expanded = _expand_datasources()
    except ValueError as exc:
        pytest.fail(str(exc))
    plot_sources = [s for s in expanded if s in ("rVV10", "MBD") or s in set(tr.MODEL_KEYS)]
    if not plot_sources:
        pytest.skip(
            "DATASOURCES has no rVV10/MBD or model keys; nothing to plot for strain comparison.",
        )

    lo, hi = float(PLOT_D_MIN), float(PLOT_D_MAX)
    if hi <= lo:
        pytest.fail(f"PLOT_D_MAX ({hi}) must exceed PLOT_D_MIN ({lo}).")

    want_models = [s for s in plot_sources if s in set(tr.MODEL_KEYS)]
    if want_models and any(m in tr._MODELS_NEEDING_LAMMPS for m in want_models):
        try:
            import lammps  # noqa: F401
        except Exception as exc:
            pytest.skip(f"LAMMPS Python module not available for model energies: {exc}")
    if want_models:
        tr._ensure_get_mcmc_inputs_importable()

    uq = _repo_root() / "uncertainty_quantification"
    if not uq.is_dir():
        pytest.skip(f"Missing uncertainty_quantification at {uq}")

    span = hi - lo
    d_grid = np.linspace(lo, hi, max(48, int(span / 0.025)))

    series_ab_aa: list[dict[str, Any]] = []
    any_ab = any_aa = False
    for src in plot_sources:
        if src == "rVV10" or src == "MBD":
            try:
                b = _dft_strain_curve_bundle(src, lo=lo, hi=hi)
            except FileNotFoundError as exc:
                pytest.skip(f"{src} DFT unavailable: {exc}")
            label = f"{src} DFT"
            entry = {
                "name": label,
                "d_ab_u": b["d_ab_u"],
                "e_ab_u_pa": b["e_ab_u_pa"],
                "d_ab_s": b["d_ab_s"],
                "e_ab_s_pa": b["e_ab_s_pa"],
                "d_aa_u": b["d_aa_u"],
                "e_aa_u_pa": b["e_aa_u_pa"],
                "d_aa_s": b["d_aa_s"],
                "e_aa_s_pa": b["e_aa_s_pa"],
            }
            series_ab_aa.append(entry)
            if b["d_ab_u"].size or b["d_ab_s"].size:
                any_ab = True
            if b["d_aa_u"].size or b["d_aa_s"].size:
                any_aa = True
        else:
            try:
                label, mb = _model_strain_curve_bundle(src, d_grid)
            except (FileNotFoundError, ValueError, OSError) as exc:
                pytest.skip(f"Model {src!r} unavailable: {exc}")
            dg = mb["d_grid"]
            series_ab_aa.append(
                {
                    "name": label,
                    "d_ab_u": dg,
                    "e_ab_u_pa": mb["e_ab_u_pa"],
                    "d_ab_s": dg,
                    "e_ab_s_pa": mb["e_ab_s_pa"],
                    "d_aa_u": dg,
                    "e_aa_u_pa": mb["e_aa_u_pa"],
                    "d_aa_s": dg,
                    "e_aa_s_pa": mb["e_aa_s_pa"],
                },
            )
            any_ab = any_aa = True

    if not any_ab:
        pytest.skip("No AB data (unstrained or strained) in range after filters; widen grid or tolerances.")
    if not any_aa:
        pytest.skip("No AA data (unstrained or strained) in range after filters; widen grid or tolerances.")

    art = _artifacts_dir()
    out_ab = art / "PES_AB_unstrained_vs_strained_a1.png"
    out_aa = art / "PES_AA_unstrained_vs_strained_a1.png"
    _plot_strain_comparison_figure(stacking="AB", lo=lo, hi=hi, series=series_ab_aa, out_path=out_ab)
    _plot_strain_comparison_figure(stacking="AA", lo=lo, hi=hi, series=series_ab_aa, out_path=out_aa)

    assert out_ab.is_file() and out_ab.stat().st_size > 2000
    assert out_aa.is_file() and out_aa.stat().st_size > 2000
    print(
        f"[test_PES] Wrote strain comparison: {out_ab} and {out_aa} "
        f"(sources={plot_sources!r}, d∈[{lo:g},{hi:g}] Å).",
        flush=True,
    )


if __name__ == "__main__":
    # Allow ``python tests/test_PES.py`` when tests/ is on sys.path
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
