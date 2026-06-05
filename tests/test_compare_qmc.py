"""
Compare interlayer PES curves: **QMC**, **rVV10** / **MBD** DFT, and empirical
models listed in ``MODEL_KEYS`` from ``test_relaxation.py``.

Unstrained primitive bilayer DFT points use :func:`test_PES.load_interlayer_dft_flat_masked`
then the same AB / AA filters as ``test_PES.py`` (``a`` ≈ 2.46 Å, hexagonal cell,
registry).  QMC uses
``data/qmc.csv`` rows for **AB** (``disregistry = 0``) and **AA** at varying
``d`` with no in-plane strain.

Each data source is shifted by the **minimum AB energy per atom** for that
source.  **AB** curves are drawn solid and **AA** dashed; **one color per
source** (AB and AA share the color).  Discrete data are connected with cubic
splines (``scipy.interpolate.CubicSpline``).

Saves ``tests/_artifacts/compare_qmc_AB_AA.png``.  Edit ``DATASOURCES`` below to
choose which curves are drawn (QMC, DFT levels, and/or empirical models).
Layer separation is restricted to ``[PLOT_D_MIN, PLOT_D_MAX]`` (default 3–4.5 Å).

Run from repo root::

    pytest tests/test_compare_qmc.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy.interpolate import CubicSpline

from blg_model_builder.geom_tools import get_aa_bilayer_atoms, get_bilayer_atoms

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_relaxation as tr  # noqa: E402

from test_PES import (  # noqa: E402
    _filter_dft_aa_unstrained_near_246,
    _filter_dft_ab_near_246,
    load_interlayer_dft_flat_masked,
)

# ---------------------------------------------------------------------------
# User: which sources to plot (order = legend / color order). Recognized
# tokens (case-insensitive except model keys, which match ``MODEL_KEYS``):
#   ``"qmc"``       — ``data/qmc.csv`` (AB + AA vs ``d``)
#   ``"rVV10"``    — unstrained filtered rVV10 DFT (same filters as ``test_PES``)
#   ``"MBD"``      — same for MBD DFT
#   ``"models"``   — every calculator in ``test_relaxation.MODEL_KEYS``
#   or any single model key, e.g. ``"tersoff_kc"``, ``"pod_energy"``, ``"tetb_pod"``
# ---------------------------------------------------------------------------
DATASOURCES: Tuple[str, ...] = (
    "qmc",
    "rVV10",
    #"MBD",
    "models",
)

# Interlayer separation range shown on the plot and used for model grids (Å).
PLOT_D_MIN = 3.0
PLOT_D_MAX = 4.0


def _clip_d_e(d: np.ndarray, e: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Keep pairs whose ``d`` lies in ``[lo, hi]`` (inclusive)."""
    d = np.asarray(d, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    if d.size == 0:
        return d, e
    if e.size != d.size:
        raise ValueError(f"d length {d.size} != e length {e.size}")
    m = (d >= lo) & (d <= hi)
    return d[m], e[m]


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifacts_dir() -> Path:
    d = _TESTS_DIR / "_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cubic_spline_fine(
    d: np.ndarray,
    e: np.ndarray,
    *,
    n: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by ``d`` and evaluate a cubic spline on a dense mesh."""
    d = np.asarray(d, dtype=float)
    e = np.asarray(e, dtype=float)
    if d.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(d)
    d_s, e_s = d[order], e[order]
    if d_s.size == 1:
        return d_s.copy(), e_s.copy()
    if np.any(np.diff(d_s) <= 0):
        uniq_d: list[float] = []
        uniq_e: list[float] = []
        for di, ei in zip(d_s, e_s):
            if not uniq_d or di > uniq_d[-1]:
                uniq_d.append(float(di))
                uniq_e.append(float(ei))
            else:
                uniq_e[-1] = min(uniq_e[-1], float(ei))
        d_s = np.asarray(uniq_d, dtype=float)
        e_s = np.asarray(uniq_e, dtype=float)
    if d_s.size < 2:
        return d_s.copy(), e_s.copy()
    cs = CubicSpline(d_s, e_s)
    n_pts = max(int(n), len(d_s) * 10)
    d_fine = np.linspace(float(d_s[0]), float(d_s[-1]), n_pts)
    return d_fine, cs(d_fine)


def _shift_by_ab_min(
    e_ab: np.ndarray,
    e_aa: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Subtract ``min(e_ab)`` from AB and AA curves (per-atom energies)."""
    if e_ab.size == 0:
        ref = 0.0
        return e_ab.copy(), e_aa, ref
    ref = float(np.min(e_ab))
    e_ab_s = e_ab - ref
    if e_aa is None or e_aa.size == 0:
        return e_ab_s, e_aa, ref
    return e_ab_s, e_aa - ref, ref


def _load_qmc_aa_ab(repo: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """AB and AA rows from ``data/qmc.csv`` (energies eV/atom)."""
    path = repo / "data" / "qmc.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing QMC reference: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    ab = df[df["stacking"].astype(str).str.upper() == "AB"].copy()
    aa = df[df["stacking"].astype(str).str.upper() == "AA"].copy()
    o_ab = np.argsort(ab["d"].astype(float).values)
    o_aa = np.argsort(aa["d"].astype(float).values)
    d_ab = ab["d"].astype(float).values[o_ab]
    e_ab = ab["energy"].astype(float).values[o_ab]
    d_aa = aa["d"].astype(float).values[o_aa]
    e_aa = aa["energy"].astype(float).values[o_aa]
    return d_ab, e_ab, d_aa, e_aa


def _load_dft_filtered(level: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(d_ab, e_tot_ab, d_aa, e_tot_aa)`` for unstrained near-equilibrium DFT.

    Uses :func:`test_PES.load_interlayer_dft_flat_masked` (flat ``mol-id`` layers,
    then the same AB / AA geometry filters as ``test_PES``).
    """
    atoms_list, energies = load_interlayer_dft_flat_masked(level)
    d_ab, e_ab, _ = _filter_dft_ab_near_246(atoms_list, energies, a0=2.46)
    d_aa, e_aa, _ = _filter_dft_aa_unstrained_near_246(atoms_list, energies, a0=2.46)
    return d_ab, e_ab, d_aa, e_aa


def _model_curves(
    model_key: str,
    d_grid: np.ndarray,
    *,
    a_lat: float = 2.46,
    n_at: int = 4,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Return ``(display_name, e/N AB, e/N AA)`` on ``d_grid`` for primitive bilayer."""
    case = tr.build_relaxation_case(model_key)
    e_ab = np.empty_like(d_grid, dtype=float)
    e_aa = np.empty_like(d_grid, dtype=float)
    calc_pod = None
    calc_tp = None
    try:
        for i, d in enumerate(d_grid):
            atoms_ab = get_bilayer_atoms(float(d), 0.0, a=float(a_lat), sc=1)
            atoms_aa = get_aa_bilayer_atoms(float(d), a=float(a_lat), sc=1)
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
    return case.energy_display_name, e_ab / n_at, e_aa / n_at


def _plot_source(
    ax: "plt.Axes",
    name: str,
    color: str,
    d_ab: np.ndarray,
    e_ab: np.ndarray,
    d_aa: np.ndarray,
    e_aa: np.ndarray,
) -> None:
    """Plot one source: shifted AB/AA with splines; same ``color``, solid AB / dashed AA."""
    e_ab_s, e_aa_s, _ref = _shift_by_ab_min(e_ab, e_aa if d_aa.size else None)
    if d_ab.size:
        d_f, e_f = _cubic_spline_fine(d_ab, e_ab_s)
        if d_f.size:
            ax.plot(d_f, e_f, "-", color=color, lw=1.8, label=f"{name} AB")
        ax.plot(d_ab, e_ab_s, "o", color=color, ms=5, mfc=color, mec=color, zorder=4)
    if d_aa.size and e_aa_s is not None:
        d_f_aa, e_f_aa = _cubic_spline_fine(d_aa, e_aa_s)
        if d_f_aa.size:
            ax.plot(d_f_aa, e_f_aa, "--", color=color, lw=1.8, label=f"{name} AA")
        ax.plot(d_aa, e_aa_s, "s", color=color, ms=4, mfc=color, mec=color, zorder=4)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_compare_qmc_ab_aa_pes_plot():
    """Write ``compare_qmc_AB_AA.png`` for sources listed in ``DATASOURCES``."""
    try:
        expanded = _expand_datasources()
    except ValueError as exc:
        pytest.fail(str(exc))
    if not expanded:
        pytest.fail("DATASOURCES expanded to an empty list; add at least one source.")

    repo = _repo_root()
    n_at = 4
    model_set = set(tr.MODEL_KEYS)
    want_models = [s for s in expanded if s in model_set]

    lo, hi = float(PLOT_D_MIN), float(PLOT_D_MAX)
    if hi <= lo:
        pytest.fail(f"PLOT_D_MAX ({hi}) must exceed PLOT_D_MIN ({lo}).")

    d_q_ab = d_q_aa = np.array([])
    e_q_ab = e_q_aa = np.array([])
    if "qmc" in expanded:
        d_q_ab, e_q_ab, d_q_aa, e_q_aa = _load_qmc_aa_ab(repo)
        d_q_ab, e_q_ab = _clip_d_e(d_q_ab, e_q_ab, lo, hi)
        d_q_aa, e_q_aa = _clip_d_e(d_q_aa, e_q_aa, lo, hi)
        if d_q_ab.size < 2:
            pytest.skip(
                f"QMC needs ≥2 AB points in [{lo}, {hi}] Å for spline (got {d_q_ab.size}).",
            )

    d_r_ab = d_r_aa = np.array([])
    e_r_ab = e_r_aa = np.array([])
    if "rVV10" in expanded:
        try:
            d_r_ab, e_r_ab, d_r_aa, e_r_aa = _load_dft_filtered("rVV10")
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"rVV10 DFT unavailable: {exc}")
        d_r_ab, e_r_ab = _clip_d_e(d_r_ab, e_r_ab, lo, hi)
        d_r_aa, e_r_aa = _clip_d_e(d_r_aa, e_r_aa, lo, hi)
        if d_r_ab.size < 2:
            pytest.skip(
                f"Too few unstrained rVV10 AB points in [{lo}, {hi}] Å after filter "
                f"(got {d_r_ab.size}).",
            )

    d_m_ab = d_m_aa = np.array([])
    e_m_ab = e_m_aa = np.array([])
    if "MBD" in expanded:
        try:
            d_m_ab, e_m_ab, d_m_aa, e_m_aa = _load_dft_filtered("MBD")
        except (FileNotFoundError, OSError, ValueError) as exc:
            pytest.skip(f"MBD DFT unavailable: {exc}")
        d_m_ab, e_m_ab = _clip_d_e(d_m_ab, e_m_ab, lo, hi)
        d_m_aa, e_m_aa = _clip_d_e(d_m_aa, e_m_aa, lo, hi)
        if d_m_ab.size < 2:
            pytest.skip(
                f"Too few unstrained MBD AB points in [{lo}, {hi}] Å after filter "
                f"(got {d_m_ab.size}).",
            )

    e_r_ab_pa = e_r_ab / n_at if d_r_ab.size else np.array([])
    e_r_aa_pa = e_r_aa / n_at if d_r_aa.size else np.array([])
    e_m_ab_pa = e_m_ab / n_at if d_m_ab.size else np.array([])
    e_m_aa_pa = e_m_aa / n_at if d_m_aa.size else np.array([])

    span = hi - lo
    d_model = np.linspace(lo, hi, max(48, int(span / 0.025)))

    if want_models:
        # Only require LAMMPS for models that actually need it.
        if any(m in tr._MODELS_NEEDING_LAMMPS for m in want_models):
            try:
                import lammps  # noqa: F401
            except Exception as exc:
                pytest.skip(
                    f"LAMMPS Python module not available for model energies: {exc}"
                )
        tr._ensure_get_mcmc_inputs_importable()

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    n_curves = len(expanded)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(8, n_curves)))

    for i, source in enumerate(expanded):
        c = colors[i]
        if source == "qmc":
            _plot_source(ax, "QMC", c, d_q_ab, e_q_ab, d_q_aa, e_q_aa)
        elif source == "rVV10":
            _plot_source(ax, "rVV10 DFT", c, d_r_ab, e_r_ab_pa, d_r_aa, e_r_aa_pa)
        elif source == "MBD":
            _plot_source(ax, "MBD DFT", c, d_m_ab, e_m_ab_pa, d_m_aa, e_m_aa_pa)
        elif source in model_set:
            try:
                label, e_ab_m, e_aa_m = _model_curves(source, d_model)
            except (FileNotFoundError, ValueError, OSError) as exc:
                pytest.skip(f"Model {source!r} unavailable: {exc}")
            _plot_source(ax, label, c, d_model, e_ab_m, d_model, e_aa_m)
        else:
            pytest.fail(f"Internal error: unexpected source {source!r}")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"Interlayer separation $d$ (Å)")
    ax.set_ylabel(r"Energy / atom minus min(AB) (eV)")
    ax.set_title(
        f"BLG interlayer PES — AB & AA, $d \\in [{lo:g}, {hi:g}]$ Å "
        "(shifted by each source's AB minimum)",
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out = _artifacts_dir() / "compare_qmc_AB_AA.png"
    fig.savefig(out)
    plt.close(fig)

    assert out.is_file() and out.stat().st_size > 5000
    print(f"[test_compare_qmc] Wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
