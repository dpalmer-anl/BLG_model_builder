"""
Potential-energy-style plots for bilayer graphene.

**rVV10** geometries and DFT totals come from :func:`load_energy_data` (``rVV10`` only).

**QMC** is read only from ``data/qmc.csv`` (not :mod:`DataLoader`); that file already
labels stackings and interlayer separations, and ``energy`` is **eV per atom**
(used as-is).

Plot 1 (rVV10, AB): in-plane strain is ``(lx - lx_min) / lx_min`` where ``lx`` is the
x Cartesian component of the first Bravais vector ``cell[0, 0]`` (Å; in-plane lattice
constant ``a`` for the usual orientation) and ``lx_min`` is that ``lx`` on the **lowest
total-energy** rVV10 configuration. rVV10 y-values are ``(E_{\mathrm{tot}} -
\min(\texttt{energies})) / N`` using the **raw** total energies returned by the loader,
with ``\min(\texttt{energies}) = \texttt{np.min(energies)}`` over the whole rVV10 set.
QMC still subtracts ``\min`` of the ``energy`` column (eV/atom) over all CSV rows.
One offset per method, no per-curve shifts.

Stacking rule: **AA** if every top atom has a bottom partner with nearly the same
``(x, y)`` (max over tops of MIC min xy distance below a small cutoff); **AB** otherwise.

Run from ``uncertainty_quantification`` (or anywhere with ``pip install -e ..``)::

    python plot_bilayer_graphene_pes.py

Also requires ``matplotlib``, ``pandas``, and ``scipy`` (cubic spline interpolation).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from blg_model_builder.DataLoader import load_energy_data

# AA: all top atoms lie (in xy) on a bottom site within this tolerance (Å).
_STACK_AA_MAX_TOP_BOTTOM_XY = 0.28

_STRAIN_NEAR_ZERO = 0.008
_STRAIN_TARGET = 0.02
_STRAIN_BAND = 0.005

# Plot 2: colour = electronic-structure method, marker = stacking
_COLOR_RVV10 = "#1f77b4"
_COLOR_QMC = "#d95f02"
_MARKER_AB = "o"
_MARKER_AA = "s"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _chdir_for_dataloader() -> None:
    root = _repo_root()
    uq = root / "uncertainty_quantification"
    os.chdir(str(uq if uq.is_dir() else root))


def interlayer_separation(atoms) -> float:
    z = atoms.positions[:, 2]
    zmid = float(np.median(z))
    bot = z < zmid
    top = ~bot
    return float(np.mean(z[top]) - np.mean(z[bot]))


def inplane_lx_cell0(atoms) -> float:
    """``cell[0, 0]`` (Å): x component of the first lattice vector (``lx``)."""
    return float(np.asarray(atoms.cell[0], dtype=float)[0])


def reference_lx_min_from_min_energy(atoms_list: list, energies: np.ndarray) -> float:
    """``lx`` on the lowest-total-energy rVV10 frame (same definition as :func:`inplane_lx_cell0`)."""
    i0 = int(np.argmin(np.asarray(energies, dtype=float)))
    return inplane_lx_cell0(atoms_list[i0])


def strain_from_lx(atoms, lx_min: float) -> float:
    """``(lx - lx_min) / lx_min`` with ``lx`` from :func:`inplane_lx_cell0`."""
    lx = inplane_lx_cell0(atoms)
    return (lx - float(lx_min)) / float(lx_min)


def load_qmc_interlayer_csv() -> pd.DataFrame:
    """Stacking, ``d`` (Å), and **eV/atom** from ``data/qmc.csv`` (no DataLoader)."""
    path = _repo_root() / "data" / "qmc.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def per_top_min_xy_mic_to_bottom(atoms) -> np.ndarray:
    """
    For each top-layer atom, minimum in-plane MIC distance (Å) to any bottom-layer atom.

    AA: every top is stacked — all values (and thus their max) are ~0.
    AB (Bernal): at least one top sits over hollow — that top's minimum stays O(1 Å),
    so the **maximum** over tops separates AA from AB (unlike the global minimum
    over all pairs, which is ~0 for both).
    """
    pos = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    z = pos[:, 2]
    zmid = np.median(z)
    itop = np.where(z >= zmid)[0]
    ibot = np.where(z < zmid)[0]
    if len(itop) == 0 or len(ibot) == 0:
        return np.array([])
    inv = np.linalg.inv(cell.T)
    fcoords = (inv @ pos.T).T
    t_cell = cell[:2, :2]
    mins: list[float] = []
    for i in itop:
        fi = fcoords[i, :2]
        fj = fcoords[ibot, :2]
        df = fj - fi[None, :]
        df -= np.round(df)
        dxy = df @ t_cell
        mins.append(float(np.min(np.linalg.norm(dxy, axis=1))))
    return np.asarray(mins, dtype=float)


def bilayer_stacking_is_aa(atoms) -> bool:
    mins = per_top_min_xy_mic_to_bottom(atoms)
    if mins.size == 0:
        return False
    return bool(np.max(mins) < _STACK_AA_MAX_TOP_BOTTOM_XY)


def bilayer_stacking_is_ab(atoms) -> bool:
    return not bilayer_stacking_is_aa(atoms)


def energy_per_atom(atoms, energy_total: float) -> float:
    return float(energy_total) / len(atoms)


def rvv10_minimum_total_energy(energies: np.ndarray) -> float:
    """``np.min(energies)`` — minimum **total** energy in the loaded rVV10 array."""
    return float(np.min(np.asarray(energies, dtype=float)))


def rvv10_shifted_energy_per_atom(
    atoms,
    energy_total: float,
    e_min_total: float,
) -> float:
    """``(E_tot - min(E_tot from dataset)) / N`` for plotting vs interlayer distance."""
    return (float(energy_total) - float(e_min_total)) / len(atoms)


def global_min_energy_per_atom_qmc(df: pd.DataFrame) -> float:
    """Minimum ``energy`` over **all** rows of ``qmc.csv`` (eV/atom)."""
    col = df["energy"].astype(float)
    return float(col.min())


def _merge_same_d(d_list: list, e_list: list) -> tuple[np.ndarray, np.ndarray]:
    """
    One point per separation bin. For duplicate ``d`` (rounded), keep **minimum**
    energy so a global minimum at ``(E - \min E_\mathrm{tot})/N = 0`` is not pulled
    upward by averaging with higher-energy structures at similar interlayer spacing.
    """
    if not d_list:
        return np.array([]), np.array([])
    buckets: OrderedDict[float, list] = OrderedDict()
    for di, ei in zip(d_list, e_list):
        key = round(float(di), 2)
        buckets.setdefault(key, []).append(float(ei))
    keys = sorted(buckets.keys())
    d_out = np.asarray(keys, dtype=float)
    e_out = np.asarray([min(buckets[k]) for k in keys], dtype=float)
    return d_out, e_out


def _cubic_spline_fine(
    d: np.ndarray,
    e: np.ndarray,
    n: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort by interlayer separation and evaluate a cubic spline on a dense mesh.

    Returns ``(layer_sep_spline, energy_spline)`` for plotting / argmin.
    """
    d = np.asarray(d, dtype=float)
    e = np.asarray(e, dtype=float)
    if d.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(d)
    d_s, e_s = d[order], e[order]
    if d_s.size == 1:
        return d_s.copy(), e_s.copy()
    # CubicSpline requires strictly increasing x
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
    e_fine = cs(d_fine)
    return d_fine, e_fine


def _plot_spline_with_markers(
    ax: plt.Axes,
    d: np.ndarray,
    e: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
    linewidth: float = 2.0,
    markersize: float = 7,
) -> None:
    """Draw a cubic spline through ``(d, e)`` and overlay discrete markers."""
    if d.size == 0:
        return
    d_fine, e_fine = _cubic_spline_fine(d, e)
    order = np.argsort(np.asarray(d, dtype=float))
    d_s = np.asarray(d, dtype=float)[order]
    e_s = np.asarray(e, dtype=float)[order]
    ax.plot(d_fine, e_fine, linestyle="-", linewidth=linewidth, color=color, label=label)
    ax.plot(
        d_s,
        e_s,
        linestyle="none",
        marker=marker,
        markersize=markersize,
        color=color,
    )


def equilibrium_interlayer_from_cubic_spline(d: np.ndarray, e: np.ndarray) -> float | None:
    """``layer_sep_spline[np.argmin(energy_spline)]`` on a dense cubic-spline mesh."""
    layer_sep_spline, energy_spline = _cubic_spline_fine(d, e, n=2000)
    if layer_sep_spline.size == 0:
        return None
    i = int(np.argmin(energy_spline))
    return float(layer_sep_spline[i])


def get_merged_pes_curves(
    atoms_r: list,
    E_r: np.ndarray,
    lx_min: float,
    e_min_total_rvv10: float,
    e_min_qmc: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Merged ``(d, E)`` with energies shifted to the method global minimum (eV/atom),
    same as :func:`plot2_aa_ab_comparison`.
    """
    (ab_d, ab_e), (aa_d, aa_e) = collect_rvv10_stacking_unstrained(
        atoms_r, E_r, lx_min, e_min_total_rvv10
    )
    dab, eab = _merge_same_d(ab_d, ab_e)
    daa, eaa = _merge_same_d(aa_d, aa_e)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "rVV10 AB": (dab, eab),
        "rVV10 AA": (daa, eaa),
    }
    df = load_qmc_interlayer_csv()
    for stacking, key in (("AB", "QMC AB"), ("AA", "QMC AA")):
        sub = df[df["stacking"].str.upper() == stacking]
        if sub.empty:
            out[key] = (np.array([]), np.array([]))
            continue
        d_u, e_u = _merge_same_d(
            sub["d"].values.tolist(),
            sub["energy"].values.astype(float).tolist(),
        )
        order = np.argsort(d_u)
        out[key] = (d_u[order], e_u[order] - e_min_qmc)
    return out


def _print_rvv10_ab_strain_diagnostics(
    atoms_list: list,
    energies: np.ndarray,
    lx_min: float,
) -> None:
    strains: list[float] = []
    lxs: list[float] = []
    for atoms, e_tot in zip(atoms_list, energies):
        if not bilayer_stacking_is_ab(atoms):
            continue
        lxs.append(inplane_lx_cell0(atoms))
        strains.append(strain_from_lx(atoms, lx_min))
    if not strains:
        print("Strain diagnostic: no AB-stacked rVV10 frames.")
        return
    s = np.asarray(strains, dtype=float)
    x = np.asarray(lxs, dtype=float)
    i_min = int(np.argmin(np.asarray(energies, dtype=float)))
    print(
        f"rVV10 strain reference: lx_min = {lx_min:.6f} Å "
        f"(cell[0,0] at global min-E frame, index {i_min})"
    )
    print(f"AB-stacked rVV10 frames: {len(strains)}")
    print(f"  lx (Å): min={x.min():.6f}  max={x.max():.6f}")
    print(f"  strain ε = (lx - lx_min)/lx_min: min={s.min():.6f}  max={s.max():.6f}")
    uniq = np.unique(np.round(s, 2))
    print(f"  unique ε (round 1e-2): {uniq}")


def _strain_bucket(eps: float, strain_target: float, strain_band: float) -> str | None:
    """
    Mutually exclusive bins: ``unstrained`` vs ``strained`` (no frame in both).
    Strained window is tensile only (eps > 0) near ``strain_target``.
    """
    if abs(eps) <= _STRAIN_NEAR_ZERO:
        return "unstrained"
    lo = strain_target - strain_band
    hi = strain_target + strain_band
    if lo <= eps <= hi and abs(eps) > _STRAIN_NEAR_ZERO:
        return "strained"
    return None


def plot1_rvv10_ab_strain(
    atoms_list: list,
    energies: np.ndarray,
    ax: plt.Axes,
    lx_min: float,
    e_min_total_rvv10: float,
    strain_target: float = _STRAIN_TARGET,
    strain_band: float = _STRAIN_BAND,
) -> None:
    _print_rvv10_ab_strain_diagnostics(atoms_list, energies, lx_min)

    unstrained_d, unstrained_e = [], []
    strained_d, strained_e = [], []
    for atoms, e_tot in zip(atoms_list, energies):
        if not bilayer_stacking_is_ab(atoms):
            continue
        d = interlayer_separation(atoms)
        eps = strain_from_lx(atoms, lx_min)
        epa = rvv10_shifted_energy_per_atom(atoms, e_tot, e_min_total_rvv10)
        bucket = _strain_bucket(eps, strain_target, strain_band)
        if bucket == "unstrained":
            unstrained_d.append(d)
            unstrained_e.append(epa)
        elif bucket == "strained":
            strained_d.append(d)
            strained_e.append(epa)

    du, eu = _merge_same_d(unstrained_d, unstrained_e)
    ds, es = _merge_same_d(strained_d, strained_e)

    if du.size:
        _plot_spline_with_markers(
            ax,
            du,
            eu,
            color="#2166ac",
            marker=_MARKER_AB,
            label="AB, ~unstrained (rVV10)",
        )
    else:
        ax.text(0.5, 0.55, "No unstrained AB rVV10 points", ha="center", va="center", transform=ax.transAxes)

    if ds.size:
        _plot_spline_with_markers(
            ax,
            ds,
            es,
            color="#b2182b",
            marker=_MARKER_AA,
            label=f"AB, ~{100 * strain_target:.0f}% tensile εxx (rVV10)",
        )
    else:
        ax.text(
            0.5,
            0.4,
            f"No AB points in +{100 * strain_target:.0f}% strain band "
            f"[{100 * (strain_target - strain_band):.1f}%, "
            f"{100 * (strain_target + strain_band):.1f}%]; adjust _STRAIN_* constants.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )

    ax.set_xlabel("Interlayer separation (Å)")
    ax.set_ylabel("Energy (eV/atom)")
    ax.set_title("rVV10 — strained AB bilayer graphene")
    ax.set_xlim(3, 4.5)
    ax.legend()
    ax.grid(True, alpha=0.35)


def collect_rvv10_stacking_unstrained(
    all_atoms,
    all_E,
    lx_min: float,
    e_min_total_rvv10: float,
) -> tuple[tuple[list, list], tuple[list, list]]:
    ab_d, ab_e = [], []
    aa_d, aa_e = [], []
    for atoms, e_tot in zip(all_atoms, all_E):
        eps = strain_from_lx(atoms, lx_min)
        if abs(eps) >= _STRAIN_NEAR_ZERO:
            continue
        d = interlayer_separation(atoms)
        epa = rvv10_shifted_energy_per_atom(atoms, e_tot, e_min_total_rvv10)
        if bilayer_stacking_is_aa(atoms):
            aa_d.append(d)
            aa_e.append(epa)
        elif bilayer_stacking_is_ab(atoms):
            ab_d.append(d)
            ab_e.append(epa)
    return (ab_d, ab_e), (aa_d, aa_e)


def plot2_aa_ab_comparison(
    ax: plt.Axes,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Plot merged PES curves (cubic splines + markers); keys from :func:`get_merged_pes_curves`."""
    plot_order: tuple[tuple[str, str, str, str], ...] = (
        ("rVV10 AB", _COLOR_RVV10, _MARKER_AB, "AB (rVV10)"),
        ("rVV10 AA", _COLOR_RVV10, _MARKER_AA, "AA (rVV10)"),
        ("QMC AB", _COLOR_QMC, _MARKER_AB, "AB (QMC)"),
        ("QMC AA", _COLOR_QMC, _MARKER_AA, "AA (QMC)"),
    )
    for key, color, marker, legend_label in plot_order:
        d, e = curves[key]
        _plot_spline_with_markers(
            ax,
            d,
            e,
            color=color,
            marker=marker,
            label=legend_label,
        )

    ax.set_xlabel("Interlayer separation (Å)")
    ax.set_ylabel("Energy (eV/atom)")
    ax.set_title("Unstrained bilayer graphene")
    ax.set_xlim(3, 4.5)
    ax.legend()
    ax.grid(True, alpha=0.35)


def main() -> None:
    _chdir_for_dataloader()
    supercells = 1
    atoms_list, energies, _ = load_energy_data(
        "interlayer", supercells, level_of_theory="rVV10"
    )
    lx_min = reference_lx_min_from_min_energy(atoms_list, energies)
    e_min_total_rvv10 = rvv10_minimum_total_energy(energies)
    shifted_all_ab = [
        rvv10_shifted_energy_per_atom(a, e, e_min_total_rvv10)
        for a, e in zip(atoms_list, energies)
        if bilayer_stacking_is_ab(a)
    ]
    if shifted_all_ab:
        print(
            f"rVV10 min shifted E/N over AB frames (pre-merge) = {min(shifted_all_ab):.6f} eV/atom "
            "(should be 0 if global minimum is AB)."
        )
    df_qmc = load_qmc_interlayer_csv()
    e_min_qmc = global_min_energy_per_atom_qmc(df_qmc)
    print(
        f"rVV10 min total energy np.min(energies) = {e_min_total_rvv10:.6f} eV "
        f"(shift; plotted as (E_tot - min) / N)."
    )
    print(
        f"QMC global min energy (subtracted on QMC curves) = {e_min_qmc:.6f} eV/atom."
    )

    curves = get_merged_pes_curves(
        atoms_list,
        energies,
        lx_min,
        e_min_total_rvv10,
        e_min_qmc,
    )
    print("Equilibrium interlayer separation (cubic spline argmin of shifted energy):")
    for name in ("rVV10 AB", "rVV10 AA", "QMC AB", "QMC AA"):
        d_arr, e_arr = curves[name]
        d_eq = equilibrium_interlayer_from_cubic_spline(d_arr, e_arr)
        if d_eq is None:
            print(f"  {name}: (no points)")
        else:
            print(f"  {name}: {d_eq:.6f} Å")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    plot1_rvv10_ab_strain(
        atoms_list,
        energies,
        ax1,
        lx_min=lx_min,
        e_min_total_rvv10=e_min_total_rvv10,
    )
    plot2_aa_ab_comparison(ax2, curves)
    fig.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bilayer_graphene_pes.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
