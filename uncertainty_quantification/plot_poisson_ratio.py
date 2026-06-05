#!/usr/bin/env python3
"""
Poisson ratio calculations for bilayer graphene from DFT (rVV10) data.

Data source
-----------
strained_bilayer_graphene_rVV10.xyz
    DFT total energies on a grid of (dx, dy, sep, stacking), where
    dx = (a1 - a0)/a0 and dy = (|a2| - a0)/a0 are lattice-vector strains
    and sep is the interlayer separation.  The grid satisfies dx >= dy.
    Stackings: AB, AA, SP.

    Frames from MD trajectories are dropped in :func:`load_strained_data` when
    atoms in the same layer do not share a common Cartesian *z* (within
    :data:`LAYER_Z_FLAT_TOL`).

Method
------
For each stacking:

1. Extract (dx, dy, sep, E) from the DFT frames.

2. Build a 3D cubic-spline interpolant E(dx, dy, sep) from the scattered
   data using RBFInterpolator (cubic kernel).  Evaluate on a regular
   N x N x N mesh (default N = 50).

3. From the mesh:

   nu_xz  (out-of-plane, x-loading)
     Fix dy = 0.  For each dx value on the mesh, find the sep index
     that minimises E -> sep*(dx).
     eps_z(dx) = (sep*(dx) - sep0) / sep0
     nu_xz = -d(eps_z)/d(dx) from a linear fit.

   nu_yz  (out-of-plane, y-loading)
     Fix dx = 0.  For each dy value on the mesh, find the sep index
     that minimises E -> sep*(dy).
     nu_yz = -d(eps_z)/d(dy) from a linear fit.

   nu_xy  (in-plane)
     Fix sep = sep0 (equilibrium separation at zero strain).
     For each dx value on the mesh, find the dy index that minimises E
     -> dy*(dx).
     nu_xy = -d(dy*)/d(dx) from a linear fit.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import RegularGridInterpolator
import ase.io

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
STRAINED_XYZ = os.path.join(HERE, "../data/strained_bilayer_graphene_rVV10.xyz")

LAT_CON  = 2.46
STACKINGS = ["AB", "AA", "SP"]
STACKING_COLORS  = {"AB": "C0", "AA": "C1", "SP": "C2"}
STACKING_MARKERS = {"AB": "o",  "AA": "s",  "SP": "^"}

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
# Strain range (±) used when evaluating the mesh slices and computing the
# Poisson ratios.  The RBF is always trained on the full data range, but the
# argmin is taken only within [-STRAIN_RANGE, +STRAIN_RANGE].
# Training data has points at 0 %, ±1 %, ±2 %; the RBF interpolates freely
# between them, so any value ≤ 0.02 is valid.
STRAIN_RANGE = 0.005   # 0.5 % — change this to adjust the evaluated strain window
LAYER_Z_FLAT_TOL = 1e-6  # Å; max |z_i - z_j| within a layer for static DFT frames

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def identify_stacking(atoms):
    """
    Identify stacking (AB, AA, SP) using the fractional (scaled) centre-of-mass
    shift between layers, which is invariant under the cell straining applied
    during data generation (scale_atoms=True).

    For the hex primitive cell a1=[a,0,0], a2=[a/2, a*sqrt(3)/2, 0] the
    high-symmetry inter-layer centre-of-mass shifts in fractional coords are:
        AA  : (0,   0  )  — no shift
        AB  : (1/3, 1/3)  — Bernal stacking
        SP  : (1/2, 0  )  — saddle-point / bridge stacking
    Using the centre-of-mass (mean fractional xy per layer) avoids the
    atom-pairing ambiguity that occurs with sorted atom-by-atom approaches
    when the SP shift reorders atoms by fractional x.
    """
    pos = atoms.get_scaled_positions()
    n   = len(pos)

    # Separate layers by fractional z
    idx    = np.argsort(pos[:, 2])
    bot_xy = pos[idx[:n//2], :2]
    top_xy = pos[idx[n//2:], :2]

    # Minimum-image centre-of-mass shift in fractional coords → [-0.5, 0.5)
    shift  = np.mean(top_xy, axis=0) - np.mean(bot_xy, axis=0)
    shift  = (shift + 0.5) % 1.0 - 0.5
    sx, sy = np.abs(shift)

    # AA: no shift; SP: one component ~0.5; AB: both ~1/3
    if sx < 0.12 and sy < 0.12:
        return "AA"
    if max(sx, sy) > 0.40:           # one component near ±0.5
        return "SP"
    return "AB"


def interlayer_sep(atoms):
    pos = atoms.get_positions()
    n   = len(pos)
    idx = np.argsort(pos[:, 2])
    return np.mean(pos[idx[n//2:], 2]) - np.mean(pos[idx[:n//2], 2])


def layers_have_uniform_z(atoms, tol: float = LAYER_Z_FLAT_TOL) -> bool:
    """
    True if each layer is planar in Cartesian *z* (static DFT), within ``tol``.

    MD snapshots have thermal / relaxation spread in *z* within a layer; those
    frames are rejected when loading the strained dataset.
    """
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
    dx   = (cell[0, 0]                    - LAT_CON) / LAT_CON
    dy   = (np.linalg.norm(cell[1, :2])   - LAT_CON) / LAT_CON
    return dx, dy

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_strained_data(z_flat_tol: float = LAYER_Z_FLAT_TOL):
    """
    Returns dict: stacking -> ndarray (N, 4) = [dx, dy, sep, E_per_atom]

    Skips MD-like frames where atoms in the same layer have differing Cartesian
    z (spread > ``z_flat_tol`` Å).
    """
    frames  = ase.io.read(STRAINED_XYZ, index=":")
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
        E   = atoms.get_potential_energy() / len(atoms)
        records[stack].append((dx, dy, sep, E))

    if n_skip_md:
        print(
            f"  load_strained_data: skipped {n_skip_md} MD-like frame(s) "
            f"(layer z spread > {z_flat_tol:g} Å)",
            flush=True,
        )

    return {s: np.array(v) for s, v in records.items() if v}

# ---------------------------------------------------------------------------
# Mesh construction and Poisson ratio extraction
# ---------------------------------------------------------------------------

def build_rbf(data):
    """
    Fit a cubic RBF interpolant to scattered (dx, dy, sep, E) data.
    Returns the fitted RBFInterpolator and the axis ranges.
    """
    X   = data[:, :3]
    E   = data[:,  3]
    rbf = RBFInterpolator(X, E, kernel="linear", degree=3,smoothing=0)

    dx_range  = (data[:, 0].min(), data[:, 0].max())
    dy_range  = (data[:, 1].min(), data[:, 1].max())
    sep_range = (3.3,4.5)

    return rbf, dx_range, dy_range, sep_range


def eval_slice_xz(rbf, strain_range, sep_range, n_strain=50, n_sep=500):
    """
    Evaluate E on a 2-D (dx × sep) slice at dy = 0.

    strain_range : (float) half-width of the strain axis, e.g. 0.005 for ±0.5 %.
    n_sep        : number of sep grid points; ~500 gives ≈ 0.004 Å resolution
                   which, combined with parabolic sub-grid refinement, is enough
                   to resolve the small interlayer shifts from in-plane strain.

    Returns dx_g, sep_g, E_slice (n_strain × n_sep).
    """
    dx_g  = np.linspace(-strain_range, strain_range, n_strain)
    sep_g = np.linspace(*sep_range, n_sep)
    DX, SEP = np.meshgrid(dx_g, sep_g, indexing="ij")
    N = n_strain * n_sep
    pts = np.column_stack([DX.ravel(), np.zeros(N), SEP.ravel()])

    return dx_g, sep_g, rbf(pts).reshape(n_strain, n_sep)


def eval_slice_yz(rbf, strain_range, sep_range, n_strain=50, n_sep=500):
    """
    Evaluate E on a 2-D (dy × sep) slice at dx = 0.
    Returns dy_g, sep_g, E_slice (n_strain × n_sep).
    """
    dy_g  = np.linspace(-strain_range, strain_range, n_strain)
    sep_g = np.linspace(*sep_range, n_sep)
    DY, SEP = np.meshgrid(dy_g, sep_g, indexing="ij")
    N = n_strain * n_sep
    pts = np.column_stack([np.zeros(N), DY.ravel(), SEP.ravel()])
    return dy_g, sep_g, rbf(pts).reshape(n_strain, n_sep)


def eval_slice_xy(rbf, strain_range, sep0, n=50):
    """
    Evaluate E on a 2-D (dx × dy) slice at sep = sep0.
    Returns dx_g, dy_g, E_slice (n × n).
    """
    dx_g = np.linspace(-strain_range, strain_range, n)
    dy_g = np.linspace(-strain_range, strain_range, n)
    DX, DY = np.meshgrid(dx_g, dy_g, indexing="ij")
    pts = np.column_stack([DX.ravel(), DY.ravel(), np.full(n*n, sep0)])
    return dx_g, dy_g, rbf(pts).reshape(n, n)


def equilibrium_sep(rbf, sep_range, n=200):
    """
    Find sep0 = argmin E(dx=0, dy=0, sep) on a fine 1-D scan.
    """
    sep_g = np.linspace(*sep_range, n)
    pts   = np.column_stack([np.zeros(n), np.zeros(n), sep_g])
    E     = rbf(pts)
    return sep_g[np.argmin(E)]


def _parabolic_min(grid, E_1d, k):
    """
    Sub-grid minimum via parabolic interpolation through grid[k-1:k+2].
    Falls back to grid[k] at boundaries or when curvature is non-positive.
    """
    if k == 0 or k == len(grid) - 1:
        return grid[k]
    h  = grid[1] - grid[0]          # assumes uniform spacing
    E0, E1, E2 = E_1d[k-1], E_1d[k], E_1d[k+1]
    denom = E0 - 2*E1 + E2
    if denom <= 0:                   # not a true minimum
        return grid[k]
    return grid[k] - 0.5 * h * (E2 - E0) / denom


def poisson_nu_xz(dx_g, sep_g, E_xz, sep0):
    """
    From the (dx × sep) slice at dy=0:
    for each dx find the sep that minimises E (grid argmin + parabolic
    sub-grid refinement) -> sep*(dx).
    nu_xz = -d(eps_z)/d(dx).
    """
    sep_star = np.array([
        _parabolic_min(sep_g, E_xz[i, :], np.argmin(E_xz[i, :]))
        for i in range(len(dx_g))
    ])
    eps_z = (sep_star - sep0) / sep0
    nz    = np.abs(dx_g) > 1e-5
    slope = np.polyfit(dx_g[nz], eps_z[nz], 1)[0]
    return -slope, sep_star, eps_z


def poisson_nu_yz(dy_g, sep_g, E_yz, sep0):
    """
    From the (dy × sep) slice at dx=0:
    for each dy find the sep that minimises E -> sep*(dy).
    nu_yz = -d(eps_z)/d(dy).
    """
    sep_star = np.array([
        _parabolic_min(sep_g, E_yz[j, :], np.argmin(E_yz[j, :]))
        for j in range(len(dy_g))
    ])
    eps_z = (sep_star - sep0) / sep0
    nz    = np.abs(dy_g) > 1e-5
    slope = np.polyfit(dy_g[nz], eps_z[nz], 1)[0]
    return -slope, sep_star, eps_z


def poisson_nu_xy(dx_g, dy_g, E_xy):
    """
    From the (dx × dy) slice at sep=sep0:
    for each dx find the dy that minimises E (grid argmin + parabolic
    sub-grid refinement) -> dy*(dx).
    nu_xy = -d(dy*)/d(dx).
    """
    dy_star = np.array([
        _parabolic_min(dy_g, E_xy[i, :], np.argmin(E_xy[i, :]))
        for i in range(len(dx_g))
    ])
    nz    = np.abs(dx_g) > 1e-5
    slope = np.polyfit(dx_g[nz], dy_star[nz], 1)[0]
    return -slope, dy_star


def analyze_stacking(stacking, data, n_grid=50, strain_range=STRAIN_RANGE):
    """
    Full Poisson ratio analysis for one stacking using the mesh approach.
    Each Poisson ratio uses a 2-D slice of the 3-D interpolant evaluated
    over [-strain_range, +strain_range] in the strain axes.
    """
    print(f"\nFitting RBF interpolant for {stacking} ({len(data)} points) ...")
    rbf, dx_range, dy_range, sep_range = build_rbf(data)

    # Equilibrium sep at zero strain
    sep0 = equilibrium_sep(rbf, sep_range)

    n_sep = 500   # fine sep grid; parabolic refinement recovers sub-grid accuracy
    print(f"  Evaluating slices over ±{strain_range*100:.2f}% strain "
          f"({n_grid}×{n_sep} for xz/yz, {n_grid}×{n_grid} for xy) ...")
    dx_g,  sep_g, E_xz = eval_slice_xz(rbf, strain_range, sep_range, n_grid, n_sep)
    dy_g,  _,     E_yz = eval_slice_yz(rbf, strain_range, sep_range, n_grid, n_sep)
    dx_g2, dy_g2, E_xy = eval_slice_xy(rbf, strain_range, sep0,      n_grid)

    nu_xz, sep_xz, eps_z_xz = poisson_nu_xz(dx_g,  sep_g, E_xz, sep0)
    nu_yz, sep_yz, eps_z_yz = poisson_nu_yz(dy_g,  sep_g, E_yz, sep0)
    nu_xy, dy_xy            = poisson_nu_xy(dx_g2, dy_g2, E_xy)

    print(f"  sep0 = {sep0:.4f} Å")
    print(f"  nu_xz (out-of-plane, x-loading, dy=0)  = {nu_xz:.4f}")
    print(f"  nu_yz (out-of-plane, y-loading, dx=0)  = {nu_yz:.4f}")
    print(f"  nu_xy (in-plane,     sep=sep0)          = {nu_xy:.4f}")

    return {
        "stacking": stacking,
        "sep0":    sep0,
        "dx_g":    dx_g,  "sep_g":  sep_g,  "E_xz": E_xz,
        "dy_g":    dy_g,  "E_yz":   E_yz,
        "dx_g2":   dx_g2, "dy_g2":  dy_g2,  "E_xy": E_xy,
        "nu_xz":   nu_xz, "sep_xz": sep_xz, "eps_z_xz": eps_z_xz,
        "nu_yz":   nu_yz, "sep_yz": sep_yz, "eps_z_yz": eps_z_yz,
        "nu_xy":   nu_xy, "dy_xy":  dy_xy,
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(results, save_dir="figures"):
    """
    3-row × 3-column figure (one column per stacking).
      Row 0: nu_xz  — sep*(dx) at dy=0
      Row 1: nu_yz  — sep*(dy) at dx=0
      Row 2: nu_xy  — dy*(dx) at sep=sep0
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Bilayer Graphene Poisson Ratios — rVV10 DFT (mesh method)", fontsize=13)

    for col, stacking in enumerate(STACKINGS):
        if stacking not in results:
            continue
        res   = results[stacking]
        color = STACKING_COLORS[stacking]
        mark  = STACKING_MARKERS[stacking]

        # ---- row 0: nu_xz ---------------------------------------------------
        ax   = axes[0, col]
        dx   = res["dx_g"]
        seps = res["sep_xz"]
        eps  = res["eps_z_xz"]
        nu   = res["nu_xz"]
        fit  = np.polyfit(dx[np.abs(dx) > 1e-5], eps[np.abs(dx) > 1e-5], 1)
        ax.scatter(dx*100, seps, color=color, marker=mark, s=25, zorder=3)
        ax.axhline(res["sep0"], color="gray", lw=0.8, ls="--",
                   label=f"sep₀ = {res['sep0']:.3f} Å")
        ax.set_ylabel("sep* (Å)")
        ax.set_xlabel(r"$\varepsilon_x$ (%)")
        ax.set_title(f"{stacking}:  $\\nu_{{xz}}$ = {nu:.4f}", fontsize=10)
        ax.legend(fontsize=7)

        ax2 = ax.twinx()
        ax2.plot(dx*100, np.polyval(fit, dx)*100, "-", color=color, lw=1.5,
                 label=rf"fit $\nu_{{xz}}$ = {nu:.3f}")
        ax2.set_ylabel(r"$\varepsilon_z$ (%)", fontsize=8)
        ax2.legend(fontsize=7, loc="upper left")

        # ---- row 1: nu_yz ---------------------------------------------------
        ax   = axes[1, col]
        dy   = res["dy_g"]
        seps = res["sep_yz"]
        eps  = res["eps_z_yz"]
        nu   = res["nu_yz"]
        fit  = np.polyfit(dy[np.abs(dy) > 1e-5], eps[np.abs(dy) > 1e-5], 1)
        ax.scatter(dy*100, seps, color=color, marker=mark, s=25, zorder=3)
        ax.axhline(res["sep0"], color="gray", lw=0.8, ls="--",
                   label=f"sep₀ = {res['sep0']:.3f} Å")
        ax.set_ylabel("sep* (Å)")
        ax.set_xlabel(r"$\varepsilon_y$ (%)")
        ax.set_title(f"{stacking}:  $\\nu_{{yz}}$ = {nu:.4f}", fontsize=10)
        ax.legend(fontsize=7)

        ax2 = ax.twinx()
        ax2.plot(dy*100, np.polyval(fit, dy)*100, "-", color=color, lw=1.5,
                 label=rf"fit $\nu_{{yz}}$ = {nu:.3f}")
        ax2.set_ylabel(r"$\varepsilon_z$ (%)", fontsize=8)
        ax2.legend(fontsize=7, loc="upper left")

        # ---- row 2: nu_xy ---------------------------------------------------
        ax  = axes[2, col]
        dx  = res["dx_g2"]
        dys = res["dy_xy"]
        nu  = res["nu_xy"]
        fit = np.polyfit(dx[np.abs(dx) > 1e-5], dys[np.abs(dx) > 1e-5], 1)
        ax.scatter(dx*100, dys*100, color=color, marker=mark, s=25,
                   zorder=3, label="mesh argmin")
        ax.plot(dx*100, np.polyval(fit, dx)*100, "-", color=color, lw=1.5,
                label=rf"$\nu_{{xy}}$ = {nu:.4f}")
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.axvline(0, color="gray", lw=0.7, ls=":")
        ax.set_ylabel(r"$\varepsilon_y^*$ (%)")
        ax.set_xlabel(r"$\varepsilon_x$ (%)")
        ax.set_title(f"{stacking}:  $\\nu_{{xy}}$ = {nu:.4f}", fontsize=10)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "poisson_ratios_dft.png")
    fig.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")
    plt.show()


def plot_energy_slices(results, stacking="AB", save_dir="figures"):
    """
    Show the 2-D energy slice surfaces used for each Poisson ratio,
    with the argmin path overlaid.
    """
    if stacking not in results:
        return
    res   = results[stacking]
    color = STACKING_COLORS[stacking]
    sep0  = res["sep0"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{stacking} stacking — energy slices used for Poisson ratios", fontsize=11)

    # Panel 0: nu_xz — E(dx, sep) at dy=0
    ax    = axes[0]
    dx_g  = res["dx_g"]
    sep_g = res["sep_g"]
    E_xz  = res["E_xz"]
    E_rel = E_xz - E_xz.min()
    im = ax.pcolormesh(dx_g*100, sep_g, E_rel.T, cmap="viridis", shading="auto")
    ax.plot(dx_g*100, res["sep_xz"], "w--", lw=1.5, label="argmin sep")
    ax.axhline(sep0, color="r", lw=1, ls=":", label=f"sep₀={sep0:.3f}")
    ax.set_xlabel(r"$\varepsilon_x$ (%)")
    ax.set_ylabel("sep (Å)")
    ax.set_title(rf"$\nu_{{xz}}$ slice: dy = 0")
    ax.legend(fontsize=7)
    fig.colorbar(im, ax=ax, label="E - E_min (eV/atom)")

    # Panel 1: nu_yz — E(dy, sep) at dx=0
    ax    = axes[1]
    dy_g  = res["dy_g"]
    E_yz  = res["E_yz"]
    E_rel = E_yz - E_yz.min()
    im = ax.pcolormesh(dy_g*100, sep_g, E_rel.T, cmap="viridis", shading="auto")
    ax.plot(dy_g*100, res["sep_yz"], "w--", lw=1.5, label="argmin sep")
    ax.axhline(sep0, color="r", lw=1, ls=":", label=f"sep₀={sep0:.3f}")
    ax.set_xlabel(r"$\varepsilon_y$ (%)")
    ax.set_ylabel("sep (Å)")
    ax.set_title(rf"$\nu_{{yz}}$ slice: dx = 0")
    ax.legend(fontsize=7)
    fig.colorbar(im, ax=ax, label="E - E_min (eV/atom)")

    # Panel 2: nu_xy — E(dx, dy) at sep=sep0
    ax    = axes[2]
    dx_g2 = res["dx_g2"]
    dy_g2 = res["dy_g2"]
    E_xy  = res["E_xy"]
    E_rel = E_xy - E_xy.min()
    im = ax.pcolormesh(dx_g2*100, dy_g2*100, E_rel.T, cmap="viridis", shading="auto")
    ax.plot(dx_g2*100, res["dy_xy"]*100, "w--", lw=1.5, label="argmin dy")
    ax.axhline(0, color="r", lw=1, ls=":")
    ax.axvline(0, color="r", lw=1, ls=":")
    ax.set_xlabel(r"$\varepsilon_x$ (%)")
    ax.set_ylabel(r"$\varepsilon_y$ (%)")
    ax.set_title(rf"$\nu_{{xy}}$ slice: sep = sep₀ = {sep0:.3f} Å")
    ax.legend(fontsize=7)
    fig.colorbar(im, ax=ax, label="E - E_min (eV/atom)")

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"energy_slices_{stacking}.png")
    fig.savefig(save_path, dpi=150)
    print(f"Energy slices saved to {save_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    w = 60
    print("\n" + "="*w)
    print(f"{'Stacking':^10} {'sep0 (Å)':^10} {'nu_xz':^10} {'nu_yz':^10} {'nu_xy':^10}")
    print("-"*w)
    for stacking in STACKINGS:
        r = results.get(stacking, {})
        sep0  = f"{r.get('sep0',  float('nan')):.4f}"
        nu_xz = f"{r.get('nu_xz', float('nan')):.4f}"
        nu_yz = f"{r.get('nu_yz', float('nan')):.4f}"
        nu_xy = f"{r.get('nu_xy', float('nan')):.4f}"
        print(f"{stacking:^10} {sep0:^10} {nu_xz:^10} {nu_yz:^10} {nu_xy:^10}")
    print("="*w)
    print("  nu_xz: -d(eps_z)/d(eps_x) at dy=0 (mesh argmin over sep)")
    print("  nu_yz: -d(eps_z)/d(eps_y) at dx=0 (mesh argmin over sep)")
    print("  nu_xy: -d(eps_y*)/d(eps_x) at sep=sep0 (mesh argmin over dy)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_GRID = 50          # mesh points per strain/sep axis; increase for finer grid

    print(f"Strain window for Poisson ratio: ±{STRAIN_RANGE*100:.2f}%  "
          f"(edit STRAIN_RANGE at the top of the file to change)")
    print("Loading strained bilayer graphene DFT data ...")
    strained_data = load_strained_data()
    for s, d in strained_data.items():
        print(f"  {s}: {len(d)} frames")

    results = {}
    for stacking in STACKINGS:
        if stacking not in strained_data:
            continue
        results[stacking] = analyze_stacking(stacking, strained_data[stacking],
                                             n_grid=N_GRID,
                                             strain_range=STRAIN_RANGE)

    print_summary(results)
    plot_all(results)
    for stacking in STACKINGS:
        plot_energy_slices(results, stacking)
