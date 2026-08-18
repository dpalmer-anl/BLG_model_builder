"""
Extract bilayer-graphene DFT elastic stiffness constants from strained energy curves.

Workflow
--------
1. Match DFT result structures to reference labels ``(stacking, mode, delta)``
   by geometry fingerprinting.
2. For each stacking and mode, fit total energies along the 1-D strain path
   ``delta`` with a quadratic polynomial relative to ``delta = 0``.
3. Convert the fitted curvature ``b2`` for each mode into Voigt stiffnesses
   ``C11, C12, C13, C33, C44`` (units eV/Å³) using the slab volume
   ``V = A d0``, then report GPa via ``1 eV/Å³ = 160.21766208 GPa``.

Elastic energy and volume
-------------------------
Strain energy density (Voigt notation):

    U = (1/2) C_ij ε_i ε_j

Total energy increment for a strained slab:

    ΔE = V U = (V/2) C_ij ε_i ε_j

Slab volume (Å³):

    V = A d0

where ``A = |a1 × a2|`` is the in-plane cell area from the unstrained reference
frame and ``d0`` is the equilibrium interlayer spacing for that stacking.

Quadratic fit per mode
----------------------
For each mode, DFT total energies ``E(delta)`` along the reference strain path
are shifted to ``ΔE(delta) = E(delta) - E(delta≈0)`` and fitted as

    ΔE(delta) ≈ b2 delta² + b1 delta + b0

near ``delta = 0``. The coefficient ``b2`` (eV) is the curvature that enters the
stiffness formulas below. (``np.polyfit`` returns this ``b2`` directly.)

Mode strain paths and stiffness formulas
--------------------------------------
Each mode applies a single scalar strain parameter ``delta`` along a fixed path.
With ``V = A d0`` and curvatures ``b2A … b2E`` from the fits:

Mode A — uniaxial in-plane (``ε_xx = delta``, all other ε = 0):

    ΔE = C11 V delta²
    C11 = b2A / V

Mode B — equibiaxial in-plane (``ε_xx = ε_yy = delta``):

    ΔE = 2 (C11 + C12) V delta²
    C12 = b2B / (2V) - C11

Mode C — uniaxial out-of-plane (``ε_zz = delta``):

    ΔE = (1/2) C33 V delta²
    C33 = 2 b2C / V

Mode D — coupled in-plane / out-of-plane (``ε_xx = ε_zz = delta``):

    When ``b2D ≈ b2A``, the difference in curvature isolates the cross term:

    C13 = (b2A - b2D) / V

Mode E — interlayer shear (``ε_yz = delta``; Voigt ``e4`` in ``gen_elastic_constant_structures.py``):

    ΔE = (3/2) C44 V delta²   (shear path in the training structures)
    C44 = 2 |b2E| / (3V)

Output units
------------
Stiffnesses are reported in eV/Å³ and converted to GPa:

    C_GPa = C_eV/Å³ × 160.21766208
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures" / "dft_elastic_fit_check"
REF_FILE = (
    "../../../DD-TETB/generate_data/Carbon_training_data/"
    "bilayer_graphene_elastic_constants_structures.xyz"
)
RESULT_FILE = "../../data/blg_elastic_constants_structures.xyz"

EV_A3_TO_GPA = 160.21766208
MODES = ("A", "B", "C", "D", "E")

# Strain tensor component(s) activated by each 1-D ``delta`` path.
MODE_STRAIN_PATH = {
    "A": "ε_xx = δ",
    "B": "ε_xx = ε_yy = δ",
    "C": "ε_zz = δ",
    "D": "ε_xx = ε_zz = δ",
    "E": "ε_yz = δ",
}

# Literature AB bilayer graphene DFT stiffnesses (GPa) for sanity check.
REFERENCE_GPA = {
    "C11": 1080.0,
    "C12": 162.0,
    "C13": -4.63,
    "C33": 33.13,
    "C44": 3.32,
}

def fingerprint(atoms):
    """
    A permutation- and order-invariant geometric fingerprint of a strained
    bilayer frame: in-plane lattice vector lengths + angle, interlayer
    separation, and relative interlayer lateral (shear) offset.
    Two frames representing the same (stacking, mode, delta) should have
    matching fingerprints regardless of atom/frame ordering.
    """
    cell = atoms.cell[:]
    a1, a2 = cell[0], cell[1]
    a1_len = np.linalg.norm(a1)
    a2_len = np.linalg.norm(a2)
    cos_ang = np.dot(a1, a2) / (a1_len * a2_len)
    ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    z = atoms.positions[:, 2]
    zmid = 0.5 * (z.min() + z.max())
    bottom = atoms[z < zmid]
    top = atoms[z >= zmid]
    interlayer_sep = top.positions[:, 2].mean() - bottom.positions[:, 2].mean()
    shear_y = top.positions[:, 1].mean() - bottom.positions[:, 1].mean()

    return np.array([a1_len, a2_len, ang, interlayer_sep, shear_y])


def cluster_reference_frames(ref_frames, tol=1e-4):
    """
    Group reference frames that are geometrically identical (this happens at
    delta=0, where all 5 modes reduce to the same structure -- e.g. mode A's
    and mode C's delta=0 frame are bit-identical). Returns a list of
    [representative_fingerprint, [list of (stacking, mode, delta, d0) labels]].
    """
    fps = np.array([fingerprint(a) for a in ref_frames])
    scale = fps.std(axis=0)
    scale[scale < 1e-8] = 1.0

    clusters = []   # list of [fp, [labels]]
    for a, fp in zip(ref_frames, fps):
        label = (a.info['stacking'], a.info['mode'], a.info['delta'], a.info['d0'])
        placed = False
        for c in clusters:
            if np.linalg.norm((fp - c[0]) / scale) < tol:
                c[1].append(label)
                placed = True
                break
        if not placed:
            clusters.append([fp, [label]])
    return clusters


def match_and_expand(result_frames, clusters, max_dist=1e-3):
    """
    For each DFT result frame, find its nearest reference cluster and emit
    ONE (stacking, mode, delta, d0, energy) record per label in that cluster
    -- this correctly duplicates a delta=0 result across all 5 modes instead
    of arbitrarily assigning it to just one.
    """
    rep_fps = np.array([c[0] for c in clusters])
    scale = rep_fps.std(axis=0)
    scale[scale < 1e-8] = 1.0

    records = []
    for a in result_frames:
        fp = fingerprint(a)
        dist = np.linalg.norm((rep_fps - fp) / scale, axis=1)
        idx = np.argmin(dist)
        if dist[idx] > max_dist:
            print(f"  WARNING: closest match has normalized distance {dist[idx]:.2e} "
                  f"(fingerprint={fp}) -- check this structure manually.")
        energy = a.get_potential_energy()
        for (stacking, mode, delta, d0) in clusters[idx][1]:
            records.append(dict(stacking=stacking, mode=mode, delta=delta, d0=d0, energy=energy))
    return records


def slab_volume(area: float, d0: float) -> float:
    """Slab volume V = A d0 (in-plane area × interlayer spacing, Å³)."""
    return float(area) * float(d0)


def quadratic_fit_delta_energy(
    deltas: np.ndarray,
    energies: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit ``ΔE = b2 δ² + b1 δ + b0`` relative to the point nearest ``δ=0``."""
    deltas = np.asarray(deltas, dtype=float)
    energies = np.asarray(energies, dtype=float)
    i0 = int(np.argmin(np.abs(deltas)))
    e_ref = float(energies[i0])
    de = energies - e_ref
    b2, b1, b0 = np.polyfit(deltas, de, 2)
    fit = np.polyval([b2, b1, b0], deltas)
    rms = float(np.sqrt(np.mean((fit - de) ** 2)))
    return float(b2), float(b1), float(b0 + e_ref), rms


def elastic_constants_from_modes(
    b2A: float,
    b2B: float,
    b2C: float,
    b2D: float,
    b2E: float,
    volume: float,
) -> dict[str, float]:
    """
    Map mode curvatures ``b2`` to Voigt stiffnesses (eV/Å³); see module docstring.
    """
    v = float(volume)
    if v <= 0.0:
        return {}

    c11 = b2A / v
    c12 = b2B / (2.0 * v) - c11
    c33 = 2.0 * b2C / v
    c13 = (b2A - b2D) / v
    c44 = 2.0 * abs(b2E) / (3.0 * v)

    return {"C11": c11, "C12": c12, "C13": c13, "C33": c33, "C44": c44}


def print_elastic_constants(
    stacking: str,
    constants: dict[str, float],
    *,
    reference_gpa: dict[str, float] | None = REFERENCE_GPA,
) -> None:
    print(f"\n{stacking} elastic constants (Voigt, eV/Å³ → GPa):")
    for name in ("C11", "C12", "C13", "C33", "C44"):
        val = constants.get(name, float("nan"))
        gpa = val * EV_A3_TO_GPA
        line = f"  {name} = {gpa:10.3f} GPa   ({val:.6f} eV/Å³)"
        if reference_gpa and name in reference_gpa and np.isfinite(gpa):
            ref = reference_gpa[name]
            line += f"    [lit. {ref:.2f} GPa, Δ={gpa - ref:+.1f}]"
        print(line)


def plot_mode_energy_fits(
    stacking: str,
    mode_data: dict,
    *,
    figures_dir: Path = FIGURES_DIR,
    dpi: int = 150,
) -> Path:
    """
    Compare DFT total energies with the quadratic ΔE(delta) fit used for each mode.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5))
    axes_flat = axes.ravel()

    for ax_idx, mode in enumerate(MODES):
        ax = axes_flat[ax_idx]
        data = mode_data.get(mode)
        if data is None:
            ax.set_title(f"mode {mode} (missing)")
            ax.axis("off")
            continue

        deltas, energies, (b2, b1, b0) = data
        i0 = int(np.argmin(np.abs(deltas)))
        e_ref = float(energies[i0])
        energies_rel = energies - e_ref
        fit = np.polyval([b2, b1, b0], deltas) - e_ref
        rms = float(np.sqrt(np.mean((fit - energies_rel) ** 2)))
        delta_line = np.linspace(float(deltas.min()), float(deltas.max()), 200)
        fit_line = np.polyval([b2, b1, b0], delta_line) - e_ref

        ax.plot(delta_line, fit_line, "-", color="C1", lw=2.0, label="quadratic fit")
        ax.plot(deltas, energies_rel, "o", color="C0", ms=7, label="DFT")
        ax.set_xlabel(r"strain $\delta$")
        ax.set_ylabel(r"$\Delta E$ relative to $\delta \approx 0$ (eV)")
        ax.set_title(
            f"mode {mode} ({MODE_STRAIN_PATH[mode]})  "
            rf"$b_2={b2:.4f}$ eV  RMS={rms:.2e} eV"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes_flat[-1].axis("off")
    fig.suptitle(f"{stacking}: DFT energy vs strain and quadratic fit", fontsize=12)
    fig.tight_layout()
    out = figures_dir / f"{stacking}_elastic_energy_fits.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ----------------------------------------------------------------------
# Load, match, label
# ----------------------------------------------------------------------
ref_frames = read(REF_FILE, index=':')
result_frames = read(RESULT_FILE, index=':')
print(f"loaded {len(ref_frames)} reference structures, {len(result_frames)} DFT result structures")

clusters = cluster_reference_frames(ref_frames)
print(f"reference set collapses to {len(clusters)} distinct geometries "
      f"({len(ref_frames)} labels -- the delta=0 points are shared across the 5 modes)")

records = match_and_expand(result_frames, clusters)

# sanity: every (stacking, mode, delta) combination in the reference set should
# have been matched exactly once
from collections import Counter
ref_keys = sorted({(s, m, round(d, 6)) for c in clusters for (s, m, d, _) in c[1]})
matched_keys = [(r['stacking'], r['mode'], round(r['delta'], 6)) for r in records]
counts = Counter(matched_keys)
missing = [k for k in ref_keys if counts.get(k, 0) == 0]
duplicated = [k for k, c in counts.items() if c > 1]
if missing:
    print(f"  WARNING: {len(missing)} reference points have no matching DFT result: {missing}")
if duplicated:
    print(f"  NOTE: {len(duplicated)} points matched more than once -- expected only for "
          f"delta=0 (shared across all 5 modes): {duplicated}")
print()

# ----------------------------------------------------------------------
# Group by (stacking, mode), fit E(delta), extract elastic constants
# ----------------------------------------------------------------------
by_group = {}
for r in records:
    key = (r['stacking'], r['mode'])
    by_group.setdefault(key, []).append(r)

results = {}   # results[stacking][mode] = (b2_eV, V0_A3)
mode_fit_data = {}  # mode_fit_data[stacking][mode] = (deltas, energies, (b2, b1, b0))
for stack in sorted({r['stacking'] for r in records}):
    ref0 = [a for a in ref_frames if a.info['stacking'] == stack and a.info['delta'] == 0.0][0]
    area = float(np.linalg.norm(np.cross(ref0.cell[0], ref0.cell[1])))
    d0 = float(ref0.info["d0"])
    V0 = slab_volume(area, d0)

    results[stack] = {}
    mode_fit_data[stack] = {}
    for mode in MODES:
        pts = sorted(by_group.get((stack, mode), []), key=lambda r: r["delta"])
        if len(pts) < 3:
            print(f"  WARNING: only {len(pts)} points for {stack}/{mode}, skipping")
            continue
        deltas = np.array([r["delta"] for r in pts])
        energies = np.array([r["energy"] for r in pts])
        b2, b1, b0, rms = quadratic_fit_delta_energy(deltas, energies)
        print(
            f"  {stack}/{mode}: n={len(pts)}  b2={b2:.6f} eV  b1={b1:.2e} eV  "
            f"fit RMS={rms:.2e} eV  ({MODE_STRAIN_PATH[mode]})"
        )
        results[stack][mode] = (b2, V0)
        mode_fit_data[stack][mode] = (deltas, energies, (b2, b1, b0))
    plot_mode_energy_fits(stack, mode_fit_data[stack])
    print(f"  slab volume V = A d0 = {V0:.4f} Å³")
    print()
# ----------------------------------------------------------------------
# Solve for C11, C12, C13, C33, C44 (eV/A^3 -> GPa)
# ----------------------------------------------------------------------
print("=" * 60)
for stack, modes in results.items():
    if not all(m in modes for m in MODES):
        print(f"\n{stack}: incomplete mode set {sorted(modes)} — skipping constants")
        continue
    b2A, V0 = modes["A"]
    b2B, _ = modes["B"]
    b2C, _ = modes["C"]
    b2D, _ = modes["D"]
    b2E, _ = modes["E"]
    constants = elastic_constants_from_modes(b2A, b2B, b2C, b2D, b2E, V0)
    print_elastic_constants(stack, constants)
