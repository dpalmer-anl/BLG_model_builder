#!/usr/bin/env python3
"""
run_uq_propagation_continuum_model.py
======================================
Fit a Nam & Koshino-style continuum Hamiltonian for twisted bilayer graphene
(TBLG) to the tight-binding (TB) eigenvalues already computed by
``run_uq_propagation_bands.py`` in ``bands/propagation/`` (or a k-mesh/mean
variant thereof), and propagate the resulting parameter uncertainty over the
ensemble.

Physics
-------
Following

    N. N. T. Nam and M. Koshino, "Lattice relaxation and energy band
    modulation in twisted bilayer graphene," Phys. Rev. B 96, 075311 (2017),

and

    M. Koshino and N. N. T. Nam, "Effective continuum model for relaxed
    twisted bilayer graphene and moiré electron-phonon interaction,"
    Phys. Rev. B 101, 195425 (2020),

the valley-``xi`` continuum Hamiltonian is a 4x4 (sublattice x layer) block
Hamiltonian::

    H = [[H_1,   U^dagger],
         [U,     H_2     ]]                                        (eq. 24)

``H_l(k)`` is the (possibly strain-shifted) monolayer Dirac Hamiltonian for
layer ``l`` (Koshino & Nam 2020, eq. 25)::

    H_l(k) = -hbar*v [R(+-theta/2) (k + e/hbar A^(l) - K^(l) xi)] . (xi*sigma_x, sigma_y)

In the supercell implementation, ``k``, ``G``, and ``K^(l)`` live in the
cell frame (moiré ``a1||x``); they are mapped to the graphene sublattice
frame before applying ``R(+theta/2)`` (layer 1) or ``R(-theta/2)`` (layer 2).
The strain pseudo-gauge ``A^(l)`` enters through the same frame map on ``u_G``.
At ``Gamma``, a gap between flat and dispersive bands requires (i) ``t ≠ t'``
from the relaxed interlayer displacement ``u^-`` (Koshino & Nam 2020, eqs.
33–34) and (ii) the intralayer strain pseudo-gauge from ``u^(l)_G``.  The rigid
``t = t' = t0`` model keeps all Γ states within ~12 meV of ``E_F`` even when
strain is included.

The strain correction only depends on the local strain tensor
``u_ij^(l) = (d_i u_j^(l) + d_j u_i^(l)) / 2``                      (eq. 26)
and on ``beta = -d ln t(d) / d ln d |_{d=a0}``                      (eq. 27).
``u^(l)(R)`` (the atomic displacement of layer ``l`` relative to the
*non-relaxed*, purely-rotated TBLG structure) is measured directly from the
relaxed ensemble trajectories (``trajectories/relaxation/``): it is the
initial (frame 0, rigid) vs. final (frame -1, relaxed) in-plane MIC
displacement of each atom.  The 6 dominant Fourier harmonics of ``u^(l)``
(at +-G1_M, +-G2_M, +-(G1_M+G2_M), the shortest moire reciprocal vectors) are
extracted from the atomic displacement field and used to build the strain
pseudo-gauge coupling in a truncated plane-wave basis.

``U`` is the interlayer coupling restricted to its 6 dominant harmonics
(eq. 32)::

    U ~= [[t, t' ],[t', t]]
       + [[t, t'*w^-xi],[t'*w^xi, t]] * exp(i*xi*G1_M . r)
       + [[t, t'*w^xi ],[t'*w^-xi, t]] * exp(i*xi*(G1_M+G2_M) . r)

with ``w = exp(2*pi*i/3)``.

Fitting parameters: ``hbar_v`` (eV.Angstrom), ``gamma0`` (eV), ``beta``
(dimensionless), ``t`` (eV), ``t'`` (eV).  A separate continuum model is
fit, by nonlinear least squares, to each ensemble sample (pairing band file
``sampleNNNN.npz`` with the relaxation trajectory at the same position in the
sorted trajectory list -- exactly the pairing used by
``run_uq_propagation_bands.py``).  Samples missing either file are skipped.

For the loss, both band structures are shifted so their own Fermi level is at
0 eV.  Flat bands are identified at the first K point as all bands within
``--flat-band-threshold-meV`` (default 5 meV) of E_F.  At K, Γ, and M the fit
targets nine scalars from the TB spectrum: flat-band width, upper band gap, and
lower band gap (three metrics × three high-symmetry points).

Parameter uncertainties are estimated from the standard nonlinear
least-squares covariance, ``cov ~= s^2 (J^T J)^-1`` where ``s^2`` is the
reduced chi-square of the fit.

Note on parameter identifiability: in eqs. 25-27, ``gamma0`` and ``beta``
only ever enter the Hamiltonian through the product ``beta*gamma0`` (the
strength of the strain pseudo-gauge coupling); ``hbar_v`` only sets the
kinetic-term slope; ``t``/``t'`` only set the interlayer coupling.  ``gamma0``
and ``beta`` are therefore *not* separately identifiable from the band
structure alone -- expect a large, strongly (anti-)correlated uncertainty
for this pair even though ``beta*gamma0`` itself is well constrained.

Usage (mirrors ``run_uq_propagation_bands.py``)
------------------------------------------------
::

    python run_uq_propagation_continuum_model.py \\
        --models 'POD_energy_POD_index*' \\
        --tb-model acsf_hoppings_sk_M_9_W_6 \\
        --tb-temperature 0.25 \\
        --temperature 0.1 \\
        --twist-angle 0.83

    # Fit just one sample and plot the comparison (temporary file, overwritten
    # each run):
    python run_uq_propagation_continuum_model.py ... --sample-index 0 \\
        --plot-comparison

Speed / accuracy
----------------
Each optimizer step diagonalizes the continuum Hamiltonian at K, Γ, and M
(three k-points × one dense ``(4*n_b) x (4*n_b)`` solve).  ``--n-shells 4``
is recommended for accurate flat-band gaps; ``--n-shells 2-3`` is faster but
may truncate the Dirac point.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse path / discovery / geometry helpers from run_uq_propagation_bands.py so
# the two scripts always agree on where files live and how samples are paired.
from run_uq_propagation_bands import (  # noqa: E402
    DEFAULT_TB_MODEL,
    DEFAULT_TB_RCUT,
    DEFAULT_TWIST_ANGLE,
    DEFAULT_TRAJECTORY_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_BANDS_DIR,
    DEFAULT_CALIBRATION_METRICS_DIR,
    DEFAULT_ENSEMBLE_SHUFFLE_SEED,
    _discover_traj_files,
    _existing_bands_npz,
    _get_recip_cell,
    _parse_tb_model_name,
    _resolve_tb_ensemble,
    _safe_filename_part,
)
from blg_model_builder.cli_hyperparams import add_hyperparam_args, collect_hyperparams  # noqa: E402,F401
from blg_model_builder.cli_model_names import add_energy_models_arg  # noqa: E402

try:
    from blg_model_builder.strain_data import LAT_CON as _LAT_CON  # graphene lattice constant (Å)
except Exception:  # pragma: no cover
    _LAT_CON = 2.46

# ---------------------------------------------------------------------------
# Physical constants / initial (paper) parameter values
# ---------------------------------------------------------------------------

A_GRAPHENE: float = float(_LAT_CON)                # Å
A0_NN: float = A_GRAPHENE / np.sqrt(3.0)            # nearest-neighbor distance, Å
R0_DECAY: float = 0.184 * A_GRAPHENE                # Slater-Koster decay length, Å

INIT_HBAR_V: float = 2.1354 * A_GRAPHENE            # eV.Å  (hbar*v/a = 2.1354 eV, Koshino 2020)
INIT_GAMMA0: float = 2.7                            # eV    (|V0_pppi|)
INIT_BETA: float = A0_NN / R0_DECAY                 # ~3.14 (eq. 27, for the SK model of eq. 6/39)
INIT_T0: float = 0.104                              # eV    (t0 in eq. 23/32-34, Koshino 2020)

PARAM_NAMES: Tuple[str, ...] = ("hbar_v", "gamma0", "beta", "t", "tprime")
SYM_POINT_NAMES: Tuple[str, ...] = ("K", "Gamma", "M")
FIT_METRIC_NAMES: Tuple[str, ...] = tuple(
    f"{pt}_{metric}"
    for pt in SYM_POINT_NAMES
    for metric in ("flat_width", "upper_gap", "lower_gap")
)

DEFAULT_FLAT_BAND_THRESHOLD_MEV: float = 5.0
DEFAULT_N_SHELLS: int = 4
DEFAULT_OUTPUT_DIR: str = os.path.join("continuum_fits", "propagation")

_OMEGA3 = np.exp(2j * np.pi / 3.0)
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rot2(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def moire_reciprocal_vectors(cell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(G1, G2)``, the in-plane moire reciprocal lattice vectors.

    For the commensurate supercells built by ``flatgraphene`` at the twist
    angles used in this project (period exactly ``L_M``), the supercell's own
    reciprocal lattice vectors *are* the moire reciprocal vectors ``G1_M``,
    ``G2_M`` of Nam & Koshino (2017), eq. 4.
    """
    recip = _get_recip_cell(np.asarray(cell, dtype=float).T)
    return np.asarray(recip[0, :2], dtype=float), np.asarray(recip[1, :2], dtype=float)


def mbz_dirac_corners(G1: np.ndarray, G2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """mBZ corners ``K1 = (G1+2 G2)/3``, ``K2 = (2 G1+G2)/3``.

    Matches ``m_k1_vec`` / ``m_k2_vec`` in TAPW_LETB ``continuum.py`` and the
    TB path node ``K = [1/3, 2/3]``.
    """
    K1 = (G1 + 2.0 * G2) / 3.0
    K2 = (2.0 * G1 + G2) / 3.0
    return K1, K2


def _graphene_reciprocal() -> Tuple[np.ndarray, np.ndarray]:
    """Unrotated graphene reciprocal vectors (same as TAPW_LETB continuum.py)."""
    a = A_GRAPHENE
    b1 = np.array([2.0 * np.pi / a, -2.0 * np.pi / (a * np.sqrt(3.0))])
    b2 = np.array([0.0, 4.0 * np.pi / (a * np.sqrt(3.0))])
    return b1, b2


def graphene_dirac_kpts(theta_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Graphene K points of the two layers and reference moiré G, ±θ/2 convention.

    ``kpt1 = -R(θ/2)(2 b1 + b2)/3``, ``kpt2 = -R(-θ/2)(2 b1 + b2)/3``,
    ``G_i^M = R(θ/2) b_i - R(-θ/2) b_i`` (TAPW_LETB ``_set_kpt`` / ``_set_moire``).
    """
    b1, b2 = _graphene_reciprocal()
    rt = _rot2(theta_rad / 2.0)
    kpt1 = -(2.0 * b1 + b2) @ rt.T / 3.0
    kpt2 = -(2.0 * b1 + b2) @ rt / 3.0
    g1_ref = b1 @ rt.T - b1 @ rt
    g2_ref = b2 @ rt.T - b2 @ rt
    return kpt1, kpt2, g1_ref, g2_ref


def _hex_star(G1: np.ndarray, G2: np.ndarray) -> List[np.ndarray]:
    return [G1, G2, G1 + G2, -G1, -G2, -(G1 + G2)]


def _oriented_moire_bases(G1: np.ndarray, G2: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """120° oriented bases drawn from the first hexagonal star (TAPW sense: det>0)."""
    star = _hex_star(G1, G2)
    bases = []
    for g1 in star:
        g1_len2 = float(np.dot(g1, g1))
        for g2 in star:
            if abs(float(np.dot(g1, g2)) + 0.5 * g1_len2) > 1e-6:
                continue
            if (g1[0] * g2[1] - g1[1] * g2[0]) <= 0.0:
                continue
            bases.append((np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)))
    return bases


def align_dirac_kpts_to_cell(
    theta_rad: float, G1: np.ndarray, G2: np.ndarray, valley: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cell-frame TAPW graphene K and moiré G.

    1. ``g1_m, g2_m`` are the oriented 120° basis whose ``K1=(g1+2g2)/3`` is
       the TB K.
    2. Graphene ``kpt1`` is the TAPW K of layer 1 rotated/scaled onto that
       basis, then snapped so ``q1(K1)=0`` on the cell G lattice.
    3. ``kpt2 = kpt1 + K2`` (TAPW identity).  Using the rotated ``kpt2``
       instead leaves both layer Dirac points on the same plane wave and
       gaps K; using a ``kpt1`` from a different G shell (wrong angle vs
       ``g1``) also gaps K even with TAPW's T matrices.
    """
    kpt1_ref, kpt2_ref, g1_ref, g2_ref = graphene_dirac_kpts(theta_rad)
    K_tb, _ = mbz_dirac_corners(G1, G2)
    bases = _oriented_moire_bases(G1, G2)
    if not bases:
        g1_m, g2_m = np.asarray(G1, dtype=float), np.asarray(G2, dtype=float)
    else:
        g1_m, g2_m = min(
            bases,
            key=lambda b: float(np.linalg.norm((b[0] + 2.0 * b[1]) / 3.0 - K_tb)),
        )
    scale = float(np.linalg.norm(g1_m) / np.linalg.norm(g1_ref))
    src = np.column_stack((g1_ref, g2_ref)) * scale
    tgt = np.column_stack((g1_m, g2_m))
    best_R = np.eye(2)
    best = np.inf
    for refl in (np.eye(2), np.diag([1.0, -1.0])):
        s = refl @ (scale * g1_ref)
        ang = np.arctan2(g1_m[1], g1_m[0]) - np.arctan2(s[1], s[0])
        R = _rot2(ang) @ refl
        score = float(np.linalg.norm(R @ src - tgt))
        if score < best:
            best = score
            best_R = R
    kpt1 = best_R @ (scale * kpt1_ref)
    K1 = (g1_m + 2.0 * g2_m) / 3.0
    K2 = (2.0 * g1_m + g2_m) / 3.0
    coeff = np.linalg.solve(np.column_stack((g1_m, g2_m)), valley * kpt1 - K1)
    G_d = np.round(coeff).astype(int)
    G_d = G_d[0] * g1_m + G_d[1] * g2_m
    kpt1 = valley * (K1 + G_d)
    kpt2 = kpt1 + K2
    return kpt1, kpt2, g1_m, g2_m, best_R


def moire_g_offset(
    kpt1: np.ndarray, G1: np.ndarray, G2: np.ndarray, valley: int,
    g1_m: Optional[np.ndarray] = None, g2_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Moiré G nearest to ``valley*kpt1 - K1``, so a plane wave sits on the
    layer-1 Dirac point at TAPW ``K1`` (the TB K after alignment)."""
    if g1_m is None or g2_m is None:
        K1, _K2 = mbz_dirac_corners(G1, G2)
    else:
        K1 = (np.asarray(g1_m, dtype=float) + 2.0 * np.asarray(g2_m, dtype=float)) / 3.0
    target = valley * kpt1 - K1
    coeff = np.linalg.solve(np.column_stack((G1, G2)), target)
    n1, n2 = np.round(coeff).astype(int)
    return n1 * G1 + n2 * G2


def plane_wave_glist(
    cb: CoupledBasis, G1: np.ndarray, G2: np.ndarray, offset: np.ndarray,
) -> np.ndarray:
    return np.array([offset + m1 * G1 + m2 * G2 for m1, m2 in cb.basis], dtype=float)


def layer_masks(atoms) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mask_layer1, mask_layer2)`` using the ``mol-id`` convention
    of ``run_uq_propagation_relaxation.py`` (1 = bottom / lower z, 2 = top)."""
    if atoms.has("mol-id"):
        mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
        u = np.unique(mol)
        if u.size == 2:
            return mol == int(u[0]), mol == int(u[1])
    z = np.asarray(atoms.get_positions()[:, 2], dtype=float)
    mid = float(np.median(z))
    return z < mid, z >= mid


def layer_displacements(atoms_init, atoms_rel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mask1, mask2, pos_init_xy, disp_xy)``.

    ``disp_xy`` is the in-plane MIC displacement (relaxed - initial) for every
    atom, i.e. the atomic displacement field ``u(R)`` of eq. 2 (Koshino 2020)
    / eq. 26 (Nam & Koshino 2017), measured directly from the relaxation
    trajectory (frame 0 = rigid/non-relaxed, frame -1 = relaxed).
    """
    from ase.geometry import find_mic  # lazy import

    pos_init = np.asarray(atoms_init.get_positions(wrap=False), dtype=float)
    pos_rel = np.asarray(atoms_rel.get_positions(wrap=False), dtype=float)
    cell = np.asarray(atoms_rel.get_cell(), dtype=float)
    mic, _ = find_mic(pos_rel - pos_init, cell, pbc=[True, True, False])
    disp_xy = np.asarray(mic, dtype=float)[:, :2]
    mask1, mask2 = layer_masks(atoms_init)
    return mask1, mask2, pos_init[:, :2], disp_xy


def fourier_disp(pos_xy: np.ndarray, disp_xy: np.ndarray, qvec: np.ndarray) -> np.ndarray:
    """Discrete atomic Fourier component ``u_q = (1/N) sum_R u(R) exp(-i q.R)``."""
    phase = np.exp(-1j * (pos_xy @ np.asarray(qvec, dtype=float)))
    return (disp_xy * phase[:, None]).mean(axis=0)


def six_dominant_uG(
    mask1: np.ndarray,
    mask2: np.ndarray,
    pos_xy: np.ndarray,
    disp_xy: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
) -> Dict[Tuple[int, int], Dict[int, np.ndarray]]:
    """Fourier components ``u^(l)_G`` of the per-layer displacement field at
    the 3 independent shortest moire reciprocal vectors ``G1, G2, G1+G2``.

    Returns a dict keyed by ``(n1, n2) in {(1,0), (0,1), (1,1)}`` mapping to
    ``{1: u1_q (complex 2-vector), 2: u2_q (complex 2-vector)}``.  The
    remaining 3 of the "6 dominant components" are the negatives of these
    and are obtained by complex conjugation (the displacement field is real).
    """
    out: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
    for key, qvec in ((1, 0), G1), ((0, 1), G2), ((1, 1), G1 + G2):
        u1 = fourier_disp(pos_xy[mask1], disp_xy[mask1], qvec)
        u2 = fourier_disp(pos_xy[mask2], disp_xy[mask2], qvec)
        out[key] = {1: u1, 2: u2}
    return out


# ---------------------------------------------------------------------------
# Truncated plane-wave basis
# ---------------------------------------------------------------------------

def build_basis(n_shells: int) -> List[Tuple[int, int]]:
    """Hexagonal-truncated set of ``(m1, m2)`` moire reciprocal lattice
    indices (``|m1| <= n``, ``|m2| <= n``, ``|m1+m2| <= n``)."""
    n = int(n_shells)
    basis = [
        (m1, m2)
        for m1 in range(-n, n + 1)
        for m2 in range(-n, n + 1)
        if abs(m1 + m2) <= n
    ]
    return sorted(basis)


@dataclass
class CoupledBasis:
    basis: List[Tuple[int, int]]
    index: Dict[Tuple[int, int], int]
    inter_pairs: List[Tuple[int, int, int]]         # (i_layer1, j_layer2, j_type in {1,2,3})
    strain_pairs: List[Tuple[int, int, Tuple[int, int]]]  # (i, j, (n1,n2) in the 3 positive dirs)


def build_coupled_basis(n_shells: int) -> CoupledBasis:
    basis = build_basis(n_shells)
    index = {mn: i for i, mn in enumerate(basis)}

    inter_pairs: List[Tuple[int, int, int]] = []
    inter_shifts = {1: (0, 0), 2: (1, 0), 3: (1, 1)}  # xi = +1
    for (m1, m2), i in index.items():
        for j_type, (s1, s2) in inter_shifts.items():
            target = (m1 + s1, m2 + s2)
            j = index.get(target)
            if j is not None:
                inter_pairs.append((i, j, j_type))

    strain_pairs: List[Tuple[int, int, Tuple[int, int]]] = []
    strain_dirs = [(1, 0), (0, 1), (1, 1)]
    for (m1, m2), i in index.items():
        for (n1, n2) in strain_dirs:
            target = (m1 + n1, m2 + n2)
            j = index.get(target)
            if j is not None:
                strain_pairs.append((i, j, (n1, n2)))
    return CoupledBasis(basis, index, inter_pairs, strain_pairs)


# ---------------------------------------------------------------------------
# Continuum Hamiltonian
# ---------------------------------------------------------------------------

def _strain_block(qvec: np.ndarray, uq: np.ndarray, beta: float, gamma0: float, xi: int) -> np.ndarray:
    """Pseudo-gauge-field correction block (eq. 26/27, see module docstring
    for the ``hbar*v``-independent simplified form) for Fourier component
    ``uq`` at wavevector ``qvec``."""
    qx, qy = float(qvec[0]), float(qvec[1])
    ux, uy = uq[0], uq[1]
    uxx = 1j * qx * ux
    uyy = 1j * qy * uy
    uxy = 0.5 * (1j * qx * uy + 1j * qy * ux)
    return (
        -0.75 * beta * gamma0 * (uxx - uyy) * _SIGMA_X
        + xi * 1.5 * beta * gamma0 * uxy * _SIGMA_Y
    )


def build_hamiltonian(
    k_gamma: np.ndarray,
    params: np.ndarray,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    xi: int = -1,
    kpt1: Optional[np.ndarray] = None,
    kpt2: Optional[np.ndarray] = None,
    glist: Optional[np.ndarray] = None,
    g1_m: Optional[np.ndarray] = None,
    g2_m: Optional[np.ndarray] = None,
    lattice_R: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Continuum Hamiltonian at moiré Bloch ``k_gamma`` (from supercell Γ).

    Kinetic + interlayer follow Koshino & Nam (2020) eqs. (23)--(25)::

        q_l^cell = k + G - valley * K_l
        q_l = R(+θ/2) R_cell→nat q_l^cell   (layer 1)
        q_l = R(-θ/2) R_cell→nat q_l^cell   (layer 2)
        H_l = -ħv (valley σ_x q_x + σ_y q_y)

    ``R_cell→nat`` aligns the supercell (moiré ``a1||x``) with the graphene
    sublattice frame; ``R(±θ/2)`` is the layer rotation from eq. (25).
    Strain pseudo-gauge blocks use the same ``R_cell→nat`` on ``G`` and ``u_G``.
    ``H = [[H1, T], [T†, H2]]`` with T the three moiré harmonics.
    """
    hbar_v, gamma0, beta, t, tprime = (float(p) for p in params)
    valley = int(np.sign(xi) or -1)
    if kpt1 is None or kpt2 is None or g1_m is None or g2_m is None or lattice_R is None:
        kpt1, kpt2, g1_m, g2_m, lattice_R = align_dirac_kpts_to_cell(
            theta_rad, G1, G2, valley=valley,
        )
    if glist is None:
        offset = moire_g_offset(kpt1, G1, G2, valley, g1_m=g1_m, g2_m=g2_m)
        glist = plane_wave_glist(cb, G1, G2, offset)

    n_b = len(glist)
    dim = 4 * n_b
    h1 = np.zeros((2 * n_b, 2 * n_b), dtype=complex)
    h2 = np.zeros((2 * n_b, 2 * n_b), dtype=complex)
    tmat = np.zeros((2 * n_b, 2 * n_b), dtype=complex)

    # Supercell moiré a1||x; σ is in the graphene (TAPW natural) frame.  lattice_R
    # maps natural→cell (~−90°); R(+θ/2), R(−θ/2) are the layer rotations in eq. (25).
    R_nat_to_cell = np.asarray(lattice_R, dtype=float)
    R_cell_to_nat = R_nat_to_cell.T
    rt_p = _rot2(theta_rad / 2.0)
    rt_m = rt_p.T
    for i in range(n_b):
        q1c = k_gamma + glist[i] - valley * kpt1
        q2c = k_gamma + glist[i] - valley * kpt2
        q1 = rt_p @ (R_cell_to_nat @ q1c)
        q2 = rt_m @ (R_cell_to_nat @ q2c)
        h1[2 * i:2 * i + 2, 2 * i:2 * i + 2] = -hbar_v * (
            valley * q1[0] * _SIGMA_X + q1[1] * _SIGMA_Y
        )
        h2[2 * i:2 * i + 2, 2 * i:2 * i + 2] = -hbar_v * (
            valley * q2[0] * _SIGMA_X + q2[1] * _SIGMA_Y
        )

    g_tr = (np.zeros(2), valley * g1_m, valley * (g1_m + g2_m))
    phase1 = _OMEGA3 ** valley
    phase2 = np.conj(phase1)
    t_blocks = (
        np.array([[t, tprime], [tprime, t]], dtype=complex),
        np.array([[t, tprime * phase2], [tprime * phase1, t]], dtype=complex),
        np.array([[t, tprime * phase1], [tprime * phase2, t]], dtype=complex),
    )
    for i in range(n_b):
        for j in range(n_b):
            dk = glist[j] - glist[i]
            for gtr, tb in zip(g_tr, t_blocks):
                if np.linalg.norm(dk - gtr) < 1e-8:
                    tmat[2 * i:2 * i + 2, 2 * j:2 * j + 2] = tb

    off1, off2 = 0, 2 * n_b
    H = np.zeros((dim, dim), dtype=complex)
    H[off1:off2, off1:off2] = h1
    H[off2:dim, off2:dim] = h2
    H[off1:off2, off2:dim] = tmat
    H[off2:dim, off1:off2] = tmat.conj().T

    Gvecs = {(1, 0): G1, (0, 1): G2, (1, 1): G1 + G2}
    for i, j, key in cb.strain_pairs:
        qvec = R_cell_to_nat @ Gvecs[key]
        for l, off in ((1, off1), (2, off2)):
            uq = uG[key][l]
            uq_nat = R_cell_to_nat @ np.asarray([uq[0], uq[1]], dtype=complex)
            block = _strain_block(qvec, uq_nat, beta, gamma0, valley)
            H[off + 2 * i:off + 2 * i + 2, off + 2 * j:off + 2 * j + 2] += block
            H[off + 2 * j:off + 2 * j + 2, off + 2 * i:off + 2 * i + 2] += block.conj().T

    return H


def continuum_bands(
    k_from_gamma: np.ndarray,
    params: np.ndarray,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
) -> np.ndarray:
    """Full eigenvalue spectrum (ascending) at every moiré k (from Γ), shape
    ``(n_k, 4*n_b)``."""
    valley = -1
    kpt1, kpt2, g1_m, g2_m, lattice_R = align_dirac_kpts_to_cell(
        theta_rad, G1, G2, valley=valley,
    )
    offset = moire_g_offset(kpt1, G1, G2, valley, g1_m=g1_m, g2_m=g2_m)
    glist = plane_wave_glist(cb, G1, G2, offset)
    out = []
    for k_gamma in k_from_gamma:
        H = build_hamiltonian(
            k_gamma, params, cb, G1, G2, theta_rad, uG,
            xi=valley, kpt1=kpt1, kpt2=kpt2, glist=glist, g1_m=g1_m, g2_m=g2_m,
            lattice_R=lattice_R,
        )
        w = np.linalg.eigvalsh(H)
        out.append(w)
    return np.array(out)


@dataclass
class ContinuumKSetup:
    kpt1: np.ndarray
    kpt2: np.ndarray
    g1_m: np.ndarray
    g2_m: np.ndarray
    lattice_R: np.ndarray
    glist: np.ndarray
    valley: int = -1


def continuum_k_setup(
    cb: CoupledBasis, G1: np.ndarray, G2: np.ndarray, theta_rad: float, valley: int = -1,
) -> ContinuumKSetup:
    kpt1, kpt2, g1_m, g2_m, lattice_R = align_dirac_kpts_to_cell(
        theta_rad, G1, G2, valley=valley,
    )
    offset = moire_g_offset(kpt1, G1, G2, valley, g1_m=g1_m, g2_m=g2_m)
    glist = plane_wave_glist(cb, G1, G2, offset)
    return ContinuumKSetup(kpt1, kpt2, g1_m, g2_m, lattice_R, glist, valley)


def continuum_evals_at_k(
    k_gamma: np.ndarray,
    params: np.ndarray,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    setup: ContinuumKSetup,
) -> np.ndarray:
    """Ascending continuum eigenvalues at one moiré k (from supercell Γ)."""
    H = build_hamiltonian(
        k_gamma, params, cb, G1, G2, theta_rad, uG,
        xi=setup.valley, kpt1=setup.kpt1, kpt2=setup.kpt2, glist=setup.glist,
        g1_m=setup.g1_m, g2_m=setup.g2_m, lattice_R=setup.lattice_R,
    )
    return np.linalg.eigvalsh(H)


def continuum_sym_evals_fermi_shifted(
    kappas: np.ndarray,
    sym_indices: Dict[str, int],
    params: np.ndarray,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    setup: ContinuumKSetup,
) -> Dict[str, np.ndarray]:
    """Fermi-shifted continuum spectra at K, Γ, M (E_F from half-filling at K)."""
    evals_k = continuum_evals_at_k(
        kappas[sym_indices["K"]], params, cb, G1, G2, theta_rad, uG, setup,
    )
    n_half = len(evals_k) // 2
    fermi = float(0.5 * (evals_k[n_half] + evals_k[n_half - 1]))
    return {
        name: continuum_evals_at_k(kappas[idx], params, cb, G1, G2, theta_rad, uG, setup) - fermi
        for name, idx in sym_indices.items()
    }


def continuum_bands_fermi_shifted(
    kappas: np.ndarray,
    params: np.ndarray,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
) -> Tuple[np.ndarray, float]:
    """Full continuum spectrum, shifted so E_F (half-filling at ``kappas[0]``)
    is 0 eV.  Shape ``(n_k, 4*n_b)``."""
    n_b = len(cb.basis)
    n_half = 2 * n_b
    evals_full = continuum_bands(kappas, params, cb, G1, G2, theta_rad, uG)
    fermi = float((evals_full[0, n_half] + evals_full[0, n_half - 1]) / 2.0)
    return evals_full - fermi, fermi


def paper_continuum_params(
    uG: Optional[Dict[Tuple[int, int], Dict[int, np.ndarray]]] = None,
) -> np.ndarray:
    """Koshino & Nam (2020) default values: ``ħv/a=2.1354 eV``, ``γ0=2.7 eV``,
    ``β≈3.14``, ``t0=0.104 eV`` (eqs. 25–27, 32–34).

    When ``uG`` is supplied, ``t`` and ``t'`` follow eqs. (33)–(34) using the
    leading ``u^- = u2 - u1`` harmonic at ``G1``.  Koshino & Nam (2020) note
    that ``t ≠ t'`` from lattice relaxation opens the ~40 meV gap at Γ between
    flat and dispersive bands; the rigid ``t = t' = t0`` case keeps all Γ
    states within ~12 meV of ``E_F`` even with strain.
    """
    params = np.array([INIT_HBAR_V, INIT_GAMMA0, INIT_BETA, INIT_T0, INIT_T0], dtype=float)
    if uG is not None:
        t, tprime = initial_tprime_t(uG)
        params[3], params[4] = t, tprime
    return params


def continuum_bands_window(
    kappas: np.ndarray,
    params: np.ndarray,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    n_each_side: int,
) -> Tuple[np.ndarray, float]:
    """``(evals_window, fermi_level)``: the ``2*n_each_side`` continuum bands
    nearest E_F, shifted so E_F (half-filling at ``kappas[0]``) is 0 eV."""
    evals_full, fermi = continuum_bands_fermi_shifted(
        kappas, params, cb, G1, G2, theta_rad, uG,
    )
    n_half = evals_full.shape[1] // 2
    lo, hi = n_half - n_each_side, n_half + n_each_side
    return evals_full[:, lo:hi], fermi


def tb_bands_window(evals: np.ndarray, n_atoms: int, band_lo: int, n_each_side: int) -> np.ndarray:
    """Select the ``2*n_each_side`` TB bands nearest E_F=0 from the (already
    Fermi-shifted) saved TB window ``evals`` of shape ``(n_k, band_hi-band_lo)``."""
    nocc = int(n_atoms) // 2
    center = nocc - int(band_lo)
    lo, hi = center - n_each_side, center + n_each_side
    if lo < 0 or hi > evals.shape[1]:
        raise ValueError(
            f"TB band window ({lo}:{hi}) out of range for saved evals with "
            f"{evals.shape[1]} bands (band_lo={band_lo}); increase n_keep in "
            "run_uq_propagation_bands.py or decrease --n-each-side."
        )
    return evals[:, lo:hi]


# ---------------------------------------------------------------------------
# Flat-band fitting targets (K, Γ, M)
# ---------------------------------------------------------------------------

def kpath_symmetry_indices(n_k: int) -> Dict[str, int]:
    """Indices of K, Γ, and M on the K–Γ–M–K path from ``run_uq_propagation_bands``.

    The path uses ``mesh_step = (n_k - 1) // 3`` points per segment; node
    indices are ``0``, ``mesh_step``, ``2 * mesh_step``.
    """
    if n_k < 4:
        raise ValueError(f"Need at least 4 k-points on the path, got {n_k}")
    mesh_step = (n_k - 1) // 3
    return {"K": 0, "Gamma": mesh_step, "M": 2 * mesh_step}


def identify_flat_band_indices(evals: np.ndarray, threshold_ev: float) -> np.ndarray:
    """Band indices (ascending-energy order) with ``|E| <= threshold`` at K."""
    evals = np.asarray(evals, dtype=float).ravel()
    mask = np.abs(evals) <= float(threshold_ev)
    if not np.any(mask):
        raise ValueError(
            f"No bands within {threshold_ev * 1e3:.3f} meV of E_F at K; "
            "increase the saved TB band window or relax --flat-band-threshold-meV."
        )
    return np.flatnonzero(mask)


def flat_band_metrics_at_k(evals: np.ndarray, flat_indices: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(flat_width, upper_gap, lower_gap)`` in eV at one k-point.

    Upper (lower) gap is measured to the nearest band *not* in ``flat_indices``
    above (below) the flat set, not merely the next ascending index (which may
    still be flat-like at Γ).
    """
    evals = np.asarray(evals, dtype=float).ravel()
    flat_indices = np.asarray(flat_indices, dtype=int)
    flat_set = set(flat_indices.tolist())
    e_flat = evals[flat_indices]
    width = float(np.max(e_flat) - np.min(e_flat))
    e_hi = float(np.max(e_flat))
    e_lo = float(np.min(e_flat))
    above = [i for i in range(evals.size) if i not in flat_set and evals[i] > e_hi + 1e-15]
    below = [i for i in range(evals.size) if i not in flat_set and evals[i] < e_lo - 1e-15]
    if not above:
        raise ValueError("Cannot compute upper band gap: no non-flat band above the flat set")
    if not below:
        raise ValueError("Cannot compute lower band gap: no non-flat band below the flat set")
    upper_gap = float(evals[min(above, key=lambda i: evals[i])] - e_hi)
    lower_gap = float(e_lo - evals[max(below, key=lambda i: evals[i])])
    return width, upper_gap, lower_gap


def sym_point_fit_metrics(
    evals_by_sym: Dict[str, np.ndarray],
    flat_indices: np.ndarray,
) -> np.ndarray:
    """Nine-element vector: width, upper gap, lower gap at K, Γ, M."""
    out: List[float] = []
    for name in SYM_POINT_NAMES:
        width, upper, lower = flat_band_metrics_at_k(evals_by_sym[name], flat_indices)
        out.extend([width, upper, lower])
    return np.asarray(out, dtype=float)


def tb_sym_evals(tb_evals: np.ndarray, sym_indices: Dict[str, int]) -> Dict[str, np.ndarray]:
    return {name: np.asarray(tb_evals[idx], dtype=float) for name, idx in sym_indices.items()}


def tb_fit_targets(
    tb_evals: np.ndarray,
    sym_indices: Dict[str, int],
    threshold_ev: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(targets, flat_indices)`` for the nine-scalar TB fit vector."""
    sym_evals = tb_sym_evals(tb_evals, sym_indices)
    flat_idx = identify_flat_band_indices(sym_evals["K"], threshold_ev)
    return sym_point_fit_metrics(sym_evals, flat_idx), flat_idx


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

INIT_ALPHA_COEFF = 2.0 * np.pi / np.sqrt(3.0)


def initial_tprime_t(uG: Dict[Tuple[int, int], Dict[int, np.ndarray]]) -> Tuple[float, float]:
    """Seed ``(t, tprime)`` from the paper's eq. 33/34, using the leading
    (``G1``) harmonic of the measured relative displacement ``u^- = u2 - u1``."""
    u1, u2 = uG[(1, 0)][1], uG[(1, 0)][2]
    u_minus = u2 - u1
    u1_amp = float(np.linalg.norm(u_minus))
    alpha = INIT_ALPHA_COEFF * (u1_amp / A_GRAPHENE)
    t = INIT_T0 * (1.0 - 2.0 * alpha)
    tprime = INIT_T0 * (1.0 + 0.5 * alpha)
    return t, tprime


def fit_continuum_model(
    kappas: np.ndarray,
    tb_evals: np.ndarray,
    sym_indices: Dict[str, int],
    threshold_ev: float,
    cb: CoupledBasis,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_rad: float,
    uG: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    setup: ContinuumKSetup,
    *,
    x0: Optional[np.ndarray] = None,
    max_nfev: int = 200,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.optimize import least_squares

    if x0 is None:
        t0, tprime0 = initial_tprime_t(uG)
        x0 = np.array([INIT_HBAR_V, INIT_GAMMA0, INIT_BETA, t0, tprime0], dtype=float)

    tb_targets, tb_flat_idx = tb_fit_targets(tb_evals, sym_indices, threshold_ev)

    def residuals(x: np.ndarray) -> np.ndarray:
        cont_evals = continuum_sym_evals_fermi_shifted(
            kappas, sym_indices, x, cb, G1, G2, theta_rad, uG, setup,
        )
        cont_flat_idx = identify_flat_band_indices(cont_evals["K"], threshold_ev)
        cont_metrics = sym_point_fit_metrics(cont_evals, cont_flat_idx)
        return cont_metrics - tb_targets

    lower = [0.5 * A_GRAPHENE, 0.2, -8.0, -1.0, -1.0]
    upper = [10.0 * A_GRAPHENE, 8.0, 8.0, 1.0, 1.0]
    result = least_squares(
        residuals, x0, bounds=(lower, upper), method="trf",
        max_nfev=int(max_nfev), verbose=int(verbose),
    )

    n_data = result.fun.size
    n_params = x0.size
    dof = max(n_data - n_params, 1)
    resid_var = 2.0 * result.cost / dof
    try:
        jtj = result.jac.T @ result.jac
        cov = resid_var * np.linalg.inv(jtj)
        perr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        cov = np.full((n_params, n_params), np.nan)
        perr = np.full(n_params, np.nan)

    mse = float(np.mean(result.fun ** 2))
    cont_evals = continuum_sym_evals_fermi_shifted(
        kappas, sym_indices, result.x, cb, G1, G2, theta_rad, uG, setup,
    )
    cont_flat_idx = identify_flat_band_indices(cont_evals["K"], threshold_ev)
    cont_metrics = sym_point_fit_metrics(cont_evals, cont_flat_idx)
    return result.x, perr, cov, mse, x0, tb_targets, cont_metrics


# ---------------------------------------------------------------------------
# Plot (temporary, overwritten each run) comparing continuum vs. TB bands for
# a single sample.
# ---------------------------------------------------------------------------

def plot_single_sample_comparison(
    path: str,
    k_dist: np.ndarray,
    k_node: np.ndarray,
    sym_labels,
    tb_evals: np.ndarray,
    continuum_fit: np.ndarray,
    continuum_paper: np.ndarray,
    *,
    title: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for b in range(tb_evals.shape[1]):
        ax.plot(k_dist, tb_evals[:, b] * 1e3, color="tab:blue", lw=1.0,
                label="tight-binding" if b == 0 else None)
    for b in range(continuum_paper.shape[1]):
        ax.plot(k_dist, continuum_paper[:, b] * 1e3, color="tab:green", lw=1.0, ls=":",
                label="continuum (paper params)" if b == 0 else None)
    for b in range(continuum_fit.shape[1]):
        ax.plot(k_dist, continuum_fit[:, b] * 1e3, color="tab:red", lw=1.0, ls="--",
                label="continuum (fit)" if b == 0 else None)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    for kn in k_node:
        ax.axvline(kn, color="gray", lw=0.5, alpha=0.5)
    labels = [str(s) for s in np.asarray(sym_labels).tolist()]
    ax.set_xticks(k_node)
    ax.set_xticklabels(labels)
    ax.set_ylabel("E (meV)")
    ax.set_ylim(-200.0, 200.0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output path helpers (mirrors run_uq_propagation_bands.py layout)
# ---------------------------------------------------------------------------

def continuum_output_path(output_dir: str, sample_index: int) -> str:
    return os.path.join(output_dir, f"sample{sample_index:04d}.npz")


def summary_csv_path(output_dir: str) -> str:
    return os.path.join(output_dir, "fit_summary.csv")


def append_summary_row(output_dir: str, sample_index: int, params: np.ndarray,
                        perr: np.ndarray, mse: float) -> None:
    path = summary_csv_path(output_dir)
    is_new = not os.path.isfile(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            header = ["sample_index"]
            for name in PARAM_NAMES:
                header += [name, f"{name}_unc"]
            header += ["mse"]
            writer.writerow(header)
        row = [sample_index]
        for v, e in zip(params, perr):
            row += [v, e]
        row += [mse]
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fit a Nam & Koshino continuum-model Hamiltonian to each TBLG "
            "TB-model ensemble band structure (bands/propagation/), and "
            "propagate the parameter uncertainty over the ensemble."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    add_energy_models_arg(p)
    p.add_argument("--ensemble-dir", default="ensembles")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--calibration-metrics-dir", default=DEFAULT_CALIBRATION_METRICS_DIR)
    p.add_argument("--calibration-target", default="energy")
    p.add_argument("--calibration-technique", default="mcmc")

    p.add_argument("--tb-model", default=DEFAULT_TB_MODEL)
    p.add_argument("--tb-temperature", type=float, default=None)
    p.add_argument("--tb-calibration-target", default="hopping")
    p.add_argument("--tb-rcut", type=float, default=DEFAULT_TB_RCUT)

    p.add_argument("--twist-angle", "--theta", type=float, default=DEFAULT_TWIST_ANGLE, dest="twist_angle")

    p.add_argument("--trajectory-dir", default=DEFAULT_TRAJECTORY_DIR)
    p.add_argument("--bands-dir", default=DEFAULT_BANDS_DIR,
                    help="Root of TB band .npz files written by run_uq_propagation_bands.py.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Directory for fitted continuum-model parameters (.npz + fit_summary.csv).")

    p.add_argument("--flat-band-threshold-meV", type=float,
                    default=DEFAULT_FLAT_BAND_THRESHOLD_MEV,
                    help="Identify flat bands at K as all bands within this "
                         f"distance of E_F (default: {DEFAULT_FLAT_BAND_THRESHOLD_MEV} meV).")
    p.add_argument("--n-shells", type=int, default=DEFAULT_N_SHELLS,
                    help="Moire reciprocal-lattice shell truncation radius for the plane-wave basis "
                         f"(default: {DEFAULT_N_SHELLS}).")
    p.add_argument("--max-nfev", type=int, default=200)
    p.add_argument("--sample-index", type=int, default=None,
                    help="Fit only this sample index (position in the sorted trajectory list). "
                         "Default: fit every sample with both a bands .npz and a trajectory.")
    p.add_argument("--plot-comparison", action="store_true",
                    help="After fitting, save a (temporary, overwritten) plot of all TB bands, "
                         "the fitted continuum bands, and the continuum bands with the paper "
                         "parameter values.")
    p.add_argument("--plot-path", default=None,
                    help="Path for --plot-comparison output (default: <output-dir>/.../"
                         "continuum_vs_tb_bands_TMP.png).")
    p.add_argument("--seed", type=int, default=DEFAULT_ENSEMBLE_SHUFFLE_SEED)

    add_hyperparam_args(p)
    args, _unknown = p.parse_known_args()

    os.chdir(HERE)

    from blg_model_builder.ensemble_io import (
        DEFAULT_CALIBRATION_METRICS_DIR as _PBF_CALIB_DIR,
        expand_model_patterns,
        resolve_ensemble_pickle,
    )
    if args.calibration_metrics_dir == DEFAULT_CALIBRATION_METRICS_DIR:
        args.calibration_metrics_dir = _PBF_CALIB_DIR

    try:
        tb_M, tb_W, tb_canonical = _parse_tb_model_name(args.tb_model)
    except ValueError as exc:
        p.error(str(exc))

    try:
        _tb_pkl_path, tb_t_used = _resolve_tb_ensemble(
            args.tb_model, args.ensemble_dir, args.tb_temperature,
            calibration_metrics_dir=args.calibration_metrics_dir,
            calibration_technique=args.calibration_technique,
            calibration_target=args.tb_calibration_target,
        )
    except FileNotFoundError as exc:
        p.error(str(exc))
    tb_t_label = f"{tb_t_used:g}"

    models = expand_model_patterns(args.models, args.ensemble_dir)
    if not models:
        p.error("No models matched --models patterns.")

    cb = build_coupled_basis(args.n_shells)
    n_b = len(cb.basis)
    threshold_ev = float(args.flat_band_threshold_meV) * 1e-3
    print(f"Plane-wave basis: {n_b} moire-G points/layer, "
          f"Hamiltonian dim = {4 * n_b} (n_shells={args.n_shells})", flush=True)
    print(f"Fit: 9 flat-band metrics at K/Γ/M "
          f"(flat-band threshold = {args.flat_band_threshold_meV:g} meV)", flush=True)

    from ase.io import read as ase_read

    for model_name in models:
        print(f"\n{'=' * 60}\n Model: {model_name}\n{'=' * 60}", flush=True)
        try:
            _pkl_path, t_used = resolve_ensemble_pickle(
                model_name, args.ensemble_dir, args.temperature,
                calibration_metrics_dir=args.calibration_metrics_dir,
                calibration_technique=args.calibration_technique,
                calibration_target=args.calibration_target,
            )
        except FileNotFoundError as exc:
            print(f"  Warning: cannot resolve ensemble for {model_name!r}: {exc}  Skipping.",
                  file=sys.stderr)
            continue
        t_label = f"{t_used:g}"

        traj_files = _discover_traj_files(args.trajectory_dir, model_name, t_label, args.twist_angle)
        bands_dir = os.path.join(
            args.bands_dir, _safe_filename_part(model_name), f"T{t_label}",
            f"{_safe_filename_part(tb_canonical)}_tbT{tb_t_label}",
            f"theta{args.twist_angle:g}deg",
        )
        if not traj_files:
            print(f"  Warning: no trajectory files under "
                  f"{os.path.join(args.trajectory_dir, _safe_filename_part(model_name), f'T{t_label}', f'theta{args.twist_angle:g}deg')}. "
                  "Skipping.", file=sys.stderr)
            continue
        if not os.path.isdir(bands_dir):
            print(f"  Warning: no bands directory {bands_dir}. Skipping.", file=sys.stderr)
            continue

        out_dir = os.path.join(
            args.output_dir, _safe_filename_part(model_name), f"T{t_label}",
            f"{_safe_filename_part(tb_canonical)}_tbT{tb_t_label}",
            f"theta{args.twist_angle:g}deg",
        )
        os.makedirs(out_dir, exist_ok=True)

        sample_indices = (
            [int(args.sample_index)] if args.sample_index is not None
            else list(range(len(traj_files)))
        )

        tb_flatband_widths: List[float] = []
        n_done = n_skipped = 0

        for sample_idx in sample_indices:
            existing = continuum_output_path(out_dir, sample_idx)
            if os.path.isfile(existing) and args.sample_index is None:
                print(f"  [skip] sample {sample_idx:04d} — fit exists", flush=True)
                n_skipped += 1
                continue

            if sample_idx >= len(traj_files):
                print(f"  [skip] sample {sample_idx:04d}: no trajectory (only "
                      f"{len(traj_files)} found)", file=sys.stderr)
                continue
            traj_path = traj_files[sample_idx]
            bands_npz = _existing_bands_npz(bands_dir, sample_idx)
            if bands_npz is None:
                print(f"  [skip] sample {sample_idx:04d}: no bands .npz in {bands_dir}",
                      file=sys.stderr)
                continue

            print(f"  sample {sample_idx:04d}: traj={os.path.basename(traj_path)}  "
                  f"bands={os.path.basename(bands_npz)} …", end="  ", flush=True)

            try:
                atoms_init = ase_read(traj_path, index=0)
                atoms_rel = ase_read(traj_path, index=-1)
            except Exception as exc:
                print(f"FAILED reading trajectory: {exc}", flush=True)
                continue

            bands_data = np.load(bands_npz)
            tb_evals = np.asarray(bands_data["evals"], dtype=float)
            kvec_red = np.asarray(bands_data["kvec"], dtype=float)
            k_dist = np.asarray(bands_data["k_dist"], dtype=float)
            k_node = np.asarray(bands_data["k_node"], dtype=float)
            sym_labels = bands_data["sym_labels"]
            n_atoms = int(bands_data["n_atoms"])
            sym_indices = kpath_symmetry_indices(len(kvec_red))

            cell = np.asarray(atoms_rel.get_cell(), dtype=float)
            recip = _get_recip_cell(cell.T)
            kvec_cart = kvec_red @ recip
            kappas = kvec_cart[:, :2]

            G1, G2 = moire_reciprocal_vectors(cell)
            theta_rad = float(np.deg2rad(args.twist_angle))

            mask1, mask2, pos_xy, disp_xy = layer_displacements(atoms_init, atoms_rel)
            uG = six_dominant_uG(mask1, mask2, pos_xy, disp_xy, G1, G2)
            max_uG_a = max(
                float(np.linalg.norm(uG[key][l]))
                for key in uG for l in (1, 2)
            )
            setup = continuum_k_setup(cb, G1, G2, theta_rad)

            try:
                params, perr, cov, mse, x0, tb_targets, cont_metrics = fit_continuum_model(
                    kappas, tb_evals, sym_indices, threshold_ev,
                    cb, G1, G2, theta_rad, uG, setup,
                    max_nfev=args.max_nfev,
                )
            except Exception as exc:
                print(f"FAILED fit: {type(exc).__name__}: {exc}", flush=True)
                continue

            continuum_full, fermi_c = continuum_bands_fermi_shifted(
                kappas, params, cb, G1, G2, theta_rad, uG,
            )
            paper_full, _ = continuum_bands_fermi_shifted(
                kappas, paper_continuum_params(uG), cb, G1, G2, theta_rad, uG,
            )
            tb_flat_idx = identify_flat_band_indices(
                tb_sym_evals(tb_evals, sym_indices)["K"], threshold_ev,
            )
            flat_tb = float(tb_targets[0])
            flat_c = float(cont_metrics[0])
            tb_flatband_widths.append(flat_tb)

            print(
                f"done  mse={mse * 1e6:.3f} meV^2  "
                f"|u_G|_max={max_uG_a * 1e3:.1f} pm  "
                f"n_flat(tb)={tb_flat_idx.size}  "
                f"flat_width@K(tb)={flat_tb * 1e3:.1f} meV  "
                f"flat_width@K(cont.)={flat_c * 1e3:.1f} meV",
                flush=True,
            )
            print(
                "    TB fit targets Γ: "
                f"width={tb_targets[3]*1e3:.1f} meV  "
                f"upper_gap={tb_targets[4]*1e3:.1f} meV  "
                f"lower_gap={tb_targets[5]*1e3:.1f} meV",
                flush=True,
            )
            print(
                "    cont. metrics Γ: "
                f"width={cont_metrics[3]*1e3:.1f} meV  "
                f"upper_gap={cont_metrics[4]*1e3:.1f} meV  "
                f"lower_gap={cont_metrics[5]*1e3:.1f} meV",
                flush=True,
            )
            paper_p = paper_continuum_params(uG)
            paper_sym = continuum_sym_evals_fermi_shifted(
                kappas, sym_indices, paper_p, cb, G1, G2, theta_rad, uG, setup,
            )
            paper_flat_idx = identify_flat_band_indices(paper_sym["K"], threshold_ev)
            _, paper_gamma_upper, _ = flat_band_metrics_at_k(
                paper_sym["Gamma"], paper_flat_idx,
            )
            print(
                f"    paper defaults (eq. 33–34 t,t' from uG): "
                f"t={paper_p[3]:.4f} eV  t'={paper_p[4]:.4f} eV  "
                f"Γ upper_gap={paper_gamma_upper * 1e3:.1f} meV",
                flush=True,
            )
            print(
                "    " + "  ".join(
                    f"{name}={v:.4f}+/-{e:.4f}" for name, v, e in zip(PARAM_NAMES, params, perr)
                ),
                flush=True,
            )

            np.savez_compressed(
                continuum_output_path(out_dir, sample_idx),
                params=params,
                param_uncertainties=perr,
                param_covariance=cov,
                param_names=np.array(PARAM_NAMES),
                param_init=x0,
                mse=np.float64(mse),
                flat_band_threshold_meV=np.float64(args.flat_band_threshold_meV),
                fit_metric_names=np.array(FIT_METRIC_NAMES),
                tb_fit_targets=tb_targets,
                continuum_fit_metrics=cont_metrics,
                tb_flat_band_indices=tb_flat_idx,
                n_shells=np.int64(args.n_shells),
                continuum_evals_full=continuum_full,
                continuum_evals_paper=paper_full,
                tb_evals_full=tb_evals,
                kvec=kvec_red,
                k_dist=k_dist,
                k_node=k_node,
                sym_labels=sym_labels,
                sym_k_indices=np.array([sym_indices[n] for n in SYM_POINT_NAMES], dtype=np.int64),
                fermi_level_continuum=np.float64(fermi_c),
                twist_angle=np.float64(args.twist_angle),
                sample_index=np.int64(sample_idx),
                n_atoms=np.int64(n_atoms),
            )
            append_summary_row(out_dir, sample_idx, params, perr, mse)
            n_done += 1

            if args.plot_comparison:
                plot_path = args.plot_path or os.path.join(out_dir, "continuum_vs_tb_bands_TMP.png")
                plot_single_sample_comparison(
                    plot_path, k_dist, k_node, sym_labels,
                    tb_evals, continuum_full, paper_full,
                    title=f"{model_name}  T={t_label}  theta={args.twist_angle:g} deg  "
                          f"sample {sample_idx:04d}",
                )
                print(f"    Wrote {plot_path}", flush=True)

        print(f"\n  {model_name}: {n_done} fit, {n_skipped} skipped (already fit).", flush=True)
        if tb_flatband_widths:
            mean_flat_tb = float(np.mean(tb_flatband_widths))
            print(
                f"  Mean TB flat-band width over {len(tb_flatband_widths)} sample(s): "
                f"{mean_flat_tb * 1e3:.1f} meV  (compare against the continuum flat-band "
                "width using the paper's default parameters, printed below, at theta=1.05 deg "
                "-- these should agree within ~50 meV).",
                flush=True,
            )
            try:
                paper_sym = continuum_sym_evals_fermi_shifted(
                    kappas, sym_indices, paper_continuum_params(uG),
                    cb, G1, G2, theta_rad, uG, setup,
                )
                paper_flat_idx = identify_flat_band_indices(paper_sym["K"], threshold_ev)
                flat_paper = float(sym_point_fit_metrics(paper_sym, paper_flat_idx)[0])
                pp = paper_continuum_params(uG)
                print(f"  Continuum flat-band width @ K (paper defaults + eq. 33–34 t,t'): "
                      f"{flat_paper * 1e3:.1f} meV  (t={pp[3]:.4f}, t'={pp[4]:.4f} eV)", flush=True)
            except Exception:
                pass

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
