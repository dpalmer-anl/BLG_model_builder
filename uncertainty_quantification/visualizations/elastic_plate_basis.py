"""Elastic plate Fourier basis coefficients for bilayer graphene (TBLG).

Ported from ``Elastic_basis_Dan/utils.py`` (``in_plane``, ``out_of_plane``)
as used by ``basis_coeff.py``.  Coordinates / displacements are in Å;
``L_AA`` is the moiré cell vector length ``|a₁|`` in Å.
"""
from __future__ import annotations

import numpy as np


def in_plane(
    top_x,
    top_y,
    top_ux,
    top_uy,
    L_AA: float,
    num_mode: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, B)`` in-plane Fourier coefficients for each mode."""
    numa = len(top_x)
    L_AA_m = float(L_AA) * 1e-10
    q = 4.0 * np.pi / np.sqrt(3.0) / L_AA_m

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    b1 = np.array([0.0, 1.0])
    b2 = np.array([np.sqrt(3) / 2, -0.5])

    def G(m: int, n: int) -> np.ndarray:
        return q * (m * b1 + n * b2)

    A1 = -a1 / np.linalg.norm(a1)
    A2 = (a1 - a2) / np.linalg.norm(a1 - a2)
    A3 = a2 / np.linalg.norm(a2)

    q1 = G(1, 0)
    q2 = G(-1, -1)
    q3 = G(0, 1)

    u_mode_vec_1 = np.zeros((num_mode, numa))
    u_mode_vec_2 = np.zeros((num_mode, numa))
    v_mode_vec_1 = np.zeros((num_mode, numa))
    v_mode_vec_2 = np.zeros((num_mode, numa))

    def calculate_for_theta(G_temp, D_temp, mode_INDEX: int) -> None:
        for i in range(numa):
            r = np.array([top_x[i] / 1e10, top_y[i] / 1e10])
            dot_product = np.dot(G_temp, r)
            u_mode_vec_1[mode_INDEX, i] = D_temp[0] * np.sin(dot_product)
            u_mode_vec_2[mode_INDEX, i] = D_temp[0] * np.cos(dot_product)
            v_mode_vec_1[mode_INDEX, i] = D_temp[1] * np.sin(dot_product)
            v_mode_vec_2[mode_INDEX, i] = D_temp[1] * np.cos(dot_product)

    if num_mode >= 1:
        calculate_for_theta(G_temp=q1, D_temp=A1, mode_INDEX=0)
    if num_mode >= 2:
        calculate_for_theta(G_temp=q2, D_temp=A2, mode_INDEX=1)
    if num_mode >= 3:
        calculate_for_theta(G_temp=q3, D_temp=A3, mode_INDEX=2)

    in_plane_1 = np.zeros(num_mode)
    in_plane_2 = np.zeros(num_mode)

    for k in range(num_mode):
        total_1 = np.concatenate((u_mode_vec_1[k, :], v_mode_vec_1[k, :]))
        total_2 = np.concatenate((u_mode_vec_2[k, :], v_mode_vec_2[k, :]))
        norm1_sq = float(np.linalg.norm(total_1) ** 2)
        norm2_sq = float(np.linalg.norm(total_2) ** 2)
        if norm1_sq <= 0.0 or norm2_sq <= 0.0:
            continue
        for j in range(int(numa)):
            u_1 = top_ux[j] * u_mode_vec_1[k, j] / norm1_sq
            u_2 = top_ux[j] * u_mode_vec_2[k, j] / norm2_sq
            v_1 = top_uy[j] * v_mode_vec_1[k, j] / norm1_sq
            v_2 = top_uy[j] * v_mode_vec_2[k, j] / norm2_sq
            in_plane_1[k] += u_1 + v_1
            in_plane_2[k] += u_2 + v_2

    return in_plane_1, in_plane_2


def out_of_plane(
    top_x,
    top_y,
    top_uz,
    L_AA: float,
    num_mode: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(C, D)`` out-of-plane Fourier coefficients (sin, cos) per mode."""
    numa = len(top_x)
    L_AA_m = float(L_AA) * 1e-10
    q = 4.0 * np.pi / np.sqrt(3.0) / L_AA_m

    b1 = np.array([0.0, 1.0])
    b2 = np.array([np.sqrt(3) / 2, -0.5])

    def G(m: int, n: int) -> np.ndarray:
        return q * (m * b1 + n * b2)

    G_vectors = [G(1, 0), G(-1, -1), G(0, 1)]

    w_mode_vec_sin = np.zeros((num_mode, numa))
    w_mode_vec_cos = np.zeros((num_mode, numa))

    for mode_idx in range(min(num_mode, len(G_vectors))):
        G_temp = G_vectors[mode_idx]
        for i in range(numa):
            r = np.array([top_x[i] / 1e10, top_y[i] / 1e10])
            dot_product = np.dot(G_temp, r)
            w_mode_vec_sin[mode_idx, i] = np.sin(dot_product)
            w_mode_vec_cos[mode_idx, i] = np.cos(dot_product)

    out_plane_sin = np.zeros(num_mode)
    out_plane_cos = np.zeros(num_mode)

    for k in range(num_mode):
        denom_sin = float(np.sum(w_mode_vec_sin[k, :] ** 2))
        denom_cos = float(np.sum(w_mode_vec_cos[k, :] ** 2))
        if denom_sin <= 0.0 or denom_cos <= 0.0:
            continue
        for j in range(numa):
            out_plane_sin[k] += top_uz[j] * w_mode_vec_sin[k, j] / denom_sin
            out_plane_cos[k] += top_uz[j] * w_mode_vec_cos[k, j] / denom_cos

    return out_plane_sin, out_plane_cos


def _mode1_G_and_dirs(L_AA: float) -> tuple[np.ndarray, np.ndarray]:
    """Mode-1 reciprocal vector ``G`` (1/m) and in-plane direction ``A1`` (unit)."""
    L_AA_m = float(L_AA) * 1e-10
    q = 4.0 * np.pi / np.sqrt(3.0) / L_AA_m
    b1 = np.array([0.0, 1.0])
    G = q * b1  # G(1, 0)
    a1 = np.array([1.0, 0.0])
    A1 = -a1 / np.linalg.norm(a1)
    return G, A1


def mode1_displacements(
    x: np.ndarray,
    y: np.ndarray,
    L_AA: float,
    *,
    A: float = 0.0,
    D: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct mode-1 elastic-plate displacements at ``(x, y)`` (Å).

    Matches the basis used by :func:`in_plane` / :func:`out_of_plane`:

    * in-plane (coefficient ``A``): ``u = A · A₁ · sin(G₁ · r)``
    * out-of-plane (coefficient ``D``): ``w = D · cos(G₁ · r)``

    Returns ``(ux, uy, uz)`` in Å, each shape ``(N,)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    G, A1 = _mode1_G_and_dirs(L_AA)
    r_m = np.column_stack((x / 1e10, y / 1e10))
    phase = r_m @ G
    sin_p = np.sin(phase)
    cos_p = np.cos(phase)
    A = float(A)
    D = float(D)
    ux = A * A1[0] * sin_p
    uy = A * A1[1] * sin_p
    uz = D * cos_p
    return ux, uy, uz


def apply_mode1_corrugation(
    atoms,
    *,
    A: float = 0.0,
    D: float = 0.0,
    d0: float = 3.4,
    opposite_layers: bool = True,
):
    """Apply mode-1 elastic-plate corrugation to a bilayer ``Atoms`` object.

    Parameters
    ----------
    atoms
        Flat (or already-built) bilayer structure.  Layers are split by ``mol-id``
        if present, otherwise by ``z`` relative to the mean.
    A, D
        Mode-1 in-plane (``A``) and out-of-plane (``D``) amplitudes in Å, as in
        :func:`top_layer_mode1_A_D`.
    d0
        Target AB-like (minimum) interlayer separation in Å after corrugation.
        Layers are rigidly shifted so ``min`` local top−bottom separation equals
        ``d0`` (default 3.4 Å).
    opposite_layers
        If True (default), bottom layer gets ``−(ux,uy,uz)`` so the relative
        field is twice the single-layer amplitude (standard continuum TBLG).

    Returns
    -------
    ase.Atoms
        Copy of ``atoms`` with corrugated positions.
    """
    out = atoms.copy()
    pos = np.asarray(out.get_positions(wrap=False), dtype=float)
    cell = np.asarray(out.get_cell(), dtype=float)
    L_AA = float(np.linalg.norm(cell[0]))
    if not np.isfinite(L_AA) or L_AA <= 0.0:
        raise ValueError(f"Invalid L_AA={L_AA} from cell[0]")

    if out.has("mol-id"):
        mol = np.asarray(out.get_array("mol-id"))
        bot = mol == 1
        top = mol == 2
        if int(np.count_nonzero(top)) == 0 or int(np.count_nonzero(bot)) == 0:
            z_mean = float(np.mean(pos[:, 2]))
            bot = pos[:, 2] <= z_mean
            top = ~bot
    else:
        z_mean = float(np.mean(pos[:, 2]))
        bot = pos[:, 2] <= z_mean
        top = ~bot

    if int(np.count_nonzero(top)) < 1 or int(np.count_nonzero(bot)) < 1:
        raise ValueError("Need atoms in both layers to apply corrugation")

    # Evaluate mode field on each layer's reference (x, y).
    ux_t, uy_t, uz_t = mode1_displacements(
        pos[top, 0], pos[top, 1], L_AA, A=A, D=D,
    )
    pos[top, 0] += ux_t
    pos[top, 1] += uy_t
    pos[top, 2] += uz_t

    if opposite_layers:
        ux_b, uy_b, uz_b = mode1_displacements(
            pos[bot, 0], pos[bot, 1], L_AA, A=A, D=D,
        )
        pos[bot, 0] -= ux_b
        pos[bot, 1] -= uy_b
        pos[bot, 2] -= uz_b

    # Local interlayer separations: each top atom vs nearest bottom atom in xy.
    top_idx = np.where(top)[0]
    bot_idx = np.where(bot)[0]
    from ase.geometry import find_mic

    local_sep = np.empty(len(top_idx), dtype=float)
    for i, it in enumerate(top_idx):
        dvecs = pos[bot_idx] - pos[it]
        dmic, _ = find_mic(dvecs, cell)
        j = int(np.argmin(np.linalg.norm(dmic[:, :2], axis=1)))
        local_sep[i] = float(pos[it, 2] - pos[bot_idx[j], 2])

    min_sep = float(np.min(local_sep))
    shift = float(d0) - min_sep
    # Expand/contract interlayer gap rigidly so AB-like (min) sep == d0.
    pos[top_idx, 2] += 0.5 * shift
    pos[bot_idx, 2] -= 0.5 * shift

    out.set_positions(pos)
    out.info["A_mode1"] = float(A)
    out.info["D_mode1"] = float(D)
    out.info["d0"] = float(d0)
    return out


def top_layer_mode1_A_D(
    initial,
    relaxed,
    *,
    num_mode: int = 3,
) -> tuple[float, float]:
    """Mode-1 in-plane ``A`` and out-of-plane ``D`` for the top layer (Å).

    Layer assignment and reference coordinates use the **initial** (unrelaxed)
    frame; displacements are the minimum-image ``r_relaxed − r_initial``
    (in-plane PBC).  ``L_AA = |a₁|`` from the initial cell, matching
    ``basis_coeff.py``.
    """
    from ase.geometry import find_mic

    pos0 = np.asarray(initial.get_positions(wrap=False), dtype=float)
    pos1 = np.asarray(relaxed.get_positions(wrap=False), dtype=float)
    if pos0.shape != pos1.shape:
        raise ValueError(
            f"initial/relaxed shape mismatch: {pos0.shape} vs {pos1.shape}"
        )

    cell = np.asarray(initial.get_cell(), dtype=float)
    L_AA = float(np.linalg.norm(cell[0]))
    if not np.isfinite(L_AA) or L_AA <= 0.0:
        raise ValueError(f"Invalid L_AA={L_AA} from cell[0]")

    z_mean = float(np.mean(pos0[:, 2]))
    top = pos0[:, 2] > z_mean
    if int(np.count_nonzero(top)) < 3:
        raise ValueError("Fewer than 3 top-layer atoms for elastic basis fit")

    mic_disp, _ = find_mic(pos1 - pos0, cell, pbc=[True, True, False])
    mic_disp = np.asarray(mic_disp, dtype=float)

    x = pos0[top, 0]
    y = pos0[top, 1]
    ux = mic_disp[top, 0]
    uy = mic_disp[top, 1]
    uz = mic_disp[top, 2]

    A, _B = in_plane(x, y, ux, uy, L_AA, num_mode=num_mode)
    _C, D = out_of_plane(x, y, uz, L_AA, num_mode=num_mode)
    return float(A[0]), float(D[0])
