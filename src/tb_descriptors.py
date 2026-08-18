from scipy.spatial.distance import cdist
import numpy as _np   # real NumPy — never aliased; used for CPU-only ops
np = _np              # alias so existing type annotations (np.ndarray) still resolve

import h5py
import pandas as pd
import scipy.spatial as spatial
from ase import Atoms
from ase.neighborlist import NeighborList

# ── GPU / CPU array library ───────────────────────────────────────────────────
# xp  = cupy  when a GPU is present, numpy otherwise.
# _np = numpy always (ASE/NeighborList/loop bookkeeping must stay on CPU).
# Three-body scatter: GPU uses cupyx.scatter_add; CPU uses scipy sparse matmul
# (orders of magnitude faster than np.add.at for large TBLG systems).
try:
    import cupy as xp
    import cupyx
    _GPU: bool = bool(xp.cuda.is_available())
    if not _GPU:
        xp = _np
        cupyx = None  # type: ignore[assignment]
except (ImportError, Exception):
    xp = _np
    cupyx = None      # type: ignore[assignment]
    _GPU = False

import scipy.sparse as _sparse  # used by CPU three-body scatter (much faster than np.add.at)
 
 
# ── Internal helpers ─────────────────────────────────────────────────────────
 
def _chebyshev(x, M: int):
    """
    Evaluate Chebyshev polynomials T_1 … T_M at every point in *x*.
    Works with both NumPy and CuPy arrays (uses xp = module-level array lib).

    Parameters
    ----------
    x : ndarray, shape (N,)  — must lie in [-1, 1]; xp-array
    M : int
    Returns
    -------
    T : ndarray, shape (N, M) — same device as *x*
    """
    if M < 1:
        raise ValueError("M must be ≥ 1")
    T = xp.empty((len(x), M), dtype=xp.float64)
    T[:, 0] = x
    if M > 1:
        T[:, 1] = 2.0 * x * x - 1.0
    for m in range(2, M):
        T[:, m] = 2.0 * x * T[:, m - 1] - T[:, m - 2]
    return T
 
 
def _add_triplet_contributions(
    three_body, p_v_cpu, q_v_cpu, s_v_cpu, pair_v, pair_r, cheb, W: int,
) -> None:
    """
    Accumulate angular (three-body) contributions for a batch of triplets
    into ``three_body`` in-place.

    Index arrays (p_v_cpu, q_v_cpu, s_v_cpu) are always CPU/numpy — they are
    transferred to the active device here.  Keeping index computation on CPU
    avoids CuPy dynamic-allocation kernels (nonzero, boolean-mask on broadcast
    views) that can trigger CUDA_ERROR_ILLEGAL_ADDRESS at module teardown.
    The large math arrays (pair_v, pair_r, cheb, three_body) stay on device.

    Parameters
    ----------
    three_body : xp array (n_pairs, M, W)
    p_v_cpu    : (n,) numpy int64 — primary bond indices
    q_v_cpu    : (n,) numpy int64 — secondary bond indices
    s_v_cpu    : (n,) numpy float64 — sign (+1 or -1)
    pair_v     : (n_pairs, 3) xp float
    pair_r     : (n_pairs,)   xp float
    cheb       : (n_pairs, M) xp float (envelope-scaled)
    W          : int
    """
    # Transfer index arrays to device (no-op when xp = _np)
    p_v = xp.asarray(p_v_cpu)
    q_v = xp.asarray(q_v_cpu)
    s_v = xp.asarray(s_v_cpu)

    cos_th = s_v * xp.einsum('nd,nd->n', pair_v[p_v], pair_v[q_v]) / (
        pair_r[p_v] * pair_r[q_v]
    )
    xp.clip(cos_th, -1.0, 1.0, out=cos_th)
    cos_pw = xp.empty((len(p_v), W), dtype=xp.float64)
    cos_pw[:, 0] = cos_th
    for w in range(1, W):
        cos_pw[:, w] = cos_pw[:, w - 1] * cos_th
    contrib = cheb[q_v][:, :, xp.newaxis] * cos_pw[:, xp.newaxis, :]  # (n, M, W)
    if _GPU:
        cupyx.scatter_add(three_body, p_v, contrib)
    else:
        _np.add.at(three_body, p_v, contrib)
    # Local variables (cos_th, cos_pw, contrib, p_v, q_v, s_v) go out of scope here.
    # CuPy's stream-ordered memory pool ensures GPU kernels complete before memory
    # is reused — no explicit del needed.


def _build_triplet_indices(pair_i: _np.ndarray, pair_j: _np.ndarray, N: int):
    """
    For every directed half-list bond p = (i → j), enumerate secondary bonds
    from **both** atom i's and atom j's complete neighbourhoods (fully vectorised,
    GPU-accelerated when CuPy is available).

    Produces a *symmetric* three-body descriptor:

        φ₃_sym(i→j) = ½ [Σ_{k∈nb(i),k≠j} f(r_ik) T(r_ik) cosʷ θ_ijk
                        + Σ_{k∈nb(j),k≠i} f(r_jk) T(r_jk) cosʷ θ_jik]

    so φ₃_sym(i→j) = φ₃_sym(j→i) → t(i→j) = t(j→i) → zero K-point gap.

    Four contributions (cos θ always uses ``pair_v[p]`` as primary direction):

      Cond 1  i-env fwd  pair_i[q]==i, pair_j[q]≠j          t_sign = +1
      Cond 2  i-env rev  pair_j[q]==i, pair_i[q]≠j          t_sign = −1
      Cond 3  j-env fwd  pair_i[q]==j, pair_j[q]≠i          t_sign = −1
      Cond 4  j-env rev  pair_j[q]==j, pair_i[q]≠i          t_sign = +1

    The caller multiplies the accumulated three-body block by 0.5.

    Parameters
    ----------
    pair_i, pair_j : (n_pairs,) CPU int arrays  (from ASE NeighborList)
    N : total number of atoms

    Returns
    -------
    t_p, t_q : (n_triplets,) xp int64  (GPU when CuPy available, else CPU)
    t_sign   : (n_triplets,) xp float64
    """
    # Transfer index arrays from CPU (ASE output) to the active device.
    # xp = cupy on GPU, numpy on CPU — no-op when xp is numpy.
    pi = xp.asarray(pair_i, dtype=xp.int32)
    pj = xp.asarray(pair_j, dtype=xp.int32)

    n = len(pi)
    if n == 0:
        return (
            xp.empty(0, dtype=xp.int64),
            xp.empty(0, dtype=xp.int64),
            xp.empty(0, dtype=xp.float64),
        )

    # ── CSR-style bond lookup (all on xp / GPU) ─────────────────────────────
    # order_center[off_c[a] : off_c[a]+cnt_c[a]]  lists bond indices q where
    # pi[q] == a (a is centre).  Likewise for order_target / off_t / cnt_t.
    order_center = xp.argsort(pi, kind='stable').astype(xp.int64)
    order_target = xp.argsort(pj, kind='stable').astype(xp.int64)

    cnt_c = xp.bincount(pi, minlength=N).astype(xp.int64)
    cnt_t = xp.bincount(pj, minlength=N).astype(xp.int64)

    off_c = xp.empty(N, dtype=xp.int64)
    off_t = xp.empty(N, dtype=xp.int64)
    off_c[0] = off_t[0] = 0
    if N > 1:
        xp.cumsum(cnt_c[:-1], out=off_c[1:])
        xp.cumsum(cnt_t[:-1], out=off_t[1:])

    # ── Vectorised expand-and-filter ────────────────────────────────────────
    # For each primary bond p, expand every secondary bond q from the group
    # selected by atom_sel[p], then discard pairs where excl_q[q]==excl_p[p].

    t_p_parts: list = []
    t_q_parts: list = []
    s_parts:   list = []

    # Single cumulative buffer, preallocated and reused across 4 _expand calls.
    _cum = xp.empty(n + 1, dtype=xp.int64)

    def _expand(atom_sel, off, cnt, order, excl_p, excl_q, sign: float) -> None:
        grp_size = cnt[atom_sel]      # (n,) secondaries per primary bond
        total    = int(grp_size.sum())   # D2H sync — needed for xp.arange(total)
        if total == 0:
            return

        # Prefix cumulative sum: _cum[p] = Σ grp_size[0:p]
        _cum[0] = 0
        xp.cumsum(grp_size, out=_cum[1:])

        # Expand each primary bond p by grp_size[p] without xp.repeat(array):
        # CuPy rejects ndarray `repeats`, so map flat index g → primary p via searchsorted.
        flat    = xp.arange(total, dtype=xp.int64)
        t_p_raw = xp.searchsorted(_cum, flat, side='right') - 1     # (total,)

        # Secondary bond index = CSR_start[p] + intra_offset
        idx_arr = off[atom_sel[t_p_raw]] + flat - _cum[t_p_raw]     # (total,)
        t_q_raw = order[idx_arr]

        # Drop the primary partner *image*, not every bond to the same atom
        # index (other periodic images of j must remain as k-legs).
        keep  = excl_q[t_q_raw] != excl_p[t_p_raw]
        n_keep = int(keep.sum())                               # D2H sync for xp.full
        if n_keep == 0:
            return
        t_p_parts.append(t_p_raw[keep])
        t_q_parts.append(t_q_raw[keep])
        s_parts.append(xp.full(n_keep, sign, dtype=xp.float64))

    # Condition 1: i-env fwd  group=center[pi], excl pj[q]==pj[p]
    _expand(pi, off_c, cnt_c, order_center, pj, pj, +1.0)
    # Condition 2: i-env rev  group=target[pi], excl pi[q]==pj[p]
    _expand(pi, off_t, cnt_t, order_target, pj, pi, -1.0)
    # Condition 3: j-env fwd  group=center[pj], excl pj[q]==pi[p]
    _expand(pj, off_c, cnt_c, order_center, pi, pj, -1.0)
    # Condition 4: j-env rev  group=target[pj], excl pi[q]==pi[p]
    _expand(pj, off_t, cnt_t, order_target, pi, pi, +1.0)

    if not t_p_parts:
        return (
            xp.empty(0, dtype=xp.int64),
            xp.empty(0, dtype=xp.int64),
            xp.empty(0, dtype=xp.float64),
        )

    return (
        xp.concatenate(t_p_parts),
        xp.concatenate(t_q_parts),
        xp.concatenate(s_parts),
    )
 
 
# ── Public API ───────────────────────────────────────────────────────────────
 
def get_acsf_hopping_descriptors(
    atoms: Atoms,
    *,
    M: int = 8,
    W: int = 3,
    r_cut: float = 6.0,
    r_inner_cut: float = 1.0,
    use_envelope: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute two-body and three-body ACSF descriptors for every directed
    neighbour pair (i, j) within *r_cut*.
 
    Parameters
    ----------
    atoms : ase.Atoms
        Structure to describe.
    M : int
        Number of Chebyshev radial basis functions (T_1 … T_M).
    W : int
        Number of angular exponents (cos^1 … cos^W).  Use ``W=0`` for a
        two-body-only descriptor (no three-body / angular block).
    r_cut : float
        Cutoff radius in Å.  Pairs with r_ij > r_cut are excluded.
    use_envelope : bool
        Multiply radial functions by the smooth cosine envelope
        f_c(r) = 0.5·(cos(π r / r_cut) + 1) so descriptors vanish at r_cut.
 
    Returns
    -------
    descriptors : ndarray, shape (n_pairs, M + M*W)
        Each row is the descriptor vector for directed pair (i, j).
        Columns 0…M-1 are the two-body block;
        columns M…M+M·W-1 are the three-body block (empty when ``W=0``).
    pair_indices : (i_array, j_array, v_array)
        ``i_array``, ``j_array``: centre and neighbour indices, shape (n_pairs,) each.
        ``v_array``: bond vectors (Angstrom) row-aligned with ``descriptors``, same as
        ASE ``NeighborList`` convention (minimal image / wrap), shape (n_pairs, 3).
        Used with cell vectors to recover ``(di, dj)`` and match HDF5 tight-binding keys.
    """
    if M < 1:
        raise ValueError("M must be ≥ 1")
    if W < 0:
        raise ValueError("W must be ≥ 0")

    n_desc = M + M * W

    N = len(atoms)
    pos  = atoms.get_positions()   # _np (ASE always returns NumPy)
    cell = atoms.get_cell()        # _np

    # ── Build neighbour list (always CPU — ASE/NeighborList are NumPy-only) ─
    nl = NeighborList(
        [r_cut / 2.0] * N,
        skin=0.0,
        self_interaction=False,
        bothways=False,  # half-list; caller adds H† when building Hamiltonian
    )
    nl.update(atoms)

    i_chunks, j_chunks, r_chunks, vec_chunks = [], [], [], []
    for i in range(N):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) == 0:
            continue
        vecs = pos[indices] - pos[i] + offsets @ cell       # (n_nb, 3)
        rs   = _np.linalg.norm(vecs, axis=1)                # (n_nb,)
        i_chunks.append(_np.full(len(indices), i, dtype=_np.int32))
        j_chunks.append(indices.astype(_np.int32))
        r_chunks.append(rs)
        vec_chunks.append(vecs)

    if not i_chunks:
        empty_idx = _np.array([], dtype=_np.int32)
        empty_v   = _np.empty((0, 3), dtype=_np.float64)
        return _np.empty((0, n_desc)), (empty_idx, empty_idx, empty_v)

    # Integer index arrays stay on CPU (used for Hamiltonian scatter indexing)
    pair_i = _np.concatenate(i_chunks)   # (n_pairs,) int32
    pair_j = _np.concatenate(j_chunks)   # (n_pairs,) int32

    # Float arrays → GPU (no-op when xp = _np)
    pair_r = xp.asarray(_np.concatenate(r_chunks),   dtype=xp.float64)  # (n_pairs,)
    pair_v = xp.asarray(_np.concatenate(vec_chunks), dtype=xp.float64)  # (n_pairs, 3)
    n_pairs = len(pair_i)

    # ── Chebyshev radial basis ──────────────────────────────────────────────
    x_ij = (2.0 * pair_r - (r_inner_cut + r_cut)) / (r_cut - r_inner_cut)
    cheb  = _chebyshev(x_ij, M)                              # (n_pairs, M)
    del x_ij

    if use_envelope:
        fc   = 0.5 * (xp.cos(_np.pi * pair_r / r_cut) + 1.0)
        cheb = cheb * fc[:, xp.newaxis]
        del fc

    two_body = cheb  # (n_pairs, M)  — alias; cheb kept for three-body use below

    if W == 0:
        del pair_r
        return two_body, (pair_i, pair_j, pair_v)

    # ── Three-body block (atom-by-atom loop) ────────────────────────────────
    # Peak GPU memory is O((bonds/atom)² × M × W) rather than O(n_triplets × M × W),
    # making the computation scalable to large supercells.
    #
    # For each atom a we form the cross-product of:
    #   Group A — primary bonds p where pair_i[p]==a (a is bond centre)
    #             vs. a's full environment (fwd: sign +1, rev: sign -1)
    #   Group B — primary bonds p where pair_j[p]==a (a is bond target)
    #             vs. a's full environment (fwd: sign -1, rev: sign +1)
    # and scatter-add the angular contributions into three_body[p].
    #
    # Exclude the primary partner *image* (same Cartesian endpoint), not the
    # partner atom index.  Index exclusion drops other periodic images of j
    # that still lie inside r_cut — wrong for small PBC training cells (N=4)
    # while large TBLG cells keep those k-legs.

    three_body = xp.zeros((n_pairs, M, W), dtype=xp.float64)

    # CSR bond lookup — built entirely on CPU.
    # All index arithmetic (nonzero, masking, concatenation) stays on CPU to avoid
    # CuPy dynamic-allocation kernels that trigger CUDA_ERROR_ILLEGAL_ADDRESS.
    order_center = _np.argsort(pair_i, kind='stable').astype(_np.int64)
    order_target = _np.argsort(pair_j, kind='stable').astype(_np.int64)
    cnt_c = _np.bincount(pair_i, minlength=N).astype(_np.int64)
    cnt_t = _np.bincount(pair_j, minlength=N).astype(_np.int64)
    off_c = _np.empty(N, dtype=_np.int64); off_c[0] = 0
    off_t = _np.empty(N, dtype=_np.int64); off_t[0] = 0
    if N > 1:
        _np.cumsum(cnt_c[:-1], out=off_c[1:])
        _np.cumsum(cnt_t[:-1], out=off_t[1:])

    # Bond vectors on CPU for image-aware k-leg exclusion.
    pair_v_cpu = pair_v.get() if hasattr(pair_v, "get") else pair_v
    pair_v_cpu = _np.asarray(pair_v_cpu, dtype=_np.float64)
    _same_image_tol2 = 1e-16  # Å²

    for a in range(N):
        n_pc = int(cnt_c[a])   # bonds where pair_i[q]==a
        n_pt = int(cnt_t[a])   # bonds where pair_j[q]==a
        n_env = n_pc + n_pt
        if n_env < 2:
            continue

        # Bond indices for atom a's environment — CPU int64 views (no copy)
        c0 = int(off_c[a]); t0 = int(off_t[a])
        pc = order_center[c0 : c0 + n_pc]
        pt = order_target[t0 : t0 + n_pt]

        q_env = _np.concatenate([pc, pt]).astype(_np.int64)  # (n_env,)
        # Displacement from a to the other end of each env bond.
        k_pos = _np.concatenate(
            [pair_v_cpu[pc], -pair_v_cpu[pt]], axis=0,
        )  # (n_env, 3)

        # ── Group A: p ∈ pc (pair_i[p]==a), primary direction = +pair_v[p] ─
        # sign: +1 for fwd secondaries (q∈pc), -1 for rev secondaries (q∈pt)
        if n_pc > 0:
            sign_A = _np.concatenate([
                _np.ones(n_pc,  dtype=_np.float64),
                -_np.ones(n_pt, dtype=_np.float64),
            ])
            j_pos = pair_v_cpu[pc]  # (n_pc, 3)
            d2 = _np.sum((k_pos[_np.newaxis, :, :] - j_pos[:, _np.newaxis, :]) ** 2, axis=2)
            mask_A = d2 > _same_image_tol2
            idx_p, idx_q = _np.nonzero(mask_A)
            if idx_p.size > 0:
                _add_triplet_contributions(
                    three_body,
                    pc[idx_p],        # CPU int64
                    q_env[idx_q],     # CPU int64
                    sign_A[idx_q],    # CPU float64
                    pair_v, pair_r, cheb, W,
                )

        # ── Group B: p ∈ pt (pair_j[p]==a), primary direction = −pair_v[p] ─
        # sign: -1 for fwd secondaries (q∈pc), +1 for rev secondaries (q∈pt)
        if n_pt > 0:
            sign_B = _np.concatenate([
                -_np.ones(n_pc, dtype=_np.float64),
                _np.ones(n_pt,  dtype=_np.float64),
            ])
            j_pos = -pair_v_cpu[pt]  # (n_pt, 3)  a → i
            d2 = _np.sum((k_pos[_np.newaxis, :, :] - j_pos[:, _np.newaxis, :]) ** 2, axis=2)
            mask_B = d2 > _same_image_tol2
            idx_p, idx_q = _np.nonzero(mask_B)
            if idx_p.size > 0:
                _add_triplet_contributions(
                    three_body,
                    pt[idx_p],        # CPU int64
                    q_env[idx_q],     # CPU int64
                    sign_B[idx_q],    # CPU float64
                    pair_v, pair_r, cheb, W,
                )

    # Free CSR tables — no longer needed after the loop.
    del order_center, order_target, cnt_c, cnt_t, off_c, off_t
    del pair_r

    # The factor of 0.5 averages i's and j's environments (ensures t(i→j) = t(j→i)).
    three_body *= 0.5 * cheb[:, :, xp.newaxis]

    # ── Assemble ────────────────────────────────────────────────────────────
    descriptors = xp.concatenate(
        [two_body, three_body.reshape(n_pairs, M * W)], axis=1
    )
    del three_body
    # pair_i / pair_j: CPU int arrays for Hamiltonian indexing
    # pair_v, descriptors: xp-arrays (GPU or CPU depending on _GPU)
    return descriptors, (pair_i, pair_j, pair_v)

def get_acsf_sk_hopping_descriptors(
    atoms: Atoms,
    *,
    M: int = 8,
    W: int = 3,
    r_cut: float = 6.0,
    r_inner_cut: float = 1.0,
    use_envelope: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute two-body and three-body ACSF descriptors for every directed
    neighbour pair (i, j) within *r_cut*.
 
    Parameters
    ----------
    atoms : ase.Atoms
        Structure to describe.
    M : int
        Number of Chebyshev radial basis functions (T_1 … T_M).
    W : int
        Number of angular exponents (cos^1 … cos^W).  Use ``W=0`` for a
        two-body-only descriptor (no three-body / angular block).
    r_cut : float
        Cutoff radius in Å.  Pairs with r_ij > r_cut are excluded.
    use_envelope : bool
        Multiply radial functions by the smooth cosine envelope
        f_c(r) = 0.5·(cos(π r / r_cut) + 1) so descriptors vanish at r_cut.
 
    Returns
    -------
    descriptors : ndarray, shape (n_pairs, M + M*W)
        Each row is the descriptor vector for directed pair (i, j).
        Columns 0…M-1 are the two-body block;
        columns M…M+M·W-1 are the three-body block (empty when ``W=0``).
    pair_indices : (i_array, j_array, v_array)
        ``i_array``, ``j_array``: centre and neighbour indices, shape (n_pairs,) each.
        ``v_array``: bond vectors (Angstrom) row-aligned with ``descriptors``, same as
        ASE ``NeighborList`` convention (minimal image / wrap), shape (n_pairs, 3).
        Used with cell vectors to recover ``(di, dj)`` and match HDF5 tight-binding keys.
    """
    descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
        atoms, M=M, W=W, r_cut=r_cut, r_inner_cut=r_inner_cut, use_envelope=use_envelope,
    )
    # pair_v is an xp-array; use xp throughout (GPU or CPU transparently)
    n  = pair_v[:, 2] / xp.linalg.norm(pair_v, axis=1)   # direction cosine with ẑ
    n2 = n[:, xp.newaxis] ** 2
    descriptors_sk = xp.concatenate(
        [descriptors * (1.0 - n2), descriptors * n2], axis=1
    )
    return descriptors_sk, (pair_i, pair_j, pair_v)

def nnmat(lattice_vectors, atomic_basis):
    """
    Build matrix which tells you relative coordinates
    of nearest neighbors to an atom i in the supercell

    Returns: nnmat [natom x 3 x 3]
    """
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = np.asarray(atomic_basis)
    nnmat = np.zeros((len(atomic_basis), 3, 3))

    # Extend atom list
    atoms = []
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            displaced_atoms = atomic_basis + lattice_vectors[np.newaxis, 0] * i + lattice_vectors[np.newaxis, 1] * j
            atoms += [list(x) for x in displaced_atoms]
    atoms = np.array(atoms)
    atomic_basis = np.array(atomic_basis)

    # Loop
    for i in range(len(atomic_basis)):
        displacements = atoms - atomic_basis[i]
        distances = np.linalg.norm(displacements,axis=1)
        ind = np.argsort(distances)
        nnmat[i] = displacements[ind[1:4]]

    return nnmat
#@njit
def ix_to_dist(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Converts displacement indices to physical distances
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    dxy - Distance in Bohr, projected in the x/y plane
    dz  - Distance in Bohr, projected onto the z axis
    """
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = np.asarray(atomic_basis)

    displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]

    displacement_vector_xy = displacement_vector[:, :2] 
    displacement_vector_z =  displacement_vector[:, -1] 

    dxy = np.linalg.norm(displacement_vector_xy, axis = 1)
    dz = np.abs(displacement_vector_z)
    return dxy, dz

def get_disp(atoms,units = "angstroms",cutoff=6,type="all"):
    if units == "bohr":
        conversion = 1.0/.529177
    elif units == "angstroms":
        conversion = 1
    positions = np.asarray(atoms.positions*conversion)
    natoms = len(atoms)
    cell = np.asarray(atoms.get_cell()*conversion)
    atom_types = np.asarray(atoms.get_array("mol-id"))

    di = []
    dj = []
    extended_coords = []
    num_lat_vec_1 = np.ceil(cutoff/(np.linalg.norm(cell[0])/2))
    num_lat_vec_2 = np.ceil(cutoff/(np.linalg.norm(cell[1])/2))
    lat_vec_iter_1 = np.arange(-num_lat_vec_1,num_lat_vec_1+1) #[-1,0,1] #
    lat_vec_iter_2 = np.arange(-num_lat_vec_2,num_lat_vec_2+1) #[-1,0,1] #

    for dx in lat_vec_iter_1:
        for dy in lat_vec_iter_2:
            extended_coords += list(positions[:, :] + cell[0, np.newaxis] * dx + cell[1, np.newaxis] * dy)
            di += [dx] * natoms
            dj += [dy] * natoms
    distances = cdist(positions, extended_coords)

    i, j = np.where((distances > 0.529)  & (distances < cutoff))
    di = np.asarray(di)[j]
    dj = np.asarray(dj)[j]
    i  = np.asarray(i)
    j  = np.asarray(j % natoms)
    if type=="all":
        disp =  di[:, np.newaxis] * cell[0] +\
                dj[:, np.newaxis] * cell[1] +\
                positions[j] - positions[i]
        return disp,i,j,di,dj
    
    elif type=="intralayer":
        intra_valid_indices = atom_types[i] == atom_types[j]
        intra_indi = i[intra_valid_indices]
        intra_indj =j[intra_valid_indices]
        intra_disp = di[intra_valid_indices, np.newaxis] * cell[0] +\
                        dj[intra_valid_indices, np.newaxis] * cell[1] +\
                        positions[intra_indj] - positions[intra_indi]
        intra_di = di[intra_valid_indices]
        intra_dj = dj[intra_valid_indices]
        return intra_disp,intra_indi,intra_indj,intra_di,intra_dj

    elif type=="interlayer":
        inter_valid_indices = atom_types[i] != atom_types[j]
        inter_indi = i[inter_valid_indices]
        inter_indj = j[inter_valid_indices]
        inter_di = di[inter_valid_indices]
        inter_dj = dj[inter_valid_indices]

        inter_disp = di[inter_valid_indices, np.newaxis] * cell[0] +\
                            dj[inter_valid_indices, np.newaxis] * cell[1] +\
                            positions[inter_indj] - positions[inter_indi]

        return inter_disp,inter_indi,inter_indj,inter_di,inter_dj

#@njit
def triangle_height(a, base):
    """
    Give area of a triangle given two displacement vectors for 2 sides
    """
     
    area = np.linalg.det(
            np.asarray([a, base, np.asarray([1, 1, 1])])
    )
    area = np.abs(area)/2
    height = 2 * area / np.linalg.norm(base)
    return height
#@njit
def t01_descriptors(lattice_vectors, atomic_basis, di, dj, ai, aj):
    # Compute NN distances
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = np.asarray(atomic_basis)
    r = di[:, np.newaxis] * lattice_vectors[0] + dj[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    a = np.linalg.norm(r, axis = 1)
    return {'a': a}
#@njit
def t02_descriptors(lattice_vectors,atomic_basis,di,dj, ai, aj):
    # Compute NNN distances
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = np.asarray(atomic_basis)
    r = di[:, np.newaxis] * lattice_vectors[0] + dj[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai]

    b = np.linalg.norm(r, axis = 1)

    # Compute h
    h1 = []
    h2 = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[aj[i]] + r[i]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        h1.append(triangle_height(nn[ind[0]], r[i]))
        h2.append(triangle_height(nn[ind[1]], r[i]))
    return {'h1': h1, 'h2': h2, 'b': b}
#@njit
def t03_descriptors(lattice_vectors,atomic_basis,di, dj, ai, aj):
    """
    Compute t03 descriptors
    """
    # Compute NNNN distances
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = np.asarray(atomic_basis)
    r = di[:, np.newaxis] * lattice_vectors[0] + dj[:, np.newaxis] * lattice_vectors[1] +\
        atomic_basis[aj] - atomic_basis[ai] # Relative coordinates
    c = np.linalg.norm(r, axis = 1)

    # All other hexagon descriptors
    l = []
    h = []
    mat = nnmat(lattice_vectors, atomic_basis)
    for i in range(len(r)):
        nn = mat[aj[i]] + r[i]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        b = nndist[ind[0]]
        d = nndist[ind[1]]
        h3 = triangle_height(nn[ind[0]], r[i])
        h4 = triangle_height(nn[ind[1]], r[i])

        nn = r[i] - mat[ai[i]]
        nndist = np.linalg.norm(nn, axis = 1)
        ind = np.argsort(nndist)
        a = nndist[ind[0]]
        e = nndist[ind[1]]
        h1 = triangle_height(nn[ind[0]], r[i])
        h2 = triangle_height(nn[ind[1]], r[i])

        l.append((a + b + d + e)/4)
        h.append((h1 + h2 + h3 + h4)/4)
    return {'c': c, 'h': h, 'l': l}
#@njit
def letb_intralayer_descriptors(atoms,cutoff=6) : #lattice_vectors, atomic_basis, di, dj, ai, aj):
    """ 
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    # Partition 
    ang_per_bohr = 1 #0.529
    disp,i,j,di,dj = get_disp(atoms,type="intralayer",cutoff=cutoff)
    distances = np.linalg.norm(disp,axis=1)/ang_per_bohr
    min_distance = min(distances)

    # NN should be within 5% of min_distance
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (distances >= 0.95 * np.sqrt(3) * min_distance) & (distances <= 1.05 * np.sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
   
    # Anything else, we zero out
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)

    # Compute descriptors
    t01 = t01_descriptors(atoms.get_cell()/ang_per_bohr, atoms.positions/ang_per_bohr, di[t01_ix], dj[t01_ix], i[t01_ix], j[t01_ix])
    t02 = t02_descriptors(atoms.get_cell()/ang_per_bohr, atoms.positions/ang_per_bohr, di[t02_ix], dj[t02_ix], i[t02_ix], j[t02_ix])
    t03 = t03_descriptors(atoms.get_cell()/ang_per_bohr, atoms.positions/ang_per_bohr, di[t03_ix], dj[t03_ix], i[t03_ix], j[t03_ix])
    return (t01, t02, t03,distances), i,j,di,dj

def letb_intralayer_descriptors_array(lattice_vectors, disp,atomic_basis, di, dj, i, j,nn_val=None) :
    """ 
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    # Partition 
    ang_per_bohr = 1 #0.529
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = np.asarray(atomic_basis)
    disp/= ang_per_bohr
    distances = np.linalg.norm(disp,axis=1)
    min_distance = min(distances)

    # NN should be within 5% of min_distance
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (distances >= 0.95 * np.sqrt(3) * min_distance) & (distances <= 1.05 * np.sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
   
    # Anything else, we zero out
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)

    # Compute descriptors
    t01 = t01_descriptors(lattice_vectors/ang_per_bohr, atomic_basis/ang_per_bohr, di[t01_ix], dj[t01_ix], i[t01_ix], j[t01_ix])
    t02 = t02_descriptors(lattice_vectors/ang_per_bohr, atomic_basis/ang_per_bohr, di[t02_ix], dj[t02_ix], i[t02_ix], j[t02_ix])
    t03 = t03_descriptors(lattice_vectors/ang_per_bohr, atomic_basis/ang_per_bohr, di[t03_ix], dj[t03_ix], i[t03_ix], j[t03_ix])
    if nn_val ==1:
        return t01["a"], t01_ix
    elif nn_val==2:
        return np.vstack([np.asarray(t02[key]) for key in t02]).T, t02_ix
    elif nn_val ==3:
        return np.vstack([np.asarray(t03[key]) for key in t03]).T, t03_ix
    else:
        return [t01, t02, t03,distances], np.concatenate((t01_ix,t02_ix,t03_ix))



def ix_to_orientation(lattice_vectors, atomic_basis, di, dj, ai, aj):
    """
    Converts displacement indices to orientations of the 
    nearest neighbor environments using definitions in 
    Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

    theta_12 - Orientation of upper-layer relative to bond length
    theta_21 - Orientation of lower-layer relative to bond length
    """
    import scipy.spatial as spatial
    displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]
    mat = nnmat(lattice_vectors, atomic_basis)

    # Compute distances and angles
    theta_12 = []
    theta_21 = []
    for disp, i, j, inn, jnn in zip(displacement_vector, ai, aj, mat[ai], mat[aj]):
        sin_jnn = np.cross(jnn[:,:2], disp[:2]) 
        sin_inn = np.cross(inn[:,:2], disp[:2]) 
        cos_jnn = np.dot(jnn[:,:2], disp[:2]) 
        cos_inn = np.dot(inn[:,:2], disp[:2]) 
        theta_jnn = np.arctan2(sin_jnn, cos_jnn)
        theta_inn = np.arctan2(sin_inn, cos_inn)

        theta_12.append(np.pi - theta_jnn[0])
        theta_21.append(theta_inn[0])
    return np.asarray(theta_12), np.asarray(theta_21)
#@njit
def letb_interlayer_descriptors(atoms,cutoff=6):
    """
    Build bi-layer descriptors given geometric quantities
        lattice_vectors - lattice_vectors of configuration
        atomic_basis - atomic basis of configuration
        di, dj - lattice_vector displacements between pair i, j
        ai, aj - basis elements for pair i, j
    """
    ang_per_bohr = 1 #0.529
    lattice_vectors = atoms.get_cell()/ang_per_bohr
    lattice_vectors = np.asarray(lattice_vectors)
    atomic_basis = atoms.positions/ang_per_bohr
    atomic_basis = np.asarray(atomic_basis)
    disp,i,j,di,dj = get_disp(atoms)
    disp/= ang_per_bohr

    dist_xy = np.linalg.norm(disp[:,:2],axis=1)
    dist_z = np.abs(disp[:,2])
    dist = np.linalg.norm(disp,axis=1)
    
    output = {
        'dxy': [], # Distance in Bohr, xy plane
        'dz': [],  # Distance in Bohr, z
        'd': [],   # Distance in Bohr 
        'theta_12': [], # Orientation of upper layer NN environment
        'theta_21': [], # Orientation of lower layer NN environment
    }
    

    # Many-body terms
    theta_12, theta_21 = ix_to_orientation(lattice_vectors, atomic_basis, di, dj, i, j)

    # Return pandas DataFrame
    #df = pd.DataFrame(output)
    atom_types = np.asarray(atoms.get_array("mol-id"))
    inter_valid_indices = atom_types[i] != atom_types[j]
    inter_indi = i[inter_valid_indices]
    inter_indj = j[inter_valid_indices]
    inter_di = di[inter_valid_indices]
    inter_dj = dj[inter_valid_indices]
    #df = df[inter_valid_indices]
    output['dxy'] = dist_xy[inter_valid_indices]
    output['dz'] = dist_z[inter_valid_indices]
    output['d'] = dist[inter_valid_indices]
    output["theta_12"] = theta_12[inter_valid_indices]
    output['theta_21'] = theta_21[inter_valid_indices]
    
    return output,inter_indi,inter_indj,inter_di,inter_dj

def letb_interlayer_descriptors_array(lattice_vectors, disp,atomic_basis, di, dj, i, j):
    output = {
        'dxy': [], # Distance in Bohr, xy plane
        'dz': [],  # Distance in Bohr, z
        'd': [],   # Distance in Bohr 
        'theta_12': [], # Orientation of upper layer NN environment
        'theta_21': [], # Orientation of lower layer NN environment
    }
    output = np.zeros((len(i),3))
    # 2-body terms
    ang_per_bohr = 1 #0.529
    disp/= ang_per_bohr
    dist_xy = np.linalg.norm(disp[:,:2],axis=1)
    dist_z = np.abs(disp[:,2])
    dist = np.linalg.norm(disp,axis=1)
    #output[:,0] = np.array(dist_xy)
    #output[:,1] = np.array(dist_z)
    output[:,0] = np.array(dist)

    # Many-body terms
    theta_12, theta_21 = ix_to_orientation(lattice_vectors/ang_per_bohr, atomic_basis/ang_per_bohr, di, dj, i, j)
    output[:,1] = np.array(theta_12)
    output[:,2] = np.array(theta_21)
   
    # Return pandas DataFrame
    #df = pd.DataFrame(output)
    # key = d,theta_12, theta_21
    return output