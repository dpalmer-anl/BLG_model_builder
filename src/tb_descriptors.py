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
 
 
def _build_triplet_indices(pair_i, N: int):
    """
    For every directed pair p = (i, j), enumerate all directed pairs
    q = (i, k) with k ≠ j (same centre atom i, different neighbour).

    CPU-only (Python loops + _np).  Returns plain NumPy int64 arrays.
    Callers move t_p / t_q to GPU themselves after this call.
    """
    # pair_i must be a CPU array — guard in case caller passes a CuPy array
    if hasattr(pair_i, 'get'):        # CuPy array
        pair_i = pair_i.get()
    pair_i = _np.asarray(pair_i)

    atom_pairs = [[] for _ in range(N)]
    for p, i in enumerate(pair_i):
        atom_pairs[i].append(p)

    t_p_chunks, t_q_chunks = [], []
    for nb in atom_pairs:
        if len(nb) < 2:
            continue
        arr = _np.asarray(nb, dtype=_np.int64)
        pp, qq = _np.meshgrid(arr, arr, indexing="ij")
        mask = pp != qq
        t_p_chunks.append(pp[mask])
        t_q_chunks.append(qq[mask])

    if not t_p_chunks:
        return _np.empty(0, dtype=_np.int64), _np.empty(0, dtype=_np.int64)

    return _np.concatenate(t_p_chunks), _np.concatenate(t_q_chunks)
 
 
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
        Number of angular exponents (cos^1 … cos^W).
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
        columns M…M+M·W-1 are the three-body block.
    pair_indices : (i_array, j_array, v_array)
        ``i_array``, ``j_array``: centre and neighbour indices, shape (n_pairs,) each.
        ``v_array``: bond vectors (Angstrom) row-aligned with ``descriptors``, same as
        ASE ``NeighborList`` convention (minimal image / wrap), shape (n_pairs, 3).
        Used with cell vectors to recover ``(di, dj)`` and match HDF5 tight-binding keys.
    """
    if M < 1 or W < 1:
        raise ValueError("M and W must be ≥ 1")

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
        return _np.empty((0, M + M * W)), (empty_idx, empty_idx, empty_v)

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

    if use_envelope:
        fc   = 0.5 * (xp.cos(_np.pi * pair_r / r_cut) + 1.0)
        cheb = cheb * fc[:, xp.newaxis]

    two_body = cheb  # (n_pairs, M)

    # ── Three-body block ────────────────────────────────────────────────────
    three_body = xp.zeros((n_pairs, M, W), dtype=xp.float64)

    # _build_triplet_indices runs on CPU; move indices to xp afterwards
    t_p_cpu, t_q_cpu = _build_triplet_indices(pair_i, N)

    if len(t_p_cpu) > 0:
        t_p = xp.asarray(t_p_cpu)   # no-op when xp=_np
        t_q = xp.asarray(t_q_cpu)

        v_p = pair_v[t_p]           # (n_triplets, 3)
        v_q = pair_v[t_q]
        r_p = pair_r[t_p]           # (n_triplets,)
        r_q = pair_r[t_q]

        cos_theta = xp.einsum("nd,nd->n", v_p, v_q) / (r_p * r_q)
        xp.clip(cos_theta, -1.0, 1.0, out=cos_theta)

        cos_pw = xp.empty((len(t_p_cpu), W), dtype=xp.float64)
        cos_pw[:, 0] = cos_theta
        for w in range(1, W):
            cos_pw[:, w] = cos_pw[:, w - 1] * cos_theta

        cheb_q  = cheb[t_q]                                  # (n_triplets, M)
        contrib = cheb_q[:, :, xp.newaxis] * cos_pw[:, xp.newaxis, :]  # (n_triplets, M, W)

        # Scatter Σ_k contributions into j-leg pair slots.
        # GPU: cupyx.scatter_add (fast).
        # CPU: scipy sparse matmul — drastically faster than np.add.at for large arrays.
        if _GPU:
            cupyx.scatter_add(three_body, t_p, contrib)
        else:
            nt = len(t_p_cpu)
            # Build (n_pairs × n_triplets) sparse indicator: entry [p, k] = 1 when t_p[k] == p
            smat = _sparse.csr_matrix(
                (_np.ones(nt, dtype=_np.float64), (t_p_cpu, _np.arange(nt))),
                shape=(n_pairs, nt),
            )
            three_body += (smat @ contrib.reshape(nt, M * W)).reshape(n_pairs, M, W)

        three_body *= cheb[:, :, xp.newaxis]

    # ── Assemble ────────────────────────────────────────────────────────────
    descriptors = xp.concatenate(
        [two_body, three_body.reshape(n_pairs, M * W)], axis=1
    )
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
        Number of angular exponents (cos^1 … cos^W).
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
        columns M…M+M·W-1 are the three-body block.
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