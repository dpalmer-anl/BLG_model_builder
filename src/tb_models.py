from scipy.spatial.distance import cdist
import numpy as np    # always real NumPy (used by non-GPU functions throughout)
import matplotlib.pyplot as plt
import torch

# ── GPU / CPU array library ───────────────────────────────────────────────────
# xp = cupy when a GPU is present, numpy otherwise.
# Only the ACSF hopping functions use xp; all other functions keep using np.
try:
    import cupy as xp
    _GPU: bool = bool(xp.cuda.is_available())
    if not _GPU:
        xp = np
except (ImportError, Exception):
    xp = np
    _GPU = False

###########################################################################################

# ACSF hoppings

##########################################################################################

def get_acsf_hoppings(
    descriptors: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a linear model to predict hopping amplitudes from descriptors.
 
    hopping(i,j) = descriptors[p] @ params
 
    Parameters
    ----------
    descriptors : ndarray, shape (n_pairs, n_features)
        Output of ``get_acsf_hopping_descriptors``.
    params : ndarray, shape (n_features,) or (n_features, n_outputs)
        Linear model weights.
        • 1-D → one scalar hopping amplitude per pair.
        • 2-D → ``n_outputs`` hopping values per pair (e.g. spin components
          or real/imaginary parts).
 
    Returns
    -------
    hoppings : ndarray, shape (n_pairs,) or (n_pairs, n_outputs)
        Predicted hopping amplitude(s) for every directed pair.
 
    Raises
    ------
    ValueError
        If the feature dimension of *descriptors* and *params* disagree.
    """
    descriptors = xp.asarray(descriptors, dtype=xp.float64)
    params      = xp.asarray(params,      dtype=xp.float64)

    n_feat_d = descriptors.shape[1] if descriptors.ndim == 2 else descriptors.shape[0]
    n_feat_p = params.shape[0]
    if n_feat_d != n_feat_p:
        raise ValueError(
            f"Feature dimension mismatch: descriptors has {n_feat_d} features "
            f"but params has leading dimension {n_feat_p}."
        )

    return descriptors @ params

def get_acsf_hoppings_sk(
    descriptors,
    params,
):
    """
    Evaluate a linear model with SK-baked descriptors to predict hopping amplitudes.

    hopping(i,j) = descriptors_sk[p] @ params

    Parameters
    ----------
    descriptors : ndarray, shape (n_pairs, 2*n_features)
        Output of ``get_acsf_sk_hopping_descriptors``.
    params : ndarray, shape (2*n_features,)
        Linear model weights.

    Returns
    -------
    hoppings : ndarray, shape (n_pairs,) — xp-array (GPU or CPU)
    """
    descriptors_sk = xp.asarray(descriptors, dtype=xp.float64)
    params         = xp.asarray(params,      dtype=xp.float64)

    n_feat_d = descriptors_sk.shape[1] if descriptors_sk.ndim == 2 else descriptors_sk.shape[0]
    n_feat_p = params.shape[0]
    if n_feat_d != n_feat_p:
        raise ValueError(
            f"Feature dimension mismatch: descriptors has {n_feat_d} features "
            f"but params has leading dimension {n_feat_p}."
        )

    return descriptors_sk @ params


###########################################################################################

# Moon-Koshino

##########################################################################################
def mk_hopping(descriptors,parameters):
    r = np.linalg.norm(descriptors,axis=1)
    [a,b,c] = parameters
    a0 = 1.42
    d0 = 3.35
    n = (descriptors[:,2]) / r
    V_p = a * np.exp(-b * (r - a0))
    V_s = c * np.exp(-b * (r - d0))
    hoppings = V_p*(1 - n**2) + V_s * n**2
    return hoppings

#################################################################################################

# LETB  

################################################################################################

def letb_intralayer(descriptors,parameters,grad=False):

    if type(parameters)==dict:
        t01_params = parameters['t01']
        t02_params = parameters['t02']
        t03_params = parameters['t03']
        
    else:
        t01_params = parameters[:2]
        t02_params = parameters[2:6]
        t03_params = parameters[6:]

    distances = descriptors[3]
    #distances = np.sqrt(distances[0]**2 + distances[1]**2)
    min_distance = min(distances)

    # NN should be within 5% of min_distance
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (distances >= 0.95 * np.sqrt(3) * min_distance) & (distances <= 1.05 * np.sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
   
    # Anything else, we zero out
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)
    dsc_array_1 = descriptors[0]["a"]
    
    dsc_array_2 = np.vstack([np.asarray(descriptors[1][key]) for key in descriptors[1]]).T
    dsc_array_3 = np.vstack([np.asarray(descriptors[2][key]) for key in descriptors[2]]).T
    hoppings = np.zeros(len(distances))
    hoppings[t01_ix] = letb_intralayer_t01(dsc_array_1,t01_params)
    hoppings[t02_ix] = letb_intralayer_t02(dsc_array_2,t02_params)
    hoppings[t03_ix] = letb_intralayer_t03(dsc_array_3,t03_params)
    hoppings[t00] = 0
    return hoppings

def letb_intralayer_t01(descriptors,t01_params):
    t01 = descriptors * t01_params[1:] +t01_params[0]
    return t01

def letb_intralayer_t02(descriptors,t02_params):
    t02 = np.dot(descriptors, t02_params[1:]) + t02_params[0]
    return t02

def letb_intralayer_t03(descriptors,t03_params):
    t03 = np.dot(descriptors, t03_params[1:]) + t03_params[0]
    return t03

def letb_interlayer(descriptors,parameters,grad=False):
    [a0, b0, c0, a3, b3, c3, a6, b6, c6, d6] = parameters
    if type(descriptors)==dict:
        r, theta12, theta21 = descriptors['d'],descriptors['theta_12'],descriptors['theta_21']
    else:
        r = descriptors[:,0]
        theta12 = descriptors[:,1]
        theta21 = descriptors[:,2]
    
    r = np.asarray(r) / 4.649 
    v0 = a0 * np.exp(-b0 * np.power(r, 2)) * np.cos(c0 * r)
    v3 = a3 * (np.power(r, 2)) * np.exp(-b3 * np.power((r - c3) , 2)) 
    v6 =  a6 * np.exp(-b6 * np.power((r - c6),2)) * np.sin(d6 * r)
    hoppings =  v0 
    hoppings += v3 * (np.cos(3 * theta12) + np.cos(3 * theta21))
    hoppings += v6 * (np.cos(6 * theta12) + np.cos(6 * theta21))

    return hoppings

def get_unique_set(array):
    unique_set = np.array([])
    for elem in array:
        if elem in unique_set:
            continue
        else:
            unique_set = np.append(unique_set,elem)
    return np.array(unique_set)

def get_hellman_feynman(atomic_basis, layer_types, lattice_vectors, eigvals,eigvec, model_type,kpoint,
                        intralayer_hopping_params,intralayer_overlap_params, interlayer_hopping_params,interlayer_overlap_params):
    """Calculate Hellman-feynman forces for a given system. Uses finite differences to calculate matrix elements derivatives 
    
    :params atomic_basis: (np.ndarray [Natoms,3]) positions of atoms in angstroms

    :params layer_types: (np.ndarray [Natoms,]) atom types expressed as integers

    :params lattice_vectors: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

    :params eigvals: (np.ndarray [natoms,]) band structure eigenvalues of system

    :params eigvec: (np.ndarray [natoms,natoms]) eigenvectors of system

    :params model_type: (str) specify which tight binding model to use. Options: [popov, mk]

    :params kpoint: (np.ndarray [3,]) kpoint to build hamiltonian and overlap with

    :returns: (np.ndarray [natoms,3]) tight binding forces on each atom"""
    #get hellman_feynman forces at single kpoint. 
    #dE/dR_i =  - Tr_i(rho_e *dS/dR_i + rho * dH/dR_i)
    #construct density matrix
    natoms = len(layer_types)
    conversion = 1.0 #/.529177 # ASE is always in angstrom, while our package wants bohr
    lattice_vectors = np.array(lattice_vectors)*conversion
    atomic_basis = atomic_basis*conversion
    nocc = natoms//2
    fd_dist = 2*np.eye(natoms)
    fd_dist[nocc:,nocc:] = 0
    occ_eigvals = 2*np.diag(eigvals)
    occ_eigvals[nocc:,nocc:] = 0
    density_matrix =  eigvec @ fd_dist  @ np.conj(eigvec).T
    #charge_density = np.diag(density_matrix)
    if overlap_model is not None:
        energy_density_matrix = eigvec @ occ_eigvals @ np.conj(eigvec).T
    del eigvec
    tot_eng = 2 * np.sum(eigvals[:nocc])

    Forces = np.zeros((natoms,3))
    layer_type_set = get_unique_set(layer_types)

    diFull = []
    djFull = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, np.newaxis] * dx + lattice_vectors[1, np.newaxis] * dy)
            diFull += [dx] * natoms
            djFull += [dy] * natoms
    distances = cdist(atomic_basis, extended_coords)

    for i_int,i_type in enumerate(layer_type_set):
        for j_int,j_type in enumerate(layer_type_set):

            if i_type==j_type:
                cutoff = models_cutoff_intralayer[model_type["intralayer"]] * conversion
                hopping_model = models_functions_intralayer[model_type["intralayer"]] #porezag_hopping
                hopping_params = intralayer_hopping_params
                overlap_model = overlap_models_functions_intralayer[model_type["intralayer"]] #porezag_overlap
                overlap_params = intralayer_overlap_params
            else:
                hopping_model = models_functions_interlayer[model_type["interlayer"]] #popov_hopping
                hopping_params = interlayer_hopping_params
                overlap_model = overlap_models_functions_interlayer[model_type["interlayer"]] #popov_overlap
                overlap_params = interlayer_overlap_params
                cutoff = models_cutoff_interlayer[model_type["interlayer"]] * conversion

            indi, indj = np.where((distances > 0.1) & (distances < cutoff))
            di = np.array(diFull)[indj]
            dj = np.array(djFull)[indj]
            i  = np.array(indi)
            j  = np.array(indj % natoms)
            valid_indices = layer_types[i] == i_type
            valid_indices &= layer_types[j] == j_type
            disp =  di[valid_indices, np.newaxis] * lattice_vectors[0] +\
                dj[valid_indices, np.newaxis] * lattice_vectors[1] +\
                atomic_basis[j[valid_indices]] - atomic_basis[i[valid_indices]]
            phases = np.exp((1.0j)*np.dot(kpoint,disp.T))

            dist = np.linalg.norm(disp,axis=1)
            #coulomb_energy = np.sum(charge_density[i[valid_indices]] * charge_density[j[valid_indices]] / dist)
            #coulomb_forces = charge_density[i[valid_indices]] * charge_density[j[valid_indices]] / dist**2 * (disp/dist)
            #check gradients of hoppings via finite difference
            grad_hop = np.zeros_like(disp)
            if overlap_model is not None:
                grad_overlap = np.zeros_like(disp)

            delta = 1e-5
            for dir_ind in range(3):
                dr = np.zeros(3)
                dr[dir_ind] +=  delta
                hop_up = hopping_model(disp+dr[np.newaxis,:],hopping_params)
                hop_dwn = hopping_model(disp-dr[np.newaxis,:],hopping_params)
                grad_hop[:,dir_ind] = (hop_up - hop_dwn)/2/delta
                if overlap_model is not None:

                    overlap_up = overlap_model(disp+dr[np.newaxis,:],overlap_params)
                    overlap_dwn = overlap_model(disp-dr[np.newaxis,:],overlap_params)
                    grad_overlap[:,dir_ind] = (overlap_up - overlap_dwn)/2/delta

            rho =  density_matrix[i[valid_indices],j[valid_indices]][:,np.newaxis] 
            if overlap_model is not None:
                energy_rho = energy_density_matrix[i[valid_indices],j[valid_indices]][:,np.newaxis]
            gradH = grad_hop * phases[:,np.newaxis] * rho
            gradH += np.conj(gradH)
            if overlap_model is not None:
                Pulay =  grad_overlap * phases[:,np.newaxis] * energy_rho
                Pulay += np.conj(Pulay)

            for atom_ind in range(natoms):
                use_ind = np.where(i[valid_indices]==atom_ind)[0]
                ave_gradH = gradH[use_ind,:]
                if overlap_model is not None:
                    ave_gradS = Pulay[use_ind,:] 
                if ave_gradH.ndim!=2:
                    Forces[atom_ind,:] -= -ave_gradH.real 
                    if overlap_model is not None:
                        Forces[atom_ind,:] -=   ave_gradS.real
                else:
                    Forces[atom_ind,:] -= -np.sum(ave_gradH,axis=0).real 
                    if overlap_model is not None:
                        Forces[atom_ind,:] -=   np.sum(ave_gradS,axis=0).real
    return Forces * conversion

def get_recip_cell(cell):
    """find reciprocal cell given real space cell
    :param cell: (np.ndarray [3,3]) real space cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector
    
    :returns: (np.ndarray [3,3]) reciprocal cell of system where recip_cell[i, j] is the jth Cartesian coordinate of the ith reciprocal cell vector"""
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return np.array([b1, b2, b3])

def k_uniform_mesh(mesh_size):
    r""" 
    Returns a uniform grid of k-points that can be passed to
    passed to function :func:`pythtb.tb_model.solve_all`.  This
    function is useful for plotting density of states histogram
    and similar.

    Returned uniform grid of k-points always contains the origin.

    :param mesh_size: Number of k-points in the mesh in each
        periodic direction of the model.
        
    :returns:

        * **k_vec** -- Array of k-vectors on the mesh that can be
    """
    # Tuple so ``mesh_size + (3,)`` and ``np.tile(..., mesh_size)`` work when
    # callers pass a list (e.g. ``k_uniform_mesh(list((nx, ny, nz)))``).
    mesh_size = tuple(int(mesh_size[i]) for i in range(len(mesh_size)))

    # get the mesh size and checks for consistency
    use_mesh=np.zeros(len(mesh_size))
    for i in range(len(mesh_size)):
        use_mesh[i] = mesh_size[i]
    # construct the mesh
    
    # get a mesh
    k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1],0:use_mesh[2]]
    # normalize the mesh
    norm=np.tile(np.array(mesh_size,dtype=float),tuple(mesh_size))
    norm=norm.reshape(mesh_size+(3,))
    norm=norm.transpose([3,0,1,2])
    k_vec=k_vec/norm
    # final reshape
    k_vec=k_vec.transpose([1,2,3,0]).reshape([int(use_mesh[0]*use_mesh[1]*use_mesh[2]),3])
    return k_vec

def k_path(sym_pts,nk,report=False):
    r"""

    Interpolates a path in reciprocal space between specified
    k-points.  In 2D or 3D the k-path can consist of several
    straight segments connecting high-symmetry points ("nodes"),
    and the results can be used to plot the bands along this path.
    
    The interpolated path that is returned contains as
    equidistant k-points as possible.

    :param kpts: Array of k-vectors in reciprocal space between
        which interpolated path should be constructed. These
        k-vectors must be given in reduced coordinates.  As a
        special case, in 1D k-space kpts may be a string:

        * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
        * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
        * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)

    :param nk: Total number of k-points to be used in making the plot.
    
    :param report: Optional parameter specifying whether printout
        is desired (default is True).

    :returns:

        * **k_vec** -- Array of (nearly) equidistant interpolated
        k-points. The distance between the points is calculated in
        the Cartesian frame, however coordinates themselves are
        given in dimensionless reduced coordinates!  This is done
        so that this array can be directly passed to function
        :func:`pythtb.tb_model.solve_all`.

        * **k_dist** -- Array giving accumulated k-distance to each
        k-point in the path.  Unlike array *k_vec* this one has
        dimensions! (Units are defined here so that for an
        one-dimensional crystal with lattice constant equal to for
        example *10* the length of the Brillouin zone would equal
        *1/10=0.1*.  In other words factors of :math:`2\pi` are
        absorbed into *k*.) This array can be used to plot path in
        the k-space so that the distances between the k-points in
        the plot are exact.

        * **k_node** -- Array giving accumulated k-distance to each
        node on the path in Cartesian coordinates.  This array is
        typically used to plot nodes (typically special points) on
        the path in k-space.
    """

    k_list=np.array(sym_pts)

    # number of nodes
    n_nodes=k_list.shape[0]

    mesh_step = nk//(n_nodes-1)
    mesh = np.linspace(0,1,mesh_step)
    step = (np.arange(0,mesh_step,1)/mesh_step)

    kvec = np.zeros((0,3))
    knode = np.zeros(n_nodes)
    for i in range(n_nodes-1):
        n1 = k_list[i,:]
        n2 = k_list[i+1,:]
        diffq = np.outer((n2 - n1),  step).T + n1

        dn = np.linalg.norm(n2-n1)
        knode[i+1] = dn + knode[i]
        if i==0:
            kvec = np.vstack((kvec,diffq))
        else:
            kvec = np.vstack((kvec,diffq))
    kvec = np.vstack((kvec,k_list[-1,:]))

    dk_ = np.zeros(np.shape(kvec)[0])
    for i in range(1,np.shape(kvec)[0]):
        dk_[i] = np.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]

    return (kvec,dk_, knode)

