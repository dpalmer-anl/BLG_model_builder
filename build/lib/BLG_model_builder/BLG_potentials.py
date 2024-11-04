import numpy as np
from BLG_model_builder.NeighborList import *
from BLG_model_builder.descriptors import *

try:
    import cupy
    if cupy.cuda.is_available():
        np = cupy
        gpu_avail = True
except:
    gpu_avail = False
#from hoomd import md
#####################################################################################################

# Python

####################################################################################################


def get_rebo_bond_order(atoms,neighbor_indices,offsets):
    G = 0
    for i in range(len(offsets)):
        G += fc(r)*angular(np.cos(theta)) # both lambda and P are zero for solid state carbon *np.exp(lambda_ijk) + P
    bij_sp = 1/np.sqrt(1+ G)
    bij_pi = G_RC + b_DH
    bij = 0.5*(bij_sp+bji_sp)+bij_pi
    return bij

"""def Tersoff_hoomd(atoms,parameters):
    nl = md.nlist.Cell()
    tersoff = md.many_body.Tersoff(default_r_cut=1.3, nlist=nl)
    tersoff.params[('A', 'B')] = dict(magnitudes=(2.0, 1.0), lambda3=5.0)
    energy = md.force.Force.energy
    forces = md.force.Force.forces
    return energy, forces"""

def get_potential_energy(atoms,parameters,pe_funcs,r_cut=10):
    #%calculation of pairwise potential
    potential_energy = 0
    
    cutoffs = r_cut*np.ones(len(atoms))
    atoms.set_pbc(True)
    nl = NeighborList(cutoffs)
    nl.update(atoms)

    norm = np.zeros((len(atoms),3))
    for i in range(len(atoms)):
        neighbor_indices, offsets = nl.get_neighbors(i) 
        norm[i,:] = get_norm(atoms,neighbor_indices,offsets) 
    atoms.set_array('norm',norm)
    for i in range(len(atoms)):
        neighbor_indices, offsets = nl.get_neighbors(i) 
        for f in pe_funcs:
            potential_energy +=f(atoms,parameters,neighbor_indices,offsets)
    return potential_energy



#@jit(nopython=True, parallel=True)
def calc_d_sw2_vectorized(A, B, p, q, sigma, cutoff, rij):
    sig_r = sigma / rij
    one_by_delta_r = 1.0 / (rij - cutoff)
    Bpq = B * sig_r ** p - sig_r ** q
    exp_sigma = np.exp(sigma * one_by_delta_r)
    E2 = A * Bpq * exp_sigma
    F = (q * sig_r ** (q + 1)) - p * B * sig_r ** (p + 1) - Bpq * (sigma * one_by_delta_r) ** 2
    F = F * (1. / sigma) * A * exp_sigma
    return E2, F

#@jit(nopython=True, parallel=True)
def calc_d_sw3_vectorized(lam, cos_beta0, gamma_ij, gamma_ik, cutoff_ij, cutoff_ik, cutoff_jk, rij, rik, rjk):
    cos_beta_ikj = (rij**2 + rik**2 - rjk**2) / (2 * rij * rik)
    cos_diff = cos_beta_ikj - cos_beta0
    exp_ij_ik = np.exp(gamma_ij / (rij - cutoff_ij) + gamma_ik / (rik - cutoff_ik))
    dij = -gamma_ij / (rij - cutoff_ij)**2
    dik = -gamma_ik / (rik - cutoff_ik)**2
    E3 = lam * exp_ij_ik * cos_diff ** 2
    dcos_drij = (rij**2 - rik**2 + rjk**2) / (2 * rij**2 * rik)
    dcos_drik = (rik**2 - rij**2 + rjk**2) / (2 * rik**2 * rij)
    dcos_drjk = -rjk / (rij * rik)
    dE3_dr = np.zeros((len(rij), 3))
    dE3_dr[:, 0] = lam * cos_diff * exp_ij_ik * (dij * cos_diff + 2 * dcos_drij)
    dE3_dr[:, 1] = lam * cos_diff * exp_ij_ik * (dik * cos_diff + 2 * dcos_drik)
    dE3_dr[:, 2] = lam * cos_diff * exp_ij_ik * 2 * dcos_drjk
    return E3, dE3_dr

#@jit(nopython=True, parallel=True)
def stillinger_weber(nl, elements_nl, coords_all, A, B, p, q, sigma, gamma, cutoff, lam, cos_beta0, cutoff_jk):
    energy = 0.0
    F = np.zeros_like(coords_all)

    for i in prange(len(nl)):
        nli = nl[i]
        elements = elements_nl[i]
        xyz_i = coords_all[nli[0]]
        elem_i = elements[0]

        num_elems = len(nli)
        xyz_j_list = np.zeros((num_elems - 1, 3))
        rij = np.zeros(num_elems - 1)
        ij_sum = np.zeros(num_elems - 1)

        for j in prange(1, num_elems):
            xyz_j_list[j - 1] = coords_all[nli[j]]
            rij[j - 1] = np.linalg.norm(xyz_j_list[j - 1] - xyz_i)
            ij_sum[j - 1] = elem_i + elements[j]

        E2, F2 = calc_d_sw2_vectorized(A[ij_sum], B[ij_sum], p[ij_sum], q[ij_sum], sigma[ij_sum], cutoff[ij_sum], rij)
        energy += 0.5 * np.sum(E2)
        F_comp = 0.5 * (F2 / rij)[:, np.newaxis] * (xyz_j_list - xyz_i)
        F[i, :] += np.sum(F_comp, axis=0)
        for j in prange(1, num_elems):
            F[nli[j], :] -= F_comp[j - 1]

        for j in prange(1, num_elems):
            elem_j = elements[j]
            xyz_j = coords_all[nli[j]]

            for k in prange(j + 1, num_elems):
                elem_k = elements[k]
                if elem_i != elem_j and elem_j == elem_k:
                    ijk_sum = 2 + -1 * (elem_i + elem_j + elem_k)
                    ik_sum = elem_i + elem_k
                    xyz_k = coords_all[nli[k]]
                    rik = xyz_k - xyz_i
                    rjk = xyz_k - xyz_j
                    norm_rij = np.linalg.norm(xyz_j - xyz_i)
                    norm_rik = np.linalg.norm(rik)
                    norm_rjk = np.linalg.norm(rjk)

                    gamma_ik = gamma[ik_sum]
                    cutoff_ik = cutoff[ik_sum]

                    E3, F3 = calc_d_sw3_vectorized(lam[ijk_sum], cos_beta0[ijk_sum], gamma[elem_i + elem_j], gamma_ik, cutoff[elem_i + elem_j], cutoff_ik, cutoff_jk[ijk_sum], norm_rij, norm_rik, norm_rjk)
                    energy += E3

                    F_comp_i = F3[:, 0] / norm_rij * (xyz_j - xyz_i)
                    F[i, :] += np.sum(F_comp_i, axis=0)
                    F[nli[j], :] -= np.sum(F_comp_i, axis=0)

                    F_comp_ik = F3[:, 1] / norm_rik * rik
                    F[i, :] += np.sum(F_comp_ik, axis=0)
                    F[nli[k], :] -= np.sum(F_comp_ik, axis=0)

                    F_comp_jk = F3[:, 2] / norm_rjk * rjk
                    F[nli[j], :] += np.sum(F_comp_jk, axis=0)
                    F[nli[k], :] -= np.sum(F_comp_jk, axis=0)

    return energy, F


def two_body_potential(rij):
    return np.where(
        rij < a * sigma,
        epsilon * A * ((B * ((sigma / rij) ** p - 1)) ** 2 - 1) * np.exp(sigma / rij - a),
        0.0
    )

def three_body_potential(rij, rik, cos_theta):
    return np.where(
        (rij < a * sigma) & (rik < a * sigma),
        epsilon * lambda_ * (cos_theta + 1/3)**2 * np.exp(gamma * ((sigma / rij - a) + (sigma / rik - a))),
        0.0
    )

def stillinger_weber_vect(atoms,params,r_cut = 5):
    [sigma, epsilon, A, B, p, q, a, lambda_, gamma] = params
    cutoffs = r_cut*np.ones(len(atoms))
    atoms.set_pbc(True)
    nl = NeighborList(cutoffs)
    nl.update(atoms)
    dist_table, disp_table = atoms.get_all_distances(mic=True,vector=True)
    dist_table = dist_table.flatten()
    potential_2b = np.sum(two_body_potential(dist_table))
    potential_3b = 0
    for i in range(len(atoms)):
        neighbor_indices, offsets = nl.get_neighbors(i) 
        ni,ni = np.meshgrid((neighbor_indices,neighbor_indices))
        ni = ni.flatten()
        nj = ni.flatten()
        angle_indices = np.stack((i*np.ones(len(ni),dtype=np.int64),ni,nj))
        angles = atoms.get_angles(angle_indices)
        potential_3b += three_body_potential(dist_table[ni],dist_table[nj],angles)
    return potential_2b + potential_3b



def get_normal_vect(atoms,n_norms=3):

    normal_vectors = np.zeros((len(atoms),3))
    for i in range(len(atoms)):
        displacements = atoms.neighbor_list.get_displacements(index_i = i)
        distances = np.linalg.norm(displacements,axis=1)
        nn_ind = np.argsort(distances)[:n_norms]
        disp_nn = displacements[nn_ind,:]
        for j in range(n_norms):
            cross_vect = np.cross(disp_nn[j,:],disp_nn[(j+1)%n_norms,:])
            cross_norm = np.linalg.norm(cross_vect)
            if np.abs(cross_norm) < 1e-6: #skip if two displacement vectors are parallel
                continue
            normal_vectors[i,:] += cross_vect/cross_norm
    return normal_vectors/n_norms

def Tersoff_depr(atoms,parameters):
    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NeighborList.NN_list(atoms)
    [m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A] = parameters

    N = len(atoms)
    
    # Calculate pairwise distances
    r_ij = atoms.neighbor_list.get_displacements()
    r_ij = np.linalg.norm(r_ij,axis=1)
    # Compute the cutoff function for each pair
    fc_ij = np.ones_like(r_ij) #cutoff_function(r_ij,R,D)

    # Compute exponential terms
    exp_term_A = A * np.exp(-lambda1 * r_ij)
    exp_term_B = B * np.exp(-lambda2 * r_ij)

    # Compute the repulsive term
    V_R = exp_term_A * fc_ij

    # Compute the attractive term
    V_A = exp_term_B * fc_ij

    # Angular function terms
    
    cos_theta_ijk = atoms.neighbor_list.get_angles()
    
    g_ijk = 1 + (c**2 / d**2) - (c**2 / (d**2 + (cos_theta_ijk - costheta0)**2))
    g_ijk *= gamma
    # Bond order term with additional exponential term
    exp_term_lambda3 = np.exp((lambda3**m) * (r_ij[:,  np.newaxis] - r_ij[ np.newaxis, :])**m)
    bond_order = (1 + np.power(beta * np.sum(fc_ij[:,  np.newaxis] 
                    * g_ijk * exp_term_lambda3, axis=1), n))**(-0.5 / n)

    bond_order = 1.33
    # Compute the potential energy
    E_pot = 0.5 * np.sum((V_R - bond_order * V_A) * fc_ij)

    return E_pot

def minimum_image_distance(positions, cell):
    """
    Calculate pairwise minimum image distances under periodic boundary conditions for non-orthogonal cells.

    :param positions: Nx3 array of atomic positions
    :param cell: 3x3 matrix describing the simulation cell vectors
    :return: NxN array of pairwise distances
    """
    # Calculate the displacement vectors between all pairs of atoms
    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    
    # Transform displacement vectors to fractional coordinates
    inv_cell = np.linalg.inv(cell)
    disp_frac = np.dot(disp, inv_cell)
    # Apply minimum image convention (shift fractional coordinates to [-0.5, 0.5))
    disp_frac = disp_frac - np.round(disp_frac)
    
    # Transform back to Cartesian coordinates
    disp_cart = np.dot(disp_frac, cell)
    distances = np.linalg.norm(disp_cart, axis=-1)
    
    return distances, disp_cart

def cutoff_function(r,R,D):
    fc = np.zeros_like(r)
    inner_ind = np.where(r<(R-D))
    fc[inner_ind] += 1
    intermed_ind = np.where((r>R-D) & (r<R+D))
    fc[intermed_ind] += 0.5*(1 - np.sin(np.pi/2*(r[intermed_ind]-R)/D))
    return fc

def Tersoff(atoms,parameters):
    # c, d, costheta0, n, beta, lambda2, B, lambda1, A
    [c, d, costheta0, n, beta, lambda2, B, lambda1, A] = parameters
    R = 1.95
    D = 0.15
    positions = atoms.positions
    cell = atoms.get_cell()
    N = len(positions)
    
    # Calculate pairwise minimum image distances and displacement vectors
    r_ij, disp_cart = minimum_image_distance(positions, cell)
    np.fill_diagonal(r_ij, R+2*D) #avoid self interaction

    # Compute the cutoff function for each pair
    fc_ij = cutoff_function(r_ij,R,D)
    
    # Compute the repulsive term
    V_R = A * np.exp(-lambda1 * r_ij) * fc_ij

    # Compute the attractive term
    V_A = -B * np.exp(-lambda2 * r_ij) * fc_ij

    # Angular function terms (cosine(theta))
    
    disp_norm = disp_cart/r_ij[:,:,np.newaxis]
    
    # Calculate cos_theta_ijk for all triplets (i, j, k)
    cos_theta_ijk = np.einsum('ijk,ilk->ijl', disp_norm, disp_norm)  # Shape: (N, N, N)
    
    # Compute g_ijk for all triplets
    g_ijk = 1 + (c**2 / d**2) - (c**2 / (d**2 + (cos_theta_ijk - costheta0)**2))
    for i in range(N):
        g_ijk[i,i,:] = 0
        g_ijk[i,:,i] = 0
        g_ijk[:,i,i] = 0
    # Calculate the ksi values by summing over k, with the exclusion of k=i and k=j
    ksi = np.einsum('ijk,ik->ij', g_ijk, fc_ij)  # Shape: (N, N)

    # Calculate bond order
    bond_order = (1 + beta**n * ksi**n) ** (-1/(2*n))  # Shape: (N, N)
    
    # Calculate total potential energy
    E_pot = 0.5 * np.sum((V_R + bond_order * V_A) * fc_ij)
    
    return E_pot
 

def Kolmogorov_Crespi_insp(atoms,parameters,cutoff=10):
    [delta,C,C0,C2,C4,z0,A6,A8,A10] =  parameters
    meV = 1e-3
    A6 *= meV
    A8 *= meV
    A10 *= meV

    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms)
    
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    normal_vectors = get_normal_vect(atoms)
    
    ni_dot_d = np.sum( normal_vectors[atoms.neighbor_list.i[valid_indices]] * disp,axis=1)
    nj_dot_d = np.sum( normal_vectors[atoms.neighbor_list.j[valid_indices]] * disp,axis=1)
    rhoij =np.sqrt(dist**2- (ni_dot_d)**2) 
    rhoji =np.sqrt(dist**2-(nj_dot_d)**2)  
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    x = dist/cutoff
    Taper = 20*np.power(x,7) - 70*np.power(x,7) + 84*np.power(x,5) - 35*np.power(x,4) +1

    V_ij = -(C+frhoij+frhoji) * (A6*np.power(dist/z0,-6)+A8*np.power(dist/z0,-8)+A10*np.power(dist/z0,-10))
    V=0.5*np.sum(V_ij*Taper)
    return V

def Kolmogorov_Crespi_insp_linear(atoms, params,return_dsc = False, cutoff = 10):
    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms)
    
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    normal_vectors = get_normal_vect(atoms)
    ni_dot_d = np.sum( normal_vectors[atoms.neighbor_list.i[valid_indices]] * disp,axis=1)
    nj_dot_d = np.sum( normal_vectors[atoms.neighbor_list.j[valid_indices]] * disp,axis=1)
    rhoij =np.sqrt(dist**2- (ni_dot_d)**2) 
    rhoji =np.sqrt(dist**2-(nj_dot_d)**2)  

    r6 = np.power(dist,-6)
    r8 = np.power(dist,-8)
    r10 = np.power(dist,-10)

    disreg2 = np.exp(-rhoij**2)*rhoij**2
    disreg4 = np.exp(-rhoij**2)*rhoij**4

    dsc = [r6 , r8, r10, r6 * disreg2, r8 * disreg2, r10 * disreg2, r6 * disreg4, r8 * disreg4, r10 * disreg4 ]
    V = np.sum(params[:,np.newaxis] * dsc)

    if return_dsc:
        return V, np.sum(dsc,axis=1)
    else:
        return V

def interlayer_potential(atoms,params,cutoff=10):
    #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
    meV = 1e-3 #in order to match lammps, scale variables
    [z0, C0, C2, C4, C, delta, lambda_val, A,reff] =  params
    C *= meV
    A *= meV
    C0 *= meV
    C2 *= meV
    C4 *= meV

    #taken from ilp BNCH potential in lammps
    sr = 0.7954443 
    d = 15.499947
    #reff = 3.681440 
    #z0 = 3.379423382381699


    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms)
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    normal_vectors = get_normal_vect(atoms)
    
    ni_dot_d = np.sum( normal_vectors[atoms.neighbor_list.i[valid_indices]] * disp,axis=1)
    nj_dot_d = np.sum( normal_vectors[atoms.neighbor_list.j[valid_indices]] * disp,axis=1)
    rhoij =np.sqrt(dist**2- (ni_dot_d)**2) 
    rhoji =np.sqrt(dist**2-(nj_dot_d)**2)  
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    x = dist/cutoff
    Taper = 20*np.power(x,7) - 70*np.power(x,7) + 84*np.power(x,5) - 35*np.power(x,4) +1
    vdw_smoothing_term = 1/(1 + np.exp(-d * (dist/(sr *reff ) - 1)))
    V_ij = np.exp(-lambda_val*(dist - z0)) * (C+frhoij+frhoji) - A*np.power(dist/z0,-6) * vdw_smoothing_term
    V = 0.5 *  np.sum(V_ij*Taper)
    return V


def vdw(atoms,params,cutoff=10):
    [A6,A8,A10] =  params
    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms)
    
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    r6 = np.power(dist,-6)
    r8 = np.power(dist,-8)
    r10 = np.power(dist,-12)
    return np.sum(A6*r6 + A8*r8 + A10*r10 )


def Kolmogorov_Crespi(atoms,parameters,cutoff=10):
    #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
    meV = 1e-3 #in order to match lammps, scale variables
    [z0, C0, C2, C4, C, delta, lambda_val, A] =  parameters
    C *= meV
    A *= meV
    C0 *= meV
    C2 *= meV
    C4 *= meV

    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms)
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    normal_vectors = get_normal_vect(atoms)
    
    ni_dot_d = np.sum( normal_vectors[atoms.neighbor_list.i[valid_indices]] * disp,axis=1)
    nj_dot_d = np.sum( normal_vectors[atoms.neighbor_list.j[valid_indices]] * disp,axis=1)
    rhoij =np.sqrt(dist**2- (ni_dot_d)**2) 
    rhoji =np.sqrt(dist**2-(nj_dot_d)**2)  
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    x = dist/cutoff
    Taper = 20*np.power(x,7) - 70*np.power(x,7) + 84*np.power(x,5) - 35*np.power(x,4) +1
    V_ij = np.exp(-lambda_val*(dist - z0)) * (C+frhoij+frhoji) - A*np.power(dist/z0,-6)
    V = 0.5 *  np.sum(V_ij*Taper)
    return V

def density_potential(atoms,density_matrix,parameters):
    #density matrix is bond order for pz orbitals
    [c, d, costheta0, n, beta, lambda2, B, lambda1, A] = parameters
    R = 1.95
    D = 0.15
    positions = atoms.positions
    cell = atoms.get_cell()
    N = len(positions)
    
    # Calculate pairwise minimum image distances and displacement vectors
    r_ij, disp_cart = minimum_image_distance(positions, cell)
    np.fill_diagonal(r_ij, R+2*D) #avoid self interaction

    # Compute the cutoff function for each pair
    fc_ij = cutoff_function(r_ij,R,D)
    
    # Compute the repulsive term
    V_R = A * np.exp(-lambda1 * r_ij) * fc_ij

    # Compute the attractive term
    V_A = -B * np.exp(-lambda2 * r_ij) * fc_ij

    # Angular function terms (cosine(theta))
    
    disp_norm = disp_cart/r_ij[:,:,np.newaxis]
    
    # Calculate cos_theta_ijk for all triplets (i, j, k)
    cos_theta_ijk = np.einsum('ijk,ilk->ijl', disp_norm, disp_norm)  # Shape: (N, N, N)
    
    # Compute g_ijk for all triplets
    g_ijk = 1 + (c**2 / d**2) - (c**2 / (d**2 + (cos_theta_ijk - costheta0)**2))
    for i in range(N):
        g_ijk[i,i,:] = 0
        g_ijk[i,:,i] = 0
        g_ijk[:,i,i] = 0
    # Calculate the ksi values by summing over k, with the exclusion of k=i and k=j
    ksi = np.einsum('ijk,ik->ij', g_ijk, fc_ij)  # Shape: (N, N)

    # Calculate bond order
    bond_order = (1 + beta**n * ksi**n) ** (-1/(2*n)) + density_matrix # Shape: (N, N)
    
    # Calculate total potential energy
    E_pot = 0.5 * np.sum((V_R + bond_order * V_A) * fc_ij)
    
    return E_pot

if __name__ == "__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    from pythtb import *
    from bilayer_letb.api import tb_model
    from BLG_model_builder.geom_tools import *
    from BLG_model_builder.Lammps_Utils import *
    import pandas as pd


    sep = 3.35
    a = 2.46
    n=5

    kc_parameters = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                              0.719345289329483, 3.293082477932360, 13.906782892134125])

    model_type = "python"


    stacking_ = ["AB","SP","Mid","AA"]
    disreg_ = [0 , 0.16667, 0.5, 0.66667]
    colors = ["blue","red","black","green"]
    d_ = np.linspace(3,5,5)
    df = pd.read_csv('../data/qmc.csv') 
    d_ab = df.loc[df['disregistry'] == 0, :]
    min_ind = np.argmin(d_ab["energy"].to_numpy())
    E0_qmc = d_ab["energy"].to_numpy()[min_ind]
    d = d_ab["d"].to_numpy()[min_ind]
    disreg = d_ab["disregistry"].to_numpy()[min_ind]
    relative_tetb_energies = []
    relative_qmc_energies = []
    E0_tegt = 1e10
    
    for i,stacking in enumerate(stacking_):
        energy_dis_tegt = []
        energy_dis_qmc = []
        energy_dis_tb = []
        d_ = []
        dis = disreg_[i]
        d_stack = df.loc[df['stacking'] == stacking, :]
        for j, row in d_stack.iterrows():

            atoms = get_bilayer_atoms(row["d"],dis)

            #total_energy = (calc.get_total_energy(atoms))/len(atoms)
            total_energy = Kolmogorov_Crespi(atoms,kc_parameters)/len(atoms)
            print(total_energy)
            
            if total_energy<E0_tegt:
                E0_tegt = total_energy

            qmc_total_energy = (row["energy"])

            energy_dis_tegt.append(total_energy)
            energy_dis_qmc.append(qmc_total_energy)
            d_.append(row["d"])


        relative_tetb_energies.append(energy_dis_tegt)
        relative_qmc_energies.append(energy_dis_qmc)
        plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " python kc",c=colors[i])
        plt.scatter(np.array(d_),np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])

    plt.xlabel(r"Interlayer Distance ($\AA$)")
    plt.ylabel("Interlayer Energy (eV)")
    plt.title(model_type)
    plt.tight_layout()
    plt.legend()
    plt.savefig("../tests/figures/interlayer_test_"+model_type+".jpg")
    plt.clf()


    #intralayer test
    a = 2.462
    # c, d, costheta0, n, beta, lambda2, B, lambda1, A
    #don't need m or lambda3 for Carbon
    model_names = ["tersoff"]
    model_files = ["../BLG_model_builder/parameters/BNC.tersoff"]
    tersoff_parameters = np.array([3.8049e4, 4.3484, -0.930, 0.72751, 1.5724e-7,  2.2119,  430.00,    3.4879,  1393.6]) #costheta0 = -0.93000
    lat_con_list = np.sqrt(3) * np.array([1.197813121272366,1.212127236580517,1.2288270377733599,1.2479125248508947,\
                        1.274155069582505,1.3027833001988072,1.3433399602385685,1.4053677932405566,\
                        1.4745526838966203,1.5294234592445326,1.5795228628230618])

    lat_con_energy = np.zeros_like(lat_con_list)
    tb_energy = np.zeros_like(lat_con_list)
    rebo_energy = np.zeros_like(lat_con_list)
    lammps_energy = np.zeros_like(lat_con_list)
    dft_energy = np.array([-5.62588911,-6.226154186,-6.804241219,-7.337927988,-7.938413961,\
                        -8.472277446,-8.961917385,-9.251954937,-9.119902805,-8.832030042,-8.432957809])

    for i,lat_con in enumerate(lat_con_list):
    
        atoms = get_monolayer_atoms(0,0,a=lat_con)
        atoms.set_array('mol-id',np.ones(len(atoms),dtype=np.int64))
        print("a = ",lat_con," natoms = ",len(atoms))
        total_energy = Tersoff(atoms,tersoff_parameters)/len(atoms)
        _,le,_ = run_lammps(atoms,model_names,model_files)
        lammps_energy[i] = le/len(atoms)
        print("Total energy = ",total_energy)
        lat_con_energy[i] = total_energy


    plt.scatter(lat_con_list/np.sqrt(3),lat_con_energy-np.min(lat_con_energy),label = "rebo fit")
    plt.scatter(lat_con_list/np.sqrt(3), dft_energy-np.min(dft_energy),label="dft results")
    #plt.scatter(lat_con_list/np.sqrt(3), lammps_energy-np.min(lammps_energy),label="tersoff lammps")
    plt.xlabel(r"nearest neighbor distance ($\AA$)")
    plt.ylabel("energy above ground state (eV/atom)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../tests/figures/intralayer_test_"+model_type+".jpg")
    plt.clf()

