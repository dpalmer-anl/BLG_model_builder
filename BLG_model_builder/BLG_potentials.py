import numpy as np
import matplotlib.pyplot as plt
from BLG_model_builder.NeighborList import *
from BLG_model_builder.descriptors import *
import warnings
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



def get_normal_vect_depr(atoms,n_norms=3):

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

def d_cutoff_dr(r, R, D):
    dfc = np.zeros_like(r)
    intermed_ind = np.where((r>R-D) & (r<R+D))
    dfc[intermed_ind] += -(0.5 * np.pi / (2 * D)) * np.cos(np.pi * (r[intermed_ind] - R) / (2 * D))
    return dfc

def Tersoff(atoms,parameters,**kwargs):
    # c, d, costheta0, n, beta, lambda2, B, lambda1, A
    warnings.simplefilter("ignore", category=RuntimeWarning)
    [c, d, costheta0, n, beta, lambda2, B, lambda1, A] = parameters
    R = 1.95
    D = 0.15

    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms)
    positions = atoms.positions
    cell = atoms.get_cell()
    atom_types = atoms.get_array("mol-id")
    N = len(positions)
    
    # Compute the attractive/repulsive term
    r_ij, disp_cart = minimum_image_distance(positions, cell)
    valid_indices = atom_types[atoms.neighbor_list.i] == atom_types[atoms.neighbor_list.j]
    r_ij[atoms.neighbor_list.i[~valid_indices],atoms.neighbor_list.j[~valid_indices]] += R+2*D
    np.fill_diagonal(r_ij, R+2*D) #avoid self interaction
    fc_ij = cutoff_function(r_ij,R,D)
    V_R = A * np.exp(-lambda1 * r_ij) 
    V_A = -B * np.exp(-lambda2 * r_ij) 
    
    # Calculate angular term for all triplets (i, j, k)
    disp_norm = disp_cart/r_ij[:,:,np.newaxis]
    cos_theta_ijk = np.einsum('ijk,ilk->ijl', disp_norm, disp_norm)  # Shape: (N, N, N)
    g_ijk = 1 + (c**2 / d**2) - (c**2 / (d**2 + (cos_theta_ijk - costheta0)**2))
    for i in range(N):
        g_ijk[i,i,:] = 0
        g_ijk[i,:,i] = 0
        g_ijk[:,i,i] = 0
    ksi = np.einsum('ijk,ik->ij', g_ijk, fc_ij)  # Shape: (N, N)

    # Calculate bond order
    bond_order = (1 + (beta)**n * ksi**n) ** (-1/(2*n))  # Shape: (N, N)
        
    # Calculate total potential energy
    E_pot = 0.5 * np.sum((V_R + bond_order * V_A) * fc_ij)

    # --- Compute forces ---
    forces = np.zeros_like(positions)

    # dE/dR_ij terms
    dV_R = -lambda1 * V_R
    dV_A = -lambda2 * V_A

    # d(bond_order)/dksi
    ksi_eps = ksi + 1e-12  # prevent div by zero
    prefactor = -0.5 * (beta**n * ksi_eps**(n - 1)) / ((1 + beta**n * ksi_eps**n) ** (1 + 1/(2*n)))

    dE_dksi = fc_ij * V_A * prefactor  # (N, N)
    dksi_drij = np.zeros_like(r_ij)  # We'll approximate this numerically or skip for now

    dE_drij = (dV_R + bond_order * dV_A) * fc_ij + (V_R + bond_order * V_A) * d_cutoff_dr(r_ij, R, D)

    # Convert scalar gradients to vector gradients (chain rule)
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_rij = np.nan_to_num(disp_cart / r_ij[:, :, None])

    force_matrix = -(dE_drij[:, :, None] * grad_rij)
    forces += np.sum(force_matrix, axis=1)  # Accumulate forces

    return E_pot, forces
 

def Kolmogorov_Crespi_insp(atoms,parameters,cutoff=10,**kwargs):
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

def Kolmogorov_Crespi_vdw(atoms,parameters,cutoff=10,**kwargs):
    #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
    meV = 1e-3 #in order to match lammps, scale variables
    [z0, C0, C2, C4, C, delta, lambda_val, A6,A8,A10,A12] =  parameters
    C *= meV
    A6 *= meV
    A8 *= meV
    A10 *= meV
    A12 *= meV
    C0 *= meV
    C2 *= meV
    C4 *= meV

    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms,cutoff=cutoff)
    #print("cutoff in pot = ",cutoff)
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    #normal_vectors = get_normal_vect(atoms)
    normal_vectors = np.zeros((len(atoms),3))
    normal_vectors[atom_types==1,2] +=1 #bottom
    normal_vectors[atom_types==2,2] -=1 #top

    ni = normal_vectors[atoms.neighbor_list.i[valid_indices]]
    nj = normal_vectors[atoms.neighbor_list.j[valid_indices]]
    ni_dot_d = np.sum( ni * disp,axis=1)
    nj_dot_d = np.sum( nj * disp,axis=1)
    rhoij =np.sqrt(dist**2- (ni_dot_d)**2) 
    rhoji =np.sqrt(dist**2-(nj_dot_d)**2)  
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    x = dist/cutoff
    Taper =  20*np.power(x,7) - 70*np.power(x,7) + 84*np.power(x,5) - 35*np.power(x,4) +1
    V_ij = np.exp(-lambda_val*(dist - z0)) * (C+frhoij+frhoji) - (A6*np.power(dist/z0,-6)+A8*np.power(dist/z0,-8)+A8*np.power(dist/z0,-10)+A12*np.power(dist/z0,-12))
    V = 0.5 *  np.sum(V_ij*Taper)

    # Force calculation
    rhat = disp/dist[:,np.newaxis]
    df_dpij2 = -1/delta**2 * np.exp(-rhoij**2/delta**2) * (C2*rhoij**2/delta**2 + C4 * rhoij**4/delta**4)\
            +  np.exp(-rhoij**2/delta**2) * (C2/delta**2 + 2 * C4 * rhoij**2/delta**4)
    df_dpji2 = -1/delta**2 * np.exp(-rhoji**2/delta**2) * (C2*rhoji**2/delta**2 + C4 * rhoji**4/delta**4)\
            +  np.exp(-rhoji**2/delta**2) * (C2/delta**2 + 2 * C4 * rhoji**2/delta**4)

    dpij2_dr = 2*disp - 2*(ni_dot_d[:,np.newaxis])*ni 
    dpji2_dr = 2*disp - 2*(nj_dot_d[:,np.newaxis])*nj

    dfrhoij_dr = df_dpij2[:,np.newaxis] * dpij2_dr
    dfrhoji_dr = df_dpji2[:,np.newaxis] * dpji2_dr

    dVij_dr = (-lambda_val * np.exp(-lambda_val*(dist - z0)) * (C+frhoij+frhoji))[:,np.newaxis] * rhat
    dVij_dr += (
        6 * A6 * (dist/z0) ** -7 +
        8 * A8 * (dist/z0) ** -9 +
        10 * A10 * (dist/z0) ** -11 +
        12 * A12 * (dist/z0) ** -13
    )[:,np.newaxis] * rhat / z0

    dVij_dr += np.exp(-lambda_val*(dist - z0))[:,np.newaxis] * (dfrhoij_dr+dfrhoji_dr) 

    dTaper_dr = rhat * (140*x**6 - 420*x**5 + 420*x**4 - 140*x**3)[:,np.newaxis]/cutoff
     
    total_deriv = dTaper_dr * V_ij[:,np.newaxis] + Taper[:,np.newaxis] * dVij_dr
    forces = np.zeros((len(atoms), 3))
    np.add.at(forces, atoms.neighbor_list.i[valid_indices], -total_deriv)
    np.add.at(forces, atoms.neighbor_list.j[valid_indices], total_deriv)

    return V,forces/2

def Kolmogorov_Crespi(atoms,parameters,cutoff=10,**kwargs):
    #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
    meV = 1e-3 #in order to match lammps, scale variables
    [z0, C0, C2, C4, C, delta, lambda_val, A] =  parameters
    C *= meV
    A *= meV
    C0 *= meV
    C2 *= meV
    C4 *= meV

    if not atoms.has('neighbor_list'):
        atoms.neighbor_list = NN_list(atoms,cutoff=cutoff)
    #print("cutoff in pot = ",cutoff)
    atoms.neighbor_list.set_cutoff(cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    valid_indices = atom_types[atoms.neighbor_list.i] != atom_types[atoms.neighbor_list.j] #select interlayer displacements only
    disp =   atoms.neighbor_list.di[valid_indices, np.newaxis] * cell[0] +\
                atoms.neighbor_list.dj[valid_indices, np.newaxis] * cell[1] +\
                positions[atoms.neighbor_list.j[valid_indices]] - positions[atoms.neighbor_list.i[valid_indices]]
    dist = np.linalg.norm(disp,axis=1)

    #normal_vectors = get_normal_vect(atoms)
    normal_vectors = np.zeros((len(atoms),3))
    normal_vectors[atom_types==1,2] +=1 #bottom
    normal_vectors[atom_types==2,2] -=1 #top

    ni_dot_d = np.sum( normal_vectors[atoms.neighbor_list.i[valid_indices]] * disp,axis=1)
    nj_dot_d = np.sum( normal_vectors[atoms.neighbor_list.j[valid_indices]] * disp,axis=1)
    rhoij =np.sqrt(dist**2- (ni_dot_d)**2) 
    rhoji =np.sqrt(dist**2-(nj_dot_d)**2)  
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    x = dist/cutoff
    Taper =  20*np.power(x,7) - 70*np.power(x,7) + 84*np.power(x,5) - 35*np.power(x,4) +1
    V_ij = np.exp(-lambda_val*(dist - z0)) * (C+frhoij+frhoji) - A*np.power(dist/z0,-6)
    V = 0.5 *  np.sum(V_ij*Taper)
    return V

def get_normal_vect(intra_disp, intra_indi ,intra_indj):
    normal_vectors = np.zeros((len(atoms),3))
    alpha = np.zeros(len(atoms))
    a0 = 1.42
    mean_length = 0.326 #average length of C pz orbital
    for i in range(len(atoms)):
        distances = np.linalg.norm(intra_disp[intra_indi==i],axis=1)
        nn_ind = np.argsort(distances)[:3]
        alpha_i[i] = mean_length * np.mean(np.abs(((distances-nn_ind-a0)/a0)))
        disp_nn = displacements[nn_ind[:2],:]
        normal_vectors[i] = np.cross(disp_nn[0,:],disp_nn[1,:])
    normal_vectors_norm = np.linalg.norm(normal_vectors,axis=1)
    normal_vectors = normal_vectors/normal_vectors_norm[:,np.newaxis]
    return normal_vectors, alpha

def Interlayer_MLP(atoms,parameters,cutoff=10,hidden_size=10,**kwargs):
    meV = 1e-3 #in order to match lammps, scale variables
    z0 = 3.35
    lambda_val = 3.293
    #construct MLP from parameters
    input_size = 3 # alpha_ij, rho_ij, inter_dist
    output_size = 2
    W1 = np.reshape(parameters[:input_size*hidden_size], (input_size, hidden_size))
    b1 = np.reshape(parameters[input_size*hidden_size:input_size*hidden_size+hidden_size], (1, hidden_size))
    W2 = np.reshape(parameters[input_size*hidden_size+hidden_size:input_size*hidden_size+hidden_size+hidden_size*output_size], (hidden_size, output_size))
    b2 = np.reshape(parameters[input_size*hidden_size+hidden_size+hidden_size*output_size:], (1, output_size))

    #separate intra and interlayer displacements
    disp, i, j, di, dj = get_disp(atoms,cutoff=cutoff)
    atom_types = atoms.get_array("mol-id")
    cell = atoms.get_cell()
    positions = atoms.positions
    intra_valid_indices = atom_types[i] == atom_types[j]
    intra_indi = i[intra_valid_indices]
    intra_indj =j[intra_valid_indices]
    intra_disp = di[intra_valid_indices, np.newaxis] * cell[0] +\
                    dj[intra_valid_indices, np.newaxis] * cell[1] +\
                    positions[intra_indj] - positions[intra_indi]
    inter_valid_indices = atom_types[i] != atom_types[j]
    inter_indi = i[inter_valid_indices]
    inter_indj = j[inter_valid_indices]
    inter_di = di[inter_valid_indices]
    inter_dj = dj[inter_valid_indices]

    inter_disp = di[inter_valid_indices, np.newaxis] * cell[0] +\
                        dj[inter_valid_indices, np.newaxis] * cell[1] +\
                        positions[inter_indj] - positions[inter_indi]
    inter_dist = np.linalg.norm(inter_disp,axis=1)

    normal_vectors,alpha = get_normal_vect(intra_disp, intra_indi ,intra_indj)
    ni_dot_d = np.sum( normal_vectors[inter_indi] * inter_disp,axis=1)
    rhoij =np.sqrt(inter_dist**2- (ni_dot_d)**2)
    alpha_ij = (alpha[inter_indi] + alpha[inter_indj])/2
    
    ni = normal_vectors[inter_indi]
    nj = normal_vectors[inter_indj]
    rhat = inter_disp/inter_dist[:,np.newaxis]
    r_alpha = inter_dist/alpha_ij

    S_ij = (np.dot(ni,rhat) * np.dot(nj,rhat))*np.exp(-r_alpha/2)*(-1-r_alpha/2 - r_alpha**2/20 + r_alpha**3/120) +\
            (np.dot(ni,nj)-np.dot(ni,rhat) * np.dot(nj,rhat))*np.exp(-r_alpha/2)*(1+r_alpha/2 - r_alpha**2/10)

    descriptors = np.stack([S_ij, rhoij, inter_dist], axis=1)
    #layer 1
    z1 = np.dot(descriptors, W1) + b1
    a1 = np.maximum(0, z1) # ReLU activation function
    
    # Layer 2 (Output)
    output = np.dot(a1, W2) + b2 
    
    x = dist/cutoff
    Taper =  20*np.power(x,7) - 70*np.power(x,7) + 84*np.power(x,5) - 35*np.power(x,4) +1
    V_ij = np.exp(-lambda_val*(inter_dist - z0)) * output[:,0] - output[:,1]*np.power(inter_dist/z0,-6)
    V = 0.5 *  np.sum(V_ij*Taper)
    return V

if __name__ == "__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    from pythtb import *
    from bilayer_letb.api import tb_model
    from BLG_model_builder.geom_tools import *
    from BLG_model_builder.Lammps_Utils import *
    import pandas as pd
    potential_forces_test=  True
    potential_test = False

    if potential_forces_test:
        h=1e-5
        tersoff_parameters = np.array([ 1.52364644, 17.31982283, -4.91503371,  2.07752859, 26.07738439,  0.13340904, 1.65483593,  4.51187634,  2.47458744])
        kc_vdw_parameters = np.array([ 3.74941839,  30.54897089, -52.52708088, -55.42460285,  -7.30650303,2.72942086,   2.39412166,   0.07028751,0,0,0])
        atoms = get_bilayer_atoms(3.35,0)
        Epot, tersoff_forces = Tersoff(atoms,tersoff_parameters)
        _,kc_vdw_forces = Kolmogorov_Crespi_vdw(atoms,kc_vdw_parameters)

        N = len(atoms)
        fd_forces_tersoff = np.zeros((N, 3))
        fd_forces_kc = np.zeros((N, 3))
        for i in range(N):
            for d in range(3):
                displaced = atoms.copy()
                displaced.positions[i, d] += h
                Ep_plus_tersoff, _ = Tersoff(displaced, tersoff_parameters)
                Ep_plus_kc,_ = Kolmogorov_Crespi_vdw(displaced,kc_vdw_parameters)

                displaced.positions[i, d] -= 2 * h
                Ep_minus_tersoff, _ = Tersoff(displaced, tersoff_parameters)
                Ep_minus_kc,_ = Kolmogorov_Crespi_vdw(displaced,kc_vdw_parameters)

                fd_forces_tersoff[i, d] = -(Ep_plus_tersoff - Ep_minus_tersoff) / (2 * h)
                fd_forces_kc[i, d] = -(Ep_plus_kc - Ep_minus_kc) / (2 * h)

        print("RMSE forces tersoff = ",np.linalg.norm(fd_forces_tersoff - tersoff_forces)/N)
        print("RMSE forces KC vdw = ",np.linalg.norm(fd_forces_kc - kc_vdw_forces)/N)
        print(fd_forces_kc)
        print(kc_vdw_forces)


    if potential_test:


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

