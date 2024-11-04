import numpy as np
import matplotlib.pyplot as plt
import flatgraphene as fg
from scipy.spatial.distance import cdist

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
        
    # get the mesh size and checks for consistency
    use_mesh=np.array(list(map(round,mesh_size)),dtype=int)
    # construct the mesh
    
    # get a mesh
    k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1],0:use_mesh[2]]
    # normalize the mesh
    norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
    norm=norm.reshape(use_mesh.tolist()+[3])
    norm=norm.transpose([3,0,1,2])
    k_vec=k_vec/norm
    # final reshape
    k_vec=k_vec.transpose([1,2,3,0]).reshape([use_mesh[0]*use_mesh[1]*use_mesh[2],3])
    return k_vec

def get_disp(atoms,units = "angstroms",cutoff=5.29):
    if units == "bohr":
        conversion = 1.0/.529177
    elif units == "angstroms":
        conversion = 1
    positions = atoms.positions*conversion
    natoms = len(atoms)
    cell = atoms.get_cell()*conversion
    atom_types = atoms.get_array("mol-id")

    di = []
    dj = []
    extended_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            extended_coords += list(positions[:, :] + cell[0, np.newaxis] * dx + cell[1, np.newaxis] * dy)
            di += [dx] * natoms
            dj += [dy] * natoms
    distances = cdist(positions, extended_coords)

    i, j = np.where((distances > 0.529)  & (distances < cutoff))
    di = np.array(di)[j]
    dj = np.array(dj)[j]
    i  = np.array(i)
    j  = np.array(j % natoms)

    disp =  di[:, np.newaxis] * cell[0] +\
            dj[:, np.newaxis] * cell[1] +\
            positions[j] - positions[i]
    return disp ,i,j,di,dj

def mk_hopping(descriptors):
    r = np.linalg.norm(descriptors,axis=1)
    (a,b,c) = (-2.7, 2.2109794066373403, 0.48)
    a0 = 1.42
    d0 = 3.35
    n = (descriptors[:,2]) / r
    V_p = a * np.exp(-b * (r - a0))
    V_s = c * np.exp(-b * (r - d0))
    hoppings = V_p*(1 - n**2) + V_s * n**2
    return hoppings

def grad_mk_hopping(disp):
    grad_hop = np.zeros_like(disp)

    delta = 1e-5
    for dir_ind in range(3):
        dr = np.zeros(3)
        dr[dir_ind] +=  delta
        hop_up = mk_hopping(disp+dr[np.newaxis,:])
        hop_dwn = mk_hopping(disp-dr[np.newaxis,:])
        grad_hop[:,dir_ind] = (hop_up - hop_dwn)/2/delta
    return grad_hop

def get_tb_energy(atoms,kpoints_reduced):
    norbs =  len(atoms)
    self_energy = -5.2887
    positions = atoms.positions
    cell = atoms.get_cell()

    tb_energy = 0
    nocc = len(atoms)//2
    disp,hop_i,hop_j,hop_di,hop_dj = get_disp(atoms)
    hoppings =  mk_hopping(disp)
    hoppings_grad = grad_mk_hopping(disp)
    recip_cell = get_recip_cell(atoms.get_cell())
    kpoints = kpoints_reduced @ recip_cell.T

    nkp = np.shape(kpoints)[0]

    wf = np.zeros((norbs,norbs,nkp),dtype=complex)

    rho_e = np.zeros_like(hoppings)
    for i in range(nkp):
        ham = self_energy * np.eye(norbs,dtype=np.complex64)

        phase = np.exp((1.0j)*np.dot(kpoints[i,:],disp.T))
        amp = hoppings * phase
        ham[hop_i,hop_j] += amp
        ham[hop_j,hop_i] += np.conj(amp)


        eigvals,wf_k = np.linalg.eigh(ham)
        tb_energy += 2 * np.sum(eigvals[:nocc])
        wf[:,:,i] = wf_k
        
        #get TB forces
        Forces = np.zeros((len(atoms),3))
        fd_dist = 2*np.eye(norbs)
        fd_dist[nocc:,nocc:] = 0

        density_matrix = wf_k @ fd_dist @ np.conj(wf_k).T

        rho =  density_matrix[hop_i,hop_j][:,np.newaxis] 
        gradH = hoppings_grad * phase[:,np.newaxis] * rho
        gradH += np.conj(gradH)

        for atom_ind in range(norbs):
            use_ind = np.squeeze(np.where(hop_i==atom_ind))
            ave_gradH = gradH[use_ind,:]
            if ave_gradH.ndim!=2:
                Forces[atom_ind,:] -= -ave_gradH.real 
            else:
                Forces[atom_ind,:] -= -np.sum(ave_gradH,axis=0).real 

        return tb_energy/nkp, Forces/nkp


if __name__=="__main__":
    
    sep = 3.35
    a = 2.46
    n=5
    theta=3.89
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                    mass=[12.01,12.01],sep=sep,h_vac=20)
    kpoints = k_uniform_mesh((5,5,1))

    un_pert_tb_energy,un_pert_tb_forces = get_tb_energy(atoms,kpoints)
    nn_dist_list = atoms.get_distances(0, np.arange(len(atoms)), mic=True) #get periodic distances for atom 0

    layer_dist= 3.35
    layer_pert = 0.01 

    pert_forces_distance = np.zeros_like(nn_dist_list)

    for i in range(len(atoms)): 
        atom_pert = np.zeros_like(atoms.positions)
        pert_dir = np.random.uniform(size=3)
        #atom_pert[i,:] += pert_dir/np.linalg.norm(pert_dir)[np.newaxis,:] * (bond_length * bond_pert)
        atom_pert[i,:] += np.array([0,0,1]) * (layer_dist * layer_pert)
        perturb_atoms_pos = atoms.positions + atom_pert
        perturb_atoms = atoms.copy()
        perturb_atoms.positions = perturb_atoms_pos
        tb_energy,tb_forces = get_tb_energy(perturb_atoms,kpoints)
        pert_forces_distance[i] = np.linalg.norm(tb_forces[0,:] - un_pert_tb_forces[0,:])

    long_ind = np.where(nn_dist_list>5)
    plt.scatter(nn_dist_list[long_ind],pert_forces_distance[long_ind])
    plt.xlabel("perturbation distance from atom 0")
    plt.ylabel("magnitude of force on atom 0")
    plt.savefig("pert_force_long_range.png")
    plt.clf()