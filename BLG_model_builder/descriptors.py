from scipy.spatial.distance import cdist
import numpy as np
import h5py
import pandas as pd
from BLG_model_builder import NeighborList
import scipy.spatial as spatial
try:
    import cupy
    if cupy.cuda.is_available():
        from cupyx.scipy.spatial.distance import cdist
        np = cupy
        gpu_avail = True
except:
    gpu_avail = False

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
    num_lat_vec_1 = cutoff//(np.linalg.norm(cell[0])/2)+1
    num_lat_vec_2 = cutoff//(np.linalg.norm(cell[1])/2)+1
    lat_vec_iter_1 = [-1,0,1] #np.arange(-num_lat_vec_1,num_lat_vec_1+1)
    lat_vec_iter_2 = [-1,0,1] #np.arange(-num_lat_vec_2,num_lat_vec_2+1)

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
