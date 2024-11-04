import numpy as np
from scipy.spatial.distance import cdist
try:
    import cupy
    if cupy.cuda.is_available():
        from cupyx.scipy.spatial.distance import cdist
        np = cupy
        gpu_avail = True
except:
    gpu_avail = False

class NN_list:
    def __init__(self,atoms,units = "angstroms",cutoff=5.29):
        self.atoms = atoms
        self.units = units
        self.cutoff = cutoff
        if self.units == "bohr":
            self.conversion = 1.0/.529177
        elif units == "angstroms":
            self.conversion = 1
        else:
            print("only angstroms or bohr accetable units")
        self.displacements = None
        self.distances = None
        self.i = None
        self.j = None
        self.di = None
        self.dj = None
        self.build_NN_list()
        

    def build_NN_list(self):
       
        positions = self.atoms.positions*self.conversion
        natoms = len(self.atoms)
        cell = self.atoms.get_cell()*self.conversion

        self.extended_di = []
        self.extended_dj = []
        extended_coords = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                extended_coords += list(positions[:, :] + cell[0, np.newaxis] * dx + cell[1, np.newaxis] * dy)
                self.extended_di += [dx] * natoms
                self.extended_dj += [dy] * natoms
        self.distance_table = cdist(positions, extended_coords)
        self.set_cutoff(self.cutoff)

    def set_cutoff(self,cutoff=None):
        self.cutoff = cutoff
        i, j = np.where((self.distance_table > 0.1)  & (self.distance_table < cutoff))
        self.di = np.array(self.extended_di)[j]
        self.dj = np.array(self.extended_dj)[j]
        self.i  = np.array(i)
        self.j  = np.array(j % len(self.atoms))

    def get_NN_list(self):
        return self.i,self.j,self.di,self.dj


    def get_distances(self,index_i,cutoff=None):
        if cutoff is not None:
            self.cutoff = cutoff
            self.set_cutoff(cutoff=cutoff)
        if self.displacements is None:
            self.get_displacements()
        if self.distances is None:
            neighbor_ind = self.j[np.where(self.i==index_i)]
            dist = self.atoms.get_distances(index_i,neighbor_ind,mic=True,vector=False)
        return dist
    
    def get_displacements(self,index_i=None,cutoff=None):
        if cutoff is not None:
            self.set_cutoff(cutoff)
        if index_i is None:
            cell = self.atoms.get_cell()
            positions = self.atoms.positions
            displacements =   self.di[:, np.newaxis] * cell[0] +\
                self.dj[:, np.newaxis] * cell[1] +\
                positions[self.j] - positions[self.i]
        else:
            neighbor_ind = self.j[np.where(self.i==index_i)]
            displacements = self.atoms.get_distances(index_i,neighbor_ind,mic=True,vector=True)
            
        return displacements
    
    def get_NN_distances(self,n_neighbors):
        self.distance_table[self.distance_table >= self.cutoff] = np.inf
    
        # Sort distances for each atom and get the indices of the smallest ones
        neighbor_indices = np.argsort(self.distance_table, axis=1)[:, :n_neighbors]
        
        # Get the corresponding neighbor atom indices
        nearest_neighbors = self.j[neighbor_indices]
        distances = self.get_distances()
        return distances[nearest_neighbors]
    
    def get_NN_displacements(self,n_neighbors):
        self.distance_table[self.distance_table >= self.cutoff] = np.inf
    
        # Sort distances for each atom and get the indices of the smallest ones
        neighbor_columns = np.argsort(self.distance_table, axis=1)[:, 1:n_neighbors+1]%len(self.atoms)
        neighbor_rows = np.arange(len(self.atoms))
        cell = self.atoms.get_cell()
        positions= self.atoms.positions
        nn_displacement = np.zeros((len(self.atoms),3,n_neighbors))
        for i in range(n_neighbors):
            nn_displacement[:,:,i] = self.di[:, np.newaxis] * cell[0] +\
                self.dj[:, np.newaxis] * cell[1] +\
                positions[neighbor_columns[:,i]] - positions[neighbor_rows]

        return nn_displacement
        
    
    def get_angles(self,return_cosine=False):
        
        # Calculate pairwise distance vectors
        rij = self.get_displacements() #positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        
        # Calculate norms of distance vectors
        rij_norms = np.linalg.norm(rij,axis=1)

        # Normalize distance vectors
        rij_normalized = rij / rij_norms[:, np.newaxis]
        
        # Calculate dot products between all pairs of normalized vectors
        cos_theta = np.einsum('ijk,ijk->ij', 
                              (rij_normalized[np.newaxis, :, :]) ,
                              (rij_normalized[:, np.newaxis, :]) )
        
        # Clip values to avoid numerical issues
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calculate angles in radians
        if return_cosine:
            return cos_theta
        else:
            return np.arccos(cos_theta)
        
if __name__=="__main__":
    import flatgraphene as fg

    sep = 3.35
    a = 2.46
    n=5
    theta=21.78
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,p=p_found,q=q_found,lat_con=a,sym=["C","C"],mass=[12.01,12.01],sep=sep,h_vac=20)

    nn_list = NN_list(atoms)
    angles = nn_list.get_angles()
    print(angles[:10]*180/np.pi)

