"""
PyTorch-optimized descriptors module for BLG model builder with Intel GPU acceleration.
This module provides JIT-compiled and vectorized descriptor calculations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np

# Intel GPU device configuration
def get_intel_gpu_device():
    """Get Intel GPU device if available, otherwise CPU."""
    if torch.xpu.is_available():
        device = torch.device("xpu:0")  # Intel GPU
        print("Using Intel GPU:", device)
        return device, True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")  # NVIDIA GPU fallback
        print("Using NVIDIA GPU:", device)
        return device, True
    else:
        device = torch.device("cpu")
        print("Using CPU:", device)
        return device, False

device, gpu_avail = get_intel_gpu_device()

# PyTorch-optimized distance calculation
@torch.jit.script
def cdist_torch(XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized pairwise Euclidean distances with Intel GPU acceleration.
    Equivalent to scipy.spatial.distance.cdist(XA, XB, metric='euclidean').
    """
    XA_norm = torch.sum(XA**2, dim=1, keepdim=True)   # shape (m, 1)
    XB_norm = torch.sum(XB**2, dim=1, keepdim=True).T   # shape (1, n)
    cross = torch.mm(XA, XB.T)                           # shape (m, n)
    D2 = XA_norm + XB_norm - 2 * cross
    return torch.sqrt(torch.clamp(D2, min=0.0))

# PyTorch-optimized nearest neighbor matrix
def nnmat_torch(lattice_vectors: torch.Tensor, atomic_basis: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized nearest neighbor matrix calculation with Intel GPU acceleration.
    """
    nnmat = torch.zeros((len(atomic_basis), 3, 3), device=device, dtype=torch.float32)

    # Create extended atom list
    atoms = []
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            displaced_atoms = atomic_basis + lattice_vectors[0] * i + lattice_vectors[1] * j
            atoms.append(displaced_atoms)
    
    atoms = torch.cat(atoms, dim=0)
    
    # Vectorized distance calculation
    for i in range(len(atomic_basis)):
        displacements = atoms - atomic_basis[i]
        distances = torch.norm(displacements, dim=1)
        ind = torch.argsort(distances)
        nnmat[i] = displacements[ind[1:4]]
    
    return nnmat

# PyTorch-optimized displacement index to distance conversion
@torch.jit.script
def ix_to_dist_torch(lattice_vectors: torch.Tensor, atomic_basis: torch.Tensor, 
                     di: torch.Tensor, dj: torch.Tensor, ai: torch.Tensor, aj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized conversion from displacement indices to physical distances with Intel GPU acceleration.
    """
    displacement_vector = di.unsqueeze(1) * lattice_vectors[0] + \
                        dj.unsqueeze(1) * lattice_vectors[1] + \
                        atomic_basis[aj] - atomic_basis[ai]
    
    displacement_vector_xy = displacement_vector[:, :2]
    displacement_vector_z = displacement_vector[:, -1]
    
    dxy = torch.norm(displacement_vector_xy, dim=1)
    dz = torch.abs(displacement_vector_z)
    
    return dxy, dz

# PyTorch-optimized triangle height calculation
@torch.jit.script
def triangle_height_torch(a: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized triangle height calculation with Intel GPU acceleration.
    """
    area = torch.abs(torch.det(torch.stack([a, base, torch.tensor([1.0, 1.0, 1.0], device=a.device, dtype=a.dtype)]))) / 2
    height = 2 * area / torch.norm(base)
    return height

# PyTorch-optimized descriptor calculations
def t01_descriptors_torch(lattice_vectors: torch.Tensor, atomic_basis: torch.Tensor, 
                         di: torch.Tensor, dj: torch.Tensor, ai: torch.Tensor, aj: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    PyTorch-optimized t01 descriptor calculation with Intel GPU acceleration.
    """
    r = di.unsqueeze(1) * lattice_vectors[0] + dj.unsqueeze(1) * lattice_vectors[1] + \
        atomic_basis[aj] - atomic_basis[ai]
    a = torch.norm(r, dim=1)
    return {'a': a}

def t02_descriptors_torch(lattice_vectors: torch.Tensor, atomic_basis: torch.Tensor, 
                         di: torch.Tensor, dj: torch.Tensor, ai: torch.Tensor, aj: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    PyTorch-optimized t02 descriptor calculation with Intel GPU acceleration.
    """
    r = di.unsqueeze(1) * lattice_vectors[0] + dj.unsqueeze(1) * lattice_vectors[1] + \
        atomic_basis[aj] - atomic_basis[ai]
    
    b = torch.norm(r, dim=1)
    
    # Compute h1 and h2 using vectorized operations
    mat = nnmat_torch(lattice_vectors, atomic_basis)
    
    def compute_h_for_pair(r_i, aj_i):
        nn = mat[aj_i] + r_i
        nndist = torch.norm(nn, dim=1)
        ind = torch.argsort(nndist)
        h1 = triangle_height_torch(nn[ind[0]], r_i)
        h2 = triangle_height_torch(nn[ind[1]], r_i)
        return h1, h2
    
    # Vectorized computation using torch.vmap equivalent
    h1_list = []
    h2_list = []
    for i in range(len(r)):
        h1, h2 = compute_h_for_pair(r[i], aj[i])
        h1_list.append(h1)
        h2_list.append(h2)
    
    h1 = torch.stack(h1_list)
    h2 = torch.stack(h2_list)
    
    return {'h1': h1, 'h2': h2, 'b': b}

def t03_descriptors_torch(lattice_vectors: torch.Tensor, atomic_basis: torch.Tensor, 
                         di: torch.Tensor, dj: torch.Tensor, ai: torch.Tensor, aj: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    PyTorch-optimized t03 descriptor calculation with Intel GPU acceleration.
    """
    r = di.unsqueeze(1) * lattice_vectors[0] + dj.unsqueeze(1) * lattice_vectors[1] + \
        atomic_basis[aj] - atomic_basis[ai]
    
    c = torch.norm(r, dim=1)
    
    # Vectorized calculation of l and h
    mat = nnmat_torch(lattice_vectors, atomic_basis)
    
    def compute_lh_for_pair(r_i, ai_i, aj_i):
        # Compute b, d, h3, h4
        nn = mat[aj_i] + r_i
        nndist = torch.norm(nn, dim=1)
        ind = torch.argsort(nndist)
        b = nndist[ind[0]]
        d = nndist[ind[1]]
        h3 = triangle_height_torch(nn[ind[0]], r_i)
        h4 = triangle_height_torch(nn[ind[1]], r_i)
        
        # Compute a, e, h1, h2
        nn = r_i - mat[ai_i]
        nndist = torch.norm(nn, dim=1)
        ind = torch.argsort(nndist)
        a = nndist[ind[0]]
        e = nndist[ind[1]]
        h1 = triangle_height_torch(nn[ind[0]], r_i)
        h2 = triangle_height_torch(nn[ind[1]], r_i)
        
        l = (a + b + d + e) / 4
        h = (h1 + h2 + h3 + h4) / 4
        
        return l, h
    
    # Vectorized computation using torch.vmap equivalent
    l_list = []
    h_list = []
    for i in range(len(r)):
        l, h = compute_lh_for_pair(r[i], ai[i], aj[i])
        l_list.append(l)
        h_list.append(h)
    
    l = torch.stack(l_list)
    h = torch.stack(h_list)
    
    return {'c': c, 'h': h, 'l': l}

# PyTorch-optimized displacement calculation
def get_disp(positions: torch.Tensor, cell: torch.Tensor, atom_types: torch.Tensor, 
                   cutoff: float = 6.0, type: str = "all") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized displacement calculation with Intel GPU acceleration.
    
    Args:
        positions: Atomic positions (N, 3)
        cell: Unit cell vectors (3, 3)
        atom_types: Atom type array (N,)
        cutoff: Distance cutoff
        type_filter: "all", "intralayer", or "interlayer"
    
    Returns:
        disp, i, j, di, dj
    """
    positions = positions.to(device=device, dtype=torch.float32)
    cell = cell.to(device=device, dtype=torch.float32)
    atom_types = atom_types.to(device=device, dtype=torch.long)
    
    natoms = len(positions)
    
    # Create extended coordinates
    di_list = []
    dj_list = []
    extended_coords = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            displaced_positions = positions + cell[0] * dx + cell[1] * dy
            extended_coords.append(displaced_positions)
            di_list.extend([dx] * natoms)
            dj_list.extend([dy] * natoms)
    
    extended_coords = torch.cat(extended_coords, dim=0)
    di = torch.tensor(di_list, device=device, dtype=torch.long)
    dj = torch.tensor(dj_list, device=device, dtype=torch.long)
    
    # Calculate distances
    distances = cdist_torch(positions, extended_coords)
    
    # Find valid pairs
    valid_mask = (distances > 0.529) & (distances < cutoff)
    i, j = torch.where(valid_mask)
    
    di_valid = di[j]
    dj_valid = dj[j]
    j_valid = j % natoms
    
    # Calculate displacements
    disp = di_valid.unsqueeze(1) * cell[0] + \
           dj_valid.unsqueeze(1) * cell[1] + \
           positions[j_valid] - positions[i]
    
    if type == "all":
        return disp, i, j_valid, di_valid, dj_valid
    
    elif type == "intralayer":
        intra_valid = atom_types[i] == atom_types[j_valid]
        return _filter_disp_torch(disp, i, j_valid, di_valid, dj_valid, intra_valid, cell, positions)
    
    elif type == "interlayer":
        inter_valid = atom_types[i] != atom_types[j_valid]
        return _filter_disp_torch(disp, i, j_valid, di_valid, dj_valid, inter_valid, cell, positions)

def _filter_disp_torch(disp: torch.Tensor, i: torch.Tensor, j: torch.Tensor, 
                      di: torch.Tensor, dj: torch.Tensor, valid_mask: torch.Tensor,
                      cell: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to filter displacements based on atom types."""
    i_filtered = i[valid_mask]
    j_filtered = j[valid_mask]
    di_filtered = di[valid_mask]
    dj_filtered = dj[valid_mask]
    
    disp_filtered = di_filtered.unsqueeze(1) * cell[0] + \
                   dj_filtered.unsqueeze(1) * cell[1] + \
                   positions[j_filtered] - positions[i_filtered]
    
    return disp_filtered, i_filtered, j_filtered, di_filtered, dj_filtered

# PyTorch-optimized orientation calculation
def ix_to_orientation_torch(lattice_vectors: torch.Tensor, atomic_basis: torch.Tensor, 
                           di: torch.Tensor, dj: torch.Tensor, ai: torch.Tensor, aj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized orientation calculation with Intel GPU acceleration.
    """
    displacement_vector = di.unsqueeze(1) * lattice_vectors[0] + \
                        dj.unsqueeze(1) * lattice_vectors[1] + \
                        atomic_basis[aj] - atomic_basis[ai]
    
    mat = nnmat_torch(lattice_vectors, atomic_basis)
    
    def compute_angles(disp, ai_i, aj_i):
        inn = mat[ai_i]
        jnn = mat[aj_i]
        
        sin_jnn = torch.cross(jnn[:, :2], disp[:2], dim=1)
        sin_inn = torch.cross(inn[:, :2], disp[:2], dim=1)
        cos_jnn = torch.dot(jnn[:, :2], disp[:2])
        cos_inn = torch.dot(inn[:, :2], disp[:2])
        
        theta_jnn = torch.atan2(sin_jnn, cos_jnn)
        theta_inn = torch.atan2(sin_inn, cos_inn)
        
        theta_12 = torch.pi - theta_jnn[0]
        theta_21 = theta_inn[0]
        
        return theta_12, theta_21
    
    # Vectorized computation using torch.vmap equivalent
    theta_12_list = []
    theta_21_list = []
    for i in range(len(displacement_vector)):
        theta_12, theta_21 = compute_angles(displacement_vector[i], ai[i], aj[i])
        theta_12_list.append(theta_12)
        theta_21_list.append(theta_21)
    
    theta_12 = torch.stack(theta_12_list)
    theta_21 = torch.stack(theta_21_list)
    
    return theta_12.to(device), theta_21.to(device)

# PyTorch-optimized LETB descriptors
def letb_intralayer_descriptors_torch(positions: torch.Tensor, cell: torch.Tensor, 
                                     atom_types: torch.Tensor, cutoff: float = 6.0) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized LETB intralayer descriptors with Intel GPU acceleration.
    """
    disp, i, j, di, dj = get_disp_torch(positions, cell, atom_types, cutoff, "intralayer")
    distances = torch.norm(disp, dim=1)
    min_distance = torch.min(distances)
    
    # Create masks for different distance ranges
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)
    t02_ix = (distances >= 0.95 * torch.sqrt(torch.tensor(3.0)) * min_distance) & (distances <= 1.05 * torch.sqrt(torch.tensor(3.0)) * min_distance)
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
    
    # Compute descriptors for each range
    t01 = t01_descriptors_torch(cell, positions, di[t01_ix], dj[t01_ix], i[t01_ix], j[t01_ix])
    t02 = t02_descriptors_torch(cell, positions, di[t02_ix], dj[t02_ix], i[t02_ix], j[t02_ix])
    t03 = t03_descriptors_torch(cell, positions, di[t03_ix], dj[t03_ix], i[t03_ix], j[t03_ix])
    
    return (t01, t02, t03, distances), i, j, di, dj

def letb_interlayer_descriptors_torch(positions: torch.Tensor, cell: torch.Tensor, 
                                     atom_types: torch.Tensor, cutoff: float = 6.0) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized LETB interlayer descriptors with Intel GPU acceleration.
    """
    disp, i, j, di, dj = get_disp_torch(positions, cell, atom_types, cutoff, "interlayer")
    
    dist_xy = torch.norm(disp[:, :2], dim=1)
    dist_z = torch.abs(disp[:, 2])
    dist = torch.norm(disp, dim=1)
    
    # Compute orientation angles
    theta_12, theta_21 = ix_to_orientation_torch(cell, positions, di, dj, i, j)
    
    output = {
        'dxy': dist_xy,
        'dz': dist_z,
        'd': dist,
        'theta_12': theta_12,
        'theta_21': theta_21,
    }
    
    return output, i, j, di, dj

# ASE atoms compatibility functions
def atoms_to_torch_tensors(atoms) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert ASE atoms object to PyTorch tensors for Intel GPU acceleration.
    
    Args:
        atoms: ASE atoms object
    
    Returns:
        positions, cell, atom_types as PyTorch tensors
    """
    # Convert to NumPy arrays first to avoid copy issues
    import numpy as np
    
    positions_np = np.array(atoms.positions)
    cell_np = np.array(atoms.get_cell())
    
    if atoms.has("mol-id"):
        atom_types_np = np.array(atoms.get_array("mol-id"))
    else:
        # Default to single layer if no mol-id
        atom_types_np = np.ones(len(atoms), dtype=np.int32)
    
    # Convert to PyTorch tensors
    positions = torch.tensor(positions_np, dtype=torch.float32, device=device)
    cell = torch.tensor(cell_np, dtype=torch.float32, device=device)
    atom_types = torch.tensor(atom_types_np, dtype=torch.long, device=device)
    
    return positions, cell, atom_types

def torch_tensors_to_atoms(positions: torch.Tensor, cell: torch.Tensor, atom_types: torch.Tensor, atoms_template):
    """
    Convert PyTorch tensors back to ASE atoms object.
    
    Args:
        positions: PyTorch tensor of positions
        cell: PyTorch tensor of cell vectors
        atom_types: PyTorch tensor of atom types
        atoms_template: Original ASE atoms object to use as template
    
    Returns:
        ASE atoms object with updated positions and cell
    """
    atoms = atoms_template.copy()
    atoms.set_positions(positions.detach().cpu().numpy())
    atoms.set_cell(cell.detach().cpu().numpy())
    atoms.set_array("mol-id", atom_types.detach().cpu().numpy())
    return atoms
