"""
PyTorch-optimized TB_Utils module for BLG model builder with Intel GPU acceleration.
This module provides JIT-compiled and vectorized tight-binding calculations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
# Intel GPU device configuration
def get_intel_gpu_device():
    """Get Intel GPU device if available, otherwise CPU."""
    #Original code commented out for debugging:
    if torch.xpu.is_available():
        device = torch.device("xpu:0")  # Intel GPU
        return device, True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")  # NVIDIA GPU fallback
        return device, True
    else:
        device = torch.device("cpu")
        return device, False

device, gpu_avail = get_intel_gpu_device()

# PyTorch-optimized distance calculation
@torch.jit.script
def cdist_torch(XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized pairwise Euclidean distances with Intel GPU acceleration.
    """
    XA_norm = torch.sum(XA**2, dim=1, keepdim=True)
    XB_norm = torch.sum(XB**2, dim=1, keepdim=True).T
    cross = torch.mm(XA, XB.T)
    D2 = XA_norm + XB_norm - 2 * cross
    return torch.sqrt(torch.clamp(D2, min=0.0))

# PyTorch-optimized Chebyshev polynomial evaluation
@torch.jit.script
def chebval_torch(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Chebyshev polynomial evaluation with Intel GPU acceleration.
    """
    if len(c) == 1:
        c0 = c[0]
        c1 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    
    return c0 + c1 * x

# PyTorch-optimized norm calculation
@torch.jit.script
def norm_torch(a: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized norm calculation with Intel GPU acceleration.
    """
    return torch.sqrt(torch.sum(a**2, dim=1))

# PyTorch-optimized Slater-Koster functions
@torch.jit.script
def SK_pz_chebyshev_torch(dR: torch.Tensor, params: torch.Tensor, aa: float = 0.529, b: float = 5.29177) -> torch.Tensor:
    """
    PyTorch-optimized Slater-Koster pz orbital calculation with Chebyshev polynomials.
    """
    r = torch.norm(dR, dim=1)
    dRn = dR / r.unsqueeze(1)
    
    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    
    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    
    y = (2*r - (b+aa))/(b-aa)
    Vpp_sigma = chebval_torch(y, Cpp_sigma)
    Vpp_pi = chebval_torch(y, Cpp_pi)
    
    Vpp_sigma -= params[0]/2
    Vpp_pi -= params[10]/2
    
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi
    return Ezz

@torch.jit.script
def SK_bond_ints_torch(r: torch.Tensor, params: torch.Tensor, aa: float = 0.529, b: float = 5.29177) -> torch.Tensor:
    """
    PyTorch-optimized Slater-Koster bond integrals with Intel GPU acceleration.
    """
    y = (2*r - (b+aa))/(b-aa)
    bond_val = chebval_torch(y, params)
    bond_val -= params[0]/2
    return bond_val

# PyTorch-optimized Popov hopping
@torch.jit.script
def popov_hopping(disp: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Popov hopping calculation with Intel GPU acceleration.
    """
    bohr_per_ang = 1.8897259886
    dR = disp * bohr_per_ang
    dRn = torch.norm(dR, dim=1)
    dRn = dR / dRn.unsqueeze(1)
    
    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = torch.norm(dR, dim=1)
    
    aa = 1.0
    b = 10.0
    y = (2.0 * r - (b + aa)) / (b - aa)
    
    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    Vpp_sigma = chebval_torch(y, Cpp_sigma)
    Vpp_pi = chebval_torch(y, Cpp_pi)
    
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    
    return Ezz

@torch.jit.script
def popov_overlap(disp: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Popov overlap calculation with Intel GPU acceleration.
    """
    bohr_per_ang = 1.8897259886
    dR = disp * bohr_per_ang
    dRn = torch.norm(dR, dim=1)
    dRn = dR / dRn.unsqueeze(1)
    
    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = torch.norm(dR, dim=1)
    
    aa = 1.0
    b = 10.0
    y = (2.0 * r - (b + aa)) / (b - aa)
    
    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    Vpp_sigma = chebval_torch(y, Cpp_sigma)
    Vpp_pi = chebval_torch(y, Cpp_pi)
    
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    
    return Ezz

# PyTorch-optimized Porezag hopping
@torch.jit.script
def porezag_hopping(disp: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Porezag hopping calculation with Intel GPU acceleration.
    """
    bohr_per_ang = 1.8897259886
    dR = disp * bohr_per_ang
    dRn = torch.norm(dR, dim=1)
    dRn = dR / dRn.unsqueeze(1)
    
    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = torch.norm(dR, dim=1)
    
    aa = 1.0
    b = 7.0
    y = (2.0 * r - (b + aa)) / (b - aa)
    
    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    Vpp_sigma = chebval_torch(y, Cpp_sigma)
    Vpp_pi = chebval_torch(y, Cpp_pi)
    
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    
    return Ezz

@torch.jit.script
def porezag_overlap(disp: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Porezag overlap calculation with Intel GPU acceleration.
    """
    bohr_per_ang = 1.8897259886
    dR = disp * bohr_per_ang
    dRn = torch.norm(dR, dim=1)
    dRn = dR / dRn.unsqueeze(1)
    
    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = torch.norm(dR, dim=1)
    
    aa = 1.0
    b = 7.0
    y = (2.0 * r - (b + aa)) / (b - aa)
    
    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    Vpp_sigma = chebval_torch(y, Cpp_sigma)
    Vpp_pi = chebval_torch(y, Cpp_pi)
    
    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    
    return Ezz

# PyTorch-optimized Moon-Koshino hopping
@torch.jit.script
def mk_hopping(descriptors: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Moon-Koshino hopping calculation with Intel GPU acceleration.
    """
    r = torch.norm(descriptors, dim=1)
    a, b, c = parameters[0], parameters[1], parameters[2]
    a0 = 1.42
    d0 = 3.35
    n = descriptors[:, 2] / r
    V_p = a * torch.exp(-b * (r - a0))
    V_s = c * torch.exp(-b * (r - d0))
    hoppings = V_p * (1 - n**2) + V_s * n**2
    return hoppings

@torch.jit.script
def mk_overlap(descriptors: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Moon-Koshino overlap calculation with Intel GPU acceleration.
    """
    r = torch.norm(descriptors, dim=1)
    a, b, c = parameters[0], parameters[1], parameters[2]
    a0 = 1.42
    d0 = 3.35
    n = descriptors[:, 2] / r
    V_p = a * torch.exp(-b * (r - a0))
    V_s = c * torch.exp(-b * (r - d0))
    hoppings = V_p * (1 - n**2) + V_s * n**2
    return hoppings

# PyTorch-optimized LETB helper functions (defined first for JIT compilation)
@torch.jit.script
def letb_intralayer_t01(descriptors: torch.Tensor, t01_params: torch.Tensor) -> torch.Tensor:
    """PyTorch-optimized t01 calculation with Intel GPU acceleration."""
    return descriptors * t01_params[1:] + t01_params[0]

@torch.jit.script
def letb_intralayer_t02(descriptors: torch.Tensor, t02_params: torch.Tensor) -> torch.Tensor:
    """PyTorch-optimized t02 calculation with Intel GPU acceleration."""
    return torch.sum(descriptors * t02_params[1:], dim=1) + t02_params[0]

@torch.jit.script
def letb_intralayer_t03(descriptors: torch.Tensor, t03_params: torch.Tensor) -> torch.Tensor:
    """PyTorch-optimized t03 calculation with Intel GPU acceleration."""
    return torch.sum(descriptors * t03_params[1:], dim=1) + t03_params[0]

# PyTorch-optimized LETB main function
@torch.jit.script
def letb_intralayer(descriptors: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor], 
                         parameters: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized LETB intralayer calculation with Intel GPU acceleration.
    """
    t01_params = parameters[:2]
    t02_params = parameters[2:6]
    t03_params = parameters[6:]
    
    distances = descriptors[3]
    min_distance = torch.min(distances)
    
    # Create masks for different distance ranges
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)
    t02_ix = (distances >= 0.95 * torch.sqrt(torch.tensor(3.0)) * min_distance) & (distances <= 1.05 * torch.sqrt(torch.tensor(3.0)) * min_distance)
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)
    
    # Compute hoppings for each range
    hoppings = torch.zeros_like(distances)
    
    # t01 hoppings
    t01_hoppings = letb_intralayer_t01(descriptors[0]["a"], t01_params)
    hoppings = torch.where(t01_ix, t01_hoppings, hoppings)
    
    # t02 hoppings
    t02_descriptors = torch.stack([descriptors[1]["h1"], descriptors[1]["h2"], descriptors[1]["b"]], dim=1)
    t02_hoppings = letb_intralayer_t02(t02_descriptors, t02_params)
    hoppings = torch.where(t02_ix, t02_hoppings, hoppings)
    
    # t03 hoppings
    t03_descriptors = torch.stack([descriptors[2]["c"], descriptors[2]["h"], descriptors[2]["l"]], dim=1)
    t03_hoppings = letb_intralayer_t03(t03_descriptors, t03_params)
    hoppings = torch.where(t03_ix, t03_hoppings, hoppings)
    
    # Zero out invalid distances
    hoppings = torch.where(t00, torch.zeros_like(hoppings), hoppings)
    
    return hoppings

@torch.jit.script
def letb_interlayer(descriptors: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized LETB interlayer calculation with Intel GPU acceleration.
    """
    # Access parameters by index instead of tuple unpacking for JIT compatibility
    a0 = parameters[0]
    b0 = parameters[1]
    c0 = parameters[2]
    a3 = parameters[3]
    b3 = parameters[4]
    c3 = parameters[5]
    a6 = parameters[6]
    b6 = parameters[7]
    c6 = parameters[8]
    d6 = parameters[9]
    
    r = descriptors[:, 0]
    theta12 = descriptors[:, 1]
    theta21 = descriptors[:, 2]
    
    r = r / 4.649
    
    v0 = a0 * torch.exp(-b0 * r**2) * torch.cos(c0 * r)
    v3 = a3 * r**2 * torch.exp(-b3 * (r - c3)**2)
    v6 = a6 * torch.exp(-b6 * (r - c6)**2) * torch.sin(d6 * r)
    
    hoppings = v0
    hoppings += v3 * (torch.cos(3 * theta12) + torch.cos(3 * theta21))
    hoppings += v6 * (torch.cos(6 * theta12) + torch.cos(6 * theta21))
    
    return hoppings
    
# PyTorch-optimized Hellman-Feynman forces
@torch.jit.script
def get_hellman_feynman_torch(eigvals: torch.Tensor, eigvec: torch.Tensor,
                             kpoint: torch.Tensor, disp: torch.Tensor,
                             grad_hop: torch.Tensor, grad_overlap: torch.Tensor,
                             hop_i: torch.Tensor, hop_j: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Hellman-Feynman force calculation following F = -ρ * dH/dR + ρ_e * dS/dR.
    
    The Hellman-Feynman theorem states: F_i = -<ψ|dH/dR_i|ψ> + <ψ|dS/dR_i|ψ>
    where:
    - ρ is the density matrix: ρ_uv = Σ_i^Nocc c*_ui c_vi
    - ρ_e is the energy-weighted density matrix: ρ_e_uv = Σ_i^Nocc ε_i c*_ui c_vi
    - dH/dR_i is the derivative of the Hamiltonian with respect to position R_i
    - dS/dR_i is the derivative of the overlap matrix with respect to position R_i
    
    Args:
        eigvals: Eigenvalues [norbs]
        eigvec: Eigenvectors [norbs, norbs]
        kpoint: K-point vector [3]
        disp: Displacement vectors [npairs, 3]
        grad_hop: Hopping gradients [npairs, 3]
        grad_overlap: Overlap gradients [npairs, 3]
        hop_i: Atom indices i [npairs]
        hop_j: Atom indices j [npairs]
    
    Returns:
        Forces: Hellman-Feynman forces [natoms, 3]
    """
    natoms = eigvec.shape[0]
    nocc = natoms // 2  # Assuming bilayer
    npairs = len(hop_i)
    # Reshape gradients from [npairs*3] to [npairs, 3]
    grad_hop = grad_hop.reshape(npairs, 3)
    grad_overlap = grad_overlap.reshape(npairs, 3)
    all_i_idx = torch.cat((hop_i, hop_j))
    all_j_idx = torch.cat([hop_j, hop_i])
    
    # Get the corresponding derivative values
    dH_values = -torch.cat((grad_hop, -grad_hop))
    dS_values = -torch.cat((grad_overlap, -grad_overlap))
    disp_values = torch.cat((disp,-disp))
    
    # Build density matrices from occupied states
    # Construct density matrix: ρ_uv = Σ_i^Nocc c*_ui c_vi
    density_matrix = 2*eigvec[:, :nocc] @ eigvec[:, :nocc].T
    energy_density_matrix = 2*eigvec[:, :nocc] @ torch.diag(eigvals[:nocc].to(eigvec.dtype)) @ eigvec[:, :nocc].T

    
    # Calculate phase factors for all pairs
    phases = torch.exp(1.0j * torch.mm(kpoint.unsqueeze(0), disp_values.T)).squeeze()  # [npairs]
    # Get density matrix elements for all pairs
    rho_ij = density_matrix[all_i_idx, all_j_idx]  # [npairs]
    rho_e_ij = energy_density_matrix[all_i_idx, all_j_idx]  # [npairs]

    # Hellmann–Feynman + Pulay: F_I = -Tr[P dH/dR_I] + Tr[W dS/dR_I]
    weight = (rho_ij * phases)          # [npairs] complex
    weight_e = (rho_e_ij * phases)      # [npairs] complex

    # Force contribution on atom i from each pair (i,j)
    f_amp = -weight.unsqueeze(1) * dH_values.to(weight.dtype) + weight_e.unsqueeze(1) * dS_values.to(weight_e.dtype)  # [npairs, 3] complex

    f_ij = f_amp

    # Initialize forces tensor as COMPLEX - will be summed over k-points before taking real part
    Forces = torch.zeros((natoms, 3), device=eigvec.device, dtype=torch.complex64)
    # Accumulate to atom i and equal-and-opposite to atom j (enforce Newton's third law)
    Forces[:,0].index_add_(0, all_i_idx, f_ij[:,0])
    Forces[:,1].index_add_(0, all_i_idx, f_ij[:,1])
    Forces[:,2].index_add_(0, all_i_idx, f_ij[:,2])
    del density_matrix, energy_density_matrix, rho_ij, rho_e_ij, weight, weight_e, f_amp, f_ij
    return Forces

# PyTorch-optimized utility functions
@torch.jit.script
def get_recip_cell_torch(cell: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized reciprocal cell calculation with Intel GPU acceleration.
    Handles both 2D and 3D cells.
    """
    a1 = cell[0, :]
    a2 = cell[1, :]
    a3 = cell[2, :]
    
    volume = torch.dot(a1, torch.cross(a2, a3, dim=0))
    
    # Handle 2D case where volume is near zero
    volume = torch.where(torch.abs(volume) < 1e-10, torch.tensor(1.0, device=cell.device), volume)
    
    b1 = 2 * torch.pi * torch.cross(a2, a3, dim=0) / volume
    b2 = 2 * torch.pi * torch.cross(a3, a1, dim=0) / volume
    b3 = 2 * torch.pi * torch.cross(a1, a2, dim=0) / volume
    
    return torch.stack([b1, b2, b3])

def k_uniform_mesh_torch(mesh_size: Tuple[int, int, int], device: torch.device = None) -> torch.Tensor:
    """
    PyTorch-optimized uniform k-mesh generation with Intel GPU acceleration.
    """
    if device is None:
        device = torch.device("cpu")
    
    nx, ny, nz = mesh_size
    
    # Create coordinate arrays
    x = torch.linspace(0, 1, nx, device=device, dtype=torch.float32)
    y = torch.linspace(0, 1, ny, device=device, dtype=torch.float32)
    z = torch.linspace(0, 1, nz, device=device, dtype=torch.float32)
    
    # Create meshgrid
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Reshape to final format
    k_vec = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    return k_vec

def k_path_torch(sym_pts: torch.Tensor, nk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized k-path generation with Intel GPU acceleration.
    """
    # number of nodes
    k_list=sym_pts
    n_nodes=k_list.shape[0]

    mesh_step = nk//(n_nodes-1)
    mesh = torch.linspace(0,1,mesh_step)
    step = (torch.arange(0,mesh_step,1)/mesh_step)

    kvec = torch.zeros((0,3))
    knode = torch.zeros(n_nodes)
    for i in range(n_nodes-1):
        n1 = k_list[i,:]
        n2 = k_list[i+1,:]
        diffq = torch.outer((n2 - n1),  step).T + n1

        dn = torch.linalg.norm(n2-n1)
        knode[i+1] = dn + knode[i]
        if i==0:
            kvec = torch.vstack((kvec,diffq))
        else:
            kvec = torch.vstack((kvec,diffq))
    kvec = torch.vstack((kvec,k_list[-1,:]))

    dk_ = torch.zeros(kvec.shape[0])
    for i in range(1,kvec.shape[0]):
        dk_[i] = torch.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]

    return (kvec,dk_, knode)

# Model function mappings for PyTorch
def _get_hopping_model_torch(model_name: str) -> Callable:
    """Get hopping model function by name for PyTorch."""
    models = {
        'popov': popov_hopping,
        'porezag': porezag_hopping,
        'mk': mk_hopping,
        'letb': letb_interlayer
    }
    return models.get(model_name, popov_hopping)

def _get_overlap_model_torch(model_name: str) -> Callable:
    """Get overlap model function by name for PyTorch."""
    models = {
        'popov': popov_overlap,
        'porezag': porezag_overlap,
        'mk': mk_overlap,
        'letb': lambda x, y: torch.zeros(len(x), device=x.device, dtype=x.dtype)  # No overlap for LETB
    }
    return models.get(model_name, popov_overlap)
