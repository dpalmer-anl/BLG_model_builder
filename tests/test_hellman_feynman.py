import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
import ase.db
from scipy.spatial import distance
from ase.build import make_supercell
import pandas as pd
import torch
from ase.optimize import FIRE
from ase.optimize import LBFGS
from ase.constraints import FixedLine

import BLG_model_builder
from BLG_model_builder.TB_Utils_torch import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors_torch import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder_torch import *
from BLG_model_builder.BLG_model_library import *
from BLG_model_builder.TETB_model_builder_torch import _build_hamiltonian_with_overlap_torch, _build_hamiltonian_no_overlap_torch, _solve_eigenvalue_with_overlap_torch, _solve_eigenvalue_no_overlap_torch
import matplotlib.pyplot as plt
import scipy.linalg
from ase import Atoms
from ase.build import graphene
from scipy.optimize import curve_fit
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
def chebval(x, c):
    """Evaluate a Chebyshev series at points x."""
    c = np.asarray(c)
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
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
def porezag_hopping(disp,params):
    """pairwise Slater Koster hopping parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    bohr_per_ang = 1.8897259886
    dR = disp * bohr_per_ang
    dRn = np.linalg.norm(dR)
    dRn = dR / dRn

    l = dRn[0]
    m = dRn[1]
    n = dRn[2]
    r = np.linalg.norm(dR)

    aa = 1.0   # [Bohr radii]
    b = 7   # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    Vpp_sigma =  chebval(y, Cpp_sigma) 
    Vpp_pi =  chebval(y, Cpp_pi) 

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat

def porezag_overlap(disp,params):
    """pairwise Slater Koster overlap parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Overlap matrix elements [eV]
    """
    bohr_per_ang = 1.8897259886
    dR = disp * bohr_per_ang
    dRn = np.linalg.norm(dR)
    dRn = dR / dRn

    l = dRn[0]
    m = dRn[1]
    n = dRn[2]
    r = np.linalg.norm(dR)

    aa = 1.0   # [Bohr radii]
    b = 7   # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Cpp_sigma = params[:10]
    Cpp_pi = params[10:]
    Vpp_sigma =  chebval(y, Cpp_sigma) 
    Vpp_pi =  chebval(y, Cpp_pi) 

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    return Ezz 

def simple_tetb_energy(atoms,kpoints):
    eV_per_hartree = 27.2114
    porezag_hopping_pp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree
    porezag_hopping_pp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree
    porezag_overlap_pp_pi =np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227])
    porezag_overlap_pp_sigma = np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751])
    hopping_params = np.append(porezag_hopping_pp_sigma,porezag_hopping_pp_pi)
    overlap_params = np.append(porezag_overlap_pp_sigma,porezag_overlap_pp_pi)

    positions = atoms.positions
    cell = atoms.cell.array
    distances, disp_table = minimum_image_distance(positions, cell)
    nkp = kpoints.shape[0]
    recip_cell = get_recip_cell(cell)
    kpoints = kpoints @ recip_cell.T
    natoms = positions.shape[0]
    nocc = natoms//2
    energy = 0
    forces = np.zeros((natoms, 3))
    fd_dist = 2*np.eye(natoms)
    fd_dist[nocc:,nocc:] = 0
    

    dr = 1e-4
    for k in range(nkp):
        phase = np.exp(1j * 2 * np.pi * np.sum(kpoints[k,:][:,None,None]* disp_table.T,axis=0))
        Ham = np.zeros((natoms,natoms),dtype=np.complex64)
        Overlap = np.eye(natoms,dtype=np.complex64)
        for i in range(natoms):
            for j in range(natoms):
                if i != j:
                    if distances[i,j] <3.7:
                        
                        Ham[i,j] = porezag_hopping(disp_table[i,j,:],hopping_params) * phase[i,j]
                        Overlap[i,j] = porezag_overlap(disp_table[i,j,:],overlap_params) * phase[i,j]

        eigvals,eigvecs = scipy.linalg.eigh(Ham,b=Overlap)
        energy += 2 * np.sum(eigvals[:nocc]) / nkp
        occ_eigvals = 2*np.diag(eigvals)
        occ_eigvals[nocc:,nocc:] = 0
        density_matrix = eigvecs @ fd_dist @ np.conj(eigvecs).T
        energy_density_matrix = eigvecs @ occ_eigvals @ np.conj(eigvecs).T

        for d in range(3):
            grad_ham = np.zeros((natoms,natoms),dtype=np.complex64)
            grad_overlap = np.zeros((natoms,natoms),dtype=np.complex64)
            disp_table_pert = disp_table.copy()
            disp_table_pert[:,:,d] += dr
            for i in range(natoms):
                for j in range(natoms):
                    if i != j:
                        if distances[i,j] <3.7:
                            pert_disp = disp_table_pert[i,j,:]
                            pert_disp[d] += dr
                            Ham_pert_up = porezag_hopping(pert_disp,hopping_params) * phase[i,j]
                            Overlap_pert_up = porezag_overlap(pert_disp,overlap_params) * phase[i,j]

                            pert_disp = disp_table_pert[i,j,:]
                            pert_disp[d] -= 2*dr
                            Ham_pert_down = porezag_hopping(pert_disp,hopping_params) * phase[i,j]
                            Overlap_pert_down = porezag_overlap(pert_disp,overlap_params) * phase[i,j]

                            grad_ham = (Ham_pert_up - Ham_pert_down) / (2 * dr)
                            grad_overlap = (Overlap_pert_up - Overlap_pert_down) / (2 * dr)

                            f_ij = grad_ham * density_matrix[i,j] + grad_overlap * energy_density_matrix[i,j]
                            forces[i,d] -= f_ij.real
                            forces[j,d] += f_ij.real
    return energy, forces


def create_dimer_system(separation=3.35):
    """Create a simple dimer system for testing force calculations."""
    # Create two carbon atoms at specified separation
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, separation]
    ])
    
    # Create a large cell to avoid periodic interactions
    cell = np.array([
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 20.0]
    ])
    
    atoms = Atoms('CC', positions=positions, cell=cell, pbc=True)
    
    # Set mol-id to distinguish layers
    atoms.set_array('mol-id', np.array([1, 2]))
    
    return atoms

def test_dimer_force_conservation():
    """Test that forces on dimer system are equal and opposite (Newton's third law)."""
    print("=" * 60)
    print("TESTING DIMER FORCE CONSERVATION")
    print("=" * 60)
    
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'interlayer'  # Use interlayer for dimer
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="hellman_feynman")
    
    # Test different separations
    separations = [3.0, 3.35, 4.0, 5.0]
    
    for sep in separations:
        print(f"\nTesting dimer at separation {sep} Å")
        atoms = create_dimer_system(sep)
        
        try:
            _, forces = calc_hf.get_tb_energy(atoms)
            
            # Check that forces are equal and opposite
            force_diff = forces[0] + forces[1]  # Should be zero
            force_magnitude = np.linalg.norm(force_diff)
            
            print(f"  Force on atom 0: {forces[0]}")
            print(f"  Force on atom 1: {forces[1]}")
            print(f"  Sum of forces: {force_diff}")
            print(f"  Magnitude of force sum: {force_magnitude:.2e}")
            
            if force_magnitude < 1e-10:
                print("  ✓ Forces are equal and opposite (Newton's third law satisfied)")
            else:
                print("  ✗ Forces are NOT equal and opposite (Newton's third law violated)")
                
        except Exception as e:
            print(f"  ✗ Error calculating forces: {e}")

def test_dimer_finite_differences():
    """Test Hellman-Feynman forces against finite differences for dimer system."""
    print("\n" + "=" * 60)
    print("TESTING DIMER FINITE DIFFERENCES")
    print("=" * 60)
    
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'interlayer'
    h = 1e-2  # Finite difference step
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="hellman_feynman")
    calc_autograd = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="autograd")
    
    # Test different separations
    separations = [3.0, 3.35, 4.0, 5.0]
    
    for sep in separations:
        print(f"\nTesting dimer at separation {sep} Å")
        atoms = create_dimer_system(sep)
        
        try:
            # Calculate Hellman-Feynman forces
            _, hf_forces = calc_hf.get_tb_energy(atoms)
            _, autograd_forces = calc_autograd.get_tb_energy(atoms)
            
            # Calculate finite difference forces
            N = len(atoms)
            fd_forces = np.zeros((N, 3))
            
            for i in range(N):
                for d in range(3):
                    # Forward difference
                    displaced_plus = atoms.copy()
                    displaced_plus.positions[i, d] += h
                    E_plus, _ = calc_hf.get_tb_energy(displaced_plus)
                    
                    # Backward difference
                    displaced_minus = atoms.copy()
                    displaced_minus.positions[i, d] -= h
                    E_minus, _ = calc_hf.get_tb_energy(displaced_minus)
                    
                    # Central difference
                    fd_forces[i, d] = -(E_plus - E_minus) / (2 * h)

            # Compare forces
            force_diff = hf_forces - autograd_forces #fd_forces
            rmse = np.linalg.norm(force_diff) / N
            max_diff = np.max(np.abs(force_diff))
            print(f"displacement vector:")
            print(f"  Atom 0: {atoms.positions[1] - atoms.positions[0]}")
            print(f"  Atom 1: {atoms.positions[0] - atoms.positions[1]}")
            print(f"  Autograd forces: ")
            print(f"    Atom 0: {autograd_forces[0]}")
            print(f"    Atom 1: {autograd_forces[1]}")
            print(f"  Hellman-Feynman forces:")
            print(f"    Atom 0: {hf_forces[0]}")
            print(f"    Atom 1: {hf_forces[1]}")
            #print(f"  Finite difference forces:")
            #print(f"    Atom 0: {fd_forces[0]}")
            #print(f"    Atom 1: {fd_forces[1]}")
            print(f"  RMSE: {rmse:.2e}")
            print(f"  Max difference: {max_diff:.2e}")
            
            if rmse < 1e-6:
                print("  ✓ Hellman-Feynman forces match finite differences")
            else:
                print("  ✗ Hellman-Feynman forces do NOT match finite differences")
                
        except Exception as e:
            print(f"  ✗ Error calculating forces: {e}")

def test_dimer_force_accuracy():
    """Test accuracy of Hellman-Feynman forces with different step sizes."""
    print("\n" + "=" * 60)
    print("TESTING DIMER FORCE ACCURACY")
    print("=" * 60)
    
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'interlayer'
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="hellman_feynman")
    
    atoms = create_dimer_system(3.35)
    
    # Test different finite difference step sizes
    step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]
    
    print(f"Testing dimer at separation 3.35 Å with different step sizes")
    
    try:
        # Calculate Hellman-Feynman forces
        _, hf_forces = calc_hf.get_tb_energy(atoms)
        print(f"Hellman-Feynman forces:")
        print(f"  Atom 0: {hf_forces[0]}")
        print(f"  Atom 1: {hf_forces[1]}")
        
        for h in step_sizes:
            print(f"\nStep size: {h}")
            
            # Calculate finite difference forces
            N = len(atoms)
            fd_forces = np.zeros((N, 3))
            
            for i in range(N):
                for d in range(3):
                    displaced_plus = atoms.copy()
                    displaced_plus.positions[i, d] += h
                    E_plus, _ = calc_hf.get_tb_energy(displaced_plus)
                    
                    displaced_minus = atoms.copy()
                    displaced_minus.positions[i, d] -= h
                    E_minus, _ = calc_hf.get_tb_energy(displaced_minus)
                    
                    fd_forces[i, d] = -(E_plus - E_minus) / (2 * h)
            
            # Compare forces
            force_diff = hf_forces - fd_forces
            rmse = np.linalg.norm(force_diff) / N
            max_diff = np.max(np.abs(force_diff))
            print(f"  FD Forces: {fd_forces}")
            print(f"  RMSE: {rmse:.2e}")
            print(f"  Max difference: {max_diff:.2e}")
            
    except Exception as e:
        print(f"  ✗ Error calculating forces: {e}")

def test_dimer_energy_consistency():
    """Test that energy is consistent with forces (F = -dE/dR)."""
    print("\n" + "=" * 60)
    print("TESTING DIMER ENERGY CONSISTENCY")
    print("=" * 60)
    
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'interlayer'
    h = 1e-4
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="hellman_feynman")
    
    atoms = create_dimer_system(3.35)
    
    try:
        # Calculate energy and forces
        energy, forces = calc_hf.get_tb_energy(atoms)
        print(f"Energy: {energy:.6f} eV")
        print(f"Forces:")
        print(f"  Atom 0: {forces[0]}")
        print(f"  Atom 1: {forces[1]}")
        
        # Test energy change with small displacement
        print(f"\nTesting energy change with displacement:")
        
        # Displace atom 0 in z direction
        displaced = atoms.copy()
        displaced.positions[0, 2] += h
        energy_plus, _ = calc_hf.get_tb_energy(displaced)
        
        displaced.positions[0, 2] -= 2 * h
        energy_minus, _ = calc_hf.get_tb_energy(displaced)
        
        # Calculate force from energy difference
        force_from_energy = -(energy_plus - energy_minus) / (2 * h)
        force_from_calculation = forces[0, 2]
        
        print(f"  Force from energy difference: {force_from_energy:.6f}")
        print(f"  Force from calculation: {force_from_calculation:.6f}")
        print(f"  Difference: {abs(force_from_energy - force_from_calculation):.2e}")
        
        if abs(force_from_energy - force_from_calculation) < 1e-6:
            print("  ✓ Energy and forces are consistent")
        else:
            print("  ✗ Energy and forces are NOT consistent")
            
    except Exception as e:
        print(f"  ✗ Error calculating forces: {e}")


def test_autograd_vs_current():
    """Test simplified force calculation against current implementation."""
    print("\n" + "=" * 60)
    print("TESTING SIMPLIFIED VS CURRENT IMPLEMENTATION")
    print("=" * 60)
    
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'interlayer'
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="hellman_feynman")
    calc_hf.model_dict["intralayer"]["overlap parameters"] = None
    calc_hf.model_dict["intralayer"]["overlap form"] = None
    calc_hf.model_dict["interlayer"]["overlap parameters"] = None
    calc_hf.model_dict["interlayer"]["overlap form"] = None
    calc_autograd = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="autograd")
    
    # Test graphene system
    print("Testing graphene system:")
    atoms = graphene(a=2.46,size=(5,5,1), vacuum=10.0)
    
    # Current implementation
    energy_current, forces_current = calc_hf.get_tb_energy(atoms)
    print("getting autograd forces")
    energy_autograd, forces_autograd = calc_autograd.get_tb_energy(atoms)
    
    print(f"Current implementation:")
    print(f"  Energy: {energy_current:.6f} eV")
    print(f"  Forces: {np.round(forces_current[:5,:], decimals=6)}")
    
    print(f"Autograd implementation:")
    print(f"  Forces: {np.round(forces_autograd[:5,:], decimals=6)}")

    print(f"Current implementation force direction:")
    print(f"   {np.round(forces_current[:5,:]/np.linalg.norm(forces_current[:5,:], axis=1)[:,np.newaxis], decimals=6)}")
    
    print(f"Autograd implementation force direction:")
    print(f"   {np.round(forces_autograd[:5,:]/np.linalg.norm(forces_autograd[:5,:], axis=1)[:,np.newaxis], decimals=6)}")
    
    print(f"force ratio:")
    print(f"  Forces: {np.round(np.nan_to_num(forces_autograd[:5,:2]/forces_current[:5,:2]), decimals=6)}")
    # Compare energies
    energy_diff = abs(energy_current - energy_autograd)
    print(f"Energy difference: {energy_diff:.2e} eV")
    
    # Compare forces
    force_diff = forces_current - forces_autograd
    force_rmse = np.linalg.norm(force_diff) / np.sqrt(len(forces_current.flatten()))
    force_max_diff = np.max(np.abs(force_diff))
    
    print(f"Force RMSE: {force_rmse:.2e}")
    print(f"Force max difference: {force_max_diff:.2e}")


    
    if force_rmse < 1e-3:
        print("  ✓ Forces are in good agreement")
    else:
        print("  ⚠ Forces show significant differences")

def test_simplified_vs_current():
    """Test simplified force calculation against current implementation."""
    print("\n" + "=" * 60)
    print("TESTING SIMPLIFIED VS CURRENT IMPLEMENTATION")
    print("=" * 60)
    
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'interlayer'
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="python", 
                           force_method="hellman_feynman")
    
    # Test graphene system
    print("Testing graphene system:")
    atoms = graphene(a=2.46,size=(3,3,1), vacuum=10.0)
    
    # Current implementation
    energy_current, forces_current = calc_hf.get_tb_energy(atoms)
    
    kmesh = calc_hf.auto_select_kpoints(torch.tensor(atoms.cell.array, dtype=torch.float32, device=device))
    kpoints = k_uniform_mesh_torch(kmesh)
    energy_simple, forces_simple = simple_tetb_energy(atoms,kpoints.numpy())
    
    print(f"Current implementation:")
    print(f"  Energy: {energy_current:.6f} eV")
    print(f"  Forces: {forces_current}")
    
    print(f"Simplified implementation:")
    print(f"  Energy: {energy_simple:.6f} eV")
    print(f"  Forces: {forces_simple}")
    
    # Compare energies
    energy_diff = abs(energy_current - energy_simple)
    print(f"Energy difference: {energy_diff:.2e} eV")
    
    # Compare forces
    force_diff = forces_current - forces_simple
    force_rmse = np.linalg.norm(force_diff) / np.sqrt(len(forces_current.flatten()))
    force_max_diff = np.max(np.abs(force_diff))
    
    print(f"Force RMSE: {force_rmse:.2e}")
    print(f"Force max difference: {force_max_diff:.2e}")
    
    if force_rmse < 1e-3:
        print("  ✓ Forces are in good agreement")
    else:
        print("  ⚠ Forces show significant differences")

def test_bilayer_graphene_relaxation():
    """Test bilayer graphene relaxation. compare interlayer separation calculated 
    by hellman-feynman and from interlayer energy"""
    model_type = "TETB"
    tb_model = "popov"
    int_type = 'full'
    
    # Create calculator with Hellman-Feynman method
    calc_hf = get_BLG_Model(int_type=int_type, energy_model=model_type, 
                           tb_model=tb_model, calc_type="lammps", 
                           force_method="hellman_feynman")
    calc_hf.model_dict["intralayer"]["potential parameters"] = np.load("../PYMC_uncertainty_quanitification/best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz")["params"] #
    calc_hf.model_dict["interlayer"]["potential parameters"] =  np.load("../PYMC_uncertainty_quanitification/best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz")["params"] #
    params = calc_hf.get_params()
    calc_hf.set_params(params)

    atoms =get_bilayer_atoms(3.55,0.0,sc=11)
    atoms.calc = calc_hf
    constraint = FixedLine(indices=range(len(atoms)), direction=[0, 0, 1])
    atoms.set_constraint(constraint)
    dyn = LBFGS(atoms,trajectory="bilayer_graphene_relaxation.traj",logfile="bilayer_graphene_relaxation.log")
    #dyn = FIRE(atoms,finc=1.1,fdec=0.5,dtmax=1,trajectory="bilayer_graphene_relaxation.traj",logfile="bilayer_graphene_relaxation.log")
    dyn.run(fmax=1e-3,steps=100)
    z =  atoms.positions[:,2]
    relaxed_interlayer_separation = np.max(z)-np.min(z)
    

    d_ = np.linspace(3.3,3.8,15)
    energy = np.zeros_like(d_)
    for i,d in enumerate(d_):
        atoms = get_bilayer_atoms(d,0.0,sc=11)
        atoms.calc = calc_hf
        energy[i] = calc_hf.get_total_energy(atoms)[0]/len(atoms)
    plt.plot(d_,energy)
    plt.xlabel("Interlayer separation (Å)")
    plt.ylabel("Interlayer energy (eV)")
    plt.savefig("bilayer_graphene_AB_relaxation.png")
    plt.clf()

    print("Relaxed interlayer separation: ",relaxed_interlayer_separation)
    print("Interlayer separation from energy: ",d_[np.argmin(energy)])
    print("Difference: ",np.abs(relaxed_interlayer_separation-d_[np.argmin(energy)]))

if __name__ == "__main__":
    # Run all tests
    #test_dimer_force_conservation()
    #test_dimer_finite_differences()
    #test_dimer_force_accuracy()
    #test_dimer_energy_consistency()
    #test_simplified_vs_current()
    #test_autograd_vs_current()
    test_bilayer_graphene_relaxation()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
