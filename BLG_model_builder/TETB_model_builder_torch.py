"""
PyTorch-optimized TETB model builder for BLG systems with Intel GPU acceleration.
This module provides JIT-compiled and vectorized tight-binding calculations.
"""

import ase.io
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import uuid
import copy
import numpy as np
import scipy.linalg
# Import PyTorch-optimized modules
from BLG_model_builder.descriptors_torch import *
from BLG_model_builder.TB_Utils_torch import *
from BLG_model_builder.Lammps_Utils import *

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

class TETB_model(Calculator):
    """
    PyTorch-optimized TETB model calculator with Intel GPU acceleration.
    """
    implemented_properties = ['energy', 'forces', 'potential_energy']
    
    def __init__(self, model_dict_input, output=None, basis="pz", kmesh=None, 
                 update_eigvals=1, use_lammps=None, force_method="hellman_feynman", **kwargs):
        """
        Initialize PyTorch-optimized TETB model with Intel GPU acceleration.
        
        Args:
            model_dict_input: Dictionary containing model parameters
            output: Output file path
            basis: Basis set type
            kmesh: K-point mesh
            update_eigvals: Frequency of eigenvalue updates
            use_lammps: Whether to use LAMMPS
            force_method: Force calculation method ("hellman_feynman" or "autograd")
                - "hellman_feynman": Use Hellman-Feynman theorem (default, faster for large systems)
                - "autograd": Use PyTorch autograd (slower but useful for verification)
        """
        Calculator.__init__(self, **kwargs)
        
        # Initialize model dictionary
        self.model_dict = {
            "interlayer": {
                "hopping form": None, "overlap form": None,
                "hopping parameters": None, "overlap parameters": None,
                "hopping kwargs": {}, "overlap kwargs": {},
                "descriptors": None, "descriptor kwargs": {},
                "use lammps": False, "potential": None,
                "potential parameters": None, "potential file writer": None
            },
            "intralayer": {
                "hopping form": None, "overlap form": None,
                "hopping parameters": None, "overlap parameters": None,
                "hopping kwargs": {}, "overlap kwargs": {},
                "descriptors": None, "descriptor kwargs": {},
                "use lammps": False, "potential": None,
                "potential parameters": None, "potential file writer": None
            }
        }
        
        self.use_lammps = use_lammps
        
        # Set force calculation method
        if force_method not in ["hellman_feynman", "autograd"]:
            raise ValueError(f"force_method must be 'hellman_feynman' or 'autograd', got '{force_method}'")
        self.force_method = force_method
        
        # Update model dictionary with input
        for m in model_dict_input:
            if self.use_lammps is None:
                if "potential" in model_dict_input[m] and type(model_dict_input[m]["potential"]) == str:
                    self.model_dict[m]["use lammps"] = True
                    self.use_lammps = True
                else:
                    self.model_dict[m]["use lammps"] = False
            
            for mz in model_dict_input[m]:
                self.model_dict[m][mz] = model_dict_input[m][mz]
        
        self.natom_types = len(self.model_dict)
        if basis == "pz":
            self.norbs_per_atom = 1
        else:
            self.norbs_per_atom = 1  # Default to 1 orbital per atom
        
        self.kmesh = kmesh
        self.nkp = 36
        self.step_index = 0
        self.update_eigvals = update_eigvals
        self.wfn = None
        self.atoms_template = None  # Will be set when first atoms object is processed
        self.use_overlap = False  # Initialize overlap flag
        
        if output is None:
            self.output = "TETB_calc_" + str(uuid.uuid4())
        else:
            self.output = output
        
        if not os.path.exists(self.output) and self.use_lammps:
            os.mkdir(self.output)
        
        if self.use_lammps:
            cwd = os.getcwd()
            os.chdir(self.output)
            self.lammps_models = []
            self.lammps_file_names = []
            for m in self.model_dict:
                file_name = m + "_residual_nkp" + str(self.nkp) + ".txt"
                self.model_dict[m]["potential file name"] = file_name
                if self.model_dict[m]["potential file writer"] is not None:
                    self.lammps_models.append(self.model_dict[m]["potential"])
                    self.lammps_file_names.append(file_name)
                    self.model_dict[m]["potential file writer"](self.model_dict[m]["potential parameters"], file_name)
            os.chdir(cwd)
    
    def set_params(self, x):
        """Set model parameters."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        start_ind = 0
        for i, m in enumerate(self.model_dict):
            if self.model_dict[m]["hopping form"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["hopping parameters"])
                self.set_param_element(m, "hopping parameters", x[start_ind:end_ind])
                start_ind = end_ind
            if self.model_dict[m]["overlap form"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["overlap parameters"])
                self.set_param_element(m, "overlap parameters", x[start_ind:end_ind])
                start_ind = end_ind
            if self.model_dict[m]["potential"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["potential parameters"])
                self.set_param_element(m, "potential parameters", x[start_ind:end_ind])
                start_ind = end_ind
    
    def set_param_element(self, outer_dict, inner_dict, params):
        """Set parameter element."""
        # Convert to PyTorch tensors and move to device
        if isinstance(params, (list, tuple)):
            params = torch.tensor(params, device=device, dtype=torch.float32)
        elif isinstance(params, np.ndarray):
            params = torch.from_numpy(params).to(device=device, dtype=torch.float32)
        
        self.model_dict[outer_dict][inner_dict] = params
        if inner_dict == "potential parameters":
            if type(self.model_dict[outer_dict]["potential"]) == str:
                self.use_lammps = True
            else:
                self.use_lammps = False
            if self.model_dict[outer_dict]["potential file writer"] is not None:
                cwd = os.getcwd()
                os.chdir(self.output)
                self.model_dict[outer_dict]["potential file writer"](
                    self.model_dict[outer_dict]["potential parameters"].cpu().numpy(),
                    self.model_dict[outer_dict]["potential file name"]
                )
                os.chdir(cwd)
    
    def get_params(self):
        """Get all parameters."""
        params = []
        for m in self.model_dict:
            if self.model_dict[m]["hopping parameters"] is not None:
                if isinstance(self.model_dict[m]["hopping parameters"], torch.Tensor):
                    params.append(self.model_dict[m]["hopping parameters"].cpu().numpy())
                else:
                    params.append(self.model_dict[m]["hopping parameters"])
            if self.model_dict[m]["overlap parameters"] is not None:
                if isinstance(self.model_dict[m]["overlap parameters"], torch.Tensor):
                    params.append(self.model_dict[m]["overlap parameters"].cpu().numpy())
                else:
                    params.append(self.model_dict[m]["overlap parameters"])
            if self.model_dict[m]["potential parameters"] is not None:
                if isinstance(self.model_dict[m]["potential parameters"], torch.Tensor):
                    params.append(self.model_dict[m]["potential parameters"].cpu().numpy())
                else:
                    params.append(self.model_dict[m]["potential parameters"])
        
        if params:
            return torch.tensor(np.concatenate(params), device=device, dtype=torch.float32)
        else:
            return torch.tensor([], device=device, dtype=torch.float32)

    def auto_select_kpoints(self,cell):
        #using empirical tests a converged (<0.01 meV/atom) k-point mesh for a 5x5 bilayer graphene supercell is (16,16,1)
        cell_length_1 = 2.46
        cell_length_2 = 2.46
        ncellsx = torch.ceil(torch.round(torch.linalg.norm(cell[0,:])/cell_length_1))
        ncellsy = torch.ceil(torch.round(torch.linalg.norm(cell[1,:])/cell_length_2))
        Kx = torch.round(16 * 5/ncellsx)
        Ky = torch.round(16 * 5/ncellsy)
        #print("auto selected kmesh = ",(Kx.item(),Ky.item(),1))
        #every 3 kpoints is exactly at dirac point which causes large changes in force
        mesh = (torch.min(torch.tensor([40,int(Kx.item())])), torch.min(torch.tensor([40,int(Ky.item())])), 1)
        if mesh[0] % 3 == 0 or mesh[1] % 3 == 0:
            mesh = (mesh[0] + 1, mesh[1] + 1, 1)
        mesh = (1,1,1)
        return mesh

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        """Calculate properties using PyTorch-optimized methods with Intel GPU acceleration."""
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        total_energy, forces = self.get_total_energy(atoms)
        self.results['forces'] = forces
        self.results['potential_energy'] = total_energy
        self.results['energy'] = total_energy
    
    def run(self, atoms):
        """Run calculation."""
        self.calculate(atoms)
    
    def get_hoppings(self, atoms):
        """
        Wrapper for get_hoppings_torch that handles ASE atoms objects.
        
        Args:
            atoms: ASE atoms object
            
        Returns:
            tuple: (hoppings, overlaps, ind_i, ind_j, di, dj)
        """
        # Convert ASE atoms to PyTorch tensors
        positions, cell, atom_types = atoms_to_torch_tensors(atoms)
        
        # Call PyTorch-optimized function
        return self.get_hoppings_torch(positions, cell, atom_types)
    
    def get_hoppings_torch(self, positions, cell, atom_types):
        """
        PyTorch-optimized hopping calculation with Intel GPU acceleration.
        """
        hoppings = torch.tensor([], device=device, dtype=torch.float32)
        grad_hoppings = torch.tensor([], device=device, dtype=torch.float32)
        overlap_elem = torch.tensor([], device=device, dtype=torch.float32)
        grad_overlaps = torch.tensor([], device=device, dtype=torch.float32)
        ind_i = torch.tensor([], device=device, dtype=torch.long)
        ind_j = torch.tensor([], device=device, dtype=torch.long)
        di = torch.tensor([], device=device, dtype=torch.long)
        dj = torch.tensor([], device=device, dtype=torch.long)
        
        for dt in self.model_dict:
            calc_hopping_form = self.model_dict[dt]["hopping form"]
            calc_model_descriptors = self.model_dict[dt]["descriptors"]
            
            if calc_hopping_form is None:
                continue
            
            descriptors, i, j, tmp_di, tmp_dj = calc_model_descriptors(
                positions, cell, atom_types, **self.model_dict[dt]["descriptor kwargs"]
            )
            
            # Ensure descriptors require gradients for autograd
            if not descriptors.requires_grad:
                descriptors = descriptors.requires_grad_(True)
            
            # Convert parameters to PyTorch tensors if they're NumPy arrays
            hopping_params = self.model_dict[dt]["hopping parameters"]
            if hasattr(hopping_params, 'numpy'):  # Already a PyTorch tensor
                hopping_params_torch = hopping_params
            else:  # NumPy array
                hopping_params_torch = torch.tensor(hopping_params, device=device, dtype=torch.float32)
            
            tmp_hops = calc_hopping_form(
                descriptors, hopping_params_torch,
                **self.model_dict[dt]["hopping kwargs"]
            )
            
            hoppings = torch.cat([hoppings, tmp_hops.flatten()])
            # Sum hoppings to create scalar for autograd
            tmp_hops_sum = torch.sum(tmp_hops)
            tmp_grad_hoppings = torch.autograd.grad(tmp_hops_sum, descriptors, create_graph=True)[0]
            grad_hoppings = torch.cat([grad_hoppings, tmp_grad_hoppings.flatten()])
            # Convert atom indices to orbital indices
            ind_i = torch.cat([ind_i, (i * self.norbs_per_atom).flatten()])
            ind_j = torch.cat([ind_j, (j * self.norbs_per_atom).flatten()])
            di = torch.cat([di, tmp_di.flatten()])
            dj = torch.cat([dj, tmp_dj.flatten()])
            
            if self.model_dict[dt]["overlap form"] is not None:
                self.use_overlap = True
            else:
                self.use_overlap = False
            
            if self.use_overlap:
                calc_overlap_form = self.model_dict[dt]["overlap form"]
                # Convert overlap parameters to PyTorch tensors if they're NumPy arrays
                overlap_params = self.model_dict[dt]["overlap parameters"]
                if hasattr(overlap_params, 'numpy'):  # Already a PyTorch tensor
                    overlap_params_torch = overlap_params
                else:  # NumPy array
                    overlap_params_torch = torch.tensor(overlap_params, device=device, dtype=torch.float32)
                
                tmp_overlaps = calc_overlap_form(
                    descriptors, overlap_params_torch,
                    **self.model_dict[dt]["overlap kwargs"]
                )
                overlap_elem = torch.cat([overlap_elem, tmp_overlaps.flatten()])
                # Sum overlaps to create scalar for autograd
                tmp_overlaps_sum = torch.sum(tmp_overlaps)
                tmp_grad_overlaps = torch.autograd.grad(tmp_overlaps_sum, descriptors, create_graph=True)[0]
                grad_overlaps = torch.cat([grad_overlaps, tmp_grad_overlaps.flatten()])
        
        if len(hoppings) == 0:
            # No hopping elements found - return empty arrays with proper shapes
            if self.use_overlap:
                return torch.tensor([], device=device, dtype=torch.float32), torch.tensor([], device=device, dtype=torch.float32), torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long)
            else:
                return torch.tensor([], device=device, dtype=torch.float32), None, torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long)
        
        if self.use_overlap:
            return hoppings/2, overlap_elem/2, ind_i, ind_j, di, dj, grad_hoppings, grad_overlaps
        else:
            return hoppings/2, None, ind_i, ind_j, di, dj, grad_hoppings, torch.zeros_like(grad_hoppings)
    
    def get_tb_energy_torch(self, positions, cell, atom_types):
        """
        PyTorch-optimized TB energy calculation with Intel GPU acceleration.
        """
        self.kmesh = self.auto_select_kpoints(cell)
        self.kpoints_reduced = k_uniform_mesh_torch(self.kmesh, device)
        self.norbs = self.norbs_per_atom * len(positions)
        tb_energy = 0.0
        # Initialize forces as COMPLEX - sum over k-points, then take real part at the end
        Forces = torch.zeros((len(positions), 3), device=device, dtype=torch.complex64)
        self_energy = -5.2887
        
        nocc = len(positions) // 2
        hoppings, overlaps, hop_i, hop_j, hop_di, hop_dj, grad_hoppings, grad_overlaps = self.get_hoppings_torch(positions, cell, atom_types)

        recip_cell = get_recip_cell_torch(cell)
        kpoints = self.kpoints_reduced @ recip_cell
        
        nkp = kpoints.shape[0]
        disp = hop_di.unsqueeze(1) * cell[0] + hop_dj.unsqueeze(1) * cell[1] + \
               positions[hop_j] - positions[hop_i]

        
        for i in range(nkp):
            ham = self_energy * torch.eye(self.norbs, device=device, dtype=torch.complex64)
            overlap = torch.eye(self.norbs, device=device, dtype=torch.complex64) if self.use_overlap else None
            
            phase = torch.exp(1.0j * torch.sum(kpoints[i, :].unsqueeze(0) * disp, dim=1))
            amp = hoppings * phase
            
            # Add hopping elements to Hamiltonian using PyTorch operations
            if self.use_overlap:
                o_amp = overlaps * phase
                ham, overlap = _build_hamiltonian_with_overlap_torch(ham, hop_i, hop_j, amp, overlap, o_amp)
            else:
                ham = _build_hamiltonian_no_overlap_torch(ham, hop_i, hop_j, amp)

            # Full diagonalization using PyTorch
            if self.use_overlap:
                eigvals, wf_k = _solve_eigenvalue_with_overlap_torch(ham, overlap)
            else:
                eigvals, wf_k = _solve_eigenvalue_no_overlap_torch(ham)
            tb_energy += 2 * torch.sum(eigvals[:nocc]) / nkp

            Forces += get_hellman_feynman_torch(
                eigvals, wf_k,
                kpoints[i, :], disp,
                grad_hoppings, grad_overlaps,
                hop_i, hop_j
            ) / nkp
            #print("force calculated for kpoint", i,"of", nkp)
            del ham,overlap,amp,phase, eigvals, wf_k
        
        # Take real part of accumulated forces after k-point summation
        Forces_real = Forces.real.to(torch.float32)
        
        self.step_index += 1
        return tb_energy.item(), Forces_real.detach().cpu().numpy()
    
    def get_tb_energy_autograd(self, positions, cell, atom_types):
        """
        PyTorch autograd-based TB energy calculation for force computation.
        Returns PyTorch tensors to maintain computational graph for autograd.
        """
        # Initialize norbs like the original method
        self.norbs = self.norbs_per_atom * len(positions)
        
        # Get kpoints exactly like the original
        kmesh = self.auto_select_kpoints(cell)
        kpoints_reduced = k_uniform_mesh_torch(kmesh)
        recip_cell = get_recip_cell_torch(cell)
        kpoints = kpoints_reduced @ recip_cell
        
        # Get hoppings exactly like the original
        hoppings, overlaps, hop_i, hop_j, hop_di, hop_dj,_,_ = self.get_hoppings_torch(positions, cell, atom_types)
        # Calculate displacement vectors exactly like the original
        disp = hop_di.unsqueeze(1) * cell[0] + hop_dj.unsqueeze(1) * cell[1] + \
               positions[hop_j] - positions[hop_i]
        
        # Use the same parameters as the original
        self_energy = -5.2887
        nkp = kpoints.shape[0]
        nocc = len(positions) // 2  # Assuming bilayer
        
        # Collect energy contributions from all kpoints
        energy_contributions = []
        for i in range(nkp):
            # Build Hamiltonian exactly like the original
            ham = self_energy * torch.eye(self.norbs, device=positions.device, dtype=torch.complex64)
            overlap = torch.eye(self.norbs, device=positions.device, dtype=torch.complex64) if self.use_overlap else None
            
            # Calculate phase exactly like the original
            phase = torch.exp(1.0j * torch.sum(kpoints[i, :].unsqueeze(0) * disp, dim=1))
            amp = hoppings * phase
            
            # Build Hamiltonian using the same helper functions as the original
            if self.use_overlap:
                o_amp = overlaps * phase
                ham, overlap = _build_hamiltonian_with_overlap_torch(ham, hop_i, hop_j, amp, overlap, o_amp)
            else:
                ham = _build_hamiltonian_no_overlap_torch(ham, hop_i, hop_j, amp)
            
            # Solve eigenvalue problem exactly like the original
            # Use the same solver as Hellman-Feynman version for consistency
            eigvals, wf_k = _solve_eigenvalue_no_overlap_torch(ham)
            
            # Store energy contribution (avoid in-place operations)
            energy_contributions.append(2 * torch.sum(eigvals[:nocc]) / nkp)
        
        # Sum all energy contributions
        tb_energy = torch.sum(torch.stack(energy_contributions))
    
        return tb_energy
    
    def get_tb_energy_autograd_wrapper(self, positions, cell, atom_types):
        """
        Wrapper for autograd-based TB energy calculation that returns forces.
        
        Args:
            positions: PyTorch tensor of atomic positions
            cell: PyTorch tensor of cell vectors
            atom_types: PyTorch tensor of atomic numbers
            
        Returns:
            tuple: (tb_energy, tb_forces) as NumPy arrays
        """
        # Ensure positions require gradients for autograd
        positions_grad = positions.clone().detach().requires_grad_(True)
        
        # Calculate energy using autograd
        tb_energy = self.get_tb_energy_autograd(positions_grad, cell, atom_types)
        
        # Calculate forces using autograd: F = -dE/dR
        autograd_forces = -torch.autograd.grad(tb_energy, positions_grad, create_graph=True)[0]
        
        # Convert to NumPy arrays for compatibility
        tb_energy_np = tb_energy.detach().cpu().numpy()
        tb_forces_np = autograd_forces.detach().cpu().numpy()
        
        return tb_energy_np, tb_forces_np
    
    def get_band_structure_torch(self, positions, cell, atom_types, kpoints):
        """
        PyTorch-optimized band structure calculation with Intel GPU acceleration.
        """
        self.norbs = self.norbs_per_atom * len(positions)
        
        hoppings, overlaps, hop_i, hop_j, hop_di, hop_dj,_,_ = self.get_hoppings_torch(positions, cell, atom_types)
        
        # Check if we have any hopping elements
        if len(hoppings) == 0:
            # No hopping elements found, return zero eigenvalues
            nkp = kpoints.shape[0]
            return torch.zeros((self.norbs, nkp), device=device, dtype=torch.float32)
        
        recip_cell = get_recip_cell_torch(cell)
        kpoint_path = kpoints @ recip_cell
        nkp = kpoint_path.shape[0]
        eigvals_k = torch.zeros((self.norbs, nkp), device=device, dtype=torch.float32)
        
        disp = hop_di.unsqueeze(1) * cell[0] + hop_dj.unsqueeze(1) * cell[1] + \
               positions[hop_j] - positions[hop_i]
        
        for i in range(nkp):
            ham = torch.zeros((self.norbs, self.norbs), device=device, dtype=torch.complex64)
            overlap = torch.eye(self.norbs, device=device, dtype=torch.complex64) if self.use_overlap else None
            
            phase = torch.exp(1.0j * torch.sum(kpoint_path[i, :].unsqueeze(0) * disp, dim=1))
            amp = hoppings * phase
            
            # Add hopping elements to Hamiltonian using PyTorch operations
            if self.use_overlap:
                o_amp = overlaps * phase
                ham, overlap = _build_hamiltonian_with_overlap_torch(ham, hop_i, hop_j, amp, overlap, o_amp)
            else:
                ham = _build_hamiltonian_no_overlap_torch(ham, hop_i, hop_j, amp)
            
            # Solve eigenvalue problem using PyTorch
            if self.use_overlap:
                eigvals, wf_k = _solve_eigenvalue_with_overlap_torch(ham, overlap)
            else:
                eigvals, wf_k = _solve_eigenvalue_no_overlap_torch(ham)
            
            eigvals_k[:, i] = eigvals
        
        return eigvals_k.detach().cpu().numpy()
    
    def get_tb_energy(self, atoms):
        """
        Wrapper for TB energy calculation that handles ASE atoms objects.
        Uses the selected force method (Hellman-Feynman or autograd).
        
        Args:
            atoms: ASE atoms object
            
        Returns:
            tuple: (tb_energy, tb_forces)
        """
        # Set atoms template if not set
        if self.atoms_template is None:
            self.atoms_template = atoms.copy()
        
        # Convert ASE atoms to PyTorch tensors
        positions, cell, atom_types = atoms_to_torch_tensors(atoms)
        
        if self.force_method == "hellman_feynman":
            # Use Hellman-Feynman method (original implementation)
            return self.get_tb_energy_torch(positions, cell, atom_types)
        elif self.force_method == "autograd":
            # Use autograd method
            return self.get_tb_energy_autograd_wrapper(positions, cell, atom_types)
        else:
            raise ValueError(f"Unknown force method: {self.force_method}")
    
    def get_band_structure(self, atoms, kpoints):
        """
        Wrapper for get_band_structure_torch that handles ASE atoms objects.
        
        Args:
            atoms: ASE atoms object
            kpoints: k-points for band structure calculation
            
        Returns:
            torch.Tensor: Band structure eigenvalues
        """
        # Convert ASE atoms to PyTorch tensors
        positions, cell, atom_types = atoms_to_torch_tensors(atoms)
        
        # Convert kpoints to PyTorch tensor if needed
        if not isinstance(kpoints, torch.Tensor):
            kpoints = torch.tensor(kpoints, device=device, dtype=torch.float32)
        
        # Call PyTorch-optimized function
        return self.get_band_structure_torch(positions, cell, atom_types, kpoints)
    
    def get_total_energy(self, atoms):
        """
        Wrapper for get_total_energy_torch that handles ASE atoms objects.
        
        Args:
            atoms: ASE atoms object
            
        Returns:
            tuple: (total_energy, forces)
        """
        if (self.model_dict["interlayer"]["hopping form"] is None and 
            self.model_dict["intralayer"]["hopping form"] is None):
            # Convert back to atoms for residual energy calculation
            residual_energy, residual_forces = self.get_residual_energy(atoms)
            return residual_energy, residual_forces
        else:
            # Convert ASE atoms to PyTorch tensors
            positions, cell, atom_types = atoms_to_torch_tensors(atoms)
            tb_energy, tb_forces = self.get_tb_energy_torch(positions, cell, atom_types)
            # Convert back to atoms for residual energy calculation
            residual_energy, residual_forces = self.get_residual_energy(atoms)
            total_energy = residual_energy + tb_energy
            forces = residual_forces + tb_forces
        return total_energy, forces
    
    
    def get_residual_energy(self, atoms):
        """Get residual energy with PyTorch compatibility."""
        residual_pe = 0
        forces = torch.zeros((len(atoms), 3), device=device, dtype=torch.float32)  # Initialize forces
        
        for m in self.model_dict:
            if self.model_dict[m]["potential"] is not None:
                if self.use_lammps:
                    cwd = os.getcwd()
                    os.chdir(self.output)
                    forces, re, residual_energy = run_lammps(atoms, self.lammps_models, self.lammps_file_names)
                    os.chdir(cwd)
                    residual_pe += re
                    break
                else:
                    ef = self.model_dict[m]["potential"](atoms, self.model_dict[m]["potential parameters"])
                    if type(ef) == tuple:
                        re = ef[0]
                        # Convert forces to PyTorch tensor if needed
                        if isinstance(ef[1], torch.Tensor):
                            forces += ef[1]
                        else:
                            forces += torch.tensor(ef[1], device=device, dtype=torch.float32)
                    else:
                        re = ef
                    residual_pe += re
        return residual_pe, forces

    def evaluate_residual_energy(self,atoms,parameters):
        start_ind = 0
        for i,m in enumerate(self.model_dict):
            if self.model_dict[m]["potential"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["potential parameters"])
                self.set_param_element(m,"potential parameters",parameters[start_ind:end_ind])
                start_ind = end_ind

        energy,_ = self.get_residual_energy(atoms)
        return energy
    
    def evaluate_total_energy(self,atoms,parameters):
        self.set_params(parameters)
        energy,_ = self.get_total_energy(atoms)
        return energy

# PyTorch-optimized helper functions
@torch.jit.script
def generalized_eigen_torch(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized generalized eigenvalue solver with Intel GPU acceleration.
    """
    Binv = torch.linalg.inv(B)
    renorm_A = Binv @ A
    eigvals, eigvecs = torch.linalg.eigh(renorm_A)
    
    # Normalize eigenvectors
    Q = eigvecs.conj().T @ B @ eigvecs
    U = torch.linalg.cholesky(torch.linalg.inv(Q))
    eigvecs = eigvecs @ U
    eigvals = torch.diag(eigvecs.conj().T @ A @ eigvecs).real
    
    return eigvals, eigvecs

@torch.jit.script
def generalized_eigvals_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized generalized eigenvalue solver with Intel GPU acceleration.
    """
    Binv = torch.linalg.inv(B)
    renorm_A = Binv @ A
    eigvals, eigvecs = torch.linalg.eigh(renorm_A)
    
    # Normalize eigenvectors
    Q = eigvecs.conj().T @ B @ eigvecs
    U = torch.linalg.cholesky(torch.linalg.inv(Q))
    eigvecs = eigvecs @ U
    eigvals = torch.diag(eigvecs.conj().T @ A @ eigvecs).real
    
    return eigvals

# PyTorch-optimized core computational functions
#@torch.jit.script
def _build_hamiltonian_no_overlap_torch(ham: torch.Tensor, hop_i: torch.Tensor, hop_j: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized function to build Hamiltonian matrix without overlap.
    """
    ham.index_put_((hop_i,hop_j), amp, accumulate=True)
    ham.index_put_((hop_j,hop_i), amp.conj(), accumulate=True)
    return ham

#@torch.jit.script
def _build_hamiltonian_with_overlap_torch(ham: torch.Tensor, hop_i: torch.Tensor, hop_j: torch.Tensor, amp: torch.Tensor, overlap: torch.Tensor, o_amp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized function to build Hamiltonian matrix with overlap.
    """
    ham.index_put_((hop_i,hop_j), amp, accumulate=True)
    ham.index_put_((hop_j,hop_i), amp.conj(), accumulate=True)
    overlap.index_put_((hop_i,hop_j), o_amp, accumulate=True)
    overlap.index_put_((hop_j,hop_i), o_amp.conj(), accumulate=True)
    return ham, overlap

@torch.jit.script
def _solve_eigenvalue_no_overlap_torch(ham: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized function to solve eigenvalue problem without overlap.
    """
    eigvals, wf_k = torch.linalg.eigh(ham)
    return eigvals, wf_k

def _solve_eigenvalue_with_overlap_torch(ham: torch.Tensor, overlap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized function to solve eigenvalue problem with overlap.
    """
    eigvals, wf_k = generalized_eigen_torch(ham, overlap)
    return eigvals, wf_k

# ASE atoms compatibility functions for PyTorch
def atoms_to_torch_tensors(atoms) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert ASE atoms object to PyTorch tensors for Intel GPU acceleration.
    
    Args:
        atoms: ASE atoms object
    
    Returns:
        positions, cell, atom_types as PyTorch tensors
    """
    import numpy as np
    
    # Convert to NumPy arrays first to avoid copy issues
    positions_np = np.array(atoms.positions)
    cell_np = np.array(atoms.get_cell())
    
    if atoms.has("mol-id"):
        atom_types_np = np.array(atoms.get_array("mol-id"))
    else:
        # Default to single layer if no mol-id
        atom_types_np = np.ones(len(atoms), dtype=np.int32)
    
    # Convert to PyTorch tensors and move to device
    positions = torch.from_numpy(positions_np).to(device=device, dtype=torch.float32)
    cell = torch.from_numpy(cell_np).to(device=device, dtype=torch.float32)
    atom_types = torch.from_numpy(atom_types_np).to(device=device, dtype=torch.long)
    
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
    atoms.set_positions(positions.cpu().numpy())
    atoms.set_cell(cell.cpu().numpy())
    atoms.set_array("mol-id", atom_types.cpu().numpy())
    return atoms


