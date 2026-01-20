import numpy as np
import ase.io
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
import scipy.linalg as spla
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder import NeighborList 
import uuid
import copy
from lammps import PyLammps
from lammps import lammps

try:
    import cupy
    import cupyx as cpx
    
    if cupy.cuda.is_available():
        np = cupy
        gpu_avail = True
        print("GPU detected, using GPU")
    else:
        gpu_avail = False
        print("GPU not detected, using CPU instead")
except:
    gpu_avail = False
    print("Cupy not available, using CPU instead")
    print("to use GPU, Cupy must be installed")

class TETB_model(Calculator):
    implemented_properties = ['energy','forces','potential_energy']
    def __init__(self,model_dict_input,output=None,basis="pz",kmesh=None,update_eigvals=1,use_lammps=None,**kwargs):
        """ construct a TETB model with a given hopping (and optional overlap) functional form, descriptors calculated from atomic positions,
            and parameters. i.e. H_TB = F[Descriptors(ase.atoms object),Param.]. Example:"""
        Calculator.__init__(self, **kwargs)
        self.model_dict = {"interlayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,
                                    "overlap parameters":None,
                                    "hopping kwargs":{},
                                    "overlap kwargs":{},
                                    "descriptors":None,
                                    "descriptor kwargs":{},
                                    "use lammps":False,
                                    "potential":None,
                                    "potential parameters":None,
                                    "potential file writer":None},
                      
                      "intralayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,
                                    "overlap parameters":None,
                                    "hopping kwargs":{},
                                    "overlap kwargs":{},
                                    "descriptors":None,
                                    "descriptor kwargs":{},
                                    "use lammps":False,
                                    "potential":None,"potential parameters":None,"potential file writer":None}}
        self.use_lammps = use_lammps
        
        for m in model_dict_input:
            if self.use_lammps is None:
                if "potential" in model_dict_input[m] and type(model_dict_input[m]["potential"]) == str:
                    self.model_dict[m]["use lammps"]=True
                    self.use_lammps=True
                else:
                    self.model_dict[m]["use lammps"]=False
                    #self.use_lammps=False

            for mz in model_dict_input[m]:
                self.model_dict[m][mz] = model_dict_input[m][mz]
                 
        self.natom_types = len(self.model_dict)
        if basis == "pz":
            self.norbs_per_atom = 1 

        self.kmesh = kmesh
        self.nkp = 36
        self.step_index = 0
        self.update_eigvals = update_eigvals
        self.wfn = None
        
        if output is None:
            self.output = "TETB_calc_"+str(uuid.uuid4())
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
                file_name = m+"_residual_nkp"+str(self.nkp)+".txt"
                self.model_dict[m]["potential file name"] = file_name
                if self.model_dict[m]["potential file writer"] is not None:
                    self.lammps_models.append(self.model_dict[m]["potential"])
                    self.lammps_file_names.append(file_name)
                    self.model_dict[m]["potential file writer"](self.model_dict[m]["potential parameters"],file_name)

            os.chdir(cwd)
            
    def auto_select_kpoints(self,atoms):
        #using empirical tests a converged (<0.01 meV/atom) k-point mesh for a 5x5 bilayer graphene supercell is (16,16,1)
        cell = atoms.get_cell()
        cell_length_1 = 2.46
        cell_length_2 = 2.46
        ncellsx = np.ceil(np.round(np.linalg.norm(cell[0,:])/cell_length_1))
        ncellsy = np.ceil(np.round(np.linalg.norm(cell[1,:])/cell_length_2))
        Kx = np.round(16 * 5/ncellsx)
        Ky = np.round(16 * 5/ncellsy)
        print("auto selected kmesh = ",(1,1,1))
        return (1,1,1)

    def set_params(self,x):
        """x is an array of the dimension of total parameters """
        start_ind = 0
        for i,m in enumerate(self.model_dict):
            if self.model_dict[m]["hopping form"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["hopping parameters"])
                self.set_param_element(m,"hopping parameters",x[start_ind:end_ind])
                start_ind = end_ind
            if self.model_dict[m]["overlap form"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["overlap parameters"])
                self.set_param_element(m,"overlap parameters",x[start_ind:end_ind])
                start_ind = end_ind
            if self.model_dict[m]["potential"] is not None:
                end_ind = start_ind + len(self.model_dict[m]["potential parameters"])
                self.set_param_element(m,"potential parameters",x[start_ind:end_ind])
                start_ind = end_ind

    def set_param_element(self,outer_dict,inner_dict,params):
        self.model_dict[outer_dict][inner_dict] = params
        if inner_dict == "potential parameters":
            if type(self.model_dict[outer_dict]["potential"]) == str:
                self.use_lammps=True
            else:
                self.use_lammps=False
            if self.model_dict[outer_dict]["potential file writer"] is not None:
                
                cwd = os.getcwd()
                os.chdir(self.output)
                self.model_dict[outer_dict]["potential file writer"](self.model_dict[outer_dict]["potential parameters"],self.model_dict[outer_dict]["potential file name"])
                os.chdir(cwd)

    def get_params(self):
        params = np.array([])
        for m in self.model_dict:
            if self.model_dict[m]["hopping parameters"] is not None:
                params = np.append(params,self.model_dict[m]["hopping parameters"])
            if self.model_dict[m]["overlap parameters"] is not None:
                params = np.append(params,self.model_dict[m]["overlap parameters"])
            if self.model_dict[m]["potential parameters"] is not None:
                params = np.append(params,self.model_dict[m]["potential parameters"])
        return np.array(params)
    
    def get_opt_params(self):
        return self.opt_params
    
    def set_opt_params(self,x):
        self.opt_params = x
    
    def get_opt_params_bounds(self):
        return self.opt_params_bounds
    
    def set_opt_params_bounds(self,bounds):
        self.opt_params_bounds = bounds
    
    def get_num_opt_params(self):
        return len(self.opt_params)
    
    def has_opt_params_bounds(self):
        if self.opt_params is not None:
            return True
        else:
            return False
        
    def update_model_params(self,x):
        self.set_params(x)

    def calculate(self, atoms, properties=None,system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        total_energy,forces = self.get_total_energy(atoms)
        self.results['forces'] = forces
        self.results['potential_energy'] = total_energy
        self.results['energy'] = total_energy
    
    def run(self,atoms):
        self.calculate(atoms)

    def get_hoppings(self,atoms):
        hoppings = np.array([])
        overlap_elem = np.array([])
        ind_i = np.array([],dtype=np.int64)
        ind_j = np.array([],dtype=np.int64)
        di = np.array([],dtype=np.int64)
        dj = np.array([],dtype=np.int64)

        for dt in self.model_dict:
            calc_hopping_form = self.model_dict[dt]["hopping form"]
            calc_model_descriptors = self.model_dict[dt]["descriptors"]
            if calc_hopping_form is None:
                continue

            descriptors,i,j,tmp_di,tmp_dj = calc_model_descriptors(atoms,**self.model_dict[dt]["descriptor kwargs"])
            tmp_hops = calc_hopping_form(descriptors,self.model_dict[dt]["hopping parameters"],**self.model_dict[dt]["hopping kwargs"])
            hoppings = np.append(hoppings , tmp_hops)
            ind_i = np.append(ind_i,i)
            ind_j = np.append(ind_j,j)
            di = np.append(di,tmp_di)
            dj = np.append(dj,tmp_dj)

            if self.model_dict[dt]["overlap form"] is not None: self.use_overlap=True 
            else: self.use_overlap=False
            
            if self.use_overlap:
                calc_overlap_form = self.model_dict[dt]["overlap form"]
                overlap_elem = np.append(overlap_elem , calc_overlap_form(descriptors,self.model_dict[dt]["overlap parameters"],**self.model_dict[dt]["overlap kwargs"]))

        if self.use_overlap:
            return hoppings/2,overlap_elem/2,ind_i,ind_j,di,dj
        else:
            return hoppings/2,None,ind_i,ind_j,di,dj
    
    def SK_hopping_transform(self,atoms):
        #need to check and see if this gives the correct answer
        self_energy = -5.2887
        ham = self_energy * np.eye(self.norbs)
        hoppings,overlaps,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        if self.use_overlap:
            overlap = np.eye(self.norbs)

        amp = hoppings
        np.add.at(ham,(hop_i,hop_j),amp)
        np.add.at(ham,(hop_j,hop_i),np.conj(amp))
        
        if self.use_overlap:
            o_amp = overlaps
            np.add.at(overlap,(hop_i,hop_j),o_amp)
            np.add.at(overlap,(hop_j,hop_i),np.conj(o_amp))
        Binv = np.linalg.inv(overlap)
        renorm_A  = Binv @ ham
        eigvals,eigvecs = np.linalg.eigh(renorm_A)
        #normalize eigenvectors s.t. eigvecs.conj().T @ B @ eigvecs = I
        Q = eigvecs.conj().T @ B @ eigvecs
        U = np.linalg.cholesky(np.linalg.inv(Q))
        rotated_ham = ham @ U
        return rotated_ham[hop_i,hop_j]

    def get_tb_energy(self,atoms):
        
        self.norbs = self.norbs_per_atom * len(atoms)
        self.tb_energy = 0
        self.Forces = np.zeros((len(atoms),3))
        self_energy = -5.2887
        positions = atoms.positions
        cell = atoms.get_cell()
        mol_id = atoms.get_array("mol-id")
        if gpu_avail:
            positions = np.asarray(positions)
            cell = np.asarray(cell)
            mol_id = np.asarray(mol_id)
        tb_energy = 0
        nocc = len(atoms)//2
        hoppings,overlaps,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        
        if self.kmesh is None:
            self.kmesh = self.auto_select_kpoints(atoms)
        self.kpoints_reduced = k_uniform_mesh(self.kmesh)
        recip_cell = get_recip_cell(cell)
        self.kpoints = self.kpoints_reduced @ recip_cell.T
        self.nkp = np.shape(self.kpoints)[0]

        fd_dist = 2*np.eye(len(atoms))
        fd_dist[nocc:,nocc:] = 0
        disp = hop_di[:, np.newaxis] * cell[0] +hop_dj[:, np.newaxis] * cell[1] +\
                        positions[hop_j] - positions[hop_i]

        for i in range(self.nkp):
            ham = self_energy * np.eye(self.norbs,dtype=np.complex64)
            if self.use_overlap:
                overlap = np.eye(self.norbs,dtype=np.complex64)

            phase = np.exp((1.0j)*np.dot(self.kpoints[i,:],disp.T))
            amp = hoppings * phase
            if gpu_avail:
                ham[hop_i,hop_j] += amp
                ham[hop_j,hop_i] += np.conj(amp)
                if self.use_overlap:
                    o_amp = overlaps * phase
                    overlap[hop_i,hop_j] +=   o_amp
                    overlap[hop_j,hop_i] +=  np.conj(o_amp)
            else:
                np.add.at(ham,(hop_i,hop_j),amp)
                np.add.at(ham,(hop_j,hop_i),np.conj(amp))
            
                if self.use_overlap:
                    o_amp = overlaps * phase
                    np.add.at(overlap,(hop_i,hop_j),o_amp)
                    np.add.at(overlap,(hop_j,hop_i),np.conj(o_amp))

            if self.use_overlap:
                if gpu_avail:
                    self.eigvals, wf_k = generalized_eigen(ham,overlap)
                else:
                    self.eigvals,wf_k = spla.eigh(ham,b=overlap)
            else:
                self.eigvals,wf_k = np.linalg.eigh(ham)
            
            efermi = (self.eigvals[self.norbs//2]+self.eigvals[(self.norbs-1)//2])/2
            self.tb_energy += 2 * np.sum(self.eigvals[:nocc])/self.nkp
            del ham
            del amp
            if self.use_overlap:
                del overlap
                del o_amp
            #self.Forces += get_hellman_feynman(atoms, disp,hop_i,hop_j,wf_k,self.kpoints[i,:],
            #    self.model_dict["interlayer"]["hopping form"], self.model_dict["interlayer"]["hopping parameters"],overlap=self.model_dict[""])/self.nkp 
            self.Forces += get_hellman_feynman(positions, mol_id, cell, self.eigvals,wf_k, {"interlayer":"popov","intralayer":"porezag"},self.kpoints[i,:],
                                                self.model_dict["intralayer"]["hopping parameters"],self.model_dict["intralayer"]["overlap parameters"],
                                                self.model_dict["interlayer"]["hopping parameters"],self.model_dict["interlayer"]["overlap parameters"])/self.nkp

        self.step_index +=1
        if gpu_avail:
            return np.asnumpy(self.tb_energy), np.asnumpy(self.Forces)
        else:
            return self.tb_energy, self.Forces

    def get_tb_energy_fd(self,atoms):
        self.norbs = self.norbs_per_atom * len(atoms)
        self.tb_energy = 0
        self.Forces = np.zeros((len(atoms),3))
        self_energy = -5.2887
        dr = 1e-3
        positions = atoms.positions
        cell = atoms.get_cell()
        tb_energy = 0
        nocc = len(atoms)//2
        hoppings,overlaps,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        recip_cell = get_recip_cell(cell)
        self.kpoints = self.kpoints_reduced @ recip_cell.T

        self.nkp = np.shape(self.kpoints)[0]

        self.wf = np.zeros((self.norbs,self.norbs,self.nkp),dtype=complex)
        disp = hop_di[:, np.newaxis] * cell[0] +hop_dj[:, np.newaxis] * cell[1] +\
                        positions[hop_j] - positions[hop_i]

        for i in range(self.nkp):
            ham = self_energy * np.eye(self.norbs,dtype=np.complex64)
            if self.use_overlap:
                overlap = np.eye(self.norbs,dtype=np.complex64)

            phase = np.exp((1.0j)*np.dot(self.kpoints[i,:],disp.T))
            amp = hoppings * phase
            np.add.at(ham,(hop_i,hop_j),amp)
            np.add.at(ham,(hop_j,hop_i),np.conj(amp))
            
            if self.use_overlap:
                o_amp = overlaps * phase
                np.add.at(overlap,(hop_i,hop_j),o_amp)
                np.add.at(overlap,(hop_j,hop_i),np.conj(o_amp))

            if self.step_index % self.update_eigvals ==0:
                #use perturbation theory to speed up relaxations/ md so you don't have to run as many diagonalizations
                if self.use_overlap:
                    if gpu_avail:
                        self.eigvals, self.wf_k = generalized_eigen(ham,overlap)
                    else:
                        self.eigvals,self.wf_k = spla.eigh(ham,b=overlap)
                else:
                    self.eigvals,self.wf_k = np.linalg.eigh(ham)
            else:
                self.eigvals = np.diag(self.wf_k.T @ ham @ self.wf_k)
            self.tb_energy += 2 * np.sum(self.eigvals[:nocc])/self.nkp
            self.wf[:,:,i] = self.wf_k
            
            for atom_ind in range(len(atoms)):
                for dir_ind in range(3):
                    atom_positions_pert = np.copy(positions)
                    atom_positions_pert[atom_ind, dir_ind] += dr
                    atoms_pert = copy.deepcopy(atoms)
                    atoms_pert.positions = atom_positions_pert
                    Energy_up,_ = self.get_tb_energy(atoms_pert)
                    
                    atom_positions_pert = np.copy(positions)
                    atom_positions_pert[atom_ind, dir_ind] -= dr
                    atoms_pert = copy.deepcopy(atoms)
                    atoms_pert.positions = atom_positions_pert
                    Energy_dwn,_ = self.get_tb_energy(atoms_pert)

                    self.Forces[atom_ind, dir_ind] += -(Energy_up - Energy_dwn) / (2 * dr)

            return self.tb_energy, self.Forces

    def get_residual_energy(self,atoms):

        residual_pe = 0
        for m in self.model_dict:
            if self.model_dict[m]["potential"] is not None:
                if self.use_lammps:
                    cwd = os.getcwd()
                    os.chdir(self.output)
                    forces,re,residual_energy = run_lammps(atoms,self.lammps_models,self.lammps_file_names)
                    os.chdir(cwd)
                    residual_pe += re
                    break
                else:
                    ef = self.model_dict[m]["potential"](atoms,self.model_dict[m]["potential parameters"])
                    if type(ef)==tuple:
                        re = ef[0]
                        forces = ef[1]
                    else:
                        re = ef
                        forces = np.zeros((len(atoms),3))
                    residual_pe += re

        return residual_pe, forces

    def relax_structure(self,atoms):
        if self.use_lammps:
            cwd = os.getcwd()
            os.chdir(self.output)
            relax_atoms,forces = pylammps_relax(atoms,self.lammps_models,self.lammps_file_names)
            os.chdir(cwd)
        else:
            print("**Warning** Cannot relax with TETB models yet")
            relax_atoms=None
            forces=None
        return relax_atoms,forces
    
    def get_band_structure(self,atoms,kpoints):
        #self.set_neighbor_list(atoms)
        self.norbs = self.norbs_per_atom * len(atoms)
        positions = atoms.positions
        cell = atoms.get_cell()
        if gpu_avail:
            positions = np.asarray(positions)
            cell = np.asarray(cell)
        hoppings,overlaps,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        recip_cell = get_recip_cell(cell)
        self.kpoint_path = kpoints @ recip_cell.T
        self.nkp = np.shape(self.kpoint_path)[0]
        eigvals_k = np.zeros((self.norbs,self.nkp))
        disp = hop_di[:, np.newaxis] * cell[0] +hop_dj[:, np.newaxis] * cell[1] +\
                        positions[hop_j] - positions[hop_i]

        for i in range(self.nkp):
            ham = np.zeros((self.norbs,self.norbs),dtype=np.complex64)
            if self.use_overlap:
                overlap = np.eye(self.norbs,dtype=np.complex64)
            
            phase = np.exp((1.0j)*np.dot(self.kpoint_path[i,:],disp.T))
            amp = hoppings * phase

            if gpu_avail:
                #cpx.scatter_add(ham,(hop_i,hop_j),amp)
                #cpx.scatter_add(ham,(hop_j,hop_i),np.conj(amp))
                
                #if self.use_overlap:
                #    o_amp = overlaps * phase
                #    cpx.scatter_add(overlap,(hop_i,hop_j),o_amp)
                #    cpx.scatter_add(overlap,(hop_j,hop_i),np.conj(o_amp))
                ham[hop_i,hop_j] += amp
                ham[hop_j,hop_i] += np.conj(amp)
                if self.use_overlap:
                    o_amp = overlaps * phase
                    overlap[hop_i,hop_j] +=   o_amp
                    overlap[hop_j,hop_i] +=  np.conj(o_amp)

            else:
                np.add.at(ham,(hop_i,hop_j),amp)
                np.add.at(ham,(hop_j,hop_i),np.conj(amp))
                
                if self.use_overlap:
                    o_amp = overlaps * phase
                    np.add.at(overlap,(hop_i,hop_j),o_amp)
                    np.add.at(overlap,(hop_j,hop_i),np.conj(o_amp))

            if self.use_overlap:
                if gpu_avail:
                    eigvals, wf_k = generalized_eigen(ham,overlap)
                else:
                    eigvals,wf_k = spla.eigh(ham,b=overlap)
            else:
                eigvals,wf_k = np.linalg.eigh(ham)
            eigvals_k[:,i] = eigvals
        
        if gpu_avail:
            return np.asnumpy(eigvals_k)
        else:
            return eigvals_k

    def get_density_matrix(self,atoms):
        tb_energy,tb_forces = self.get_tb_energy(atoms)
        return self.density_matrix

    def get_hamiltonian(self,atoms):
        self.norbs = self.norbs_per_atom * len(atoms)
        positions = atoms.positions
        cell = atoms.get_cell()
        hoppings,overlaps,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        recip_cell = get_recip_cell(cell)
        self.kpoints = self.kpoints_reduced @ recip_cell.T

        self.nkp = np.shape(self.kpoints)[0]
        disp = hop_di[:, np.newaxis] * cell[0] +hop_dj[:, np.newaxis] * cell[1] +\
                        positions[hop_j] - positions[hop_i]
        ham_k = np.zeros((self.norbs,self.norbs,self.nkp),dtype=complex)

        for i in range(self.nkp):
            phase = np.exp((1.0j)*np.dot(self.kpoints[i,:],disp.T))
            amp = hoppings * phase
            np.add.at(ham_k,(hop_i,hop_j,i),amp)
            np.add.at(ham_k,(hop_j,hop_i,i),np.conj(amp))
        return ham_k
    
    def get_total_energy(self,atoms):
        
        #self.set_neighbor_list(atoms)
        if not atoms.has("mol-id"):
            pos = atoms.positions
            mean_z = np.mean(pos[:,2])
            top_ind = np.where(pos[:,2]>mean_z)
            mol_id = np.ones(len(atoms),dtype=np.int64)
            mol_id[top_ind] = 2
            atoms.set_array("mol-id",mol_id)

        if self.model_dict["interlayer"]["hopping form"] is None and self.model_dict["intralayer"]["hopping form"] is None:
            residual_energy, residual_forces = self.get_residual_energy(atoms)
            return residual_energy, residual_forces
        else:
            tb_energy,tb_forces  = self.get_tb_energy(atoms)
            residual_energy, residual_forces = self.get_residual_energy(atoms)
            total_energy = residual_energy + tb_energy
            forces = residual_forces + tb_forces
            return  total_energy , forces
    #def get_potential_energy(self,atoms):
    #    return self.get_total_energy(atoms)[0]
        
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
    
    def set_neighbor_list(self,atoms):
        atoms.neighbor_list = NeighborList.NN_list(atoms)

def generalized_eigen(A,B):
    """generalized eigen value solver using cupy. equivalent to scipy.linalg.eigh(A,B=B) """
    Binv = np.linalg.inv(B)
    renorm_A  = Binv @ A
    eigvals,eigvecs = np.linalg.eigh(renorm_A)
    #normalize eigenvectors s.t. eigvecs.conj().T @ B @ eigvecs = I
    Q = eigvecs.conj().T @ B @ eigvecs
    U = np.linalg.cholesky(np.linalg.inv(Q))
    eigvecs = eigvecs @ U 
    eigvals = np.diag(eigvecs.conj().T @ A @ eigvecs).real

    return eigvals,eigvecs


if __name__=="__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    from pythtb import *
    from bilayer_letb.api import tb_model
    from BLG_model_builder.BLG_potentials import *
    from BLG_model_builder.geom_tools import *
    from BLG_model_builder.Lammps_Utils import *

    sep = 3.35
    a = 2.46
    n=5
    theta=3.89
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                    mass=[12.01,12.01],sep=sep,h_vac=20)
    
    
    interlayer_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406,
                                -103.18388323245665, 1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
    intralayer_params = np.array([0.14687637217609084, 4.683462616941604, 12433.64356176609,\
            12466.479169306709, 19.121905577450008, 30.504342033258325,\
            4.636516235627607 , 1.3641304165817836, 1.3878198074813923])
    
    eV_per_hartree = 27.2114
    interlayer_hopping_fxn = SK_pz_chebyshev
    interlayer_overlap_fxn = SK_pz_chebyshev
    intralayer_hopping_fxn = SK_pz_chebyshev
    intralayer_overlap_fxn = SK_pz_chebyshev
    hopping_model_name ="popov"

    popov_hopping_pp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863]) *eV_per_hartree #np.load("../BLG_model_builder/parameters/popov_hoppings_pp_sigma.npz")["parameters"]
    popov_hopping_pp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree #np.load("../BLG_model_builder/parameters/popov_hoppings_pp_pi.npz")["parameters"]
    popov_overlap_pp_sigma = np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997,0.0921727, -0.0268106, 0.0002240, 0.0040319,-0.0022450, 0.0005596])  #np.load("../BLG_model_builder/parameters/popov_overlap_pp_sigma.npz")["parameters"]
    popov_overlap_pp_pi = np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,0.0156376, 0.0025976, -0.0039498, 0.0020581,-0.0007114, 0.0001427])  #np.load("../BLG_model_builder/parameters/popov_overlap_pp_pi.npz")["parameters"]
    interlayer_hopping_params = np.append(popov_hopping_pp_sigma,popov_hopping_pp_pi)
    interlayer_overlap_params = np.append(popov_overlap_pp_sigma,popov_overlap_pp_pi)

    porezag_hopping_pp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree #np.load("../BLG_model_builder/parameters/porezag_hoppings_pp_sigma.npz")["parameters"]
    porezag_hopping_pp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree #np.load("../BLG_model_builder/parameters/porezag_hoppings_pp_pi.npz")["parameters"]
    porezag_overlap_pp_sigma = np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751]) #np.load("../BLG_model_builder/parameters/porezag_overlap_pp_sigma.npz")["parameters"]
    porezag_overlap_pp_pi = np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227])  #np.load("../BLG_model_builder/parameters/porezag_overlap_pp_pi.npz")["parameters"]
    intralayer_hopping_params = np.append(porezag_hopping_pp_sigma,porezag_hopping_pp_pi)
    intralayer_overlap_params = np.append(porezag_overlap_pp_sigma,porezag_overlap_pp_pi)
    #similar format to lammps pair_coeff
    popov_model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177},"cutoff":5.29177,
                                "potential":Kolmogorov_Crespi_insp,"potential parameters":interlayer_params},
    
            "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239},"cutoff":3.704239,
                                "potential":None,"potential parameters":None,"potential file writer":None}}
    
    mk_model_dict = model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29},"cutoff":5.29,
                                "potential":None,"potential parameters":None},
    
            "intralayer":{"hopping form":None,"overlap form":None,
                                "hopping parameters":None,"overlap parameters":None,
                                "descriptors":None,"descriptor kwargs":None,"cutoff":None,
                                "potential":None,"potential parameters":None}}

    
    """model_dict = {"letb intralayer":{"hopping form":letb_intralayer,"overlap form":None,
                                    "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                    "descriptors":letb_intralayer_descriptors},
                "letb interlayer":{"hopping form":letb_interlayer,"overlap form":None,
                                    "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                    "descriptors":letb_interlayer_descriptors}}
    
    model_dict = {"porezag intralayer":{"hopping form":porezag_hopping,"overlap form":porezag_overlap,
                                    "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                    "descriptors":"displacement"},
                "popov interlayer":{"hopping form":popov_hopping,"overlap form":popov_overlap,
                                    "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                    "descriptors":"displacement"}}"""
    
    calc = TETB_model(mk_model_dict)
    #energy = calc.get_total_energy(atoms)

    Gamma = [0,   0,   0]
    K = [1/3,2/3,0]
    Kprime = [2/3,1/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=60
    (kvec,k_dist, k_node) = k_path(sym_pts,nk)

    erange = 0.8
    evals = calc.get_band_structure(atoms,kvec)
    fig, ax = plt.subplots()
    label=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
    # specify horizontal axis details
    # set range of horizontal axis
    ax.set_xlim(k_node[0],k_node[-1])
    # put tickmarks and labels at node positions
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    # add vertical lines at node positions
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    # put title
    ax.set_title("21.78 twist graphene")
    ax.set_xlabel("Path in k-space")
    
    nbands = np.shape(evals)[0]
    efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
    fermi_ind = (nbands)//2

    for n in range(np.shape(evals)[0]):
        ax.plot(k_dist,evals[n,:]-efermi,color="black")
        
    # make an PDF figure of a plot
    #fig.tight_layout()
    ax.set_ylim(-erange,erange)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_xticks(k_node)
    ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
    ax.set_xlim(k_node[0], k_node[-1])
    fig.savefig("theta_21_78_graphene.png", bbox_inches='tight')
    plt.clf()

        
