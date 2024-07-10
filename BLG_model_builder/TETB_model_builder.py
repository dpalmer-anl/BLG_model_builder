import numpy as np
import ase.io
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
import TETB_GRAPHENE
import scipy.linalg as spla
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from TB_Utils import *
from Lammps_Utils import *
from descriptors import *

class TETB_model(Calculator):
    def __init__(self,model_dict,output="TETB_calc",basis="pz",kmesh=(11,11,1)):
        """ construct a TETB model with a given hopping (and optional overlap) functional form, descriptors calculated from atomic positions,
            and parameters. i.e. H_TB = F[Descriptors(R),Param.]. Example:
        
        model_dict = {"interlayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,
                                    "overlap parameters":None,
                                    "descriptors":"displacement",
                                    "potential":"reg/dep/poly 10.0 0",
                                    "potential parameters":None,
                                    "potential file writer":write_kcinsp},
                      
                      "intralayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,
                                    "overlap parameters":None,
                                    "descriptors":"displacement"}}"""
        self.model_dict = model_dict
        self.natom_types = len(model_dict)
        if basis == "pz":
            self.norbs_per_atom = 1 
        self.kpoints_reduced = k_uniform_mesh(kmesh)
        self.nkp = np.shape(self.kpoints_reduced)[0]
        
        self.output = output
        
        if not os.path.exists(self.output):
            os.mkdir(self.output)

        cwd = os.getcwd()
        os.chdir(self.output)
        self.lammps_models = []
        self.lammps_file_names = []
        for m in model_dict:
            file_name = m+"_residual_nkp"+str(self.nkp)+".txt"
            self.lammps_models.append(self.model_dict[m]["potential"])
            self.lammps_file_names.append(file_name)
            self.model_dict[m]["potential file writer"](self.model_dict[m]["potential parameters"],file_name)
        os.chdir(cwd)

    
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

            descriptors,i,j,tmp_di,tmp_dj = calc_model_descriptors(atoms,**self.model_dict[dt]["descriptor kwargs"])
            tmp_hops = calc_hopping_form(descriptors,self.model_dict[dt]["hopping parameters"])
            hoppings = np.append(hoppings , tmp_hops)
            ind_i = np.append(ind_i,i)
            ind_j = np.append(ind_j,j)
            di = np.append(di,tmp_di)
            dj = np.append(dj,tmp_dj)

            if model_dict[dt]["overlap form"] is not None: self.use_overlap=True 
            else: self.use_overlap=False
            
            if self.use_overlap:
                calc_overlap_form = self.model_dict[dt]["overlap form"]
                overlap_elem = np.append(overlap_elem , calc_overlap_form(descriptors,self.model_dict[dt]["overlap parameters"]))

        if self.use_overlap:
            return hoppings,overlap_elem,ind_i,ind_j,di,dj
        else:
            return hoppings/2,ind_i,ind_j,di,dj
    
    def get_tb_energy(self,atoms):
        self.norbs = self.norbs_per_atom * len(atoms)
        positions = atoms.positions
        cell = atoms.get_cell()
        tb_energy = 0
        nocc = len(atoms)//2
        hoppings,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        recip_cell = get_recip_cell(atoms.get_cell())
        self.kpoints = self.kpoints_reduced @ recip_cell.T

        self.nkp = np.shape(self.kpoints)[0]
        for i in range(self.nkp):
            ham = np.zeros((self.norbs,self.norbs),dtype=np.complex64)

            for n,h in enumerate(hoppings):
                disp = hop_di[n] * cell[0] + hop_dj[n] * cell[1] + positions[hop_j[n],:] - positions[hop_i[n],:]
                phase = np.exp((1.0j)*np.dot(self.kpoints[i,:],disp))
                ham[hop_i[n],hop_j[n]] += h * phase
                ham[hop_j[n],hop_i[n]] += h * np.conj(phase) 

            #amp = hoppings * phases
            #ham[hop_i,hop_j] += amp
            #ham[hop_j,hop_i] += np.conj(amp)

            eigvals,_ = np.linalg.eigh(ham)
            tb_energy += 2 * np.sum(eigvals[:nocc])
        return tb_energy/self.nkp

    def get_residual_energy(self,atoms):
        cwd = os.getcwd()
        os.chdir(self.output)
        forces,residual_pe,residual_energy = run_lammps(atoms,self.lammps_models,self.lammps_file_names)
        os.chdir(cwd)
        return residual_pe
    
    def get_band_structure(self,atoms,kpoints):
        self.norbs = self.norbs_per_atom * len(atoms)
        positions = atoms.positions
        cell = atoms.get_cell()
        hoppings,hop_i,hop_j,hop_di,hop_dj =  self.get_hoppings(atoms)
        recip_cell = get_recip_cell(atoms.get_cell())
        self.kpoint_path = kpoints @ recip_cell.T
        self.nkp = np.shape(self.kpoint_path)[0]
        eigvals_k = np.zeros((self.norbs,self.nkp))

        for i in range(self.nkp):
            ham = np.zeros((self.norbs,self.norbs),dtype=np.complex64)

            for n,h in enumerate(hoppings):
                disp = hop_di[n] * cell[0] + hop_dj[n] * cell[1] + positions[hop_j[n],:] - positions[hop_i[n],:]
                phase = np.exp((1.0j)*np.dot(self.kpoint_path[i,:],disp))
                ham[hop_i[n],hop_j[n]] += h * phase
                ham[hop_j[n],hop_i[n]] += h * np.conj(phase)  

            #amp = hoppings * phases
            #ham[hop_i,hop_j] += amp
            #ham[hop_j,hop_i] += np.conj(amp)

            eigvals,_ = np.linalg.eigh(ham)
            eigvals_k[:,i] = eigvals
        return eigvals_k
    
    def get_total_energy(self,atoms):
        if not atoms.has("mol-id"):
            pos = atoms.positions
            mean_z = np.mean(pos[:,2])
            top_ind = np.where(pos[:,2]>mean_z)
            mol_id = np.ones(len(atoms),dtype=np.int64)
            mol_id[top_ind] = 2
            atoms.set_array("mol-id",mol_id)

        return  self.get_residual_energy(atoms) + self.get_tb_energy(atoms) 
    

if __name__=="__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    from pythtb import *
    from bilayer_letb.api import tb_model

    sep = 3.35
    a = 2.46
    n=5
    theta=21.78
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                    mass=[12.01,12.01],sep=sep,h_vac=20)
    intralayer_potential = np.load("parameters/mnml_hopping_rebo_residual_params.npz")["params"]
    interlayer_potential = np.load("parameters/mnml_hopping_kcinsp_residual_params.npz")["params"]
    #similar format to lammps pair_coeff
    model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer"},
                                "potential":"reg/dep/poly 10.0 0","potential parameters":interlayer_potential,
                                "potential file writer":write_kcinsp},

                "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer"},
                                "potential":"airebo 3","potential parameters":intralayer_potential,"potential file writer":write_rebo}}

    
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
    
    calc = TETB_model(model_dict)
    energy = calc.get_total_energy(atoms)

    Gamma = [0,   0,   0]
    K = [1/3,2/3,0]
    Kprime = [2/3,1/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=60
    (kvec,k_dist, k_node) = k_path(sym_pts,nk)

    erange = 5
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

    
