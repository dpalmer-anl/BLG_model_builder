from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ase.io

intralayer_potential = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])

interlayer_potential = np.array([0.719345289329483, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                        3.379423382381699,3.293082477932360, 13.906782892134125,0])
interlayer_potential[1:] *=0

model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                        "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                        "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer"},
                        "potential":"rebo","potential parameters":intralayer_potential,"potential file writer":write_rebo},
                        #"potential":"reg/dep/poly 12.0 0","potential parameters":interlayer_potential,
                        #"potential file writer":write_kcinsp},

        "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                        "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                        "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer"},
                        "potential":"rebo","potential parameters":intralayer_potential,"potential file writer":write_rebo}}

calc = TETB_model(model_dict)

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
all_tb_energies = []
E0_tegt = 1e10

for i,stacking in enumerate(stacking_):
    energy_dis_tegt = []
    energy_dis_qmc = []
    energy_dis_tb = []
    d_ = []
    dis = disreg_[i]
    d_stack = df.loc[df['stacking'] == stacking, :]
    for j, row in d_stack.iterrows():
        #if row["d"] > 5.01:
        #    continue
        atoms = get_bilayer_atoms(row["d"],dis)
        pos = atoms.positions
        mean_z = np.mean(pos[:,2])
        top_ind = np.where(pos[:,2]>mean_z)

        #(norbs,nbands,kpoints)
        tb_energy, wf = calc.get_tb_energy(atoms,return_wf=True)
        nocc = len(atoms)//2
        occ_wf = wf[:,:nocc,:]
        density_k = np.sum(np.conj(occ_wf) * occ_wf,axis=1) 
        density = np.mean(density_k,axis=1).real
        n_electrons = len(atoms) #1 electron per atom
        #density *= n_electrons
        print(density)
        plt.scatter(pos[top_ind,0],pos[top_ind,1],c=density[top_ind])
        plt.xlim((0,20))
        plt.ylim((0,20))
        plt.clim((0.493,0.507))
        plt.colorbar()
        plt.savefig("figures/density_d_"+str(row["d"])+"_s_"+stacking+".png")
        plt.clf()
