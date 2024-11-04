import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import BLG_model_builder
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *

if __name__=="__main__":

    mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
    cutoff = 10
    model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":cutoff},"cutoff":cutoff},
                "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":cutoff},"cutoff":cutoff}}


    calc = TETB_model(model_dict)

    stacking_ = ["AB","SP","Mid","AA"]
    disreg_ = [0 , 0.16667, 0.5, 0.66667]
    df = pd.read_csv('../data/qmc.csv') 

    
    for i,stacking in enumerate(stacking_):
        print(stacking," Stacking")
        model_energy = []
        tb_energy = [] 
        density_sfe = []
        d_ = []
        dis = disreg_[i]
        d_stack = df.loc[df['stacking'] == stacking, :]
        for j, row in d_stack.iterrows():

            d_.append(row["d"])
            atoms = get_bilayer_atoms(row["d"],row["disregistry"])
            total_energy = (calc.get_total_energy(atoms))/len(atoms)

            model_energy.append(total_energy)
            tmp_tb_energy, _, wf = calc.get_tb_energy(atoms,return_wf = True)
            tmp_tb_energy /= len(atoms)
            tb_energy.append(tmp_tb_energy)

            nocc = len(atoms)//2
            fd_dist = 2*np.eye(len(atoms))
            fd_dist[nocc:,nocc:] = 0
            #occ_eigvals = 2*np.diag(eigvals)
            #occ_eigvals[nocc:,nocc:] = 0
            density_matrix = np.zeros((len(atoms),len(atoms)),dtype=complex)
            nkp = np.shape(wf)[-1]
            for i in range(nkp):
                density_matrix += wf[:,:,i] @ fd_dist @ np.conj(wf[:,:,i]).T / nkp

            density_sfe.append(np.diag(density_matrix.real))
        density_sfe = np.array(density_sfe)
        plt.plot(d_,density_sfe[:,0],label=stacking+" A")
        plt.plot(d_,density_sfe[:,1],label=stacking+" B")
        plt.legend()

        model_energy = np.array(model_energy)
        model_energy -= model_energy[-1] #np.min(model_energy)
        model_energy -= np.min(model_energy)
        tb_energy =np.array(tb_energy) - np.min(np.array(tb_energy))

        qmc_energy = df["energy"].to_numpy()
        qmc_energy -= np.min(qmc_energy)
    plt.xlabel("interlayer separation")
    plt.ylabel("electron density")
    plt.savefig("density_sfe.png")
