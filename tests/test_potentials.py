import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
import ase.db
import flatgraphene as fg
from scipy.spatial import distance
from ase.build import make_supercell
import pandas as pd
import torch
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

from ase import Atoms
from ase.build import graphene
from scipy.optimize import curve_fit

def get_twist_geom(t,sep,a=2.46):
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=20)
    return atoms

if __name__=="__main__":
    """ run mcmc
    $ export MPIEXEC_OPTIONS="--bind-to core --map-by slot:PE=<num_openmp_processes> port-bindings"
    $ mpiexec -np <num_mpi_workers> ${MPIEXEC_OPTIONS} python script.py
    """
    csfont = {'fontname':'serif',"size":18}

    test_intralayer_lat_con=False
    test_intralayer=False
    test_interlayer=False
    convergence_test = False
    interlayer_strain=False
    interlayer_tb_energy = False


    if interlayer_tb_energy:
        calc = get_BLG_Model(int_type="interlayer",energy_model="TETB",tb_model="popov",calc_type="python")
        layer_sep = np.linspace(2,7,20)
        stacking_ = [0,0.667]
        tb_energy_stack = []
        for s in stacking_:
            tb_energy = []
            for i,ls in enumerate(layer_sep):
                atoms = get_bilayer_atoms(ls,s,sc=1)
                pos = atoms.positions
                mean_z = np.mean(pos[:,2])
                top_ind = np.where(pos[:,2]>mean_z)
                bot_ind = np.where(pos[:,2]<mean_z)
                d = np.mean(np.abs(pos[top_ind,2]-pos[bot_ind,2]))
                mol_id = np.ones(len(atoms),dtype=np.int64)
                mol_id[top_ind] = 2
                atoms.set_array("mol-id",mol_id)

                top_layer_ind = np.where(pos[:,2]>mean_z)
                top_pos = np.squeeze(pos[top_layer_ind,:])
                bot_layer_ind = np.where(pos[:,2]<mean_z)
                bot_pos = np.squeeze(pos[bot_layer_ind,:])

                e,f = calc.get_tb_energy(atoms)
                tb_energy.append(e/len(atoms))
            tb_energy_stack.append(tb_energy)
        plt.plot(layer_sep,np.array(tb_energy_stack[0])-tb_energy_stack[0][-1],label="d = "+str(stacking_[0]))
        paper_ls = np.array([3.322368421052632, 3.3421052631578947, 3.3684210526315788, 3.394736842105263, 
                            3.427631578947368, 3.4671052631578947, 3.5657894736842106, 3.6644736842105265, 
                            3.796052631578948, 4.006578947368421, 4.197368421052632, 4.440789473684211, 4.690789473684211, 
                            4.9868421052631575, 5.315789473684211, 5.723684210526315, 6.217105263157895, 6.651315789473684, 
                            6.967105263157895])

        paper_energy = np.array([0.0951917693169093,0.08774729376633073,0.08207664551449546,0.07463294761727013,
                                0.06648080751524199,0.05797483513748911,0.04238366305835509,0.029983980340923228,
                                0.020070455393803643,0.009102432499688923,0.004160445439840732,0.002061559039442565,
                                0.0006726701505536808,0.00035305462237152085,0.00039193729003357547,0.0007947617270125523,
                                0.0008530857285056342,0.0005498009207415416,0.0005871282816971302])
        plt.plot(paper_ls,paper_energy,label="paper data")
        plt.plot(layer_sep,np.array(tb_energy_stack[1])-tb_energy_stack[0][-1],label="d = "+str(stacking_[1]))
        plt.xlabel("layer sep (ang.)")
        plt.ylabel("tb energy (eV/atom)")
        plt.ylim((-0.1,0.1))
        plt.legend()
        plt.savefig("figures/tb_energy_popov.png")
        plt.clf()

        """plt.plot(layer_sep, np.array([-8.29293243, -8.3542576,  -8.36050491, -8.36082055, -8.36082055, -8.36082055,
 -8.36082055, -8.36082055, -8.36082055, -8.36082055]),label="d = "+str(0))
        plt.plot(layer_sep, np.array([-8.26851619, -8.35887626, -8.36655478, -8.36697586, -8.36697586, -8.36697586,
 -8.36697586, -8.36697586, -8.36697586, -8.36697586]),label="d = "+str(0.667))
        plt.xlabel("layer sep (ang.)")
        plt.ylabel("tb energy (eV/atom)")
        plt.legend()
        plt.savefig("figures/tb_energy_popov.png")
        plt.clf()"""

    if convergence_test:

        calc = get_BLG_Model(int_type="interlayer",energy_model="Classical",tb_model=None,calc_type="python")
        lammps_calc = get_BLG_Model(int_type="interlayer",energy_model="Classical",tb_model=None,calc_type="lammps")
        
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
        E0_tegt = 0

        ncells = np.arange(1,7,1)
        total_energy = np.zeros(len(ncells))
        tb_energy = np.zeros(len(ncells))
        residual_energy = np.zeros(len(ncells))
        lammps_energy = np.zeros(len(ncells))

        cutoff = 10
        for i,n in enumerate(ncells):
            print("ncells = ",n)
            atoms = get_bilayer_atoms(3.5,0,sc=n)
            tmp_tb_energy,_ = calc.get_tb_energy(atoms)
            tb_energy[i] = tmp_tb_energy/len(atoms)
            tmp_residual_energy,_ = calc.get_residual_energy(atoms)
            residual_energy[i] = tmp_residual_energy/len(atoms)
            tmp_energy,tmp_forces = calc.get_total_energy(atoms)
            total_energy[i] = tmp_energy/len(atoms)

            le,lf = lammps_calc.get_total_energy(atoms)
            lammps_energy[i] = le/len(atoms)
        plt.plot(ncells,total_energy-total_energy[-1],label="total python")
        plt.plot(ncells,lammps_energy-lammps_energy[-1],label="total lammps")
        #plt.plot(ncells,tb_energy-tb_energy[-1],label="tb")
        #plt.plot(ncells,residual_energy-residual_energy[-1],label="residual")
        plt.legend()
        plt.savefig("cell_convergence.png")
        plt.clf()


    if test_interlayer:
        model_type = "TETB" #"Classical"
        tb_model = "popov" #"MK"
        int_type = 'interlayer'

        cutoff = 7
        mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
        #A, B, n, m
        ave_bond_order = -32
        epsilon = 0.02 * ave_bond_order
        sigma = 3.5 / 2**(1/6)
        
        interlayer_params = np.array([4*epsilon*sigma**12, 4*epsilon*sigma**6, -12, -6]) 
        calc = get_BLG_Model(int_type=int_type,energy_model=model_type,tb_model=tb_model,calc_type="python")
        """model_dict =  {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"all","cutoff":cutoff},"cutoff":cutoff,
                                    "potential":TETB_react_interlayer,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,"overlap parameters":None,
                                    "descriptors":None,"descriptor kwargs":{"type":"all","cutoff":None},"cutoff":None,
                                    "potential":None,"potential parameters":None}} """
        #calc = TETB_model(model_dict)
        #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
        """params = np.array([ 2.53540643,  3.03014623e3, -5.46704907e-6, -8.93881508e-5,
  7.97828994e1,  1.21479622,  2.78766240, -5.79453081,
 -2.45235024e3])
        
        calc.set_param_element("interlayer","potential parameters",params)"""

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
            residual_energy = []
            d_ = []
            dis = disreg_[i]
            d_stack = df.loc[df['stacking'] == stacking, :]
            for j, row in d_stack.iterrows():
                atoms = get_bilayer_atoms(row["d"],dis,sc=5)
                #theta = 5.09
                #atoms = get_twist_geom(theta,3.35)

                tmp_total_energy,_ = calc.get_total_energy(atoms)
                total_energy = (tmp_total_energy)/len(atoms)
                residual_energy_tmp,_ = calc.get_residual_energy(atoms)
                residual_energy_tmp /= len(atoms)
                residual_energy.append(residual_energy_tmp)
                tb_energy,tb_forces = calc.get_tb_energy(atoms)
                tb_energy /= len(atoms)
                #total_energy = tb_energy + residual_energy_tmp
                #total_energy = Kolmogorov_Crespi(atoms,kc_parameters)
                print(total_energy)
                
                if total_energy<E0_tegt:
                    E0_tegt = total_energy

                qmc_total_energy = (row["energy"])

                energy_dis_tegt.append(total_energy)
                energy_dis_qmc.append(qmc_total_energy)
                energy_dis_tb.append(tb_energy)
                d_.append(row["d"])

            relative_tetb_energies.append(energy_dis_tegt)
            relative_qmc_energies.append(energy_dis_qmc)
            #plt.scatter(np.array(d_),np.array(residual_energy)-(residual_energy[-1]),label=stacking + " residual energy",c=colors[i],marker="*")
            plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " "+model_type,c=colors[i])
            plt.plot(d_,(np.array(energy_dis_qmc)-E0_qmc)-(np.array(energy_dis_tb)-energy_dis_tb[-1]),label=stacking + " QMC Diff. "+model_type,c=colors[i],linestyle="dashed")
            #plt.scatter(np.array(d_),np.array(energy_dis_tb)-(energy_dis_tb[-1]),label=stacking + " TB",c=colors[i],marker=",")
            plt.scatter(np.array(d_),np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])

        layer_sep = np.array([3.3266666666666667,3.3466666666666667,3.3866666666666667,3.4333333333333336,3.5,3.5733333333333333,3.6466666666666665,3.7666666666666666,3.9466666666666668,4.113333333333333,4.3533333333333335,4.54,4.76,5.013333333333334,5.16])
        popov_energies_sep = np.array([ 0.0953237410071943, 0.08884892086330941, 0.07877697841726625, 0.06582733812949645, 0.05323741007194249, 0.042086330935251826, 0.03237410071942448, 0.02230215827338132, 0.01151079136690649, 0.007194244604316571, 0.0025179856115108146, 0.0010791366906475058, 0.0007194244604316752, 0.00035971223021584453, 1.3877787807814457e-17])
        #plt.plot(layer_sep,popov_energies_sep,label="popov tb energies")
        plt.xlabel(r"Interlayer Distance ($\AA$)",**csfont)
        plt.ylabel(model_type+" Interlayer Energy (eV/atom)",**csfont)
        plt.title(model_type,**csfont)
        plt.tight_layout()
        plt.legend()
        plt.savefig("figures/interlayer_test"+model_type+".jpg")
        plt.clf()

    if test_intralayer_lat_con:
        def quadratic_function(x,a,b,c):
            return a*(x-b)**2 + c

        model_type = "Classical"
        tb_model = None #"MK"
        int_type = 'intralayer'

        calc = get_BLG_Model(int_type=int_type,energy_model=model_type,tb_model=tb_model)
        lammps_calc = get_BLG_Model(int_type=int_type,energy_model=model_type,tb_model=tb_model,calc_type="lammps")
        a = 2.462
        lat_con_list_dft = np.sqrt(3) * np.array([1.3027833001988072,1.3433399602385685,1.4053677932405566,\
                            1.4745526838966203,1.5294234592445326,1.5795228628230618])

        lat_con_list = np.linspace(np.min(lat_con_list_dft),np.max(lat_con_list_dft),15)

        lat_con_energy = np.zeros_like(lat_con_list)
        tb_energy = np.zeros_like(lat_con_list)
        rebo_energy = np.zeros_like(lat_con_list)
        lammps_energy = np.zeros_like(lat_con_list)
        dft_energy = np.array([-8.472277446,-8.961917385,-9.251954937,-9.119902805,-8.832030042,-8.432957809])

        for i,lat_con in enumerate(lat_con_list):
        
            atoms = get_monolayer_atoms(0,0,a=lat_con)
            atoms.set_array('mol-id',np.ones(len(atoms),dtype=np.int64))
            print("a = ",lat_con," natoms = ",len(atoms))
            total_energy,f = calc.get_total_energy(atoms)
            total_energy /= len(atoms)
            #tb_energy_geom,tb_forces = calc_obj.run_tight_binding(atoms)
            #tb_energy[i] = tb_energy_geom/len(atoms)
            lammps_e,lammps_f = lammps_calc.get_total_energy(atoms)
            lammps_energy[i] = lammps_e/len(atoms)
            #rebo_energy[i] = total_energy/len(atoms)
            #total_energy = tote + tb_energy_geom
            lat_con_energy[i] = total_energy
        
        fit_min_ind = np.argmin(lat_con_energy)
        initial_guess = (1.0, 1.42, np.min(lat_con_energy))  # Initial parameter guess
        rebo_params, covariance = curve_fit(quadratic_function, lat_con_list, lat_con_energy, p0=initial_guess)
        rebo_min = np.min(lat_con_energy*len(atoms))

        dft_min_ind = np.argmin(dft_energy)
        initial_guess = (1.0, 1.42, np.min(dft_energy))  # Initial parameter guess
        dft_params, covariance = curve_fit(quadratic_function, lat_con_list_dft, dft_energy, p0=initial_guess)
        dft_min = dft_params[-1]

        print("rebo fit minimum energy = ",str(rebo_params[-1]))
        print("rebo fit minimum lattice constant = ",str(lat_con_list[fit_min_ind]))
        print("rebo young's modulus = ",str(rebo_params[0]))
        print("DFT minimum energy = ",str(dft_params[-1]))
        print("DFT minimum lattice constant = ",str(lat_con_list_dft[dft_min_ind]))
        print("DFT young's modulus = ",str(dft_params[0]))

        plt.plot(lat_con_list/np.sqrt(3),lat_con_energy-np.min(lat_con_energy),label = "python tersoff fit")
        plt.plot(lat_con_list/np.sqrt(3),lammps_energy-np.min(lammps_energy),label = "lammps tersoff fit",marker="*")
        #plt.plot(lat_con_list/np.sqrt(3),tb_energy-tb_energy[fit_min_ind],label = "tight binding energy")
        #plt.plot(lat_con_list/np.sqrt(3),rebo_energy - rebo_energy[fit_min_ind],label="rebo corrective energy")
        #plt.plot(lat_con_list_dft/np.sqrt(3), dft_energy-np.min(dft_energy),label="dft results")
        plt.xlabel(r"nearest neighbor distance ($\AA$)")
        plt.ylabel("energy above ground state (eV/atom)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("rebo_lat_con_"+model_type+".jpg")
        plt.clf()

    if test_intralayer:
        model_type = "Classical"
        tb_model = None #"MK"
        int_type = 'intralayer'

        calc = get_BLG_Model(int_type=int_type,energy_model=model_type,tb_model=tb_model,calc_type="python")

        db = ase.db.connect('../data/monolayer_nkp121.db')
        energy = []
        
        nconfig=0
        dft_min = 1e8
        for row in db.select():
            if row.data.total_energy<dft_min:
                dft_min = row.data.total_energy
        tegtb_energy = []
        dft_energy = []   
        nn_dist = []
        atoms_id =[]
        unstrained_atoms = get_monolayer_atoms(0,0,a=2.462)
        unstrained_cell = unstrained_atoms.get_cell()
        
        for row in db.select():
    
            atoms = db.get_atoms(id = row.id)
            atoms_id.append(row.id)

            e,_ = calc.get_total_energy(atoms)
            e /= len(atoms)
            tegtb_energy.append(e)
            dft_energy.append(row.data.total_energy)
            nconfig+=1

            pos = atoms.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            nn_dist.append(average_distance)
        dft_min = np.min(dft_energy)
        rebo_min_ind = np.argmin(tegtb_energy)
        rebo_min = tegtb_energy[rebo_min_ind]

        rms_tetb  = []
        rms_rebo = []
        for i,e in enumerate(tegtb_energy):
            line = np.linspace(0,1,10)
            ediff_line = line*((dft_energy[i]-dft_min) - (e-rebo_min)) + (e-rebo_min)
            tmp_rms = np.linalg.norm((dft_energy[i]-dft_min) - (e-rebo_min))/(dft_energy[i]-dft_min)

            #if tmp_rms >0.15:
            #    del db[atoms_id[i]]
            #    continue
            print("dft energy (eV/atom) = ",dft_energy[i]-dft_min)
            print("tegtb energy (eV/atom) = ",e-rebo_min)
            print("\n")
            average_distance = nn_dist[i]
            if nn_dist[i] > 1.5 or (dft_energy[i]-dft_min)>0.4:
                continue
            rms_tetb.append(tmp_rms)

            if i==0:
                plt.scatter(average_distance,e-rebo_min,color="red",label=model_type)
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue",label="DFT")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
            else:
                plt.scatter(average_distance,e-rebo_min,color="red")
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
        
        print("rms tetb ",rms_tetb)

        rms_tetb = np.array(rms_tetb)
        rms_rebo = np.array(rms_rebo)
        rms_tetb = rms_tetb[rms_tetb<1e3]
        rms_rebo = rms_rebo[rms_rebo<1e3]
        rms_tetb = np.mean(rms_tetb)
        rms_rebo = np.mean(rms_rebo)
        #rms_tetb = np.mean(np.abs(np.array(tegtb_energy)-rebo_min-(np.array(dft_energy)-dft_min)))
        #rms_rebo = np.mean(np.abs(np.array(rebo_energy)-emprebo_min-(np.array(dft_energy)-dft_min)))
        print("average rms tetb = ",rms_tetb)
        
        print("average difference in tetb energy across all configurations = "+str(rms_tetb)+" (eV/atom)")
        print("average difference in rebo energy across all configurations = "+str(rms_rebo)+" (eV/atom)")
        plt.xlabel(r"average nearest neighbor distance ($\AA$)",**csfont)
        plt.ylabel("energy (eV/atom)",**csfont)
        plt.title(model_type+" intralayer energy",**csfont)

        plt.legend()
        #plt.colorbar().set_label('RMS', rotation=270,**csfont)
        #plt.clim((1e-5,1e-4))
        plt.tight_layout()
        plt.savefig("figures/intralayer_test_"+model_type+".jpg")
        plt.clf()

    if interlayer_strain:
        model_type = "Classical"
        tb_model = None #"MK"
        int_type = 'interlayer'
        csfont = {'fontname':'serif',"size":25}

        
        strain_percent = np.array([-0.02,0,0.02])
        layer_sep = np.linspace(3,5,15)
        Eshift =  0.002
        linestyle = ["dotted","dashed","dashdot"]
        marker = ["s","*","x"]
        label = ["TETB(nkp = 121)","classical"]
        models = [("TETB","popov","interlayer"),("Classical",None,"interlayer")]
        for j,m in enumerate(models):
            calc_obj = get_BLG_Model(int_type=m[2],energy_model=m[0],tb_model=m[1],calc_type="python")
            for strain_ind,t in enumerate(strain_percent):
                layer_energies = np.zeros(len(layer_sep))
                for k,d in enumerate(layer_sep):
                    atoms = get_bilayer_atoms(d,0,a = (1+t)*2.46 ,sc=1)
                    atoms.calc = calc_obj
                    energy,_ = calc_obj.get_total_energy(atoms)
                    layer_energies[k] = energy/len(atoms)
                plt.plot(layer_sep,layer_energies-layer_energies[-1]-Eshift,label=r" $\epsilon =$ "+str(t),linestyle=linestyle[strain_ind],marker=marker[strain_ind])
            plt.legend()
            plt.xlabel(r"Interlayer separation $(\AA)$",**csfont)
            plt.ylabel(r"Interlayer Energy ($eV$)",**csfont)
            plt.title(label[j],**csfont)
            plt.tight_layout()
            plt.savefig("figures/strained_interlayer_energy_"+label[j]+".png")
            plt.clf() 
