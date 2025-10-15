import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import argparse
import pickle
from lammps import PyLammps
import ase.io
import os
import flatgraphene as fg
from ase.optimize import FIRE
from ase.optimize import LBFGS
from ase.calculators.calculator import Calculator, all_changes
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from BLG_model_builder.TB_Utils_torch import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors_torch import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder_torch import *
from BLG_model_builder.BLG_model_library import *
from model_fit import *
import uuid
from ensemble_plotter import read_params_and_predictions

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

def get_relaxed_struct(atoms,calc,theta):
    """ evaluate corrective potential energy, forces in lammps 
    """
    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        sym = atoms.get_chemical_symbols()
        top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        mol_id[top_layer_ind] += 1
        atoms.set_array("mol-id",mol_id)
    if calc.model_dict["interlayer"]["hopping form"] is None:
        relax_atoms,forces = calc.relax_structure(atoms)
    else:
        output = calc.output
        #read in TETB relaxed structures first and relax from there
        atoms = get_twist_geom(theta) #ase.io.read(os.path.join(output,"theta_"+str(theta)+".traj"))
        atoms.calc = calc
        #dyn = FIRE(atoms,finc=1.1,fdec=0.5,dtmax=1,trajectory=os.path.join(output,"mcmc_theta_"+str(theta)+".traj"),
        #           logfile=os.path.join(output,"mcmc_theta_"+str(theta)+".log"), max_steps=100)
        dyn = LBFGS(atoms,trajectory=os.path.join(output,"mcmc_theta_"+str(theta)+".traj"),
                   logfile=os.path.join(output,"mcmc_theta_"+str(theta)+".log"))
        dyn.run(fmax=1e-3,steps=50)

        relax_atoms = ase.io.read(os.path.join(output,"mcmc_theta_"+str(theta)+".traj"),index=-1)
        forces = relax_atoms.get_forces()
    return relax_atoms,forces

def get_twist_geom(theta,layer_sep=3.35,a=2.46):
    #comp is 2d vector for compression percentage along cell vectors
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                        mass=[12.01,12.01],sep=layer_sep,h_vac=20)
    return atoms

if __name__=="__main__":
    #define hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--energy_model',type=str,default="Classical")
    parser.add_argument('-t','--tb_model',type=str,default='MK')
    parser.add_argument('-nv','--nn_val',type=str,default="1")
    parser.add_argument('-q','--qoi',type=str,default="relax_atoms")
    parser.add_argument('-i','--ensemble_index',type=str,default="0")
    parser.add_argument('-n','--npartitions',type=str,default="1")
    parser.add_argument('-a','--theta',type=str,default="2.88")
    parser.add_argument('-u','--uq_type',type=str,default="mcmc")
    args = parser.parse_args() 
    #works for MK, letb interlayer, letb intralayer nn 1, classical interlayer energy, letb intralayer nn 2, letb intralayer nn 3
    #check  classical intralayer energy, tetb intralayer, tetb interlayer
    ################## Adjustable, set model ##########################################
    int_type = "full"
    energy_model = args.energy_model
    tb_model = args.tb_model
    nn_val = int(args.nn_val)
    uq_type = args.uq_type

    if energy_model !="Classical" and energy_model!="TETB":
        print("Energy model must be 'Classical' or 'TETB'")
        exit()
    if tb_model !="LETB" and tb_model!="MK" and tb_model!='None' and tb_model != "popov":
        print("tb model must be 'None', 'LETB' or 'MK' or 'popov'")
        exit()

    print("int type = ",int_type)
    print("energy model = ",str(energy_model))
    print("tb_model = ",str(tb_model))

    if energy_model =="None": energy_model=None
    if tb_model =="None": tb_model = None
    #define model name
    model_name = str(energy_model)+"_energy_"+str(int_type)+"_"+str(tb_model)
    model_name = model_name.replace("full_","")
    model_name = model_name.replace("None_energy_","")
    model_name = model_name.replace("_None","")
    if model_name =="intralayer_LETB":
        model_name = model_name + "_NN_val_"+str(nn_val)

    Total_energy_type = args.energy_model
    hopping_model = args.tb_model
    qoi = args.qoi
    twist_angles = np.array([0.88,0.99,1.08,1.2,1.47,1.89,2.88])
    n_ensembles = 75
    theta = float(args.theta)

    #with open("ensembles/Optimal_Temperature_weight_models.pkl", 'rb') as file:
    #    opt_temp_weight = pickle.load(file)
    opt_ensemble = {"mcmc":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_ensemble_T_2.0.pkl",
                    "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_ensemble_T_10.0.pkl",
                    "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_ensemble_T_0.2.pkl",
                    "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_ensemble_T_3.0.pkl",
                    "TETB_energy_interlayer_popov":"ensembles/TETB_energy_interlayer_popov/TETB_energy_interlayer_popov_ensemble_T_2.0.txt",
                    "TETB_energy_intralayer_popov":"ensembles/TETB_energy_intralayer_popov/TETB_energy_intralayer_popov_ensemble_T_0.0001.pkl",
                    "MK":"ensembles/MK/MK_ensemble_T_0.01.pkl"},
                    
                    "cv":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_CV_ensemble_p_0.8.pkl",
                       "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_CV_ensemble_p_0.2.pkl",
                       "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_CV_ensemble_p_0.5.pkl",
                        "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_CV_ensemble_p_0.5.pkl",
                        "MK":"ensembles/MK/MK_CV_ensemble_p_0.9.pkl"}}

    if qoi == "relax_atoms":
        
        interlayer_name = str(energy_model)+"_energy_interlayer_"+str(tb_model)
        interlayer_name = interlayer_name.replace("_None","")
        filename = opt_ensemble[uq_type][interlayer_name]
        #with open(filename, 'rb') as file:
        #    ensemble_dict = pickle.load(file)
        #interlayer_ensemble = ensemble_dict["ensemble"]["energy"]
        interlayer_ensemble,ypred = read_params_and_predictions(filename)
        interlayer_calc,xdata,ydata,ydn, interlayer_params,params_std,param_bounds, ypred_bestfit = get_MCMC_inputs("interlayer",energy_model,tb_model,interlayer_name,1)
        """ypred = ensemble_dict["ypred_samples"]["energy"]
        interlayer_ensemble = []
        d_ = np.array([3,3.2,3.35,3.5,3.65,3.8,4,4.5,5,6,7])
        d_ = np.hstack((d_,d_,d_,d_))
        
        print("\n\n interlayer best fit params = ",interlayer_params["energy"])
        print(uq_type+" interlayer ensemble mean = ",np.mean(interlayer_ensemble_full,axis=0))
        print(uq_type+" interlayer ensemble std = ",np.std(interlayer_ensemble_full,axis=0))

        for n in range(np.shape(interlayer_ensemble_full)[0]):
            d_min = np.min(d_)
            d_max = np.max(d_)
            d_eq = d_[np.argmin(ypred[n,:])]
            if d_eq > d_min and d_eq < d_max:
                #local minimum present
                interlayer_ensemble.append(interlayer_ensemble_full[n,:])
        interlayer_ensemble = np.array(interlayer_ensemble_full)
        deltaE=0.021
        min_ind = np.argmin(ydata["energy"])

        if energy_model=="TETB":
            tb_calc = get_BLG_Model(int_type="interlayer",energy_model=energy_model,tb_model=tb_model,calc_type="lammps")
            tb_calc.model_dict["intralayer"]["potential parameters"] = np.load("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz")["params"]
            tb_calc.model_dict["interlayer"]["potential parameters"] = np.load("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz")["params"]
            
            ytotal,_ = tb_calc.get_total_energy(xdata["energy"][0])
            tb_energy = [0]
            for x in xdata["energy"][1:]:
                tbe,_ = 0,0 #tb_calc.get_tb_energy(x)
                tb_energy = np.append(tb_energy,tbe)
                yt,_ = tb_calc.get_total_energy(x)
                ytotal = np.append(ytotal,yt)
            #tb_energy *= len(xdata["energy"][0])/len(tb_xdata["energy"][0])
            tb_energy -= tb_energy[-1]
            ypred = ypred + tb_energy[np.newaxis,:]
           

        ci = 0.64
        lower_bound = np.quantile(ypred - ypred[:,min_ind][:,np.newaxis],(1-ci)/2,axis=0)
        upper_bound = np.quantile(ypred- ypred[:,min_ind][:,np.newaxis],1-(1-ci)/2,axis=0)
        ypred_std = (upper_bound - lower_bound)/4
        plt.scatter(d_,(ydata["energy"]-ydata["energy"][min_ind])/len(x) - deltaE,label="qmc",color="black")
        #plt.errorbar(d_,(np.mean(ypred-ypred[:,min_ind][:,np.newaxis],axis=0))/len(x) - deltaE,yerr = ypred_std,fmt="o",label="ypred",color="red")
        plt.scatter(d_,(ytotal-ytotal[min_ind])/len(x) - deltaE,label="best fit",color="blue")
        plt.xlabel("layer sep")
        plt.ylabel("Energy (eV/atom)")
        plt.legend()
        plt.savefig("figures/"+interlayer_name+"_"+uq_type+".png")
        plt.clf()
        exit()"""
        
        if np.shape(interlayer_ensemble)[0] > n_ensembles:
            indices = np.arange(np.shape(interlayer_ensemble)[0])
            selected = np.random.choice(indices, size=n_ensembles, replace=False)
            interlayer_ensemble = interlayer_ensemble[selected,:]

        intralayer_name = str(energy_model)+"_energy_intralayer_"+str(tb_model)
        intralayer_name = intralayer_name.replace("None_energy_","")
        intralayer_name = intralayer_name.replace("_None","")
        
        filename = opt_ensemble[uq_type][intralayer_name]
        with open(filename, 'rb') as file:
            ensemble_dict = pickle.load(file)
        intralayer_ensemble = ensemble_dict["ensemble"]["energy"]
        
        """intralayer_ensemble = []
        nn_dist = []
        intralayer_calc,xdata,ydata,ydata_noise, intralayer_params,params_std,param_bounds, ypred_bestfit = get_MCMC_inputs("intralayer",energy_model,tb_model,intralayer_name,1)

        print("\n\n intralayer best fit params = ",intralayer_params["energy"])
        print(uq_type+" intralayer ensemble mean = ",np.mean(intralayer_ensemble_full,axis=0))
        print(uq_type+" intralayer ensemble std = ",np.std(intralayer_ensemble_full,axis=0))

        if np.ndim(ensemble_dict["ypred_samples"]) ==1:
            ypred,_ = evaluate_ensemble(ensemble_dict["ensemble"],xdata, ydata, intralayer_calc)
            ypred = ypred["energy"]
            
        else:
            ypred = ensemble_dict["ypred_samples"]["energy"]


        for atoms in xdata["energy"]:
            pos = atoms.positions
            distances = cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            nn_dist.append(average_distance)
        nn_dist = np.array(nn_dist)

        for n in range(np.shape(intralayer_ensemble_full)[0]):
            nn_min = np.min(nn_dist)
            nn_max = np.max(nn_dist)
            nn_eq = nn_dist[np.argmin(ypred[n,:])]
            if nn_eq > nn_min and nn_eq < nn_max:
                #local minimum present
                intralayer_ensemble.append(intralayer_ensemble_full[n,:])
        intralayer_ensemble = np.array(intralayer_ensemble_full)"""
        
        """if energy_model=="TETB":
            tb_calc = get_BLG_Model(int_type="intralayer",energy_model=energy_model,tb_model=tb_model,calc_type="python")
            tb_xdata,_,_ = get_training_data("intralayer energy") 
            tb_energy,_ = tb_calc.get_tb_energy(tb_xdata["energy"][0])
            tb_energy,_ = tb_calc.get_tb_energy(tb_xdata["energy"][0])
            for x in tb_xdata["energy"][1:]:
                tbe,_ = tb_calc.get_tb_energy(x)
                tb_energy = np.append(tb_energy,tbe)
            ypred = ypred + tb_energy[np.newaxis,:]

        plt.scatter(nn_dist,ydata["energy"]/len(xdata["energy"][0]),label="dft",color='black')
        min_ind = np.argmin(ydata["energy"])
        plt.errorbar(nn_dist,(np.mean(ypred-ypred[:,min_ind][:,np.newaxis],axis=0))/len(xdata["energy"][0]),
                     yerr = (np.std(ypred-ypred[:,min_ind][:,np.newaxis],axis=0))/len(xdata["energy"][0]),fmt="o",label="ypred")
        plt.xlabel("average nn dist")
        plt.ylabel("Energy (eV/atom)")
        plt.legend()
        plt.savefig("figures/"+intralayer_name+"_"+uq_type+".png")
        plt.clf()
        """

        if np.shape(intralayer_ensemble)[0] > n_ensembles:
            indices = np.arange(np.shape(intralayer_ensemble)[0])
            selected = np.random.choice(indices, size=n_ensembles, replace=False)
            intralayer_ensemble = intralayer_ensemble[selected,:]

        if hopping_model=="MK":
            filename = opt_ensemble[uq_type]["MK"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            interlayer_band_ensemble = ensemble_dict["ensemble"]["hoppings"]
            if n_ensembles<np.shape(interlayer_band_ensemble)[0]:
                indices = np.arange(np.shape(interlayer_band_ensemble)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                interlayer_band_ensemble = interlayer_band_ensemble[selected,:]

        model_name = str(energy_model)+"_energy_"+str(tb_model)
        model_name = model_name.replace("full_","")
        model_name = model_name.replace("None_energy_","")
        model_name = model_name.replace("_None","")

        #twist_angles = np.array([0.88,0.99,1.08,1.12,1.16,1.2,1.47,1.89,2.88])
        twist_angles = [theta]

        for i,t in enumerate(twist_angles):
            relaxed_atoms_list = []
            atoms = get_twist_geom(t)
            #atoms = ase.io.read("starting_TETB_configs/theta_"+str(t)+".traj")
            
            run_indices = np.array_split(np.arange(n_ensembles),int(args.npartitions))
            for j in run_indices[int(args.ensemble_index)]:
                id = uuid.uuid4()
                calc = get_BLG_Model(int_type="full",energy_model=energy_model,
                tb_model=tb_model,output=model_name+"_t_"+str(t)+"_"+str(id),update_eigvals=1,
                calc_type="lammps",force_method="autograd")
                calc.model_dict["intralayer"]["potential parameters"] = np.load("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz")["params"] #intralayer_ensemble[j,:] # 
                calc.model_dict["interlayer"]["potential parameters"] =  np.load("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz")["params"] #interlayer_ensemble[j,:] #

                if energy_model=="TETB":
                    if tb_model == "MK":
                        calc.model_dict["interlayer"]["hopping parameters"] =  interlayer_band_ensemble[j,:] #
                params = calc.get_params()
                calc.set_params(params)
                
                #try:
                relaxed_atoms,forces = get_relaxed_struct(atoms,calc,t)
                
                #except:
                #    continue
                relaxed_atoms_list.append(relaxed_atoms.copy())
                exit()

            ase.io.write("relaxed_atoms_"+energy_model+"_theta_"+str(t)+"_"+uq_type+"_ensemble.xyz",relaxed_atoms_list,format="extxyz")

    if qoi == "band_structure":
        
        
        relaxed_atoms_list = ase.io.read("relaxed_atoms_"+energy_model+"_theta_"+str(args.theta)+"_"+uq_type+"_ensemble.xyz",format="extxyz",index=":")
        if hopping_model=="MK":
            filename = opt_ensemble[uq_type]["MK"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            interlayer_band_ensemble = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(interlayer_band_ensemble)[0]:
                indices = np.arange(np.shape(interlayer_band_ensemble)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                interlayer_band_ensemble = interlayer_band_ensemble[selected,:]

        elif hopping_model=="LETB":
            filename = opt_ensemble[uq_type]["interlayer_LETB"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            interlayer_band_ensemble = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(interlayer_band_ensemble)[0]:
                indices = np.arange(np.shape(interlayer_band_ensemble)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                interlayer_band_ensemble = interlayer_band_ensemble[selected,:]

            filename = opt_ensemble[uq_type]["intralayer_LETB_nn_val_1"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            intralayer_nn1 = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(intralayer_nn1)[0]:
                indices = np.arange(np.shape(intralayer_nn1)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                intralayer_nn1 = intralayer_nn1[selected,:]

            filename = opt_ensemble[uq_type]["intralayer_LETB_nn_val_2"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            intralayer_nn2 = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(intralayer_nn2)[0]:
                indices = np.arange(np.shape(intralayer_nn2)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                intralayer_nn2 = intralayer_nn2[selected,:]

            filename = opt_ensemble[uq_type]["intralayer_LETB_nn_val_3"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            intralayer_nn3 = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(intralayer_nn3)[0]:
                indices = np.arange(np.shape(intralayer_nn3)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                intralayer_nn3 = intralayer_nn3[selected,:]

            intralayer_band_ensemble = np.hstack((intralayer_nn1,intralayer_nn2,intralayer_nn3))
        
        Gamma = [0,   0,   0]
        K = [1/3,2/3,0]
        Kprime = [2/3,1/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nk=60
        (kvec,k_dist, k_node) = k_path(sym_pts,nk)

        n_ensembles = np.min(np.asarray([len(relaxed_atoms_list),np.shape(interlayer_band_ensemble)[0]]))
        run_indices = np.array_split(np.arange(n_ensembles),int(args.npartitions))
        calc = get_BLG_Model(int_type="full",energy_model=energy_model,tb_model=tb_model,output=model_name+"_t_"+str(args.theta),calc_type="python")
        for i in run_indices[int(args.ensemble_index)]:
            if os.path.exists("ensemble_bands/band_structure_"+energy_model+"_"+tb_model+"_"+uq_type+"_t_"+str(args.theta)+"_"+str(i)):
                continue
            calc.model_dict["interlayer"]["hopping parameters"] = interlayer_band_ensemble[int(i),:]
            if tb_model=="LETB":
                calc.model_dict["intralayer"]["hopping parameters"] = intralayer_band_ensemble[int(i),:]
            #calc = TETB_model(model_dict)
            print("number of atoms = ",len(relaxed_atoms_list[int(i)]))
            evals = calc.get_band_structure(relaxed_atoms_list[int(i)],kvec)
            np.savez("ensemble_bands/band_structure_"+energy_model+"_"+tb_model+"_"+uq_type+"_t_"+str(args.theta)+"_"+str(i),evals=evals,kvec=kvec,k_dist=k_dist,k_node=k_node)
            del evals


