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
#from BLG_model_builder.TB_Utils_torch import *
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
#from BLG_model_builder.descriptors_torch import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
#from BLG_model_builder.TETB_model_builder_torch import *
from BLG_model_builder.TETB_model_builder import *
from BLG_model_builder.BLG_model_library import *
from model_fit import *
import uuid

try:
    import cupy
    import cupyx as cpx
    if cupy.cuda.is_available():
        np = cupy
        
        gpu_avail = True
    else:
        gpu_avail = False
except:
    gpu_avail = False

def get_relaxed_struct(atoms,calc,theta):
    """ evaluate corrective potential energy, forces in lammps 
    """
    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        #sym = atoms.get_chemical_symbols()
        #top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        pos = np.asarray(atoms.positions)
        mean_z = np.mean(pos[:,2])
        top_layer_ind = np.where(pos[:,2]>mean_z)[0]
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
        dyn.run(fmax=1e-3,steps=150)

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
    parser.add_argument('-c','--calc_dir',type=str,default="")
    parser.add_argument('-o','--optimal_hyperparam',type=str,default="marginal_likelihood")
    args = parser.parse_args() 
    #works for MK, letb interlayer, letb intralayer nn 1, classical interlayer energy, letb intralayer nn 2, letb intralayer nn 3
    #check  classical intralayer energy, tetb intralayer, tetb interlayer
    ################## Adjustable, set model ##########################################
    int_type = "full"
    energy_model = args.energy_model
    tb_model = args.tb_model
    nn_val = int(args.nn_val)
    uq_type = args.uq_type
    optimal_hyperparam = args.optimal_hyperparam

    if energy_model !="Classical" and energy_model!="TETB":
        print("Energy model must be 'Classical' or 'TETB'")
        exit()
    # if tb_model !="LETB" and tb_model!="MK" and tb_model!='None' and tb_model != "popov" and tb_model != "MLP_SK":
    #     print("tb model must be 'None', 'LETB' or 'MK' or 'popov'")
    #     exit()

    print("int type = ",int_type)
    print("energy model = ",str(energy_model))
    print("tb_model = ",str(tb_model))
    print("optimal_hyperparam = ",str(optimal_hyperparam))

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
    twist_angles = np.array([0.83,0.88,0.93,0.99,1.08,1.12,1.16,1.2,1.47,1.89,2.88])
    n_ensembles = 75
    theta = float(args.theta)

    #with open("ensembles/Optimal_Temperature_weight_models.pkl", 'rb') as file:
    #    opt_temp_weight = pickle.load(file)
    if optimal_hyperparam == "miscalibration_area":
        opt_ensemble = {"mcmc":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_ensemble_T_2.0.pkl",
                        "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_ensemble_T_10.0.pkl",
                        "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_ensemble_T_0.2.pkl",
                        "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_ensemble_T_3.0.pkl",
                        "TETB_energy_interlayer_popov":"ensembles/TETB_energy_interlayer_popov/TETB_energy_interlayer_popov_ensemble_T_0.5.pkl",
                        "TETB_energy_intralayer_popov":"ensembles/TETB_energy_intralayer_popov/TETB_energy_intralayer_popov_ensemble_T_0.0001.pkl",
                        "MK":"ensembles/MK/MK_ensemble_T_0.5.pkl",
                        "intralayer_LETB_NN_val_1":"ensembles/intralayer_LETB_NN_val_1/intralayer_LETB_NN_val_1_ensemble_T_10.0.pkl",
                        "intralayer_LETB_NN_val_2":"ensembles/intralayer_LETB_NN_val_2/intralayer_LETB_NN_val_2_ensemble_T_0.5.pkl",
                        "intralayer_LETB_NN_val_3":"ensembles/intralayer_LETB_NN_val_3/intralayer_LETB_NN_val_3_ensemble_T_1.0.pkl",
                        "interlayer_LETB":"ensembles/interlayer_LETB/interlayer_LETB_ensemble_T_7.0.pkl",
                        "MLP_SK_15_2":"ensembles/MLP_SK_15_2/MLP_SK_15_2_ensemble_T_4.0.pkl",
                        },
                        
                        "cv":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_CV_ensemble_p_0.8.pkl",
                        "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_CV_ensemble_p_0.2.pkl",
                        "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_CV_ensemble_p_0.5.pkl",
                            "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_CV_ensemble_p_0.5.pkl",
                            "MK":"ensembles/MK/MK_CV_ensemble_p_0.9.pkl"}}
    
    elif optimal_hyperparam == "marginal_likelihood":
        opt_ensemble = {"mcmc":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_ensemble_T_0.5.pkl",
                        "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_ensemble_T_2.0.pkl",
                        "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_ensemble_T_0.2.pkl",
                        "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_ensemble_T_3.0.pkl",
                        "TETB_energy_interlayer_popov":"ensembles/TETB_energy_interlayer_popov/TETB_energy_interlayer_popov_ensemble_T_0.5.pkl",
                        "TETB_energy_intralayer_popov":"ensembles/TETB_energy_intralayer_popov/TETB_energy_intralayer_popov_ensemble_T_0.0001.pkl",
                        "MK":"ensembles/MK/MK_ensemble_T_0.5.pkl",
                        "intralayer_LETB_NN_val_1":"ensembles/intralayer_LETB_NN_val_1/intralayer_LETB_NN_val_1_ensemble_T_7.0.pkl",
                        "intralayer_LETB_NN_val_2":"ensembles/intralayer_LETB_NN_val_2/intralayer_LETB_NN_val_2_ensemble_T_0.5.pkl",
                        "intralayer_LETB_NN_val_3":"ensembles/intralayer_LETB_NN_val_3/intralayer_LETB_NN_val_3_ensemble_T_1.0.pkl",
                        "interlayer_LETB":"ensembles/interlayer_LETB/interlayer_LETB_ensemble_T_4.0.pkl",
                        "MLP_SK_15_2":"ensembles/MLP_SK_15_2/MLP_SK_15_2_ensemble_T_7.0.pkl",
                        "POD_energy":"ensembles/POD_energy/POD_energy_ensemble_T_0.5.pkl",
                        },
                        
                        "cv":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_CV_ensemble_p_0.8.pkl",
                        "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_CV_ensemble_p_0.2.pkl",
                        "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_CV_ensemble_p_0.5.pkl",
                            "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_CV_ensemble_p_0.5.pkl",
                            "MK":"ensembles/MK/MK_CV_ensemble_p_0.9.pkl"}}

    if qoi == "relax_atoms":
        
        interlayer_name = str(energy_model)+"_energy_interlayer_"+str(tb_model)
        interlayer_name = interlayer_name.replace("_None","")
        filename = opt_ensemble[uq_type][interlayer_name]
        with open(filename, 'rb') as file:
           ensemble_dict = pickle.load(file)
        interlayer_ensemble = np.asarray(ensemble_dict["ensemble"]["energy"])
        #interlayer_ensemble,ypred = read_params_and_predictions(filename)
        #interlayer_calc,xdata,ydata,ydn, interlayer_params,params_std,param_bounds, ypred_bestfit = get_MCMC_inputs("interlayer",energy_model,tb_model,interlayer_name,1)
        
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
        intralayer_ensemble = np.asarray(ensemble_dict["ensemble"]["energy"])
        

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
            #atoms = get_twist_geom(t)
            atoms = ase.io.read("starting_TETB_configs/theta_"+str(t)+".traj")
            
            run_indices = np.array_split(np.arange(n_ensembles),int(args.npartitions))
            for j in run_indices[int(args.ensemble_index)]:
                id = uuid.uuid4()
                calc = get_BLG_Model(int_type="full",energy_model=energy_model,
                tb_model=tb_model,output=model_name+"_t_"+str(t)+"_"+str(id),update_eigvals=1,
                calc_type="lammps")
                #calc.model_dict["intralayer"]["potential parameters"] = np.load("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz")["params"] 
                calc.model_dict["intralayer"]["potential parameters"] = intralayer_ensemble[j,:]
                #calc.model_dict["interlayer"]["potential parameters"] =  np.load("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz")["params"] 
                calc.model_dict["interlayer"]["potential parameters"] = interlayer_ensemble[j,:]
                if energy_model=="TETB":
                    if tb_model == "MK":
                        calc.model_dict["interlayer"]["hopping parameters"] =  interlayer_band_ensemble[j,:] #
                params = calc.get_params()
                calc.set_params(params)
                
                #try:
                relaxed_atoms,forces = get_relaxed_struct(atoms,calc,t)
                z  = relaxed_atoms.get_positions()[:,2]
                print("layer sep max = ",np.max(z)-np.min(z))
                
                #except:
                #    continue
                relaxed_atoms_list.append(relaxed_atoms.copy())
                #exit()

            #ase.io.write("relaxed_atoms_"+energy_model+"_theta_"+str(t)+"_"+uq_type+"_ensemble.xyz",relaxed_atoms_list,format="extxyz")

    if qoi == "rerelax":
        
        calc_dir = args.calc_dir
        theta_val = float(calc_dir.split("_")[-2])
        atoms_file = os.path.join(calc_dir,"mcmc_theta_"+str(theta_val)+".traj")
        intralayer_file = os.path.join(calc_dir,"intralayer_residual_nkp36.txt")
        interlayer_file = os.path.join(calc_dir,"interlayer_residual_nkp36.txt")

        atoms = ase.io.read(atoms_file)
        forces = atoms.get_forces()
        if np.max(forces) > 1e-3:
            exit()
        #make sure this is correct
        intralayer_params = read_Tersoff(intralayer_file)
        interlayer_params = read_kc_insp(interlayer_file)

        calc = get_BLG_Model(int_type="full",energy_model=energy_model,
            tb_model=tb_model,output=calc_dir,update_eigvals=1,calc_type="lammps")
        
        calc.model_dict["intralayer"]["potential parameters"] = intralayer_params
        calc.model_dict["interlayer"]["potential parameters"] = interlayer_params

        params = calc.get_params()
        calc.set_params(params)
        relaxed_atoms,forces = get_relaxed_struct(atoms,calc,theta_val)

    if qoi == "band_structure":
        
        
        relaxed_atoms_list = ase.io.read("relaxed_structures/relaxed_atoms_"+energy_model+"_theta_"+str(args.theta)+"_"+uq_type+"_ensemble.xyz",format="extxyz",index=":")
        if hopping_model=="MK":
            hopping_model = "MK"
            filename = opt_ensemble[uq_type]["MK"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            interlayer_band_ensemble = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(interlayer_band_ensemble)[0]:
                indices = np.arange(np.shape(interlayer_band_ensemble)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                interlayer_band_ensemble = interlayer_band_ensemble[selected,:]
        elif "MLP_SK" in hopping_model:
            filename = opt_ensemble[uq_type]["MLP_SK_15_2"]
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

            filename = opt_ensemble[uq_type]["intralayer_LETB_NN_val_1"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            intralayer_nn1 = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(intralayer_nn1)[0]:
                indices = np.arange(np.shape(intralayer_nn1)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                intralayer_nn1 = intralayer_nn1[selected,:]

            filename = opt_ensemble[uq_type]["intralayer_LETB_NN_val_2"]
            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            intralayer_nn2 = np.asarray(ensemble_dict["ensemble"]["hoppings"])
            if n_ensembles<np.shape(intralayer_nn2)[0]:
                indices = np.arange(np.shape(intralayer_nn2)[0])
                selected = np.random.choice(indices, size=n_ensembles, replace=False)
                intralayer_nn2 = intralayer_nn2[selected,:]

            filename = opt_ensemble[uq_type]["intralayer_LETB_NN_val_3"]
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
        calc = get_BLG_Model(int_type="full",energy_model=energy_model,tb_model=hopping_model,output=model_name+"_t_"+str(args.theta),calc_type="python")
        #calc = get_BLG_Model(int_type="full",energy_model="TETB",tb_model="popov",calc_type="python")
        
        for i in run_indices[int(args.ensemble_index)]:
            if os.path.exists("ensemble_bands/band_structure_"+energy_model+"_"+tb_model+"_"+uq_type+"_t_"+str(args.theta)+"_"+str(i)):
                continue
            calc.model_dict["interlayer"]["hopping parameters"] = interlayer_band_ensemble[int(i),:]
            if tb_model=="LETB":
                calc.model_dict["intralayer"]["hopping parameters"] = intralayer_band_ensemble[int(i),:]
            #calc = TETB_model(model_dict)
            print("number of atoms = ",len(relaxed_atoms_list[int(i)]))
            relaxed_atoms = relaxed_atoms_list[int(i)]
            z = np.asarray(relaxed_atoms.positions[:,2])
            mean_z = np.mean(z)
            top_ind = np.where(z>mean_z)[0]
            bottom_ind = np.where(z<mean_z)[0]
            mol_id = np.ones(len(relaxed_atoms))
            mol_id[top_ind] = 2
            if gpu_avail:
                relaxed_atoms.set_array("mol-id",np.asnumpy(mol_id))
            else:
                relaxed_atoms.set_array("mol-id",mol_id)
            evals = calc.get_band_structure(relaxed_atoms,kvec)
            np.savez("ensemble_bands/band_structure_"+energy_model+"_"+tb_model+"_"+uq_type+"_t_"+str(args.theta)+"_"+str(i)+"_"+optimal_hyperparam,evals=evals,kvec=kvec,k_dist=k_dist,k_node=k_node)
            del evals


