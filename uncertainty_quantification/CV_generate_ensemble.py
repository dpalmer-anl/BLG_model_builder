import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import scipy.optimize
from sklearn.model_selection import KFold, RepeatedKFold
import arviz as az
import os
import time
import pickle
import subprocess
import argparse
from tqdm import tqdm

from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
from BLG_model_builder.BLG_model_library import *

from model_fit import *

def get_CV_ensemble(xdata, ydata, calc, w0, p0,bounds,p_subset,nsamples = 10,shift_data=False):
    #p_subset = % of training data 
    ensemble = {}
    for key in calc:
        potential_samples = []
        if key=="energy":
            Eshift_xdata = xdata[key][np.argmin(ydata[key])]
            Eshift = -calc[key](Eshift_xdata,p0[key])
        else:
            Eshift = 0
        for i in tqdm(range(nsamples)):
            indices = np.arange(len(ydata[key]))
            n_select = int(len(indices) * p_subset) 
            train_index = np.random.choice(indices, size=n_select, replace=False)
            if type(xdata[key])==list:
                xdata_train = [xdata[key][index] for index in train_index]
            else:
                if np.ndim(xdata[key]) > 1:
                    xdata_train = xdata[key][train_index,:]
                else:
                    xdata_train = xdata[key][train_index]
            ydata_train = ydata[key][train_index] 
            best_fit_params,_ = fit_model(calc[key],xdata_train,ydata_train,p0[key],tb_energy = Eshift,shift_data=False,bounds=bounds[key])
            potential_samples.append(best_fit_params)
        ensemble[key] = np.array(potential_samples)
    return ensemble

def get_TETB_CV_ensemble(xdata, ydata, calc, w0, p0,bounds,p_subset,tb_ensemble,nsamples = 30,shift_data=False):
    #p_subset = % of training data 
    ensemble = {}
    for key in ["energy"]:
        potential_samples = []
        Eshift_xdata = xdata[key][np.argmin(ydata[key])]
        print(np.append(p0["hoppings"],p0[key]))
        Eshift = -calc[key](Eshift_xdata,np.append(p0["hoppings"],p0[key]))
        for i in tqdm(range(nsamples)):
            indices = np.arange(len(ydata[key]))
            n_select = int(len(indices) * p_subset) 
            train_index = np.random.choice(indices, size=n_select, replace=False)
            if type(xdata[key])==list:
                xdata_train = [xdata[key][index] for index in train_index]
            else:
                xdata_train = xdata[key][train_index,:]
            ydata_train = ydata[key][train_index] 
            tb_calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="python")
            calc.update(get_BLG_Evaluator(int_type=int_type,energy_model=energy_model,tb_model=tb_model,energy_type="residual"))
            use_params = np.append(tb_ensemble["hoppings"][i,:],p0)
            tb_calc.set_params(use_params)
            tb_energy,_ = tb_calc.get_tb_energy(xdata_train[0])
            for x in xdata_train[1:]:
                tbe,_ = tb_calc.get_tb_energy(x)
                tb_energy = np.append(tb_energy,tbe)
            #tb_energy -= [np.argmin(np.abs(tb_energy))]

            #tbw_ = [0.1,0.25,0.35,0.5,0.6,0.75,0.85,1.0]
            tbw_ = [1.0]
            curr_params = p0[key]
            for tbw in tbw_:
                tb_energy_use = tb_energy.copy() * tbw + Eshift
                best_fit_params,_ = fit_model(calc[key],xdata_train,ydata_train,curr_params,tb_energy=tb_energy_use,shift_data=False,bounds=bounds[key])
                curr_params = best_fit_params.copy()

            potential_samples.append(best_fit_params)
        ensemble[key] = np.array(potential_samples).copy()
    ensemble["hoppings"] = tb_ensemble["hoppings"].copy()
    return ensemble
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--interaction_type',type=str,default="interlayer")
    parser.add_argument('-e','--energy_model',type=str,default='Classical')
    parser.add_argument('-t','--tb_model',type=str,default="None")
    parser.add_argument('-n','--nn_val',type=str,default="1")
    parser.add_argument('-p','--p_subset',type=str,default="0.5")
    args = parser.parse_args() 
    ################## Adjustable, set model ##########################################
    seed = 135726
    #np.random.seed(seed)

    int_type = args.interaction_type
    energy_model = args.energy_model
    tb_model = args.tb_model
    nn_val = int(args.nn_val)
    p_subset = float(args.p_subset)
    print("int type = ",int_type)
    print("energy model = ",str(energy_model))
    print("tb_model = ",str(tb_model))

    if energy_model =="None": energy_model=None
    if tb_model =="None": tb_model = None
    ####################################################################################
    #define model name
    model_name = str(energy_model)+"_energy_"+str(int_type)+"_"+str(tb_model)
    model_name = model_name.replace("full_","")
    model_name = model_name.replace("None_energy_","")
    model_name = model_name.replace("_None","")
    if model_name =="intralayer_LETB":
        model_name = model_name + "_NN_val_"+str(nn_val)
    
    if energy_model is not None:
        shift_data=True
    else:
        shift_data=False

    calc,xdata,ydata,ydata_noise, params,params_std,param_bounds, ypred_bestfit = get_MCMC_inputs(int_type,energy_model,tb_model,model_name,nn_val)
    w0={}
    relative_weight = 1/len(calc.keys())
    for nm,key in enumerate(calc):
        w0[key] = relative_weight **((nm+1)%2) * (1-relative_weight)**(nm%2)

    """if energy_model =="TETB":
        filename = "ensembles/MK/MK_CV_ensemble_p_"+str(p_subset)+".pkl"
        with open(filename, 'rb') as file:
            tb_ensemble_data = pickle.load(file)
        tb_ensemble = tb_ensemble_data["ensemble"]
        ensemble_samples = get_TETB_CV_ensemble(xdata, ydata, calc, w0, params,param_bounds,p_subset,tb_ensemble,shift_data=shift_data)
        print("not working version = ",ensemble_samples)
        intralayer_name = str(energy_model)+"_energy_intralayer_"+str(tb_model)
        intralayer_name = intralayer_name.replace("None_energy_","")
        intralayer_name = intralayer_name.replace("_None","")
        
        ensemble_samples = {}
        ensemble_samples["energy"] = np.array([[ 3.8049e4, 4.34999127, -9.29707912e-1,  7.27282331e-1,  8.86074006e-8, 2.2680887,  4.29999968e2,  3.50288017,  1.3936e3],
                                               [ 3.8049e4,4.34999017, -9.29703933e-1,  7.27279511e-1,  8.86133604e-8, 2.268077,  4.29999968e2,  3.50287793,  1.3936e3],
                                               [ 3.80490000e4, 4.34998980, -9.29704113e-1,  7.27279323e-1,  8.86109893e-8, 2.26808529,  4.29999968e2,  3.50287673,  1.3936e3]])
        ensemble_samples.update(tb_ensemble)
        print("working version = ",ensemble_samples)
        

    else:"""
    
    ensemble_samples = get_CV_ensemble(xdata, ydata, calc, w0, params,param_bounds,p_subset,shift_data=shift_data)

    
    
    ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)

    filename = "ensembles/"+model_name+"/"+model_name+"_CV_ensemble_p_"+str(p_subset)+".pkl"
    ensemble_dict = {"ensemble":ensemble_samples, "ypred_samples":ypred_samples,"ydata":ydata,"xdata":xdata}
    print("saving model")
    if os.path.exists(filename):
        subprocess.call("rm -rf "+filename,shell=True)
    with open(filename, 'wb') as file:
        pickle.dump(ensemble_dict, file)
