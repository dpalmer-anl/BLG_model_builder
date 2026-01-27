import numpy as np
from multiprocessing import Pool
import emcee
import os
import corner
import argparse
import pickle
import subprocess
from tqdm import tqdm
import uuid

from model_fit import *
from get_MCMC_inputs import *

def get_SubSamp_ensemble(xdata, ydata, calc, p0,bounds,p_subset,nsamples = 50,fit_mlp=False,input_dim=1,output_dim=2,hidden_dim=15):
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
            if fit_mlp:
                mlp_model = MLP(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim)
                best_fit_params,_ = fit_torch_mlp(calc[key],mlp_model,xdata_train,ydata_train,ypred_shift=None,num_epochs=500)
            else:
                best_fit_params,_ = fit_model(calc[key],xdata_train,ydata_train,p0[key],tb_energy = Eshift,shift_data=False,bounds=bounds[key])
            potential_samples.append(best_fit_params)
        ensemble[key] = np.array(potential_samples)
    return ensemble
 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name',type=str,default="MLP_SK")
    parser.add_argument('-p','--p_subset',type=str,default="0.5")
    args = parser.parse_args() 
    ################## Adjustable, set model ##########################################
    seed = 135726
    #np.random.seed(seed)

    model_name = args.model_name
    p_subset = float(args.p_subset)
    calc,xdata,ydata,ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
    if model_name=="MLP_SK":
        fit_mlp = True
        input_dim = 1
        output_dim = 2
        hidden_dim = 15
        calc = {"hoppings": MLP_SK_hoppings_torch}

    elif model_name=="Interlayer_MLP":
        fit_mlp = True
        input_dim = 2
        output_dim = 2
        hidden_dim = 15
        calc = {"energy": Interlayer_MLP_torch}
    else:
        fit_mlp = False
        input_dim = None
        output_dim = None
        hidden_dim = None
    print("model name = ",model_name)

    ensemble_samples = get_SubSamp_ensemble(xdata, ydata, calc, params,bounds,p_subset,
                                            fit_mlp=fit_mlp,input_dim=input_dim,output_dim=output_dim,hidden_dim=hidden_dim)

    calc,xdata,ydata,ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")                      
    ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)

    filename = "ensembles/"+model_name+"/"+model_name+"_SubSamp_ensemble_p_"+str(p_subset)+".pkl"
    ensemble_dict = {"ensemble":ensemble_samples, "ypred_samples":ypred_samples,"ydata":ydata,"xdata":xdata}
    print("saving model")
    if os.path.exists(filename):
        subprocess.call("rm -rf "+filename,shell=True)
    with open(filename, 'wb') as file:
        pickle.dump(ensemble_dict, file)
