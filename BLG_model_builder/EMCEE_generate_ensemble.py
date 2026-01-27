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


def get_residual_error_hopping(ydata,ypred): 
    return (ydata - ypred)

def get_residual_error_energy(ydata,ypred,tb_energy=None):
    shift_ind = np.argmin(ydata) 
    if tb_energy is not None:
        ypred = ypred + tb_energy
    ypred_shift = ypred[shift_ind] 
    ydata_shift = ydata[shift_ind] 
    ypred_scaled = np.nan_to_num((ypred-ypred_shift))
    ydata_scaled = np.nan_to_num((ydata-ydata_shift))
        
    return (ydata_scaled - ypred_scaled)

def get_residual_error(ydata,ypred,key,tb_energy):
    if key =="energy":
        return get_residual_error_energy(ydata,ypred,tb_energy=tb_energy)
    elif key=="hoppings":
        return get_residual_error_hopping(ydata,ypred)

def log_likelihood(theta, x, y, noise, T, model,weights,key,tb_energy): 
    
    ypred = get_prediction(model,x,theta)
    if np.isnan(ypred).any() or np.isinf(ypred).any():
        return -np.inf
    residual = get_residual_error(y, ypred,key,tb_energy)
    sigma2 = T**2 + noise**2
    return -0.5 * np.sum(residual ** 2 / sigma2)*weights

def logprior_uniform(x: np.ndarray, bounds: np.ndarray) -> float:
    l_bounds, u_bounds = bounds
    if all(np.less(x, u_bounds)) and all(np.greater(x, l_bounds)):
        ret = 0.0
    else:
        ret = -np.inf
    return ret

def log_probability(theta, x, y, noise, T, model,weights, theta_best_fit, bounds,tb_energy):
    log_prob = 0
    theta_ind = 0
    if "hoppings" in model.keys() and "energy" in model.keys():
        use_TETB=True
    else:
        use_TETB=False
    for key in model:
        #theta is an array, so need to split it into individual parts of model
        if use_TETB and key=="energy":
            partial_theta = theta.copy()
            low_bound = np.append(bounds["hoppings"][:,0].copy(),bounds["energy"][:,0].copy())
            up_bound = np.append(bounds["hoppings"][:,1].copy(),bounds["energy"][:,1].copy())
        else:
            partial_theta = theta[theta_ind:theta_ind + len(theta_best_fit[key])]
            low_bound = bounds[key][:,0]
            up_bound = bounds[key][:,1]
        theta_ind += len(theta_best_fit[key])
        lpu = logprior_uniform(partial_theta,(low_bound,up_bound))
        lle = log_likelihood(partial_theta, x[key], y[key], noise[key], T[key], model[key], weights[key],key,tb_energy)
        log_prob +=  lpu + lle 
    
    return log_prob

def get_MCMC_ensemble(x, y, noise, T, model, weights, theta_best_fit, bounds, 
                        N_samples=700,step_size=1e-3,tb_energy=None):
    """Generate Markov Chain Monte Carlo ensemble, given x,y data, 
    aleatoric noise, sampling temperature, cost fxn weights, best fit parameters and standard deviations. 
    works for multi-part cost functions i.e. Cost(hoppings) + Cost(energy)

    :param x: (dict) dict of xdata with keys "energy" and/or "hoppings". dict entries must be arrays

    :param y: (dict) dict of ydata with keys "energy" and/or "hoppings". dict entries must be arrays

    :param noise: (dict) dict of ydata aleatoric noise with keys "energy" and/or "hoppings". dict entries must be arrays

    :param T: (dict) dict of sampling Temperature weight with keys "energy" and/or "hoppings". Sampling Temp = T * T0, dict entries must be floats

    :param model: (dict) dict of fxns with keys "energy" and/or "hoppings". dict entries must be functions

    :param weights: (dict) dict of cost fxn weights with keys "energy" and/or "hoppings". dict entries must be floats

    :param theta_best_fit: (dict) dict of best fit model parameters with keys "energy" and/or "hoppings". dict entries must be arrays

    :param bounds: (dict) dict of bounds for the model parameters with keys "energy" and/or "hoppings". dict entries must be arrays

    :returns: (dict) Ensemble samples with keys "energy" and/or "hoppings".
     """
    #os.environ["OMP_NUM_THREADS"] = "1"
    nwalkers = 0
    ndim = {}
    for key in model:

        nw = int(2*len(theta_best_fit[key]))
        nd = len(theta_best_fit[key])
        nwalkers += nw
        os.environ["OMP_NUM_THREADS"] = str(nwalkers)
        ndim[key]= nd

    for it, key in enumerate(model):
        #just start walkers around max. liklihood parameters to have shorter burnin time
        if it ==0:
            scale = 1e-5 * np.abs(theta_best_fit[key])
            theta_walkers = np.random.normal(loc=theta_best_fit[key],scale=scale,size=(nwalkers,ndim[key]))
            print("starting walkers")
            
        else:
            scale = 1e-5 * np.abs(theta_best_fit[key])
            theta_walkers = np.append(theta_walkers,np.random.normal(loc=theta_best_fit[key],scale=scale,size=(nwalkers,ndim[key])),axis=1)

    move = emcee.moves.StretchMove(a=step_size)
    #move = emcee.moves.GaussianMove(0.1)
    
    nsteps = N_samples
    print("running ",nsteps)
    #with Pool() as pool:

    sampler = emcee.EnsembleSampler(
            nwalkers, np.shape(theta_walkers)[1], log_probability, 
            args=(x, y, noise, T, model, weights, theta_best_fit, bounds,tb_energy),
            moves=move) #, pool=pool) #i think pool is giving pretty bad performance on the cluster
    sampler.run_mcmc(theta_walkers, nsteps, progress=True)

    acceptance_fraction = sampler.acceptance_fraction
    print("Mean acceptance fraction: {:.8f}".format(np.mean(sampler.acceptance_fraction)))
    samples = sampler.get_chain( flat=True)
    samples = samples[::50]

    print("Shape of ensemble = ",np.shape(samples))
    sample_dict = {}
    theta_ind = 0
    for key in model:        
        sample_dict[key] = samples[:,theta_ind:theta_ind + len(theta_best_fit[key])]
        theta_ind += len(theta_best_fit[key])
    return sample_dict, np.mean(acceptance_fraction)

def train_test_split(xdata,ydata,ydata_noise,test_size=0.4):
    xdata_train = {}
    xdata_test = {}
    ydata_train = {}
    ydata_test = {}
    ydata_noise_train = {}
    ydata_noise_test = {}
    
    for key in ydata:
        indices = np.arange(len(ydata[key]))
        n_select = int(len(indices) * test_size) 
        selected = np.random.choice(indices, size=n_select, replace=False)
        not_selected = np.setdiff1d(indices, selected)

        if type(xdata[key])==list:
            xdata_train[key] = [xdata[key][ns] for ns in not_selected] 
            xdata_test[key] = [xdata[key][s] for s in selected]
        else:
            xdata_train[key] =  xdata[key][not_selected]
            xdata_test[key] = xdata[key][selected]
        ydata_train[key] = ydata[key][not_selected]
        ydata_test[key] = ydata[key][selected]
        ydata_noise_train[key] = ydata_noise[key][not_selected]
        ydata_noise_test[key] = ydata_noise[key][selected]
    return xdata_train, xdata_test, ydata_train, ydata_test, ydata_noise_train, ydata_noise_test

if __name__=="__main__":
    from time import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name',type=str,default="MK")
    parser.add_argument('-B','--beta',type=str,default="1")
    args = parser.parse_args() 
    ################## Adjustable, set model ##########################################
    seed = 135726
    #np.random.seed(seed)
    model_name = args.model_name
    Temperature_weight = float(args.beta)
    print("running MCMC for ",model_name)
    ################## Adjustable, set MCMC hyperparameters ############################
    Prior_weight = 1.0
    relative_weight = 1.0
    test_size = 0.4
    ####################################################################################

    calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name)

    #xdata_train, xdata_test, ydata_train, ydata_test,ydata_noise_train,ydata_noise_test = train_test_split(xdata, ydata, ydata_noise, test_size=test_size)

    if not os.path.exists("ensembles/"+model_name+"/"):
        os.mkdir("ensembles/"+model_name+"/")
    Temperature = {}
    w0={}
    tb_energy = 0
    relative_weight = 0.85
    for nm,key in enumerate(calc):
    
        if "TETB" in model_name and key == "energy":
            tb_energy,_ = calc[key].get_tb_energy(xdata[key][0])
            for x in xdata[key][1:]:
                tbe,_ = calc[key].get_tb_energy(x)
                tb_energy = np.append(tb_energy,tbe)

        print("getting residual error")
        residual = get_residual_error(ydata[key],ypred_bestfit[key],key,np.zeros_like(ydata[key])) 
        print("MAE (eV)= ",np.mean(np.abs(residual)))
        Temperature[key] = Temperature_weight * np.sum((residual)**2) * (2/len(params[key])) 
        w0[key] = relative_weight **((nm+1)%2) * (1-relative_weight)**(nm%2)

    step_size = 0.01
    start = time()

    ensemble_samples,acceptance_fraction = get_MCMC_ensemble(xdata, ydata, 
                                    ydata_noise, Temperature, calc, w0, 
                                    params,bounds,step_size=step_size,tb_energy=tb_energy)

    print("getting ypred samples")
    
    ypred_samples,_ = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
    for key in calc:
        print(key," shape ", np.shape(ypred_samples[key]))

    end =  time()
    print("total time = ",end-start)
    filename = "ensembles/"+model_name+"/"+model_name+"_ensemble_T_"+str(Temperature_weight)+".pkl"
    ensemble_dict = {"ensemble":ensemble_samples, "ypred_samples":ypred_samples,"ydata":ydata,"xdata":xdata}
    print("saving model")
    if os.path.exists(filename):
        subprocess.call("rm -rf "+filename,shell=True)
    with open(filename, 'wb') as file:
        pickle.dump(ensemble_dict, file)
