import numpy as np
from multiprocessing import Pool
import emcee
import os
import corner
import argparse
import pickle
import subprocess
from tqdm import tqdm

from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
from BLG_model_builder.BLG_model_library import *
from model_fit import *
import uuid
#import jax.numpy as np

def get_residual_error_hopping(ydata,ypred): 
    ydata_shift = 0 #np.mean(ydata) 
    ypred_scaled = np.nan_to_num((ypred-ydata_shift)) #np.std(ydata)
    ydata_scaled = np.nan_to_num((ydata-ydata_shift)) #np.std(ydata)
    return (ydata_scaled - ypred_scaled)

def get_residual_error_energy(ydata,ypred,tb_energy=None):
    shift_ind = np.argmin(ydata)  #choose point closest to zero
    #if np.argmin(ypred) != shift_ind:
    #    #return np.inf
    if tb_energy is not None:
        ypred = ypred + tb_energy
    ypred_shift = ypred[shift_ind] 
    ydata_shift = ydata[shift_ind] 
    ypred_scaled = np.nan_to_num((ypred-ypred_shift)) #np.std(ydata)
    ydata_scaled = np.nan_to_num((ydata-ydata_shift)) #np.std(ydata)

    """if np.random.rand()<0.05:
        print("MAE = ",np.mean(np.abs((ydata_scaled - ypred_scaled))))
        plt.scatter(ydata-ydata_shift,ypred-ypred_shift)
        plt.plot(ydata-ydata_shift,ydata-ydata_shift,c="black")
        plt.savefig("figures/mcmc_step.png")
        plt.clf()

        d_ = np.array([3,3.2,3.35,3.5,3.65,3.8,4,4.5,5,6,7])
        d_ = np.hstack((d_,d_,d_,d_))
        xdata,_,ydn = get_training_data("interlayer energy")

        deltaE=0.01
        plt.scatter(d_,(ydata-ydata[shift_ind]-deltaE)/100,label="qmc",color="black")
        plt.errorbar(d_,(ypred-ypred[shift_ind]-deltaE)/100,fmt="o",label="ypred")
        #plt.scatter(d_,(tb_energy-tb_energy[-1]-deltaE)/100,marker="x",label="TB energy",color="red")
        plt.xlabel("layer sep")
        plt.ylabel("Energy (eV/atom)")
        plt.legend()
        plt.savefig("figures/interlayer_pes_sample.png")
        plt.clf()"""
        
        
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
    #return -0.5 * np.sum(residual ** 2 / sigma2 + np.log(2*np.pi*sigma2))*weights
    #print("mean sigma^2 = ",np.mean(sigma2))
    #print("mean residual (eV) = ",np.linalg.norm(residual)/len(residual))
    filename = os.environ["FILENAME"]
    with open(filename, "a+") as f:
        f.write("PARAMS:\n")
        f.write((" ").join([str(x) for x in theta]))
        f.write("\nYPRED:\n")
        f.write((" ").join([str(y) for y in ypred]))
        f.write("\n")
    return -0.5 * np.sum(residual ** 2 / sigma2)*weights

def logprior_uniform(x: np.ndarray, bounds: np.ndarray) -> float:
    l_bounds, u_bounds = bounds
    if all(np.less(x, u_bounds)) and all(np.greater(x, l_bounds)):
        ret = 0.0
    else:
        ret = -np.inf
    return ret

def find_bounds(params, params_std, x, y, noise, T, model,weights,key):
    mle = log_likelihood(params, x, y, noise, T, model,weights,key)
    mle_mult = 10
    param_step_val = 0.2
    
    upper_bound = np.zeros(len(params))
    lower_bound = np.zeros(len(params))
    for i in range(len(params)):
        #find upper bound
        tmp_params = params.copy()
        lle= mle
        #if upper_bound < param_step_val + 0.01*param_step_val:
        #    param_step_val *= 0.1

        while (np.abs(lle) < np.abs(mle_mult * mle)) and (tmp_params[i] < params[i] + 5 * params_std[i]): 
            tmp_params[i] += param_step_val * params_std[i]
            lle = log_likelihood(tmp_params, x, y, noise, T, model,weights,key)
        upper_bound[i] = tmp_params[i] #- param_step_val * params_std[i]

        #find lower bound
        tmp_params = params.copy()
        lle= mle
        while (np.abs(lle) < np.abs(mle_mult * mle)) and (tmp_params[i] > (params[i] - 5 * params_std[i])): 
            tmp_params[i] -= param_step_val * params_std[i]
            lle = log_likelihood(tmp_params, x, y, noise, T, model,weights,key)
        lower_bound[i] = tmp_params[i] #- param_step_val * params_std[i]

    return (lower_bound,upper_bound)

def log_probability(theta, x, y, noise, T, model,weights, theta_best_fit, theta_std, bounds,tb_energy):
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

def get_MCMC_ensemble(x, y, noise, T, model, weights, theta_best_fit, theta_std, bounds, 
                        N_samples=75,thin=5,step_size=1e-3,use_EMCEE=True,burnin=5,tb_energy=None):
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

    :param theta_std: (dict) dict of best fit model parameters standard deviations with keys "energy" and/or "hoppings". 
                            used to define prior distributions. dict entries must be arrays

    :returns: (dict) Ensemble samples with keys "energy" and/or "hoppings".
     """
    #os.environ["OMP_NUM_THREADS"] = "1"
    nwalkers = 0
    ndim = {}
    for key in model:
        #bounds[key] = find_bounds(theta_best_fit[key], theta_std[key], x[key], y[key], noise[key], T[key], model[key],weights[key],key)
        
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
            print(np.mean(theta_walkers,axis=0))
            
        else:
            scale = 1e-5 * np.abs(theta_best_fit[key])
            theta_walkers = np.append(theta_walkers,np.random.normal(loc=theta_best_fit[key],scale=scale,size=(nwalkers,ndim[key])),axis=1)

    if use_EMCEE:
        move = emcee.moves.StretchMove(a=step_size)
        #move = emcee.moves.GaussianMove(0.1)
        
        nsteps = int(2*thin*N_samples/nd)+burnin
        print("running ",nsteps)
        #with Pool() as pool:

        sampler = emcee.EnsembleSampler(
                nwalkers, np.shape(theta_walkers)[1], log_probability, 
                args=(x, y, noise, T, model, weights, theta_best_fit, theta_std, bounds,tb_energy),
                moves=move) #, pool=pool) #i think pool is giving pretty bad performance on the cluster
        sampler.run_mcmc(theta_walkers, nsteps, progress=False) #, thin=thin)

        acceptance_fraction = sampler.acceptance_fraction
        print("Mean acceptance fraction: {:.8f}".format(np.mean(sampler.acceptance_fraction)))
        samples = sampler.get_chain( flat=True)
        samples = samples[burnin:,:]
    else:
        log_prob_array, samples,acceptance_fraction = run_MCMC_sampler(x, y, noise, T, model, weights, theta_best_fit, theta_std, bounds,step_size,int(N_samples*thin)+burnin)
        samples = samples[burnin:,:]
        samples = samples[::thin,:]

    print("Shape of ensemble = ",np.shape(samples))
    sample_dict = {}
    theta_ind = 0
    for key in model:        
        sample_dict[key] = samples[:,theta_ind:theta_ind + len(theta_best_fit[key])]
        theta_ind += len(theta_best_fit[key])
    return sample_dict, np.mean(acceptance_fraction)

def run_MCMC_sampler(x, y, noise, T, model, weights, theta_best_fit, theta_std, bounds,step_size,N_samples):
    for it, key in enumerate(model):
        #just start walkers around max. liklihood parameters to have shorter burnin time
        if it ==0:
            theta_rand = np.random.normal(scale = step_size, size = (N_samples,len(theta_best_fit[key]))) * theta_std[key][np.newaxis,:]
            theta = theta_best_fit[key]
        else:
            theta_rand = np.append(theta_rand,np.random.normal(scale = step_size, size = (N_samples,len(theta_best_fit[key]))) * theta_std[key][np.newaxis,:],axis=1)
            theta = np.append(theta,theta_best_fit[key],axis=0)

    threshold_rand = np.random.rand(N_samples)
    current_log_prob = log_probability(theta, x, y, noise, T, model, weights, theta_best_fit, theta_std, bounds)
    chain_vals = []
    log_prob_array = []
    y_pred_samples = []
    accepted = 0
    for n in tqdm(range(N_samples)):
        trial_step_theta = theta + theta_rand[n,:]
        log_prob = log_probability(trial_step_theta,x, y, noise, T, model, weights, theta_best_fit, theta_std, bounds)
        proposal_prob = log_prob - current_log_prob
        if threshold_rand[n] < np.exp(proposal_prob):
            theta = trial_step_theta.copy()
            current_log_prob = log_prob.copy()
            accepted +=1
        log_prob_array.append(log_prob)
        chain_vals.append(trial_step_theta)
        #y_pred_samples.append(y_pred)
    print("acceptance fraction = ",accepted/N_samples)
    return np.array(log_prob_array), np.array(chain_vals), accepted/N_samples 

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
        selected = jax.random.choice(key, indices, shape=(n_select,), replace=False)
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

def get_ks_metric(y_pred_samples_dict,ydata_dict,w0):
    ks_metric = 0
    ci_array = np.linspace(0.,0.95,15)
    for key in ydata_dict:
        y_pred_samples = y_pred_samples_dict[key]
        ydata = ydata_dict[key]
        shift_ind = np.argmin(ydata)  #choose point closest to zero
        ypred_samples_shift = y_pred_samples[:,shift_ind] 
        ydata_shift = ydata[shift_ind] 
        y_pred_samples = (y_pred_samples-ypred_samples_shift[:,np.newaxis])/np.std(ydata)
        ydata = (ydata-ydata_shift)/np.std(ydata)

        percent_within_bounds = np.zeros_like(ci_array)
        for index,ci in enumerate(ci_array):
            lower_bound = np.quantile(y_pred_samples,(1-ci)/2,axis=0)
            upper_bound = np.quantile(y_pred_samples,1-(1-ci)/2,axis=0)
            within_ind = (ydata >= lower_bound[np.newaxis,:]) & (ydata <= upper_bound[np.newaxis,:])
            percent_within_bounds[index] = sum(sum(within_ind))/len(ydata) #counting number of true elements, divided by total num elements

        ks_metric += w0[key] * np.trapz(2*(percent_within_bounds - ci_array),ci_array)
    print("KS METRIC = " ,ks_metric)
    return ks_metric, ci_array, percent_within_bounds

def get_coverage(y_pred_samples_dict,ydata_dict,w0,ci):
    coverage = 0
    for key in ydata_dict:
        y_pred_samples = y_pred_samples_dict[key]
        ydata = ydata_dict[key]

        shift_ind = np.argmin(ydata)  #choose point closest to zero
        ypred_samples_shift = y_pred_samples[:,shift_ind] 
        ydata_shift = ydata[shift_ind] 
        y_pred_samples = (y_pred_samples-ypred_samples_shift[:,np.newaxis])/np.std(ydata)
        ydata = (ydata-ydata_shift)/np.std(ydata)


        lower_bound = np.quantile(y_pred_samples,(1-ci)/2,axis=0)
        upper_bound = np.quantile(y_pred_samples,1-(1-ci)/2,axis=0)
        
        within_ind = (ydata >= lower_bound[np.newaxis,:]) & (ydata <= upper_bound[np.newaxis,:])
        percent_within_bounds = sum(sum(within_ind))/len(ydata) #counting number of true elements, divided by total num elements

        coverage += w0[key] * percent_within_bounds
    print("coverage for ci ("+str(ci)+") = " ,coverage)
    return coverage


def get_uncertainty_correlation(y_pred_samples_dict,ydata_dict,w0,n_bins = 10):
    metric = 0
    ci = 1-(0.64/2)
    for key in ydata_dict:
        if key=="energy":
            min_ind = np.argmin(ydata_dict[key])
            ypred_mean = np.mean(y_pred_samples_dict[key] - y_pred_samples_dict[key][:,min_ind][:,np.newaxis] ,axis=0)
            #ypred_std = np.std(y_pred_samples_dict[key],axis=0)
            lower_bound = np.quantile(y_pred_samples_dict[key]- y_pred_samples_dict[key][:,min_ind][:,np.newaxis],(1-ci)/2,axis=0)
            upper_bound = np.quantile(y_pred_samples_dict[key]- y_pred_samples_dict[key][:,min_ind][:,np.newaxis],1-(1-ci)/2,axis=0)
            ypred_std = upper_bound - lower_bound
            residual_error = np.sqrt(np.power(((ypred_mean - ypred_mean[min_ind])) - (ydata_dict[key] - ydata_dict[key][min_ind]),2))
        else:
            ypred_mean = np.mean(y_pred_samples_dict[key]  ,axis=0)
            #ypred_std = np.std(y_pred_samples_dict[key],axis=0)
            lower_bound = np.quantile(y_pred_samples_dict[key],(1-ci)/2,axis=0)
            upper_bound = np.quantile(y_pred_samples_dict[key],1-(1-ci)/2,axis=0)
            ypred_std = upper_bound - lower_bound
            residual_error = np.sqrt(np.power((ypred_mean ) - ydata_dict[key] ,2))
        
        
        #R^2 test
        metric += w0[key] * np.linalg.norm(ypred_std - residual_error) #units are in eV or whatever ypred units are
    print("uq error metric = ",metric)
    return metric

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--interaction_type',type=str,default="interlayer")
    parser.add_argument('-e','--energy_model',type=str,default='Classical')
    parser.add_argument('-t','--tb_model',type=str,default="None")
    parser.add_argument('-n','--nn_val',type=str,default="1")
    parser.add_argument('-B','--beta',type=str,default="1")
    #parser.add_argument('-T',"--temperature_weight",type=str,default='1.0')
    args = parser.parse_args() 
    #works for MK, letb interlayer, letb intralayer nn 1, classical interlayer energy, letb intralayer nn 2, letb intralayer nn 3
    #check  classical intralayer energy, tetb intralayer, tetb interlayer
    ################## Adjustable, set model ##########################################
    seed = 135726
    #np.random.seed(seed)
    use_EMCEE = True
    from time import time
    int_type = args.interaction_type
    energy_model = args.energy_model
    tb_model = args.tb_model
    nn_val = int(args.nn_val)
    print("int type = ",int_type)
    print("energy model = ",str(energy_model))
    print("tb_model = ",str(tb_model))

    if energy_model =="None": energy_model=None
    if tb_model =="None": tb_model = None
    ################## Adjustable, set MCMC hyperparameters ############################
    Temperature_weight = float(args.beta)
    Prior_weight = 1.0
    relative_weight = 1.0
    test_size = 0.4
    ####################################################################################
    #define model name
    model_name = str(energy_model)+"_energy_"+str(int_type)+"_"+str(tb_model)
    model_name = model_name.replace("full_","")
    model_name = model_name.replace("None_energy_","")
    model_name = model_name.replace("_None","")
    if model_name =="intralayer_LETB":
        model_name = model_name + "_NN_val_"+str(nn_val)

    calc,xdata,ydata,ydata_noise, params,params_std,param_bounds, ypred_bestfit = get_MCMC_inputs(int_type,energy_model,tb_model,model_name,nn_val)
    #xdata_train, xdata_test, ydata_train, ydata_test,ydata_noise_train,ydata_noise_test = train_test_split(xdata, ydata, ydata_noise, test_size=test_size)

    if not os.path.exists("ensembles/"+model_name+"/"):
        os.mkdir("ensembles/"+model_name+"/")
    try:
        os.remove("ensembles/"+model_name+"/"+model_name+"_ensemble_T_"+str(Temperature_weight)+".txt")
    except:
        dummy=0
    os.environ["FILENAME"] = "ensembles/"+model_name+"/"+model_name+"_ensemble_T_"+str(Temperature_weight)+".txt"
    Temperature = {}
    w0={}
    #relative_weight = 1/len(calc.keys())
    relative_weight = 0.85
    for nm,key in enumerate(calc):
    
        if energy_model=="TETB" and key == "energy" and tb_model=="popov":
            tb_calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="python")
            if int_type=="interlayer":
                tb_xdata,_,_ = get_training_data(int_type+" energy",supercells=5) 
            else:
                tb_xdata = xdata
            tb_energy,_ = tb_calc.get_tb_energy(tb_xdata[key][0])
            for x in tb_xdata[key][1:]:
                tbe,_ = tb_calc.get_tb_energy(x)
                tb_energy = np.append(tb_energy,tbe)
            
            #if int_type=="interlayer":
            #    #tb_energy *= len(xdata[key][0])/len(tb_xdata[key][0])
            #    tb_energy -= tb_energy[-1]

        print("getting residual error")
        residual = get_residual_error(ydata[key],ypred_bestfit[key],key,np.zeros_like(ydata[key])) 
        print("MAE (eV)= ",np.linalg.norm(residual)/len(residual))
        Temperature[key] = Temperature_weight * np.sum((residual)**2) * (2/len(params[key])) 
        w0[key] = relative_weight **((nm+1)%2) * (1-relative_weight)**(nm%2)

    """if energy_model=="TETB":
        #use_params = np.append(params["hoppings"],params["energy"])
        xaxis_data = []
        y_pred = []
        tb_energy = []
        residual_energy = []

        tetb_calc =  get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="lammps")
        
        if int_type=="interlayer":
            tetb_calc.model_dict["interlayer"]["potential parameters"] = params["energy"] #np.array([3.43845,34.0449,-17.1697,17.2296,-23.0449,3.07926,-1.54847,10.784,-7.1459]) #
            tb_xdata,_,_ = get_training_data(int_type+" energy",supercells=5) 
        else:
            tetb_calc.model_dict["intralayer"]["potential parameters"] = params["energy"]
            tb_xdata = xdata 
        #for x in xdata["energy"]:
        for x in tb_xdata["energy"]:
            #tetb_calc.set_params(use_params)
            tbe,_ = tetb_calc.get_tb_energy(x)
            tb_energy.append(tbe)
            #er,_ = tetb_calc.get_residual_energy(x)
            #residual_energy.append(er)
            
            yval,_ = tetb_calc.get_total_energy(x) #calc["energy"](x,params["energy"]) #
            y_pred.append(yval)

            pos = x.positions
            if int_type=="intralayer":
                distances = cdist(pos, pos)
                np.fill_diagonal(distances, np.inf)
                min_distances = np.min(distances, axis=1)
                average_distance = np.mean(min_distances)
                xaxis_data.append(average_distance)
            elif int_type=="interlayer":
                mean_z = np.mean(pos[:,2])
                top_ind = np.where(pos[:,2]>mean_z)
                xaxis_data.append(2*np.mean(pos[top_ind,2]-mean_z))
        ydata_full = ydata["energy"] #+ np.array(tb_energy)
        min_ind = np.argmin((ydata_full))
        #plt.scatter(xaxis_data,Eshift,label="tb energy")
        tb_energy -= tb_energy[-1]
        
        plt.scatter(xaxis_data,(ydata_full-ydata_full[min_ind])/len(x),label="ydata")
        #plt.scatter(xaxis_data,np.array(tb_energy)/len(x),label="tb energy")
        
        #composite_energy = np.array(tb_energy) + np.array(residual_energy)
        #plt.scatter(xaxis_data,composite_energy - composite_energy[min_ind],label="composite energy")
        #plt.scatter(xaxis_data,np.array(residual_energy)-residual_energy[min_ind],label="residual energy")
        plt.scatter(xaxis_data,(np.array(y_pred)-np.array(y_pred)[min_ind])/len(x),label="ypred")
        plt.legend()
        plt.savefig("figures/test_"+int_type+"_fit.png")
        plt.clf()
        exit()"""

    #if not use_EMCEE:
    if not os.path.exists("step_sizes/"+model_name+"_ensemble.pkl"):
        print("finding proper step size")
        acceptance_fraction = 0
        step_size = 1
        while acceptance_fraction < 0.2 or acceptance_fraction > 0.5:
            if acceptance_fraction > 0.5:
                step_size *= 10
            elif acceptance_fraction < 0.2:
                step_size /= 5

            print("step size = ",step_size)
            ensemble_samples,acceptance_fraction = get_MCMC_ensemble(xdata, ydata, 
                                        ydata_noise, Temperature, calc, w0, 
                                        params,params_std,param_bounds,N_samples=200,thin=1,burnin=100,
                                        step_size=step_size,use_EMCEE=use_EMCEE,tb_energy=tb_energy)
        dict = {"step_size":step_size,"acceptance_fraction":acceptance_fraction}
        with open("step_sizes/"+model_name+"_ensemble.pkl", 'wb') as file:
            pickle.dump(dict, file)
    else:
        filename = "step_sizes/"+model_name+"_ensemble.pkl"
        with open(filename, 'rb') as file:
            dict = pickle.load(file)
        step_size = dict["step_size"]
    start = time()

    ensemble_samples,acceptance_fraction = get_MCMC_ensemble(xdata, ydata, 
                                    ydata_noise, Temperature, calc, w0, 
                                    params,params_std,param_bounds,step_size=step_size,use_EMCEE=use_EMCEE,tb_energy=tb_energy)

    """else:
        ensemble_samples,_ = get_MCMC_ensemble(xdata, ydata, 
                                        ydata_noise, Temperature, calc, w0, 
                                        params,params_std,param_bounds)"""
    
    for key in calc:
        print("ensemble mean "+key,np.mean(ensemble_samples[key],axis=0))
        print("ensemble std  "+key,np.std(ensemble_samples[key],axis=0))
    print("getting ypred samples")
    #try:
    ypred_samples,_ = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
    for key in calc:
        print(key," shape ", np.shape(ypred_samples[key]))
    #except:
    #    ypred_samples = []

    #filename = "ensembles/"+model_name+"/"+model_name+"_ensemble_T_"+str(Temperature_weight)+"_"+str(uuid.uuid4())+".pkl"
    end =  time()
    print("total time = ",end-start)
    filename = "ensembles/"+model_name+"/"+model_name+"_ensemble_T_"+str(Temperature_weight)+".pkl"
    ensemble_dict = {"ensemble":ensemble_samples, "ypred_samples":ypred_samples,"ydata":ydata,"xdata":xdata}
    print("saving model")
    if os.path.exists(filename):
        subprocess.call("rm -rf "+filename,shell=True)
    with open(filename, 'wb') as file:
        pickle.dump(ensemble_dict, file)

    run_all_temps = False
    if run_all_temps:
        T_weight_array = np.array([0.01,0.1,0.2,0.5,0.75,1,1.5,2.0,3,4,5,10,50,100,500,1000])
        for i,Tw in enumerate(T_weight_array):
            Temperature = {}
            w0={}
            relative_weight = 1/len(calc.keys())
            for nm,key in enumerate(calc):
                
                residual = get_residual_error(ydata[key],ypred_bestfit[key],key) 
                print("MAE (eV)= ",np.linalg.norm(residual)/len(residual))
                Temperature[key] = Temperature_weight * np.sum((residual)**2) * (2/len(params[key])) 
                w0[key] = relative_weight **((nm+1)%2) * (1-relative_weight)**(nm%2)
            print("Tw = ",Tw)
            ensemble_samples = get_MCMC_ensemble(xdata_train, ydata_train, 
                                                ydata_noise_train, Temperature, calc, w0, 
                                                params,params_std,param_bounds)
            
            print("ensemble mean = ",np.mean(ensemble_samples[key],axis=0))
            print("ensemble std = ",np.std(ensemble_samples[key],axis=0))

            ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata_test, ydata_test, calc)
            
            filename = "ensembles/"+model_name+"/"+model_name+"_ensemble_T_"+str(Tw)+".pkl"
            ensemble_dict = {"ensemble":ensemble_samples, "ypred_samples":ypred_samples,"ydata_test":ydata_test,"xdata_test":xdata_test}
            if os.path.exists(filename):
                subprocess.call("rm -rf "+filename,shell=True)
            with open(filename, 'wb') as file:
                pickle.dump(ensemble_dict, file)


