import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
import time
import pickle
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from BLG_model_builder.MLP import *
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
from BLG_model_builder.BLG_model_library import *

def fit_torch_mlp(function,model,xdata,ydata,ypred_shift=None,num_epochs=1000, learning_rate=0.001):
    x, y = torch.tensor(xdata,dtype=torch.float32), torch.tensor(ydata,dtype=torch.float32)
    if ypred_shift is None:
        ypred_shift = torch.zeros_like(y,dtype=torch.float32)
    else:
        ypred_shift = torch.tensor(ypred_shift,dtype=torch.float32)
    dataset = TensorDataset(x, y, ypred_shift)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    criterion = nn.MSELoss()  # Mean squared error for hopping values
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining MLP with {sum(p.numel() for p in model.parameters())} parameters...")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch, ypred_shift_batch in train_loader:
            # Forward pass: some function of the MLP and the input data
            pred = function(x_batch, model) + ypred_shift_batch
            
            # Compute loss on hopping values
            loss = criterion(pred, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

    ypred_bestfit = function(x, model) + ypred_shift
    flat_list = []
    params = dict(model.named_parameters())
    for i in [0, 2, 4]:
        weight_key = f'mlp.{i}.weight'
        bias_key = f'mlp.{i}.bias'
        w = params[weight_key].detach().cpu().numpy().T
        flat_list.append(w.ravel())
        b = params[bias_key].detach().cpu().numpy().ravel()
        flat_list.append(b)
        
    # Combine all into one 1D array
    all_params_numpy = np.concatenate(flat_list)
    return all_params_numpy, ypred_bestfit.detach().numpy()

def fit_model(method,xdata,ydata,p0,tb_energy=0,bounds=None,shift_data=False,minimizer="differential_evolution",**kwargs):
    ym = get_mean(ydata)
    ys = get_std(ydata)
    loss_fxn = get_loss_fxn(method,xdata,ydata, ym,ys,tb_energy,shift_data=shift_data)
    if minimizer=="differential_evolution":
        result = scipy.optimize.differential_evolution(loss_fxn,bounds,strategy="randtobest1bin")
    elif minimizer=="SLSQP":
        result = scipy.optimize.minimize(loss_fxn,p0,method="SLSQP",bounds=bounds)
    elif minimizer=="Nelder-Mead":
        result = scipy.optimize.minimize(loss_fxn,p0,method="Nelder-Mead",bounds=bounds)
    elif minimizer=="L-BFGS-B":
        result = scipy.optimize.minimize(loss_fxn,p0,method="L-BFGS-B",bounds=bounds)
    else:
        raise ValueError("Invalid minimizer: "+minimizer)
    popt = result.x
    ypred_bestfit = get_prediction(method,xdata,popt) + tb_energy
    #pcov = result.hess_inv.todense()
    return np.array(popt), ypred_bestfit

def get_loss_fxn(method,xdata,ydata, ym,ys,Eshift,shift_data=False):
    def func(params):
        ypred = get_prediction(method,xdata,params) + Eshift
        shift_ind = np.argmin(ydata) 
        ypred_shift = ypred[shift_ind] 
        ydata_shift = ydata[shift_ind] 
        if shift_data:
            ypred_scaled = np.nan_to_num((ypred-ypred_shift)/(ydata))
            ydata_scaled = np.nan_to_num((ydata-ydata_shift)/(ydata))
        else:
            ypred_scaled =  ypred
            ydata_scaled = ydata

        if type(ydata_scaled)==list:
            loss = 0
            for i in range(len(xdata)):
                loss += np.linalg.norm(ypred_scaled[i] - ydata_scaled[i])
            return loss
        elif type(ydata_scaled)==np.ndarray:
            return np.linalg.norm(ydata_scaled - ypred_scaled)
    return func

def get_prediction(method,xdata,params):
    if type(xdata)==list:
        y_pred = method(xdata[0],params)
        for x in xdata[1:]:
            yval = method(x,params)
            y_pred = np.append(y_pred,yval)
    elif type(xdata)==np.ndarray:
        y_pred = method(xdata,params)
    return y_pred

def worker(args):
    theta, model,x = args
    return get_prediction(model, x, theta)

def evaluate_ensemble(ensemble_samples,x, y, model):
    ypred_samples = {}
    clean_ensemble_samples = {}
    if "hoppings" in model.keys() and "energy" in model.keys():
        use_TETB=True
    else:
        use_TETB=False
    for key in ensemble_samples:
        if use_TETB and key=="energy":
            theta = np.hstack((ensemble_samples["hoppings"],ensemble_samples["energy"]))
            ypred_samples_list = [worker(theta[n,:], model[key], x[key]) for n in range(np.shape(theta)[0])]
        else:
            theta = ensemble_samples[key]
            ypred_samples_list = [worker((theta[n,:], model[key], x[key])) for n in range(np.shape(theta)[0])]
        ypreds = np.vstack(np.squeeze(ypred_samples_list))
        nan_ind = np.isnan(ypreds).any(axis=1)
        ypreds = ypreds[~nan_ind]
        ypred_samples[key] = ypreds
        clean_ensemble_samples[key] = ensemble_samples[key][~nan_ind]
        print("shape of cleaned "+key+" ensemble = ", np.shape(clean_ensemble_samples[key]))
    return ypred_samples, clean_ensemble_samples


if __name__=="__main__":
    int_type = "full"
    energy_model = None
    tb_model = "MLP_tb"
    model_name = str(energy_model)+"_energy_"+str(int_type)+"_"+str(tb_model)
    model_name = model_name.replace("full_","")
    model_name = model_name.replace("None_energy_","")
    model_name = model_name.replace("_None","")
    
    calc,xdata,ydata,ydata_noise, params,params_std,bounds,ypred_bestfit = get_MCMC_inputs(int_type=int_type,energy_model=energy_model,tb_model=tb_model,model_name=model_name)
    r = np.linalg.norm(xdata["hoppings"],axis=1)
    plt.scatter(r,ydata["hoppings"])
    plt.scatter(r,ypred_bestfit["hoppings"])
    plt.show()