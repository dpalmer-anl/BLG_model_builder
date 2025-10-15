import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import os
from scipy.spatial.distance import cdist
from EMCEE_generate_ensemble import get_MCMC_inputs, train_test_split, get_residual_error, get_ks_metric, get_uncertainty_correlation
from model_fit import *

def read_params_and_predictions(filename):
    """
    Reads a text file with blocks of:
    PARAMS:
    <numbers>
    YPRED:
    <numbers>
    
    Returns:
        params_array (np.ndarray): shape (n_steps, n_params)
        ypred_array (np.ndarray): shape (n_steps, n_preds)
    """
    params = []
    ypreds = []
    
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        if lines[i].startswith("PARAMS:"):
            # Next line should have params
            i += 1
            params_line = lines[i].split()
            params.append([float(x) for x in params_line])
        
        elif lines[i].startswith("YPRED:"):
            # Next line should have ypreds
            i += 1
            ypred_line = lines[i].split()
            ypreds.append([float(x) for x in ypred_line])
        
        i += 1
    
    return np.array(params), np.array(ypreds)

if __name__=="__main__":
    model_tuple = [("interlayer","Classical",None),("intralayer","Classical",None), ("interlayer",None,"LETB"),
                   ("intralayer",None,"LETB",1), ("intralayer",None,"LETB",2),("intralayer",None,"LETB",3),
                   ("full",None,"MK"),
                   ("interlayer","TETB","MK"), ("intralayer","TETB","MK")]
    #model_tuple = [("interlayer","Classical",None),("intralayer","Classical",None)]
    #model_tuple = [("full",None,"MK")]
    model_tuple = [ ("interlayer","TETB","popov")] #("interlayer","TETB","popov"),
    """model_tuple = [("interlayer",None,"LETB"),
                   ("intralayer",None,"LETB",1), ("intralayer",None,"LETB",2),("intralayer",None,"LETB",3),
                   ("full",None,"MK")]"""
    

    T_weight_array = np.array([1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5,1,1.5,2.0,3,4,5]) #,10,50,100] ) #,500,1000])   #np.array([0.01,0.1,0.2,0.5,0.75,1,1.5,2.0,3,4,5])  
    N_folds_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    uq_param_array = [T_weight_array,N_folds_array]
    uq_types = ["mcmc"] #,"cv"]
    ensemble_suffix = ["ensemble_T","CV_ensemble_p"]
    param_name = ["Temperature weight","Percent of training data"]
    use_pickle = True

    for mt in model_tuple:
        print(mt)
        
        int_type = mt[0]
        energy_model = mt[1]
        tb_model = mt[2]
        if len(mt)==4:
            nn_val = mt[-1]
        else:
            nn_val=1

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


        if energy_model=="TETB" and tb_model=="popov":
            if os.path.exists("tb_energy.npz"):
                tb_energy = np.load("tb_energy.npz")["tb_energy"]
            else:
                tb_calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="python")
                if int_type=="interlayer":
                    tb_xdata,_,_ = get_training_data(int_type+" energy",supercells=5) 
                else:
                    tb_xdata = xdata
                tb_energy,_ = tb_calc.get_tb_energy(tb_xdata["energy"][0])
                for x in tb_xdata["energy"][1:]:
                    tbe,_ = tb_calc.get_tb_energy(x)
                    tb_energy = np.append(tb_energy,tbe)
                np.savez("tb_energy",tb_energy=tb_energy)


        for uq_ind,uqt in enumerate(uq_types):
            calc,xdata,ydata,ydata_noise, params,params_std,param_bounds, ypred_bestfit = get_MCMC_inputs(int_type,energy_model,tb_model,model_name,nn_val,calc_type="python")
            Temperature = {}
            w0={}
            if len(calc.keys())==2:
                relative_weight=0.5
            else:
                relative_weight = 1.0
            for nm,key in enumerate(calc):
                w0[key] = relative_weight **((nm+1)%2) * (1-relative_weight)**(nm%2)

            ks_metric = np.zeros(len(uq_param_array[uq_ind]))
            uncertainty_error_metric = np.zeros(len(uq_param_array[uq_ind]))
            
            for temp_ind,Temperature_weight in enumerate(uq_param_array[uq_ind]):
                print("Temp weight = ",Temperature_weight)
                if not os.path.exists("figures/ks_metric_plots/"+model_name):
                    os.mkdir("figures/ks_metric_plots/"+model_name)

                if use_pickle:
                    filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".pkl"
                    with open(filename, 'rb') as file:
                        ensemble_dict = pickle.load(file)

                    ensemble_samples = ensemble_dict["ensemble"]
                    """for key in calc.keys():
                        plt.plot(ensemble_samples[key][:,1])
                        plt.savefig("figures/convergence_check/"+model_name+"_ks_metric_"+ensemble_suffix[uq_ind]+str(Temperature_weight)+".png")
                        plt.clf()"""
                    ypred_samples = ensemble_dict["ypred_samples"]
                    if ypred_samples ==[]:
                        ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
                    ydata = ensemble_dict["ydata"]
                    ypred_samples = {"energy":ypred_samples["energy"]+np.squeeze(tb_energy[:,np.newaxis])}
                else:
                    filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".txt"
                    ensemble_samples_array, ypred_samples_array = read_params_and_predictions(filename)
                    ensemble_samples = {"energy":ensemble_samples_array}
                    ypred_samples = {"energy":ypred_samples_array+np.squeeze(tb_energy[:,np.newaxis])}
                
                #if energy_model=="TETB" and tb_model=="popov":
                #    n_samples = np.shape(ypred_samples["energy"])[0]
                #    for n in range(n_samples):
                #        ypred_samples["energy"][n,:] +=  tb_energy

                ks, ci_array, percent_within_bounds = get_ks_metric(ypred_samples,ydata,w0)
                ks_metric[temp_ind] = ks
                plt.plot(ci_array,percent_within_bounds,label="observed")
                plt.plot(ci_array,ci_array,label="expected")
                plt.xlabel("Confidence interval")
                plt.ylabel("CDF")
                plt.title(" ".join(model_name.split("_")))
                plt.legend()
                plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_ks_metric_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".png")
                plt.clf()
                #coverage = get_coverage(ensemble["ypred_samples"],ydata,w0,ci)

                uncertainty_error_metric[temp_ind] = get_uncertainty_correlation(ypred_samples,ydata,w0)
            plt.plot(uq_param_array[uq_ind],ks_metric)
            plt.plot(uq_param_array[uq_ind],np.zeros_like(uq_param_array[uq_ind]),linestyle="dashed",color="black")
            plt.xlabel(param_name[uq_ind])
            plt.ylabel("KS metric")
            plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_ks_metric"+ensemble_suffix[uq_ind]+".png")
            plt.clf()

            plt.plot(uq_param_array[uq_ind],uncertainty_error_metric)
            plt.plot(uq_param_array[uq_ind],np.zeros_like(uq_param_array[uq_ind]),linestyle="dashed",color="black")
            plt.xlabel(param_name[uq_ind])
            plt.ylabel("UQ correlation metric")
            plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_uq_correlation_metric"+ensemble_suffix[uq_ind]+".png")
            plt.clf()

            
            xaxis_data = []
            for x in xdata["energy"]:
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

            for temp_ind,Temperature_weight in enumerate(uq_param_array[uq_ind]):
                if use_pickle:
                    filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".pkl"
                    with open(filename, 'rb') as file:
                        ensemble_dict = pickle.load(file)

                    ensemble_samples = ensemble_dict["ensemble"]
                    ypred_samples_array = ensemble_dict["ypred_samples"]["energy"]
                    if ypred_samples ==[]:
                        ypred_samples_array,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
                    ydata = ensemble_dict["ydata"]
                    ypred_samples = {"energy":ypred_samples_array+np.squeeze(tb_energy[:,np.newaxis])}
                else: 
                    filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".txt"
                    ensemble_samples_array, ypred_samples_array = read_params_and_predictions(filename)
                    ensemble_samples = {"energy":ensemble_samples_array}
                    ypred_samples = {"energy":ypred_samples_array+np.squeeze(tb_energy[:,np.newaxis])}

                for key in ydata:
                    min_ind = np.argmin(ydata[key])
                    ypred_mean = np.mean(ypred_samples[key] - ypred_samples[key][:,min_ind][:,np.newaxis],axis=0)
                    ypred_std = np.std(ypred_samples[key]- ypred_samples[key][:,min_ind][:,np.newaxis],axis=0)
                    residual_error = np.sqrt(np.power(((ypred_mean - ypred_mean[min_ind]) - (ydata[key] - ydata[key][min_ind])),2))
                    residual_error_scaled = (residual_error - np.min(residual_error)) / (np.max(residual_error) - np.min(residual_error))
                    ypred_std_scaled = (ypred_std - np.min(residual_error))/ (np.max(residual_error) - np.min(residual_error))

                    
                    #plt.errorbar(ydata[key]-np.min(ydata[key]),ypred_mean - np.min(ypred_mean),yerr = ypred_std,fmt="o")
                    #plt.plot(np.linspace(np.min(ydata[key])-np.min(ydata[key]),np.max(ydata[key])-np.min(ydata[key]),10) , np.linspace(np.min(ydata[key])-np.min(ydata[key]),np.max(ydata[key])-np.min(ydata[key]),10) )
                    plt.scatter(xaxis_data,ydata[key]-np.min(ydata[key]),label="ydata",color="black")
                    plt.errorbar(xaxis_data,ypred_mean-np.min(ypred_mean),yerr=ypred_std,label="ypred",fmt="o")
                    #plt.scatter(xaxis_data,tb_energy-tb_energy[-1],label="tb energy")
                    plt.legend()
                    #plt.scatter(residual_error,ypred_std)
                    #plt.plot(np.linspace(np.min(residual_error),np.max(residual_error),10),np.linspace(np.min(residual_error),np.max(residual_error),10))
                    #plt.errorbar(np.linalg.norm(xdata[key],axis=1),np.mean(ypred_samples[key],axis=0),yerr=np.std(ypred_samples[key],axis=0),fmt="o")
                    plt.xlabel("expected error")
                    plt.ylabel("observed uncertainty")
                    #plt.xlabel("ydata")
                    #plt.ylabel("ypred")
                    plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_uq_correlation_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".png")
                    plt.clf()

                    """plt.hist(ypred_samples[key][0,:],bins=25)
                    plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_hist0_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".png")
                    plt.clf()"""


