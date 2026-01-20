import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import os
from model_fit import *
from get_MCMC_inputs import get_MCMC_inputs

plt.rcParams["font.family"] = "serif"
# Optional: also set the font for mathematical text
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["font.size"] = 16

def get_ks_metric(y_pred_samples_dict,ydata_dict,w0=1,burnin = 0.25):
    ks_metric = 0
    ci_array = np.linspace(0.,0.95,15)
    for key in ydata_dict:
        burnin_step = int(burnin * len(y_pred_samples_dict[key]))
        y_pred_samples = y_pred_samples_dict[key][burnin_step:,:]
        ydata = ydata_dict[key]
        shift_ind = np.argmin(ydata)  #choose point closest to zero
        ypred_samples_shift = y_pred_samples[:,shift_ind] 
        ydata_shift = ydata[shift_ind]
        if key == "energy":
            y_pred_samples = (y_pred_samples-ypred_samples_shift[:,np.newaxis]) #/np.std(ydata)
            ydata = (ydata-ydata_shift) #/np.std(ydata)

        percent_within_bounds = np.zeros_like(ci_array)
        for index,ci in enumerate(ci_array):
            lower_bound = np.quantile(y_pred_samples,(1-ci)/2,axis=0)
            upper_bound = np.quantile(y_pred_samples,1-(1-ci)/2,axis=0)
            within_ind = (ydata >= lower_bound[np.newaxis,:]) & (ydata <= upper_bound[np.newaxis,:])
            percent_within_bounds[index] = sum(sum(within_ind))/len(ydata) #counting number of true elements, divided by total num elements
        ci_array = np.append(ci_array,1)
        percent_within_bounds = np.append(percent_within_bounds,1)
        ks_metric += w0[key] * np.trapezoid(2*(percent_within_bounds - ci_array),ci_array)
    
    return ks_metric, ci_array, percent_within_bounds

def get_negative_log_likelihood(ydata,ypred_samples,w0=1):
    for key in ydata.keys():
        min_index = np.argmin(ydata[key])
        ydata_shift = ydata[key] - ydata[key][min_index]
        ypred_mean = np.mean( ypred_samples[key]-ypred_samples[key][:,min_index][:,np.newaxis],axis=0)
        ypred_std = np.std(ypred_samples[key]-ypred_samples[key][:,min_index][:,np.newaxis],axis=0)
        ydata_shift = np.delete(ydata_shift,min_index)
        ypred_mean = np.delete(ypred_mean,min_index)
        ypred_std = np.delete(ypred_std,min_index)
        negative_log_likelihood =  -np.mean(0.5*np.log( ypred_std**2) + (ydata_shift - ypred_mean)**2 / (2 * ypred_std**2))
    return negative_log_likelihood

def get_Hessian(cost_fn, theta_opt, eps=1e-8):
    """
    Laplace-approximate parameter covariance.
    """
    theta = np.asarray(theta_opt)
    n = theta.size
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps

            fpp = cost_fn(theta + ei + ej)
            fpm = cost_fn(theta + ei - ej)
            fmp = cost_fn(theta - ei + ej)
            fmm = cost_fn(theta - ei - ej)

            H_ij = (fpp - fpm - fmp + fmm) / (4 * eps**2)
            H[i, j] = H[j, i] = H_ij
    # Symmetrize explicitly
    H = 0.5 * (H + H.T)

    return H

if __name__=="__main__":
    plot_ks_metric = True
    plot_confidence_interval = False
    plot_negative_log_likelihood = False
    plot_uq_correlation = False
    plot_convergence = True

    model_names = ["MLP_SK"]
    

    T_weight_array = np.array([1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5,1,1.5,2.0,3,4,5,7,10,15,20,30,50]) #,10,50,100] ) #,500,1000])   #np.array([0.01,0.1,0.2,0.5,0.75,1,1.5,2.0,3,4,5])  
    N_folds_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    uq_param_array = [T_weight_array,N_folds_array]
    uq_types = ["mcmc"] #,"cv"]
    ensemble_suffix = ["ensemble_T","CV_ensemble_p"]
    param_name = [r"$ln(T/T_{0})$","Percent of training data"]
    use_pickle = True

    if plot_ks_metric:

        for model_name in model_names:
            print(model_name)

            for uq_ind,uqt in enumerate(uq_types):
                calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
                Temperature = {}
                w0={}
                if len(calc.keys())==2:
                    relative_weight=0.5
                else:
                    relative_weight = 1.0
                for nm,key in enumerate(calc):
                    w0[key] = relative_weight **((nm+1)%2) * (1-relative_weight)**(nm%2)

                ks_metric = np.zeros(len(uq_param_array[uq_ind]))
                mae = np.zeros(len(uq_param_array[uq_ind]))
                average_cost = np.zeros(len(uq_param_array[uq_ind]))
                std_cost = np.zeros(len(uq_param_array[uq_ind]))
                for temp_ind,Temperature_weight in enumerate(uq_param_array[uq_ind]):
                    print("Temp weight = ",Temperature_weight)
                    if not os.path.exists("figures/ks_metric_plots/"+model_name):
                        os.mkdir("figures/ks_metric_plots/"+model_name)

                    filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".pkl"
                    with open(filename, 'rb') as file:
                        ensemble_dict = pickle.load(file)

                    ensemble_samples = ensemble_dict["ensemble"]
                    ypred_samples = ensemble_dict["ypred_samples"]
                    if ypred_samples ==[]:
                        ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
                    ydata = ensemble_dict["ydata"]
                    for key in ypred_samples:
                        ypred_samples[key] = ypred_samples[key]+yshift_data[key]
                        mae[temp_ind] = np.mean(np.abs(np.mean(ypred_samples[key],axis=0) - ydata[key]))
                        average_cost[temp_ind] = np.mean((ypred_samples[key]-ydata[key][np.newaxis,:])**2)
                        std_cost[temp_ind] = np.std((ypred_samples[key]-ydata[key][np.newaxis,:])**2)

                    ks, ci_array, percent_within_bounds = get_ks_metric(ypred_samples,ydata,w0)
                    ks_metric[temp_ind] = ks
                        
                if uqt=="mcmc":
                    plt.plot(np.log10(uq_param_array[uq_ind]),ks_metric,label=uqt)
                    plt.plot(np.log10(uq_param_array[uq_ind]),np.zeros_like(uq_param_array[uq_ind]),linestyle="dashed",color="black")
                else:
                    plt.plot(uq_param_array[uq_ind],ks_metric,label=uqt)
                    plt.plot(uq_param_array[uq_ind],np.zeros_like(uq_param_array[uq_ind]),linestyle="dashed",color="black")
                plt.xlabel(param_name[uq_ind])
                plt.ylabel(r"$\mathcal{M}$")
                plt.legend()
                plt.tight_layout()
                plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_ks_metric"+ensemble_suffix[uq_ind]+".png")
                plt.clf()
                print("Minimum KS Temperature weight = ",uq_param_array[uq_ind][np.argmin(np.abs(ks_metric))])

                if uqt=="mcmc":
                    plt.plot(np.log10(uq_param_array[uq_ind]),mae,label=uqt)
                else:
                    plt.plot(uq_param_array[uq_ind],mae,label=uqt)
                plt.xlabel(param_name[uq_ind])
                plt.ylabel(r"MAE")
                plt.legend()
                plt.tight_layout()
                plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_mae_"+ensemble_suffix[uq_ind]+".png")
                plt.clf()

                if uqt=="mcmc":
                    plt.plot((uq_param_array[uq_ind]),average_cost,label=uqt)
                    plt.plot((uq_param_array[uq_ind]),average_cost+std_cost,linestyle="dashed",color="black")
                    plt.plot((uq_param_array[uq_ind]),average_cost-std_cost,linestyle="dashed",color="black")
                    plt.xlabel(r"$T/T_0$")
                else:
                    plt.plot(uq_param_array[uq_ind],average_cost,label=uqt)
                    plt.plot(uq_param_array[uq_ind],average_cost+std_cost,linestyle="dashed",color="black")
                    plt.plot(uq_param_array[uq_ind],average_cost-std_cost,linestyle="dashed",color="black")
                    plt.xlabel("Percent of training data")
                plt.ylabel(r"$\langle C(\theta) \rangle$")
                plt.legend()
                plt.tight_layout()
                plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_average_cost_"+ensemble_suffix[uq_ind]+".png")
                plt.clf()

    if plot_confidence_interval:
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
                if os.path.exists("tb_energy"+model_name+".npz"):
                    tb_energy = np.load("tb_energy"+model_name+".npz")["tb_energy"]
                else:
                    tb_calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="python")
                    xdata,_,_ = get_training_data(int_type+" energy",supercells=20)
                    tb_energy,_ = tb_calc.get_tb_energy(xdata["energy"][0])
                    for x in xdata["energy"][1:]:
                        tbe,_ = tb_calc.get_tb_energy(x)
                        tb_energy = np.append(tb_energy,tbe)
                    np.savez("tb_energy"+model_name,tb_energy=tb_energy)


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
                        ypred_samples = ensemble_dict["ypred_samples"]
                        if ypred_samples ==[]:
                            ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
                        ydata = ensemble_dict["ydata"]
                        if energy_model=="TETB":
                            ypred_samples = {"energy":ypred_samples["energy"]+np.squeeze(tb_energy[:,np.newaxis])}

                    else:
                        filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".txt"
                        ensemble_samples_array, ypred_samples_array = read_params_and_predictions(filename)
                        ensemble_samples = {"energy":ensemble_samples_array}
                        if energy_model=="TETB":
                            ypred_samples = {"energy":ypred_samples_array+np.squeeze(tb_energy[:,np.newaxis])}

                    ks, ci_array, percent_within_bounds = get_ks_metric(ypred_samples,ydata,w0)
                    plt.plot(ci_array,percent_within_bounds,label="observed")
                    plt.plot(ci_array,ci_array,label="expected")
                    plt.xlabel(r"Confidence interval ($\alpha$)")
                    plt.ylabel(r"$\hat{C}_{\alpha}$")
                    #plt.title(" ".join(model_name.split("_")))
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("figures/ks_metric_plots/"+model_name+"/"+model_name+"_"+key+"_ks_metric_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".png")
                    plt.clf()
    
    if plot_negative_log_likelihood:
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
                if os.path.exists("tb_energy"+model_name+".npz"):
                    tb_energy = np.load("tb_energy"+model_name+".npz")["tb_energy"]
                else:
                    tb_calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="python")
                    xdata,_,_ = get_training_data(int_type+" energy",supercells=20)
                    tb_energy,_ = tb_calc.get_tb_energy(xdata["energy"][0])
                    for x in xdata["energy"][1:]:
                        tbe,_ = tb_calc.get_tb_energy(x)
                        tb_energy = np.append(tb_energy,tbe)
                    natoms = len(xdata["energy"][0])
                    np.savez("tb_energy"+model_name,tb_energy=tb_energy)


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

                nll_metric = np.zeros(len(uq_param_array[uq_ind]))
                
                for temp_ind,Temperature_weight in enumerate(uq_param_array[uq_ind]):

                    if use_pickle:
                        filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".pkl"
                        with open(filename, 'rb') as file:
                            ensemble_dict = pickle.load(file)

                        ensemble_samples = ensemble_dict["ensemble"]
                        ypred_samples = ensemble_dict["ypred_samples"]
                        ypred_samples = ensemble_dict["ypred_samples"]
                        if ypred_samples ==[]:
                            ypred_samples,ensemble_samples = evaluate_ensemble(ensemble_samples,xdata, ydata, calc)
                        ydata = ensemble_dict["ydata"]
                        if energy_model=="TETB":
                            ypred_samples = {"energy":ypred_samples["energy"]+np.squeeze(tb_energy[:,np.newaxis])}

                    else:
                        filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".txt"
                        ensemble_samples_array, ypred_samples_array = read_params_and_predictions(filename)
                        ensemble_samples = {"energy":ensemble_samples_array}
                        if energy_model=="TETB":
                            ypred_samples = {"energy":ypred_samples_array+np.squeeze(tb_energy[:,np.newaxis])}


                    negative_log_likelihood = get_negative_log_likelihood(ydata,ypred_samples)
                    nll_metric[temp_ind] = negative_log_likelihood
                plt.plot(uq_param_array[uq_ind],nll_metric,label=uqt)
                #plt.plot(uq_param_array[uq_ind],np.zeros_like(uq_param_array[uq_ind]),linestyle="dashed",color="black")
                plt.xlabel(param_name[uq_ind])
                plt.ylabel(r"$\mathcal{NLL}$")
                plt.legend()
                plt.tight_layout()
                plt.savefig("figures/negative_log_likelihood_plots/"+model_name+"_"+key+"_negative_log_likelihood"+ensemble_suffix[uq_ind]+".png")
                plt.clf()
                print("Minimum NLL Temperature weight = ",uq_param_array[uq_ind][np.argmin(nll_metric)])

    if plot_convergence:
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
                if os.path.exists("tb_energy"+model_name+".npz"):
                    tb_energy = np.load("tb_energy"+model_name+".npz")["tb_energy"]
                else:
                    tb_calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type="python")
                    xdata,_,_ = get_training_data(int_type+" energy",supercells=20)
                    tb_energy,_ = tb_calc.get_tb_energy(xdata["energy"][0])
                    for x in xdata["energy"][1:]:
                        tbe,_ = tb_calc.get_tb_energy(x)
                        tb_energy = np.append(tb_energy,tbe)
                    np.savez("tb_energy"+model_name,tb_energy=tb_energy)


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
                        ypred_samples = ensemble_dict["ypred_samples"]
                        ypred_trace = ypred_samples[key] - ypred_samples[key][:,np.argmin(ydata[key])][:,np.newaxis]
                        ypred_trace = np.mean(ypred_trace,axis=0)
                        ypred_ave_evolve = np.array([np.mean(ypred_trace[:i]) for i in range(len(ypred_trace))])
                        
                        for key in calc.keys():
                            # plt.plot(ypred_ave_evolve)
                            # plt.ylabel("average Energy (meV/atom)")
                            # plt.xlabel("step")
                            # plt.savefig("figures/convergence_check/"+model_name+"_"+ensemble_suffix[uq_ind]+str(Temperature_weight)+".png")
                            # plt.clf()

                            plt.plot(ensemble_samples[key][:,0])
                            plt.ylabel("ensemble sample ")
                            plt.xlabel("step")
                            plt.savefig("figures/convergence_check/"+model_name+"_"+ensemble_suffix[uq_ind]+str(Temperature_weight)+"ensemble_sample.png")
                            plt.clf()

    if plot_uq_correlation:
        for mt in model_tuple:
            print(mt)
            
            int_type = mt[0]
            energy_model = mt[1]
            tb_model = mt[2]
            if len(mt)==4:
                nn_val = mt[-1]
            else:
                nn_val=1
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
                    if energy_model=="TETB":
                        ypred_samples = {"energy":ypred_samples["energy"]+np.squeeze(tb_energy[:,np.newaxis])}

                else: 
                    filename = "ensembles/"+model_name+"/"+model_name+"_"+ensemble_suffix[uq_ind]+"_"+str(Temperature_weight)+".txt"
                    ensemble_samples_array, ypred_samples_array = read_params_and_predictions(filename)
                    ensemble_samples = {"energy":ensemble_samples_array}
                    if energy_model=="TETB":
                        ypred_samples = {"energy":ypred_samples["energy"]+np.squeeze(tb_energy[:,np.newaxis])}


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

                   


