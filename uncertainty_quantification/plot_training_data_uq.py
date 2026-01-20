import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import os
from scipy.spatial.distance import cdist
from EMCEE_generate_ensemble import get_MCMC_inputs, train_test_split, get_residual_error
from model_fit import *

plt.rcParams["font.family"] = "serif"
# Optional: also set the font for mathematical text
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["font.size"] = 16

if __name__ == "__main__":
    plot_model_name = ["Classical_energy_interlayer","Classical_energy_intralayer",
                        "TETB_energy_interlayer_popov","TETB_energy_intralayer_popov",
                        "MK","intralayer_LETB_NN_val_1","intralayer_LETB_NN_val_2","intralayer_LETB_NN_val_3","interlayer_LETB","MLP_SK"]
    plot_model_name = ["MLP_SK"]

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
                    "MLP_SK":"ensembles/MLP_SK/MLP_SK_ensemble_T_4.0.pkl",
                    },
                    
                    "cv":{"Classical_energy_interlayer":"ensembles/Classical_energy_interlayer/Classical_energy_interlayer_CV_ensemble_p_0.8.pkl",
                       "Classical_energy_intralayer":"ensembles/Classical_energy_intralayer/Classical_energy_intralayer_CV_ensemble_p_0.2.pkl",
                       "TETB_energy_interlayer_MK":"ensembles/TETB_energy_interlayer_MK/TETB_energy_interlayer_MK_CV_ensemble_p_0.5.pkl",
                        "TETB_energy_intralayer_MK":"ensembles/TETB_energy_intralayer_MK/TETB_energy_intralayer_MK_CV_ensemble_p_0.5.pkl",
                        "MK":"ensembles/MK/MK_CV_ensemble_p_0.9.pkl"}}
    uq_type = "mcmc"
    ####################################

    # plot interlayer ensemble

    #########################################################

    if "Classical_energy_interlayer" in plot_model_name:
        model_name = "Classical_energy_interlayer"
        filename = opt_ensemble[uq_type][model_name]
        calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
        
        with open(filename, 'rb') as file:
            ensemble_dict = pickle.load(file)
        interlayer_ensemble = ensemble_dict["ensemble"]["energy"]
        ypred = ensemble_dict["ypred_samples"]["energy"]

        ypred = ypred + yshift_data["energy"][np.newaxis,:]
            
        ypred = ypred/4 #len(xdata["energy"][0])
        ci = 0.64
        lower_bound = np.quantile(ypred - ypred[:,min_ind][:,np.newaxis],(1-ci)/2,axis=0)
        upper_bound = np.quantile(ypred- ypred[:,min_ind][:,np.newaxis],1-(1-ci)/2,axis=0)
        ypred_std = (upper_bound - lower_bound)

        plt.scatter(d_,(ydata["energy"]-ydata["energy"][min_ind])/len(xdata["energy"][0]) - deltaE,label="qmc",color="black")
        plt.errorbar(d_,(np.mean(ypred-ypred[:,min_ind][:,np.newaxis],axis=0)) - deltaE,yerr = ypred_std,fmt="o",label="ypred",color="red")
        #plt.scatter(d_,(ytotal-ytotal[min_ind])/len(xdata["energy"][0]) - deltaE,label="best fit",color="blue")
        plt.xlabel("layer sep")
        plt.ylabel("Energy (eV/atom)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/"+interlayer_name+"_"+uq_type+".png")
        plt.clf()

    #########################################################

    # plot intralayer ensemble

    #########################################################

    # plot TETB interlayer ensemble

    #########################################################

    # plot TETB intralayer ensemble

    #########################################################

    # plot MK ensemble

    #########################################################

    if "MK" in plot_model_name:
        model_name = "MK"
        filename = opt_ensemble[uq_type][model_name]
        calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
        intralayer_ind = np.where(xdata["hoppings"][:,2] < 0.2)[0]
        interlayer_ind = np.where(xdata["hoppings"][:,2] > 0.2)[0]
        ypred_bestfit = ypred_bestfit["hoppings"]

        with open(filename, 'rb') as file:
            ensemble_dict = pickle.load(file)
        mk_ensemble = ensemble_dict["ensemble"]["hoppings"]
        ypred = ensemble_dict["ypred_samples"]["hoppings"]

        print(uq_type+" MK ensemble mean = ",np.mean(mk_ensemble,axis=0))
        print(uq_type+" MK ensemble std = ",np.std(mk_ensemble,axis=0))
        
        xaxis_data = np.linalg.norm(xdata["hoppings"],axis=1)
        ci = 0.64
        lower_bound = np.quantile(ypred ,(1-ci)/2,axis=0)
        upper_bound = np.quantile(ypred,1-(1-ci)/2,axis=0)
        ypred_std = (upper_bound - lower_bound)

        plt.scatter(xaxis_data[intralayer_ind],ydata["hoppings"][intralayer_ind],label="DFT",color="black")
        plt.errorbar(xaxis_data[intralayer_ind],np.mean(ypred,axis=0)[intralayer_ind],yerr = ypred_std[intralayer_ind],fmt="o",label="ypred",color="red")
        plt.scatter(xaxis_data[intralayer_ind],ypred_bestfit[intralayer_ind],label="best fit",color="blue")
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/MK_intralayer_"+uq_type+".png")
        plt.clf()

        plt.scatter(xaxis_data[interlayer_ind],ydata["hoppings"][interlayer_ind],label="DFT",color="black")
        plt.errorbar(xaxis_data[interlayer_ind],np.mean(ypred,axis=0)[interlayer_ind],yerr = ypred_std[interlayer_ind],fmt="o",label="ypred",color="red")
        plt.scatter(xaxis_data[interlayer_ind],ypred_bestfit[interlayer_ind],label="best fit",color="blue")
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/MK_interlayer_"+uq_type+".png")
        plt.clf()

    #########################################################

    # plot LETB ensemble

    #########################################################

    if "intralayer_LETB_NN_val_1" in plot_model_name:
        model_name = "intralayer_LETB_NN_val_"
        nn_vals = [1,2,3]
        for nn in nn_vals:
            model_name = model_name + str(nn)
            filename = opt_ensemble[uq_type]["intralayer_LETB_NN_val_"+str(nn)]
            calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
            ypred_bestfit = ypred_bestfit["hoppings"]

            with open(filename, 'rb') as file:
                ensemble_dict = pickle.load(file)
            letb_ensemble = ensemble_dict["ensemble"]["hoppings"]
            ypred = ensemble_dict["ypred_samples"]["hoppings"]

            print(uq_type+" LETB ensemble mean = ",np.mean(letb_ensemble,axis=0))
            print(uq_type+" LETB ensemble std = ",np.std(letb_ensemble,axis=0))
            if nn==1:
                xaxis_data = xdata["hoppings"]
            elif nn==2:
                xaxis_data = xdata["hoppings"][:,-1]
            elif nn==3:
                xaxis_data = xdata["hoppings"][:,0]
            ci = 0.64
            lower_bound = np.quantile(ypred ,(1-ci)/2,axis=0)
            upper_bound = np.quantile(ypred,1-(1-ci)/2,axis=0)
            ypred_std = (upper_bound - lower_bound)

            if nn==1:
                plt.scatter(xaxis_data,ydata["hoppings"],label="DFT",color="black")
                plt.errorbar(xaxis_data,np.mean(ypred,axis=0),yerr = ypred_std,fmt="o",label="ypred",color="red")
                plt.scatter(xaxis_data,ypred_bestfit,label="best fit",color="blue")
            else:
                plt.scatter(xaxis_data,ydata["hoppings"],color="black")
                plt.errorbar(xaxis_data,np.mean(ypred,axis=0),yerr = ypred_std,fmt="o",color="red")
                plt.scatter(xaxis_data,ypred_bestfit,color="blue")
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/LETB_intralayer_"+uq_type+".png")
        plt.clf()

    if "interlayer_LETB" in plot_model_name:
        model_name = "interlayer_LETB"
        filename = opt_ensemble[uq_type][model_name]
        calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
        ypred_bestfit = ypred_bestfit["hoppings"]

        with open(filename, 'rb') as file:
            ensemble_dict = pickle.load(file)
        letb_ensemble = ensemble_dict["ensemble"]["hoppings"]
        ypred = ensemble_dict["ypred_samples"]["hoppings"]

        print(uq_type+" LETB ensemble mean = ",np.mean(letb_ensemble,axis=0))
        print(uq_type+" LETB ensemble std = ",np.std(letb_ensemble,axis=0))
        
        xaxis_data = xdata["hoppings"][:,0] #np.linalg.norm(xdata["hoppings"],axis=1)
        ci = 0.64
        lower_bound = np.quantile(ypred ,(1-ci)/2,axis=0)
        upper_bound = np.quantile(ypred,1-(1-ci)/2,axis=0)
        ypred_std = (upper_bound - lower_bound)
        plt.scatter(xaxis_data,ydata["hoppings"],label="DFT",color="black")
        plt.errorbar(xaxis_data,np.mean(ypred,axis=0),yerr = ypred_std,fmt="o",label="ypred",color="red")
        plt.scatter(xaxis_data,ypred_bestfit,label="best fit",color="blue")
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/LETB_interlayer_"+uq_type+".png")
        plt.clf()

    if "MLP_SK" in plot_model_name:
        model_name = "MLP_SK"
        filename = opt_ensemble[uq_type][model_name]
        calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name,calc_type="python")
        intralayer_ind = np.where(xdata["hoppings"][:,2] < 0.2)[0]
        interlayer_ind = np.where(xdata["hoppings"][:,2] > 0.2)[0]
        ypred_bestfit = ypred_bestfit["hoppings"]

        with open(filename, 'rb') as file:
            ensemble_dict = pickle.load(file)
        mlp_sk_ensemble = ensemble_dict["ensemble"]["hoppings"]
        ypred = ensemble_dict["ypred_samples"]["hoppings"]
        
        xaxis_data = np.linalg.norm(xdata["hoppings"],axis=1)
        ci = 0.64
        lower_bound = np.quantile(ypred ,(1-ci)/2,axis=0)
        upper_bound = np.quantile(ypred,1-(1-ci)/2,axis=0)
        ypred_std = (upper_bound - lower_bound)

        plt.scatter(xaxis_data[intralayer_ind],ydata["hoppings"][intralayer_ind],label="DFT",color="black")
        plt.errorbar(xaxis_data[intralayer_ind],np.mean(ypred,axis=0)[intralayer_ind],yerr = ypred_std[intralayer_ind],fmt="o",label="ypred",color="red")
        plt.scatter(xaxis_data[intralayer_ind],ypred_bestfit[intralayer_ind],label="best fit",color="blue")
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/MLP_SK_intralayer_"+uq_type+".png")
        plt.clf()

        plt.scatter(xaxis_data[interlayer_ind],ydata["hoppings"][interlayer_ind],label="DFT",color="black")
        plt.errorbar(xaxis_data[interlayer_ind],np.mean(ypred,axis=0)[interlayer_ind],yerr = ypred_std[interlayer_ind],fmt="o",label="ypred",color="red")
        plt.scatter(xaxis_data[interlayer_ind],ypred_bestfit[interlayer_ind],label="best fit",color="blue")
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/MLP_SK_interlayer_"+uq_type+".png")
        plt.clf()

        r_lin = np.linspace(1.40,6,20)
        r_disp = np.array([r_lin,np.zeros_like(r_lin),np.zeros_like(r_lin)]).T #sigma bonding only

        parameters = np.load("best_fit_params/MLP_SK_best_fit_params.npz")["params"]
        ypred_lin = calc["hoppings"](r_disp,parameters)
        ypred_lin_ensemble = np.zeros((len(mlp_sk_ensemble),len(r_lin)))
        for i,param in enumerate(mlp_sk_ensemble):
            ypred_lin_ensemble[i] = calc["hoppings"](r_disp,param)

        ypred_lin_mean = np.mean(ypred_lin_ensemble,axis=0)
        ypred_lin_std = np.std(ypred_lin_ensemble,axis=0)
        plt.scatter(r_lin,ypred_lin,label="best fit",color="blue")
        plt.scatter(xaxis_data[intralayer_ind],ydata["hoppings"][intralayer_ind],label="DFT",color="black")
        plt.plot(r_lin,ypred_lin_mean,label="ypred",color="red")
        plt.fill_between(r_lin,ypred_lin_mean - ypred_lin_std,ypred_lin_mean + ypred_lin_std,color="red",alpha=0.5)
        plt.xlabel("hopping distance")
        plt.ylabel("hopping energy (eV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/MLP_SK_linear_"+uq_type+".png")
        plt.clf()