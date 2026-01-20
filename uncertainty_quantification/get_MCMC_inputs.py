import numpy as np 
from BLG_model_builder.MLP import *
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
from BLG_model_builder.BLG_model_library import *
from model_fit import *
import matplotlib.pyplot as plt
import os

def get_MCMC_inputs(model_name,calc_type="lammps"):
    """ model_name: string, name of the model to get the inputs for"""
    """ calc_type: string, type of calculation to get the inputs for"""
    """
    Returns:
    calc: function for calculting predictions
    xdata: x data
    ydata: y data
    ydata_noise: y data noise
    params: best fit parameters for calc
    bounds: bounds for the parameters
    ypred_bestfit: best fit predictions for calc
    """
    ############################################################

    #total energy models

    ############################################################
    #classical interatomic potentials
    if model_name=="Classical_energy_interlayer":
        key="energy"
        calc = get_BLG_Evaluator(int_type="interlayer",energy_model="Classical",tb_model=None,calc_type=calc_type)
        xdata,ydata,ydata_noise = get_training_data("interlayer energy")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["energy"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            fit_model(calc["energy"],xdata["energy"],ydata["energy"],p0,shift_data=True,bounds=bounds["energy"])
        
    elif model_name=="Classical_energy_intralayer":
        key="energy"
        calc = get_BLG_Evaluator(int_type="intralayer",energy_model="Classical",tb_model=None,calc_type=calc_type)
        xdata,ydata,ydata_noise = get_training_data("intralayer energy")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["energy"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            fit_model(calc["energy"],xdata["energy"],ydata["energy"],p0,shift_data=True,bounds=bounds["energy"])

    elif model_name=="MLP_energy_interlayer":
        key="energy"
        calc = get_BLG_Evaluator(int_type="interlayer",energy_model="MLP",tb_model=None,calc_type=calc_type)
        xdata,ydata,ydata_noise = get_training_data("interlayer energy")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            torch_mlp_func() #
            fit_model(torch_mlp_func,xdata["energy"],ydata["energy"],minimizer="torch_mlp")

    elif model_name=="MLP_energy_intralayer":
        key="energy"
        calc = get_BLG_Evaluator(int_type="intralayer",energy_model="MLP",tb_model=None,calc_type=calc_type)
        xdata,ydata,ydata_noise = get_training_data("intralayer energy")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            torch_mlp_func() #
            fit_model(torch_mlp_func,xdata["energy"],ydata["energy"],minimizer="torch_mlp")

    #TETB models
    elif model_name=="TETB_energy_interlayer_popov":
        key="energy"
        calc = get_BLG_Evaluator(int_type="interlayer",energy_model="TETB",tb_model="popov",calc_type=calc_type)
        xdata,ydata,ydata_noise = get_training_data("interlayer energy")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["energy"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            tb_energy = []
            for x in xdata["energy"]:
                tbe,_ = calc["energy"].get_tb_energy(x)
                tb_energy.append(tbe)
            fit_model(calc["energy"],xdata["energy"],ydata["energy"],p0,tb_energy=np.array(tb_energy),shift_data=True,bounds=bounds["energy"])

    elif model_name=="TETB_energy_intralayer_popov":
        key="energy"
        calc = get_BLG_Evaluator(int_type="intralayer",energy_model="TETB",tb_model="popov",calc_type=calc_type)
        xdata,ydata,ydata_noise = get_training_data("intralayer energy")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["energy"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            tb_energy = []
            for x in xdata["energy"]:
                tbe,_ = calc["energy"].get_tb_energy(x)
                tb_energy.append(tbe)
            fit_model(calc["energy"],xdata["energy"],ydata["energy"],p0,tb_energy=np.array(tb_energy),shift_data=True,bounds=bounds["energy"])

    elif model_name=="TETB_MLP_energy_interlayer":
        key="energy"
        print(model_name+" Not implemented yet. Try another model.")
        exit()
        calc = get_BLG_Evaluator(int_type="interlayer",energy_model="TETB",tb_model="MLP",calc_type=calc_type)
        energy_xdata,energy_ydata,energy_ydata_noise = get_training_data("interlayer energy")
        hopping_xdata,hopping_ydata,hopping_ydata_noise = get_training_data("interlayer hopping")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            torch_mlp_func() #
            fit_torch_mlp(torch_mlp_func,model,xdata["energy"],ydata["energy"])

    elif model_name=="TETB_MLP_energy_intralayer":
        key="energy"
        print(model_name+" Not implemented yet. Try another model.")
        exit()
        calc = get_BLG_Evaluator(int_type="intralayer",energy_model="TETB",tb_model="MLP",calc_type=calc_type)
        energy_xdata,energy_ydata,energy_ydata_noise = get_training_data("intralayer energy")
        hopping_xdata,hopping_ydata,hopping_ydata_noise = get_training_data("intralayer hopping")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            torch_mlp_func() #
            fit_model(torch_mlp_func,xdata["energy"],ydata["energy"],minimizer="torch_mlp")

    #hopping models
    elif model_name=="MK":
        key="hoppings"
        calc = {"hoppings": mk_hopping}
        xdata,ydata,ydata_noise = get_training_data("MK hoppings")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["hoppings"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            fit_model(calc["hoppings"],xdata["hoppings"],ydata["hoppings"],p0,shift_data=False,bounds=bounds["hoppings"])

    elif model_name=="interlayer_LETB":
        key="hoppings"
        calc = {"hoppings": letb_interlayer}
        xdata,ydata,ydata_noise = get_training_data("interlayer_LETB hoppings")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["hoppings"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            fit_model(calc["hoppings"],xdata["hoppings"],ydata["hoppings"],p0,shift_data=False,bounds=bounds["hoppings"])

    elif model_name=="intralayer_LETB_NN_val_1":
        key="hoppings"
        calc = {"hoppings": letb_intralayer_t01}
        xdata,ydata,ydata_noise = get_training_data("intralayer_LETB_NN_val_1 hoppings")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["hoppings"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            fit_model(calc["hoppings"],xdata["hoppings"],ydata["hoppings"],p0,shift_data=False,bounds=bounds["hoppings"])

    elif model_name=="intralayer_LETB_NN_val_2":
        key="hoppings"
        calc = {"hoppings": letb_intralayer_t02}
        xdata,ydata,ydata_noise = get_training_data("intralayer_LETB_NN_val_2 hoppings")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["hoppings"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            fit_model(calc["hoppings"],xdata["hoppings"],ydata["hoppings"],p0,shift_data=False,bounds=bounds["hoppings"])

    elif model_name=="intralayer_LETB_NN_val_3":
        key="hoppings"
        calc = {"hoppings": letb_intralayer_t03}
        xdata,ydata,ydata_noise = get_training_data("intralayer_LETB_NN_val_3 hoppings")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            initial_estimate_dict = np.load("best_fit_params/"+model_name+"_best_fit_params_estimate.npz")
            p0 = initial_estimate_dict["params"]
            bounds["hoppings"] = initial_estimate_dict["bounds"]
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            best_fit_params,ypred_bestfit = fit_model(calc["hoppings"],xdata["hoppings"],ydata["hoppings"],p0,shift_data=False,bounds=bounds["hoppings"])
            np.savez("best_fit_params/"+model_name+"_best_fit_params",params=best_fit_params,bounds=bounds["hoppings"], ypred_bestfit=ypred_bestfit)

    elif model_name=="MLP_SK":
        key="hoppings"
        input_dim = 1
        output_dim = 2
        hidden_dim = 15
        model = MLP_numpy(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        calc = {"hoppings": get_MLP_SK_hoppings_func(model)} #this function is a numpy/cupy version of the pytorch MLP
        xdata,ydata,ydata_noise = get_training_data("MLP_SK hoppings")
        yshift_data = np.zeros_like(ydata["hoppings"])
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            torch_func = MLP_SK_hoppings_torch #this is the pytorch MLP hopping function
            model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            best_fit_params,ypred_bestfit = fit_torch_mlp(torch_func,model, xdata["hoppings"],ydata["hoppings"])
            bounds = [[-1e4,1e4]]*len(best_fit_params)
            np.savez("best_fit_params/"+model_name+"_best_fit_params",params=best_fit_params,bounds=bounds, ypred_bestfit=ypred_bestfit, yshift_data=yshift_data)

    elif model_name=="MLP_LETB":
        key="hoppings"
        calc = {"hoppings": MLP_LETB_hoppings}
        xdata,ydata,ydata_noise = get_training_data("MLP_LETB hoppings")
        if not os.path.exists("best_fit_params/"+model_name+"_best_fit_params.npz") :
            print("fitting model")
            #fit model and write to (model_name)_best_fit_params.npz
            torch_func = MLP_hoppings_torch
            model = MLP(input_dim=3, hidden_dim=64, output_dim=1)
            best_fit_params,ypred_bestfit = fit_torch_mlp(torch_mlp_func,model,xdata["hoppings"],ydata["hoppings"])
            bounds = [[-1e4,1e4]]*len(best_fit_params)
            np.savez("best_fit_params/"+model_name+"_best_fit_params",params=best_fit_params,bounds=bounds, ypred_bestfit=ypred_bestfit)
    
    #these lines work for all models
    best_fit_data = np.load("best_fit_params/"+model_name+"_best_fit_params.npz")
    params = {key:best_fit_data["params"]}
    bounds = {key:best_fit_data["bounds"]}
    ypred_bestfit = {key:best_fit_data["ypred_bestfit"]}
    yshift_data = {key:best_fit_data["yshift_data"]}
    return calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds

if __name__=="__main__":
    model_name = "MLP_SK"
    calc, xdata, ydata, ydata_noise, yshift_data, ypred_bestfit, params, bounds = get_MCMC_inputs(model_name)
    r = np.linalg.norm(xdata["hoppings"],axis=1)
    plt.scatter(r,ydata["hoppings"])
    plt.scatter(r,ypred_bestfit)
    plt.show()