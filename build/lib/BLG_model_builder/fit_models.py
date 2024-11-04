import numpy as np
import scipy.optimize 
from TB_Utils import *
from Lammps_Utils import *
from descriptors import *
from BLG_model_builder.TETB_model_builder import TETB_model
from BLG_model_builder.geom_tools import *

def build_Loss_func(xdata,ydata,models,methods,param_names,weights=None):
    def loss_func(params,weights=weights):
        """xdata, ydata, model all lists for different types of fitting data """
        if weights is None:
            weights = 1/len(models)
        rmse = 0
        for i,m in enumerate(models):
            m.set_params(param_names[i][0],param_names[i][1],params)
            yfit = []
            for x in xdata[i]:
                yfit.append(methods[i](x))
            yfit = np.arrary(yfit)
            rmse += weights[i] * np.linalg.norm((ydata[i]-yfit)/ydata)

        return rmse
    return loss_func

def fit_model(xdata,ydata,models,methods,p0,param_names,bounds=None,weights=None):
    Loss_func = build_Loss_func(xdata,ydata,models,methods,param_names,weights=weights)
    popt,pcov = scipy.optimize.minimize(Loss_func,p0, method="Nelder-Mead",bounds=bounds)
    return popt

if __name__=="__main__":
    mk_dict = np.load("parameters/MK_tb_params.npz")
    a = mk_dict["a"]
    b=mk_dict["b"]
    c=mk_dict["c"]
    mk_params = np.array([a,b,c])
    interlayer_params = np.array([0.719345289329483, 6.074935002291668, 18.184672181803677,
             13.394207130830571, 0.003559135312169, 3.379423382381699,
             1.6724670937654809 ,13.646628785353208, 0.7907544823937784])
    intralayer_params = np.array([0.14687637217609084, 4.683462616941604, 12433.64356176609,\
             12466.479169306709, 19.121905577450008, 30.504342033258325,\
             4.636516235627607 , 1.3641304165817836, 1.3878198074813923])

    model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer"},
                                "potential":"reg/dep/poly 12.0 1","potential parameters":interlayer_params,
                                "potential file writer":write_kcinsp},
    
               "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer"},
                                "potential":"airebo 3","potential parameters":intralayer_params,"potential file writer":write_rebo}}
    
    calc = TETB_model(model_dict)

    loss = "interlayer energy"
    if loss=="total":
        models = [calc.get_total_energy,calc.get_total_energy,mk_hopping,mk_hopping]
        xdata = [interlayer_atoms, intralayer_atoms, interlayer_disp, intralayer_disp ]
        ydata = [interlayer_energy, intralayer_energy, interlayer_hoppings, intralayer_hoppings]
        param_names = [["interlayer","potential parameters"],["intralayer","potential parameters"],
                       ["interlayer","hopping parameters"],["intralayer","hopping parameters"]]

    elif loss=="interlayer energy":

        df = pd.read_csv('../data/qmc.csv')
        interlayer_atoms = []
        if os.path.exists("residual_interlayer_energies.npz"):
            interlayer_residual = np.squeeze(np.load("residual_interlayer_energies.npz")["interlayer_residual"])
        else:
            interlayer_residual = []
        for i, row in df.iterrows():
            print(i)
            atoms = get_bilayer_atoms(row['d'], row['disregistry'])
            if not os.path.exists("residual_interlayer_energies.npz"):
                tb_energy = calc.get_tb_energy(atoms)
                interlayer_residual.append(row["energy"]*len(atoms) - tb_energy)

            interlayer_atoms.append(atoms)
        np.savez("residual_interlayer_energies",interlayer_residual=np.array(interlayer_residual))
        

        models = [calc]
        methods = [calc.get_residual_energy]
        xdata = [interlayer_atoms]
        ydata = [interlayer_residual]
        param_names = [["interlayer","potential parameters"]]
        p0 = [0.719345289329483, 6.074935002291668, 18.184672181803677,
             13.394207130830571, 0.003559135312169, 3.379423382381699,
             1.6724670937654809 ,13.646628785353208, 0.7907544823937784]

    elif loss=="intralayer energy":
        models = [calc]
        methods = [calc.get_residual_energy]
        xdata = [intralayer_atoms]
        ydata = [intralayer_residual]
        param_names = [["intralayer","potential parameters"]]

    popt = fit_model(xdata,ydata,models,methods,p0,param_names)

    
        