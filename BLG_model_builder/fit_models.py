import numpy as np
import scipy.optimize 
from TB_Utils import *
from Lammps_Utils import *
from descriptors import *
from TETB_model_builder import TETB_model

def build_Loss_func(xdata,ydata,models,param_names,weights=None):
    def loss_func(params):
        """xdata, ydata, model all lists for different types of fitting data """
        if weights is None:
            weights = 1/len(models)
        rmse = 0
        for i,m in enumerate(models):
            m[param_names[i][0]][param_names[i][1]] = params
            yfit = []
            for x in xdata:
                yfit.append(m(xdata))
            yfit = np.arrary(yfit)
            rmse += weights[i] * np.linalg.norm((ydata-yfit)/ydata)

        return rmse
    return loss_func

def fit_model(xdata,ydata,models,p0,param_names,bounds=None,weights=None):
    Loss_func = build_Loss_func(xdata,ydata,models,param_names,weights=weights)
    popt,pcov = scipy.optimize.minimize(Loss_func,p0, method="Nelder-Mead",bounds=bounds)
    return popt

if __name__=="__main__":

    model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                "hopping descriptors":"displacement",
                                "potential":"reg/dep/poly 10.0 0","potential parameters":[],"potential file writer":write_kcinsp},

                "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":np.array([-2.7, 2.2109794066373403, 0.48]),"overlap parameters":None,
                                "hopping descriptors":"displacement",
                                "potential":"airebo 3","potential parameters":[],"potential file writer":write_rebo}}
    
    calc = TETB_model(model_dict)

    loss = "interlayer energy"
    if loss=="total":
        models = [calc.get_total_energy,calc.get_total_energy,calc.get_hoppings,calc.get_hoppings]
        xdata = [interlayer_atoms, intralayer_atoms, interlayer_disp, intralayer_disp ]
        ydata = [interlayer_energy, intralayer_energy, interlayer_hoppings, intralayer_hoppings]
        param_names = [["interlayer","potential parameters"],["intralayer","potential parameters"],
                       ["interlayer","hopping parameters"],["intralayer","hopping parameters"]]

    elif loss=="interlayer energy":
        models = [calc.get_total_energy]
        xdata = [interlayer_atoms]
        ydata = [interlayer_energy]
        param_names = ["interlayer potential parameters"]

    elif loss=="intralayer energy":
        models = [calc.get_total_energy]
        xdata = [intralayer_atoms]
        ydata = [intralayer_energy]
        param_names = ["intralayer potential parameters"]

    popt = fit_model(xdata,ydata,models,param_names)

    
        