import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.optimize
from loguru import logger
from kliff import parallel
from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.dataset.weight import Weight
from kliff.error import report_import_error
from datetime import datetime
import subprocess
import time

from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
#from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI

    mpi4py_avail = True
except ImportError:
    mpi4py_avail = False

try:
    from geodesicLM import geodesiclm

    geodesicLM_avail = True
except ImportError:
    geodesicLM_avail = False

def build_loss_func_kwargs(xdata,ydata,method,method_kwargs={},weights=None):
    def loss_func(params,weights=weights):
        if weights is None:
            weights = 1
        #print("FITTING PARAMETERS  = ",params)
        rmse=0
        yfit_list = []
        yab_list = []
        xdata_list = []
        for i,x in enumerate(xdata):
            yfit = method(x,params,**method_kwargs)
            rmse +=  np.linalg.norm(weights *(ydata[i]-yfit))
            yfit_list = np.append(yfit_list,yfit)
            yab_list = np.append(yab_list,ydata[i])
            
            #xdata_list = np.append(xdata_list,x[:,2])
        if np.isnan(rmse):
            rmse = np.inf
        
        """plt.scatter(np.array(xdata_list),yab_list,label="ab initio")
        plt.scatter(np.array(xdata_list),yfit_list,label="yfit")
        plt.legend()
        plt.savefig("letb_hopping_best_fit.png")
        plt.clf()"""
        return rmse
    return loss_func

def build_loss_func_array(xdata,ydata,method,method_kwargs={},weights=None,return_array=False):
    def loss_func(params,weights=weights):
        if weights is None:
            weights = 1
        #print("FITTING PARAMETERS  = ",params)
        rmse=0
        rmse_array = []
        yfit = method(xdata,params,**method_kwargs)
        tmp_rmse = np.linalg.norm(weights *(ydata-yfit)) 
        rmse +=  tmp_rmse
        rmse_array.append(tmp_rmse)

        if np.isnan(rmse):
            rmse = np.inf
        if return_array:
            return np.array(rmse_array)
        else:
            return rmse
    return loss_func
    
def build_loss_func_TETB(xdata_energy,xdata_tb,
                        ydata_energy,ydata_tb,
                        model,energy_method,tb_method,
                        weight=0.5,fit_params_str=None):

    def loss_func(params,weight=weight):
        """xdata, ydata, model all lists for different types of fitting data """
        
        energy_weight = weight
        tb_weight = 1 - weight

        if fit_params_str is not None:
            model.model_dict[fit_params_str[0]][fit_params_str[1]] = params
            if model.model_dict[fit_params_str[0]]["potential file writer"] is not None:
                
                cwd = os.getcwd()
                os.chdir(model.output)
                model.model_dict[fit_params_str[0]]["potential file writer"](model.model_dict[fit_params_str[0]]["potential parameters"],
                                                                            model.model_dict[fit_params_str[0]]["potential file name"])
                os.chdir(cwd)
        else:
            model.set_params(params)
        print("FITTING PARAMETERS  = ",params)

        #energy RMSE
        yfit_energy = []
        for x in xdata_energy:
            yfit_energy.append(energy_method(x))


        yfit_energy = np.array(yfit_energy)
        y_abinitio_energy = ydata_energy - np.min(ydata_energy)
        y_ab_min_ind = np.argmin(y_abinitio_energy)
        yfit_energy -= yfit_energy[y_ab_min_ind]

        energy_rmse =  np.linalg.norm((y_abinitio_energy[~y_ab_min_ind]-yfit_energy[~y_ab_min_ind])/y_abinitio_energy[~y_ab_min_ind]) / len(ydata_energy)

        #TB RMSE

        interlayer = np.array(np.abs(xdata_tb[:,2]) > 1)
        interlayer_xdata_tb = xdata_tb[interlayer]
        hoppings_fit = tb_method(interlayer_xdata_tb,params[:3])
        tb_rmse = np.linalg.norm((hoppings_fit - ydata_tb[interlayer])/ydata_tb[interlayer]) / len(ydata_tb[interlayer]) /2
        plt.scatter(np.linalg.norm(interlayer_xdata_tb,axis=1),hoppings_fit,c="red")
        plt.scatter(np.linalg.norm(interlayer_xdata_tb,axis=1),ydata_tb[interlayer],c="black")

        intralayer_xdata_tb = xdata_tb[~interlayer]
        hoppings_fit = tb_method(intralayer_xdata_tb,params[-3:])
        tb_rmse += np.linalg.norm((hoppings_fit - ydata_tb[~interlayer])/ydata_tb[~interlayer]) / len(ydata_tb[~interlayer]) /2

        #print("TB RMSE = ",tb_rmse)
        #print("Energy RMSE = ",energy_rmse)
        rmse = energy_weight * energy_rmse + tb_weight * tb_rmse

        if np.isnan(rmse):
            rmse = np.inf
            print("potential not stable")

        return rmse
    return loss_func


def build_Loss_func(xdata,ydata,model,method,weights=None,p0=None,fit_params_ind=None,fit_params_str=None,tb_energy=0):
    def loss_func(params,weights=weights):
        """xdata, ydata, model all lists for different types of fitting data """
        if weights is None:
            weights = np.ones(len(xdata))
        rmse = 0
        if fit_params_ind is not None:
            use_params = p0
            use_params[fit_params_ind] = params 
        else:
            use_params = params
        #print(fit_params_str)
        if fit_params_str is not None:
            model.model_dict[fit_params_str[0]][fit_params_str[1]] = params
            if model.model_dict[fit_params_str[0]]["potential file writer"] is not None:
                
                cwd = os.getcwd()
                os.chdir(model.output)
                model.model_dict[fit_params_str[0]]["potential file writer"](model.model_dict[fit_params_str[0]]["potential parameters"],model.model_dict[fit_params_str[0]]["potential file name"])
                os.chdir(cwd)
        else:
            model.set_params(use_params)
        print("FITTING PARAMETERS  = ",params)

            
        yfit = []
        d = []
        for x in xdata:
            yfit.append(method(x))
            """pos = x.positions
            mean_z = np.mean(pos[:,2])
            top_ind = np.where(pos[:,2]>mean_z)
            bot_ind = np.where(pos[:,2]<mean_z)
            d.append(np.mean(pos[top_ind,2]-pos[bot_ind,2]))
        d = np.array(d)"""
        yfit = np.array(yfit) + tb_energy
        y_abinitio = ydata - np.min(ydata)
        y_ab_min_ind = np.argmin(y_abinitio)
        
        yfit = np.array(yfit)
        yfit -= yfit[y_ab_min_ind]

        rmse +=  np.linalg.norm(weights *(y_abinitio[~y_ab_min_ind]-yfit[~y_ab_min_ind])) #/y_abinitio[~y_ab_min_ind])

        #yabinitio_diff_grid = y_abinitio[:,np.newaxis] - y_abinitio[np.newaxis,:]
        #yfit_diff_grid = yfit[:,np.newaxis] - yfit[np.newaxis,:]
        #rmse += np.linalg.norm((yabinitio_diff_grid - yfit_diff_grid)) 

        """if np.isclose(d[np.argmin(yfit)], np.min(d)) or np.isclose(d[np.argmin(yfit)], np.max(d)):
            #not stable
            rmse = np.inf
            print("potential not stable")
        if np.argmin(yfit) != y_ab_min_ind:

            yfit_min_ind = np.argmin(yfit)
            rmse += 1e6 * np.abs(y_abinitio[yfit_min_ind]-yfit[yfit_min_ind])
            print("potential incorrect minimum")"""
        if np.isnan(rmse):
            rmse = np.inf
            print("potential not stable")
        """

        yabinitio_2nd_diff_grid = (y_abinitio[:,np.newaxis] - 2*y_abinitio[np.newaxis,:])[:,:,np.newaxis] + y_abinitio[np.newaxis,np.newaxis,:]
        yfit_2nd_diff_grid = (yfit[:,np.newaxis] - 2*yfit[np.newaxis,:])[:,:,np.newaxis] + yfit[np.newaxis,np.newaxis,:]
        upper_tri_ind = np.where(yabinitio_diff_grid>1e-5)
        #rmse += np.linalg.norm((yabinitio_2nd_diff_grid[upper_tri_ind] - yfit_2nd_diff_grid[upper_tri_ind])/yabinitio_2nd_diff_grid[upper_tri_ind])"""
        """if np.random.rand() < 0.05:
            
            plt.scatter(d,y_abinitio,label="abinitio")
            plt.scatter(d,yfit,label="fit")
            #plt.scatter(d,np.array(tb_energy) - tb_energy[-1],label="tb energy")
            #plt.ylim((np.min(y_abinitio),np.max(y_abinitio)))
            plt.title("RMSE = "+str(rmse))
            plt.savefig("figures/current_fit_kcinsp.png")
            plt.clf()"""

        return rmse/len(xdata)
    return loss_func


        
class LossModel:
    """
    Loss function class to optimize the physics-based potential parameters.

    Args:
        calculator: Calculator to compute prediction from atomic configuration using
            a potential model.
        nprocs: Number of processes to use..
        residual_fn: function to compute residual, e.g. :meth:`energy_forces_residual`,
            :meth:`energy_residual`, and :meth:`forces_residual`. See the documentation
            of :meth:`energy_forces_residual` for the signature of the function.
            Default to :meth:`energy_forces_residual`.
        residual_data: data passed to ``residual_fn``; can be used to fine tune the
            residual function. Default to
            {
                "normalize_by_natoms": True,
            }
            See the documentation of :meth:`energy_forces_residual` for more.
    """

    scipy_minimize_methods = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]
    scipy_minimize_methods_not_supported_args = ["bounds"]
    scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
    scipy_least_squares_methods_not_supported_args = ["bounds"]

    def __init__(
        self,
        LossFxn,
        calculator,
        nprocs: int = 1,
    ):
        
        self.LossFxn = LossFxn
        self.calculator = calculator #Class that has methods .get_num_opt_params(), .get_opt_params_bounds(), .get_opt_params()
        self.nprocs = nprocs


        logger.debug(f"`{self.__class__.__name__}` instantiated.")

    def minimize(self, method: str = "L-BFGS-B", **kwargs):
        """
        Minimize the loss.

        Args:
            method: minimization methods as specified at:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

            kwargs: extra keyword arguments that can be used by the scipy optimizer
        """
        kwargs = self._adjust_kwargs(method, **kwargs)

        logger.info(f"Start minimization using method: {method}.")
        result = self._scipy_optimize(method, **kwargs)
        logger.info(f"Finish minimization using method: {method}.")

        # update final optimized parameters
        #self.calculator.update_model_params(result.x)

        return result

    def _adjust_kwargs(self, method, **kwargs):
        """
        Check kwargs and adjust them as necessary.
        """

        if method in self.scipy_least_squares_methods:
            # check support status
            for i in self.scipy_least_squares_methods_not_supported_args:
                if i in kwargs:
                    raise LossError(
                        f"Argument `{i}` should not be set via the `minimize` method. "
                        "It it set internally."
                    )

            # adjust bounds
            if self.calculator.has_opt_params_bounds():
                if method in ["trf", "dogbox"]:
                    bounds = self.calculator.get_opt_params_bounds()
                    lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
                    ub = [b[1] if b[1] is not None else np.inf for b in bounds]
                    bounds = (lb, ub)
                    kwargs["bounds"] = bounds
                else:
                    raise LossError(f"Method `{method}` cannot handle bounds.")

        elif method in self.scipy_minimize_methods:
            # check support status
            for i in self.scipy_minimize_methods_not_supported_args:
                if i in kwargs:
                    raise LossError(
                        f"Argument `{i}` should not be set via the `minimize` method. "
                        "It it set internally."
                    )

            # adjust bounds
            if isinstance(self.calculator, _WrapperCalculator):
                calculators = self.calculator.calculators
            else:
                calculators = [self.calculator]
            for calc in calculators:
                if calc.has_opt_params_bounds():
                    if method in ["L-BFGS-B", "TNC", "SLSQP"]:
                        bounds = self.calculator.get_opt_params_bounds()
                        kwargs["bounds"] = bounds
                    else:
                        raise LossError(f"Method `{method}` cannot handle bounds.")
        else:
            raise LossError(f"Minimization method `{method}` not supported.")

        return kwargs

    def _scipy_optimize(self, method, **kwargs):
        """
        Minimize the loss use scipy.optimize.least_squares or scipy.optimize.minimize
        methods. A user should not call this function, but should call the ``minimize``
        method.
        """

        size = parallel.get_MPI_world_size()

        if size > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            logger.info(f"Running in MPI mode with {size} processes.")

            if self.nprocs > 1:
                logger.warning(
                    f"Argument `nprocs = {self.nprocs}` provided at initialization is "
                    f"ignored. When running in MPI mode, the number of processes "
                    f"provided along with the `mpiexec` (or `mpirun`) command is used."
                )

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                # geodesic LM
                if method == "geodesiclm":
                    if not geodesicLM_avail:
                        report_import_error("geodesiclm")
                    else:
                        minimize_fn = geodesiclm
                else:
                    minimize_fn = scipy.optimize.least_squares
                func = self._get_loss

            elif method in self.scipy_minimize_methods:
                minimize_fn = scipy.optimize.minimize
                func = self._get_loss_MPI

            if rank == 0:
                result = minimize_fn(func, x, method=method, **kwargs)
                # notify other process to break func
                break_flag = True
                for i in range(1, size):
                    comm.send(break_flag, dest=i, tag=i)
            else:
                func(x)
                result = None

            result = comm.bcast(result, root=0)

            return result

        else:
            # 1. running MPI with 1 process
            # 2. running without MPI at all
            # both cases are regarded as running without MPI

            if self.nprocs == 1:
                logger.info("Running in serial mode.")
            else:
                logger.info(
                    f"Running in multiprocessing mode with {self.nprocs} processes."
                )

                # Maybe one thinks he is using MPI because nprocs is used
                if mpi4py_avail:
                    logger.warning(
                        "`mpi4py` detected. If you try to run in MPI mode, you should "
                        "execute your code via `mpiexec` (or `mpirun`). If not, ignore "
                        "this message."
                    )

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                if method == "geodesiclm":
                    if not geodesicLM_avail:
                        report_import_error("geodesiclm")
                    else:
                        minimize_fn = geodesiclm
                else:
                    minimize_fn = scipy.optimize.least_squares

                func = self._get_loss
            elif method in self.scipy_minimize_methods:
                minimize_fn = scipy.optimize.minimize
                func = self._get_loss

            result = minimize_fn(func, x, method=method, **kwargs)
            return result

    def _get_loss(self, x):
        """
        Compute the loss in serial or multiprocessing mode.

        This is a callable for optimizing method in scipy.optimize.minimize,
        which is passed as the first positional argument.

        Args:
            x: 1D array, optimizing parameter values
        """
        #start = time.time()
        loss = self.LossFxn(x)
        #end = time.time()
        #print("time for loss function  = ",end-start)

        return loss

class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg

