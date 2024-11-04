import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
import ase.db
import time
import matplotlib.pyplot as plt

from KLIFF_LOSS import *
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import run_BLG_uq
import emcee

def _get_loglikelihood(xdata,ydata,ase_calc,methods):
    """Compute the log-likelihood from the cost function. It has an option to temper the
    cost by specifying ``T``.
    """
    T = 1
    LossFxn = build_Loss_func(xdata,ydata,ase_calc,methods)
    def loglikelihood(x):
        cost = LossFxn(x)
        logl = -cost / T
        return logl
    return loglikelihood

if __name__=="__main__":
        nkp = 121
        bound_frac = 0.25
        interlayer_db =  ase.db.connect('../data/bilayer_nkp'+str(nkp)+'.db')
        intralayer_db = db = ase.db.connect('../data/monolayer_nkp'+str(nkp)+'.db')
        ase_atom_list,energies = run_BLG_uq.create_Dataset(interlayer_db,intralayer_db)
        rebo_params = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])

        kc_params = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                                        0.719345289329483, 3.293082477932360, 13.906782892134125])


        model_dict = {"interlayer":{"hopping form":None,
                                "potential":"kolmogorov/crespi/full 10.0 0","potential parameters":kc_params,
                                "potential file writer":write_kc},

                        "intralayer":{"hopping form":None,
                                "potential":"rebo","potential parameters":rebo_params,"potential file writer":write_rebo}}
        param_names = ['z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A','Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2', 'BIJc_CC3','Beta_CC1', 
                'Beta_CC2','Beta_CC3']
        ase_calc = TETB_model(model_dict) 
        opt_param = ase_calc.get_params()

        xdata = [ase_atom_list]
        ydata = [energies]
        methods = [ase_calc.get_total_energy]
        
        ndim = len(opt_param)
        ase_calc.set_params(opt_param)
        
        log_prob_fn = _get_loglikelihood(xdata,ydata,ase_calc,methods)
        nwalkers = 2*ndim
        nsteps = 100

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)
        sampler.run_mcmc(np.tile(opt_param,(nwalkers,1)), nsteps)
        log_probs = sampler.get_log_prob(flat=True)
        np.savez("log_probs",log_probs=log_probs)
        costs = -1*log_probs
        plt.scatter(np.arange(len(costs)),costs)
        plt.xlabel("mcmc step")
        plt.ylabel("Cost")
        plt.title("mcmc chain")
        plt.savefig("mcmc_cost_chain.png")
        plt.clf()
        exit()
        
        nsamples = 50
        loss = np.zeros(nsamples)
        LossFxn = build_Loss_func(xdata,ydata,ase_calc,methods)
        loss[0] = LossFxn(opt_param)
        param_dev = np.zeros(nsamples)
        for n in range(1,nsamples):

            new_param = opt_param
            param_pert = np.random.uniform(1-bound_frac,1+bound_frac,size=ndim)
            param_dev[n] = np.mean(1-param_pert)
            new_param *= param_pert
            ase_calc.set_params(new_param)
            LossFxn = build_Loss_func(xdata,ydata,ase_calc,methods)
            loss[n] = LossFxn(new_param)

        plt.scatter(param_dev,loss)
        plt.xlabel("arb. param deviation percent")
        plt.ylabel("Cost")
        plt.title("Cost surface")
        plt.savefig("cost_surface_full.png")
        plt.clf()
        exit()
        
        nsamples = 15
        frac_vec = np.linspace(1-bound_frac,1+bound_frac,nsamples)
        for i in range(ndim):
            loss = np.zeros_like(frac_vec)
            for j,frac in enumerate(frac_vec):
                  new_param = opt_param
                  new_param[i] *= frac
                  ase_calc.set_params(new_param)
                  LossFxn = build_Loss_func(xdata,ydata,ase_calc,methods)
                  loss[j] = LossFxn(new_param)
            plt.plot(opt_param[i]*frac_vec,loss)
            plt.title(param_names[i]+" cost surface")
            plt.xlabel(param_names[i])
            plt.ylabel("Cost")
            plt.savefig(param_names[i]+"_cost_surface.png")
            plt.clf()

