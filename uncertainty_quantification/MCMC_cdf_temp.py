import numpy as np
import subprocess
import scipy.stats
import scipy.special
from run_BLG_uq import *
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.descriptors import *
import matplotlib as mpl

def cdf(r,rmse,ensemble):
    #r is confidence interval, rmse is standard deviation i.e. 64% confidence interval
    z = scipy.stats.norm.ppf((1 + r) / 2)
    npoints = np.shape(ensemble)[1]
    contained = 0
    for n in range(npoints):
        mean = np.mean(ensemble[:,n])
        contained_ind = np.where( (ensemble[:,n] <= (mean + z*rmse[n])) & (ensemble[:,n] >= (mean - z*rmse[n])))
        contained += len((contained_ind)[0])

    contained_percent = contained / np.prod(np.shape(ensemble))
    #print("contained Percent for confidence interval "+str(r)+" = "+str(contained_percent))
    return contained_percent


if __name__=="__main__":
    T_ = [0.1,0.25,0.5,1.0,1.5,2,4,6,8,10,15,20,25,30,35,40,50] #,60,75,100,200,300,400,500]) #,"best_fit_rmse"
    #T_ = np.array([1.0,2,3,4,5,6,7,8,9,10])
    r = np.linspace(0.05,0.99,15)
    z = scipy.stats.norm.ppf((1 + r) / 2)
    best_fit_params = np.array([-2.7, 2.2109794066373403, 0.48])
    expected_cdf = scipy.special.erf(r/np.sqrt(2))

    hopping_data = hopping_training_data(hopping_type="all")
    xdata_list = hopping_data["disp"]
    ydata_list = hopping_data["hopping"]
    xdata = xdata_list[0]
    ydata = ydata_list[0]
    for i in range(1,len(xdata_list)):
        xdata = np.vstack((xdata,xdata_list[i]))
        ydata = np.append(ydata,ydata_list[i])

    best_fit_hoppings = mk_hopping(xdata,best_fit_params)
    best_fit_rmse = np.sqrt(np.power(best_fit_hoppings - ydata,2))

    cdf_diff_integral = np.zeros_like(T_)
    tb_model = "TB_MK"
    for i,T_val in enumerate(T_):

        executable = "python run_BLG_uq.py -u "+tb_model+" -T "+str(T_val)
        subprocess.call(executable,shell=True)


        ensemble = np.load("../uncertainty_quantification/ensembles/"+tb_model+"_interlayer_mcmc_T_"+str(T_val)+"_ensemble.npz")["ensembles"]
        n_ensembles = np.shape(ensemble)[0]

        ensemble_hoppings = np.zeros((n_ensembles,len(ydata)))
        for n in range(n_ensembles):
            tb_params = ensemble[n,:]
            ensemble_hoppings[n,:] = mk_hopping(xdata,tb_params)
        mean_hoppings = np.mean(ensemble_hoppings,axis=0)
        std_hoppings = np.std(ensemble_hoppings,axis=0)
        calc_cdf = np.zeros_like(r)
        for j,r_val in enumerate(r):
            calc_cdf[j] = cdf(r_val, best_fit_rmse, ensemble_hoppings )

        cdf_diff_integral[i] = np.mean(np.abs(calc_cdf - scipy.special.erf(z/np.sqrt(2))))
        if type(T_val)==str:
            plt.plot(z,calc_cdf,label="calculated cdf T = "+str(T_val)+"*T0", color="red")
        else:    
            plt.plot(z,calc_cdf,label="calculated cdf T = "+str(T_val)+"*T0", c = mpl.colormaps["plasma"](T_val/50))
    #plt.colorbar()
    plt.plot(z,scipy.special.erf(z/np.sqrt(2)),label="expected cdf",marker="*",color="black")
    #plt.legend()
    plt.xlabel("z")
    plt.ylabel("CDF")
    plt.savefig("figures/"+tb_model+"_ensemble_cdf.png")
    plt.clf()

    plt.plot(T_,cdf_diff_integral)
    plt.xlabel("Temperature")
    plt.ylabel("CDF integral deviation")
    plt.savefig("figures/"+tb_model+"_ensemble_cdf_integral_deviation.png")
    plt.clf()

    for i,T_val in enumerate(T_):

        ensemble = np.load("../uncertainty_quantification/ensembles/"+tb_model+"_interlayer_mcmc_T_"+str(T_val)+"_ensemble.npz")["ensembles"]
        n_ensembles = np.shape(ensemble)[0]

        ensemble_hoppings = np.zeros((n_ensembles,len(ydata)))
        for n in range(n_ensembles):
            tb_params = ensemble[n,:]
            ensemble_hoppings[n,:] = mk_hopping(xdata,tb_params)
        mean_hoppings = np.mean(ensemble_hoppings,axis=0)
        std_hoppings = np.std(ensemble_hoppings,axis=0)
        if type(T_val)==str:
            plt.scatter(best_fit_rmse,std_hoppings,label="T = "+str(T_val)+"*T0", color = "red",s=0.5)
        else:
            plt.scatter(best_fit_rmse,std_hoppings,label="T = "+str(T_val)+"*T0", c = mpl.colormaps["plasma"](T_val/50),s=0.5)
    #plt.colorbar()
    best_fit_rmse = np.sort(best_fit_rmse)
    plt.plot(best_fit_rmse,best_fit_rmse,label="expected cdf",color="black")
    #plt.legend()
    plt.xlabel("RMSE")
    plt.ylabel("RMV")
    plt.xlim(0,np.max(best_fit_rmse))
    plt.ylim(0,np.max(best_fit_rmse))
    plt.savefig("figures/"+tb_model+"_ensemble_rmse_rmv.png")
    plt.clf()

    
    