import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
from scipy.optimize import curve_fit
import scipy.special
import scipy.stats

def quadratic(x,a,b,c):
    return a*(x-b)**2 +c

def get_interlayer_sep(d,energies):
    min_ind = np.argmin(energies)
    c0 = np.min(energies)
    b0 = d[min_ind]
    a0 = 1
    popt,pcov = curve_fit(quadratic,d[min_ind-1:min_ind+2],energies[min_ind-1:min_ind+2],p0=[a0,b0,c0])
    sep = popt[1]

    return sep

def hopping_training_data(hopping_type="interlayer"):
    data = []
    flist = glob.glob('../data/hoppings/*.hdf5',recursive=True)
    eV_per_hart=27.2114
    hoppings = np.zeros((1,1))
    disp_array = np.zeros((1,3))
    for f in flist:
        if ".hdf5" in f:
            with h5py.File(f, 'r') as hdf:
                # Unpack hdf
                lattice_vectors = np.array(hdf['lattice_vectors'][:]) #* 1.88973
                atomic_basis =    np.array(hdf['atomic_basis'][:])    #* 1.88973
                tb_hamiltonian = hdf['tb_hamiltonian']
                tij = np.array(tb_hamiltonian['tij'][:]) #* eV_per_hart
                di  = np.array(tb_hamiltonian['displacementi'][:])
                dj  = np.array(tb_hamiltonian['displacementj'][:])
                ai  = np.array(tb_hamiltonian['atomi'][:])
                aj  = np.array(tb_hamiltonian['atomj'][:])
                displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]
                
            hoppings = np.append(hoppings,tij)
            disp_array = np.vstack((disp_array,displacement_vector)) 
    hoppings = hoppings[1:]
    disp_array = disp_array[1:,:]
    if hopping_type=="interlayer":
        type_ind = np.where(disp_array[:,2] > 1) # Inter-layer hoppings only, allows for buckling
    else:
        type_ind = np.where(disp_array[:,2] < 1)
    return {"hopping":hoppings[type_ind],"disp":disp_array[type_ind]}

def plot_parameter_dist(ensembles):
    # dimension = (nwalkers, niterations, nparams)
    n_params = np.shape(ensembles)[1]
    param_names = ['z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A','Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2','BIJc_CC3', 'Beta_CC1', 
            'Beta_CC2', 'Beta_CC3'] 

    #param_names = [ 'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'] 
    
    params0 = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923,3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,0.719345289329483, 3.293082477932360, 13.906782892134125])
    for n in range(n_params):
        param_dist = ensembles[:,n]
        
        #plt.hist(param_dist.flatten(),bins = 50,range = (0.5*(params0[n])-params0[n], params0[n]+0.5*(params0[n])), density=True,histtype="stepfilled")
        #plt.hist(param_dist,bins = 50,range = (params0[n]-3*(np.std(param_dist)), params0[n]+3*(np.std(param_dist))), density=True,histtype="stepfilled",stacked=True)
        plt.hist(param_dist,density=True,histtype="stepfilled",stacked=True)
        if n==0:
            plt.xlim(3.3777,3.382)
        plt.title(param_names[n]+" Distribution")
        plt.ylabel("counts/ total counts")
        plt.xlabel(param_names[n]+" value")
        plt.savefig("figures/MCMC"+param_names[n]+"histogram.png")
        plt.clf()
        #exit()

def get_gsfe_graphene(calc,calc_tb_energy=False):
    stacking_ = ["AB","SP","Mid","AA"]
    disreg_ = [0 , 0.16667, 0.5, 0.66667]
    df = pd.read_csv('../data/qmc.csv') 

    model_energy = []
    tb_energy = [] 
    d_ = []
    for i, row in df.iterrows():
        #if row["d"] > 5.01:
        #    continue
        d_.append(row["d"])
        atoms = get_bilayer_atoms(row["d"],row["disregistry"])
        total_energy = (calc.get_total_energy(atoms))/len(atoms)

        model_energy.append(total_energy)
        if calc_tb_energy:
            tmp_tb_energy = calc.get_tb_energy(atoms)/len(atoms)
            tb_energy.append(tmp_tb_energy)
    model_energy = np.array(model_energy)
    model_energy -= model_energy[-1] #np.min(model_energy)
    df["model energy"] = model_energy - np.min(model_energy)
    interlayer_sep = get_interlayer_sep(d_,model_energy)
    
    #df["energy"] -= np.min(df["energy"])

    if calc_tb_energy:
        df["tb energy"] =np.array(tb_energy) - np.min(np.array(tb_energy))
    qmc_energy = df["energy"].to_numpy()
    qmc_energy -= np.min(qmc_energy)
    return qmc_energy, model_energy, interlayer_sep

def plot_gsfe(qmc_energies,model_energies,std,filename):
    stacking_ = ["AB","SP","Mid","AA"]
    colors = ["blue","red","black","green"]
    nd = 11
    qmc_ind = 0
    model_ind = 1
    d_ = np.array([3,3.2,3.35,3.5,3.65,3.8,4,4.5,5,6,7])
    stacking_eq_layer_sep = []
    for i,s in enumerate(stacking_):

        qmc_energy_stack = qmc_energies[int(i*nd) : int((i+1)*nd)] #- qmc_energies[int((i+1)*nd)-1]
        model_energy_stack = model_energies[int(i*nd) : int((i+1)*nd)] - np.min(model_energies) #[int((i+1)*nd)-1]
        std_energy_stack = std[int(i*nd) : int((i+1)*nd)]

        plt.scatter(d_,qmc_energy_stack,label=s + " qmc",c=colors[i])
        plt.plot(d_,model_energy_stack,c=colors[i],label = s+" model")
        plt.fill_between(d_, model_energy_stack-std_energy_stack, model_energy_stack+std_energy_stack ,alpha=0.3, facecolor=colors[i])
    plt.legend()
    plt.xlabel(r"Interlayer Separation $\AA$")
    plt.ylabel("Interlayer Energy (eV)")
    plt.title("Classical KC potential UQ for Bilayer Graphene with Uncertainty")
    print("saving figure")
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()

    for i,s in enumerate(stacking_):

        qmc_energy_stack = qmc_energies[int(i*nd) : int((i+1)*nd)] #- qmc_energies[int((i+1)*nd)-1]
        model_energy_stack = model_energies[int(i*nd) : int((i+1)*nd)] - np.min(model_energies) #[int((i+1)*nd)-1]
        std_energy_stack = std[int(i*nd) : int((i+1)*nd)]

        plt.plot(d_,std_energy_stack,c=colors[i],label = s+" estimated model uncertainty")
        plt.plot(d_,-std_energy_stack,c=colors[i])
        plt.errorbar(d_,np.zeros(len(d_)),yerr=np.abs(qmc_energy_stack-model_energy_stack),label=s+" True model error",c="black")
        plt.fill_between(d_, -std_energy_stack, std_energy_stack ,alpha=0.3, facecolor=colors[i])
        plt.xlabel(r"Interlayer Separation $\AA$")
        plt.ylabel("Interlayer Energy Uncertainty (eV)")
        plt.legend()
        plt.tight_layout()
        plt.title(s+" Stacking")
        plt.savefig("figures/"+s+"_stacking_uq.png",bbox_inches='tight')
        plt.clf()
    return stacking_eq_layer_sep

def get_fermi_energy(eigvals):
    nvals = np.shape(eigvals)[0]
    lumo = np.min(eigvals[nvals//2,:])
    homo = np.max(eigvals[nvals//2-1,:])
    return (homo+lumo)/2

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

    import os
    uq_type = "MCMC"
    model_type="interlayer"
    

    interlayer_potential = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                              0.719345289329483, 3.293082477932360, 13.906782892134125])
    intralayer_potential = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])
    if model_type=="interlayer":

        model_dict = {"interlayer":{"hopping form":None,
                            "potential":"kolmogorov/crespi/full 10.0 0","potential parameters":interlayer_potential,
                            "potential file writer":write_kc}}

    elif model_type=="intralayer":

        model_dict = {"intralayer":{"hopping form":None,
                            "potential":"rebo","potential parameters":intralayer_potential,"potential file writer":write_rebo}}
    
    calc = TETB_model(model_dict)
    param0 = calc.get_params()
    

    if uq_type=="MCMC":
        ensembles = np.load("../uncertainty_quantification/ensembles/Classical_energy_interlayer_mcmc_ensemble.npz")["ensembles"]
        
        n_ensembles =  np.shape(ensembles)[0]
        nparams = np.shape(ensembles)[1]
        moving_std=np.zeros((n_ensembles-1,nparams))
        for i in range(1,n_ensembles-1):
            moving_std[i,:] = np.std(ensembles[:i,:],axis=0)
        for j in range(nparams):
            plt.plot(moving_std[:,j]/moving_std[-1,j])
        plt.xlabel("MCMC step")
        plt.ylabel(r"$\sigma(\theta)$")
        plt.savefig("figures/std_params.png")
        plt.clf()
    elif uq_type=="bootstrap":
        ensembles = np.load("../uncertainty_quantification/ensembles/Classical_energy_interlayer_bootstrap_ensemble.npz")["samples"]

    elif uq_type=="cv":
        ensembles = np.load("../uncertainty_quantification/ensembles/Classical_energy_interlayer_Kfold_n_8_ensemble.npz")["samples"]

    n_ensembles =  np.shape(ensembles)[0]
    plot_parameter_dist(ensembles)



    classical=False
    TB_popov=False
    TB_MK=False
    TB_LETB=True

    if classical:
        if os.path.exists("classical_ensemble_gsfe.npz"):
                
            data = np.load("classical_ensemble_gsfe.npz")
            ensemble_gsfe = data["ensemble_gsfe"]
            qmc_gsfe = data["qmc_gsfe"]
            ensemble_interlayer_sep = data["ensemble_interlayer_sep"]

        else:
            ensemble_gsfe = []
            ensemble_interlayer_sep = []
            for i in range(n_ensembles):
                calc.set_params(ensembles[i,:])
                qmc_gsfe,model_gsfe,interlayer_sep = get_gsfe_graphene(calc)
                ensemble_gsfe.append(model_gsfe)
                ensemble_interlayer_sep.append(interlayer_sep)
            ensemble_gsfe = np.array(ensemble_gsfe)
            
            np.savez("classical_ensemble_gsfe",qmc_gsfe=qmc_gsfe,ensemble_gsfe=ensemble_gsfe,ensemble_interlayer_sep=ensemble_interlayer_sep)

        print("AB interlayer separation mean, std",np.mean(ensemble_interlayer_sep),np.std(ensemble_interlayer_sep))
        ensemble_gsfe_std = np.std(ensemble_gsfe,axis=0)
        ensemble_gsfe_mean = np.mean(ensemble_gsfe,axis=0)
        filename = "figures/blg_classical_"+uq_type+"_uq.png"
        plot_gsfe(qmc_gsfe,ensemble_gsfe_mean,ensemble_gsfe_std,filename)

        calc.set_params(interlayer_potential)
        qmc_energy, model_energy, _ = get_gsfe_graphene(calc)
        best_fit_rmse = np.sqrt(np.power(qmc_energy - model_energy,2))

        plt.scatter(np.array(best_fit_rmse),ensemble_gsfe_std,label="calibration curve")
        plt.plot(np.linspace(0,np.max(ensemble_gsfe_std),10), np.linspace(0,np.max(best_fit_rmse),10) ,label="perfect calibration")
        plt.ylabel(r"$\sigma^{2}$")
        plt.xlabel("Best Fit RMSE")
        plt.title("Error based calibration curve, "+uq_type+" ensemble")
        plt.savefig("figures/"+uq_type+"_interlayer_ensemble_error_calibration_curve.png")
        plt.clf()


        r = np.linspace(0.05,0.99,15)
        z = scipy.stats.norm.ppf((1 + r) / 2)
        expected_cdf = scipy.special.erf(r/np.sqrt(2))
        calc_cdf = np.zeros_like(r)
        for j,r_val in enumerate(r):
            calc_cdf[j] = cdf(r_val, best_fit_rmse, ensemble_gsfe)

        plt.plot(z,scipy.special.erf(z/np.sqrt(2)),label="expected cdf",marker="*",color="black")
        plt.plot(z,calc_cdf,label="calculated cdf")
        
        plt.legend()
        plt.xlabel("z")
        plt.ylabel("CDF")
        plt.savefig("figures/"+uq_type+"_interlayer_ensemble_cdf_curve.png")
        plt.clf()


    if TB_popov:
        bondint_ensembles={'popov_hopping_pp_sigma' :  np.load("../uncertainty_quantification/ensembles/TB_sk_interlayer_mcmc_popov_hopping_pp_sigma_ensemble.npz")["ensembles"],
                        'popov_hopping_pp_pi': np.load("../uncertainty_quantification/ensembles/TB_sk_interlayer_mcmc_popov_hopping_pp_pi_ensemble.npz")["ensembles"],
                        'popov_overlap_pp_pi': np.load("../uncertainty_quantification/ensembles/TB_sk_interlayer_mcmc_popov_overlap_pp_pi_ensemble.npz")["ensembles"],
                        'popov_overlap_pp_sigma': np.load("../uncertainty_quantification/ensembles/TB_sk_interlayer_mcmc_popov_overlap_pp_sigma_ensemble.npz")["ensembles"],  
                        'porezag_hopping_pp_sigma':  np.load("../uncertainty_quantification/ensembles/TB_sk_intralayer_mcmc_porezag_hopping_pp_sigma_ensemble.npz")["ensembles"],
                        'porezag_hopping_pp_pi':  np.load("../uncertainty_quantification/ensembles/TB_sk_intralayer_mcmc_porezag_hopping_pp_pi_ensemble.npz")["ensembles"],
                        'porezag_overlap_pp_pi': np.load("../uncertainty_quantification/ensembles/TB_sk_intralayer_mcmc_porezag_overlap_pp_pi_ensemble.npz")["ensembles"],
                        'porezag_overlap_pp_sigma': np.load("../uncertainty_quantification/ensembles/TB_sk_intralayer_mcmc_porezag_overlap_pp_sigma_ensemble.npz")["ensembles"]} 

        atoms = get_bilayer_atoms(3.35,0)
        n_ensembles = np.shape(bondint_ensembles["popov_hopping_pp_sigma"])[0]
        Gamma = [0,   0,   0]
        K = [1/3,2/3,0]
        Kprime = [2/3,1/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nkp=60

        eigvals_ensemble = np.zeros((n_ensembles,len(atoms),nkp+1))
        interlayer_hopping_fxn = SK_pz_chebyshev
        interlayer_overlap_fxn = SK_pz_chebyshev
        intralayer_hopping_fxn = SK_pz_chebyshev
        intralayer_overlap_fxn = SK_pz_chebyshev
        for i in range(n_ensembles):
            interlayer_hopping_params = np.append(bondint_ensembles['popov_hopping_pp_sigma'][i,:],bondint_ensembles['popov_hopping_pp_pi'][i,:])
            interlayer_overlap_params = 0*np.append(bondint_ensembles['popov_overlap_pp_sigma'][i,:],bondint_ensembles['popov_overlap_pp_pi'][i,:])
            intralayer_hopping_params = np.append(bondint_ensembles['porezag_hopping_pp_sigma'][i,:],bondint_ensembles['porezag_hopping_pp_pi'][i,:])
            intralayer_overlap_params = 0*np.append(bondint_ensembles['porezag_overlap_pp_sigma'][i,:],bondint_ensembles['porezag_overlap_pp_pi'][i,:])

            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177},"cutoff":5.29177},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239},"cutoff":3.704239}}

            calc = TETB_model(model_dict)
            (kvec,k_dist, k_node) = k_path(sym_pts,nkp)
            eigvals = calc.get_band_structure(atoms,kvec)
            eigvals_ensemble[i,:,:] = eigvals - get_fermi_energy(eigvals)
        
        mean_eigvals = np.mean(eigvals_ensemble,axis=0)
        std_eigvals = np.std(eigvals_ensemble,axis=0)


        fig, ax = plt.subplots()

        label=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(k_node[0],k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        # add vertical lines at node positions
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n],linewidth=0.5, color='k')
        # put title
        ax.set_title("AB bilayer graphene, popov UQ")
        ax.set_xlabel("Path in k-space")
        
        nbands = len(atoms)

        for n in range(nbands):
            ax.plot(k_dist,mean_eigvals[n,:],c="red")
            ax.fill_between(k_dist,mean_eigvals[n,:]-std_eigvals[n,:],mean_eigvals[n,:]+std_eigvals[n,:],alpha=0.3,label="std")
        print("mean error for eigval ",np.mean(std_eigvals))
        # make an PDF figure of a plot
        #fig.tight_layout()
        erange = 2
        ax.set_ylim(-erange,erange)
        ax.set_ylabel(r'$E - E_F$ (eV)')
        ax.set_xticks(k_node)
        ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
        ax.set_xlim(k_node[0], k_node[-1])
        fig.savefig("ab_graphene_band_structure_popov_uq.png")
        plt.clf()

    if TB_MK:
        csfont = {'fontname':'serif',"size":18}
        interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
        intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")
        xdata = np.concatenate((interlayer_hopping_data['disp'],intralayer_hopping_data['disp']),axis=0)
        ydata = np.concatenate((interlayer_hopping_data['hopping'],intralayer_hopping_data["hopping"]),axis=0)

        ensemble = np.load("../uncertainty_quantification/ensembles/TB_MK_interlayer_mcmc_ensemble.npz")["ensembles"]
        n_ensembles = np.shape(ensemble)[0]

        ensemble_hoppings = np.zeros((n_ensembles,len(ydata)))
        for n in range(n_ensembles):
            tb_params = ensemble[n,:]
            ensemble_hoppings[n,:] = mk_hopping(xdata,tb_params)
        mean_hoppings = np.mean(ensemble_hoppings,axis=0)
        std_hoppings = np.std(ensemble_hoppings,axis=0)
        plt.scatter(np.linalg.norm(xdata,axis=1),mean_hoppings,c=std_hoppings)
        plt.xlabel(r"r $(\AA)$",**csfont)
        plt.ylabel("<t> (eV)",**csfont)
        plt.colorbar()
        plt.title("MK hopping mean value",**csfont)
        plt.savefig("figures/mk_hopping_uncertainty.png", bbox_inches = 'tight')
        plt.clf()

        plt.scatter(np.linalg.norm(xdata,axis=1),std_hoppings)
        plt.xlabel(r"r $(\AA)$",**csfont)
        plt.ylabel(r"$\sigma(t) (eV)$",**csfont)
        plt.title("MK hopping uncertainty",**csfont)
        plt.savefig("figures/mk_hopping_uncertainty_std.png", bbox_inches = 'tight')
        plt.clf()


        atoms = get_bilayer_atoms(3.35,0)
        
        Gamma = [0,   0,   0]
        K = [1/3,2/3,0]
        Kprime = [2/3,1/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nkp=60

        eigvals_ensemble = np.zeros((n_ensembles,len(atoms),nkp+1))
        for i in range(n_ensembles):
            cutoff = 10
            mk_params = ensemble[i,:]
            model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":cutoff}},
                    "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":cutoff}}}

            calc = TETB_model(model_dict)
            (kvec,k_dist, k_node) = k_path(sym_pts,nkp)
            eigvals = calc.get_band_structure(atoms,kvec)
            eigvals_ensemble[i,:,:] = eigvals - get_fermi_energy(eigvals)
        
        mean_eigvals = np.mean(eigvals_ensemble,axis=0)
        std_eigvals = np.std(eigvals_ensemble,axis=0)


        fig, ax = plt.subplots()

        label=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(k_node[0],k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        # add vertical lines at node positions
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n],linewidth=0.5, color='k')
        # put title
        ax.set_title("AB bilayer graphene, MK UQ")
        ax.set_xlabel("Path in k-space")
        
        nbands = len(atoms)

        for n in range(nbands):
            ax.plot(k_dist,mean_eigvals[n,:],c="red")
            ax.fill_between(k_dist,mean_eigvals[n,:]-std_eigvals[n,:],mean_eigvals[n,:]+std_eigvals[n,:],alpha=0.3,label="std")
        print("mean error for eigval ",np.mean(std_eigvals))
        # make an PDF figure of a plot
        #fig.tight_layout()
        erange = 2
        ax.set_ylim(-erange,erange)
        ax.set_ylabel(r'$E - E_F$ (eV)')
        ax.set_xticks(k_node)
        ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
        ax.set_xlim(k_node[0], k_node[-1])
        fig.savefig("figures/ab_graphene_band_structure_MK_uq.png")
        plt.clf()

        dist_to_fermi = np.abs(mean_eigvals)
        plt.scatter(dist_to_fermi,std_eigvals,s=5)
        plt.xlabel(r"$|\epsilon_{ik} - E_{f}|$ (eV)")
        plt.ylabel(r"$\sigma(\epsilon_{ik})$ (eV)")
        plt.savefig("figures/MK_dist_to_fermi_level_std.png")
        plt.clf()


    if TB_LETB:
        csfont = {'fontname':'serif',"size":18}
        interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
        intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")
        xdata = np.concatenate((interlayer_hopping_data['disp'],intralayer_hopping_data['disp']),axis=0)
        ydata = np.concatenate((interlayer_hopping_data['hopping'],intralayer_hopping_data["hopping"]),axis=0)

        interlayer_ensemble = np.load("../uncertainty_quantification/ensembles/LETB_interlayer_mcmc_T_1_ensemble.npz")["ensembles"]
        intralayer_ensemble = np.load("../uncertainty_quantification/ensembles/LETB_intralayer_mcmc_T_1_ensemble.npz")["ensembles"]
        n_ensembles = np.shape(interlayer_ensemble)[0]

        """ensemble_hoppings = np.zeros((n_ensembles,len(ydata)))
        for n in range(n_ensembles):
            tb_params = ensemble[n,:]
            ensemble_hoppings[n,:] = mk_hopping(xdata,tb_params)
        mean_hoppings = np.mean(ensemble_hoppings,axis=0)
        std_hoppings = np.std(ensemble_hoppings,axis=0)
        plt.scatter(np.linalg.norm(xdata,axis=1),mean_hoppings,c=std_hoppings)
        plt.xlabel(r"r $(\AA)$",**csfont)
        plt.ylabel("<t> (eV)",**csfont)
        plt.colorbar()
        plt.title("letb hopping mean value",**csfont)
        plt.savefig("figures/letb_hopping_uncertainty.png", bbox_inches = 'tight')
        plt.clf()

        plt.scatter(np.linalg.norm(xdata,axis=1),std_hoppings)
        plt.xlabel(r"r $(\AA)$",**csfont)
        plt.ylabel(r"$\sigma(t) (eV)$",**csfont)
        plt.title("LETB hopping uncertainty",**csfont)
        plt.savefig("figures/letb_hopping_uncertainty_std.png", bbox_inches = 'tight')
        plt.clf()"""


        atoms = get_bilayer_atoms(3.35,0)
        
        Gamma = [0,   0,   0]
        K = [1/3,2/3,0]
        Kprime = [2/3,1/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nkp=60
        (kvec,k_dist, k_node) = k_path(sym_pts,nkp)

        eigvals_ensemble = np.zeros((n_ensembles,len(atoms),nkp+1))

        fig, ax = plt.subplots()

        label=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(k_node[0],k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        # add vertical lines at node positions
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n],linewidth=0.5, color='k')
        ax.set_title("AB bilayer graphene, MK UQ")
        ax.set_xlabel("Path in k-space")
        for i in range(n_ensembles):
            cutoff = 10
            interlayer_hopping_params = interlayer_ensemble[i,:]
            intralayer_hopping_params = intralayer_ensemble[i,:]

            model_dict = {"interlayer":{"hopping form":letb_interlayer,"overlap form":None,
                                        "hopping parameters":interlayer_hopping_params,"overlap parameters":None,
                                        "descriptors":letb_interlayer_descriptors},
            
                    "intralayer":{"hopping form":letb_intralayer,"overlap form":None,
                                        "hopping parameters":intralayer_hopping_params,"overlap parameters":None,
                                        "descriptors":letb_intralayer_descriptors}}

            calc = TETB_model(model_dict)
            eigvals = calc.get_band_structure(atoms,kvec)
            eigvals_ensemble[i,:,:] = eigvals - get_fermi_energy(eigvals)

            nbands = len(atoms)
            if i//50:
                for n in range(nbands):
                    ax.plot(k_dist,eigvals[n,:],c="black",linewidth=0.01,alpha=0.3)
 
        erange = 2
        ax.set_ylim(-erange,erange)
        ax.set_ylabel(r'$E - E_F$ (eV)')
        ax.set_xticks(k_node)
        ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
        ax.set_xlim(k_node[0], k_node[-1])
        fig.savefig("figures/ab_graphene_band_structure_letb_all_bands.png")
        plt.clf()
        
        
        mean_eigvals = np.mean(eigvals_ensemble,axis=0)
        std_eigvals = np.std(eigvals_ensemble,axis=0)


        fig, ax = plt.subplots()

        label=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(k_node[0],k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        # add vertical lines at node positions
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n],linewidth=0.5, color='k')
        # put title
        ax.set_title("AB bilayer graphene, LETB UQ")
        ax.set_xlabel("Path in k-space")
        
        nbands = len(atoms)

        for n in range(nbands):
            ax.plot(k_dist,mean_eigvals[n,:],c="red")
            ax.fill_between(k_dist,mean_eigvals[n,:]-std_eigvals[n,:],mean_eigvals[n,:]+std_eigvals[n,:],alpha=0.3,label="std")
        print("mean error for eigval ",np.mean(std_eigvals))
        # make an PDF figure of a plot
        #fig.tight_layout()
        erange = 2
        ax.set_ylim(-erange,erange)
        ax.set_ylabel(r'$E - E_F$ (eV)')
        ax.set_xticks(k_node)
        ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
        ax.set_xlim(k_node[0], k_node[-1])
        fig.savefig("figures/ab_graphene_band_structure_letb_uq.png")
        plt.clf()

        dist_to_fermi = np.abs(mean_eigvals)
        plt.scatter(dist_to_fermi,std_eigvals,s=5)
        plt.xlabel(r"$|\epsilon_{ik} - E_{f}|$ (eV)")
        plt.ylabel(r"$\sigma(\epsilon_{ik})$ (eV)")
        plt.savefig("figures/letb_dist_to_fermi_level_std.png")
        plt.clf()


        

