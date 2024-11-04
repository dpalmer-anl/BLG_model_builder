#from kliff.calculators import Calculator as Kliff_calc
#from kliff.dataset.weight import MagnitudeInverseWeight
#from kliff.loss import Loss
#from kliff.models.parameter_transform import LogParameterTransform
from kliff.uq import MCMC, get_T0, autocorr, mser, rhat
from kliff.dataset.extxyz import read_extxyz, write_extxyz
from kliff.dataset.dataset import Configuration
from mcmc import MCMC
from schwimmbad import MPIPool
from multiprocessing import Pool
import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
import ase.db
import time
from KLIFF_LOSS import *
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import argparse
import emcee
from sklearn.model_selection import KFold, RepeatedKFold
import math

def BootStrap(Nsamples,xdata,ydata,model,methods,weights=None):
    nparams = len(model.get_params())
    potential_samples = np.zeros((Nsamples,nparams))
    for i in range(Nsamples):
        xdata_part = []
        ydata_part = []
        for j in range(len(xdata)):
            #leave one out bootstrap
            """ind = np.random.randint(len(ydata[j])) #figure out proper way to sample, this is leave one out
            xdata_part.append([x for i,x in enumerate(xdata[j]) if i!=ind])
            ydata_part.append([y for i,y in enumerate(ydata[j]) if i!=ind])"""
            #parameter length bootstrap
            ind = np.random.randint(len(ydata[j]),size=len(ydata[j]))
            xdata_part.append([xdata[j][index] for index in ind])
            ydata_part.append([ydata[j][index] for index in ind])
        loss_fxn = build_Loss_func(xdata_part,ydata_part,model,methods,weights=weights)
        loss = LossModel(loss_fxn, model)
        result = loss.minimize()
        params = result.x
        with open("BootStrap_ensemble.txt","a+") as f:
            param_str = [str(p) for p in params]
            f.write(" ".join(param_str)+"\n")
        potential_samples[i,:] = params
    return potential_samples

def Kfold_CV(Nsplits,xdata,ydata,model,methods,weights=None,fit_params_str=None):
    nsamples = 20
    n_repeats = nsamples//Nsplits
    nparams = len(model.model_dict[fit_params_str[0]][fit_params_str[1]])
    potential_samples = np.zeros((int(Nsplits*n_repeats),nparams))
    test_loss = np.zeros(int(Nsplits*n_repeats))

    kf = RepeatedKFold(n_splits=Nsplits, n_repeats=n_repeats)
    for i, (train_index, test_index) in enumerate(kf.split(xdata)):
        xdata_train = [xdata[index] for index in train_index]
        ydata_train = [ydata[index] for index in train_index]
        loss_fxn = build_Loss_func(xdata_train,ydata_train,model,methods,weights=weights,fit_params_str=fit_params_str)
        loss = LossModel(loss_fxn, model)
        result = loss.minimize()
        params = result.x
        potential_samples[i,:] = params

        xdata_test = [xdata[index] for index in test_index]
        ydata_test = [ydata[index] for index in test_index]
        loss_fxn = build_Loss_func(xdata_test,ydata_test,model,model.get_total_energy,weights=weights,fit_params_str=fit_params_str)
        test_loss[i] = loss_fxn(params)
    return potential_samples,test_loss

def Kfold_CV_kwargs(Nsplits,xdata,ydata,methods,method_kwargs=None):
    nsamples = 20
    n_repeats = nsamples//Nsplits
    nparams = len(model.model_dict[fit_params_str[0]][fit_params_str[1]])
    potential_samples = np.zeros((int(Nsplits*n_repeats),nparams))
    test_loss = np.zeros(int(Nsplits*n_repeats))

    kf = RepeatedKFold(n_splits=Nsplits, n_repeats=n_repeats)
    for i, (train_index, test_index) in enumerate(kf.split(xdata)):
        xdata_train = [xdata[index] for index in train_index]
        ydata_train = [ydata[index] for index in train_index]
        loss_fxn = build_loss_func_kwargs(xdata_train,ydata_train,methods,**method_kwargs)
        loss = LossModel(loss_fxn, model)
        result = loss.minimize()
        params = result.x
        potential_samples[i,:] = params

        xdata_test = [xdata[index] for index in test_index]
        ydata_test = [ydata[index] for index in test_index]
        loss_fxn = build_loss_func_kwargs(xdata_test,ydata_test,methods,**method_kwargs)
        test_loss[i] = loss_fxn(params)
    return potential_samples,test_loss

def LOO_CV(xdata,ydata,model,methods,weights=None):
    nparams = len(model.get_params())
    potential_samples = np.zeros((Nsamples,Nsamples,nparams))
    loocv_loss = np.zeros(nparams)

    for j in range(len(xdata)):
        for k in range(len(xdata)):
            #parameter length bootstrap
            xdata_part = xdata[j][~k]
            ydata_part = ydata[j][~k]
            loss_fxn = build_Loss_func(xdata_part,ydata_part,model,methods,weights=weights)
            loss = LossModel(loss_fxn, model)
            result = loss.minimize()
            params = result.x
            potential_samples[i,k,:] = params
            loocv_loss[k] = ()
    return potential_samples

def hopping_training_data(hopping_type="interlayer",units="ang"):
    data = []
    flist = glob.glob('../data/hoppings/*.hdf5',recursive=True)
    eV_per_hart=27.2114
    ang_per_bohr = 0.529
    if units == "bohr":
        conv = 1/ang_per_bohr
    else:
        conv=1
    #hoppings = np.zeros((1,1))
    disp_list = []
    hoppings = []
    atoms_list = []
    i_list = []
    j_list = []
    di_list = []
    dj_list = []
    for f in flist:
        if ".hdf5" in f:
            with h5py.File(f, 'r') as hdf:
                # Unpack hdf
                lattice_vectors = np.array(hdf['lattice_vectors'][:]) * conv
                atomic_basis =    np.array(hdf['atomic_basis'][:])   * conv
                atoms =  ase.Atoms("C"*np.shape(atomic_basis)[0],positions =atomic_basis,cell=lattice_vectors)
                mean_z = np.mean(atomic_basis[:,2])
                top_ind = np.where(atomic_basis[:,2]>mean_z)
                mol_id = np.ones(len(atoms),dtype=np.int64)
                mol_id[top_ind] = 2
                atoms.set_array("mol-id",mol_id)
                atoms_list.append(atoms)
                tb_hamiltonian = hdf['tb_hamiltonian']
                tij = np.array(tb_hamiltonian['tij'][:]) #* eV_per_hart
                di  = np.array(tb_hamiltonian['displacementi'][:])
                dj  = np.array(tb_hamiltonian['displacementj'][:])
                ai  = np.array(tb_hamiltonian['atomi'][:])
                aj  = np.array(tb_hamiltonian['atomj'][:])
                displacement_vector = (di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai])*conv

                if hopping_type=="interlayer":
                    type_ind = np.where(mol_id[ai]!=mol_id[aj])
                    hoppings.append(tij[type_ind])
                    i_list.append(ai[type_ind])
                    j_list.append(aj[type_ind])
                    di_list.append(di[type_ind])
                    dj_list.append(dj[type_ind])
                    disp_list.append(np.squeeze(displacement_vector[type_ind,:]))
                elif hopping_type=="intralayer":
                    type_ind = np.where(mol_id[ai]==mol_id[aj])
                    hoppings.append(tij[type_ind])
                    i_list.append(ai[type_ind])
                    j_list.append(aj[type_ind])
                    di_list.append(di[type_ind])
                    dj_list.append(dj[type_ind])
                    disp_list.append(np.squeeze(displacement_vector[type_ind,:]))
                else:
                    hoppings.append(tij)
                    i_list.append(ai)
                    j_list.append(aj)
                    di_list.append(di)
                    dj_list.append(dj)
                    disp_list.append(np.squeeze(displacement_vector))

    return {"hopping":hoppings,"atoms":atoms_list,"i":i_list,"j":j_list,"di":di_list,"dj":dj_list,"disp":disp_list}

def create_Dataset(interlayer_db,intralayer_db):

    interlayer_atom_list = []
    interlayer_energies = []
    intralayer_atom_list = []
    intralayer_energies = []
    for i,row in enumerate(interlayer_db.select()):
        atoms = interlayer_db.get_atoms(id = row.id)
        pos = atoms.positions
        mean_z = np.mean(pos[:,2])
        top_ind = np.where(pos[:,2]>mean_z)
        mol_id = np.ones(len(atoms),dtype=np.int64)
        mol_id[top_ind] = 2
        atoms.set_array("mol-id",mol_id)

        top_layer_ind = np.where(pos[:,2]>mean_z)
        top_pos = np.squeeze(pos[top_layer_ind,:])
        bot_layer_ind = np.where(pos[:,2]<mean_z)
        bot_pos = np.squeeze(pos[bot_layer_ind,:])

        interlayer_atom_list.append(atoms)
        interlayer_energies.append(row.data.total_energy*len(atoms))

    for i,row in enumerate(intralayer_db.select()):
        atoms = intralayer_db.get_atoms(id = row.id)
        atoms.set_array("mol-id",np.ones(len(atoms),dtype=np.int64))

        intralayer_atom_list.append(atoms)
        intralayer_energies.append(row.data.total_energy*len(atoms))

    return interlayer_atom_list,interlayer_energies,intralayer_atom_list,intralayer_energies

def logprior_fxn():
    min_ind = np.argmin(energies)
    min_d = d_[min_ind]
    if min_d<3.01 or min_d >7:
        return np.inf
    else:
        return 1

def get_bounds(loss,opt_params):
    #find bounds s.t. cost increases n times
    C0 = loss(opt_params)
    n = 1/C0/10
    bound_list = []
    #param_names = ['z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A']
    for i in range(len(opt_params)):

        new_params = opt_params.copy()
        new_params[i] += new_params[i]  * 0.005 
        cost_up = loss(new_params)
        curve = (cost_up - C0)/(opt_params[i]  * 0.005)**2
        #print("Cost curvature for "+param_names[i]+" = ",curve)

        dtheta = np.sqrt(n*C0/curve)
        upper_bound = opt_params[i] + dtheta
        if np.isnan(upper_bound):
            upper_bound = opt_params[i] + 0.005 * opt_params[i]
        lower_bound = opt_params[i] - dtheta
        if np.isnan(lower_bound):
            lower_bound = opt_params[i] - 0.005 * opt_params[i]

        bound_list.append((lower_bound,upper_bound))
    return bound_list
 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u','--use_model',type=str,default="Classical_energy")
    parser.add_argument('-m','--model_type',type=str,default='interlayer')
    parser.add_argument('-q','--uq_type',type=str,default="mcmc")
    parser.add_argument('-b','--bondint_model_name',type=str,default='popov_hopping_pp_sigma')
    parser.add_argument('-n',"--nsplits",type=str,default='2')
    parser.add_argument('-T',"--temperature_weight",type=str,default='1')
    args = parser.parse_args() 
    #define minimal cost model parameters

    use_model = args.use_model
    model_type = args.model_type
    uq_type =args.uq_type
    hyper_param_str = use_model+"_"+model_type+"_"+uq_type+"_T_"+str(args.temperature_weight)

    nkp = 121
    bound_frac = 5e-1
    interlayer_db =  ase.db.connect('../data/bilayer_nkp'+str(nkp)+'.db')
    intralayer_db = db = ase.db.connect('../data/monolayer_nkp'+str(nkp)+'.db')
    interlayer_atom_list,interlayer_energies,intralayer_atom_list,intralayer_energies = create_Dataset(interlayer_db,intralayer_db)

    if use_model=="TETB":
        rebo_parameters = np.load("../uncertainty_quantification/parameters/intralayer_energy_parameters_mk_nkp121.npz")["parameters"]
        interlayer_params = np.load("../uncertainty_quantification/parameters/interlayer_energy_parameters_mk_nkp121.npz")["parameters"]
        
        
        hopping_model_name ="MK"

        if hopping_model_name =="popov":
            eV_per_hartree = 27.2114
            interlayer_hopping_fxn = SK_pz_chebyshev
            interlayer_overlap_fxn = SK_pz_chebyshev
            intralayer_hopping_fxn = SK_pz_chebyshev
            intralayer_overlap_fxn = SK_pz_chebyshev
            popov_hopping_pp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863]) *eV_per_hartree #np.load("../BLG_model_builder/parameters/popov_hoppings_pp_sigma.npz")["parameters"]
            popov_hopping_pp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree #np.load("../BLG_model_builder/parameters/popov_hoppings_pp_pi.npz")["parameters"]
            popov_overlap_pp_pi = np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,0.0156376, 0.0025976, -0.0039498, 0.0020581,-0.0007114, 0.0001427])  #np.load("../BLG_model_builder/parameters/popov_overlap_pp_sigma.npz")["parameters"]
            popov_overlap_pp_sigma = np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997, 0.0921727, -0.0268106, 0.0002240, 0.0040319, -0.0022450, 0.0005596])  #np.load("../BLG_model_builder/parameters/popov_overlap_pp_pi.npz")["parameters"]
            interlayer_hopping_params = np.append(popov_hopping_pp_sigma,popov_hopping_pp_pi)
            interlayer_overlap_params = np.append(popov_overlap_pp_sigma,popov_overlap_pp_pi)

            porezag_hopping_pp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree #np.load("../BLG_model_builder/parameters/porezag_hoppings_pp_sigma.npz")["parameters"]
            porezag_hopping_pp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree #np.load("../BLG_model_builder/parameters/porezag_hoppings_pp_pi.npz")["parameters"]
            porezag_overlap_pp_pi =np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227]) #np.load("../BLG_model_builder/parameters/porezag_overlap_pp_sigma.npz")["parameters"]
            porezag_overlap_pp_sigma = np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751])  #np.load("../BLG_model_builder/parameters/porezag_overlap_pp_pi.npz")["parameters"]
            intralayer_hopping_params = np.append(porezag_hopping_pp_sigma,porezag_hopping_pp_pi)
            intralayer_overlap_params = np.append(porezag_overlap_pp_sigma,porezag_overlap_pp_pi)

        elif hopping_model_name=="MK":
            interlayer_hopping_fxn = mk_hopping
            intralayer_hopping_fxn = mk_hopping
            interlayer_overlap_fxn = None
            intralayer_overlap_fxn = None
            interlayer_hopping_params = np.array([-2.7, 2.2109794066373403, 0.48])
            intralayer_hopping_params = np.array([-2.7, 2.2109794066373403, 0.48])
            interlayer_overlap_params = None
            intralayer_overlap_params = None
            intralayer_cutoff = 6
            interlayer_cutoff = 6
            bound_list = []
            for i in range(len(interlayer_hopping_params)):
                bound_pair = (interlayer_hopping_params[i]-bound_frac*interlayer_hopping_params[i], interlayer_hopping_params[i]+bound_frac*interlayer_hopping_params[i])
                low_bound = np.min(bound_pair)
                up_bound = np.max(bound_pair)
                bound_list.append((low_bound,up_bound))


        #bound_list = []
        if model_type=="interlayer":
            fit_params_str= None #["interlayer","potential parameters"]
            for i in range(len(interlayer_params)):
                bound_pair = (interlayer_params[i]-bound_frac*interlayer_params[i],interlayer_params[i]+bound_frac*interlayer_params[i])
                #low_bound = 10**(np.log(interlayer_params[i])-1)
                #up_bound = 10**(np.log(interlayer_params[i])+1)
                bound_list.append((np.min(bound_pair),np.max(bound_pair)))

            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":interlayer_cutoff},"cutoff":interlayer_cutoff,
                                    "potential":interlayer_potential,"potential parameters":interlayer_params},
                                    #"potential":"reg/dep/poly 10.0 0","potential parameters":interlayer_params,
                                    #"potential file writer":write_kcinsp},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":intralayer_cutoff},"cutoff":intralayer_cutoff,
                                    "potential":None,"potential parameters":None}}
                                    #"potential":"rebo","potential parameters":rebo_parameters,"potential file writer":write_rebo}}


            xdata_energy = interlayer_atom_list
            ydata_energy = interlayer_energies
            weights = []
            df = pd.read_csv('../data/qmc.csv')
            for i, row in df.iterrows():
                weights.append(1/row['energy_err'])
            

        elif model_type=="intralayer":
            fit_params_str= None #["intralayer","potential parameters"]
            for i in range(len(intralayer_params)):
                bound_pair = (intralayer_params[i]-bound_frac*intralayer_params[i],intralayer_params[i]+bound_frac*intralayer_params[i])
                #low_bound = 10**(np.log(intralayer_params[i])-1)
                #up_bound = 10**(np.log(intralayer_params[i])+1)
                bound_list.append((np.min(bound_pair),np.max(bound_pair)))

            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":6},"cutoff":6,
                                    "potential":None,"potential parameters":None},

                        "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":6},"cutoff":6,
                                "potential":"rebo","potential parameters":rebo_parameters,"potential file writer":write_rebo}}
            xdata_energy = intralayer_atom_list
            ydata_energy = intralayer_energies

        for i in range(len(intralayer_hopping_params)):
                bound_pair = (intralayer_hopping_params[i]-bound_frac*intralayer_hopping_params[i], intralayer_hopping_params[i]+bound_frac*intralayer_hopping_params[i])
                low_bound = np.min(bound_pair)
                up_bound = np.max(bound_pair)
                bound_list.append((low_bound,up_bound))

        hopping_data = hopping_training_data(hopping_type="all")
        xdata_tb_list = hopping_data["disp"]
        xdata_tb = xdata_tb_list[0]
        ydata_tb_list = hopping_data["hopping"]
        ydata_tb = ydata_tb_list[0]

        for i in range(1,len(ydata_tb_list)):
            xdata_tb = np.vstack((xdata_tb,xdata_tb_list[i]))
            ydata_tb = np.append(ydata_tb,ydata_tb_list[i])

        ase_calc = TETB_model(model_dict) 
        opt_params = ase_calc.get_params()
        ase_calc.set_opt_params(opt_params)
        ase_calc.set_opt_params_bounds(np.array(bound_list))
        #ase_calc.set_opt_params_bounds([(-np.inf,np.inf)]*len(opt_params))

        
        energy_method = ase_calc.get_total_energy
        tb_method = mk_hopping #ase_calc.get_hoppings
        energy_best_fit_rmse = 6.491542548624808e-5
        tb_best_fit_rmse = 0.24680382797271483
        weight = tb_best_fit_rmse / (tb_best_fit_rmse + energy_best_fit_rmse)
        print("Relative weighting = ")
        LossFxn = build_loss_func_TETB(xdata_energy,xdata_tb,
                        ydata_energy,ydata_tb,
                        ase_calc,energy_method,tb_method,
                        weight=weight,fit_params_str=fit_params_str)

        loss = LossModel(LossFxn, ase_calc)
    

    elif use_model=="Classical_energy":
        rebo_params = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])

        #from Mick's qmc kc paper kc_params = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
        #                                0.719345289329483, 3.293082477932360, 13.906782892134125])
        kc_params = np.array([3.40401668, 18.1835987, 13.3939433, 3.08220226e-3, 6.07208011, 6.87317037e-1, 3.30175581, 13.9135780])

        bound_list = []
        #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
        if model_type=="interlayer":
            fit_params_str= ["interlayer","potential parameters"]
            """for i in range(len(kc_params)):
                #bound_pair = (kc_params[i]-bound_frac*kc_params[i],kc_params[i]+bound_frac*kc_params[i])
                low_bound = 10**(np.log10(kc_params[i])-1)
                up_bound = 10**(np.log10(kc_params[i])+1)
                bound_list.append((low_bound,up_bound))
            bound_list = [(3,4),(1,100),(1,100),(1e-6,10),(1e-6,10),(1e-1,4),(2,4),(1e-2,1e6)]"""
            #model_dict = {"interlayer":{"hopping form":None,
            #                    "potential":"kolmogorov/crespi/full 10.0 0","potential parameters":kc_params,
            #                    "potential file writer":write_kc}}
            model_dict = {"interlayer":{"hopping form":None,"overlap form":None,
                                "potential":Kolmogorov_Crespi,"potential parameters":kc_params}}
            xdata = interlayer_atom_list
            ydata = interlayer_energies
            weights = []
            df = pd.read_csv('../data/qmc.csv')
            for i, row in df.iterrows():
                weights.append(1/row['energy_err'])
            weights = [weights]
        elif model_type=="intralayer":
            fit_params_str= ["intralayer","potential parameters"]
            """for i in range(len(rebo_params)):
                #bound_pair = (rebo_params[i]-bound_frac*rebo_params[i],rebo_params[i]+bound_frac*rebo_params[i])
                low_bound = 10**(np.log10(rebo_params[i])-1)
                up_bound = 10**(np.log10(rebo_params[i])+1)
                bound_list.append((low_bound,up_bound))"""

            model_dict = {"intralayer":{"hopping form":None,
                                "potential":"rebo","potential parameters":rebo_params,"potential file writer":write_rebo}}
            xdata = intralayer_atom_list
            ydata = intralayer_energies
        
        ase_calc = TETB_model(model_dict) 
        opt_params = ase_calc.get_params()

        ase_calc.set_opt_params(opt_params)
        #ase_calc.set_opt_params_bounds(np.array(bound_list))

        
        methods = ase_calc.get_total_energy

        #loss fxn with no bounds
        LossFxn = build_Loss_func(xdata,ydata,ase_calc,methods)
        bound_list = get_bounds(LossFxn,opt_params)

        ase_calc.set_opt_params_bounds(np.array(bound_list))
        LossFxn = build_Loss_func(xdata,ydata,ase_calc,methods)
        loss = LossModel(LossFxn, ase_calc)
    
    if use_model=="TB_sk":
        eV_per_hartree = 27.2114
        ang_per_bohr = 0.529
        bondint_model_name =args.bondint_model_name
        hyper_param_str += "_"+bondint_model_name

        bondint_params={'popov_hopping_pp_sigma' : np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863]) *eV_per_hartree ,
                        'popov_hopping_pp_pi': np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree ,
                        'popov_overlap_pp_pi': np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,0.0156376, 0.0025976, -0.0039498, 0.0020581,-0.0007114, 0.0001427])  ,
                        'popov_overlap_pp_sigma': np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997, 0.0921727, -0.0268106, 0.0002240, 0.0040319, -0.0022450, 0.0005596]),  

                        'porezag_hopping_pp_sigma': np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree ,
                        'porezag_hopping_pp_pi': np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree ,
                        'porezag_overlap_pp_pi':np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227]) ,
                        'porezag_overlap_pp_sigma': np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751])} 

        bondint_cutoffs = {'popov_hopping_pp_sigma' : 5.29177 ,
                        'popov_hopping_pp_pi': 5.29177 ,
                        'popov_overlap_pp_pi': 5.29177  ,
                        'popov_overlap_pp_sigma': 5.29177 ,  

                        'porezag_hopping_pp_sigma': 3.704239,
                        'porezag_hopping_pp_pi': 3.704239 ,
                        'porezag_overlap_pp_pi':3.704239 ,
                        'porezag_overlap_pp_sigma': 3.704239} 

        opt_params = bondint_params[bondint_model_name]
        bound_list = []
        for i in range(len(opt_params)):
            bound_pair = (opt_params[i]-bound_frac*opt_params[i], opt_params[i]+bound_frac*opt_params[i])
            low_bound = np.min(bound_pair)
            up_bound = np.max(bound_pair)
            bound_list.append((low_bound,up_bound))

        data = np.loadtxt("../data/"+bondint_model_name+".txt")
        xdata = data[:,0] *ang_per_bohr
        ydata = data[:,1] * eV_per_hartree

        methods = SK_bond_ints
        method_kwargs  ={'b':bondint_cutoffs[bondint_model_name]}
        LossFxn = build_loss_func_kwargs(xdata,ydata,methods,method_kwargs)


    if use_model=="TB_MK":
        opt_params = np.array([-2.7, 2.2109794066373403, 0.48])

        hopping_data = hopping_training_data(hopping_type="all")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata = xdata_list[0]
        ydata = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata = np.vstack((xdata,xdata_list[i]))
            ydata = np.append(ydata,ydata_list[i])
        bound_list = []
        for i in range(len(opt_params)):
            bound_pair = (opt_params[i]-bound_frac*opt_params[i], opt_params[i]+bound_frac*opt_params[i])
            low_bound = np.min(bound_pair)
            up_bound = np.max(bound_pair)
            bound_list.append((low_bound,up_bound))

        methods = mk_hopping
        LossFxn = build_loss_func_array(xdata,ydata,methods,return_array=True)

        best_fit_rmse = LossFxn(opt_params)
        if args.temperature_weight=="best_fit_rmse":
            LossFxn = build_loss_func_array(xdata,ydata,methods,weights = best_fit_rmse)

    if use_model=="LETB":
        bound_frac = 0.25
        if model_type == "interlayer":
            interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
            opt_params = np.load("../parameters/letb_interlayer_parameters.npz")["parameters"]
            bound_list = []
            for i in range(len(opt_params)):
                bound_list.append((opt_params[i] - bound_frac*opt_params[i],opt_params[i] + bound_frac*opt_params[i]))
            atoms_list = interlayer_hopping_data["atoms"]
            i_list = interlayer_hopping_data["i"]
            j_list = interlayer_hopping_data["j"]
            di_list = interlayer_hopping_data["di"]
            dj_list = interlayer_hopping_data["dj"]
            disp_list = interlayer_hopping_data["disp"]

            xdata = []
            for atom_i in range(len(atoms_list)):
                cell = atoms_list[atom_i].get_cell()
                pos = atoms_list[atom_i].positions
                dsc = letb_interlayer_descriptors_array(cell,disp_list[atom_i],pos, di_list[atom_i], dj_list[atom_i], i_list[atom_i], j_list[atom_i])
                xdata.append(dsc)
            
            methods = letb_interlayer
            ydata = interlayer_hopping_data['hopping']

        elif model_type == "intralayer":
            intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")
            opt_params = np.load("../parameters/letb_intralayer_parameters.npz")["parameters"]
            bound_list = []
            for i in range(len(opt_params)):
                bound_list.append((opt_params[i] - bound_frac*opt_params[i],opt_params[i] + bound_frac*opt_params[i]))
            atoms_list = intralayer_hopping_data["atoms"]
            i_list = intralayer_hopping_data["i"]
            j_list = intralayer_hopping_data["j"]
            di_list = intralayer_hopping_data["di"]
            dj_list = intralayer_hopping_data["dj"]
            disp_list = intralayer_hopping_data["disp"]

            xdata = []
            for atom_i in range(len(atoms_list)):
                cell = atoms_list[atom_i].get_cell()
                pos = atoms_list[atom_i].positions
                dsc = letb_intralayer_descriptors_array(cell,disp_list[atom_i],pos, di_list[atom_i], dj_list[atom_i], i_list[atom_i], j_list[atom_i])
                xdata.append(dsc)
            ydata = intralayer_hopping_data["hopping"]
            methods = letb_intralayer

        LossFxn = build_loss_func_kwargs(xdata,ydata,methods)
        bound_list = get_bounds(LossFxn,opt_params)



    if uq_type == "bootstrap":

        nsamples = 100  # Number of samples to generate
        initial_guess = opt_params
        
        bootstrap_samples = BootStrap(nsamples,xdata,ydata,ase_calc,methods,weights=None)

        np.savez("ensembles/"+hyper_param_str+"_ensemble",samples = bootstrap_samples)

    elif uq_type =="Kfold":
        Nsplits = int(args.nsplits)
        hyper_param_str += "_n_"+str(Nsplits)
        initial_guess = opt_params
        
        cv_samples, test_loss = Kfold_CV_kwargs(Nsplits,xdata,ydata,methods)

        np.savez("ensembles/"+hyper_param_str+"_ensemble",samples = cv_samples,test_loss=test_loss)

    if uq_type =="mcmc":
        #define hyperparameters
        ndim = len(opt_params)
        bounds = np.array(bound_list)
        print("opt params = ",opt_params)
        print("bounds = ", bounds)

        ensemble_size = 200
        nwalkers = 2 * ndim
        auto_corr_length = 50
        burnin = 500
        iterations = ensemble_size//nwalkers * auto_corr_length + burnin

        ntemps = 1
        start = time.time()
        #run monte carlo sampling
        if args.temperature_weight=="best_fit_rmse":
            T = 1
        else:
            T0 = 2*LossFxn(opt_params)/ndim 
            T = T0* float(args.temperature_weight)
        print("T = ",T)


        Tladder = np.array([T])
        sampler = MCMC(LossFxn, ndim,bounds,nwalkers=nwalkers,  T=Tladder) #,logprior_args=(bounds,), #logprior_fn, logprior_args,

        
        p0 = np.empty((nwalkers, ndim))
        for ii, bound in enumerate(bounds):
            p0[ :, ii] = np.random.uniform(low=bounds[ii,0],high=bounds[ii,1],size= ( nwalkers))
        #sampler.pool = MPIPool()
        #sampler.pool = Pool(40)
        sampler.run_mcmc(p0, iterations)
        #sampler.pool.close() 

        # Retrieve the chain
        chain = sampler.chain
        end = time.time()
        
        param_ind =0
        mean_param = np.zeros(np.shape(chain)[1])
        std_param = np.zeros(np.shape(chain)[1])
        for i in range(1,len(mean_param)):
            mean_param[i] = np.mean(chain[:,:i,param_ind])
            std_param[i] = np.std(chain[:,:i,param_ind])


        plt.plot(np.arange(len(mean_param)),mean_param,label="mean")
        plt.fill_between(np.arange(len(mean_param)),mean_param+std_param,mean_param-std_param,alpha=0.3,label="std")
        plt.legend()
        plt.xlabel("MCMC step")
        plt.ylabel("accumulated mean/std")
        plt.savefig("figures/"+use_model+"_mean_param_"+str(param_ind)+"_burnin.png")
        plt.clf()

        param_ind =0
        mean_param = np.zeros(np.shape(chain)[1])
        std_param = np.zeros(np.shape(chain)[1])
        for i in range(burnin+1,len(mean_param)):
            mean_param[i] = np.mean(chain[:,burnin:i,param_ind])
            std_param[i] = np.std(chain[:,burnin:i,param_ind])


        plt.plot(np.arange(burnin+1,len(mean_param)),mean_param[burnin+1:],label="mean")
        plt.fill_between(np.arange(burnin+1,len(mean_param)),mean_param[burnin+1:]+std_param[burnin+1:],mean_param[burnin+1:]-std_param[burnin+1:],alpha=0.3,label="std")
        plt.legend()
        plt.xlabel("MCMC step")
        plt.ylabel("accumulated mean/std")
        plt.savefig("figures/"+use_model+"_mean_param_"+str(param_ind)+"_no_burnin.png")
        plt.clf()
        # Estimate the autocorrelation length for each temperature
        chain_no_burnin = np.squeeze(chain[:, burnin:,:])
        niter = np.shape(chain_no_burnin)[1]
        ensembles = chain_no_burnin[:,0,:]
        for i in range(int(niter/auto_corr_length)):
            part_ensembles = chain_no_burnin[:,int(i*auto_corr_length),:]
            ensembles = np.append(ensembles,part_ensembles,axis=0)
        print(np.shape(ensembles))
        np.savez("ensembles/"+hyper_param_str+"_ensemble",ensembles = ensembles)

        accept_frac = sampler.acceptance_fraction
        print("acceptance fraction = ",np.mean(accept_frac))
