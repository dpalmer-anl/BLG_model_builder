import numpy as np
import scipy.optimize 
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.TETB_model_builder import TETB_model
from BLG_model_builder.geom_tools import *
from BLG_model_builder.BLG_potentials import *
import pandas as pd
import os
import ase.db
import glob
import h5py
from KLIFF_LOSS import *
from shgo import shgo
#from geodesicLM import geodesiclm

def hopping_training_data(hopping_type=None):
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
                print(np.linalg.norm(displacement_vector,axis=1))
            hoppings = np.append(hoppings,tij)
            disp_array = np.vstack((disp_array,displacement_vector)) 
    hoppings = hoppings[1:]
    disp_array = disp_array[1:,:]
    if hopping_type=="interlayer":
        type_ind = np.where(disp_array[:,2] > 1) # Inter-layer hoppings only, allows for buckling
    elif hopping_type=="intralayer":
        type_ind = np.where(disp_array[:,2] < 1)
    else:
        type_ind = np.where(disp_array[:,2]>-1e6)
    return hoppings[type_ind],disp_array[type_ind]

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
        intralayer_energies.append(row.data.total_energy)


    return interlayer_atom_list,interlayer_energies,intralayer_atom_list,intralayer_energies

def fit_model(xdata,ydata,model,methods,p0,bounds=None,weights=None,fit_type="normal",tb_energy=0):


    if fit_type=="one_at_a_time":
        p0 = np.zeros_like(p0)
        oaat_p0 = np.zeros_like(p0)
        for i in range(len(p0)):
            Loss_func = build_Loss_func(xdata,ydata,model,methods,weights=weights,p0=p0,fit_params_ind=i,tb_energy=tb_energy)
            popt = scipy.optimize.minimize(Loss_func,[0], method="Nelder-Mead",bounds=bounds)
            oaat_p0[i] = popt.x

        popt = scipy.optimize.minimize(Loss_func,oaat_p0, method="L-BFGS-B",bounds=bounds)
        min_params = popt.x

    if fit_type=="scan":
        param_bounds = (-1e8,1e8)
        bounds = [param_bounds]*len(p0)
        nsamples = 10
        params = np.zeros((nsamples,len(p0)))
        loss = np.zeros(nsamples)
        for i in range(nsamples):
            p0_rand = np.random.uniform(low=param_bounds[0],high=param_bounds[1],size=len(p0))
            p0_rand[5] = 3.35
            model.set_params(p0_rand)
            Loss_func = build_Loss_func(xdata,ydata,model,methods,weights=weights,tb_energy=tb_energy)

            y_ab_min_ind = np.argmin(ydata)
            y_abinitio = ydata - np.min(ydata)
            yfit = []
            for x in xdata:
                yfit.append(methods(x))
            yfit = np.array(yfit) + tb_energy
            yfit -= yfit[y_ab_min_ind]
            print(yfit)

            y_ab_scale = np.max(y_abinitio)
            y_fit_scale = np.max(yfit) - np.min(yfit)
            
            #put y fit on same energy scale as y_ab_initio
            print("y ab scale ",y_ab_scale)
            print("y fit scale ",y_fit_scale)
            ratio = y_ab_scale/y_fit_scale
            print("ratio", ratio)
            print("p0 original",p0_rand)
            p0_rand[6:] *=ratio
            print("p0 ",p0_rand)

            popt = scipy.optimize.minimize(Loss_func,p0_rand, method="Nelder-Mead",bounds = bounds)
            loss[i] = Loss_func(popt.x)
        min_ind = np.argmin(loss)
        min_params = params[min_ind,:]

    else:
        
        Loss_func = build_Loss_func(xdata,ydata,model,methods,weights=weights,tb_energy=tb_energy)
        popt = scipy.optimize.minimize(Loss_func,p0, method="L-BFGS-B",bounds=bounds)

        min_params = popt.x
    return min_params

if __name__=="__main__":
    eV_per_hartree = 27.2114
    mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
 
    interlayer_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406,
                                   -103.18388323245665, 1.8220964068356134, 2.70634766,0.0003125,-6.25e-5]) 
    #interlayer_params = np.load("../uncertainty_quantification/parameters/interlayer_energy_parameters_popov_nkp121.npz")["parameters"]
    
    intralayer_params = np.array([0.14687637217609084, 4.683462616941604, 12433.64356176609,\
                12466.479169306709, 19.121905577450008, 30.504342033258325,\
                4.636516235627607 , 1.3641304165817836, 1.3878198074813923])
    hopping_model_name="MK"
    if hopping_model_name == "MK":
        interlayer_hopping_fxn = mk_hopping
        interlayer_overlap_fxn = None
        interlayer_hopping_params = mk_params
        interlayer_overlap_params = None
        interlayer_cutoff = 5.29177
        
        intralayer_hopping_fxn = mk_hopping
        intralayer_overlap_fxn = None
        intralayer_hopping_params = mk_params
        intralayer_overlap_params = None
        intralayer_cutoff=5.29177

    elif hopping_model_name == "popov":
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
        interlayer_cutoff = 5.29177

        porezag_hopping_pp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree #np.load("../BLG_model_builder/parameters/porezag_hoppings_pp_sigma.npz")["parameters"]
        porezag_hopping_pp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree #np.load("../BLG_model_builder/parameters/porezag_hoppings_pp_pi.npz")["parameters"]
        porezag_overlap_pp_pi =np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227]) #np.load("../BLG_model_builder/parameters/porezag_overlap_pp_sigma.npz")["parameters"]
        porezag_overlap_pp_sigma = np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751])  #np.load("../BLG_model_builder/parameters/porezag_overlap_pp_pi.npz")["parameters"]
        intralayer_hopping_params = np.append(porezag_hopping_pp_sigma,porezag_hopping_pp_pi)
        intralayer_overlap_params = np.append(porezag_overlap_pp_sigma,porezag_overlap_pp_pi)
        intralayer_cutoff=3.704239

    
    nkp = 121
    interlayer_db =  ase.db.connect('../data/bilayer_nkp'+str(nkp)+'.db')
    intralayer_db =  ase.db.connect('../data/monolayer_nkp'+str(nkp)+'.db')
    interlayer_atoms,interlayer_energy,intralayer_atoms,intralayer_energy = create_Dataset(interlayer_db,intralayer_db)


    loss = "intralayer energy"
    
    if loss=="interlayer hopping":
        model_dict = {"interlayer":{"hopping form":popov_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer"},
                                "potential":Kolmogorov_Crespi_insp,"potential parameters":interlayer_params},
    
               "intralayer":{"hopping form":porezag_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer"},
                                "potential":None,"potential parameters":None,"potential file writer":None}}
    
        calc = TETB_model(model_dict)

        calc.model_dict["interlayer"]["hopping form"] = popov_hopping
        interlayer_hoppings,interlayer_disp = hopping_training_data(hopping_type="interlayer")
        methods = [popov_hopping] #[mk_hopping]
        xdata = [interlayer_disp]
        ydata = [interlayer_hoppings]

    elif loss=="intralayer hopping":
        model_dict = {"interlayer":{"hopping form":popov_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer"},
                                "potential":Kolmogorov_Crespi_insp,"potential parameters":interlayer_params},
    
               "intralayer":{"hopping form":porezag_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer"},
                                "potential":None,"potential parameters":None,"potential file writer":None}}
    
        calc = TETB_model(model_dict)

        calc.model_dict["intralayer"]["hopping form"] = popov_hopping
        intralayer_hoppings,intralayer_disp = hopping_training_data(hopping_type="intralayer")
        methods = [porezag_hopping] #[mk_hopping]
        xdata = [intralayer_disp]
        ydata = [intralayer_hoppings]

    elif loss=="hopping":
        calc.model_dict["intralayer"]["hopping form"] = mk_hopping
        hoppings,disp = hopping_training_data()
        methods = [mk_hopping]
        xdata = [disp]
        ydata = [hoppings]

    elif loss=="interlayer energy":
        model_form = "ILP"
        if model_form == "KCinsp":
            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":interlayer_cutoff},"cutoff":interlayer_cutoff,
                                    "potential":Kolmogorov_Crespi_insp,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":intralayer_cutoff},"cutoff":intralayer_cutoff,
                                    "potential":None,"potential parameters":None,"potential file writer":None}}
            p0 = interlayer_params

        if model_form =="KC":
            interlayer_params = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                                        0.719345289329483, 3.293082477932360, 13.906782892134125])
            model_dict = {"interlayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,"overlap parameters":None,
                                    "potential":Kolmogorov_Crespi,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":None,"overlap form":None,
                                    "hopping parameters":None,"overlap parameters":None,
                                    "potential":None,"potential parameters":None,"potential file writer":None}}
            p0 = interlayer_params

        if model_form =="ILP":
            
            interlayer_params = np.array([3.38709827, 1.21309991e1, 1.33976555e1, 1.37023469e-2, 6.09444612, 1.0682735, 3.91962370, 1.38524965e1,3.681440])
            
            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":interlayer_cutoff},"cutoff":interlayer_cutoff,
                                    "potential":interlayer_potential,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":intralayer_cutoff},"cutoff":intralayer_cutoff,
                                    "potential":None,"potential parameters":None,"potential file writer":None}}
            p0 = interlayer_params

        elif model_form == "vdw":
            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177},"cutoff":5.29177,
                                    "potential":vdw,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239},"cutoff":3.704239,
                                    "potential":"rebo","potential parameters":intralayer_potential,"potential file writer":write_rebo}}
            p0 = np.array([-10,0,0])
    
        calc = TETB_model(model_dict)

        df = pd.read_csv('../data/qmc.csv')
        interlayer_atoms = []

        interlayer_energy = []
        interlayer_tb_energy = []
        d = []
        weights = []
        bounds = [(-1e8,1e8),] * len(interlayer_params)
        dmin = 3.43

        lambda_val = 3.293082477932360
        z0 = 3.379423382381699
        A = 0.25
        tb_model_energy = []

        for i, row in df.iterrows():

            atoms = get_bilayer_atoms(row['d'], row['disregistry'])
            d.append(row["d"])
            if model_form != "KC":
                tb_energy, tb_forces = calc.get_tb_energy(atoms)
                print("tb energy = ",tb_energy)
            else:
                tb_energy = 0
            interlayer_energy.append(row["energy"]*len(atoms))
            interlayer_tb_energy.append(tb_energy)

            interlayer_atoms.append(atoms)
            weights.append(1/np.abs(row['d']-dmin))

            disp,_,_,_,_ = get_disp(atoms,type="interlayer")
            dist = np.linalg.norm(disp,axis=1)
            tb_model_energy.append(np.sum(-A*np.exp(-lambda_val * (dist-z0)))/len(atoms))

        """plt.scatter(d,np.array(interlayer_tb_energy)-interlayer_tb_energy[-1],label="tb energy")
        plt.scatter(d,np.array(tb_model_energy)-tb_model_energy[-1],label="e^-l(r-z0)")
        plt.legend()
        plt.savefig("tb_energy_model.png")
        exit()"""
        weights = np.array(weights)
        calc.model_dict["interlayer"]["hopping form"] = None
        calc.model_dict["intralayer"]["hopping form"] = None
        calc.model_dict["interlayer"]["overlap form"] = None
        calc.model_dict["intralayer"]["overlap form"] = None
        methods = calc.get_residual_energy
        xdata = interlayer_atoms
        ydata = interlayer_energy
        tb_energy = interlayer_tb_energy
        weights = None #[weights]
        
        

    elif loss=="intralayer energy":
        intralayer_params  = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,14522.273379352482,7.855493960028371,
                                        40.609282094464604,4.62769509546907,0.7945927858501145,2.2242248220983427])
        model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":interlayer_cutoff},"cutoff":interlayer_cutoff,
                                    "potential":None,"potential parameters":None},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":intralayer_cutoff},"cutoff":intralayer_cutoff,
                                    "potential":"rebo","potential parameters":intralayer_params,"potential file writer":write_rebo}}
        p0 = intralayer_params
    
        calc = TETB_model(model_dict)
        intralayer_residual = []

        intralayer_tb_energy = []
        intralayer_energy_list = []
        for i,a in enumerate(intralayer_atoms):
            tb_energy, tb_forces = calc.get_tb_energy(a)
            intralayer_residual.append(intralayer_energy[i]*len(a) - tb_energy)
            intralayer_tb_energy.append(tb_energy)
            intralayer_energy_list.append(intralayer_energy[i]*len(a))



        calc.model_dict["interlayer"]["hopping form"] = None
        calc.model_dict["intralayer"]["hopping form"] = None
        methods = calc.get_residual_energy
        xdata = intralayer_atoms
        ydata = np.array(intralayer_energy_list)
        p0 = intralayer_params
        weights = None
        tb_energy = np.array(intralayer_tb_energy)
        

    bounds = None
    print(tb_energy)
    popt = fit_model(xdata,ydata,calc,methods,p0,bounds=bounds,weights=weights,fit_type="normal",tb_energy=tb_energy)
    print("final parameters = ",popt)
    #popt = p0
    np.savez("parameters/"+"_".join(loss.split(" "))+"_parameters_"+hopping_model_name+"_nkp"+str(nkp),parameters = popt)
    #np.savez("parameters/"+"_".join(loss.split(" "))+"_parameters_classical",parameters = popt)

    
        
