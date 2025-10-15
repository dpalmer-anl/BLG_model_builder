import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
import ase.db
import flatgraphene as fg
import pandas as pd
import BLG_model_builder
from BLG_model_builder.TB_Utils_torch import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors_torch import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder_torch import *
import matplotlib.pyplot as plt

def get_tb_model(int_type,tb_model):
    model_dict = {"interlayer":{"hopping form":None,"overlap form":None,
                                "hopping parameters":None,"overlap parameters":None,
                                "descriptors":None,"descriptor kwargs":None,"cutoff":None},
    
            "intralayer":{"hopping form":None,"overlap form":None,
                                "hopping parameters":None,"overlap parameters":None,
                                "descriptors":None,"descriptor kwargs":None,"cutoff":None}}

    if tb_model=="popov":
        nparams = 20
        eV_per_hartree = 27.2114
        model_dict["interlayer"]["hopping form"] = popov_hopping
        model_dict["interlayer"]["overlap form"] = popov_overlap
        model_dict["intralayer"]["hopping form"] = porezag_hopping
        model_dict["intralayer"]["overlap form"] = porezag_overlap
        popov_hopping_pp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863]) *eV_per_hartree
        popov_hopping_pp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree
        popov_overlap_pp_pi = np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,0.0156376, 0.0025976, -0.0039498, 0.0020581,-0.0007114, 0.0001427])
        popov_overlap_pp_sigma = np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997, 0.0921727, -0.0268106, 0.0002240, 0.0040319, -0.0022450, 0.0005596])
        
        popov_hopping_pp_sigma_bounds = np.array([-1*np.ones_like(popov_hopping_pp_sigma),np.ones_like(popov_hopping_pp_sigma)])
        popov_hopping_pp_pi_bounds = np.array([-1*np.ones_like(popov_hopping_pp_sigma),np.ones_like(popov_hopping_pp_sigma)])
        popov_overlap_pp_sigma_bounds = np.array([-1*np.ones_like(popov_hopping_pp_sigma),np.ones_like(popov_hopping_pp_sigma)])
        popov_overlap_pp_pi_bounds = np.array([-1*np.ones_like(popov_hopping_pp_sigma),np.ones_like(popov_hopping_pp_sigma)])
        
        model_dict["interlayer"]["hopping parameters"] = np.append(popov_hopping_pp_sigma,popov_hopping_pp_pi)
        model_dict["interlayer"]["overlap parameters"] = np.append(popov_overlap_pp_sigma,popov_overlap_pp_pi)

        porezag_hopping_pp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree
        porezag_hopping_pp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree
        porezag_overlap_pp_pi =np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227])
        porezag_overlap_pp_sigma = np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751])
        model_dict["intralayer"]["hopping parameters"] = np.append(porezag_hopping_pp_sigma,porezag_hopping_pp_pi)
        model_dict["intralayer"]["overlap parameters"] = np.append(porezag_overlap_pp_sigma,porezag_overlap_pp_pi)

        model_dict["interlayer"]["hopping bounds"] = np.array([-1*np.ones(nparams),np.ones(nparams)])
        model_dict["interlayer"]["overlap bounds"] = np.array([-1*np.ones(nparams),np.ones(nparams)])
        model_dict["intralayer"]["hopping bounds"] = np.array([-1*np.ones(nparams),np.ones(nparams)])
        model_dict["intralayer"]["overlap bounds"] = np.array([-1*np.ones(nparams),np.ones(nparams)])


        model_dict["interlayer"]["descriptors"] = get_disp
        model_dict["interlayer"]["descriptor kwargs"] = {"type":"interlayer","cutoff":5.29}
        model_dict["interlayer"]["cutoff"] = 5.29
        model_dict["intralayer"]["descriptors"] = get_disp
        model_dict["intralayer"]["descriptor kwargs"] = {"type":"intralayer","cutoff":3.7}
        model_dict["intralayer"]["cutoff"] = 3.7

    elif tb_model=="MK":
        model_dict["interlayer"]["hopping form"] = mk_hopping
        model_dict["interlayer"]["hopping parameters"] = np.array([-2.92500706,  4.95594733,  0.34230107])
        model_dict["interlayer"]["descriptors"] = get_disp
        model_dict["interlayer"]["descriptor kwargs"] = {"type":"all","cutoff":5.29}
        model_dict["interlayer"]["cutoff"] = 5.29
        model_dict["interlayer"]["hopping bounds"] = np.array([[-5,-1e-5],
                            [1e-5,5],
                            [1e-5,5]])

    elif tb_model =="LETB":
        model_dict["interlayer"]["hopping form"] = letb_interlayer
        model_dict["intralayer"]["hopping form"] = letb_intralayer
        model_dict["interlayer"]["cutoff"] = 10
        model_dict["interlayer"]["descriptors"] = letb_interlayer_descriptors
        model_dict["intralayer"]["cutoff"] = 10
        model_dict["intralayer"]["descriptors"] = letb_intralayer_descriptors
        # a0, b0, c0, a3, b3, c3, a6, b6, c6, d6
        model_dict["interlayer"]["hopping parameters"] = np.load("../parameters/letb_interlayer_parameters.npz")["parameters"]
        model_dict["interlayer"]["hopping bounds"] = np.array([[-10,10], #a0
                                                [1e-5,10], #b0
                                                [1e-5,10], #c0
                                                [-10,10], #a3
                                                [1e-5,10], #b3
                                                [1e-5,10], #c3
                                                [-10,10], #a6
                                                [1e-5,10], #b6
                                                [1e-5,10], #c6
                                                [0,5] #d6
                                                ])
        model_dict["intralayer"]["hopping parameters"] = np.load("../parameters/letb_intralayer_parameters.npz")["parameters"]
        model_dict["interlayer"]["hopping bounds"] = np.array([[-15,-5], #t01 a
                                                [1e-5,10], #t01 b
                                                [1e-5,5], #t02 a
                                                [-5,-1e-5], #t02 b
                                                [-5,-1e-5], #t02 c
                                                [1e-5,5], #t02 d
                                                [-3,-1e-5], #t03 a
                                                [1e-5,5], #t03 b
                                                [-5,-1e-5], #t03 c
                                                [1e-5,5], #t03 d
                                                ])
        
    return model_dict

def get_energy_model(int_type,energy_model,calc_type):
    model_dict = {"interlayer":{"potential":None,"potential parameters":None,"potential file writer":None},
            "intralayer":{"potential":None,"potential parameters":None,"potential file writer":None}}
    use_rebo = False
    use_tersoff = True
    if energy_model=="Classical":
        if int_type == "interlayer" or int_type=="full":
            interlayer_params = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                                    0.719345289329483, 3.293082477932360, 13.906782892134125])
            #z0, C0, C2, C4, C, delta, lambda_val, A
            
            if calc_type =="lammps":
                model_dict["interlayer"]["potential"] = "kolmogorov/crespi/full 10.0 0"
                model_dict["interlayer"]["potential parameters"] = interlayer_params
                model_dict["interlayer"]["potential file writer"] = write_kc

            elif calc_type=="python":
                model_dict["interlayer"]["potential"] = Kolmogorov_Crespi
                model_dict["interlayer"]["potential parameters"] = interlayer_params

        if int_type=="intralayer" or int_type=="full":
            
            if calc_type=="lammps":
                if use_rebo:
                    intralayer_params = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                                30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])
                    model_dict["intralayer"]["potential"] = "rebo"
                    model_dict["intralayer"]["potential parameters"] = intralayer_params
                    model_dict["intralayer"]["potential file writer"] = write_rebo
                elif use_tersoff:
                    intralayer_params = np.array([3.8049e4, 4.3484, -0.93000, 0.72751, 1.5724e-7,  2.2119,  430.00,   3.4879,  1393.6])
                    model_dict["intralayer"]["potential"] = "tersoff"
                    model_dict["intralayer"]["potential parameters"] = intralayer_params
                    model_dict["intralayer"]["potential file writer"] = write_Tersoff

            elif calc_type=="python":
                #c, d, costheta0, n, beta, lambda2, B, lambda1, A
                intralayer_params = np.array([3.8049e4, 4.3484, -0.93000, 0.72751, 1.5724e-7,  2.2119,  430.00,   3.4879,  1393.6])
                model_dict["intralayer"]["potential"] = Tersoff
                model_dict["intralayer"]["potential parameters"] = intralayer_params

    elif energy_model=="TETB": 
        if int_type=="interlayer" or int_type=="full":
            model_dict["interlayer"]["potential parameters"] = np.array([3.4384,34.0449, -17.1697, 17.22962, -23.044, 3.07925, -1.5484, 10.7840, -7.145953])
            # delta,C,C0,C2,C4,z0,A6,A8,A10
            model_dict["interlayer"]["potential bounds"] = np.array([[2,4],[-10000,10000],[-10000,10000],[-100,100],[-10000,10000],[1e-2,10],[1e-2,10],[-10000,10000],[-10000,10000],[-10000,10000],[-10000,10000]])
            if calc_type == "python":
                model_dict["interlayer"]["potential"] = Kolmogorov_Crespi_insp
                #model_dict["interlayer"]["potential"] = Kolmogorov_Crespi #_vdw
            else:
                model_dict["interlayer"]["potential"] = "reg/dep/poly 10.0 0"
                model_dict["interlayer"]["potential file writer"] = write_kcinsp

                #model_dict["interlayer"]["potential"] = "kolmogorov/crespi/full 10.0 0"
                #model_dict["interlayer"]["potential file writer"] = write_kc

        if int_type=="intralayer" or int_type=="full":
            if calc_type =="python":
                intralayer_params = np.array([ 4.0381772,  16.79935969, -5.1949946,  1.40171327, 26.16975888,  0.04642429,2.94080086,  5.1350953,   2.19817537])
                model_dict["intralayer"]["potential"] = Tersoff
                model_dict["intralayer"]["potential parameters"] = intralayer_params
            else:
                if use_rebo:
                    model_dict["intralayer"]["potential"] = "rebo"
                    model_dict["intralayer"]["potential parameters"] = np.load("../uncertainty_quantification/parameters/intralayer_energy_parameters_popov_nkp121.npz")["parameters"]
                    model_dict["intralayer"]["potential file writer"] = write_rebo
                elif use_tersoff:
                    intralayer_params = np.array([ 4.0381772,  16.79935969, -5.1949946,  1.40171327, 26.16975888,  0.04642429,2.94080086,  5.1350953,   2.19817537])
                    model_dict["intralayer"]["potential"] = "tersoff"
                    model_dict["intralayer"]["potential parameters"] = intralayer_params
                    model_dict["intralayer"]["potential file writer"] = write_Tersoff
            
    return model_dict 

def get_BLG_Model(int_type="interlayer",energy_model="Classical",tb_model="MK",calc_type="lammps",output=None,update_eigvals = 1,**kwargs):
    model_dict = {"interlayer":{"hopping form":None,"overlap form":None,
                                "hopping parameters":None,"overlap parameters":None,
                                "descriptors":None,"descriptor kwargs":None,"cutoff":None,
                                "potential":None,"potential parameters":None},
    
            "intralayer":{"hopping form":None,"overlap form":None,
                                "hopping parameters":None,"overlap parameters":None,
                                "descriptors":None,"descriptor kwargs":None,"cutoff":None,
                                "potential":None,"potential parameters":None}}

    if tb_model is not None:
        tb_model_dict = get_tb_model(int_type,tb_model) #,**kwargs)
        for lt in tb_model_dict.keys():
            for et in tb_model_dict[lt].keys():
                model_dict[lt][et] = tb_model_dict[lt][et]
    if energy_model is not None:
        
        energy_model_dict = get_energy_model(int_type,energy_model,calc_type)
        for lt in energy_model_dict.keys():
            for et in energy_model_dict[lt].keys():
                model_dict[lt][et] = energy_model_dict[lt][et]
    if calc_type=="lammps":
        use_lammps=True
    else:
        use_lammps=None
    
    ase_calc = TETB_model(model_dict,output=output,update_eigvals=update_eigvals,use_lammps=use_lammps,**kwargs) 
    opt_params = ase_calc.get_params()
    #ase_calc.set_params(opt_params)
    return ase_calc

def get_BLG_Evaluator(int_type="interlayer",energy_model="Classical",tb_model="MK",nn_val=None,calc_type="python",energy_type="total",**kwargs):
    """return most efficient method to evaluate interaction type """
    #TB only evaluators
    evaluator = {}
    if energy_model is None:
        if tb_model == "MK":
            evaluator["hoppings"] = mk_hopping

        elif int_type=="interlayer":
            if tb_model =="SK":  
                evaluator["hoppings"] = calc.SK_hopping_transform
            elif tb_model=="LETB":
                evaluator["hoppings"] = letb_interlayer
            elif tb_model=="polynomial":
                evaluator["hoppings"] = poly_func

        elif int_type=="intralayer":
            if tb_model =="SK":
                evaluator["hoppings"] = calc.SK_hopping_transform
            elif tb_model=="LETB":
                if nn_val==1:
                    evaluator["hoppings"] = letb_intralayer_t01
                elif nn_val==2:
                    evaluator["hoppings"] = letb_intralayer_t02
                elif nn_val==3:
                    evaluator["hoppings"] = letb_intralayer_t03
                else:
                    evaluator["hoppings"] = letb_intralayer
            elif tb_model=="polynomial":
                evaluator["hoppings"] = poly_func
    elif energy_model=="Classical":
        calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type=calc_type)
        if tb_model is None:
            evaluator["energy"] = calc.evaluate_total_energy

    elif energy_model=="TETB":
        calc = get_BLG_Model(int_type=int_type,energy_model=energy_model,tb_model=tb_model,calc_type=calc_type)
        if tb_model=="MK":
            evaluator["hoppings"] = mk_hopping
        elif tb_model=="popov":
             dummy = None
        if energy_type=="total":
            evaluator["energy"] = calc.evaluate_total_energy 
        elif energy_type=="residual":
            evaluator["energy"] = calc.evaluate_residual_energy 
        
    else:
        print("Error: energy must be set to None, Classical, or TETB")
        exit()
    return evaluator
    

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

def create_Dataset(interlayer_df,intralayer_db, supercells):

    interlayer_atom_list = []
    interlayer_energies = []
    interlayer_uncertainties = []
    intralayer_atom_list = []
    intralayer_energies = []
    stacking_ = ["AB","SP","Mid","AA"]
    disreg_ = [0 , 0.16667, 0.5, 0.66667]
    
    for i,stacking in enumerate(stacking_):
        dis = disreg_[i]
        d_stack = interlayer_df.loc[interlayer_df['stacking'] == stacking, :]
        for j, row in d_stack.iterrows():
            atoms = get_bilayer_atoms(row["d"],dis,sc=supercells)
            pos = atoms.positions
            mean_z = np.mean(pos[:,2])
            top_ind = np.where(pos[:,2]>mean_z)
            bot_ind = np.where(pos[:,2]<mean_z)
            d = np.mean(np.abs(pos[top_ind,2]-pos[bot_ind,2]))
            mol_id = np.ones(len(atoms),dtype=np.int64)
            mol_id[top_ind] = 2
            atoms.set_array("mol-id",mol_id)

            top_layer_ind = np.where(pos[:,2]>mean_z)
            top_pos = np.squeeze(pos[top_layer_ind,:])
            bot_layer_ind = np.where(pos[:,2]<mean_z)
            bot_pos = np.squeeze(pos[bot_layer_ind,:])

            interlayer_atom_list.append(atoms)
            interlayer_energies.append(row["energy"]*len(atoms))
            interlayer_uncertainties.append(row["energy_err"]*len(atoms))

    for i,row in enumerate(intralayer_db.select()):
        atoms = intralayer_db.get_atoms(id = row.id)
        atoms.set_array("mol-id",np.ones(len(atoms),dtype=np.int64))

        intralayer_atom_list.append(atoms)
        intralayer_energies.append(row.data.total_energy*len(atoms))

    return interlayer_atom_list,interlayer_energies,interlayer_uncertainties,intralayer_atom_list,intralayer_energies

def get_training_data(model_name,supercells=5,nn_val=None):
    xdata = {} 
    ydata = {} 
    ydata_noise = {}
    if model_name == "interlayer energy":
        interlayer_df =  pd.read_csv('../data/qmc.csv') 
        intralayer_db =  ase.db.connect('../data/monolayer_nkp121.db')
        interlayer_atom_list,interlayer_energies,interlayer_uncertainties,intralayer_atom_list,intralayer_energies = create_Dataset(interlayer_df,intralayer_db,supercells)
        xdata["energy"] = interlayer_atom_list
        ydata["energy"] = np.array(interlayer_energies)
        ydata["energy"] -= np.min(ydata["energy"]) #only want interlayer energies
        ydata_noise["energy"] = np.array(interlayer_uncertainties)

    elif model_name == "intralayer energy":
        interlayer_df =  pd.read_csv('../data/qmc.csv') 
        intralayer_db =  ase.db.connect('../data/monolayer_nkp121.db')
        interlayer_atom_list,interlayer_energies,interlayer_uncertainties,intralayer_atom_list,intralayer_energies = create_Dataset(interlayer_df,intralayer_db,supercells)
        xdata["energy"] = intralayer_atom_list
        ydata["energy"] = np.array(intralayer_energies)-np.min(intralayer_energies)
        ydata_noise["energy"] = np.zeros_like(ydata["energy"])

    elif model_name == "MK hoppings":
        hopping_data = hopping_training_data(hopping_type="all")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = xdata_list[0]
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.vstack((xdata["hoppings"],xdata_list[i]))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])

    elif model_name == "popov hoppings":
        hopping_data = hopping_training_data(hopping_type="interlayer")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = xdata_list[0]
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.vstack((xdata["hoppings"],xdata_list[i]))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])
    
    elif model_name == "porezag hoppings":
        hopping_data = hopping_training_data(hopping_type="intralayer")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = xdata_list[0]
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.vstack((xdata["hoppings"],xdata_list[i]))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])

    elif model_name =="TETB_energy_interlayer_popov hoppings":
        hopping_data = hopping_training_data(hopping_type="interlayer")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = xdata_list[0]
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.vstack((xdata["hoppings"],xdata_list[i]))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hopings"])

        interlayer_df =  pd.read_csv('../data/qmc.csv') 
        intralayer_db = db = ase.db.connect('../data/monolayer_nkp121.db')
        interlayer_atom_list,interlayer_energies,interlayer_uncertainties,intralayer_atom_list,intralayer_energies = create_Dataset(interlayer_df,intralayer_db,supercells)
        xdata["hoppings"] = interlayer_atom_list
        ydata["hoppings"] = np.array(interlayer_energies)
        ydata["hoppings"] -= ydata["hoppings"][-1] #only want interlayer energies
        ydata_noise["hoppings"] = np.array(interlayer_uncertainties)

    elif model_name =="TETB_energy_intralayer_porezag hoppings":
        hopping_data = hopping_training_data(hopping_type="intralayer")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = xdata_list[0]
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.vstack((xdata["hoppings"],xdata_list[i]))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])

        interlayer_df =  pd.read_csv('../data/qmc.csv') 
        intralayer_db = db = ase.db.connect('../data/monolayer_nkp121.db')
        interlayer_atom_list,interlayer_energies,interlayer_uncertainties,intralayer_atom_list,intralayer_energies = create_Dataset(interlayer_df,intralayer_db,supercells)
        xdata["hoppings"] = intralayer_atom_list
        ydata["hoppings"] = np.array(intralayer_energies)
        ydata_noise["hoppings"] = np.zeros_like(intralayer_energies)

    elif "intralayer_LETB hoppings" in model_name:
        intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")

        atoms_list = intralayer_hopping_data["atoms"]
        i_list = intralayer_hopping_data["i"]
        j_list = intralayer_hopping_data["j"]
        di_list = intralayer_hopping_data["di"]
        dj_list = intralayer_hopping_data["dj"]
        disp_list = intralayer_hopping_data["disp"]

        cell = atoms_list[0].get_cell()
        pos = atoms_list[0].positions
        dsc, ix = letb_intralayer_descriptors_array(cell,disp_list[0],pos, di_list[0], dj_list[0], i_list[0], j_list[0],nn_val=nn_val)
        xdata["hoppings"] = dsc
        ydata["hoppings"] = intralayer_hopping_data['hopping'][0][ix]
        for atom_i in range(1,len(atoms_list)):
            cell = atoms_list[atom_i].get_cell()
            pos = atoms_list[atom_i].positions

            dsc,ix = letb_intralayer_descriptors_array(cell,disp_list[atom_i],pos, di_list[atom_i], dj_list[atom_i], i_list[atom_i], j_list[atom_i],nn_val=nn_val)
            if dsc.ndim <2:
                xdata["hoppings"] = np.append(xdata["hoppings"],dsc)
            else:
                xdata["hoppings"] = np.vstack((xdata["hoppings"],dsc))
            ydata["hoppings"] = np.append(ydata["hoppings"],intralayer_hopping_data['hopping'][atom_i][ix])

        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])

    elif model_name == "interlayer_LETB hoppings":
        interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
        atoms_list = interlayer_hopping_data["atoms"]
        i_list = interlayer_hopping_data["i"]
        j_list = interlayer_hopping_data["j"]
        di_list = interlayer_hopping_data["di"]
        dj_list = interlayer_hopping_data["dj"]
        disp_list = interlayer_hopping_data["disp"]

        cell = atoms_list[0].get_cell()
        pos = atoms_list[0].positions
        dsc = letb_interlayer_descriptors_array(cell,disp_list[0],pos, di_list[0], dj_list[0], i_list[0], j_list[0])
        xdata["hoppings"] = dsc
        ydata["hoppings"] = interlayer_hopping_data['hopping'][0]
        for atom_i in range(1,len(atoms_list)):
            cell = atoms_list[atom_i].get_cell()
            pos = atoms_list[atom_i].positions
            dsc = letb_interlayer_descriptors_array(cell,disp_list[atom_i],pos, di_list[atom_i], dj_list[atom_i], i_list[atom_i], j_list[atom_i])
            xdata["hoppings"] = np.vstack((xdata["hoppings"],dsc))

            ydata["hoppings"] = np.concatenate((ydata["hoppings"],interlayer_hopping_data['hopping'][atom_i]))
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])
        
    elif model_name =="interlayer_polynomial hoppings":
        hopping_data = hopping_training_data(hopping_type="interlayer")
        xdata_list =hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = np.linalg.norm(xdata_list[0],axis=1)
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.append(xdata["hoppings"],np.linalg.norm(xdata_list[i],axis=1))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])
    elif model_name =="intralayer_polynomial hoppings":
        hopping_data = hopping_training_data(hopping_type="intralayer")
        xdata_list = hopping_data["disp"]
        ydata_list = hopping_data["hopping"]
        xdata["hoppings"] = np.linalg.norm(xdata_list[0],axis=1)
        ydata["hoppings"] = ydata_list[0]
        for i in range(1,len(xdata_list)):
            xdata["hoppings"] = np.append(xdata["hoppings"],np.linalg.norm(xdata_list[i],axis=1))
            ydata["hoppings"] = np.append(ydata["hoppings"],ydata_list[i])
        ydata_noise["hoppings"] = np.zeros_like(ydata["hoppings"])

    return xdata, ydata, ydata_noise


