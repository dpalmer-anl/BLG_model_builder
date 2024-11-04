import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import argparse
from lammps import PyLammps
import ase.io
import os
import flatgraphene as fg
from ase.calculators.calculator import Calculator, all_changes
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


def get_relaxed_struct(atoms,calc):
    """ evaluate corrective potential energy, forces in lammps 
    """
    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        sym = atoms.get_chemical_symbols()
        top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        mol_id[top_layer_ind] += 1
        atoms.set_array("mol-id",mol_id)

    relax_atoms,forces = calc.relax_structure(atoms)
    return relax_atoms,forces

def get_twist_geom(theta,layer_sep=3.35,a=2.46):
    #comp is 2d vector for compression percentage along cell vectors
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                        mass=[12.01,12.01],sep=layer_sep,h_vac=20)
    return atoms

if __name__=="__main__":
    #define hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--energy_model',type=str,default="Classical")
    parser.add_argument('-t','--tb_model',type=str,default='MK')
    parser.add_argument('-q','--qoi',type=str,default="relax_atoms")
    parser.add_argument('-i','--ensemble_index',type=str,default="0")
    parser.add_argument('-n','--npartitions',type=str,default="1")
    parser.add_argument('-a','--theta',type=str,default="1.05")

    args = parser.parse_args() 
    Total_energy_type = args.energy_model
    hopping_model = args.tb_model
    qoi = args.qoi
    twist_angles = np.array([0.88,0.99,1.08,1.2,1.47,1.89,2.88])


    if hopping_model =="popov":
        eV_per_hartree = 27.2114
        interlayer_hopping_fxn = SK_pz_chebyshev
        interlayer_overlap_fxn = SK_pz_chebyshev
        intralayer_hopping_fxn = SK_pz_chebyshev
        intralayer_overlap_fxn = SK_pz_chebyshev
        hopping_model_name ="popov"

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
        interlayer_dsc_kwargs = {"type":"interlayer","cutoff":5.29177}
        intralayer_dsc_kwargs = {"type":"intralayer","cutoff":3.704239}

    elif hopping_model =="MK":
        mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
        
        cutoff = 10
        model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":cutoff},"cutoff":cutoff},
                    "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":cutoff},"cutoff":cutoff}}

    elif hopping_model == "letb":
        interlayer_hopping_params = np.load("../parameters/letb_interlayer_parameters.npz")["parameters"]
        intralayer_hopping_params = np.load("../letb_intralayer_parameters.npz")

        model_dict = {"interlayer":{"hopping form":letb_interlayer,"overlap form":None,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":None,
                                    "descriptors":letb_interlayer_descriptors},
        
                "intralayer":{"hopping form":letb_intralayer,"overlap form":None,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":None,
                                    "descriptors":letb_intralayer_descriptors}}

    calc = TETB_model(model_dict)
    param0 = calc.get_params()

    uq_type = "cv"
    if uq_type=="cv":
        interlayer_ensembles = np.load("ensembles/Classical_energy_interlayer_Kfold_n_8_ensemble.npz")["samples"]
        intralayer_ensembles = np.load("ensembles/Classical_energy_intralayer_Kfold_n_8_ensemble.npz")["samples"]
    elif uq_type=="mcmc":
        interlayer_ensembles = np.load("ensembles/Classical_energy_interlayer_mcmc_ensemble.npz")["ensembles"]
        intralayer_ensembles = np.load("ensembles/Classical_energy_intralayer_mcmc_ensemble.npz")["ensembles"]
    n_ensembles = np.min(np.array([np.shape(interlayer_ensembles)[0],np.shape(intralayer_ensembles)[0]]))
    print(n_ensembles)
    
    if qoi == "relax_atoms":

        if Total_energy_type == "Classical":
        
            interlayer_potential =  np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                                    0.719345289329483, 3.293082477932360, 13.906782892134125])
            intralayer_potential = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                            30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])
            model_dict = {"interlayer":{"hopping form":None,
                                    "potential":"kolmogorov/crespi/full 10.0 0","potential parameters":interlayer_potential,
                                    "potential file writer":write_kc},

                            "intralayer":{"hopping form":None,
                                    "potential":"rebo","potential parameters":intralayer_potential,"potential file writer":write_rebo}}
        calc = TETB_model(model_dict)

        twist_angles = np.array([0.88,0.99,1.08,1.12,1.16,1.2,1.47,1.89,2.88])
        twist_angles = [args.theta]

        for i,t in enumerate(twist_angles):
            relaxed_atoms_list = []
            for j in range(n_ensembles):

                calc.model_dict["intralayer"]["potential parameters"] = intralayer_ensembles[j,:]
                calc.model_dict["interlayer"]["potential parameters"] = interlayer_ensembles[j,:]
                params = calc.get_params()
                atoms = get_twist_geom(t)
                calc.set_params(params)
                relaxed_atoms,forces = get_relaxed_struct(atoms,calc)
                relaxed_atoms_list.append(relaxed_atoms)

            ase.io.write("relaxed_atoms_theta_"+str(t)+"_"+uq_type+"_ensemble.xyz",relaxed_atoms_list,format="extxyz")

    if qoi == "band_structure":
        relaxed_atoms_list = ase.io.read("relaxed_atoms_theta_"+str(t)+"_"+uq_type+"_ensemble.xyz",format="extxyz",index=":")
        band_ensemble = np.load("ensembles/TB_"+hopping_model+"_interlayer_mcmc_ensemble.npz")["ensembles"]
        ensemble_size = np.min(np.array([len(relaxed_atoms_list),np.shape(band_ensemble)[0]]))
        Gamma = [0,   0,   0]
        K = [1/3,2/3,0]
        Kprime = [2/3,1/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nk=60
        (kvec,k_dist, k_node) = k_path(sym_pts,nk)


        run_indices = np.array_split(np.arange(ensemble_size),int(args.npartitions))
        n_band_ensembles =  np.shape(band_ensemble)
        calc = TETB_model(model_dict)
        for i in run_indices[int(args.ensemble_index)]:
            if os.path.exists("ensemble_bands/band_structure_"+str(args.theta)+"_"+str(i)):
                continue
            model_dict["interlayer"]["hopping parameters"] = band_ensemble[int(i),:]
            calc = TETB_model(model_dict)
            print("number of atoms = ",len(relaxed_atoms_list[int(i)]))
            evals = calc.get_band_structure(relaxed_atoms_list[int(i)],kvec)
            np.savez("ensemble_bands/band_structure_"+str(args.theta)+"_"+str(i),evals=evals,kvec=kvec,k_dist=k_dist,k_node=k_node)
            del calc
            del evals


