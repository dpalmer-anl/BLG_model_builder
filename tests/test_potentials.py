import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
import ase.db
import flatgraphene as fg
from scipy.spatial import distance
from ase.build import make_supercell
import pandas as pd
import BLG_model_builder
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import matplotlib.pyplot as plt


if __name__=="__main__":
    """ run mcmc
    $ export MPIEXEC_OPTIONS="--bind-to core --map-by slot:PE=<num_openmp_processes> port-bindings"
    $ mpiexec -np <num_mpi_workers> ${MPIEXEC_OPTIONS} python script.py
    """
    csfont = {'fontname':'serif',"size":18}

    test_intralayer_lat_con=False
    test_intralayer=True
    test_interlayer=False
    convergence_test = False

    model_type = "TETB" #"Classical"
    tb_model = "MK"
    eV_per_hartree = 27.2114

    if model_type=="Classical":
        interlayer_potential = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                              0.719345289329483, 3.293082477932360, 13.906782892134125])
        intralayer_potential = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])
        model_dict = {"interlayer":{"hopping form":None,
                                "potential":"kolmogorov/crespi/full 10.0 0","potential parameters":interlayer_potential,
                                "potential file writer":write_kc},

                        "intralayer":{"hopping form":None,
                                "potential":"rebo","potential parameters":intralayer_potential,"potential file writer":write_rebo}}


    elif model_type == "TETB":
        if tb_model=="MK":
            mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
            interlayer_params = np.load("../uncertainty_quantification/parameters/interlayer_energy_parameters_popov_nkp121.npz")["parameters"]

            intralayer_params = np.array([0.14687637217609084, 4.683462616941604, 12433.64356176609,\
                12466.479169306709, 19.121905577450008, 30.504342033258325,\
                4.636516235627607 , 1.3641304165817836, 1.3878198074813923])
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

        elif tb_model=="popov":
            interlayer_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406,
                                   -103.18388323245665, 1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
            intralayer_params = np.array([0.14687637217609084, 4.683462616941604, 12433.64356176609,\
                12466.479169306709, 19.121905577450008, 30.504342033258325,\
                4.636516235627607 , 1.3641304165817836, 1.3878198074813923])
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


    if convergence_test:
        
        stacking_ = ["AB","SP","Mid","AA"]
        disreg_ = [0 , 0.16667, 0.5, 0.66667]
        colors = ["blue","red","black","green"]
        d_ = np.linspace(3,5,5)
        df = pd.read_csv('../data/qmc.csv') 
        d_ab = df.loc[df['disregistry'] == 0, :]
        min_ind = np.argmin(d_ab["energy"].to_numpy())
        E0_qmc = d_ab["energy"].to_numpy()[min_ind]
        d = d_ab["d"].to_numpy()[min_ind]
        disreg = d_ab["disregistry"].to_numpy()[min_ind]
        relative_tetb_energies = []
        relative_qmc_energies = []
        all_tb_energies = []
        E0_tegt = 0

        ncells = np.arange(1,14,1)
        tb_energy = np.zeros(len(ncells))
        for i,n in enumerate(ncells):
            atoms = get_bilayer_atoms(3.5,0,sc=n)
            tb_energy[i] = calc.get_tb_energy(atoms)/len(atoms)
        plt.plot(ncells,tb_energy)
        plt.savefig("cell_convergence.png")
        plt.clf()


    if test_interlayer:
        use_lammps=False
        if use_lammps:
            md_calc = "lammps"
            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177},"cutoff":5.29177,
                                    "potential":"reg/dep/poly 10.0 0","potential parameters":interlayer_params,
                                    "potential file writer":write_kcinsp},
                                    #"potential":Kolmogorov_Crespi_insp,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239},"cutoff":3.704239,
                                    "potential":"rebo","potential parameters":intralayer_params,"potential file writer":write_rebo}}
        else:
            md_calc = "python"
            interlayer_params = np.load("../uncertainty_quantification/parameters/interlayer_energy_parameters_mk_nkp121.npz")["parameters"]
            #'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
            interlayer_params[1] /= 1.30
            model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177},"cutoff":5.29177,
                                    "potential":interlayer_potential,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239},"cutoff":3.704239,
                                    "potential":None,"potential parameters":None}}
            
        print("Model calcuator = ",md_calc)
        print("Model type = TETB interlayer")
        calc = TETB_model(model_dict)

        stacking_ = ["AB","SP","Mid","AA"]
        disreg_ = [0 , 0.16667, 0.5, 0.66667]
        colors = ["blue","red","black","green"]
        d_ = np.linspace(3,5,5)
        df = pd.read_csv('../data/qmc.csv') 
        d_ab = df.loc[df['disregistry'] == 0, :]
        min_ind = np.argmin(d_ab["energy"].to_numpy())
        E0_qmc = d_ab["energy"].to_numpy()[min_ind]
        d = d_ab["d"].to_numpy()[min_ind]
        disreg = d_ab["disregistry"].to_numpy()[min_ind]
        relative_tetb_energies = []
        relative_qmc_energies = []
        all_tb_energies = []
        E0_tegt = 1e10
        
        for i,stacking in enumerate(stacking_):
            energy_dis_tegt = []
            energy_dis_qmc = []
            energy_dis_tb = []
            residual_energy = []
            d_ = []
            dis = disreg_[i]
            d_stack = df.loc[df['stacking'] == stacking, :]
            for j, row in d_stack.iterrows():
                atoms = get_bilayer_atoms(row["d"],dis)

                total_energy = (calc.get_total_energy(atoms))/len(atoms)
                residual_energy_tmp = calc.get_residual_energy(atoms)/len(atoms)
                residual_energy.append(residual_energy_tmp)
                tb_energy,tb_forces = calc.get_tb_energy(atoms)
                tb_energy /= len(atoms)
                #total_energy = tb_energy + residual_energy_tmp
                #total_energy = Kolmogorov_Crespi(atoms,kc_parameters)
                print(total_energy)
                
                if total_energy<E0_tegt:
                    E0_tegt = total_energy

                qmc_total_energy = (row["energy"])

                energy_dis_tegt.append(total_energy)
                energy_dis_qmc.append(qmc_total_energy)
                energy_dis_tb.append(tb_energy)
                d_.append(row["d"])

            relative_tetb_energies.append(energy_dis_tegt)
            relative_qmc_energies.append(energy_dis_qmc)
            #plt.scatter(np.array(d_),np.array(residual_energy)-(residual_energy[-1]),label=stacking + " residual energy",c=colors[i],marker="*")
            plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " "+model_type,c=colors[i])
            #plt.plot(d_,(np.array(energy_dis_tegt)-E0_tegt)-(np.array(energy_dis_qmc)-E0_qmc),label=stacking + " QMC Diff. "+model_type,c=colors[i])
            #plt.scatter(np.array(d_),np.array(energy_dis_tb)-(energy_dis_tb[-1]),label=stacking + " TB",c=colors[i],marker=",")
            plt.scatter(np.array(d_),np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])

        layer_sep = np.array([3.3266666666666667,3.3466666666666667,3.3866666666666667,3.4333333333333336,3.5,3.5733333333333333,3.6466666666666665,3.7666666666666666,3.9466666666666668,4.113333333333333,4.3533333333333335,4.54,4.76,5.013333333333334,5.16])
        popov_energies_sep = np.array([ 0.0953237410071943, 0.08884892086330941, 0.07877697841726625, 0.06582733812949645, 0.05323741007194249, 0.042086330935251826, 0.03237410071942448, 0.02230215827338132, 0.01151079136690649, 0.007194244604316571, 0.0025179856115108146, 0.0010791366906475058, 0.0007194244604316752, 0.00035971223021584453, 1.3877787807814457e-17])
        #plt.plot(layer_sep,popov_energies_sep,label="popov tb energies")
        plt.xlabel(r"Interlayer Distance ($\AA$)",**csfont)
        plt.ylabel("TETB Interlayer Energy (eV)",**csfont)
        plt.title(model_type,**csfont)
        plt.tight_layout()
        plt.legend()
        plt.savefig("figures/interlayer_test"+model_type+".jpg")
        plt.clf()

    if test_intralayer_lat_con:
        a = 2.462
        lat_con_list = np.sqrt(3) * np.array([1.197813121272366,1.212127236580517,1.2288270377733599,1.2479125248508947,\
                            1.274155069582505,1.3027833001988072,1.3433399602385685,1.4053677932405566,\
                            1.4745526838966203,1.5294234592445326,1.5795228628230618])

        lat_con_energy = np.zeros_like(lat_con_list)
        tb_energy = np.zeros_like(lat_con_list)
        rebo_energy = np.zeros_like(lat_con_list)
        dft_energy = np.array([-5.62588911,-6.226154186,-6.804241219,-7.337927988,-7.938413961,\
                            -8.472277446,-8.961917385,-9.251954937,-9.119902805,-8.832030042,-8.432957809])

        for i,lat_con in enumerate(lat_con_list):
        
            atoms = get_monolayer_atoms(0,0,a=lat_con)
            atoms.set_array('mol-id',np.ones(len(atoms),dtype=np.int64))
            print("a = ",lat_con," natoms = ",len(atoms))
            total_energy = calc.get_total_energy(atoms)/len(atoms)
            #tb_energy_geom,tb_forces = calc_obj.run_tight_binding(atoms)
            #tb_energy[i] = tb_energy_geom/len(atoms)
            #lammps_forces,lammps_pe,tote = calc_obj.run_lammps(atoms)
            #rebo_energy[i] = total_energy/len(atoms)
            #total_energy = tote + tb_energy_geom
            lat_con_energy[i] = total_energy
        """fit_min_ind = np.argmin(lat_con_energy)
        initial_guess = (1.0, 1.0, 1.0)  # Initial parameter guess
        rebo_params, covariance = curve_fit(quadratic_function, lat_con_list, lat_con_energy, p0=initial_guess)
        rebo_min = np.min(lat_con_energy*len(atoms))

        dft_min_ind = np.argmin(dft_energy)
        initial_guess = (1.0, 1.0, 1.0)  # Initial parameter guess
        dft_params, covariance = curve_fit(quadratic_function, lat_con_list, dft_energy, p0=initial_guess)
        dft_min = dft_params[-1]

        print("rebo fit minimum energy = ",str(rebo_params[-1]))
        print("rebo fit minimum lattice constant = ",str(lat_con_list[fit_min_ind]))
        print("rebo young's modulus = ",str(rebo_params[0]))
        print("DFT minimum energy = ",str(dft_params[-1]))
        print("DFT minimum lattice constant = ",str(lat_con_list[dft_min_ind]))
        print("DFT young's modulus = ",str(dft_params[0]))"""

        plt.plot(lat_con_list/np.sqrt(3),lat_con_energy-np.min(lat_con_energy),label = "rebo fit")
        #plt.plot(lat_con_list/np.sqrt(3),tb_energy-tb_energy[fit_min_ind],label = "tight binding energy")
        #plt.plot(lat_con_list/np.sqrt(3),rebo_energy - rebo_energy[fit_min_ind],label="rebo corrective energy")
        plt.plot(lat_con_list/np.sqrt(3), dft_energy-np.min(dft_energy),label="dft results")
        plt.xlabel(r"nearest neighbor distance ($\AA$)")
        plt.ylabel("energy above ground state (eV/atom)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("rebo_lat_con_"+model_type+".jpg")
        plt.clf()

    if test_intralayer:
        mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
        rebo_parameters = np.load("../uncertainty_quantification/parameters/intralayer_energy_parameters_mk_nkp121.npz")["parameters"]
        model_dict  ={"interlayer":{"hopping form":mk_hopping,"hopping parameters":mk_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer"},
                                "potential":None,"potential parameters":None},
            "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                "hopping parameters":mk_params,"overlap parameters":None,
                                "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer"},
                                "potential":"rebo","potential parameters":rebo_parameters,"potential file writer":write_rebo}}

        calc = TETB_model(model_dict)
        db = ase.db.connect('../data/monolayer_nkp121.db')
        energy = []
        
        nconfig=0
        dft_min = 1e8
        for row in db.select():
            if row.data.total_energy<dft_min:
                dft_min = row.data.total_energy
        tegtb_energy = []
        dft_energy = []   
        nn_dist = []
        atoms_id =[]
        unstrained_atoms = get_monolayer_atoms(0,0,a=2.462)
        unstrained_cell = unstrained_atoms.get_cell()
        
        for row in db.select():
    
            atoms = db.get_atoms(id = row.id)
            atoms_id.append(row.id)

            e = calc.get_total_energy(atoms)/len(atoms)
            tegtb_energy.append(e)
            dft_energy.append(row.data.total_energy)
            nconfig+=1

            pos = atoms.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            nn_dist.append(average_distance)
        dft_min = np.min(dft_energy)
        rebo_min_ind = np.argmin(tegtb_energy)
        rebo_min = tegtb_energy[rebo_min_ind]

        rms_tetb  = []
        rms_rebo = []
        for i,e in enumerate(tegtb_energy):
            line = np.linspace(0,1,10)
            ediff_line = line*((dft_energy[i]-dft_min) - (e-rebo_min)) + (e-rebo_min)
            tmp_rms = np.linalg.norm((dft_energy[i]-dft_min) - (e-rebo_min))/(dft_energy[i]-dft_min)

            #if tmp_rms >0.15:
            #    del db[atoms_id[i]]
            #    continue
            print("dft energy (eV/atom) = ",dft_energy[i]-dft_min)
            print("tegtb energy (eV/atom) = ",e-rebo_min)
            print("\n")
            average_distance = nn_dist[i]
            if nn_dist[i] > 1.5 or (dft_energy[i]-dft_min)>0.4:
                continue
            rms_tetb.append(tmp_rms)

            if i==0:
                plt.scatter(average_distance,e-rebo_min,color="red",label=model_type)
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue",label="DFT")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
            else:
                plt.scatter(average_distance,e-rebo_min,color="red")
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
        
        print("rms tetb ",rms_tetb)

        rms_tetb = np.array(rms_tetb)
        rms_rebo = np.array(rms_rebo)
        rms_tetb = rms_tetb[rms_tetb<1e3]
        rms_rebo = rms_rebo[rms_rebo<1e3]
        rms_tetb = np.mean(rms_tetb)
        rms_rebo = np.mean(rms_rebo)
        #rms_tetb = np.mean(np.abs(np.array(tegtb_energy)-rebo_min-(np.array(dft_energy)-dft_min)))
        #rms_rebo = np.mean(np.abs(np.array(rebo_energy)-emprebo_min-(np.array(dft_energy)-dft_min)))
        print("average rms tetb = ",rms_tetb)
        
        print("average difference in tetb energy across all configurations = "+str(rms_tetb)+" (eV/atom)")
        print("average difference in rebo energy across all configurations = "+str(rms_rebo)+" (eV/atom)")
        plt.xlabel(r"average nearest neighbor distance ($\AA$)",**csfont)
        plt.ylabel("energy (eV/atom)",**csfont)
        plt.title(model_type+" intralayer energy",**csfont)

        plt.legend()
        #plt.colorbar().set_label('RMS', rotation=270,**csfont)
        #plt.clim((1e-5,1e-4))
        plt.tight_layout()
        plt.savefig("figures/intralayer_test_"+model_type+".jpg")
        plt.clf()
