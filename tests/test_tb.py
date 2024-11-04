import numpy as np
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
import h5py
import glob

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

if __name__=="__main__":
    TB_model ="LETB" #"MK" "LETB" or "popov"

    #define twisted graphene system
    sep = 3.35
    a = 2.46
    n=5
    theta=9.4
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta,a_tol=2e-1)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                    mass=[12.01,12.01],sep=sep,h_vac=20)

    if TB_model =="MK":
        mk_params = np.array([-2.7, 2.2109794066373403, 0.48])
        cutoff = 10
        model_dict = {"interlayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":cutoff},"cutoff":cutoff},
                    "intralayer":{"hopping form":mk_hopping,"overlap form":None,
                                    "hopping parameters":mk_params,"overlap parameters":None,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":cutoff},"cutoff":cutoff}}

        """interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
        intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")
        xdata = np.concatenate((interlayer_hopping_data['disp'],intralayer_hopping_data['disp']),axis=0)
        r = np.linalg.norm(xdata,axis=1)
        ydata = np.concatenate((interlayer_hopping_data['hopping'],intralayer_hopping_data["hopping"]),axis=0)

        plt.scatter(r,ydata,label="ab initio")
        plt.scatter(r,mk_hopping(xdata,mk_params),label="MK")
        plt.xlabel("r")
        plt.ylabel("t")
        plt.savefig("figures/MK_hoppings_fit.png")
        plt.clf()"""

    elif TB_model =="popov":
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

        model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "hopping kwargs":{"b":5.29177},"overlap kwargs":{"b":5.29177},
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177}},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "hopping kwargs":{"b":3.704239},"overlap kwargs":{"b":3.704239},
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239}}}

    elif TB_model=="LETB":
        
        interlayer_hopping_params = np.load("../parameters/letb_interlayer_parameters.npz")["parameters"]
        intralayer_hopping_params = np.load("../parameters/letb_intralayer_parameters.npz")["parameters"]

        model_dict = {"interlayer":{"hopping form":letb_interlayer,"overlap form":None,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":None,
                                    "descriptors":letb_interlayer_descriptors},
        
                "intralayer":{"hopping form":letb_intralayer,"overlap form":None,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":None,
                                    "descriptors":letb_intralayer_descriptors}}

    calc = TETB_model(model_dict)
    Gamma = [0,   0,   0]
    K = [1/3,2/3,0]
    Kprime = [2/3,1/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=60
    (kvec,k_dist, k_node) = k_path(sym_pts,nk)

    erange = 1.5
    evals = calc.get_band_structure(atoms,kvec)
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
    ax.set_title(str(theta)+" twist graphene")
    ax.set_xlabel("Path in k-space")
    
    nbands = np.shape(evals)[0]
    efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
    fermi_ind = (nbands)//2

    for n in range(np.shape(evals)[0]):
        ax.plot(k_dist,evals[n,:]-efermi,color="black")
        
    # make an PDF figure of a plot
    #fig.tight_layout()
    ax.set_ylim(-erange,erange)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_xticks(k_node)
    ax.set_xticklabels(["K", "$\Gamma$", "M", "$K^\prime$"])
    ax.set_xlim(k_node[0], k_node[-1])
    fig.savefig("figures/"+TB_model+"_theta_"+str(theta)+"_graphene.png", bbox_inches='tight')
    plt.clf()
