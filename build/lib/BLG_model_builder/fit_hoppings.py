import numpy as np
import scipy
from scipy.optimize import curve_fit
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import random


def hopping_training_data(hopping_type="interlayer"):
    data = []
    # flist = subprocess.Popen(["ls", dataset],
    #                       stdout=subprocess.PIPE).communicate()[0]
    # flist = flist.decode('utf-8').split("\n")[:-1]
    # flist = [dataset+x for x in flist]
    flist = glob.glob('../TETB_GRAPHENE/data/hoppings/*.hdf5',recursive=True)
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
    return hoppings[type_ind],disp_array[type_ind]

def fit_hoppings(dft_hoppings,descriptors,hopping_function,init_params):

    popt,pcov = curve_fit(hopping_function,descriptors,dft_hoppings,p0=init_params)
    return popt

def mk_hopping(descriptors,a,b,c):
    #descriptors = displacements
    #parameters = a,b,c
    ang_per_bohr = 0.529177249 # Ang/bohr radius
    #[a,b,c] = parameters #units = [eV, 1/Angstroms, eV]
    z_val = descriptors[:,2]

    a0 = 2.68 * ang_per_bohr
    d0 = 6.33 * ang_per_bohr

    d = np.linalg.norm(descriptors,axis=1)
    n_sq = np.power(z_val,2) / d

    V_p = a * np.exp(-b * (d - a0))
    V_s = c * np.exp(-b * (d - d0))
    

    hop_val = V_p*(1 - n_sq) + V_s * n_sq

    return hop_val

if __name__=="__main__":
    # fit interlayer parameters
    rcut = 10
    interlayer_hoppings,interlayer_disp = hopping_training_data(hopping_type="interlayer")
    intralayer_hoppings,intralayer_disp = hopping_training_data(hopping_type="intralayer")
    all_hoppings = np.append(interlayer_hoppings,intralayer_hoppings)
    all_disp = np.vstack((interlayer_disp,intralayer_disp))

    init_params = np.array([-2.7, 2.2109794066373403, 0.48])
    interlayer_params = fit_hoppings(all_hoppings,all_disp,mk_hopping,init_params)
    print(interlayer_params)
    [a,b,c] = interlayer_params
    interlayer_fit_hoppings = mk_hopping(interlayer_disp,a,b,c)

    plt.scatter(np.linalg.norm(interlayer_disp,axis=1) / .529177,interlayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(interlayer_disp,axis=1)/ .529177,interlayer_fit_hoppings,label="SK")
    plt.xlabel("distance (bohr)")
    plt.ylabel("hoppings (eV)")
    plt.legend()
    plt.title("interlayer hoppings fit")
    plt.savefig("interlayer_hoppings.png")
    plt.clf()

    intralayer_fit_hoppings = mk_hopping(intralayer_disp,a,b,c)
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings.png")
    plt.clf()

    nn_dist = 1.42
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.xlim(0.95*nn_dist,1.05*nn_dist)
    plt.ylim(-85,-70)
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings_1nn.png")
    plt.clf()

    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.xlim(0.95*nn_dist*np.sqrt(3),1.05*nn_dist*np.sqrt(3))
    plt.ylim(-5,10)
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings_2nn.png")
    plt.clf()

    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.xlim(0.95*nn_dist*2,1.05*nn_dist*2)
    plt.ylim(-15,0)
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings_3nn.png")
    plt.clf()

    np.savez("MK_tb_params",a=a,b=b,c=c,distance_units="angstroms",energy_units="eV")
    
