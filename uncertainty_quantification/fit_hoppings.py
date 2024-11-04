import numpy as np
import scipy
from scipy.optimize import curve_fit
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import random
from BLG_model_builder.TB_Utils import *


def hopping_training_data(hopping_type="interlayer"):
    data = []
    # flist = subprocess.Popen(["ls", dataset],
    #                       stdout=subprocess.PIPE).communicate()[0]
    # flist = flist.decode('utf-8').split("\n")[:-1]
    # flist = [dataset+x for x in flist]
    flist = glob.glob('../../TETB_GRAPHENE/data/hoppings/*.hdf5',recursive=True)
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

def sk_hopping(dR,sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,
               pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10):

    """pairwise Slater Koster Interlayer hopping parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    Cpp_sigma = np.array([sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10])
    Cpp_pi = np.array([pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10])

    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)

    aa = 1.0 * .529177  # [Bohr radii]
    b = 10.0 * .529177  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat

def sk_chebval(y,sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10):
    bond_int = np.array([sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10]) 
    return np.polynomial.chebyshev.chebval(y, bond_int) - sp1/2

def sk_poly(r,a6,a8,a10,a12,a14):
    exponents = -1*np.array([6,8,10,12,14])
    coeff = np.array([a6,a8,a10,a12,a14])
    bond_int = np.sum(np.power(r[:,np.newaxis],exponents[np.newaxis,:]) * coeff,axis=1)
    return bond_int

def sk_exp(r,a,b,r1,r2):
    return a*np.exp(-r/r1) - b*np.exp(-r/r2)

def sk_relu_real(r,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10):
    coeff_slope = np.array([a1,a2,a3,a4,a5])
    coeff_shift = np.array([a6,a7,a8,a9,a10])
    #relu = np.maximum(0,r[:,np.newaxis]+coeff_shift[np.newaxis,:])
    #bond_int = np.sum(relu * coeff_slope,axis=1)
    relu = np.maximum(0, (r[:,np.newaxis]*coeff_slope[np.newaxis,:])+coeff_shift)
    bond_int = np.sum(relu ,axis=1)

    return bond_int

def sk_relu(r,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10):
    exponents = np.array([a1,a2,a3,a4,a5])
    coeff = np.array([a6,a7,a8,a9,a10])
    bond_int = np.sum(np.power(r[:,np.newaxis],exponents[np.newaxis,:]) * coeff,axis=1)
    return bond_int



if __name__=="__main__":
    # fit interlayer parameters
    eV_per_hartree = 27.2114
    ang_per_bohr = 1/1.8897259886
    rcut = 10
    aa = 1.0 * ang_per_bohr  
    b = rcut * ang_per_bohr

    interlayer_hoppings,interlayer_disp = hopping_training_data(hopping_type="interlayer")
    intralayer_hoppings,intralayer_disp = hopping_training_data(hopping_type="intralayer")
    model_type = "porezag cheby"
    if model_type =="popov cheby":
        

        Cpp_sigma=np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863])*eV_per_hartree
    
        popov_pp_sigma = np.loadtxt("../data/popov_hopping_pp_sigma.txt")
        popov_pp_sigma_hopping = popov_pp_sigma[:,1] * eV_per_hartree 
        popov_pp_sigma_dist = popov_pp_sigma[:,0] * ang_per_bohr
        y = (2.0 * popov_pp_sigma_dist - (b + aa)) / (b - aa)
        parameters_pp_sigma = fit_hoppings(popov_pp_sigma_hopping,y,sk_chebval,Cpp_sigma)
        np.savez("parameters/popov_hoppings_pp_sigma",parameters = parameters_pp_sigma)

        Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree
        popov_pp_pi = np.loadtxt("../data/popov_hopping_pp_pi.txt")
        popov_pp_pi_hopping = popov_pp_pi[:,1] * eV_per_hartree 
        popov_pp_pi_dist = popov_pp_pi[:,0] * ang_per_bohr
        y = (2.0 * popov_pp_pi_dist - (b + aa)) / (b - aa)
        parameters_pp_pi = fit_hoppings(popov_pp_pi_hopping,y,sk_chebval,Cpp_pi)
        np.savez("parameters/popov_hoppings_pp_pi",parameters = parameters_pp_pi)

        npts = 100
        r = np.linspace(aa,b, npts)
        y = (2.0 * r - (b + aa)) / (b - aa)
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_pi)-parameters_pp_pi[0]/2,label = "pp pi fit")
        plt.scatter(popov_pp_pi_dist,popov_pp_pi_hopping,label="yabinitio")
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_sigma)-parameters_pp_sigma[0]/2,label = "pp sigma fit")
        plt.scatter(popov_pp_sigma_dist,popov_pp_sigma_hopping,label="yabinitio")
        
        plt.xlabel("r")
        plt.ylabel("energy")
        plt.legend()
        plt.title("interlayer hopping fit")
        plt.savefig("figures/interlayer_hopping_fit.png")
        plt.clf()

        #same for overlap
        Cpp_sigma=np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863])*eV_per_hartree

        popov_pp_sigma = np.loadtxt("../data/popov_overlap_pp_sigma.txt")
        popov_pp_sigma_overlap = popov_pp_sigma[:,1] * eV_per_hartree 
        popov_pp_sigma_dist = popov_pp_sigma[:,0] * ang_per_bohr
        y = (2.0 * popov_pp_sigma_dist - (b + aa)) / (b - aa)
        parameters_pp_sigma = fit_hoppings(popov_pp_sigma_overlap,y,sk_chebval,Cpp_sigma)
        np.savez("parameters/popov_overlap_pp_sigma",parameters = parameters_pp_sigma)

        Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree
        popov_pp_pi = np.loadtxt("../data/popov_overlap_pp_pi.txt")
        popov_pp_pi_overlap = popov_pp_pi[:,1] * eV_per_hartree 
        popov_pp_pi_dist = popov_pp_pi[:,0] * ang_per_bohr
        y = (2.0 * popov_pp_pi_dist - (b + aa)) / (b - aa)
        parameters_pp_pi = fit_hoppings(popov_pp_pi_overlap,y,sk_chebval,Cpp_pi)
        np.savez("parameters/popov_overlap_pp_pi",parameters = parameters_pp_pi)

        npts = 100
        r = np.linspace(aa,b, npts)
        y = (2.0 * r - (b + aa)) / (b - aa)
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_pi)-parameters_pp_pi[0]/2,label = "pp pi fit")
        plt.scatter(popov_pp_pi_dist,popov_pp_pi_overlap,label="yabinitio")
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_sigma)-parameters_pp_sigma[0]/2,label = "pp sigma fit")
        plt.scatter(popov_pp_sigma_dist,popov_pp_sigma_overlap,label="yabinitio")
        
        plt.xlabel("r")
        plt.ylabel("energy")
        plt.legend()
        plt.title("interlayer overlap fit")
        plt.savefig("figures/interlayer_overlap_fit.png")
        plt.clf()


    elif model_type=="porezag cheby":
        Cpp_sigma=np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863])*eV_per_hartree

        porezag_pp_sigma = np.loadtxt("../data/porezag_hopping_pp_sigma.txt")
        porezag_pp_sigma_hopping = porezag_pp_sigma[:,1] * eV_per_hartree 
        porezag_pp_sigma_dist = porezag_pp_sigma[:,0] * ang_per_bohr
        y = (2.0 * porezag_pp_sigma_dist - (b + aa)) / (b - aa)
        parameters_pp_sigma = fit_hoppings(porezag_pp_sigma_hopping,y,sk_chebval,Cpp_sigma)
        np.savez("parameters/porezag_hoppings_pp_sigma",parameters = parameters_pp_sigma)

        Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree
        porezag_pp_pi = np.loadtxt("../data/porezag_hopping_pp_pi.txt")
        porezag_pp_pi_hopping = porezag_pp_pi[:,1] * eV_per_hartree 
        porezag_pp_pi_dist = porezag_pp_pi[:,0] * ang_per_bohr
        y = (2.0 * porezag_pp_pi_dist - (b + aa)) / (b - aa)
        parameters_pp_pi = fit_hoppings(porezag_pp_pi_hopping,y,sk_chebval,Cpp_pi)
        np.savez("parameters/porezag_hoppings_pp_pi",parameters = parameters_pp_pi)

        npts = 100
        r = np.linspace(aa,b, npts)
        y = (2.0 * r - (b + aa)) / (b - aa)
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_pi)-parameters_pp_pi[0]/2,label = "pp pi fit")
        plt.scatter(porezag_pp_pi_dist,porezag_pp_pi_hopping,label="yabinitio")
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_sigma)-parameters_pp_sigma[0]/2,label = "pp sigma fit")
        plt.scatter(porezag_pp_sigma_dist,porezag_pp_sigma_hopping,label="yabinitio")
        
        plt.xlabel("r")
        plt.ylabel("energy")
        plt.legend()
        plt.title("intralayer hopping fit")
        plt.savefig("figures/intralayer_hopping_fit.png")
        plt.clf()

        #same for overlap
        Cpp_sigma=np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863])*eV_per_hartree

        porezag_pp_sigma = np.loadtxt("../data/porezag_overlap_pp_sigma.txt")
        porezag_pp_sigma_overlap = porezag_pp_sigma[:,1] * eV_per_hartree 
        porezag_pp_sigma_dist = porezag_pp_sigma[:,0] * ang_per_bohr
        y = (2.0 * porezag_pp_sigma_dist - (b + aa)) / (b - aa)
        parameters_pp_sigma = fit_hoppings(porezag_pp_sigma_overlap,y,sk_chebval,Cpp_sigma)
        np.savez("parameters/porezag_overlap_pp_sigma",parameters = parameters_pp_sigma)

        Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree
        porezag_pp_pi = np.loadtxt("../data/porezag_overlap_pp_pi.txt")
        porezag_pp_pi_overlap = porezag_pp_pi[:,1] * eV_per_hartree 
        porezag_pp_pi_dist = porezag_pp_pi[:,0] * ang_per_bohr
        y = (2.0 * porezag_pp_pi_dist - (b + aa)) / (b - aa)
        parameters_pp_pi = fit_hoppings(porezag_pp_pi_overlap,y,sk_chebval,Cpp_pi)
        np.savez("parameters/porezag_overlap_pp_pi",parameters = parameters_pp_pi)

        npts = 100
        r = np.linspace(aa,b, npts)
        y = (2.0 * r - (b + aa)) / (b - aa)
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_pi)-parameters_pp_pi[0]/2,label = "pp pi fit")
        plt.scatter(porezag_pp_pi_dist,porezag_pp_pi_overlap,label="yabinitio")
        plt.plot(r,np.polynomial.chebyshev.chebval(y,parameters_pp_sigma)-parameters_pp_sigma[0]/2,label = "pp sigma fit")
        plt.scatter(porezag_pp_sigma_dist,porezag_pp_sigma_overlap,label="yabinitio")
        
        plt.xlabel("r")
        plt.ylabel("energy")
        plt.legend()
        plt.title("intralayer overlap fit")
        plt.savefig("figures/intralayer_overlap_fit.png")
        plt.clf()


    elif model_type =="MK":
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
        
