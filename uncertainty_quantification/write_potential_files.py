import numpy as np
import os
from BLG_model_builder.BLG_model_library import *
from model_fit import fit_model

file_names = ["Classical_energy_interlayer_best_fit_params_estimate.npz",
            "Classical_energy_intralayer_best_fit_params_estimate.npz",
            "MK_best_fit_params_estimate.npz",   
            "TETB_energy_interlayer_MK_best_fit_params_estimate.npz",
            "TETB_energy_interlayer_popov_best_fit_params_estimate.npz",
            "TETB_energy_intralayer_MK_best_fit_params_estimate.npz",
            "TETB_energy_intralayer_popov_best_fit_params_estimate.npz",
            "interlayer_LETB_best_fit_params_estimate.npz",
            "intralayer_LETB_NN_val_1_best_fit_params_estimate.npz",
            "intralayer_LETB_NN_val_2_best_fit_params_estimate.npz",
            "intralayer_LETB_NN_val_3_best_fit_params_estimate.npz"]

#'z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'
Classical_energy_interlayer = np.array([3.35797189, 18.61604471, 14.68599064,-0.49209699, 6.07493500,0.72973189, 3.39420934,12.30297446])
#Classical_energy_interlayer_bounds = np.array([[2,4],[-100,100],[-100,100],[-100,100],[-100,100],[1e-2,10],[1e-2,10],[-100,100]])
Classical_energy_interlayer_bounds = np.array([[1,1e2],[0,1e5],[0,1e5],[-1e5,0],[-1e5,1e5],[1e-2,1e5],[1e-2,1e5],[1e-4,1e5]])
np.savez("best_fit_params/Kolmogorov_Crespi_best_fit_params_estimate",params= Classical_energy_interlayer,bounds = Classical_energy_interlayer_bounds)
if os.path.exists("best_fit_params/Kolmogorov_Crespi_best_fit_params.npz"):
    data = np.load("best_fit_params/Kolmogorov_Crespi_best_fit_params.npz")
    params = data["params"]
    bounds = Classical_energy_interlayer_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/Kolmogorov_Crespi_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

# C0 C2 C4 C delta lambda A z0 — B and eta are fixed at 0 in DRIPASECalculator (like cutoffs)
drip_interlayer = np.array([15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238, 3.34])
drip_interlayer_bounds = np.array(
    [[0, 1e5], [0, 1e5], [0, 1e5], [-1e5, 1e5], [1e-2, 1e5], [1e-2, 1e5], [1e-4, 1e5], [1, 1e2]]
)
np.savez("best_fit_params/DRIP_best_fit_params_estimate",params= drip_interlayer,bounds = drip_interlayer_bounds)


# m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A
# LAMMPS pair_tersoff requires: c,d,n,beta,lambda2,B,R,D,lambda1,A,gamma >= 0; m in {1,3}; D <= R
Classical_energy_intralayer_params = np.array([3, 1.0, 0.0, 38049.0, 4.3484, -0.57058, 0.72751,
                                        1.5724e-7, 2.2119, 346.74, 2.85, 0.15, 3.4879, 1393.6])
# Bounds: m fixed; gamma,c,d,n,beta,lam2,B,R,D,lam1,A > 0; costheta0 in [-1,1]; enforce D < R via R_min > D_max
Classical_energy_intralayer_bounds = np.array([
    [3, 3],           # m: must be 1 or 3
    [1e-6, 1e2],      # gamma >= 0
    [0, 1e2],         # lambda3 >= 0
    [1e-5, 1e6],      # c >= 0
    [1e-5, 1e5],      # d >= 0
    [-1, 1],          # costheta0
    [1e-5, 1e2],      # n >= 0
    [1e-10, 1e2],     # beta >= 0
    [1e-5, 1e2],      # lambda2 >= 0
    [1e-5, 1e5],      # B >= 0
    [2.0, 3.5],       # R >= 0 (R_min chosen so R > D)
    [0.01, 0.4],      # D >= 0, D < R (D_max < R_min)
    [1e-5, 1e2],      # lambda1 >= 0
    [1e-5, 1e5],      # A >= 0
])
np.savez("best_fit_params/Tersoff_best_fit_params_estimate",params= Classical_energy_intralayer_params,bounds = Classical_energy_intralayer_bounds)
if os.path.exists("best_fit_params/Tersoff_best_fit_params.npz"):
    data = np.load("best_fit_params/Tersoff_best_fit_params.npz")
    params = data["params"]
    bounds = Classical_energy_intralayer_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/Tersoff_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

MK = np.array([-2.92500706,  4.95594733,  0.34230107])
MK_bounds = np.array([[-1e2, -1.e-5], [ 1e-5,  1e2], [ 1e-5,  1e2]])
np.savez("best_fit_params/MK_best_fit_params_estimate",params=MK, bounds = MK_bounds)
if os.path.exists("best_fit_params/MK_best_fit_params.npz"):
    data = np.load("best_fit_params/MK_best_fit_params.npz")
    params = data["params"]
    bounds = MK_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/MK_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

#delta,C,C0,C2,C4,z0,A6 ,A8,A10
#TETB_energy_interlayer_MK = np.array([3.43845303, 34.04495658, -17.16974743, 17.22962837, -23.0448948, 3.07925665, -1.54847667, 10.78402167, -7.14595312])
#TETB_energy_interlayer_MK_bounds = np.array([[1e-2,10], [-10000,10000], [-10000,10000], 
#                                            [-10000,10000], [-10000,10000], [2,4], [-10000,10000], [-10000,10000], [-10000,10000] ])
TETB_energy_interlayer_MK = np.array([3.78930829,  50.76326033,  15.84972824,  37.62929632, -17.67148694, 0.78781926,   4.86061691,   5.14393366]) #,0,0,0])
TETB_energy_interlayer_MK_bounds = np.array([[2,4],[-10000,10000],[-10000,10000],[-100,100],[-10000,10000],[1e-2,10],[1e-2,10],[1e-4,1000]]) #,[-10000,10000],[-10000,10000],[-10000,10000]])
np.savez("best_fit_params/TETB_energy_interlayer_MK_best_fit_params_estimate",params = TETB_energy_interlayer_MK, bounds = TETB_energy_interlayer_MK_bounds)

TETB_energy_interlayer_popov = np.array([3.43845,34.0449,-17.1697,17.2296,-23.0449,3.07926,-1.54847,10.784,-7.1459]) #kc inspired form
TETB_energy_interlayer_popov_bounds = np.array([[1e-4,1e3],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[1e-4,1e3],[-1e5,1e5],[-1e5,1e5],[-1e5,1e5]])
np.savez("best_fit_params/TETB_energy_interlayer_popov_best_fit_params_estimate",params = TETB_energy_interlayer_popov, bounds = TETB_energy_interlayer_popov_bounds)
if os.path.exists("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz"):
    data = np.load("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz")
    params = data["params"]
    bounds = TETB_energy_interlayer_popov_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/TETB_energy_interlayer_popov_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)
##c, d, costheta0, n, beta, lambda2, B, lambda1, A
#TETB_energy_intralayer_MK = np.array([ 4.0381772,  16.79935969, -0.93,  1.40171327, 26.16975888,  0.04642429, 2.94080086,  5.1350953,   2.19817537])
TETB_energy_intralayer_MK = np.array([3.8049e4, 4.3484, -0.93000, 0.72751, 1.5724e-7,  2.2119,  430.00,   3.4879,  1393.6])
TETB_energy_intralayer_MK_bounds = np.array([[1e-5,  1e6],[ 1e-5,  1e2],[-1,  1],
                                                [ 1e-5,  5],[ 1e-10,  100],[ 1e-5,  100],[ 1e-5,  1e5],[ 1e-5,  100],[ 1e-5,  1e5]])
np.savez("best_fit_params/TETB_energy_intralayer_MK_best_fit_params_estimate",params = TETB_energy_intralayer_MK, bounds = TETB_energy_intralayer_MK_bounds)

TETB_energy_intralayer_popov = np.array([3.8049e4, 4.3484, -0.93000, 0.720060119, 2.65270138e-7,  2.2119,  430.00,   3.4879,  1393.6])
TETB_energy_intralayer_popov_bounds = np.array([[1e-5,  1e6],[ 1e-5,  1e2],[-1,  1],
                                                [ 1e-5,  1e3],[ 1e-10,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5]])
np.savez("best_fit_params/TETB_energy_intralayer_popov_best_fit_params_estimate",params = TETB_energy_intralayer_popov, bounds = TETB_energy_intralayer_popov_bounds)
if os.path.exists("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz"):
    data = np.load("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz")
    params = data["params"]
    bounds = TETB_energy_intralayer_popov_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/TETB_energy_intralayer_popov_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)
#
interlayer_LETB = np.array([ 2.38399524e-1,  2.12105173,  1.87047453, -3.97325344e-2,
                            3.72127933,  5.14865154e-1, -5.91880658e-3,  6.06170635,1.52121159,  1.73030803])
interlayer_LETB_bounds = np.array([[-1.e3,  1.e5],[ 1.e-5,  1e5],[ 1.e-5,  1e5],[-1e1,  1e1],[ 1.e-5,  1e5],
                                    [ 1.e-5,  1e5],[-1e2,  -1e-6],[ 1.e-5,  1e5],[ 1.e-5,  1e5],[ 1e-5,  1e5]])
np.savez("best_fit_params/interlayer_LETB_best_fit_params_estimate",params = interlayer_LETB, bounds = interlayer_LETB_bounds)
if os.path.exists("best_fit_params/interlayer_LETB_best_fit_params.npz"):
    data = np.load("best_fit_params/interlayer_LETB_best_fit_params.npz")
    params = data["params"]
    bounds = interlayer_LETB_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/interlayer_LETB_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

intralayer_LETB_NN_val_1 = np.array([ -10.5, 5.012701434614247])
intralayer_LETB_NN_val_1_bounds = np.array([[-20,-5],[1,10]])
np.savez("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params_estimate",params = intralayer_LETB_NN_val_1, bounds = intralayer_LETB_NN_val_1_bounds)
if os.path.exists("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params.npz"):
    data = np.load("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params.npz")
    params = data["params"]
    bounds = intralayer_LETB_NN_val_1_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)
intralayer_LETB_NN_val_2 = np.array([ 1.56592635, -0.55782822, -0.00575551, -0.17259223]) 
intralayer_LETB_NN_val_2_bounds = np.array([[-1e2,1e2],[-1e2,1e2],[-1e2,1e2],[-1e2,1e2]])
np.savez("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params_estimate",params = intralayer_LETB_NN_val_2, bounds = intralayer_LETB_NN_val_2_bounds)
if os.path.exists("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params.npz"):
    data = np.load("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params.npz")
    params = data["params"]
    bounds = intralayer_LETB_NN_val_2_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

intralayer_LETB_NN_val_3 = np.array([-1.18627235, -0.05553831,  0.1048468,   0.26656029])
intralayer_LETB_NN_val_3_bounds = np.array([[-1e2,1e2],[-1e2,1e2],[-1e2,1e2],[-1e2,1e2]])
np.savez("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params_estimate",params = intralayer_LETB_NN_val_3, bounds = intralayer_LETB_NN_val_3_bounds)
if os.path.exists("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params.npz"):
    data = np.load("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params.npz")
    params = data["params"]
    bounds = intralayer_LETB_NN_val_3_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

# ACSF_hoppings
M = 10
W = 3
r_cut = 6.0
use_envelope = True
ACSF_hoppings_params = np.random.normal(0, 1, size=M+M*W)
ACSF_hoppings_bounds = np.array([[-1e2, 1e2]]*(M+M*W))
np.savez("best_fit_params/ACSF_hoppings_best_fit_params_estimate",params = ACSF_hoppings_params, bounds = ACSF_hoppings_bounds)
if os.path.exists("best_fit_params/ACSF_hoppings_best_fit_params.npz"):
    data = np.load("best_fit_params/ACSF_hoppings_best_fit_params.npz")
    params = data["params"]
    bounds = ACSF_hoppings_bounds
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/ACSF_hoppings_best_fit_params",params=params,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)



eV_per_hartree = 27.2114
popov_hopping_pp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863]) *eV_per_hartree
popov_hopping_pp_sigma_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=5.29177)}
xdata,ydata,ydata_noise = get_training_data("popov_hopping_pp_sigma")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],popov_hopping_pp_sigma)
np.savez("best_fit_params/popov_hopping_pp_sigma_best_fit_params",params = popov_hopping_pp_sigma, bounds = popov_hopping_pp_sigma_bounds
                                                                ,ypred_bestfit=ypred_bestfit)   

popov_hopping_pp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])*eV_per_hartree
popov_hopping_pp_pi_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=5.29177)}
xdata,ydata,ydata_noise = get_training_data("popov_hopping_pp_pi")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],popov_hopping_pp_pi)
np.savez("best_fit_params/popov_hopping_pp_pi_best_fit_params",params = popov_hopping_pp_pi, bounds = popov_hopping_pp_pi_bounds,
                                                                ypred_bestfit=ypred_bestfit)

popov_overlap_pp_pi = np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,0.0156376, 0.0025976, -0.0039498, 0.0020581,-0.0007114, 0.0001427])
popov_overlap_pp_pi_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=5.29177)}
xdata,ydata,ydata_noise = get_training_data("popov_overlap_pp_pi")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],popov_overlap_pp_pi)
np.savez("best_fit_params/popov_overlap_pp_pi_best_fit_params",params = popov_overlap_pp_pi, bounds = popov_overlap_pp_pi_bounds,
                                                                ypred_bestfit=ypred_bestfit)

popov_overlap_pp_sigma = np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997, 0.0921727, -0.0268106, 0.0002240, 0.0040319, -0.0022450, 0.0005596])
popov_overlap_pp_sigma_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=5.29177)}
xdata,ydata,ydata_noise = get_training_data("popov_overlap_pp_sigma")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],popov_overlap_pp_sigma)
np.savez("best_fit_params/popov_overlap_pp_sigma_best_fit_params",params = popov_overlap_pp_sigma, bounds = popov_overlap_pp_sigma_bounds,
                                                                ypred_bestfit=ypred_bestfit)

#these parameters don't seem to fit the data well enough, so just refit it here
porezag_hopping_pp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906]) * eV_per_hartree
porezag_hopping_pp_sigma_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=7*ang_per_bohr)}

xdata,ydata,ydata_noise = get_training_data("porezag_hopping_pp_sigma")
best_fit_params,ypred_bestfit = fit_model(calc["hoppings"],xdata["hoppings"],ydata["hoppings"],
                                            porezag_hopping_pp_sigma,shift_data=False,bounds=porezag_hopping_pp_sigma_bounds)
np.savez("best_fit_params/porezag_hopping_pp_sigma_best_fit_params",params = porezag_hopping_pp_sigma, bounds = porezag_hopping_pp_sigma_bounds,
                                                                ypred_bestfit=ypred_bestfit)

porezag_hopping_pp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, -0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855]) * eV_per_hartree
porezag_hopping_pp_pi_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=7*ang_per_bohr)}
xdata,ydata,ydata_noise = get_training_data("porezag_hopping_pp_pi")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],porezag_hopping_pp_pi)
np.savez("best_fit_params/porezag_hopping_pp_pi_best_fit_params",params = porezag_hopping_pp_pi, bounds = porezag_hopping_pp_pi_bounds,
                                                                ypred_bestfit=ypred_bestfit)

porezag_overlap_pp_pi =np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227])
porezag_overlap_pp_pi_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=7*ang_per_bohr)}
xdata,ydata,ydata_noise = get_training_data("porezag_overlap_pp_pi")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],porezag_overlap_pp_pi)
np.savez("best_fit_params/porezag_overlap_pp_pi_best_fit_params",params = porezag_overlap_pp_pi, bounds = porezag_overlap_pp_pi_bounds,
                                                                ypred_bestfit=ypred_bestfit)

porezag_overlap_pp_sigma = np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751])
porezag_overlap_pp_sigma_bounds = np.array([[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6],[-1e6,1e6]])
calc = {"hoppings": SK_bond_ints(aa=0.529, b=7*ang_per_bohr)}
xdata,ydata,ydata_noise = get_training_data("porezag_overlap_pp_sigma")
ypred_bestfit = calc["hoppings"](xdata["hoppings"],porezag_overlap_pp_sigma)
np.savez("best_fit_params/porezag_overlap_pp_sigma_best_fit_params",params = porezag_overlap_pp_sigma, bounds = porezag_overlap_pp_sigma_bounds,
                                                                ypred_bestfit=ypred_bestfit)

