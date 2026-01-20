import numpy as np
import os

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
np.savez("best_fit_params/Classical_energy_interlayer_best_fit_params_estimate",params= Classical_energy_interlayer,bounds = Classical_energy_interlayer_bounds)
if os.path.exists("best_fit_params/Classical_energy_interlayer_best_fit_params.npz"):
    data = np.load("best_fit_params/Classical_energy_interlayer_best_fit_params.npz")
    params = data["params"]
    bounds = Classical_energy_interlayer_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/Classical_energy_interlayer_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

#c, d, costheta0, n, beta, lambda2, B, lambda1, A
Classical_energy_intralayer = np.array([3.8049e4, 4.3484, -0.93000, 0.72751, 1.5724e-7,  2.2119,  430.00,   3.4879,  1393.6])
Classical_energy_intralayer_bounds = np.array([[1e-5,  1e6],[ 1e-5,  1e5],[-1,  1],
                                                [ 1e-5,  1e5],[ 1e-10,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5]])
np.savez("best_fit_params/Classical_energy_intralayer_best_fit_params_estimate",params= Classical_energy_intralayer,bounds = Classical_energy_intralayer_bounds)
if os.path.exists("best_fit_params/Classical_energy_intralayer_best_fit_params.npz"):
    data = np.load("best_fit_params/Classical_energy_intralayer_best_fit_params.npz")
    params = data["params"]
    bounds = Classical_energy_intralayer_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/Classical_energy_intralayer_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

MK = np.array([-2.92500706,  4.95594733,  0.34230107])
MK_bounds = np.array([[-1e2, -1.e-5], [ 1e-5,  1e2], [ 1e-5,  1e2]])
np.savez("best_fit_params/MK_best_fit_params_estimate",params=MK, bounds = MK_bounds)
if os.path.exists("best_fit_params/MK_best_fit_params.npz"):
    data = np.load("best_fit_params/MK_best_fit_params.npz")
    params = data["params"]
    bounds = MK_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/MK_best_fit_params",params=params,covariance=covariance,
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
np.savez("best_fit_params/TETB_energy_interlayer_popov_best_fit_params",params = TETB_energy_interlayer_popov, bounds = TETB_energy_interlayer_popov_bounds,covariance = np.ones_like(TETB_energy_interlayer_popov),ypred_bestfit=np.zeros(44))
if os.path.exists("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz"):
    data = np.load("best_fit_params/TETB_energy_interlayer_popov_best_fit_params.npz")
    params = data["params"]
    bounds = TETB_energy_interlayer_popov_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/TETB_energy_interlayer_popov_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)
##c, d, costheta0, n, beta, lambda2, B, lambda1, A
#TETB_energy_intralayer_MK = np.array([ 4.0381772,  16.79935969, -0.93,  1.40171327, 26.16975888,  0.04642429, 2.94080086,  5.1350953,   2.19817537])
TETB_energy_intralayer_MK = np.array([3.8049e4, 4.3484, -0.93000, 0.72751, 1.5724e-7,  2.2119,  430.00,   3.4879,  1393.6])
TETB_energy_intralayer_MK_bounds = np.array([[1e-5,  1e6],[ 1e-5,  1e2],[-1,  1],
                                                [ 1e-5,  5],[ 1e-10,  100],[ 1e-5,  100],[ 1e-5,  1e5],[ 1e-5,  100],[ 1e-5,  1e5]])
np.savez("best_fit_params/TETB_energy_intralayer_MK_best_fit_params_estimate",params = TETB_energy_intralayer_MK, bounds = TETB_energy_intralayer_MK_bounds)

TETB_energy_intralayer_popov = np.array([3.8049e4, 4.3484, -0.93000, 0.72751, 1.5724e-7,  2.2119,  430.00,   3.4879,  1393.6])
TETB_energy_intralayer_popov_bounds = np.array([[1e-5,  1e6],[ 1e-5,  1e2],[-1,  1],
                                                [ 1e-5,  1e3],[ 1e-10,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5],[ 1e-5,  1e5]])
np.savez("best_fit_params/TETB_energy_intralayer_popov_best_fit_params_estimate",params = TETB_energy_intralayer_popov, bounds = TETB_energy_intralayer_popov_bounds)
if os.path.exists("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz"):
    data = np.load("best_fit_params/TETB_energy_intralayer_popov_best_fit_params.npz")
    params = data["params"]
    bounds = TETB_energy_intralayer_popov_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/TETB_energy_intralayer_popov_best_fit_params",params=params,covariance=covariance,
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
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/interlayer_LETB_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

intralayer_LETB_NN_val_1 = np.array([ -10.5, 5.012701434614247])
intralayer_LETB_NN_val_1_bounds = np.array([[-20,-5],[1,10]])
np.savez("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params_estimate",params = intralayer_LETB_NN_val_1, bounds = intralayer_LETB_NN_val_1_bounds)
if os.path.exists("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params.npz"):
    data = np.load("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params.npz")
    params = data["params"]
    bounds = intralayer_LETB_NN_val_1_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/intralayer_LETB_NN_val_1_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)
intralayer_LETB_NN_val_2 = np.array([ 1.56592635, -0.55782822, -0.00575551, -0.17259223]) 
intralayer_LETB_NN_val_2_bounds = np.array([[-1e2,1e2],[-1e2,1e2],[-1e2,1e2],[-1e2,1e2]])
np.savez("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params_estimate",params = intralayer_LETB_NN_val_2, bounds = intralayer_LETB_NN_val_2_bounds)
if os.path.exists("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params.npz"):
    data = np.load("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params.npz")
    params = data["params"]
    bounds = intralayer_LETB_NN_val_2_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/intralayer_LETB_NN_val_2_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)

intralayer_LETB_NN_val_3 = np.array([-1.18627235, -0.05553831,  0.1048468,   0.26656029])
intralayer_LETB_NN_val_3_bounds = np.array([[-1e2,1e2],[-1e2,1e2],[-1e2,1e2],[-1e2,1e2]])
np.savez("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params_estimate",params = intralayer_LETB_NN_val_3, bounds = intralayer_LETB_NN_val_3_bounds)
if os.path.exists("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params.npz"):
    data = np.load("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params.npz")
    params = data["params"]
    bounds = intralayer_LETB_NN_val_3_bounds
    covariance = data["covariance"]
    ypred_bestfit = data["ypred_bestfit"]
    np.savez("best_fit_params/intralayer_LETB_NN_val_3_best_fit_params",params=params,covariance=covariance,
                                                                    bounds=bounds, ypred_bestfit=ypred_bestfit)
