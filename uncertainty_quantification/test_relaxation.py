import ase.io 
import flatgraphene as fg 
from blg_model_builder.pod_interface import *
import numpy as np
from ase.optimize import LBFGS

rcut = 5.0
hyperparams =  dict(
    nelements=1, rin=1.0, rcut=rcut,
    besseldegree=4, inversedegree=8, nbesselpars=3,
    besselparams=[1e-3, 2.0, 4.0],
    onebody=1, nrbf2=20, nrbf3=20, nrbf4=20, P3=4, P4=3, 
    # nrbf33 = 6,   # number of radial functions for the cross term
    # P33 = 3,   # max angular degree of the cross product
    # nrbf34 = 4,
    # P34 = 2,
    n_vdw = 5, rcut_vdw = 10.0,
)
hyperparam_str = "_".join([f"{k}_{v}" for k, v in hyperparams.items()])
data = np.load("best_fit_params/POD_NN_energy_body_order_"+hyperparam_str+"_best_fit_params.npz")
coeffs = data["params"]
pod = PODInterface(hyperparams)
pod.set_coefficients(coeffs)

p_found, q_found, theta_comp = fg.twist.find_p_q(2.88)
atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=2.46,sym=["C","C"],
                                        mass=[12.01,12.01],sep=3.35,h_vac=20)
calc = pod.get_ase_calculator()
atoms.calc = calc
dyn = LBFGS(atoms,trajectory="test_tblg.traj",
                   logfile="test_twist.log")
dyn.run(fmax=1e-3,steps=150)
print("max layer separation = ",np.max(atoms.positions[:,2])-np.min(atoms.positions[:,2]))