import numpy as np
import flatgraphene as fg
import pandas as pd
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.BLG_potentials import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *
import matplotlib.pyplot as plt

def get_energy_density(atoms,calc):
    _, _, rho_e, disp = calc.get_tb_energy(atoms,return_wf = True)
    return rho_e, disp


def get_twist_geom(t,sep,a=2.46):
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=20)
    return atoms

if __name__=="__main__":
    eV_per_hartree = 27.2114
    kmesh = (1,1,1)
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
    model_dict = {"interlayer":{"hopping form":interlayer_hopping_fxn,"overlap form":interlayer_overlap_fxn,
                                    "hopping parameters":interlayer_hopping_params,"overlap parameters":interlayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"interlayer","cutoff":5.29177},"cutoff":5.29177,
                                    "potential":Kolmogorov_Crespi_insp,"potential parameters":interlayer_params},
        
                "intralayer":{"hopping form":intralayer_hopping_fxn,"overlap form":intralayer_overlap_fxn,
                                    "hopping parameters":intralayer_hopping_params,"overlap parameters":intralayer_overlap_params,
                                    "descriptors":get_disp,"descriptor kwargs":{"type":"intralayer","cutoff":3.704239},"cutoff":3.704239,
                                    "potential":None,"potential parameters":None}}
    calc = TETB_model(model_dict,kmesh=kmesh)
    theta = [5.09,3.89] #,2.88]
    for t in theta:
        print(t)
        atoms = get_twist_geom(t,sep=3.35)
        rho_e,disp = get_energy_density(atoms,calc)

        plt.scatter(np.linalg.norm(disp,axis=1),rho_e,label="theta = "+str(t),s=10)
    plt.xlabel(r"$r_{ij}$")
    plt.ylim(-1,1)
    plt.ylabel(r"$T_{ij}$")
    plt.legend()
    plt.title("local electronic kinetic energy as a function of twist angle")
    plt.savefig("electron_kinetic_energy_twist.png")
    plt.clf()