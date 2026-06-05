import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import flatgraphene as fg

import ase.io
from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs
from blg_model_builder.potentials import PODASECalculator
from scipy.interpolate import CubicSpline

csfont = {'fontname':'sans-serif',"size":15}


def gsfe_path(s, E_AB, E_SP, E_AA):
    """
    Interpolates GSFE using cubic splines.
    Milestones:
    s=0   : AB
    s=1/6 : SP
    s=1/3 : AB
    s=2/3 : AA
    s=1   : AB
    """
    # 1. Define the milestones (knots)
    s_knots = np.array([0, 1/6, 1/3, 2/3, 1.0])
    e_knots = np.array([E_AB, E_SP, E_AB, E_AA, E_AB])
    
    # 2. Create the Periodic Cubic Spline
    # bc_type='periodic' ensures the derivative at s=0 matches s=1
    cs = CubicSpline(s_knots, e_knots, bc_type='periodic')
    
    return cs(s)

calc, xdata_train, xdata_test, xdata, ydata_train, ydata_test, ydata, ypred_bestfit, params, bounds = \
        get_MCMC_inputs("POD_energy")

rcut = 5.0
hyperparams =  {"species": ["C"],
                "bessel_polynomial_degree": 4, "inverse_polynomial_degree": 8, 
                "twobody_number_radial_basis_functions": 10, 
                "threebody_number_radial_basis_functions": 8, "threebody_angular_degree": 4, 
                "fourbody_number_radial_basis_functions": 6, "fourbody_angular_degree": 3, 
                "fivebody_number_radial_basis_functions": 4, "fivebody_angular_degree": 3, 
                "sixbody_number_radial_basis_functions": 3, "sixbody_angular_degree": 2, 
                "sevenbody_number_radial_basis_functions": 2, "sevenbody_angular_degree": 2}
pod_calc = PODASECalculator(hyperparams, params["energy"], elements=["C"], cutoff=rcut)

layer_sep = 3.43
stacking_sym_pts = ["AB","SP","AA"]
disregistry_sym_pts = [0.0, 0.166667, 0.666667]

dft_atoms = ase.io.read("../data/strained_bilayer_graphene_rVV10.xyz",index=":")
min_atoms = fg.shift.make_graphene(['A','B'],'hex',1,1,2.46,2,3.43,sym=["C","C"],
                        mass=[12.01,12.01],mol_id=None,h_vac=20)
min_energy = np.min(ydata["energy"])
min_atoms.calc = pod_calc
min_pod_energy = min_atoms.get_potential_energy()


min_atom_cell = min_atoms.get_cell()
min_atom_pos = min_atoms.get_positions()
disregistry_list = []
stacking_list = []
strain_x = []
strain_y = []
layer_sep_list = []

for i, atoms in enumerate(dft_atoms):
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    disreg_norm = np.linalg.norm(cell[0]) * np.sqrt(3)
    cell_diff_2d = cell[:2,:2] / min_atom_cell[:2,:2]
    dx = cell_diff_2d[0,0] - 1
    dy = cell_diff_2d[1,1] - 1

    strain_x.append(dx)
    strain_y.append(dy)

    if np.round(np.linalg.norm((pos[2,:2] -pos[0,:2]))/disreg_norm,decimals=2) ==0:
        stacking_list.append("AA")
        disregistry_list.append(0.66667)
    elif np.round(np.linalg.norm((pos[2,:2] -pos[0,:2]))/disreg_norm,decimals=2) == 0.33:
        stacking_list.append("AB")
        disregistry_list.append(0.0)
    else:
        stacking_list.append("SP")
        disregistry_list.append(0.1666667)

    layer_sep_list.append(np.abs(pos[2,2]-pos[0,2]))

strain_y = np.array(strain_y)
strain_x = np.array(strain_x)
disregistry_list = np.array(disregistry_list)
layer_sep_list = np.array(layer_sep_list)
use_strain_y = 0.0

unique_strain_x = np.unique(np.round(strain_x,3))
unique_disregistry = np.unique(np.round(disregistry_list,2))
unique_layer_sep = np.unique(np.round(layer_sep_list,3))
unique_strain_y = np.unique(np.round(strain_y,3))

s_array = np.linspace(0,1,100)

my_cmap = cm.viridis
my_norm = colors.Normalize(vmin=0.0, vmax=0.04)
fig, ax = plt.subplots(figsize=(8, 5))
for sx in unique_strain_x:
    if sx < 0.0:
        continue
    use_ind = np.where((np.round(strain_x,3) == np.round(sx,3)) \
        & (np.round(strain_y,3) == np.round(use_strain_y,3)) \
        & (np.round(layer_sep_list,3) == np.round(layer_sep,3)))[0]

    use_disregistry = disregistry_list[use_ind]
    use_atoms_list = [dft_atoms[i] for i in use_ind]
    
    pod_energy_list = []
    dft_energy_list = []
    
    for atoms in use_atoms_list:

        dft_energy_list.append((atoms.get_potential_energy() - min_energy)/len(atoms))
        atoms_copy = atoms.copy()
        atoms_copy.calc = pod_calc
        pod_energy = (atoms_copy.get_potential_energy() - min_pod_energy)/len(atoms_copy)
        pod_energy_list.append(pod_energy)

    if len(dft_energy_list) < 3:
        continue

    
    dft_energy_list = np.sort(dft_energy_list[:3])
    pod_energy_list = np.sort(pod_energy_list[:3])

    gamma_pod = gsfe_path(s_array, pod_energy_list[0], pod_energy_list[1], pod_energy_list[2])
    gamma_dft = gsfe_path(s_array, dft_energy_list[0], dft_energy_list[1], dft_energy_list[2])


    plot_dft_energy = np.zeros(4)
    plot_dft_energy[:3] = np.sort(dft_energy_list[:3].copy())
    plot_dft_energy[3] = plot_dft_energy[0]
    plot_pod_energy = np.zeros(4)
    plot_pod_energy[:3] = np.sort(pod_energy_list[:3].copy())
    plot_pod_energy[3] = plot_pod_energy[0]
    ax.plot(s_array, gamma_pod, color = my_cmap(my_norm(sx)),zorder=1)
    ax.scatter(np.sort(np.append(use_disregistry[:3],[1])), plot_pod_energy,marker="o",color = my_cmap(my_norm(sx)),zorder=1)
    #ax.plot(s_array, gamma_dft, color = my_cmap(my_norm(sx)),zorder=1)
    ax.scatter(np.sort(np.append(use_disregistry[:3],[1])), plot_dft_energy,marker="x",color="black",s=60,zorder=2)

ax.set_xlabel("Disregistry ",**csfont)
ax.set_ylabel("GSFE (eV/atom)",**csfont)
ax.axvline(x=0.0, color='black', linestyle='dotted', linewidth=1)
ax.axvline(x=0.1666667, color='black', linestyle='dotted', linewidth=1)
ax.axvline(x=0.33333, color='black', linestyle='dotted', linewidth=1)
ax.axvline(x=0.6666667, color='black', linestyle='dotted', linewidth=1)
ax.axvline(x=1.0, color='black', linestyle='dotted', linewidth=1)
plt.xticks([0, 0.1666667,0.33333, 0.6666667, 1.0], ['AB', 'SP','BA', 'AA', 'AB'])
cbar = fig.colorbar(cm.ScalarMappable(norm=my_norm, cmap=my_cmap), ax=ax)
cbar.set_label(label="strain along x direction",fontsize=csfont["size"],fontfamily=csfont["fontname"])

handle1 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                markersize=8, label='DFT')

handle2 = mlines.Line2D([], [], color='black', marker='_', linestyle='None',
                        markersize=8, label="POD")

# Create the legend using the custom handles
plt.legend(handles=[handle1, handle2])
fig.tight_layout()
plt.savefig("figures/GSFE_intralayer_PES_strain_x.png")
plt.clf()



    