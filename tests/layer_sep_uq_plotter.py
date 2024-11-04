import numpy as np
import matplotlib.pyplot as plt
import ase.io
from scipy.interpolate import LinearNDInterpolator

def get_max_layer_sep(atoms):
    pos = atoms.positions
    mean_z = np.mean(pos[:,2])
    top_ind = np.where(pos[:,2]>mean_z)
    return np.max(2*np.abs(mean_z-pos[top_ind,2]))

def get_struct_cross_sect(atoms):
    
    cell = atoms.get_cell()
    cell = np.abs(cell)
    atoms.set_cell(cell)
    atoms.wrap()
    pos = atoms.positions
    xy_lim = cell[0,:] + cell[1,:]
    
    npoints = 60
    mesh = np.linspace(0,1,npoints)
    path = xy_lim[:2,np.newaxis] @ mesh[:,np.newaxis].T

    mean_z = np.mean(pos[:,2])
    top_layer_ind = np.where(pos[:,2]>mean_z)
    top_pos = np.squeeze(pos[top_layer_ind,:])
    bot_layer_ind = np.where(pos[:,2]<mean_z)
    bot_pos = np.squeeze(pos[bot_layer_ind,:])

    interp = LinearNDInterpolator(list(zip(top_pos[:,0],top_pos[:,1])), top_pos[:,2])
    zpath_top = interp(path.T)

    interp = LinearNDInterpolator(list(zip(bot_pos[:,0],bot_pos[:,1])), bot_pos[:,2])
    zpath_bot = interp(path.T)

    return mesh*np.linalg.norm(xy_lim[:2]),zpath_top-mean_z
    #plt.plot(mesh*np.linalg.norm(xy_lim[:2]),zpath_top-mean_z,color = color,label=label,linestyle=line_style)

if __name__=="__main__":
    twist_angles = np.array([0.88,0.99,1.08,1.12,1.16,1.2,1.47,1.89,2.88])
    #twist_angles = np.array([0.88,1.08,1.47])
    layer_sep_angles_mean = np.zeros_like(twist_angles)
    layer_sep_angles_std = np.zeros_like(twist_angles)
     
    for i,t in enumerate(twist_angles):
        atoms_list = ase.io.read("../uncertainty_quantification/relaxed_atoms_theta_"+str(t)+"_cv_ensemble.xyz",format="extxyz",index=":")
        layer_sep_ensemble = np.zeros(len(atoms_list))
        layer_z_ensemble = np.zeros((60,len(atoms_list)))
        for j,a in enumerate(atoms_list):
            layer_sep_ensemble[j] = get_max_layer_sep(a)
            path, layer_z = get_struct_cross_sect(a)
            layer_z_ensemble[:,j] = layer_z

        layer_sep_angles_mean[i] = np.mean(layer_sep_ensemble)
        layer_sep_angles_std[i] = np.std(layer_sep_ensemble)
        layer_z_mean = np.mean(layer_z_ensemble, axis = 1)
        layer_z_std = np.std(layer_z_ensemble, axis = 1)

        plt.plot(path,layer_z_mean,label="ensemble mean")
        plt.fill_between(path,layer_z_mean-layer_z_std,layer_z_mean+layer_z_std,label="95% confidence interval",alpha=0.3)
        label = ["AA","AB","SP","BA","AA"]
        #xy_lim = atoms_list[0].get_cell()[0,:] + atoms_list[0].get_cell()[1,:]
        #plt.xticks(np.linspace(0,1,5)*np.linalg.norm(xy_lim),label)
        plt.ylabel("z position, top layer")
        plt.legend()
        plt.title(r"$\theta=$"+str(t))
        plt.savefig("figures/theta_"+str(t)+"cross_section_uq.png")
        plt.clf()
        print("std "+str(t)+" = ",np.std(layer_sep_ensemble))

    plt.scatter(twist_angles,layer_sep_angles_mean,label="ensemble mean")
    plt.fill_between(twist_angles,layer_sep_angles_mean-layer_sep_angles_std,layer_sep_angles_mean+layer_sep_angles_std,label="ensemble std",alpha=0.3)
    plt.xlabel("twist angle")
    plt.ylabel("max interlayer separation")
    plt.legend()
    plt.savefig("figures/twist_angle_layer_sep_uq.png")

