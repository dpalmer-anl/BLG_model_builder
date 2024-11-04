import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flatgraphene as fg
from BLG_model_builder.TB_Utils import *
from BLG_model_builder.Lammps_Utils import *
from BLG_model_builder.descriptors import *
from BLG_model_builder.geom_tools import *
from BLG_model_builder.TETB_model_builder import *

def plot_bands(evals_mean,evals_std,kdat,efermi=None,colors='black',title='',label="",linestyle="solid",erange=5,figname="bands.png"):
    fig, ax = plt.subplots()
    (kvec,k_dist, k_node) = kdat
    
    ticklabel=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
    #ticklabel=('K','G', 'M','K')
    # specify horizontal axis details
    # set range of horizontal axis
    ax.set_xlim(k_node[0],k_node[-1])
    # put tickmarks and labels at node positions
    ax.set_xticks(k_node)
    ax.set_xticklabels(ticklabel,**csfont)
    plt.yticks(fontsize=16)
    # add vertical lines at node positions
    for n in range(len(k_node)):
      ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    # put title
    ax.set_title(title,**csfont)
    ax.set_xlabel("Path in k-space",**csfont)
    ax.set_ylabel("Band energy",**csfont)
    

    if not efermi:
        nbands = np.shape(evals_mean)[0]
        efermi = np.mean([evals_mean[nbands//2,0],evals_mean[(nbands-1)//2,0]])
        fermi_ind = (nbands)//2
    else:
        ediff = np.array(evals_mean).copy()
        ediff -= efermi
        fermi_ind = np.argmin(np.abs(ediff))-1
    
    # plot first and second band
    for n in range(np.shape(evals_mean)[0]):
        if n ==0:
            ax.plot(k_dist,evals_mean[n,:]-efermi,c=colors,label=label+" mean",linestyle=linestyle)
            plt.fill_between(k_dist,(evals_mean[n,:]-evals_std[n,:])-efermi, (evals_mean[n,:]+evals_std[n,:])-efermi,alpha=0.3,facecolor=colors,label="STD")
        else:
            ax.plot(k_dist,evals_mean[n,:]-efermi,c=colors,linestyle=linestyle)
            plt.fill_between(k_dist,(evals_mean[n,:]-evals_std[n,:])-efermi, (evals_mean[n,:]+evals_std[n,:])-efermi,alpha=0.3,facecolor=colors)
    fig.tight_layout()
    ax.set_ylim(-erange,erange)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig.savefig(figname,dpi=1200)
    plt.clf()

def get_band_width(eigenvalues):
    fermi_ind = np.shape(eigenvalues)[0]//2
    cond_band_max = np.max(eigenvalues[fermi_ind,:])
    val_band_min = np.min(eigenvalues[fermi_ind-1,:])
    band_width = np.abs(cond_band_max - val_band_min)
    return band_width

def get_recip_cell(cell):
    periodicR1 = cell[0,:]
    periodicR2 = cell[1,:]
    periodicR3 = cell[2,:]
    V = np.dot(periodicR1,np.cross(periodicR2,periodicR3))
    b1 = 2*np.pi*np.cross(periodicR3,periodicR1)/V
    b2 = 2*np.pi*np.cross(periodicR2,periodicR3)/V
    b3 = 2*np.pi*np.cross(periodicR1,periodicR2)/V
    return np.stack((b1,b2,b3),axis=0)

def get_fermi_velocity_mesh(eigenvalues,dk,cell):
    """input kmesh = kmesh[0,:] = K
        kmesh[1,:] = K + np.array([dk,0,0])
        kmesh[2,:] = K + np.array([-dk,0,0])
        kmesh[3,:] = K + np.array([0,dk,0])
        kmesh[4,:] = K + np.array([0,-dk,0])"""
    fermi_ind = np.shape(eigenvalues)[0]//2
    recip_cell = get_recip_cell(cell)
    dk_ = dk * np.linalg.norm(cell,axis=0)

    homo = eigenvalues[fermi_ind-1,:]
    lumo = eigenvalues[fermi_ind,:]

    fermi_vel = np.zeros(3)
    fermi_vel[0] = np.abs(homo[1]-homo[0]) / dk_[0] /2 + np.abs(homo[2]-homo[0]) / dk_[0] /2 
    fermi_vel[1] = np.abs(homo[3]-homo[0]) / 2 / dk_[1] +  np.abs(homo[4]-homo[0]) / 2 / dk_[1]
    return np.linalg.norm(fermi_vel)

def get_fermi_velocity(eigenvalues,cell,kpoints):
    recip_cell = get_recip_cell(cell)
    kpoints = kpoints@recip_cell.T
    fermi_ind = np.shape(eigenvalues)[0]//2
    homo_band = eigenvalues[fermi_ind-1,:]
    lumo_band = eigenvalues[fermi_ind,:]

    #K point is at index 0
    dk = np.linalg.norm(kpoints[0,:]-kpoints[1,:])
    fermi_vel = np.abs(homo_band[0] - homo_band[1])/dk

    return fermi_vel

                
def get_band_gaps(eigenvalues):
    fermi_ind = np.shape(eigenvalues)[0]//2
    
    #make sure these are right bands
    upper_disp_band = eigenvalues[fermi_ind+2,:]
    upper_flat_band = eigenvalues[fermi_ind+1,:]
    lower_disp_band =  eigenvalues[fermi_ind-3,:]
    lower_flat_band =  eigenvalues[fermi_ind-2,:]

    if np.min(lower_flat_band) < np.max(lower_disp_band):
        homo_gap = 0
    else:
        homo_gap = np.min(lower_flat_band)-np.max(lower_disp_band)
    if np.max(upper_flat_band) > np.min(upper_disp_band):
        lumo_gap = 0
    else:
        lumo_gap = np.max(upper_flat_band) - np.min(upper_disp_band)
    return np.abs(homo_gap),np.abs(lumo_gap)

def get_twist_geom(theta,layer_sep=3.35,a=2.46):
    #comp is 2d vector for compression percentage along cell vectors
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                        mass=[12.01,12.01],sep=layer_sep,h_vac=20)
    return atoms

def get_fermi_energy(eigvals):
    nvals = np.shape(eigvals)[0]
    lumo = np.min(eigvals[nvals//2,:])
    homo = np.max(eigvals[nvals//2-1,:])
    return (homo+lumo)/2

if __name__=="__main__":
    csfont = {'fontname':'serif',"size":13}

    theta_list = np.array([0.88,0.99,1.08,1.12,1.16,1.20,1.47,1.89,2.88])
    mean_fermi_vel = np.zeros(len(theta_list))
    std_fermi_vel = np.zeros(len(theta_list))
    mean_band_width = np.zeros(len(theta_list))
    std_band_width = np.zeros(len(theta_list))
    mean_homo_gap = np.zeros(len(theta_list))
    std_homo_gap = np.zeros(len(theta_list))
    mean_lumo_gap = np.zeros(len(theta_list))
    std_lumo_gap = np.zeros(len(theta_list))
    fermi_vel_magic_angle = 0.99 * np.ones((len(theta_list),24))
    erange =     [0.1 ,0.1 ,0.1 ,0.1,0.1,0.1 ,0.1 ,0.4 ,0.4]
    dk = 1e-2

    for j,t in enumerate(theta_list):
        atoms = get_twist_geom(t)
        cell = atoms.get_cell()
        files = glob.glob("../uncertainty_quantification/ensemble_bands_wrong/band_structure_"+str(t)+"_*.npz",recursive=True)
        ensemble_evals = [] # np.zeros((len(files),len(atoms),61))
        homo_gap = np.zeros(len(files))
        lumo_gap = np.zeros(len(files))
        fermi_vel = np.zeros(len(files))
        band_width = np.zeros(len(files))
        print(20*"=",str(t),20*"=")
        for i,f in enumerate(files):
            data = np.load(f)
            evals = data["evals"]
            evals -= get_fermi_energy(evals)
            if np.shape(evals)[0] != len(atoms):
                continue
            kvec= data["kvec"]
            k_dist=data["k_dist"]
            k_node=data["k_node"]
            kdat = [kvec,k_dist,k_node]
            #ensemble_evals[i,:,:] = evals
            ensemble_evals.append(evals)
            band_width[i] = get_band_width(evals)
            homo_gap[i], lumo_gap[i] = get_band_gaps(evals)
            fermi_vel[i] = get_fermi_velocity(evals,cell,kvec)
            fermi_vel_magic_angle[j,i] = get_fermi_velocity(evals,cell,kvec)
        evals_mean = np.mean(ensemble_evals,axis=0)
        evals_std = np.std(ensemble_evals,axis=0)
        mean_fermi_vel[j] = np.mean(fermi_vel)
        std_fermi_vel[j] = np.std(fermi_vel)

        mean_band_width[j] = np.mean(band_width)
        std_band_width[j] = np.std(band_width)

        mean_homo_gap[j] = np.mean(homo_gap)
        std_homo_gap[j] = np.std(homo_gap)

        mean_lumo_gap[j] = np.mean(lumo_gap)
        std_lumo_gap[j] = np.std(lumo_gap)

        theta_str = "_".join(str(t).split("."))
        plot_bands(evals_mean,evals_std,kdat,efermi=None,colors='black',title='',label="",linestyle="solid",erange=erange[j],figname="figures/band_structure"+theta_str+"MK_classical_UQ.jpg")


        plt.plot(theta_list,mean_fermi_vel,color = "black",linestyle="solid",marker="o",label="Mean")
        plt.plot(theta_list,std_fermi_vel,color = "blue",linestyle="solid",label="STD")
        plt.fill_between(theta_list,mean_fermi_vel-std_fermi_vel,mean_fermi_vel+std_fermi_vel,alpha=0.3,facecolor="black")
        plt.legend()
        plt.xlabel("twist angle (degrees)",**csfont)
        plt.ylabel("fermi velocity (eV*Angstroms)",**csfont)
        plt.savefig("figures/fermi_velocities_uq_mk_classical.jpg")
        plt.clf()

        plt.plot(theta_list,mean_band_width,color = "black",linestyle="solid",marker="o",label="Mean")
        plt.plot(theta_list,std_band_width,color = "blue",linestyle="solid",label="STD")
        plt.fill_between(theta_list,mean_band_width-std_band_width,mean_band_width+std_band_width,alpha=0.3,facecolor="black")
        plt.legend()
        plt.xlabel("twist angle (degrees)",**csfont)
        plt.ylabel("Band Width (eV)",**csfont)
        plt.savefig("figures/band_width_uq_mk_classical.jpg")
        plt.clf()
        
        plt.plot(theta_list,mean_homo_gap,color = "black",linestyle="solid",marker="o",label="Mean")
        plt.plot(theta_list,std_homo_gap,color = "blue",linestyle="solid",label="STD")
        lower_homo_gap = mean_homo_gap-std_homo_gap
        neg_ind = np.where(lower_homo_gap<0)
        lower_homo_gap[neg_ind] = 0
        plt.fill_between(theta_list,lower_homo_gap,mean_homo_gap+std_homo_gap,alpha=0.3,facecolor="black")
        plt.legend()
        plt.xlabel("twist angle (degrees)",**csfont)
        plt.ylabel("Homo Gap (eV)",**csfont)
        plt.savefig("figures/homo_gap_uq_mk_classical.jpg")
        plt.clf()

        plt.plot(theta_list,mean_lumo_gap,color = "black",linestyle="solid",marker="o",label="Mean")
        plt.plot(theta_list,std_lumo_gap,color = "blue",linestyle="solid",label="STD")
        lower_lumo_gap = mean_lumo_gap-std_lumo_gap
        neg_ind = np.where(lower_lumo_gap<0)
        lower_lumo_gap[neg_ind] = 0
        plt.fill_between(theta_list,lower_lumo_gap,mean_lumo_gap+std_lumo_gap,alpha=0.3,facecolor="black")
        plt.legend()
        plt.xlabel("twist angle (degrees)",**csfont)
        plt.ylabel("Lumo Gap (eV)",**csfont)
        plt.savefig("figures/lumo_gap_uq_mk_classical.jpg")
        plt.clf()

    magic_angle_ensemble = theta_list[np.argmin(fermi_vel_magic_angle,axis=0)]
    print(magic_angle_ensemble)
    print("average magic angle = ",np.mean(magic_angle_ensemble))
    print("std magic angle = ",np.std(magic_angle_ensemble))

