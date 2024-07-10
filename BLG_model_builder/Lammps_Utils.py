import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
from lammps import lammps
import re
import glob
import TETB_GRAPHENE
from ase.neighborlist import NeighborList

def init_pylammps_depr(atoms,kc_file = None,rebo_file = None):
    """ create pylammps object and calculate corrective potential energy 
    """
    ntypes = len(set(atoms.get_chemical_symbols()))
    data_file = "tegt.data"
    ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
    L = PyLammps(verbose=False)
    L.command("units		metal")
    L.command("atom_style	full")
    L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
    L.command("box tilt large")

    L.command("read_data "+data_file)

    L.command("group top type 1")
    L.command("mass 1 12.0100")

    if ntypes ==2:
        L.command("group bottom type 2")
        L.command("mass 2 12.0100")

    L.command("velocity	all create 0.0 87287 loop geom")
    # Interaction potential for carbon atoms
    ######################## Potential defition ########################
    
    if ntypes ==2:
        L.command("pair_style       hybrid/overlay reg/dep/poly 10.0 0 airebo 3")
        L.command("pair_coeff       * *   reg/dep/poly  "+kc_file+"   C C") # long-range 
        L.command("pair_coeff      * * airebo "+rebo_file+" C C")

    else:
        L.command("pair_style       airebo 3")
        L.command("pair_coeff      * * "+rebo_file+" C")

    ####################################################################
    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")
    return L

def init_pylammps(atoms,model_names,model_files):
    """ create pylammps object and calculate corrective potential energy 
    """
    ntypes = len(set(atoms.get_chemical_symbols()))
    masses = list(set(atoms.get_masses()))
    data_file = "tegt.data"
    ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
    L = PyLammps(verbose=False)
    L.command("units		metal")
    L.command("atom_style	full")
    L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
    L.command("box tilt large")

    L.command("read_data "+data_file)
    for n in range(ntypes):
        L.command("group group_num_"+str(n+1)+" type "+str(n+1))
        L.command("mass "+str(n+1)+" "+str(masses[n%len(masses)]))

    if len(model_names)>1:
        pair_style = "pair_style       hybrid/overlay " 
    else:
        pair_style = "pair_style "
    pair_coeff = []
    for i,m in enumerate(model_names):
        #L.command("group group_num_"+str(i+1)+" type "+str(i+1))
        #L.command("mass "+str(i+1)+" 12.0100") #get mass of group 1 automatically

        pair_style += m+" "
        model_name = m.split(" ")[0]
        pair_coeff.append("pair_coeff * * "+model_name+" "+model_files[i]+" "+"C "*ntypes)

    L.command("velocity	all create 0.0 87287 loop geom")
    L.command(pair_style)
    for i,m in enumerate(model_names):
        L.command(pair_coeff[i])

    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")
    return L

def init_pylammps_classical_depr(atoms,kc_file = None,rebo_file = None):
    """ create pylammps object and calculate corrective potential energy 
    """
    ntypes = len(set(atoms.get_chemical_symbols()))
    data_file = "tegt.data"
    ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
    L = PyLammps(verbose=False)
    L.command("units		metal")
    L.command("atom_style	full")
    L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
    L.command("box tilt large")

    L.command("read_data "+data_file)

    L.command("group top type 1")
    L.command("mass 1 12.0100")

    L.command("velocity	all create 0.0 87287 loop geom")
    # Interaction potential for carbon atoms
    ######################## Potential defition ########################

    L.command("pair_style       hybrid/overlay kolmogorov/crespi/full 10.0 0 rebo ")
    L.command("pair_coeff       * *   kolmogorov/crespi/full  "+kc_file+"    C") # long-range 
    L.command("pair_coeff      * * rebo "+rebo_file+" C ")

    ####################################################################
    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")
    return L

def run_lammps(atoms,model_names,model_files):
    """ evaluate corrective potential energy, forces in lammps 
    """

    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        sym = atoms.get_chemical_symbols()
        top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        mol_id[top_layer_ind] += 1
        atoms.set_array("mol-id",mol_id)

    L = init_pylammps(atoms,model_names,model_files)

    forces = np.zeros((atoms.get_global_number_of_atoms(),3))

    L.run(0)
    pe = L.eval("pe")
    ke = L.eval("ke")
    for i in range(atoms.get_global_number_of_atoms()):
        forces[i,:] = L.atoms[i].force
    #del L

    return forces,pe,pe+ke


##############################################################################################

# Write Potential files for KC, KCinsp, and rebo

##############################################################################################

def write_kcinsp(params,kc_file):
    """write kc inspired potential """
    params = params[:9]
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['', "delta","C","C0 ","C2","C4","z0","A6","A8","A10"])
    with open(kc_file, 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
        
def write_kc(params,kc_file):
    """write kc inspired potential """
    
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['','z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'])
    with open(kc_file, 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
    

def check_keywords(string):
    """check to see which keywords are in string """
    keywords = ['Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2', 'BIJc_CC3','Beta_CC1', 
                'Beta_CC2','Beta_CC3']
    
    for k in keywords:
        if k in string:
            return True, k
        
    return False,k

def write_rebo(params,rebo_file):
    """write rebo potential given list of parameters. assumed order is
    Q_CC , alpha_CC, A_CC, BIJc_CC1, BIJc_CC2 ,BIJc_CC3, Beta_CC1, Beta_CC2,Beta_CC3
    
    :param params: (list) list of rebo parameters
    """
    keywords = [ 'Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2','BIJc_CC3', 'Beta_CC1', 
            'Beta_CC2', 'Beta_CC3']
    param_dict=dict(zip(keywords,params))
    #copy pre-writtten rebo file to desired location then rewrite parameters
    repo_root = os.path.join("/".join(TETB_GRAPHENE.__file__.split("/")[:-1]),"parameters")
    subprocess.call("cp " +os.path.join(repo_root,"CH_pz.rebo")+" "+rebo_file,shell=True)
    with open(rebo_file, 'r') as f:
        lines = f.readlines()
        new_lines=[]
        for i,l in enumerate(lines):
            
            in_line,line_key = check_keywords(l)
            
            if in_line:
                nl = str(np.squeeze(param_dict[line_key]))+" "+line_key+" \n"
                new_lines.append(nl)
            else:
                new_lines.append(l)
    with open(rebo_file, 'w') as f:        
        f.writelines(new_lines)

#####################################################################################################

# Python

####################################################################################################

def get_norm(atoms,neighbor_indices,offsets):    
    positions = atoms.positions
    cell = atoms.get_cell()
    midplane = np.mean(positions[:,2])

    neighbors_positions = positions[neighbor_indices[1:],:] + np.dot(offsets[1:], cell)
    dis = neighbors_positions - positions[neighbor_indices[0],:]
    inplane = dis[neighbor_indices[:3],:] #gets the distance vector of the inplane nearest neighbors
    print("inplane ",inplane)
    prod1 = np.cross(inplane[0,:], inplane[1,:])
    print("prod1 ",prod1)
    prod2 = np.cross(inplane[1,:], inplane[2,:])
    prod3 = np.cross(inplane[2,:], inplane[0,:])
    
    #normalizing the products
    prod1 = np.sign(prod1[2])*prod1/np.linalg.norm(prod1) #%making sure all three normals are positive
    prod2 = np.sign(prod2[2])*prod2/np.linalg.norm(prod2)
    prod3 = np.sign(prod3[2])*prod3/np.linalg.norm(prod3)

    
    normal =(prod1+prod2+prod3)/3
    
    if positions[neighbor_indices[0],2] > midplane: #%if atom "r" is in upper layer of graphene, normal is downward
        normal= -normal

    return normal

def getvdW(rijvec, ni, nj, curConstants):

    C0 = curConstants[0]
    C2 = curConstants[1]
    C4 = curConstants[2]
    C = curConstants[3]
    delta = curConstants[4]
    A6 = curConstants[5]
    A8 = curConstants[6]
    A10 = curConstants[7]
   
    z0 = 3.34 #AB bilayer experimentnal separation in Angstrom
    
    rij = np.linalg.norm(rijvec,axis=1)
 
    rhoij =np.sqrt(rij**2-(np.dot(ni,rijvec))**2) 
    rhoji =np.sqrt(rij**2-(np.dot(nj,rijvec))**2)  
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0*np.power(rhoij/delta,0)+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0*np.power(rhoji/delta,0)+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    
    V=-(C+frhoij+frhoji)*(A6*np.power(rij/z0,-6)+A8*np.power(rij/z0,-8)+A10*np.power(rij/z0,-10))

    return V


def KCinsp(atoms,parameters,neighbor_indices,offsets):
    norm = atoms.get_array('norm')
    positions = atoms.positions
    cell = atoms.get_cell()
    layer_index = atoms.get_array("mol-id")
    self_index = neighbor_indices[0]
    neighbor_indices = neighbor_indices[1:]
    offsets = offsets[1:]
    neighbors_positions = positions[neighbor_indices,:] + np.dot(offsets, cell)
    displacements = neighbors_positions - positions[self_index,:] #displacements between all neighbors of atom i over pbc

    use_ind = np.where(layer_index[self_index]!=layer_index[neighbor_indices])
    potential_energy = getKCinsp(displacements,norm[self_index],norm[neighbor_indices[use_ind]],parameters)
    return potential_energy

def fc(r):
    Dij_min = 1.7
    Dij_max = 2
    if r<Dij_min:
        return 1
    elif r>Dij_min and r<Dij_max:
        return (1+np.cos((r-Dij_min)/(Dij_max-Dij_min)))/2
    else:
        return 0

def get_bond_order(atoms,neighbor_indices,offsets):
    G = 0
    for i in range(len(offsets)):
        G += fc(r)*angular(np.cos(theta))*np.exp(lambda_ijk) + P
    bij_sp = 1/np.sqrt(1+ G)
    bij_pi = G_RC + b_DH
    bij = 0.5*(bij_sp+bji_sp)+bij_pi
    return bij

def REBO(atoms,parameters,neighbor_indices,offsets):
    [ Q ,alpha, A,BIJc_CC1, BIJc_CC2,BIJc_CC3, Beta_CC1, 
            Beta_CC2, Beta_CC3] = parameters
    positions = atoms.positions
    cell = atoms.get_cell()
    self_index = neighbor_indices[0]
    neighbor_indices = neighbor_indices[1:]
    offsets = offsets[1:]
    neighbors_positions = positions[neighbor_indices,:] + np.dot(offsets, cell)
    displacements = neighbors_positions - positions[self_index,:]

    potential_energy = 0
    for i in range(len(offsets)):
        r = np.linalg.norm(displacements[i,:])
        Vr = fc(r)*(1+Q/r)*A*np.exp(-alpha/r)
        Va = fc(r) * (BIJc_CC1*np.exp(-Beta_CC1*r) + BIJc_CC2*np.exp(-Beta_CC2*r) + BIJc_CC3*np.exp(-Beta_CC3*r))
        bond_order = get_bond_order(r)
        potential_energy += Va - bond_order * Vr
    return potential_energy

def get_potential_energy(atoms,parameters,pe_funcs,r_cut=10):
    #%calculation of pairwise potential
    potential_energy = 0
    
    cutoffs = r_cut*np.ones(len(atoms))
    atoms.set_pbc(True)
    nl = NeighborList(cutoffs)
    nl.update(atoms)

    norm = np.zeros((len(atoms),3))
    for i in range(len(atoms)):
        neighbor_indices, offsets = nl.get_neighbors(i) 
        norm[i,:] = get_norm(atoms,neighbor_indices,offsets) 
    atoms.set_array('norm',norm)
    for i in range(len(atoms)):
        neighbor_indices, offsets = nl.get_neighbors(i) 
        for f in pe_funcs:
            potential_energy +=f(atoms,parameters,neighbor_indices,offsets)
    return potential_energy

def pairwiseKC(atoms, parameters, tol=0.26, r_cut=10):
    xyz = atoms.get_positions()
    cell_vecs = atoms.get_cell()
    periodicR1 = cell_vecs[0]
    periodicR2 = cell_vecs[1]
    natoms = xyz.shape[0]
    uplayer = np.zeros((int(len(xyz)/2),3))
    downlayer = np.zeros((int(len(xyz)/2),3))
    
    #dividing upper layer and down layer for plotting
    a=0
    b=0
    zmean = np.mean(xyz[:,2])
    
    for i in range(len(xyz)):
        if xyz[i,2] > zmean:
            uplayer[a,:]=xyz[i,:]
            a+=1
        else: 
            downlayer[b,:]=xyz[i,:]
            b+=1
    
    meanup = np.mean(uplayer[:,2])
    meandown = np.mean(downlayer[:,2])
    sep = meanup - meandown

    xyzperiodic = np.zeros((xyz.shape[0]*9,3)) #%creating 9 boxes of atoms
    indices = [0, -1, 1] #%to force the first atom at 000
    for   periodicI in range (0,3,1):
        i2 = indices[periodicI]
        for periodicJ in range (0,3,1):
            j2 = indices[periodicJ]
            for i in range (0,xyz.shape[0],1): #%goint to each atom one by one
                index = natoms*(periodicI)+natoms*3*(periodicJ)+i #%index of all the 9 boxes
                xyzperiodic[index,:] = xyz[i,:] + i2*periodicR1 + j2*periodicR2#%will not append them to make 9 blocks   
    
    
    normal = np.zeros((xyz.shape[0],3)) #%normal is a vector
    for i in range(xyz.shape[0]):
        normal[i,:]=getNormal(xyzperiodic[i,:], xyzperiodic, tol, sep/2)
 
    #%calculation of pairwise potential
    potential_energy = 0
    positions = atoms.positions
    cell = atoms.get_cell()
    cutoffs = r_cut*np.ones(len(atoms))
    atoms.set_pbc(True)
    nl = NeighborList(cutoffs, self_interaction=False)
    nl.update(atoms)
    for i in range(len(atoms)):
        neighbor_indices, offsets = nl.get_neighbors(i) 
        neighbors_positions = positions[indices,:] + np.dot(offsets, cell)
        displacements = neighbors_positions - positions[i,:] #displacements between all neighbors of atom i over


    pairwiseEnergy = np.zeros((xyz.shape[0]))
    
    for periodicI in [-1, 0, 1]:
        for periodicJ in [-1, 0, 1]:    

            for i in range (0,xyz.shape[0], 1): #%going to each atom one by one
                ri = xyz[i,:]      #%+ periodicI*periodicR1' + periodicJ*periodicR2';
                normi = normal[i,:] 
                
                for j in range (i+1, xyz.shape[0], 1): #% avoiding 'i' to avoid self interaction
                    rj = xyz[j,:] +periodicI*periodicR1 + periodicJ*periodicR2  #; %it is searching within xyz
                    dist = np.linalg.norm(rj - ri)
                    vertical = np.abs(rj[2]-ri[2]) 
                    normj = normal[j, :]
                        
                    #%INTERLAYER INTERACTION, NOT APPLYING R_CUT, RIGIDLY CHOOSING NEIGHBOR
                    if dist < r_cut and vertical > tol:#only interlayer pairwise energy
                         Vij = getvdW(ri, rj,normi, normj,  parameters)
                         Vrepel += Vij #got rid of double counting                   
                         #pairwiseEnergy[i] += 0.5*Vij #
                         #pairwiseEnergy[j] += 0.5*Vij
    
  
    return Vrepel/len(xyz) #meV/atom 

if __name__ == "__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    from pythtb import *
    from bilayer_letb.api import tb_model

    sep = 3.35
    a = 2.46
    n=5
    theta=21.78
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                    mass=[12.01,12.01],sep=sep,h_vac=20)
    parameters = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
                                        1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
    pe = get_potential_energy(atoms,parameters,[KCinsp])
