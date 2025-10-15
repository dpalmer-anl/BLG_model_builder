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
import uuid
import TETB_GRAPHENE
from ase.neighborlist import NeighborList

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
        L.command("mass "+str(n+1)+" 12.01")

    if len(model_names)>1:
        pair_style = "pair_style       hybrid/overlay " 
        pair_coeff = []
    else:
        pair_style = "pair_style       hybrid/overlay zero 10 "
        pair_coeff = ["pair_coeff * * zero"]
    
    for i,m in enumerate(model_names):
        #L.command("group group_num_"+str(i+1)+" type "+str(i+1))
        #L.command("mass "+str(i+1)+" 12.0100") #get mass of group 1 automatically

        pair_style += m+" "
        model_name = m.split(" ")[0]
        pair_coeff.append("pair_coeff * * "+model_name+" "+model_files[i]+" "+"C "*ntypes)
    
    L.command("velocity	all create 0.0 87287 loop geom")
    L.command(pair_style)
    for i,m in enumerate(pair_coeff):
        L.command(m)

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

def pylammps_relax(atoms,model_names,model_files):
    """ create pylammps object and calculate corrective potential energy """
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
        L.command("mass "+str(n+1)+" 12.01")

    if len(model_names)>1:
        pair_style = "pair_style       hybrid/overlay " 
        pair_coeff = []
    else:
        pair_style = "pair_style       hybrid/overlay zero 10 "
        pair_coeff = ["pair_coeff * * zero"]
    
    for i,m in enumerate(model_names):
        #L.command("group group_num_"+str(i+1)+" type "+str(i+1))
        #L.command("mass "+str(i+1)+" 12.0100") #get mass of group 1 automatically

        pair_style += m+" "
        model_name = m.split(" ")[0]
        pair_coeff.append("pair_coeff * * "+model_name+" "+model_files[i]+" "+"C "*ntypes)

    L.command("velocity	all create 0.0 87287 loop geom")
    L.command(pair_style)
    for i,m in enumerate(pair_coeff):
        L.command(m)

    L.command("timestep 0.00025")
    L.command("thermo 100")
    L.command("fix 1 all nve")
    L.command("min_style fire")
    L.command("minimize       1e-8 1e-9 3000 10000")
    L.command("dump mydump2 all custom 1 dump.tblg id type x y z fx fy fz")
    pe = L.eval("pe")
    ke = L.eval("ke")
    forces = np.zeros((len(atoms),3))
    positions = np.zeros((len(atoms),3))
    for i in range(atoms.get_global_number_of_atoms()):
        forces[i,:] = L.atoms[i].force.copy()
        positions[i,:] =  L.atoms[i].position
    
    atoms.set_positions(positions)
    del L
    return atoms,forces


def evaluate_lammps(atoms,model_names,model_files):
    """ evaluate corrective potential energy, forces in lammps 
    """

    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        sym = atoms.get_chemical_symbols()
        top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        mol_id[top_layer_ind] += 1
        atoms.set_array("mol-id",mol_id)
    mol_id = atoms.get_array("mol-id")
    positions=atoms.positions.copy()

    ntypes = len(set(atoms.get_chemical_symbols()))
    masses = list(set(atoms.get_masses()))
    
    L = PyLammps(verbose=False) #,cmdargs=["-log", "none"])
    L.command("units		metal")
    L.command("atom_style	full")
    L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
    L.command("box tilt large")
    a, b, c = atoms.cell.copy()
    xlo, ylo, zlo = 0, 0, 0
    xhi = np.linalg.norm(a)
    xy  = np.dot(b, a) / xhi
    xz  = np.dot(c, a) / xhi
    yhi = np.linalg.norm(b)
    yz  = np.dot(c, b) / yhi
    zhi = np.linalg.norm(c)

    L.command(f"region myreg prism {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} {xy} {xz} {yz} units box")
    L.command(f"create_box {ntypes} myreg")
    L.lmp.create_atoms(len(atoms),list(np.arange(1,1+len(atoms),dtype=np.int8)),list(np.ones(len(atoms),np.int8)),list(positions.flatten()))
    mol = L.lmp.numpy.extract_atom("molecule")
    mol[:] = mol_id.copy()

    for n in range(ntypes):
        L.command("group group_num_"+str(n+1)+" type "+str(n+1))
        L.command("mass "+str(n+1)+" 12.01")

    if len(model_names)>1:
        pair_style = "pair_style       hybrid/overlay " 
        pair_coeff = []
    else:
        pair_style = "pair_style       hybrid/overlay zero 10 "
        pair_coeff = ["pair_coeff * * zero"]
    
    for i,m in enumerate(model_names):
        #L.command("group group_num_"+str(i+1)+" type "+str(i+1))
        #L.command("mass "+str(i+1)+" 12.0100") #get mass of group 1 automatically

        pair_style += m+" "
        model_name = m.split(" ")[0]
        pair_coeff.append("pair_coeff * * "+model_name+" "+model_files[i]+" "+"C "*ntypes)
    
    L.command("velocity	all create 0.0 87287 loop geom")
    L.command(pair_style)
    for i,m in enumerate(pair_coeff):
        L.command(m)

    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")

    forces = np.zeros((atoms.get_global_number_of_atoms(),3))

    L.run(0)
    pe = L.eval("pe")
    ke = L.eval("ke")
    for i in range(atoms.get_global_number_of_atoms()):
        forces[i,:] = L.atoms[i].force

    return forces,pe,pe+ke

    


##############################################################################################

# Write Potential files for KC, KCinsp, and rebo

##############################################################################################

def write_kcinsp(params,kc_file):
    """write kc inspired potential """
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
    
def write_Tersoff(params,tersoff_file):
    # tunable parameters: c, d, costheta0, n, beta, lambda2, B, lambda1, A
    # fixed parameters: m, gamma, lambda3, R, D
    params_str_1 = " ".join([str(x) for x in params[:-2]])
    params_str_2 = " ".join([str(x) for x in params[-2:]])
    with open(tersoff_file, 'w+') as f:
        f.write("# format of a single entry (one or more lines):\n\
                #   element 1, element 2, element 3,m, gamma, lambda3, c, d, costheta0, n,beta, lambda2, B, R, D, lambda1, A\n")
        f.write("C C C 3.0 1.0 0.0 "+params_str_1+" 1.95 0.15 "+params_str_2+"\n\n")
    
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
    params = np.squeeze(params)
    keywords = [ 'Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2','BIJc_CC3', 'Beta_CC1', 
            'Beta_CC2', 'Beta_CC3']
    param_dict=dict(zip(keywords,params))
    #copy pre-writtten rebo file to desired location then rewrite parameters
    repo_root = os.path.join("/".join(TETB_GRAPHENE.__file__.split("/")[:-1]),"parameters")
    subprocess.call("cp " +os.path.join(repo_root,"CH.rebo")+" "+rebo_file,shell=True)
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
        

def write_airebo(params,rebo_file):
    """write rebo potential given list of parameters. assumed order is
    Q_CC , alpha_CC, A_CC, BIJc_CC1, BIJc_CC2 ,BIJc_CC3, Beta_CC1, Beta_CC2,Beta_CC3
    
    :param params: (list) list of rebo parameters
    """
    params = np.squeeze(params)
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


