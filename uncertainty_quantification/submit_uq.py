from pyexpat import model
import subprocess
from datetime import datetime
import numpy as np
import glob
import os
import ase.io

def submit_batch_file_uiuc_cc(executable,batch_options,
                                 conda_env='blg_uq'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+' '+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("\nsource activate "+conda_env+"\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)
        

def submit_batch_file_delta(executable,batch_options,
                                 conda_env='myenv'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+'='+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")
        
        f.write("\nconda activate "+conda_env+"\n")
        f.write("conda deactivate\n")
        f.write("\nconda activate "+conda_env+"\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)

def submit_batch_file_aurora(executable,batch_options,
                                 conda_env='/lus/flare/projects/qmchamm/dpalmer3/venv/bin/activate',
                                 dir='/lus/flare/projects/qmchamm/dpalmer3/BLG_model_builder/PYMC_uncertainty_quanitification'):

    sbatch_file="job"+str(hash(datetime.now()) )+".qsub"
    batch_copy = batch_options.copy()

    prefix="#PBS "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+' '+str(batch_copy[key])+"\n")

        f.write("\nsource "+conda_env+"\n")
        f.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
        f.write("export OMP_NUM_THREADS=1\n")
        for m in modules:
            f.write("module load "+m+"\n")
        f.write('cd '+dir+"\n")
        f.write(executable)
    subprocess.call("qsub "+sbatch_file,shell=True)

if __name__=="__main__":

    mcmc_uq = True 
    cv_uq = False
    relaxation = False
    band_structure = False
    rerelax = False

    batch_options_uiuc_cc= {
                 '--partition':'qmchamm',
                 #'--partition':'secondary',
                 '--nodes':1,
                 '--ntasks':1,
                 '--cpus-per-task':1,
                 '--time':'48:00:00',
                 '--output':'uq.log',
                 '--job-name':'uq',
                 'modules':['anaconda/2023-Mar/3']
        }
    batch_options_delta = {
            "--nodes":"1",
            "--time":"48:00:00",
            "--account":"bcmp-delta-gpu",
            "--partition":"gpuA100x4,gpuA40x4",
            #"--partition":"gpuA100x8",
            "--job-name":"prod",
            "--gpus-per-task":"1",
            "--cpus-per-task":"1",
            "--ntasks-per-node":"1",
            "--mem":"208g",
            "modules":['pytorch-conda/2.8']}

    batch_options_delta_cpu = {
            "--nodes":"1",
            "--time":"8:00:00",
            "--account":"bcmp-delta-cpu",
            "--partition":"cpu",
            "--job-name":"prod",
            "--cpus-per-task":"1",
            "--ntasks-per-node":"1",
            "--mem":"20g",
            "modules":['pytorch-conda/2.8']}
    
    batch_options_aurora = {
            "-A": "qmchamm",
            "-N": "job_name",
            "-l": "select=1:ncpus=1:ngpus=1,walltime=06:00:00,filesystems=flare",
            "-q": "prod",
            "modules":['frameworks']}

    batch_options = batch_options_uiuc_cc


    """int_type = ['interlayer','intralayer'] 
    energy_model = ['Classical','TETB']
    tb_model = ['MK','popov','LETB']
    calc_type = ['python','lammps']"""

    model_names = ["MK","intralayer_LETB_NN_val_1","intralayer_LETB_NN_val_2","intralayer_LETB_NN_val_3","interlayer_LETB","POD_SK"]
    model_names = ["Tersoff+DRIP", "Tersoff+Kolmogorov_Crespi","POD_energy","TETB_POD"]
    model_names = "ACSF_hoppings_sk" # "POD_energy" #

    if mcmc_uq:
        T_weight_array = np.array([1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5,1,1.5,2.0,3,4,5,7,10,15,20,30,50,100,150,200,300,500]) 
        T_weight_array = np.array([1e-5,1e-4,1e-3,0.01,0.1,1,5,10,25,50,100,150,200,300,500]) 
        T_weight_array = np.array([np.round(10**i,8) for i in np.linspace(-4,6,20)])
        T_weight_array = np.array([1e-2,5e-2, 0.1, 0.25,0.5,0.75,1,1.25,1.5, 1.75, 2,3,4,5,10,20,30,40,50]) #np.linspace(0.1,2,8)
        M_array = [8,10,12]
        W_array = [6]
        from blg_model_builder.pod_model_selection import load_use_pod_model_hashes

        _pod_hashes = load_use_pod_model_hashes()
        pod_index_array = list(range(len(_pod_hashes)))

        if model_names =="ACSF_hoppings_sk":
            for M in M_array:
                for W in W_array:
                    for TW in T_weight_array:
                        executable = "python run_MCMC.py -m "+model_names +" -M "+str(M)+" -W "+str(W)+ " -B "+str(TW)
                        batch_options["--job-name"]=model_names+"_M_"+str(M)+"_W_"+str(W)+"_T_"+str(TW)
                        batch_options["--output"]= model_names+"_M_"+str(M)+"_W_"+str(W)+"_T_"+str(TW)+".log"
                        print(executable)
                        #submit_batch_file_uiuc_cc(executable,batch_options)
                        subprocess.call(executable,shell=True)
                        #exit()
                            
        elif model_names == "POD_energy" or model_names == "TETB_POD":
            for pod_index in pod_index_array:
                #if pod_index>6:
                #    continue
                for TW in T_weight_array:
                    executable = "python run_MCMC.py -m "+model_names +" --POD-index "+str(pod_index)+" -B "+str(TW)
                    batch_options["--job-name"]=model_names+"_POD_index_"+str(pod_index)
                    batch_options["--output"]= model_names+"_POD_index_"+str(pod_index)+".log"
                    print(executable)
                    submit_batch_file_uiuc_cc(executable,batch_options)
                    #subprocess.call(executable,shell=True)
                    #exit()

    if cv_uq:
        psubset_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        for mt in model_tuple:
            for p in psubset_arr:
                if len(mt)==4:
                    nn_val = mt[-1]
                else:
                    nn_val=1
                hyper_param_str = "int_"+str(mt[0])+"_e_"+str(mt[1])+"_tb_"+str(mt[2])+"_p_"+str(p)+"_nn_"+str(nn_val)

                executable = "python run_SubSamp.py -i "+str(mt[0])+" -e "+str(mt[1])+" -t "+str(mt[2]) +" -p "+str(p)+" -n "+str(nn_val)
                batch_options["--job-name"]=hyper_param_str
                batch_options["--output"]= hyper_param_str+".log"
                print(executable)
                #submit_batch_file_uiuc_cc(executable,batch_options)
                subprocess.call(executable,shell=True)

    if relaxation:
        uq_arr = ["mcmc"] #,"cv"]
        twist_angle = np.array([0.83,0.88,0.93,0.99,1.05,1.08,1.12,1.16,1.2,1.47]) #
        from blg_model_builder.pod_model_selection import load_use_pod_model_hashes

        _pod_hashes = load_use_pod_model_hashes()
        model_names = [
            f"POD_energy_POD_index_{i}_{h}" for i, h in enumerate(_pod_hashes)
        ]
        for mt in model_names:
            if "POD_index_0" in mt or "POD_index_9" in mt:
            
                for t in twist_angle:
                    hyper_param_str = mt+"_a_"+str(t)

                    executable = "python run_uq_propagation_relaxation.py --models "+str(mt)+" --twist-angle "+str(t)
                    #batch_options["-N"]=hyper_param_str
                    batch_options["--output"]= hyper_param_str+".log"
                    batch_options["--job-name"]=hyper_param_str
                
                    print(executable)
                    submit_batch_file_uiuc_cc(executable,batch_options)
                    #exit()
                    #subprocess.call(executable,shell=True)

    if rerelax:
        files = glob.glob("TETB_relaxations/TETB_energy_popov_t_*",recursive=True)
        print("num files = ",len(files))
        njobs=0
        for file in files:
            theta_val = float(file.split("_")[-2])
            hyper_param_str = "e_TETB_tb_popov_uq_mcmc_a_"+str(theta_val)
            batch_options["--output"]= hyper_param_str+".log"
            
            wall_time="3:00:00"
            batch_options["--time"] = wall_time
            atoms_file = os.path.join(file,"mcmc_theta_"+str(theta_val)+".traj")
            try:
                atoms = ase.io.read(atoms_file)
            except Exception as e:
                print(f"failed with exception {e}")
                continue
            forces = atoms.get_forces()
            if np.abs(np.max(forces)) < 1e-3:
                continue
            #print("tol = ",1e-3,"max force = ",np.max(forces)," mean force = ",np.mean(forces))
            executable = "python run_uq_propagation.py -c "+file+" -q rerelax -t popov -e TETB -u mcmc" 
            #print(executable)
            njobs +=1
            submit_batch_file_delta(executable,batch_options)
            #exit()
        print("njobs = ",njobs)

    if band_structure:
        uq_arr = ["mcmc"] #
        twist_angle = np.array([0.83,0.88,0.93,0.99,1.05,1.08,1.12,1.16,1.2,1.47])
        relaxation_model = "POD_energy_POD_index_0_09fdb1c2b98eb30e"
        tb_model = "ACSF_hoppings_sk_M_12_W_6"
        for t in twist_angle:
            hyper_param_str = relaxation_model+"_tb_"+tb_model+"_a_"+str(t)

            executable = "python run_uq_propagation_bands.py --models "+relaxation_model+" --tb-model "+tb_model+" --twist-angle "+str(t)
            batch_options["--job-name"]=hyper_param_str
            batch_options["--output"]= hyper_param_str+".log"
            print(executable)
            submit_batch_file_delta(executable,batch_options)
            #exit()


    


