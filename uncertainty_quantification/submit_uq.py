import subprocess
from datetime import datetime
import numpy as np
import glob
import os
import ase.io

def submit_batch_file_uiuc_cc(executable,batch_options,
                                 conda_env='my.anaconda'):

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

    mcmc_uq = False
    cv_uq = False
    relaxation = False
    band_structure = True
    rerelax = False

    batch_options_uiuc_cc= {
                 '--partition':'qmchamm',
                 '--nodes':1,
                 '--ntasks':1,
                 '--cpus-per-task':1,
                 '--time':'72:00:00',
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
    
    batch_options_aurora = {
            "-A": "qmchamm",
            "-N": "job_name",
            "-l": "select=1:ncpus=1:ngpus=1,walltime=06:00:00,filesystems=flare",
            "-q": "prod",
            "modules":['frameworks']}

    batch_options = batch_options_delta


    """int_type = ['interlayer','intralayer'] 
    energy_model = ['Classical','TETB']
    tb_model = ['MK','popov','LETB']
    calc_type = ['python','lammps']"""

    model_tuple = [("interlayer","Classical",None),("intralayer","Classical",None),#("interlayer",None,"LETB"),("intralayer",None,"LETB","1"),
                   #("intralayer",None,"LETB","2"),("intralayer",None,"LETB","3"),
                   ("full",None,"MK"),
                   ("interlayer","TETB","MK"), ("intralayer","TETB","MK")]
    #model_tuple = [("interlayer","TETB","MK"), ("intralayer","TETB","MK")]
    model_tuple = [("interlayer","Classical",None),("intralayer","Classical",None)]
    model_tuple = [("interlayer",None,"LETB")]
    #model_tuple = [("interlayer",None,"LETB"),("intralayer",None,"LETB","1"),
    #               ("intralayer",None,"LETB","2"),("intralayer",None,"LETB","3"),
    #               ("full",None,"MK")]
    model_tuple = [("full",None,"MK")]

    if mcmc_uq:
        T_weight_array = np.array([1e-5,1e-4,1e-3,0.02,0.04,0.075]) #,0.01,0.1,0.2,0.5,0.75,1,1.5,2.0,3,4,5,10,50,100,200,500,1000])  
        nchains = 1
        for n in range(nchains):
            for mt in model_tuple:
                for TW in T_weight_array:
                    if len(mt)==4:
                        nn_val = mt[-1]
                    else:
                        nn_val=1
                    hyper_param_str = "int_"+str(mt[0])+"_e_"+str(mt[1])+"_tb_"+str(mt[2])+"_TW_"+str(TW)+"_nn_"+str(nn_val)

                    executable = "python EMCEE_generate_ensemble.py -i "+str(mt[0])+" -e "+str(mt[1])+" -t "+str(mt[2]) +" -B "+str(TW)+" -n "+str(nn_val)
                    batch_options["--job-name"]=hyper_param_str
                    batch_options["--output"]= hyper_param_str+".log"
                    print(executable)
                    #submit_batch_file_uiuc_cc(executable,batch_options)
                    subprocess.call(executable,shell=True)

    if cv_uq:
        psubset_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        for mt in model_tuple:
            for p in psubset_arr:
                if len(mt)==4:
                    nn_val = mt[-1]
                else:
                    nn_val=1
                hyper_param_str = "int_"+str(mt[0])+"_e_"+str(mt[1])+"_tb_"+str(mt[2])+"_p_"+str(p)+"_nn_"+str(nn_val)

                executable = "python CV_generate_ensemble.py -i "+str(mt[0])+" -e "+str(mt[1])+" -t "+str(mt[2]) +" -p "+str(p)+" -n "+str(nn_val)
                batch_options["--job-name"]=hyper_param_str
                batch_options["--output"]= hyper_param_str+".log"
                print(executable)
                #submit_batch_file_uiuc_cc(executable,batch_options)
                subprocess.call(executable,shell=True)

    if relaxation:
        uq_arr = ["mcmc"] #,"cv"]
        twist_angle = np.array([1.08,1.12,1.16,1.2,1.47,1.89,2.88]) #0.88,0.99
        model_tuple = [("TETB","popov")] #[("Classical","None")] #,("TETB","MK")] 
        npartitions = 4
        for mt in model_tuple:
            for u in uq_arr:
                for t in twist_angle:
                    for i in range(npartitions):
                        hyper_param_str = "e_"+str(mt[0])+"_tb_"+str(mt[1])+"_uq_"+u+"_a_"+str(t)+"_i_"+str(i)

                        executable = "python run_uq_propagation.py -e "+str(mt[0])+" -t "+str(mt[1]) + " -a "+str(t) + " -u "+u + " -q relax_atoms -n "+str(npartitions) +" -i "+str(i)
                        #batch_options["-N"]=hyper_param_str
                        batch_options["--output"]= hyper_param_str+".log"
                        
                        print(executable)
                        submit_batch_file_delta(executable,batch_options)
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
        twist_angle = np.array([0.88,0.99,1.08,1.12,1.16,1.2,1.47,1.89,2.88])
        model_tuple = [("TETB","popov")] #("Classical","None")
        for mt in model_tuple:
            for u in uq_arr:
                if u=="cv":
                    npartitions=2
                if u =="mcmc":
                    npartitions = 2
                for t in twist_angle:
                    for i in range(npartitions):
                        hyper_param_str = "e_"+str(mt[0])+"_tb_"+str(mt[1])+"_uq_"+u+"_a_"+str(t)+"_i_"+str(i)

                        executable = "python run_uq_propagation.py -e "+str(mt[0])+" -t "+str(mt[1]) + " -a "+str(t) + " -u "+u+ " -q band_structure"+" -n "+str(npartitions) + " -i "+str(i)
                        batch_options["--job-name"]=hyper_param_str
                        batch_options["--output"]= hyper_param_str+".log"
                        print(executable)
                        submit_batch_file_delta(executable,batch_options)


    


