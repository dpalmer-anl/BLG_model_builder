import subprocess
from datetime import datetime
import numpy as np
def submit_batch_file_perlmutter(executable,batch_options,
                                 conda_env='$PSCRATCH/mypythonev'):

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

        f.write("\nsource activate "+conda_env+"\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)

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

        f.write("\nsource activate "+conda_env+"\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)


if __name__=="__main__":
    tb_uq = False
    te_uq = False
    cv_uq = False
    relax = True
    band_prod = False

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
            "modules":['anaconda3_gpu/23.9.0']}
    batch_options = batch_options_uiuc_cc

    if te_uq:
        use_model_list = ["Classical_energy"] #"TB_sk", "TETB"
        Temperature = [0.1,0.25,0.5,1.0,1.5,2,4,6,8,10,20,30,40,50]
        model_type_list = ["interlayer","intralayer"]
        uq_type_list = ["mcmc"] #,"bootstrap"]
        for u in use_model_list:
            for m in model_type_list:
                for q in uq_type_list:
                    for T in Temperature:
                        hyper_param_str = u+"_"+m+"_"+q
                        executable = "python run_BLG_uq.py -u "+u+" -m "+m+" -q "+q+" -T "+str(T)
                        batch_options["--job-name"]=hyper_param_str
                        batch_options["--output"]= hyper_param_str+".log"
                        #print(executable)
                        submit_batch_file_uiuc_cc(executable,batch_options)

    if tb_uq:
    
        bond_ints = ['popov_hopping_pp_sigma','popov_hopping_pp_pi','popov_overlap_pp_pi','popov_overlap_pp_sigma',
        'porezag_hopping_pp_sigma','porezag_hopping_pp_pi','porezag_overlap_pp_pi','porezag_overlap_pp_sigma']
        model_type_list = ["interlayer","interlayer","interlayer","interlayer",
                            "intralayer","intralayer","intralayer","intralayer"]

        for i,b in enumerate(bond_ints):
            hyper_param_str = "TB_sk"+"_"+"mcmc"
            executable = "python run_BLG_uq.py -u TB_sk -m "+model_type_list[i]+" -q mcmc -b"+b
            subprocess.call(executable,shell=True)

    if tetb_uq:
        Temperature = [0.1,0.25,0.5,1.0,1.5,2,4,6,8,10,20,30,40,50]
        use_model = "TETB"
        model_type_list = ["interlayer","intralayer"]
        uq_type_list = ["mcmc"] #,"bootstrap"]
        for m in model_type_list:
            for q in uq_type_list:
                for T in Temperature:
                    hyper_param_str = use_model+"_"+m+"_"+q
                    executable = "python run_BLG_uq.py -u "+use_model+" -m "+m+" -q "+q+" -T "+str(T)
                    batch_options["--job-name"]=hyper_param_str
                    batch_options["--output"]= hyper_param_str+".log"
                    #print(executable)
                    submit_batch_file_uiuc_cc(executable,batch_options)

    if cv_uq:
        use_model_list = ["Classical_energy", "TETB"]
        model_type_list = ["interlayer","intralayer"]
        uq_type_list = ["Kfold"]
        n_split_list = [2,3,4,5,6,7,8]
        for u in use_model_list:
            for m in model_type_list:
                for q in uq_type_list:
                    for n in n_split_list:
                        hyper_param_str = u+"_"+m+"_"+q+"_nsplit_"+str(n)
                        executable = "python run_BLG_uq.py -u "+u+" -m "+m+" -q "+q+" -n "+str(n)
                        batch_options["--job-name"]=hyper_param_str
                        batch_options["--output"]= hyper_param_str+".log"
                        #print(executable)
                        submit_batch_file_uiuc_cc(executable,batch_options)

    if relax:
        energy_model = "Classical" #"TB_sk", "TETB"

        theta = [1.12,1.16] #np.array([0.88,0.99,1.08,1.2,1.47,1.89,2.88])
        for t in theta:
            hyper_param_str = "_a_"+str(t)+"_e_"+energy_model
            executable = "python ensemble_qoi_production.py -a "+str(t)+" -q relax_atoms -e "+energy_model
            batch_options["--job-name"]=hyper_param_str
            batch_options["--output"]= hyper_param_str+".log"
            #print(executable)
            submit_batch_file_delta(executable,batch_options)

    if band_prod:
        energy_model = "Classical" #"TB_sk", "TETB"
        tb_model = "MK"
        n_partitions = 10
        config_ind = []
        theta = np.array([0.88,0.99,1.08,1.2,1.47,1.89,2.88])


        for t in theta:
            for n in range(n_partitions):
                hyper_param_str = str(n)+" _n_"+str(n_partitions)+"_a_"+str(t)+"_q_band_structure_t_"+tb_model+"_e_"+energy_model
                executable = "python ensemble_qoi_production.py -i "+str(n)+" -n "+str(n_partitions)+" -a "+str(t)+" -q band_structure -t "+tb_model+" -e "+energy_model
                batch_options["--job-name"]=hyper_param_str
                batch_options["--output"]= hyper_param_str+".log"
                #print(executable)
                submit_batch_file_delta(executable,batch_options)

