import numpy as np 
import subprocess

uq= "mcmc" #"subsamp" #

if uq=="mcmc":
    execs = ["python EMCEE_generate_ensemble.py -m MK -B ",
        "python EMCEE_generate_ensemble.py -m interlayer_LETB -B ",
        "python EMCEE_generate_ensemble.py -m intralayer_LETB_NN_val_1 -B ",
        "python EMCEE_generate_ensemble.py -m intralayer_LETB_NN_val_2 -B ", 
        "python EMCEE_generate_ensemble.py -m intralayer_LETB_NN_val_3 -B "] #, 
        #"python EMCEE_generate_ensemble.py -m MLP_SK_5_1 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_10_1 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_15_1 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_20_1 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_5_2 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_10_2 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_15_2 -B ",
        #"python EMCEE_generate_ensemble.py -m MLP_SK_20_2 -B "] 

    execs = ["python EMCEE_generate_ensemble.py -m MLP_SK_5_1 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_10_1 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_15_1 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_20_1 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_5_2 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_10_2 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_15_2 -B ",
            "python EMCEE_generate_ensemble.py -m MLP_SK_20_2 -B "]
    T_weight = np.array([1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5,1,1.5,2.0,3,4,5,7,10,15,20,30,50])
    for exec in execs:
        for T in T_weight:
            subprocess.call(exec+str(T),shell=True)

if uq=="subsamp":
    execs = ["python SubSamp_generate_ensemble.py -m MK -p ",
        "python SubSamp_generate_ensemble.py -m interlayer_LETB -p ",
        "python SubSamp_generate_ensemble.py -m intralayer_LETB_NN_val_1 -p ",
        "python SubSamp_generate_ensemble.py -m intralayer_LETB_NN_val_2 -p ", 
        "python SubSamp_generate_ensemble.py -m intralayer_LETB_NN_val_3 -p ", 
        "python SubSamp_generate_ensemble.py -m MLP_SK -p "] 

    execs = ["python SubSamp_generate_ensemble.py -m MLP_SK -p "]
    p_subset = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #,1.0])
    for exec in execs:
        for p in p_subset:
            subprocess.call(exec+str(p),shell=True)