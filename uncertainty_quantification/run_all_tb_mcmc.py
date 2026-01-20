import numpy as np 
import subprocess

execs = ["python EMCEE_generate_ensemble.py -i full -e None -t MK -B ",
        "python EMCEE_generate_ensemble.py -i interlayer -e None -t LETB -B ",
        "python EMCEE_generate_ensemble.py -i intralayer -e None -t LETB -n 1 -B ",
        "python EMCEE_generate_ensemble.py -i intralayer -e None -t LETB -n 2 -B ", 
        "python EMCEE_generate_ensemble.py -i intralayer -e None -t LETB -n 3 -B "] 

execs = ["python EMCEE_generate_ensemble.py -m MLP_SK -B "]

T_weight = np.array([1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5,1,1.5,2.0,3,4,5,7,10,15,20,30,50])
for exec in execs:
    for T in T_weight:
        subprocess.call(exec+str(T),shell=True)