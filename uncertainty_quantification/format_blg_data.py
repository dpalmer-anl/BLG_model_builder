import numpy as np
import ase.io
import pandas as pd
from BLG_model_builder.geom_tools import *

stacking_ = ["AB","SP","Mid","AA"]
disreg_ = [0 , 0.16667, 0.5, 0.66667]
df = pd.read_csv('../data/qmc.csv') 
atoms_list = []
for i,stacking in enumerate(stacking_):
    dis = disreg_[i]
    d_stack = df.loc[df['stacking'] == stacking, :]
    for j, row in d_stack.iterrows():
        atoms = get_bilayer_atoms(row["d"],dis)
        atoms.info = {"energy":row["energy"]}
        print(atoms.info)
        atoms_list.append(atoms)

ase.io.write("bilayer_graphene_data.xyz",atoms_list,format="extxyz")