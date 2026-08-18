import numpy as np
import flatgraphene as fg
from scipy.spatial import distance
from ase.build import make_supercell
import pandas as pd
import ase
import matplotlib.pyplot as plt

from blg_model_builder.strain_data import LAT_CON

def get_monolayer_atoms(dx,dy,a=2.462,sc=4):
    atoms=fg.shift.make_layer("A","rect",sc,sc,a,7.0,"B",12.01,1)
    curr_cell=atoms.get_cell()
    atoms.set_array('mol-id',np.ones(len(atoms),dtype=np.int8))
    curr_cell[-1,-1]=14
    atoms.set_cell(curr_cell)
    return ase.Atoms(atoms) 

def get_basis(a, d, c, disregistry, zshift='CM'):

    '''
    `disregistry` is defined such that the distance to disregister from AB to AB again is 1.0,
    which corresponds to 3*bond_length = 3/sqrt(3)*lattice_constant = sqrt(3)*lattice_constant
    so we convert the given `disregistry` to angstrom
    '''
    disregistry_ang = 3**0.5*a*disregistry
    orig_basis = np.array([
        [0, 0, 0],
        [0, a/3**0.5, 0],
        [0, a/3**0.5 + disregistry_ang, d],
        [a/2, a/(2*3**0.5) + disregistry_ang, d]
        ])

    # for open boundary condition in the z-direction
    # move the first layer to the middle of the cell
    if zshift == 'first_layer':
        z = c/2
    # or move the center of mass to the middle of the cell
    elif zshift == 'CM':
        z = c/2 - d/2
    shift_vector = np.array([0, 0, z])
    shifted_basis = orig_basis + shift_vector
    return shifted_basis.tolist()

def get_lattice_vectors(a, c):
    return [
        [a, 0, 0],
        [1/2*a, 1/2*3**0.5*a, 0],
        [0, 0, c]
        ]

def get_bilayer_atoms(d,disregistry, a=LAT_CON, c=20, sc=1,zshift='CM'):
    '''All units should be in angstroms'''
    symbols = ["C","C","C","C"]
    atoms = ase.Atoms(
        symbols=symbols,
        positions=get_basis(a, d, c, disregistry, zshift=zshift),
        cell=get_lattice_vectors(a, c),
        pbc=[1, 1, 1],
        tags=[0, 0, 1, 1],
        )
    atoms.set_array("mol-id",np.array([1,1,2,2],dtype=np.int8))  
    atoms = make_supercell(atoms, [[sc, 0, 0], [0, sc, 0], [0, 0, 1]])
    return atoms


def get_aa_bilayer_atoms(d, a=LAT_CON, c=20, sc=1, zshift="CM"):
    """Primitive AA-stacked bilayer graphene (same ``(x,y)`` in both layers).

    Unlike :func:`get_bilayer_atoms` with ``disregistry != 0``, AA here means each
    top-layer carbon sits directly above a bottom-layer carbon with **no** in-plane
    registry shift.  Vertical spacing between mean layer heights is ``d``; the
    same ``zshift`` convention as :func:`get_basis` applies (default ``'CM'``).

    Parameters
    ----------
    d : float
        Interlayer separation (Å), i.e. difference in mean ``z`` between layers.
    a, c, sc, zshift
        Same meaning as in :func:`get_bilayer_atoms`.
    """
    y1 = float(a / 3.0 ** 0.5)
    if zshift == "first_layer":
        z0 = float(c) / 2.0
    else:
        # ``'CM'`` (default) and any other value follow :func:`get_basis`.
        z0 = float(c) / 2.0 - float(d) / 2.0
    positions = [
        [0.0, 0.0, z0],
        [0.0, y1, z0],
        [0.0, 0.0, z0 + float(d)],
        [0.0, y1, z0 + float(d)],
    ]
    atoms = ase.Atoms(
        symbols=["C", "C", "C", "C"],
        positions=positions,
        cell=get_lattice_vectors(a, c),
        pbc=[1, 1, 1],
        tags=[0, 0, 1, 1],
    )
    atoms.set_array("mol-id", np.array([1, 1, 2, 2], dtype=np.int8))
    atoms = make_supercell(atoms, [[sc, 0, 0], [0, sc, 0], [0, 0, 1]])
    return atoms
