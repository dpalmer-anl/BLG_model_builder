"""
DataLoader.py - Centralized data loading and descriptor precomputation for MCMC.

Provides functions to load training data (hoppings and energies) and to
precompute geometry-dependent descriptors so that the MCMC loop avoids
redundant neighbor-list and descriptor computation at each step.

"""

import numpy as np
import ase.db
import ase.io
import pandas as pd
import glob
import h5py
from ase.calculators.singlepoint import SinglePointCalculator
from blg_model_builder.tb_descriptors import *
from blg_model_builder.geom_tools import *
# precompute_pod_descriptors depended on the C++ potential_ext extension (removed).
# It is no longer available.  The MCMC loop now uses PODLammpsCalculator.evaluate_batch.

eV_per_hartree = 27.2114
ang_per_bohr = 0.529
TEST_SIZE = 0.2
np.random.seed(42) #need to be consistent for train/test split

# ---------------------------------------------------------------------------
# Train/test split utility
# ---------------------------------------------------------------------------

def train_test_split(xdata, ydata):
    """Split xdata and ydata into train/test using a single random split.

    Uses one canonical split so that xdata[k][i] and ydata[k][i] stay aligned
    across all keys (essential when forces must match structures).
    """
    xdata_train, xdata_test = {}, {}
    ydata_train, ydata_test = {}, {}

    ydata_keys = list(ydata.keys())
    xdata_keys = list(xdata.keys())

    # Use a single canonical length and split for all keys
    n_total = len(xdata[xdata_keys[0]])
    for k in list(xdata_keys[1:]) + ydata_keys:
        other_len = len(ydata[k]) if k in ydata else len(xdata[k])
        if other_len != n_total:
            raise ValueError(
                f"train_test_split: key '{k}' has length {other_len}, expected {n_total}. "
                "All keys must have the same length for aligned train/test split."
            )

    n_select = int(n_total * TEST_SIZE)
    selected = np.random.choice(n_total, size=n_select, replace=False)
    not_selected = np.setdiff1d(np.arange(n_total), selected)

    for xkey in xdata_keys:
        if isinstance(xdata[xkey], list):
            xdata_train[xkey] = [xdata[xkey][ns] for ns in not_selected]
            xdata_test[xkey] = [xdata[xkey][s] for s in selected]
        else:
            xdata_train[xkey] = xdata[xkey][not_selected]
            xdata_test[xkey] = xdata[xkey][selected]

    for ykey in ydata_keys:
        if isinstance(ydata[ykey], list):
            ydata_train[ykey] = [ydata[ykey][i] for i in not_selected]
            ydata_test[ykey] = [ydata[ykey][i] for i in selected]
        else:
            ydata_train[ykey] = ydata[ykey][not_selected]
            ydata_test[ykey] = ydata[ykey][selected]

    return (xdata_train, xdata_test, ydata_train, ydata_test)

# ---------------------------------------------------------------------------
# Low-level data loaders
# ---------------------------------------------------------------------------

def _as_repeat_tuple(supercells):
    """Normalize supercell spec to a 3-tuple for Atoms.repeat()."""
    if isinstance(supercells, (tuple, list, np.ndarray)):
        if len(supercells) != 3:
            raise ValueError(f"supercells must be an int or a length-3 tuple, got {supercells!r}")
        return tuple(int(x) for x in supercells)
    return (int(supercells), int(supercells), 1)


def _make_supercell_with_tiled_sp(primitive_atoms, supercells):
    """Repeat atoms while scaling E and tiling forces periodically.

    Parameters
    ----------
    primitive_atoms : ase.Atoms
        Must have a single-point energy; forces optional.
    supercells : int or (int, int, int)
        Repeat factors for (a, b, c).

    Returns
    -------
    supercell_atoms : ase.Atoms
        Repeated atoms with a SinglePointCalculator attached.
    scaled_energy : float
    tiled_forces : np.ndarray
        Shape (n_super, 3). Zeros if primitive forces unavailable.
    """
    rep = _as_repeat_tuple(supercells)
    prim = primitive_atoms
    n_prim = len(prim)
    if n_prim == 0:
        raise ValueError("primitive_atoms has zero atoms; cannot build supercell")

    # Extract reference properties from the primitive BEFORE repeating.
    energy_prim = float(prim.get_potential_energy())
    try:
        forces_prim = np.asarray(prim.get_forces(), dtype=float)
    except Exception:
        forces_prim = None

    supercell_atoms = prim.repeat(rep)
    n_super = len(supercell_atoms)
    if n_super % n_prim != 0:
        raise ValueError(
            f"Supercell atom count ({n_super}) not divisible by primitive count ({n_prim}); "
            f"repeat factors were {rep}"
        )
    mult = n_super // n_prim

    scaled_energy = energy_prim * mult
    if forces_prim is None:
        tiled_forces = np.zeros((n_super, 3), dtype=float)
    else:
        if forces_prim.shape != (n_prim, 3):
            raise ValueError(f"Expected primitive forces shape {(n_prim, 3)}, got {forces_prim.shape}")
        # ASE repeat order replicates the atom list in blocks; tiling matches that ordering.
        tiled_forces = np.tile(forces_prim, (mult, 1))

    supercell_atoms.calc = SinglePointCalculator(supercell_atoms, energy=scaled_energy, forces=tiled_forces)
    return supercell_atoms, scaled_energy, tiled_forces


def load_hopping_data(hopping_type="interlayer", units="ang"):
    """Load raw hopping training data from HDF5 files.

    Returns the dict produced by ``hopping_training_data`` with keys
    'hopping', 'atoms', 'i', 'j', 'di', 'dj', 'disp'.
    """
    data = []
    flist = glob.glob('../data/hoppings/*.hdf5',recursive=True)
    eV_per_hart=27.2114
    ang_per_bohr = 0.529
    if units == "bohr":
        conv = 1/ang_per_bohr
    else:
        conv=1
    #hoppings = np.zeros((1,1))
    disp_list = []
    hoppings = []
    atoms_list = []
    i_list = []
    j_list = []
    di_list = []
    dj_list = []
    for f in flist:
        if ".hdf5" in f:
            with h5py.File(f, 'r') as hdf:
                # Unpack hdf
                lattice_vectors = np.array(hdf['lattice_vectors'][:]) * conv
                atomic_basis =    np.array(hdf['atomic_basis'][:])   * conv
                atoms = ase.Atoms("C" * np.shape(atomic_basis)[0], positions=atomic_basis, cell=lattice_vectors)
                # PBC required for POD_TB to find 2nd/3rd NN in-plane hoppings via periodic images
                atoms.pbc = [True, True, False]  # in-plane periodic, out-of-plane non-periodic for bilayer
                mean_z = np.mean(atomic_basis[:,2])
                top_ind = np.where(atomic_basis[:,2]>mean_z)
                mol_id = np.ones(len(atoms),dtype=np.int64)
                mol_id[top_ind] = 2
                atoms.set_array("mol-id",mol_id)
                atoms_list.append(atoms)
                tb_hamiltonian = hdf['tb_hamiltonian']
                tij = np.array(tb_hamiltonian['tij'][:]) #* eV_per_hart
                di  = np.array(tb_hamiltonian['displacementi'][:])
                dj  = np.array(tb_hamiltonian['displacementj'][:])
                ai  = np.array(tb_hamiltonian['atomi'][:])
                aj  = np.array(tb_hamiltonian['atomj'][:])
                displacement_vector = (di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai])*conv

                if hopping_type=="interlayer":
                    type_ind = np.where(mol_id[ai]!=mol_id[aj])
                    hoppings.append(tij[type_ind])
                    i_list.append(ai[type_ind])
                    j_list.append(aj[type_ind])
                    di_list.append(di[type_ind])
                    dj_list.append(dj[type_ind])
                    disp_list.append(np.squeeze(displacement_vector[type_ind,:]))
                elif hopping_type=="intralayer":
                    type_ind = np.where(mol_id[ai]==mol_id[aj])
                    hoppings.append(tij[type_ind])
                    i_list.append(ai[type_ind])
                    j_list.append(aj[type_ind])
                    di_list.append(di[type_ind])
                    dj_list.append(dj[type_ind])
                    disp_list.append(np.squeeze(displacement_vector[type_ind,:]))
                else:
                    hoppings.append(tij)
                    i_list.append(ai)
                    j_list.append(aj)
                    di_list.append(di)
                    dj_list.append(dj)
                    disp_list.append(np.squeeze(displacement_vector))

    return {"hopping":hoppings,"atoms":atoms_list,"i":i_list,"j":j_list,"di":di_list,"dj":dj_list,"disp":disp_list}


def load_energy_data(int_type="interlayer", supercells=1, level_of_theory="rVV10"):
    """Load energy training data (atoms list, energies, forces).

    Parameters
    ----------
    int_type : str
        ``"interlayer"`` or ``"intralayer"``.
    supercells : int or tuple
        Passed to :func:`_make_supercell_with_tiled_sp`.
    level_of_theory : str
        Interlayer / intralayer reference data:

        - ``"rVV10"`` — strained bilayer / monolayer extxyz (default).
        - ``"MBD"`` — bilayer total energies from ``../data/bilayer_graphene_MBD.xyz``.
        - ``"QMC"`` — interlayer only, from ``../data/qmc.csv``.
        - ``"r2SCAN"`` — intralayer only, ASE db.

    Returns
    -------
    atoms_list : list of ase.Atoms
    energies : np.ndarray
    forces : list of ndarray
        One ``(n_atoms, 3)`` array per configuration (zeros if unavailable).
    """

    if int_type == "interlayer":
        if level_of_theory == "rVV10":
            interlayer_atom_list_primitive = ase.io.read(
                "../data/strained_bilayer_graphene_rVV10.xyz",
                format="extxyz",
                index=":",
            )
            interlayer_energies = np.zeros(len(interlayer_atom_list_primitive))
            interlayer_atom_list = []
            interlayer_forces = []
            for i in range(len(interlayer_atom_list_primitive)):
                atoms, E, F = _make_supercell_with_tiled_sp(interlayer_atom_list_primitive[i], supercells)
                z = atoms.positions[:,2]
                mean_z = np.mean(z)
                top_ind = np.where(z > mean_z)[0]
                mol_id = np.ones(len(atoms),dtype=np.int8)
                mol_id[top_ind] = 2
                atoms.set_array("mol-id", mol_id)
                interlayer_atom_list.append(atoms)
                interlayer_energies[i] = E
                interlayer_forces.append(F)

            #interlayer_energies -= np.min(interlayer_energies)
            return interlayer_atom_list, interlayer_energies, interlayer_forces

        elif level_of_theory == "MBD":
            interlayer_atom_list_primitive = ase.io.read(
                "../data/bilayer_graphene_MBD.xyz",
                format="extxyz",
                index=":",
            )
            n_prim = len(interlayer_atom_list_primitive)
            interlayer_energies = np.zeros(n_prim)
            interlayer_atom_list = []
            interlayer_forces = []
            for i in range(n_prim):
                atoms, E, F = _make_supercell_with_tiled_sp(
                    interlayer_atom_list_primitive[i], supercells,
                )
                z = atoms.positions[:, 2]
                mean_z = np.mean(z)
                top_ind = np.where(z > mean_z)[0]
                mol_id = np.ones(len(atoms), dtype=np.int8)
                mol_id[top_ind] = 2
                atoms.set_array("mol-id", mol_id)
                interlayer_atom_list.append(atoms)
                interlayer_energies[i] = E
                interlayer_forces.append(F)
            return interlayer_atom_list, interlayer_energies, interlayer_forces

        elif level_of_theory == "QMC":
            interlayer_df = pd.read_csv("../data/qmc.csv")
            interlayer_atom_list = []
            interlayer_energies = []
            stacking_ = ["AB","SP","Mid","AA"]
            disreg_ = [0 , 0.16667, 0.5, 0.66667]
            
            for i,stacking in enumerate(stacking_):
                dis = disreg_[i]
                d_stack = interlayer_df.loc[interlayer_df['stacking'] == stacking, :]
                for j, row in d_stack.iterrows():
                    atoms = get_bilayer_atoms(row["d"],dis,sc=supercells)
                    pos = atoms.positions
                    mean_z = np.mean(pos[:,2])
                    top_ind = np.where(pos[:,2]>mean_z)
                    bot_ind = np.where(pos[:,2]<mean_z)
                    d = np.mean(np.abs(pos[top_ind,2]-pos[bot_ind,2]))
                    mol_id = np.ones(len(atoms),dtype=np.int64)
                    mol_id[top_ind] = 2
                    atoms.set_array("mol-id",mol_id)

                    top_layer_ind = np.where(pos[:,2]>mean_z)
                    top_pos = np.squeeze(pos[top_layer_ind,:])
                    bot_layer_ind = np.where(pos[:,2]<mean_z)
                    bot_pos = np.squeeze(pos[bot_layer_ind,:])

                    interlayer_atom_list.append(atoms)
                    interlayer_energies.append(row["energy"]*len(atoms))
            interlayer_energies = np.array(interlayer_energies)
            interlayer_energies -= np.min(interlayer_energies)
            # QMC reference has no forces; use zeros with correct shape.
            n_cfg = len(interlayer_atom_list)
            n_atoms = len(interlayer_atom_list[0]) if n_cfg > 0 else 0
            interlayer_forces = np.zeros((n_cfg, n_atoms, 3))
            return interlayer_atom_list, interlayer_energies, interlayer_forces

        raise ValueError(
            f"interlayer level_of_theory must be 'rVV10', 'MBD', or 'QMC'; got {level_of_theory!r}",
        )

    elif int_type == "intralayer":
        if level_of_theory == "rVV10":
            intralayer_atom_list_primitive = ase.io.read(
                "../data/strained_monolayer_graphene_rVV10.xyz",
                format="extxyz",
                index=":",
            )
            intralayer_atom_list = []
            intralayer_energies = np.zeros(len(intralayer_atom_list_primitive))
            intralayer_forces = []
            for i in range(len(intralayer_atom_list_primitive)):
                atoms, E, F = _make_supercell_with_tiled_sp(intralayer_atom_list_primitive[i], supercells)
                mol_id = np.ones(len(atoms),dtype=np.int8)
                atoms.set_array("mol-id", mol_id)
                intralayer_atom_list.append(atoms)
                intralayer_energies[i] = E
                intralayer_forces.append(F)
            intralayer_energies -= np.min(intralayer_energies)
            return intralayer_atom_list, intralayer_energies, intralayer_forces

        if level_of_theory == "MBD":
            raise ValueError(
                "intralayer MBD is not implemented: add a dataset and branch in load_energy_data.",
            )

        elif level_of_theory == "r2SCAN":
            intralayer_db = ase.db.connect("../data/monolayer_nkp121.db")
            intralayer_atom_list = []
            intralayer_energies = []
            for i,row in enumerate(intralayer_db.select()):
                atoms = intralayer_db.get_atoms(id = row.id) * (supercells,supercells,1)
                atoms.set_array("mol-id",np.ones(len(atoms),dtype=np.int64))

                intralayer_atom_list.append(atoms)
                intralayer_energies.append(row.data.total_energy*len(atoms))
            n_cfg = len(intralayer_atom_list)
            n_atoms = len(intralayer_atom_list[0]) if n_cfg > 0 else 0
            intralayer_forces = np.zeros((n_cfg, n_atoms, 3))

            return intralayer_atom_list, intralayer_energies, intralayer_forces

        raise ValueError(
            f"intralayer level_of_theory must be 'rVV10' or 'r2SCAN'; got {level_of_theory!r}",
        )
    else:
        raise ValueError(f"int_type must be 'interlayer' or 'intralayer', got '{int_type}'")


def _concat_hopping_data(hopping_data):
    """Stack displacement and hopping arrays from all configurations."""
    xdata_list = hopping_data["disp"]
    ydata_list = hopping_data["hopping"]
    xdata = xdata_list[0]
    ydata = ydata_list[0]
    for i in range(1, len(xdata_list)):
        xdata = np.vstack((xdata, xdata_list[i]))
        ydata = np.append(ydata, ydata_list[i])
    return xdata, ydata


def _displacement_to_di_dj(disp, pos_i, pos_j, cell):
    """Recover (di, dj) from displacement vector: disp = di*cell[0] + dj*cell[1] + (pos_j - pos_i)."""
    residual = disp - (pos_j - pos_i)
    A = np.column_stack([cell[0], cell[1]])
    di_dj, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)
    return int(round(di_dj[0])), int(round(di_dj[1]))


def _ydata_hopping_aligned_to_acsf_order(
    atoms,
    pair_i,
    pair_j,
    pair_v,
    ref_i,
    ref_j,
    ref_di,
    ref_dj,
    ref_tij,
):
    """Map reference TB hoppings to ACSF descriptor row order.

    For each row ``r`` of ``get_acsf_hopping_descriptors``, recover ``(di, dj)``
    from the bond vector ``pair_v[r]`` and look up
    ``tij`` for key ``(pair_i[r], pair_j[r], di, dj)``, matching ``load_hopping_data``.

    Returns
    -------
    y_row : ndarray, shape (n_pairs,)
        ``tij`` where a matching reference pair exists, else ``nan``.
    mask : ndarray, dtype bool
        ``True`` where a reference hop was found (finite ``y_row``).
    """
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pair_i = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pair_j = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    pair_v = np.asarray(pair_v, dtype=float).reshape(-1, 3)

    ref_i = np.asarray(ref_i, dtype=np.int64).reshape(-1)
    ref_j = np.asarray(ref_j, dtype=np.int64).reshape(-1)
    ref_di = np.asarray(ref_di, dtype=np.int64).reshape(-1)
    ref_dj = np.asarray(ref_dj, dtype=np.int64).reshape(-1)
    ref_tij = np.asarray(ref_tij, dtype=float).reshape(-1)

    ref_map = {}
    for p in range(ref_i.size):
        key = (int(ref_i[p]), int(ref_j[p]), int(ref_di[p]), int(ref_dj[p]))
        ref_map[key] = float(ref_tij[p])

    n_pairs = pair_i.size
    y_row = np.full(n_pairs, np.nan, dtype=np.float64)
    for r in range(n_pairs):
        di, dj = _displacement_to_di_dj(
            pair_v[r],
            pos[pair_i[r]],
            pos[pair_j[r]],
            cell,
        )
        key = (int(pair_i[r]), int(pair_j[r]), di, dj)
        if key in ref_map:
            y_row[r] = ref_map[key]
    mask = np.isfinite(y_row)
    return y_row, mask

# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def load_data_for_model(model_name, supercells=1, nn_val=None, level_of_theory="rVV10", **kwargs):
    """Load (xdata, ydata, ydata_noise) dicts for a given model name.

    Parameters
    ----------
    model_name : str
        One of the recognized model keys (see below).
    supercells : int
        Supercell multiplier for energy datasets.
    nn_val : int or None
        Nearest-neighbor shell index for LETB intralayer models.
    level_of_theory : str
        Energy reference: ``"rVV10"`` (default), ``"MBD"`` (bilayer MBD extxyz),
        ``"QMC"``, or ``"r2SCAN"`` (intralayer only) as supported by
        :func:`load_energy_data`.
    **kwargs : dict
        Passed through; for ``model_name == "ACSF_hoppings"`` use optional
        ``acsf_M``, ``acsf_W`` (or ``M``, ``W``), ``acsf_r_cut`` (or ``r_cut``),
        ``acsf_use_envelope`` (or ``use_envelope``). For ``TETB_POD``, use
        ``tb_M`` / ``tb_W`` (same fallbacks as above) for ACSF hopping descriptors;
        POD radial/angular counts use ``pod_M`` / ``pod_W`` (see ``get_MCMC_inputs``).

    Returns
    -------
    xdata, ydata : dict
        Dicts with keys ``"hoppings"`` and/or ``"energy"`` and ``"forces"``.
    """
    xdata, ydata = {}, {}

    # ----- Tight-binding hopping data -----
    if model_name == "MK":
        hopping_data = load_hopping_data(hopping_type="all")
        xdata["hopping"], ydata["hopping"] = _concat_hopping_data(hopping_data)
        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name.startswith("ACSF_hoppings_sk"):
        # SK variant: SK physics is baked into the descriptors via
        # get_acsf_sk_hopping_descriptors, so xdata["hopping"] has shape
        # (n_pairs, 2*n_feat) and get_acsf_hoppings_sk is just descriptors @ params.
        hopping_data = load_hopping_data(hopping_type="all")
        M = int(kwargs.get("acsf_M", kwargs.get("M", 10)))
        W = int(kwargs.get("acsf_W", kwargs.get("W", 3)))
        r_cut = float(kwargs.get("acsf_r_cut", kwargs.get("r_cut", 6.0)))
        use_envelope = bool(
            kwargs.get("acsf_use_envelope", kwargs.get("use_envelope", True))
        )
        xdata["hopping"] = []
        xdata["hopping_dist"] = []
        ydata["hopping"] = []
        for k, atoms in enumerate(hopping_data["atoms"]):
            dsc_sk, (pair_i, pair_j, pair_v) = get_acsf_sk_hopping_descriptors(
                atoms, M=M, W=W, r_cut=r_cut, use_envelope=use_envelope
            )
            y_row, mask = _ydata_hopping_aligned_to_acsf_order(
                atoms,
                pair_i,
                pair_j,
                pair_v,
                hopping_data["i"][k],
                hopping_data["j"][k],
                hopping_data["di"][k],
                hopping_data["dj"][k],
                hopping_data["hopping"][k],
            )
            pv = np.asarray(pair_v, dtype=float)
            dist = np.linalg.norm(pv[mask], axis=1)
            xdata["hopping"].append(dsc_sk[mask])
            xdata["hopping_dist"].append(dist)
            ydata["hopping"].append(y_row[mask])

        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name.startswith("ACSF_hoppings"):
        hopping_data = load_hopping_data(hopping_type="all")
        M = int(kwargs.get("acsf_M", kwargs.get("M", 10)))
        W = int(kwargs.get("acsf_W", kwargs.get("W", 3)))
        r_cut = float(kwargs.get("acsf_r_cut", kwargs.get("r_cut", 6.0)))
        use_envelope = bool(
            kwargs.get("acsf_use_envelope", kwargs.get("use_envelope", True))
        )
        xdata["hopping"] = []
        xdata["hopping_dist"] = []
        xdata["hopping_disp"] = []
        ydata["hopping"] = []
        for k, atoms in enumerate(hopping_data["atoms"]):
            dsc, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
                atoms, M=M, W=W, r_cut=r_cut, use_envelope=use_envelope
            )
            y_row, mask = _ydata_hopping_aligned_to_acsf_order(
                atoms,
                pair_i,
                pair_j,
                pair_v,
                hopping_data["i"][k],
                hopping_data["j"][k],
                hopping_data["di"][k],
                hopping_data["dj"][k],
                hopping_data["hopping"][k],
            )
            pv = np.asarray(pair_v, dtype=float)
            dist = np.linalg.norm(pv[mask], axis=1)
            xdata["hopping"].append(dsc[mask])
            xdata["hopping_dist"].append(dist)
            xdata["hopping_disp"].append(pv[mask])
            ydata["hopping"].append(y_row[mask])

        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name == "LETB_interlayer":
        hopping_data = load_hopping_data(hopping_type="interlayer")
        atoms_list = hopping_data["atoms"]
        i_list, j_list = hopping_data["i"], hopping_data["j"]
        di_list, dj_list = hopping_data["di"], hopping_data["dj"]
        disp_list = hopping_data["disp"]

        cell = atoms_list[0].get_cell()
        pos = atoms_list[0].positions
        dsc = letb_interlayer_descriptors_array(
            cell, disp_list[0], pos, di_list[0], dj_list[0], i_list[0], j_list[0]
        )
        xdata["hopping"] = dsc
        ydata["hopping"] = hopping_data["hopping"][0]
        for k in range(1, len(atoms_list)):
            cell = atoms_list[k].get_cell()
            pos = atoms_list[k].positions
            dsc = letb_interlayer_descriptors_array(
                cell, disp_list[k], pos, di_list[k], dj_list[k], i_list[k], j_list[k]
            )
            xdata["hopping"] = np.vstack((xdata["hopping"], dsc))
            ydata["hopping"] = np.concatenate((ydata["hopping"], hopping_data["hopping"][k]))
        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name.startswith("LETB_intralayer"):
        if nn_val is None:
            raise ValueError("nn_val must be specified for LETB intralayer models")
        hopping_data = load_hopping_data(hopping_type="intralayer")
        atoms_list = hopping_data["atoms"]
        i_list, j_list = hopping_data["i"], hopping_data["j"]
        di_list, dj_list = hopping_data["di"], hopping_data["dj"]
        disp_list = hopping_data["disp"]

        cell = atoms_list[0].get_cell()
        pos = atoms_list[0].positions
        dsc, ix = letb_intralayer_descriptors_array(
            cell, disp_list[0], pos, di_list[0], dj_list[0], i_list[0], j_list[0],
            nn_val=nn_val,
        )
        xdata["hopping"] = dsc
        ydata["hopping"] = hopping_data["hopping"][0][ix]
        for k in range(1, len(atoms_list)):
            cell = atoms_list[k].get_cell()
            pos = atoms_list[k].positions
            dsc, ix = letb_intralayer_descriptors_array(
                cell, disp_list[k], pos, di_list[k], dj_list[k], i_list[k], j_list[k],
                nn_val=nn_val,
            )
            if dsc.ndim < 2:
                xdata["hopping"] = np.append(xdata["hopping"], dsc)
            else:
                xdata["hopping"] = np.vstack((xdata["hopping"], dsc))
            ydata["hopping"] = np.append(ydata["hopping"], hopping_data["hopping"][k][ix])
        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    # ----- Energy data -----
    elif model_name.startswith("POD_energy"):
        # Combined interlayer + intralayer for single POD_NN model.
        # Forces: list of (n_atoms_i, 3) per config (n_atoms can differ between configs).
        inter_atoms, inter_E, inter_F = load_energy_data(
            "interlayer", supercells, level_of_theory=level_of_theory,
        )
        atoms_list = inter_atoms
        energies = inter_E
        inter_F_list = [np.asarray(inter_F[i]) for i in range(len(inter_atoms))]
        forces = inter_F_list
        xdata["energy"] = atoms_list
        natoms_list = [len(atoms) for atoms in atoms_list]
        ydata["energy"] = energies
        ydata["forces"] = forces
        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name in ("Kolmogorov_Crespi", "DRIP"):
        atoms_list, energies, forces = load_energy_data("interlayer", supercells, level_of_theory=level_of_theory)
        xdata["energy"] = atoms_list
        ydata["energy"] = energies
        ydata["forces"] = forces
        if level_of_theory == "QMC":
            #not enough data for train/test split
            xdata_train = xdata["energy"]
            xdata_test = xdata["energy"]
            ydata_train = ydata["energy"]
            ydata_test = ydata["energy"]
        else:
            xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name == "Tersoff+Kolmogorov_Crespi" or model_name == "Tersoff+DRIP":
        inter_atoms, inter_E, inter_F = load_energy_data("interlayer", 2,level_of_theory=level_of_theory)
        intra_atoms, intra_E, intra_F = load_energy_data(
            "intralayer", supercells, level_of_theory=level_of_theory,
        )
        atoms_list = inter_atoms #+ intra_atoms
        energies = inter_E #np.concatenate([inter_E, intra_E])
        inter_F_list = [np.asarray(inter_F[i]) for i in range(len(inter_atoms))]
        #intra_F_list = [np.asarray(intra_F[i]) for i in range(len(intra_atoms))]
        forces = inter_F_list #+ intra_F_list
        #E_min = np.min(energies)
        #energies = energies - E_min
        xdata["energy"] = atoms_list
        ydata["energy"] = energies
        ydata["forces"] = forces
        if level_of_theory == "QMC":
            #not enough data for train/test split
            xdata_train = xdata["energy"]
            xdata_test = xdata["energy"]
            ydata_train = ydata["energy"]
            ydata_test = ydata["energy"]
        else:
            xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name == "Tersoff":
        atoms_list, energies, forces = load_energy_data("intralayer", supercells, level_of_theory=level_of_theory)
        xdata["energy"] = atoms_list
        ydata["energy"] = energies
        ydata["forces"] = forces
        xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata)

    elif model_name.startswith("TETB_POD"):
        # Hopping (pair samples / structures) and energy (interlayer configs) come
        # from different datasets with different lengths.  ``train_test_split``
        # requires every key in *xdata* to have the same length, so we split each
        # modality independently and merge the train/test dicts (same pattern as
        # if hopping and energy were unrelated tables).
        #
        # Hopping *x* must be ACSF descriptors (same as ``ACSF_hoppings``), not raw
        # displacement vectors from ``_concat_hopping_data`` (which are 3-D).
        # Use ``tb_M`` / ``tb_W`` (or ``acsf_M`` / ``acsf_W``, then top-level ``M`` /
        # ``W``) so POD basis size can differ from the TB descriptor width.
        hopping_data = load_hopping_data(hopping_type="all")
        M_tb = int(kwargs.get("tb_M", kwargs.get("acsf_M", kwargs.get("M", 10))))
        W_tb = int(kwargs.get("tb_W", kwargs.get("acsf_W", kwargs.get("W", 3))))
        r_cut = float(kwargs.get("acsf_r_cut", kwargs.get("r_cut", 6.0)))
        use_envelope = bool(
            kwargs.get("acsf_use_envelope", kwargs.get("use_envelope", True))
        )
        xdata_hop = {"hopping": [], "hopping_dist": []}
        ydata_hop = {"hopping": []}
        for k, atoms in enumerate(hopping_data["atoms"]):
            dsc, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
                atoms, M=M_tb, W=W_tb, r_cut=r_cut, use_envelope=use_envelope
            )
            y_row, mask = _ydata_hopping_aligned_to_acsf_order(
                atoms,
                pair_i,
                pair_j,
                pair_v,
                hopping_data["i"][k],
                hopping_data["j"][k],
                hopping_data["di"][k],
                hopping_data["dj"][k],
                hopping_data["hopping"][k],
            )
            pv = np.asarray(pair_v, dtype=float)
            dist = np.linalg.norm(pv[mask], axis=1)
            xdata_hop["hopping"].append(dsc[mask])
            xdata_hop["hopping_dist"].append(dist)
            ydata_hop["hopping"].append(y_row[mask])
        xdata_train_h, xdata_test_h, ydata_train_h, ydata_test_h = train_test_split(
            xdata_hop, ydata_hop
        )

        inter_atoms, inter_E, inter_F = load_energy_data(
            "interlayer", supercells, level_of_theory=level_of_theory,
        )
        xdata_e, ydata_e = {}, {}
        xdata_e["energy"] = inter_atoms
        ydata_e["energy"] = inter_E
        ydata_e["forces"] = [np.asarray(inter_F[i]) for i in range(len(inter_atoms))]
        xdata_train_e, xdata_test_e, ydata_train_e, ydata_test_e = train_test_split(
            xdata_e, ydata_e
        )

        xdata = {**xdata_hop, **xdata_e}
        ydata = {**ydata_hop, **ydata_e}
        xdata_train = {**xdata_train_h, **xdata_train_e}
        xdata_test = {**xdata_test_h, **xdata_test_e}
        ydata_train = {**ydata_train_h, **ydata_train_e}
        ydata_test = {**ydata_test_h, **ydata_test_e}
    else:
        raise ValueError(f"Unknown model_name '{model_name}' for data loading")

    return xdata_train, xdata_test, xdata, ydata_train, ydata_test, ydata
