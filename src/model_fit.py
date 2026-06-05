from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
import time
import pickle
import subprocess
import ase.io
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ase.calculators.singlepoint import SinglePointCalculator

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data


DTYPE = torch.float64


def fit_acsf_linear_hopping(
    xdata,
    ydata,
    *,
    ridge: float = 0.1,
    rcond: float | None = None,
):
    """Fit ``t_ij = X_ij @ w`` by (ridge) least squares for ACSF descriptor rows.

    ``get_acsf_hoppings(descriptors, w)`` is linear in ``w``; this stacks all
    training pairs and solves the normal equations or dense ``lstsq``.

    Parameters
    ----------
    xdata : list of ndarray
        ACSF matrices per structure, each ``(n_k, n_features)``.
    ydata : list of ndarray
        Target hoppings per structure, each ``(n_k,)`` aligned with ``xdata[k]``.
    ridge : float, optional
        Tikhonov regularization ``λ``. If ``λ > 0``, solves
        ``(XᵀX + λ I) w = Xᵀ y``. Use a small value (e.g. ``1e-8``) if the
        design matrix is rank-deficient.
    rcond : float, optional
        Cutoff for small singular values in ``numpy.linalg.lstsq`` when
        ``ridge == 0``. Default ``1e-12``.

    Returns
    -------
    params : ndarray, shape (n_features,)
        Weights ``w`` for ``get_acsf_hoppings``.
    ypred_bestfit : list of ndarray
        Predictions ``X_k @ w`` for each training structure (same lengths as ``ydata``).
    """
    if not isinstance(xdata, list) or not isinstance(ydata, list):
        raise TypeError("fit_acsf_linear_hopping expects list-valued xdata and ydata")
    if len(xdata) != len(ydata):
        raise ValueError("xdata and ydata must have the same number of structures")
    if len(xdata) == 0:
        raise ValueError("no training structures for ACSF linear fit")

    desc_list = [np.asarray(d, dtype=np.float64) for d in xdata]
    y_list = [np.asarray(y, dtype=np.float64) for y in ydata]

    n_feat = None
    for k, (Xk, yk) in enumerate(zip(desc_list, y_list)):
        if Xk.ndim != 2:
            raise ValueError(f"xdata[{k}] must be 2-D, got shape {Xk.shape}")
        if yk.ndim != 1:
            yk = yk.reshape(-1)
        if Xk.shape[0] != yk.shape[0]:
            raise ValueError(
                f"xdata[{k}] has {Xk.shape[0]} rows but ydata[{k}] has length {yk.shape[0]}"
            )
        if n_feat is None:
            n_feat = Xk.shape[1]
        elif Xk.shape[1] != n_feat:
            raise ValueError(
                f"Inconsistent n_features: got {Xk.shape[1]} at index {k}, expected {n_feat}"
            )

    if n_feat is None or n_feat == 0:
        raise ValueError("descriptor dimension is zero")

    X_flat = np.vstack(desc_list)
    y_flat = np.concatenate([y.reshape(-1) for y in y_list])

    if X_flat.shape[0] < n_feat:
        print(
            f"warning: ACSF linear fit has {X_flat.shape[0]} rows and {n_feat} features (underdetermined)"
        )

    print(f"fitting ACSF linear model with {n_feat} parameters ({X_flat.shape[0]} pair samples)")

    if ridge > 0.0:
        xtx = X_flat.T @ X_flat
        rhs = X_flat.T @ y_flat
        w = np.linalg.solve(xtx + ridge * np.eye(n_feat, dtype=np.float64), rhs)
    else:
        rc = 1e-12 if rcond is None else rcond
        w, _, rank, _ = np.linalg.lstsq(X_flat, y_flat, rcond=rc)
        
        if rank < n_feat:
            print(f"warning: design matrix rank {rank} < n_features {n_feat}")

    ypred_bestfit = [Xk @ w for Xk in desc_list]

    return np.asarray(w, dtype=np.float64), ypred_bestfit




def expand_ypred_to_ydata_shape(ypred_filtered, ydata_train, kept_indices, config_indices):
    """Expand ypred (from filtered fit) to match ydata_train shape.

    Pairs without descriptors get np.nan. Same structure as ydata_train.
    config_indices[i] = original config index for ypred_filtered[i].
    """
    ypred_expanded = []
    for k in range(len(ydata_train)):
        n_pairs = len(ydata_train[k])
        y_full = np.full(n_pairs, np.nan, dtype=np.float64)
        # Find which filtered entry corresponds to this config
        for i, cfg_idx in enumerate(config_indices):
            if cfg_idx == k:
                idx_kept = kept_indices[i]
                y_full[idx_kept] = ypred_filtered[i]
                break
        ypred_expanded.append(y_full)
    return ypred_expanded

def fit_pod(
    hyperparams_str,
    atoms_list,
    lammps_exec="/mnt/c/Users/Daniel/Documents/research/lammps/build/lmp",
    *,
    regularization: float = 1e-12,
    weight_energy: float = 1000.0,
    weight_force: float = 1.0,
):
    """Fit a POD potential via LAMMPS ``fitpod``.

    Parameters
    ----------
    hyperparams_str : str
        Content of the ``C_param.pod`` descriptor file.
    atoms_list : list of ase.Atoms
        Training structures with energies / forces in a
        ``SinglePointCalculator``.
    lammps_exec : str
        Path to the LAMMPS executable.
    regularization : float, optional
        L2 Tikhonov regularization for the LAMMPS ``fitpod`` least-squares
        solve.  Larger values constrain coefficient magnitudes more tightly,
        which improves stability on out-of-distribution (OOD) geometries
        (e.g. twisted bilayer graphene) at the cost of slightly worse
        in-distribution accuracy.  Typical range: ``1e-12`` (essentially
        unregularized) to ``1e-2`` (aggressive).  Default ``1e-12`` preserves
        the original behaviour; ``1e-4`` is a good starting point for
        production models that must evaluate OOD structures.
    weight_energy : float, optional
        Relative weight of the energy residuals in the least-squares fit
        (``fitting_weight_energy`` in the LAMMPS data file).  Default 1000.
    weight_force : float, optional
        Relative weight of the force residuals (``fitting_weight_force``).
        Default 1.
    """
    original_dir = os.getcwd()
    os.mkdir("tmp_pod_fit")
    os.chdir("tmp_pod_fit")
    try:
        os.mkdir("TrainingData")
        os.mkdir("TestData")
        ase.io.write("TrainingData/C_data.xyz",atoms_list, format="extxyz")
        ase.io.write("TestData/C_data.xyz",atoms_list, format="extxyz")
        with open("C_param.pod", "w") as f:
            f.write(hyperparams_str)

        with open("C_data.pod", "w") as f:
            f.write(
                f"file_format extxyz\n"
                f"                    file_extension xyz\n"
                f"                    path_to_training_data_set 'TrainingData/'\n"
                f"                    path_to_test_data_set 'TestData/'\n"
                f"                    fitting_weight_energy {weight_energy!r}\n"
                f"                    fitting_weight_force {weight_force!r}\n"
                f"                    fitting_regularization_parameter {regularization!r}\n"
                f"                    error_analysis_for_training_data_set 1\n"
                f"                    error_analysis_for_test_data_set 0\n"
                f"                    basename_for_output_files C\n"
                f"                    precision_for_pod_coefficients 12"
            )

        with open("fit.pod", "w") as f:
            f.write("units metal\n\
                    fitpod C_param.pod C_data.pod")

        subprocess.call(lammps_exec+" -in fit.pod",shell=True)
        best_fit_params = np.loadtxt("C_coefficients.pod",skiprows=1)
    finally:
        os.chdir(original_dir)
        import shutil
        if os.path.isdir("tmp_pod_fit"):
            shutil.rmtree("tmp_pod_fit")
    return best_fit_params


def fit_tetb_residual_pod(
    atoms_list: List[Any],
    e_dft: Sequence[float],
    f_dft: List[Any],
    *,
    M: int,
    W: int,
    r_cut: float,
    pod_hyperparams: Dict[str, Any],
    pod_cutoff: float = 5.0,
    kpoints=None,
    tb_solver_method: str = "diagonalization",
    valence_charge: float = 1.0,
    ewald_cutoff: float = 12.0,
    pppm_accuracy: float = 1e-4,
    best_fit_dir: str = "best_fit_params",
    xdata_hopping: Optional[List[Any]] = None,
    ydata_hopping: Optional[List[Any]] = None,
    tb_params: Optional[np.ndarray] = None,
    elements: Optional[List[str]] = None,
    lammps_exec: Optional[str] = None,
    ridge_acsf: float = 0.1,
    regularization: float = 1e-12,
    weight_energy: float = 1000.0,
    weight_force: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Fit the POD *residual* potential for a TETB-style model (TB + Ewald + POD).

    The total model is ``E = E_band(TB) + E_ewald(q) + E_POD`` with analogous
    forces.  This routine:

    1. Resolves ACSF linear hopping weights ``tb_params`` at the requested
       ``(M, W, r_cut)``.  If ``tb_params`` is not supplied, loads
       ``{best_fit_dir}/ACSF_hoppings_M_{M}_W_{W}_best_fit_params.npz`` when
       present; otherwise fits with :func:`fit_acsf_linear_hopping` from
       ``xdata_hopping`` / ``ydata_hopping`` (required in that case) and saves
       the cache file.
    2. For every structure, evaluates **TB band + Coulomb (Mulliken)** using
       :class:`~blg_model_builder.lammps_interface.TETB_PODLammpsCalculator`
       with **zero** POD coefficients (so the LAMMPS part is Ewald-only in
       practice).
    3. Forms residuals ``E_res = E_DFT - E_TB - E_ewald`` and
       ``F_res = F_DFT - F_TB - F_ewald``.
    4. Attaches those residuals via :class:`~ase.calculators.singlepoint.SinglePointCalculator`
       on copied ``Atoms`` and calls :func:`fit_pod` so LAMMPS ``fitpod`` trains
       the POD on ``(E_res, F_res)``.

    Parameters
    ----------
    atoms_list
        DFT reference geometries (``ase.Atoms``), one per configuration.
    e_dft
        Total DFT energies in eV for **each whole structure** (same convention
        as ``Atoms`` / ``load_energy_data``: supercell-extensive totals, not
        per-atom), length ``len(atoms_list)``.
    f_dft
        DFT force arrays, each ``(n_atoms, 3)`` in eV/Å (required).  Residual
        forces are ``F_DFT - F_TB - F_ewald``.
    M, W, r_cut
        ACSF hopping descriptor hyperparameters (same as ``ACSF_hoppings``).
    pod_hyperparams
        POD descriptor dict (keys as in ``pod_hyperparams_to_str`` /
        ``ncoeff_from_params``).  ``species`` is defaulted to *elements* if absent.
    pod_cutoff
        POD ``rcut`` in Å (descriptor file and :class:`TETB_PODLammpsCalculator`).
    kpoints
        Passed to ``TETB_PODLammpsCalculator``. ``None`` (default) selects an
        automatic MP mesh from each structure; pass ``[nx, ny, nz]`` or an
        explicit ``(n_kp, 3)`` Cartesian **k**-array to override.
    tb_solver_method, valence_charge, ewald_cutoff, pppm_accuracy
        Same meaning as in ``TETB_PODLammpsCalculator``.
    best_fit_dir
        Directory for ``ACSF_hoppings_M_*_W_*_best_fit_params.npz`` cache.
    xdata_hopping, ydata_hopping
        Training descriptor / hopping lists for ``fit_acsf_linear_hopping`` when
        no cache exists and ``tb_params`` is not passed.
    tb_params
        If given, skips load/fit and uses these ACSF weights directly.
    elements
        Chemical symbols for POD / TB calculator (default ``["C"]``).
    lammps_exec
        LAMMPS executable for :func:`fit_pod`.  Defaults to the same path as
        ``fit_pod`` when *None*.
    ridge_acsf
        Ridge regularisation for ``fit_acsf_linear_hopping`` when fitting TB.
    regularization, weight_energy, weight_force
        Passed to :func:`fit_pod` on the residual (same knobs as standalone POD).

    Returns
    -------
    tb_params : ndarray
        ACSF hopping weights used for the subtraction step.
    pod_coeffs : ndarray
        Coefficients returned by ``fit_pod``.
    atoms_residual : list of ase.Atoms
        Structures carrying ``SinglePointCalculator`` with residual energies
        and forces (the ``fitpod`` training targets).
    """
    from blg_model_builder.lammps_interface import TETB_PODLammpsCalculator
    from blg_model_builder.potentials import (
        ncoeff_from_params,
        pod_hyperparams_to_str,
    )

    elems = list(elements or ["C"])
    n_cfg = len(atoms_list)
    e_dft_arr = np.asarray(e_dft, dtype=np.float64).reshape(n_cfg)

    if n_cfg == 0:
        raise ValueError("fit_tetb_residual_pod: atoms_list is empty")

    if len(e_dft_arr) != n_cfg:
        raise ValueError(
            f"fit_tetb_residual_pod: len(e_dft)={len(e_dft_arr)} != len(atoms_list)={n_cfg}"
        )

    if len(f_dft) != n_cfg:
        raise ValueError(
            f"fit_tetb_residual_pod: len(f_dft)={len(f_dft)} != len(atoms_list)={n_cfg}"
        )

    # ── 1. ACSF hopping weights ─────────────────────────────────────────────
    os.makedirs(best_fit_dir, exist_ok=True)
    acsf_npz = os.path.join(
        best_fit_dir, f"ACSF_hoppings_M_{int(M)}_W_{int(W)}_best_fit_params.npz",
    )

    if tb_params is not None:
        tb_params_arr = np.asarray(tb_params, dtype=np.float64)
    elif os.path.isfile(acsf_npz):
        tb_params_arr = np.asarray(
            np.load(acsf_npz, allow_pickle=True)["params"], dtype=np.float64,
        )
    else:
        if xdata_hopping is None or ydata_hopping is None:
            raise ValueError(
                "fit_tetb_residual_pod: no ACSF cache at "
                f"{acsf_npz!r} and tb_params not given — supply "
                "xdata_hopping and ydata_hopping so fit_acsf_linear_hopping can run."
            )
        tb_params_arr, _ = fit_acsf_linear_hopping(
            xdata_hopping, ydata_hopping, ridge=ridge_acsf,
        )
        np.savez(
            acsf_npz,
            params=tb_params_arr,
            bounds=np.array([[ -1e8, 1e8]] * len(tb_params_arr), dtype=float),
            acsf_M=np.int32(M),
            acsf_W=np.int32(W),
            allow_pickle=True,
        )

    # ── POD hyperparams (ensure species for ncoeff) ────────────────────────
    pod_hp = dict(pod_hyperparams)
    if "species" not in pod_hp:
        pod_hp["species"] = elems

    n_pod = ncoeff_from_params(pod_hp)
    pod_zero = np.zeros(n_pod, dtype=np.float64)

    tb_hp = {"M": int(M), "W": int(W), "r_cut": float(r_cut)}
    # Zero POD coefficients → TB band + Ewald (no POD residual term).
    calc = TETB_PODLammpsCalculator(
        tb_params=tb_params_arr,
        pod_params=pod_zero,
        tb_hyperparams=tb_hp,
        pod_hyperparams=pod_hp,
        pod_cutoff=float(pod_cutoff),
        elements=elems,
        kpoints=kpoints,
        tb_solver_method=tb_solver_method,
        ewald_cutoff=float(ewald_cutoff),
        pppm_accuracy=float(pppm_accuracy),
        valence_charge=float(valence_charge),
        shift=0.0,
    )

    atoms_residual: List[Any] = []
    try:
        for i, atoms in enumerate(atoms_list):
            ref = calc.calculate(atoms)
            e_tb_ew = float(ref["energy"])
            f_tb_ew = np.asarray(ref["forces"], dtype=np.float64)

            e_res = float(e_dft_arr[i]) - e_tb_ew
            f_dft_i = np.asarray(f_dft[i], dtype=np.float64)
            if f_dft_i.shape != f_tb_ew.shape:
                raise ValueError(
                    f"fit_tetb_residual_pod: forces shape mismatch at index {i}: "
                    f"DFT {f_dft_i.shape} vs TB+Ewald {f_tb_ew.shape}"
                )
            f_res = f_dft_i - f_tb_ew

            ac = atoms.copy()
            ac.calc = SinglePointCalculator(ac, energy=e_res, forces=f_res)
            atoms_residual.append(ac)
    finally:
        calc.close()

    hp_str = pod_hyperparams_to_str(pod_hp, pod_cutoff, elems)
    lmp = (
        lammps_exec
        if lammps_exec is not None
        else "/mnt/c/Users/Daniel/Documents/research/lammps/build/lmp"
    )
    pod_coeffs = fit_pod(
        hp_str,
        atoms_residual,
        lammps_exec=lmp,
        regularization=regularization,
        weight_energy=weight_energy,
        weight_force=weight_force,
    )

    return tb_params_arr, np.asarray(pod_coeffs, dtype=np.float64), atoms_residual


def fit_torch(
    model,
    xdata,
    ydata,
    zero_shift_data: bool = False,
    ydata_forces=None,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    force_weight: float = 1.0
):
    """
    Train a PODLinearModel (or any nn.Module whose forward(atoms) returns
    (energy_tensor, forces_tensor_or_None)) by minimising a weighted
    energy + force MSE loss.

    Loss
    ----
    L = w_e * MSE(E_pred, E_ref)
      + w_f * MSE(F_pred_flat, F_ref_flat)   # only when ydata_forces given

    with
        w_e = 1 / (N_train * Var(E_ref))
        w_f = force_weight / (N_train * N_force_components * Var(F_ref_flat))

    Parameters
    ----------
    model : nn.Module
        Must implement ``model.forward(atoms) -> (E: Tensor, F: Tensor | None)``.
        ``E`` must carry a ``grad_fn`` with respect to ``model.parameters()``.
    xdata : list[ase.Atoms]
        One entry per training configuration.
    ydata : array-like, shape (N,)
        Reference energies.
    zero_shift_data : bool
        If True, subtract the energy of the lowest-energy configuration from
        both reference and predicted energies before computing the loss.
    ydata_forces : list[array (natom_i, 3)] or None
        Reference forces per configuration.  Pass None to train on energies
        only.
    num_epochs : int
    learning_rate : float
    batch_size : int
    force_weight : float
        Relative weight of the force term (default 1.0).

    Returns
    -------
    flat_params   : np.ndarray, shape (nparams,)
        Fitted model parameters as a flat array.
    ypred_bestfit : np.ndarray, shape (N,)
        Model energies at the fitted parameters, zero-shifted if requested.
    """

    # ── Reference energies ────────────────────────────────────────────────
    ydata_np = np.asarray(ydata, dtype=np.float64)
    N_train  = len(xdata)

    if zero_shift_data:
        min_ind   = int(np.argmin(ydata_np))
        min_atoms = xdata[min_ind]
        ydata_np  = ydata_np - ydata_np[min_ind]

    y = torch.tensor(ydata_np, dtype=DTYPE)

    y_var = torch.var(y)
    if y_var < 1e-12:
        y_var = torch.tensor(1.0, dtype=DTYPE)
    w_e = 1.0 / torch.linalg.norm(y) #(N_train * y_var)

    # ── Reference forces ──────────────────────────────────────────────────
    use_forces = (ydata_forces is not None) and (force_weight > 0.0)
    if use_forces:
        y_forces = [
            torch.tensor(np.asarray(f, dtype=np.float64), dtype=DTYPE)
            for f in ydata_forces
        ]
        f_flat_np = np.concatenate([np.ravel(f) for f in ydata_forces])
        f_var     = float(np.var(f_flat_np))
        if f_var < 1e-12:
            f_var = 1.0
        w_f =  force_weight / torch.linalg.norm(torch.tensor(f_flat_np, dtype=DTYPE)) #(N_train * len(f_flat_np) * f_var)
    else:
        y_forces = None
        w_f      = 0.0

    # ── DataLoader ────────────────────────────────────────────────────────
    indices = list(range(N_train))

    def collate_fn(batch_indices):
        return (
            [xdata[i]    for i in batch_indices],
            y[batch_indices],
            [y_forces[i] for i in batch_indices] if use_forces else None,
        )

    train_loader = torch.utils.data.DataLoader(
        indices,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-7)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining with {n_params} parameters, "
          f"N_train={N_train}, use_forces={use_forces}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mae  = 0.0

        for x_batch, y_batch, yf_batch in train_loader:

            preds_E = []
            preds_F = []

            for i, atoms_i in enumerate(x_batch):
                # Do NOT wrap in torch.no_grad() — gradients must flow
                # through E_i back to model.parameters().
                E_i, F_i = model.forward(atoms_i)
                

                if not torch.is_tensor(E_i):
                    raise RuntimeError(
                        f"model.forward() returned a non-tensor energy "
                        f"({type(E_i)}).  The energy must be a torch.Tensor "
                        f"with a grad_fn to allow loss.backward()."
                    )

                # .to(DTYPE) is a no-op when dtype already matches;
                # it does NOT break the computation graph.
                preds_E.append(E_i.squeeze().to(DTYPE))

                if use_forces and F_i is not None:
                    if not torch.is_tensor(F_i):
                        raise RuntimeError(
                            f"model.forward() returned non-tensor forces "
                            f"({type(F_i)})."
                        )
                    preds_F.append(F_i.to(DTYPE))

            # torch.stack preserves grad_fn → shape (batch,)
            pred_E = torch.stack(preds_E)
           

            # zero-shift: subtract predicted min — must stay in the graph
            if zero_shift_data:
                E_min, _ = model.forward(min_atoms)
                pred_E   = pred_E - E_min.squeeze().to(DTYPE)
            
            loss = w_e * criterion(pred_E, y_batch)

            with torch.no_grad():
                mae = torch.mean(torch.abs(pred_E - y_batch))

            if use_forces and preds_F and yf_batch is not None:
                pred_F_flat   = torch.cat([f.reshape(-1) for f in preds_F])
                target_F_flat = torch.cat(
                    [t.to(DTYPE).reshape(-1) for t in yf_batch])
                if pred_F_flat.shape != target_F_flat.shape:
                    raise ValueError(
                        f"Force dimension mismatch: predicted {pred_F_flat.numel()} components "
                        f"vs target {target_F_flat.numel()}. "
                        "This usually indicates xdata and ydata_forces are misaligned "
                        "(different structure at the same index). Ensure train_test_split "
                        "uses a single canonical split for all keys.")
                loss = loss + w_f * criterion(pred_F_flat, target_F_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae  += mae.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_mae  = epoch_mae  / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:>{len(str(num_epochs))}}/{num_epochs}]  "
                f"Loss: {avg_loss:.6f}  "
                f"MAE: {avg_mae:.6f} eV  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # ── Final predictions (no grad needed) ────────────────────────────────
    model.eval()
    preds_final = []

    with torch.no_grad():
        if zero_shift_data:
            E_min_final, _ = model.forward(min_atoms)
            E_min_final    = E_min_final.squeeze().to(DTYPE)
        else:
            E_min_final = torch.tensor(0.0, dtype=DTYPE)

        for atoms_i in xdata:
            E_i, _ = model.forward(atoms_i)
            preds_final.append(E_i.squeeze().to(DTYPE) - E_min_final)

    model.train()

    ypred_bestfit = (
        torch.stack(preds_final).squeeze().detach().cpu().numpy()
    )

    if hasattr(model, "get_coefficients"):
        flat_params = model.get_coefficients()
    else:
        flat_params = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in model.parameters()]
        )

    return flat_params, ypred_bestfit


def _energy_force_loss_weights(ydata, ydata_forces):
    """Positive weights for energy+force fitting (same spirit as EMCEE log_likelihood).

    Previously w_e = sum(ydata) was used; for negative DFT energies that sum is
    negative, so minimizing loss *increased* the energy residual and drove inf energies.
    """
    if ydata_forces is None:
        return 1.0, 0.0
    ydata_arr = np.asarray(ydata, dtype=float)
    n = max(len(ydata_arr), 1)
    y_std_e = float(np.std(ydata_arr))
    if y_std_e < 1e-12:
        y_std_e = 1.0
    w_e = 1.0 / (n * y_std_e ** 2)
    flat = np.concatenate([np.ravel(np.asarray(f, dtype=float)) for f in ydata_forces])
    n_f = max(len(flat), 1)
    y_std_f = float(np.std(flat))
    if y_std_f < 1e-12:
        y_std_f = 1.0
    w_f = 1.0 / (n_f * y_std_f ** 2)
    return w_e, w_f


def fit_model(method,xdata,ydata,p0,ydata_forces=None,zero_shift_data=False,bounds=None,minimizer="L-BFGS-B",**kwargs):
    print("fitting model with "+str(len(p0))+" parameters")
    w_e, w_f = _energy_force_loss_weights(ydata, ydata_forces)
    w_f = 0

    loss_fxn = get_loss_fxn(method,xdata,ydata,ydata_forces=ydata_forces,zero_shift_data=zero_shift_data,w_e=w_e,w_f=w_f)
    if minimizer=="differential_evolution":
        result = scipy.optimize.differential_evolution(loss_fxn,bounds,strategy="randtobest1bin")
    elif minimizer=="SLSQP":
        result = scipy.optimize.minimize(loss_fxn,p0,method="SLSQP",bounds=bounds)
    elif minimizer=="Nelder-Mead":
        result = scipy.optimize.minimize(loss_fxn,p0,method="Nelder-Mead",bounds=bounds)
    elif minimizer=="L-BFGS-B":
        result = scipy.optimize.minimize(loss_fxn,p0,method="L-BFGS-B",bounds=bounds)
    else:
        raise ValueError("Invalid minimizer: "+minimizer)
    popt = result.x
    ypred_bestfit = get_prediction(method,xdata,popt) 
    #pcov = result.hess_inv.todense()
    return np.array(popt), ypred_bestfit

def get_loss_fxn(method,xdata,ydata,ydata_forces=None,zero_shift_data=False,w_e=1.0, w_f = 1.0):
    def func(params):
        ypred = get_prediction(method,xdata,params) 
        if isinstance(ypred, tuple):
            ypred_ = ypred[0]
            ypred_forces = ypred[1]
        else:
            ypred_ = ypred
            ypred_forces = None
        
        if zero_shift_data:
            shift_ind = np.argmin(ydata) 
            ypred_shift = ypred_[shift_ind] 
            ydata_shift = ydata[shift_ind] 
            ypred_scaled = np.nan_to_num((ypred_-ypred_shift)) 
            ydata_scaled = np.nan_to_num((ydata-ydata_shift)) 
        else:
            # Use the energy component only (ypred_ not ypred, which may be a
            # (energy, forces) tuple when the model also returns forces).
            ypred_scaled = ypred_
            ydata_scaled = ydata
        if type(ydata_scaled)==list:
            loss = 0
            for i in range(len(xdata)):
                loss += np.linalg.norm(ypred_scaled[i] - ydata_scaled[i])
                if ypred_forces is not None:
                    loss += np.linalg.norm(ypred_forces[i] - ydata_forces[i])
            return loss
        elif type(ydata_scaled)==np.ndarray:
            # Guard against unexpected non-array types (e.g. residual tuple from
            # a model that returns (energy, forces)).
            try:
                ypred_scaled = np.asarray(ypred_scaled, dtype=float)
            except (ValueError, TypeError):
                return 1e300
            if not np.all(np.isfinite(ypred_scaled)) or not np.all(np.isfinite(ydata_scaled)):
                return 1e300
            if ypred_forces is not None:
                for i in range(len(xdata)):
                    fi = ypred_forces[i]
                    if fi is None:
                        return 1e300
                    if not np.all(np.isfinite(fi)):
                        return 1e300
            loss = w_e * np.linalg.norm(ydata_scaled - ypred_scaled)
            if ypred_forces is not None:
                for i in range(len(xdata)):
                    loss += w_f * np.linalg.norm(ypred_forces[i] - ydata_forces[i])
            return loss
    return func

def get_prediction(method, xdata, params):
    """method(x, params) may return scalar/array or (energy, forces) tuple."""
    
    if isinstance(xdata, list):
        
        y_pred_energy = []
        y_pred_forces = []
        for x in xdata:
            yval = method(x, params)
            if isinstance(yval, (list, tuple)) and len(yval) >= 2:
                y_pred_energy.append(yval[0])
                y_pred_forces.append(yval[1])
            else:
                y_pred_energy.append(yval)
                y_pred_forces.append(None)

        if all(f is None for f in y_pred_forces):
            return y_pred_energy
        return np.array(y_pred_energy), y_pred_forces  # forces: list (ragged n_atoms per config)
        
    y_pred = method(xdata, params)
    return y_pred


if __name__=="__main__":
    int_type = "full"
    energy_model = None
    tb_model = "MLP_tb"
    model_name = str(energy_model)+"_energy_"+str(int_type)+"_"+str(tb_model)
    model_name = model_name.replace("full_","")
    model_name = model_name.replace("None_energy_","")
    model_name = model_name.replace("_None","")
    
    calc,xdata,ydata,ydata_noise, params,params_std,bounds,ypred_bestfit = get_MCMC_inputs(int_type=int_type,energy_model=energy_model,tb_model=tb_model,model_name=model_name)
    r = np.linalg.norm(xdata["hoppings"],axis=1)
    plt.scatter(r,ydata["hoppings"])
    plt.scatter(r,ypred_bestfit["hoppings"])
    plt.show()