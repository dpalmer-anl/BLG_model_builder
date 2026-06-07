from blg_model_builder.DataLoader import load_data_for_model
from blg_model_builder.tb_models import (
    create_tb_model,
    get_acsf_hoppings,
    get_acsf_hoppings_sk,
    letb_interlayer,
    letb_intralayer_t01,
    letb_intralayer_t02,
    letb_intralayer_t03,
    mk_hopping,
)
from blg_model_builder.potentials import *
from blg_model_builder.lammps_interface import (
    TersoffLammpsCalculator,
    KolmogorovCrespiLammpsCalculator,
    DRIPLammpsCalculator,
    TersoffKCLammpsCalculator,
    TersoffDRIPLammpsCalculator,
    PODLammpsCalculator,
    TETB_PODLammpsCalculator,
)
from blg_model_builder.allegro_interface import (
    AllegroCalculator,
    allegro_bounds_from_params,
)
from blg_model_builder.model_fit import (
    fit_acsf_linear_hopping,
    fit_model,
    fit_pod,
    fit_tetb_residual_pod,
    get_prediction,
)
from typing import Optional, Dict, Any, Tuple, Callable
import warnings
import numpy as np
import os
import re
import matplotlib.pyplot as plt

# ``np.savez`` does not create parent directories and does not accept ``allow_pickle``
# (that keyword is only for ``numpy.load`` / ``numpy.save``).
_BEST_FIT_PARAMS_SUBDIR = "best_fit_params"

# Last LAMMPS calculator built by :func:`get_MCMC_inputs` (UQ propagation scripts).
_UQ_LAMMPS_RUNTIME: Dict[str, Any] = {}


def _register_uq_lammps_runtime(
    calc_obj: Any,
    set_params_fn: Optional[Callable] = None,
) -> None:
    _UQ_LAMMPS_RUNTIME.clear()
    _UQ_LAMMPS_RUNTIME["calc_obj"] = calc_obj
    if set_params_fn is not None:
        _UQ_LAMMPS_RUNTIME["set_params_fn"] = set_params_fn


def get_uq_lammps_runtime() -> Dict[str, Any]:
    """Return ``calc_obj`` / ``set_params_fn`` from the latest :func:`get_MCMC_inputs` call."""
    return dict(_UQ_LAMMPS_RUNTIME)

# Default POD descriptor radial settings (``POD_energy`` and TETB+POD POD block).
POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE = 4
POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE = 8


def _ensure_best_fit_params_dir() -> None:
    os.makedirs(_BEST_FIT_PARAMS_SUBDIR, exist_ok=True)


def _energy_ypred_for_cache(ypred):
    """``fit_model`` / ``get_prediction`` may return ``(energies, forces)``; cache MAE uses 1-D energies."""
    if isinstance(ypred, tuple) and len(ypred) >= 1:
        ypred = ypred[0]
    return np.asarray(ypred, dtype=float)


def _append_shift_param(params_arr, bounds_arr, ydata):
    """Append an energy-shift scalar to *params_arr* and *bounds_arr*.

    The shift is initialized to ``mean(ydata)`` so the optimizer starts with a
    reasonable absolute energy offset.  Bounds span ±10× the ydata range to give
    the sampler freedom while keeping the prior finite.

    Parameters
    ----------
    params_arr : np.ndarray, shape (N,)
    bounds_arr : np.ndarray, shape (N, 2)   each row is [lo, hi]
    ydata : array-like   training energies

    Returns
    -------
    new_params : np.ndarray, shape (N+1,)
    new_bounds : np.ndarray, shape (N+1, 2)
    """
    y = np.asarray(ydata, dtype=float)
    shift0 = float(np.mean(y))
    rng = float(np.ptp(y)) if np.ptp(y) > 0 else 1.0
    shift_bounds = np.array([[float(np.min(y)) - 10 * rng,
                              float(np.max(y)) + 10 * rng]])
    return np.append(params_arr, shift0), np.vstack([bounds_arr, shift_bounds])


def _make_batch_evaluator(calc_obj, xdata_atoms, set_params_fn=None):
    """Return an ``evaluator(atoms, params) -> (energy, forces)`` backed by batch LAMMPS evaluation.

    ``prepare_batch`` is called once on ``xdata_atoms`` (the full dataset).
    On each unique ``params`` vector, ``evaluate_batch`` is called **once** for
    all ~400 configurations; subsequent calls with the same params return the
    cached per-atom result instantly.  This is the key MCMC speedup: one
    LAMMPS pass per proposal instead of N_configs passes.

    Parameters
    ----------
    calc_obj : LammpsCalculatorBase
        Calculator whose ``prepare_batch`` / ``evaluate_batch`` will be used.
    xdata_atoms : list[ase.Atoms]
        Full training+test atom list.  The evaluator uses ``id(atoms)``
        look-up, so the objects must be the same Python instances that will
        later be passed by ``get_prediction``.
    set_params_fn : callable, optional
        If provided, called as ``set_params_fn(params)`` instead of
        ``calc_obj.set_parameters(params)``.  Useful for hybrid calculators
        where ``set_parameters`` takes two separate vectors.
    """
    calc_obj.prepare_batch(xdata_atoms)
    _idx = {id(a): i for i, a in enumerate(xdata_atoms)}
    _set = set_params_fn if set_params_fn is not None else calc_obj.set_parameters
    state = {"params": None, "energies": None, "forces": None}

    def evaluator(atoms, params):
        params_arr = np.asarray(params, dtype=np.float64)
        if state["params"] is None or not np.array_equal(params_arr, state["params"]):
            _set(params_arr.tolist())
            state["energies"], state["forces"] = calc_obj.evaluate_batch()
            state["params"] = params_arr.copy()
        i = _idx[id(atoms)]
        e = state["energies"][i]
        f = state["forces"][i]
        # forces[i] is np.full(NaN) when LAMMPS failed for that config.
        if f is None:
            f = np.full((len(atoms), 3), np.nan)
        return float(e), np.asarray(f, dtype=float)

    return evaluator


def build_tetb_pod_hyperparams_from_data_kw(
    data_kw: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Resolve TB (ACSF) and POD descriptor dicts plus a tag for cache / ensemble paths.

    The tag has the form ``tb_M_<m>_W_<w>_pod_M_<pm>_W_<pw>`` where ``(m, w)``
    are the resolved ACSF hopping basis sizes and ``(pm, pw)`` are the resolved
    POD primary knobs (twobody radial count and threebody angular degree).

    Parameters
    ----------
    data_kw
        Same keys as :func:`get_MCMC_inputs` uses for ``TETB_POD`` (``M``, ``W``,
        ``tb_M``, ``tb_W``, ``pod_M``, ``pod_W``, optional ``tb_hyperparams`` /
        ``pod_hyperparams`` overrides, etc.).

    Returns
    -------
    tb_hyperparams, pod_hyperparams, tag
        Merged hyperparameter dicts and the tag string (no ``TETB_POD_`` prefix).
    """
    _tb_M = int(data_kw.get("tb_M", data_kw.get("acsf_M", data_kw.get("M", 10))))
    _tb_W = int(data_kw.get("tb_W", data_kw.get("acsf_W", data_kw.get("W", 3))))
    tb_hyperparams = {
        "M": _tb_M,
        "W": _tb_W,
        "r_cut": float(data_kw.get("r_cut", data_kw.get("acsf_r_cut", 6.0))),
    }
    _pod_M = int(data_kw.get("pod_M", data_kw.get("M", 10)))
    _pod_W = int(data_kw.get("pod_W", data_kw.get("W", 4)))
    pod_hyperparams = {
        "species": ["C"],
        "bessel_polynomial_degree": POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE,
        "inverse_polynomial_degree": POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE,
        "twobody_number_radial_basis_functions": _pod_M,
        "threebody_number_radial_basis_functions": int(
            data_kw.get("pod_M", data_kw.get("M", 3))
        ),
        "threebody_angular_degree": int(data_kw.get("pod_W", data_kw.get("W", 4))),
        "fourbody_number_radial_basis_functions": int(
            data_kw.get("pod_M", data_kw.get("M", 2))
        ),
        "fourbody_angular_degree": int(data_kw.get("pod_W", data_kw.get("W", 3))),
        "fivebody_number_radial_basis_functions": 4,
        "fivebody_angular_degree": 3,
        "sixbody_number_radial_basis_functions": 3,
        "sixbody_angular_degree": 2,
        "sevenbody_number_radial_basis_functions": 2,
        "sevenbody_angular_degree": 2,
    }
    _tb_ov = data_kw.get("tb_hyperparams")
    if isinstance(_tb_ov, dict):
        tb_hyperparams = {**tb_hyperparams, **_tb_ov}
    _pod_ov = data_kw.get("pod_hyperparams")
    if isinstance(_pod_ov, dict):
        pod_hyperparams = {**pod_hyperparams, **_pod_ov}
    _tb_M = int(tb_hyperparams.get("M", tb_hyperparams.get("acsf_M", 10)))
    _tb_W = int(tb_hyperparams.get("W", tb_hyperparams.get("acsf_W", 3)))
    pod_m = int(pod_hyperparams["twobody_number_radial_basis_functions"])
    pod_w = int(pod_hyperparams["threebody_angular_degree"])
    tag = f"tb_M_{_tb_M}_W_{_tb_W}_pod_M_{pod_m}_W_{pod_w}"
    return tb_hyperparams, pod_hyperparams, tag


def get_MCMC_inputs(model_name, calc_type="python", supercells=1,
                    hyperparameters: Optional[Dict[str, Any]] = None,
                    M: Optional[int] = None,
                    W: Optional[int] = None,
                    tb_M: Optional[int] = None,
                    tb_W: Optional[int] = None,
                    pod_M: Optional[int] = None,
                    pod_W: Optional[int] = None,
                    **kwargs):
    """Load data and fitted models for UQ / MCMC.

    Extra keyword arguments are forwarded to :func:`load_data_for_model`. For
    ``model_name == "ACSF_hoppings"`` you can set ACSF basis size with e.g.
    ``M=8``, ``W=4`` or ``acsf_M``, ``acsf_W``, and cutoff with ``r_cut`` /
    ``acsf_r_cut``. Optional ``M`` and ``W`` arguments override those keys after
    ``kwargs`` (handy when calling this function explicitly). ``hyperparameters``
    is merged first, then ``kwargs``, then ``M`` / ``W`` if provided.

    ACSF best-fit caches are stored per ``(M, W)`` as
    ``best_fit_params/ACSF_hoppings_M_<M>_W_<W>_best_fit_params.npz``.

    For ``model_name == "TETB_POD"`` (or an extended tag
    ``TETB_POD_tb_M_<m>_W_<w>_pod_M_<pm>_W_<pw>`` from the ensemble drivers),
    tight-binding (ACSF) and POD descriptor sizes are controlled separately:
    use ``tb_M`` / ``tb_W`` (or ``acsf_M`` / ``acsf_W`` / ``r_cut`` in ``kwargs``)
    for hoppings, and ``pod_M`` / ``pod_W`` (or top-level ``M`` / ``W`` as defaults
    for the POD block) for the POD potential — same pattern as ``ACSF_hoppings``
    vs ``POD_energy``, combined in one model.

    Best-fit caches are written as
    ``best_fit_params/TETB_POD_tb_M_<m>_W_<w>_pod_M_<pm>_W_<pw>_best_fit_params.npz``.
    A legacy ``TETB_POD_best_fit_params.npz`` is still read if the tagged file is absent.

    Reference energies are **total** DFT energies (eV) for each ASE structure,
    same as in ``load_energy_data`` / ``DataLoader``. The fitted POD residual
    is added to TB + Ewald so the sum matches those totals; no extra additive
    ``mean(E_DFT)`` shift is applied (unlike classical potentials that use
    ``_append_shift_param``).
    """
    _UQ_LAMMPS_RUNTIME.clear()
    data_kw: Dict[str, Any] = {}
    if hyperparameters is not None:
        data_kw.update(hyperparameters)
    data_kw.update(kwargs)
    if M is not None:
        data_kw["M"] = M
    if W is not None:
        data_kw["W"] = W
    if tb_M is not None:
        data_kw["tb_M"] = tb_M
    if tb_W is not None:
        data_kw["tb_W"] = tb_W
    if pod_M is not None:
        data_kw["pod_M"] = pod_M
    if pod_W is not None:
        data_kw["pod_W"] = pod_W

    skip_diagnostics = bool(data_kw.pop("skip_diagnostics", False))

    # If the model name encodes M and W (e.g. ACSF_hoppings_sk_M_8_W_6),
    # those values take priority over whatever arrived in data_kw / kwargs
    # so that descriptor computation and cache file naming are always consistent
    # with the model that was actually requested.
    if model_name.startswith("ACSF_hoppings"):
        _mw = re.search(r"[_\-]M[_\-](\d+)[_\-]W[_\-](\d+)", model_name, re.IGNORECASE)
        if _mw:
            data_kw["M"] = int(_mw.group(1))
            data_kw["W"] = int(_mw.group(2))

    load_model_name = model_name
    if model_name.startswith("TETB_POD"):
        load_model_name = "TETB_POD"
    xdata_train, xdata_test, xdata, ydata_train, ydata_test, ydata = load_data_for_model(
        load_model_name, supercells, **data_kw
    )
    calc = {}
    params = {}
    bounds = {}
    ypred_bestfit = {}
    _ensure_best_fit_params_dir()
    if model_name.startswith("MK"):
        calc["hopping"] = create_tb_model("MK", data_kw)
        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/MK_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/MK_best_fit_params.npz")
            params["hopping"] = data["params"]
            bounds["hopping"] = data["bounds"]
            ypred_bestfit["hopping"] = data["ypred_bestfit"]
        else:
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/MK_best_fit_params_estimate.npz")
            p0 = data["params"]
            bounds["hopping"] = data["bounds"]
            params["hopping"], ypred_bestfit["hopping"] = fit_model(calc["hopping"],xdata_train["hopping"],ydata_train["hopping"],p0,yshift=0,zero_shift_data=False,bounds=bounds["hopping"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/MK_best_fit_params.npz",
                params=params["hopping"],
                bounds=bounds["hopping"],
                ypred_bestfit=ypred_bestfit["hopping"],
            )

    elif model_name.startswith("ACSF_hoppings_sk"):
        # ── Slater-Koster ACSF hopping model ────────────────────────────────
        # get_acsf_sk_hopping_descriptors bakes the SK physics into the
        # descriptor columns so xdata["hopping"] has shape (n_pairs, 2*n_feat).
        # get_acsf_hoppings_sk(descriptors_sk, params) = descriptors_sk @ params,
        # identical interface to get_acsf_hoppings — params shape (2*n_feat,).

        from blg_model_builder.tb_models import get_acsf_hoppings_sk

        calc["hopping"] = get_acsf_hoppings_sk

        _acsf_M = int(data_kw.get("acsf_M", data_kw.get("M", 10)))
        _acsf_W = int(data_kw.get("acsf_W", data_kw.get("W", 3)))
        acsf_mw_tag = f"M_{_acsf_M}_W_{_acsf_W}"
        acsf_sk_npz = (
            f"{_BEST_FIT_PARAMS_SUBDIR}/ACSF_hoppings_sk_{acsf_mw_tag}_best_fit_params.npz"
        )

        def _sk_descriptor_width(x_hop_list):
            for blk in x_hop_list:
                a = np.asarray(blk)
                if a.ndim == 2 and a.shape[0] > 0:
                    return int(a.shape[1])
            return None

        n_feat = _sk_descriptor_width(xdata_train["hopping"])
        _sk_param_bound = 1e4

        def _sk_hopping_bounds(n_par: int) -> np.ndarray:
            return np.array([[-_sk_param_bound, _sk_param_bound]] * n_par, dtype=float)

        def _sk_ypred_from_params(x_hop_list, w: np.ndarray):
            return [
                np.asarray(get_acsf_hoppings_sk(np.asarray(blk, dtype=float), w), dtype=float)
                for blk in x_hop_list
            ]

        def _normalize_sk_ypred(yp, x_hop_list, w: np.ndarray):
            if isinstance(yp, list):
                return [np.asarray(b, dtype=float).ravel() for b in yp]
            yp_arr = np.asarray(yp)
            if yp_arr.dtype == object and yp_arr.size > 0:
                return [np.asarray(b, dtype=float).ravel() for b in yp_arr.ravel()]
            if yp_arr.ndim == 0 or yp_arr.size == 0:
                return _sk_ypred_from_params(x_hop_list, w)
            return [np.asarray(yp_arr, dtype=float).ravel()]

        if os.path.exists(acsf_sk_npz):
            data = np.load(acsf_sk_npz, allow_pickle=True)
            params["hopping"] = np.asarray(data["params"], dtype=float)
            n_par = int(params["hopping"].shape[0])
            if "bounds" in data.files:
                bounds["hopping"] = np.asarray(data["bounds"], dtype=float)
            else:
                bounds["hopping"] = _sk_hopping_bounds(n_par)
            if "ypred_bestfit" in data.files:
                ypred_bestfit["hopping"] = _normalize_sk_ypred(
                    data["ypred_bestfit"],
                    xdata_train["hopping"],
                    params["hopping"],
                )
            else:
                ypred_bestfit["hopping"] = _sk_ypred_from_params(
                    xdata_train["hopping"], params["hopping"],
                )
        else:
            bounds["hopping"] = _sk_hopping_bounds(n_feat)
            params["hopping"], ypred_bestfit["hopping"] = fit_acsf_linear_hopping(
                xdata_train["hopping"],
                ydata_train["hopping"],
            )
            n_par = int(np.asarray(params["hopping"]).shape[0])
            if int(bounds["hopping"].shape[0]) != n_par:
                bounds["hopping"] = _sk_hopping_bounds(n_par)
            np.savez(
                acsf_sk_npz,
                params=params["hopping"],
                bounds=bounds["hopping"],
                ypred_bestfit=np.array(ypred_bestfit["hopping"], dtype=object),
                acsf_M=np.int32(_acsf_M),
                acsf_W=np.int32(_acsf_W),
            )

    elif model_name.startswith("ACSF_hoppings"):
        calc["hopping"] = create_tb_model(model_name, data_kw)

        _acsf_M = int(data_kw.get("acsf_M", data_kw.get("M", 10)))
        _acsf_W = int(data_kw.get("acsf_W", data_kw.get("W", 3)))
        acsf_mw_tag = f"M_{_acsf_M}_W_{_acsf_W}"
        acsf_best_npz = f"{_BEST_FIT_PARAMS_SUBDIR}/ACSF_hoppings_{acsf_mw_tag}_best_fit_params.npz"

        def _acsf_descriptor_width(x_hop_list):
            for blk in x_hop_list:
                a = np.asarray(blk)
                if a.ndim == 2 and a.shape[0] > 0:
                    return int(a.shape[1])
            return None

        n_feat = _acsf_descriptor_width(xdata_train["hopping"])
        # Uniform coefficient bounds — same scale as ``POD_energy`` (±1e4).
        _acsf_param_bound = 1e4

        def _acsf_hopping_bounds(n_par: int) -> np.ndarray:
            return np.array([[-_acsf_param_bound, _acsf_param_bound]] * n_par, dtype=float)

        def _acsf_hopping_ypred_from_params(x_hop_list, w: np.ndarray):
            return [
                np.asarray(get_acsf_hoppings(np.asarray(blk, dtype=float), w), dtype=float)
                for blk in x_hop_list
            ]

        def _normalize_acsf_ypred_bestfit(yp, x_hop_list, w: np.ndarray):
            """Return a list of 1-D prediction arrays (per structure)."""
            if isinstance(yp, list):
                return [np.asarray(b, dtype=float).ravel() for b in yp]
            yp_arr = np.asarray(yp)
            if yp_arr.dtype == object and yp_arr.size > 0:
                return [np.asarray(b, dtype=float).ravel() for b in yp_arr.ravel()]
            if yp_arr.ndim == 0 or yp_arr.size == 0:
                return _acsf_hopping_ypred_from_params(x_hop_list, w)
            return [np.asarray(yp_arr, dtype=float).ravel()]

        if os.path.exists(acsf_best_npz):
            data = np.load(acsf_best_npz, allow_pickle=True)
            params["hopping"] = np.asarray(data["params"], dtype=float)
            n_par = int(params["hopping"].shape[0])
            if "bounds" in data.files:
                bounds["hopping"] = np.asarray(data["bounds"], dtype=float)
            else:
                bounds["hopping"] = _acsf_hopping_bounds(n_par)
            if "ypred_bestfit" in data.files:
                ypred_bestfit["hopping"] = _normalize_acsf_ypred_bestfit(
                    data["ypred_bestfit"],
                    xdata_train["hopping"],
                    params["hopping"],
                )
            else:
                ypred_bestfit["hopping"] = _acsf_hopping_ypred_from_params(
                    xdata_train["hopping"], params["hopping"],
                )

        else:
            bounds["hopping"] = _acsf_hopping_bounds(n_feat)
            params["hopping"], ypred_bestfit["hopping"] = fit_acsf_linear_hopping(
                xdata_train["hopping"],
                ydata_train["hopping"],
            )
            n_par = int(np.asarray(params["hopping"]).shape[0])
            if int(bounds["hopping"].shape[0]) != n_par:
                bounds["hopping"] = _acsf_hopping_bounds(n_par)
            np.savez(
                acsf_best_npz,
                params=params["hopping"],
                bounds=bounds["hopping"],
                ypred_bestfit=np.array(ypred_bestfit["hopping"], dtype=object),
                acsf_M=np.int32(_acsf_M),
                acsf_W=np.int32(_acsf_W),
            )

        # Bond lengths: ‖r_ij‖ for each descriptor row (aligned with y_train in DataLoader ACSF branch).
        # yp = ypred_bestfit["hopping"]
        # if isinstance(yp, np.ndarray) and yp.dtype == object:
        #     yp = list(yp)
        # labeled_pred, labeled_dft = False, False
        # for i in range(len(yp)):
        #     dist = np.asarray(xdata_train["hopping_dist"][i], dtype=float).ravel()
        #     y_pred = np.asarray(yp[i], dtype=float).ravel()
        #     y_dft = np.asarray(ydata_train["hopping"][i], dtype=float).ravel()
        #     n = min(dist.size, y_pred.size, y_dft.size)
        #     dist, y_pred, y_dft = dist[:n], y_pred[:n], y_dft[:n]
        #     kw_p = {"c": "tab:red", "s": 14, "alpha": 0.45}
        #     kw_d = {"c": "black", "s": 14, "alpha": 0.45}
        #     if not labeled_pred:
        #         kw_p["label"] = "Predicted"
        #         labeled_pred = True
        #     if not labeled_dft:
        #         kw_d["label"] = "DFT"
        #         labeled_dft = True
        #     plt.scatter(dist, y_pred, **kw_p)
        #     plt.scatter(dist, y_dft, **kw_d)
        # plt.legend()
        # plt.xlabel("Bond distance (Å)")
        # plt.ylabel("Hopping")
        # plt.tight_layout()
        # plt.show()

    elif model_name == "LETB_interlayer":
        calc["hopping"] = create_tb_model("LETB_interlayer", data_kw)
        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/interlayer_LETB_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/interlayer_LETB_best_fit_params.npz")
            params["hopping"] = data["params"]
            bounds["hopping"] = data["bounds"]
            ypred_bestfit["hopping"] = data["ypred_bestfit"]
        else:
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/interlayer_LETB_best_fit_params_estimate.npz")
            p0 = data["params"]
            bounds["hopping"] = data["bounds"]
            params["hopping"], ypred_bestfit["hopping"] = fit_model(calc["hopping"],xdata_train["hopping"],ydata_train["hopping"],p0,yshift=0,zero_shift_data=False,bounds=bounds["hopping"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/interlayer_LETB_best_fit_params.npz",
                params=params["hopping"],
                bounds=bounds["hopping"],
                ypred_bestfit=ypred_bestfit["hopping"],
            )
            
    elif model_name.startswith("LETB_intralayer"):
        nn_val = int(data_kw.get("nn_val", 1))
        if nn_val not in (1, 2, 3):
            raise ValueError(f"Unknown nn_val {nn_val!r} for LETB intralayer models")
        data_kw["nn_val"] = nn_val
        tb_letb = create_tb_model("LETB_intralayer", data_kw)
        calc["hopping"] = tb_letb
        model_name = tb_letb.canonical_name

        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/{model_name}_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/{model_name}_best_fit_params.npz")
            params["hopping"] = data["params"]
            bounds["hopping"] = data["bounds"]
            ypred_bestfit["hopping"] = data["ypred_bestfit"]
        else:
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/{model_name}_best_fit_params_estimate.npz")
            p0 = data["params"]
            bounds["hopping"] = data["bounds"]
            params["hopping"], ypred_bestfit["hopping"] = fit_model(calc["hopping"],xdata_train["hopping"],ydata_train["hopping"],p0,yshift=0,zero_shift_data=False,bounds=bounds["hopping"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/{model_name}_best_fit_params.npz",
                params=params["hopping"],
                bounds=bounds["hopping"],
                ypred_bestfit=ypred_bestfit["hopping"],
            )
            
    # ----- Energy data -----
    elif model_name.startswith("POD_energy"):

        # POD descriptor hyperparameters can be provided as a full dict (e.g. from the
        # hyperparameter search CSV), otherwise build from legacy M/W + per-body knobs.
        rcut = float(data_kw.get("pod_cutoff", data_kw.get("rcut", 6.0)))
        pod_hp_override = data_kw.get("pod_hyperparams")
        if isinstance(pod_hp_override, dict) and pod_hp_override:
            hyperparams = dict(pod_hp_override)
            hyperparams.setdefault("species", ["C"])
        else:
            # M / W are legacy shorthands: all per-body kwargs fall back to them.
            _M = int(data_kw.get("M", 10))
            _W = int(data_kw.get("W", 4))
            hyperparams = {
                "species": ["C"],
                "bessel_polynomial_degree": POD_DEFAULT_BESSEL_POLYNOMIAL_DEGREE,
                "inverse_polynomial_degree": POD_DEFAULT_INVERSE_POLYNOMIAL_DEGREE,
                "twobody_number_radial_basis_functions": int(data_kw.get("two_body_radial",   _M)),
                "threebody_number_radial_basis_functions": int(data_kw.get("three_body_radial", _M)),
                "threebody_angular_degree":                int(data_kw.get("three_body_angular", _W)),
                "fourbody_number_radial_basis_functions":  int(data_kw.get("four_body_radial",  _M)),
                "fourbody_angular_degree":                 int(data_kw.get("four_body_angular",  _W)),
                # 5-7 body: configurable for stability tuning; default historical values.
                # Set to 0 to disable high-body terms (fewer coefficients, better OOD).
                "fivebody_number_radial_basis_functions": int(data_kw.get("five_body_radial",   4)),
                "fivebody_angular_degree":                int(data_kw.get("five_body_angular",   3)),
                "sixbody_number_radial_basis_functions":  int(data_kw.get("six_body_radial",     3)),
                "sixbody_angular_degree":                 int(data_kw.get("six_body_angular",     2)),
                "sevenbody_number_radial_basis_functions": int(data_kw.get("seven_body_radial",  2)),
                "sevenbody_angular_degree":                int(data_kw.get("seven_body_angular",  2)),
            }
        ncoeffs = ncoeff_from_params(hyperparams)

        _regularization   = float(data_kw.get("regularization",   1e-12))
        _weight_energy    = float(data_kw.get("weight_energy",    1000.0))
        _weight_force     = float(data_kw.get("weight_force",     1.0))
        _include_intra    = bool(data_kw.get("include_intralayer", False))
        _pod_hash         = str(data_kw.get("pod_hash", "")).strip()

        # Encode fit settings into the cache filename so that models with
        # different regularization or training data are not confused.
        _reg_tag   = f"reg{_regularization:.0e}"
        _intra_tag = "_intra" if _include_intra else ""
        _hash_tag  = f"_{_pod_hash}" if _pod_hash else ""
        cache_path = (
            f"{_BEST_FIT_PARAMS_SUBDIR}/"
            f"POD_energy_{ncoeffs}_{_reg_tag}{_intra_tag}{_hash_tag}_best_fit_params.npz"
        )

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            params["energy"] = data["params"]
            bounds["energy"] = np.array([len(params["energy"]) * [-1e4, 1e4]])
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
        else:
            hyperparams_str = pod_hyperparams_to_str(hyperparams, rcut, ["C"])
            params["energy"] = fit_pod(
                hyperparams_str,
                xdata_train["energy"],
                regularization=_regularization,
                weight_energy=_weight_energy,
                weight_force=_weight_force,
            )
            bounds["energy"] = np.array([len(params["energy"]) * [-1e4, 1e4]])

        # Build batch-backed calculator over the full dataset.
        calc_obj = PODLammpsCalculator(
            hyperparams, params["energy"], elements=["C"], cutoff=rcut,
        )
        _register_uq_lammps_runtime(calc_obj)
        calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])

        # Compute (or recompute) best-fit predictions on the training set.
        calc_obj.set_parameters(params["energy"].tolist())
        e_bf, f_bf = calc_obj.evaluate_batch(list(xdata_train["energy"]))
        ypred_bestfit["energy"] = e_bf
        ypred_bestfit["forces"] = f_bf

        if not os.path.exists(cache_path):
            np.savez(
                cache_path,
                params=params["energy"], bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
                **{k: v for k, v in hyperparams.items() if k != "species"},
            )

    elif model_name == "Kolmogorov_Crespi":
        _n_kc = KolmogorovCrespiLammpsCalculator.N_FITTED_PARAMS  # 8

        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Kolmogorov_Crespi_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Kolmogorov_Crespi_best_fit_params.npz")
            params["energy"] = data["params"]
            bounds["energy"] = data["bounds"]
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
            # Back-compat: old cache files have no shift param — append it.
            if len(params["energy"]) == _n_kc:
                params["energy"], bounds["energy"] = _append_shift_param(
                    params["energy"], bounds["energy"], ydata_train["energy"]
                )
            calc_obj = KolmogorovCrespiLammpsCalculator(params["energy"], elements=["C"])
            calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])
        else:
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Kolmogorov_Crespi_best_fit_params_estimate.npz")
            p0 = data["params"]
            bounds["energy"] = data["bounds"]
            p0, bounds["energy"] = _append_shift_param(
                p0, bounds["energy"], ydata_train["energy"]
            )
            calc_obj = KolmogorovCrespiLammpsCalculator(p0, elements=["C"])
            calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])
            params["energy"], ypred_bestfit["energy"] = fit_model(
                calc["energy"], xdata_train["energy"], ydata_train["energy"], p0,
                ydata_forces=ydata_train["forces"], zero_shift_data=False, bounds=bounds["energy"],
            )
            ypred_bestfit["energy"] = _energy_ypred_for_cache(ypred_bestfit["energy"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/Kolmogorov_Crespi_best_fit_params.npz",
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
            )

        if not skip_diagnostics:
            layer_sep = []
            for atoms in xdata_train["energy"]:
                layer_sep.append(np.abs(atoms.positions[0, 2] - atoms.positions[3, 2]))
            plt.scatter(layer_sep, ypred_bestfit["energy"] - np.min(ypred_bestfit["energy"]))
            plt.scatter(layer_sep, ydata_train["energy"] - np.min(ydata_train["energy"]))
            plt.ylim(-0.05, 0.3)
            plt.show()

            _, _, qmc_xdata, _, _, qmc_data = load_data_for_model(
                model_name, supercells, level_of_theory="QMC"
            )
            calc_obj.set_parameters(params["energy"].tolist())
            qmc_energy, _ = calc_obj.evaluate_batch(list(qmc_xdata["energy"]))
            qmc_layer_sep = [
                np.abs(a.positions[0, 2] - a.positions[3, 2])
                for a in qmc_xdata["energy"]
            ]
            n_atoms = len(qmc_xdata["energy"][0])
            plt.scatter(qmc_layer_sep, (qmc_energy - np.min(qmc_energy)) / n_atoms - 0.021)
            plt.scatter(
                qmc_layer_sep,
                (np.asarray(qmc_data["energy"]) - np.min(qmc_data["energy"])) / n_atoms - 0.021,
            )
            plt.show()
    
    elif model_name == "DRIP":
        _n_drip = DRIPLammpsCalculator.N_FITTED_PARAMS  # 8

        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/DRIP_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/DRIP_best_fit_params.npz")
            params["energy"] = data["params"]
            bounds["energy"] = data["bounds"]
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
            # Back-compat: old cache files have no shift param — append it.
            if len(params["energy"]) == _n_drip:
                params["energy"], bounds["energy"] = _append_shift_param(
                    params["energy"], bounds["energy"], ydata_train["energy"]
                )
            calc_obj = DRIPLammpsCalculator(params["energy"], elements=["C"])
            calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])
        else:
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/DRIP_best_fit_params_estimate.npz")
            p0 = data["params"]
            bounds["energy"] = data["bounds"]
            p0, bounds["energy"] = _append_shift_param(
                p0, bounds["energy"], ydata_train["energy"]
            )
            calc_obj = DRIPLammpsCalculator(p0, elements=["C"])
            calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])
            params["energy"], ypred_bestfit["energy"] = fit_model(
                calc["energy"], xdata_train["energy"], ydata_train["energy"], p0,
                ydata_forces=ydata_train["forces"], zero_shift_data=False, bounds=bounds["energy"],
            )
            ypred_bestfit["energy"] = _energy_ypred_for_cache(ypred_bestfit["energy"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/DRIP_best_fit_params.npz",
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
            )

        if not skip_diagnostics:
            layer_sep = []
            for atoms in xdata_train["energy"]:
                layer_sep.append(np.abs(atoms.positions[0, 2] - atoms.positions[3, 2]))
            plt.scatter(layer_sep, ypred_bestfit["energy"] - np.min(ypred_bestfit["energy"]))
            plt.scatter(layer_sep, ydata_train["energy"] - np.min(ydata_train["energy"]))
            plt.show()

            _, _, qmc_xdata, _, _, qmc_data = load_data_for_model(
                model_name, 5, level_of_theory="QMC"
            )
            calc_obj.set_parameters(params["energy"].tolist())
            qmc_energy, _ = calc_obj.evaluate_batch(list(qmc_xdata["energy"]))
            qmc_layer_sep = [
                np.abs(a.positions[0, 2] - a.positions[3, 2])
                for a in qmc_xdata["energy"]
            ]
            n_atoms = len(qmc_xdata["energy"][0])
            plt.scatter(qmc_layer_sep, (qmc_energy - np.min(qmc_energy)) / n_atoms - 0.021)
            plt.show()

    elif model_name == "Tersoff+DRIP":
        # Parameter vector layout: [DRIP params (N_DRIP=8), Tersoff params (14), shift (1)]
        # TersoffDRIPLammpsCalculator evaluates both via hybrid/overlay in a single LAMMPS run.
        _nd = TersoffDRIPLammpsCalculator.N_FITTED_PARAMS   # 8  — DRIP split index
        _nt = TersoffDRIPLammpsCalculator.N_TERSOFF_PARAMS  # 14
        _n_phys = _nd + _nt                                 # 22

        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+DRIP_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+DRIP_best_fit_params.npz", allow_pickle=True)
            params["energy"] = data["params"]
            bounds["energy"] = data["bounds"]
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
            # Back-compat: old cache has no shift — append it.
            if len(params["energy"]) == _n_phys:
                params["energy"], bounds["energy"] = _append_shift_param(
                    params["energy"], bounds["energy"], ydata_train["energy"]
                )
        else:
            drip_data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/DRIP_best_fit_params_estimate.npz")
            tersoff_data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params_estimate.npz")
            p0 = np.concatenate([drip_data["params"][:_nd], tersoff_data["params"][:_nt]])
            b0 = np.vstack([drip_data["bounds"][:_nd], tersoff_data["bounds"][:_nt]])
            params["energy"], bounds["energy"] = _append_shift_param(
                p0, b0, ydata_train["energy"]
            )

        calc_obj = TersoffDRIPLammpsCalculator(
            tersoff_params=params["energy"][_nd:_n_phys].tolist(),
            drip_params=params["energy"][:_nd].tolist(),
            elements=["C"],
            shift=float(params["energy"][-1]),
        )

        def _set_tersoff_drip(params_flat):
            p = np.asarray(params_flat, dtype=np.float64)
            calc_obj.set_parameters(
                tersoff_params=p[_nd:_n_phys].tolist(),
                drip_params=p[:_nd].tolist(),
                shift=float(p[-1]),
            )

        _register_uq_lammps_runtime(calc_obj, _set_tersoff_drip)
        calc["energy"] = _make_batch_evaluator(
            calc_obj, xdata["energy"], set_params_fn=_set_tersoff_drip,
        )

        if not os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+DRIP_best_fit_params.npz"):
            params["energy"], ypred_bestfit["energy"] = fit_model(
                calc["energy"], xdata_train["energy"], ydata_train["energy"],
                params["energy"],
                ydata_forces=ydata_train["forces"], zero_shift_data=False,
                bounds=bounds["energy"],
            )
            ypred_bestfit["energy"] = _energy_ypred_for_cache(ypred_bestfit["energy"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+DRIP_best_fit_params.npz",
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
            )

        if not skip_diagnostics:
            layer_sep = [
                np.abs(a.positions[0, 2] - a.positions[3, 2])
                for a in xdata_train["energy"]
            ]
            plt.scatter(layer_sep, ypred_bestfit["energy"] )
            plt.scatter(layer_sep, ydata_train["energy"] )
            plt.ylim(-0.05, 0.3)
            plt.show()
            plt.clf()

            ydata_train_shifted = ydata_train["energy"]-np.mean(ydata_train["energy"])
            ypred_bestfit_shifted = ypred_bestfit["energy"]-np.mean(ypred_bestfit["energy"])
            plt.scatter(ydata_train["energy"]-np.mean(ydata_train["energy"]),ypred_bestfit["energy"]-np.mean(ypred_bestfit["energy"]))
            plt.plot(np.linspace(np.min(ydata_train_shifted),np.max(ydata_train_shifted),100),np.linspace(np.min(ydata_train_shifted),np.max(ydata_train_shifted),100))
            plt.xlabel("DFT energy (eV)")
            plt.ylabel("Tersoff+DRIP energy (eV)")
            plt.savefig("figures/Tersoff+DRIP_energy_scatter.png")
            plt.clf()

            _, _, qmc_xdata, _, _, qmc_data = load_data_for_model(
                model_name, 2, level_of_theory="QMC"
            )
            _set_tersoff_drip(params["energy"])
            qmc_energy, _ = calc_obj.evaluate_batch(list(qmc_xdata["energy"]))
            qmc_energy = np.asarray(qmc_energy, dtype=float)
            qmc_e = np.asarray(qmc_data["energy"], dtype=float)
            n_atoms_arr = np.array([len(a) for a in qmc_xdata["energy"]], dtype=float)
            qmc_layer_sep = [
                np.abs(a.positions[0, 2] - a.positions[3, 2])
                for a in qmc_xdata["energy"]
            ]
            y_model = (qmc_energy - np.nanmin(qmc_energy)) / n_atoms_arr - 0.021
            y_qmc = (qmc_e - np.nanmin(qmc_e)) / n_atoms_arr - 0.021
            mask = np.isfinite(y_model) & np.isfinite(y_qmc)
            plt.scatter(np.asarray(qmc_layer_sep)[mask], y_model[mask], label="Tersoff+DRIP")
            plt.scatter(np.asarray(qmc_layer_sep)[mask], y_qmc[mask], label="QMC")
            plt.legend()
            plt.show()

    elif model_name == "Tersoff+Kolmogorov_Crespi":
        # Parameter vector layout: [KC params (N_KC=8), Tersoff params (14), shift (1)]
        # TersoffKCLammpsCalculator uses hybrid/overlay in a single LAMMPS run.
        _nkc = TersoffKCLammpsCalculator.N_KC_PARAMS        # 8
        _nt  = TersoffKCLammpsCalculator.N_TERSOFF_PARAMS   # 14
        _n_phys = _nkc + _nt                                # 22

        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+Kolmogorov_Crespi_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+Kolmogorov_Crespi_best_fit_params.npz")
            params["energy"] = data["params"]
            bounds["energy"] = data["bounds"]
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
            # Back-compat: old cache has no shift — append it.
            if len(params["energy"]) == _n_phys:
                params["energy"], bounds["energy"] = _append_shift_param(
                    params["energy"], bounds["energy"], ydata_train["energy"]
                )
        else:
            kc_data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Kolmogorov_Crespi_best_fit_params_estimate.npz")
            tersoff_data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params_estimate.npz")
            kc_p0 = kc_data["params"][:_nkc]
            p0 = np.concatenate([kc_p0, tersoff_data["params"][:_nt]])
            b0 = np.vstack([kc_data["bounds"][:_nkc], tersoff_data["bounds"][:_nt]])
            params["energy"], bounds["energy"] = _append_shift_param(
                p0, b0, ydata_train["energy"]
            )

        calc_obj = TersoffKCLammpsCalculator(
            tersoff_params=params["energy"][_nkc:_n_phys].tolist(),
            kc_params=params["energy"][:_nkc].tolist(),
            elements=["C"],
            shift=float(params["energy"][-1]),
        )

        def _set_tersoff_kc(params_flat):
            p = np.asarray(params_flat, dtype=np.float64)
            calc_obj.set_parameters(
                tersoff_params=p[_nkc:_n_phys].tolist(),
                kc_params=p[:_nkc].tolist(),
                shift=float(p[-1]),
            )

        _register_uq_lammps_runtime(calc_obj, _set_tersoff_kc)
        calc["energy"] = _make_batch_evaluator(
            calc_obj, xdata["energy"], set_params_fn=_set_tersoff_kc,
        )

        if not os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+Kolmogorov_Crespi_best_fit_params.npz"):
            params["energy"], ypred_bestfit["energy"] = fit_model(
                calc["energy"], xdata_train["energy"], ydata_train["energy"],
                params["energy"],
                ydata_forces=ydata_train["forces"], zero_shift_data=False,
                bounds=bounds["energy"],
            )
            ypred_bestfit["energy"] = _energy_ypred_for_cache(ypred_bestfit["energy"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff+Kolmogorov_Crespi_best_fit_params.npz",
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
            )

        if not skip_diagnostics:
            layer_sep = [
                np.abs(a.positions[0, 2] - a.positions[3, 2])
                for a in xdata_train["energy"]
            ]
            plt.scatter(layer_sep, ypred_bestfit["energy"] - np.min(ypred_bestfit["energy"]))
            plt.scatter(layer_sep, ydata_train["energy"] - np.min(ydata_train["energy"]))
            plt.show()

            _, _, qmc_xdata, _, _, qmc_data = load_data_for_model(
                model_name, 5, level_of_theory="QMC"
            )
            _set_tersoff_kc(params["energy"])
            qmc_energy, _ = calc_obj.evaluate_batch(list(qmc_xdata["energy"]))
            qmc_energy = np.asarray(qmc_energy, dtype=float)
            n_atoms = len(qmc_xdata["energy"][0])
            qmc_layer_sep = [
                np.abs(a.positions[0, 2] - a.positions[3, 2])
                for a in qmc_xdata["energy"]
            ]
            plt.scatter(qmc_layer_sep, (qmc_energy - np.min(qmc_energy)) / n_atoms - 0.021)
            plt.show()

    elif model_name == "Tersoff":
        _n_tersoff = TersoffLammpsCalculator.N_FITTED_PARAMS  # 14

        if os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params.npz"):
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params.npz")
            params["energy"] = data["params"]
            bounds["energy"] = data["bounds"]
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
            # Back-compat: old cache has no shift — append it.
            if len(params["energy"]) == _n_tersoff:
                params["energy"], bounds["energy"] = _append_shift_param(
                    params["energy"], bounds["energy"], ydata_train["energy"]
                )
        else:
            data = np.load(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params_estimate.npz")
            params["energy"] = data["params"]
            bounds["energy"] = data["bounds"]
            params["energy"], bounds["energy"] = _append_shift_param(
                params["energy"], bounds["energy"], ydata_train["energy"]
            )

        calc_obj = TersoffLammpsCalculator(params["energy"], elements=["C"])
        calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])

        if not os.path.exists(f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params.npz"):
            params["energy"], ypred_bestfit["energy"] = fit_model(
                calc["energy"], xdata_train["energy"], ydata_train["energy"],
                params["energy"], yshift=0, zero_shift_data=False, bounds=bounds["energy"],
            )
            ypred_bestfit["energy"] = _energy_ypred_for_cache(ypred_bestfit["energy"])
            np.savez(
                f"{_BEST_FIT_PARAMS_SUBDIR}/Tersoff_best_fit_params.npz",
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
            )

        if not skip_diagnostics:
            e_dft = np.asarray(ydata_train["energy"])
            e_pred = np.asarray(ypred_bestfit["energy"])
            plt.scatter(e_dft - np.min(e_dft), e_pred - np.min(e_pred))
            lim = np.max(e_dft - np.min(e_dft))
            plt.plot([0, lim], [0, lim])
            plt.show()

            
    elif model_name.startswith("TETB_POD"):
        tb_hyperparams, pod_hyperparams, tetb_tag = (
            build_tetb_pod_hyperparams_from_data_kw(data_kw)
        )
        _tb_M = int(tb_hyperparams.get("M", tb_hyperparams.get("acsf_M", 10)))
        _tb_W = int(tb_hyperparams.get("W", tb_hyperparams.get("acsf_W", 3)))

        pod_cutoff = float(data_kw.get("pod_cutoff", 6.0))
        # Omitted / None → calculator auto k-mesh; set ``kpoints`` to [nx,ny,nz] or (n_k,3) Cartesian to override.
        kpoints = data_kw.get("kpoints", None)
        tb_solver_method = str(data_kw.get("tb_solver_method", "diagonalization"))
        ewald_cutoff = float(data_kw.get("ewald_cutoff", 12.0))
        pppm_accuracy = float(data_kw.get("pppm_accuracy", 1e-4))
        valence_charge = float(data_kw.get("valence_charge", 1.0))

        # ── Resolve TB M, W for cache file naming ────────────────────────────
        tb_mw_tag   = f"M_{_tb_M}_W_{_tb_W}"
        tb_best_npz = f"{_BEST_FIT_PARAMS_SUBDIR}/ACSF_hoppings_{tb_mw_tag}_best_fit_params.npz"

        # Pre-fitted ACSF weights (optional — :func:`fit_tetb_residual_pod` can fit
        # from ``xdata_train[\"hopping\"]`` / ``ydata_train[\"hopping\"]`` if absent).
        tb_bounds_arr = None
        if os.path.exists(tb_best_npz):
            tb_data = np.load(tb_best_npz, allow_pickle=True)
            tb_params_arr = np.asarray(tb_data["params"], dtype=np.float64)
            if "bounds" in tb_data.files:
                tb_bounds_arr = np.asarray(tb_data["bounds"], dtype=float)
        else:
            tb_params_arr = None

        n_pod = ncoeff_from_params(pod_hyperparams)
        tetb_cache = (
            f"{_BEST_FIT_PARAMS_SUBDIR}/TETB_POD_{tetb_tag}_best_fit_params.npz"
        )
        legacy_tetb = f"{_BEST_FIT_PARAMS_SUBDIR}/TETB_POD_best_fit_params.npz"
        tetb_path = None
        if os.path.isfile(tetb_cache):
            tetb_path = tetb_cache
        elif os.path.isfile(legacy_tetb):
            tetb_path = legacy_tetb
            warnings.warn(
                f"TETB_POD: loading legacy {legacy_tetb!r}; new fits are saved as "
                f"{os.path.basename(tetb_cache)!r}.",
                UserWarning,
                stacklevel=2,
            )
        had_tetb_cache = tetb_path is not None

        if had_tetb_cache:
            cache_data = np.load(tetb_path, allow_pickle=True)
            pe = np.asarray(cache_data["params"], dtype=np.float64).ravel()
            be = np.asarray(cache_data["bounds"], dtype=np.float64)
            if pe.shape[0] == n_pod + 1:
                warnings.warn(
                    "TETB_POD: ignoring legacy last parameter in cached "
                    f"{tetb_path!r} (duplicate energy shift vs DFT totals). "
                    "Delete the cache and refit if energies still look wrong.",
                    UserWarning,
                    stacklevel=2,
                )
                pe = pe[:n_pod]
            if be.shape[0] == n_pod + 1:
                be = be[:n_pod]
            params["energy"] = pe
            bounds["energy"] = be
            if tb_params_arr is None:
                raise FileNotFoundError(
                    f"TETB_POD: cache {tetb_path!r} exists but {tb_best_npz!r} is missing; "
                    "restore the ACSF cache or delete the TETB_POD cache to refit."
                )
        else:
            # Residual POD via LAMMPS fitpod (TB+Ewald subtracted from DFT first).
            tb_out, pod_coeffs, _ = fit_tetb_residual_pod(
                list(xdata_train["energy"]),
                ydata_train["energy"],
                ydata_train["forces"],
                M=int(_tb_M),
                W=int(_tb_W),
                r_cut=float(tb_hyperparams["r_cut"]),
                pod_hyperparams=pod_hyperparams,
                pod_cutoff=pod_cutoff,
                kpoints=kpoints,
                tb_solver_method=tb_solver_method,
                valence_charge=valence_charge,
                ewald_cutoff=ewald_cutoff,
                pppm_accuracy=pppm_accuracy,
                best_fit_dir="best_fit_params",
                xdata_hopping=xdata_train.get("hopping"),
                ydata_hopping=ydata_train.get("hopping"),
                tb_params=tb_params_arr,
                elements=["C"],
                lammps_exec=data_kw.get("lammps_exec"),
                ridge_acsf=float(data_kw.get("ridge_acsf", 0.1)),
            )
            tb_params_arr = tb_out
            params["energy"] = np.asarray(pod_coeffs, dtype=np.float64).reshape(-1)
            bounds["energy"] = np.column_stack(
                [np.full(n_pod, -1e4, dtype=np.float64),
                 np.full(n_pod,  1e4, dtype=np.float64)],
            )

        # ── ACSF hopping block for EMCEE (joint cost over hoppings + POD energy + forces)
        if tb_bounds_arr is None or (
            hasattr(tb_bounds_arr, "shape") and tb_bounds_arr.shape[0] != len(tb_params_arr)
        ):
            if os.path.exists(tb_best_npz):
                td = np.load(tb_best_npz, allow_pickle=True)
                if "bounds" in td.files:
                    tb_bounds_arr = np.asarray(td["bounds"], dtype=float)
        n_tb_acsf = int(np.asarray(tb_params_arr).size)
        if tb_bounds_arr is None or tb_bounds_arr.shape[0] != n_tb_acsf:
            tb_bounds_arr = np.array([[-1e8, 1e8]] * n_tb_acsf, dtype=float)
        params["hopping"] = np.asarray(tb_params_arr, dtype=np.float64).reshape(-1)
        bounds["hopping"] = np.asarray(tb_bounds_arr, dtype=float).reshape(n_tb_acsf, 2)
        ypred_bestfit["hopping"] = get_prediction(
            get_acsf_hoppings, xdata_train["hopping"], params["hopping"]
        )
        calc["hopping"] = get_acsf_hoppings

        # ── Build the calculator ─────────────────────────────────────────────
        calc_obj = TETB_PODLammpsCalculator(
            tb_params      = tb_params_arr,
            pod_params     = params["energy"][:n_pod],
            tb_hyperparams = tb_hyperparams,
            pod_hyperparams= pod_hyperparams,
            pod_cutoff     = pod_cutoff,
            elements       = ["C"],
            kpoints        = kpoints,
            tb_solver_method = tb_solver_method,
            ewald_cutoff   = ewald_cutoff,
            pppm_accuracy  = pppm_accuracy,
            valence_charge = valence_charge,
            shift          = 0.0,
        )

        def _set_tetb_joint(params_flat):
            """Joint MCMC: ``n_tb + n_pod`` or optional ``+1`` shift; POD-only: ``n_pod`` or ``+1``."""
            p = np.asarray(params_flat, dtype=np.float64).ravel()
            npl = int(n_pod)
            if p.size == n_tb_acsf + npl + 1:
                calc_obj.set_tb_parameters(p[:n_tb_acsf])
                calc_obj.set_parameters(p[n_tb_acsf:])
            elif p.size == n_tb_acsf + npl:
                calc_obj.set_tb_parameters(p[:n_tb_acsf])
                calc_obj.set_parameters(p[n_tb_acsf:])
            elif p.size == npl + 1 or p.size == npl:
                calc_obj.set_parameters(p)
            else:
                raise ValueError(
                    f"TETB_POD: expected {n_tb_acsf + npl} or {n_tb_acsf + npl + 1} "
                    f"(TB+POD [+shift]) or {npl} / {npl + 1} (POD only), got {p.size}."
                )

        _register_uq_lammps_runtime(calc_obj, _set_tetb_joint)
        calc["energy"] = _make_batch_evaluator(
            calc_obj, xdata["energy"], set_params_fn=_set_tetb_joint,
        )

        # Best-fit predictions on the training set (same pattern as POD_energy).
        calc_obj.set_parameters(params["energy"].tolist())
        e_bf, f_bf = calc_obj.evaluate_batch(list(xdata_train["energy"]))
        ypred_bestfit["energy"] = e_bf
        ypred_bestfit["forces"] = f_bf
        if not had_tetb_cache:
            np.savez(
                tetb_cache,
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
            )

    elif model_name.startswith("Allegro_energy"):
        import blg_model_builder.energy_registry  # noqa: F401 — register Allegro_energy
        from blg_model_builder.model_registry import cache_basename, make_hyperparams

        hp = make_hyperparams(model_name, **data_kw)
        ckpt = hp["allegro_checkpoint"]
        r_max = hp["allegro_r_max"]
        device = hp["allegro_device"]
        bound_scale = hp["allegro_bound_scale"]
        ckpt_tag = hp["allegro_ckpt_tag"]

        cache_path = f"{_BEST_FIT_PARAMS_SUBDIR}/{cache_basename(model_name, hp)}.npz"

        calc_obj = AllegroCalculator(ckpt, r_max=r_max, device=device)

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            params["energy"] = np.asarray(data["params"], dtype=float)
            bounds["energy"] = np.asarray(data["bounds"], dtype=float)
            ypred_bestfit["energy"] = _energy_ypred_for_cache(data["ypred_bestfit"])
            calc_obj.set_parameters(params["energy"])
        else:
            params["energy"] = calc_obj.get_parameters()
            bounds["energy"] = allegro_bounds_from_params(
                params["energy"], bound_scale=bound_scale,
            )
            calc_obj.set_parameters(params["energy"])
            e_bf, f_bf = calc_obj.evaluate_batch(list(xdata_train["energy"]))
            ypred_bestfit["energy"] = e_bf
            ypred_bestfit["forces"] = f_bf
            np.savez(
                cache_path,
                params=params["energy"],
                bounds=bounds["energy"],
                ypred_bestfit=ypred_bestfit["energy"],
                allegro_checkpoint=ckpt,
                allegro_ckpt_tag=ckpt_tag,
                allegro_r_max=r_max,
            )

        _register_uq_lammps_runtime(calc_obj)
        calc["energy"] = _make_batch_evaluator(calc_obj, xdata["energy"])

    else:
        raise ValueError(f"Unknown model_name '{model_name}' for data loading")
    return calc, xdata_train, xdata_test, xdata, ydata_train, ydata_test, ydata, ypred_bestfit, params, bounds