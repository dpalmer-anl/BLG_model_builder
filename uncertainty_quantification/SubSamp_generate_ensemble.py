"""
Bootstrap-style ensemble from repeated random subsamples of the *training* set.

Loads data and calculators via :func:`get_MCMC_inputs.get_MCMC_inputs` (same
supported ``model_name`` values as MCMC). Each ensemble member refits on a
random fraction ``p_subset`` of the training data. Saves under
``ensembles/<model_name>/<model_name>_SubSamp_ensemble_p_<p>.pkl`` with the same
payload shape as :mod:`EMCEE_generate_ensemble` (including test-set predictions
when derivable). For ``TETB_POD``, ``model_name`` is expanded to
``TETB_POD_tb_M_<m>_W_<w>_pod_M_<pm>_W_<pw>`` to match best-fit caches and MCMC outputs.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from blg_model_builder_v2.potentials import pod_hyperparams_to_str

from EMCEE_generate_ensemble import evaluate_ensemble, slice_ypred_test_from_full
from get_MCMC_inputs import get_MCMC_inputs, build_tetb_pod_hyperparams_from_data_kw
from model_fit import (
    fit_acsf_linear_hopping,
    fit_model,
    fit_pod,
    fit_tetb_residual_pod,
)


def _train_n_configs(ytrain: Dict[str, Any], calc: Dict[str, Any]) -> int:
    """Number of training structures (prefer ``energy`` list length if present)."""
    if "energy" in calc and "energy" in ytrain:
        ye = ytrain["energy"]
        if isinstance(ye, list):
            return len(ye)
        return int(np.asarray(ye).shape[0])
    key = next(iter(calc))
    yk = ytrain[key]
    if isinstance(yk, list):
        return len(yk)
    return int(np.asarray(yk).shape[0])


def _subset_indices(n: int, p_subset: float, rng: np.random.Generator) -> np.ndarray:
    n_select = int(round(n * float(p_subset)))
    n_select = max(1, min(n, n_select))
    return rng.choice(np.arange(n, dtype=int), size=n_select, replace=False)


def _list_subset(seq: List[Any], idx: np.ndarray) -> List[Any]:
    return [seq[int(i)] for i in idx]


def _resolve_fit_kind(model_name: str, calc: Dict[str, Any]) -> str:
    if "hopping" in calc and "energy" in calc:
        return "tetb_pod"
    if model_name.startswith("POD_energy"):
        return "pod_energy"
    if model_name.startswith("ACSF_hoppings"):
        return "acsf_hoppings"
    return "generic"


def _pod_energy_hyperparams_str(M: int, W: int) -> str:
    """Mirror ``get_MCMC_inputs`` ``POD_energy`` descriptor dict (same ``M``, ``W`` keys)."""
    rcut = 6.0
    data_kw = {"M": int(M), "W": int(W)}
    hyperparams = {
        "species": ["C"],
        "bessel_polynomial_degree": 4,
        "inverse_polynomial_degree": 8,
        "twobody_number_radial_basis_functions": int(data_kw.get("M", data_kw.get("M", 10))),
        "threebody_number_radial_basis_functions": int(data_kw.get("M", data_kw.get("M", 3))),
        "threebody_angular_degree": int(data_kw.get("W", data_kw.get("W", 4))),
        "fourbody_number_radial_basis_functions": int(data_kw.get("M", data_kw.get("M", 2))),
        "fourbody_angular_degree": int(data_kw.get("W", data_kw.get("W", 3))),
        "fivebody_number_radial_basis_functions": 4,
        "fivebody_angular_degree": 3,
        "sixbody_number_radial_basis_functions": 3,
        "sixbody_angular_degree": 2,
        "sevenbody_number_radial_basis_functions": 2,
        "sevenbody_angular_degree": 2,
    }
    return pod_hyperparams_to_str(hyperparams, rcut, ["C"])


_DEFAULT_LMP = "/mnt/c/Users/Daniel/Documents/research/lammps/build/lmp"


def get_subsamp_ensemble(
    *,
    fit_kind: str,
    xtrain: Dict[str, Any],
    ytrain: Dict[str, Any],
    calc: Dict[str, Any],
    params: Dict[str, np.ndarray],
    bounds: Dict[str, np.ndarray],
    p_subset: float,
    nsamples: int,
    rng: np.random.Generator,
    pod_hyperparams_str: Optional[str],
    lammps_exec: Optional[str],
    tetb_pod_kw: Optional[Dict[str, Any]],
    ridge_acsf: float,
) -> Dict[str, np.ndarray]:
    """Return dict of stacked parameter samples (one row per subsample refit)."""
    n_cfg = _train_n_configs(ytrain, calc)
    blocks: Dict[str, List[np.ndarray]] = {k: [] for k in calc}

    for _ in tqdm(range(nsamples), desc="SubSamp refits"):
        idx = _subset_indices(n_cfg, p_subset, rng)

        if fit_kind == "tetb_pod":
            kw = dict(tetb_pod_kw or {})
            atoms_sub = _list_subset(list(xtrain["energy"]), idx)
            e_sub = np.asarray([ytrain["energy"][int(i)] for i in idx], dtype=float)
            f_sub = [ytrain["forces"][int(i)] for i in idx]
            xhop_sub = _list_subset(list(xtrain["hopping"]), idx)
            yhop_sub = [ytrain["hopping"][int(i)] for i in idx]
            tb_out, pod_coeffs, _ = fit_tetb_residual_pod(
                atoms_sub,
                e_sub,
                f_sub,
                xdata_hopping=xhop_sub,
                ydata_hopping=yhop_sub,
                lammps_exec=lammps_exec,
                **kw,
            )
            n_pod = int(np.asarray(pod_coeffs, dtype=float).size)
            energy_params = np.asarray(pod_coeffs, dtype=np.float64).reshape(-1)
            blocks["hopping"].append(np.asarray(tb_out, dtype=np.float64).reshape(-1))
            blocks["energy"].append(energy_params)

        elif fit_kind == "pod_energy":
            if pod_hyperparams_str is None:
                raise ValueError("pod_hyperparams_str is required for POD_energy")
            atoms_sub = _list_subset(list(xtrain["energy"]), idx)
            le = lammps_exec if lammps_exec is not None else _DEFAULT_LMP
            pod_p = fit_pod(pod_hyperparams_str, atoms_sub, lammps_exec=le)
            blocks["energy"].append(np.asarray(pod_p, dtype=np.float64).reshape(-1))

        elif fit_kind == "acsf_hoppings":
            x_sub = [xtrain["hopping"][int(i)] for i in idx]
            y_sub = [ytrain["hopping"][int(i)] for i in idx]
            w, _ = fit_acsf_linear_hopping(x_sub, y_sub, ridge=ridge_acsf)
            blocks["hopping"].append(np.asarray(w, dtype=np.float64).reshape(-1))

        else:
            for key in calc:
                xk = xtrain[key]
                yk = ytrain[key]
                if isinstance(yk, list):
                    x_sub = [xk[int(i)] for i in idx]
                    y_sub = [yk[int(i)] for i in idx]
                else:
                    x_sub = np.asarray(xk)[idx]
                    y_sub = np.asarray(yk)[idx]

                p0 = np.asarray(params[key], dtype=float).reshape(-1)
                b = np.asarray(bounds[key], dtype=float)
                yf_sub = None
                if key == "energy" and "forces" in ytrain:
                    yf_sub = [ytrain["forces"][int(i)] for i in idx]

                p_fit, _ = fit_model(
                    calc[key],
                    x_sub,
                    y_sub,
                    p0,
                    ydata_forces=yf_sub,
                    zero_shift_data=False,
                    bounds=b,
                )
                blocks[key].append(np.asarray(p_fit, dtype=np.float64).reshape(-1))

    return {k: np.vstack(blocks[k]) for k in calc}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subsample-training bootstrap ensembles (SubSamp tag).",
    )
    parser.add_argument("-m", "--model_name", type=str, default="ACSF_hoppings")
    parser.add_argument("-p", "--p_subset", type=float, default=0.5)
    parser.add_argument("-n", "--nsamples", type=int, default=30)
    parser.add_argument("-M", "--M", type=int, default=10)
    parser.add_argument("-W", "--W", type=int, default=6)
    parser.add_argument(
        "--tb-M", type=int, default=None, dest="tb_M",
        help="TETB_POD: ACSF hopping M (defaults to -M when unset).",
    )
    parser.add_argument(
        "--tb-W", type=int, default=None, dest="tb_W",
        help="TETB_POD: ACSF hopping W (defaults to -W when unset).",
    )
    parser.add_argument(
        "--pod-M", type=int, default=None, dest="pod_M",
        help="TETB_POD: POD primary M (defaults to -M when unset).",
    )
    parser.add_argument(
        "--pod-W", type=int, default=None, dest="pod_W",
        help="TETB_POD: POD primary W (defaults to -W when unset).",
    )
    parser.add_argument("--seed", type=int, default=135726)
    parser.add_argument(
        "--nn-val",
        type=int,
        default=None,
        dest="nn_val",
        help="Required for model_name starting with LETB_intralayer (nn_val=1,2,3).",
    )
    parser.add_argument(
        "--lammps-exec",
        type=str,
        default=None,
        help="LAMMPS binary for fit_pod / TETB residual POD (defaults to model_fit.fit_pod path if unset).",
    )
    parser.add_argument(
        "--pod-cutoff",
        type=float,
        default=6.0,
        help="POD cutoff (Å) for TETB_POD (passed to fit_tetb_residual_pod).",
    )
    parser.add_argument(
        "--ridge-acsf",
        type=float,
        default=0.1,
        help="Ridge λ for fit_acsf_linear_hopping (and TB refit inside TETB residual POD).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    model_name = args.model_name
    if model_name in ("ACSF_hoppings", "POD_energy", "KC_energy"):
        model_name = f"{model_name}_M_{args.M}_W_{args.W}"
    elif model_name == "TETB_POD":
        _tag_kw: Dict[str, Any] = {"M": args.M, "W": args.W}
        if args.tb_M is not None:
            _tag_kw["tb_M"] = args.tb_M
        if args.tb_W is not None:
            _tag_kw["tb_W"] = args.tb_W
        if args.pod_M is not None:
            _tag_kw["pod_M"] = args.pod_M
        if args.pod_W is not None:
            _tag_kw["pod_W"] = args.pod_W
        _, _, _tetb_tag = build_tetb_pod_hyperparams_from_data_kw(_tag_kw)
        model_name = f"TETB_POD_{_tetb_tag}"

    hyperparameters = None
    if model_name.startswith("LETB_intralayer"):
        if args.nn_val is None:
            raise SystemExit(
                "LETB_intralayer models require --nn-val (1, 2, or 3), matching get_MCMC_inputs."
            )
        hyperparameters = {"nn_val": int(args.nn_val)}

    if model_name == "DRIP" or model_name == "Tersoff+DRIP":
        sc = 2
    else:
        sc = 1

    gkw: Dict[str, Any] = {"supercells": sc, "M": args.M, "W": args.W}
    if args.tb_M is not None:
        gkw["tb_M"] = args.tb_M
    if args.tb_W is not None:
        gkw["tb_W"] = args.tb_W
    if args.pod_M is not None:
        gkw["pod_M"] = args.pod_M
    if args.pod_W is not None:
        gkw["pod_W"] = args.pod_W
    if hyperparameters is not None:
        gkw["hyperparameters"] = hyperparameters

    print("[SubSamp] loading:", model_name, "| kwargs:", gkw, flush=True)
    calc, xdata_train, xdata_test, xdata, ydata_train, ydata_test, ydata, ypred_bestfit, params, bounds = get_MCMC_inputs(
        model_name,
        **gkw,
    )

    fit_kind = _resolve_fit_kind(model_name, calc)
    pod_hyperparams_str = None
    if fit_kind == "pod_energy":
        pod_hyperparams_str = _pod_energy_hyperparams_str(args.M, args.W)

    tetb_pod_kw: Optional[Dict[str, Any]] = None
    if fit_kind == "tetb_pod":
        _fit_kw = {
            k: v for k, v in gkw.items()
            if k not in ("supercells", "hyperparameters")
        }
        tb_hp, pod_hp, _ = build_tetb_pod_hyperparams_from_data_kw(_fit_kw)
        tetb_pod_kw = {
            "M": int(tb_hp["M"]),
            "W": int(tb_hp["W"]),
            "r_cut": float(tb_hp["r_cut"]),
            "pod_hyperparams": pod_hp,
            "pod_cutoff": float(args.pod_cutoff),
            "kpoints": None,
            "tb_solver_method": "diagonalization",
            "valence_charge": 1.0,
            "ewald_cutoff": 12.0,
            "pppm_accuracy": 1e-4,
            "best_fit_dir": "best_fit_params",
            "elements": ["C"],
            "ridge_acsf": float(args.ridge_acsf),
        }

    lmp = args.lammps_exec
    if lmp is None and (fit_kind == "pod_energy" or fit_kind == "tetb_pod"):
        lmp = _DEFAULT_LMP

    ensemble_samples = get_subsamp_ensemble(
        fit_kind=fit_kind,
        xtrain=xdata_train,
        ytrain=ydata_train,
        calc=calc,
        params=params,
        bounds=bounds,
        p_subset=args.p_subset,
        nsamples=args.nsamples,
        rng=rng,
        pod_hyperparams_str=pod_hyperparams_str,
        lammps_exec=lmp,
        tetb_pod_kw=tetb_pod_kw,
        ridge_acsf=float(args.ridge_acsf),
    )

    print("[SubSamp] evaluating ensemble on full / test splits …", flush=True)
    ypred_samples, _ = evaluate_ensemble(ensemble_samples, xdata, ydata, calc)
    ens_keys = list(ensemble_samples.keys())
    ypred_samples_test = slice_ypred_test_from_full(
        ypred_samples, xdata, xdata_test, ens_keys,
    )
    if ypred_samples_test is None:
        ypred_samples_test, _ = evaluate_ensemble(
            ensemble_samples, xdata_test, ydata_test, calc,
        )

    os.makedirs(os.path.join("ensembles", model_name), exist_ok=True)
    filename = os.path.join(
        "ensembles",
        model_name,
        f"{model_name}_SubSamp_ensemble_p_{args.p_subset}.pkl",
    )
    ensemble_dict = {
        "ensemble": ensemble_samples,
        "ypred_samples": ypred_samples,
        "ypred_samples_test": ypred_samples_test,
        "ydata": ydata,
        "ydata_test": ydata_test,
        "xdata": xdata,
        "xdata_test": xdata_test,
        "model_name": model_name,
        "p_subset": float(args.p_subset),
        "nsamples": int(args.nsamples),
        "fit_kind": fit_kind,
        "ypred_bestfit": ypred_bestfit,
    }
    print("[SubSamp] saving:", filename, flush=True)
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "wb") as f:
        pickle.dump(ensemble_dict, f)


if __name__ == "__main__":
    main()
