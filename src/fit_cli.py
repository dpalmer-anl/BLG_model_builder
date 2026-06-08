"""
``fit_model`` CLI: fit one or more models and cache their best-fit parameters.

This runs only the *fitting* stage of the workflow — it calls
:func:`blg_model_builder.get_MCMC_inputs.get_MCMC_inputs`, which fits each model
to the training data and writes the best-fit cache under ``best_fit_params/``
(the same cache later reused by MCMC / SubSamp ensembles).  No sampling is done.

Any model-specific hyperparameter can be supplied on the command line via a
dedicated flag or the generic ``--set KEY=VALUE`` / bare ``--KEY VALUE`` syntax
(see :mod:`blg_model_builder.cli_hyperparams`).

Examples
--------
    python fit_model.py --models POD_energy --two_body_radial 2 --three_body_angular 4
    python fit_model.py --models ACSF_hoppings -M 10 -W 6 --acsf_r_cut 7.0
    python fit_model.py --models Allegro_energy --allegro_bound_scale 50
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from blg_model_builder.cli_hyperparams import add_hyperparam_args, collect_hyperparams
from blg_model_builder.get_MCMC_inputs import (
    build_tetb_pod_hyperparams_from_data_kw,
    get_MCMC_inputs,
)


def _ravel(y: Any) -> np.ndarray:
    if isinstance(y, list):
        parts = [np.asarray(b, dtype=float).ravel() for b in y]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    arr = np.asarray(y)
    if arr.dtype == object:
        parts = [np.asarray(b, dtype=float).ravel() for b in arr.ravel()]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    return arr.ravel().astype(float)


def _mae(y_true: Any, y_pred: Any) -> float:
    a = _ravel(y_true)
    b = _ravel(y_pred)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _expand_model_name(model_name: str, args, mcmc_kw: Dict[str, Any]) -> str:
    """Apply the same name tagging used by the MCMC / SubSamp drivers."""
    if model_name in ("ACSF_hoppings", "ACSF_hoppings_sk", "POD_energy", "KC_energy"):
        if model_name == "POD_energy" and args.pod_index is not None:
            from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

            _, _, pod_hash = pod_hyperparams_for_index(int(args.pod_index))
            return f"POD_energy_POD_index_{int(args.pod_index)}_{pod_hash}"
        return f"{model_name}_M_{args.M}_W_{args.W}"
    if model_name == "TETB_POD":
        _, _, tag = build_tetb_pod_hyperparams_from_data_kw(mcmc_kw)
        if args.pod_index is not None:
            from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

            _, _, pod_hash = pod_hyperparams_for_index(int(args.pod_index))
            return f"TETB_POD_{tag}_POD_index_{int(args.pod_index)}_{pod_hash}"
        return f"TETB_POD_{tag}"
    if model_name == "Allegro_energy":
        from blg_model_builder.allegro_interface import (
            checkpoint_tag,
            resolve_allegro_checkpoint,
        )

        ckpt = resolve_allegro_checkpoint(args.allegro_checkpoint)
        return f"Allegro_energy_ckpt_{checkpoint_tag(ckpt)}"
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit model(s) and cache best-fit parameters (no sampling).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-m", "--models", nargs="+", required=True,
        help="One or more model names (e.g. POD_energy ACSF_hoppings Allegro_energy).",
    )
    parser.add_argument("-M", "--M", type=int, default=10)
    parser.add_argument("-W", "--W", type=int, default=6)
    parser.add_argument("--supercells", type=int, default=None,
                        help="Supercell multiplier (default: 2 for DRIP-based, else 1).")
    parser.add_argument("--nn-val", type=int, default=None, dest="nn_val",
                        help="Required for LETB_intralayer models (1, 2, or 3).")
    parser.add_argument("--level-of-theory", type=str, default="rVV10",
                        dest="level_of_theory")
    parser.add_argument("--POD-index", type=int, default=None, dest="pod_index",
                        help="Select a POD descriptor by index into use_pod_models_hash.txt.")
    parser.add_argument("--allegro-checkpoint", type=str, default=None,
                        dest="allegro_checkpoint",
                        help="Path to a trained Allegro .ckpt (for Allegro_energy).")
    parser.add_argument("--allegro-r-max", type=float, default=None,
                        dest="allegro_r_max",
                        help="Neighbor cutoff (Å) for the Allegro calculator.")
    add_hyperparam_args(parser)
    args, _unknown = parser.parse_known_args()
    cli_hyperparams = collect_hyperparams(args, _unknown)
    if cli_hyperparams:
        print(f"[fit] CLI hyperparameters: {cli_hyperparams}", flush=True)

    failures: List[str] = []
    for raw_name in args.models:
        kw: Dict[str, Any] = {"M": args.M, "W": args.W}
        if args.pod_index is not None:
            from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

            pod_hp, pod_cutoff, pod_hash = pod_hyperparams_for_index(int(args.pod_index))
            kw["pod_hyperparams"] = pod_hp
            kw["pod_cutoff"] = pod_cutoff
            kw["pod_hash"] = pod_hash
        if args.allegro_checkpoint is not None:
            kw["allegro_checkpoint"] = args.allegro_checkpoint
        if args.allegro_r_max is not None:
            kw["allegro_r_max"] = args.allegro_r_max
        if args.nn_val is not None:
            kw["nn_val"] = args.nn_val
        kw["level_of_theory"] = args.level_of_theory
        kw.update(cli_hyperparams)

        model_name = _expand_model_name(raw_name, args, kw)
        sc = args.supercells
        if sc is None:
            sc = 2 if raw_name in ("DRIP", "Tersoff+DRIP") else 1

        print(f"\n=== Fitting {model_name} (supercells={sc}) ===", flush=True)
        try:
            out = get_MCMC_inputs(model_name, supercells=sc, **kw)
            calc, _xtr, _xte, _x, ytrain, _yte, _y, ypred_bestfit, params, _bounds = out
            n_params = sum(len(params[k]) for k in calc)
            print(f"  fitted: n_params={n_params}", flush=True)
            for key in calc:
                if key in ypred_bestfit and key in ytrain:
                    print(f"  {key} best-fit MAE = {_mae(ytrain[key], ypred_bestfit[key]):.6g}",
                          flush=True)
        except Exception as exc:  # noqa: BLE001 — report per-model and continue
            import traceback

            failures.append(raw_name)
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    if failures:
        raise SystemExit(f"Fitting failed for: {failures}")
    print("\nAll requested models fitted.", flush=True)


if __name__ == "__main__":
    main()
