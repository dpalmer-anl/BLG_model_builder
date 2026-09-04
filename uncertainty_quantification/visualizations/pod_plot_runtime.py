"""Shared POD calculator helpers for visualization scripts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from blg_model_builder.cli_model_names import expand_ensemble_model_name
from blg_model_builder.potentials import (
    PODD3LammpsCalculator,
    PODLammpsCalculator,
    ncoeff_from_params,
)

HERE = Path(__file__).resolve().parent
UQ_DIR = HERE.parent

import sys

if str(UQ_DIR) not in sys.path:
    sys.path.insert(0, str(UQ_DIR))

from uq_model_runtime import build_uq_calculator, mcmc_kw_for_model  # noqa: E402
from run_uq_propagation_relaxation import expand_models_for_relaxation  # noqa: E402

POD_FAMILY_PREFIXES = ("POD_energy", "PODD3_energy")


def chdir_for_dataloader() -> None:
    uq = UQ_DIR
    os.chdir(str(uq if uq.is_dir() else uq.parent))


def resolve_model_names(
    model_patterns: list[str],
    args,
    cli_hyperparams: dict,
    ensemble_dir: str,
) -> list[str]:
    expanded = expand_models_for_relaxation(model_patterns, ensemble_dir)
    seen: set[str] = set()
    out: list[str] = []
    for pattern in expanded:
        name = expand_ensemble_model_name(pattern, args, cli_hyperparams)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def is_pod_family_model(model_name: str) -> bool:
    return model_name.startswith(POD_FAMILY_PREFIXES)


def pod_family_load_name(model_name: str) -> str:
    if model_name.startswith("PODD3_energy"):
        return "PODD3_energy"
    if model_name.startswith("POD_energy"):
        return "POD_energy"
    raise ValueError(f"Not a POD-family model: {model_name!r}")


def model_plot_label(model_name: str, load_name: str) -> str:
    if model_name.startswith(load_name):
        return (
            model_name.split("_POD_index_")[0]
            if "_POD_index_" in model_name
            else model_name
        )
    return load_name


def energy_per_atom_total(atoms, energy_total: float) -> float:
    return float(energy_total) / len(atoms)


def evaluate_model_energies_on_atoms(calc_obj, atoms_list: list) -> np.ndarray:
    if not atoms_list:
        return np.array([], dtype=float)
    if hasattr(calc_obj, "prepare_batch"):
        calc_obj.prepare_batch(atoms_list)
        energies, _ = calc_obj.evaluate_batch()
        return np.asarray(energies, dtype=float).ravel()
    return np.asarray(
        [float(calc_obj.get_potential_energy(a)) for a in atoms_list],
        dtype=float,
    )


def _pod_best_fit_npz(model_name: str, extra_kw: dict | None) -> Path:
    load_name = pod_family_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), **(extra_kw or {})}
    pod_hp = data_kw.get("pod_hyperparams")
    if not isinstance(pod_hp, dict) or not pod_hp:
        raise ValueError(
            f"Could not resolve POD hyperparameters for {model_name!r}."
        )
    hyperparams = dict(pod_hp)
    hyperparams.setdefault("species", ["C"])
    ncoeffs = ncoeff_from_params(hyperparams)
    regularization = float(data_kw.get("regularization", 1e-12))
    include_intra = bool(data_kw.get("include_intralayer", False))
    pod_hash = str(data_kw.get("pod_hash", "")).strip()
    reg_tag = f"reg{regularization:.0e}"
    intra_tag = "_intra" if include_intra else ""
    hash_tag = f"_{pod_hash}" if pod_hash else ""
    path = (
        UQ_DIR
        / "best_fit_params"
        / f"{load_name}_{ncoeffs}_{reg_tag}{intra_tag}{hash_tag}_best_fit_params.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"{load_name} best-fit parameters not found: {path}")
    return path


def build_pod_family_calculator(
    model_name: str, extra_kw: dict | None,
) -> PODLammpsCalculator | PODD3LammpsCalculator:
    load_name = pod_family_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), **(extra_kw or {})}
    pod_hp = data_kw.get("pod_hyperparams")
    if not isinstance(pod_hp, dict) or not pod_hp:
        raise ValueError(
            f"Could not resolve POD hyperparameters for {model_name!r}."
        )
    hyperparams = dict(pod_hp)
    hyperparams.setdefault("species", ["C"])
    rcut = float(data_kw.get("pod_cutoff", data_kw.get("rcut", 6.0)))
    params = np.asarray(
        np.load(_pod_best_fit_npz(model_name, extra_kw))["params"], dtype=float,
    )
    if load_name == "PODD3_energy":
        return PODD3LammpsCalculator(
            hyperparams,
            params,
            elements=["C"],
            cutoff=rcut,
            d3_damping=str(data_kw.get("d3_damping", "zerom")),
            d3_functional=str(data_kw.get("d3_functional", "pbe")),
            d3_cutoff=float(data_kw.get("d3_cutoff", 30.0)),
            d3_cn_cutoff=float(data_kw.get("d3_cn_cutoff", 20.0)),
        )
    return PODLammpsCalculator(
        hyperparams, params, elements=["C"], cutoff=rcut,
    )


def build_model_calculator(model_name: str, extra_kw: dict | None):
    """Return ``(calculator, close_fn)``."""
    if is_pod_family_model(model_name):
        calc = build_pod_family_calculator(model_name, extra_kw)
        return calc, getattr(calc, "close", lambda: None)
    calc_obj, _set_params_fn, _load_name = build_uq_calculator(
        model_name, extra_kw=extra_kw,
    )
    return calc_obj, getattr(calc_obj, "close", lambda: None)


# Backward-compatible private aliases (older import names).
_build_model_calculator = build_model_calculator
_chdir_for_dataloader = chdir_for_dataloader
_is_pod_family_model = is_pod_family_model
_model_plot_label = model_plot_label
_pod_family_load_name = pod_family_load_name
_resolve_model_names = resolve_model_names
