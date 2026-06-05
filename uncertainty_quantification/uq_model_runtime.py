"""
Shared model resolution for UQ propagation scripts.

Supports LAMMPS-backed ensembles for:

* ``POD_energy`` (and ``POD_energy_POD_index_<i>_<hash>``)
* ``TETB_POD`` (and tagged / ``POD_index`` variants)
* ``Tersoff+DRIP``
* ``Tersoff+Kolmogorov_Crespi`` (Tersoff + Kolmogorov–Crespi)
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

_RE_POD_INDEX = re.compile(r"^POD_energy_POD_index_(\d+)_([0-9a-f]+)$", re.I)
_RE_TETB_POD_INDEX = re.compile(
    r"^TETB_POD_(?P<tag>tb_M_\d+_W_\d+_pod_M_\d+_W_\d+)_POD_index_(\d+)_([0-9a-f]+)$",
    re.I,
)
_RE_TETB_TAG = re.compile(r"^TETB_POD_(tb_M_\d+_W_\d+_pod_M_\d+_W_\d+)$", re.I)
_RE_TB_POD_MW = re.compile(
    r"tb_M_(\d+)_W_(\d+)_pod_M_(\d+)_W_(\d+)", re.I,
)

UQ_LAMMPS_MODELS = frozenset(
    {
        "POD_energy",
        "TETB_POD",
        "Tersoff+DRIP",
        "Tersoff+Kolmogorov_Crespi",
    }
)


def is_uq_lammps_model(model_name: str) -> bool:
    """True if ``model_name`` (ensemble folder) is supported for UQ propagation."""
    if model_name in UQ_LAMMPS_MODELS:
        return True
    if model_name.startswith("POD_energy"):
        return True
    if model_name.startswith("TETB_POD"):
        return True
    return False


def resolve_load_name(model_name: str) -> str:
    """``model_name`` passed to :func:`get_MCMC_inputs` for data / calculator setup."""
    if model_name.startswith("POD_energy"):
        return model_name if model_name == "POD_energy" else "POD_energy"
    if model_name.startswith("TETB_POD"):
        return "TETB_POD"
    if model_name in UQ_LAMMPS_MODELS:
        return model_name
    raise ValueError(
        f"Unsupported model for UQ propagation: {model_name!r}. "
        f"Expected one of {sorted(UQ_LAMMPS_MODELS)} or a POD_energy / TETB_POD ensemble folder."
    )


def _tetb_mw_from_tag(tag: str) -> Dict[str, int]:
    m = _RE_TB_POD_MW.search(tag)
    if not m:
        return {}
    return {
        "tb_M": int(m.group(1)),
        "tb_W": int(m.group(2)),
        "pod_M": int(m.group(3)),
        "pod_W": int(m.group(4)),
    }


def _pod_hyperparams_from_hash(pod_hash: str) -> Optional[Dict[str, Any]]:
    """
    Return ``{"pod_hyperparams": ..., "pod_cutoff": ..., "pod_hash": ...}`` by
    looking up *pod_hash* directly in the CSV (no index file needed).
    Returns ``None`` if the hash is not found.
    """
    from pathlib import Path

    import pandas as pd

    csv_path = Path(__file__).resolve().parent / "pod_hyperparam_search_results.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    match = df[df["hash"].astype(str) == str(pod_hash)]
    if match.empty:
        return None
    row = match.iloc[0]
    from pod_model_selection import pod_hyperparams_from_row

    pod_hp, pod_cutoff, h = pod_hyperparams_from_row(row)
    return {"pod_hyperparams": pod_hp, "pod_cutoff": pod_cutoff, "pod_hash": h}


def mcmc_kw_for_model(model_name: str) -> Dict[str, Any]:
    """
    Keyword arguments for :func:`get_MCMC_inputs` from an ensemble folder name.

    For ``POD_energy_POD_index_<i>_<hash>`` folders the **hash** embedded in
    the folder name is used to look up hyperparameters directly from the search
    CSV.  This ensures the calculator always matches the ensemble that was
    generated for that hash, regardless of how ``use_pod_models_hash.txt`` has
    changed since the ensemble was produced.  The index is used only as a
    fallback when the hash is not in the CSV.
    """
    import sys

    m = _RE_POD_INDEX.match(model_name)
    if m:
        folder_hash = m.group(2)
        kw = _pod_hyperparams_from_hash(folder_hash)
        if kw is not None:
            return kw
        # Hash not in CSV — fall back to index-based lookup
        pod_index = int(m.group(1))
        try:
            from pod_model_selection import pod_hyperparams_for_index

            pod_hp, pod_cutoff, pod_hash = pod_hyperparams_for_index(pod_index)
        except (FileNotFoundError, IndexError, KeyError) as exc:
            print(
                f"Warning: POD_index lookup failed ({exc}); using default POD M/W.",
                file=sys.stderr,
            )
            return {"M": 10, "W": 6}
        return {
            "pod_hyperparams": pod_hp,
            "pod_cutoff": pod_cutoff,
            "pod_hash": pod_hash,
        }

    m = _RE_TETB_POD_INDEX.match(model_name)
    if m:
        kw = dict(_tetb_mw_from_tag(m.group("tag")))
        folder_hash = m.group(3)
        hash_kw = _pod_hyperparams_from_hash(folder_hash)
        if hash_kw is not None:
            kw.update(hash_kw)
        else:
            pod_index = int(m.group(2))
            try:
                from pod_model_selection import pod_hyperparams_for_index

                pod_hp, pod_cutoff, pod_hash = pod_hyperparams_for_index(pod_index)
                kw["pod_hyperparams"] = pod_hp
                kw["pod_cutoff"] = pod_cutoff
                kw["pod_hash"] = pod_hash
            except (FileNotFoundError, IndexError, KeyError) as exc:
                print(
                    f"Warning: TETB_POD POD_index lookup failed ({exc}); "
                    "using tag M/W only.",
                    file=sys.stderr,
                )
        return kw

    m = _RE_TETB_TAG.match(model_name)
    if m:
        return _tetb_mw_from_tag(m.group(1))

    return {}


def apply_uq_parameters(
    calc_obj: Any,
    theta: np.ndarray,
    set_params_fn: Optional[Callable[[np.ndarray], None]] = None,
) -> None:
    """Set calculator parameters from one MCMC ensemble draw."""
    p = np.asarray(theta, dtype=float).ravel()
    if set_params_fn is not None:
        set_params_fn(p)
    else:
        calc_obj.set_parameters(p.tolist())


def build_uq_lammps_calculator(
    model_name: str,
) -> Tuple[Any, Optional[Callable[[np.ndarray], None]], str]:
    """
    Build a LAMMPS calculator for elasticity / relaxation UQ.

    Returns ``(calc_obj, set_params_fn, load_name)``. ``set_params_fn`` is non-``None``
    for hybrid models (Tersoff+KC, Tersoff+DRIP, TETB_POD).
    """
    if not is_uq_lammps_model(model_name):
        raise ValueError(f"Not a UQ LAMMPS model: {model_name!r}")

    from get_MCMC_inputs import get_MCMC_inputs, get_uq_lammps_runtime

    load_name = resolve_load_name(model_name)
    data_kw = {**mcmc_kw_for_model(model_name), "skip_diagnostics": True}
    get_MCMC_inputs(load_name, supercells=1, **data_kw)
    meta = get_uq_lammps_runtime()
    calc_obj = meta.get("calc_obj")
    if calc_obj is None:
        raise RuntimeError(
            f"get_MCMC_inputs({load_name!r}) did not register a LAMMPS calculator "
            f"for UQ (folder {model_name!r})."
        )
    return calc_obj, meta.get("set_params_fn"), load_name
