"""
Shared model resolution for UQ propagation scripts.

Supports ensembles for:

* LAMMPS-backed: ``POD_energy``, ``PODD3_energy``, ``TETB_POD``, ``Tersoff+DRIP``,
  ``Tersoff+Kolmogorov_Crespi``
* Python-backed: ``Allegro_energy`` (NequIP/Allegro checkpoint)
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

_RE_POD_INDEX = re.compile(r"^POD(?:D3)?_energy_POD_index_(\d+)_([0-9a-f]+)$", re.I)
_RE_TETB_POD_INDEX = re.compile(
    r"^TETB_POD_(?P<tag>tb_M_\d+_W_\d+_pod_M_\d+_W_\d+)_POD_index_(\d+)_([0-9a-f]+)$",
    re.I,
)
_RE_TETB_TAG = re.compile(r"^TETB_POD_(tb_M_\d+_W_\d+_pod_M_\d+_W_\d+)$", re.I)
_RE_TB_POD_MW = re.compile(
    r"tb_M_(\d+)_W_(\d+)_pod_M_(\d+)_W_(\d+)", re.I,
)
_RE_ALLEGRO_CKPT = re.compile(r"^Allegro_energy_ckpt_([0-9a-f]+)$", re.I)

UQ_LAMMPS_MODELS = frozenset(
    {
        "POD_energy",
        "PODD3_energy",
        "TETB_POD",
        "Tersoff+DRIP",
        "Tersoff+Kolmogorov_Crespi",
    }
)

UQ_PYTHON_MODELS = frozenset({"Allegro_energy"})


def is_uq_lammps_model(model_name: str) -> bool:
    """True if ``model_name`` is a LAMMPS-backed UQ propagation model."""
    if model_name in UQ_LAMMPS_MODELS:
        return True
    if model_name.startswith("POD_energy") or model_name.startswith("PODD3_energy"):
        return True
    if model_name.startswith("TETB_POD"):
        return True
    return False


def is_uq_python_model(model_name: str) -> bool:
    """True if ``model_name`` is a Python-backed UQ propagation model."""
    return model_name.startswith("Allegro_energy")


def is_uq_energy_model(model_name: str) -> bool:
    """True if ``model_name`` is supported for relaxation / elasticity UQ."""
    return is_uq_lammps_model(model_name) or is_uq_python_model(model_name)


def resolve_load_name(model_name: str) -> str:
    """``model_name`` passed to :func:`get_MCMC_inputs` for data / calculator setup."""
    if model_name.startswith("Allegro_energy"):
        return "Allegro_energy"
    if model_name.startswith("PODD3_energy"):
        return model_name if model_name == "PODD3_energy" else "PODD3_energy"
    if model_name.startswith("POD_energy"):
        return model_name if model_name == "POD_energy" else "POD_energy"
    if model_name.startswith("TETB_POD"):
        return "TETB_POD"
    if model_name in UQ_LAMMPS_MODELS:
        return model_name
    raise ValueError(
        f"Unsupported model for UQ propagation: {model_name!r}. "
        f"Expected LAMMPS model, Allegro_energy*, or POD/TETB/PODD3 folder name."
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


def _pod_hyperparams_from_csv_identity(
    pod_index: int,
    pod_hash: str,
) -> Dict[str, Any]:
    """Resolve and validate ``POD_index`` + hash against the search CSV/cache."""
    from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

    pod_hp, pod_cutoff, h = pod_hyperparams_for_index(
        int(pod_index),
        require_fit_cache=True,
    )
    if str(h).lower() != str(pod_hash).lower():
        raise ValueError(
            f"Ensemble folder identifies POD_index={int(pod_index)} "
            f"hash={pod_hash}, but CSV row {int(pod_index)} has hash={h}. "
            "Regenerate the MCMC ensemble for the current CSV."
        )
    return {"pod_hyperparams": pod_hp, "pod_cutoff": pod_cutoff, "pod_hash": h}


def mcmc_kw_for_model(model_name: str) -> Dict[str, Any]:
    """
    Keyword arguments for :func:`get_MCMC_inputs` from an ensemble folder name.

    For ``POD_energy_POD_index_<i>_<hash>`` folders, row ``i`` in
    ``pod_hyperparam_search.csv`` must contain the same hash and have an
    associated search fit cache.

    For ``Allegro_energy_ckpt_<tag>`` folders, the checkpoint path is resolved
    by scanning the repository for a ``.ckpt`` file with matching tag.
    """
    m = _RE_ALLEGRO_CKPT.match(model_name)
    if m:
        from blg_model_builder.allegro_interface import resolve_allegro_checkpoint_by_tag

        tag = m.group(1)
        ckpt = resolve_allegro_checkpoint_by_tag(tag)
        return {"allegro_checkpoint": str(ckpt), "allegro_ckpt_tag": tag.lower()}

    m = _RE_POD_INDEX.match(model_name)
    if m:
        pod_index = int(m.group(1))
        folder_hash = m.group(2)
        return _pod_hyperparams_from_csv_identity(pod_index, folder_hash)

    m = _RE_TETB_POD_INDEX.match(model_name)
    if m:
        kw = dict(_tetb_mw_from_tag(m.group("tag")))
        pod_index = int(m.group(2))
        folder_hash = m.group(3)
        kw.update(_pod_hyperparams_from_csv_identity(pod_index, folder_hash))
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
    extra_kw: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[Callable[[np.ndarray], None]], str]:
    """
    Build a LAMMPS calculator for elasticity / relaxation UQ.

    Returns ``(calc_obj, set_params_fn, load_name)``. ``set_params_fn`` is non-``None``
    for hybrid models (Tersoff+KC, Tersoff+DRIP, TETB_POD).

    Uses ``calculator_only=True`` so training data are not loaded and POD
    descriptors / batch evaluators are not computed — only the calculator is
    needed for ensemble parameter propagation.

    ``extra_kw`` (e.g. CLI hyperparameters) overrides the keyword arguments
    derived from the ensemble folder name before they are forwarded to
    :func:`get_MCMC_inputs`.
    """
    if not is_uq_lammps_model(model_name):
        raise ValueError(f"Not a UQ LAMMPS model: {model_name!r}")

    from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs, get_uq_lammps_runtime

    load_name = resolve_load_name(model_name)
    data_kw = {
        **mcmc_kw_for_model(model_name),
        "skip_diagnostics": True,
    }
    if extra_kw:
        data_kw.update(extra_kw)
    data_kw["calculator_only"] = True
    get_MCMC_inputs(load_name, supercells=1, **data_kw)
    meta = get_uq_lammps_runtime()
    calc_obj = meta.get("calc_obj")
    if calc_obj is None:
        raise RuntimeError(
            f"get_MCMC_inputs({load_name!r}) did not register a LAMMPS calculator "
            f"for UQ (folder {model_name!r})."
        )
    return calc_obj, meta.get("set_params_fn"), load_name


def build_uq_python_calculator(
    model_name: str,
    extra_kw: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[Callable[[np.ndarray], None]], str]:
    """Build a Python-backed calculator (Allegro) for UQ propagation.

    Same ``calculator_only`` fast path as :func:`build_uq_lammps_calculator`.

    ``extra_kw`` (e.g. CLI hyperparameters) overrides the derived keyword
    arguments before they are forwarded to :func:`get_MCMC_inputs`.
    """
    if not is_uq_python_model(model_name):
        raise ValueError(f"Not a UQ Python model: {model_name!r}")

    from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs, get_uq_lammps_runtime

    load_name = resolve_load_name(model_name)
    data_kw = {
        **mcmc_kw_for_model(model_name),
        "skip_diagnostics": True,
    }
    if extra_kw:
        data_kw.update(extra_kw)
    data_kw["calculator_only"] = True
    get_MCMC_inputs(load_name, supercells=1, **data_kw)
    meta = get_uq_lammps_runtime()
    calc_obj = meta.get("calc_obj")
    if calc_obj is None:
        raise RuntimeError(
            f"get_MCMC_inputs({load_name!r}) did not register a calculator "
            f"for UQ (folder {model_name!r})."
        )
    return calc_obj, meta.get("set_params_fn"), load_name


def build_uq_calculator(
    model_name: str,
    extra_kw: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[Callable[[np.ndarray], None]], str]:
    """Dispatch to LAMMPS or Python calculator builder."""
    if is_uq_python_model(model_name):
        return build_uq_python_calculator(model_name, extra_kw)
    return build_uq_lammps_calculator(model_name, extra_kw)
