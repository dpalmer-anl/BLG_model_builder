"""Register LAMMPS and Python-backed energy models in the central registry."""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from blg_model_builder.allegro_interface import (
    _DEFAULT_R_MAX,
    checkpoint_tag,
    resolve_allegro_checkpoint,
)
from blg_model_builder.model_registry import ModelSpec, register

_RE_ALLEGRO_CKPT = re.compile(r"^Allegro_energy(?:_ckpt_([0-9a-f]+))?$", re.I)


def _parse_allegro(model_name: str) -> Tuple[str, Dict[str, Any]]:
    m = _RE_ALLEGRO_CKPT.match(model_name.replace(" ", "_"))
    if not m:
        return "Allegro_energy", {}
    tag = m.group(1)
    hp: Dict[str, Any] = {}
    if tag:
        hp["allegro_ckpt_tag"] = tag.lower()
    return "Allegro_energy", hp


def _allegro_make_hyperparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = hp.get("allegro_checkpoint")
    if ckpt is None and hp.get("allegro_ckpt_tag"):
        from blg_model_builder.allegro_interface import resolve_allegro_checkpoint_by_tag

        ckpt = str(resolve_allegro_checkpoint_by_tag(hp["allegro_ckpt_tag"]))
    elif ckpt is None:
        ckpt = str(resolve_allegro_checkpoint(None))
    else:
        ckpt = str(resolve_allegro_checkpoint(ckpt))
    return {
        "allegro_checkpoint": ckpt,
        "allegro_r_max": float(hp.get("allegro_r_max", hp.get("r_max", _DEFAULT_R_MAX))),
        "allegro_device": str(hp.get("allegro_device", hp.get("device", "cpu"))),
        "allegro_bound_scale": float(hp.get("allegro_bound_scale", 1e2)),
        "allegro_ckpt_tag": hp.get("allegro_ckpt_tag") or checkpoint_tag(ckpt),
    }


def _allegro_cache_basename(model_name: str, hp: Dict[str, Any]) -> str:
    tag = hp.get("allegro_ckpt_tag") or checkpoint_tag(hp["allegro_checkpoint"])
    return f"Allegro_energy_ckpt_{tag}_best_fit_params"


def _allegro_load_data_name(model_name: str, hp: Dict[str, Any]) -> str:
    return "Allegro_energy"


def _register_energy_models() -> None:
    register(
        ModelSpec(
            name="Allegro_energy",
            kind="energy",
            fit_kind="allegro",
            match=lambda n: n.startswith("Allegro_energy"),
            parse_name=_parse_allegro,
            make_hyperparams=_allegro_make_hyperparams,
            cache_basename=_allegro_cache_basename,
            load_data_name=_allegro_load_data_name,
            description="Allegro E(3)-equivariant potential (Python checkpoint)",
        )
    )


_register_energy_models()
