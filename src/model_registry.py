"""
Central registry for BLG energy and tight-binding models.

Each registered model exposes a consistent schema for hyperparameters,
cache naming, and fit dispatch so UQ scripts do not duplicate if/elif chains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

ModelKind = Literal["hopping", "energy", "hybrid"]
FitKind = Literal[
    "acsf_linear",
    "generic",
    "pod",
    "tetb_pod",
    "lammps",
]

_ACSF_MW_RE = re.compile(
    r"(?i)(acsf[_\-]hoppings(?:[_\-]sk)?)[_\-]m[_\-](\d+)[_\-]w[_\-](\d+)"
)
_ACSF_BASE_RE = re.compile(r"(?i)^acsf[_\-]hoppings(?:[_\-]sk)?$")


@dataclass
class ModelSpec:
    """Metadata and factory hooks for one registered model."""

    name: str
    kind: ModelKind
    fit_kind: FitKind
    match: Callable[[str], bool]
    parse_name: Callable[[str], Tuple[str, Dict[str, Any]]]
    make_hyperparams: Callable[..., Dict[str, Any]]
    cache_basename: Callable[[str, Dict[str, Any]], str]
    load_data_name: Callable[[str, Dict[str, Any]], str]
    # Optional: build a TB model instance (hopping models only).
    tb_factory: Optional[Callable[[Dict[str, Any]], Any]] = None
    description: str = ""


_REGISTRY: Dict[str, ModelSpec] = {}
_ORDERED: List[ModelSpec] = []


def register(spec: ModelSpec) -> ModelSpec:
    """Register a model spec (first match wins for ``resolve_model_spec``)."""
    _REGISTRY[spec.name] = spec
    if spec not in _ORDERED:
        _ORDERED.append(spec)
    return spec


def list_registered_models() -> List[str]:
    return [s.name for s in _ORDERED]


def resolve_model_spec(model_name: str) -> ModelSpec:
    """Return the first registered spec whose ``match`` accepts *model_name*."""
    for spec in _ORDERED:
        if spec.match(model_name):
            return spec
    raise ValueError(f"Unknown model_name {model_name!r}; registered: {list_registered_models()}")


def parse_model_name(model_name: str) -> Tuple[str, Dict[str, Any]]:
    """Canonical base name + hyperparameter dict parsed from *model_name*."""
    spec = resolve_model_spec(model_name)
    return spec.parse_name(model_name)


def make_hyperparams(model_name: str, **overrides: Any) -> Dict[str, Any]:
    """Build validated hyperparameters for *model_name* with optional overrides."""
    spec = resolve_model_spec(model_name)
    _, hp = spec.parse_name(model_name)
    hp.update(overrides)
    return spec.make_hyperparams(hp)


def load_data_model_name(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> str:
    """Name passed to :func:`DataLoader.load_data_for_model`."""
    spec = resolve_model_spec(model_name)
    hp = dict(hyperparameters or {})
    _, parsed = spec.parse_name(model_name)
    hp = {**parsed, **hp}
    return spec.load_data_name(model_name, hp)


def cache_basename(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> str:
    spec = resolve_model_spec(model_name)
    hp = dict(hyperparameters or {})
    _, parsed = spec.parse_name(model_name)
    hp = {**parsed, **hp}
    return spec.cache_basename(model_name, hp)


# ── ACSF helpers ─────────────────────────────────────────────────────────────

def _parse_acsf_mw(model_name: str, sk: bool = False) -> Tuple[str, Dict[str, Any]]:
    base = "ACSF_hoppings_sk" if sk else "ACSF_hoppings"
    m = _ACSF_MW_RE.search(model_name.replace(" ", "_"))
    if m:
        return base, {"M": int(m.group(2)), "W": int(m.group(3))}
    if _ACSF_BASE_RE.match(model_name.replace(" ", "_")) or model_name.startswith(base):
        return base, {}
    return base, {}


def _acsf_make_hyperparams(hp: Dict[str, Any], *, sk: bool = False) -> Dict[str, Any]:
    base = "ACSF_hoppings_sk" if sk else "ACSF_hoppings"
    return {
        "M": int(hp.get("acsf_M", hp.get("M", 10))),
        "W": int(hp.get("acsf_W", hp.get("W", 3))),
        "r_cut": float(hp.get("acsf_r_cut", hp.get("r_cut", 6.0))),
        "use_envelope": bool(hp.get("acsf_use_envelope", hp.get("use_envelope", True))),
        "model_base": base,
    }


def _acsf_cache_basename(model_name: str, hp: Dict[str, Any], *, sk: bool = False) -> str:
    base = "ACSF_hoppings_sk" if sk else "ACSF_hoppings"
    M = int(hp.get("M", 10))
    W = int(hp.get("W", 3))
    return f"{base}_M_{M}_W_{W}_best_fit_params"


def _acsf_load_data_name(model_name: str, hp: Dict[str, Any], *, sk: bool = False) -> str:
    base = "ACSF_hoppings_sk" if sk else "ACSF_hoppings"
    M = int(hp.get("M", 10))
    W = int(hp.get("W", 3))
    return f"{base}_M_{M}_W_{W}"


# ── Deferred registration (populated by tb_models / lammps_interface imports) ───

def ensure_tb_models_registered() -> None:
    """Import tb_models to trigger TB model registration."""
    import blg_model_builder.tb_models  # noqa: F401


def ensure_energy_models_registered() -> None:
    """Import lammps_interface energy registry hooks if present."""
    try:
        import blg_model_builder.energy_registry  # noqa: F401
    except ImportError:
        pass
