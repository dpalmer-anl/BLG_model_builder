"""
Shared CLI model-name handling for BLG UQ workflow scripts.

Users should be able to pass the exact folder name under ``ensembles/``
(e.g. ``ACSF_hoppings_sk_M_12_W_1``, ``POD_energy_POD_index_9_<hash>``) on
any entry point.  Bare base names (``ACSF_hoppings_sk``, ``POD_energy``) are
expanded with ``-M`` / ``-W`` / ``--POD-index`` only when the name does not
already encode those tags.

Also normalises common flag aliases so ``--models`` / ``--model`` /
``--model_name`` / ``-m`` select the model instead of being mis-parsed as
hyperparameters, and ``--tb-models`` aliases ``--tb-model``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from blg_model_builder.cli_hyperparams import collect_hyperparams

__all__ = [
    "RESERVED_CLI_KEYS",
    "add_energy_models_arg",
    "add_model_name_arg",
    "add_tb_model_arg",
    "collect_workflow_hyperparams",
    "expand_ensemble_model_name",
    "is_full_ensemble_name",
]

# Keys that select models on the CLI — never forward to get_MCMC_inputs kwargs.
RESERVED_CLI_KEYS = frozenset(
    {
        "model",
        "model_name",
        "models",
        "tb_model",
        "tb_models",
    }
)

_ACSF_MW_IN_NAME = re.compile(r"[_\-]M[_\-](\d+)[_\-]W[_\-](\d+)", re.I)
_POD_INDEX_IN_NAME = re.compile(r"POD_index_\d+_", re.I)
_TETB_POD_TAG_IN_NAME = re.compile(r"^TETB_POD_tb_M_\d+_W_\d+_pod_M_\d+_W_\d+", re.I)
_ALLEGRO_CKPT_IN_NAME = re.compile(r"^Allegro_energy_ckpt_", re.I)

_BARE_EXPANDABLE = frozenset(
    {
        "ACSF_hoppings",
        "ACSF_hoppings_sk",
        "POD_energy",
        "KC_energy",
        "TETB_POD",
        "Allegro_energy",
    }
)


def is_full_ensemble_name(name: str) -> bool:
    """True when *name* already looks like an ``ensembles/<name>/`` folder."""
    n = str(name).strip()
    if not n:
        return False
    if _POD_INDEX_IN_NAME.search(n):
        return True
    if _TETB_POD_TAG_IN_NAME.match(n):
        return True
    if _ALLEGRO_CKPT_IN_NAME.match(n):
        return True
    if n.startswith(("ACSF_hoppings", "ACSF_hoppings_sk")) and _ACSF_MW_IN_NAME.search(n):
        return True
    if n.startswith("POD_energy") and _ACSF_MW_IN_NAME.search(n):
        return True
    if n.startswith("LETB_intralayer"):
        return True
    return n not in _BARE_EXPANDABLE


def expand_ensemble_model_name(
    model_name: str,
    args,
    mcmc_kw: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Map a CLI model name to the canonical ``ensembles/<name>/`` folder name.

    If *model_name* already encodes M/W, POD index, TETB tags, or an Allegro
    checkpoint tag, it is returned unchanged.
    """
    name = str(model_name).strip()
    if is_full_ensemble_name(name):
        return name

    kw = dict(mcmc_kw or {})

    if name in ("ACSF_hoppings", "ACSF_hoppings_sk", "POD_energy", "KC_energy"):
        pod_index = getattr(args, "pod_index", None)
        if name == "POD_energy" and pod_index is not None:
            from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

            _, _, pod_hash = pod_hyperparams_for_index(int(pod_index))
            return f"POD_energy_POD_index_{int(pod_index)}_{pod_hash}"
        return f"{name}_M_{int(args.M)}_W_{int(args.W)}"

    if name == "TETB_POD":
        from blg_model_builder.get_MCMC_inputs import build_tetb_pod_hyperparams_from_data_kw

        tag_kw: Dict[str, Any] = {"M": int(args.M), "W": int(args.W)}
        for key in ("tb_M", "tb_W", "pod_M", "pod_W"):
            val = getattr(args, key, None)
            if val is not None:
                tag_kw[key] = val
        tag_kw.update({k: v for k, v in kw.items() if k in tag_kw or k.startswith(("tb_", "pod_"))})
        _, _, tag = build_tetb_pod_hyperparams_from_data_kw(tag_kw)
        pod_index = getattr(args, "pod_index", None)
        if pod_index is not None:
            pod_hash = str(kw.get("pod_hash", "unknown"))
            if pod_hash == "unknown":
                from blg_model_builder.pod_model_selection import pod_hyperparams_for_index

                _, _, pod_hash = pod_hyperparams_for_index(int(pod_index))
            return f"TETB_POD_{tag}_POD_index_{int(pod_index)}_{pod_hash}"
        return f"TETB_POD_{tag}"

    if name == "Allegro_energy":
        from blg_model_builder.allegro_interface import (
            checkpoint_tag,
            resolve_allegro_checkpoint,
        )

        ckpt = resolve_allegro_checkpoint(getattr(args, "allegro_checkpoint", None))
        return f"Allegro_energy_ckpt_{checkpoint_tag(ckpt)}"

    return name


def collect_workflow_hyperparams(
    args,
    unknown: Optional[Sequence[str]] = None,
    *,
    extra_reserved: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Like :func:`collect_hyperparams` but drop model-selection CLI keys."""
    hp = collect_hyperparams(args, unknown)
    reserved = set(RESERVED_CLI_KEYS)
    if extra_reserved:
        reserved.update(extra_reserved)
    for key in reserved:
        hp.pop(key, None)
    return hp


def add_model_name_arg(parser, *, default: str = "MK", required: bool = False) -> None:
    """Register ``-m`` / ``--model_name`` / ``--models`` (single model)."""
    kwargs: Dict[str, Any] = dict(
        type=str,
        default=default,
        dest="model_name",
        help=(
            "Model name — use the exact ``ensembles/<name>/`` folder name "
            "(e.g. ACSF_hoppings_sk_M_12_W_1) or a bare base name "
            "(ACSF_hoppings_sk, POD_energy) with ``-M`` / ``-W``.  "
            "``--models`` and ``--model`` are aliases for ``-m``."
        ),
    )
    if required:
        kwargs["required"] = True
        kwargs.pop("default", None)
    parser.add_argument("-m", "--model_name", "--models", "--model", **kwargs)


def add_energy_models_arg(parser, *, required: bool = True) -> None:
    """Register ``--models`` for scripts that accept one or more ensemble patterns."""
    parser.add_argument(
        "--models",
        nargs="+",
        required=required,
        help=(
            "One or more model folder names under ``--ensemble-dir``; glob "
            "wildcards supported.  Use exact names from ``ensembles/`` "
            "(e.g. POD_energy_POD_index_9_<hash>, ACSF_hoppings_sk_M_12_W_1)."
        ),
    )


def add_tb_model_arg(parser, *, default: str) -> None:
    """Register ``--tb-model`` / ``--tb-models`` for band-structure propagation."""
    parser.add_argument(
        "--tb-model",
        "--tb-models",
        default=default,
        dest="tb_model",
        help=(
            "ACSF hopping ensemble model — exact ``ensembles/<name>/`` folder "
            "name (e.g. ACSF_hoppings_sk_M_12_W_1).  "
            "``--tb-models`` is an alias for ``--tb-model``."
        ),
    )
