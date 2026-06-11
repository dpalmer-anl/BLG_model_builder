"""
Generic command-line hyperparameter parsing shared by all BLG workflow CLIs.

Every workflow entry point (``fit_model.py``, ``run_MCMC.py`` /
``EMCEE_generate_ensemble.py``, ``run_SubSamp.py`` /
``SubSamp_generate_ensemble.py``, and the ``run_uq_propagation_*`` scripts)
forwards an arbitrary hyperparameter dict into
:func:`blg_model_builder.get_MCMC_inputs.get_MCMC_inputs` (and from there into
:func:`blg_model_builder.DataLoader.load_data_for_model`).  Model selection
uses the exact ``ensembles/<name>/`` folder names via
:mod:`blg_model_builder.cli_model_names` (``-m`` / ``--models`` / ``--model`` /
``--tb-model`` / ``--tb-models``).

Two equivalent syntaxes are supported:

* ``--set KEY=VALUE`` (repeatable) or ``--hyperparams KEY=VALUE KEY=VALUE``
* bare ``--KEY VALUE`` / ``--KEY=VALUE`` (captured from unknown CLI tokens)

Values are coerced to ``int`` / ``float`` / ``bool`` / ``None`` / JSON when they
parse cleanly, otherwise kept as ``str``.  Dashes in keys are normalized to
underscores (``--two-body-radial`` and ``--two_body_radial`` are equivalent).

Examples
--------
``python fit_model.py --models POD_energy --two_body_radial 2 --three_body_angular 4``
``python run_MCMC.py -m POD_energy -B 0.001 --set regularization=1e-10``
``python run_MCMC.py -m ACSF_hoppings -M 10 -W 6 --acsf_r_cut 7.0``
``python run_MCMC.py -m Allegro_energy --allegro_bound_scale 50``
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "coerce_value",
    "add_hyperparam_args",
    "collect_hyperparams",
    "parse_known_with_hyperparams",
]


def coerce_value(text: str) -> Any:
    """Convert a CLI string to the most natural Python type.

    Order: bool → None → int → float → JSON (list/dict) → str.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s and s[0] in "[{\"":
        try:
            return json.loads(s)
        except (ValueError, json.JSONDecodeError):
            pass
    return s


def _normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def add_hyperparam_args(parser) -> None:
    """Register ``--set`` / ``--hyperparams`` on an ``argparse`` parser.

    Bare ``--KEY VALUE`` tokens are also accepted, but only when the caller
    parses with :func:`parse_known_with_hyperparams` (or passes the leftover
    ``unknown`` list to :func:`collect_hyperparams`).
    """
    parser.add_argument(
        "--set",
        "-S",
        dest="set_hyperparams",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set a model hyperparameter (repeatable). Forwarded to "
        "get_MCMC_inputs / load_data_for_model. E.g. --set two_body_radial=2.",
    )
    parser.add_argument(
        "--hyperparams",
        "--hyperparam",
        dest="hyperparams_list",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="One or more KEY=VALUE hyperparameters (space-separated). "
        "E.g. --hyperparams two_body_radial=2 three_body_angular=4.",
    )


def _parse_kv_list(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid hyperparameter {item!r}; expected KEY=VALUE."
            )
        key, value = item.split("=", 1)
        key = _normalize_key(key)
        if not key:
            raise ValueError(f"Invalid hyperparameter {item!r}; empty key.")
        out[key] = coerce_value(value)
    return out


def _parse_unknown_tokens(unknown: Sequence[str]) -> Dict[str, Any]:
    """Parse leftover ``--key value`` / ``--key=value`` / ``--flag`` tokens."""
    out: Dict[str, Any] = {}
    tokens = list(unknown or [])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(
                f"Unrecognized CLI argument {tok!r}. Use --key value or "
                f"--set key=value for hyperparameters."
            )
        body = tok[2:]
        if "=" in body:
            key, value = body.split("=", 1)
            out[_normalize_key(key)] = coerce_value(value)
            i += 1
            continue
        key = _normalize_key(body)
        # Peek at the next token: if it is a value (not another --flag), consume
        # it; a leading '-' that parses as a negative number still counts.
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            is_value = not nxt.startswith("--") and not (
                nxt.startswith("-") and not _looks_numeric(nxt)
            )
            if is_value:
                out[key] = coerce_value(nxt)
                i += 2
                continue
        out[key] = True
        i += 1
    return out


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def collect_hyperparams(
    args, unknown: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Merge ``--set``, ``--hyperparams``, and unknown ``--key value`` tokens.

    Precedence (later wins): ``--hyperparams`` < ``--set`` < bare ``--key``.
    """
    hp: Dict[str, Any] = {}
    hp.update(_parse_kv_list(getattr(args, "hyperparams_list", []) or []))
    hp.update(_parse_kv_list(getattr(args, "set_hyperparams", []) or []))
    hp.update(_parse_unknown_tokens(unknown or []))
    return hp


def parse_known_with_hyperparams(parser) -> Tuple[Any, Dict[str, Any]]:
    """``parser.parse_known_args()`` + :func:`collect_hyperparams`.

    Returns ``(args, hyperparams_dict)``.
    """
    args, unknown = parser.parse_known_args()
    hp = collect_hyperparams(args, unknown)
    return args, hp
