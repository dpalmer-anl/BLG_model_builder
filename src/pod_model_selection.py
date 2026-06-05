"""
POD model selection for UQ ensembles.

``use_pod_models_hash.txt`` lists hyperparameter-search hashes (one per line).
``POD_index`` is a 0-based index into that file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _pod_search_dir() -> Path:
    """Resolve pod_hyperparam_search directory (CWD or package-relative)."""
    cwd = Path.cwd()
    local = cwd / "pod_hyperparam_search"
    if local.is_dir():
        return local
    pkg = Path(__file__).resolve().parent.parent
    return pkg / "uncertainty_quantification" / "pod_hyperparam_search"


_POD_DIR = _pod_search_dir()
USE_POD_MODELS_HASH_FILE = _POD_DIR / "use_pod_models_hash.txt"
POD_SEARCH_RESULTS_CSV = _POD_DIR / "pod_hyperparam_search_results.csv"

DEFAULT_SELECTION_SEED = 42
DEFAULT_MIN_NCOEFF = 50
DEFAULT_N_MODELS = 10


def _meets_all_criteria_bool(val: Any) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def load_use_pod_model_hashes(path: Optional[Path] = None) -> list[str]:
    path = Path(path) if path is not None else USE_POD_MODELS_HASH_FILE
    if not path.is_file():
        raise FileNotFoundError(f"POD model list not found: {path}")
    hashes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hashes.append(line.split()[0])
    if not hashes:
        raise ValueError(f"No hashes in {path}")
    return hashes


def pod_row_for_index(
    pod_index: int,
    *,
    csv_path: Optional[Path] = None,
    hash_list_path: Optional[Path] = None,
) -> pd.Series:
    hashes = load_use_pod_model_hashes(hash_list_path)
    idx = int(pod_index)
    if idx < 0 or idx >= len(hashes):
        raise IndexError(f"POD_index={idx} out of range for {len(hashes)} entries")
    target = hashes[idx]
    df = pd.read_csv(csv_path or POD_SEARCH_RESULTS_CSV)
    match = df[df["hash"].astype(str) == str(target)]
    if match.empty:
        raise KeyError(f"Hash {target!r} (POD_index={idx}) not found in CSV")
    return match.iloc[0]


def pod_hyperparams_from_row(row: pd.Series) -> tuple[dict[str, Any], float, str]:
    pod_hp = {
        "species": ["C"],
        "bessel_polynomial_degree": int(row["bessel_polynomial_degree"]),
        "inverse_polynomial_degree": int(row["inverse_polynomial_degree"]),
        "twobody_number_radial_basis_functions": int(row["two_body_radial"]),
        "threebody_number_radial_basis_functions": int(row["three_body_radial"]),
        "threebody_angular_degree": int(row["three_body_angular"]),
        "fourbody_number_radial_basis_functions": int(row["four_body_radial"]),
        "fourbody_angular_degree": int(row["four_body_angular"]),
        "fivebody_number_radial_basis_functions": int(row["five_body_radial"]),
        "fivebody_angular_degree": int(row["five_body_angular"]),
        "sixbody_number_radial_basis_functions": int(row["six_body_radial"]),
        "sixbody_angular_degree": int(row["six_body_angular"]),
        "sevenbody_number_radial_basis_functions": int(row["seven_body_radial"]),
        "sevenbody_angular_degree": int(row["seven_body_angular"]),
    }
    return pod_hp, float(row["rcut"]), str(row["hash"])


def pod_hyperparams_for_index(pod_index: int, **kwargs: Any) -> tuple[dict[str, Any], float, str]:
    row = pod_row_for_index(pod_index, **kwargs)
    return pod_hyperparams_from_row(row)
