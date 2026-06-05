"""
POD model selection for UQ ensembles.

``use_pod_models_hash.txt`` lists up to 10 hyperparameter-search hashes (one per
line). ``POD_index`` is a 0-based index into that file, not into all rows with
``meets_all_criteria`` in the CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
USE_POD_MODELS_HASH_FILE = _HERE / "use_pod_models_hash.txt"
POD_SEARCH_RESULTS_CSV = _HERE / "pod_hyperparam_search_results.csv"

DEFAULT_SELECTION_SEED = 42
DEFAULT_MIN_NCOEFF = 50
DEFAULT_N_MODELS = 10


def _meets_all_criteria_bool(val: Any) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def load_use_pod_model_hashes(path: Optional[Path] = None) -> list[str]:
    """Read hashes from ``use_pod_models_hash.txt`` (comments and blank lines skipped)."""
    path = Path(path) if path is not None else USE_POD_MODELS_HASH_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"POD model list not found: {path}. "
            "Run: python pod_model_selection.py"
        )
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
    """CSV row for ``POD_index`` using ``use_pod_models_hash.txt``."""
    hashes = load_use_pod_model_hashes(hash_list_path)
    idx = int(pod_index)
    if idx < 0 or idx >= len(hashes):
        raise IndexError(
            f"POD_index={idx} out of range for {len(hashes)} entries in "
            f"{hash_list_path or USE_POD_MODELS_HASH_FILE}"
        )
    target = hashes[idx]
    df = pd.read_csv(csv_path or POD_SEARCH_RESULTS_CSV)
    match = df[df["hash"].astype(str) == str(target)]
    if match.empty:
        raise KeyError(
            f"Hash {target!r} (POD_index={idx}) not found in "
            f"{csv_path or POD_SEARCH_RESULTS_CSV}"
        )
    return match.iloc[0]


def pod_hyperparams_from_row(row: pd.Series) -> tuple[dict[str, Any], float, str]:
    """Return (pod_hyperparams, pod_cutoff, hash) from a search-results row."""
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
    """Convenience: row lookup + hyperparameter dict for ``POD_index``."""
    row = pod_row_for_index(pod_index, **kwargs)
    return pod_hyperparams_from_row(row)


def select_and_write_use_pod_models(
    *,
    n_models: int = DEFAULT_N_MODELS,
    min_ncoeff: int = DEFAULT_MIN_NCOEFF,
    seed: int = DEFAULT_SELECTION_SEED,
    csv_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> list[str]:
    """
    Pick ``n_models`` hashes: best test RMSE first, then ``n_models - 1`` at random
    from the remaining pool (``meets_all_criteria``, ``ncoeff > min_ncoeff``).
    """
    csv_path = Path(csv_path) if csv_path is not None else POD_SEARCH_RESULTS_CSV
    out_path = Path(out_path) if out_path is not None else USE_POD_MODELS_HASH_FILE

    df = pd.read_csv(csv_path)
    ok = df["meets_all_criteria"].map(_meets_all_criteria_bool)
    pool = df[ok & (df["ncoeff"] > min_ncoeff)].sort_values(
        "test_energy_rmse_per_atom_eV", ascending=True
    )
    if len(pool) < n_models:
        raise ValueError(
            f"Only {len(pool)} candidates with ncoeff>{min_ncoeff} and "
            f"meets_all_criteria; need {n_models}."
        )

    best = pool.iloc[0]
    rest = pool.iloc[1:]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rest), size=n_models - 1, replace=False)
    others = rest.iloc[np.sort(pick)]

    chosen: list[str] = [str(best["hash"])]
    chosen.extend(str(r["hash"]) for _, r in others.iterrows())

    lines = [
        "# use_pod_models_hash.txt — POD_index is 0-based line index (comments skipped)",
        f"# Selected: best test RMSE + {n_models - 1} random (seed={seed}, ncoeff>{min_ncoeff})",
        "# index  hash  ncoeff  test_rmse_eV_per_atom",
    ]
    for i, h in enumerate(chosen):
        row = pool[pool["hash"].astype(str) == h].iloc[0]
        rmse = float(row["test_energy_rmse_per_atom_eV"])
        lines.append(
            f"# {i}  {h}  ncoeff={int(row['ncoeff'])}  rmse={rmse:.6g}"
        )
    lines.extend(chosen)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return chosen


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Select POD models for UQ ensembles.")
    p.add_argument("--n", type=int, default=DEFAULT_N_MODELS)
    p.add_argument("--min-ncoeff", type=int, default=DEFAULT_MIN_NCOEFF)
    p.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    hashes = select_and_write_use_pod_models(
        n_models=args.n,
        min_ncoeff=args.min_ncoeff,
        seed=args.seed,
        csv_path=args.csv,
        out_path=args.out,
    )
    print(f"Wrote {len(hashes)} hashes to {args.out or USE_POD_MODELS_HASH_FILE}")
    for i, h in enumerate(hashes):
        print(f"  POD_index={i}  {h}")
