"""
Merge rcut=6 POD hyperparameter-search results with rcut=7 tightened results.

Appends the rcut=7 CSV/JSON rows after rcut=6 so ``POD_index`` is 0-based row
order in the merged ``pod_hyperparam_search.csv`` (rcut=6 → 0..N-1, rcut=7 →
N..N+M-1).  Copies rcut=6 fit caches into ``pod_hyperparam_search_cache/``
without overwriting existing files (rcut=7 caches are kept on name clashes).

Run from ``uncertainty_quantification/pod_hyperparam_search``::

    python merge_pod_hyperparam_search_results.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_RCUT6_DIR = _HERE / "pod_results_rcut6"
_RCUT6_CSV = _RCUT6_DIR / "pod_hyperparam_search_results_tightened_rcut6.csv"
_RCUT6_JSON = _RCUT6_DIR / "pod_hyperparam_search_results_tightened.json"
_RCUT6_CACHE = _RCUT6_DIR / "pod_hyperparam_search_cache"

_RCUT7_CSV = _HERE / "pod_hyperparam_search_results_tightened.csv"
_RCUT7_JSON = _HERE / "pod_hyperparam_search_results_tightened.json"
_RCUT7_CACHE = _HERE / "pod_hyperparam_search_cache"

_OUT_CSV = _HERE / "pod_hyperparam_search.csv"
_OUT_JSON = _HERE / "pod_hyperparam_search.json"


def _load_json_list(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def merge_csv() -> pd.DataFrame:
    if not _RCUT6_CSV.is_file():
        raise FileNotFoundError(_RCUT6_CSV)
    if not _RCUT7_CSV.is_file():
        raise FileNotFoundError(_RCUT7_CSV)
    df6 = pd.read_csv(_RCUT6_CSV)
    df7 = pd.read_csv(_RCUT7_CSV)
    if list(df6.columns) != list(df7.columns):
        raise ValueError("rcut6/rcut7 CSV column mismatch")
    overlap = set(df6["hash"]) & set(df7["hash"])
    if overlap:
        raise ValueError(f"duplicate hash(es) in rcut6/rcut7 CSV: {sorted(overlap)[:5]}")
    merged = pd.concat([df6, df7], ignore_index=True)
    merged.to_csv(_OUT_CSV, index=False)
    print(
        f"[merge] CSV  rcut6={len(df6)}  rcut7={len(df7)}  "
        f"total={len(merged)}  -> {_OUT_CSV.name}",
        flush=True,
    )
    return merged


def merge_json() -> list[dict]:
    if not _RCUT6_JSON.is_file():
        raise FileNotFoundError(_RCUT6_JSON)
    if not _RCUT7_JSON.is_file():
        raise FileNotFoundError(_RCUT7_JSON)
    j6 = _load_json_list(_RCUT6_JSON)
    j7 = _load_json_list(_RCUT7_JSON)
    h6 = {str(r.get("hash", "")) for r in j6}
    h7 = {str(r.get("hash", "")) for r in j7}
    overlap = h6 & h7
    if overlap:
        raise ValueError(f"duplicate hash(es) in rcut6/rcut7 JSON: {sorted(overlap)[:5]}")
    merged = j6 + j7
    with open(_OUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)
    print(
        f"[merge] JSON rcut6={len(j6)}  rcut7={len(j7)}  "
        f"total={len(merged)}  -> {_OUT_JSON.name}",
        flush=True,
    )
    return merged


def merge_cache() -> tuple[int, int, int]:
    if not _RCUT6_CACHE.is_dir():
        raise FileNotFoundError(_RCUT6_CACHE)
    _RCUT7_CACHE.mkdir(parents=True, exist_ok=True)
    copied = skipped = missing_src = 0
    for src in sorted(_RCUT6_CACHE.rglob("*.npz")):
        rel = src.relative_to(_RCUT6_CACHE)
        dst = _RCUT7_CACHE / rel
        if not src.is_file():
            missing_src += 1
            continue
        if dst.is_file():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(
        f"[merge] cache copied={copied}  skipped(existing)={skipped}  "
        f"into {_RCUT7_CACHE.name}/",
        flush=True,
    )
    return copied, skipped, missing_src


def main() -> None:
    merged = merge_csv()
    merge_json()
    merge_cache()
    n6 = int((merged["rcut"] == 6.0).sum()) if "rcut" in merged.columns else 0
    n7 = int((merged["rcut"] == 7.0).sum()) if "rcut" in merged.columns else 0
    print(
        f"[merge] POD_index map: rcut=6 -> 0..{n6 - 1}, rcut=7 -> {n6}..{n6 + n7 - 1}",
        flush=True,
    )


if __name__ == "__main__":
    main()
