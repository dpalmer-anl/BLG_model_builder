#!/usr/bin/env python3
"""Re-evaluate meets_all_criteria in pod_hyperparam_search_results.csv."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

SEP_MIN = 3.0
SEP_MAX = 4.0
TBLG_THETAS = [21.78, 9.43]


def _parse_bool(val: Any) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def _has_error(val: Any) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(val).strip())


def _parse_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _sep_in_band(value: Optional[float]) -> bool:
    return value is not None and SEP_MIN <= value <= SEP_MAX


def _separation_flags(
    ab: Optional[float],
    aa: Optional[float],
    sep_min: Optional[float],
    sep_max: Optional[float],
) -> dict[str, bool]:
    ab_ok = _sep_in_band(ab)
    aa_ok = _sep_in_band(aa)
    sep_min_ok = _sep_in_band(sep_min) if sep_min is not None else ab_ok
    sep_max_ok = _sep_in_band(sep_max) if sep_max is not None else aa_ok
    ab_lt_aa = (
        ab is not None
        and aa is not None
        and float(ab) < float(aa)
    )
    ok = ab_ok and aa_ok and sep_min_ok and sep_max_ok and ab_lt_aa
    return {
        "ab_in_target": ab_ok,
        "aa_in_target": aa_ok,
        "sep_min_in_target": sep_min_ok,
        "sep_max_in_target": sep_max_ok,
        "ab_lt_aa": ab_lt_aa,
        "separations_ok": ok,
    }


def reevaluate_csv(csv_path: Path, output_path: Path | None = None) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    flag_keys = (
        "ab_in_target", "aa_in_target", "ab_lt_aa",
        "sep_min_in_target", "sep_max_in_target", "separations_ok",
    )
    for theta in TBLG_THETAS:
        for key in flag_keys:
            c = f"tblg_{theta}_{key}"
            if c not in df.columns:
                df[c] = False
            else:
                df[c] = df[c].astype(bool)

    n_ok = 0
    for i in range(len(df)):
        rmse = _parse_float(df.at[i, "test_energy_rmse_per_atom_eV"])
        meets = rmse is not None
        if meets:
            for theta in TBLG_THETAS:
                prefix = f"tblg_{theta}_"
                if _has_error(df.at[i, f"{prefix}error"]):
                    meets = False
                    break
                if f"{prefix}converged" in cols and not _parse_bool(df.at[i, f"{prefix}converged"]):
                    meets = False
                    break
                ab = _parse_float(df.at[i, f"{prefix}ab_separation"])
                aa = _parse_float(df.at[i, f"{prefix}aa_separation"])
                smin = (
                    _parse_float(df.at[i, f"{prefix}sep_min"])
                    if f"{prefix}sep_min" in cols
                    else None
                )
                smax = (
                    _parse_float(df.at[i, f"{prefix}sep_max"])
                    if f"{prefix}sep_max" in cols
                    else None
                )
                flags = _separation_flags(ab, aa, smin, smax)
                for key in flag_keys:
                    df.at[i, f"{prefix}{key}"] = flags[key]
                if not flags["separations_ok"]:
                    meets = False
                    break
        df.at[i, "meets_all_criteria"] = meets
        if meets:
            n_ok += 1

    out = output_path or csv_path
    df.to_csv(out, index=False)
    return n_ok, len(df)


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parent
    path = base / "pod_hyperparam_search_results.csv"
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else path
    ok, total = reevaluate_csv(path, out_path)
    print(f"Updated {out_path.name}: {ok}/{total} meet_all_criteria")
    print(f"  Criteria: [{SEP_MIN}, {SEP_MAX}] Å for AB, AA, min/max layer sep; AB < AA; converged; no LAMMPS error")
