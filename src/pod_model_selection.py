"""POD model selection from the hyperparameter-search CSV and its fit caches.

``POD_index`` is the 0-based row index in ``pod_hyperparam_search.csv``
(rcut=6 rows first, then rcut=7). No separate hash-list files are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

def _pod_search_dir() -> Path:
    """Resolve pod_hyperparam_search directory (CWD or package-relative)."""
    cwd = Path.cwd()
    local = cwd / "pod_hyperparam_search"
    if local.is_dir():
        return local
    # Also accept being run from inside pod_hyperparam_search/.
    if cwd.name == "pod_hyperparam_search" and (cwd / "pod_hyperparam_search_cache").is_dir():
        return cwd
    pkg = Path(__file__).resolve().parent.parent
    return pkg / "uncertainty_quantification" / "pod_hyperparam_search"


def _pod_dir() -> Path:
    """Current search-results directory (resolved at call time, not import time)."""
    return _pod_search_dir()


POD_SEARCH_RESULTS_CSV = _pod_search_dir() / "pod_hyperparam_search.csv"
POD_SEARCH_RESULTS_CSV_LEGACY = (
    _pod_search_dir() / "pod_hyperparam_search_results_tightened.csv"
)

def _meets_all_criteria_bool(val: Any) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def _search_csv_paths(csv_path: Optional[Path] = None) -> list[Path]:
    if csv_path is not None:
        path = Path(csv_path)
        return [path] if path.is_file() else []
    pod_dir = _pod_dir()
    canonical = pod_dir / "pod_hyperparam_search.csv"
    legacy = pod_dir / "pod_hyperparam_search_results_tightened.csv"
    if canonical.is_file():
        return [canonical]
    if legacy.is_file():
        return [legacy]
    return []


def load_pod_search_results(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the POD hyperparameter-search CSV (merged rcut=6 + rcut=7 by default)."""
    paths = _search_csv_paths(csv_path)
    if not paths:
        expected = Path(csv_path) if csv_path is not None else POD_SEARCH_RESULTS_CSV
        raise FileNotFoundError(f"POD hyperparameter search CSV not found: {expected}")
    df = pd.read_csv(paths[0])
    required = {"hash", "meets_all_criteria"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{paths[0]} is missing required column(s): {sorted(missing)}"
        )
    return df


def pod_row_for_index(
    pod_index: int,
    *,
    csv_path: Optional[Path] = None,
    hash_list_path: Optional[Path] = None,
) -> pd.Series:
    """
    Return the hyperparameter-search row for ``POD_index``.

    ``POD_index`` is the 0-based row number in the search-results CSV
    (``pod_hyperparam_search.csv`` by default).  The optional
    ``hash_list_path`` argument is ignored (kept for backward compatibility).
    """
    del hash_list_path  # CSV row order is the source of truth for POD_index.
    idx = int(pod_index)
    paths = _search_csv_paths(csv_path)
    if not paths:
        raise FileNotFoundError(
            f"No POD hyperparameter search CSV found "
            f"(expected {POD_SEARCH_RESULTS_CSV.name})"
        )
    path = paths[0]
    df = load_pod_search_results(path)
    if 0 <= idx < len(df):
        return df.iloc[idx]
    raise IndexError(
        f"POD_index={idx} out of range for {len(df)} rows in {path.name}"
    )


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


def pod_hyperparams_for_index(
    pod_index: int,
    *,
    require_fit_cache: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Any], float, str]:
    """Resolve a CSV row and verify its associated fit cache by default."""
    row = pod_row_for_index(pod_index, **kwargs)
    resolved = pod_hyperparams_from_row(row)
    pod_hash = resolved[2]
    if require_fit_cache and not any(
        path.is_file() for path in pod_hyperparam_search_cache_paths(pod_hash)
    ):
        tried = ", ".join(str(p) for p in pod_hyperparam_search_cache_paths(pod_hash))
        raise FileNotFoundError(
            f"No POD fit cache associated with CSV POD_index={int(pod_index)} "
            f"hash={pod_hash}; tried: {tried}"
        )
    return resolved


def pod_energy_best_fit_cache_path(
    hyperparams: dict[str, Any],
    *,
    regularization: float = 1e-12,
    include_intralayer: bool = False,
    pod_hash: str = "",
    best_fit_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Path for ``POD_energy`` best-fit cache (same naming as :func:`get_MCMC_inputs`)."""
    from blg_model_builder.potentials import ncoeff_from_params

    ncoeffs = ncoeff_from_params(hyperparams)
    reg_tag = f"reg{float(regularization):.0e}"
    intra_tag = "_intra" if include_intralayer else ""
    hash_tag = f"_{str(pod_hash).strip()}" if str(pod_hash).strip() else ""
    root = Path(best_fit_dir) if best_fit_dir is not None else _default_uq_root()
    return (
        root
        / "best_fit_params"
        / f"POD_energy_{ncoeffs}_{reg_tag}{intra_tag}{hash_tag}_best_fit_params.npz"
    )


def pod_hyperparam_search_cache_paths(pod_hash: str) -> list[Path]:
    """Candidate fit-cache paths associated with a search-CSV hash."""
    h = str(pod_hash).strip()
    if not h:
        return []
    # Resolve at call time so CWD / chdir in callers is respected.
    pod_dir = _pod_dir()
    cache_root = pod_dir / "pod_hyperparam_search_cache"
    paths = [
        cache_root / "POD_energy" / f"{h}.npz",
        cache_root / "POD_energy_MBD" / f"{h}.npz",
        cache_root / f"{h}.npz",
    ]
    # Also pick up other LOT / model subdirs (e.g. POD_energy_DFTD3).
    if cache_root.is_dir():
        for sub in sorted(cache_root.glob(f"*/{h}.npz")):
            if sub not in paths:
                paths.append(sub)
    return paths


def load_pod_hyperparam_search_fit(pod_hash: str) -> Optional[np.ndarray]:
    """
    Load POD coefficients from ``pod_hyperparam_search_cache`` (hyperparameter search).

    Returns ``None`` when no cache file exists for *pod_hash*.
    """
    h = str(pod_hash).strip()
    if not h:
        return None
    for path in pod_hyperparam_search_cache_paths(h):
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=True)
        if "pod_params" in data.files:
            return np.asarray(data["pod_params"], dtype=float).ravel()
        if "params" in data.files:
            return np.asarray(data["params"], dtype=float).ravel()
    return None


def write_pod_energy_best_fit_caches_from_tightened_csv(
    *,
    csv_path: Optional[Path] = None,
    force: bool = False,
    supercells: int = 1,
) -> list[Path]:
    """
    Write ``best_fit_params/POD_energy_*_best_fit_params.npz`` for every row in the
    tightened hyperparameter-search CSV.

    Coefficients are taken from ``pod_hyperparam_search_cache`` when available;
    otherwise :func:`get_MCMC_inputs` falls back to ``fit_pod``.  Each ``.npz`` uses
    the same keys as the ``POD_energy`` branch of :func:`get_MCMC_inputs`
    (``params``, ``bounds``, ``ypred_bestfit``, plus hyperparameter fields).
    """
    import os

    from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs

    csv_path = Path(csv_path) if csv_path is not None else POD_SEARCH_RESULTS_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"POD search results not found: {csv_path}")

    uq_root = _default_uq_root()
    prev_cwd = os.getcwd()
    os.chdir(uq_root)
    written: list[Path] = []
    try:
        df = pd.read_csv(csv_path)
        for idx in range(len(df)):
            row = df.iloc[idx]
            pod_hp, rcut, pod_hash = pod_hyperparams_from_row(row)
            reg = float(row.get("regularization", 1e-12))
            include_intra = bool(row.get("include_intralayer", False))
            out_path = pod_energy_best_fit_cache_path(
                pod_hp,
                regularization=reg,
                include_intralayer=include_intra,
                pod_hash=pod_hash,
            )
            if out_path.is_file() and not force:
                print(f"[export] POD_index={idx}  exists  {out_path.name}", flush=True)
                written.append(out_path)
                continue
            if out_path.is_file() and force:
                out_path.unlink()

            search_params = load_pod_hyperparam_search_fit(pod_hash)
            src = "search cache" if search_params is not None else "fit_pod"
            print(
                f"[export] POD_index={idx}  hash={pod_hash}  ({src}) …",
                flush=True,
            )
            kw: dict[str, Any] = {
                "pod_hyperparams": pod_hp,
                "pod_cutoff": rcut,
                "pod_hash": pod_hash,
                "regularization": reg,
                "weight_energy": float(row.get("weight_energy", 1000.0)),
                "weight_force": float(row.get("weight_force", 1.0)),
                "include_intralayer": include_intra,
                "skip_diagnostics": True,
            }
            get_MCMC_inputs("POD_energy", supercells=supercells, **kw)
            if not out_path.is_file():
                raise RuntimeError(
                    f"get_MCMC_inputs did not write expected cache: {out_path}"
                )
            written.append(out_path)
            print(f"[export] wrote {out_path}", flush=True)
    finally:
        os.chdir(prev_cwd)
    return written


def _default_uq_root() -> Path:
    return _pod_dir().parent


def find_pod_energy_ensemble_folder(
    pod_hash: str,
    ensemble_dir: Union[str, Path],
    *,
    pod_index: Optional[int] = None,
) -> Optional[str]:
    """Return ensemble subfolder name for a POD hash (exact index name first)."""
    root = Path(ensemble_dir)
    h = str(pod_hash).lower()
    if pod_index is not None:
        exact = f"POD_energy_POD_index_{int(pod_index)}_{pod_hash}"
        if (root / exact).is_dir():
            return exact

    matches: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("POD_energy_POD_index_"):
            continue
        if h in name.lower():
            matches.append(name)
    if not matches:
        return None
    if pod_index is not None:
        for name in matches:
            if name.startswith(f"POD_energy_POD_index_{int(pod_index)}_"):
                return name
    return matches[0]


def pod_energy_ensemble_names_from_csv(
    ensemble_dir: Union[str, Path] = "ensembles",
    *,
    csv_path: Optional[Path] = None,
    passing_only: bool = True,
    require_fit_cache: bool = True,
) -> list[str]:
    """Map POD search CSV rows to existing POD ensemble folders.

    CSV row number is preserved as ``POD_index``. By default only rows with
    ``meets_all_criteria`` and an associated hyperparameter-search fit cache
    are eligible for propagation.
    """
    df = load_pod_search_results(csv_path)
    names: list[str] = []
    for pod_index, row in df.iterrows():
        if passing_only and not _meets_all_criteria_bool(row["meets_all_criteria"]):
            continue
        pod_hash = str(row["hash"])
        if require_fit_cache and not any(
            path.is_file() for path in pod_hyperparam_search_cache_paths(pod_hash)
        ):
            print(
                f"Warning: no POD search fit cache for CSV row {pod_index} "
                f"hash={pod_hash!r}; skipping.",
                flush=True,
            )
            continue
        folder = find_pod_energy_ensemble_folder(
            pod_hash, ensemble_dir, pod_index=int(pod_index),
        )
        if folder is None:
            print(
                f"Warning: no ensemble folder for CSV POD_index={pod_index} "
                f"hash={pod_hash!r} under {ensemble_dir!r}",
                flush=True,
            )
            continue
        names.append(folder)
    return names


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect tightened-CSV POD rows or export their fit caches."
    )
    p.add_argument(
        "--export-pod-best-fit-from-search",
        action="store_true",
        help="Write best_fit_params/POD_energy_* caches from tightened CSV rows "
        "(coefficients from pod_hyperparam_search_cache when available).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --export-pod-best-fit-from-search, overwrite existing caches.",
    )
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    if args.export_pod_best_fit_from_search:
        paths = write_pod_energy_best_fit_caches_from_tightened_csv(
            csv_path=args.csv,
            force=args.force,
        )
        print(f"Exported {len(paths)} POD_energy best-fit cache(s).", flush=True)
    else:
        rows = load_pod_search_results(args.csv)
        for pod_index, row in rows.iterrows():
            print(
                f"POD_index={pod_index} hash={row['hash']} "
                f"pass={row['meets_all_criteria']}"
            )
