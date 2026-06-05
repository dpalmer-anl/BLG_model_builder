"""
Shared helpers for verifying physical requirements in
`tests/physical_properties_models.md` across implemented BLG models.

This module is intentionally lightweight and repo-relative so it works both in
WSL and on native Linux.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import datetime

def _ensure_importable_package() -> None:
    """
    Prefer the *installed* package (e.g. conda env `blg_uq`).
    Only fall back to the repo `src/` layout if import fails.
    """
    try:
        import blg_model_builder  # noqa: F401
        return
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # If still not importable, create a minimal namespace package pointing at src/.
    try:
        import blg_model_builder  # noqa: F401
        return
    except Exception:
        import types

        pkg = types.ModuleType("blg_model_builder")
        pkg.__path__ = [str(src)]  # type: ignore[attr-defined]
        sys.modules["blg_model_builder"] = pkg


_ensure_importable_package()

# Physical-property checks require the LAMMPS Python module.
# Do NOT skip: fail loudly so missing installs are caught in CI and locally.
try:
    import lammps  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The LAMMPS Python module is required for physical-property tests. "
        "Build LAMMPS with -DWITH_PYTHON=yes and run `make install-python`. "
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc

from blg_model_builder.geom_tools import get_bilayer_atoms
from blg_model_builder.potentials import (
    DRIPASECalculator,
    PODASECalculator,
    TersoffDRIPASECalculator,
    TersoffKolmogorovCrespiASECalculator,
    ncoeff_from_params,
)

# -----------------------------------------------------------------------------
# Artifacts (plots / CSV)
# -----------------------------------------------------------------------------

def artifacts_dir() -> Path:
    """
    Directory for test artifacts (CSV/PNG).

    Override with env var `BLG_TEST_ARTIFACTS_DIR`. Defaults to `tests/_artifacts`.
    """
    env = os.environ.get("BLG_TEST_ARTIFACTS_DIR")
    if env:
        d = Path(env).expanduser()
    else:
        d = Path(__file__).resolve().parent / "_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timestamp_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def save_scan_csv(d_grid: np.ndarray, energies: np.ndarray, *, stem: str) -> Path:
    out = artifacts_dir() / f"{stem}.csv"
    arr = np.column_stack([np.asarray(d_grid, float), np.asarray(energies, float)])
    header = "d_A,energy_eV"
    np.savetxt(out, arr, delimiter=",", header=header, comments="")
    return out


def save_scan_plot(d_grid: np.ndarray, energies: np.ndarray, *, stem: str, title: str) -> Optional[Path]:
    """
    Save a simple energy vs separation plot. Returns the PNG path or None if
    matplotlib isn't available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out = artifacts_dir() / f"{stem}.png"
    d = np.asarray(d_grid, float)
    e = np.asarray(energies, float)
    e0 = np.nanmin(e)
    plt.figure(figsize=(6.0, 4.0), dpi=160)
    plt.plot(d, e - e0, marker="o", lw=1.2, ms=3.2)
    plt.xlabel("Interlayer separation d (Å)")
    plt.ylabel("ΔE (eV) relative to min")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


# -----------------------------------------------------------------------------
# Paths / parameter loading
# -----------------------------------------------------------------------------

def _repo_root() -> Path:
    # tests/physical_properties_harness.py -> repo root
    return Path(__file__).resolve().parents[1]


def best_fit_params_dir() -> Path:
    """
    Prefer repo-local `uncertainty_quantification/best_fit_params`.
    Many scripts also write to `best_fit_params/` relative to CWD; we support
    that as a secondary option.
    """
    root = _repo_root()
    cand = [
        root / "uncertainty_quantification" / "best_fit_params",
        root / "best_fit_params",
    ]
    for d in cand:
        if d.exists() and d.is_dir():
            return d
    # Return preferred location even if it doesn't exist yet (caller can decide).
    return cand[0]


def _npz_params(path: Path) -> np.ndarray:
    d = np.load(str(path), allow_pickle=True)
    if "params" not in d:
        raise KeyError(f"{path} missing 'params' key")
    return np.asarray(d["params"], dtype=float).ravel()


def load_best_fit_estimate_tersoff() -> np.ndarray:
    pdir = best_fit_params_dir()
    f = pdir / "Tersoff_best_fit_params_estimate.npz"
    if f.exists():
        return _npz_params(f)

    # Fallback to the known good default used in existing tests:
    # m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A
    return np.asarray(
        [
            3.0,
            1.0,
            0.0,
            38049.0,
            4.3484,
            -0.57058,
            0.72751,
            1.5724e-7,
            2.2119,
            346.74,
            2.85,
            0.15,
            3.4879,
            1393.6,
        ],
        dtype=float,
    )


def load_best_fit_estimate_kc() -> np.ndarray:
    pdir = best_fit_params_dir()
    f = pdir / "Kolmogorov_Crespi_best_fit_params_estimate.npz"
    if f.exists():
        return _npz_params(f)

    # Fallback: values hard-coded in tests as "estimate".
    # [z0,C0,C2,C4,C,delta,lambda,A]
    return np.asarray(
        [
            3.416084,
            20.021583,
            10.9055107,
            4.2756354,
            1.0010836e-2,
            0.8447122,
            2.9360584,
            14.3132588,
        ],
        dtype=float,
    )


def load_best_fit_estimate_drip() -> np.ndarray:
    pdir = best_fit_params_dir()
    f = pdir / "DRIP_best_fit_params_estimate.npz"
    if f.exists():
        return _npz_params(f)

    # Fallback: parameters used in write_potential_files.py.
    # [C0, C2, C4, C, delta, lambda, A, z0]
    return np.asarray([15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238, 3.34], dtype=float)


def _fit_and_save_pod_energy_params(hyperparams: dict, save_path: Path) -> np.ndarray:
    """Fit a POD_energy model and save coefficients to *save_path*.

    Changes to ``uncertainty_quantification/`` so that DataLoader resolves
    ``../data`` correctly, then restores the working directory on exit.
    Requires training data in ``data/`` and a LAMMPS executable for ``fitpod``.
    """
    uq_dir = _repo_root() / "uncertainty_quantification"
    if not uq_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot auto-fit: uncertainty_quantification/ not found at {uq_dir}."
        )
    orig_dir = os.getcwd()
    try:
        os.chdir(str(uq_dir))
        if str(uq_dir) not in sys.path:
            sys.path.insert(0, str(uq_dir))

        from blg_model_builder.potentials import pod_hyperparams_to_str  # type: ignore
        from blg_model_builder.DataLoader import load_data_for_model  # type: ignore
        from model_fit import fit_pod  # type: ignore

        xdata_tr, _, _, _, _, _ = load_data_for_model("POD_energy", supercells=1)
        atoms_list = xdata_tr["energy"]
        rcut = 6.0
        hyperparams_str = pod_hyperparams_to_str(hyperparams, rcut, ["C"])
        print(
            f"[physical_properties_harness] Fitting POD_energy "
            f"(ncoeffs={int(ncoeff_from_params(hyperparams))}) — this may take a while …",
            flush=True,
        )
        best_fit = fit_pod(hyperparams_str, atoms_list)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(save_path), params=best_fit)
        print(f"[physical_properties_harness] Saved POD_energy params → {save_path}", flush=True)
        return np.asarray(best_fit, dtype=float).ravel()
    finally:
        os.chdir(orig_dir)


def load_pod_best_fit_energy_params(
    hyperparams: dict,
    *,
    require_file: bool = True,
) -> np.ndarray:
    """
    Load POD best-fit coefficients from:
        uncertainty_quantification/best_fit_params/POD_energy_<ncoeffs>_best_fit_params.npz

    If the file does not exist, an automatic fit is attempted via ``fitpod``.
    *require_file* controls whether a ``FileNotFoundError`` is raised when the
    file is missing **and** the auto-fit also fails.  Set ``require_file=False``
    to receive an empty array instead.
    """
    hp = dict(hyperparams)
    if "species" not in hp:
        hp["species"] = ["C"]
    ncoeffs = int(ncoeff_from_params(hp))
    pdir = best_fit_params_dir()
    f = pdir / f"POD_energy_{ncoeffs}_best_fit_params.npz"
    if f.exists():
        params = _npz_params(f)
        if params.size != ncoeffs:
            raise ValueError(
                f"{f} has params length {params.size}, expected {ncoeffs} from hyperparams"
            )
        return params

    # File missing — attempt auto-fit before giving up.
    print(
        f"[physical_properties_harness] {f.name} not found; attempting auto-fit …",
        flush=True,
    )
    try:
        return _fit_and_save_pod_energy_params(hp, f)
    except Exception as fit_exc:
        if require_file:
            raise FileNotFoundError(
                f"Missing POD best-fit coefficients: {f}. "
                f"Auto-fit also failed: {fit_exc}. "
                f"Generate it under {pdir} manually."
            ) from fit_exc
        return np.array([])


# -----------------------------------------------------------------------------
# Geometry helpers / metrics
# -----------------------------------------------------------------------------

def layer_tags_from_mol_id(atoms) -> list[int]:
    """Convert atoms.arrays['mol-id'] to 0-based layer indices for calculators."""
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    return np.searchsorted(np.unique(mol), mol).astype(int).tolist()


def make_blg_ab(d: float, *, a: float = 2.46, sc: int = 1):
    """AB-stacked bilayer graphene using existing builder convention."""
    return get_bilayer_atoms(d, 0.0, a=float(a), sc=int(sc))


def make_blg_aa(d: float, *, a: float = 2.46, sc: int = 1):
    """
    AA-stacked bilayer graphene.

    In this repo's convention, `disregistry` is defined so AB→AB is 1.0 and the
    AB structure is at disregistry=0.0. The in-plane shift from AB to AA is a
    single bond length, i.e. (a1+a2)/3, which corresponds to disregistry=1/3.
    """
    return get_bilayer_atoms(d, 1.0 / 3.0, a=float(a), sc=int(sc))


@dataclass(frozen=True)
class LayerMetrics:
    separation: float
    buckling_layer0: float
    buckling_layer1: float


def compute_layer_metrics(atoms) -> LayerMetrics:
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float)
    mol = np.asarray(atoms.get_array("mol-id"), dtype=int).ravel()
    layers = np.searchsorted(np.unique(mol), mol).astype(int)
    if np.unique(layers).size != 2:
        raise ValueError("Expected exactly 2 layers from mol-id")
    z0 = pos[layers == 0, 2]
    z1 = pos[layers == 1, 2]
    separation = float(z1.mean() - z0.mean())
    return LayerMetrics(
        separation=separation,
        buckling_layer0=float(np.std(z0)),
        buckling_layer1=float(np.std(z1)),
    )


def perturb_positions(atoms, *, sigma: float = 0.01, seed: int = 0):
    out = atoms.copy()
    rng = np.random.default_rng(int(seed))
    out.set_positions(out.get_positions() + rng.normal(scale=float(sigma), size=(len(out), 3)))
    return out


# -----------------------------------------------------------------------------
# Model constructors
# -----------------------------------------------------------------------------

def make_calc_tersoff_kc_best_fit_estimate():
    te = load_best_fit_estimate_tersoff()
    kc = load_best_fit_estimate_kc()
    # Use a generous KC cutoff (consistent with default in potentials.py hybrid class).
    return te, kc


def make_calc_tersoff_drip_best_fit_estimate():
    te = load_best_fit_estimate_tersoff()
    drip = load_best_fit_estimate_drip()
    return te, drip


# POD hyperparams: keep in one place for BLG physical checks.
POD_BLG_HYPERPARAMS = {
    "species": ["C"],
    "bessel_polynomial_degree": 4,
    "inverse_polynomial_degree": 8,
    "twobody_number_radial_basis_functions": 10,
    "threebody_number_radial_basis_functions": 8,
    "threebody_angular_degree": 4,
    "fourbody_number_radial_basis_functions": 6,
    "fourbody_angular_degree": 3,
    "fivebody_number_radial_basis_functions": 4,
    "fivebody_angular_degree": 3,
    "sixbody_number_radial_basis_functions": 3,
    "sixbody_angular_degree": 2,
    "sevenbody_number_radial_basis_functions": 2,
    "sevenbody_angular_degree": 2,
}


# -----------------------------------------------------------------------------
# Rigid scans
# -----------------------------------------------------------------------------

def rigid_scan_interlayer_minimum(
    atoms_builder: Callable[[float], object],
    calc_builder: Callable[[object], object],
    *,
    d_grid: np.ndarray,
) -> Tuple[float, np.ndarray]:
    e = np.empty_like(d_grid, dtype=float)
    for i, d in enumerate(d_grid):
        atoms = atoms_builder(float(d))
        atoms.calc = calc_builder(atoms)
        e[i] = float(atoms.get_potential_energy())
    d_min = float(d_grid[int(np.argmin(e))])
    return d_min, e


def relax_atoms(atoms, *, backend: str = "ase", ftol: float = 1e-5, maxiter: int = 500):
    """
    Relax using the calculator's `relax_structure` implementation (patched onto ASE Atoms).
    Returns a new Atoms instance.
    """
    if atoms.calc is None:
        raise ValueError("atoms.calc must be set before relax_atoms()")
    return atoms.relax_structure(ftol=float(ftol), maxiter=int(maxiter), relax_backend=str(backend))


def assert_no_buckling(metrics: LayerMetrics, *, max_std: float = 0.02):
    """Fail if per-layer z-std indicates in-plane buckling."""
    assert metrics.buckling_layer0 <= max_std, f"layer0 buckling std(z)={metrics.buckling_layer0:.4f} Å"
    assert metrics.buckling_layer1 <= max_std, f"layer1 buckling std(z)={metrics.buckling_layer1:.4f} Å"


