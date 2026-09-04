#!/usr/bin/env python3
"""
Lightweight smoke tests for all non-TETB models in the UQ workflow.

Run from ``uncertainty_quantification/`` with the ``blg_uq`` conda env::

    python smoke_test_models.py

Verifies wiring through ``get_MCMC_inputs`` (data load + best-fit cache/fit).
Skips TETB_POD. LAMMPS-backed models require a working LAMMPS Python module.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ENERGY_MODELS = [
    "Tersoff",
    "Kolmogorov_Crespi",
    "DRIP",
    "Tersoff+DRIP",
    "Tersoff+Kolmogorov_Crespi",
    "POD_energy",
]

OPTIONAL_ENERGY_MODELS = [
    ("Allegro_energy", {"allegro_device": "cpu"}),
    ("POD+extep+ILP", {}),
    ("POD+LJ_continuum", {}),
]

TB_MODELS = [
    "MK",
    "ACSF_hoppings_M_8_W_6",
    "ACSF_hoppings_sk_M_8_W_6",
    "LETB_interlayer",
]

TB_KW = {
    "LETB_intralayer": {"nn_val": 1},
}


def _test_model(name: str, **kw) -> tuple[bool, str]:
    try:
        from blg_model_builder.get_MCMC_inputs import get_MCMC_inputs

        sc = 2 if name in ("DRIP", "Tersoff+DRIP", "Tersoff+Kolmogorov_Crespi") else 1
        out = get_MCMC_inputs(name, supercells=sc, skip_diagnostics=True, **kw)
        calc, *_rest, params, bounds = out
        keys = list(calc.keys())
        npar = sum(len(params[k]) for k in calc)
        return True, f"calc_keys={keys} n_params={npar}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}"


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    for name in TB_MODELS:
        ok, msg = _test_model(name, **TB_KW.get(name, {}))
        results.append((name, ok, msg))

    for nn in (1, 2, 3):
        name = "LETB_intralayer"
        ok, msg = _test_model(name, nn_val=nn)
        results.append((f"{name}_nn{nn}", ok, msg))

    for name in ENERGY_MODELS:
        ok, msg = _test_model(name, M=8, W=6)
        results.append((name, ok, msg))

    for name, kw in OPTIONAL_ENERGY_MODELS:
        try:
            import nequip  # noqa: F401
        except ImportError:
            results.append((name, True, "SKIP (nequip-allegro not installed)"))
            continue
        ok, msg = _test_model(name, **kw)
        results.append((name, ok, msg))

    print("\n=== Smoke test results ===")
    failed = 0
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg[:200]}")
        if not ok:
            failed += 1

    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
