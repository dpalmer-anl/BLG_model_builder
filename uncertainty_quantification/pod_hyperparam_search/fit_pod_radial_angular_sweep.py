"""
Sweep POD potential fits (LAMMPS fitpod) over shared 2-/3-body radial counts and
3-body angular degree.

Mirrors ``get_MCMC_inputs.py`` / ``POD_energy`` when no cache exists:
``hyperparams_str = pod_hyperparams_to_str(...)`` then
``params = fit_pod(hyperparams_str, xdata_train["energy"])``, then
``PODASECalculator(hyperparams, params, elements=["C"], cutoff=rcut)``.

Run from anywhere; the script ``chdir``\ s to this folder so ``tmp_pod_fit`` and
data paths match the working example.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from blg_model_builder.DataLoader import load_data_for_model
from blg_model_builder.potentials import PODASECalculator, pod_hyperparams_to_str
from blg_model_builder.model_fit import fit_pod


# Identical template to ``get_MCMC_inputs.py`` / ``POD_energy`` (lines 238–245).
def pod_hyperparams_template() -> Dict[str, Any]:
    # return {
    #     "species": ["C"],
    #     "bessel_polynomial_degree": 4,
    #     "inverse_polynomial_degree": 8,
    #     "twobody_number_radial_basis_functions": 10,
    #     "threebody_number_radial_basis_functions": 8,
    #     "threebody_angular_degree": 4,
    #     "fourbody_number_radial_basis_functions": 6,
    #     "fourbody_angular_degree": 3,
    #     "fivebody_number_radial_basis_functions": 4,
    #     "fivebody_angular_degree": 3,
    #     "sixbody_number_radial_basis_functions": 3,
    #     "sixbody_angular_degree": 2,
    #     "sevenbody_number_radial_basis_functions": 2,
    #     "sevenbody_angular_degree": 2,
    # }
    return {
        "species": ["C"],
        "bessel_polynomial_degree": 4,
        "inverse_polynomial_degree": 8,
        "twobody_number_radial_basis_functions": 10,
        "threebody_number_radial_basis_functions": 8,
        "threebody_angular_degree": 4,
        "fourbody_number_radial_basis_functions": 1,
        "fourbody_angular_degree": 1,
        "fivebody_number_radial_basis_functions": 1,
        "fivebody_angular_degree": 1,
        "sixbody_number_radial_basis_functions": 1,
        "sixbody_angular_degree": 1,
        "sevenbody_number_radial_basis_functions": 1,
        "sevenbody_angular_degree": 1,
    }


def mae_energy_per_atom(
    calc: PODASECalculator,
    atoms_list: list,
    energies_ref: np.ndarray,
) -> float:
    """Mean absolute error on total energy / n_atoms (eV/atom)."""
    e_ref = np.asarray(energies_ref, dtype=float)
    pred = np.array([calc.get_potential_energy(at) for at in atoms_list], dtype=float)
    nat = np.array([len(at) for at in atoms_list], dtype=float)
    return float(np.mean(np.abs(pred / nat - e_ref / nat)))


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="POD fitpod sweep: test MAE vs shared 2b/3b radial count per angular degree."
    )
    parser.add_argument("--supercells", type=int, default=1)
    parser.add_argument("--rcut", type=float, default=6.0)
    parser.add_argument("--n-rad-min", type=int, default=6)
    parser.add_argument("--n-rad-max", type=int, default=10)
    parser.add_argument("--l3-min", type=int, default=1)
    parser.add_argument("--l3-max", type=int, default=6)
    parser.add_argument(
        "--lmp",
        type=str,
        default=None,
        help="LAMMPS executable for fit_pod (optional). If omitted, uses fit_pod default "
        "(same as get_MCMC_inputs, which does not pass lammps_exec).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/pod_radial_angular_sweep_mae.png",
    )
    args = parser.parse_args()

    n_rad_values = list(range(args.n_rad_min, args.n_rad_max + 1))
    l3_values = list(range(args.l3_min, args.l3_max + 1))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    np.random.seed(42)
    xdata_train, xdata_test, _, _, ydata_test, _ = load_data_for_model(
        "POD_energy",
        supercells=args.supercells,
    )

    plt.figure(figsize=(8.5, 5.5))

    for l3 in l3_values:
        maes: list[float] = []
        for n_rad in n_rad_values:
            hyperparams = pod_hyperparams_template()
            hyperparams["twobody_number_radial_basis_functions"] = int(n_rad)
            hyperparams["threebody_number_radial_basis_functions"] = int(n_rad)
            hyperparams["threebody_angular_degree"] = int(l3)

            hyperparams_str = pod_hyperparams_to_str(hyperparams, args.rcut, ["C"])
            print(f"threebody_angular_degree={l3}  n_rad={n_rad}")

            try:
                if args.lmp is not None:
                    params_energy = fit_pod(
                        hyperparams_str,
                        xdata_train["energy"],
                        lammps_exec=args.lmp,
                    )
                else:
                    params_energy = fit_pod(
                        hyperparams_str,
                        xdata_train["energy"],
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  fit_pod failed: {exc}")
                if os.path.isdir(os.path.join(script_dir, "tmp_pod_fit")):
                    shutil.rmtree(
                        os.path.join(script_dir, "tmp_pod_fit"),
                        ignore_errors=True,
                    )
                maes.append(float("nan"))
            else:
                params_energy = np.atleast_1d(
                    np.squeeze(np.asarray(params_energy, dtype=float))
                ).ravel()
                try:
                    eval_calc = PODASECalculator(
                        hyperparams,
                        params_energy,
                        elements=["C"],
                        cutoff=args.rcut,
                    )
                    mae = mae_energy_per_atom(
                        eval_calc,
                        xdata_test["energy"],
                        ydata_test["energy"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  evaluation failed: {exc}")
                    mae = float("nan")
                maes.append(mae)
                print(f"  test MAE (eV/atom) = {mae:.6g}")
            finally:
                try:
                    os.chdir(script_dir)
                except OSError:
                    pass

        plt.plot(
            n_rad_values,
            maes,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"threebody_angular_degree = {l3}",
        )

    plt.xlabel(r"$n_\mathrm{rad}$ (2-body = 3-body radial basis count)")
    plt.ylabel("Test MAE (eV / atom)")
    plt.title("POD linear fit (LAMMPS fitpod): test energy MAE vs radial basis")
    plt.legend(title="3-body angular")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
