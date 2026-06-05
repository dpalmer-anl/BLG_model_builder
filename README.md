blg-model-builder-v2
====================

Utilities for bilayer-graphene interatomic models, descriptor construction,
data loading, and Bayesian uncertainty quantification.  Interatomic potentials
are evaluated through the official **LAMMPS Python module**
(`from lammps import lammps`).

The package is **pure Python** — no C++ compilation required.

---

## Interatomic potential calculators

All calculators are ASE-compatible (implement `calculate`, `get_potential_energy`,
`get_forces`, `relax_structure`, `set_parameters`, `prepare_batch`, `evaluate_batch`).

| Role | Class | Free parameters / notes |
|---|---|---|
| Intralayer (Tersoff) | `TersoffLammpsCalculator` | 14 floats: BNC Tersoff row for C–C–C |
| Interlayer only (Kolmogorov–Crespi full) | `KolmogorovCrespiLammpsCalculator` | 8 floats: KC Full pair parameters |
| Interlayer only (DRIP) | `DRIPLammpsCalculator` | 8 floats: DRIP pair parameters |
| **Hybrid** Tersoff + KC | `TersoffKCLammpsCalculator` | 14 Tersoff + 8 KC (`hybrid/overlay` Tersoff + `kolmogorov/crespi/full`) |
| **Hybrid** Tersoff + DRIP | `TersoffDRIPLammpsCalculator` | 14 Tersoff + 8 DRIP (`hybrid/overlay` Tersoff + `drip`) |
| **POD** (ML-POD linear potential) | `PODLammpsCalculator` | **Fixed** multi-body descriptor (`.pod` content from a hyperparameter dict) + **N** linear `fitpod` / least-squares coefficients. `N` = `ncoeff_from_params(hyperparams)` |
| Tight-binding + POD residual (electronic + short-range) | `TETB_PODLammpsCalculator` (alias `TETB_PODASECalculator`) | TB parameters, POD `hyperparams` and `pod_params`, cutoffs, k-points, Ewald, etc. — for semi-empirical BLG with a POD interlayer/intralayer correction |

Backward-compatible names (`*ASECalculator` aliases) are re-exported from
`blg_model_builder_v2.potentials` (see the module docstring in `src/potentials.py`).

### Tersoff

Models **intralayer** covalent C–C bonding in a classical many-body Tersoff form
(`pair_style tersoff`).  Often combined with a separate interlayer model via a
**hybrid** calculator (Tersoff+KC or Tersoff+DRIP) for BLG.  The 14-parameter
ordering matches a LAMMPS Tersoff table row; see [Parameter reference](#tersoff-14-values).

### Kolmogorov–Crespi (full) and DRIP (interlayer-only)

- **KC full** — registry-dependent interlayer energy between graphene layers
  (`pair_style kolmogorov/crespi/full` in the INTERLAYER package).  Requires
  `atom_style full` and per-atom `mol-id` to distinguish layers.
- **DRIP** — alternative interlayer potential (`drip`); same molecular-ID
  requirements.

Use the **stand-alone** 8-float calculators when you only need the interlayer
term; use **hybrid** classes when a single `Atoms` object should run Tersoff for
in-plane and KC/DRIP for out-of-plane in one LAMMPS instance.

### POD (Potential-Optimized Descriptor)

POD is a **linear** potential in a fixed, multi-body **descriptor** basis.  In
LAMMPS, `pair pod` evaluates energies and forces from a coefficient vector; the
**architecture** of the basis (body orders, radial counts, angular degree,
Bessel / inverse polynomial degrees) is set by a **hyperparameter dict**, not
by the coefficient vector.  Coefficient count is

`len(pod_params) == N == ncoeff_from_params(hyperparams)`.

**Typical use**

- Fit `pod_params` to DFT (or other) training data (e.g. LAMMPS `fitpod` or
  the project’s `uncertainty_quantification` / `get_MCMC_inputs` workflows).
- Evaluate new structures with a fixed `hyperparams` + `pod_params` and a
  real-space **cutoff** `rcut` that must match the descriptor and `pair pod`
  input.

**Python helpers (same module)**

- `ncoeff_from_params(hyperparams)` — coefficient dimension for a given architecture.
- `pod_hyperparams_to_str(hyperparams, cutoff, elements)` — text for a `.pod` file.
- `init_pod_coefficients(hyperparams)` — zero vector of the correct length.

A minimal POD constructor call looks like:

```python
import numpy as np
from blg_model_builder_v2.potentials import PODLammpsCalculator, ncoeff_from_params

hp = {
    "species": ["C"],
    "bessel_polynomial_degree": 4,
    "inverse_polynomial_degree": 8,
    "twobody_number_radial_basis_functions": 10,
    "threebody_number_radial_basis_functions": 4,
    "threebody_angular_degree": 2,
    "fourbody_number_radial_basis_functions": 0,
    "fourbody_angular_degree": 0,
    "fivebody_number_radial_basis_functions": 0,
    "fivebody_angular_degree": 0,
    "sixbody_number_radial_basis_functions": 0,
    "sixbody_angular_degree": 0,
    "sevenbody_number_radial_basis_functions": 0,
    "sevenbody_angular_degree": 0,
}
p = np.asarray([...], dtype=float)  # length ncoeff_from_params(hp), e.g. from fitpod
calc = PODLammpsCalculator(hp, p, elements=["C"], cutoff=6.0)
```

> **POD limitation:** behavior outside the DFT (or training) **volume of
> configuration space** (e.g. very large, twisted supercells) can be poor;
> the linear model is only as reliable as the data and the chosen basis size.

Build LAMMPS with **`-DPKG_ML-POD=yes`**; see [Troubleshooting](#troubleshooting).

### TETB + POD

`TETB_PODLammpsCalculator` couples an empirical tight-binding (electronic) part
to a **POD** short-range / correction block.  It is the most flexible but also
the heaviest to fit and to run.  See `src/potentials.py` and
`get_MCMC_inputs` TETB+POD model tags in `uncertainty_quantification/`.

---

## How it works

Each calculator maintains a persistent LAMMPS instance (output suppressed with
`-screen none -log none`).  Hot path for MCMC:

1. **`set_parameters(params)`** — writes a temporary parameter file and
   re-runs `pair_style` / `pair_coeff` if parameters changed.
2. **Single-point (`calculate(atoms)`)** — injects positions via `scatter_atoms`
   when only positions changed; uses `clear` + `read_data` when atom count or
   cell changes.  Runs `run 0` and gathers energy / forces.
3. **Batch (`prepare_batch` + `evaluate_batch`)** — writes all ~400 structures
   to individual LAMMPS data files once, then iterates with `clear` / `read_data`
   / `run 0` per structure.
4. **Relaxation (`relax_structure(atoms)`)** — two backends:
   - `relax_backend='lammps'` (default): LAMMPS `min_style fire` / `minimize`.
   - `relax_backend='ase'`: ASE `FIRE` optimizer with position injection.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | |
| LAMMPS Python module | see below |
| ASE ≥ 3.22 | `pip install ase` |
| NumPy ≥ 1.22 | installed automatically |

---

## Installation

### 1 — Build LAMMPS with Python support

```bash
git clone https://github.com/cesmix-mit/lammps.git
cd lammps && mkdir build && cd build

cmake ../cmake \
    -DBUILD_SHARED_LIBS=yes \
    -DLAMMPS_EXCEPTIONS=yes \
    -DPKG_INTERLAYER=yes \
    -DPKG_MANYBODY=yes \
    -DPKG_ML-POD=yes \
    -DPKG_MOLECULE=yes \
    -DPKG_KSPACE=yes\
    -DWITH_PYTHON=yes

cmake --build . -- -j$(nproc)

# Install the lammps Python module into the active Python environment
make install-python
```

> **MOLECULE package**: `kolmogorov/crespi/full` and `drip` require
> `atom_style full` to distinguish interlayer atoms.  Build with
> `-DPKG_MOLECULE=yes` or the KC/DRIP calculators will error at runtime.

### 2 — Install blg-model-builder-v2

```bash
# From the repo root (pure Python, no compilation):
pip install -e .

# Or use the helper script:
bash clean_install.sh
```

---

## Quick usage

```python
from blg_model_builder_v2.potentials import (
    TersoffLammpsCalculator,
    TersoffKCLammpsCalculator,
    PODLammpsCalculator,
)

# Tersoff single-point energy + forces
calc = TersoffLammpsCalculator(params=[...14 values...], elements=["C"])
energy = calc.get_potential_energy(atoms)
forces = calc.get_forces(atoms)

# Relaxation (LAMMPS FIRE backend)
relaxed = calc.relax_structure(atoms, relax_backend="lammps")

# Batch evaluation for MCMC
calc.prepare_batch(atoms_list, elements=["C"])
energies, forces_list = calc.evaluate_batch(params=[...])

# Hybrid Tersoff + Kolmogorov–Crespi (14 + 8 parameters)
# from blg_model_builder_v2.potentials import TersoffKCLammpsCalculator
# calc = TersoffKCLammpsCalculator(tersoff_14, kc_8, kc_cutoff=20.0, elements=["C"])

# POD: fixed descriptor dict `hp` + fitted coefficient vector `p` (length ncoeff)
# from blg_model_builder_v2.potentials import PODLammpsCalculator, ncoeff_from_params
# assert len(p) == ncoeff_from_params(hp)
# calc = PODLammpsCalculator(hp, p, elements=["C"], cutoff=6.0)
```

---

## Parameter reference

Vectors are **ordered lists of floats** passed to `set_parameters` / `evaluate_batch`
in the same order the corresponding calculator expects (see `src/lammps_interface.py`).

### Tersoff (14 values)
```
[m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A]
```
Column order matches a LAMMPS `.tersoff` file (minus element symbols).  Used by
`TersoffLammpsCalculator` and as the first block in `TersoffKCLammpsCalculator` /
`TersoffDRIPLammpsCalculator`.

### Kolmogorov–Crespi full (8 values)
```
[z0, C0, C2, C4, C, delta, lambda, A]
```
Written to a KC file for `pair_style kolmogorov/crespi/full` (stand-alone
`KolmogorovCrespiLammpsCalculator` or the KC block of `TersoffKCLammpsCalculator`).

### DRIP (8 values)
```
[C0, C2, C4, C, delta, lambda, A, z0]
```
Physical DRIP parameters (stand-alone `DRIPLammpsCalculator` or the DRIP block
of `TersoffDRIPLammpsCalculator`).

### Hybrid Tersoff + interlayer
- **`TersoffKCLammpsCalculator`**: `tersoff_params` (14) and `kc_params` (8); optional
  `kc_cutoff` (Å, default 20) for the KC sub-style.
- **`TersoffDRIPLammpsCalculator`**: `tersoff_params` (14) and `drip_params` (8);
  optional `cutoff` / `rhocut` / `ncut` for the DRIP sub-style (see class docstring).

### POD: hyperparameters + coefficient vector
The **fit** is only in the **coefficient vector** `pod_params` of length `N`, where
`N = ncoeff_from_params(hyperparams)`.  The `hyperparams` **dict** fixes the
descriptor: species, Bessel / inverse polynomial degrees, and, for each body
order, radial basis counts and (where applicable) angular degrees.  A complete
key list and counting logic are in `ncoeff_from_params` (see `src/potentials.py`).

A separate scalar **`cutoff` / `rcut` (Å)** is passed to `PODLammpsCalculator(..., cutoff=...)`
and must be consistent with the string produced by `pod_hyperparams_to_str` for
`fitpod` / I/O.  Fitting is typically done in the `uncertainty_quantification/`
workflows; evaluation from Python is always `PODLammpsCalculator(hyperparams, pod_params, elements, cutoff)`.

### TETB + POD
Parameter layout is model-specific (TB matrix elements, POD block, Ewald, etc.).  
Construct via `TETB_PODLammpsCalculator` with keyword arguments as in
`get_MCMC_inputs` / the module docstring in `src/potentials.py`.

---

## Uncertainty quantification: ensemble scripts

Run these from the `uncertainty_quantification/` directory so paths to `data/`
and `best_fit_params/` resolve as in `get_MCMC_inputs`. You need a working
LAMMPS build (and `emcee` for the MCMC script). Both tools write **pickled
ensembles** under `uncertainty_quantification/ensembles/<model_name>/` for
downstream calibration and comparison scripts.

### `EMCEE_generate_ensemble.py` — MCMC posteriors with [emcee](https://emcee.readthedocs.io/)

**Typical workflow**

1. Ensure `get_MCMC_inputs` has already produced the relevant best-fit caches
   (or the script loads data and reference fits as for other UQ entry points).
2. Choose a `model_name` (e.g. `Tersoff+DRIP`, `TETB_POD`, or taggable models
   like `POD_energy` / `ACSF_hoppings` with `-M` / `-W`).
3. Run **serial** for debugging, then scale with **multiprocessing** on one node
   or **MPI** on a cluster (each worker holds its own LAMMPS instance).

**Examples**

```bash
cd uncertainty_quantification

# Serial (default)
python EMCEE_generate_ensemble.py -m Tersoff+DRIP

# Use all CPU cores on one machine
python EMCEE_generate_ensemble.py -m Tersoff+DRIP --parallel

# Cap worker count
python EMCEE_generate_ensemble.py -m Tersoff+DRIP --parallel --n-workers 8

# Multi-node (requires schwimmbad)
mpiexec -n 8 python EMCEE_generate_ensemble.py -m Tersoff+DRIP --mpi

# TETB+POD (M/W drive TB/POD basis tags; see script help for --tb-M, --pod-W, etc.)
python EMCEE_generate_ensemble.py -m TETB_POD -B 0.0001 -M 10 -W 6
```

The script docstring and `--help` describe the cost function, temperature
scaling, and thread limits (`OMP_NUM_THREADS=1` per worker to avoid oversubscription).

### `SubSamp_generate_ensemble.py` — **subsample refit** ensembles

**Typical workflow**

1. Pick `model_name` and descriptor knobs (`-M`, `-W`, optional TETB/POD
   overrides) consistent with `get_MCMC_inputs`.
2. Set **`-p` / `--p_subset`**: fraction of training configs per refit (e.g. `0.5`).
3. Set **`-n` / `--nsamples`**: number of refit draws (e.g. `30`–`100`).

**Examples**

```bash
cd uncertainty_quantification

# Default model (see script: often ACSF_hoppings), 50% of training per refit, 30 members
python SubSamp_generate_ensemble.py

# POD_energy-style tag with explicit M/W; 200 refits on 40% of training points
python SubSamp_generate_ensemble.py -m POD_energy -M 15 -W 6 -p 0.4 -n 200
```

**Output** is written to:

`ensembles/<model_name>/<model_name>_SubSamp_ensemble_p_<p>.pkl`

For `TETB_POD`, `model_name` is expanded to a tag like
`TETB_POD_tb_M_..._pod_M_...` so paths line up with best-fit caches and EMCEE outputs.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'lammps'`**
Build LAMMPS with `-DWITH_PYTHON=yes` and run `make install-python`.

**`atom attribute molecule`**
Rebuild LAMMPS with `-DPKG_MOLECULE=yes` for KC/DRIP potentials.

**`Could not find pair_pod`**
Rebuild LAMMPS with `-DPKG_ML-POD=yes`.
