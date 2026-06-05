# BLG_model_builder

Utilities for bilayer-graphene interatomic models, tight-binding hoppings,
descriptor construction, data loading, and Bayesian uncertainty quantification.
Interatomic potentials are evaluated through the **LAMMPS Python module**
(`from lammps import lammps`).

Install as an editable package:

```bash
pip install -e .
```

Import as `blg_model_builder` (maps to the `src/` directory).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Installable package `blg_model_builder` (potentials, TB models, UQ machinery) |
| `uncertainty_quantification/` | CLI entry points (`submit_uq.py`, `run_uq_propagation_*.py`, `run_MCMC.py`) |
| `uncertainty_quantification/visualizations/` | Plotting scripts |
| `uncertainty_quantification/pod_hyperparam_search/` | POD hyperparameter search tools and results |
| `data/` | Training / test structures and hopping HDF5 |
| `tests/` | pytest suite |
| `docs/add_blg_model.md` | Checklist for adding new models |

---

## Models implemented

### Energy / force (LAMMPS calculators)

| Model | Class |
|-------|-------|
| Tersoff (intralayer) | `TersoffLammpsCalculator` |
| Kolmogorov–Crespi full | `KolmogorovCrespiLammpsCalculator` |
| DRIP | `DRIPLammpsCalculator` |
| Tersoff + KC | `TersoffKCLammpsCalculator` |
| Tersoff + DRIP | `TersoffDRIPLammpsCalculator` |
| POD energy | `PODLammpsCalculator` |
| TETB + POD (hybrid) | `TETB_PODLammpsCalculator` |

Backward-compatible `*ASECalculator` aliases are re-exported from `blg_model_builder.potentials`.

### Tight-binding hopping models

All hopping models use class-based `TBModelBase` implementations in `blg_model_builder.tb_models`:

| Model | Class |
|-------|-------|
| Moon–Koshino | `MKHoppingModel` |
| ACSF linear | `ACSFHoppingModel` |
| ACSF + SK descriptors | `ACSFHoppingSKModel` |
| LETB interlayer | `LETBInterlayerModel` |
| LETB intralayer (NN/NNN/NNNN) | `LETBIntralayerModel` |

Model metadata and hyperparameter schemas are registered in `blg_model_builder.model_registry`.

---

## UQ workflow

### 1. Generate MCMC ensemble

```bash
cd uncertainty_quantification
python run_MCMC.py -m ACSF_hoppings -M 10 -W 6 -B 0.01
python run_MCMC.py -m POD_energy --POD-index 9 -B 0.001
```

### 2. Generate subsample ensemble

```bash
python run_SubSamp.py -m ACSF_hoppings_M_10_W_6 -p 0.5 -n 30
```

### 3. Propagate uncertainty

```bash
python run_uq_propagation_relaxation.py --models POD_energy_POD_index_9_*
python run_uq_propagation_elasticity.py --models Tersoff+DRIP --n-samples 10
python run_uq_propagation_bands.py --models POD_energy_* --tb-model acsf_hoppings_M_10_W_6
```

### 4. HPC submission

`submit_uq.py` chains relaxation and band jobs (uses `run_MCMC.py` for MCMC).

---

## Adding a new model

See [docs/add_blg_model.md](docs/add_blg_model.md) for the full checklist (class → registry → DataLoader → model_fit → get_MCMC_inputs → ensembles → propagation).

Quick smoke test after adding a model:

```bash
cd uncertainty_quantification
python smoke_test_models.py
```

---

## LAMMPS build requirements

- `PKG_INTERLAYER` — KC full, DRIP
- `PKG_ML-POD` — POD potentials
- `PKG_MOLECULE` — layer tags for interlayer models

---

## Package modules (selected)

| Module | Role |
|--------|------|
| `blg_model_builder.lammps_interface` | LAMMPS calculator classes |
| `blg_model_builder.tb_models` | TB hopping model classes |
| `blg_model_builder.tb_descriptors` | ACSF / LETB descriptors |
| `blg_model_builder.DataLoader` | Training data loading |
| `blg_model_builder.get_MCMC_inputs` | MCMC data + best-fit setup |
| `blg_model_builder.model_fit` | Fitting routines |
| `blg_model_builder.EMCEE_generate_ensemble` | MCMC ensemble generation |
| `blg_model_builder.SubSamp_generate_ensemble` | Bootstrap ensembles |
| `blg_model_builder.ensemble_io` | Ensemble pickle I/O utilities |
| `blg_model_builder.model_registry` | Model registration and hyperparameter schema |
