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
| Allegro (Python) | `AllegroCalculator` |

Allegro uses the [NequIP/Allegro](https://github.com/mir-group/allegro) Python stack
(checkpoint load + flat `get_parameters` / `set_parameters`). Install with
`pip install -e ".[allegro]"` or `pip install nequip-allegro`. Train via
`uncertainty_quantification/initial_allegro_tests/fit_allegro.py`.

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

> **CLI hyperparameters:** every command below accepts generic model
> hyperparameter flags — `--KEY VALUE`, `--set KEY=VALUE`, or
> `--hyperparams KEY=VALUE ...` — forwarded to `get_MCMC_inputs` /
> `load_data_for_model`. E.g. `--two_body_radial 2` (POD), `--acsf_r_cut 7.0`
> (ACSF), `--allegro_bound_scale 50` (Allegro). See
> [`docs/add_blg_model.md`](docs/add_blg_model.md#cli-hyperparameter-specification-required-for-every-model).

### 0. Fit and cache best-fit parameters (no sampling)

```bash
cd uncertainty_quantification
python fit_model.py --models POD_energy --two_body_radial 2 --three_body_angular 4
python fit_model.py --models ACSF_hoppings -M 10 -W 6 --acsf_r_cut 7.0
```

### 1. Generate MCMC ensemble

```bash
cd uncertainty_quantification
python run_MCMC.py -m ACSF_hoppings -M 10 -W 6 -B 0.01
python run_MCMC.py -m POD_energy --POD-index 9 -B 0.001 --set regularization=1e-10
python run_MCMC.py -m Allegro_energy -B 0.001
python run_MCMC.py -m Allegro_energy --allegro-checkpoint initial_allegro_tests/allegro_blg_output/best-v2.ckpt
```

### 2. Generate subsample ensemble

```bash
python run_SubSamp.py -m ACSF_hoppings_M_10_W_6 -p 0.5 -n 30
```

### 3. Propagate uncertainty

```bash
python run_uq_propagation_relaxation.py --models POD_energy_POD_index_9_*
python run_uq_propagation_relaxation.py --models 'Allegro_energy_ckpt_*' --relax-backend ase
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
- `EXTRA-PAIR` for D3 correction

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
