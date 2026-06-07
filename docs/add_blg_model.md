# Adding a model to BLG_model_builder

Follow this checklist when integrating a new energy/force or tight-binding model.

## 1. Implement the model class

### Tight-binding (hopping)

1. Add a subclass of `TBModelBase` in [`src/tb_models.py`](../src/tb_models.py):
   - `evaluate(descriptors, params)` — core physics
   - `set_parameters` / `get_parameters` — inherited from base
   - `cache_basename()` — best-fit NPZ name under `best_fit_params/`
   - Optional `descriptors(atoms)` delegating to `tb_descriptors`

2. Register a `ModelSpec` in `_register_tb_models()` at the bottom of `tb_models.py`:
   - `match`, `parse_name`, `make_hyperparams`, `cache_basename`, `load_data_name`, `tb_factory`

### Energy / force (LAMMPS)

1. Add a subclass of `LammpsCalculatorBase` in [`src/lammps_interface.py`](../src/lammps_interface.py):
   - `_atom_style`, `_write_potential_files`, `_pair_style_commands`
   - `set_parameters` / `get_parameters`

## 2. Data loading

Add a branch in [`src/DataLoader.py`](../src/DataLoader.py) `load_data_for_model()` or route through `model_registry.load_data_model_name()`.

## 3. Fitting

Add or reuse a fit function in [`src/model_fit.py`](../src/model_fit.py):

| Model type | Fit function |
|------------|--------------|
| Linear ACSF | `fit_acsf_linear_hopping` |
| Generic scipy | `fit_model` |
| POD | `fit_pod` |
| TETB+POD | `fit_tetb_residual_pod` |
| Allegro | train via `initial_allegro_tests/fit_allegro.py`; load weights with `load_allegro_parameters` |

Wire the fit in [`src/get_MCMC_inputs.py`](../src/get_MCMC_inputs.py) (or extend registry-driven setup).

## 4. MCMC / subsample ensembles

- EMCEE: `python run_MCMC.py -m <model_name> ...`
- SubSamp: `python run_SubSamp.py -m <model_name> ...`

Ensure CLI name expansion in `EMCEE_generate_ensemble.main()` handles hyperparameter tags (`_M_`, `_W_`, `POD_index`, etc.).

## 5. UQ propagation

| Model kind | Script |
|------------|--------|
| LAMMPS energy | `run_uq_propagation_relaxation.py`, `run_uq_propagation_elasticity.py` |
| Python energy (Allegro) | `run_uq_propagation_relaxation.py` (`--relax-backend ase`) |
| TB hopping | `run_uq_propagation_bands.py` (`--tb-model`) |

Register LAMMPS models in `uq_model_runtime.UQ_LAMMPS_MODELS`; Python models in `UQ_PYTHON_MODELS`.

## 6. Smoke test

```bash
cd uncertainty_quantification
python smoke_test_models.py
```

Add your model to `TB_MODELS` or `ENERGY_MODELS` in that script.

## 7. Documentation

Update [`README.md`](../README.md) model catalog.

## Hyperparameter schema

Use `blg_model_builder.model_registry.make_hyperparams(model_name, **overrides)` for consistent dicts:

```python
from blg_model_builder.model_registry import make_hyperparams
hp = make_hyperparams("ACSF_hoppings_M_10_W_6", r_cut=6.0)
```
