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

## 4. Fit / MCMC / subsample ensembles

- Fit only (cache best-fit params): `python fit_model.py --models <model_name> ...`
- EMCEE: `python run_MCMC.py -m <model_name> ...`
- SubSamp: `python run_SubSamp.py -m <model_name> ...`

Ensure CLI name expansion in `EMCEE_generate_ensemble.main()` (and
`fit_cli._expand_model_name`) handles hyperparameter tags (`_M_`, `_W_`,
`POD_index`, etc.). All of these accept generic hyperparameter flags — see
[CLI hyperparameter specification](#cli-hyperparameter-specification-required-for-every-model).

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

## CLI hyperparameter specification (REQUIRED for every model)

Every workflow entry point accepts **generic hyperparameter flags** so any
model-specific knob can be set from the command line without adding a dedicated
argument. This is implemented once in
[`src/cli_hyperparams.py`](../src/cli_hyperparams.py) and wired into:

- `fit_model.py` (fit + cache best-fit params, no sampling)
- `run_MCMC.py` / `EMCEE_generate_ensemble.py`
- `run_SubSamp.py` / `SubSamp_generate_ensemble.py`
- `run_uq_propagation_relaxation.py`, `run_uq_propagation_elasticity.py`, `run_uq_propagation_bands.py`

### Three equivalent syntaxes

```bash
# 1. bare --KEY VALUE  (dashes and underscores are interchangeable)
python fit_model.py --models POD_energy --two_body_radial 2 --three_body_angular 4

# 2. --set KEY=VALUE  (repeatable)
python run_MCMC.py -m POD_energy -B 0.001 --set regularization=1e-10 --set weight_force=1.0

# 3. --hyperparams KEY=VALUE KEY=VALUE ...
python run_SubSamp.py -m ACSF_hoppings -M 10 -W 6 --hyperparams acsf_r_cut=7.0 use_envelope=false
```

Values are auto-coerced (`int` / `float` / `bool` / `none` / JSON / `str`).
Everything collected is forwarded as `**kwargs` into `get_MCMC_inputs` →
`load_data_for_model`, overriding the explicit flags (`-M`, `-W`, …).

### Per-model examples

| Model | Tunable keys (read from `data_kw`) | Example |
|-------|------------------------------------|---------|
| `ACSF_hoppings` / `_sk` | `acsf_M`/`M`, `acsf_W`/`W`, `acsf_r_cut`/`r_cut`, `acsf_use_envelope`/`use_envelope` | `--acsf_r_cut 7.0 --use_envelope false` |
| `POD_energy` | `two_body_radial`, `three_body_radial`, `three_body_angular`, `four_body_radial`, `four_body_angular`, `five_body_*`, `six_body_*`, `seven_body_*`, `regularization`, `weight_energy`, `weight_force`, `rcut`/`pod_cutoff`, `include_intralayer` | `--two_body_radial 2 --regularization 1e-10` |
| `TETB_POD` | `tb_M`, `tb_W`, `pod_M`, `pod_W`, `pod_cutoff`, plus POD keys above | `--tb_M 10 --pod_M 12` |
| `Allegro_energy` | `allegro_checkpoint`, `allegro_r_max`, `allegro_device`, `allegro_bound_scale` | `--allegro_bound_scale 50 --allegro_device cuda` |

### Requirement when adding a NEW model

To make a new model controllable from the CLI **for free**, read each
hyperparameter from the `data_kw` / `kwargs` dict in
`DataLoader.load_data_for_model()` and `get_MCMC_inputs()` (use
`data_kw.get("my_key", <default>)`), exactly like the POD / ACSF / Allegro
branches do. Do **not** read hyperparameters from hard-coded constants or add a
bespoke `argparse` flag in each CLI — the generic mechanism already plumbs the
value through. Document the new keys in the table above.

> Note: the generic flags change the *values* used for fitting/evaluation but do
> not, by themselves, change the ensemble-folder / cache name. Folder tags still
> come from `-M`/`-W`/`--POD-index`/checkpoint hash (see
> `EMCEE_generate_ensemble.main()` and `fit_cli._expand_model_name`). Cache files
> that encode descriptor sizes (e.g. POD `ncoeffs`) stay distinct automatically;
> if you introduce a new size-changing key, fold it into the cache basename.
