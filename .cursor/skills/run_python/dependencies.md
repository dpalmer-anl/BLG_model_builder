# BLG_model_builder — conda dependencies

## Package source of truth

Core deps from `pyproject.toml`: `numpy>=1.22`, `ase>=3.22`.

Install the repo editable in any working env:

```bash
conda run -n blg_uq pip install -e .
```

## Dependency groups by task

| Task | Extra imports beyond base |
|------|---------------------------|
| Base / most UQ scripts | `emcee`, `matplotlib`, `pandas`, `scipy`, `tqdm` |
| TB descriptors / hopping data | `h5py` |
| pytest | `pytest` |
| LAMMPS calculators | `lammps` (requires LAMMPS built with `make install-python`) |
| Allegro | `torch`, `nequip`, `allegro` — use `allegro_env` or install `[allegro]` extra |
| MCMC parallel pools | `schwimmbad` (optional) |

## Import-check snippets

Copy and extend as needed:

```bash
# Base UQ
conda run -n blg_uq python -c "import numpy, ase, emcee, matplotlib, pandas, scipy, tqdm, blg_model_builder"

# Tests
conda run -n blg_uq python -c "import pytest, blg_model_builder"

# LAMMPS potentials
conda run -n blg_uq python -c "from lammps import lammps; import blg_model_builder"

# Allegro
conda run -n allegro_env python -c "import torch, nequip, allegro, blg_model_builder"
```

## Create-env one-liners

Paste from repo root. Adjust Python version if needed.

**Standard UQ + MCMC (`blg_uq`):**

```bash
conda create -n blg_uq python=3.11 numpy ase scipy matplotlib pandas h5py emcee tqdm pytest -y && conda run -n blg_uq pip install -e .
```

**Tests only (minimal):**

```bash
conda create -n blg_uq python=3.11 numpy ase pytest -y && conda run -n blg_uq pip install -e ".[test]"
```

**Allegro workflows (`allegro_env`):**

```bash
conda create -n allegro_env python=3.11 -y && conda run -n allegro_env pip install -e ".[allegro]"
```

**LAMMPS note:** `lammps` is not a conda-forge drop-in for this project — build LAMMPS with `PKG_INTERLAYER`, `PKG_ML-POD`, `PKG_MOLECULE`, then `make install-python`, and install into the active env. See `README.md` and `src/lammps_interface.py`.
