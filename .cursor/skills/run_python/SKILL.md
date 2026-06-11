---
name: run-python
description: >-
  Run Python, pip, and pytest only inside conda environments. Defaults to the
  blg_uq env for BLG_model_builder. Verifies dependencies before running,
  scans other conda envs if needed, and provides a create-env command when
  none work. Use when executing Python scripts, tests, pip installs, or any
  python/pytest/pip shell command in this project.
---

# Run Python (conda)

Never invoke bare `python`, `pip`, or `pytest`. Always run through conda.

## Default environment

**`blg_uq`** — used by `tests/conftest.py`, `uncertainty_quantification/submit_uq.py`, and project docs.

## Command pattern

Prefer `conda run` (no shell activation needed):

```bash
conda run -n blg_uq python script.py [args]
conda run -n blg_uq pip install -e .
conda run -n blg_uq pytest -v
```

Working directory matters for UQ scripts — run from `uncertainty_quantification/` when the script expects it (see README).

### Windows (conda not in PATH)

Locate conda, then use the same `conda run` pattern:

```powershell
# Common locations — use whichever exists
$conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
if (-not (Test-Path $conda)) { $conda = "$env:USERPROFILE\anaconda3\Scripts\conda.exe" }
& $conda run -n blg_uq python script.py
```

If `conda` is missing entirely, stop and give the user the create-env one-liner from [dependencies.md](dependencies.md).

## Workflow (follow in order)

```
Task Progress:
- [ ] 1. Infer required imports for the command
- [ ] 2. Verify blg_uq has them
- [ ] 3. If not, scan other conda envs
- [ ] 4. If none work, ask user or provide create command
- [ ] 5. Run via conda run -n <env>
```

### Step 1 — Infer dependencies

From the target script's imports and task context:

| Always (this repo) | `numpy`, `ase`, `blg_model_builder` |
| UQ / MCMC | `emcee`, `matplotlib`, `pandas`, `scipy`, `tqdm` |
| TB descriptors | `h5py`, `scipy`, `pandas` |
| Tests | `pytest` |
| LAMMPS potentials | `lammps` (LAMMPS Python module) |
| Allegro | `torch`, `nequip`, `allegro` |

See [dependencies.md](dependencies.md) for the full map and create-env commands.

### Step 2 — Verify `blg_uq`

Build an import check from step 1, then run:

```bash
conda run -n blg_uq python -c "import numpy, ase, blg_model_builder"
```

Add other modules as needed, e.g.:

```bash
conda run -n blg_uq python -c "import emcee, matplotlib, pandas, scipy, tqdm, blg_model_builder"
```

Also confirm the package is installed when running project code:

```bash
conda run -n blg_uq python -c "import blg_model_builder; print(blg_model_builder.__file__)"
```

If imports succeed, use `blg_uq` and proceed to step 5.

### Step 3 — Scan other envs

```bash
conda env list
```

For each env (skip `base` unless nothing else exists), run the same import check from step 2. Use the first env that passes.

Known alternate env in this repo: **`allegro_env`** (NequIP/Allegro training only).

### Step 4 — No working env

Do **one** of the following (prefer asking if unsure which extras are needed):

1. **Ask the user** which conda env to use.
2. **Give a paste-ready one-liner** from [dependencies.md](dependencies.md) matched to the task (base UQ vs tests vs Allegro).

Do not fall back to system Python.

### Step 5 — Execute

```bash
conda run -n <env> python <script> [args]
```

For editable install after env creation:

```bash
conda run -n blg_uq pip install -e .
```

## Examples

**Smoke test (from `uncertainty_quantification/`):**

```bash
conda run -n blg_uq python smoke_test_models.py
```

**pytest (from repo root, after `pip install -e .`):**

```bash
conda run -n blg_uq pytest -v tests/
```

**MCMC:**

```bash
cd uncertainty_quantification
conda run -n blg_uq python run_MCMC.py -m ACSF_hoppings -M 10 -W 6 -B 0.01
```

## Additional resources

- Dependency groups and env-creation one-liners: [dependencies.md](dependencies.md)
