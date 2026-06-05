# BLG Model Builder v2 — Project Progress & Agent Handoff Document

> **Purpose:** This document is written specifically for an LLM agent to understand the project, avoid rediscovering known pitfalls, and continue development efficiently. Read this entire file before touching any code.

---

## 1. Project Overview

`BLG_model_builder_v2` is a Python package for **uncertainty quantification (UQ) of bilayer graphene (BLG) models**. It:

- Wraps a suite of interatomic potentials via LAMMPS for fast energy/force evaluation
- Provides ASE-compatible calculators for geometry relaxation and property evaluation
- Runs MCMC (via `emcee`) to generate posterior distributions over model parameters
- Propagates uncertainty through physical observables: geometry relaxations, mechanical properties (Poisson ratio, GSFE), and electronic band structures of twisted BLG (TBLG)

**System:** The code runs inside **WSL (Windows Subsystem for Linux)** on the user's Windows machine. All build steps and Python execution must use `wsl -e bash -c "..."`.

---

## 2. Repository Layout

```
BLG_model_builder_v2/
├── clean_install.sh                   # Full clean build script (pip install -e .)
├── pyproject.toml / setup.py          # pybind11 build configuration
├── src/
│   ├── potential_wrapper.cpp          # C++ pybind11 LAMMPS wrapper (THE key file)
│   ├── potentials.py                  # Python ASE calculator interface
│   ├── DataLoader.py                  # Training data loading and descriptor precomputation
│   ├── tb_models.py                   # Moon-Koshino, LETB tight-binding models
│   ├── tb_descriptors.py              # Descriptor computation for TB models
│   ├── geom_tools.py                  # Geometry utilities
│   ├── POD_TB_tight_binding_wrapper.py  # Python wrapper for C++ TB hopping
│   ├── POD_TB_tight_binding.cpp/h     # C++ tight-binding hopping computation
│   ├── POD_TB_bindings.cpp            # pybind11 bindings for TB
│   └── bindings.cpp                   # Additional pybind11 bindings
├── uncertainty_quantification/
│   ├── EMCEE_generate_ensemble.py     # MCMC ensemble generation (main UQ entry point)
│   ├── get_MCMC_inputs.py             # Load data, fit best-params, return calc/data
│   ├── model_fit.py                   # fit_pod (LAMMPS fitpod), fit_model (scipy), fit_torch
│   ├── run_uq_propagation.py          # Post-MCMC UQ propagation
│   ├── write_potential_files.py       # Write LAMMPS potential files from params
│   ├── submit_uq.py                   # HPC job submission helper
│   ├── plot_*.py                      # Plotting scripts
│   └── test_relaxation.py             # A second relaxation test (in uq/ folder)
├── tests/
│   ├── test_relaxation.py             # Primary relaxation test (2.88° TBLG)
│   ├── test_pod_finite_diff.py        # Force consistency verification (VERIFIED PASSING)
│   ├── test_calculators.py            # Calculator tests
│   ├── test_interface.py              # Interface tests
│   ├── test_tetb.py                   # TETB model tests
│   └── C_coefficients.pod             # Ground-truth POD coefficients from fitpod
└── patches_lammps/                    # LAMMPS source patches (KC Full fix)
```

**Compiled module name:** `blg_model_builder_v2.potential_ext` (the C++ extension). Exposes: `TersoffCalculator`, `KolmogorovCrespiCalculator`, `DRIPCalculator`, `PODCalculator`, `PODCoulCalculator`.

---

## 3. Models Supported

### 3.1 Interatomic Potentials (via LAMMPS)

| Calculator | File in potential_wrapper.cpp | Purpose |
|---|---|---|
| `TersoffASECalculator` | `TersoffCalculator` | In-plane C-C bonds |
| `KolmogorovCrespiASECalculator` | `KolmogorovCrespiCalculator` | Interlayer (KC Full) |
| `DRIPASECalculator` | `DRIPCalculator` | Interlayer (DRIP) |
| `PODASECalculator` | `PODCalculator` | Full ML potential (in+interlayer) |
| `TETBASECalculator` | `PODCoulCalculator` | TB + POD residual + Coulomb |

**POD potential** uses LAMMPS's `pair_style pod` which wraps the EAPOD (Environment Adaptive Proper Orthogonal Decomposition) library. Fitted with LAMMPS's built-in `fitpod` command.

### 3.2 Tight-Binding Models (Python)

| Model | Location | Description |
|---|---|---|
| Moon-Koshino (MK) | `tb_models.py: mk_hopping` | Parametric 3-param SK model |
| LETB | `tb_models.py: letb_intralayer` | Distance-dependent TB |
| POD-TB | `POD_TB_tight_binding_wrapper.py` | Linear POD-like TB descriptors for hoppings |

---

## 4. Build System

### 4.1 LAMMPS Requirements

LAMMPS must be compiled as a **shared library** with the `ML-POD` package enabled:

```bash
cd $LAMMPS_ROOT/build
cmake ../cmake -DBUILD_SHARED_LIBS=on -DPKG_ML-POD=yes -DPKG_MOLECULE=yes ...
cmake --build . --parallel
```

The build script auto-detects LAMMPS at `../lammps` relative to the repo:

```bash
cd /mnt/c/Users/Daniel/Documents/research/BLG_model_builder_v2
./clean_install.sh
```

### 4.2 Critical LAMMPS Source Patch (MUST BE APPLIED)

**File:** `$LAMMPS_ROOT/src/ML-POD/pair_pod.cpp`

In `PairPOD::coeff()`, after destroying temp memory, reset `nijmax`:

```cpp
memory->destroy(fastpodptr->tmpmem);
memory->destroy(fastpodptr->tmpint);
nijmax = 0;    // temp memory was freed; force re-allocation on next compute()
```

**Without this patch:** `pair_coeff` frees temp memory but leaves stale `nijmax`. Next `compute()` skips re-allocation and dereferences freed pointers → **segmentation fault at address ~0x1e0** in `PairPOD::lammpsNeighborList → ti1[nij]`.

### 4.3 Rebuild Workflow After LAMMPS Changes

```bash
cd /mnt/c/Users/Daniel/Documents/research/lammps/build
make -j$(nproc)  # or: cmake --build . --parallel
cd /mnt/c/Users/Daniel/Documents/research/BLG_model_builder_v2
./clean_install.sh
```

---

## 5. C++ Architecture (`potential_wrapper.cpp`)

### 5.1 Design Principles

1. **No subclassing or manual replacement of `lmp->force->pair`.** Replacing the pair pointer after `pair_style` is called leaves dangling pointers in LAMMPS's neighbor-list system → segfault. Always use LAMMPS commands.

2. **File-based `pair_coeff + run 0`** for every coefficient update. Using EAPOD's `inject_coefficients()` / `mknewcoeff()` only partially updates internal state, leaving stale precomputed tables → wrong forces.

3. **`atom_modify map array`** (NOT `atom_modify map yes`). With `map yes`, LAMMPS defaults to a hash map for large systems. `PairPOD::compute()` uses `atom->map_array` directly which is null for hash maps → segfault.

4. **One `clear` + full rebuild per geometry change** (via `setup_pod_lammps`). This is slow but correct. A future optimization would be to move atoms directly rather than rebuilding.

### 5.2 PODCalculator Command Flow

For each `calculate(atoms)` call:

```
_geometry_changed() → True
  → _setup(atoms):
      ase_to_lammps(atoms) → pos_lammps, types, lammps_cell
      set_geometry(pos, types, box, pod_content, coeff_content, elements):
        write pod_content → /tmp/*.pod_desc
        write coeff_content → /tmp/*.pod_coeff  (file format: "model_coefficients: N 0 0\n v1\n v2\n ...")
        setup_pod_lammps():
          lmp_cmd("clear")
          lmp_cmd("units metal"), ("atom_style atomic"), ("atom_modify map array"), ...
          make_periodic_box()          → "region box prism/block ..." + "create_box"
          create_atoms single (×N)     → one lmp_cmd per atom — SLOW for N>1000
          set masses
          lmp_cmd("pair_style pod")
          lmp_cmd("pair_coeff * * pod_file coeff_file C")  → PairPOD::coeff() → nijmax=0
          lmp_cmd("run 0")             → builds neighbor list, allocates tmpmem, computes E/F
          lmp_cmd("compute pod_glob ...") → for descriptor extraction only
  → compute(coeffs):
      write same coeffs to coeff_file (redundant but correct)
      pair_coeff + run 0               → nijmax reset to 0 by coeff(), recomputed by run 0
      energy = lmp->force->pair->eng_vdwl
      forces = {lmp->atom->f[map(i+1)][α] for i in range(N)}
  → lammps_to_ase_forces(forces_lammps, atoms)  → rotate back to ASE frame
```

### 5.3 Energy Extraction

`lmp->force->pair->eng_vdwl` is the total potential energy. In `PairPOD::compute()` (blockMode=0, hard-coded):

```cpp
evdwl = fastpodptr->peratomenergyforce2(fij1, rij1, tmp, ti1, tj1, nij);
ev_tally_full(i, 2.0*evdwl, 0.0, ...);
// → eng_vdwl += 0.5 * 2.0 * evdwl = evdwl  (per atom, summed over all atoms)
```

Energy is computed only if `eflag_global = 1`. For `run 0`, LAMMPS always outputs thermo at step 0 (default behavior), so `eflag` IS set and energy IS computed.

**NOTE on blockMode:** `PairPOD::compute()` has two code paths (`blockMode=0` and `blockMode=1`). Only `blockMode=0` is active (hard-coded). The `grow_pairs()` function and `rij`/`fij` arrays at lines ~663+ are **DEAD CODE** that is never executed. The `nijmax` discussion involving `grow_pairs` in comments or older analysis is irrelevant.

### 5.4 Coordinate Transformation (ASE ↔ LAMMPS)

The transformation is a **pure rotation** (QR decomposition of the cell matrix):

- **Forward (ASE → LAMMPS):**
  - `frac = atoms.get_scaled_positions()` → fractional coords
  - `pos_lammps = frac @ lammps_cell` → LAMMPS Cartesian (x along A, y in AB-plane)
  - `lammps_cell = [[ax,0,0],[bx,by,0],[cx,cy,cz]]` computed via Gram-Schmidt

- **Force rotation (LAMMPS → ASE):**
  - `f_ase = f_lammps @ R` where `R = np.vstack([ex, ey, ez])`
  - `ex = A/|A|`, `ey = normalized(B - (B·ex)ex)`, `ez = ex × ey`
  - This is EXACT for any cell (no approximation)

### 5.5 Neighbor Periodicity

The moiré supercell for 2.88° TBLG has a 20 Å vacuum gap in z. With `pbc 1 1 1` in the POD file and a LAMMPS box of ~49×42×43 Å, the vacuum ensures no z-periodic images are within the 5 Å cutoff. This is correct for a 2D-like system.

---

## 6. Python Interface (`potentials.py`)

### 6.1 PODASECalculator

**Does NOT inherit from** `ase.calculators.calculator.Calculator`. It is a custom class with the right method signatures:

- `calculate(atoms)` → `{"energy": float, "forces": np.ndarray(N,3)}`
- `get_potential_energy(atoms)` → calls `calculate(atoms)`
- `get_forces(atoms)` → calls `calculate(atoms)`

**Geometry caching:** Uses `_geometry_changed(atoms, last_pos, last_cell)` with `np.allclose(atol=1e-10)` to detect when LAMMPS needs to be rebuilt.

**Important:** `_geometry_changed` stores positions in ASE Cartesian frame (`atoms.get_positions()`), not LAMMPS frame.

### 6.2 `ase_to_lammps(atoms, element_order)`

Returns `(pos_lammps, types, lammps_cell, element_order)`. The `lammps_cell` is the **row-major** LAMMPS cell matrix:
```python
lammps_cell = [[ax, 0,  0 ],
               [bx, by, 0 ],
               [cx, cy, cz]]
```

### 6.3 `ncoeff_from_params(hyperparams)`

Python port of EAPOD's `crossindices` count function. This MUST match the C++ EAPOD coefficient count. The coefficient file `C_coefficients.pod` is the ground truth — always verify with an `assert len(params) == ncoeff_from_params(hyperparams)` check.

### 6.4 Energy Scale

The POD potential is trained on **absolute DFT total energies** (not cohesive or formation energies). For graphene:
- Onebody coefficient (first in file): ≈ −169.8 eV per atom
- Total energy scale: ≈ −254 eV/atom for a relaxed bilayer graphene structure
- The 1588-atom moiré cell (2.88°) gives ≈ −403,700 eV total

This is **not a bug**. Forces are the gradient of this energy surface and are physically correct. Verified by finite-difference test: max error 3.4×10⁻⁴ eV/Å at δ=10⁻³ Å.

---

## 7. POD Potential Fitting

Fitting uses LAMMPS's built-in `fitpod` command via `model_fit.py:fit_pod()`:

1. Creates `TrainingData/` and `TestData/` directories with XYZ files
2. Writes `C_param.pod` (hyperparameter file) and `C_data.pod` (fitting config)
3. Runs `lammps -in fit.pod` which calls `fitpod`
4. Reads `C_coefficients.pod` (output: `model_coefficients: N 0 0\n v1\n v2\n ...`)

**CRITICAL:** The file `tests/C_coefficients.pod` is the **ground-truth coefficient file** produced by `fitpod` with exactly `tests/C_param.pod`. Never use coefficients from `uncertainty_quantification/best_fit_params/*.npz` for the relaxation test — those may be from a different training run with different hyperparameters.

The coefficient file format:
```
model_coefficients: 184 0 0
-169.806290972864
-20.261876996170
...
```
(184 = N coefficients, then one value per line)

---

## 8. Bugs Discovered and Fixed

### Bug 1: PairPOD nijmax not reset in coeff() → Segfault ✅ FIXED

**Symptom:** Signal: Segmentation fault (11), address 0x1e0, in `PairPOD::lammpsNeighborList`, accessing `ti1[nij]`.

**Root cause:** `PairPOD::coeff()` calls `memory->destroy(fastpodptr->tmpint)` (freeing the buffer) but does not reset `PairPOD::nijmax`. On the next `compute()` call: since `nijmax` (stale) > `jnum` (current neighbor count), the `if (nijmax < jnum)` check fails, `allocate_temp_memory` is skipped, and the code immediately dereferences the freed pointer via `&fastpodptr->tmpint[2*nijmax]` → segfault.

**Fix:** In `$LAMMPS_ROOT/src/ML-POD/pair_pod.cpp`, `PairPOD::coeff()`:
```cpp
memory->destroy(fastpodptr->tmpmem);
memory->destroy(fastpodptr->tmpint);
nijmax = 0;    // temp memory was freed; force re-allocation on next compute()
```

**After this fix:** LAMMPS must be recompiled and the Python package rebuilt.

---

### Bug 2: atom_modify map yes → Null pointer segfault ✅ FIXED

**Symptom:** Same segfault address 0x1e0, same stack trace.

**Root cause:** `atom_modify map yes` lets LAMMPS choose the map type. For large systems (typically >1000 atoms), LAMMPS defaults to a **hash map** (`map_style = MAP_HASH`). `PairPOD::compute()` accesses `atom->map_array` (the array-based map) which is **null** for hash maps → null pointer dereference.

**Fix:** Use `atom_modify map array` everywhere in `potential_wrapper.cpp`. Never use `atom_modify map yes`.

---

### Bug 3: PairPODDirect replace hack → Dangling pointers ✅ FIXED (removed)

**Symptom:** Same segfault, different internal state.

**Root cause:** Older code created a `PairPODDirect` subclass, then deleted `lmp->force->pair` and replaced it with a new `PairPODDirect` instance. LAMMPS's neighbor-list system keeps raw pointers to the pair style; deleting `lmp->force->pair` invalidates those pointers → dangling pointer dereference.

**Fix:** Removed `PairPODDirect` entirely. Use LAMMPS commands (`pair_style pod` + `pair_coeff`) to let LAMMPS manage the `PairPOD` instance internally.

---

### Bug 4: inject_coefficients/mknewcoeff → Wrong forces ✅ FIXED

**Symptom:** Simulation produced physically wrong results (forces inconsistent with expected behavior).

**Root cause:** EAPOD's `inject_coefficients()` and `mknewcoeff()` only partially update EAPOD's internal state (coefficient arrays), leaving stale precomputed tables. Subsequent `compute()` calls use updated coefficients but stale tables → wrong forces.

**Fix:** Always use file-based `pair_coeff + run 0`. Write coefficients to a `/tmp/*.pod_coeff` file, then issue `pair_coeff * * pod_file coeff_file elements` + `run 0`. This triggers `PairPOD::coeff()` which calls `new EAPOD(...)` from scratch.

---

### Bug 5: EMCEE Temperature as scalar when it needs to be a dict ✅ FIXED

**Symptom:** Runtime error or wrong MCMC behavior when multiple observable types used.

**Root cause:** `log_probability` was building a `Temperature` dict from `C0` keys, but `log_likelihood` expected a scalar `T`. Mismatch in interface.

**Fix:** `log_likelihood` now handles `T` as either a scalar or a dict:
```python
if isinstance(T, dict):
    T_energy = T.get(key, T.get("energy", 1.0))
    T_forces = T.get("forces", T_energy)
else:
    T_energy = T_forces = T
```

---

### Bug 6: LBFGS optimizer diverges for bilayer graphene ✅ DIAGNOSED, PARTIALLY FIXED

**Symptom:**
- Step 0: E = −403,700 eV, fmax = 0.167 eV/Å (correct, near equilibrium)
- Step 1: fmax jumps to 2.07 eV/Å (12× increase after first step)
- Step 9: E → +125,000 eV, fmax → 2.3×10⁶ eV/Å (catastrophic divergence)
- Step 10+: NaN/Inf energies, simulation completely broken

**Root cause (CONFIRMED by finite-difference test):** The forces are physically CORRECT (consistent with energy to 3.4×10⁻⁴ eV/Å). The issue is the optimizer.

ASE's LBFGS default `H0=70 Å/eV` corresponds to assuming a spring constant of 0.014 eV/Å². Graphene's in-plane C-C bonds have spring constants **~40 eV/Å²** (2800× stiffer). LBFGS tries to take steps of `H0 × force = 70 × 0.167 = 11.7 Å`, which maxstep clips to 0.2 Å. Even at 0.2 Å, the step **overshoots** the in-plane equilibrium (~0.04 Å displacement). After 2 oscillating steps, the LBFGS Hessian estimate goes invalid, and step 3 takes a large step that pushes atoms into a high-energy configuration (+520 eV), causing runaway divergence.

**Fix:** Use FIRE optimizer instead of LBFGS:
```python
from ase.optimize import FIRE
dyn = FIRE(atoms, maxstep=0.1, dt=0.1, dtmax=1.0, logfile='-')
dyn.run(fmax=1e-3, steps=500)
```

**Known issue:** As of 2026-03-27, `dyn.run()` hangs in `test_relaxation.py` despite individual FIRE steps working correctly in `test_relaxation_debug.py`. The `logfile='-'` parameter was added but did not fix the hang. This requires further investigation (see Section 10).

---

### Bug 7: Coefficient file loading in test_relaxation.py ✅ FIXED

**Symptom:** Wrong coefficients loaded for the POD potential test.

**Root cause:** Previous code loaded coefficients from `uncertainty_quantification/best_fit_params/*.npz`, which may be from a different training run with different hyperparameters.

**Fix:** Load directly from `tests/C_coefficients.pod` (the fitpod output):
```python
params = np.loadtxt(os.path.join(os.path.dirname(__file__), "C_coefficients.pod"), skiprows=1)
assert len(params) == ncoeffs, "mismatch between C_coefficients.pod and hyperparams"
```

---

## 9. What Works (Verified)

| Component | Status | Notes |
|---|---|---|
| `TersoffASECalculator` | ✅ Working | Standard Tersoff for in-plane C |
| `KolmogorovCrespiASECalculator` | ✅ Working | KC Full for interlayer |
| `DRIPASECalculator` | ✅ Working | DRIP for interlayer |
| `PODASECalculator` | ✅ Working | Forces verified by finite-diff test |
| `PODCoulCalculator` / `TETBASECalculator` | ✅ Compiles, untested at scale |  |
| `PODTorchCalculator` | ✅ Compiles, gradient flow works | No finite-diff test yet |
| LAMMPS neighbor list (map array) | ✅ Fixed | Must use `map array` |
| PairPOD nijmax segfault | ✅ Fixed | LAMMPS source patched |
| POD coefficient loading | ✅ Fixed | File-based pair_coeff |
| Force consistency | ✅ Verified | Max error 3.4e-4 eV/Å |
| EMCEE MCMC framework | ✅ Working | Temperature dict handled |
| `fit_pod` (LAMMPS fitpod) | ✅ Working | Produces C_coefficients.pod |
| `fit_torch` | ✅ Working | Adam + ReduceLROnPlateau |
| `get_MCMC_inputs` | ✅ Working | Loads data, fits best params |
| `ncoeff_from_params` | ✅ Verified | Matches EAPOD count |
| ASE↔LAMMPS coordinate transform | ✅ Correct (proven) | Pure rotation |

---

## 10. What Needs Fixing

### 10.1 FIRE Optimizer Hangs in test_relaxation.py (HIGH PRIORITY)

**Symptom:** `dyn.run(fmax=1e-3, steps=500)` hangs indefinitely with zero output, even with `logfile='-'`.

**Diagnostic findings:**
- `test_relaxation_debug.py` shows individual force computation takes ~0.35s ✅
- Energy computation takes ~0.15s ✅  
- `dyn.step(f)` is instantaneous ✅
- `dyn.run()` hangs completely ✗

**Suspected causes:**
1. ASE version compatibility (warning seen: `"Please do not pass forces to step(). This argument will be removed in ase 3.28.0."`)
2. `dyn.run()` may call `atoms.get_potential_energy()` → `calc.calculate(None)` → `ValueError: atoms must be provided` (but this should raise an exception, not hang)
3. The hang may be at the first `atoms.get_forces()` call inside `dyn.run()` if ASE's internal caching mechanism blocks

**Next steps:**
1. Check ASE version: `python3 -c "import ase; print(ase.__version__)"`
2. Try: replace custom calculator with proper `ase.calculators.calculator.Calculator` subclass
3. Try: `LBFGS(atoms, H0=0.025, maxstep=0.02)` (correct H0 for graphene in-plane stiffness)
4. Try: run a LAMMPS `minimize` directly to verify potential correctness (see Section 11)

### 10.2 TRUST LAMMPS — Verify Python Relaxation Against LAMMPS minimize

Per the project requirement: **always compare Python-side relaxations against LAMMPS's built-in minimizer**.

To do this:
1. Write the TBLG structure to an XYZ file
2. Write a LAMMPS input script that loads the structure, sets up POD, and runs `minimize 1e-6 1e-8 1000 10000`
3. Compare the final energy and geometry to the Python-side result

If LAMMPS minimize converges but Python FIRE diverges → bug in Python calculator or optimizer interface.
If both diverge → the POD potential is not appropriate for full structural relaxation of this geometry (trained on too-narrow distribution, or only for interlayer modes).

**Template LAMMPS minimize script:**
```lammps
units metal
atom_style atomic
atom_modify map array
boundary p p p
newton on
read_data tblg.lammps   # or read_dump / create_atoms
pair_style pod
pair_coeff * * C_param.pod C_coefficients.pod C
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
minimize 1e-6 1e-8 1000 10000
write_dump all custom final.dump id type x y z fx fy fz
```

### 10.3 PODASECalculator Performance (MEDIUM PRIORITY)

The current implementation issues one `create_atoms single` command per atom. For 1588 atoms, this is 1588 LAMMPS input-parser calls per geometry step. For an optimization with 500 steps, that's ~800,000 parser calls total.

**Better approach:** Keep the LAMMPS instance alive across geometry changes and use `set atom i x y z` commands or directly manipulate `lmp->atom->x[i]` via C++ to update positions without `clear`. However, this requires carefully tracking when the LAMMPS state needs a full reset vs. a position-only update.

**Current workaround:** The existing approach is correct though slow. Only optimize if performance becomes a bottleneck.

### 10.4 TETBASECalculator Full Test (LOW PRIORITY)

`TETBASECalculator` combines TB hopping (C++ pybind11), PODCoulCalculator (LAMMPS), and k-point summation (numpy/CuPy). Not fully tested at scale. The TB hopping part and PODCoulCalculator compile and run, but end-to-end validation with realistic TBLG is pending.

### 10.5 MCMC for POD Energy+Forces (LOW PRIORITY)

`EMCEE_generate_ensemble.py` supports the `get_C0` function and Temperature dict, but has not been tested with the POD energy+forces model. The `log_likelihood` function handles the energy+forces case, but the `get_MCMC_inputs` loading path for POD energy models needs verification.

---

## 11. The "Trust LAMMPS" Principle

**Key lesson:** When debugging simulation instability, ALWAYS run an equivalent computation directly in LAMMPS first. The LAMMPS-native computation is ground truth. If it succeeds, the bug is in the Python/C++ interface. If it fails, the bug is in the potential itself (fitting, extrapolation, etc.).

### For geometry relaxation:
```bash
# 1. Convert Python atoms to LAMMPS data file
from ase.io import write
write("tblg.lammps", atoms, format="lammps-data")

# 2. Run LAMMPS minimize
lmp -in minimize.in

# 3. Compare final energy with Python result
```

### For potential verification:
```bash
# At a specific geometry, compute forces in both LAMMPS and Python
# LAMMPS: run 0 and dump forces
# Python: calc.get_forces(atoms)
# Tolerance: max|F_lammps - F_python| < 1e-6 eV/Å (should be machine precision)
```

### For fitting verification:
```bash
# After fitpod produces C_coefficients.pod:
# 1. Load C_coefficients.pod into Python (np.loadtxt, skiprows=1)
# 2. Compare PODASECalculator.get_forces(training_atoms) against
#    LAMMPS forces on same training_atoms (from fitpod error output or direct run)
```

---

## 12. Key Implementation Details (Do Not Rediscover)

### 12.1 POD File Format vs. Inline Format

The C++ `write_coeff_content_to_file` function distinguishes two formats:

- **File format** (from EAPOD/fitpod): `"model_coefficients: N 0 0\n v1\n v2\n ..."`
- **Inline format** (from Python): `"model_coefficients: v1 v2 v3 ..."`

The distinction is made by checking if exactly 3 tokens follow `model_coefficients:` with the last two being 0.0. **Fragile edge case:** if coefficients 2 and 3 happen to be exactly 0.0 with only 3 coefficients total, the format detection misfires. In practice this is very unlikely for POD coefficients.

### 12.2 LAMMPS Box Convention

LAMMPS prism (triclinic) box convention:
```
region box prism xlo xhi ylo yhi zlo zhi xy xz yz
```
Where:
- `a = (xhi-xlo, 0, 0)`  
- `b = (xy, yhi-ylo, 0)`  → `xy = bx`, `yhi = by`
- `c = (xz, yz, zhi-zlo)` → `xz = cx`, `yz = cy`, `zhi = cz`

This maps directly to the `lammps_cell` matrix rows from `ase_to_lammps`.

### 12.3 Force Map Convention

`collect_forces` uses `lmp->atom->map(i + 1)` (1-based global atom ID → 0-based local index). This is correct because:
- Atoms are created as `create_atoms N single x y z` with sequential IDs starting at 1
- `atom_modify map array` ensures `map_array` is populated
- In serial MPI (single process), `nlocal = N` and the map always returns valid indices

### 12.4 Energy Tally in PairPOD

```cpp
evdwl = fastpodptr->peratomenergyforce2(...);  // energy for atom i
ev_tally_full(i, 2.0*evdwl, 0.0, ...);        // eng_vdwl += 0.5 * 2*evdwl = evdwl
```
The factor of 2 in `ev_tally_full` compensates for the 0.5 factor in `ev_tally_full`'s implementation. Net result: `eng_vdwl` = sum of per-atom energies = total potential energy. No double-counting.

### 12.5 LBFGS Is Wrong for Graphene — Always Use FIRE

Graphene in-plane spring constant ≈ 40 eV/Å². LBFGS default `H0=70 Å/eV` assumes k ≈ 0.014 eV/Å² (2800× smaller). The initial step size before maxstep clipping is `H0 × fmax = 70 × 0.167 = 11.7 Å` — **enormous**. Even with maxstep=0.2 Å, the step overshoots the in-plane equilibrium (typical displacement ~0.04 Å), causing fmax to increase rather than decrease. After 2-3 oscillating steps, the LBFGS Hessian goes bad and divergence follows.

Use `FIRE` (or `LBFGS(H0=0.025, maxstep=0.02)` as a fallback).

### 12.6 Moiré Supercell Size for 2.88°

```python
fg.twist.find_p_q(2.88)  # → p=1, q=23, theta_comp=2.8759°
# Total atoms = 1588
# Cell: ~49×42×43 Å (including 20 Å vacuum in z)
```

### 12.7 FIRE API Change in ASE 3.28+

```
UserWarning: Please do not pass forces to step(). This argument will be removed in ase 3.28.0.
```

Do NOT pass forces to `dyn.step()`. Use `dyn.run()` directly (though see Section 10.1 for the hanging bug).

---

## 13. Dependencies

```
Python packages: numpy, scipy, torch, emcee, ase, flatgraphene, h5py, pandas, tqdm, pybind11
LAMMPS: compiled with ML-POD, MOLECULE packages; shared library mode
C++ compiler: g++ (WSL)
CUDA/CuPy: optional for TETBASECalculator GPU acceleration
```

---

## 14. Test Files

| Test | Status | What it checks |
|---|---|---|
| `tests/test_pod_finite_diff.py` | ✅ PASSING | Force consistency: max err 3.4e-4 eV/Å |
| `tests/test_relaxation.py` | ⚠️ BROKEN (FIRE hangs) | 2.88° TBLG relaxation with POD |
| `tests/test_relaxation_debug.py` | ✅ PASSING | Individual FIRE steps work, timing verified |
| `tests/count_atoms.py` | ✅ | Counts atoms in moiré cell |
| `tests/test_calculators.py` | Unknown | Basic calculator tests |
| `tests/test_interface.py` | Unknown | Interface tests |

**Run the finite-diff test to verify calculator health:**
```bash
wsl -e bash -c "source /home/dpalmer/miniconda3/etc/profile.d/conda.sh && conda activate blg_uq && cd /mnt/c/Users/Daniel/Documents/research/BLG_model_builder_v2/tests && python test_pod_finite_diff.py"
```

---

## 15. Quick Reference: Common Tasks

### Build from scratch
```bash
cd /mnt/c/Users/Daniel/Documents/research/lammps/build && make -j$(nproc)
cd /mnt/c/Users/Daniel/Documents/research/BLG_model_builder_v2 && ./clean_install.sh
```

### Run finite-difference verification
```bash
# In WSL:
conda activate blg_uq
cd tests && python test_pod_finite_diff.py
# Expected: ✓ PASS with max err ~3e-4 eV/Å
```

### Fit a POD potential
```python
from uncertainty_quantification.model_fit import fit_pod
params = fit_pod(hyperparams_str, atoms_list, lammps_exec="/path/to/lmp")
# Produces C_coefficients.pod in current directory
```

### Evaluate POD potential at a geometry
```python
from blg_model_builder_v2.potentials import PODASECalculator, ncoeff_from_params
import numpy as np
hyperparams = {...}  # Must match C_coefficients.pod
params = np.loadtxt("C_coefficients.pod", skiprows=1)
calc = PODASECalculator(hyperparams, params, elements=["C"], cutoff=5.0)
atoms.calc = calc
E = atoms.get_potential_energy()
F = atoms.get_forces()
```

### Run MCMC
```bash
# In uncertainty_quantification/:
python EMCEE_generate_ensemble.py -m POD -B 1.0
```

---

## 16. Appendix: LAMMPS PairPOD Internal Structure

Understanding this prevents future segfaults:

```
PairPOD (lmp->force->pair after "pair_style pod")
├── nijmax                    ← Per-atom max neighbor count. RESET TO 0 in coeff().
│                               After run 0, = max(numneigh[i] for i in atoms).
│                               After grow_pairs() (DEAD CODE in blockMode=0), would be Nij_total.
│
├── fastpodptr (EAPOD*)       ← EAPOD instance. Deleted and re-created in coeff().
│   ├── tmpmem[9*nijmax]      ← rij, fij, tmp buffers. Freed in coeff(). Re-allocated in compute().
│   └── tmpint[4*nijmax]      ← ai, aj, ti, tj buffers. Freed in coeff(). Re-allocated in compute().
│
└── PairPOD::compute() flow (blockMode=0):
    for each atom i:
      jnum = numneigh[i]
      if (nijmax < jnum):        ← nijmax=0 after coeff(), so always true on first atom
        nijmax = MAX(nijmax, jnum)
        fastpodptr->allocate_temp_memory(nijmax)  ← grows as needed
      rij1 = &tmpmem[0]
      fij1 = &tmpmem[3*nijmax]  ← safe: nijmax >= jnum at this point
      lammpsNeighborList(rij1, ai1, ...)
      evdwl = fastpodptr->peratomenergyforce2(fij1, rij1, ...)
      ev_tally_full(i, 2*evdwl, ...)
      tallyforce(f, fij1, ai1, aj1, nij)
```
