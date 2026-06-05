"""
config.py — Central hyperparameter configuration for the MLP-POD module.

All tunable knobs live here. Edit this file to change any aspect of the
data loading, descriptor computation, MLP architecture, training, or MCMC.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_FILE     = "../data/strained_bilayer_graphene_rVV10.xyz"
DATA_FRACTION = 0.08     # fraction of total structures to use (keep small for speed)
TEST_FRACTION = 0.2      # fraction of the selected subset used for testing
RANDOM_SEED   = 42

# ── POD descriptor hyperparameters ────────────────────────────────────────────
# Hash 09fdb1c2b98eb30e — index 0 from use_pod_models_hash.txt
# ncoeff = 53, test RMSE = 0.00268 eV/atom (best model in the hyperparameter search)
POD_HASH     = "09fdb1c2b98eb30e"
POD_CUTOFF   = 6.0
POD_ELEMENTS = ["C"]
POD_HYPERPARAMS = {
    "bessel_polynomial_degree":               2,
    "inverse_polynomial_degree":              8,
    "twobody_number_radial_basis_functions":  12,
    "threebody_number_radial_basis_functions": 8,
    "threebody_angular_degree":               2,
    "fourbody_number_radial_basis_functions":  4,
    "fourbody_angular_degree":                2,
    "fivebody_number_radial_basis_functions":  0,
    "fivebody_angular_degree":                0,
    "sixbody_number_radial_basis_functions":   0,
    "sixbody_angular_degree":                 0,
    "sevenbody_number_radial_basis_functions": 0,
    "sevenbody_angular_degree":               0,
}
# n_desc = 53 for single-element C with the above hyperparams

# ── MLP architecture ──────────────────────────────────────────────────────────
MLP_HIDDEN_DIM = 30      # hidden layer width
MLP_N_LAYERS   = 2       # number of hidden layers (default: 2)
MLP_ACTIVATION = "silu"  # activation function: "silu", "tanh", "relu"
# Total parameters with n_desc=53, hidden_dim=30, n_layers=2:
#   W1: 30×53=1590  b1: 30  W2: 30×30=900  b2: 30  Wout: 1×30=30  bout: 1  → 2581

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS     = 1500
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 16
COMPUTE_FORCES = False   # True: include force loss using LAMMPS Jacobians (slower)
FORCE_WEIGHT   = 0.1     # relative weight of force MSE when COMPUTE_FORCES=True

# ── MCMC (emcee) ──────────────────────────────────────────────────────────────
# Temperature T = T_WEIGHT * C0 * (2 / n_params)  [T0 formula from EMCEE_generate_ensemble.py]
# where C0 = SSE at the best-fit parameters.
N_WALKERS  = None   # None → auto: 2 * n_params (emcee EnsembleSampler minimum)
N_STEPS    = 50    # MCMC steps per walker
T_WEIGHT   = 1.0    # scales the temperature (T0 when T_WEIGHT=1)
STEP_SIZE  = 2.0    # emcee StretchMove 'a' parameter

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR         = "best_fit_params"   # descriptor cache + model checkpoint
ENSEMBLE_SAVE_DIR = "ensembles"         # MCMC ensemble pickles
FIGURES_DIR       = "figures"           # output figures

# Derived filenames (do not edit unless you also change the scripts)
DESCRIPTOR_CACHE  = f"{CACHE_DIR}/descriptors_cache_{POD_HASH}.npz"
MODEL_CHECKPOINT  = f"{CACHE_DIR}/mlp_pod_model_{POD_HASH}.pt"
ENSEMBLE_FILE     = f"{ENSEMBLE_SAVE_DIR}/mlp_pod_ensemble_{POD_HASH}.pkl"
