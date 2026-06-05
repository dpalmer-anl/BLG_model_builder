"""Backward-compatible shim — use ``run_MCMC.py`` or ``blg_model_builder.EMCEE_generate_ensemble``."""
from blg_model_builder.EMCEE_generate_ensemble import *  # noqa: F401,F403
from blg_model_builder.EMCEE_generate_ensemble import main

if __name__ == "__main__":
    main()
