"""Backward-compatible shim — use ``run_SubSamp.py`` or ``blg_model_builder.SubSamp_generate_ensemble``."""
from blg_model_builder.SubSamp_generate_ensemble import *  # noqa: F401,F403
from blg_model_builder.SubSamp_generate_ensemble import main

if __name__ == "__main__":
    main()
