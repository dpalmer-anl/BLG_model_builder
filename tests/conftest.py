"""
conftest.py – pytest configuration for BLG_model_builder_v2 tests.

blg-model-builder-v2 is a pure-Python package.  Install it in editable mode:

    pip install -e .

Interatomic potential calculators (KC, Tersoff, DRIP, POD) require the
LAMMPS Python module.  Install it via `make install-python` in the LAMMPS
build tree, then run:

    conda activate blg_uq
    cd tests/
    pytest -v
"""
