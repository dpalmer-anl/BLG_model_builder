"""
setup.py — minimal shim for editable installs.

blg-model-builder-v2 is now a pure-Python package.
All interatomic potentials use the LAMMPS Python module.
All C++ pybind11 extensions (potential_ext, POD_TB_cpp) have been removed.

Install with:
    pip install -e .
"""
from setuptools import setup

setup()
