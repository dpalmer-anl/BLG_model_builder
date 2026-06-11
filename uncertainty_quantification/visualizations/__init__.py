"""Plotting scripts for BLG UQ workflows."""

from __future__ import annotations

import sys
from pathlib import Path

_UQ_DIR = Path(__file__).resolve().parent.parent


def ensure_uq_dir_on_path() -> Path:
    """
    Add ``uncertainty_quantification/`` to ``sys.path``.

    Scripts under ``visualizations/`` are often launched as
    ``python visualizations/<script>.py``, which puts only ``visualizations/``
    on ``sys.path``.  Local modules such as ``uq_model_runtime`` live one level
    up and require this bootstrap.
    """
    uq_dir = str(_UQ_DIR)
    if uq_dir not in sys.path:
        sys.path.insert(0, uq_dir)
    return _UQ_DIR


ensure_uq_dir_on_path()
