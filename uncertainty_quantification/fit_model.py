#!/usr/bin/env python3
"""CLI entry point for fitting models and caching best-fit parameters."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

from blg_model_builder.fit_cli import main

if __name__ == "__main__":
    main()
