#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
cd /mnt/c/Users/Daniel/Documents/research/BLG_model_builder
conda run -n blg_uq python scripts/verify_hbn_pod_split.py
conda run -n blg_uq pytest tests/test_pod_extep_ilp_hbn.py -v --tb=short
