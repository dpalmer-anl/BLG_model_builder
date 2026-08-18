#!/bin/bash
#SBATCH --job-name=pod_aa_ab_sep
#SBATCH --output=pod_aa_ab_sep_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#
# Refit POD_index 27 from pod_hyperparam_search.csv with rcut=5 Å on 80% of
# the training set, relax TBLG at 1.2° and 0.99°, write AA/AB layer-separation
# plot under figures/.
#
# Usage (from uncertainty_quantification/):
#   chmod +x run_pod_aa_ab_sep_vs_twist.sh
#   ./run_pod_aa_ab_sep_vs_twist.sh
#   # or:
#   sbatch run_pod_aa_ab_sep_vs_twist.sh
#
# Optional env:
#   LAMMPS_EXECUTABLE=/path/to/lmp   # needed only if refitting (no cache yet)
#   CONDA_ENV=blg_uq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-blg_uq}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Activate conda if available (cluster submit helpers often use source activate).
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "${CONDA_ENV}" ]]; then
  :
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# Prefer a working LAMMPS; do not blindly take a broken ``lmp`` on PATH
# (e.g. ~/.local/bin/lmp missing liblammps.so.0).
if [[ -z "${LAMMPS_EXECUTABLE:-}" ]]; then
  if [[ -x /mnt/c/Users/Daniel/Documents/research/lammps/build/lmp ]]; then
    export LAMMPS_EXECUTABLE="/mnt/c/Users/Daniel/Documents/research/lammps/build/lmp"
    # Ensure the companion shared library is found when running fitpod
    export LD_LIBRARY_PATH="/mnt/c/Users/Daniel/Documents/research/lammps/build:${LD_LIBRARY_PATH:-}"
  elif command -v lmp >/dev/null 2>&1; then
    export LAMMPS_EXECUTABLE="$(command -v lmp)"
  fi
fi

echo "[run] cwd=$(pwd)"
echo "[run] python=$(command -v python)"
echo "[run] LAMMPS_EXECUTABLE=${LAMMPS_EXECUTABLE:-"(unset; python will probe)"}"
echo "[run] starting plot_pod_best_aa_ab_sep_vs_twist.py"

python -u visualizations/plot_pod_best_aa_ab_sep_vs_twist.py \
  --pod-index 27 \
  --rcut 5 \
  --train-frac 0.8 \
  --twist-angles 1.2 0.99 \
  --relax-backend ase \
  --relax-ftol 1e-3 \
  --relax-maxiter 2000 \
  "$@"

echo "[run] done"
