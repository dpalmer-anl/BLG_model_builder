#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --account=bcmp-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --job-name=example_gpu_bands
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=208g
#SBATCH --output=example_gpu_bands.log
#SBATCH --ntasks-per-node=1
module load gcc/11.4.0
module load cuda/11.8.0
module load slurm-env/0.1
module load cudnn/8.9.0.131
module load openmpi/4.1.6
module load cue-login-env/1.0
module load default-s11
module load anaconda3_gpu/23.9.0

source activate myenv
export OMP_NUM_THREADS=1
python test_tb.py
