#!/bin/bash
#SBATCH --job-name=handcdo-optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}


./.venv/bin/python scripts/run_optuna_round.py \
  --study-name handcdo-mujoco \
  --storage sqlite:///outputs/handcdo_optuna.db \
  --n-trials 20 \
  --n-grasp-trials 4 \
  --output-dir outputs \
  --seed 0 \
  --tools hammer,spoon,knife \
  --backend mujoco
