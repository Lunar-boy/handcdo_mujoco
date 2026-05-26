#!/bin/bash
#SBATCH --job-name=handcdo-mujoco
#SBATCH --array=0-99%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}



./.venv/bin/python scripts/evaluate_design_batch.py \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --designs-per-task 5 \
  --design-dir outputs/designs \
  --results-dir outputs/results \
  --config configs/default_eval.yaml
