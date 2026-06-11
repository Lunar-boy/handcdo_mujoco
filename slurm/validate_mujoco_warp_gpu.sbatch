#!/bin/bash
#SBATCH --job-name=warp-gpu-validate
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --verbose
#SBATCH --output=logs/warp_gpu_validate_%j.out
#SBATCH --error=logs/warp_gpu_validate_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/warp_gpu_validation

echo "=== Host ==="
hostname
date
echo "PWD=$(pwd)"

echo "=== Slurm/GPU ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true


module purge
ml load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0


# Override with: sbatch --export=ALL,VENV_PATH=/path/to/venv slurm/validate_mujoco_warp_gpu.sbatch
VENV_PATH="${VENV_PATH:-.venv}"

if [ -x "${VENV_PATH}/bin/python" ]; then
  source "${VENV_PATH}/bin/activate"
  PYTHON="${VENV_PATH}/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "=== Python ==="
"${PYTHON}" -V
"${PYTHON}" -m pip show mujoco mujoco-warp warp-lang || true

echo "=== MuJoCo Warp GPU validation ==="
"${PYTHON}" scripts/validate_mujoco_warp_gpu.py \
  --strict \
  --results-dir outputs/warp_gpu_validation \
  --tool hammer \
  --n-grasps 2 \
  --nworld 2 \
  --wrench-steps 2 \
  --warmup-steps 1 \
  --readback-interval 1

echo "=== Done ==="
date
