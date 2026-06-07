#!/bin/bash
#SBATCH --job-name=handcdo-warp-capella-smoke
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=700000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:4
#SBATCH --verbose
#SBATCH --time=24:00:00
#SBATCH --output=logs/mujoco_warp_capella_smoke_%j.out
#SBATCH --error=logs/mujoco_warp_capella_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

# Adjust environment setup to the actual HPC module/venv policy.
# module purge
# module load Python/3.11
# module load CUDA
# source .venv/bin/activate

hostname
date
nvidia-smi || true

python3 -V

python3 -m pytest -q -m "not gpu and not slow"

RUN_GPU_TESTS=1 python3 -m pytest -q -m gpu

python3 scripts/benchmark_mujoco_warp.py \
  --output-dir "outputs/mujoco_warp_capella_smoke_${SLURM_JOB_ID}" \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 8 \
  --nconmax 64 \
  --njmax 128 \
  --require-warp \
  --overwrite
