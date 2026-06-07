#!/bin/bash
#SBATCH --job-name=handcdo-warp-alpha-sweep
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=990000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:8
#SBATCH --verbose
#SBATCH --time=72:00:00
#SBATCH --output=logs/mujoco_warp_alpha_sweep_%j.out
#SBATCH --error=logs/mujoco_warp_alpha_sweep_%j.err

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

RUN_GPU_TESTS=1 python3 -m pytest -q -m gpu

python3 scripts/benchmark_mujoco_warp.py \
  --output-dir "outputs/mujoco_warp_alpha_sweep_${SLURM_JOB_ID}" \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 100 \
  --warmup-steps 10 \
  --cpu-repeats 1 \
  --warp-repeats 3 \
  --sweep-nworld 8,32,64,128 \
  --sweep-nconmax 64,128,256 \
  --sweep-njmax 128,256,512 \
  --require-warp \
  --overwrite
