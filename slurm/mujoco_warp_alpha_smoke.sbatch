#!/bin/bash
#SBATCH --job-name=handcdo-warp-alpha-smoke
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH --time=00:30:00
#SBATCH --output=logs/mujoco_warp_capella_smoke_%j.out
#SBATCH --error=logs/mujoco_warp_capella_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

echo "=== Host ==="
hostname
date

echo "=== Slurm ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "=== GPU ==="
nvidia-smi || true

# Optional: load site modules if your HPC requires them.
module purge

ml load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
# Use project virtual environment by default.
# Override with: sbatch --export=ALL,VENV_PATH=/path/to/venv slurm/mujoco_warp_capella_smoke.sbatch
VENV_PATH="${VENV_PATH:-.venv}"

if [ ! -x "${VENV_PATH}/bin/python" ]; then
  echo "ERROR: virtual environment not found or not executable: ${VENV_PATH}/bin/python"
  echo "Create it first, for example:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install --upgrade pip"
  echo "  .venv/bin/python -m pip install -e '.[test,warp]'"
  exit 2
fi

source "${VENV_PATH}/bin/activate"

PYTHON="${VENV_PATH}/bin/python"

echo "=== Python ==="
which python
"${PYTHON}" -V
"${PYTHON}" -m pip show pytest mujoco mujoco-warp || true

echo "=== CPU/default tests ==="
"${PYTHON}" -m pytest -q -m "not gpu and not slow"

echo "=== GPU marker tests ==="
RUN_GPU_TESTS=1 "${PYTHON}" -m pytest -q -m gpu

echo "=== MuJoCo Warp Capella smoke ==="
"${PYTHON}" scripts/benchmark_mujoco_warp.py \
  --output-dir "outputs/mujoco_warp_capella_smoke_${SLURM_JOB_ID}" \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 4 \
  --nconmax 64 \
  --njmax 128 \
  --require-warp \
  --overwrite

echo "=== Done ==="
date