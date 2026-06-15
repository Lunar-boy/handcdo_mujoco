#!/bin/bash -l
#SBATCH --job-name=handcdo-fast
#SBATCH --array=0-99%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DESIGN_DIR="${DESIGN_DIR:-outputs/designs}"
RESULTS_DIR="${RESULTS_DIR:-outputs/fast/results}"
DESIGNS_PER_TASK="${DESIGNS_PER_TASK:-5}"
CONFIG_PATH="${CONFIG_PATH:-configs/eval_fast.yaml}"
TOOLS="${TOOLS:-hammer,spoon,knife}"
SEED="${SEED:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "${RESULTS_DIR}" logs

unset -f module ml __lmod_internal_module_cmd 2>/dev/null || true

if [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
elif [[ -r /usr/share/lmod/lmod/init/bash ]]; then
  source /usr/share/lmod/lmod/init/bash
elif [[ -r /etc/profile ]]; then
  source /etc/profile
fi


module purge
ml load release/25.06 GCCcore/13.3.0 Python/3.12.3

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi
PYTHON="${PYTHON:-python3}"

echo "hostname: $(hostname)"
echo "working directory: $(pwd)"
echo "task ID: ${TASK_ID}"
echo "config path: ${CONFIG_PATH}"
echo "design dir: ${DESIGN_DIR}"
echo "results dir: ${RESULTS_DIR}"
echo "designs per task: ${DESIGNS_PER_TASK}"
echo "tools: ${TOOLS}"
echo "seed: ${SEED}"
echo "Python executable: $(${PYTHON} -c 'import sys; print(sys.executable)')"

"${PYTHON}" scripts/evaluate_design_batch.py \
  --task-id "${TASK_ID}" \
  --designs-per-task "${DESIGNS_PER_TASK}" \
  --design-dir "${DESIGN_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --config "${CONFIG_PATH}" \
  --tools "${TOOLS}" \
  --seed "${SEED}"
