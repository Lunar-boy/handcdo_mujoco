#!/bin/bash -l
#SBATCH --job-name=handcdo-medium
#SBATCH --array=0-19%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DESIGN_DIR="${DESIGN_DIR:-outputs/designs}"
DESIGN_IDS="${DESIGN_IDS:-outputs/medium/design_ids.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medium}"
RESULTS_DIR="${RESULTS_DIR:-outputs/medium/results}"
CONFIG_PATH="${CONFIG_PATH:-configs/eval_medium.yaml}"
FIDELITY="${FIDELITY:-medium}"
TOOLS="${TOOLS:-hammer,spoon,knife}"
BACKEND="${BACKEND:-mujoco_cpu}"
SEED="${SEED:-0}"
DESIGNS_PER_TASK="${DESIGNS_PER_TASK:-5}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}" logs

if [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
elif [[ -r /usr/share/lmod/lmod/init/bash ]]; then
  source /usr/share/lmod/lmod/init/bash
elif [[ -r /etc/profile ]]; then
  source /etc/profile
fi

module purge
ml load release/25.06 GCCcore/13.3.0 Python/3.12.3

if [[ ! -f "${DESIGN_IDS}" ]]; then
  echo "ERROR: design-id file not found: ${DESIGN_IDS}" >&2
  echo "Create it first, for example:" >&2
  echo "  python3 scripts/select_top_designs.py --input-csv outputs/fast/results.csv --top-k 100 --output-design-ids ${DESIGN_IDS}" >&2
  exit 1
fi

START=$((TASK_ID * DESIGNS_PER_TASK + 1))
END=$((START + DESIGNS_PER_TASK - 1))
CHUNK_DIR="${OUTPUT_DIR}/slurm_chunks"
mkdir -p "${CHUNK_DIR}"
CHUNK_IDS="${CHUNK_DIR}/design_ids_${TASK_ID}.txt"
sed -n "${START},${END}p" "${DESIGN_IDS}" > "${CHUNK_IDS}"

if [[ ! -s "${CHUNK_IDS}" ]]; then
  echo "No design IDs assigned to task ${TASK_ID}; exiting successfully."
  echo "Source design-id file: ${DESIGN_IDS}"
  echo "Requested line range: ${START}-${END}"
  echo "Chunk file: ${CHUNK_IDS}"
  exit 0
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi
PYTHON="${PYTHON:-python3}"
TASK_SEED=$((SEED + TASK_ID * DESIGNS_PER_TASK))

echo "hostname: $(hostname)"
echo "working directory: $(pwd)"
echo "task ID: ${TASK_ID}"
echo "config path: ${CONFIG_PATH}"
echo "design dir: ${DESIGN_DIR}"
echo "design-id file: ${DESIGN_IDS}"
echo "results dir: ${RESULTS_DIR}"
echo "output dir: ${OUTPUT_DIR}"
echo "selected chunk file: ${CHUNK_IDS}"
echo "requested line range: ${START}-${END}"
echo "fidelity: ${FIDELITY}"
echo "backend: ${BACKEND}"
echo "tools: ${TOOLS}"
echo "seed: ${TASK_SEED}"
echo "Python executable: $(${PYTHON} -c 'import sys; print(sys.executable)')"

"${PYTHON}" scripts/reevaluate_designs.py \
  --design-dir "${DESIGN_DIR}" \
  --design-ids "${CHUNK_IDS}" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG_PATH}" \
  --fidelity "${FIDELITY}" \
  --tools "${TOOLS}" \
  --backend "${BACKEND}" \
  --seed "${TASK_SEED}"
