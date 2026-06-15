#!/bin/bash -l
#SBATCH --job-name=handcdo-optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

MODULE_RELEASE="${MODULE_RELEASE:-release/25.06}"
MODULE_GCCCORE="${MODULE_GCCCORE:-GCCcore/13.3.0}"
MODULE_PYTHON="${MODULE_PYTHON:-Python/3.12.3}"

STUDY_NAME="${STUDY_NAME:-handcdo-mujoco}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/optuna}"
STORAGE="${STORAGE:-sqlite:///${OUTPUT_DIR}/handcdo_optuna.db}"
N_TRIALS="${N_TRIALS:-20}"
N_GRASP_TRIALS="${N_GRASP_TRIALS:-4}"
SEED="${SEED:-0}"
TOOLS="${TOOLS:-hammer,spoon,knife}"
BACKEND="${BACKEND:-mujoco_cpu}"
CONFIG_PATH="${CONFIG_PATH:-configs/default_eval.yaml}"
SEARCH_SPACE="${SEARCH_SPACE:-configs/search_space.yaml}"

if [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
elif [[ -r /usr/share/lmod/lmod/init/bash ]]; then
  source /usr/share/lmod/lmod/init/bash
elif [[ -r /etc/profile ]]; then
  source /etc/profile
fi

module purge
ml load "${MODULE_RELEASE}" "${MODULE_GCCCORE}" "${MODULE_PYTHON}"

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python not found or not executable."
  echo "Run: python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -e '.[test]'"
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${SEARCH_SPACE}" ]]; then
  echo "ERROR: SEARCH_SPACE does not exist: ${SEARCH_SPACE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/results"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Python: $(realpath .venv/bin/python)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Storage: ${STORAGE}"
echo "Study: ${STUDY_NAME}"
echo "Trials: ${N_TRIALS}"
echo "Grasp trials: ${N_GRASP_TRIALS}"
echo "Tools: ${TOOLS}"
echo "Backend: ${BACKEND}"
echo "Config: ${CONFIG_PATH}"
echo "Search space: ${SEARCH_SPACE}"

./.venv/bin/python scripts/run_optuna_round.py \
  --study-name "${STUDY_NAME}" \
  --storage "${STORAGE}" \
  --n-trials "${N_TRIALS}" \
  --n-grasp-trials "${N_GRASP_TRIALS}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --tools "${TOOLS}" \
  --backend "${BACKEND}" \
  --config "${CONFIG_PATH}" \
  --search-space "${SEARCH_SPACE}"