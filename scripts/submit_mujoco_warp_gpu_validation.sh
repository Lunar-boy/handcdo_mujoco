#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch not found; run this on a Slurm login node or load Slurm first" >&2
  exit 127
fi

mkdir -p logs outputs/warp_gpu_validation

echo "Created runtime directories: logs outputs/warp_gpu_validation"
echo "Submitting: sbatch $* slurm/validate_mujoco_warp_gpu.sbatch"
sbatch "$@" slurm/validate_mujoco_warp_gpu.sbatch
