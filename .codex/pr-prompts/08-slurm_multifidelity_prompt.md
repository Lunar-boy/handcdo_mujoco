# PR 8 Prompt: Slurm Multi-Fidelity Launch Templates

Implement PR 8: add Slurm launch templates for the existing multi-fidelity workflow.

## Current repository facts

- This repository is a CPU-only MuJoCo reproduction of the HandCDO optimization infrastructure.
- PR7 already added the multi-fidelity pipeline:
  - `configs/eval_fast.yaml`
  - `configs/eval_medium.yaml`
  - `configs/eval_high.yaml`
  - `scripts/select_top_designs.py`
  - `scripts/reevaluate_designs.py`
  - `scripts/merge_multifidelity_results.py`
- `scripts/evaluate_design_batch.py` evaluates contiguous slices of all designs under `outputs/designs`.
- `scripts/reevaluate_designs.py` evaluates an explicit design-id list and records fidelity metadata.
- README currently uses:
  - `outputs/fast/results`
  - `outputs/medium/results`
  - `outputs/high/results`

## Goal

Add Slurm templates that execute the existing deterministic three-stage workflow:

1. Fast full-screening over all generated designs.
2. Medium re-evaluation over selected top-k design IDs.
3. High re-evaluation over selected top-k design IDs.

## Required changes

### 1. Add files

Add the following files:

- `slurm/eval_fast_array.sbatch`
- `slurm/eval_medium_array.sbatch`
- `slurm/eval_high_array.sbatch`
- `slurm/README.md`

### 2. Keep all Slurm scripts CPU-only

All Slurm scripts must remain CPU-only.

Do not add:

- `#SBATCH --gres=gpu`
- CUDA-specific assumptions
- Isaac Sim
- Apptainer
- MJX
- Warp

### 3. Fast Slurm script behavior

`slurm/eval_fast_array.sbatch` should run broad full-screening over all generated designs.

Use:

- Script: `scripts/evaluate_design_batch.py`
- Default config: `configs/eval_fast.yaml`
- Default design dir: `outputs/designs`
- Default results dir: `outputs/fast/results`
- Default designs per task: `5`
- Default tools: `hammer,spoon,knife`
- Task index source: `SLURM_ARRAY_TASK_ID`

Allow overrides through environment variables:

- `DESIGN_DIR`
- `RESULTS_DIR`
- `DESIGNS_PER_TASK`
- `CONFIG_PATH`
- `TOOLS`
- `SEED`

Do not expose `BACKEND` in this script unless `scripts/evaluate_design_batch.py` is also updated to support `--backend`.

Suggested fast Slurm resources:

```bash
#SBATCH --job-name=handcdo-fast
#SBATCH --array=0-99%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
```

### 4. Medium Slurm script behavior

`slurm/eval_medium_array.sbatch` should re-evaluate only the selected medium-stage design IDs.

Use:

- Script: `scripts/reevaluate_designs.py`
- Default design-id list: `outputs/medium/design_ids.txt`
- Default config: `configs/eval_medium.yaml`
- Default design dir: `outputs/designs`
- Default output dir: `outputs/medium`
- Default results dir: `outputs/medium/results`
- Default fidelity: `medium`
- Default backend: `mujoco_cpu`
- Default tools: `hammer,spoon,knife`

Important requirements:

- Use `scripts/reevaluate_designs.py`, not `scripts/evaluate_design_batch.py`.
- Split the design-id file by Slurm array index so each task evaluates a deterministic chunk.
- Do not silently re-evaluate all designs if the design-id file is missing.
- Fail clearly with an actionable error message if `outputs/medium/design_ids.txt` does not exist.
- Empty chunks should exit successfully with a clear message.

Allow overrides through environment variables:

- `DESIGN_DIR`
- `DESIGN_IDS`
- `OUTPUT_DIR`
- `RESULTS_DIR`
- `CONFIG_PATH`
- `FIDELITY`
- `TOOLS`
- `BACKEND`
- `SEED`
- `DESIGNS_PER_TASK`

Suggested medium Slurm resources:

```bash
#SBATCH --job-name=handcdo-medium
#SBATCH --array=0-19%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
```

### 5. High Slurm script behavior

`slurm/eval_high_array.sbatch` should re-evaluate only the selected high-stage design IDs.

Use:

- Script: `scripts/reevaluate_designs.py`
- Default design-id list: `outputs/high/design_ids.txt`
- Default config: `configs/eval_high.yaml`
- Default design dir: `outputs/designs`
- Default output dir: `outputs/high`
- Default results dir: `outputs/high/results`
- Default fidelity: `high`
- Default backend: `mujoco_cpu`
- Default tools: `hammer,spoon,knife`

Important requirements:

- Use `scripts/reevaluate_designs.py`, not `scripts/evaluate_design_batch.py`.
- Split the design-id file by Slurm array index so each task evaluates a deterministic chunk.
- Do not silently re-evaluate all designs if the design-id file is missing.
- Fail clearly with an actionable error message if `outputs/high/design_ids.txt` does not exist.
- Empty chunks should exit successfully with a clear message.

Allow overrides through environment variables:

- `DESIGN_DIR`
- `DESIGN_IDS`
- `OUTPUT_DIR`
- `RESULTS_DIR`
- `CONFIG_PATH`
- `FIDELITY`
- `TOOLS`
- `BACKEND`
- `SEED`
- `DESIGNS_PER_TASK`

Suggested high Slurm resources:

```bash
#SBATCH --job-name=handcdo-high
#SBATCH --array=0-9%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
```

### 6. Environment setup in all scripts

Each Slurm script should:

- Use `set -euo pipefail`.
- Activate `.venv` if `.venv/bin/activate` exists.
- Otherwise fall back to `python3`.
- Set:
  - `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}`
  - `MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}`
  - `OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}`
- Create runtime output directories with `mkdir -p`.
- Print useful runtime information:
  - hostname
  - working directory
  - task ID
  - config path
  - design dir
  - results dir
  - Python executable
  - selected chunk file for medium/high scripts

Important Slurm note:

- Document that `logs/` should be created before `sbatch`, because Slurm may open stdout/stderr before the script body runs.
- Do not rely only on `mkdir -p logs` inside the script when `#SBATCH --output=logs/%A_%a.out` is used.

### 7. Chunking behavior for medium/high

If needed, create temporary per-task design-id files under the relevant output directory.

Examples:

- `outputs/medium/slurm_chunks/design_ids_${SLURM_ARRAY_TASK_ID}.txt`
- `outputs/high/slurm_chunks/design_ids_${SLURM_ARRAY_TASK_ID}.txt`

Chunking requirements:

- Deterministic.
- Based on `SLURM_ARRAY_TASK_ID` and `DESIGNS_PER_TASK`.
- Preserve the ordering in the original design-id file.
- Empty chunks should exit with status 0 and a clear message.
- Do not duplicate evaluation logic already present in `handcdo.multifidelity`.

Recommended shell logic:

```bash
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DESIGNS_PER_TASK="${DESIGNS_PER_TASK:-5}"
START=$((TASK_ID * DESIGNS_PER_TASK + 1))
END=$((START + DESIGNS_PER_TASK - 1))

CHUNK_DIR="${OUTPUT_DIR}/slurm_chunks"
mkdir -p "${CHUNK_DIR}"
CHUNK_IDS="${CHUNK_DIR}/design_ids_${TASK_ID}.txt"

sed -n "${START},${END}p" "${DESIGN_IDS}" > "${CHUNK_IDS}"

if [[ ! -s "${CHUNK_IDS}" ]]; then
  echo "No design IDs assigned to task ${TASK_ID}; exiting successfully."
  exit 0
fi
```

Then pass `--design-ids "${CHUNK_IDS}"` to `scripts/reevaluate_designs.py`.

### 8. Add `slurm/README.md`

The README should explain the full expected workflow:

1. Generate designs.
2. Create `logs/`.
3. Submit fast array.
4. Collect fast results.
5. Select medium design IDs.
6. Submit medium array.
7. Collect medium results.
8. Select high design IDs.
9. Submit high array.
10. Collect high results.
11. Merge multi-fidelity CSVs.

Include example commands:

```bash
python3 scripts/generate_designs.py   --n-designs 500   --output-dir outputs/designs   --seed 10

mkdir -p logs

sbatch slurm/eval_fast_array.sbatch

python3 scripts/collect_results.py   --results-dir outputs/fast/results   --output-csv outputs/fast/results.csv

python3 scripts/select_top_designs.py   --input-csv outputs/fast/results.csv   --top-k 100   --output-design-ids outputs/medium/design_ids.txt

sbatch slurm/eval_medium_array.sbatch

python3 scripts/collect_results.py   --results-dir outputs/medium/results   --output-csv outputs/medium/results.csv

python3 scripts/select_top_designs.py   --input-csv outputs/medium/results.csv   --top-k 20   --output-design-ids outputs/high/design_ids.txt

sbatch slurm/eval_high_array.sbatch

python3 scripts/collect_results.py   --results-dir outputs/high/results   --output-csv outputs/high/results.csv

python3 scripts/merge_multifidelity_results.py   --fast-csv outputs/fast/results.csv   --medium-csv outputs/medium/results.csv   --high-csv outputs/high/results.csv   --output-csv outputs/multifidelity_results.csv
```

Also explain how to change:

- Slurm array size.
- Array concurrency cap.
- Partition/account directives.
- Designs per task.
- Result directories.
- Tool list.
- Config paths.

Clearly state:

- These scripts are cluster templates and may need local `#SBATCH --partition`, `#SBATCH --account`, or module-load lines.
- Fast fidelity is for broad screening only.
- Medium fidelity is for re-ranking.
- High fidelity should be used for final reporting whenever available.
- Do not compare final hand quality using mixed-fidelity scores unless the fidelity level is explicitly stated.

### 9. Tests and validation

Run:

```bash
bash -n slurm/eval_fast_array.sbatch
bash -n slurm/eval_medium_array.sbatch
bash -n slurm/eval_high_array.sbatch
pytest -q
```

Also verify that the Slurm scripts contain no GPU directives:

```bash
! grep -R --line-number -- '--gres=gpu\|nvidia-smi\|module load cuda' slurm/
```

If shell chunking logic is added, document a lightweight dry-run test using a small synthetic design-id file.

Example:

```bash
mkdir -p outputs/medium
printf "design_a\ndesign_b\ndesign_c\n" > outputs/medium/design_ids.txt

SLURM_ARRAY_TASK_ID=0 DESIGNS_PER_TASK=2 bash -n slurm/eval_medium_array.sbatch
```

## Out of scope

Do not implement:

- GPU support.
- Isaac Sim integration.
- Apptainer integration.
- MJX or Warp.
- Surrogate modeling.
- Changes to the scientific scoring function.
- Changes to PR7 CSV merge semantics.
- Mesh asset generation or downloading.
