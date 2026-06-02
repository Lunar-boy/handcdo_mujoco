Implement PR 8: add Slurm scripts for fast/medium/high evaluation.

Current repository facts:
- Existing `slurm/eval_array.sbatch` supports CPU-only array evaluation.
- README says each array task reads `SLURM_ARRAY_TASK_ID`, evaluates assigned designs, catches per-design failures, and writes partial JSON results.

Goal:
Add separate Slurm scripts for fast, medium, and high fidelity evaluation without requiring GPU resources.

Required changes:
1. Add:
   - `slurm/eval_fast_array.sbatch`
   - `slurm/eval_medium_array.sbatch`
   - `slurm/eval_high_array.sbatch`

2. Keep them CPU-only.

Suggested fast:
```bash
#SBATCH --job-name=handcdo-fast
#SBATCH --array=0-199%64
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
```

Suggested medium:
```bash
#SBATCH --job-name=handcdo-medium
#SBATCH --array=0-99%32
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
```

Suggested high:
```bash
#SBATCH --job-name=handcdo-high
#SBATCH --array=0-49%16
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
```

3. Each script should:
   - Activate `.venv` if present.
   - Set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
   - Create logs and output dirs.
   - Call `scripts/evaluate_design_batch.py`.
   - Use the correct config:
     - fast: `configs/eval_fast.yaml`
     - medium: `configs/eval_medium.yaml`
     - high: `configs/eval_high.yaml`
   - Write to separate result directories:
     - `outputs/results_fast`
     - `outputs/results_medium`
     - `outputs/results_high`

4. Add a short `slurm/README.md`:
   - Explain when to use each script.
   - Explain how to modify array size and concurrency.
   - Explain that these are templates and may need local partition/account changes.

5. Optional: add environment variables:
   - `DESIGN_DIR`
   - `RESULTS_DIR`
   - `DESIGNS_PER_TASK`
   - `CONFIG_PATH`
   - `BACKEND`

Out of scope:
- Do not add GPU directives.
- Do not add Apptainer or Isaac.
- Do not implement MJX.

Validation:
```bash
bash -n slurm/eval_fast_array.sbatch
bash -n slurm/eval_medium_array.sbatch
bash -n slurm/eval_high_array.sbatch
pytest -q
```
