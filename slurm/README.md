# Slurm Multi-Fidelity Templates

These templates run the existing CPU-only three-stage multi-fidelity workflow:

1. Fast full-screening over all generated designs.
2. Medium re-evaluation over selected top-k design IDs.
3. High re-evaluation over selected top-k design IDs.

They are cluster templates and may need local `#SBATCH --partition`, `#SBATCH --account`, or module-load lines. They intentionally do not request GPUs, CUDA, Isaac Sim, Apptainer, MJX, or Warp.

## Workflow

Generate candidate designs:

```bash
python3 scripts/generate_designs.py \
  --n-designs 500 \
  --output-dir outputs/designs \
  --seed 10
```

Create the Slurm log directory before submitting jobs:

```bash
mkdir -p logs
```

Slurm may open stdout and stderr before the script body runs, so do not rely only on `mkdir -p logs` inside a script when using `#SBATCH --output=logs/%A_%a.out`.

Submit fast full-screening:

```bash
sbatch slurm/eval_fast_array.sbatch
```

Collect fast results:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/fast/results \
  --output-csv outputs/fast/results.csv
```

Select medium-stage design IDs:

```bash
python3 scripts/select_top_designs.py \
  --input-csv outputs/fast/results.csv \
  --top-k 100 \
  --output-design-ids outputs/medium/design_ids.txt
```

Submit medium re-evaluation:

```bash
sbatch slurm/eval_medium_array.sbatch
```

Collect medium results:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/medium/results \
  --output-csv outputs/medium/results.csv
```

Select high-stage design IDs:

```bash
python3 scripts/select_top_designs.py \
  --input-csv outputs/medium/results.csv \
  --top-k 20 \
  --output-design-ids outputs/high/design_ids.txt
```

Submit high re-evaluation:

```bash
sbatch slurm/eval_high_array.sbatch
```

Collect high results:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/high/results \
  --output-csv outputs/high/results.csv
```

Merge all available fidelity CSVs:

```bash
python3 scripts/merge_multifidelity_results.py \
  --fast-csv outputs/fast/results.csv \
  --medium-csv outputs/medium/results.csv \
  --high-csv outputs/high/results.csv \
  --output-csv outputs/multifidelity_results.csv
```

## Template Knobs

Change the Slurm array size by editing the first number in `#SBATCH --array`, for example `0-199%32`.

Change the array concurrency cap by editing the number after `%`, for example `0-199%16`.

Add cluster-specific partition or account settings with local directives such as:

```bash
#SBATCH --partition=cpu
#SBATCH --account=my_project
```

Change designs per task at submit time:

```bash
DESIGNS_PER_TASK=10 sbatch slurm/eval_fast_array.sbatch
DESIGNS_PER_TASK=10 sbatch slurm/eval_medium_array.sbatch
DESIGNS_PER_TASK=5 sbatch slurm/eval_high_array.sbatch
```

Change result directories, tool lists, or config paths with environment variables:

```bash
RESULTS_DIR=outputs/fast_hammer/results \
TOOLS=hammer \
CONFIG_PATH=configs/eval_fast.yaml \
sbatch slurm/eval_fast_array.sbatch
```

Medium and high templates also accept `DESIGN_IDS`, `OUTPUT_DIR`, `FIDELITY`, `BACKEND`, and `SEED`:

```bash
DESIGN_IDS=outputs/medium/design_ids.txt \
OUTPUT_DIR=outputs/medium \
RESULTS_DIR=outputs/medium/results \
BACKEND=mujoco_cpu \
SEED=1000 \
sbatch slurm/eval_medium_array.sbatch
```

The fast template intentionally does not expose `BACKEND` because `scripts/evaluate_design_batch.py` does not currently support a `--backend` flag.

## Medium and High Chunking

The medium and high templates read a selected design-id file and split it deterministically by `SLURM_ARRAY_TASK_ID` and `DESIGNS_PER_TASK`. Per-task chunk files are written under:

```text
outputs/medium/slurm_chunks/design_ids_<task_id>.txt
outputs/high/slurm_chunks/design_ids_<task_id>.txt
```

Missing design-id files fail clearly instead of falling back to all designs. Empty chunks exit successfully with a message.

A lightweight syntax and chunking dry run:

```bash
mkdir -p outputs/medium
printf "design_a\ndesign_b\ndesign_c\n" > outputs/medium/design_ids.txt
SLURM_ARRAY_TASK_ID=0 DESIGNS_PER_TASK=2 bash -n slurm/eval_medium_array.sbatch
```

## Fidelity Interpretation

Fast fidelity is for broad screening only. Medium fidelity is for re-ranking. High fidelity should be used for final reporting whenever available.

Do not compare final hand quality using mixed-fidelity scores unless the fidelity level is explicitly stated. The merged multi-fidelity CSV keeps separate score columns so ranking changes across fidelity levels remain inspectable.
