# handcdo_mujoco

`handcdo_mujoco` is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

This is not an exact reproduction of the original hardware, unpublished hand generator, Isaac Sim pipeline, UR5e setup, OptiTrack evaluation, or physical fabrication workflow. Those parts are intentionally out of scope. The goal here is a robust research codebase for parametric hand design sampling, primitive MJCF generation, MuJoCo grasp/wrench evaluation, TPE optimization, Slurm array execution, result collection, and Random Forest plus SHAP parameter analysis.

The evaluation score is an approximation of the paper’s simulation-based wrench stability score. Geometry is deliberately simplified and uses MuJoCo primitives so the workflow can run on CPU HPC nodes. The code is structured so more accurate meshes, real deformation kernels, MJX/JAX, or GPU simulation can be added later.

Tool geometry defaults to the original primitive hammer, spoon, and knife models. Optional `tool.mode: hybrid` configuration can add separate visual and collision meshes from `assets/tools/<tool_name>/`; when assets are missing it logs a warning and falls back to primitives without changing optimization semantics. Collision meshes should be convex or low-complexity for stable MuJoCo contact. This infrastructure does not perform convex decomposition. See [docs/tool_geometry.md](docs/tool_geometry.md).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[test]"
```

On headless clusters, MuJoCo’s Python package runs CPU simulation without Isaac Sim, ROS, or a GPU.

## Local Demo

Generate five random designs:

```bash
python3 scripts/generate_designs.py \
  --n-designs 5 \
  --output-dir outputs/designs \
  --seed 0
```

Evaluate them locally in one batch task:

```bash
python3 scripts/evaluate_design_batch.py \
  --task-id 0 \
  --designs-per-task 5 \
  --design-dir outputs/designs \
  --results-dir outputs/results \
  --config configs/default_eval.yaml
```

Collect result JSON files into CSV:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/results \
  --output-csv outputs/results.csv
```

## Slurm Array

Generate a larger candidate batch first:

```bash
python3 scripts/generate_designs.py --n-designs 500 --output-dir outputs/designs --seed 10
```

Submit the array:

```bash
mkdir -p logs
sbatch slurm/eval_array.sbatch
```

Each array task reads `SLURM_ARRAY_TASK_ID`, evaluates its assigned designs, catches per-design failures, and writes partial JSON results under `outputs/results`.

## Optuna TPE Optimization

Run a resumable 20-trial local optimization:

```bash
python3 scripts/run_optuna_round.py \
  --study-name handcdo-mujoco \
  --storage sqlite:///outputs/handcdo_optuna.db \
  --n-trials 20 \
  --n-grasp-trials 4 \
  --output-dir outputs \
  --seed 0 \
  --tools hammer,spoon,knife \
  --backend mujoco_cpu
```

The outer objective maximizes the average best grasp stability score across hammer, spoon, and knife. The inner grasp optimizer uses Optuna TPE when available and falls back to random search if necessary.
`mujoco` remains accepted as a legacy alias for the same CPU MuJoCo backend.

## Baseline Benchmark

Run this before changing contact geometry to freeze the current MuJoCo CPU baseline. Future geometry modes can be compared against this baseline with the comparison helper. The generated benchmark outputs under `outputs/` should not be committed.

```bash
python3 scripts/run_baseline_benchmark.py \
  --n-designs 2 \
  --n-grasp-trials 1 \
  --tools hammer \
  --seed 0 \
  --backend mujoco_cpu \
  --config configs/default_eval.yaml \
  --output-dir outputs/smoke_baseline

python3 scripts/compare_benchmarks.py \
  --left outputs/smoke_baseline/results.csv \
  --right outputs/smoke_baseline/results.csv \
  --output-dir outputs/smoke_baseline_compare
```

## Optional MuJoCo Warp benchmark

The repository remains CPU-only by default. MuJoCo Warp is an optional benchmark-only dependency for GPU compatibility and throughput diagnostics; it is not a production backend and does not replace CPU MuJoCo scoring. The benchmark does not compute grasp scores or prove score equivalence.

Install the optional package only in environments where you want to run the GPU diagnostics:

```bash
python3 -m pip install -e ".[warp]"
```

The CLI help and default tests work without CUDA, JAX, MJX, MuJoCo Warp, Slurm, or a GPU:

```bash
python3 scripts/benchmark_mujoco_warp.py --help

python3 scripts/benchmark_mujoco_warp.py \
  --output-dir outputs/mujoco_warp_local_cpu_smoke \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --steps 5 \
  --warmup-steps 1 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 2 \
  --overwrite
```

On CPU-only systems, the benchmark writes `availability.json`, `benchmark_results.json`, and `benchmark_results.csv`, runs the CPU timing if MuJoCo is installed, and records MuJoCo Warp as skipped unless `--require-warp` is passed. The original generated MJCF is preserved under `model/original_model.xml`; any benchmark-local compatibility copy is written as `model/warp_model.xml` with rewrites recorded in the JSON output.

Capella is the recommended PR10 GPU smoke target:

```bash
sbatch slurm/mujoco_warp_capella_smoke.sbatch
```

Alpha is reserved for heavier optional sweeps or later batched-backend experiments:

```bash
sbatch slurm/mujoco_warp_alpha_sweep.sbatch
```

Do not treat these diagnostics as physical validation, H100 correctness, or speedup evidence unless the corresponding GPU-node benchmark logs were actually produced and reported.

## SHAP Analysis

```bash
python3 scripts/run_shap_analysis.py \
  --input-csv outputs/results.csv \
  --output-dir outputs/analysis
```

Outputs include:

- `outputs/analysis/feature_importance.csv`
- `outputs/analysis/shap_importance.csv`, when SHAP succeeds
- `outputs/analysis/shap_summary.png`, when SHAP succeeds
- `outputs/analysis/optimization_convergence.png`
- `outputs/analysis/best_design.json`

If SHAP cannot import or fails on a given environment, the script writes `shap_unavailable.txt` and still saves the Random Forest importance and convergence plot.

## Surrogate candidate proposal

After collecting enough random, Optuna, or multi-fidelity MuJoCo results, a lightweight sklearn surrogate can be used as a cheap proposal filter. The surrogate predicts scores from design parameters only, ranks newly sampled candidates, and writes the predicted top candidates as normal design JSON files. These predictions are not physically validated scores; proposed candidates must still be evaluated with MuJoCo, and final reporting should use high-fidelity MuJoCo scores whenever available.

Training writes `surrogate_diagnostics.json` next to the model. Treat this as a sanity check, not physical validation: high train R2 alone is not enough, and low or negative cross-validation R2 means the surrogate ranking may be unreliable.

Train a reusable surrogate:

```bash
python3 scripts/propose_surrogate_candidates.py \
  --mode train-only \
  --results-csv outputs/multifidelity_results.csv \
  --search-space configs/search_space.yaml \
  --target best_available_score \
  --model-type random_forest \
  --output-dir outputs/surrogate_proposals \
  --seed 0
```

Reuse an existing model to propose candidates:

```bash
python3 scripts/propose_surrogate_candidates.py \
  --mode propose-only \
  --model-path outputs/surrogate_proposals/model/surrogate_model.joblib \
  --search-space configs/search_space.yaml \
  --n-random 10000 \
  --top-k 200 \
  --output-dir outputs/surrogate_proposals_seed1 \
  --seed 1 \
  --exclude-existing
```

Train and propose in one command:

```bash
python3 scripts/propose_surrogate_candidates.py \
  --mode train-propose \
  --results-csv outputs/multifidelity_results.csv \
  --search-space configs/search_space.yaml \
  --target best_available_score \
  --model-type random_forest \
  --n-random 10000 \
  --top-k 200 \
  --output-dir outputs/surrogate_proposals \
  --seed 0 \
  --exclude-existing
```

Rerunning into the same proposal output directory fails by default to prevent stale `proposed_designs` from contaminating later evaluations. Pass `--overwrite` when you intentionally want to replace `proposed_candidates.csv`, `manifest.json`, and only the `proposed_designs/` directory.

The proposed design directory can be passed directly to the existing evaluator:

```bash
python3 scripts/evaluate_design_batch.py \
  --task-id 0 \
  --designs-per-task 10 \
  --design-dir outputs/surrogate_proposals/proposed_designs \
  --results-dir outputs/surrogate_proposals/fast_results \
  --config configs/eval_fast.yaml \
  --tools hammer,spoon,knife \
  --seed 100
```

## Inspect A Best Design

```bash
cat outputs/analysis/best_design.json
```

Generated MJCF and design JSON files are stored under:

```text
outputs/designs/<design_id>/design.json
outputs/designs/<design_id>/model.xml
```

## Multi-fidelity workflow

The repository supports a deterministic three-stage multi-fidelity evaluation workflow. The goal is to cheaply screen many candidate hands at low fidelity, re-rank the best candidates at medium fidelity, and compute final scores for a small set of high-quality designs at high fidelity.

The three fidelity levels are:

- **Fast**: low simulation budget, primitive geometry, and a small grasp-search budget. Use this for broad screening.
- **Medium**: higher simulation budget, fingertip pads, palm pad grid, and medium grasp-search budget. Use this for re-ranking.
- **High**: highest simulation budget, fingertip pads, denser palm pad grid, and optional hybrid tool geometry. Use this for final ranking.

Final conclusions should be based on high-fidelity scores whenever available. The merged CSV keeps separate score columns for each fidelity level so ranking drift between fast, medium, and high evaluation remains visible.

### 1. Generate candidate designs

```bash
python3 scripts/generate_designs.py \
  --n-designs 500 \
  --output-dir outputs/designs \
  --seed 10
```

This writes one directory per design under:

```text
outputs/designs/<design_id>/
```

Each design directory contains its generated design JSON and, when requested by the evaluation config, the corresponding MJCF model.

### 2. Run fast evaluation

```bash
python3 scripts/evaluate_design_batch.py \
  --task-id 0 \
  --designs-per-task 500 \
  --design-dir outputs/designs \
  --results-dir outputs/fast/results \
  --config configs/eval_fast.yaml
```

Collect the fast-fidelity result JSON files into a CSV:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/fast/results \
  --output-csv outputs/fast/results.csv
```

### 3. Select top candidates for medium fidelity

```bash
python3 scripts/select_top_designs.py \
  --input-csv outputs/fast/results.csv \
  --top-k 100 \
  --output-design-ids outputs/medium/design_ids.txt
```

By default, failed rows and rows with invalid scores are ignored. Ties are broken deterministically by `design_id`.

### 4. Run medium re-evaluation

```bash
python3 scripts/reevaluate_designs.py \
  --design-dir outputs/designs \
  --design-ids outputs/medium/design_ids.txt \
  --results-dir outputs/medium/results \
  --output-dir outputs/medium \
  --config configs/eval_medium.yaml \
  --fidelity medium \
  --tools hammer,spoon,knife \
  --backend mujoco_cpu \
  --seed 1000
```

Collect medium-fidelity results:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/medium/results \
  --output-csv outputs/medium/results.csv
```

### 5. Select top candidates for high fidelity

```bash
python3 scripts/select_top_designs.py \
  --input-csv outputs/medium/results.csv \
  --top-k 20 \
  --output-design-ids outputs/high/design_ids.txt
```

### 6. Run high re-evaluation

```bash
python3 scripts/reevaluate_designs.py \
  --design-dir outputs/designs \
  --design-ids outputs/high/design_ids.txt \
  --results-dir outputs/high/results \
  --output-dir outputs/high \
  --config configs/eval_high.yaml \
  --fidelity high \
  --tools hammer,spoon,knife \
  --backend mujoco_cpu \
  --seed 2000
```

Collect high-fidelity results:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/high/results \
  --output-csv outputs/high/results.csv
```

### 7. Merge fidelity levels

```bash
python3 scripts/merge_multifidelity_results.py \
  --fast-csv outputs/fast/results.csv \
  --medium-csv outputs/medium/results.csv \
  --high-csv outputs/high/results.csv \
  --output-csv outputs/multifidelity_results.csv
```

The merged CSV contains separate score columns:

```text
hand_score_fast
hand_score_medium
hand_score_high
best_available_score
```

`best_available_score` uses the highest available fidelity in this order:

```text
high > medium > fast
```

The merged file also preserves per-fidelity metadata where available, including:

```text
fidelity
backend
config_path
n_grasp_trials
sampler
seed
```

### Custom fidelity inputs

The merge script also supports custom fidelity names:

```bash
python3 scripts/merge_multifidelity_results.py \
  --input fast=outputs/fast/results.csv \
  --input medium=outputs/medium/results.csv \
  --input high=outputs/high/results.csv \
  --output-csv outputs/multifidelity_results.csv
```

### Notes

- Fast fidelity is intended only for broad screening.
- Medium fidelity is intended for re-ranking.
- High fidelity should be used for final reporting whenever available.
- Do not compare final hand quality using mixed-fidelity scores unless the fidelity level is explicitly stated.
- Generated output directories such as `outputs/fast`, `outputs/medium`, `outputs/high`, and `outputs/multifidelity_results.csv` should not be committed.

## Tests

```bash
pytest
```

Tests cover design-space bounds, deterministic JSON round trips, MJCF loadability when MuJoCo is installed, wrench-score bounds, and robust result collection from partial or failed batch outputs.

## Project Layout

```text
handcdo_mujoco/
├── configs/
├── handcdo/
├── scripts/
├── slurm/
├── tests/
└── outputs/
```

## Scope Notes

- No Isaac Sim integration.
- No ROS dependency.
- No GPU-only libraries.
- No real robot control, fabrication, OptiTrack, or physical validation.
- Primitive hammer, spoon, and knife models are placeholders for simulation infrastructure tests.
- Hybrid tool geometry supports optional visual/collision meshes while preserving primitive fallback and the default primitive path.
- Palm kernels are approximated as configurable local contact pads, with a clean interface for replacing them with real mesh deformation later.
