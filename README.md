# handcdo_mujoco

`handcdo_mujoco` is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

This is not an exact reproduction of the original hardware, unpublished hand generator, Isaac Sim pipeline, UR5e setup, OptiTrack evaluation, or physical fabrication workflow. Those parts are intentionally out of scope. The goal here is a robust research codebase for parametric hand design sampling, primitive MJCF generation, MuJoCo grasp/wrench evaluation, TPE optimization, Slurm array execution, result collection, and Random Forest plus SHAP parameter analysis.

The evaluation score is an approximation of the paper’s simulation-based wrench stability score. Geometry is deliberately simplified and uses MuJoCo primitives so the workflow can run on CPU HPC nodes. The code is structured so more accurate meshes, real deformation kernels, MJX/JAX, or GPU simulation can be added later.

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

## Inspect A Best Design

```bash
cat outputs/analysis/best_design.json
```

Generated MJCF and design JSON files are stored under:

```text
outputs/designs/<design_id>/design.json
outputs/designs/<design_id>/model.xml
```

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
- Palm kernels are approximated as configurable local contact pads, with a clean interface for replacing them with real mesh deformation later.
