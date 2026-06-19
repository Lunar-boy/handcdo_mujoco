# handcdo_mujoco

`handcdo_mujoco` is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

 The goal here is a robust research codebase for parametric hand design sampling, primitive MJCF generation, MuJoCo grasp/wrench evaluation, TPE optimization, Slurm array execution, result collection, and Random Forest plus SHAP parameter analysis.

The evaluation score is an approximation of the paper’s simulation-based wrench stability score. 

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
sbatch slurm/eval_fast_array.sbatch
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

## Paper-like evaluation protocol

An explicit reproduction protocol defines the canonical 12 Cartesian force/torque directions, normalized stable-time scoring, best-grasp aggregation per tool, and mean aggregation across hammer, spoon, and knife.

Run the cheap deterministic protocol smoke test:

```bash
python3 scripts/run_paper_eval_smoke.py \
  --config configs/eval_paper_protocol.yaml \
  --num-designs 2 \
  --output outputs/paper_eval_smoke
```

Add `--backend mujoco_cpu` to run the same protocol through CPU MuJoCo. Outputs include nested `results.json`, per-wrench `results.csv`, and `run_config.json`. The default geometry is `configs/eval_paper_like.yaml`.

This protocol approximates the paper's stability objective in MuJoCo. Exact scores may differ because the simulator backend, tool meshes, contact parameters, and grasp optimizer are not identical. See [docs/reproduction.md](docs/reproduction.md) for the schema, larger-run guidance, and reproduction boundary.

## Palm outline design parameters

The morphology search includes bounded rigid palm outline parameters:

- `palm_half_x`: rigid palm body half extent in local X.
- `palm_half_y`: base rigid palm body half extent in local Y.
- `palm_half_z`: rigid palm body half thickness.
- `palm_aspect_ratio`: multiplier applied to the local-Y outline extent.
- `palm_polygon_sides`: convex outline vertex count, chosen from 6, 8, 10, or 12.

Legacy design JSON files that omit these fields load with the previous fixed palm defaults `(0.085, 0.115, 0.032)`, aspect ratio `1.0`, and 8 sides. `palm_kernel_max_height` controls surface deformation only; it does not change the rigid palm body size.

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
- **Stable high**: `configs/eval_high.yaml` uses the highest standard simulation budget, fingertip pads, a denser palm pad grid, and optional hybrid tool geometry. Use this for final ranking that must remain comparable with existing high-fidelity runs.
- **Paper-like**: `configs/eval_paper_like.yaml` keeps the stable-high simulation and grasp budgets while enabling ellipsoid fingertips with local convex patch colliders, outline-aware palm tiled mesh colliders, and hybrid tool geometry with primitive fallback. This is the closest currently available geometry approximation to the paper, not a numerically equivalent reproduction.

Final conclusions should be based on high-fidelity scores whenever available. The merged CSV keeps separate score columns for each fidelity level so ranking drift between fast, medium, and high evaluation remains visible.

The paper-like config changes contact geometry. Its scores are not directly comparable with `eval_high.yaml`, older `pad_grid` runs, or `capsule_tip_pad` evaluations. Keep the config name with reported results and re-evaluate the same design IDs and seeds when comparing geometry tiers.

An additional explicit convex-patch configuration is available for palm-contact ablations or selected-design re-evaluation. It maps the existing palm kernel parameters to a bounded deterministic grid of height-varying MuJoCo box patches. Use `configs/eval_palm_convex_patches.yaml` for a complete high-fidelity evaluation configuration; `configs/geometry_palm_convex_patches.yaml` is geometry-only. This is a CPU-only contact approximation, not deformable mesh or soft-body simulation, and it does not replace the default high-fidelity configuration. See [docs/geometry_config.md](docs/geometry_config.md).

The rigid palm body is a deterministic convex 2D outline extruded into a closed MuJoCo mesh. Finger and thumb bases are derived from named boundary frames, while fixed inward insets keep the default layout close to the previous box model. Gaussian palm parameters affect only the contact pads, patches, and surface colliders; they no longer resize the rigid palm body. This is an architectural approximation of the paper's parametric palm generator, not a fabrication-grade or numerically equivalent reproduction.

The optional `palm.mode: tiled_mesh_colliders` mode evaluates the Gaussian palm surface on deterministic local mesh colliders. `mesh_collider_domain: outline` clips boundary tiles to the convex `PalmBodySpec` outline, while `bbox` preserves the legacy rectangular height-field behavior. This is closer to the paper's surface-pad mesh plus small-collider idea than the box-based modes, but it remains a static CPU-only MuJoCo approximation rather than VHACD, non-convex geometry processing, fabrication export, or deformable simulation. Use `configs/eval_palm_tiled_mesh_colliders.yaml` for selected-design evaluation; existing default contact modes are unchanged.

### Palm surface mesh export

Export a selected design's Gaussian-deformed palm top surface as a static visual OBJ/STL mesh:

```bash
python3 scripts/export_palm_surface_mesh.py \
  --design-json outputs/designs/<design_id>/design.json \
  --output-dir outputs/designs/<design_id>/meshes \
  --resolution 32 \
  --formats obj,stl
```

The exporter uses the existing `palm_kernel_*` design parameters and does not alter evaluation or MuJoCo collision geometry. It is not runtime deformable simulation or fabrication-ready full-hand generation. Batch usage and geometry details are documented in [docs/geometry_config.md](docs/geometry_config.md).

Export the corresponding local collision tiles without running evaluation:

```bash
python3 scripts/export_palm_mesh_colliders.py \
  --design-json outputs/designs/<design_id>/design.json \
  --output-dir outputs/designs/<design_id>/meshes/collision \
  --resolution 6 \
  --collider-type quad_frustum \
  --domain outline \
  --format stl
```

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

Tests cover design-space bounds, deterministic JSON round trips, MJCF loadability when MuJoCo is installed, canonical wrench directions, normalized stable-time and aggregation bounds, deterministic protocol smoke evaluation, and robust result collection from partial or failed batch outputs.

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
- Hybrid tool geometry supports optional visual/collision meshes while preserving primitive fallback and the default primitive path.
- Palm kernels are approximated as configurable local contact pads or optional deterministic convex-patch fields; neither mode is real mesh deformation.
