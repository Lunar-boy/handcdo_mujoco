# PR 3 Prompt — Baseline Benchmark and Regression Comparison

Implement PR 3: add a reproducible baseline benchmark workflow and benchmark comparison tools.

## Context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

PR1 introduced the simulator backend abstraction with `mujoco_cpu`.
PR2 introduced `GeometryConfig` and threaded it through MJCF generation, evaluation, and optimization as schema/plumbing only.

This PR must freeze the current MuJoCo CPU baseline before PR4/PR5/PR6 start changing contact geometry.

Do **not** add Isaac Sim, ROS, GPU-only dependencies, MJX, MuJoCo-Warp, real robot control, OptiTrack, fabrication assets, or large mesh assets in this PR.

This PR must not change evaluator behavior, scoring behavior, default geometry, MJCF generation semantics, or optimizer behavior.

## Current repository facts

- `handcdo.slurm_batch.generate_designs(...)` already creates deterministic design JSON files and a `manifest.json`.
- `handcdo.optimize_hand.evaluate_design(...)` already accepts:
  - `config: EvaluationConfig | None`
  - `geometry_config: GeometryConfig | None`
  - `backend_name: str`
  - `backend: SimulatorBackend | None`
- `handcdo.collect_results.collect_results(...)` already flattens JSON result files into CSV rows containing:
  - `design_id`
  - `hand_score`
  - `failed`
  - `error`
  - design parameters
  - per-tool columns like `{tool}_best_score`
- `configs/default_eval.yaml` contains simulation, wrench, grasp, and output sections.
- Default benchmark should use `backend=mujoco_cpu` and `configs/default_eval.yaml`.

## Goal

Add a reproducible benchmark workflow that:

1. Generates or reuses a fixed set of hand designs.
2. Evaluates those designs with fixed tools, fixed seeds, fixed backend, and fixed config.
3. Writes result JSON files, a collected CSV, and a metadata JSON.
4. Provides a comparison script to compare two benchmark CSV files.
5. Supports future regression checks after contact geometry changes.

## Required changes

### 1. Add script wrapper

Add:

```text
scripts/run_baseline_benchmark.py
```

This should be a thin wrapper that imports and calls a main function from a proper package module, for example:

```text
handcdo/baseline_benchmark.py
```

Avoid putting all implementation logic directly inside the script wrapper.

### 2. Implement benchmark module

Create:

```text
handcdo/baseline_benchmark.py
```

Expose reusable functions where practical:

```python
run_baseline_benchmark(...)
write_benchmark_metadata(...)
get_git_metadata(...)
get_environment_metadata(...)
```

### 3. Benchmark CLI

Suggested CLI:

```bash
python3 scripts/run_baseline_benchmark.py \
  --n-designs 20 \
  --n-grasp-trials 4 \
  --tools hammer,spoon,knife \
  --seed 0 \
  --backend mujoco_cpu \
  --config configs/default_eval.yaml \
  --search-space configs/search_space.yaml \
  --output-dir outputs/baselines/current
```

Required arguments/options:

- `--n-designs`, default `20`
- `--n-grasp-trials`, default `4`
- `--tools`, default `hammer,spoon,knife`
- `--seed`, default `0`
- `--backend`, default `mujoco_cpu`, choices `mujoco,mujoco_cpu`
- `--config`, default `configs/default_eval.yaml`
- `--search-space`, default `configs/search_space.yaml`
- `--output-dir`, default `outputs/baselines/current`
- Optional `--design-dir`
  - If supplied and contains design JSON files, reuse those designs.
  - If omitted, generate designs under `<output-dir>/designs`.
- Optional `--reuse-designs`
  - If true, do not regenerate existing designs if `<design-dir>/manifest.json` exists.
- Optional `--sampler`
  - default should come from config if present; otherwise keep existing behavior.

### 4. Output layout

The benchmark must write:

```text
<output-dir>/
  designs/
    manifest.json
    <design_id>/design.json
  results/
    <design_id>.json
  results.csv
  metadata.json
```

Do not commit generated outputs.

### 5. Design generation/reuse behavior

If no reusable design directory is provided:

- Use `DesignSpace.from_yaml(search_space)` if `search_space` is provided.
- Generate exactly `n_designs` designs deterministically from `seed`.
- Prefer reusing existing `handcdo.slurm_batch.generate_designs(...)` instead of duplicating design sampling logic.
- Write `manifest.json`.

If a design directory is provided:

- Load designs from `*/design.json`, sorted deterministically.
- If `manifest.json` exists, prefer its ordering.
- If fewer than `n_designs` designs are available, raise a clear `ValueError`.
- Evaluate only the first `n_designs` designs.

### 6. Evaluation behavior

For each design:

- Load config once from `--config`.
- Parse:

```python
eval_config = EvaluationConfig.from_dict(config_data)
geometry_config = GeometryConfig.from_dict(config_data)
```

- Call `evaluate_design(...)` directly.
- Pass:
  - `tools`
  - `n_grasp_trials`
  - `output_dir`
  - `result_dir=<output-dir>/results`
  - `seed=seed + design_index`
  - `config=eval_config`
  - `geometry_config=geometry_config`
  - `backend_name=backend`

The benchmark must continue to catch per-design exceptions and write failed result JSONs, following the existing batch-evaluation behavior.

### 7. Collect results

After evaluation, call the existing collection logic:

```python
collect_results(<output-dir>/results, <output-dir>/results.csv)
```

Do not implement a second incompatible CSV schema.

### 8. Metadata JSON

Write:

```text
<output-dir>/metadata.json
```

Required fields:

```json
{
  "benchmark_schema_version": 1,
  "timestamp_utc": "...",
  "git": {
    "commit": "... or null",
    "branch": "... or null",
    "dirty": true
  },
  "environment": {
    "python_version": "...",
    "platform": "...",
    "mujoco_version": "... or null",
    "numpy_version": "... or null",
    "pandas_version": "... or null"
  },
  "benchmark": {
    "seed": 0,
    "n_designs": 20,
    "n_grasp_trials": 4,
    "tools": ["hammer", "spoon", "knife"],
    "backend": "mujoco_cpu",
    "config_path": "configs/default_eval.yaml",
    "search_space_path": "configs/search_space.yaml",
    "design_dir": "...",
    "results_dir": "...",
    "results_csv": "..."
  }
}
```

Optional but preferred:

- SHA256 hash of config file.
- SHA256 hash of search-space file.
- Number of successful designs.
- Number of failed designs.

Git metadata must fail gracefully when the repo is unavailable or git is not installed.

### 9. Add comparison script wrapper

Add:

```text
scripts/compare_benchmarks.py
```

This should call a package module, for example:

```text
handcdo/compare_benchmarks.py
```

### 10. Implement benchmark comparison module

Create:

```text
handcdo/compare_benchmarks.py
```

Suggested CLI:

```bash
python3 scripts/compare_benchmarks.py \
  --left outputs/baselines/current/results.csv \
  --right outputs/baselines/new_geometry/results.csv \
  --output-dir outputs/baseline_compare \
  --score-column hand_score
```

Required options:

- `--left`
- `--right`
- `--output-dir`
- `--score-column`, default `hand_score`
- `--top-k`, default `5,10`
- Optional `--fail-on-regression`
- Optional `--max-mean-drop`, default disabled
- Optional `--min-spearman`, default disabled

### 11. Comparison behavior

The comparison script must:

- Read two CSV files.
- Require both to contain:
  - `design_id`
  - selected `score-column`
- Join on `design_id` using inner join for common designs.
- Track:
  - number of rows in left
  - number of rows in right
  - number of common designs
  - design IDs only in left
  - design IDs only in right
- Compare selected score column.
- Also compare common per-tool score columns ending in `_best_score` if present in both files.
- Sort top-k by selected score column descending.

Compute at least:

- mean score left/right
- median score left/right
- mean delta: `right - left`
- median delta: `right - left`
- min/max delta
- top-k overlap count and ratio for each k
- Spearman rank correlation if scipy is importable; otherwise set it to `null` and include a warning
- Optional Pearson correlation using pandas/numpy if straightforward

If `n_common == 0`, raise a clear `ValueError`.

For top-k:

- use `effective_k = min(k, n_common)`
- overlap ratio is `len(intersection) / effective_k`

### 12. Comparison outputs

Write:

```text
<output-dir>/comparison_summary.json
<output-dir>/joined_scores.csv
```

`joined_scores.csv` should include:

- `design_id`
- left score
- right score
- delta
- shared tool score columns where available
- failed/error columns if available

Use clear suffixes such as `_left` and `_right`.

### 13. Regression gate behavior

By default, comparison is report-only and exits successfully.

If `--fail-on-regression` is set:

- Fail with nonzero exit code if `mean_delta < -max_mean_drop`, when `--max-mean-drop` is provided.
- Fail with nonzero exit code if `spearman < min_spearman`, when `--min-spearman` is provided and Spearman is available.
- Print a clear message explaining which threshold failed.

### 14. Add tests

Add:

```text
tests/test_benchmark_compare.py
tests/test_baseline_benchmark.py
```

Required comparison tests with tiny synthetic CSVs:

- self-comparison gives mean delta `0.0`.
- top-k overlap is computed correctly.
- `effective_k = min(k, n_common)` works.
- missing `design_id` raises a clear error.
- missing selected score column raises a clear error.
- no common designs raises a clear error.
- designs only in left/right are counted.
- scipy absence does not fail the comparison; Spearman becomes `null` or is omitted with warning.
- per-tool `_best_score` columns are compared when present.

Required benchmark tests:

- metadata writer includes schema version, seed, tools, backend, config path, Python version.
- git metadata helper fails gracefully outside git.
- design loading from an existing design directory is deterministic.
- benchmark output paths are created correctly.
- Avoid long MuJoCo simulation in unit tests.
- If a real MuJoCo smoke test is added, skip gracefully when MuJoCo is unavailable.

### 15. Update README

Add a short section:

```markdown
## Baseline Benchmark
```

Explain:

- Run this before changing contact geometry.
- It freezes the current MuJoCo CPU baseline.
- Compare future geometry modes against this baseline.
- Generated outputs should not be committed.

Include commands:

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

### 16. Validation

Run:

```bash
pytest -q
python3 scripts/run_baseline_benchmark.py --n-designs 2 --n-grasp-trials 1 --tools hammer --seed 0 --backend mujoco_cpu --output-dir outputs/smoke_baseline
python3 scripts/compare_benchmarks.py --left outputs/smoke_baseline/results.csv --right outputs/smoke_baseline/results.csv --output-dir outputs/smoke_baseline_compare
```
