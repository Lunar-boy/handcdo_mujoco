Implement PR 3: add baseline benchmark and regression comparison scripts.

Current repository facts:
- README documents generation, batch evaluation, collection, Optuna, and SHAP workflows.
- Tests already cover design-space bounds, deterministic JSON round trips, MJCF loadability when MuJoCo is installed, wrench-score bounds, and robust result collection.
- Current outputs are design JSON, MJCF model XML, partial JSON results, and collected CSV.

Goal:
Add a reproducible benchmark workflow that freezes the current MuJoCo CPU baseline and lets future geometry changes be compared against it.

Required changes:
1. Add script:
   - `scripts/run_baseline_benchmark.py`

2. Script behavior:
   - Generate or load a fixed set of designs.
   - Evaluate fixed tools with fixed seeds.
   - Collect result JSON into CSV.
   - Save a metadata JSON with:
     - git commit if available
     - timestamp
     - seed
     - n_designs
     - n_grasp_trials
     - tools
     - backend
     - config path
     - Python version
     - optional MuJoCo version if import succeeds

3. Suggested CLI:
```bash
python3 scripts/run_baseline_benchmark.py \
  --n-designs 20 \
  --n-grasp-trials 4 \
  --tools hammer,spoon,knife \
  --seed 0 \
  --backend mujoco_cpu \
  --config configs/default_eval.yaml \
  --output-dir outputs/baselines/current
```

4. Add script:
   - `scripts/compare_benchmarks.py`

5. `compare_benchmarks.py` behavior:
   - Input two CSV files.
   - Join by `design_id`.
   - Compare hand scores.
   - Compute:
     - mean score A/B
     - median score A/B
     - score delta mean
     - top-k overlap for k=5,10 if possible
     - Spearman rank correlation if scipy is installed, otherwise skip gracefully
   - Write:
     - `comparison_summary.json`
     - `joined_scores.csv`

6. Add tests:
   - `tests/test_benchmark_compare.py`
   - Use tiny synthetic CSVs.
   - Test top-k overlap.
   - Test missing columns produce clear error.
   - Test scipy absence does not fail the whole script.

7. Update README:
   - Add a short "Baseline Benchmark" section.
   - Explain that this is used before changing contact geometry.

Out of scope:
- Do not change evaluator.
- Do not change geometry.
- Do not implement multi-fidelity pipeline yet.

Validation:
```bash
pytest -q
python3 scripts/run_baseline_benchmark.py --n-designs 2 --n-grasp-trials 1 --tools hammer --seed 0 --backend mujoco_cpu --output-dir outputs/smoke_baseline
python3 scripts/compare_benchmarks.py --left outputs/smoke_baseline/results.csv --right outputs/smoke_baseline/results.csv --output-dir outputs/smoke_baseline_compare
```
