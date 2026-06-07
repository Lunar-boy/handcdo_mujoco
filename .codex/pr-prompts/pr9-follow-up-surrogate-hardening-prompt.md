# PR 9 Follow-up: Harden Surrogate Proposal Workflow

Implement a small follow-up patch on top of PR9 for the CPU-only MuJoCo HandCDO reproduction.

This is a cleanup/hardening PR for the surrogate-assisted candidate proposal workflow. It must remain an additive analysis/engineering change.

Do **not** change:

- MuJoCo evaluation semantics;
- wrench scoring;
- grasp optimization;
- Optuna objective direction;
- design parameter bounds;
- existing result JSON schema;
- existing evaluator behavior;
- baseline evaluation scores.

The goal is to make the PR9 surrogate workflow safer and more useful for repeated HPC experiments.

---

## Existing PR9 context

PR9 already added:

- `handcdo/surrogate/features.py`
- `handcdo/surrogate/train.py`
- `handcdo/surrogate/propose.py`
- `scripts/propose_surrogate_candidates.py`
- README documentation
- tests for training and proposal

The current workflow trains a sklearn surrogate from prior MuJoCo results, samples candidate designs from `DesignSpace`, predicts scores, writes top-k candidates to:

```text
<output-dir>/proposed_candidates.csv
<output-dir>/manifest.json
<output-dir>/proposed_designs/<design_id>/design.json
```

The proposed designs are then evaluated by the existing MuJoCo batch evaluator.

This follow-up patch must fix four issues:

1. prevent stale `proposed_designs` from contaminating later evaluations;
2. make `best_available_score` failure-aware;
3. add CLI modes for train-only, propose-only, and combined train+propose;
4. add surrogate quality diagnostics.

---

# 1. Prevent stale `proposed_designs` contamination

## Problem

Currently, reusing the same `--output-dir` can leave old `proposed_designs/<design_id>/design.json` files in place. Existing evaluators glob all `*/design.json` files under the design directory, so stale candidates can be accidentally evaluated.

This is a real workflow bug.

## Required behavior

Add explicit output overwrite behavior to `propose_candidates`.

Update the API to include:

```python
def propose_candidates(
    model_path: str | Path,
    search_space: str | Path | None,
    n_random: int,
    top_k: int,
    output_dir: str | Path,
    seed: int = 0,
    existing_csv: str | Path | None = None,
    exclude_existing: bool = True,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    ...
```

Behavior:

- `output_dir` itself may exist.
- Before writing `proposed_designs`, inspect:

```text
<output-dir>/proposed_designs
```

- If it exists and is non-empty:
  - if `overwrite=False`, fail clearly with a `FileExistsError`;
  - if `overwrite=True`, delete only `<output-dir>/proposed_designs` and recreate it.
- Do **not** delete the entire `output_dir`, because it may contain model artifacts, diagnostics, logs, or user notes.
- Also handle existing `proposed_candidates.csv` and `manifest.json`:
  - if either exists and `overwrite=False`, fail clearly;
  - if `overwrite=True`, replace them.

Use safe standard-library operations only. `shutil.rmtree` is acceptable, but only on the exact `proposed_designs` directory.

Update `manifest.json` to include:

```json
{
  "overwrite": true_or_false
}
```

## CLI

Add:

```text
--overwrite
--no-overwrite
```

Default should be `--no-overwrite`.

## Tests

Add tests covering:

- rerunning proposal into the same output directory with default overwrite disabled raises `FileExistsError`;
- rerunning with `overwrite=True` removes stale design JSONs and writes exactly the new top-k candidates;
- stale `proposed_designs/<old_id>/design.json` is not present after overwrite;
- existing `proposed_candidates.csv` and `manifest.json` are protected when overwrite is false.

---

# 2. Make `best_available_score` failure-aware

## Problem

The existing multi-fidelity merge may compute `best_available_score` using high → medium → fast score priority without fully checking whether the selected fidelity failed. This can let failed high-fidelity rows with `hand_score_high = 0.0` override successful lower-fidelity scores.

For surrogate training, this target can be noisy or misleading.

## Required behavior

Add a failure-aware score resolver for training.

Implement in `handcdo/surrogate/features.py` or `handcdo/surrogate/train.py`:

```python
def compute_failure_aware_best_score(df: pd.DataFrame) -> pd.Series:
    ...
```

Behavior:

- For each row, choose the highest-fidelity valid score in this order:

```text
hand_score_high
hand_score_medium
hand_score_fast
hand_score
```

- A fidelity score is valid only if:
  - the score column exists;
  - the score is numeric and not NaN;
  - the corresponding failed column does not indicate failure.

Failure columns:

```text
failed_high
failed_medium
failed_fast
failed
```

- Treat these values as truthy failure:
  - Python/CSV booleans: `True`
  - strings case-insensitively equal to: `"true"`, `"1"`, `"yes"`, `"y"`
  - numeric values not equal to zero
- Treat missing failed columns as "not failed" for that fidelity.
- If no valid score exists for a row, return NaN for that row.

Update training target behavior:

- If requested target is exactly `best_available_score`, do **not** blindly use the CSV column.
- Instead, compute the failure-aware best score from available fidelity columns.
- If the CSV also has `best_available_score`, metadata should record both:
  - `target_requested: "best_available_score"`
  - `target_used: "failure_aware_best_available_score"`
  - `csv_best_available_score_present: true_or_false`
- If no fidelity score columns are available to compute it, fail clearly.
- Drop rows where the computed score is NaN.
- Preserve explicit target behavior for `hand_score_high`, `hand_score_medium`, `hand_score_fast`, and `hand_score`, but continue applying per-fidelity failure filtering.

Default target selection:

- Keep default priority as before:

```text
best_available_score
hand_score_high
hand_score_medium
hand_score_fast
hand_score
```

- If `best_available_score` exists or any multi-fidelity score columns exist, prefer the failure-aware best score.
- Metadata should make clear which target was actually used.

## Tests

Add tests covering:

- high failed + medium valid → target uses medium;
- high valid + medium valid → target uses high;
- high failed + medium failed + fast valid → target uses fast;
- all fidelities failed → row is dropped;
- CSV `best_available_score` disagrees with failure-aware score → training uses failure-aware score;
- explicit `target="hand_score_high"` still excludes `failed_high=True` rows;
- boolean/string/numeric failed markers are parsed correctly.

---

# 3. Add CLI modes: train-only, propose-only, train+propose

## Problem

The current CLI performs train + propose in a single run. This is okay for smoke tests but inefficient for real experiments. Users may want to train one surrogate once, then reuse it for multiple proposal seeds/top-k values.

## Required behavior

Refactor `scripts/propose_surrogate_candidates.py` to support three modes:

```text
train-propose
train-only
propose-only
```

### CLI mode argument

Add:

```text
--mode {train-propose,train-only,propose-only}
```

Default:

```text
--mode train-propose
```

### train-propose mode

Behavior should match current PR9 behavior, with the new overwrite and diagnostics behavior.

Required arguments:

- `--results-csv`
- `--output-dir`

Optional:

- `--search-space`
- `--target`
- `--model-type`
- `--n-random`
- `--top-k`
- `--seed`
- `--existing-csv`
- `--exclude-existing` / `--no-exclude-existing`
- `--min-rows`
- `--overwrite`

Training output:

```text
<output-dir>/model/surrogate_model.joblib
<output-dir>/model/surrogate_metadata.json
<output-dir>/model/surrogate_diagnostics.json
```

Proposal output:

```text
<output-dir>/proposed_candidates.csv
<output-dir>/manifest.json
<output-dir>/proposed_designs/<design_id>/design.json
```

### train-only mode

Train the surrogate and write only model artifacts and diagnostics.

Required arguments:

- `--results-csv`
- `--output-dir`

Do not require:

- `--n-random`
- `--top-k`

Do not write:

- `proposed_candidates.csv`
- `manifest.json`
- `proposed_designs/`

Print the model path and diagnostics path.

### propose-only mode

Use an existing model to propose candidates.

Required arguments:

- `--model-path`
- `--output-dir`
- `--n-random`
- `--top-k`

Optional:

- `--search-space`
- `--seed`
- `--existing-csv`
- `--exclude-existing` / `--no-exclude-existing`
- `--overwrite`

Do not require:

- `--results-csv`

Do not train a new model.

If `--search-space` is omitted, use the search space recorded in the model metadata if available; otherwise fall back to the default `DesignSpace()` only if the existing code already allows this safely.

### Argument validation

Fail clearly when:

- `--mode train-only` is used with missing `--results-csv`;
- `--mode propose-only` is used with missing `--model-path`;
- `--mode propose-only` is used without `--n-random` or `--top-k`;
- `--mode train-only` is given proposal-only args that are irrelevant, unless you intentionally ignore them and document that behavior;
- `--top-k > --n-random`.

## Tests

Add CLI tests, preferably using the script through `subprocess.run` with the repo virtual environment where possible, or by testing a `main(argv)` function directly.

Cover:

- train-only writes model + metadata + diagnostics and no proposals;
- propose-only uses an existing model and writes proposals;
- train-propose writes both model artifacts and proposals;
- missing required arguments fail with non-zero exit or `SystemExit`;
- overwrite behavior works through CLI.

---

# 4. Add surrogate quality diagnostics

## Problem

The surrogate workflow currently writes model metadata but not quality diagnostics. Users cannot easily tell whether the surrogate is meaningful enough to guide expensive MuJoCo evaluations.

## Required behavior

Add lightweight deterministic diagnostics during training.

Implement a function such as:

```python
def compute_surrogate_diagnostics(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    min_cv_rows: int = 10,
) -> dict[str, Any]:
    ...
```

Diagnostics should be cheap and CPU-only.

### Required diagnostics fields

Write:

```text
<model-output-dir>/surrogate_diagnostics.json
```

Fields:

```json
{
  "n_rows_used": 0,
  "n_features_raw": 0,
  "target_mean": 0.0,
  "target_std": 0.0,
  "target_min": 0.0,
  "target_max": 0.0,
  "train_r2": 0.0,
  "train_mae": 0.0,
  "cv_enabled": true,
  "cv_folds": 5,
  "cv_r2_mean": 0.0,
  "cv_r2_std": 0.0,
  "cv_mae_mean": 0.0,
  "cv_mae_std": 0.0,
  "diagnostic_warnings": []
}
```

Rules:

- Always compute in-sample `train_r2` and `train_mae`.
- If there are enough valid rows, compute deterministic K-fold CV:
  - use `KFold(shuffle=True, random_state=seed)`;
  - use up to 5 folds;
  - require at least 2 folds;
  - avoid folds that are too small.
- If there are too few rows, set `cv_enabled=false`, CV fields to `None`, and add a warning.
- If target variance is zero or near-zero, handle R² safely and add a warning.
- Avoid crashing diagnostics for small synthetic test data.
- Do not add heavy plotting or SHAP here.

### Metadata

Update `surrogate_metadata.json` to include:

```json
{
  "diagnostics_path": ".../surrogate_diagnostics.json"
}
```

### CLI

All modes that train a model should print the diagnostics path.

### README

Update the surrogate section to explain:

- `surrogate_diagnostics.json` is a sanity check, not physical validation;
- high train R² alone is not enough;
- low or negative CV R² means surrogate ranking may be unreliable;
- proposed candidates must still be evaluated with MuJoCo.

## Tests

Add tests covering:

- diagnostics JSON is written during training;
- diagnostics includes required keys;
- small dataset disables CV gracefully;
- larger synthetic dataset enables CV;
- constant-target dataset does not crash and emits a warning;
- CLI train-only prints or at least writes diagnostics.

---

# 5. README update

Update the surrogate candidate proposal section.

Include examples for all three modes.

## Train only

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

## Propose only

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

## Train and propose

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

## Overwrite

Document that rerunning into the same output directory requires:

```bash
--overwrite
```

Otherwise the command should fail to prevent stale candidate contamination.

---

# 6. Tests and validation

Run:

```bash
pytest -q
```

If the system Python lacks pytest, use the repo environment:

```bash
.venv/bin/pytest -q
```

Also run a tiny CLI smoke test that:

1. trains on a synthetic CSV;
2. runs propose-only from the saved model;
3. writes exactly 3 proposed design JSONs;
4. loads each with `HandDesign.from_json`.

Do not commit generated outputs, model artifacts, logs, caches, or virtual environments.

---

# 7. Expected final summary

In the PR summary, explicitly report:

- stale proposal directory protection implemented;
- failure-aware best score implemented;
- train-only/propose-only/train-propose CLI modes added;
- diagnostics JSON added;
- tests added;
- exact test command and result.

Also state clearly that:

- no MuJoCo evaluation semantics changed;
- no scoring semantics changed;
- no Optuna objective changed;
- no generated output artifacts were committed.
