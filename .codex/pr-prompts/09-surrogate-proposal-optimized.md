# PR 9: Surrogate-Assisted Candidate Proposal

Implement PR 9: add a lightweight surrogate-assisted candidate proposal workflow for the CPU-only MuJoCo HandCDO reproduction.

## Repository context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from arXiv:2604.27557.

Existing code already supports:

- random hand design generation;
- batch MuJoCo design evaluation;
- result JSON collection into CSV;
- Optuna TPE hand optimization;
- fast/medium/high multi-fidelity evaluation;
- Slurm array templates;
- Random Forest + SHAP post-hoc analysis.

Do **not** duplicate MuJoCo simulation, grasp optimization, result collection, design loading, or MJCF generation logic.

Reuse existing abstractions wherever possible:

- `DesignSpace`
- `ParameterSpec`
- `HandDesign`
- `stable_design_id`
- `ensure_dir`
- `write_json`
- existing CSV result schema from `collect_results`
- existing multi-fidelity score columns when available

This PR adds a proposal helper, not a new simulator and not a replacement for Optuna.

---

## Goal

Add a lightweight surrogate workflow that:

1. reads previously collected simulation results;
2. trains a tabular sklearn regressor to predict design score from design parameters;
3. samples many random candidate designs from `DesignSpace`;
4. predicts candidate scores;
5. writes the predicted top-k candidates as normal `design.json` files;
6. leaves all proposed candidates to be re-evaluated by the existing MuJoCo pipeline.

The surrogate is only a cheap proposal filter. It must not be treated as final evaluation.

---

## Research and engineering interpretation

Classify this PR as an analysis/engineering change.

It must not change:

- MuJoCo evaluation semantics;
- wrench scoring;
- grasp optimization;
- design parameter bounds;
- Optuna objective direction;
- result collection semantics;
- existing CLI behavior.

It may add new output files under `outputs/`.

---

## Required changes

### 1. Add package module

Add:

```text
handcdo/surrogate/__init__.py
handcdo/surrogate/train.py
handcdo/surrogate/propose.py
```

Optionally add:

```text
handcdo/surrogate/features.py
```

only if it avoids duplicating feature-selection/preprocessing logic between training and proposal.

---

## 2. Dependencies

`scikit-learn` is already a project dependency. Use sklearn only; do not add neural-network, GPU, Bayesian deep learning, or simulator dependencies.

You may use `joblib` for sklearn model serialization, since it is a standard sklearn dependency. Do not add it explicitly to `pyproject.toml` unless tests prove it is unavailable.

Do not add heavy dependencies.

---

## 3. Feature schema

Use only design-parameter columns from the active `DesignSpace`.

Do **not** train on metadata or score-leakage columns, including:

- `design_id`
- `hand_score`
- `hand_score_fast`
- `hand_score_medium`
- `hand_score_high`
- `best_available_score`
- any column ending in `_best_score`
- `failed`
- `failed_*`
- `error`
- `error_*`
- `fidelity`
- `backend`
- `config_path`
- `n_grasp_trials`
- `sampler`
- `seed`
- suffixed metadata columns from multi-fidelity merges

The model input feature columns must be exactly the names in `space.specs` that exist in the training CSV.

Categorical parameters must be encoded with `OneHotEncoder(handle_unknown="ignore")`.

Numeric parameters should be converted to numeric and imputed robustly.

Prefer a sklearn `Pipeline` / `ColumnTransformer` so the same preprocessing is used at proposal time.

---

## 4. Target column selection

Implement target selection with an explicit override:

```python
def resolve_target_column(df, requested_target: str | None = None) -> str:
    ...
```

Behavior:

- If `requested_target` is provided, require that column.
- Otherwise choose the first available column from:

```text
best_available_score
hand_score_high
hand_score_medium
hand_score_fast
hand_score
```

- Convert the target to numeric.
- Drop rows with NaN target.

Failed-row handling:

- For plain `hand_score`, exclude rows where `failed` is truthy when the `failed` column exists.
- For `hand_score_fast`, exclude `failed_fast` if it exists.
- For `hand_score_medium`, exclude `failed_medium` if it exists.
- For `hand_score_high`, exclude `failed_high` if it exists.
- For `best_available_score`, drop NaN scores and exclude rows that are clearly failed at all available fidelity levels when such failed columns exist.
- Fail clearly if no valid training rows remain.

---

## 5. Training API

Implement:

```python
def train_surrogate(
    results_csv: str | Path,
    output_dir: str | Path,
    search_space: str | Path | None = "configs/search_space.yaml",
    target: str | None = None,
    model_type: str = "random_forest",
    seed: int = 0,
    min_rows: int = 5,
) -> Path:
    ...
```

Behavior:

- Read `results_csv` with pandas.
- Load `DesignSpace.from_yaml(search_space)` if provided, otherwise `DesignSpace()`.
- Select the target column using `resolve_target_column`.
- Filter failed/invalid rows as described above.
- Build features using only active design-space parameter columns.
- Require at least `min_rows` valid rows.
- Support:
  - `model_type="random_forest"` using `RandomForestRegressor`
  - `model_type="extra_trees"` using `ExtraTreesRegressor`
- Use deterministic `random_state=seed`.
- Save a serialized sklearn pipeline/bundle under:

```text
<output_dir>/surrogate_model.joblib
```

- Save metadata under:

```text
<output_dir>/surrogate_metadata.json
```

Metadata must include:

- `results_csv`
- `target`
- `model_type`
- `seed`
- `n_rows_total`
- `n_rows_used`
- `feature_columns`
- `categorical_columns`
- `numeric_columns`
- `search_space`
- `sklearn_model_class`

Return the model path.

---

## 6. Candidate proposal API

Implement:

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
) -> list[dict[str, Any]]:
    ...
```

Behavior:

- Validate `n_random > 0`.
- Validate `top_k > 0`.
- Validate `top_k <= n_random`.
- Load the serialized model/pipeline and metadata.
- Load the active `DesignSpace`.
- Sample `n_random` candidate designs using a single `np.random.default_rng(seed)`.
- Use `DesignSpace.sample(rng=rng)` and `HandDesign.to_json()`; do not hand-roll design IDs or JSON schema.
- De-duplicate candidate `design_id`s deterministically.
- If `exclude_existing=True`, exclude any design IDs from:
  - `existing_csv`, if provided;
  - otherwise the training CSV recorded in metadata, if it contains `design_id`.
- Convert candidate parameter dictionaries to the same feature schema used during training.
- Predict surrogate scores.
- Sort by:
  1. predicted score descending;
  2. `design_id` ascending for deterministic tie-breaking.
- Write exactly `top_k` selected candidates unless there are not enough unique non-existing candidates, in which case fail clearly.

Write:

```text
<output_dir>/proposed_candidates.csv
<output_dir>/manifest.json
<output_dir>/proposed_designs/<design_id>/design.json
```

`proposed_candidates.csv` must include:

- `rank`
- `design_id`
- `predicted_score`
- all design parameter columns

`manifest.json` must include:

- `model_path`
- `search_space`
- `seed`
- `n_random`
- `top_k`
- `exclude_existing`
- `selected_design_ids`

---

## 7. CLI wrapper

Add:

```text
scripts/propose_surrogate_candidates.py
```

This should be a thin CLI wrapper around `handcdo.surrogate.train` and `handcdo.surrogate.propose`.

Suggested usage:

```bash
python3 scripts/propose_surrogate_candidates.py \
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

Supported args:

- `--results-csv`
- `--search-space`
- `--target`, optional
- `--model-type`, choices `random_forest`, `extra_trees`
- `--n-random`
- `--top-k`
- `--output-dir`
- `--seed`
- `--existing-csv`, optional
- `--exclude-existing` / `--no-exclude-existing`
- `--min-rows`, default `5`

The CLI should:

1. train the surrogate into `<output-dir>/model/`;
2. propose candidates into `<output-dir>/`;
3. print the output CSV path and proposed design directory.

The proposed design directory should be directly usable as `--design-dir` for existing evaluation scripts.

Example validation after proposal:

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

---

## 8. README update

Add a short section:

```markdown
## Surrogate candidate proposal
```

Explain:

- The surrogate is trained from already evaluated MuJoCo results.
- It predicts candidate scores only for cheap ranking.
- Proposed candidates must still be evaluated with MuJoCo.
- The workflow is useful after collecting enough random, Optuna, or multi-fidelity simulation data.
- Final reporting should still use high-fidelity MuJoCo scores when available.

Include the CLI example and the follow-up `evaluate_design_batch.py` example.

Do not claim that surrogate predictions are physically validated.

---

## 9. Tests

Add focused tests that do not require expensive MuJoCo simulation.

Suggested files:

```text
tests/test_surrogate_train.py
tests/test_surrogate_propose.py
```

Cover:

### Training

- synthetic CSV with valid design-space columns trains successfully;
- model file and metadata file are written;
- default target selection prefers `best_available_score` when available;
- explicit `--target hand_score` works;
- failed rows are excluded;
- NaN/non-numeric targets are dropped;
- missing target fails clearly;
- too few valid rows fails clearly.

### Proposal

- proposal writes exactly `top_k` rows;
- each selected candidate has a `proposed_designs/<design_id>/design.json`;
- fixed seed produces deterministic `proposed_candidates.csv`;
- candidates are sorted by predicted score descending and design_id ascending for ties;
- existing design IDs are excluded when requested;
- invalid `top_k > n_random` fails clearly.

### Integration shape

- proposed `design.json` files can be loaded with `HandDesign.from_json`;
- `proposed_candidates.csv` includes `rank`, `design_id`, `predicted_score`, and all active design parameter columns.

---

## 10. Validation

Run:

```bash
pytest -q
```

Also run a lightweight CLI smoke test using a tiny synthetic CSV if practical.

Do not commit generated outputs, model artifacts, logs, caches, or virtual environments.

---

## Out of scope

Do not implement:

- neural networks;
- Gaussian-process Bayesian optimization;
- uncertainty acquisition functions unless explicitly requested later;
- active learning loops;
- replacement of Optuna/TPE;
- changes to MuJoCo simulation;
- changes to wrench scoring;
- changes to grasp optimization;
- Isaac Sim;
- ROS;
- GPU-only dependencies;
- MJX or MuJoCo-Warp;
- physical robot evaluation;
- fabrication workflow.
