# PR 7: Multi-Fidelity Evaluation Pipeline

Implement PR 7: add a deterministic multi-fidelity evaluation pipeline for the CPU-only MuJoCo HandCDO reproduction.

## Repository context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from arXiv:2604.27557.

Existing code already supports:

- random hand design generation;
- batch design evaluation;
- result JSON collection into CSV;
- Optuna-based hand optimization;
- typed geometry configuration;
- fingertip pads;
- palm pad grids;
- optional hybrid tool geometry with primitive fallback.

Do **not** duplicate MJCF generation, MuJoCo simulation, grasp optimization, result collection, or design loading logic.

Reuse existing abstractions wherever possible:

- `HandDesign`
- `EvaluationConfig`
- `GeometryConfig`
- `evaluate_design`
- `collect_results`
- `read_yaml`
- `write_json`
- `ensure_dir`

The goal of this PR is to add a reliable experiment protocol, not a new simulator.

---

## Goal

Add a three-stage multi-fidelity workflow:

1. **Fast evaluation**
   - low simulation budget;
   - primitive/capsule geometry;
   - low grasp optimization budget;
   - used for broad screening.

2. **Medium re-evaluation**
   - default-ish simulation budget;
   - fingertip pads and palm pad grid;
   - medium grasp optimization budget;
   - used for re-ranking top fast candidates.

3. **High re-evaluation**
   - larger simulation budget;
   - fingertip pads and higher-resolution palm pad grid;
   - optional hybrid tool geometry with primitive fallback;
   - higher grasp optimization budget;
   - used for final ranking.

Definitions:

- "fidelity" means a named evaluation configuration, not a surrogate model.
- This PR must not implement surrogate modeling, active learning, MJX, Warp, GPU simulation, Isaac Sim, real mesh assets, real robot evaluation, or fabrication logic.
- All outputs must be deterministic for fixed design files, config, tools, backend, and seed.

---

## Required changes

### 1. Add standalone evaluation configs

Add:

- `configs/eval_fast.yaml`
- `configs/eval_medium.yaml`
- `configs/eval_high.yaml`

Each config must include:

- `simulation`
- `wrench`
- `grasp`
- `output`
- `geometry`

Use the following starting values unless they conflict with existing schema constraints.

#### `configs/eval_fast.yaml`

```yaml
simulation:
  timestep: 0.002
  settle_steps: 120
  close_steps: 180
  wrench_steps: 120
  control_stiffness: 6.0

wrench:
  translation_threshold: 0.045
  rotation_threshold_rad: 0.55

grasp:
  n_trials: 2
  sampler: random

output:
  save_mjcf: true

geometry:
  finger:
    mode: capsule
    fingertip_pad_enabled: false

  palm:
    mode: box_pads

  tool:
    mode: primitive
```

#### `configs/eval_medium.yaml`

```yaml
simulation:
  timestep: 0.002
  settle_steps: 250
  close_steps: 350
  wrench_steps: 250
  control_stiffness: 6.0

wrench:
  translation_threshold: 0.045
  rotation_threshold_rad: 0.55

grasp:
  n_trials: 4
  sampler: tpe

output:
  save_mjcf: true

geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
    fingertip_pad_thickness: 0.004
    fingertip_pad_friction: [1.4, 0.03, 0.003]

  palm:
    mode: pad_grid
    pad_resolution: 3
    pad_friction: [1.4, 0.02, 0.002]
    max_num_pad_geoms: 16

  tool:
    mode: primitive
```

#### `configs/eval_high.yaml`

```yaml
simulation:
  timestep: 0.002
  settle_steps: 350
  close_steps: 450
  wrench_steps: 400
  control_stiffness: 6.0

wrench:
  translation_threshold: 0.045
  rotation_threshold_rad: 0.55

grasp:
  n_trials: 8
  sampler: tpe

output:
  save_mjcf: true

geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
    fingertip_pad_thickness: 0.004
    fingertip_pad_friction: [1.4, 0.03, 0.003]

  palm:
    mode: pad_grid
    pad_resolution: 4
    pad_friction: [1.4, 0.02, 0.002]
    max_num_pad_geoms: 16

  tool:
    mode: hybrid
    collision_margin: 0.001
```

If the existing schema uses slightly different field names, adapt minimally and keep the same semantics.

---

### 2. Add `handcdo/multifidelity.py`

Add a testable package module:

```text
handcdo/multifidelity.py
```

Implement the functions below.

---

#### `select_top_designs`

```python
def select_top_designs(
    input_csv: str | Path,
    top_k: int,
    output_design_ids: str | Path,
    score_column: str = "hand_score",
    include_failed: bool = False,
) -> list[str]:
    ...
```

Behavior:

- Read the input CSV.
- Require `design_id` and the selected score column.
- By default, exclude rows with `failed == True`.
- Convert the selected score column to numeric.
- Drop rows with NaN scores.
- Sort by score descending and `design_id` ascending for deterministic tie-breaking.
- If `top_k` exceeds the valid row count, write all valid rows.
- Write one design id per line.
- Ensure the output file is newline-terminated.
- Return the selected design ids.
- Fail clearly if required columns are missing.

---

#### `load_design_ids`

```python
def load_design_ids(path: str | Path) -> list[str]:
    ...
```

Behavior:

- Read newline-separated design ids.
- Strip whitespace.
- Ignore empty lines.
- Preserve order.
- Reject duplicate IDs with a clear error.

---

#### `reevaluate_designs`

Recommended signature:

```python
def reevaluate_designs(
    *,
    design_dir: str | Path,
    design_ids: list[str],
    results_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    fidelity: str,
    tools: list[str],
    backend: str = "mujoco_cpu",
    seed: int = 0,
    n_grasp_trials: int | None = None,
    sampler: str | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> list[dict[str, Any]]:
    ...
```

Behavior:

- For each design id, load:

```text
<design_dir>/<design_id>/design.json
```

- If a design JSON is missing, fail clearly with `FileNotFoundError` or `ValueError` mentioning the missing design id.
- Read config via `read_yaml`.
- Build `EvaluationConfig.from_dict(config_data)`.
- Build `GeometryConfig.from_dict(config_data)`.
- Use `config_data["grasp"]["n_trials"]` unless `n_grasp_trials` override is provided.
- Use `config_data["grasp"]["sampler"]` unless `sampler` override is provided.
- Call existing `evaluate_design`.
- Do not reimplement simulation or scoring.
- Add these top-level scalar fields to every result payload:
  - `fidelity`
  - `backend`
  - `config_path`
  - `n_grasp_trials`
  - `sampler`
  - `seed`
- Re-write the JSON result after adding metadata, because `evaluate_design` writes the initial payload before metadata exists.
- Return all payloads.

Important:

- Keep the output result schema backward-compatible.
- Do not introduce a new result format.
- If existing `evaluate_design` returns only a path rather than a payload, read the written JSON, add metadata, and write it back.

---

#### `merge_multifidelity_results`

```python
def merge_multifidelity_results(
    inputs: Mapping[str, str | Path],
    output_csv: str | Path,
):
    ...
```

Behavior:

- Accept a mapping from fidelity name to CSV path.
- Expected fidelity names are usually `fast`, `medium`, and `high`, but the function should not be hard-coded to only these names.
- Outer-join by `design_id`.
- Preserve score columns with deterministic suffixes:
  - `hand_score_fast`
  - `hand_score_medium`
  - `hand_score_high`
- Preserve per-fidelity fields where present:
  - `failed_<fidelity>`
  - `error_<fidelity>`
  - `backend_<fidelity>`
  - `config_path_<fidelity>`
  - `n_grasp_trials_<fidelity>`
  - `sampler_<fidelity>`
  - `seed_<fidelity>`
- Preserve design parameter columns only once.
- If a parameter column appears in multiple fidelity CSVs, use the highest fidelity available in this priority order:
  - `high`
  - `medium`
  - `fast`
  - then any other fidelity in lexical order.
- Add `best_available_score`, using the highest fidelity score available in this priority order:
  - `high`
  - `medium`
  - `fast`
  - then any other fidelity in lexical order.
- Write the merged CSV.
- Return the merged rows or DataFrame.

---

### 3. Add CLI wrappers

Add thin CLI wrappers around `handcdo.multifidelity`:

- `scripts/select_top_designs.py`
- `scripts/reevaluate_designs.py`
- `scripts/merge_multifidelity_results.py`

These scripts should contain argument parsing only and should delegate implementation to `handcdo.multifidelity`.

---

#### Required CLI: `select_top_designs.py`

Example:

```bash
python3 scripts/select_top_designs.py \
  --input-csv outputs/fast/results.csv \
  --top-k 100 \
  --output-design-ids outputs/medium/design_ids.txt
```

Supported args:

- `--input-csv`
- `--top-k`
- `--output-design-ids`
- `--score-column`, default `hand_score`
- `--include-failed`, default false

---

#### Required CLI: `reevaluate_designs.py`

Example:

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

Supported args:

- `--design-dir`
- `--design-ids`
- `--results-dir`
- `--output-dir`
- `--config`
- `--fidelity`
- `--tools`, comma-separated
- `--backend`, default `mujoco_cpu`
- `--seed`, default `0`
- `--n-grasp-trials`, optional override
- `--sampler`, optional override
- `--tool-assets-dir`, default `assets/tools`

---

#### Required CLI: `merge_multifidelity_results.py`

Example:

```bash
python3 scripts/merge_multifidelity_results.py \
  --fast-csv outputs/fast/results.csv \
  --medium-csv outputs/medium/results.csv \
  --high-csv outputs/high/results.csv \
  --output-csv outputs/multifidelity_results.csv
```

Supported args:

- `--fast-csv`, optional
- `--medium-csv`, optional
- `--high-csv`, optional
- `--input`, repeatable custom mapping in the form `fidelity=path/to/results.csv`
- `--output-csv`

At least one input CSV must be provided.

---

### 4. Update result collection

Update `handcdo.collect_results.flatten_result()` so it preserves top-level scalar metadata fields when present:

- `fidelity`
- `backend`
- `config_path`
- `n_grasp_trials`
- `sampler`
- `seed`

Keep backward compatibility with existing result JSON files that do not contain these fields.

Do not break existing CSV output columns.

---

### 5. Optional compatibility improvement

If safe and low-risk, update batch evaluation so it honors `grasp.sampler` from config when calling `evaluate_design`.

Preserve current default behavior when `grasp.sampler` is absent.

---

### 6. Add tests

Add focused unit tests that do not require expensive MuJoCo simulation.

---

#### `tests/test_multifidelity_select.py`

Cover:

- selecting top-k from a synthetic CSV;
- ignoring failed rows by default;
- including failed rows when `include_failed=True`;
- deterministic tie-breaking by `design_id`;
- clear failure when required columns are missing;
- newline-terminated output file.

---

#### `tests/test_multifidelity_merge.py`

Cover:

- merging synthetic fast, medium, and high CSVs;
- preserving:
  - `hand_score_fast`
  - `hand_score_medium`
  - `hand_score_high`
- computing `best_available_score`;
- preserving metadata suffixes;
- preserving parameter columns once;
- handling missing fidelity CSVs gracefully.

---

#### `tests/test_multifidelity_reevaluate.py`

Cover:

- `load_design_ids` strips whitespace and rejects duplicates;
- missing design JSON fails clearly;
- metadata fields are added to result payloads;
- use monkeypatch or fake `evaluate_design` where possible to avoid expensive MuJoCo simulation.

All existing tests must continue to pass.

---

### 7. Update README

Add a section:

```markdown
## Multi-fidelity workflow
```

Include this staged example:

```bash
# 1. Generate designs
python3 scripts/generate_designs.py \
  --n-designs 500 \
  --output-dir outputs/designs \
  --seed 10

# 2. Fast evaluation
python3 scripts/evaluate_design_batch.py \
  --task-id 0 \
  --designs-per-task 500 \
  --design-dir outputs/designs \
  --results-dir outputs/fast/results \
  --config configs/eval_fast.yaml

python3 scripts/collect_results.py \
  --results-dir outputs/fast/results \
  --output-csv outputs/fast/results.csv

# 3. Select top candidates for medium fidelity
python3 scripts/select_top_designs.py \
  --input-csv outputs/fast/results.csv \
  --top-k 100 \
  --output-design-ids outputs/medium/design_ids.txt

# 4. Medium re-evaluation
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

python3 scripts/collect_results.py \
  --results-dir outputs/medium/results \
  --output-csv outputs/medium/results.csv

# 5. Select top candidates for high fidelity
python3 scripts/select_top_designs.py \
  --input-csv outputs/medium/results.csv \
  --top-k 20 \
  --output-design-ids outputs/high/design_ids.txt

# 6. High re-evaluation
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

python3 scripts/collect_results.py \
  --results-dir outputs/high/results \
  --output-csv outputs/high/results.csv

# 7. Merge all fidelity levels
python3 scripts/merge_multifidelity_results.py \
  --fast-csv outputs/fast/results.csv \
  --medium-csv outputs/medium/results.csv \
  --high-csv outputs/high/results.csv \
  --output-csv outputs/multifidelity_results.csv
```

Also mention:

- fast fidelity is intended for broad screening only;
- final ranking should be based on high-fidelity scores when available;
- merged CSV keeps separate fidelity scores to make ranking drift visible.

---

## Validation

Run:

```bash
pytest -q
```

Also verify at least one CLI path using a synthetic CSV:

```bash
python3 scripts/select_top_designs.py \
  --input-csv <synthetic-or-existing-csv> \
  --top-k 5 \
  --output-design-ids outputs/top5.txt
```

If existing lightweight generated designs are available, also smoke-test:

```bash
python3 scripts/reevaluate_designs.py \
  --design-dir outputs/designs \
  --design-ids outputs/top5.txt \
  --results-dir outputs/smoke/results \
  --output-dir outputs/smoke \
  --config configs/eval_fast.yaml \
  --fidelity fast \
  --tools hammer \
  --backend mujoco_cpu \
  --seed 0
```

Do not commit generated outputs.

---

## Out of scope

Do not implement:

- surrogate modeling;
- active learning acquisition functions;
- MJX;
- Warp;
- GPU simulation;
- Isaac Sim;
- mesh asset generation;
- physical robot evaluation;
- fabrication workflow;
- generated experiment outputs.

The PR should remain small, deterministic, testable, and compatible with the existing CPU-only MuJoCo infrastructure.
