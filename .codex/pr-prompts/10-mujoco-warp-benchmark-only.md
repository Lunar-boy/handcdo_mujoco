# PR 10: MuJoCo Warp Benchmark Only

Implement PR 10: add an optional **MuJoCo Warp / MJWarp benchmark scaffold only**.

This PR must not integrate MuJoCo Warp into the normal HandCDO optimization pipeline. It exists only to test whether the current repo-generated MJCF scenes are compatible with MuJoCo Warp and whether H100-class GPU throughput is promising enough to justify a later experimental backend PR.

---

## Repository context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from arXiv:2604.27557.

Existing code already supports:

- deterministic hand design generation;
- `HandDesign` / `DesignSpace` search-space abstractions;
- MJCF generation from hand designs;
- CPU MuJoCo grasp and wrench evaluation;
- batch evaluation on CPU;
- result JSON collection into CSV;
- Optuna TPE hand optimization;
- fast/medium/high multi-fidelity evaluation;
- Slurm array templates;
- Random Forest + SHAP analysis;
- surrogate-assisted candidate proposal.

Do **not** duplicate design loading, MJCF generation, CPU MuJoCo simulation, grasp sampling, wrench scoring, result collection, or backend registry logic.

Reuse existing abstractions wherever possible:

- `DesignSpace`
- `HandDesign`
- `write_design_model` / existing MJCF generation helpers
- `EvaluationConfig`
- `GeometryConfig`
- existing tool names: `hammer`, `spoon`, `knife`
- `ensure_dir`
- `write_json`

---

## Naming clarification

Use the precise name **MuJoCo Warp** or **MJWarp** in code comments and documentation.

Prefer the standalone package:

```python
import mujoco_warp as mjw
```

Do not call this PR a JAX/MJX differentiable-physics PR.

Do not require JAX.

Do not implement automatic differentiation.

Do not use the name `mjx_warp` for the main module or script unless the implementation actually goes through the `mujoco.mjx` API. For this PR, prefer:

```text
handcdo/benchmarks/mujoco_warp.py
scripts/benchmark_mujoco_warp.py
```

---

## Goal

Add a benchmark-only workflow that can answer:

1. Is MuJoCo Warp importable in the current environment?
2. Can the current repo generate an MJCF model suitable for a benchmark scene?
3. Can CPU MuJoCo load the same MJCF model?
4. Can MuJoCo Warp transfer the model/data to the device?
5. What is the CPU MuJoCo timing for a fixed scene?
6. If MuJoCo Warp is available, what is the GPU timing for the same fixed scene?
7. How do `nworld`, `nconmax`, `naconmax`, and `njmax` affect success/failure and throughput?
8. Which unsupported-feature or capacity errors occur?

This PR must not claim physical validation, score equivalence, or speedup unless the benchmark actually measures it.

---

## Required changes

Add:

```text
handcdo/benchmarks/__init__.py
handcdo/benchmarks/mujoco_warp.py
scripts/benchmark_mujoco_warp.py
tests/test_mujoco_warp_benchmark.py
```

Optional README update:

```markdown
## Optional MuJoCo Warp benchmark
```

Do **not** add:

```text
handcdo/backends/mujoco_warp.py
```

Do **not** modify:

```text
handcdo/backends/registry.py
handcdo/backends/__init__.py
handcdo/optimize_hand.py
handcdo/optimize_grasp.py
handcdo/slurm_batch.py
```

Do **not** add `mujoco_warp` to normal `--backend` choices.

---

## Dependency policy

Default installation and default tests must remain CPU-only.

Do not add GPU-only packages to `[project.dependencies]`.

Allowed optional extra:

```toml
[project.optional-dependencies]
warp = [
  "mujoco-warp",
]
```

Only add the optional extra if it is needed and does not break CPU-only environments.

The benchmark must fail gracefully if optional dependencies are absent.

Implement an availability helper:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class WarpAvailability:
    available: bool
    reason: str | None
    package: str | None
    version: str | None


def check_warp_available() -> WarpAvailability:
    ...
```

`check_warp_available()` must not raise for ordinary missing-dependency cases.

---

## Benchmark scope

This PR may implement:

- optional dependency import checks;
- deterministic benchmark design generation;
- MJCF generation for one hand/tool scene;
- CPU MuJoCo model loading;
- CPU MuJoCo short rollout timing;
- optional MuJoCo Warp model/data creation;
- optional MuJoCo Warp stepping;
- wall-clock timing;
- steps/sec reporting;
- structured failure and unsupported-feature reporting;
- output JSON/CSV summaries.

This PR must not implement:

- a production MuJoCo Warp backend;
- replacement of `evaluate_design`;
- replacement of `optimize_grasp_for_tool`;
- replacement of wrench scoring;
- score correlation claims;
- JAX autodiff;
- differentiable physics;
- policy learning;
- RL environment creation;
- Isaac Sim integration;
- ROS;
- GPU Slurm integration;
- physical robot evaluation;
- fabrication workflow.

---

## Benchmark input design

Support two ways to choose the fixed design.

### 1. User-provided design JSON

```bash
--design-json path/to/design.json
```

### 2. Deterministic generated design

```bash
--search-space configs/search_space.yaml
--seed 0
```

If `--design-json` is not provided, load the search space and sample exactly one design deterministically.

Write the selected/generated design and generated MJCF under the benchmark output directory.

Do not commit generated outputs.

---

## CLI

Add:

```text
scripts/benchmark_mujoco_warp.py
```

The script should be a thin wrapper around `handcdo.benchmarks.mujoco_warp`.

Suggested CLI:

```bash
python3 scripts/benchmark_mujoco_warp.py \
  --design-json outputs/designs/<design_id>/design.json \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --output-dir outputs/mujoco_warp_benchmark \
  --seed 0 \
  --cpu-repeats 3 \
  --warp-repeats 3 \
  --warmup-steps 10 \
  --steps 100 \
  --nworld 64 \
  --nconmax 64 \
  --njmax 128
```

Required args:

- `--output-dir`

Optional args:

- `--design-json`
- `--search-space`, default `configs/search_space.yaml`
- `--config`, default `configs/eval_fast.yaml`
- `--tool`, default `hammer`
- `--seed`, default `0`
- `--cpu-repeats`, default `3`
- `--warp-repeats`, default `3`
- `--warmup-steps`, default `10`
- `--steps`, default `100`
- `--nworld`, default `64`
- `--nconmax`, default `64`
- `--naconmax`, optional
- `--njmax`, default `128`
- `--require-warp`, action flag
- `--skip-cpu`, action flag
- `--overwrite`, action flag

CLI behavior:

- `--help` must work without importing GPU dependencies.
- Missing MuJoCo Warp without `--require-warp` must print a clear message, write availability metadata, and exit successfully after the CPU benchmark if CPU benchmark is enabled.
- Missing MuJoCo Warp with `--require-warp` must exit nonzero with a clear error.
- CPU benchmark must remain runnable on a normal CPU-only environment.
- Invalid numeric arguments such as `steps <= 0`, `nworld <= 0`, `repeats <= 0`, `nconmax <= 0`, or `njmax <= 0` must fail clearly.

---

## Implementation details

Implement focused functions similar to:

```python
@dataclass(frozen=True)
class WarpBenchmarkConfig:
    output_dir: Path
    design_json: Path | None = None
    search_space: Path = Path("configs/search_space.yaml")
    config_path: Path = Path("configs/eval_fast.yaml")
    tool: str = "hammer"
    seed: int = 0
    cpu_repeats: int = 3
    warp_repeats: int = 3
    warmup_steps: int = 10
    steps: int = 100
    nworld: int = 64
    nconmax: int | None = 64
    naconmax: int | None = None
    njmax: int = 128
    require_warp: bool = False
    skip_cpu: bool = False
    overwrite: bool = False
```

```python
def build_fixed_benchmark_model(config: WarpBenchmarkConfig) -> tuple[HandDesign, Path]:
    ...
```

```python
def run_cpu_mujoco_timing(
    mjcf_path: str | Path,
    steps: int,
    warmup_steps: int,
    repeats: int,
) -> dict[str, Any]:
    ...
```

```python
def run_warp_timing(
    mjcf_path: str | Path,
    steps: int,
    warmup_steps: int,
    repeats: int,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
) -> dict[str, Any]:
    ...
```

```python
def run_benchmark(config: WarpBenchmarkConfig) -> dict[str, Any]:
    ...
```

Use `time.perf_counter()` for timing.

For CPU timing, load the generated MJCF with CPU MuJoCo and repeatedly call `mujoco.mj_step`.

For Warp timing, load the same MJCF with CPU MuJoCo, transfer model/data to MuJoCo Warp, create batched data with `nworld`, and repeatedly call `mjw.step`.

For Warp, separate these stages in the result payload:

- import availability;
- model loading success;
- device transfer success;
- data creation success;
- warmup success;
- measured stepping success;
- total measured seconds;
- `steps`;
- `nworld`;
- total simulated world-steps;
- steps/sec total;
- steps/sec per world if meaningful.

Catch exceptions and record them as structured failure payloads.

Do not hide unsupported-feature errors.

---

## Score comparison policy

Do not implement a fake grasp score.

Do not claim score correlation by default.

For this PR, timing and compatibility are sufficient.

If a comparable score is implemented later, it must:

- reuse the same fixed design;
- reuse the same fixed tool;
- reuse the same fixed grasp samples;
- document whether the same disturbance protocol is used;
- output CPU score and Warp score separately;
- mark the comparison as experimental.

---

## Output files

Write:

```text
<output-dir>/availability.json
<output-dir>/benchmark_results.json
<output-dir>/benchmark_results.csv
<output-dir>/design/design.json
<output-dir>/model/model.xml
```

`availability.json` should include:

- `warp_available`
- `reason`
- `package`
- `version`
- Python version
- platform
- CUDA/GPU device info if safely available without raising

`benchmark_results.json` should include:

- input arguments;
- generated or loaded design id;
- MJCF path;
- CPU timing result;
- Warp timing result or skip reason;
- failure count;
- exceptions as strings;
- timestamp.

`benchmark_results.csv` should contain one row per measured backend:

- `backend`
- `available`
- `success`
- `seconds_mean`
- `seconds_std`
- `steps`
- `nworld`
- `total_sim_steps`
- `steps_per_second_total`
- `steps_per_second_per_world`
- `failure_count`
- `error`

Do not write outputs outside `--output-dir`.

Do not commit generated benchmark outputs.

---

## README update

Add a short section:

```markdown
## Optional MuJoCo Warp benchmark
```

Explain:

- The repository remains CPU-only by default.
- MuJoCo Warp is optional and intended for GPU throughput experiments.
- The benchmark is not a production backend.
- The benchmark does not replace CPU MuJoCo scoring.
- The benchmark is useful only after CPU regression tests are stable.
- Default pytest must not require CUDA, JAX, MJX, or MuJoCo Warp.

Include install example:

```bash
python3 -m pip install -e ".[test]"
python3 -m pip install -e ".[warp]"
```

Include smoke example:

```bash
python3 scripts/benchmark_mujoco_warp.py --help
python3 scripts/benchmark_mujoco_warp.py \
  --output-dir outputs/mujoco_warp_benchmark \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 8
```

Do not claim speedup unless measured.

---

## Tests

Add focused tests that do not require GPU or MuJoCo Warp.

Suggested file:

```text
tests/test_mujoco_warp_benchmark.py
```

Cover:

1. `check_warp_available()` does not raise when MuJoCo Warp is absent.
2. Importing `handcdo.benchmarks.mujoco_warp` works without GPU dependencies.
3. `python3 scripts/benchmark_mujoco_warp.py --help` works.
4. Missing MuJoCo Warp without `--require-warp` exits successfully and writes `availability.json`.
5. Missing MuJoCo Warp with `--require-warp` exits nonzero.
6. Benchmark config validation rejects invalid `steps <= 0`.
7. Benchmark config validation rejects invalid `nworld <= 0`.
8. Output schema helpers produce rows with required CSV columns.
9. No changes are made to backend registry choices.

Tests must not require:

- H100;
- CUDA;
- JAX;
- MJX;
- `mujoco_warp`;
- internet access.

If CPU MuJoCo is unavailable in a test environment, CPU rollout tests should skip cleanly.

---

## Validation

Run:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
python3 scripts/benchmark_mujoco_warp.py \
  --output-dir outputs/mujoco_warp_benchmark_smoke \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --steps 5 \
  --warmup-steps 1 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 2
```

The final command must succeed on CPU-only systems by skipping Warp gracefully unless `--require-warp` is passed.

Do not commit:

- benchmark outputs;
- generated MJCF files;
- generated design JSON files;
- logs;
- caches;
- virtual environments.

---

## Out of scope

Do not implement:

- a production `mujoco_warp` backend;
- changes to `handcdo/backends/registry.py`;
- changes to `optimize_hand.py` backend choices;
- changes to `evaluate_design`;
- changes to `optimize_grasp_for_tool`;
- changes to wrench scoring;
- changes to result collection semantics;
- JAX autodiff;
- differentiable physics;
- reinforcement learning;
- neural policies;
- Isaac Sim;
- ROS;
- Slurm GPU submission scripts;
- physical robot evaluation;
- fabrication or mesh manufacturing workflow.
