# PR 10: MuJoCo Warp Compatibility and Throughput Benchmark Only

Implement PR 10: add an optional **MuJoCo Warp / MJWarp compatibility and throughput benchmark scaffold only**.

This PR must not integrate MuJoCo Warp into the normal HandCDO optimization pipeline. It exists only
to test whether the current repository-generated MJCF scenes are compatible with MuJoCo Warp and
whether the known Capella/Alpha HPC GPU nodes provide promising enough throughput to justify a later
experimental backend PR.

This PR is a benchmark and diagnostics PR, not a scientific validation PR and not a production backend PR.

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

Do **not** duplicate design loading, MJCF generation, CPU MuJoCo simulation, grasp sampling, wrench
scoring, result collection, or backend registry logic.

Reuse existing abstractions wherever possible:

- `DesignSpace`
- `HandDesign`
- `write_design_model` / existing MJCF generation helpers
- `EvaluationConfig`
- `GeometryConfig`
- `GraspParams`
- `sample_random_grasp`
- existing tool names: `hammer`, `spoon`, `knife`
- `ensure_dir`
- `write_json`
- existing YAML/JSON utilities

---

## Paper alignment

The paper uses parametric hand design, grasp optimization, and wrench-based stability evaluation over
multiple Cartesian force/torque disturbance directions.

This PR does **not** need to reproduce the full paper evaluation in MuJoCo Warp.

For PR 10, the correct scientific role is:

1. verify whether the repository-generated hand/tool MJCF scenes can be loaded by CPU MuJoCo;
2. verify whether those same scenes, or benchmark-local compatible copies, can be transferred to MuJoCo Warp;
3. measure simple stepping throughput;
4. optionally measure a deterministic contact-smoke scene;
5. report compatibility failures clearly;
6. provide optional Capella/Alpha Slurm validation helpers.

This PR must not claim physical validity, score equivalence, CPU-vs-Warp score correlation, or final
speedup conclusions unless those are actually measured and explicitly marked as experimental.

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

Do not use the name `mjx_warp` for the main module or script unless the implementation actually goes
through the `mujoco.mjx` API.

For this PR, use:

```text
handcdo/benchmarks/mujoco_warp.py
scripts/benchmark_mujoco_warp.py
```

---

## Known HPC GPU scheduler profiles

The user's HPC has the following GPU scheduler profiles.

### Capella GPU profile

```yaml
SCHEDULER_PARAMETERS_CAPELLA: "--partition=capella --nodes=1 --ntasks=1 --cpus-per-task=56 --mem=700000 --cpu-freq=High --gres=gpu:4 --verbose --time=24:00:00 --licenses=''"
```

Equivalent direct Slurm options:

```bash
--partition=capella
--nodes=1
--ntasks=1
--cpus-per-task=56
--mem=700000
--cpu-freq=High
--gres=gpu:4
--verbose
--time=24:00:00
```

### Alpha GPU profile

```yaml
SCHEDULER_PARAMETERS_ALPHA: "--partition=alpha --nodes=1 --ntasks=1 --cpus-per-task=44 --mem=990000 --cpu-freq=High --gres=gpu:8 --verbose --time=72:00:00 --licenses=''"
```

Equivalent direct Slurm options:

```bash
--partition=alpha
--nodes=1
--ntasks=1
--cpus-per-task=44
--mem=990000
--cpu-freq=High
--gres=gpu:8
--verbose
--time=72:00:00
```

Use Capella for short PR10 smoke validation by default.

Use Alpha only for heavier optional sweeps or PR11+ batched backend experiments.

Do not require either scheduler profile for default local tests.

---

## Goal

Add a benchmark-only workflow that can answer:

1. Is MuJoCo Warp importable in the current environment?
2. Is a Warp-compatible GPU/device visible, if this can be checked safely?
3. Can the current repo generate an MJCF model for one fixed hand/tool scene?
4. Can CPU MuJoCo load the generated MJCF model?
5. Does the generated MJCF need benchmark-local compatibility rewrites before Warp transfer?
6. Can MuJoCo Warp transfer the model/data to the device?
7. Can MuJoCo Warp step the model for a simple load/step scene?
8. Can MuJoCo Warp step a deterministic contact-smoke scene, if requested?
9. What is the CPU MuJoCo timing for the same fixed benchmark scene?
10. If MuJoCo Warp is available, what is the GPU timing for the same fixed benchmark scene?
11. How do `nworld`, `nconmax`, `naconmax`, and `njmax` affect success/failure and throughput?
12. Which unsupported-feature, capacity, device, or import errors occur?
13. Can the same benchmark be run locally in CPU-only mode and optionally on Capella/Alpha through Slurm?

This PR must not claim score equivalence or final speedup conclusions.

---

## Required changes

Add:

```text
handcdo/benchmarks/__init__.py
handcdo/benchmarks/mujoco_warp.py
scripts/benchmark_mujoco_warp.py
tests/test_mujoco_warp_benchmark.py
```

Recommended optional Slurm helper scripts:

```text
slurm/mujoco_warp_capella_smoke.sbatch
slurm/mujoco_warp_alpha_sweep.sbatch
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

Do **not** change the default CPU-only workflow.

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
    device_count: int | None = None
    device_names: list[str] | None = None
    cuda_available: bool | None = None


def check_warp_available() -> WarpAvailability:
    ...
```

`check_warp_available()` must not raise for ordinary missing-dependency, missing-CUDA, or missing-device cases.

Device probing must be best effort only.

If GPU/device information cannot be queried safely, record the reason and continue.

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

Use `DesignSpace.from_yaml(...)` and `DesignSpace.sample(seed=...)` where possible.

Write the selected/generated design under the benchmark output directory.

Do not commit generated outputs.

---

## Model generation and output path policy

The existing `write_design_model(...)` helper writes models under a repository-specific layout such as:

```text
<some-output-dir>/designs/<design_id>/design.json
<some-output-dir>/designs/<design_id>/model_<tool>.xml
```

Do not modify the global behavior of `write_design_model(...)`.

For the benchmark, it is acceptable to:

1. call `write_design_model(...)` into a benchmark-local staging directory; and then
2. copy or reference the generated files under canonical benchmark paths.

The benchmark output directory should contain:

```text
<output-dir>/availability.json
<output-dir>/benchmark_results.json
<output-dir>/benchmark_results.csv
<output-dir>/design/design.json
<output-dir>/model/original_model.xml
<output-dir>/model/warp_model.xml
```

Where:

- `original_model.xml` is the raw generated MJCF from the current repo generator;
- `warp_model.xml` is the benchmark-local MJCF after optional compatibility rewrites;
- if no rewrite is applied, `warp_model.xml` may be identical to `original_model.xml`;
- `benchmark_results.json` must record whether the two files differ.

Do not write outputs outside `--output-dir`.

Do not commit generated benchmark outputs.

---

## Warp compatibility rewrite policy

The benchmark may create a benchmark-local MJCF copy that is more likely to be accepted by MuJoCo Warp.

Do not modify the global MJCF generator defaults.

Before calling MuJoCo Warp transfer functions, inspect the generated MJCF and/or loaded `mujoco.MjModel`
for features that are known to be unsupported or risky in MuJoCo Warp.

At minimum:

- If the generated MJCF uses `integrator="implicitfast"`, rewrite only the benchmark-local MJCF copy to
  `integrator="Euler"` unless the user passes `--no-warp-xml-rewrite`.
- Record every rewrite in `benchmark_results.json` under `mjcf_rewrites`.
- Preserve the original generated MJCF under the output directory for debugging.
- If MuJoCo Warp still rejects the model, record the exact exception and do not hide it.
- Do not silently delete geoms, joints, contacts, actuators, or tool bodies to make the model pass.
- Do not rewrite scientific semantics beyond minimal compatibility rewrites unless explicitly recorded.

Suggested result field:

```json
{
  "mjcf_rewrites": [
    {
      "field": "option.integrator",
      "old": "implicitfast",
      "new": "Euler",
      "reason": "benchmark-local MuJoCo Warp compatibility"
    }
  ]
}
```

---

## Benchmark scene tiers

Support two benchmark scene tiers.

### 1. `load_step`

Default mode.

Behavior:

- generate/load one hand-tool MJCF;
- load with CPU MuJoCo;
- transfer to MuJoCo Warp if available;
- run plain stepping only;
- do not set a grasp;
- do not claim contact realism;
- do not compute a score.

This mode is for minimal model compatibility and stepping throughput.

### 2. `contact_smoke`

Optional mode.

Behavior:

- reuse existing `GraspParams` / `sample_random_grasp`;
- set a deterministic tool pose and deterministic actuator controls;
- run a short close/settle phase before timing;
- create at least a plausible hand-tool contact scenario;
- do not compute or report a scientific grasp score;
- report only timing, contact count/capacity failures, and compatibility.

The `contact_smoke` mode may reuse internal CPU helper logic if available, but it must not duplicate the
full CPU evaluator or wrench scoring pipeline.

If exact helper functions are private or inconvenient to import, implement the minimum deterministic setup
locally and document it in the output as:

```json
"scene_mode_semantics": "contact_smoke_not_score_equivalent"
```

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
  --scene-mode load_step \
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
- `--scene-mode`, choices `load_step`, `contact_smoke`, default `load_step`
- `--cpu-repeats`, default `3`
- `--warp-repeats`, default `3`
- `--warmup-steps`, default `10`
- `--steps`, default `100`
- `--nworld`, default `64`
- `--nconmax`, default `64`
- `--naconmax`, optional
- `--njmax`, default `128`
- `--sweep-nworld`, optional comma-separated ints
- `--sweep-nconmax`, optional comma-separated ints
- `--sweep-njmax`, optional comma-separated ints
- `--require-warp`, action flag
- `--skip-cpu`, action flag
- `--overwrite`, action flag
- `--no-warp-xml-rewrite`, action flag

CLI behavior:

- `--help` must work without importing GPU dependencies.
- Missing MuJoCo Warp without `--require-warp` must print a clear message, write availability metadata,
  and exit successfully after the CPU benchmark if CPU benchmark is enabled.
- Missing MuJoCo Warp with `--require-warp` must exit nonzero with a clear error.
- CPU benchmark must remain runnable on a normal CPU-only environment.
- Invalid numeric arguments such as `steps <= 0`, `nworld <= 0`, `repeats <= 0`, `nconmax <= 0`, or
  `njmax <= 0` must fail clearly.
- Sweep args must parse deterministically and reject invalid values clearly.
- Running into an existing output directory should fail unless `--overwrite` is passed.

---

## Capacity sweep

Add optional lightweight sweeps.

Example:

```bash
python3 scripts/benchmark_mujoco_warp.py \
  --output-dir outputs/mujoco_warp_sweep \
  --tool hammer \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --sweep-nworld 1,8,32,64 \
  --sweep-nconmax 32,64,128 \
  --sweep-njmax 64,128,256
```

If no sweep args are provided, run exactly one configuration from `--nworld`, `--nconmax`, `--naconmax`,
and `--njmax`.

For each row, record:

- `backend`
- `scene_mode`
- `nworld`
- `nconmax`
- `naconmax`
- `njmax`
- `available`
- `success`
- `failure_stage`
- `exception_type`
- `exception_message`
- `seconds_mean`
- `seconds_std`
- `steps`
- `warmup_steps`
- `repeats`
- `total_sim_steps`
- `total_world_steps`
- `steps_per_second_total`
- `world_steps_per_second`
- `steps_per_second_per_world`
- `max_contacts_observed`, if safely available
- `max_constraints_observed`, if safely available

Keep default validation fast.

Do not run a large sweep in the default pytest suite.

---

## Optional Slurm helper scripts

Add optional Slurm helper scripts only if they can remain cluster-specific and non-default.

### `slurm/mujoco_warp_capella_smoke.sbatch`

Use Capella for PR10 smoke.

Suggested file body:

```bash
#!/bin/bash
#SBATCH --job-name=handcdo-warp-capella-smoke
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=700000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:4
#SBATCH --verbose
#SBATCH --time=24:00:00
#SBATCH --output=logs/mujoco_warp_capella_smoke_%j.out
#SBATCH --error=logs/mujoco_warp_capella_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

# Adjust environment setup to the actual HPC module/venv policy.
# module purge
# module load Python/3.11
# module load CUDA
# source .venv/bin/activate

hostname
date
nvidia-smi || true

python3 -V

pytest -q -m "not gpu and not slow"

RUN_GPU_TESTS=1 pytest -q -m gpu

python3 scripts/benchmark_mujoco_warp.py \
  --output-dir "outputs/mujoco_warp_capella_smoke_${SLURM_JOB_ID}" \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 8 \
  --nconmax 64 \
  --njmax 128 \
  --require-warp \
  --overwrite
```

### `slurm/mujoco_warp_alpha_sweep.sbatch`

Use Alpha only for heavier optional sweeps.

Suggested file body:

```bash
#!/bin/bash
#SBATCH --job-name=handcdo-warp-alpha-sweep
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=990000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:8
#SBATCH --verbose
#SBATCH --time=72:00:00
#SBATCH --output=logs/mujoco_warp_alpha_sweep_%j.out
#SBATCH --error=logs/mujoco_warp_alpha_sweep_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

# Adjust environment setup to the actual HPC module/venv policy.
# module purge
# module load Python/3.11
# module load CUDA
# source .venv/bin/activate

hostname
date
nvidia-smi || true

python3 -V

RUN_GPU_TESTS=1 pytest -q -m gpu

python3 scripts/benchmark_mujoco_warp.py \
  --output-dir "outputs/mujoco_warp_alpha_sweep_${SLURM_JOB_ID}" \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 100 \
  --warmup-steps 10 \
  --cpu-repeats 1 \
  --warp-repeats 3 \
  --sweep-nworld 8,32,64,128 \
  --sweep-nconmax 64,128,256 \
  --sweep-njmax 128,256,512 \
  --require-warp \
  --overwrite
```

Do not make these Slurm scripts part of default pytest.

Do not require Alpha sweep for PR10 acceptance.

---

## Implementation details

Implement focused functions similar to:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class WarpBenchmarkConfig:
    output_dir: Path
    design_json: Path | None = None
    search_space: Path = Path("configs/search_space.yaml")
    config_path: Path = Path("configs/eval_fast.yaml")
    tool: str = "hammer"
    seed: int = 0
    scene_mode: str = "load_step"
    cpu_repeats: int = 3
    warp_repeats: int = 3
    warmup_steps: int = 10
    steps: int = 100
    nworld: int = 64
    nconmax: int | None = 64
    naconmax: int | None = None
    njmax: int = 128
    sweep_nworld: tuple[int, ...] | None = None
    sweep_nconmax: tuple[int, ...] | None = None
    sweep_njmax: tuple[int, ...] | None = None
    require_warp: bool = False
    skip_cpu: bool = False
    overwrite: bool = False
    no_warp_xml_rewrite: bool = False
```

```python
def validate_config(config: WarpBenchmarkConfig) -> None:
    ...
```

```python
def build_fixed_benchmark_model(config: WarpBenchmarkConfig) -> dict[str, Any]:
    ...
```

Return a payload containing at least:

```python
{
    "design": design,
    "original_mjcf_path": original_mjcf_path,
    "warp_mjcf_path": warp_mjcf_path,
    "mjcf_rewrites": rewrites,
}
```

```python
def prepare_warp_compatible_mjcf(
    original_mjcf_path: str | Path,
    output_path: str | Path,
    allow_rewrite: bool = True,
) -> dict[str, Any]:
    ...
```

```python
def run_cpu_mujoco_timing(
    mjcf_path: str | Path,
    steps: int,
    warmup_steps: int,
    repeats: int,
    scene_mode: str,
    seed: int,
) -> dict[str, Any]:
    ...
```

```python
def run_warp_timing(
    mjcf_path: str | Path,
    steps: int,
    warmup_steps: int,
    repeats: int,
    scene_mode: str,
    seed: int,
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

```python
def write_benchmark_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    ...
```

Use `time.perf_counter()` for timing.

Catch exceptions and record them as structured failure payloads.

Do not hide unsupported-feature errors.

---

## Timing correctness

For CPU timing:

- use `time.perf_counter()`;
- exclude model compilation/loading from measured step timing unless explicitly reporting load time separately;
- report load time and step time separately;
- keep one `mujoco.MjModel` per benchmark scene and create/reset `mujoco.MjData` deterministically for
  repeats where appropriate.

For MuJoCo Warp timing:

- separate import time, CPU model load time, `mjw.put_model` time, data allocation time, warmup time, and measured step time;
- call a safe device synchronization before starting and after finishing measured timing;
- if using the `warp` package directly, call `warp.synchronize()` when available;
- if synchronization is unavailable, record `"synchronized": false` and a warning;
- report whether CUDA graph capture was used;
- default `capture_graph=False` in PR10 unless implemented cleanly and tested.

Suggested timing fields:

```json
{
  "load_seconds": 0.0,
  "transfer_seconds": 0.0,
  "allocation_seconds": 0.0,
  "warmup_seconds": 0.0,
  "seconds_mean": 0.0,
  "seconds_std": 0.0,
  "synchronized": true
}
```

---

## MuJoCo Warp usage policy

The exact MuJoCo Warp API may vary by version.

Use lazy imports inside the Warp-only functions.

The module import itself must not require `mujoco_warp`.

Try the expected standalone API first:

```python
import mujoco
import mujoco_warp as mjw

mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
mj_data = mujoco.MjData(mj_model)

# Pseudocode only; adapt to actual installed API.
m = mjw.put_model(mj_model)
d = mjw.make_data(m, nworld=nworld, nconmax=nconmax, naconmax=naconmax, njmax=njmax)
mjw.step(m, d)
```

If the installed API differs, implement a narrow compatibility helper and record the package version in
`availability.json`.

Do not import JAX.

Do not use `mujoco.mjx` unless this is explicitly refactored and renamed in a later PR.

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

For PR10, any `contact_smoke` result must be labeled as non-score-equivalent:

```json
"score_semantics": "none_benchmark_only"
```

or:

```json
"scene_mode_semantics": "contact_smoke_not_score_equivalent"
```

---

## Output files

Write:

```text
<output-dir>/availability.json
<output-dir>/benchmark_results.json
<output-dir>/benchmark_results.csv
<output-dir>/design/design.json
<output-dir>/model/original_model.xml
<output-dir>/model/warp_model.xml
```

`availability.json` should include:

- `warp_available`
- `reason`
- `package`
- `version`
- Python version
- platform
- CUDA/GPU device info if safely available without raising
- timestamp
- Slurm environment variables if present, including `SLURM_JOB_ID`, `SLURM_JOB_PARTITION`,
  `SLURM_CPUS_PER_TASK`, `SLURM_GPUS`, and `CUDA_VISIBLE_DEVICES`

`benchmark_results.json` should include:

- input arguments;
- generated or loaded design id;
- original MJCF path;
- Warp-compatible MJCF path;
- MJCF rewrites;
- scene mode;
- CPU timing result;
- Warp timing result or skip reason;
- sweep results, if applicable;
- failure count;
- exceptions as strings;
- timestamp;
- detected scheduler profile, if running under Slurm.

`benchmark_results.csv` should contain one row per measured backend and per capacity configuration:

- `backend`
- `available`
- `success`
- `scene_mode`
- `nworld`
- `nconmax`
- `naconmax`
- `njmax`
- `seconds_mean`
- `seconds_std`
- `steps`
- `warmup_steps`
- `repeats`
- `total_sim_steps`
- `total_world_steps`
- `steps_per_second_total`
- `world_steps_per_second`
- `steps_per_second_per_world`
- `failure_count`
- `failure_stage`
- `exception_type`
- `error`

Do not write outputs outside `--output-dir`.

Do not commit generated benchmark outputs.

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
8. Sweep argument parser rejects invalid values.
9. Output schema helpers produce rows with required CSV columns.
10. MJCF rewrite helper rewrites only benchmark-local files and records rewrite metadata.
11. No changes are made to backend registry choices.
12. CPU smoke benchmark skips cleanly if CPU MuJoCo is unavailable.
13. Optional Slurm helper scripts, if added, contain the expected Capella/Alpha scheduler parameters.

Tests must not require:

- H100;
- CUDA;
- JAX;
- MJX;
- `mujoco_warp`;
- internet access;
- Slurm.

If CPU MuJoCo is unavailable in a test environment, CPU rollout tests should skip cleanly.

Do not run a large capacity sweep in tests.

---

## Validation matrix

This PR must support three validation levels.

### Level 1: local CPU-only validation

This is the default validation path for local VS Code + Codex development.

Run:

```bash
pytest -q -m "not gpu and not slow"
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

Expected behavior on CPU-only systems:

- tests pass;
- CLI help works;
- CPU benchmark runs if CPU MuJoCo is installed;
- MuJoCo Warp is skipped gracefully if unavailable;
- `availability.json`, `benchmark_results.json`, and `benchmark_results.csv` are written.

### Level 2: Capella GPU smoke validation

Use Capella as the default HPC GPU smoke target for PR10.

Submit:

```bash
sbatch \
  --partition=capella \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=56 \
  --mem=700000 \
  --cpu-freq=High \
  --gres=gpu:4 \
  --verbose \
  --time=24:00:00 \
  slurm/mujoco_warp_capella_smoke.sbatch
```

The job should run:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu

python3 scripts/benchmark_mujoco_warp.py \
  --output-dir outputs/mujoco_warp_capella_smoke_${SLURM_JOB_ID} \
  --config configs/eval_fast.yaml \
  --tool hammer \
  --scene-mode contact_smoke \
  --steps 20 \
  --warmup-steps 2 \
  --cpu-repeats 1 \
  --warp-repeats 1 \
  --nworld 8 \
  --nconmax 64 \
  --njmax 128 \
  --require-warp \
  --overwrite
```

### Level 3: Alpha GPU optional sweep validation

Use Alpha only for heavier optional sweeps.

Submit:

```bash
sbatch \
  --partition=alpha \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=44 \
  --mem=990000 \
  --cpu-freq=High \
  --gres=gpu:8 \
  --verbose \
  --time=72:00:00 \
  slurm/mujoco_warp_alpha_sweep.sbatch
```

Do not require Alpha validation for PR10 acceptance.

Do not claim GPU correctness, H100 compatibility, or speedup unless the relevant Capella/Alpha validation
level has been run successfully.

---

## README update

Add a short section:

```markdown
## Optional MuJoCo Warp benchmark
```

Explain:

- The repository remains CPU-only by default.
- MuJoCo Warp is optional and intended for GPU compatibility and throughput experiments.
- The benchmark is not a production backend.
- The benchmark does not replace CPU MuJoCo scoring.
- The benchmark does not prove score equivalence.
- Default pytest must not require CUDA, JAX, MJX, MuJoCo Warp, or Slurm.
- Capella is the recommended PR10 GPU smoke target.
- Alpha is for heavier optional sweeps or later PRs.

Do not claim speedup unless measured.

---

## Acceptance criteria

This PR is complete when:

1. Default `pytest -q -m "not gpu and not slow"` passes on CPU-only systems.
2. The benchmark module imports without MuJoCo Warp installed.
3. The benchmark script `--help` works without GPU dependencies.
4. A CPU-only smoke benchmark writes `availability.json`, `benchmark_results.json`, and `benchmark_results.csv`.
5. Missing MuJoCo Warp is reported as a skip, not as a crash, unless `--require-warp` is passed.
6. No production backend is added.
7. No backend registry behavior changes.
8. No default optimization/evaluation CLI behavior changes.
9. The original generated MJCF is preserved.
10. Any benchmark-local MJCF rewrites are recorded.
11. Warp failures are structured and visible.
12. README clearly states that this is optional and benchmark-only.
13. GPU/HPC validation is documented as optional and separate from default local validation.
14. If Slurm helper scripts are added, they match the known Capella/Alpha scheduler profiles.

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
- physical robot evaluation;
- fabrication or mesh manufacturing workflow.
