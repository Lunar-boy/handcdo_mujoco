# AGENTS.md

This repository is a CPU-first MuJoCo reproduction of the optimization infrastructure from
“Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

The project is developed primarily from a local laptop using VS Code + Codex. The local machine may
have only a weak GPU such as MX350. Therefore, default development and validation must remain
CPU-only. Optional MuJoCo Warp / GPU validation must be guarded and run only when explicitly
requested, preferably through Slurm on the known Capella or Alpha GPU nodes.

---

## Core project goal

Preserve a robust CPU-first research codebase for:

- parametric hand design generation;
- MJCF generation from hand designs;
- CPU MuJoCo grasp and wrench evaluation;
- design batch evaluation;
- Optuna TPE hand optimization;
- multi-fidelity evaluation;
- Slurm CPU/GPU-compatible validation helpers;
- result collection and analysis;
- surrogate-assisted candidate proposal.

GPU-related work is allowed only when the active PR prompt asks for it. GPU work must remain optional
unless the user explicitly requests an experimental GPU PR.

---

## Default local development environment

Assume ordinary Codex work is performed on a local CPU-only or weak-GPU machine.

Default validation commands:

```bash
pytest -q -m "not gpu and not slow"
python3 scripts/benchmark_mujoco_warp.py --help
```

If `scripts/benchmark_mujoco_warp.py` does not exist yet, only run the pytest command.

Do not require CUDA, H100, JAX, MJX, MuJoCo Warp, Isaac Sim, ROS, or Slurm for default tests.

If a command cannot be run because a dependency is missing, report exactly what was missing and which
validation level was skipped.

---

## Local vs GPU/HPC validation

Default validation must be runnable on a normal CPU-only developer machine.

GPU-related code must be written so that:

- importing modules does not require GPU dependencies;
- parsing CLI flags does not require GPU dependencies;
- default pytest does not require GPU dependencies;
- missing optional GPU dependencies produce clear skip/error messages;
- no GPU code path is executed unless explicitly requested.

MuJoCo Warp tests must be optional and guarded by explicit markers or environment variables:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

Do not claim that MuJoCo Warp functionality, H100 behavior, or GPU throughput is validated unless a
Slurm GPU job or equivalent allocated GPU-node test has actually been run.

If only local CPU tests were run, report this explicitly:

```text
GPU/HPC tests were not run because this environment has no allocated CUDA GPU.
```

For laptops with weak GPUs such as MX350, treat GPU tests as unavailable for performance conclusions.
A local GPU smoke test may be useful only for import/device sanity, not for H100 throughput claims.

---

## Pytest markers

Recommended pytest markers:

```text
gpu: requires CUDA-capable GPU and optional MuJoCo Warp dependencies
slow: long-running benchmark or sweep
slurm: intended to run inside a Slurm allocation
capella: intended for Capella GPU validation
alpha: intended for Alpha GPU validation
```

Default test commands must exclude GPU and slow tests:

```bash
pytest -q -m "not gpu and not slow"
```

GPU tests must skip unless explicitly enabled:

```python
import os
import pytest

pytestmark = pytest.mark.gpu

def test_gpu_only_feature():
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU tests.")
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

## Recommended validation levels

### Level 1: local CPU-only validation

Use this from VS Code + Codex on the laptop.

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

- default tests pass;
- CLI help works;
- CPU benchmark runs if CPU MuJoCo is installed;
- MuJoCo Warp is skipped gracefully if unavailable;
- output JSON/CSV files are written.

### Level 2: Capella GPU smoke validation

Use Capella for PR10 GPU smoke validation.

Direct Slurm submission command:

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

Inside the allocated job, run:

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

### Level 3: Alpha GPU heavier validation

Use Alpha only for heavier optional sweeps or PR11+ batched backend work.

Direct Slurm submission command:

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

Inside the allocated job, use larger optional sweeps only if PR10 smoke has already passed on Capella.

Do not require Alpha validation for PR10 acceptance.

---

## MuJoCo Warp policy

For PR10:

- implement benchmark-only compatibility and throughput diagnostics;
- do not add a production backend;
- do not modify backend registry behavior;
- do not change default CLI behavior;
- default tests must remain CPU-only;
- GPU validation is optional and should use Capella first.

For PR11 or later:

- an experimental backend may be added only if the prompt explicitly requests it;
- CPU backend behavior must remain unchanged;
- missing MuJoCo Warp must not break imports;
- H100 throughput claims require actual GPU-node benchmark logs from Capella or Alpha.

Do not use `mujoco.mjx` or JAX unless the prompt explicitly requests a JAX/MJX implementation.
Prefer standalone MuJoCo Warp where applicable:

```python
import mujoco_warp as mjw
```

---

## Backend stability policy

Do not change CPU backend semantics unless the active task explicitly asks for it.

The following behavior must remain stable unless a prompt explicitly says otherwise:

- `mujoco` remains a CPU MuJoCo alias;
- `mujoco_cpu` remains the explicit CPU backend;
- default optimization uses CPU MuJoCo;
- default tests pass without GPU;
- result JSON/CSV schema changes must be backward-compatible or clearly documented.

For benchmark-only PRs, do not add new normal `--backend` choices.

---

## MJCF and geometry policy

Reuse existing MJCF generation helpers whenever possible.

Do not duplicate design loading, MJCF generation, CPU MuJoCo evaluation, grasp sampling, wrench
scoring, or result collection logic unless a narrow benchmark-local helper is necessary.

If benchmark-local MJCF rewrites are needed for MuJoCo Warp compatibility:

- do not modify the global MJCF generator defaults;
- preserve the original generated MJCF;
- write a benchmark-local rewritten copy;
- record every rewrite in JSON output;
- do not silently delete bodies, joints, geoms, contacts, actuators, or tools.

Known example: if the generated MJCF uses `integrator="implicitfast"` and MuJoCo Warp rejects it,
rewrite only the benchmark-local copy to a compatible integrator such as `Euler`, and record the rewrite.

---

## Slurm script policy

Slurm scripts may be added as optional helpers for Capella and Alpha.

Recommended optional files:

```text
slurm/mujoco_warp_capella_smoke.sbatch
slurm/mujoco_warp_alpha_sweep.sbatch
```

Slurm scripts must not be executed automatically from default pytest.

Do not commit large benchmark outputs produced by these scripts.

Use `logs/` for Slurm stdout/stderr and `outputs/` for benchmark result files.

---

## Output and generated files

Do not commit generated outputs.

Do not commit:

- benchmark outputs;
- generated MJCF files;
- generated design JSON files;
- Slurm logs;
- caches;
- virtual environments;
- downloaded models or datasets;
- large binary artifacts.

If rerunning into an existing output directory may mix stale results, fail by default unless
`--overwrite` is passed.

---

## Code quality expectations

Prefer small, testable functions.

Use typed dataclasses for structured configs and results when useful.

Avoid hidden side effects at import time.

Avoid top-level optional GPU imports.

Use lazy imports inside optional code paths.

Record structured failures instead of swallowing exceptions.

When adding CLI scripts:

- keep the script thin;
- place implementation in importable package modules;
- ensure `--help` works without optional GPU dependencies;
- validate numeric arguments clearly.

---

## Reporting expectations

When summarizing changes, distinguish between:

- implemented but not run;
- run locally on CPU;
- skipped because optional dependency is absent;
- run on Capella GPU;
- run on Alpha GPU;
- failed with a specific error.

Do not imply GPU correctness or speedup without GPU benchmark logs.

A correct summary may say:

```text
Validated locally with CPU-only tests. Capella/Alpha GPU MuJoCo Warp validation was not run in this environment.
```

---

## Safety boundaries

Do not implement real robot control, fabrication instructions, or hardware deployment code unless a
future prompt explicitly requests it.

Do not claim physical validity from simplified MuJoCo benchmarks.

Do not present benchmark-only results as final scientific conclusions.
