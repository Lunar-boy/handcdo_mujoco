# PR10 Follow-up Prompt: Fix Merge Blockers for MuJoCo Warp Benchmark Scaffold

You are working in the existing `handcdo_mujoco` repository after PR10 has already implemented the benchmark-only MuJoCo Warp scaffold.

This is a **follow-up fix prompt**, not a new feature PR.

Your task is to fix the remaining merge blockers from PR10 review and make the PR locally mergeable under CPU-only validation, while keeping GPU/HPC validation optional.

Do **not** expand scope into a production MuJoCo Warp backend.

---

## Current PR10 state

PR10 already added or edited files similar to:

```text
handcdo/benchmarks/mujoco_warp.py
handcdo/benchmarks/__init__.py
scripts/benchmark_mujoco_warp.py
tests/test_mujoco_warp_benchmark.py
slurm/mujoco_warp_capella_smoke.sbatch
slurm/mujoco_warp_alpha_sweep.sbatch
README.md
pyproject.toml
```

Current validation reported:

```text
pytest -q failed because pytest is not installed: pytest: command not found
python3 -m pytest failed because pytest is not installed: No module named pytest
python3 -m py_compile handcdo/benchmarks/mujoco_warp.py scripts/benchmark_mujoco_warp.py tests/test_mujoco_warp_benchmark.py passed
python3 scripts/benchmark_mujoco_warp.py --help passed
CPU-only smoke wrote expected output files under outputs/mujoco_warp_local_cpu_smoke
CPU MuJoCo timing did not run because mujoco is not installed
MuJoCo Warp was skipped because mujoco_warp is not installed
GPU/HPC tests were not run because no allocated CUDA GPU is available
```

The implementation looks close to mergeable, but the review found several blockers.

---

## Merge blockers to fix

### Blocker 1: Real pytest validation was not run

`pyproject.toml` already contains or should contain:

```toml
[project.optional-dependencies]
test = ["pytest>=7.4"]
```

Make sure the README and validation instructions use:

```bash
python3 -m pip install -e ".[test]"
python3 -m pytest -q -m "not gpu and not slow"
```

Do not rely on plain `pytest`, because some environments may not put the pytest console script on `PATH`.

All Slurm scripts and docs should prefer:

```bash
python3 -m pytest ...
```

instead of:

```bash
pytest ...
```

---

### Blocker 2: Slurm scripts call `pytest -m gpu`, but the repo has no actual GPU marker test

Current Slurm scripts likely contain something like:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

This is fragile because the repo may have zero tests marked `gpu`, which can make the job fail or make the command meaningless.

Fix this by adding one minimal optional GPU smoke test.

Add:

```text
tests/test_mujoco_warp_gpu_smoke.py
```

Recommended content:

```python
from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.gpu


def test_mujoco_warp_gpu_import_and_device() -> None:
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU tests.")

    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    warp = pytest.importorskip("warp")

    try:
        device = warp.get_device("cuda")
    except TypeError:
        device = warp.get_device()
    except Exception as exc:  # pragma: no cover - depends on GPU runtime
        pytest.fail(f"Could not get CUDA device through warp: {type(exc).__name__}: {exc}")

    assert device is not None
```

This test must:

- be marked `pytest.mark.gpu`;
- skip unless `RUN_GPU_TESTS=1`;
- use `pytest.importorskip(...)` for optional packages;
- not run during default local validation;
- not require CUDA unless explicitly enabled;
- fail only when explicitly enabled and a usable CUDA device cannot be obtained.

---

### Blocker 3: Slurm scripts should use `python3 -m pytest`

Update:

```text
slurm/mujoco_warp_capella_smoke.sbatch
slurm/mujoco_warp_alpha_sweep.sbatch
```

Replace plain `pytest` calls with:

```bash
python3 -m pytest -q -m "not gpu and not slow"
RUN_GPU_TESTS=1 python3 -m pytest -q -m gpu
```

For Capella, keep this scheduler profile:

```bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=700000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:4
#SBATCH --verbose
#SBATCH --time=24:00:00
```

For Alpha, keep this scheduler profile:

```bash
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=990000
#SBATCH --cpu-freq=High
#SBATCH --gres=gpu:8
#SBATCH --verbose
#SBATCH --time=72:00:00
```

Do not make Slurm validation part of default local pytest.

---

### Blocker 4: README still has an outdated “No GPU-only libraries” style statement

Search `README.md` for phrases like:

```text
No GPU-only libraries.
```

or equivalent.

Replace with wording that preserves the CPU-first guarantee but allows optional benchmark-only MuJoCo Warp:

```markdown
- No GPU-only libraries are required by the default installation or default test suite.
- Optional MuJoCo Warp support is benchmark-only and not a production backend.
```

Make sure README does **not** imply that the repository has zero optional GPU dependencies after PR10.

---

### Blocker 5: README test instructions should use the real local merge gate

Update the README test section so it uses:

```bash
python3 -m pip install -e ".[test]"
python3 -m pytest -q -m "not gpu and not slow"
```

If there is an Optional MuJoCo Warp section, it should clearly state:

```markdown
Default tests do not require CUDA, Slurm, MuJoCo Warp, JAX, MJX, or H100.

GPU tests are optional and must be explicitly enabled:

```bash
RUN_GPU_TESTS=1 python3 -m pytest -q -m gpu
```
```

Keep the wording clear that laptop GPU timings, especially on weak GPUs like MX350, should not be interpreted as H100-class throughput.

---

## Optional but recommended improvement: clarify availability semantics

If `handcdo/benchmarks/mujoco_warp.py` currently treats `mujoco_warp` import success as `available=True` even when CUDA device probing fails, that is acceptable for PR10.

However, if you can make the result schema clearer without breaking tests, prefer fields like:

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
    usable_for_gpu_benchmark: bool | None = None
```

Do not make this change if it causes unnecessary churn.

The minimum requirement is that missing device or missing CUDA must not crash imports or default tests.

---

## Required files to inspect and update

Inspect and update as needed:

```text
README.md
pyproject.toml
tests/test_mujoco_warp_benchmark.py
tests/test_mujoco_warp_gpu_smoke.py
slurm/mujoco_warp_capella_smoke.sbatch
slurm/mujoco_warp_alpha_sweep.sbatch
handcdo/benchmarks/mujoco_warp.py
```

Do not edit unrelated files.

Do not modify:

```text
handcdo/backends/registry.py
handcdo/backends/__init__.py
handcdo/optimize_hand.py
handcdo/optimize_grasp.py
```

Do not add:

```text
handcdo/backends/mujoco_warp.py
```

Do not add `mujoco_warp` to normal `--backend` choices.

---

## Acceptance criteria

After your changes:

1. Default CPU-only tests can be run with:

   ```bash
   python3 -m pip install -e ".[test]"
   python3 -m pytest -q -m "not gpu and not slow"
   ```

2. `python3 scripts/benchmark_mujoco_warp.py --help` works without MuJoCo Warp, CUDA, JAX, MJX, or Slurm.

3. CPU-only smoke works and skips Warp gracefully when `mujoco_warp` is absent:

   ```bash
   python3 scripts/benchmark_mujoco_warp.py      --output-dir outputs/mujoco_warp_local_cpu_smoke      --config configs/eval_fast.yaml      --tool hammer      --steps 5      --warmup-steps 1      --cpu-repeats 1      --warp-repeats 1      --nworld 2      --overwrite
   ```

4. The new GPU smoke test is not run by default:

   ```bash
   python3 -m pytest -q -m "not gpu and not slow"
   ```

5. The GPU smoke test is selected only when explicitly requested:

   ```bash
   RUN_GPU_TESTS=1 python3 -m pytest -q -m gpu
   ```

6. Slurm scripts use `python3 -m pytest`, not bare `pytest`.

7. README no longer says or implies that optional GPU benchmark dependencies do not exist.

8. README clearly distinguishes:

   ```text
   local CPU-only validation
   optional local GPU smoke
   optional Capella GPU smoke
   optional Alpha sweep
   ```

9. No production MuJoCo Warp backend is added.

10. Backend registry behavior remains unchanged.

---

## Validation commands to run locally

Run these locally after editing:

```bash
python3 -m pip install -e ".[test]"
python3 -m pytest -q -m "not gpu and not slow"
python3 scripts/benchmark_mujoco_warp.py --help
python3 scripts/benchmark_mujoco_warp.py   --output-dir outputs/mujoco_warp_local_cpu_smoke   --config configs/eval_fast.yaml   --tool hammer   --steps 5   --warmup-steps 1   --cpu-repeats 1   --warp-repeats 1   --nworld 2   --overwrite
```

If `mujoco` is installed by `.[test]`, the CPU timing path should run.

If `mujoco_warp` is not installed, the Warp path should be skipped gracefully.

Do not run GPU/HPC validation locally unless a CUDA GPU and the optional dependencies are actually available.

---

## Optional Capella validation

After local validation passes, the user may run:

```bash
sbatch slurm/mujoco_warp_capella_smoke.sbatch
```

This is optional for PR10 merge readiness.

Do not claim Capella/H100 correctness unless this job actually succeeds and the resulting logs and JSON outputs are available.

---

## Expected final report

At the end, report:

- files changed;
- whether `python3 -m pip install -e ".[test]"` succeeded;
- whether `python3 -m pytest -q -m "not gpu and not slow"` passed;
- whether CLI help passed;
- whether CPU-only smoke passed;
- whether `mujoco` CPU timing actually ran or was skipped;
- whether `mujoco_warp` was skipped;
- whether GPU/HPC validation was run or not.

Use explicit wording such as:

```text
GPU/HPC tests were not run because this environment has no allocated CUDA GPU.
```

Do not say the PR is fully GPU-validated unless a Slurm GPU job was actually run.
