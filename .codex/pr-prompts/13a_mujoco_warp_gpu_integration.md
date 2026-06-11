# CodeX Prompt: PR13a — Add real MuJoCo Warp GPU integration smoke test and HPC validation script

Repository:

```text
Lunar-boy/handcdo_mujoco
```

Target branch context:

```text
Base this work on the branch after PR #18 is merged, or on the latest PR #18 branch if it has not been merged yet.
```

PR title suggestion:

```text
PR13a: Add real MuJoCo Warp GPU integration smoke test and HPC validation script
```

This PR must focus on **validation infrastructure**, not on rewriting the MuJoCo Warp backend.

---

## 0. Background

PR #18 introduced an experimental true MuJoCo Warp fixed-random batch scoring path. CPU-only tests cover metadata, capability gating, failure handling, warmup reporting, capture reporting, partial-chunk reset behavior, and fake backend control flow.

However, the current default test suite skips GPU tests unless explicitly enabled:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

This is correct for normal CPU-only CI, but it means we do not yet have a strong real-GPU integration test that validates the complete `MujocoWarpBackend.evaluate_grasps_batch(...)` path on an actual CUDA/MuJoCo-Warp environment.

This PR must add:

1. A **real GPU integration smoke test** for the MuJoCo Warp backend.
2. A **HPC validation script** that can be submitted on a Slurm GPU node and saves reproducible logs/metadata.
3. Documentation of how to run both.

---

## 1. Scope

### In scope

Add or update files such as:

```text
tests/test_mujoco_warp_gpu_integration.py
scripts/validate_mujoco_warp_gpu.py
scripts/slurm/validate_mujoco_warp_gpu.sbatch
docs/mujoco_warp_gpu_validation.md
```

If the repository has a different convention for scripts/docs/test locations, follow the existing convention.

### Out of scope

Do not rewrite:

```text
handcdo/backends/mujoco_warp.py
handcdo/warp_utils.py
handcdo/backends/mujoco_cpu.py
```

unless a tiny compatibility fix is absolutely required to make the real GPU test runnable.

Do not change the meaning of:

```text
score_semantics = "experimental_non_equivalent"
sequential_fallback = False
include_in_multifidelity = False
```

Do not introduce CPU fallback.

Do not claim CPU equivalence.

Do not make GPU tests run by default in CPU-only environments.

---

## 2. Required GPU integration smoke test

Create a new gated test file, recommended:

```text
tests/test_mujoco_warp_gpu_integration.py
```

The test must be marked:

```python
pytestmark = pytest.mark.gpu
```

and must skip unless:

```bash
RUN_GPU_TESTS=1
```

Recommended skip logic:

```python
if os.environ.get("RUN_GPU_TESTS") != "1":
    pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU integration tests.")
```

The test must also gracefully skip if required runtime packages are missing or no CUDA device is available:

```python
mujoco
mujoco_warp
warp
```

Do not fail CPU-only CI because MuJoCo Warp or CUDA is unavailable.

---

## 3. What the GPU integration test must validate

The test must exercise the real backend path, not only utility functions.

It should call:

```python
MujocoWarpBackend(...).evaluate_grasps_batch(...)
```

with a small real design/tool/grasp setup.

Minimum required assertions:

```python
len(evaluations) == len(grasps)
backend.last_batch_metadata is not None
metadata["backend"] == "mujoco_warp"
metadata["experimental"] is True
metadata["score_semantics"] == "experimental_non_equivalent"
metadata["sequential_fallback"] is False
metadata["include_in_multifidelity"] is False
metadata["failure_count"] == 0
metadata["failure_reason"] is None
metadata["true_batched_scoring"] is True
metadata["per_world_state_init"] is True
metadata["num_grasps"] == len(grasps)
metadata["nworld"] >= len(grasps) or metadata["num_chunks"] >= 1
```

For each returned `GraspEvaluation`:

```python
evaluation.tool == tool_name
evaluation.failed is False
evaluation.error is None
isinstance(evaluation.score, float)
len(evaluation.wrench_results) == 12
```

If the real implementation may occasionally fail due to MuJoCo-Warp alpha limitations, the test may be written as a strict smoke test only when an environment variable is enabled:

```bash
RUN_GPU_TESTS=1 RUN_STRICT_WARP_INTEGRATION=1 pytest -q -m gpu
```

But the default `RUN_GPU_TESTS=1` path should still run a meaningful check. Do not let it silently pass without calling the backend.

---

## 4. Minimal design/tool/grasp setup

Prefer using existing repository factories/fixtures/utilities for:

```text
HandDesign
ToolSpec
GraspParameters
EvaluationConfig
```

Search the existing tests first and reuse current helper patterns.

The test should avoid large assets and long runtimes. Use:

```python
nworld = 2
num_grasps = 2
wrench_steps = 1 or 2
settle_steps = as small as allowed
warmup_steps = 1
readback_interval = 1
capture_graph = False
```

The goal is not benchmark accuracy; it is end-to-end GPU path validation.

If there are existing preset designs/tools under `outputs/designs`, `examples`, or test fixtures, use them. If not, build a minimal valid hand/tool design using existing dataclasses/builders.

Do not hard-code absolute machine-specific paths.

---

## 5. Required HPC validation script

Add a script, recommended:

```text
scripts/validate_mujoco_warp_gpu.py
```

It should be runnable as:

```bash
python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --tool hammer \
  --n-grasps 2 \
  --nworld 2 \
  --wrench-steps 2 \
  --warmup-steps 1 \
  --readback-interval 1
```

The script must:

1. Import and print runtime environment information:
   - Python version
   - platform
   - `mujoco` version if available
   - `mujoco_warp` version if available
   - `warp` version if available
   - CUDA device info if available
2. Run a small real `MujocoWarpBackend.evaluate_grasps_batch(...)` evaluation.
3. Save a JSON file containing:
   - timestamp
   - command-line arguments
   - environment info
   - backend metadata
   - per-grasp result summary
   - pass/fail status
   - exception traceback if failed
4. Exit with:
   - code `0` if validation succeeds;
   - code `1` if validation fails;
   - code `2` if runtime prerequisites are missing and `--allow-skip` is not set.
5. Support:
   ```bash
   --allow-skip
   ```
   so that environments without GPU/MuJoCo-Warp can record a skipped validation JSON instead of hard failing.

Do not silently swallow backend failures. If the backend fails, write the JSON report and return a nonzero exit code.

---

## 6. Required Slurm script

Add an example Slurm script, recommended:

```text
scripts/slurm/validate_mujoco_warp_gpu.sbatch
```

It should be generic and editable, not hard-coded to one private account.

Recommended template:

```bash
#!/bin/bash
#SBATCH --job-name=warp-gpu-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/warp_gpu_validate_%j.out
#SBATCH --error=logs/warp_gpu_validate_%j.err

set -euo pipefail

mkdir -p logs outputs/warp_gpu_validation

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Optional: activate venv or load modules here.
# source .venv/bin/activate
# module load ...

python3 -m pip show mujoco mujoco-warp warp-lang || true

python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --tool hammer \
  --n-grasps 2 \
  --nworld 2 \
  --wrench-steps 2 \
  --warmup-steps 1 \
  --readback-interval 1
```

If the repository already has Slurm script style conventions, follow them.

---

## 7. Documentation

Add documentation, recommended:

```text
docs/mujoco_warp_gpu_validation.md
```

It must explain:

1. Why GPU tests are optional.
2. How to run CPU tests:
   ```bash
   pytest -q
   ```
3. How to run GPU smoke tests:
   ```bash
   RUN_GPU_TESTS=1 pytest -q -m gpu
   ```
4. How to run only the new integration test:
   ```bash
   RUN_GPU_TESTS=1 pytest -q tests/test_mujoco_warp_gpu_integration.py
   ```
5. How to run the validation script locally:
   ```bash
   python3 scripts/validate_mujoco_warp_gpu.py --results-dir outputs/warp_gpu_validation
   ```
6. How to submit the Slurm script:
   ```bash
   sbatch scripts/slurm/validate_mujoco_warp_gpu.sbatch
   ```
7. Expected pass/fail/skip behavior.
8. What metadata fields must be checked:
   ```text
   true_batched_scoring
   per_world_state_init
   failure_count
   failure_reason
   sequential_fallback
   score_semantics
   readback_interval
   warmup_executed_steps
   capture_graph_enabled
   capture_graph_reason
   ```
9. Current limitations:
   - MuJoCo Warp path remains experimental.
   - Results are not claimed CPU-equivalent.
   - GPU tests require real CUDA and MuJoCo-Warp runtime.
   - Graph capture may remain disabled if unsupported by the dynamic readback path.

---

## 8. Pytest marker configuration

If the repo does not already register the `gpu` marker, add it to the appropriate config file:

```text
pyproject.toml
pytest.ini
setup.cfg
```

Example:

```toml
[tool.pytest.ini_options]
markers = [
    "gpu: tests requiring a CUDA GPU and MuJoCo Warp runtime",
]
```

Do not break existing pytest config.

---

## 9. Robust skip/fail policy

The GPU integration test must distinguish:

### Skip cases

Skip when:

```text
RUN_GPU_TESTS != "1"
mujoco/mujoco_warp/warp missing
CUDA device unavailable
```

### Fail cases

Fail when `RUN_GPU_TESTS=1` and prerequisites are available, but:

```text
backend evaluation raises unexpectedly
metadata reports failure_count > 0
true_batched_scoring is not True
per_world_state_init is not True
sequential_fallback is True
wrench_results length is not 12
```

This prevents false green results on real GPU environments.

---

## 10. Validation commands

Run these CPU-safe checks:

```bash
python3 -m py_compile \
  tests/test_mujoco_warp_gpu_integration.py \
  scripts/validate_mujoco_warp_gpu.py

pytest -q
```

Confirm that the default suite still passes with the GPU tests skipped:

```bash
pytest -q -rs
```

If GPU is available, run:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

Then run:

```bash
python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --tool hammer \
  --n-grasps 2 \
  --nworld 2 \
  --wrench-steps 2 \
  --warmup-steps 1 \
  --readback-interval 1
```

---

## 11. Acceptance criteria

This PR is complete only if:

1. Default `pytest -q` remains CPU-safe and does not require CUDA.
2. A new real backend GPU integration test exists and is gated by `RUN_GPU_TESTS=1`.
3. The GPU integration test calls `MujocoWarpBackend.evaluate_grasps_batch(...)`, not only low-level utility functions.
4. The test asserts successful metadata and per-grasp `wrench_results`.
5. A standalone validation script writes structured JSON reports.
6. A Slurm script exists for HPC GPU validation.
7. Documentation explains how to run CPU tests, GPU tests, and HPC validation.
8. No CPU fallback is introduced.
9. No CPU equivalence claim is introduced.
10. The MuJoCo Warp path remains explicitly experimental.

---

## 12. Final PR summary

After implementation, update the PR body or add a PR comment with:

```text
Summary:
- Added gated real MuJoCo Warp GPU backend integration smoke test.
- Added local/HPC validation script that writes structured JSON metadata.
- Added Slurm template for GPU validation.
- Documented RUN_GPU_TESTS=1 workflow and skip/fail policy.

Validation:
- pytest -q
- pytest -q -rs
- python3 -m py_compile ...
- If available: RUN_GPU_TESTS=1 pytest -q -m gpu
- If available: python3 scripts/validate_mujoco_warp_gpu.py ...
```

Do not merge automatically.
