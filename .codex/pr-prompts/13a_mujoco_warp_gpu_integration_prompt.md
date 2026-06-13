# CodeX Prompt: PR13a — Add real MuJoCo Warp GPU integration smoke test and HPC validation

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
PR13a: Add real MuJoCo Warp GPU integration smoke test and HPC validation
```

This PR must focus on **validation infrastructure**, not on rewriting the MuJoCo Warp backend.

---

## 0. Background

PR #18 introduced an experimental true MuJoCo Warp fixed-random batch scoring path. CPU-only tests already cover metadata, capability gating, failure handling, warmup reporting, capture reporting, partial-chunk reset behavior, and fake-backend control flow.

The default test suite correctly skips GPU tests unless explicitly enabled:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

This PR must add a real-GPU validation harness that exercises the complete:

```python
MujocoWarpBackend.evaluate_grasps_batch(...)
```

path on an actual CUDA/MuJoCo-Warp environment while preserving CPU-only CI behavior.

Add:

1. A gated **real GPU integration smoke test**.
2. A local/HPC **validation module + thin CLI wrapper** that writes structured JSON reports.
3. A generic **Slurm GPU validation template** following the repository's root-level `slurm/` convention.
4. Documentation describing CPU, GPU, local, and Slurm validation workflows.

---

## 1. Scope

### In scope

Add or update files such as:

```text
tests/test_mujoco_warp_gpu_integration.py
handcdo/validation/__init__.py
handcdo/validation/mujoco_warp_gpu.py
scripts/validate_mujoco_warp_gpu.py
slurm/validate_mujoco_warp_gpu.sbatch
docs/mujoco_warp_gpu_validation.md
```

Use the repository's existing style and naming conventions.

### Out of scope

Do **not** rewrite:

```text
handcdo/backends/mujoco_warp.py
handcdo/warp_utils.py
handcdo/backends/mujoco_cpu.py
```

unless a tiny compatibility fix is absolutely required to make the real GPU test runnable. If such a fix is necessary, keep it minimal and explain it in the PR summary.

Do not change the meaning of:

```text
score_semantics = "experimental_non_equivalent"
sequential_fallback = False
include_in_multifidelity = False
```

Do not introduce CPU fallback.

Do not claim CPU equivalence.

Do not make GPU tests run by default in CPU-only environments.

Do not add the MuJoCo Warp path to the multi-fidelity optimization pool.

The MuJoCo Warp path must remain explicitly experimental.

---

## 2. Repository-specific constraints

Follow these repository-specific constraints exactly.

### 2.1 Slurm path convention

Use:

```text
slurm/validate_mujoco_warp_gpu.sbatch
```

Do **not** create:

```text
scripts/slurm/validate_mujoco_warp_gpu.sbatch
```

Keep logs under:

```text
logs/
```

Keep validation JSON reports under:

```text
outputs/warp_gpu_validation/
```

### 2.2 Real API names

Use repository API names that actually exist:

```python
from handcdo.design_space import DesignSpace
from handcdo.batch_eval import sample_fixed_random_grasps
from handcdo.geometry_config import GeometryConfig
from handcdo.mujoco_eval import EvaluationConfig
from handcdo.backends.mujoco_warp import MujocoWarpBackend
```

Use `GraspParams` if a direct grasp dataclass is needed. Do **not** invent `GraspParameters`.

Prefer deterministic repository helpers over ad-hoc fixtures:

```python
design = DesignSpace().sample(seed=13)
tool_name = "hammer"
grasps = sample_fixed_random_grasps(
    n_grasp_trials=2,
    seed=13,
    design_id=design.design_id,
    tool_name=tool_name,
)
config = EvaluationConfig(close_steps=1, settle_steps=1, wrench_steps=1)
geometry_config = GeometryConfig()
```

If the exact helper signatures differ, inspect the current repository and adapt to the real signatures. Do not invent new public APIs unless necessary.

### 2.3 Thin CLI wrapper

The script:

```text
scripts/validate_mujoco_warp_gpu.py
```

must be a thin wrapper.

Put reusable implementation in:

```text
handcdo/validation/mujoco_warp_gpu.py
```

The wrapper should only handle repository-root import setup if needed and call the validation module's `main()`.

### 2.4 Pytest markers

The repository already has pytest markers such as:

```text
gpu
slow
slurm
capella
alpha
```

Confirm that `gpu` remains registered. Do not duplicate marker entries in `pyproject.toml`.

---

## 3. Required GPU integration smoke test

Create:

```text
tests/test_mujoco_warp_gpu_integration.py
```

The file must be marked:

```python
pytestmark = pytest.mark.gpu
```

The test must skip unless:

```bash
RUN_GPU_TESTS=1
```

Recommended pattern:

```python
if os.environ.get("RUN_GPU_TESTS") != "1":
    pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU integration tests.")
```

The test must gracefully skip if any required runtime package is missing:

```text
mujoco
mujoco_warp
warp
```

The test must gracefully skip if no CUDA device is available.

Do not fail CPU-only CI because MuJoCo Warp or CUDA is unavailable.

---

## 4. What the GPU integration test must validate

The test must exercise the real backend path, not only utility functions.

It must call:

```python
backend = MujocoWarpBackend(
    nworld=2,
    warmup_steps=1,
    capture_graph=False,
    readback_interval=1,
)
evaluations = backend.evaluate_grasps_batch(
    design,
    tool_name,
    grasps,
    config,
    geometry_config=geometry_config,
)
```

If the real method signature differs, inspect and use the actual signature. Do not bypass `MujocoWarpBackend.evaluate_grasps_batch(...)`.

Minimum assertions in strict-success mode:

```python
assert len(evaluations) == len(grasps)

metadata = backend.last_batch_metadata
assert metadata is not None
assert metadata["backend"] == "mujoco_warp"
assert metadata["experimental"] is True
assert metadata["score_semantics"] == "experimental_non_equivalent"
assert metadata["sequential_fallback"] is False
assert metadata["include_in_multifidelity"] is False
assert metadata["failure_count"] == 0
assert metadata["failure_reason"] is None
assert metadata["true_batched_scoring"] is True
assert metadata["per_world_state_init"] is True
assert metadata["num_grasps"] == len(grasps)
assert metadata["nworld"] == 2
assert metadata["num_chunks"] >= 1
```

For each returned `GraspEvaluation`:

```python
assert evaluation.tool == tool_name
assert evaluation.failed is False
assert evaluation.error is None
assert isinstance(evaluation.score, float)
assert len(evaluation.wrench_results) == 12
```

The test should also check that the metadata reports warmup/readback truthfully:

```python
assert metadata["warmup_requested_steps"] == 1
assert metadata["warmup_executed_steps"] in (0, 1)
assert metadata["readback_interval"] == 1
assert metadata["capture_graph_requested"] is False
assert metadata["capture_graph_enabled"] is False
```

Do not compare scores against CPU MuJoCo. This PR is not a CPU equivalence PR.

---

## 5. Two-level GPU test policy

Use a two-level policy to avoid false green results while still being practical on alpha MuJoCo Warp environments.

### 5.1 Default GPU mode

Command:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

Default GPU mode must:

1. Import runtime prerequisites or skip.
2. Confirm CUDA device availability or skip.
3. Instantiate `MujocoWarpBackend`.
4. Call `evaluate_grasps_batch(...)`.

If the installed MuJoCo Warp runtime lacks true fixed-grasp batching support and the backend raises a repository-specific capability error such as:

```text
MujocoWarpCapabilityError
NotImplementedError
```

then the test should inspect `backend.last_batch_metadata`, assert that it truthfully reports the failure, and then `pytest.xfail(...)`, unless strict mode is enabled.

Expected truthful failure metadata in this case includes:

```python
assert metadata is not None
assert metadata["backend"] == "mujoco_warp"
assert metadata["experimental"] is True
assert metadata["score_semantics"] == "experimental_non_equivalent"
assert metadata["sequential_fallback"] is False
assert metadata["true_batched_scoring"] is False
assert metadata["failure_count"] == len(grasps)
assert metadata["failure_reason"]
```

If available, also assert:

```python
capabilities = metadata.get("warp_capabilities", {})
assert capabilities.get("supports_true_fixed_grasp_batching") is False
```

Unexpected exceptions must still fail.

Do not let the default `RUN_GPU_TESTS=1` path silently pass without calling the backend.

### 5.2 Strict GPU mode

Command:

```bash
RUN_GPU_TESTS=1 RUN_STRICT_WARP_INTEGRATION=1 pytest -q -m gpu
```

Strict mode must require full success:

```python
metadata["failure_count"] == 0
metadata["failure_reason"] is None
metadata["true_batched_scoring"] is True
metadata["per_world_state_init"] is True
metadata["sequential_fallback"] is False
len(evaluation.wrench_results) == 12 for every evaluation
```

In strict mode, capability failures must be hard failures, not xfail.

---

## 6. Minimal design/tool/grasp setup

Use a very small deterministic smoke configuration.

Recommended values:

```python
nworld = 2
num_grasps = 2
wrench_steps = 1
close_steps = 1
settle_steps = 1
warmup_steps = 1
readback_interval = 1
capture_graph = False
tool_name = "hammer"
```

The goal is end-to-end GPU path validation, not benchmark accuracy.

Do not use large assets or long runtimes.

Do not hard-code absolute machine-specific paths.

Do not depend on files outside the repository unless existing repository APIs already do so.

---

## 7. Required validation module and CLI

Add reusable implementation:

```text
handcdo/validation/mujoco_warp_gpu.py
```

Add thin wrapper:

```text
scripts/validate_mujoco_warp_gpu.py
```

The wrapper should be runnable as:

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

The validation implementation must:

1. Collect runtime environment information:
   - Python version
   - platform
   - current working directory
   - relevant environment variables, including `CUDA_VISIBLE_DEVICES`
   - `mujoco` version if available
   - `mujoco_warp` version if available
   - `warp` version if available
   - CUDA device information if available
2. Run a small real `MujocoWarpBackend.evaluate_grasps_batch(...)` evaluation.
3. Save a structured JSON report containing:
   - timestamp
   - command-line arguments
   - environment info
   - backend metadata
   - per-grasp result summary
   - status: `passed`, `failed`, or `skipped`
   - exception type/message/traceback if failed
   - skip reason if skipped
4. Return well-defined exit codes:
   - `0`: validation succeeded;
   - `0`: validation skipped because prerequisites are missing and `--allow-skip` was set, after writing a skipped JSON report;
   - `1`: backend validation failed after prerequisites were available;
   - `2`: prerequisites missing and `--allow-skip` was not set.
5. Support:
   ```bash
   --allow-skip
   ```
6. Support:
   ```bash
   --strict
   ```
   which applies the same full-success policy as `RUN_STRICT_WARP_INTEGRATION=1`.

Do not silently swallow backend failures. Always write a JSON report before exiting when possible.

Recommended report filename:

```text
mujoco_warp_gpu_validation_<timestamp>.json
```

Use filesystem-safe timestamps.

---

## 8. Runtime prerequisite detection

Implement robust prerequisite detection shared by the test and validation module where practical.

The code should distinguish:

```text
missing package
package import failed
CUDA unavailable
backend capability failure
unexpected backend exception
```

Do not assume a single fixed Warp API. Probe defensively.

Examples of acceptable checks:

```python
import importlib.util

if importlib.util.find_spec("mujoco") is None:
    ...
if importlib.util.find_spec("mujoco_warp") is None:
    ...
if importlib.util.find_spec("warp") is None:
    ...
```

For CUDA/Warp device checks, use the installed `warp` API defensively. If the API differs, skip with a clear reason rather than crashing during prerequisite detection.

The backend evaluation itself may still fail if the runtime is present but incompatible. That is a validation failure or xfail depending on strict/default mode.

---

## 9. Required Slurm script

Add:

```text
slurm/validate_mujoco_warp_gpu.sbatch
```

It should be generic and editable, not hard-coded to one private account.

Template:

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
echo "PWD: $(pwd)"
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

If existing Slurm scripts in the repo have additional style conventions, follow them while preserving the path and output conventions above.

---

## 10. Documentation

Add:

```text
docs/mujoco_warp_gpu_validation.md
```

It must explain:

1. Why GPU tests are optional.
2. How to run CPU tests:
   ```bash
   pytest -q
   ```
3. How to confirm GPU tests are skipped by default:
   ```bash
   pytest -q -rs
   ```
4. How to run GPU smoke tests:
   ```bash
   RUN_GPU_TESTS=1 pytest -q -m gpu
   ```
5. How to run strict GPU smoke tests:
   ```bash
   RUN_GPU_TESTS=1 RUN_STRICT_WARP_INTEGRATION=1 pytest -q -m gpu
   ```
6. How to run only the new integration test:
   ```bash
   RUN_GPU_TESTS=1 pytest -q tests/test_mujoco_warp_gpu_integration.py
   ```
7. How to run the validation script locally:
   ```bash
   python3 scripts/validate_mujoco_warp_gpu.py --results-dir outputs/warp_gpu_validation
   ```
8. How to run it in strict mode:
   ```bash
   python3 scripts/validate_mujoco_warp_gpu.py \
     --results-dir outputs/warp_gpu_validation \
     --strict
   ```
9. How to allow skipped validation in CPU-only environments:
   ```bash
   python3 scripts/validate_mujoco_warp_gpu.py \
     --results-dir outputs/warp_gpu_validation \
     --allow-skip
   ```
10. How to submit the Slurm script:
    ```bash
    sbatch slurm/validate_mujoco_warp_gpu.sbatch
    ```
11. Expected pass/fail/skip/xfail behavior.
12. What metadata fields are important:
    ```text
    backend
    experimental
    score_semantics
    include_in_multifidelity
    true_batched_scoring
    per_world_state_init
    failure_count
    failure_reason
    sequential_fallback
    num_grasps
    num_chunks
    nworld
    readback_interval
    warmup_requested_steps
    warmup_executed_steps
    capture_graph_requested
    capture_graph_enabled
    capture_graph_reason
    warp_capabilities
    ```
13. Current limitations:
    - MuJoCo Warp path remains experimental.
    - Results are not claimed CPU-equivalent.
    - GPU tests require real CUDA and MuJoCo-Warp runtime.
    - Graph capture may remain disabled if unsupported by the dynamic readback path.
    - Default GPU mode may xfail on installed MuJoCo Warp runtimes that cannot support true fixed-grasp batching.
    - Strict mode is the mode that proves the current environment can run the real end-to-end path successfully.

---

## 11. Robust skip/fail/xfail policy

The GPU integration test must distinguish:

### Skip cases

Skip when:

```text
RUN_GPU_TESTS != "1"
mujoco missing
mujoco_warp missing
warp missing
CUDA device unavailable
```

### Xfail cases in default GPU mode only

Xfail when:

```text
RUN_GPU_TESTS=1
runtime prerequisites are present
backend is actually called
backend truthfully refuses because true fixed-grasp batching is unsupported by the installed MuJoCo Warp runtime
RUN_STRICT_WARP_INTEGRATION is not set
```

The xfail must only happen after asserting useful failure metadata.

### Fail cases

Fail when:

```text
RUN_GPU_TESTS=1
runtime prerequisites are present
unexpected exception occurs
metadata is missing
metadata is internally inconsistent
sequential_fallback is True
score_semantics is changed away from experimental_non_equivalent
include_in_multifidelity is True
wrench_results length is not 12 in strict-success mode
```

Strict mode must fail instead of xfail for capability failures.

This prevents false green results on real GPU environments while preserving practical behavior for alpha runtimes.

---

## 12. Validation commands

Run these CPU-safe checks:

```bash
python3 -m py_compile \
  tests/test_mujoco_warp_gpu_integration.py \
  handcdo/validation/mujoco_warp_gpu.py \
  scripts/validate_mujoco_warp_gpu.py

pytest -q
```

Confirm that the default suite still passes with GPU tests skipped:

```bash
pytest -q -rs
```

If GPU is available, run default GPU mode:

```bash
RUN_GPU_TESTS=1 pytest -q -m gpu
```

If the environment should support full real MuJoCo Warp integration, run strict GPU mode:

```bash
RUN_GPU_TESTS=1 RUN_STRICT_WARP_INTEGRATION=1 pytest -q -m gpu
```

Then run the validation script:

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

Optional strict validation script run:

```bash
python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --tool hammer \
  --n-grasps 2 \
  --nworld 2 \
  --wrench-steps 2 \
  --warmup-steps 1 \
  --readback-interval 1 \
  --strict
```

Optional CPU-only skip-report run:

```bash
python3 scripts/validate_mujoco_warp_gpu.py \
  --results-dir outputs/warp_gpu_validation \
  --allow-skip
```

---

## 13. Acceptance criteria

This PR is complete only if:

1. Default `pytest -q` remains CPU-safe and does not require CUDA.
2. A new real backend GPU integration test exists and is gated by `RUN_GPU_TESTS=1`.
3. The GPU integration test calls `MujocoWarpBackend.evaluate_grasps_batch(...)`, not only low-level utility functions.
4. Default GPU mode either:
   - succeeds with truthful metadata and per-grasp `wrench_results`, or
   - xfails only after actually calling the backend and verifying truthful capability-failure metadata.
5. Strict GPU mode requires full success and does not xfail capability failures.
6. A reusable validation implementation exists under `handcdo/validation/`.
7. `scripts/validate_mujoco_warp_gpu.py` is a thin wrapper.
8. The validation script writes structured JSON reports for passed, failed, and skipped outcomes.
9. The Slurm script exists at `slurm/validate_mujoco_warp_gpu.sbatch`.
10. Documentation explains CPU tests, default GPU tests, strict GPU tests, local validation, Slurm validation, and skip/fail/xfail behavior.
11. No CPU fallback is introduced.
12. No CPU equivalence claim is introduced.
13. The MuJoCo Warp path remains explicitly experimental.
14. `include_in_multifidelity` remains `False`.
15. Existing pytest marker config is not duplicated.

---

## 14. Final PR summary

After implementation, update the PR body or add a PR comment with:

```text
Summary:
- Added gated real MuJoCo Warp GPU backend integration smoke test.
- Added reusable local/HPC MuJoCo Warp GPU validation module.
- Added thin validation CLI wrapper that writes structured JSON reports.
- Added Slurm template under slurm/ for GPU validation.
- Documented RUN_GPU_TESTS=1, strict mode, skip/xfail/fail policy, and validation report semantics.

Validation:
- python3 -m py_compile tests/test_mujoco_warp_gpu_integration.py handcdo/validation/mujoco_warp_gpu.py scripts/validate_mujoco_warp_gpu.py
- pytest -q
- pytest -q -rs
- If available: RUN_GPU_TESTS=1 pytest -q -m gpu
- If available: RUN_GPU_TESTS=1 RUN_STRICT_WARP_INTEGRATION=1 pytest -q -m gpu
- If available: python3 scripts/validate_mujoco_warp_gpu.py --results-dir outputs/warp_gpu_validation --tool hammer --n-grasps 2 --nworld 2 --wrench-steps 2 --warmup-steps 1 --readback-interval 1
```

Do not merge automatically.
