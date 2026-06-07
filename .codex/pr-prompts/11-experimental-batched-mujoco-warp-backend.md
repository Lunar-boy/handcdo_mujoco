# PR 11: Experimental Batched MuJoCo Warp Backend

Implement PR 11: add an **experimental batched MuJoCo Warp backend** for H100-class GPU throughput evaluation.

This PR depends on PR 10. Do not start this PR until the MuJoCo Warp benchmark scaffold exists and has shown that repo-generated MJCF scenes can at least be loaded, transferred, and stepped by MuJoCo Warp in a simple smoke benchmark.

---

## Repository context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from arXiv:2604.27557.

Existing code already supports:

- `DesignSpace` and `HandDesign`;
- MJCF generation from hand designs;
- CPU MuJoCo grasp/wrench evaluation;
- backend abstraction through `SimulatorBackend`;
- Optuna TPE or random grasp search;
- batch design evaluation;
- multi-fidelity evaluation;
- surrogate-assisted proposal.

The current backend registry supports only CPU MuJoCo aliases:

```text
mujoco -> mujoco_cpu
mujoco_cpu -> mujoco_cpu
```

This PR may add an experimental `mujoco_warp` backend, but CPU behavior must remain unchanged.

---

## Goal

Add an optional experimental GPU backend that can accelerate large numbers of grasp/wrench evaluations by using MuJoCo Warp batched simulation.

The purpose is H100 throughput, not differentiable physics.

The main value of this PR is not a single-grasp `evaluate_grasp()` wrapper. The main value is a batched path that can evaluate many fixed grasp candidates and/or wrench rollouts in parallel.

---

## Technical direction

Use the standalone MuJoCo Warp package when available:

```python
import mujoco_warp as mjw
```

Do not require JAX.

Do not implement automatic differentiation.

Do not rename this backend `mjx_warp` unless the implementation actually goes through the `mujoco.mjx` API. Prefer:

```text
mujoco_warp
```

---

## Dependency policy

Default installation and default tests must remain CPU-only.

Do not add `mujoco-warp`, CUDA, JAX, or GPU dependencies to `[project.dependencies]`.

If PR 10 already added an optional extra, reuse it:

```toml
[project.optional-dependencies]
warp = [
  "mujoco-warp",
]
```

If the optional extra does not exist, add it.

Missing MuJoCo Warp must produce a clear runtime error only when the experimental backend is explicitly requested.

Default `pytest -q` must pass without GPU, CUDA, JAX, MJX, or MuJoCo Warp.

---

## Required changes

Add or update:

```text
handcdo/backends/mujoco_warp.py
handcdo/backends/batched.py
handcdo/backends/registry.py
scripts/evaluate_design_batch_warp.py
tests/test_mujoco_warp_backend_optional.py
```

Optional, if it avoids mixing too much GPU-specific orchestration into backend code:

```text
handcdo/warp_batch_eval.py
```

Do not remove or change the CPU backend behavior.

Do not change default CLI behavior.

Do not make `mujoco_warp` the default backend.

---

## Backend registry

Add an experimental alias:

```python
_BACKEND_ALIASES = {
    "mujoco": "mujoco_cpu",
    "mujoco_cpu": "mujoco_cpu",
    "mujoco_warp": "mujoco_warp",
}
```

Behavior:

- `get_backend("mujoco")` must still return the CPU backend.
- `get_backend("mujoco_cpu")` must still return the CPU backend.
- `get_backend("mujoco_warp")` may import `MujocoWarpBackend` lazily.
- Missing MuJoCo Warp must not break importing `handcdo.backends`.
- Missing MuJoCo Warp should raise a clear `RuntimeError` or `ImportError` only when `mujoco_warp` is explicitly requested or used.

Do not import `mujoco_warp` at top level of `handcdo/backends/__init__.py`.

---

## Backend protocol design

Keep the existing `SimulatorBackend` protocol unchanged for CPU compatibility.

Add a separate optional batched protocol:

```python
from typing import Protocol

class BatchedSimulatorBackend(Protocol):
    name: str

    def evaluate_grasps_batch(
        self,
        design: HandDesign,
        tool_name: str,
        grasps: list[GraspParams],
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None = None,
        tool_assets_dir: str | Path = "assets/tools",
    ) -> list[GraspEvaluation]:
        ...
```

Optionally add a helper:

```python
def supports_batched_grasps(backend: object) -> bool:
    ...
```

Do not force the CPU backend to implement the batched protocol unless a simple deterministic fallback is useful and does not change behavior.

---

## Experimental MuJoCo Warp backend

Implement:

```text
handcdo/backends/mujoco_warp.py
```

Suggested class:

```python
class MujocoWarpBackend:
    name = "mujoco_warp"

    def __init__(
        self,
        nworld: int = 64,
        nconmax: int | None = 64,
        naconmax: int | None = None,
        njmax: int = 128,
        warmup_steps: int = 0,
        capture_graph: bool = False,
    ) -> None:
        ...

    def evaluate_grasp(...):
        ...

    def evaluate_grasps_batch(...):
        ...
```

`evaluate_grasp()` may be implemented as a compatibility wrapper around `evaluate_grasps_batch(..., [grasp], ...)`, but it should be documented as not the performance-critical path.

The backend must:

- generate/load the same MJCF as CPU MuJoCo for the same `HandDesign`, tool, and `GeometryConfig`;
- use MuJoCo CPU only for model construction/loading and reference naming utilities;
- use MuJoCo Warp for the measured simulation stepping;
- support fixed `nworld` batch size;
- process a list of grasps in chunks of at most `nworld`;
- return one `GraspEvaluation` per input grasp;
- record structured failures per grasp when possible;
- fail clearly for unsupported global model features.

---

## Batched evaluation semantics

The first working version may support only **batched fixed-grasp evaluation**, not full Optuna-TPE batching.

Reason: TPE is sequential/adaptive, while Warp throughput is most useful when many predetermined grasp samples can be evaluated in parallel.

Required behavior:

1. For `sampler="random"`, generate all random grasp candidates first, then evaluate them in batches using `evaluate_grasps_batch` when available.
2. For `sampler="tpe"`, keep existing sequential CPU-style objective behavior unless a safe batched TPE design is explicitly implemented later.
3. Do not change the semantics of CPU random or CPU TPE evaluation.
4. If `mujoco_warp` is requested with `sampler="tpe"`, either:
   - use sequential `evaluate_grasp()` with a clear warning that this will not fully use H100 throughput; or
   - fail clearly and tell the user to use `sampler="random"` for batched Warp evaluation.

Prefer adding a dedicated script for Warp batched random-grasp evaluation instead of silently changing the existing Optuna path.

---

## Wrench scoring policy

The final score should remain conceptually consistent with the existing CPU wrench score:

- close hand;
- settle;
- run disturbance tests over the existing wrench directions;
- measure translation/rotation failure relative to thresholds;
- aggregate normalized stability durations.

However, the first experimental implementation may simplify only if it is clearly marked as experimental and does not claim CPU equivalence.

Do not implement fake score correlation.

If exact CPU-equivalent wrench scoring is not implemented, output must clearly mark:

```json
"score_semantics": "experimental_non_equivalent"
```

If exact CPU-equivalent scoring is implemented, output may mark:

```json
"score_semantics": "intended_cpu_equivalent"
```

Do not use Warp scores as final scientific conclusions until CPU-vs-Warp regression comparisons exist.

---

## CLI

Add:

```text
scripts/evaluate_design_batch_warp.py
```

This script should be explicitly experimental.

Suggested CLI:

```bash
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results \
  --config configs/eval_fast.yaml \
  --tools hammer,spoon,knife \
  --n-grasp-trials 128 \
  --sampler random \
  --nworld 64 \
  --nconmax 64 \
  --njmax 128 \
  --seed 0
```

Supported args:

- `--design-dir`, default `outputs/designs`
- `--design-ids`, optional text file with one design id per line
- `--results-dir`, required
- `--config`, default `configs/eval_fast.yaml`
- `--tools`, default `hammer,spoon,knife`
- `--n-grasp-trials`, default `64`
- `--sampler`, choices `random`, optionally `tpe`; default `random`
- `--nworld`, default `64`
- `--nconmax`, default `64`
- `--naconmax`, optional
- `--njmax`, default `128`
- `--warmup-steps`, default `0`
- `--capture-graph`, action flag, default false
- `--seed`, default `0`
- `--max-designs`, optional
- `--require-warp`, action flag
- `--continue-on-error`, action flag, default true

CLI behavior:

- `--help` must work without importing MuJoCo Warp.
- Missing MuJoCo Warp must produce a clear error when the script actually attempts GPU evaluation.
- The script must write one result JSON per design, compatible with existing result collection where practical.
- The script must not silently overwrite existing results unless an explicit `--overwrite` flag is provided.

---

## Output schema

For each design result JSON, preserve the existing high-level shape where practical:

```json
{
  "design_id": "...",
  "parameters": {...},
  "hand_score": 0.0,
  "tool_results": [...],
  "failed": false,
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "intended_cpu_equivalent",
  "warp_metadata": {...}
}
```

`warp_metadata` should include:

- `nworld`
- `nconmax`
- `naconmax`
- `njmax`
- `warmup_steps`
- `capture_graph`
- `batch_size`
- `num_grasps`
- `num_chunks`
- `seconds_total`
- `grasps_per_second` if measurable
- `world_steps_per_second` if measurable
- `failure_count`
- optional device info if safely available

Per-tool result should preserve the existing shape where practical:

```json
{
  "tool": "hammer",
  "best_score": 0.0,
  "best_grasp": {...},
  "trials": [...]
}
```

Each trial should include:

- grasp parameters;
- score;
- wrench results if implemented;
- failed flag;
- error if failed;
- backend metadata if needed.

---

## CPU-vs-Warp comparison helper

If practical, add a lightweight comparison helper:

```text
scripts/compare_cpu_warp_results.py
```

This is optional in this PR.

If added, it should compare existing result CSV/JSON files and report:

- overlapping design ids;
- CPU score;
- Warp score;
- absolute score difference;
- rank drift;
- failed rows;
- backend metadata.

Do not require this helper if it makes the PR too large.

---

## README update

Add a short section:

```markdown
## Experimental MuJoCo Warp backend
```

Explain:

- This backend is optional and experimental.
- It is intended for H100/NVIDIA GPU throughput experiments.
- It is most useful for batched random-grasp/wrench evaluation.
- It is not the default backend.
- CPU MuJoCo remains the reference implementation.
- Scientific conclusions should still be validated against CPU MuJoCo until regression comparisons are stable.
- Default tests and default installation remain CPU-only.

Include install example:

```bash
python3 -m pip install -e ".[test]"
python3 -m pip install -e ".[warp]"
```

Include smoke example:

```bash
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results_smoke \
  --config configs/eval_fast.yaml \
  --tools hammer \
  --n-grasp-trials 8 \
  --sampler random \
  --nworld 8 \
  --seed 0
```

Do not claim speedup unless measured.

---

## Tests

Add focused tests that do not require GPU or MuJoCo Warp.

Suggested file:

```text
tests/test_mujoco_warp_backend_optional.py
```

Cover:

1. Importing `handcdo.backends` works without MuJoCo Warp installed.
2. `get_backend("mujoco")` still returns CPU backend.
3. `get_backend("mujoco_cpu")` still returns CPU backend.
4. `get_backend("mujoco_warp")` fails clearly if MuJoCo Warp is absent.
5. `scripts/evaluate_design_batch_warp.py --help` works without MuJoCo Warp installed.
6. `supports_batched_grasps()` returns false for CPU backend unless a CPU fallback is intentionally implemented.
7. `MujocoWarpBackend` construction validates invalid `nworld <= 0`, `nconmax <= 0`, or `njmax <= 0`.
8. Output schema helpers include required metadata fields.
9. Existing CPU tests still pass unchanged.

Optional GPU tests should be skipped unless MuJoCo Warp and a CUDA device are available.

Use skip markers or runtime checks for GPU-dependent tests.

Default test suite must not require:

- H100;
- CUDA;
- JAX;
- MJX;
- `mujoco_warp`;
- internet access.

---

## Validation

On CPU-only environment:

```bash
pytest -q
python3 scripts/evaluate_design_batch_warp.py --help
```

On H100/GPU environment with optional dependency installed:

```bash
python3 -m pip install -e ".[warp]"
python3 scripts/evaluate_design_batch_warp.py \
  --design-dir outputs/designs \
  --results-dir outputs/warp_results_smoke \
  --config configs/eval_fast.yaml \
  --tools hammer \
  --n-grasp-trials 8 \
  --sampler random \
  --nworld 8 \
  --nconmax 64 \
  --njmax 128 \
  --seed 0
```

Then collect results if compatible:

```bash
python3 scripts/collect_results.py \
  --results-dir outputs/warp_results_smoke \
  --output-csv outputs/warp_results_smoke.csv
```

Do not commit generated outputs, logs, caches, model files, or virtual environments.

---

## Out of scope

Do not implement:

- JAX autodiff;
- differentiable physics;
- MJX-JAX backend;
- Isaac Sim;
- ROS;
- RL training;
- policy learning;
- neural controllers;
- fabrication workflow;
- physical robot evaluation;
- changes to design-space bounds;
- changes to CPU MuJoCo scoring semantics;
- changes to Optuna objective direction;
- changes to default backend behavior;
- GPU Slurm production templates unless explicitly requested later.

---

## Success criteria

This PR is successful if:

1. CPU-only installation and tests still pass.
2. `mujoco_warp` is available only when explicitly requested.
3. The experimental backend can evaluate fixed random grasp batches on H100 when dependencies are installed.
4. The output schema remains compatible with existing result analysis where practical.
5. CPU MuJoCo remains the reference backend.
6. The README clearly warns that the backend is experimental and intended for throughput exploration.
