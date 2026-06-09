# Codex Prompt: PR11-c — Fixed Random-Grasp Batch Orchestration

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR11-a and PR11-b have merged:

```bash
git checkout main
git pull origin main
git checkout -b pr11c-fixed-random-grasp-batch-orchestration
```

## Goal

Add deterministic fixed random-grasp batch orchestration without implementing real MuJoCo Warp physics yet.

This stage prepares the dataflow that the future Warp backend will use:

1. generate all random grasp candidates before evaluation;
2. pass fixed grasps to a batched backend;
3. preserve result order and length;
4. aggregate best grasp per tool/design;
5. test the orchestration using a dummy batched backend.

Do not change the existing CPU Optuna/TPE path.

Do not change default CPU behavior.

Do not add a public CLI yet.

## Why this stage exists

TPE is sequential and adaptive, so it should remain on the existing CPU/sequential path for now.

Random grasp sampling can be generated upfront and evaluated in batch. That is the first safe target for MuJoCo Warp throughput experiments.

This PR should prove the orchestration layer before real GPU stepping is added.

## Required changes

Add one of the following, depending on existing structure:

```text
handcdo/batch_eval.py
```

or:

```text
handcdo/warp_batch_eval.py
```

Update only if needed:

```text
handcdo/backends/mujoco_warp.py
```

Add tests:

```text
tests/test_fixed_random_grasp_batch_orchestration.py
tests/test_warp_batch_result_schema.py
```

## Core helper

Add a helper function like:

```python
from pathlib import Path

from handcdo.backends.batched import BatchedSimulatorBackend
from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


def evaluate_fixed_grasps_batched(
    backend: BatchedSimulatorBackend,
    design: HandDesign,
    tool_name: str,
    grasps: list[GraspParams],
    config: EvaluationConfig | None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> list[GraspEvaluation]:
    ...
```

Rules:

- Preserve input ordering.
- Preserve input length.
- Return an empty list for empty input.
- Do not drop failed trials.
- Failed trials should produce structured failure results when the existing data model supports it; otherwise raise clear exceptions.
- Keep random grasp generation outside the backend. The backend receives fixed `grasps`.
- Do not call `backend.evaluate_grasp()` inside this helper.
- Do not implement fake batching.

## Random grasp generation helper

If the repository already has a random grasp sampler, reuse it.

Add a deterministic helper only if needed, for example:

```python
def sample_fixed_random_grasps(
    n_grasp_trials: int,
    seed: int,
    ...
) -> list[GraspParams]:
    ...
```

Rules:

- Same seed must produce same candidate list.
- Different seed should usually produce different candidate list.
- Sampling must be independent of backend availability.
- Sampling must not import `mujoco_warp`.

## Tool-level aggregation helper

Add a helper only if useful:

```python
def summarize_tool_batch_results(
    tool_name: str,
    grasps: list[GraspParams],
    evaluations: list[GraspEvaluation],
) -> dict:
    ...
```

Required behavior:

- Validate `len(grasps) == len(evaluations)`.
- Select the best non-failed grasp by score.
- Preserve all trials in output if existing CPU schema does so.
- Include `failure_count`.
- Include conservative metadata:

```json
{
  "backend": "mujoco_warp",
  "experimental": true,
  "score_semantics": "experimental_non_equivalent"
}
```

Do not claim CPU-equivalent scoring.

## Sequential fallback policy

Do not add generic CPU fallback here.

If a dummy backend is needed for tests, define it inside tests.

If `allow_sequential_fallback` already exists on `MujocoWarpBackend`, do not use it in this orchestration layer.

The orchestration layer should require a true batched backend object exposing `evaluate_grasps_batch`.

## Tests

Add CPU-only tests using a dummy batched backend.

Cover:

1. Empty grasp list returns empty result list.
2. Input length and output length match.
3. Output order matches input order.
4. The helper calls `evaluate_grasps_batch()` exactly once for the full list.
5. The helper does not call `evaluate_grasp()`.
6. Failed evaluations are not dropped.
7. Best-grasp summary ignores failed trials when possible.
8. Deterministic random grasp generation works if a helper is added.
9. Importing the helper does not require `mujoco_warp`.
10. Existing CPU backend tests still pass.

## Validation

Run:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
```

## Out of scope

Do not implement:

- real MuJoCo Warp stepping;
- per-world state initialization;
- wrench disturbance simulation in Warp;
- public batch CLI;
- TPE batching;
- CPU-vs-Warp comparison;
- JAX/MJX/autodiff;
- Isaac Sim;
- Slurm production scripts.

## Success criteria

This stage is successful if:

1. Fixed random-grasp batch orchestration exists and is CPU-testable.
2. It preserves ordering, length, and failure information.
3. It does not fake batching through `evaluate_grasp()`.
4. CPU-only `pytest -q` passes.
5. CPU MuJoCo remains the reference backend and default workflow.

## Additional tests:
1. The same `(design_id, tool_name, seed, n_grasp_trials)` produces the same grasp list regardless of backend.
2. Tool-level aggregation must not include failed trials when selecting `best_grasp`, but must preserve failed trials in the serialized `trials` list.