# Codex Prompt: PR11-a — Warp Utilities and Batched Backend Protocol

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Create a branch:

```bash
git checkout main
git pull origin main
git checkout -b pr11a-warp-utils-batched-protocol
```

## Goal

Implement only the low-risk infrastructure layer for PR11:

1. a batched backend protocol;
2. optional MuJoCo Warp utility extraction from the PR10 benchmark scaffold.

Do not implement the actual MuJoCo Warp backend yet.

Do not change CPU MuJoCo behavior.

Do not change default CLI behavior.

## Repository context

The repository currently has:

- CPU MuJoCo evaluation as the reference backend;
- a single-grasp `SimulatorBackend` interface;
- existing PR10 MuJoCo Warp benchmark scaffolding;
- optional `warp` dependency in `pyproject.toml`;
- CPU-only default tests.

PR10 already contains useful MuJoCo Warp helper logic such as:

- availability checking;
- safe optional import;
- device probing;
- MJCF compatibility rewrite;
- MuJoCo Warp model/data allocation helpers;
- synchronization helpers;
- benchmark metadata collection.

This PR should make those utilities reusable without changing benchmark behavior.

## Required changes

Add:

```text
handcdo/backends/batched.py
```

Optionally add:

```text
handcdo/warp_utils.py
```

Update only if necessary for helper reuse:

```text
handcdo/benchmarks/mujoco_warp.py
```

Add focused tests:

```text
tests/test_batched_backend_protocol.py
tests/test_warp_utils_optional.py
```

## Batched backend protocol

Create a separate protocol. Do not modify the existing `SimulatorBackend` protocol.

Suggested shape:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


@runtime_checkable
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


def supports_batched_grasps(backend: object) -> bool:
    return callable(getattr(backend, "evaluate_grasps_batch", None))
```

Rules:

- `supports_batched_grasps()` must not instantiate a backend.
- It must not import `mujoco_warp`.
- It should return `False` for plain objects.
- It should return `True` for objects exposing callable `evaluate_grasps_batch`.
- Do not force the CPU backend to implement this protocol in PR11-a.

## Warp utility extraction

Inspect the existing PR10 MuJoCo Warp benchmark code.

If the benchmark already has reusable logic, factor it into `handcdo/warp_utils.py`.

Candidate utilities:

```text
WarpAvailability
check_warp_available
availability_payload
prepare_warp_compatible_mjcf
make_warp_data
synchronize_warp
```

Implementation rules:

- Preserve current PR10 benchmark behavior.
- Preserve benchmark output schema unless unavoidable.
- Do not import `mujoco_warp` at module import time.
- All optional dependency imports must be lazy.
- CPU-only imports must work without `mujoco_warp`, CUDA, JAX, or MJX.
- Do not commit generated outputs, caches, benchmark logs, model files, or virtualenv artifacts.

## Dependency policy

Do not add `mujoco-warp`, CUDA, JAX, MJX, or GPU packages to `[project.dependencies]`.

Keep or reuse the optional extra:

```toml
[project.optional-dependencies]
warp = [
  "mujoco-warp",
]
```

Do not change the default install path.

## Tests

Add CPU-only tests covering:

1. `handcdo.backends.batched` imports without `mujoco_warp`.
2. `supports_batched_grasps()` returns `False` for plain objects.
3. `supports_batched_grasps()` returns `True` for a dummy object with callable `evaluate_grasps_batch`.
4. Existing CPU backend imports still work.
5. Warp utility availability checks do not raise when `mujoco_warp` is missing.
6. Existing benchmark help still works without importing `mujoco_warp`.

No GPU, network, H100, JAX, MJX, or MuJoCo Warp installation may be required for these tests.

## Validation

Run:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
```

## Out of scope

Do not implement:

- `handcdo/backends/mujoco_warp.py`;
- `mujoco_warp` backend registry alias;
- real batched grasp simulation;
- new Warp batch evaluation CLI;
- CPU-vs-Warp comparison;
- score equivalence claims;
- TPE batching;
- JAX, MJX, autodiff, Isaac Sim, RL, ROS, or Slurm production templates.

## Success criteria

This stage is successful if:

1. The batched protocol exists and is independent from `SimulatorBackend`.
2. Reusable Warp helper logic is available without optional dependency import failures.
3. PR10 benchmark behavior remains intact.
4. `pytest -q` passes in a CPU-only environment.



## Additional guardrail:
- If `handcdo/warp_utils.py` is added, keep public helper names stable and do not make later PRs depend on private benchmark helpers such as `_make_warp_data()` or `_synchronize()`.
- Add a regression test that PR10 benchmark CSV columns remain unchanged after utility extraction.