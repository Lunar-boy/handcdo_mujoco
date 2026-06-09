# Codex Prompt: PR11a — Warp Utilities and Batched Backend Protocol

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Create a new branch before making changes:

```bash
git checkout main
git pull origin main
git checkout -b pr11a-warp-utils-batched-protocol
```

## Goal

Implement only the infrastructure preparation layer for PR11.

Do not implement the actual MuJoCo Warp backend yet.
Do not change CPU MuJoCo behavior.
Do not change default CLI behavior.

This PR should be a low-risk refactor and protocol addition.

## Repository context

The repository currently has:

- CPU MuJoCo evaluation as the reference backend.
- `SimulatorBackend` with single-grasp `evaluate_grasp()`.
- A PR10 MuJoCo Warp benchmark scaffold in `handcdo/benchmarks/mujoco_warp.py`.
- Optional `warp` dependency in `pyproject.toml`.
- CPU-only default tests.

PR10 already contains useful MuJoCo Warp helper logic such as:

- availability checking;
- device probing;
- MJCF compatibility rewrite;
- `mujoco_warp.put_model` / `put_data` / `make_data` API handling;
- Warp synchronization;
- benchmark output metadata.

## Required changes

Add:

```text
handcdo/backends/batched.py
```

Optionally add:

```text
handcdo/warp_utils.py
```

Update the PR10 benchmark only if required to reuse factored helpers:

```text
handcdo/benchmarks/mujoco_warp.py
```

Add focused tests:

```text
tests/test_batched_backend_protocol.py
tests/test_warp_utils_optional.py
```

## Batched protocol

Create a separate optional protocol. Do not modify the existing `SimulatorBackend` protocol.

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
    ...
```

`supports_batched_grasps()` should return `True` only when the object exposes a callable `evaluate_grasps_batch`.

Do not force the CPU backend to implement this protocol in this stage.

## Warp utility extraction

Inspect `handcdo/benchmarks/mujoco_warp.py`.

If helper logic is duplicated or likely needed by PR11, factor reusable pieces into `handcdo/warp_utils.py`.

Candidate utilities:

```text
WarpAvailability
check_warp_available
prepare_warp_compatible_mjcf
make_warp_data
synchronize_warp
```

Rules:

- Preserve current benchmark behavior.
- Do not change benchmark output schema unless unavoidable.
- Do not import `mujoco_warp` at module import time.
- All optional dependency imports must remain lazy.
- CPU-only imports must still work without `mujoco_warp`, CUDA, JAX, or MJX.
- No generated outputs, benchmark logs, caches, virtualenvs, or model files should be committed.

## Tests

Add CPU-only tests covering:

1. `handcdo.backends.batched` imports without `mujoco_warp`.
2. `supports_batched_grasps()` returns `False` for plain objects and CPU backend objects unless CPU backend already intentionally implements batching.
3. `supports_batched_grasps()` returns `True` for a dummy object with callable `evaluate_grasps_batch`.
4. Warp utility availability checks do not raise when `mujoco_warp` is missing.
5. The existing benchmark help still works without importing `mujoco_warp`.

Use no GPU requirement and no network requirement.

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
- new Warp batch evaluation CLI;
- real batched grasp simulation;
- CPU-vs-Warp comparison;
- score equivalence claims;
- JAX, MJX, autodiff, Isaac Sim, RL, ROS, or Slurm production templates.

## Success criteria

This stage is successful if:

1. The batched protocol exists and is independent from `SimulatorBackend`.
2. Reusable Warp helper logic is available without optional dependency import failures.
3. PR10 benchmark behavior remains intact.
4. `pytest -q` passes in a CPU-only environment.
