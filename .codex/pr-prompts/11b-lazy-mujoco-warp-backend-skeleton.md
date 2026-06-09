# Codex Prompt: PR11-b — Lazy MuJoCo Warp Backend Skeleton

You are working in the `Lunar-boy/handcdo_mujoco` repository.

Start from latest `main` after PR11-a has merged:

```bash
git checkout main
git pull origin main
git checkout -b pr11b-lazy-mujoco-warp-backend
```

## Goal

Add an experimental `mujoco_warp` backend skeleton and lazy registry alias.

This stage is about:

- dependency behavior;
- constructor validation;
- backend registration;
- interface shape.

It must not attempt real batched physics scoring yet.

## Required changes

Add or update:

```text
handcdo/backends/mujoco_warp.py
handcdo/backends/registry.py
tests/test_mujoco_warp_backend_optional.py
```

Do not change CPU backend behavior.

Do not change default CLI behavior.

Do not make `mujoco_warp` the default backend.

## Backend registry requirements

Update aliases so that:

```python
_BACKEND_ALIASES = {
    "mujoco": "mujoco_cpu",
    "mujoco_cpu": "mujoco_cpu",
    "mujoco_warp": "mujoco_warp",
}
```

Expected behavior:

- `get_backend("mujoco")` still returns the CPU backend.
- `get_backend("mujoco_cpu")` still returns the CPU backend.
- `get_backend("mujoco_warp")` lazily imports and returns the experimental Warp backend.
- Importing `handcdo.backends` must not import `mujoco_warp`.
- Missing `mujoco_warp` must not break CPU imports.
- Missing `mujoco_warp` should raise a clear error only when the experimental backend is explicitly requested or constructed.

## Backend skeleton

Create:

```text
handcdo/backends/mujoco_warp.py
```

Suggested class:

```python
from __future__ import annotations

from pathlib import Path

from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


class MujocoWarpUnavailableError(RuntimeError):
    pass


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
        allow_sequential_fallback: bool = False,
    ) -> None:
        ...
```

Constructor validation must happen before importing `mujoco_warp`, so CPU-only tests can validate bad values.

Validate:

- `nworld > 0`
- `nconmax is None or nconmax > 0`
- `naconmax is None or naconmax > 0`
- `njmax > 0`
- `warmup_steps >= 0`
- `capture_graph` is boolean-like
- `allow_sequential_fallback` is boolean-like

After validation, the backend may lazily check optional dependency availability through the PR11-a/PR10 utilities.

## Methods

Add method stubs:

```python
def evaluate_grasp(
    self,
    design: HandDesign,
    tool_name: str,
    grasp: GraspParams,
    config: EvaluationConfig | None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> GraspEvaluation:
    ...
```

```python
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

For this stage, these methods may raise:

```python
NotImplementedError(
    "Experimental MuJoCo Warp grasp evaluation is not implemented in PR11-b; use PR11-c/PR11-d."
)
```

Do not implement fake batching.

Do not call CPU backend and label the result as Warp.

Do not silently fall back to CPU.

## Lazy optional dependency behavior

The following must work without `mujoco_warp` installed:

```python
import handcdo.backends
from handcdo.backends import get_backend
get_backend("mujoco")
get_backend("mujoco_cpu")
```

Requesting the experimental backend may fail clearly:

```python
get_backend("mujoco_warp")
```

The error message must explain how to install the optional extra, for example:

```text
MuJoCo Warp backend requires the optional warp extra. Install with:
python3 -m pip install -e ".[warp]"
```

Do not expose a raw `ModuleNotFoundError: mujoco_warp` traceback to normal users.

## Dependency policy

Do not add `mujoco-warp`, CUDA, JAX, or GPU packages to `[project.dependencies]`.

Reuse the optional extra if present:

```toml
[project.optional-dependencies]
warp = [
  "mujoco-warp",
]
```

## Tests

Add CPU-only tests covering:

1. Importing `handcdo.backends` works without `mujoco_warp`.
2. `get_backend("mujoco")` returns CPU backend.
3. `get_backend("mujoco_cpu")` returns CPU backend.
4. `get_backend("mujoco_warp")` either returns the skeleton backend when optional dependency is available or fails clearly when absent.
5. Missing `mujoco_warp` does not break CPU backend construction.
6. Invalid constructor values are rejected before optional dependency import is required.
7. `supports_batched_grasps()` returns `True` for `MujocoWarpBackend` only if the class exposes callable `evaluate_grasps_batch`.

Make tests robust to both environments:

- `mujoco_warp` absent;
- `mujoco_warp` installed.

GPU must not be required.

## Validation

Run:

```bash
pytest -q
python3 scripts/benchmark_mujoco_warp.py --help
```

## Out of scope

Do not implement:

- real batch initialization;
- real Warp stepping for grasp scoring;
- random-grasp orchestration;
- new batch evaluation CLI;
- score aggregation;
- CPU-vs-Warp comparison;
- TPE batching;
- JAX/MJX/autodiff;
- Slurm templates.

## Success criteria

This stage is successful if:

1. CPU behavior is unchanged.
2. `mujoco_warp` is available only when explicitly requested.
3. Constructor validation is tested.
4. No optional GPU dependency is required for default tests.
