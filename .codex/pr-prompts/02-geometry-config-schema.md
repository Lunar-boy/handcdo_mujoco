# PR 2 Prompt — Geometry Configuration Schema

Implement PR 2: add a typed geometry configuration schema and thread it through the MJCF/evaluation call path, while preserving default MJCF output exactly.

## Context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

Do **not** add Isaac Sim, ROS, GPU-only dependencies, MJX, MuJoCo-Warp, real robot control, OptiTrack, fabrication assets, or large mesh assets in this PR.

This PR is **schema-only / plumbing-only**. It prepares future PRs for configurable finger, palm, and tool contact geometry. It must not change the default generated MJCF XML or simulation scoring behavior.

## Current repository facts

- `handcdo.mjcf_generator.build_mjcf_xml(hand, tool=None, fixed_tool=False)` currently hardcodes geometry choices.
- Finger links are generated as capsule geoms.
- Palm is generated as a box.
- Palm pads are generated as box geoms.
- Hammer/spoon/knife are generated using simple MuJoCo primitive geoms.
- `handcdo.mujoco_eval.EvaluationConfig.from_dict(...)` currently parses simulation and wrench settings only.
- Batch/Optuna config files are currently read as YAML dictionaries and passed into `EvaluationConfig.from_dict(...)`.

## Goal

Introduce a small, typed `GeometryConfig` layer that can later control tool, palm, and finger contact geometry.

The default config must preserve the current XML output byte-for-byte for equivalent calls.

## Required changes

### 1. Create a new module

Create:

```text
handcdo/geometry_config.py
```

### 2. Define frozen dataclasses

Define:

- `FingerContactConfig`
- `PalmContactConfig`
- `ToolContactConfig`
- `GeometryConfig`

Use:

```python
from __future__ import annotations
```

Suggested schema:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FingerContactConfig:
    mode: str = "capsule"  # capsule | capsule_tip_pad | local_convex_patches
    fingertip_pad_enabled: bool = False
    fingertip_pad_shape: str = "box"  # box | ellipsoid | capsule | convex_mesh
    fingertip_pad_thickness: float = 0.004
    fingertip_pad_friction: tuple[float, float, float] = (1.4, 0.03, 0.003)


@dataclass(frozen=True)
class PalmContactConfig:
    mode: str = "box_pads"  # box_pads | pad_grid | convex_patches
    pad_resolution: int = 2
    pad_friction: tuple[float, float, float] = (1.4, 0.02, 0.002)
    max_num_pad_geoms: int = 16


@dataclass(frozen=True)
class ToolContactConfig:
    mode: str = "primitive"  # primitive | hybrid | convex_mesh
    friction: tuple[float, float, float] | None = None
    collision_margin: float = 0.001


@dataclass(frozen=True)
class GeometryConfig:
    finger: FingerContactConfig = field(default_factory=FingerContactConfig)
    palm: PalmContactConfig = field(default_factory=PalmContactConfig)
    tool: ToolContactConfig = field(default_factory=ToolContactConfig)
```

### 3. Implement config parsing

Implement:

```python
GeometryConfig.from_dict(data: dict[str, Any] | None) -> GeometryConfig
```

Accepted input formats:

Canonical format:

```yaml
geometry:
  finger:
    mode: capsule
  palm:
    mode: box_pads
  tool:
    mode: primitive
```

Also support direct legacy/shortcut sections:

```yaml
finger_contact:
palm_contact:
tool_contact:
```

Optional direct aliases may also be accepted:

```yaml
finger:
palm:
tool:
```

Parsing rules:

- `None` or `{}` returns `GeometryConfig()`.
- If `geometry:` exists, parse from `data["geometry"]`.
- If both `geometry:` and direct sections exist, `geometry:` takes precedence.
- Convert friction lists to `tuple[float, float, float]`.
- Validate enum-like values:
  - `FingerContactConfig.mode in {"capsule", "capsule_tip_pad", "local_convex_patches"}`
  - `FingerContactConfig.fingertip_pad_shape in {"box", "ellipsoid", "capsule", "convex_mesh"}`
  - `PalmContactConfig.mode in {"box_pads", "pad_grid", "convex_patches"}`
  - `ToolContactConfig.mode in {"primitive", "hybrid", "convex_mesh"}`
- Validate numeric fields:
  - `fingertip_pad_thickness > 0`
  - `pad_resolution >= 1`
  - `max_num_pad_geoms >= 1`
  - `collision_margin >= 0`
- Unknown keys inside known config sections should raise a clear `ValueError` so YAML typos do not silently pass.
- Error messages should include the offending field name and value.

### 4. Update `build_mjcf_xml(...)`

Update `handcdo.mjcf_generator.build_mjcf_xml(...)`.

Add optional argument:

```python
geometry_config: GeometryConfig | None = None
```

Inside the function, normalize:

```python
geometry_config = geometry_config or GeometryConfig()
```

Pass `geometry_config.finger`, `geometry_config.palm`, and `geometry_config.tool` to helper functions where useful.

Do not change generated geometry in this PR.

With default config, XML output must be exactly identical to the current output.

### 5. Update MJCF helper functions as no-op plumbing

Update helper functions only as no-op plumbing, for example:

```python
_add_digit(..., finger_config: FingerContactConfig | None = None)
_add_tool(..., tool_config: ToolContactConfig | None = None)
```

Optionally add:

```python
_add_palm_geoms(..., palm_config: PalmContactConfig | None = None)
```

These helpers must still emit exactly the same XML for default config.

### 6. Update `write_design_model(...)`

Update `write_design_model(...)`:

```python
geometry_config: GeometryConfig | None = None
```

Pass through to `build_mjcf_xml(...)`.

### 7. Thread geometry config through evaluation path

Thread geometry config through evaluation path now, even though it is a no-op:

```python
handcdo.mujoco_eval.evaluate_grasp(
    ...,
    geometry_config: GeometryConfig | None = None,
)
```

Pass it into `build_mjcf_xml(...)`.

If backend abstraction from PR1 exists, update the backend protocol/wrapper accordingly.

If PR1 has not been applied yet, keep this change local to the existing MuJoCo path.

### 8. Thread geometry config through optimization path

Thread geometry config through hand/grasp optimization where present:

```python
optimize_grasp_for_tool(..., geometry_config: GeometryConfig | None = None)
evaluate_design(..., geometry_config: GeometryConfig | None = None)
```

In `run_optuna(...)`, parse:

```python
geometry_config = GeometryConfig.from_dict(config_data)
```

Then pass it into `evaluate_design(...)`.

In `slurm_batch.evaluate_task(...)`, parse the same config dictionary with:

```python
geometry_config = GeometryConfig.from_dict(config_data)
```

Then pass it into `evaluate_design(...)`.

Default behavior must remain unchanged when config has no `geometry` section.

### 9. Add config files

Add:

```text
configs/geometry_fast.yaml
configs/geometry_medium.yaml
configs/geometry_high.yaml
```

For this PR, all three files may be parsed, but only default/fast should be expected to preserve current behavior.

Example `configs/geometry_fast.yaml`:

```yaml
geometry:
  finger:
    mode: capsule
    fingertip_pad_enabled: false
  palm:
    mode: box_pads
    pad_resolution: 2
  tool:
    mode: primitive
```

Example `configs/geometry_medium.yaml`:

```yaml
geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
    fingertip_pad_thickness: 0.004
    fingertip_pad_friction: [1.4, 0.03, 0.003]
  palm:
    mode: pad_grid
    pad_resolution: 3
    pad_friction: [1.4, 0.02, 0.002]
  tool:
    mode: primitive
```

Example `configs/geometry_high.yaml`:

```yaml
geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
    fingertip_pad_thickness: 0.004
    fingertip_pad_friction: [1.4, 0.03, 0.003]
  palm:
    mode: pad_grid
    pad_resolution: 4
    max_num_pad_geoms: 16
  tool:
    mode: hybrid
    collision_margin: 0.001
```

Note: `medium` and `high` may request modes that are parsed but not implemented yet.

This PR should not alter geometry for those modes unless a future PR implements them.

If an unimplemented-but-valid mode is passed into `build_mjcf_xml` in this PR, raise `NotImplementedError` with a clear message, except for default modes:

- finger: `capsule`
- palm: `box_pads`
- tool: `primitive`

### 10. Add tests

Add:

```text
tests/test_geometry_config.py
```

Required tests:

- `GeometryConfig.from_dict(None) == GeometryConfig()`.
- Parse canonical `geometry.finger/palm/tool` YAML-like dict.
- Parse legacy `finger_contact/palm_contact/tool_contact` sections.
- Convert friction lists to tuples.
- Invalid mode raises `ValueError`.
- Invalid fingertip pad shape raises `ValueError`.
- Invalid numeric values raise `ValueError`.
- Unknown key inside a config section raises `ValueError`.
- `build_mjcf_xml(hand, tool=tool) == build_mjcf_xml(hand, tool=tool, geometry_config=GeometryConfig())`.
- Default XML still loads with MuJoCo if MuJoCo is installed; skip gracefully if MuJoCo is unavailable.
- `evaluate_grasp(..., geometry_config=GeometryConfig())` preserves the existing evaluation path.

### 11. Update docs

Add a short README note or:

```text
docs/geometry_config.md
```

Explain:

- This PR only adds schema and no-op plumbing.
- Full paper-level Gaussian surface deformation kernels are not implemented yet.
- Current palm kernels are still approximated as simple local contact pads.
- Future PRs will implement fingertip pads, palm pad grids, and hybrid tool collision.

## Out of scope

- Do not add fingertip pads yet.
- Do not add palm grid geometry yet.
- Do not add tool mesh/hybrid collision yet.
- Do not implement Gaussian surface deformation kernels.
- Do not change simulation scoring.
- Do not change optimizer semantics.
- Do not change default MJCF XML.
- Do not add Isaac Sim, ROS, GPU-only dependencies, MJX, or MuJoCo-Warp.
- Do not commit generated outputs, logs, databases, caches, virtual environments, or large mesh assets.

## Validation

Run:

```bash
pytest -q
python3 scripts/generate_designs.py --n-designs 1 --output-dir outputs/smoke_geom_config --seed 0
python3 scripts/evaluate_design_batch.py   --task-id 0   --designs-per-task 1   --design-dir outputs/smoke_geom_config   --results-dir outputs/smoke_geom_config_results   --config configs/geometry_fast.yaml   --tools hammer   --seed 0
```
