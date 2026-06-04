# PR 6 Prompt — Hybrid Tool Visual/Collision Mesh Configuration

Implement PR 6: add a hybrid tool geometry path for hammer, spoon, and knife while preserving the primitive tool path as the default and as the exact fallback.

## Goal

Prepare the codebase for real tool visual/collision mesh assets without requiring any large mesh assets to be committed.

This PR should make `geometry.tool.mode: hybrid` runnable. If no mesh assets are present, hybrid mode must fall back to the existing primitive tool geoms and keep the current evaluation/optimization semantics unchanged.

This PR is infrastructure only. It is not convex decomposition, not Isaac/URDF/USD support, and not paper-level tool asset reconstruction.

## Current repository facts

* `ToolContactConfig.mode` already accepts `primitive`, `hybrid`, and `convex_mesh`.
* `geometry_high.yaml` already requests `tool.mode: hybrid`.
* `handcdo.mjcf_generator._add_tool(...)` currently hardcodes primitive hammer, spoon, and knife geoms.
* `build_mjcf_xml(...)` currently creates the MJCF root and then the `worldbody`.
* Mesh assets must be declared under root-level `<asset>`, not under `worldbody`.
* `mujoco_eval._load_model(...)` writes generated XML to a temporary file before calling `mujoco.MjModel.from_xml_path(...)`; therefore relative mesh paths are fragile. For this PR, resolved absolute mesh paths should be used in `<mesh file="...">` whenever hybrid mesh assets are emitted.

## Required changes

### 1. Add tool geometry resolver module

Create:

```text
handcdo/tool_geometry.py
```

Define:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolGeometryAsset:
    name: str
    visual_mesh: Path | None
    collision_meshes: tuple[Path, ...]
    primitive_fallback: bool = True
```

Implement:

```python
def resolve_tool_geometry(
    tool_name: str,
    assets_dir: Path = Path("assets/tools"),
) -> ToolGeometryAsset:
    ...
```

Resolver behavior:

* Supported tool names are `hammer`, `spoon`, and `knife`.
* Unknown tool names should raise `ValueError` with a clear message.
* Search under:

```text
assets/tools/<tool_name>/
```

* Recognize optional visual mesh files with deterministic priority:

```text
visual.stl
visual.obj
tool_visual.stl
tool_visual.obj
```

* Recognize collision mesh files by deterministic sorted glob order:

```text
collision*.stl
collision*.obj
collider*.stl
collider*.obj
```

* Ignore `.gitkeep`.
* Return absolute resolved paths for any discovered mesh files.
* Do not require assets to exist.

### 2. Add placeholder asset directories only

Add:

```text
assets/tools/hammer/.gitkeep
assets/tools/spoon/.gitkeep
assets/tools/knife/.gitkeep
```

Do not add large mesh files.

### 3. Update tool geometry validation

Update `handcdo.mjcf_generator._ensure_supported_geometry_config(...)`.

Supported after this PR:

* `tool.mode == "primitive"`
* `tool.mode == "hybrid"`

Still unsupported:

* `tool.mode == "convex_mesh"` should raise `NotImplementedError`.

For a manually constructed unknown tool mode, raise `ValueError`.

Do not change supported finger or palm behavior from PR4/PR5.

### 4. Refactor MJCF asset support cleanly

Update `build_mjcf_xml(...)` to create a root-level `<asset>` element before `worldbody` when hybrid mesh assets are actually emitted.

Avoid emitting an empty `<asset>` element in default primitive mode, so default XML remains exactly unchanged.

Refactor `_add_tool(...)` so it can add mesh declarations and tool body geoms. Suggested signature:

```python
def _add_tool(
    world_parent: ET.Element,
    tool: ToolSpec,
    fixed: bool = False,
    tool_config: ToolContactConfig | None = None,
    asset_parent: ET.Element | None = None,
    tool_assets_dir: Path = Path("assets/tools"),
) -> None:
    ...
```

Alternatively, use a small internal helper object, but the implementation must keep root-level mesh declarations under `<asset>` and tool geoms under `worldbody/body`.

### 5. Preserve primitive behavior exactly

Extract the current hardcoded primitive tool geoms into a helper:

```python
def _add_primitive_tool_geoms(body: ET.Element, tool: ToolSpec, friction: tuple[float, float, float]) -> None:
    ...
```

Rules:

* `tool.mode == "primitive"` must emit exactly the same tool XML as before for the default config.
* `tool_config.friction` should override `tool.friction` when it is not `None`.
* Existing primitive geom names must remain unchanged:

  * `hammer_handle`, `hammer_head`
  * `spoon_handle`, `spoon_bowl`
  * `knife_handle`, `knife_blade`
* Unsupported tool names must still raise `ValueError`.

### 6. Implement hybrid mode semantics

For `tool.mode == "hybrid"`:

1. Resolve `ToolGeometryAsset`.
2. If no visual mesh and no collision meshes exist:

   * fall back to the primitive tool geoms;
   * emit no `<asset>` element solely for this fallback;
   * log a warning through `logging.getLogger(__name__)`;
   * do not add XML comments.
3. If a visual mesh exists:

   * add one `<mesh>` asset for the visual mesh;
   * add one visual `<geom type="mesh">` under the tool body;
   * set `contype="0"` and `conaffinity="0"` on the visual geom.
4. If collision meshes exist:

   * add one `<mesh>` asset per collision mesh;
   * add one collision `<geom type="mesh">` per collision mesh;
   * set `contype="1"` and `conaffinity="1"` on collision geoms;
   * set friction from `tool_config.friction` if provided, otherwise from `tool.friction`;
   * distribute the tool mass deterministically across collision geoms, for example `tool.mass / len(collision_meshes)`.
5. If a visual mesh exists but no collision mesh exists:

   * use primitive geoms for collision;
   * add the visual mesh as non-colliding visual geometry.
6. Mesh asset names and geom names must be deterministic and safe for MJCF:

   * `hammer_visual_mesh`
   * `hammer_collision_mesh_0`
   * `hammer_collision_0`
   * etc.

For this PR, do not attempt to infer inertia from the mesh. Keep MuJoCo `inertiafromgeom="true"` behavior.

### 7. Path handling

Because the current evaluator writes XML to a temporary file before loading it, all emitted mesh file paths must be absolute resolved paths.

For example:

```python
file=str(mesh_path.resolve())
```

Document that this is chosen for robust in-memory/temp-file MuJoCo evaluation and that a future export-oriented PR may add portable relative-path support.

### 8. Config/documentation updates

Update:

```text
docs/tool_geometry.md
docs/geometry_config.md
README.md
```

Explain:

* `tool.mode: primitive` is the default and preserves the original primitive hammer/spoon/knife models.
* `tool.mode: hybrid` can add visual/collision mesh assets when present.
* Missing assets fall back to primitive collision.
* Visual and collision meshes should be separate.
* Collision meshes should be convex or low-complexity for MuJoCo contact stability.
* This PR does not implement convex decomposition.
* This PR does not change optimization semantics.
* `configs/geometry_high.yaml` becomes runnable even when no real tool assets exist, because hybrid mode falls back to primitives.

### 9. Tests

Add:

```text
tests/test_tool_hybrid_collision.py
```

Required tests:

1. Default primitive preservation:

   * `build_mjcf_xml(hand, tool=get_tool("hammer"))`
   * equals `build_mjcf_xml(hand, tool=get_tool("hammer"), geometry_config=GeometryConfig())`.
   * XML contains `hammer_handle` and `hammer_head`.
   * XML does not contain `<asset>` solely because of default primitive mode.

2. Primitive mode for all tools:

   * hammer contains `hammer_handle`, `hammer_head`;
   * spoon contains `spoon_handle`, `spoon_bowl`;
   * knife contains `knife_handle`, `knife_blade`.

3. Hybrid missing-assets fallback:

   * use `GeometryConfig(tool=ToolContactConfig(mode="hybrid"))`;
   * point resolver/assets_dir to an empty temporary tool-assets directory or monkeypatch the default path;
   * XML contains primitive tool geoms;
   * XML does not contain `<geom type="mesh">`;
   * XML loads with MuJoCo if MuJoCo is available.

4. Hybrid visual-only behavior:

   * create a tiny temporary visual mesh file;
   * no collision mesh files;
   * XML contains a visual `<mesh>` asset and visual `<geom type="mesh" contype="0" conaffinity="0">`;
   * XML still contains primitive collision geoms.

5. Hybrid collision mesh behavior:

   * create one or two tiny temporary collision mesh files;
   * XML contains root-level `<asset><mesh ... /></asset>`;
   * XML contains `<geom type="mesh">` collision geoms;
   * collision geoms have `contype="1"` and `conaffinity="1"`;
   * mass is assigned deterministically;
   * friction uses `tool_config.friction` if provided.

6. Resolver tests:

   * unknown tool raises `ValueError`;
   * missing directories return no meshes and `primitive_fallback=True`;
   * collision mesh ordering is deterministic;
   * `.gitkeep` is ignored.

7. Validation tests:

   * `tool.mode == "hybrid"` is accepted by `build_mjcf_xml`.
   * `tool.mode == "convex_mesh"` raises `NotImplementedError`.
   * manually constructed unknown tool mode raises `ValueError`.

8. Config smoke test:

   * `configs/geometry_high.yaml` can be parsed;
   * generating hammer XML with it no longer raises solely because of `tool.mode: hybrid`;
   * with missing assets, it falls back to primitive tool geoms.

Use MuJoCo load tests only as optional smoke tests with `pytest.importorskip("mujoco")`.

### 10. Benchmark/regression note

Because this PR should not change behavior when assets are missing, document a small benchmark check:

```bash
python3 scripts/run_baseline_benchmark.py \
  --n-designs 2 \
  --n-grasp-trials 1 \
  --tools hammer \
  --seed 0 \
  --backend mujoco_cpu \
  --config configs/default_eval.yaml \
  --output-dir outputs/baselines/pr6_primitive

python3 scripts/run_baseline_benchmark.py \
  --n-designs 2 \
  --n-grasp-trials 1 \
  --tools hammer \
  --seed 0 \
  --backend mujoco_cpu \
  --config configs/geometry_high.yaml \
  --design-dir outputs/baselines/pr6_primitive/designs \
  --output-dir outputs/baselines/pr6_hybrid_fallback

python3 scripts/compare_benchmarks.py \
  --left outputs/baselines/pr6_primitive/results.csv \
  --right outputs/baselines/pr6_hybrid_fallback/results.csv \
  --output-dir outputs/baseline_compare/pr6_hybrid_fallback \
  --score-column hand_score
```

Do not commit generated benchmark outputs.

## Out of scope

* Do not add large mesh files.
* Do not implement convex decomposition.
* Do not implement Isaac Sim, URDF, USD, ROS, or fabrication export.
* Do not alter grasp optimization, hand optimization, scoring, wrench directions, or SHAP analysis.
* Do not replace primitive tool models as the default.

## Validation

Run:

```bash
pytest -q
```

Optionally run the PR6 primitive-vs-hybrid-fallback benchmark comparison above.
