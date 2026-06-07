# PR6 Follow-up Prompt — Validate Real Hybrid Mesh Loading and Thread `tool_assets_dir`

Implement a small follow-up commit on top of PR6.

## Goal

Strengthen PR6 hybrid tool visual/collision infrastructure by proving that hybrid mesh XML can actually load in MuJoCo with a minimal valid mesh file, and by threading `tool_assets_dir` through the evaluation path so tests and future users are not restricted to the default `assets/tools` directory.

PR6 already implemented:

- `handcdo/tool_geometry.py`
- hybrid `tool.mode`
- root-level MJCF `<asset>` emission
- primitive fallback when assets are missing
- absolute mesh paths for temporary-file MuJoCo loading
- tests for XML structure and fallback behavior

This follow-up should not change optimization semantics or default primitive behavior.

---

## Problem to fix

Current PR6 tests create empty `.obj` files for visual/collision mesh cases. Those tests verify XML structure, but they do not prove MuJoCo can load an actual mesh-based hybrid tool model.

Also, `build_mjcf_xml(...)` accepts `tool_assets_dir`, but `evaluate_grasp(...)` still calls `build_mjcf_xml(...)` without passing a tool asset directory. This makes custom tool-asset directories hard to use outside direct XML-generation tests.

---

## Required changes

### 1. Add a tiny valid OBJ mesh helper for tests

In `tests/test_tool_hybrid_collision.py`, add a helper that writes a minimal valid triangular OBJ mesh.

Suggested helper:

```python
def _write_tiny_triangle_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0.000 0.000 0.000",
                "v 0.020 0.000 0.000",
                "v 0.000 0.020 0.000",
                "f 1 2 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
```

If MuJoCo requires a non-degenerate 3D surface rather than a single triangle in the installed version, use a tiny tetrahedron-like OBJ instead:

```text
v 0.000 0.000 0.000
v 0.020 0.000 0.000
v 0.000 0.020 0.000
v 0.000 0.000 0.020
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
```

Use whichever one actually loads reliably with `mujoco.MjModel.from_xml_string(...)`.

Do not add permanent mesh assets to the repository. These meshes should be generated only in temporary test directories.

---

### 2. Replace empty mesh test files with valid mesh files

Update existing hybrid visual-only and collision mesh tests so they write valid minimal OBJ content instead of empty files.

Affected tests likely include:

- `test_hybrid_visual_only_adds_noncolliding_visual_and_primitive_collision`
- `test_hybrid_collision_meshes_are_root_assets_with_deterministic_mass_and_friction`

The existing XML-structure assertions should remain.

---

### 3. Add MuJoCo load smoke test for hybrid visual-only mode

Add a test like:

```python
def test_hybrid_visual_only_with_valid_mesh_loads_with_mujoco_when_available(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    visual = tmp_path / "hammer" / "visual.obj"
    _write_tiny_mesh_obj(visual)

    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nmesh >= 1
    assert model.ngeom > 0
```

Expected semantics:

- visual mesh is present;
- primitive collision fallback is still present because no collision meshes exist;
- MuJoCo can load the XML.

---

### 4. Add MuJoCo load smoke test for hybrid collision mesh mode

Add a test like:

```python
def test_hybrid_collision_mesh_with_valid_mesh_loads_with_mujoco_when_available(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    collision = tmp_path / "hammer" / "collision_0.obj"
    _write_tiny_mesh_obj(collision)

    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nmesh >= 1
    assert model.ngeom > 0
```

Expected semantics:

- collision mesh is present;
- primitive hammer collision geoms should not be emitted when collision meshes exist;
- MuJoCo can load the XML.

If MuJoCo rejects a flat mesh, switch the helper to the tiny tetrahedron OBJ above.

---

### 5. Thread `tool_assets_dir` through `evaluate_grasp(...)`

Update `handcdo.mujoco_eval.evaluate_grasp(...)`:

```python
def evaluate_grasp(
    design: HandDesign,
    tool_name: str,
    grasp: GraspParams,
    config: EvaluationConfig | None = None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = Path("assets/tools"),
) -> GraspEvaluation:
    ...
```

Then call:

```python
xml = build_mjcf_xml(
    build_hand_model(design),
    tool=tool,
    geometry_config=geometry_config,
    tool_assets_dir=Path(tool_assets_dir),
)
```

Use `from pathlib import Path` if not already available.

Default behavior must remain unchanged.

---

### 6. Update backend protocol and wrapper if needed

If `SimulatorBackend.evaluate_grasp(...)` currently mirrors `mujoco_eval.evaluate_grasp(...)`, update it to accept the new optional `tool_assets_dir` argument.

Update `MujocoCpuBackend.evaluate_grasp(...)` to pass `tool_assets_dir` through to `handcdo.mujoco_eval.evaluate_grasp(...)`.

Keep backward compatibility:

- existing calls without `tool_assets_dir` must still work;
- existing fake-backend tests should remain simple.

---

### 7. Thread `tool_assets_dir` through optimization path only if minimally invasive

If the existing optimization stack already carries geometry/evaluation config cleanly, add optional `tool_assets_dir` parameters to:

- `optimize_grasp_for_tool(...)`
- `evaluate_design(...)`

Then pass it through to the backend/evaluator.

Default:

```python
tool_assets_dir: str | Path = Path("assets/tools")
```

Do not add a large CLI/config refactor unless it is straightforward. The minimal acceptable result is that direct `evaluate_grasp(...)` tests can use a temporary custom tool asset directory.

---

### 8. Add a direct evaluator test using custom `tool_assets_dir`

Add a lightweight test that verifies `evaluate_grasp(...)` can receive a custom temporary tool-assets directory.

Because real simulation can be slow, keep this as a smoke test and skip gracefully if MuJoCo is unavailable.

Example shape:

```python
def test_evaluate_grasp_accepts_custom_tool_assets_dir_for_hybrid_mesh(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    collision = tmp_path / "hammer" / "collision_0.obj"
    _write_tiny_mesh_obj(collision)

    design = DesignSpace().sample(seed=...)
    grasp = GraspParams(...)
    result = evaluate_grasp(
        design,
        "hammer",
        grasp,
        config=EvaluationConfig(settle_steps=1, close_steps=1, wrench_steps=1),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    assert result.design_id == design.design_id
    assert result.tool == "hammer"
```

Use the existing canonical way to construct a minimal `GraspParams` in the repository. Keep step counts tiny so the test remains fast.

If this is too slow or too coupled, add a smaller unit test by monkeypatching `build_mjcf_xml` and asserting `tool_assets_dir` is forwarded. Prefer the real MuJoCo smoke test when available.

---

### 9. Documentation update

Update `docs/tool_geometry.md` or `docs/geometry_config.md` with one sentence:

- Hybrid mesh loading is covered by tests using temporary minimal OBJ files.
- `evaluate_grasp(...)` accepts `tool_assets_dir` for custom tool asset locations.
- Exported XML still uses absolute mesh paths for robust temporary-file MuJoCo loading and is not portable across machines.

---

## Non-goals

Do not:

- add real hammer/spoon/knife mesh assets;
- add large binary assets;
- implement convex decomposition;
- implement `convex_mesh`;
- change default primitive XML;
- change scoring, wrench directions, optimizer behavior, or benchmark schemas;
- introduce Isaac Sim, URDF, USD, ROS, or GPU dependencies.

---

## Validation

Run:

```bash
pytest -q
git diff --check
```

If MuJoCo is installed, the new smoke tests should actually load hybrid visual/collision mesh XML with `mujoco.MjModel.from_xml_string(...)`.

Expected result:

- all tests pass;
- default primitive XML remains unchanged;
- missing-assets fallback still emits no `<asset>`;
- valid temporary OBJ mesh assets produce loadable hybrid MJCF;
- `evaluate_grasp(..., tool_assets_dir=...)` works or is at least verified to forward the parameter correctly.
