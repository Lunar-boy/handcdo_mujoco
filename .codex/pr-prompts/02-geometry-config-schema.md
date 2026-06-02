Implement PR 2: add a geometry configuration schema without changing default MJCF output.

Current repository facts:
- `handcdo.mjcf_generator.build_mjcf_xml(hand, tool=None, fixed_tool=False)` currently hardcodes geometry choices.
- Finger links are generated as capsule geoms.
- Palm is generated as a box.
- Palm pads are generated as box geoms.
- Hammer/spoon/knife are generated using simple MuJoCo primitives.
- `EvaluationConfig` currently only parses simulation and wrench settings.

Goal:
Introduce a small, typed geometry configuration layer that can later control tool, palm, and finger contact geometry. This PR should not alter default behavior.

Required changes:
1. Create a new module:
   - `handcdo/geometry_config.py`

2. Define dataclasses:
   - `FingerContactConfig`
   - `PalmContactConfig`
   - `ToolContactConfig`
   - `GeometryConfig`

3. Suggested fields:
```python
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
    finger: FingerContactConfig = FingerContactConfig()
    palm: PalmContactConfig = PalmContactConfig()
    tool: ToolContactConfig = ToolContactConfig()
```

Use `default_factory` where required by dataclasses.

4. Implement:
   - `GeometryConfig.from_dict(data: dict | None) -> GeometryConfig`
   - It should accept either:
     - root-level `geometry:`
     - or direct `finger_contact`, `palm_contact`, `tool_contact` sections.
   - Unknown values should raise clear `ValueError` only if they are used as enum-like modes.

5. Update `handcdo.mjcf_generator.build_mjcf_xml(...)`:
   - Add optional argument `geometry_config: GeometryConfig | None = None`.
   - Default should preserve current output.
   - Pass config to helper functions, but do not change geometry yet.

6. Update `write_design_model(...)`:
   - Add optional `geometry_config=None`.
   - Pass through to `build_mjcf_xml`.

7. Update `handcdo.mujoco_eval.evaluate_grasp(...)`:
   - It may still ignore geometry config in this PR unless easy to thread through.
   - If threading config, keep default identical.

8. Add config files:
   - `configs/geometry_fast.yaml`
   - `configs/geometry_medium.yaml`
   - `configs/geometry_high.yaml`

For now they may be parsed but only fast/default should produce current behavior.

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

9. Add tests:
   - `tests/test_geometry_config.py`
   - Test default config.
   - Test parsing from YAML-like dict.
   - Test invalid mode if validation is implemented.
   - Test `build_mjcf_xml(..., geometry_config=GeometryConfig())` still contains capsule fingers and box palm pads.

Out of scope:
- Do not add fingertip pads yet.
- Do not add palm grid yet.
- Do not add tool mesh/hybrid collision yet.
- Do not change simulation scoring.

Validation:
```bash
pytest -q
python3 scripts/generate_designs.py --n-designs 1 --output-dir outputs/smoke_geom_config --seed 0
```
