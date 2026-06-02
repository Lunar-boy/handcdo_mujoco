Implement PR 4: add optional fingertip pad contact geometry.

Current repository facts:
- `handcdo.hand_model.LinkSpec` has `fingertip: bool = False`.
- In `build_hand_model`, the last link of each finger/thumb is marked as fingertip.
- Current MJCF generator writes every link as a capsule.
- Current default behavior should remain capsule-only.

Goal:
Add an optional fingertip pad geom for fingertip links, controlled by `GeometryConfig.finger`, while preserving current default behavior.

Required changes:
1. Ensure PR 2 geometry config exists. If not, implement the minimal required part:
   - `GeometryConfig`
   - `FingerContactConfig`
   - `build_mjcf_xml(..., geometry_config=None)`

2. Update `handcdo.mjcf_generator._add_digit(...)`:
   - Add optional `geometry_config` or `finger_config`.
   - Keep existing capsule geom for every link.
   - If `link.fingertip` and `finger_config.fingertip_pad_enabled` is true, add an extra pad geom attached to the same body.

3. Start with `box` pad only if needed for stability.
   - If implementing multiple shapes, support `box` and `ellipsoid`.
   - Unknown pad shape should raise `ValueError`.

4. Suggested pad placement:
   - Local x near the distal end of the fingertip link.
   - Keep dimensions conservative.
   - Use fields from `FingerContactConfig`.
   - Do not use mesh files in this PR.

5. Suggested XML:
```xml
<geom
  name="{link.name}_tip_pad"
  type="box"
  pos="..."
  size="..."
  density="400"
  friction="1.4 0.03 0.003"
  contype="1"
  conaffinity="1"
/>
```

6. Add config:
   - Update `configs/geometry_medium.yaml`:
```yaml
geometry:
  finger:
    mode: capsule_tip_pad
    fingertip_pad_enabled: true
    fingertip_pad_shape: box
    fingertip_pad_thickness: 0.004
    fingertip_pad_friction: [1.4, 0.03, 0.003]
  palm:
    mode: box_pads
  tool:
    mode: primitive
```

7. Add tests:
   - `tests/test_fingertip_pad_geometry.py`
   - Build a deterministic design.
   - Generate default XML and assert no `_tip_pad` geoms exist.
   - Generate XML with fingertip pads enabled and assert `_tip_pad` geoms exist.
   - Assert non-fingertip links do not receive tip pads.
   - If MuJoCo is installed, assert the XML loads.

8. Update README or `docs/geometry_modes.md`:
   - Explain `capsule` vs `capsule_tip_pad`.

Out of scope:
- Do not replace entire finger with mesh.
- Do not add low-poly convex finger surface pieces.
- Do not add MJX.
- Do not change scoring or optimizer behavior.

Validation:
```bash
pytest -q
python3 scripts/generate_designs.py --n-designs 1 --output-dir outputs/smoke_tip_pad --seed 0
```
