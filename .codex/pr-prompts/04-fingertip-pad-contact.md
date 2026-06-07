# PR 4 Prompt — Optional Fingertip Pad Contact Geometry

Implement PR 4: add optional primitive fingertip pad contact geometry for fingertip links.

## Context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

The paper models hand morphology as a searchable design space including finger kinematics, fingertip geometry, palm/finger surface geometry, and contact-relevant fine geometry. In this repository, we intentionally approximate those fine geometries using simple MuJoCo primitives rather than paper-level deformation kernels or large mesh assets.

Do **not** add Isaac Sim, ROS, GPU-only dependencies, MJX, MuJoCo-Warp, real robot control, OptiTrack, fabrication assets, or large mesh assets in this PR.

## Current repository facts

- `handcdo.hand_model.LinkSpec` has `fingertip: bool = False`.
- `handcdo.hand_model.build_hand_model(...)` marks only the last link of each finger/thumb as `fingertip=True`.
- `handcdo.mjcf_generator._add_digit(...)` currently writes every link as a capsule geom.
- `handcdo.mjcf_generator._ensure_supported_geometry_config(...)` currently rejects `finger.mode != "capsule"` and rejects `fingertip_pad_enabled=True`.
- `GeometryConfig`, `FingerContactConfig`, `PalmContactConfig`, and `ToolContactConfig` already exist from PR2.
- Default behavior must remain capsule-only and must preserve generated default MJCF XML exactly.

## Goal

Add an optional fingertip pad geom for fingertip links, controlled by `GeometryConfig.finger`, while preserving default behavior.

This PR implements the first contact-geometry approximation for fingertips:

- default: capsule-only finger links;
- non-default: capsule link plus an extra primitive `_tip_pad` geom on each fingertip link body.

## Required behavior

### 1. Supported finger modes

Update `handcdo.mjcf_generator._ensure_supported_geometry_config(...)`.

Supported after this PR:

- `finger.mode == "capsule"` with `fingertip_pad_enabled == False`
  - current default behavior;
  - must produce exactly the same XML as before.

- `finger.mode == "capsule_tip_pad"` with `fingertip_pad_enabled == True`
  - generate normal capsule link geoms;
  - additionally generate one `_tip_pad` geom only for links where `link.fingertip is True`.

Still unsupported:

- `finger.mode == "local_convex_patches"` should raise `NotImplementedError`.
- `palm.mode == "pad_grid"` and tool hybrid/mesh modes should remain unsupported until their later PRs unless already implemented.

For `finger.mode == "capsule_tip_pad"` with `fingertip_pad_enabled == False`, prefer treating it as capsule-only and document this behavior. Do not emit `_tip_pad` geoms.

### 2. Update `_add_digit(...)`

Update:

```python
_add_digit(parent: ET.Element, digit: DigitSpec, finger_config: FingerContactConfig | None = None) -> None
```

Rules:

- Keep the existing capsule geom for every link.
- If `finger_config is None`, use `FingerContactConfig()`.
- If `link.fingertip` and `finger_config.fingertip_pad_enabled` is true, add an extra geom attached to the same body as the capsule.
- Add the pad geom before creating the child `{link.name}_tip` body.
- Do not add pads to non-fingertip links.

### 3. Implement helper for pad geom

Add a small helper, for example:

```python
def _add_fingertip_pad_geom(parent: ET.Element, link: LinkSpec, finger_config: FingerContactConfig) -> None:
    ...
```

Use only fields already present in `FingerContactConfig`.

Supported shapes in this PR:

- required: `box`;
- optional: `ellipsoid`.

Do not implement `capsule` or `convex_mesh` in this PR. These are valid schema values from PR2 but should raise `NotImplementedError` during MJCF generation if requested.

If a manually constructed config contains a truly unknown shape, raise `ValueError`.

### 4. Deterministic pad placement

Use deterministic, conservative geometry derived from the link dimensions.

For a box pad:

```python
thickness = finger_config.fingertip_pad_thickness
pad_half_x = min(0.008, max(0.003, 0.28 * link.length))
pad_half_y = max(0.003, 0.75 * link.radius)
pad_half_z = 0.5 * thickness

pos = (
    max(0.0, link.length - pad_half_x),
    0.0,
    -(link.radius + pad_half_z * 0.5),
)
size = (pad_half_x, pad_half_y, pad_half_z)
```

Rationale:

- local x is near the distal end of the capsule;
- local z offset places the pad on the palmar/closing side of the fingertip approximation;
- dimensions scale with link length/radius and remain conservative;
- MuJoCo `box` size values are half-extents.

Use:

```xml
<geom
  name="{link.name}_tip_pad"
  type="box"
  pos="..."
  size="..."
  density="400"
  friction="..."
  contype="1"
  conaffinity="1"
/>
```

Set friction using:

```python
friction=_vec(finger_config.fingertip_pad_friction)
```

Do not hardcode friction in the helper except as the dataclass default.

For an optional ellipsoid implementation, use MuJoCo `type="ellipsoid"` with `size` interpreted as radii. If not implemented, raise `NotImplementedError`.

### 5. Config files

Add a runnable config that does not depend on later PRs:

```text
configs/geometry_tip_pad.yaml
```

with:

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

Do not rely on `configs/geometry_medium.yaml` for this PR's smoke test if it still contains `palm.mode: pad_grid`, because `pad_grid` belongs to a later PR and may still raise `NotImplementedError` before PR5.

### 6. Tests

Add:

```text
tests/test_fingertip_pad_geometry.py
```

Required tests:

1. Build a deterministic design with `DesignSpace().sample(seed=...)`.

2. Generate XML with default `GeometryConfig()` and assert:
   - no `_tip_pad` appears;
   - XML equals the default call without `geometry_config`.

3. Generate XML with:

```python
GeometryConfig(
    finger=FingerContactConfig(
        mode="capsule_tip_pad",
        fingertip_pad_enabled=True,
        fingertip_pad_shape="box",
    )
)
```

and assert:
   - `_tip_pad` geoms exist;
   - number of `_tip_pad` geoms equals `sum(link.fingertip for digit in hand.digits for link in digit.links)`;
   - every `_tip_pad` name corresponds to a fingertip link name;
   - no non-fingertip link receives a `_tip_pad`.

4. Parse XML with `xml.etree.ElementTree` and assert each pad geom has:
   - `type="box"`;
   - `density="400"`;
   - `contype="1"`;
   - `conaffinity="1"`;
   - friction equal to the config value.

5. If MuJoCo is installed, assert the tip-pad XML loads:

```python
mujoco.MjModel.from_xml_string(xml)
```

6. Add tests for unsupported shapes:
   - `capsule` and `convex_mesh` raise `NotImplementedError` during MJCF generation.
   - a manually constructed unknown shape raises `ValueError`.

7. Existing PR2 tests that expect future modes to raise `NotImplementedError` must be updated carefully:
   - `capsule_tip_pad + enabled=True` should no longer raise;
   - palm `pad_grid` and tool `hybrid` should still raise if not yet implemented.

### 7. Documentation

Update `docs/geometry_config.md` or README.

Explain:

- `capsule`: original default finger geometry, one capsule per link.
- `capsule_tip_pad`: keeps the capsule and adds one primitive contact pad on each fingertip link.
- This is not the paper-level Gaussian surface deformation kernel or mesh collider decomposition.
- This mode is intended as a CPU-friendly MuJoCo primitive approximation for contact experiments.

### 8. Out of scope

- Do not replace entire fingers with mesh.
- Do not add low-poly convex finger surface pieces.
- Do not implement Gaussian deformation kernels.
- Do not add MJX, MuJoCo-Warp, Isaac Sim, ROS, or GPU-only dependencies.
- Do not change scoring or optimizer semantics.
- Do not change default generated MJCF XML.
- Do not commit generated outputs, logs, databases, caches, virtual environments, or mesh assets.

## Validation

Run:

```bash
pytest -q

python3 scripts/generate_designs.py \
  --n-designs 1 \
  --output-dir outputs/smoke_tip_pad \
  --seed 0

python3 scripts/evaluate_design_batch.py \
  --task-id 0 \
  --designs-per-task 1 \
  --design-dir outputs/smoke_tip_pad \
  --results-dir outputs/smoke_tip_pad_results \
  --config configs/geometry_tip_pad.yaml \
  --tools hammer \
  --seed 0
```
