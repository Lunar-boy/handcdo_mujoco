# PR 5 Prompt — Optional Palm Pad Grid Geometry

Implement PR 5: add optional palm pad grid contact geometry for the palm.

## Context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from “Function-based Parametric Co-Design Optimization of Dexterous Hands” (arXiv:2604.27557).

The paper models palm surface geometry using parametric surface deformation kernels and decomposed contact/collision pieces. In this repository, we intentionally approximate palm surface contact using simple MuJoCo primitive geoms. Do **not** implement paper-level Gaussian surface deformation, mesh deformation, convex mesh decomposition, Isaac Sim, ROS, GPU-only simulation, MJX, MuJoCo-Warp, fabrication assets, or real robot hardware code in this PR.

PR2 introduced `GeometryConfig` and `PalmContactConfig` with `palm.mode` values including `box_pads`, `pad_grid`, and `convex_patches`. PR3 introduced baseline benchmark and comparison tools. PR4 implemented optional fingertip pads. This PR implements only the `palm.mode == "pad_grid"` primitive approximation.

## Current repository facts

- `handcdo.hand_model.PalmPad` stores `name`, `pos`, and `size`.
- `handcdo.hand_model.build_hand_model(...)` currently creates two `palm_kernel_pad_*` box pads.
- `handcdo.mjcf_generator._add_palm_geoms(...)` currently writes:
  - one `palm_geom` box;
  - all `hand.palm_pads` as box geoms.
- `handcdo.mjcf_generator._ensure_supported_geometry_config(...)` currently rejects `palm.mode != "box_pads"`.
- `PalmContactConfig` already has:
  - `mode: str = "box_pads"`
  - `pad_resolution: int = 2`
  - `pad_friction: tuple[float, float, float] = (1.4, 0.02, 0.002)`
  - `max_num_pad_geoms: int = 16`
- Default behavior must preserve the generated default MJCF XML exactly.

## Goal

Add a `pad_grid` palm contact mode controlled by `GeometryConfig.palm`.

Supported after this PR:

- `palm.mode == "box_pads"`
  - existing behavior;
  - emit existing `hand.palm_pads`;
  - default XML must remain exactly unchanged.

- `palm.mode == "pad_grid"`
  - emit a deterministic low-resolution grid of small box pads on the top surface of the palm;
  - do **not** emit the existing `palm_kernel_pad_1` / `palm_kernel_pad_2` pads in this mode;
  - keep `palm_geom` unchanged.

Still unsupported:

- `palm.mode == "convex_patches"` must raise `NotImplementedError`.

## Required implementation

### 1. Update supported geometry validation

Update `handcdo.mjcf_generator._ensure_supported_geometry_config(...)` so that `palm.mode in {"box_pads", "pad_grid"}` is accepted.

Keep `convex_patches` unsupported with a clear `NotImplementedError`.

Do not change supported tool modes in this PR. `tool.mode == "hybrid"` should remain unsupported until the later tool-collision PR.

### 2. Refactor palm geometry helpers without changing default XML

Refactor palm generation around this structure:

```python
def _add_palm_geoms(
    parent: ET.Element,
    hand: HandModel,
    palm_config: PalmContactConfig | None = None,
) -> None:
    palm_config = palm_config or PalmContactConfig()

    # Always emit the existing palm_geom first.
    ET.SubElement(
        parent,
        "geom",
        name="palm_geom",
        type="box",
        size=_vec(hand.palm_size),
        density="700",
        friction="1.2 0.02 0.002",
    )

    if palm_config.mode == "box_pads":
        _add_palm_box_pads(parent, hand, palm_config)
    elif palm_config.mode == "pad_grid":
        _add_palm_pad_grid(parent, hand, palm_config)
    elif palm_config.mode == "convex_patches":
        raise NotImplementedError(...)
    else:
        raise ValueError(...)
```

For `box_pads`, preserve the current XML exactly. In particular, do not change existing pad names, positions, sizes, density, friction, `contype`, or `conaffinity`.

### 3. Implement `_add_palm_box_pads(...)`

Move the existing loop over `hand.palm_pads` into:

```python
def _add_palm_box_pads(
    parent: ET.Element,
    hand: HandModel,
    palm_config: PalmContactConfig,
) -> None:
    ...
```

For now, keep the existing box-pad behavior exactly. `palm_config.pad_friction` does not need to alter default `box_pads` behavior in this PR, because preserving default XML is more important.

### 4. Implement `_add_palm_pad_grid(...)`

Add:

```python
def _add_palm_pad_grid(
    parent: ET.Element,
    hand: HandModel,
    palm_config: PalmContactConfig,
) -> None:
    ...
```

Behavior:

- Read `resolution = palm_config.pad_resolution`.
- Validate rather than silently clamp:
  - `resolution >= 2`
  - `resolution * resolution <= palm_config.max_num_pad_geoms`
- If invalid, raise `ValueError` with a clear message including `pad_resolution` and `max_num_pad_geoms`.
- Generate exactly `resolution * resolution` geoms.
- Name geoms exactly:

```text
palm_grid_pad_r{row}_c{col}
```

with zero-based row/column indices.

### 5. Deterministic grid geometry

Use MuJoCo box half-extents correctly.

Let:

```python
palm_half_x, palm_half_y, palm_half_z = hand.palm_size
resolution = palm_config.pad_resolution
pad_half_z = 0.0025  # 5 mm full height
```

Use a conservative inset so pads stay inside the palm footprint:

```python
margin_x = min(0.010, 0.15 * palm_half_x)
margin_y = min(0.010, 0.15 * palm_half_y)
usable_half_x = max(0.001, palm_half_x - margin_x)
usable_half_y = max(0.001, palm_half_y - margin_y)
```

For each grid cell:

```python
cell_half_x = usable_half_x / resolution
cell_half_y = usable_half_y / resolution
pad_half_x = 0.85 * cell_half_x
pad_half_y = 0.85 * cell_half_y
z = palm_half_z + pad_half_z
```

Centers should be deterministic and symmetric:

```python
x = -usable_half_x + (2 * col + 1) * cell_half_x
y = -usable_half_y + (2 * row + 1) * cell_half_y
pos = (x, y, z)
size = (pad_half_x, pad_half_y, pad_half_z)
```

This places the bottom face of each pad approximately on the palm top surface and leaves small gaps between adjacent pads.

### 6. Grid geom attributes

Each grid pad should be:

```xml
<geom
  name="palm_grid_pad_r{row}_c{col}"
  type="box"
  pos="..."
  size="..."
  density="500"
  friction="..."
  contype="1"
  conaffinity="1"
/>
```

Use:

```python
friction=_vec(palm_config.pad_friction)
```

Do not hardcode grid friction except through the dataclass default.

### 7. Config files

Update:

```text
configs/geometry_medium.yaml
```

so it is runnable after PR5:

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
    max_num_pad_geoms: 16
  tool:
    mode: primitive
```

`configs/geometry_high.yaml` may keep `palm.mode: pad_grid` with `pad_resolution: 4`, but if it still uses `tool.mode: hybrid`, it is not expected to run end-to-end until the later tool-hybrid PR.

Optionally add a minimal runnable config:

```text
configs/geometry_palm_grid.yaml
```

with default capsule fingers, `palm.mode: pad_grid`, and primitive tools.

### 8. Tests

Add:

```text
tests/test_palm_pad_grid.py
```

Required tests:

1. Default behavior:
   - `build_mjcf_xml(hand)` equals `build_mjcf_xml(hand, geometry_config=GeometryConfig())`.
   - Default XML contains `palm_kernel_pad_1` and `palm_kernel_pad_2`.
   - Default XML does not contain `palm_grid_pad_`.

2. Pad-grid behavior:
   - Build XML with:

```python
GeometryConfig(
    palm=PalmContactConfig(
        mode="pad_grid",
        pad_resolution=3,
        pad_friction=(1.6, 0.04, 0.004),
    )
)
```

   - XML contains exactly 9 `palm_grid_pad_...` geoms.
   - XML does not contain `palm_kernel_pad_1` or `palm_kernel_pad_2`.
   - Every grid pad has:
     - `type="box"`
     - `density="500"`
     - `contype="1"`
     - `conaffinity="1"`
     - `friction` equal to `_vec(palm_config.pad_friction)`.

3. Geometry placement:
   - Parse XML with `xml.etree.ElementTree`.
   - Check all grid pad positions are within palm footprint:
     - `abs(x) < hand.palm_size[0]`
     - `abs(y) < hand.palm_size[1]`
   - Check `z > hand.palm_size[2]`.
   - Check size components are positive.

4. Validation:
   - `pad_resolution=1` raises `ValueError`.
   - `pad_resolution=5` with default `max_num_pad_geoms=16` raises `ValueError`.
   - `PalmContactConfig(mode="convex_patches")` raises `NotImplementedError`.

5. Integration with fingertip pads:
   - A config with `finger.mode="capsule_tip_pad"`, `fingertip_pad_enabled=True`, and `palm.mode="pad_grid"` should emit both `_tip_pad` and `palm_grid_pad_` geoms.

6. MuJoCo load smoke test:
   - If `mujoco` is installed, assert:

```python
mujoco.MjModel.from_xml_string(xml)
```

loads for the pad-grid XML.
   - Skip gracefully if MuJoCo is unavailable.

Update any existing tests that expected `palm.mode == "pad_grid"` to raise `NotImplementedError`.

### 9. Documentation

Update `docs/geometry_config.md`.

Explain:

- `palm.mode: box_pads` is the original two-pad approximation based on `palm_kernel_pad_*`.
- `palm.mode: pad_grid` replaces those two local pads with a low-resolution grid of primitive box contact pads.
- `pad_grid` is a CPU-friendly MuJoCo approximation of palm surface contact.
- It is not the paper-level Gaussian surface deformation kernel, not mesh deformation, and not convex mesh decomposition.
- Higher `pad_resolution` can better approximate distributed palm contact but increases contact count and simulation cost.

### 10. Benchmark and regression comparison

Because PR5 changes contact geometry and increases the number of palm contact geoms, this PR should include a small benchmark/regression comparison workflow using the PR3 benchmark tools.

The goal is not to require score equality. The goal is to make the geometry-induced behavior change visible and reproducible.

After implementing PR5, run or document the following comparison:

1. Generate or reuse one deterministic design set.
2. Evaluate the same designs with the original/default geometry.
3. Evaluate the same designs with `pad_grid` geometry.
4. Compare the collected CSV files with `scripts/compare_benchmarks.py`.

Suggested commands:

```bash
python3 scripts/run_baseline_benchmark.py   --n-designs 5   --n-grasp-trials 1   --tools hammer   --seed 0   --backend mujoco_cpu   --config configs/default_eval.yaml   --output-dir outputs/baselines/pr5_default

python3 scripts/run_baseline_benchmark.py   --n-designs 5   --n-grasp-trials 1   --tools hammer   --seed 0   --backend mujoco_cpu   --config configs/geometry_medium.yaml   --design-dir outputs/baselines/pr5_default/designs   --output-dir outputs/baselines/pr5_palm_grid

python3 scripts/compare_benchmarks.py   --left outputs/baselines/pr5_default/results.csv   --right outputs/baselines/pr5_palm_grid/results.csv   --output-dir outputs/baseline_compare/pr5_palm_grid   --score-column hand_score
```

Do not commit generated benchmark outputs.

If this is too slow for CI, keep it as a documented local smoke/regression procedure. Unit tests should remain lightweight.

The PR description should briefly report whether the pad-grid smoke benchmark ran successfully and whether the comparison output was generated. It does not need to assert that palm-grid scores are better than default scores.

### 11. Out of scope

- Do not implement Gaussian kernel deformation.
- Do not implement mesh collision or convex mesh decomposition.
- Do not add finger surface patches beyond PR4 fingertip pads.
- Do not change tool geometry.
- Do not change wrench scoring.
- Do not change optimizer semantics.
- Do not change default generated MJCF XML.
- Do not add Isaac Sim, Isaac Lab, ROS, GPU-only dependencies, MJX, or MuJoCo-Warp.
- Do not commit generated outputs, logs, databases, caches, virtual environments, or large mesh assets.

## Validation

Run:

```bash
pytest -q

python3 scripts/generate_designs.py   --n-designs 1   --output-dir outputs/smoke_palm_grid   --seed 0

python3 scripts/evaluate_design_batch.py   --task-id 0   --designs-per-task 1   --design-dir outputs/smoke_palm_grid   --results-dir outputs/smoke_palm_grid_results   --config configs/geometry_medium.yaml   --tools hammer   --seed 0
```

Also run or document the benchmark/regression comparison in Section 10 when practical.
