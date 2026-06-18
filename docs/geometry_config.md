# Geometry Configuration

This repository now has a typed `GeometryConfig` schema for finger, palm, and tool contact geometry settings.

The default `finger.mode: capsule` geometry emits one capsule per link and preserves the original generated MJCF.

The optional `finger.mode: capsule_tip_pad` geometry keeps every capsule link and, when `fingertip_pad_enabled: true`, adds one primitive box contact pad to each fingertip link. The pad dimensions and placement are deterministically derived from the fingertip link dimensions. Setting `capsule_tip_pad` with `fingertip_pad_enabled: false` is treated as capsule-only geometry and does not emit pad geoms.

Use `configs/geometry_tip_pad.yaml` for a runnable tip-pad configuration. Other fingertip pad shapes remain reserved by the schema but are not implemented.

These modes are CPU-friendly MuJoCo primitive approximations for contact experiments. They are not the paper-level Gaussian surface deformation kernel, mesh deformation, or convex mesh decomposition.

The default `palm.mode: box_pads` geometry keeps the original two local `palm_kernel_pad_*` box pads. The optional `palm.mode: pad_grid` geometry replaces those two local pads with a deterministic, low-resolution grid of primitive box contact pads across the palm's top surface. Higher `pad_resolution` values can better approximate distributed palm contact, but increase contact count and simulation cost. `max_num_pad_geoms` limits the permitted grid size.

Use `configs/geometry_medium.yaml` for a runnable configuration combining fingertip pads and a palm pad grid.

The optional `palm.mode: convex_patches` geometry converts the existing `palm_kernel_*` design parameters into a deterministic grid of local box contact patches. Patch heights sample two Gaussian-like kernel bumps across the palm surface, while `convex_patch_resolution`, `convex_patch_max_height`, `convex_patch_base_thickness`, `convex_patch_min_height`, and `convex_patch_margin_ratio` control grid density and bounds. The total patch count is fixed at `convex_patch_resolution^2` and must not exceed `max_num_pad_geoms`.

This is a CPU-only, MuJoCo-stable local convex patch approximation of paper-style palm surface deformation. It is not full mesh deformation, soft-body simulation, or a claim of numerical equivalence to the paper. The default and existing fast, medium, and high configurations remain unchanged.

`configs/geometry_palm_convex_patches.yaml` is a geometry-only reference configuration. `configs/eval_palm_convex_patches.yaml` combines the same geometry with the complete simulation and grasp settings from `configs/eval_high.yaml`; use that full config for high-fidelity re-evaluation.

For example, re-evaluate a fixed design set with the convex-patch geometry:

```bash
python3 scripts/reevaluate_designs.py \
  --design-dir outputs/designs \
  --design-ids outputs/high/design_ids.txt \
  --results-dir outputs/palm_convex_patches/results \
  --output-dir outputs/palm_convex_patches \
  --config configs/eval_palm_convex_patches.yaml \
  --fidelity palm_convex_patches \
  --tools hammer,spoon,knife \
  --backend mujoco_cpu \
  --seed 40
```

For an ablation, evaluate the same design IDs and seeds with `box_pads`, `pad_grid`, and `convex_patches`, then compare success counts, runtime, rank correlation, top-k overlap, and selected best designs. Scores from different geometry modes should be labeled and should not be treated as directly interchangeable.

## Palm surface mesh export

The optional palm surface exporter generates a static, vertex-level deformed palm top-surface mesh. It applies the same two Gaussian-kernel height-field idea used by `convex_patches`, but writes OBJ/STL visual meshes instead of changing MuJoCo collision geometry. `palm_kernel_max_height` remains the design amplitude, and an optional export cap only limits that amplitude.

```bash
python3 scripts/export_palm_surface_mesh.py \
  --design-dir outputs/designs \
  --design-ids outputs/high/design_ids.txt \
  --output-dir outputs/palm_surface_mesh_exports \
  --resolution 32 \
  --formats obj,stl
```

Outputs are written under `outputs/palm_surface_mesh_exports/<design_id>/`. The reference settings in `configs/palm_surface_mesh_export.yaml` document the available mesh options but are not wired into evaluation.

This is not runtime deformable simulation, full fabrication-ready hand generation, or a replacement for MuJoCo collision geoms. Simulation collision remains controlled exclusively by `geometry.palm.mode`, so exporting a mesh does not change evaluation scores.

The default `tool.mode: primitive` preserves the original primitive hammer, spoon, and knife models. `tool.mode: hybrid` discovers optional visual and collision meshes under `assets/tools/<tool_name>/`. Missing mesh assets fall back exactly to primitive tool geometry, so `configs/geometry_high.yaml` is runnable without real tool assets and does not change optimization semantics in that fallback case.

Hybrid visual and collision meshes should be separate. Visual meshes are non-colliding; collision meshes should be convex or low-complexity for stable MuJoCo contact. Hybrid mode does not perform convex decomposition. See [Tool Geometry](tool_geometry.md) for file naming and path details.
