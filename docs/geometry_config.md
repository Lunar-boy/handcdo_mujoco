# Geometry Configuration

This repository now has a typed `GeometryConfig` schema for finger, palm, and tool contact geometry settings.

The default `finger.mode: capsule` geometry emits one capsule per link and preserves the original generated MJCF.

The optional `finger.mode: capsule_tip_pad` geometry keeps every capsule link and, when `fingertip_pad_enabled: true`, adds one primitive box contact pad to each fingertip link. The pad dimensions and placement are deterministically derived from the fingertip link dimensions. Setting `capsule_tip_pad` with `fingertip_pad_enabled: false` is treated as capsule-only geometry and does not emit pad geoms.

Use `configs/geometry_tip_pad.yaml` for a runnable tip-pad configuration. Other fingertip pad shapes remain reserved by the schema but are not implemented.

These modes are CPU-friendly MuJoCo primitive approximations for contact experiments. They are not the paper-level Gaussian surface deformation kernel, mesh deformation, or convex mesh decomposition.

The default `palm.mode: box_pads` geometry keeps the original two local `palm_kernel_pad_*` box pads. The optional `palm.mode: pad_grid` geometry replaces those two local pads with a deterministic, low-resolution grid of primitive box contact pads across the palm's top surface. Higher `pad_resolution` values can better approximate distributed palm contact, but increase contact count and simulation cost. `max_num_pad_geoms` limits the permitted grid size.

Use `configs/geometry_medium.yaml` for a runnable configuration combining fingertip pads and a palm pad grid.

The default `tool.mode: primitive` preserves the original primitive hammer, spoon, and knife models. `tool.mode: hybrid` discovers optional visual and collision meshes under `assets/tools/<tool_name>/`. Missing mesh assets fall back exactly to primitive tool geometry, so `configs/geometry_high.yaml` is runnable without real tool assets and does not change optimization semantics in that fallback case.

Hybrid visual and collision meshes should be separate. Visual meshes are non-colliding; collision meshes should be convex or low-complexity for stable MuJoCo contact. Hybrid mode does not perform convex decomposition. See [Tool Geometry](tool_geometry.md) for file naming and path details.
