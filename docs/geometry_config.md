# Geometry Configuration

This repository now has a typed `GeometryConfig` schema for finger, palm, and tool contact geometry settings.

The default `finger.mode: capsule` geometry emits one capsule per link and preserves the original generated MJCF.

The optional `finger.mode: capsule_tip_pad` geometry keeps every capsule link and, when `fingertip_pad_enabled: true`, adds one primitive box contact pad to each fingertip link. The pad dimensions and placement are deterministically derived from the fingertip link dimensions. Setting `capsule_tip_pad` with `fingertip_pad_enabled: false` is treated as capsule-only geometry and does not emit pad geoms.

Use `configs/geometry_tip_pad.yaml` for a runnable tip-pad configuration. Other fingertip pad shapes remain reserved by the schema but are not implemented.

This mode is a CPU-friendly MuJoCo primitive approximation for contact experiments. It is not the paper-level Gaussian surface deformation kernel or mesh collider decomposition. Current palm kernels remain approximated as simple local contact pads, and palm pad grids and hybrid tool collision remain future work.
