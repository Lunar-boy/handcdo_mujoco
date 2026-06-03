# Geometry Configuration

This repository now has a typed `GeometryConfig` schema for finger, palm, and tool contact geometry settings.

This is schema and plumbing only. The current MJCF generator still emits the same capsule finger links, box palm, box palm pads, and primitive tool collision geometry for the default configuration.

Full paper-level Gaussian surface deformation kernels are not implemented yet. Current palm kernels remain approximated as simple local contact pads.

Future PRs are expected to implement fingertip pads, palm pad grids, and hybrid tool collision. Until then, valid non-default geometry modes parse successfully but raise `NotImplementedError` when used for MJCF generation.
