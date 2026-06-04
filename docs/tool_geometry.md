# Tool Geometry

The default `tool.mode: primitive` preserves the original primitive hammer, spoon, and knife models and remains the standard optimization path.

`tool.mode: hybrid` optionally discovers mesh files under:

```text
assets/tools/<tool_name>/
```

Visual mesh priority is `visual.stl`, `visual.obj`, `tool_visual.stl`, then `tool_visual.obj`. Collision meshes are discovered deterministically from `collision*.stl`, `collision*.obj`, `collider*.stl`, and `collider*.obj`.

Visual and collision meshes should be separate. Visual meshes are emitted as non-colliding, massless MuJoCo geoms. Collision meshes receive the configured tool friction and split the tool mass evenly. Use convex or low-complexity collision meshes for MuJoCo contact stability.

If no meshes exist, hybrid mode logs a warning and falls back to the original primitive tool geoms. If only a visual mesh exists, it is added while primitive geoms remain as collision geometry. This keeps `configs/geometry_high.yaml` runnable without committed mesh assets and preserves current optimization semantics when assets are missing.

Generated MJCF uses resolved absolute mesh paths because evaluation writes XML to a temporary file before MuJoCo loads it. A future export-oriented change may add portable relative-path support.

This infrastructure does not implement convex decomposition, paper-level tool reconstruction, Isaac Sim, URDF, or USD support. Do not commit large mesh assets.

## Regression Check

Generated benchmark outputs under `outputs/` should not be committed.

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
