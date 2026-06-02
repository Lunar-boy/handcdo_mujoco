Implement PR 6: add a hybrid tool collision configuration while preserving primitive fallback.

Current repository facts:
- `_add_tool(...)` in `handcdo.mjcf_generator` currently hardcodes hammer, spoon, and knife using primitive MuJoCo geoms.
- README states primitive hammer, spoon, and knife models are placeholders for simulation infrastructure tests.
- The repo likely does not yet contain real tool mesh assets.

Goal:
Prepare the codebase for tool visual/collision mesh assets, but keep current primitive tool generation as the default and as a fallback.

Required changes:
1. Create or extend module:
   - `handcdo/tool_geometry.py` or extend `handcdo/tools.py`

2. Add a small dataclass:
```python
@dataclass(frozen=True)
class ToolGeometryAsset:
    name: str
    visual_mesh: Path | None
    collision_meshes: tuple[Path, ...]
    primitive_fallback: bool = True
```

3. Add a resolver:
```python
resolve_tool_geometry(tool_name: str, assets_dir: Path = Path("assets/tools")) -> ToolGeometryAsset
```

4. Update `_add_tool(...)`:
   - Accept `tool_config` or `geometry_config`.
   - `mode: primitive`: use current behavior exactly.
   - `mode: hybrid`: if collision mesh assets exist, add them as mesh assets/geoms; otherwise fallback to primitive and optionally add a comment in XML or log a warning.
   - Do not require mesh assets to exist for tests to pass.

5. If adding MJCF mesh support:
   - Add `<asset><mesh ... /></asset>` properly.
   - Make sure relative paths are valid from the generated XML location or use absolute paths carefully.
   - Keep mass/friction assigned.
   - Avoid non-convex/high-resolution mesh assumptions.

6. Add placeholder directory structure with `.gitkeep`, not large assets:
```text
assets/tools/hammer/.gitkeep
assets/tools/spoon/.gitkeep
assets/tools/knife/.gitkeep
```

7. Add tests:
   - Default primitive XML still contains `hammer_handle`, `hammer_head`, etc.
   - Hybrid mode with missing assets falls back to primitive.
   - Unsupported tool still raises `ValueError`.
   - If a tiny temporary mesh is used in tests, ensure MJCF XML includes `<mesh>` and `<geom type="mesh">`.

8. Update docs:
   - `docs/tool_geometry.md`
   - Explain `primitive` vs `hybrid`.
   - Explain that visual meshes and collision meshes should be separate.

Out of scope:
- Do not add large mesh files.
- Do not implement convex decomposition.
- Do not implement Isaac/URDF/USD.
- Do not alter optimization semantics.

Validation:
```bash
pytest -q
```
