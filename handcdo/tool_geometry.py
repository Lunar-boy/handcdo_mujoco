from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_TOOLS = {"hammer", "knife", "spoon"}
VISUAL_MESH_NAMES = ("visual.stl", "visual.obj", "tool_visual.stl", "tool_visual.obj")
COLLISION_MESH_PATTERNS = ("collision*.stl", "collision*.obj", "collider*.stl", "collider*.obj")


@dataclass(frozen=True)
class ToolGeometryAsset:
    name: str
    visual_mesh: Path | None
    collision_meshes: tuple[Path, ...]
    primitive_fallback: bool = True


def resolve_tool_geometry(
    tool_name: str,
    assets_dir: Path = Path("assets/tools"),
) -> ToolGeometryAsset:
    if tool_name not in SUPPORTED_TOOLS:
        raise ValueError(f"Unknown tool {tool_name!r}; choose from {sorted(SUPPORTED_TOOLS)}")

    tool_dir = assets_dir / tool_name
    visual_mesh = next(
        (path.resolve() for name in VISUAL_MESH_NAMES if (path := tool_dir / name).is_file()),
        None,
    )
    collision_meshes = tuple(
        path.resolve()
        for pattern in COLLISION_MESH_PATTERNS
        for path in sorted(tool_dir.glob(pattern))
        if path.is_file() and path.name != ".gitkeep"
    )
    return ToolGeometryAsset(
        name=tool_name,
        visual_mesh=visual_mesh,
        collision_meshes=collision_meshes,
    )
