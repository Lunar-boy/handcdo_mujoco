from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    mass: float
    friction: tuple[float, float, float]
    reference_pos: tuple[float, float, float]
    reference_quat: tuple[float, float, float, float]
    force_limit: float
    torque_limit: float


TOOL_LIBRARY: dict[str, ToolSpec] = {
    "hammer": ToolSpec("hammer", 0.55, (1.1, 0.02, 0.002), (0.11, 0.0, 0.055), (1.0, 0.0, 0.0, 0.0), 18.0, 0.55),
    "spoon": ToolSpec("spoon", 0.08, (0.9, 0.015, 0.001), (0.10, 0.0, 0.052), (1.0, 0.0, 0.0, 0.0), 8.0, 0.20),
    "knife": ToolSpec("knife", 0.14, (0.8, 0.015, 0.001), (0.11, 0.0, 0.052), (1.0, 0.0, 0.0, 0.0), 10.0, 0.25),
}


def get_tool(name: str) -> ToolSpec:
    try:
        return TOOL_LIBRARY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool {name!r}; choose from {sorted(TOOL_LIBRARY)}") from exc


def tool_names() -> list[str]:
    return sorted(TOOL_LIBRARY)
