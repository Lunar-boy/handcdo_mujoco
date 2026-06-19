from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from .geometry_config import PalmContactConfig
from .hand_model import HandModel
from .polygon_geometry import clip_convex_polygon, polygon_area
from .utils import ensure_dir, write_json


@dataclass(frozen=True)
class PalmSurfaceMeshConfig:
    resolution: int = 32
    margin_ratio: float = 0.0
    include_skirt: bool = True
    skirt_depth: float = 0.003
    max_height_cap: float | None = None

    def __post_init__(self) -> None:
        if self.resolution < 1:
            raise ValueError(f"resolution must be >= 1; got {self.resolution!r}")
        if not 0.0 <= self.margin_ratio < 1.0:
            raise ValueError(f"margin_ratio must be in [0, 1); got {self.margin_ratio!r}")
        if self.skirt_depth <= 0.0:
            raise ValueError(f"skirt_depth must be > 0; got {self.skirt_depth!r}")
        if self.max_height_cap is not None and self.max_height_cap < 0.0:
            raise ValueError(f"max_height_cap must be >= 0; got {self.max_height_cap!r}")


@dataclass(frozen=True)
class PalmSurfaceCell:
    row: int
    col: int
    vertices_2d: tuple[tuple[float, float], ...]
    heights: tuple[float, ...]
    base_z: float
    top_vertices: tuple[tuple[float, float, float], ...]
    bottom_vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


def compute_palm_height_field(
    hand: HandModel,
    resolution: int,
    margin_ratio: float = 0.0,
    max_height_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, and Gaussian-kernel height grids for the palm top surface."""
    config = PalmSurfaceMeshConfig(
        resolution=resolution,
        margin_ratio=margin_ratio,
        max_height_cap=max_height_cap,
    )
    palm_half_x, palm_half_y, _ = hand.palm_size
    usable_half_x = palm_half_x * (1.0 - config.margin_ratio)
    usable_half_y = palm_half_y * (1.0 - config.margin_ratio)
    x = np.linspace(-usable_half_x, usable_half_x, config.resolution + 1)
    y = np.linspace(-usable_half_y, usable_half_y, config.resolution + 1)
    X, Y = np.meshgrid(x, y)

    H = _evaluate_palm_heights(hand, X, Y, config.max_height_cap)
    return X, Y, H


def build_outline_palm_surface_cells(
    hand: HandModel,
    palm_config: PalmContactConfig,
) -> tuple[PalmSurfaceCell, ...]:
    """Build closed, height-deformed grid cells clipped to the palm outline."""
    resolution = palm_config.mesh_collider_resolution
    outline = hand.palm_body.outline_vertices_2d
    min_x = min(x for x, _ in outline)
    max_x = max(x for x, _ in outline)
    min_y = min(y for _, y in outline)
    max_y = max(y for _, y in outline)
    margin = palm_config.mesh_collider_margin_ratio
    margin_x = 0.5 * (max_x - min_x) * margin
    margin_y = 0.5 * (max_y - min_y) * margin
    x_values = np.linspace(min_x + margin_x, max_x - margin_x, resolution + 1)
    y_values = np.linspace(min_y + margin_y, max_y - margin_y, resolution + 1)
    top_z = float(hand.palm_body.half_extents[2])
    bottom_z = top_z - palm_config.mesh_collider_thickness
    cells: list[PalmSurfaceCell] = []
    eps = 1e-12

    for row in range(resolution):
        for col in range(resolution):
            rectangle = (
                (float(x_values[col]), float(y_values[row])),
                (float(x_values[col + 1]), float(y_values[row])),
                (float(x_values[col + 1]), float(y_values[row + 1])),
                (float(x_values[col]), float(y_values[row + 1])),
            )
            vertices_2d = clip_convex_polygon(rectangle, outline)
            if len(vertices_2d) < 3 or abs(polygon_area(vertices_2d)) <= eps:
                continue
            x = np.asarray([vertex[0] for vertex in vertices_2d], dtype=float)
            y = np.asarray([vertex[1] for vertex in vertices_2d], dtype=float)
            heights_array = _evaluate_palm_heights(hand, x, y, None)
            heights = tuple(float(value) for value in heights_array)
            top_vertices = tuple(
                (point[0], point[1], top_z + height)
                for point, height in zip(vertices_2d, heights, strict=True)
            )
            bottom_vertices = tuple(
                (point[0], point[1], bottom_z) for point in vertices_2d
            )
            cells.append(
                PalmSurfaceCell(
                    row=row,
                    col=col,
                    vertices_2d=vertices_2d,
                    heights=heights,
                    base_z=bottom_z,
                    top_vertices=top_vertices,
                    bottom_vertices=bottom_vertices,
                    faces=_closed_prism_faces(len(vertices_2d)),
                )
            )
    return tuple(cells)


def _evaluate_palm_heights(
    hand: HandModel,
    X: np.ndarray,
    Y: np.ndarray,
    max_height_cap: float | None,
) -> np.ndarray:
    params = hand.design.params
    design_max_height = float(params["palm_kernel_max_height"])
    max_height = design_max_height
    if max_height_cap is not None:
        max_height = min(design_max_height, max_height_cap)
    height = np.zeros_like(X, dtype=float)
    if max_height <= 0.0:
        return height
    for index in (1, 2):
        angle = float(params[f"palm_kernel_center_angle_{index}"])
        center_radius = 0.035 + float(params[f"palm_kernel_center_offset_{index}"])
        center_x = center_radius * np.cos(angle)
        center_y = center_radius * np.sin(angle)
        spread = max(float(params[f"palm_kernel_spread_{index}"]), 1e-6)
        intensity = float(params[f"palm_kernel_intensity_ratio_{index}"])
        distance_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
        height += intensity * max_height * np.exp(-distance_sq / (2.0 * spread**2))
    return np.clip(height, 0.0, max_height)


def _closed_prism_faces(vertex_count: int) -> tuple[tuple[int, int, int], ...]:
    faces: list[tuple[int, int, int]] = []
    for index in range(1, vertex_count - 1):
        faces.append((0, index, index + 1))
        faces.append(
            (vertex_count, vertex_count + index + 1, vertex_count + index)
        )
    for index in range(vertex_count):
        next_index = (index + 1) % vertex_count
        faces.extend(
            (
                (index, vertex_count + index, vertex_count + next_index),
                (index, vertex_count + next_index, next_index),
            )
        )
    return tuple(faces)


def build_palm_surface_mesh(
    hand: HandModel,
    config: PalmSurfaceMeshConfig | None = None,
) -> trimesh.Trimesh:
    """Build a static visual mesh of the Gaussian-deformed palm top surface."""
    config = config or PalmSurfaceMeshConfig()
    X, Y, H = compute_palm_height_field(
        hand,
        resolution=config.resolution,
        margin_ratio=config.margin_ratio,
        max_height_cap=config.max_height_cap,
    )
    top_z = float(hand.palm_size[2])
    vertices = np.column_stack((X.ravel(), Y.ravel(), top_z + H.ravel())).tolist()
    faces: list[tuple[int, int, int]] = []
    width = config.resolution + 1

    for row in range(config.resolution):
        for col in range(config.resolution):
            lower_left = row * width + col
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    if config.include_skirt:
        perimeter = _top_perimeter_indices(config.resolution)
        bottom_z = top_z - config.skirt_depth
        bottom_start = len(vertices)
        vertices.extend((vertices[index][0], vertices[index][1], bottom_z) for index in perimeter)
        bottom_center = len(vertices)
        vertices.append((0.0, 0.0, bottom_z))

        for offset, top_index in enumerate(perimeter):
            next_offset = (offset + 1) % len(perimeter)
            next_top_index = perimeter[next_offset]
            bottom_index = bottom_start + offset
            next_bottom_index = bottom_start + next_offset
            faces.extend(
                (
                    (top_index, bottom_index, next_bottom_index),
                    (top_index, next_bottom_index, next_top_index),
                    (bottom_center, next_bottom_index, bottom_index),
                )
            )

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def export_palm_surface_mesh(
    hand: HandModel,
    output_dir: str | Path,
    config: PalmSurfaceMeshConfig | None = None,
    formats: tuple[str, ...] = ("obj", "stl"),
) -> dict[str, Path]:
    """Export a deformed palm visual mesh and JSON metadata."""
    config = config or PalmSurfaceMeshConfig()
    normalized_formats = tuple(dict.fromkeys(item.strip().lower().lstrip(".") for item in formats))
    unsupported = [item for item in normalized_formats if item not in {"obj", "stl"}]
    if not normalized_formats:
        raise ValueError("At least one export format is required")
    if unsupported:
        raise ValueError(f"Unsupported palm surface mesh format(s): {', '.join(unsupported)}")

    output_path = ensure_dir(output_dir)
    mesh = build_palm_surface_mesh(hand, config)
    exported: dict[str, Path] = {}
    for file_type in normalized_formats:
        path = output_path / f"palm_surface_visual.{file_type}"
        mesh.export(path, file_type=file_type)
        exported[file_type] = path

    _, _, height = compute_palm_height_field(
        hand,
        resolution=config.resolution,
        margin_ratio=config.margin_ratio,
        max_height_cap=config.max_height_cap,
    )
    metadata_path = output_path / "palm_surface_metadata.json"
    metadata = {
        "type": "palm_surface_visual_mesh",
        **asdict(config),
        "design_id": hand.design.design_id,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "palm_kernel_max_height": float(hand.design.params["palm_kernel_max_height"]),
        "formats": list(normalized_formats),
        "note": "visual/export mesh only; simulation collision is unchanged",
    }
    write_json(metadata_path, metadata)
    exported["metadata"] = metadata_path
    return exported


def _top_perimeter_indices(resolution: int) -> list[int]:
    width = resolution + 1
    bottom = list(range(width))
    right = [row * width + resolution for row in range(1, width)]
    top = [resolution * width + col for col in range(resolution - 1, -1, -1)]
    left = [row * width for row in range(resolution - 1, 0, -1)]
    return bottom + right + top + left
