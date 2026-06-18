from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from .hand_model import HandModel
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

    params = hand.design.params
    design_max_height = float(params["palm_kernel_max_height"])
    max_height = design_max_height
    if config.max_height_cap is not None:
        max_height = min(design_max_height, config.max_height_cap)

    H = np.zeros_like(X, dtype=float)
    if max_height <= 0.0:
        return X, Y, H

    for index in (1, 2):
        angle = float(params[f"palm_kernel_center_angle_{index}"])
        center_radius = 0.035 + float(params[f"palm_kernel_center_offset_{index}"])
        center_x = center_radius * np.cos(angle)
        center_y = center_radius * np.sin(angle)
        spread = max(float(params[f"palm_kernel_spread_{index}"]), 1e-6)
        intensity = float(params[f"palm_kernel_intensity_ratio_{index}"])
        distance_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
        H += intensity * max_height * np.exp(-distance_sq / (2.0 * spread**2))

    return X, Y, np.clip(H, 0.0, max_height)


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
