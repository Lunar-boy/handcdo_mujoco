from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .geometry_config import PalmContactConfig
from .hand_model import HandModel
from .palm_mesh_deformation import (
    PalmSurfaceCell,
    build_outline_palm_surface_cells,
    compute_palm_height_field,
)
from .utils import ensure_dir


@dataclass(frozen=True)
class PalmMeshCollider:
    name: str
    mesh: trimesh.Trimesh


def build_palm_tiled_mesh_colliders(
    hand: HandModel,
    palm_config: PalmContactConfig,
) -> list[PalmMeshCollider]:
    """Build closed local palm colliders from the shared deformed height field."""
    _validate_palm_mesh_collider_config(palm_config)
    if palm_config.mesh_collider_domain == "outline":
        colliders = _build_outline_colliders(hand, palm_config)
        _validate_actual_collider_count(colliders, palm_config)
        return colliders

    colliders = _build_bbox_colliders(hand, palm_config)
    _validate_actual_collider_count(colliders, palm_config)
    return colliders


def _build_bbox_colliders(
    hand: HandModel,
    palm_config: PalmContactConfig,
) -> list[PalmMeshCollider]:
    resolution = palm_config.mesh_collider_resolution
    X, Y, H = compute_palm_height_field(
        hand,
        resolution=resolution,
        margin_ratio=palm_config.mesh_collider_margin_ratio,
    )
    top_z = float(hand.palm_size[2])
    bottom_z = top_z - palm_config.mesh_collider_thickness
    colliders: list[PalmMeshCollider] = []

    for row in range(resolution):
        for col in range(resolution):
            top_quad = np.asarray(
                [
                    (X[row, col], Y[row, col], top_z + H[row, col]),
                    (X[row, col + 1], Y[row, col + 1], top_z + H[row, col + 1]),
                    (X[row + 1, col + 1], Y[row + 1, col + 1], top_z + H[row + 1, col + 1]),
                    (X[row + 1, col], Y[row + 1, col], top_z + H[row + 1, col]),
                ],
                dtype=float,
            )
            if palm_config.mesh_collider_type == "quad_frustum":
                colliders.append(
                    PalmMeshCollider(
                        name=f"palm_tile_r{row:02d}_c{col:02d}",
                        mesh=_build_quad_frustum(top_quad, bottom_z),
                    )
                )
            else:
                triangles = ((0, 1, 2), (0, 2, 3))
                for triangle_index, indices in enumerate(triangles):
                    colliders.append(
                        PalmMeshCollider(
                            name=(
                                f"palm_tile_r{row:02d}_c{col:02d}_"
                                f"tri{triangle_index}"
                            ),
                            mesh=_build_triangular_prism(top_quad[list(indices)], bottom_z),
                        )
                    )

    return colliders


def _build_outline_colliders(
    hand: HandModel,
    palm_config: PalmContactConfig,
) -> list[PalmMeshCollider]:
    colliders: list[PalmMeshCollider] = []
    for cell in build_outline_palm_surface_cells(hand, palm_config):
        if palm_config.mesh_collider_type == "quad_frustum":
            colliders.append(
                PalmMeshCollider(
                    name=f"palm_tile_r{cell.row:02d}_c{cell.col:02d}",
                    mesh=_mesh_from_cell(cell),
                )
            )
            continue
        for triangle_index in range(1, len(cell.top_vertices) - 1):
            indices = (0, triangle_index, triangle_index + 1)
            top_triangle = np.asarray(
                [cell.top_vertices[index] for index in indices],
                dtype=float,
            )
            colliders.append(
                PalmMeshCollider(
                    name=(
                        f"palm_tile_r{cell.row:02d}_c{cell.col:02d}_"
                        f"tri{triangle_index - 1}"
                    ),
                    mesh=_build_triangular_prism(top_triangle, cell.base_z),
                )
            )
    return colliders


def _mesh_from_cell(cell: PalmSurfaceCell) -> trimesh.Trimesh:
    vertices = np.asarray(cell.top_vertices + cell.bottom_vertices, dtype=float)
    faces = np.asarray(cell.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def export_palm_tiled_mesh_colliders(
    colliders: list[PalmMeshCollider],
    output_dir: str | Path,
    file_format: str = "stl",
) -> list[Path]:
    """Export one mesh file per local collider."""
    normalized_format = file_format.strip().lower().lstrip(".")
    if normalized_format not in {"obj", "stl"}:
        raise ValueError(f"Unsupported palm mesh collider format: {file_format!r}")
    output_path = ensure_dir(output_dir)
    paths: list[Path] = []
    for collider in colliders:
        path = output_path / f"{collider.name}.{normalized_format}"
        collider.mesh.export(path, file_type=normalized_format)
        paths.append(path)
    return paths


def _validate_palm_mesh_collider_config(palm_config: PalmContactConfig) -> None:
    resolution = palm_config.mesh_collider_resolution
    if resolution < 2:
        raise ValueError(f"mesh_collider_resolution must be >= 2; got {resolution!r}")
    if palm_config.mesh_collider_type not in {"quad_frustum", "triangular_prism"}:
        raise ValueError(
            "mesh_collider_type must be one of: quad_frustum, triangular_prism; "
            f"got {palm_config.mesh_collider_type!r}"
        )
    if palm_config.mesh_collider_domain not in {"bbox", "outline"}:
        raise ValueError(
            "mesh_collider_domain must be one of: bbox, outline; "
            f"got {palm_config.mesh_collider_domain!r}"
        )
    if palm_config.mesh_collider_thickness <= 0:
        raise ValueError(
            "mesh_collider_thickness must be > 0; "
            f"got {palm_config.mesh_collider_thickness!r}"
        )
    if not 0 <= palm_config.mesh_collider_margin_ratio < 0.5:
        raise ValueError(
            "mesh_collider_margin_ratio must be >= 0 and < 0.5; "
            f"got {palm_config.mesh_collider_margin_ratio!r}"
        )
    if palm_config.max_num_mesh_colliders <= 0:
        raise ValueError(
            "max_num_mesh_colliders must be > 0; "
            f"max_num_mesh_colliders={palm_config.max_num_mesh_colliders!r}"
        )


def _validate_actual_collider_count(
    colliders: list[PalmMeshCollider],
    palm_config: PalmContactConfig,
) -> None:
    if len(colliders) > palm_config.max_num_mesh_colliders:
        raise ValueError(
            "tiled palm collider count must not exceed max_num_mesh_colliders; "
            f"got collider_count={len(colliders)!r}, "
            f"max_num_mesh_colliders={palm_config.max_num_mesh_colliders!r}"
        )


def _build_quad_frustum(top_vertices: np.ndarray, bottom_z: float) -> trimesh.Trimesh:
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = bottom_z
    vertices = np.vstack((top_vertices, bottom_vertices))
    faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _build_triangular_prism(top_vertices: np.ndarray, bottom_z: float) -> trimesh.Trimesh:
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = bottom_z
    vertices = np.vstack((top_vertices, bottom_vertices))
    faces = np.asarray(
        [
            (0, 1, 2),
            (3, 5, 4),
            (0, 3, 4),
            (0, 4, 1),
            (1, 4, 5),
            (1, 5, 2),
            (2, 5, 3),
            (2, 3, 0),
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
