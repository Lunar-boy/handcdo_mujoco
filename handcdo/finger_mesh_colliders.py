from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .finger_mesh_deformation import (
    compute_fingertip_height_field,
    fingertip_contact_half_extents,
)
from .geometry_config import FingerContactConfig
from .hand_model import DigitSpec, LinkSpec
from .utils import ensure_dir


@dataclass(frozen=True)
class FingerMeshCollider:
    name: str
    mesh: trimesh.Trimesh


def build_fingertip_local_mesh_colliders(
    digit: DigitSpec,
    link: LinkSpec,
    finger_config: FingerContactConfig,
) -> list[FingerMeshCollider]:
    """Build closed convex tiles for one terminal fingertip's palmar surface."""
    _validate_finger_mesh_collider_config(finger_config)
    if not link.fingertip:
        return []

    resolution = finger_config.local_patch_resolution
    max_height = (
        finger_config.fingertip_pad_thickness
        if finger_config.local_patch_max_height is None
        else finger_config.local_patch_max_height
    )
    X, Y, H = compute_fingertip_height_field(
        link,
        resolution=resolution,
        margin_ratio=finger_config.local_patch_margin_ratio,
        max_height=max_height,
        min_height=finger_config.local_patch_min_height,
        finger_config=finger_config,
    )
    _, _, contact_half_z = fingertip_contact_half_extents(link, finger_config)
    top_z = -contact_half_z
    bottom_z = top_z + finger_config.local_patch_thickness
    colliders: list[FingerMeshCollider] = []

    for row in range(resolution):
        for col in range(resolution):
            surface_quad = np.asarray(
                [
                    (X[row, col], Y[row, col], top_z - H[row, col]),
                    (X[row, col + 1], Y[row, col + 1], top_z - H[row, col + 1]),
                    (
                        X[row + 1, col + 1],
                        Y[row + 1, col + 1],
                        top_z - H[row + 1, col + 1],
                    ),
                    (X[row + 1, col], Y[row + 1, col], top_z - H[row + 1, col]),
                ],
                dtype=float,
            )
            prefix = f"{link.name}_local_patch_r{row:02d}_c{col:02d}"
            if finger_config.local_patch_collider_type == "quad_frustum":
                colliders.append(
                    FingerMeshCollider(
                        name=prefix,
                        mesh=_build_quad_frustum(surface_quad, bottom_z),
                    )
                )
            else:
                for triangle_index, indices in enumerate(((0, 1, 2), (0, 2, 3))):
                    colliders.append(
                        FingerMeshCollider(
                            name=f"{prefix}_tri{triangle_index}",
                            mesh=_build_triangular_prism(
                                surface_quad[list(indices)],
                                bottom_z,
                            ),
                        )
                    )

    return colliders


def export_fingertip_local_mesh_colliders(
    colliders: list[FingerMeshCollider],
    output_dir: str | Path,
    file_format: str = "stl",
) -> list[Path]:
    """Export one mesh file per fingertip collider."""
    normalized_format = file_format.strip().lower().lstrip(".")
    if normalized_format not in {"obj", "stl"}:
        raise ValueError(f"Unsupported fingertip mesh collider format: {file_format!r}")
    output_path = ensure_dir(output_dir)
    paths: list[Path] = []
    for collider in colliders:
        path = output_path / f"{collider.name}.{normalized_format}"
        collider.mesh.export(path, file_type=normalized_format)
        paths.append(path)
    return paths


def _validate_finger_mesh_collider_config(finger_config: FingerContactConfig) -> None:
    resolution = finger_config.local_patch_resolution
    if resolution < 2:
        raise ValueError(f"local_patch_resolution must be >= 2; got {resolution!r}")
    if finger_config.local_patch_collider_type not in {
        "quad_frustum",
        "triangular_prism",
    }:
        raise ValueError(
            "local_patch_collider_type must be one of: quad_frustum, triangular_prism; "
            f"got {finger_config.local_patch_collider_type!r}"
        )
    if finger_config.local_patch_thickness <= 0:
        raise ValueError(
            "local_patch_thickness must be > 0; "
            f"got {finger_config.local_patch_thickness!r}"
        )
    if not 0 <= finger_config.local_patch_margin_ratio < 0.5:
        raise ValueError(
            "local_patch_margin_ratio must be >= 0 and < 0.5; "
            f"got {finger_config.local_patch_margin_ratio!r}"
        )
    collider_count = resolution**2
    if finger_config.local_patch_collider_type == "triangular_prism":
        collider_count *= 2
    if collider_count > finger_config.max_num_local_patch_colliders:
        raise ValueError(
            "fingertip collider count must not exceed max_num_local_patch_colliders; "
            f"got collider_count={collider_count!r}, "
            f"max_num_local_patch_colliders="
            f"{finger_config.max_num_local_patch_colliders!r}"
        )


def _build_quad_frustum(surface_vertices: np.ndarray, bottom_z: float) -> trimesh.Trimesh:
    bottom_vertices = surface_vertices.copy()
    bottom_vertices[:, 2] = bottom_z
    vertices = np.vstack((surface_vertices, bottom_vertices))
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
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    ).convex_hull


def _build_triangular_prism(
    surface_vertices: np.ndarray,
    bottom_z: float,
) -> trimesh.Trimesh:
    bottom_vertices = surface_vertices.copy()
    bottom_vertices[:, 2] = bottom_z
    vertices = np.vstack((surface_vertices, bottom_vertices))
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
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    ).convex_hull
