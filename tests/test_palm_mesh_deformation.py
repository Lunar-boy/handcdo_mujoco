from __future__ import annotations

import json
import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.geometry_config import PalmContactConfig
from handcdo.hand_model import build_hand_model
from handcdo.palm_mesh_deformation import (
    PalmSurfaceMeshConfig,
    build_outline_palm_surface_cells,
    build_palm_surface_mesh,
    compute_palm_height_field,
    export_palm_surface_mesh,
)
from handcdo.polygon_geometry import is_point_in_convex_polygon, polygon_area


def _hand_with_height(height: float):
    design = DesignSpace().sample(seed=71)
    params = design.to_dict()
    params["palm_kernel_max_height"] = height
    return build_hand_model(HandDesign(params))


def test_outline_domain_clips_cells_to_palm_outline():
    hand = _hand_with_height(0.02)
    config = PalmContactConfig(
        mode="tiled_mesh_colliders",
        mesh_collider_domain="outline",
        mesh_collider_resolution=4,
        max_num_mesh_colliders=64,
    )

    cells = build_outline_palm_surface_cells(hand, config)
    repeated = build_outline_palm_surface_cells(hand, config)

    assert cells
    assert cells == repeated
    for cell in cells:
        assert len(cell.vertices_2d) >= 3
        assert len(set(cell.vertices_2d)) >= 3
        assert all(
            math.isfinite(coordinate)
            for vertex in cell.vertices_2d
            for coordinate in vertex
        )
        assert polygon_area(cell.vertices_2d) > 1e-12
        assert all(
            is_point_in_convex_polygon(
                vertex,
                hand.palm_body.outline_vertices_2d,
                eps=1e-9,
            )
            for vertex in cell.vertices_2d
        )


def test_zero_design_height_produces_flat_height_field_and_top_surface():
    hand = _hand_with_height(0.0)
    _, _, height = compute_palm_height_field(hand, resolution=8)
    mesh = build_palm_surface_mesh(hand, PalmSurfaceMeshConfig(resolution=8, include_skirt=False))

    assert np.array_equal(height, np.zeros_like(height))
    assert np.all(mesh.vertices[:, 2] == pytest.approx(hand.palm_size[2]))


def test_positive_design_height_increases_max_z():
    hand = _hand_with_height(0.02)
    mesh = build_palm_surface_mesh(hand, PalmSurfaceMeshConfig(resolution=16, include_skirt=False))

    assert mesh.vertices[:, 2].max() > hand.palm_size[2]


def test_max_height_cap_caps_instead_of_overriding_design_height():
    low_hand = _hand_with_height(0.005)
    _, _, low_uncapped = compute_palm_height_field(low_hand, resolution=16)
    _, _, low_high_cap = compute_palm_height_field(low_hand, resolution=16, max_height_cap=0.02)
    assert np.array_equal(low_uncapped, low_high_cap)

    high_hand = _hand_with_height(0.035)
    _, _, capped = compute_palm_height_field(high_hand, resolution=16, max_height_cap=0.01)
    assert capped.max() <= 0.01


def test_resolution_eight_has_expected_top_mesh_counts():
    hand = _hand_with_height(0.01)
    mesh = build_palm_surface_mesh(hand, PalmSurfaceMeshConfig(resolution=8, include_skirt=False))

    assert len(mesh.vertices) == 81
    assert len(mesh.faces) == 128


def test_same_design_and_config_produce_deterministic_mesh():
    hand = _hand_with_height(0.02)
    config = PalmSurfaceMeshConfig(resolution=8)
    first = build_palm_surface_mesh(hand, config)
    second = build_palm_surface_mesh(hand, config)

    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert first.is_watertight


@pytest.mark.parametrize("file_type", ["obj", "stl"])
def test_exported_mesh_can_be_reloaded(tmp_path, file_type):
    hand = _hand_with_height(0.02)
    paths = export_palm_surface_mesh(
        hand,
        tmp_path,
        config=PalmSurfaceMeshConfig(resolution=8),
        formats=(file_type,),
    )

    loaded = trimesh.load(paths[file_type], force="mesh", process=False)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.vertices) > 0
    assert len(loaded.faces) > 0
    assert metadata["note"] == "visual/export mesh only; simulation collision is unchanged"
    assert metadata["height_max"] > 0.0
