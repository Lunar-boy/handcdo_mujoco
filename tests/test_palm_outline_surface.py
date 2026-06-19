from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("trimesh")

from handcdo.design_space import DesignSpace
from handcdo.geometry_config import GeometryConfig, PalmContactConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml
from handcdo.palm_mesh_colliders import build_palm_tiled_mesh_colliders
from handcdo.palm_mesh_deformation import build_outline_palm_surface_cells
from handcdo.polygon_geometry import (
    clip_convex_polygon,
    is_point_in_convex_polygon,
    polygon_area,
)


def _hand():
    return build_hand_model(DesignSpace().sample(seed=430))


def _outline_config(**overrides) -> PalmContactConfig:
    values = {
        "mode": "tiled_mesh_colliders",
        "mesh_collider_domain": "outline",
        "mesh_collider_resolution": 4,
        "mesh_collider_type": "quad_frustum",
        "mesh_collider_thickness": 0.003,
        "mesh_collider_margin_ratio": 0.0,
        "max_num_mesh_colliders": 64,
    }
    values.update(overrides)
    return PalmContactConfig(**values)


def test_convex_polygon_clipping_normalizes_orientation_and_stays_inside():
    subject = ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0))
    clockwise_clip = ((-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))

    clipped = clip_convex_polygon(subject, clockwise_clip)

    assert polygon_area(clipped) == pytest.approx(4.0)
    assert all(is_point_in_convex_polygon(vertex, clockwise_clip) for vertex in clipped)


def test_outline_surface_cells_are_clipped_and_include_boundary_polygons():
    hand = _hand()
    cells = build_outline_palm_surface_cells(hand, _outline_config())
    outline = hand.palm_body.outline_vertices_2d

    assert cells
    assert all(
        is_point_in_convex_polygon(vertex, outline, eps=1e-8)
        for cell in cells
        for vertex in cell.vertices_2d
    )
    assert any(len(cell.vertices_2d) != 4 for cell in cells)
    assert all(len(cell.heights) == len(cell.vertices_2d) for cell in cells)
    assert all(len(cell.faces) >= 8 for cell in cells)


def test_outline_colliders_do_not_extend_outside_palm_outline():
    hand = _hand()
    outline = hand.palm_body.outline_vertices_2d
    colliders = build_palm_tiled_mesh_colliders(hand, _outline_config())

    assert colliders
    assert all(collider.mesh.is_watertight for collider in colliders)
    assert all(
        is_point_in_convex_polygon(
            (float(vertex[0]), float(vertex[1])),
            outline,
            eps=1e-8,
        )
        for collider in colliders
        for vertex in collider.mesh.vertices
    )
    assert any(len(collider.mesh.vertices) > 8 for collider in colliders)


def test_outline_triangular_prisms_are_watertight_and_respect_actual_limit():
    hand = _hand()
    config = _outline_config(
        mesh_collider_type="triangular_prism",
        max_num_mesh_colliders=64,
    )
    colliders = build_palm_tiled_mesh_colliders(hand, config)

    assert len(colliders) > 2 * config.mesh_collider_resolution**2
    assert all(collider.mesh.is_watertight for collider in colliders)

    with pytest.raises(ValueError, match="collider_count"):
        build_palm_tiled_mesh_colliders(
            hand,
            _outline_config(
                mesh_collider_type="triangular_prism",
                max_num_mesh_colliders=len(colliders) - 1,
            ),
        )


def test_bbox_domain_preserves_rectangular_grid_behavior():
    hand = _hand()
    config = _outline_config(
        mesh_collider_domain="bbox",
        max_num_mesh_colliders=16,
    )

    colliders = build_palm_tiled_mesh_colliders(hand, config)

    assert len(colliders) == 16
    assert all(len(collider.mesh.vertices) == 8 for collider in colliders)
    assert any(
        not is_point_in_convex_polygon(
            (float(vertex[0]), float(vertex[1])),
            hand.palm_body.outline_vertices_2d,
        )
        for collider in colliders
        for vertex in collider.mesh.vertices
    )


def test_outline_tiled_colliders_generate_mjcf():
    hand = _hand()
    xml = build_mjcf_xml(
        hand,
        geometry_config=GeometryConfig(palm=_outline_config()),
    )
    root = ET.fromstring(xml)
    meshes = [
        mesh
        for mesh in root.findall("./asset/mesh")
        if mesh.attrib.get("name", "").startswith("palm_tile_")
    ]

    assert meshes
    assert root.find("./asset/mesh[@name='palm_body_mesh']") is not None
    assert all("vertex" in mesh.attrib and "face" in mesh.attrib for mesh in meshes)
