from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.geometry_config import GeometryConfig, PalmContactConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml
from handcdo.palm_outline import build_palm_outline_body


def _design_with(**overrides: float) -> HandDesign:
    params = DesignSpace().sample(seed=401).to_dict()
    params.update(overrides)
    return HandDesign(params)


def _digit(hand, name: str):
    return next(digit for digit in hand.digits if digit.name == name)


def test_palm_body_mesh_has_valid_closed_structure():
    palm = build_palm_outline_body(
        n_fingers=3,
        finger_side_offsets=(0.0, 0.0, 0.0),
        finger_normal_offsets=(0.0, 0.0, 0.0),
        finger_angles=(0.0, 0.0, 0.0),
        thumb_side_offset=0.0,
        thumb_normal_offset=0.0,
        thumb_angle=0.0,
    )

    assert len(palm.outline_vertices_2d) >= 4
    assert len(palm.vertices) == 2 * len(palm.outline_vertices_2d)
    assert len(palm.faces) >= 4
    assert all(math.isfinite(value) for vertex in palm.vertices for value in vertex)
    assert all(0 <= index < len(palm.vertices) for face in palm.faces for index in face)
    assert all(len(set(face)) == 3 for face in palm.faces)

    edge_counts: dict[tuple[int, int], int] = {}
    for face in palm.faces:
        for start, end in zip(face, face[1:] + face[:1], strict=True):
            edge = tuple(sorted((start, end)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert set(edge_counts.values()) == {2}


def test_palm_kernel_height_does_not_change_rigid_body_size():
    low = build_hand_model(_design_with(palm_kernel_max_height=0.0))
    high = build_hand_model(_design_with(palm_kernel_max_height=0.035))

    assert low.palm_body.half_extents == pytest.approx(high.palm_body.half_extents)
    assert low.palm_size == pytest.approx((0.085, 0.115, 0.032))


def test_finger_base_frame_is_outline_bound_and_responds_to_offsets():
    baseline = build_hand_model(
        _design_with(finger_side_offset_1=0.0, finger_normal_offset_1=0.0)
    )
    shifted = build_hand_model(
        _design_with(finger_side_offset_1=0.02, finger_normal_offset_1=0.01)
    )
    frame = baseline.palm_body.base_frames["finger1"]

    assert _digit(baseline, "finger1").base_pos != _digit(shifted, "finger1").base_pos
    assert frame.anchor_2d[0] == pytest.approx(baseline.palm_size[0])
    assert frame.pos[0] == pytest.approx(0.038)
    assert abs(frame.pos[1]) > 0.02


def test_thumb_base_frame_is_outline_bound_and_responds_to_offsets():
    baseline = build_hand_model(
        _design_with(thumb_side_offset=0.0, thumb_normal_offset=0.0)
    )
    shifted = build_hand_model(
        _design_with(thumb_side_offset=0.02, thumb_normal_offset=0.01)
    )
    frame = baseline.palm_body.base_frames["thumb"]

    assert _digit(baseline, "thumb").base_pos != _digit(shifted, "thumb").base_pos
    assert frame.anchor_2d[1] == pytest.approx(-baseline.palm_size[1])
    assert frame.pos[:2] == pytest.approx((-0.022, -0.068))


def test_mjcf_contains_outline_palm_mesh_asset_and_geom():
    root = ET.fromstring(build_mjcf_xml(build_hand_model(_design_with())))
    mesh = root.find("./asset/mesh[@name='palm_body_mesh']")
    geom = root.find(".//geom[@name='palm_geom']")

    assert mesh is not None
    assert "vertex" in mesh.attrib
    assert "face" in mesh.attrib
    assert geom is not None
    assert geom.attrib["type"] == "mesh"
    assert geom.attrib["mesh"] == "palm_body_mesh"


@pytest.mark.parametrize(
    "palm_config",
    [
        PalmContactConfig(mode="box_pads"),
        PalmContactConfig(mode="pad_grid", pad_resolution=2),
        PalmContactConfig(
            mode="convex_patches",
            convex_patch_resolution=2,
            max_num_pad_geoms=4,
        ),
        PalmContactConfig(
            mode="tiled_mesh_colliders",
            mesh_collider_resolution=2,
            max_num_mesh_colliders=4,
        ),
    ],
)
def test_existing_palm_contact_modes_generate_with_outline_body(palm_config):
    xml = build_mjcf_xml(
        build_hand_model(_design_with()),
        geometry_config=GeometryConfig(palm=palm_config),
    )

    assert 'name="palm_body_mesh"' in xml
    assert 'name="palm_geom" type="mesh"' in xml
