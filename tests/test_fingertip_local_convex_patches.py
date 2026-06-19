from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from handcdo.design_space import DesignSpace
from handcdo.geometry_config import FingerContactConfig, GeometryConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml


def _config(body_shape: str = "capsule", **overrides) -> GeometryConfig:
    values = {
        "mode": "local_convex_patches",
        "fingertip_body_shape": body_shape,
        "fingertip_pad_enabled": True,
        "local_patch_resolution": 2,
        "local_patch_collider_type": "quad_frustum",
        "local_patch_max_height": 0.003,
        "max_num_local_patch_colliders": 96,
    }
    values.update(overrides)
    return GeometryConfig(finger=FingerContactConfig(**values))


def _local_patch_geoms(root: ET.Element) -> list[ET.Element]:
    return [
        geom
        for geom in root.iter("geom")
        if "_local_patch_" in geom.attrib.get("name", "")
    ]


def test_local_patch_config_fields_parse_and_validate():
    config = GeometryConfig.from_dict(
        {
            "geometry": {
                "finger": {
                    "mode": "local_convex_patches",
                    "local_patch_resolution": 3,
                    "local_patch_collider_type": "triangular_prism",
                    "local_patch_thickness": 0.003,
                    "local_patch_margin_ratio": 0.1,
                    "local_patch_max_height": 0.004,
                    "local_patch_min_height": 0.0002,
                    "max_num_local_patch_colliders": 100,
                    "local_patch_export": True,
                    "local_patch_export_dir": "outputs/finger_colliders",
                }
            }
        }
    )

    assert config.finger.local_patch_resolution == 3
    assert config.finger.local_patch_collider_type == "triangular_prism"
    assert config.finger.local_patch_thickness == 0.003
    assert config.finger.local_patch_margin_ratio == 0.1
    assert config.finger.local_patch_max_height == 0.004
    assert config.finger.local_patch_min_height == 0.0002
    assert config.finger.max_num_local_patch_colliders == 100
    assert config.finger.local_patch_export is True
    assert config.finger.local_patch_export_dir == "outputs/finger_colliders"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_patch_resolution", 1),
        ("local_patch_collider_type", "vhacd"),
        ("local_patch_thickness", 0),
        ("local_patch_margin_ratio", 0.5),
        ("local_patch_min_height", -0.001),
        ("local_patch_max_height", 0),
        ("max_num_local_patch_colliders", 0),
    ],
)
def test_invalid_local_patch_config_fields_raise_value_error(field, value):
    with pytest.raises(ValueError, match=field):
        GeometryConfig.from_dict(
            {
                "geometry": {
                    "finger": {
                        "mode": "local_convex_patches",
                        field: value,
                    }
                }
            }
        )


def test_total_hand_collider_limit_is_enforced():
    hand = build_hand_model(DesignSpace().sample(seed=303))
    terminal_count = sum(
        link.fingertip for digit in hand.digits for link in digit.links
    )
    config = _config(max_num_local_patch_colliders=terminal_count * 4 - 1)

    with pytest.raises(ValueError, match="total collider count"):
        build_mjcf_xml(hand, geometry_config=config)


def test_mjcf_emits_unique_matching_assets_and_terminal_only_geoms():
    hand = build_hand_model(DesignSpace().sample(seed=304))
    root = ET.fromstring(build_mjcf_xml(hand, geometry_config=_config()))
    asset = root.find("asset")
    assert asset is not None
    meshes = [
        mesh
        for mesh in asset.findall("mesh")
        if "_local_patch_" in mesh.attrib["name"]
    ]
    geoms = _local_patch_geoms(root)
    terminal_links = {
        link.name
        for digit in hand.digits
        for link in digit.links
        if link.fingertip
    }
    proximal_links = {
        link.name
        for digit in hand.digits
        for link in digit.links
        if not link.fingertip
    }

    expected_count = len(terminal_links) * 4
    mesh_names = [mesh.attrib["name"] for mesh in meshes]
    geom_names = [geom.attrib["name"] for geom in geoms]
    assert len(meshes) == expected_count
    assert len(geoms) == expected_count
    assert len(set(mesh_names)) == expected_count
    assert set(mesh_names) == set(geom_names)
    assert all(geom.attrib["type"] == "mesh" for geom in geoms)
    assert all(geom.attrib["mesh"] == geom.attrib["name"] for geom in geoms)
    assert all(any(name.startswith(link) for link in terminal_links) for name in geom_names)
    assert all(not any(name.startswith(link) for link in proximal_links) for name in geom_names)
    assert "_tip_pad" not in ET.tostring(root, encoding="unicode")


def test_default_config_emits_no_local_patch_assets():
    hand = build_hand_model(DesignSpace().sample(seed=305))
    xml = build_mjcf_xml(hand)

    assert "_local_patch_" not in xml


@pytest.mark.parametrize("body_shape", ["capsule", "ellipsoid"])
def test_local_patch_xml_loads_with_mujoco_for_both_tip_shapes(body_shape):
    mujoco = pytest.importorskip("mujoco")
    hand = build_hand_model(DesignSpace().sample(seed=306))
    xml = build_mjcf_xml(hand, geometry_config=_config(body_shape))

    model = mujoco.MjModel.from_xml_string(xml)

    assert model.nmesh > 0
    assert model.ngeom > 0
