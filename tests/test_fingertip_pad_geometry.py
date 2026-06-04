from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from handcdo.design_space import DesignSpace
from handcdo.geometry_config import FingerContactConfig, GeometryConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import _vec, build_mjcf_xml


def _tip_pad_geoms(xml: str) -> list[ET.Element]:
    return [
        geom
        for geom in ET.fromstring(xml).iter("geom")
        if geom.attrib.get("name", "").endswith("_tip_pad")
    ]


def test_default_geometry_emits_no_tip_pads_and_preserves_xml_exactly():
    hand = build_hand_model(DesignSpace().sample(seed=40))

    default_xml = build_mjcf_xml(hand)
    configured_xml = build_mjcf_xml(hand, geometry_config=GeometryConfig())

    assert "_tip_pad" not in default_xml
    assert configured_xml == default_xml


def test_tip_pads_are_added_only_to_fingertip_links_with_expected_attributes():
    hand = build_hand_model(DesignSpace().sample(seed=41))
    finger_config = FingerContactConfig(
        mode="capsule_tip_pad",
        fingertip_pad_enabled=True,
        fingertip_pad_shape="box",
        fingertip_pad_friction=(1.6, 0.04, 0.004),
    )
    xml = build_mjcf_xml(hand, geometry_config=GeometryConfig(finger=finger_config))
    pad_geoms = _tip_pad_geoms(xml)
    fingertip_links = {
        link.name: link
        for digit in hand.digits
        for link in digit.links
        if link.fingertip
    }
    non_fingertip_links = {
        link.name
        for digit in hand.digits
        for link in digit.links
        if not link.fingertip
    }

    assert len(pad_geoms) == sum(link.fingertip for digit in hand.digits for link in digit.links)
    assert {geom.attrib["name"] for geom in pad_geoms} == {
        f"{link_name}_tip_pad" for link_name in fingertip_links
    }
    assert all(f"{link_name}_tip_pad" not in xml for link_name in non_fingertip_links)

    for geom in pad_geoms:
        link = fingertip_links[geom.attrib["name"].removesuffix("_tip_pad")]
        pad_half_x = min(0.008, max(0.003, 0.28 * link.length))
        pad_half_y = max(0.003, 0.75 * link.radius)
        pad_half_z = 0.5 * finger_config.fingertip_pad_thickness
        assert geom.attrib["type"] == "box"
        assert geom.attrib["density"] == "400"
        assert geom.attrib["contype"] == "1"
        assert geom.attrib["conaffinity"] == "1"
        assert geom.attrib["friction"] == _vec(finger_config.fingertip_pad_friction)
        assert geom.attrib["pos"] == _vec(
            (max(0.0, link.length - pad_half_x), 0.0, -(link.radius + pad_half_z * 0.5))
        )
        assert geom.attrib["size"] == _vec((pad_half_x, pad_half_y, pad_half_z))


def test_capsule_tip_pad_mode_without_enabled_flag_is_capsule_only():
    hand = build_hand_model(DesignSpace().sample(seed=42))
    config = GeometryConfig(finger=FingerContactConfig(mode="capsule_tip_pad", fingertip_pad_enabled=False))

    assert "_tip_pad" not in build_mjcf_xml(hand, geometry_config=config)


def test_tip_pad_xml_loads_with_mujoco_when_available():
    mujoco = pytest.importorskip("mujoco")
    hand = build_hand_model(DesignSpace().sample(seed=43))
    config = GeometryConfig(
        finger=FingerContactConfig(mode="capsule_tip_pad", fingertip_pad_enabled=True, fingertip_pad_shape="box")
    )

    model = mujoco.MjModel.from_xml_string(build_mjcf_xml(hand, geometry_config=config))

    assert model.ngeom > 0


@pytest.mark.parametrize("shape", ["capsule", "convex_mesh", "ellipsoid"])
def test_unimplemented_tip_pad_shapes_raise_not_implemented(shape):
    hand = build_hand_model(DesignSpace().sample(seed=44))
    config = GeometryConfig(
        finger=FingerContactConfig(mode="capsule_tip_pad", fingertip_pad_enabled=True, fingertip_pad_shape=shape)
    )

    with pytest.raises(NotImplementedError, match=shape):
        build_mjcf_xml(hand, geometry_config=config)


def test_unknown_manually_constructed_tip_pad_shape_raises_value_error():
    hand = build_hand_model(DesignSpace().sample(seed=45))
    config = GeometryConfig(
        finger=FingerContactConfig(mode="capsule_tip_pad", fingertip_pad_enabled=True, fingertip_pad_shape="sphere")
    )

    with pytest.raises(ValueError, match="Unknown fingertip pad shape 'sphere'"):
        build_mjcf_xml(hand, geometry_config=config)
