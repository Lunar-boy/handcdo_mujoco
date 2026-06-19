from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.geometry_config import FingerContactConfig, GeometryConfig
from handcdo.hand_model import JointSpec, LinkSpec, build_hand_model
from handcdo.mjcf_generator import _fingertip_contact_half_extents, _vec, build_mjcf_xml


def _tip_pad_geoms(xml: str) -> list[ET.Element]:
    return [
        geom
        for geom in ET.fromstring(xml).iter("geom")
        if geom.attrib.get("name", "").endswith("_tip_pad")
    ]


def _design_with_tip_scales(scale_y: float, scale_z: float) -> HandDesign:
    params = DesignSpace().sample(seed=46).to_dict()
    params["fingertip_scale_y"] = scale_y
    params["fingertip_scale_z"] = scale_z
    return HandDesign(params)


def _tip_pad_for_first_digit(xml: str, hand) -> ET.Element:
    link = hand.digits[0].links[-1]
    geom = ET.fromstring(xml).find(f".//geom[@name='{link.name}_tip_pad']")
    assert geom is not None
    return geom


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
        contact_half_x, contact_half_y, contact_half_z = _fingertip_contact_half_extents(link)
        pad_half_x = min(0.008, max(0.003, 0.55 * contact_half_x))
        pad_half_y = max(0.003, 0.75 * contact_half_y)
        pad_half_z = 0.5 * finger_config.fingertip_pad_thickness
        assert geom.attrib["type"] == "box"
        assert geom.attrib["density"] == "400"
        assert geom.attrib["contype"] == "1"
        assert geom.attrib["conaffinity"] == "1"
        assert geom.attrib["friction"] == _vec(finger_config.fingertip_pad_friction)
        assert geom.attrib["pos"] == _vec(
            (
                max(0.0, link.length - contact_half_x),
                0.0,
                -(contact_half_z + pad_half_z * 0.5),
            )
        )
        assert geom.attrib["size"] == _vec((pad_half_x, pad_half_y, pad_half_z))


def test_tip_pad_width_uses_anisotropic_fingertip_half_y():
    config = GeometryConfig(
        finger=FingerContactConfig(
            fingertip_body_shape="ellipsoid",
            fingertip_pad_enabled=True,
            fingertip_pad_shape="box",
        )
    )
    narrow_hand = build_hand_model(_design_with_tip_scales(0.8, 1.2))
    wide_hand = build_hand_model(_design_with_tip_scales(1.4, 1.2))

    narrow_pad = _tip_pad_for_first_digit(build_mjcf_xml(narrow_hand, geometry_config=config), narrow_hand)
    wide_pad = _tip_pad_for_first_digit(build_mjcf_xml(wide_hand, geometry_config=config), wide_hand)
    narrow_half_y = float(narrow_pad.attrib["size"].split()[1])
    wide_half_y = float(wide_pad.attrib["size"].split()[1])

    assert narrow_half_y == pytest.approx(0.75 * 0.009 * 0.8)
    assert wide_half_y == pytest.approx(0.75 * 0.009 * 1.4)
    assert wide_half_y > narrow_half_y


def test_tip_pad_placement_uses_anisotropic_fingertip_half_z():
    config = GeometryConfig(
        finger=FingerContactConfig(
            fingertip_body_shape="ellipsoid",
            fingertip_pad_enabled=True,
            fingertip_pad_shape="box",
        )
    )
    shallow_hand = build_hand_model(_design_with_tip_scales(1.1, 0.8))
    deep_hand = build_hand_model(_design_with_tip_scales(1.1, 1.4))

    shallow_pad = _tip_pad_for_first_digit(build_mjcf_xml(shallow_hand, geometry_config=config), shallow_hand)
    deep_pad = _tip_pad_for_first_digit(build_mjcf_xml(deep_hand, geometry_config=config), deep_hand)
    shallow_z = float(shallow_pad.attrib["pos"].split()[2])
    deep_z = float(deep_pad.attrib["pos"].split()[2])

    assert shallow_z == pytest.approx(-(0.009 * 0.8 + 0.001))
    assert deep_z == pytest.approx(-(0.009 * 1.4 + 0.001))
    assert deep_z < shallow_z


def test_contact_half_extents_fall_back_to_capsule_dimensions():
    link = LinkSpec(
        name="tip",
        length=0.02,
        radius=0.007,
        joint=JointSpec(name="joint", axis=(0.0, 1.0, 0.0), range=(-0.25, 1.0)),
        fingertip=True,
    )

    assert _fingertip_contact_half_extents(link) == pytest.approx((0.0056, 0.007, 0.007))


def test_capsule_tip_pad_mode_without_enabled_flag_is_capsule_only():
    hand = build_hand_model(DesignSpace().sample(seed=42))
    config = GeometryConfig(finger=FingerContactConfig(mode="capsule_tip_pad", fingertip_pad_enabled=False))

    assert "_tip_pad" not in build_mjcf_xml(hand, geometry_config=config)


@pytest.mark.parametrize("body_shape", ["capsule", "ellipsoid"])
def test_tip_pad_xml_loads_with_mujoco_when_available(body_shape):
    mujoco = pytest.importorskip("mujoco")
    hand = build_hand_model(_design_with_tip_scales(0.9, 1.3))
    config = GeometryConfig(
        finger=FingerContactConfig(
            mode="capsule_tip_pad",
            fingertip_body_shape=body_shape,
            fingertip_pad_enabled=True,
            fingertip_pad_shape="box",
        )
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
