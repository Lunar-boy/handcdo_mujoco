from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.geometry_config import FingerContactConfig, GeometryConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml


def _design_with_tip_scales(scale_y: float, scale_z: float) -> HandDesign:
    params = DesignSpace().sample(seed=50).to_dict()
    params["fingertip_scale_y"] = scale_y
    params["fingertip_scale_z"] = scale_z
    return HandDesign(params)


def _ellipsoid_config() -> GeometryConfig:
    return GeometryConfig(
        finger=FingerContactConfig(fingertip_body_shape="ellipsoid")
    )


def test_default_fingertip_bodies_remain_single_capsules():
    hand = build_hand_model(_design_with_tip_scales(0.8, 1.4))
    xml = build_mjcf_xml(hand, geometry_config=GeometryConfig())
    root = ET.fromstring(xml)

    assert not [
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("name", "").endswith("_tip_ellipsoid")
    ]
    for digit in hand.digits:
        for link in digit.links:
            if link.fingertip:
                geom = root.find(f".//geom[@name='{link.name}']")
                assert geom is not None
                assert geom.attrib["type"] == "capsule"


def test_opt_in_ellipsoid_geometry_emits_terminal_shafts_and_tips():
    hand = build_hand_model(_design_with_tip_scales(0.9, 1.3))
    root = ET.fromstring(build_mjcf_xml(hand, geometry_config=_ellipsoid_config()))

    for digit in hand.digits:
        terminal_link = digit.links[-1]
        tip_geometry = terminal_link.fingertip_geometry
        assert tip_geometry is not None

        tip = root.find(f".//geom[@name='{terminal_link.name}_tip_ellipsoid']")
        assert tip is not None
        assert tip.attrib["type"] == "ellipsoid"

        if tip_geometry.shaft_length > 1e-8:
            shaft = root.find(f".//geom[@name='{terminal_link.name}_shaft']")
            assert shaft is not None
            assert shaft.attrib["type"] == "capsule"

    for digit in hand.digits:
        for link in digit.links[:-1]:
            geom = root.find(f".//geom[@name='{link.name}']")
            assert geom is not None
            assert geom.attrib["type"] == "capsule"


def test_ellipsoid_tip_uses_independent_y_and_z_scales():
    scale_y = 0.8
    scale_z = 1.4
    hand = build_hand_model(_design_with_tip_scales(scale_y, scale_z))
    root = ET.fromstring(build_mjcf_xml(hand, geometry_config=_ellipsoid_config()))
    link = hand.digits[0].links[-1]
    geom = root.find(f".//geom[@name='{link.name}_tip_ellipsoid']")

    assert geom is not None
    half_x, half_y, half_z = (float(value) for value in geom.attrib["size"].split())
    assert half_x == pytest.approx(link.fingertip_geometry.half_x)
    assert half_y == pytest.approx(0.009 * scale_y)
    assert half_z == pytest.approx(0.009 * scale_z)
    assert half_y != half_z
    assert half_y != pytest.approx(link.radius)
    assert half_z != pytest.approx(link.radius)


def test_ellipsoid_fingertip_xml_loads_with_mujoco_when_available():
    mujoco = pytest.importorskip("mujoco")
    hand = build_hand_model(_design_with_tip_scales(0.9, 1.3))

    model = mujoco.MjModel.from_xml_string(
        build_mjcf_xml(hand, geometry_config=_ellipsoid_config())
    )

    assert model.ngeom > 0


def test_invalid_fingertip_body_shape_is_rejected():
    with pytest.raises(ValueError, match="fingertip_body_shape"):
        GeometryConfig.from_dict(
            {"geometry": {"finger": {"fingertip_body_shape": "bad"}}}
        )
