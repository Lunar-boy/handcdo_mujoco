from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from handcdo.design_space import DesignSpace
from handcdo.geometry_config import FingerContactConfig, GeometryConfig, PalmContactConfig
from handcdo.grasp_sampling import sample_random_grasp
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import _vec, build_mjcf_xml
from handcdo.mujoco_eval import EvaluationConfig, evaluate_grasp
from handcdo.tools import get_tool


def _grid_pad_geoms(xml: str) -> list[ET.Element]:
    return [
        geom
        for geom in ET.fromstring(xml).iter("geom")
        if geom.attrib.get("name", "").startswith("palm_grid_pad_")
    ]


def _convex_patch_geoms(xml: str) -> list[ET.Element]:
    return [
        geom
        for geom in ET.fromstring(xml).iter("geom")
        if geom.attrib.get("name", "").startswith("palm_convex_patch_")
    ]


def test_default_geometry_preserves_original_palm_pads_and_xml_exactly():
    hand = build_hand_model(DesignSpace().sample(seed=50))

    default_xml = build_mjcf_xml(hand)
    configured_xml = build_mjcf_xml(hand, geometry_config=GeometryConfig())

    assert configured_xml == default_xml
    assert "palm_kernel_pad_1" in default_xml
    assert "palm_kernel_pad_2" in default_xml
    assert "palm_grid_pad_" not in default_xml


def test_pad_grid_replaces_original_pads_with_expected_geoms():
    hand = build_hand_model(DesignSpace().sample(seed=51))
    palm_config = PalmContactConfig(mode="pad_grid", pad_resolution=3, pad_friction=(1.6, 0.04, 0.004))
    xml = build_mjcf_xml(hand, geometry_config=GeometryConfig(palm=palm_config))
    grid_pads = _grid_pad_geoms(xml)

    assert len(grid_pads) == 9
    assert "palm_kernel_pad_1" not in xml
    assert "palm_kernel_pad_2" not in xml
    assert {geom.attrib["name"] for geom in grid_pads} == {
        f"palm_grid_pad_r{row}_c{col}" for row in range(3) for col in range(3)
    }
    for geom in grid_pads:
        assert geom.attrib["type"] == "box"
        assert geom.attrib["density"] == "500"
        assert geom.attrib["contype"] == "1"
        assert geom.attrib["conaffinity"] == "1"
        assert geom.attrib["friction"] == _vec(palm_config.pad_friction)


def test_pad_grid_positions_and_sizes_are_within_palm_footprint():
    hand = build_hand_model(DesignSpace().sample(seed=52))
    config = GeometryConfig(palm=PalmContactConfig(mode="pad_grid", pad_resolution=4))

    for geom in _grid_pad_geoms(build_mjcf_xml(hand, geometry_config=config)):
        x, y, z = (float(value) for value in geom.attrib["pos"].split())
        size = tuple(float(value) for value in geom.attrib["size"].split())
        assert abs(x) < hand.palm_size[0]
        assert abs(y) < hand.palm_size[1]
        assert z > hand.palm_size[2]
        assert all(component > 0 for component in size)


@pytest.mark.parametrize(
    "palm_config",
    [
        PalmContactConfig(mode="pad_grid", pad_resolution=1),
        PalmContactConfig(mode="pad_grid", pad_resolution=5),
    ],
)
def test_invalid_pad_grid_resolution_raises_value_error(palm_config):
    hand = build_hand_model(DesignSpace().sample(seed=53))

    with pytest.raises(ValueError, match=r"pad_resolution=.*max_num_pad_geoms="):
        build_mjcf_xml(hand, geometry_config=GeometryConfig(palm=palm_config))


def test_convex_palm_patches_generate_deterministic_height_varying_box_geoms():
    hand = build_hand_model(DesignSpace().sample(seed=54))
    palm_config = PalmContactConfig(
        mode="convex_patches",
        convex_patch_resolution=4,
        max_num_pad_geoms=16,
        pad_friction=(1.6, 0.04, 0.004),
    )
    config = GeometryConfig(palm=palm_config)

    xml = build_mjcf_xml(hand, geometry_config=config)
    patches = _convex_patch_geoms(xml)

    assert xml == build_mjcf_xml(hand, geometry_config=config)
    assert len(patches) == 16
    assert "palm_kernel_pad_1" not in xml
    assert "palm_grid_pad_" not in xml
    assert len({geom.attrib["size"].split()[2] for geom in patches}) > 1
    assert {geom.attrib["name"] for geom in patches} == {
        f"palm_convex_patch_r{row}_c{col}" for row in range(4) for col in range(4)
    }
    for geom in patches:
        assert geom.attrib["type"] == "box"
        assert geom.attrib["density"] == "500"
        assert geom.attrib["contype"] == "1"
        assert geom.attrib["conaffinity"] == "1"
        assert geom.attrib["friction"] == _vec(palm_config.pad_friction)


@pytest.mark.parametrize(
    "palm_config",
    [
        PalmContactConfig(mode="convex_patches", convex_patch_resolution=1),
        PalmContactConfig(mode="convex_patches", convex_patch_resolution=5),
    ],
)
def test_invalid_convex_patch_resolution_raises_value_error(palm_config):
    hand = build_hand_model(DesignSpace().sample(seed=54))

    with pytest.raises(ValueError, match=r"convex_patch_resolution=.*max_num_pad_geoms="):
        build_mjcf_xml(hand, geometry_config=GeometryConfig(palm=palm_config))


def test_unknown_manually_constructed_palm_mode_raises_value_error():
    hand = build_hand_model(DesignSpace().sample(seed=54))
    config = GeometryConfig(palm=PalmContactConfig(mode="surface"))

    with pytest.raises(ValueError, match="Unknown palm contact mode 'surface'"):
        build_mjcf_xml(hand, geometry_config=config)


def test_pad_grid_integrates_with_fingertip_pads():
    hand = build_hand_model(DesignSpace().sample(seed=55))
    config = GeometryConfig(
        finger=FingerContactConfig(mode="capsule_tip_pad", fingertip_pad_enabled=True),
        palm=PalmContactConfig(mode="pad_grid", pad_resolution=3),
    )
    xml = build_mjcf_xml(hand, geometry_config=config)

    assert "_tip_pad" in xml
    assert "palm_grid_pad_" in xml


def test_pad_grid_xml_loads_with_mujoco_when_available():
    mujoco = pytest.importorskip("mujoco")
    hand = build_hand_model(DesignSpace().sample(seed=56))
    config = GeometryConfig(palm=PalmContactConfig(mode="pad_grid", pad_resolution=3))

    model = mujoco.MjModel.from_xml_string(build_mjcf_xml(hand, geometry_config=config))

    assert model.ngeom > 0


def test_convex_patches_xml_loads_with_mujoco_when_available():
    mujoco = pytest.importorskip("mujoco")
    hand = build_hand_model(DesignSpace().sample(seed=57))
    config = GeometryConfig(
        palm=PalmContactConfig(
            mode="convex_patches",
            convex_patch_resolution=4,
            max_num_pad_geoms=16,
        )
    )

    model = mujoco.MjModel.from_xml_string(
        build_mjcf_xml(hand, tool=get_tool("hammer"), geometry_config=config)
    )

    assert model.ngeom >= 16
    assert model.nu > 0


def test_evaluate_grasp_accepts_convex_patches_when_mujoco_available():
    pytest.importorskip("mujoco")
    design = DesignSpace().sample(seed=58)
    geometry_config = GeometryConfig(
        palm=PalmContactConfig(
            mode="convex_patches",
            convex_patch_resolution=2,
            max_num_pad_geoms=4,
        )
    )

    result = evaluate_grasp(
        design,
        "hammer",
        sample_random_grasp(seed=0),
        config=EvaluationConfig(settle_steps=1, close_steps=1, wrench_steps=1),
        geometry_config=geometry_config,
    )

    assert result.design_id == design.design_id
    assert result.tool == "hammer"
    assert math.isfinite(result.score)
