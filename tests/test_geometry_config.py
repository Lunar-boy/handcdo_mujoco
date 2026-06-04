from __future__ import annotations

import pytest

from handcdo.design_space import DesignSpace
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import sample_random_grasp
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml
from handcdo.mujoco_eval import EvaluationConfig, evaluate_grasp
from handcdo.tools import get_tool


def test_geometry_config_from_none_is_default():
    assert GeometryConfig.from_dict(None) == GeometryConfig()


def test_geometry_config_parses_canonical_geometry_sections():
    config = GeometryConfig.from_dict(
        {
            "geometry": {
                "finger": {"mode": "capsule", "fingertip_pad_thickness": 0.005},
                "palm": {"mode": "box_pads", "pad_resolution": 3},
                "tool": {"mode": "primitive", "collision_margin": 0.002},
            },
            "finger_contact": {"mode": "local_convex_patches"},
        }
    )

    assert config.finger.mode == "capsule"
    assert config.finger.fingertip_pad_thickness == 0.005
    assert config.palm.pad_resolution == 3
    assert config.tool.collision_margin == 0.002


def test_geometry_config_parses_legacy_sections():
    config = GeometryConfig.from_dict(
        {
            "finger_contact": {"mode": "capsule_tip_pad", "fingertip_pad_enabled": True},
            "palm_contact": {"mode": "pad_grid", "pad_resolution": 4},
            "tool_contact": {"mode": "hybrid"},
        }
    )

    assert config.finger.mode == "capsule_tip_pad"
    assert config.finger.fingertip_pad_enabled is True
    assert config.palm.mode == "pad_grid"
    assert config.tool.mode == "hybrid"


def test_geometry_config_parses_direct_alias_sections():
    config = GeometryConfig.from_dict(
        {
            "finger": {"mode": "capsule"},
            "palm": {"mode": "box_pads"},
            "tool": {"mode": "primitive"},
        }
    )

    assert config == GeometryConfig()


def test_geometry_config_converts_friction_lists_to_tuples():
    config = GeometryConfig.from_dict(
        {
            "geometry": {
                "finger": {"fingertip_pad_friction": [1, 2, 3]},
                "palm": {"pad_friction": [4, 5, 6]},
                "tool": {"friction": [7, 8, 9]},
            }
        }
    )

    assert config.finger.fingertip_pad_friction == (1.0, 2.0, 3.0)
    assert config.palm.pad_friction == (4.0, 5.0, 6.0)
    assert config.tool.friction == (7.0, 8.0, 9.0)


def test_geometry_config_invalid_mode_raises_value_error():
    with pytest.raises(ValueError, match="geometry.finger.mode='spheres'"):
        GeometryConfig.from_dict({"geometry": {"finger": {"mode": "spheres"}}})


def test_geometry_config_invalid_fingertip_pad_shape_raises_value_error():
    with pytest.raises(ValueError, match="geometry.finger.fingertip_pad_shape='sphere'"):
        GeometryConfig.from_dict({"geometry": {"finger": {"fingertip_pad_shape": "sphere"}}})


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"geometry": {"finger": {"fingertip_pad_thickness": 0}}}, "fingertip_pad_thickness"),
        ({"geometry": {"palm": {"pad_resolution": 0}}}, "pad_resolution"),
        ({"geometry": {"palm": {"max_num_pad_geoms": 0}}}, "max_num_pad_geoms"),
        ({"geometry": {"tool": {"collision_margin": -0.001}}}, "collision_margin"),
    ],
)
def test_geometry_config_invalid_numeric_values_raise_value_error(payload, field):
    with pytest.raises(ValueError, match=field):
        GeometryConfig.from_dict(payload)


def test_geometry_config_unknown_key_inside_section_raises_value_error():
    with pytest.raises(ValueError, match="geometry.palm.padd_resolution=3"):
        GeometryConfig.from_dict({"geometry": {"palm": {"padd_resolution": 3}}})


def test_default_geometry_config_preserves_mjcf_xml_exactly():
    design = DesignSpace().sample(seed=4)
    hand = build_hand_model(design)
    tool = get_tool("hammer")

    assert build_mjcf_xml(hand, tool=tool) == build_mjcf_xml(hand, tool=tool, geometry_config=GeometryConfig())


def test_default_xml_still_loads_with_mujoco_when_available():
    mujoco = pytest.importorskip("mujoco")
    design = DesignSpace().sample(seed=5)
    xml = build_mjcf_xml(build_hand_model(design), tool=get_tool("hammer"), geometry_config=GeometryConfig())

    model = mujoco.MjModel.from_xml_string(xml)

    assert model.nbody > 1
    assert model.nu > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"geometry": {"finger": {"mode": "local_convex_patches"}}},
        {"geometry": {"tool": {"mode": "convex_mesh"}}},
    ],
)
def test_valid_future_geometry_modes_raise_not_implemented_during_mjcf_generation(payload):
    design = DesignSpace().sample(seed=6)
    config = GeometryConfig.from_dict(payload)

    with pytest.raises(NotImplementedError):
        build_mjcf_xml(build_hand_model(design), tool=get_tool("hammer"), geometry_config=config)


def test_evaluate_grasp_accepts_default_geometry_config_when_mujoco_available():
    pytest.importorskip("mujoco")
    design = DesignSpace().sample(seed=7)
    grasp = sample_random_grasp(seed=0)
    config = EvaluationConfig(settle_steps=1, close_steps=1, wrench_steps=1)

    result = evaluate_grasp(design, "hammer", grasp, config=config, geometry_config=GeometryConfig())

    assert result.design_id == design.design_id
    assert result.tool == "hammer"
