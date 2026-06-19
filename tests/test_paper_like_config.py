from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from handcdo.design_space import DesignSpace
from handcdo.geometry_config import GeometryConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml
from handcdo.mujoco_eval import EvaluationConfig
from handcdo.tools import get_tool
from handcdo.utils import read_yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_eval_config(name: str) -> tuple[EvaluationConfig, GeometryConfig]:
    config_data = read_yaml(ROOT / "configs" / name)
    return (
        EvaluationConfig.from_dict(config_data),
        GeometryConfig.from_dict(config_data),
    )


def test_paper_like_config_loads_through_evaluation_config_path():
    eval_config, geometry = _load_eval_config("eval_paper_like.yaml")

    assert eval_config.settle_steps == 350
    assert geometry.finger.mode == "local_convex_patches"
    assert geometry.finger.fingertip_body_shape == "ellipsoid"
    assert geometry.palm.mode == "tiled_mesh_colliders"
    assert geometry.palm.mesh_collider_domain == "outline"
    assert geometry.tool.mode == "hybrid"


def test_paper_like_config_generates_bounded_expected_mjcf():
    _, geometry = _load_eval_config("eval_paper_like.yaml")
    hand = build_hand_model(DesignSpace().sample(seed=0))
    root = ET.fromstring(
        build_mjcf_xml(
            hand,
            tool=get_tool("hammer"),
            geometry_config=geometry,
        )
    )

    palm_meshes = [
        mesh
        for mesh in root.findall("./asset/mesh")
        if mesh.attrib.get("name", "").startswith("palm_tile_")
    ]
    palm_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("name", "").startswith("palm_tile_")
    ]
    finger_meshes = [
        mesh
        for mesh in root.findall("./asset/mesh")
        if "_local_patch_" in mesh.attrib.get("name", "")
    ]
    finger_geoms = [
        geom
        for geom in root.iter("geom")
        if "_local_patch_" in geom.attrib.get("name", "")
    ]

    assert root.find("./asset/mesh[@name='palm_body_mesh']") is not None
    assert palm_meshes
    assert len(palm_meshes) == len(palm_geoms)
    assert len(palm_meshes) <= geometry.palm.max_num_mesh_colliders
    assert finger_meshes
    assert len(finger_meshes) == len(finger_geoms)
    assert len(finger_meshes) <= geometry.finger.max_num_local_patch_colliders


def test_existing_high_config_still_loads_and_generates_mjcf():
    _, geometry = _load_eval_config("eval_high.yaml")
    hand = build_hand_model(DesignSpace().sample(seed=0))
    xml = build_mjcf_xml(hand, tool=get_tool("hammer"), geometry_config=geometry)

    assert geometry.finger.mode == "capsule_tip_pad"
    assert geometry.palm.mode == "pad_grid"
    assert "palm_body_mesh" in xml
    assert "palm_grid_pad_" in xml
