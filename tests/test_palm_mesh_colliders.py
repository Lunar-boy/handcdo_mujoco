from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.geometry_config import GeometryConfig, PalmContactConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml
from handcdo.palm_mesh_colliders import (
    build_palm_tiled_mesh_colliders,
    export_palm_tiled_mesh_colliders,
)
from handcdo.tools import get_tool


def _hand_with_height(height: float):
    design = DesignSpace().sample(seed=81)
    params = design.to_dict()
    params["palm_kernel_max_height"] = height
    return build_hand_model(HandDesign(params))


def _config(
    collider_type: str = "quad_frustum",
    resolution: int = 4,
    **overrides,
) -> PalmContactConfig:
    values = {
        "mode": "tiled_mesh_colliders",
        "mesh_collider_resolution": resolution,
        "mesh_collider_type": collider_type,
        "mesh_collider_thickness": 0.003,
        "mesh_collider_margin_ratio": 0.0,
        "max_num_mesh_colliders": (
            2 * resolution**2 if collider_type == "triangular_prism" else resolution**2
        ),
    }
    values.update(overrides)
    return PalmContactConfig(**values)


@pytest.mark.parametrize(
    ("collider_type", "expected_count"),
    [("quad_frustum", 16), ("triangular_prism", 32)],
)
def test_resolution_four_produces_expected_closed_colliders(
    collider_type, expected_count
):
    colliders = build_palm_tiled_mesh_colliders(
        _hand_with_height(0.02),
        _config(collider_type),
    )

    assert len(colliders) == expected_count
    assert all(len(collider.mesh.vertices) > 0 for collider in colliders)
    assert all(len(collider.mesh.faces) > 0 for collider in colliders)
    assert all(collider.mesh.is_watertight for collider in colliders)


def test_collider_names_follow_deterministic_grid_order():
    quad = build_palm_tiled_mesh_colliders(_hand_with_height(0.01), _config())
    prisms = build_palm_tiled_mesh_colliders(
        _hand_with_height(0.01),
        _config("triangular_prism"),
    )

    assert quad[0].name == "palm_tile_r00_c00"
    assert quad[-1].name == "palm_tile_r03_c03"
    assert prisms[0].name == "palm_tile_r00_c00_tri0"
    assert prisms[1].name == "palm_tile_r00_c00_tri1"
    assert prisms[-1].name == "palm_tile_r03_c03_tri1"


def test_zero_height_produces_flat_top_vertices():
    hand = _hand_with_height(0.0)
    colliders = build_palm_tiled_mesh_colliders(hand, _config())

    for collider in colliders:
        assert np.all(collider.mesh.vertices[:4, 2] == pytest.approx(hand.palm_size[2]))


def test_positive_height_raises_at_least_one_top_vertex():
    hand = _hand_with_height(0.02)
    colliders = build_palm_tiled_mesh_colliders(hand, _config())

    assert max(collider.mesh.vertices[:4, 2].max() for collider in colliders) > hand.palm_size[2]


def test_same_design_and_config_produce_identical_colliders():
    hand = _hand_with_height(0.02)
    config = _config("triangular_prism")
    first = build_palm_tiled_mesh_colliders(hand, config)
    second = build_palm_tiled_mesh_colliders(hand, config)

    assert [collider.name for collider in first] == [collider.name for collider in second]
    for first_collider, second_collider in zip(first, second, strict=True):
        assert np.array_equal(first_collider.mesh.vertices, second_collider.mesh.vertices)
        assert np.array_equal(first_collider.mesh.faces, second_collider.mesh.faces)


@pytest.mark.parametrize("file_format", ["obj", "stl"])
def test_exported_colliders_can_be_reloaded(tmp_path, file_format):
    colliders = build_palm_tiled_mesh_colliders(
        _hand_with_height(0.02),
        _config(resolution=2),
    )

    paths = export_palm_tiled_mesh_colliders(
        colliders,
        tmp_path,
        file_format=file_format,
    )

    assert len(paths) == 4
    for path in paths:
        loaded = trimesh.load(path, force="mesh", process=False)
        assert isinstance(loaded, trimesh.Trimesh)
        assert len(loaded.vertices) > 0
        assert len(loaded.faces) > 0


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(resolution=1), "mesh_collider_resolution"),
        (_config(mesh_collider_thickness=0.0), "mesh_collider_thickness"),
        (_config(mesh_collider_margin_ratio=0.5), "mesh_collider_margin_ratio"),
        (_config(mesh_collider_type="vhacd"), "mesh_collider_type"),
        (_config(max_num_mesh_colliders=15), "collider_count"),
    ],
)
def test_invalid_direct_configs_raise_clear_errors(config, message):
    with pytest.raises(ValueError, match=message):
        build_palm_tiled_mesh_colliders(_hand_with_height(0.01), config)


def test_mjcf_contains_one_inline_mesh_asset_and_geom_per_collider():
    hand = _hand_with_height(0.02)
    palm_config = _config()
    xml = build_mjcf_xml(
        hand,
        geometry_config=GeometryConfig(palm=palm_config),
    )
    root = ET.fromstring(xml)
    meshes = [
        mesh
        for mesh in root.findall("./asset/mesh")
        if mesh.attrib["name"].startswith("palm_tile_")
    ]
    geoms = [
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("name", "").startswith("palm_tile_")
    ]

    assert len(meshes) == 16
    assert len(geoms) == 16
    assert all("vertex" in mesh.attrib and "face" in mesh.attrib for mesh in meshes)
    assert all(geom.attrib["type"] == "mesh" for geom in geoms)
    assert all(geom.attrib["friction"] == "1.4 0.02 0.002" for geom in geoms)
    assert "palm_kernel_pad_1" not in xml
    assert "palm_convex_patch_" not in xml


def test_mjcf_export_option_writes_design_scoped_stl_files(tmp_path):
    hand = _hand_with_height(0.01)
    config = GeometryConfig(
        palm=_config(
            resolution=2,
            mesh_collider_export=True,
            mesh_collider_export_dir=str(tmp_path),
        )
    )

    build_mjcf_xml(hand, geometry_config=config)

    exported = sorted((tmp_path / hand.design.design_id).glob("*.stl"))
    assert len(exported) == 4


def test_tiled_mesh_collider_xml_loads_with_mujoco_when_available():
    mujoco = pytest.importorskip("mujoco")
    hand = _hand_with_height(0.02)
    config = GeometryConfig(palm=_config())

    model = mujoco.MjModel.from_xml_string(
        build_mjcf_xml(hand, tool=get_tool("hammer"), geometry_config=config)
    )

    assert model.nmesh >= 16
    assert model.ngeom >= 16
    assert model.nu > 0
