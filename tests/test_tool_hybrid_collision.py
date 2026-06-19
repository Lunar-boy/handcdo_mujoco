from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from handcdo.design_space import DesignSpace
from handcdo.grasp_sampling import sample_random_grasp
from handcdo.geometry_config import GeometryConfig, ToolContactConfig
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import _vec, build_mjcf_xml
from handcdo.mujoco_eval import EvaluationConfig, evaluate_grasp
from handcdo.tool_geometry import resolve_tool_geometry
from handcdo.tools import get_tool


def _hand():
    return build_hand_model(DesignSpace().sample(seed=60))


def _hybrid_config(friction: tuple[float, float, float] | None = None) -> GeometryConfig:
    return GeometryConfig(tool=ToolContactConfig(mode="hybrid", friction=friction))


def _geom(root: ET.Element, name: str) -> ET.Element:
    geom = root.find(f".//geom[@name='{name}']")
    assert geom is not None
    return geom


def _tool_mesh_assets(root: ET.Element) -> list[ET.Element]:
    asset = root.find("asset")
    assert asset is not None
    return [
        mesh
        for mesh in asset.findall("mesh")
        if mesh.attrib.get("name", "").startswith(("hammer_", "spoon_", "knife_"))
    ]


def _write_tiny_mesh_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0.000 0.000 0.000",
                "v 0.020 0.000 0.000",
                "v 0.000 0.020 0.000",
                "v 0.000 0.000 0.020",
                "f 1 3 2",
                "f 1 2 4",
                "f 1 4 3",
                "f 2 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_default_primitive_tool_geometry_is_preserved_exactly():
    hand = _hand()
    tool = get_tool("hammer")

    default_xml = build_mjcf_xml(hand, tool=tool)
    configured_xml = build_mjcf_xml(hand, tool=tool, geometry_config=GeometryConfig())

    assert configured_xml == default_xml
    assert "hammer_handle" in default_xml
    assert "hammer_head" in default_xml
    assert 'name="palm_body_mesh"' in default_xml


@pytest.mark.parametrize(
    ("tool_name", "geom_names"),
    [
        ("hammer", ("hammer_handle", "hammer_head")),
        ("spoon", ("spoon_handle", "spoon_bowl")),
        ("knife", ("knife_handle", "knife_blade")),
    ],
)
def test_primitive_mode_preserves_named_geoms_for_all_tools(tool_name, geom_names):
    xml = build_mjcf_xml(_hand(), tool=get_tool(tool_name))

    assert all(name in xml for name in geom_names)
    assert '<geom type="mesh"' not in xml


def test_primitive_mode_uses_tool_config_friction_override():
    friction = (1.7, 0.05, 0.005)
    config = GeometryConfig(tool=ToolContactConfig(friction=friction))
    root = ET.fromstring(build_mjcf_xml(_hand(), tool=get_tool("hammer"), geometry_config=config))

    assert _geom(root, "hammer_handle").attrib["friction"] == _vec(friction)
    assert _geom(root, "hammer_head").attrib["friction"] == _vec(friction)


def test_hybrid_missing_assets_falls_back_without_tool_mesh_asset(tmp_path, caplog):
    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    assert "hammer_handle" in xml
    assert "hammer_head" in xml
    assert '<geom type="mesh"' not in xml
    assert _tool_mesh_assets(ET.fromstring(xml)) == []
    assert "using primitive fallback" in caplog.text


def test_hybrid_missing_assets_xml_loads_with_mujoco_when_available(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    model = mujoco.MjModel.from_xml_string(xml)

    assert model.ngeom > 0


def test_hybrid_visual_only_adds_noncolliding_visual_and_primitive_collision(tmp_path):
    visual = tmp_path / "hammer" / "visual.obj"
    _write_tiny_mesh_obj(visual)

    root = ET.fromstring(
        build_mjcf_xml(
            _hand(),
            tool=get_tool("hammer"),
            geometry_config=_hybrid_config(),
            tool_assets_dir=tmp_path,
        )
    )
    asset = root.find("asset")
    assert asset is not None
    mesh = asset.find("mesh[@name='hammer_visual_mesh']")
    assert mesh is not None
    visual_geom = _geom(root, "hammer_visual")

    assert mesh.attrib == {"name": "hammer_visual_mesh", "file": str(visual.resolve())}
    assert visual_geom.attrib["type"] == "mesh"
    assert visual_geom.attrib["mesh"] == "hammer_visual_mesh"
    assert visual_geom.attrib["mass"] == "0"
    assert visual_geom.attrib["contype"] == "0"
    assert visual_geom.attrib["conaffinity"] == "0"
    assert _geom(root, "hammer_handle") is not None
    assert _geom(root, "hammer_head") is not None


def test_hybrid_collision_meshes_are_root_assets_with_deterministic_mass_and_friction(tmp_path):
    tool_dir = tmp_path / "hammer"
    tool_dir.mkdir()
    collision_paths = [tool_dir / "collision_b.obj", tool_dir / "collision_a.obj"]
    for path in collision_paths:
        _write_tiny_mesh_obj(path)
    friction = (1.8, 0.06, 0.006)
    tool = get_tool("hammer")

    root = ET.fromstring(
        build_mjcf_xml(
            _hand(),
            tool=tool,
            geometry_config=_hybrid_config(friction),
            tool_assets_dir=tmp_path,
        )
    )
    asset = root.find("asset")
    assert asset is not None
    meshes = _tool_mesh_assets(root)

    assert root.find("worldbody") is not None
    assert list(root).index(asset) < list(root).index(root.find("worldbody"))
    assert [mesh.attrib["name"] for mesh in meshes] == [
        "hammer_collision_mesh_0",
        "hammer_collision_mesh_1",
    ]
    assert [mesh.attrib["file"] for mesh in meshes] == [
        str((tool_dir / "collision_a.obj").resolve()),
        str((tool_dir / "collision_b.obj").resolve()),
    ]
    assert root.find(".//geom[@name='hammer_handle']") is None
    assert root.find(".//geom[@name='hammer_head']") is None
    for index in range(2):
        geom = _geom(root, f"hammer_collision_{index}")
        assert geom.attrib["type"] == "mesh"
        assert geom.attrib["mesh"] == f"hammer_collision_mesh_{index}"
        assert geom.attrib["contype"] == "1"
        assert geom.attrib["conaffinity"] == "1"
        assert geom.attrib["mass"] == str(tool.mass / 2)
        assert geom.attrib["friction"] == _vec(friction)


def test_hybrid_visual_only_with_valid_mesh_loads_with_mujoco_when_available(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    visual = tmp_path / "hammer" / "visual.obj"
    _write_tiny_mesh_obj(visual)

    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )
    model = mujoco.MjModel.from_xml_string(xml)

    assert "hammer_handle" in xml
    assert "hammer_head" in xml
    assert model.nmesh >= 1
    assert model.ngeom > 0


def test_hybrid_collision_mesh_with_valid_mesh_loads_with_mujoco_when_available(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    collision = tmp_path / "hammer" / "collision_0.obj"
    _write_tiny_mesh_obj(collision)

    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )
    model = mujoco.MjModel.from_xml_string(xml)

    assert "hammer_collision_0" in xml
    assert "hammer_handle" not in xml
    assert "hammer_head" not in xml
    assert model.nmesh >= 1
    assert model.ngeom > 0


def test_resolver_rejects_unknown_tool(tmp_path):
    with pytest.raises(ValueError, match="Unknown tool 'fork'"):
        resolve_tool_geometry("fork", tmp_path)


def test_resolver_missing_directory_returns_primitive_fallback(tmp_path):
    asset = resolve_tool_geometry("spoon", tmp_path)

    assert asset.name == "spoon"
    assert asset.visual_mesh is None
    assert asset.collision_meshes == ()
    assert asset.primitive_fallback is True


def test_resolver_uses_visual_priority_and_deterministic_collision_order(tmp_path):
    tool_dir = tmp_path / "knife"
    tool_dir.mkdir()
    for name in (
        ".gitkeep",
        "tool_visual.stl",
        "visual.obj",
        "collider_b.obj",
        "collision_b.stl",
        "collision_a.stl",
        "collider_a.obj",
    ):
        (tool_dir / name).write_text("", encoding="utf-8")

    asset = resolve_tool_geometry("knife", tmp_path)

    assert asset.visual_mesh == (tool_dir / "visual.obj").resolve()
    assert asset.collision_meshes == tuple(
        (tool_dir / name).resolve()
        for name in ("collision_a.stl", "collision_b.stl", "collider_a.obj", "collider_b.obj")
    )


def test_hybrid_is_accepted_and_convex_mesh_remains_unimplemented(tmp_path):
    build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    with pytest.raises(NotImplementedError, match="convex_mesh"):
        build_mjcf_xml(
            _hand(),
            tool=get_tool("hammer"),
            geometry_config=GeometryConfig(tool=ToolContactConfig(mode="convex_mesh")),
        )


def test_unknown_manually_constructed_tool_mode_raises_value_error():
    config = GeometryConfig(tool=ToolContactConfig(mode="surface"))

    with pytest.raises(ValueError, match="Unknown tool contact mode 'surface'"):
        build_mjcf_xml(_hand(), tool=get_tool("hammer"), geometry_config=config)


def test_geometry_high_config_generates_with_missing_asset_fallback(tmp_path):
    config_data = yaml.safe_load(Path("configs/geometry_high.yaml").read_text(encoding="utf-8"))
    config = GeometryConfig.from_dict(config_data)

    xml = build_mjcf_xml(
        _hand(),
        tool=get_tool("hammer"),
        geometry_config=config,
        tool_assets_dir=tmp_path,
    )

    assert config.tool.mode == "hybrid"
    assert "hammer_handle" in xml
    assert "hammer_head" in xml
    assert _tool_mesh_assets(ET.fromstring(xml)) == []


def test_evaluate_grasp_accepts_custom_tool_assets_dir_for_hybrid_mesh(tmp_path):
    pytest.importorskip("mujoco")
    collision = tmp_path / "hammer" / "collision_0.obj"
    _write_tiny_mesh_obj(collision)
    design = DesignSpace().sample(seed=61)

    result = evaluate_grasp(
        design,
        "hammer",
        sample_random_grasp(seed=0),
        config=EvaluationConfig(settle_steps=1, close_steps=1, wrench_steps=1),
        geometry_config=_hybrid_config(),
        tool_assets_dir=tmp_path,
    )

    assert result.design_id == design.design_id
    assert result.tool == "hammer"
