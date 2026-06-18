from __future__ import annotations

import logging
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .design_space import HandDesign
from .geometry_config import FingerContactConfig, GeometryConfig, PalmContactConfig, ToolContactConfig
from .hand_model import DigitSpec, HandModel, LinkSpec, build_hand_model
from .tool_geometry import ToolGeometryAsset, resolve_tool_geometry
from .tools import ToolSpec, get_tool
from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def _vec(xs: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{x:.8g}" for x in xs)


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _ensure_supported_geometry_config(geometry_config: GeometryConfig) -> None:
    if geometry_config.finger.mode not in {"capsule", "capsule_tip_pad"}:
        raise NotImplementedError(f"finger contact mode {geometry_config.finger.mode!r} is not implemented yet")
    if geometry_config.palm.mode not in {"box_pads", "pad_grid", "convex_patches"}:
        raise ValueError(f"Unknown palm contact mode {geometry_config.palm.mode!r}")
    if geometry_config.tool.mode == "convex_mesh":
        raise NotImplementedError(f"tool contact mode {geometry_config.tool.mode!r} is not implemented yet")
    if geometry_config.tool.mode not in {"primitive", "hybrid"}:
        raise ValueError(f"Unknown tool contact mode {geometry_config.tool.mode!r}")


def _add_fingertip_pad_geom(parent: ET.Element, link: LinkSpec, finger_config: FingerContactConfig) -> None:
    shape = finger_config.fingertip_pad_shape
    if shape == "box":
        thickness = finger_config.fingertip_pad_thickness
        pad_half_x = min(0.008, max(0.003, 0.28 * link.length))
        pad_half_y = max(0.003, 0.75 * link.radius)
        pad_half_z = 0.5 * thickness
        pos = (
            max(0.0, link.length - pad_half_x),
            0.0,
            -(link.radius + pad_half_z * 0.5),
        )
        size = (pad_half_x, pad_half_y, pad_half_z)
    elif shape in {"ellipsoid", "capsule", "convex_mesh"}:
        raise NotImplementedError(f"fingertip pad shape {shape!r} is not implemented yet")
    else:
        raise ValueError(f"Unknown fingertip pad shape {shape!r}")

    ET.SubElement(
        parent,
        "geom",
        name=f"{link.name}_tip_pad",
        type="box",
        pos=_vec(pos),
        size=_vec(size),
        density="400",
        friction=_vec(finger_config.fingertip_pad_friction),
        contype="1",
        conaffinity="1",
    )


def _add_digit(parent: ET.Element, digit: DigitSpec, finger_config: FingerContactConfig | None = None) -> None:
    finger_config = finger_config or FingerContactConfig()
    base = ET.SubElement(
        parent,
        "body",
        name=digit.name,
        pos=_vec(digit.base_pos),
        euler=f"0 0 {digit.base_yaw:.8g}",
    )
    current = base
    for link in digit.links:
        ET.SubElement(
            current,
            "joint",
            name=link.joint.name,
            type="hinge",
            axis=_vec(link.joint.axis),
            range=_vec((math.degrees(link.joint.range[0]), math.degrees(link.joint.range[1]))),
            damping="0.18",
            armature="0.002",
            limited="true",
        )
        ET.SubElement(
            current,
            "geom",
            name=link.name,
            type="capsule",
            fromto=_vec((0.0, 0.0, 0.0, link.length, 0.0, 0.0)),
            size=f"{link.radius:.8g}",
            density="650",
            friction="1.1 0.02 0.002",
            contype="1",
            conaffinity="1",
        )
        if link.fingertip and finger_config.fingertip_pad_enabled:
            _add_fingertip_pad_geom(current, link, finger_config)
        current = ET.SubElement(current, "body", name=f"{link.name}_tip", pos=_vec((link.length, 0.0, 0.0)))


def _add_palm_box_pads(parent: ET.Element, hand: HandModel, palm_config: PalmContactConfig) -> None:
    _ = palm_config
    for pad in hand.palm_pads:
        ET.SubElement(parent, "geom", name=pad.name, type="box", pos=_vec(pad.pos), size=_vec(pad.size), density="500", friction="1.4 0.02 0.002")


def _add_palm_pad_grid(parent: ET.Element, hand: HandModel, palm_config: PalmContactConfig) -> None:
    resolution = palm_config.pad_resolution
    if resolution < 2 or resolution * resolution > palm_config.max_num_pad_geoms:
        raise ValueError(
            "palm pad_grid requires pad_resolution >= 2 and "
            "pad_resolution * pad_resolution <= max_num_pad_geoms; "
            f"got pad_resolution={resolution!r}, max_num_pad_geoms={palm_config.max_num_pad_geoms!r}"
        )

    palm_half_x, palm_half_y, palm_half_z = hand.palm_size
    pad_half_z = 0.0025
    margin_x = min(0.010, 0.15 * palm_half_x)
    margin_y = min(0.010, 0.15 * palm_half_y)
    usable_half_x = max(0.001, palm_half_x - margin_x)
    usable_half_y = max(0.001, palm_half_y - margin_y)
    cell_half_x = usable_half_x / resolution
    cell_half_y = usable_half_y / resolution
    size = (0.85 * cell_half_x, 0.85 * cell_half_y, pad_half_z)
    z = palm_half_z + pad_half_z

    for row in range(resolution):
        for col in range(resolution):
            pos = (
                -usable_half_x + (2 * col + 1) * cell_half_x,
                -usable_half_y + (2 * row + 1) * cell_half_y,
                z,
            )
            ET.SubElement(
                parent,
                "geom",
                name=f"palm_grid_pad_r{row}_c{col}",
                type="box",
                pos=_vec(pos),
                size=_vec(size),
                density="500",
                friction=_vec(palm_config.pad_friction),
                contype="1",
                conaffinity="1",
            )


def _add_palm_convex_patches(parent: ET.Element, hand: HandModel, palm_config: PalmContactConfig) -> None:
    resolution = palm_config.convex_patch_resolution
    if resolution < 2 or resolution * resolution > palm_config.max_num_pad_geoms:
        raise ValueError(
            "palm convex_patches requires convex_patch_resolution >= 2 and "
            "convex_patch_resolution^2 <= max_num_pad_geoms; "
            f"got convex_patch_resolution={resolution!r}, "
            f"max_num_pad_geoms={palm_config.max_num_pad_geoms!r}"
        )

    palm_half_x, palm_half_y, palm_half_z = hand.palm_size
    usable_half_x = palm_half_x * (1.0 - palm_config.convex_patch_margin_ratio)
    usable_half_y = palm_half_y * (1.0 - palm_config.convex_patch_margin_ratio)
    cell_half_x = usable_half_x / resolution
    cell_half_y = usable_half_y / resolution
    patch_half_x = 0.85 * cell_half_x
    patch_half_y = 0.85 * cell_half_y
    params = hand.design.params
    max_height = (
        palm_config.convex_patch_max_height
        if palm_config.convex_patch_max_height is not None
        else float(params["palm_kernel_max_height"])
    )
    eps = 1e-6
    kernels = []
    for index in (1, 2):
        angle = float(params[f"palm_kernel_center_angle_{index}"])
        radius = 0.035 + float(params[f"palm_kernel_center_offset_{index}"])
        kernels.append(
            (
                radius * math.cos(angle),
                radius * math.sin(angle),
                max(eps, float(params[f"palm_kernel_spread_{index}"])),
                float(params[f"palm_kernel_intensity_ratio_{index}"]),
            )
        )

    for row in range(resolution):
        for col in range(resolution):
            x = -usable_half_x + (2 * col + 1) * cell_half_x
            y = -usable_half_y + (2 * row + 1) * cell_half_y
            local_height = 0.0
            for center_x, center_y, spread, intensity in kernels:
                distance_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                local_height += intensity * max_height * math.exp(-distance_sq / (2.0 * spread**2))
            local_height = max(palm_config.convex_patch_min_height, local_height)
            if max_height > 0:
                local_height = min(max_height, local_height)

            patch_half_z = palm_config.convex_patch_base_thickness + 0.5 * local_height
            ET.SubElement(
                parent,
                "geom",
                name=f"palm_convex_patch_r{row}_c{col}",
                type="box",
                pos=_vec((x, y, palm_half_z + patch_half_z)),
                size=_vec((patch_half_x, patch_half_y, patch_half_z)),
                density="500",
                friction=_vec(palm_config.pad_friction),
                contype="1",
                conaffinity="1",
            )


def _add_palm_geoms(parent: ET.Element, hand: HandModel, palm_config: PalmContactConfig | None = None) -> None:
    palm_config = palm_config or PalmContactConfig()
    ET.SubElement(parent, "geom", name="palm_geom", type="box", size=_vec(hand.palm_size), density="700", friction="1.2 0.02 0.002")
    if palm_config.mode == "box_pads":
        _add_palm_box_pads(parent, hand, palm_config)
    elif palm_config.mode == "pad_grid":
        _add_palm_pad_grid(parent, hand, palm_config)
    elif palm_config.mode == "convex_patches":
        _add_palm_convex_patches(parent, hand, palm_config)
    else:
        raise ValueError(f"Unknown palm contact mode {palm_config.mode!r}")


def _add_primitive_tool_geoms(
    body: ET.Element,
    tool: ToolSpec,
    friction: tuple[float, float, float],
) -> None:
    fr = _vec(friction)
    if tool.name == "hammer":
        ET.SubElement(body, "geom", name="hammer_handle", type="capsule", fromto="-0.10 0 0 0.11 0 0", size="0.012", mass=str(tool.mass * 0.45), friction=fr)
        ET.SubElement(body, "geom", name="hammer_head", type="box", pos="0.12 0 0.03", size="0.035 0.018 0.018", mass=str(tool.mass * 0.55), friction=fr)
    elif tool.name == "spoon":
        ET.SubElement(body, "geom", name="spoon_handle", type="capsule", fromto="-0.11 0 0 0.07 0 0", size="0.006", mass=str(tool.mass * 0.55), friction=fr)
        ET.SubElement(body, "geom", name="spoon_bowl", type="ellipsoid", pos="0.095 0 0.005", size="0.032 0.020 0.006", mass=str(tool.mass * 0.45), friction=fr)
    elif tool.name == "knife":
        ET.SubElement(body, "geom", name="knife_handle", type="capsule", fromto="-0.10 0 0 0.015 0 0", size="0.010", mass=str(tool.mass * 0.55), friction=fr)
        ET.SubElement(body, "geom", name="knife_blade", type="box", pos="0.085 0 0", size="0.070 0.010 0.0025", mass=str(tool.mass * 0.45), friction=fr)
    else:
        raise ValueError(f"Unsupported tool {tool.name}")


def _add_hybrid_tool_geoms(
    body: ET.Element,
    tool: ToolSpec,
    friction: tuple[float, float, float],
    asset_parent: ET.Element,
    geometry_asset: ToolGeometryAsset,
) -> None:
    if geometry_asset.visual_mesh is not None:
        mesh_name = f"{tool.name}_visual_mesh"
        ET.SubElement(asset_parent, "mesh", name=mesh_name, file=str(geometry_asset.visual_mesh.resolve()))
        ET.SubElement(
            body,
            "geom",
            name=f"{tool.name}_visual",
            type="mesh",
            mesh=mesh_name,
            mass="0",
            contype="0",
            conaffinity="0",
        )

    if geometry_asset.collision_meshes:
        geom_mass = tool.mass / len(geometry_asset.collision_meshes)
        for index, mesh_path in enumerate(geometry_asset.collision_meshes):
            mesh_name = f"{tool.name}_collision_mesh_{index}"
            ET.SubElement(asset_parent, "mesh", name=mesh_name, file=str(mesh_path.resolve()))
            ET.SubElement(
                body,
                "geom",
                name=f"{tool.name}_collision_{index}",
                type="mesh",
                mesh=mesh_name,
                mass=str(geom_mass),
                friction=_vec(friction),
                contype="1",
                conaffinity="1",
            )
    else:
        _add_primitive_tool_geoms(body, tool, friction)


def _add_tool(
    world_parent: ET.Element,
    tool: ToolSpec,
    fixed: bool = False,
    tool_config: ToolContactConfig | None = None,
    asset_parent: ET.Element | None = None,
    tool_assets_dir: Path = Path("assets/tools"),
    geometry_asset: ToolGeometryAsset | None = None,
) -> None:
    tool_config = tool_config or ToolContactConfig()
    friction = tool_config.friction if tool_config.friction is not None else tool.friction
    body = ET.SubElement(world_parent, "body", name="tool", pos=_vec(tool.reference_pos), quat=_vec(tool.reference_quat))
    if not fixed:
        ET.SubElement(body, "freejoint", name="tool_free")

    if tool_config.mode == "primitive":
        _add_primitive_tool_geoms(body, tool, friction)
    elif tool_config.mode == "hybrid":
        geometry_asset = geometry_asset or resolve_tool_geometry(tool.name, tool_assets_dir)
        if geometry_asset.visual_mesh is None and not geometry_asset.collision_meshes:
            LOGGER.warning("No hybrid mesh assets found for tool=%s; using primitive fallback", tool.name)
            _add_primitive_tool_geoms(body, tool, friction)
        else:
            if asset_parent is None:
                raise ValueError("Hybrid tool mesh assets require a root-level MJCF asset element")
            _add_hybrid_tool_geoms(body, tool, friction, asset_parent, geometry_asset)
    elif tool_config.mode == "convex_mesh":
        raise NotImplementedError(f"tool contact mode {tool_config.mode!r} is not implemented yet")
    else:
        raise ValueError(f"Unknown tool contact mode {tool_config.mode!r}")


def build_mjcf_xml(
    hand: HandModel,
    tool: ToolSpec | None = None,
    fixed_tool: bool = False,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: Path = Path("assets/tools"),
) -> str:
    geometry_config = geometry_config or GeometryConfig()
    _ensure_supported_geometry_config(geometry_config)
    tool_geometry_asset = None
    if tool is not None and geometry_config.tool.mode == "hybrid":
        tool_geometry_asset = resolve_tool_geometry(tool.name, tool_assets_dir)
    root = ET.Element("mujoco", model=f"handcdo_{hand.design.design_id}")
    ET.SubElement(root, "compiler", angle="degree", coordinate="local", inertiafromgeom="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "size", njmax="500", nconmax="200")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", limited="true")
    ET.SubElement(default, "geom", solref="0.012 1", solimp="0.9 0.95 0.001", margin="0.001")
    asset = None
    if tool_geometry_asset is not None and (
        tool_geometry_asset.visual_mesh is not None or tool_geometry_asset.collision_meshes
    ):
        asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="top", pos="0 0 1.0")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0.6 0.6 0.02", pos="0 0 -0.04", friction="1 0.01 0.001")
    palm = ET.SubElement(world, "body", name="palm", pos="0 0 0")
    _add_palm_geoms(palm, hand, palm_config=geometry_config.palm)
    for digit in hand.digits:
        _add_digit(palm, digit, finger_config=geometry_config.finger)
    if tool is not None:
        _add_tool(
            world,
            tool,
            fixed=fixed_tool,
            tool_config=geometry_config.tool,
            asset_parent=asset,
            tool_assets_dir=tool_assets_dir,
            geometry_asset=tool_geometry_asset,
        )
    actuators = ET.SubElement(root, "actuator")
    for joint in hand.joint_names:
        ET.SubElement(actuators, "position", name=f"{joint}_pos", joint=joint, kp="6.0", ctrlrange="-0.3 1.35", ctrllimited="true")
    _indent(root)
    return ET.tostring(root, encoding="unicode")


def write_design_model(
    design: HandDesign,
    output_dir: str | Path,
    tool_name: str | None = None,
    fixed_tool: bool = False,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: Path = Path("assets/tools"),
) -> Path:
    hand = build_hand_model(design)
    tool = get_tool(tool_name) if tool_name else None
    design_dir = ensure_dir(Path(output_dir) / "designs" / design.design_id)
    design.to_json(design_dir / "design.json")
    xml = build_mjcf_xml(
        hand,
        tool=tool,
        fixed_tool=fixed_tool,
        geometry_config=geometry_config,
        tool_assets_dir=tool_assets_dir,
    )
    model_path = design_dir / ("model.xml" if tool is None else f"model_{tool.name}.xml")
    model_path.write_text(xml, encoding="utf-8")
    return model_path
