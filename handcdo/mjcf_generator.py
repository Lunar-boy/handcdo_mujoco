from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .design_space import HandDesign
from .geometry_config import FingerContactConfig, GeometryConfig, PalmContactConfig, ToolContactConfig
from .hand_model import DigitSpec, HandModel, LinkSpec, build_hand_model
from .tools import ToolSpec, get_tool
from .utils import ensure_dir


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
    if geometry_config.palm.mode != "box_pads":
        raise NotImplementedError(f"palm contact mode {geometry_config.palm.mode!r} is not implemented yet")
    if geometry_config.tool.mode != "primitive":
        raise NotImplementedError(f"tool contact mode {geometry_config.tool.mode!r} is not implemented yet")


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


def _add_palm_geoms(parent: ET.Element, hand: HandModel, palm_config: PalmContactConfig | None = None) -> None:
    _ = palm_config
    ET.SubElement(parent, "geom", name="palm_geom", type="box", size=_vec(hand.palm_size), density="700", friction="1.2 0.02 0.002")
    for pad in hand.palm_pads:
        ET.SubElement(parent, "geom", name=pad.name, type="box", pos=_vec(pad.pos), size=_vec(pad.size), density="500", friction="1.4 0.02 0.002")


def _add_tool(parent: ET.Element, tool: ToolSpec, fixed: bool = False, tool_config: ToolContactConfig | None = None) -> None:
    _ = tool_config
    body = ET.SubElement(parent, "body", name="tool", pos=_vec(tool.reference_pos), quat=_vec(tool.reference_quat))
    if not fixed:
        ET.SubElement(body, "freejoint", name="tool_free")
    fr = _vec(tool.friction)
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


def build_mjcf_xml(
    hand: HandModel,
    tool: ToolSpec | None = None,
    fixed_tool: bool = False,
    geometry_config: GeometryConfig | None = None,
) -> str:
    geometry_config = geometry_config or GeometryConfig()
    _ensure_supported_geometry_config(geometry_config)
    root = ET.Element("mujoco", model=f"handcdo_{hand.design.design_id}")
    ET.SubElement(root, "compiler", angle="degree", coordinate="local", inertiafromgeom="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "size", njmax="500", nconmax="200")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", limited="true")
    ET.SubElement(default, "geom", solref="0.012 1", solimp="0.9 0.95 0.001", margin="0.001")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="top", pos="0 0 1.0")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0.6 0.6 0.02", pos="0 0 -0.04", friction="1 0.01 0.001")
    palm = ET.SubElement(world, "body", name="palm", pos="0 0 0")
    _add_palm_geoms(palm, hand, palm_config=geometry_config.palm)
    for digit in hand.digits:
        _add_digit(palm, digit, finger_config=geometry_config.finger)
    if tool is not None:
        _add_tool(world, tool, fixed=fixed_tool, tool_config=geometry_config.tool)
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
) -> Path:
    hand = build_hand_model(design)
    tool = get_tool(tool_name) if tool_name else None
    design_dir = ensure_dir(Path(output_dir) / "designs" / design.design_id)
    design.to_json(design_dir / "design.json")
    xml = build_mjcf_xml(hand, tool=tool, fixed_tool=fixed_tool, geometry_config=geometry_config)
    model_path = design_dir / ("model.xml" if tool is None else f"model_{tool.name}.xml")
    model_path.write_text(xml, encoding="utf-8")
    return model_path
