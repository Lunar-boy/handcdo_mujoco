from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PalmBaseFrame:
    name: str
    pos: tuple[float, float, float]
    yaw: float
    tangent: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    anchor_2d: tuple[float, float]


@dataclass(frozen=True)
class PalmOutlineSpec:
    half_x: float
    half_y: float
    half_z: float
    polygon_sides: int
    aspect_ratio: float
    corner_rounding: float = 0.0


@dataclass(frozen=True)
class PalmBodySpec:
    outline_vertices_2d: tuple[tuple[float, float], ...]
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    half_extents: tuple[float, float, float]
    base_frames: dict[str, PalmBaseFrame]
    outline_spec: PalmOutlineSpec


def build_palm_outline_body(
    *,
    n_fingers: int,
    finger_side_offsets: tuple[float, ...],
    finger_normal_offsets: tuple[float, ...],
    finger_angles: tuple[float, ...],
    thumb_side_offset: float,
    thumb_normal_offset: float,
    thumb_angle: float,
    half_x: float = 0.085,
    half_y: float = 0.115,
    half_z: float = 0.032,
    polygon_sides: int = 8,
    aspect_ratio: float = 1.0,
) -> PalmBodySpec:
    """Build a convex extruded palm and outline-local digit base frames.

    The zero-offset digit bases retain fixed inward insets from their boundary
    anchors. This keeps the initial layout close to the previous box model while
    making all configurable offsets relative to the outline tangent and normal.
    """
    _validate_inputs(
        n_fingers=n_fingers,
        finger_side_offsets=finger_side_offsets,
        finger_normal_offsets=finger_normal_offsets,
        finger_angles=finger_angles,
        half_x=half_x,
        half_y=half_y,
        half_z=half_z,
        polygon_sides=polygon_sides,
        aspect_ratio=aspect_ratio,
    )
    outline_spec = PalmOutlineSpec(
        half_x=half_x,
        half_y=half_y,
        half_z=half_z,
        polygon_sides=polygon_sides,
        aspect_ratio=aspect_ratio,
        corner_rounding=0.18 * min(half_x, half_y * aspect_ratio) if polygon_sides == 8 else 0.0,
    )
    outline = _build_outline(outline_spec)
    vertices, faces = _extrude_outline(outline, half_z)
    base_frames = _build_base_frames(
        outline,
        n_fingers=n_fingers,
        finger_side_offsets=finger_side_offsets,
        finger_normal_offsets=finger_normal_offsets,
        finger_angles=finger_angles,
        thumb_side_offset=thumb_side_offset,
        thumb_normal_offset=thumb_normal_offset,
        thumb_angle=thumb_angle,
    )
    actual_half_x = max(abs(x) for x, _ in outline)
    actual_half_y = max(abs(y) for _, y in outline)
    return PalmBodySpec(
        outline_vertices_2d=outline,
        vertices=vertices,
        faces=faces,
        half_extents=(actual_half_x, actual_half_y, half_z),
        base_frames=base_frames,
        outline_spec=outline_spec,
    )


def _validate_inputs(
    *,
    n_fingers: int,
    finger_side_offsets: tuple[float, ...],
    finger_normal_offsets: tuple[float, ...],
    finger_angles: tuple[float, ...],
    half_x: float,
    half_y: float,
    half_z: float,
    polygon_sides: int,
    aspect_ratio: float,
) -> None:
    if n_fingers not in {2, 3}:
        raise ValueError(f"n_fingers must be 2 or 3; got {n_fingers!r}")
    for name, values in (
        ("finger_side_offsets", finger_side_offsets),
        ("finger_normal_offsets", finger_normal_offsets),
        ("finger_angles", finger_angles),
    ):
        if len(values) < n_fingers:
            raise ValueError(f"{name} must contain at least {n_fingers} values")
        if not all(math.isfinite(float(value)) for value in values[:n_fingers]):
            raise ValueError(f"{name} must contain only finite values")
    if not all(math.isfinite(value) and value > 0.0 for value in (half_x, half_y, half_z)):
        raise ValueError("palm half extents must be finite and > 0")
    if polygon_sides < 4:
        raise ValueError(f"polygon_sides must be >= 4; got {polygon_sides!r}")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be finite and > 0; got {aspect_ratio!r}")


def _build_outline(spec: PalmOutlineSpec) -> tuple[tuple[float, float], ...]:
    half_y = spec.half_y * spec.aspect_ratio
    if spec.polygon_sides == 8:
        clip = spec.corner_rounding
        return (
            (-spec.half_x + clip, -half_y),
            (spec.half_x - clip, -half_y),
            (spec.half_x, -half_y + clip),
            (spec.half_x, half_y - clip),
            (spec.half_x - clip, half_y),
            (-spec.half_x + clip, half_y),
            (-spec.half_x, half_y - clip),
            (-spec.half_x, -half_y + clip),
        )

    # Start at the lower-most point and proceed counterclockwise. Scaling a
    # regular polygon by the requested half extents keeps the outline convex.
    raw = [
        (
            math.cos(-0.5 * math.pi + 2.0 * math.pi * index / spec.polygon_sides),
            math.sin(-0.5 * math.pi + 2.0 * math.pi * index / spec.polygon_sides),
        )
        for index in range(spec.polygon_sides)
    ]
    max_abs_x = max(abs(x) for x, _ in raw)
    max_abs_y = max(abs(y) for _, y in raw)
    return tuple(
        (spec.half_x * x / max_abs_x, half_y * y / max_abs_y)
        for x, y in raw
    )


def _extrude_outline(
    outline: tuple[tuple[float, float], ...],
    half_z: float,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    count = len(outline)
    vertices = tuple((x, y, -half_z) for x, y in outline) + tuple(
        (x, y, half_z) for x, y in outline
    )
    faces: list[tuple[int, int, int]] = []

    for index in range(1, count - 1):
        faces.append((0, index + 1, index))
        faces.append((count, count + index, count + index + 1))
    for index in range(count):
        next_index = (index + 1) % count
        faces.extend(
            (
                (index, next_index, count + next_index),
                (index, count + next_index, count + index),
            )
        )
    return vertices, tuple(faces)


def _build_base_frames(
    outline: tuple[tuple[float, float], ...],
    *,
    n_fingers: int,
    finger_side_offsets: tuple[float, ...],
    finger_normal_offsets: tuple[float, ...],
    finger_angles: tuple[float, ...],
    thumb_side_offset: float,
    thumb_normal_offset: float,
    thumb_angle: float,
) -> dict[str, PalmBaseFrame]:
    slots = (-0.035, 0.035) if n_fingers == 2 else (-0.045, 0.0, 0.045)
    front_edge = _most_outward_edge(outline, direction=(1.0, 0.0))
    frames: dict[str, PalmBaseFrame] = {}
    for index, slot in enumerate(slots):
        anchor = _point_on_edge_near_coordinate(front_edge, slot, coordinate=1)
        tangent, outward_normal = _edge_frame(front_edge)
        if tangent[1] < 0.0:
            tangent = (-tangent[0], -tangent[1])
        pos_2d = _offset_from_anchor(
            anchor,
            tangent,
            outward_normal,
            side_offset=finger_side_offsets[index],
            normal_offset=-0.047 + finger_normal_offsets[index],
        )
        frames[f"finger{index + 1}"] = PalmBaseFrame(
            name=f"finger{index + 1}",
            pos=(pos_2d[0], pos_2d[1], 0.020),
            yaw=math.atan2(outward_normal[1], outward_normal[0]) + finger_angles[index],
            tangent=(tangent[0], tangent[1], 0.0),
            outward_normal=(outward_normal[0], outward_normal[1], 0.0),
            anchor_2d=anchor,
        )

    thumb_edge = _most_outward_edge(outline, direction=(0.0, -1.0))
    thumb_anchor = _point_on_edge_near_coordinate(thumb_edge, -0.022, coordinate=0)
    thumb_tangent, thumb_normal = _edge_frame(thumb_edge)
    if thumb_tangent[0] < 0.0:
        thumb_tangent = (-thumb_tangent[0], -thumb_tangent[1])
    thumb_pos_2d = _offset_from_anchor(
        thumb_anchor,
        thumb_tangent,
        thumb_normal,
        side_offset=thumb_side_offset,
        normal_offset=-0.047 + thumb_normal_offset,
    )
    frames["thumb"] = PalmBaseFrame(
        name="thumb",
        pos=(thumb_pos_2d[0], thumb_pos_2d[1], 0.012),
        yaw=-1.15 + thumb_angle,
        tangent=(thumb_tangent[0], thumb_tangent[1], 0.0),
        outward_normal=(thumb_normal[0], thumb_normal[1], 0.0),
        anchor_2d=thumb_anchor,
    )
    return frames


def _most_outward_edge(
    vertices: tuple[tuple[float, float], ...],
    direction: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    edges = tuple(zip(vertices, vertices[1:] + vertices[:1], strict=True))
    return max(
        edges,
        key=lambda edge: (
            0.5 * (edge[0][0] + edge[1][0]) * direction[0]
            + 0.5 * (edge[0][1] + edge[1][1]) * direction[1],
            _edge_length(edge),
        ),
    )


def _point_on_edge_near_coordinate(
    edge: tuple[tuple[float, float], tuple[float, float]],
    target: float,
    *,
    coordinate: int,
) -> tuple[float, float]:
    start, end = edge
    delta = end[coordinate] - start[coordinate]
    if abs(delta) < 1e-12:
        t = 0.5
    else:
        t = min(1.0, max(0.0, (target - start[coordinate]) / delta))
    return (
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
    )


def _edge_frame(
    edge: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    start, end = edge
    length = _edge_length(edge)
    tangent = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
    # Vertices are counterclockwise, so the exterior lies to the right.
    outward_normal = (tangent[1], -tangent[0])
    return tangent, outward_normal


def _edge_length(
    edge: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    return math.hypot(edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])


def _offset_from_anchor(
    anchor: tuple[float, float],
    tangent: tuple[float, float],
    outward_normal: tuple[float, float],
    *,
    side_offset: float,
    normal_offset: float,
) -> tuple[float, float]:
    return (
        anchor[0] + side_offset * tangent[0] + normal_offset * outward_normal[0],
        anchor[1] + side_offset * tangent[1] + normal_offset * outward_normal[1],
    )
