from __future__ import annotations

from dataclasses import dataclass
import math

from .design_space import DEFAULT_PALM_OUTLINE_PARAMETERS, HandDesign
from .palm_outline import PalmBodySpec, build_palm_outline_body


@dataclass(frozen=True)
class JointSpec:
    name: str
    axis: tuple[float, float, float]
    range: tuple[float, float]


@dataclass(frozen=True)
class FingertipGeometry:
    half_x: float
    half_y: float
    half_z: float
    shaft_length: float


@dataclass(frozen=True)
class LinkSpec:
    name: str
    length: float
    radius: float
    joint: JointSpec
    fingertip: bool = False
    fingertip_geometry: FingertipGeometry | None = None


@dataclass(frozen=True)
class DigitSpec:
    name: str
    base_pos: tuple[float, float, float]
    base_yaw: float
    links: tuple[LinkSpec, ...]


@dataclass(frozen=True)
class PalmPad:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class HandModel:
    design: HandDesign
    palm_body: PalmBodySpec
    digits: tuple[DigitSpec, ...]
    palm_pads: tuple[PalmPad, ...]

    @property
    def palm_size(self) -> tuple[float, float, float]:
        return self.palm_body.half_extents

    @property
    def joint_names(self) -> list[str]:
        return [link.joint.name for digit in self.digits for link in digit.links]


def build_hand_model(design: HandDesign) -> HandModel:
    p = design.params
    n_fingers = int(p["finger_number"])
    palm_body = build_palm_outline_body(
        n_fingers=n_fingers,
        finger_side_offsets=tuple(
            p[f"finger_side_offset_{index}"] for index in range(1, n_fingers + 1)
        ),
        finger_normal_offsets=tuple(
            p[f"finger_normal_offset_{index}"] for index in range(1, n_fingers + 1)
        ),
        finger_angles=tuple(
            p[f"finger_angle_{index}"] for index in range(1, n_fingers + 1)
        ),
        thumb_side_offset=p["thumb_side_offset"],
        thumb_normal_offset=p["thumb_normal_offset"],
        thumb_angle=p["thumb_angle"],
        half_x=p.get("palm_half_x", DEFAULT_PALM_OUTLINE_PARAMETERS["palm_half_x"]),
        half_y=p.get("palm_half_y", DEFAULT_PALM_OUTLINE_PARAMETERS["palm_half_y"]),
        half_z=p.get("palm_half_z", DEFAULT_PALM_OUTLINE_PARAMETERS["palm_half_z"]),
        polygon_sides=int(
            round(
                p.get(
                    "palm_polygon_sides",
                    DEFAULT_PALM_OUTLINE_PARAMETERS["palm_polygon_sides"],
                )
            )
        ),
        aspect_ratio=p.get(
            "palm_aspect_ratio",
            DEFAULT_PALM_OUTLINE_PARAMETERS["palm_aspect_ratio"],
        ),
    )
    palm_size = palm_body.half_extents
    base_lengths = [0.044, 0.034, 0.027, 0.022]
    lengths = [
        max(0.018, base_lengths[i] + p[f"added_link_length_{i + 1}"])
        for i in range(4)
    ]
    fingers: list[DigitSpec] = []
    for idx in range(1, n_fingers + 1):
        link_count = 3 if p["finger_code"] == "1-1-1" else 4
        links = []
        for j in range(link_count):
            fingertip = j == link_count - 1
            radius = 0.009
            fingertip_geometry = None
            if fingertip:
                radius *= 0.5 * (p["fingertip_scale_y"] + p["fingertip_scale_z"])
                half_x = min(0.010, max(0.004, 0.25 * lengths[min(j, 3)]))
                fingertip_geometry = FingertipGeometry(
                    half_x=half_x,
                    half_y=0.009 * p["fingertip_scale_y"],
                    half_z=0.009 * p["fingertip_scale_z"],
                    shaft_length=max(0.0, lengths[min(j, 3)] - 2.0 * half_x),
                )
            links.append(
                LinkSpec(
                    name=f"finger{idx}_link{j + 1}",
                    length=lengths[min(j, 3)],
                    radius=radius,
                    joint=JointSpec(
                        name=f"finger{idx}_joint{j + 1}",
                        axis=(0.0, 1.0, 0.0),
                        range=(-0.25, 1.35 if j else 1.1),
                    ),
                    fingertip=fingertip,
                    fingertip_geometry=fingertip_geometry,
                )
            )
        fingers.append(
            DigitSpec(
                name=f"finger{idx}",
                base_pos=palm_body.base_frames[f"finger{idx}"].pos,
                base_yaw=palm_body.base_frames[f"finger{idx}"].yaw,
                links=tuple(links),
            )
        )

    thumb_links = []
    thumb_link_count = 3 if p["thumb_code"] == "1-22" else 4
    for j in range(thumb_link_count):
        axis = (0.0, 1.0, 0.0) if j else (0.0, 0.0, 1.0)
        fingertip = j == thumb_link_count - 1
        thumb_link_length = max(0.018, lengths[min(j, 3)] * 0.9)
        fingertip_geometry = None
        if fingertip:
            half_x = min(0.010, max(0.004, 0.25 * thumb_link_length))
            fingertip_geometry = FingertipGeometry(
                half_x=half_x,
                half_y=0.0105 * p["fingertip_scale_y"],
                half_z=0.0105 * p["fingertip_scale_z"],
                shaft_length=max(0.0, thumb_link_length - 2.0 * half_x),
            )
        thumb_links.append(
            LinkSpec(
                name=f"thumb_link{j + 1}",
                length=thumb_link_length,
                radius=0.010 if not fingertip else 0.0105 * p["fingertip_scale_y"],
                joint=JointSpec(name=f"thumb_joint{j + 1}", axis=axis, range=(-0.7, 1.2)),
                fingertip=fingertip,
                fingertip_geometry=fingertip_geometry,
            )
        )
    fingers.append(
        DigitSpec(
            name="thumb",
            base_pos=palm_body.base_frames["thumb"].pos,
            base_yaw=palm_body.base_frames["thumb"].yaw,
            links=tuple(thumb_links),
        )
    )

    pads: list[PalmPad] = []
    for i in (1, 2):
        angle = p[f"palm_kernel_center_angle_{i}"]
        offset = p[f"palm_kernel_center_offset_{i}"]
        spread = p[f"palm_kernel_spread_{i}"]
        intensity = p[f"palm_kernel_intensity_ratio_{i}"]
        r = 0.035 + offset
        pads.append(
            PalmPad(
                name=f"palm_kernel_pad_{i}",
                pos=(r * math.cos(angle), r * math.sin(angle), palm_size[2] + 0.006),
                size=(spread * 0.45, spread * 0.32, 0.004 + 0.012 * intensity * p["palm_kernel_max_height"]),
            )
        )
    return HandModel(
        design=design,
        palm_body=palm_body,
        digits=tuple(fingers),
        palm_pads=tuple(pads),
    )
