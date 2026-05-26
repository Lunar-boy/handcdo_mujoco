from __future__ import annotations

from dataclasses import dataclass
import math

from .design_space import HandDesign


@dataclass(frozen=True)
class JointSpec:
    name: str
    axis: tuple[float, float, float]
    range: tuple[float, float]


@dataclass(frozen=True)
class LinkSpec:
    name: str
    length: float
    radius: float
    joint: JointSpec
    fingertip: bool = False


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
    palm_size: tuple[float, float, float]
    digits: tuple[DigitSpec, ...]
    palm_pads: tuple[PalmPad, ...]

    @property
    def joint_names(self) -> list[str]:
        return [link.joint.name for digit in self.digits for link in digit.links]


def build_hand_model(design: HandDesign) -> HandModel:
    p = design.params
    palm_size = (
        0.085 + 0.5 * p["palm_kernel_max_height"],
        0.115,
        0.032 + p["palm_kernel_max_height"],
    )
    base_lengths = [0.044, 0.034, 0.027, 0.022]
    lengths = [
        max(0.018, base_lengths[i] + p[f"added_link_length_{i + 1}"])
        for i in range(4)
    ]
    fingers: list[DigitSpec] = []
    n_fingers = int(p["finger_number"])
    y_slots = [-0.035, 0.035] if n_fingers == 2 else [-0.045, 0.0, 0.045]
    for idx, y in enumerate(y_slots, start=1):
        side = p[f"finger_side_offset_{idx}"]
        normal = p[f"finger_normal_offset_{idx}"]
        yaw = p[f"finger_angle_{idx}"]
        link_count = 3 if p["finger_code"] == "1-1-1" else 4
        links = []
        for j in range(link_count):
            fingertip = j == link_count - 1
            radius = 0.009
            if fingertip:
                radius *= 0.5 * (p["fingertip_scale_y"] + p["fingertip_scale_z"])
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
                )
            )
        fingers.append(
            DigitSpec(
                name=f"finger{idx}",
                base_pos=(0.038 + normal, y + side, 0.02),
                base_yaw=yaw,
                links=tuple(links),
            )
        )

    thumb_links = []
    thumb_link_count = 3 if p["thumb_code"] == "1-22" else 4
    for j in range(thumb_link_count):
        axis = (0.0, 1.0, 0.0) if j else (0.0, 0.0, 1.0)
        thumb_links.append(
            LinkSpec(
                name=f"thumb_link{j + 1}",
                length=max(0.018, lengths[min(j, 3)] * 0.9),
                radius=0.010 if j < thumb_link_count - 1 else 0.0105 * p["fingertip_scale_y"],
                joint=JointSpec(name=f"thumb_joint{j + 1}", axis=axis, range=(-0.7, 1.2)),
                fingertip=j == thumb_link_count - 1,
            )
        )
    fingers.append(
        DigitSpec(
            name="thumb",
            base_pos=(-0.022 + p["thumb_normal_offset"], -0.068 + p["thumb_side_offset"], 0.012),
            base_yaw=-1.15 + p["thumb_angle"],
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
    return HandModel(design=design, palm_size=palm_size, digits=tuple(fingers), palm_pads=tuple(pads))
