from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WrenchDirection:
    name: str
    force: tuple[float, float, float]
    torque: tuple[float, float, float]


def canonical_wrench_directions() -> tuple[WrenchDirection, ...]:
    return (
        WrenchDirection("+Fx", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        WrenchDirection("-Fx", (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        WrenchDirection("+Fy", (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        WrenchDirection("-Fy", (0.0, -1.0, 0.0), (0.0, 0.0, 0.0)),
        WrenchDirection("+Fz", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        WrenchDirection("-Fz", (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
        WrenchDirection("+Tx", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        WrenchDirection("-Tx", (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        WrenchDirection("+Ty", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        WrenchDirection("-Ty", (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        WrenchDirection("+Tz", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        WrenchDirection("-Tz", (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    )


WRENCH_DIRECTIONS: tuple[tuple[str, np.ndarray, np.ndarray], ...] = tuple(
    (legacy_name, np.asarray(direction.force), np.asarray(direction.torque))
    for legacy_name, direction in zip(
        (
            "force_pos_x",
            "force_neg_x",
            "force_pos_y",
            "force_neg_y",
            "force_pos_z",
            "force_neg_z",
            "torque_pos_x",
            "torque_neg_x",
            "torque_pos_y",
            "torque_neg_y",
            "torque_pos_z",
            "torque_neg_z",
        ),
        canonical_wrench_directions(),
    )
)


@dataclass(frozen=True)
class WrenchDirectionResult:
    direction: str
    stable_steps: int
    total_steps: int
    normalized_duration: float
    max_translation: float
    max_rotation_rad: float
    failed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_stable_time(stable_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 0.0
    return float(np.clip(stable_steps / total_steps, 0.0, 1.0))


def rotation_error_from_mats(r0: np.ndarray, r1: np.ndarray) -> float:
    rel = r0.reshape(3, 3).T @ r1.reshape(3, 3)
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cos_angle))


def aggregate_wrench_results(results: list[WrenchDirectionResult]) -> float:
    if not results:
        return 0.0
    return float(np.clip(np.mean([r.normalized_duration for r in results]), 0.0, 1.0))
