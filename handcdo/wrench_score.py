from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np


WRENCH_DIRECTIONS: tuple[tuple[str, np.ndarray, np.ndarray], ...] = (
    ("force_pos_x", np.array([1.0, 0.0, 0.0]), np.zeros(3)),
    ("force_neg_x", np.array([-1.0, 0.0, 0.0]), np.zeros(3)),
    ("force_pos_y", np.array([0.0, 1.0, 0.0]), np.zeros(3)),
    ("force_neg_y", np.array([0.0, -1.0, 0.0]), np.zeros(3)),
    ("force_pos_z", np.array([0.0, 0.0, 1.0]), np.zeros(3)),
    ("force_neg_z", np.array([0.0, 0.0, -1.0]), np.zeros(3)),
    ("torque_pos_x", np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("torque_neg_x", np.zeros(3), np.array([-1.0, 0.0, 0.0])),
    ("torque_pos_y", np.zeros(3), np.array([0.0, 1.0, 0.0])),
    ("torque_neg_y", np.zeros(3), np.array([0.0, -1.0, 0.0])),
    ("torque_pos_z", np.zeros(3), np.array([0.0, 0.0, 1.0])),
    ("torque_neg_z", np.zeros(3), np.array([0.0, 0.0, -1.0])),
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


def rotation_error_from_mats(r0: np.ndarray, r1: np.ndarray) -> float:
    rel = r0.reshape(3, 3).T @ r1.reshape(3, 3)
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cos_angle))


def aggregate_wrench_results(results: list[WrenchDirectionResult]) -> float:
    if not results:
        return 0.0
    return float(np.clip(np.mean([r.normalized_duration for r in results]), 0.0, 1.0))
