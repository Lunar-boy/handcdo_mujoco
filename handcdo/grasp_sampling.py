from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraspParams:
    dx: float
    dy: float
    dz: float
    yaw: float
    pitch: float
    roll: float
    closure: float
    thumb_closure: float
    spread_bias: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def sample_random_grasp(seed: int | None = None, rng: np.random.Generator | None = None) -> GraspParams:
    rng = rng or np.random.default_rng(seed)
    return GraspParams(
        dx=float(rng.uniform(-0.025, 0.025)),
        dy=float(rng.uniform(-0.035, 0.035)),
        dz=float(rng.uniform(-0.012, 0.018)),
        yaw=float(rng.uniform(-0.45, 0.45)),
        pitch=float(rng.uniform(-0.25, 0.25)),
        roll=float(rng.uniform(-0.25, 0.25)),
        closure=float(rng.uniform(0.45, 1.25)),
        thumb_closure=float(rng.uniform(0.35, 1.15)),
        spread_bias=float(rng.uniform(-0.18, 0.18)),
    )


def optuna_suggest_grasp(trial: Any) -> GraspParams:
    return GraspParams(
        dx=trial.suggest_float("dx", -0.025, 0.025),
        dy=trial.suggest_float("dy", -0.035, 0.035),
        dz=trial.suggest_float("dz", -0.012, 0.018),
        yaw=trial.suggest_float("yaw", -0.45, 0.45),
        pitch=trial.suggest_float("pitch", -0.25, 0.25),
        roll=trial.suggest_float("roll", -0.25, 0.25),
        closure=trial.suggest_float("closure", 0.45, 1.25),
        thumb_closure=trial.suggest_float("thumb_closure", 0.35, 1.15),
        spread_bias=trial.suggest_float("spread_bias", -0.18, 0.18),
    )
