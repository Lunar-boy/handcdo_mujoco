from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


@runtime_checkable
class BatchedSimulatorBackend(Protocol):
    name: str

    def evaluate_grasps_batch(
        self,
        design: HandDesign,
        tool_name: str,
        grasps: list[GraspParams],
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None = None,
        tool_assets_dir: str | Path = "assets/tools",
    ) -> list[GraspEvaluation]:
        ...


def supports_batched_grasps(backend: object) -> bool:
    return callable(getattr(backend, "evaluate_grasps_batch", None))
