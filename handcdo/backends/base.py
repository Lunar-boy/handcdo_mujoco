from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from handcdo.design_space import HandDesign
    from handcdo.grasp_sampling import GraspParams
    from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


class SimulatorBackend(Protocol):
    name: str

    def evaluate_grasp(
        self,
        design: "HandDesign",
        tool_name: str,
        grasp: "GraspParams",
        config: "EvaluationConfig | None",
    ) -> "GraspEvaluation":
        ...
