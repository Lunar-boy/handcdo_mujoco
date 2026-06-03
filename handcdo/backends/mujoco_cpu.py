from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from handcdo.design_space import HandDesign
    from handcdo.geometry_config import GeometryConfig
    from handcdo.grasp_sampling import GraspParams
    from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


class MujocoCpuBackend:
    name = "mujoco_cpu"

    def evaluate_grasp(
        self,
        design: "HandDesign",
        tool_name: str,
        grasp: "GraspParams",
        config: "EvaluationConfig | None",
        geometry_config: "GeometryConfig | None" = None,
    ) -> "GraspEvaluation":
        from handcdo.mujoco_eval import evaluate_grasp

        return evaluate_grasp(design, tool_name, grasp, config, geometry_config=geometry_config)
