from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .design_space import HandDesign
from .geometry_config import GeometryConfig
from .grasp_sampling import GraspParams
from .hand_model import build_hand_model
from .mjcf_generator import build_mjcf_xml
from .tools import ToolSpec, get_tool
from .wrench_score import (
    WRENCH_DIRECTIONS,
    WrenchDirectionResult,
    aggregate_wrench_results,
    normalized_stable_time,
    rotation_error_from_mats,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    settle_steps: int = 250
    close_steps: int = 350
    wrench_steps: int = 250
    timestep: float = 0.002
    translation_threshold: float = 0.045
    rotation_threshold_rad: float = 0.55
    force_magnitude: float | None = None
    torque_magnitude: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvaluationConfig":
        data = data or {}
        sim = data.get("simulation", data)
        wrench = data.get("wrench", {})
        return cls(
            settle_steps=int(sim.get("settle_steps", 250)),
            close_steps=int(sim.get("close_steps", 350)),
            wrench_steps=int(sim.get("wrench_steps", 250)),
            timestep=float(sim.get("timestep", 0.002)),
            translation_threshold=float(wrench.get("translation_threshold", sim.get("translation_threshold", 0.045))),
            rotation_threshold_rad=float(wrench.get("rotation_threshold_rad", sim.get("rotation_threshold_rad", 0.55))),
            force_magnitude=_optional_float(wrench.get("force_magnitude")),
            torque_magnitude=_optional_float(wrench.get("torque_magnitude")),
        )


@dataclass(frozen=True)
class GraspEvaluation:
    design_id: str
    tool: str
    grasp: dict[str, float]
    score: float
    wrench_results: list[dict[str, Any]]
    failed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_model(xml: str):
    import mujoco

    with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        return mujoco.MjModel.from_xml_path(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _set_tool_pose(model: Any, data: Any, tool: ToolSpec, grasp: GraspParams) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tool_free")
    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr : qpos_addr + 3] = np.array(tool.reference_pos) + np.array([grasp.dx, grasp.dy, grasp.dz])
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([grasp.roll, grasp.pitch, grasp.yaw]), "XYZ")
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat


def _close_hand(model: Any, data: Any, grasp: GraspParams, steps: int) -> None:
    import mujoco

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        base = grasp.thumb_closure if name.startswith("thumb") else grasp.closure
        data.ctrl[i] = np.clip(base + grasp.spread_bias * (i % 2), -0.25, 1.3)
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _settle(model: Any, data: Any, steps: int) -> None:
    import mujoco

    for _ in range(steps):
        mujoco.mj_step(model, data)


def _run_wrench_tests(model: Any, data: Any, tool: ToolSpec, config: EvaluationConfig) -> list[WrenchDirectionResult]:
    import mujoco

    tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = data.ctrl.copy()
    results: list[WrenchDirectionResult] = []
    force_magnitude = tool.force_limit if config.force_magnitude is None else config.force_magnitude
    torque_magnitude = tool.torque_limit if config.torque_magnitude is None else config.torque_magnitude
    for name, force_dir, torque_dir in WRENCH_DIRECTIONS:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = ctrl0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        start_pos = data.xpos[tool_body].copy()
        start_mat = data.xmat[tool_body].copy()
        stable_steps = config.wrench_steps
        max_trans = 0.0
        max_rot = 0.0
        failed = False
        for step in range(config.wrench_steps):
            scale = (step + 1) / config.wrench_steps
            data.xfrc_applied[tool_body, :3] = force_dir * force_magnitude * scale
            data.xfrc_applied[tool_body, 3:] = torque_dir * torque_magnitude * scale
            mujoco.mj_step(model, data)
            trans = float(np.linalg.norm(data.xpos[tool_body] - start_pos))
            rot = rotation_error_from_mats(start_mat, data.xmat[tool_body])
            max_trans = max(max_trans, trans)
            max_rot = max(max_rot, rot)
            if trans > config.translation_threshold or rot > config.rotation_threshold_rad:
                stable_steps = step
                failed = True
                break
        results.append(
            WrenchDirectionResult(
                direction=name,
                stable_steps=int(stable_steps),
                total_steps=int(config.wrench_steps),
                normalized_duration=normalized_stable_time(stable_steps, config.wrench_steps),
                max_translation=max_trans,
                max_rotation_rad=max_rot,
                failed=failed,
            )
        )
    data.xfrc_applied[:] = 0.0
    return results


def evaluate_grasp(
    design: HandDesign,
    tool_name: str,
    grasp: GraspParams,
    config: EvaluationConfig | None = None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = Path("assets/tools"),
) -> GraspEvaluation:
    config = config or EvaluationConfig()
    geometry_config = geometry_config or GeometryConfig()
    tool = get_tool(tool_name)
    try:
        import mujoco

        xml = build_mjcf_xml(
            build_hand_model(design),
            tool=tool,
            geometry_config=geometry_config,
            tool_assets_dir=Path(tool_assets_dir),
        )
        model = _load_model(xml)
        model.opt.timestep = config.timestep
        data = mujoco.MjData(model)
        _set_tool_pose(model, data, tool, grasp)
        mujoco.mj_forward(model, data)
        _close_hand(model, data, grasp, config.close_steps)
        _settle(model, data, config.settle_steps)
        wrench = _run_wrench_tests(model, data, tool, config)
        score = aggregate_wrench_results(wrench)
        return GraspEvaluation(
            design_id=design.design_id,
            tool=tool_name,
            grasp=grasp.to_dict(),
            score=score,
            wrench_results=[r.to_dict() for r in wrench],
        )
    except Exception as exc:
        LOGGER.exception("Simulation failed for design=%s tool=%s", design.design_id, tool_name)
        return GraspEvaluation(
            design_id=design.design_id,
            tool=tool_name,
            grasp=grasp.to_dict(),
            score=0.0,
            wrench_results=[],
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
