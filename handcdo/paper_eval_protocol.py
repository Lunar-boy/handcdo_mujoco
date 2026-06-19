from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .backends import SimulatorBackend, get_backend
from .design_space import DesignSpace, HandDesign
from .geometry_config import GeometryConfig
from .grasp_sampling import GraspParams
from .mujoco_eval import EvaluationConfig, GraspEvaluation
from .optimize_grasp import optimize_grasp_for_tool
from .tools import get_tool
from .utils import ensure_dir, read_yaml, write_json
from .wrench_score import canonical_wrench_directions, normalized_stable_time


@dataclass(frozen=True)
class WrenchEvaluationResult:
    direction: str
    stable_time: float
    normalized_stable_time: float
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WrenchEvaluationResult":
        return cls(**data)


@dataclass(frozen=True)
class ToolEvaluationResult:
    tool: str
    grasp_id: str
    wrench_results: tuple[WrenchEvaluationResult, ...]
    mean_score: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolEvaluationResult":
        return cls(
            tool=str(data["tool"]),
            grasp_id=str(data["grasp_id"]),
            wrench_results=tuple(WrenchEvaluationResult.from_dict(item) for item in data["wrench_results"]),
            mean_score=float(data["mean_score"]),
        )


@dataclass(frozen=True)
class DesignEvaluationResult:
    design_id: str
    geometry_config: str
    tools: tuple[ToolEvaluationResult, ...]
    aggregate_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignEvaluationResult":
        return cls(
            design_id=str(data["design_id"]),
            geometry_config=str(data["geometry_config"]),
            tools=tuple(ToolEvaluationResult.from_dict(item) for item in data["tools"]),
            aggregate_score=float(data["aggregate_score"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DesignEvaluationResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class WrenchEvaluator(Protocol):
    def evaluate_grasp(
        self,
        design: HandDesign,
        tool_name: str,
        grasp: GraspParams,
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None = None,
        tool_assets_dir: str | Path = "assets/tools",
    ) -> GraspEvaluation:
        ...


@dataclass(frozen=True)
class PaperEvaluationConfig:
    config_path: str
    geometry_config_path: str
    tools: tuple[str, ...]
    candidates_per_tool: int
    sampler: str
    seed: int
    evaluation: EvaluationConfig
    geometry: GeometryConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PaperEvaluationConfig":
        path = Path(path)
        data = read_yaml(path)
        protocol = data.get("evaluation", {})
        geometry_path = Path(str(data["geometry_config"]))
        if not geometry_path.is_absolute():
            candidates = (Path.cwd() / geometry_path, path.parent.parent / geometry_path, path.parent / geometry_path)
            geometry_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        geometry_data = read_yaml(geometry_path)
        rollout = protocol.get("rollout", {})
        stability = protocol.get("stability", {})
        wrench = protocol.get("wrench", {})
        grasp = protocol.get("grasp", {})
        timestep = float(rollout.get("timestep", 0.002))
        settle_steps = _seconds_to_steps(float(rollout.get("settle_time", 0.5)), timestep)
        wrench_steps = _seconds_to_steps(float(rollout.get("perturb_time", 1.0)), timestep)
        max_steps = int(rollout.get("max_steps", settle_steps + wrench_steps))
        if settle_steps + wrench_steps > max_steps:
            raise ValueError("evaluation.rollout.max_steps is smaller than settle plus perturb steps")
        eval_data = {
            "simulation": {
                "timestep": timestep,
                "settle_steps": settle_steps,
                "close_steps": int(rollout.get("close_steps", geometry_data.get("simulation", {}).get("close_steps", 350))),
                "wrench_steps": wrench_steps,
            },
            "wrench": {
                "translation_threshold": stability.get("displacement_threshold", 0.05),
                "rotation_threshold_rad": stability.get("rotation_threshold_rad", 0.8),
                "force_magnitude": wrench.get("force_magnitude", 1.0),
                "torque_magnitude": wrench.get("torque_magnitude", 0.05),
            },
        }
        tools = tuple(str(tool) for tool in protocol.get("tools", ("hammer", "spoon", "knife")))
        for tool in tools:
            get_tool(tool)
        if wrench.get("directions", "canonical_12") != "canonical_12":
            raise ValueError("Only evaluation.wrench.directions=canonical_12 is supported")
        return cls(
            config_path=str(path),
            geometry_config_path=str(geometry_path),
            tools=tools,
            candidates_per_tool=int(grasp.get("candidates_per_tool", 8)),
            sampler=str(grasp.get("sampler", "random")),
            seed=int(grasp.get("seed", 0)),
            evaluation=EvaluationConfig.from_dict(eval_data),
            geometry=GeometryConfig.from_dict(geometry_data),
        )


class DeterministicSmokeBackend:
    name = "deterministic_smoke"

    def evaluate_grasp(
        self,
        design: HandDesign,
        tool_name: str,
        grasp: GraspParams,
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None = None,
        tool_assets_dir: str | Path = "assets/tools",
    ) -> GraspEvaluation:
        del geometry_config, tool_assets_dir
        config = config or EvaluationConfig()
        wrench_results = []
        grasp_dict = grasp.to_dict()
        for direction in canonical_wrench_directions():
            unit_value = _deterministic_unit_value(design.design_id, tool_name, direction.name, grasp_dict)
            stable_steps = int(round(unit_value * config.wrench_steps))
            score = normalized_stable_time(stable_steps, config.wrench_steps)
            wrench_results.append(
                {
                    "direction": direction.name,
                    "stable_steps": stable_steps,
                    "total_steps": config.wrench_steps,
                    "normalized_duration": score,
                    "max_translation": 0.0,
                    "max_rotation_rad": 0.0,
                    "failed": stable_steps < config.wrench_steps,
                }
            )
        score = aggregate_tool_score([item["normalized_duration"] for item in wrench_results])
        return GraspEvaluation(
            design_id=design.design_id,
            tool=tool_name,
            grasp=grasp_dict,
            score=score,
            wrench_results=wrench_results,
        )


def aggregate_tool_score(wrench_scores: list[float] | tuple[float, ...]) -> float:
    if not wrench_scores:
        return 0.0
    return float(np.clip(np.mean(wrench_scores), 0.0, 1.0))


def aggregate_best_grasp(grasp_scores: list[list[float]] | tuple[tuple[float, ...], ...]) -> float:
    if not grasp_scores:
        return 0.0
    return max(aggregate_tool_score(scores) for scores in grasp_scores)


def evaluate_design_protocol(
    design: HandDesign,
    config: PaperEvaluationConfig,
    *,
    backend: SimulatorBackend | WrenchEvaluator | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> DesignEvaluationResult:
    backend = backend or get_backend("mujoco_cpu")
    tool_results: list[ToolEvaluationResult] = []
    for index, tool_name in enumerate(config.tools):
        optimized = optimize_grasp_for_tool(
            design,
            tool_name,
            n_trials=config.candidates_per_tool,
            seed=config.seed + 1009 * (index + 1),
            config=config.evaluation,
            geometry_config=config.geometry,
            sampler=config.sampler,
            backend=backend,
            tool_assets_dir=tool_assets_dir,
        )
        best = optimized["best_grasp"]
        wrench_results = tuple(_convert_wrench_result(item, config.evaluation.timestep) for item in best.get("wrench_results", []))
        grasp = best.get("grasp", {})
        tool_results.append(
            ToolEvaluationResult(
                tool=tool_name,
                grasp_id=_grasp_id(grasp),
                wrench_results=wrench_results,
                mean_score=aggregate_tool_score([item.normalized_stable_time for item in wrench_results]),
            )
        )
    return DesignEvaluationResult(
        design_id=design.design_id,
        geometry_config=config.geometry_config_path,
        tools=tuple(tool_results),
        aggregate_score=aggregate_tool_score([result.mean_score for result in tool_results]),
    )


def run_paper_evaluation(
    config_path: str | Path,
    *,
    num_designs: int,
    output_dir: str | Path,
    seed: int | None = None,
    backend_name: str = "deterministic_smoke",
    search_space_path: str | Path = "configs/search_space.yaml",
) -> list[DesignEvaluationResult]:
    config = PaperEvaluationConfig.from_yaml(config_path)
    if seed is not None:
        config = PaperEvaluationConfig(
            config_path=config.config_path,
            geometry_config_path=config.geometry_config_path,
            tools=config.tools,
            candidates_per_tool=config.candidates_per_tool,
            sampler=config.sampler,
            seed=seed,
            evaluation=config.evaluation,
            geometry=config.geometry,
        )
    backend: SimulatorBackend | WrenchEvaluator
    backend = DeterministicSmokeBackend() if backend_name == "deterministic_smoke" else get_backend(backend_name)
    space = DesignSpace.from_yaml(search_space_path)
    designs = [space.sample(seed=config.seed + index) for index in range(num_designs)]
    results = [evaluate_design_protocol(design, config, backend=backend) for design in designs]
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "results.json", [result.to_dict() for result in results])
    _write_wrench_csv(output_dir / "results.csv", results)
    write_json(
        output_dir / "run_config.json",
        {
            "protocol_config": str(config_path),
            "geometry_config": config.geometry_config_path,
            "backend": backend_name,
            "seed": config.seed,
            "num_designs": num_designs,
            "tools": list(config.tools),
            "candidates_per_tool": config.candidates_per_tool,
        },
    )
    return results


def _convert_wrench_result(data: dict[str, Any], timestep: float) -> WrenchEvaluationResult:
    stable_steps = int(data.get("stable_steps", 0))
    total_steps = int(data.get("total_steps", 0))
    failed = bool(data.get("failed", stable_steps < total_steps))
    return WrenchEvaluationResult(
        direction=_canonical_direction_name(str(data["direction"])),
        stable_time=stable_steps * timestep,
        normalized_stable_time=normalized_stable_time(stable_steps, total_steps),
        failure_reason="stability_threshold_exceeded" if failed else None,
    )


def _write_wrench_csv(path: Path, results: list[DesignEvaluationResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_id",
                "geometry_config",
                "aggregate_score",
                "tool",
                "grasp_id",
                "tool_mean_score",
                "direction",
                "stable_time",
                "normalized_stable_time",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for design in results:
            for tool in design.tools:
                for wrench in tool.wrench_results:
                    writer.writerow(
                        {
                            "design_id": design.design_id,
                            "geometry_config": design.geometry_config,
                            "aggregate_score": design.aggregate_score,
                            "tool": tool.tool,
                            "grasp_id": tool.grasp_id,
                            "tool_mean_score": tool.mean_score,
                            **asdict(wrench),
                        }
                    )


def _seconds_to_steps(seconds: float, timestep: float) -> int:
    if timestep <= 0.0:
        raise ValueError("evaluation.rollout.timestep must be positive")
    return max(1, int(round(seconds / timestep)))


def _grasp_id(grasp: dict[str, Any]) -> str:
    blob = json.dumps(grasp, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _deterministic_unit_value(design_id: str, tool: str, direction: str, grasp: dict[str, float]) -> float:
    blob = json.dumps(
        {"design_id": design_id, "tool": tool, "direction": direction, "grasp": grasp},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
    return integer / float((1 << 64) - 1)


def _canonical_direction_name(name: str) -> str:
    aliases = {
        "force_pos_x": "+Fx",
        "force_neg_x": "-Fx",
        "force_pos_y": "+Fy",
        "force_neg_y": "-Fy",
        "force_pos_z": "+Fz",
        "force_neg_z": "-Fz",
        "torque_pos_x": "+Tx",
        "torque_neg_x": "-Tx",
        "torque_pos_y": "+Ty",
        "torque_neg_y": "-Ty",
        "torque_pos_z": "+Tz",
        "torque_neg_z": "-Tz",
    }
    return aliases.get(name, name)
