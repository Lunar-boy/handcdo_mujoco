from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


INSTALL_HINT = 'python3 -m pip install -e ".[warp]"'
NOT_IMPLEMENTED_MESSAGE = (
    "Experimental MuJoCo Warp grasp evaluation is not implemented in PR11-b; use PR11-c/PR11-d."
)


class MujocoWarpUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MujocoWarpBackendConfig:
    nworld: int = 64
    nconmax: int | None = 64
    naconmax: int | None = None
    njmax: int = 128
    warmup_steps: int = 0
    capture_graph: bool = False
    allow_sequential_fallback: bool = False


class MujocoWarpBackend:
    name = "mujoco_warp"

    def __init__(
        self,
        nworld: int = 64,
        nconmax: int | None = 64,
        naconmax: int | None = None,
        njmax: int = 128,
        warmup_steps: int = 0,
        capture_graph: bool = False,
        allow_sequential_fallback: bool = False,
    ) -> None:
        config = MujocoWarpBackendConfig(
            nworld=nworld,
            nconmax=nconmax,
            naconmax=naconmax,
            njmax=njmax,
            warmup_steps=warmup_steps,
            capture_graph=capture_graph,
            allow_sequential_fallback=allow_sequential_fallback,
        )
        _validate_config(config)
        self.config = config
        self._ensure_available()

    def evaluate_grasp(
        self,
        design: HandDesign,
        tool_name: str,
        grasp: GraspParams,
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None = None,
        tool_assets_dir: str | Path = "assets/tools",
    ) -> GraspEvaluation:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def evaluate_grasps_batch(
        self,
        design: HandDesign,
        tool_name: str,
        grasps: list[GraspParams],
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None = None,
        tool_assets_dir: str | Path = "assets/tools",
    ) -> list[GraspEvaluation]:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @staticmethod
    def _ensure_available() -> None:
        from handcdo.warp_utils import check_warp_available

        availability = check_warp_available()
        if availability.available:
            return
        reason = availability.reason or "mujoco_warp is unavailable"
        raise MujocoWarpUnavailableError(
            "MuJoCo Warp backend requires the optional warp extra. Install with:\n"
            f"{INSTALL_HINT}\n"
            f"Availability check failed: {reason}"
        )


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_optional_positive_int(value: int | None, name: str) -> None:
    if value is None:
        return
    _validate_positive_int(value, name)


def _validate_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_bool_like(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_config(config: MujocoWarpBackendConfig) -> None:
    _validate_positive_int(config.nworld, "nworld")
    _validate_optional_positive_int(config.nconmax, "nconmax")
    _validate_optional_positive_int(config.naconmax, "naconmax")
    _validate_positive_int(config.njmax, "njmax")
    _validate_nonnegative_int(config.warmup_steps, "warmup_steps")
    _validate_bool_like(config.capture_graph, "capture_graph")
    _validate_bool_like(config.allow_sequential_fallback, "allow_sequential_fallback")
