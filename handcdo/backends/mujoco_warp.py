from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


INSTALL_HINT = 'python3 -m pip install -e ".[warp]"'
NOT_IMPLEMENTED_MESSAGE = (
    "Single-grasp MuJoCo Warp evaluation is not implemented; CPU MuJoCo remains the reference backend."
)
TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE = (
    "True per-world fixed-grasp initialization is not available for this MuJoCo Warp API; "
    "refusing to report fake batched scores."
)
EXPERIMENTAL_SCORE_SEMANTICS = "experimental_non_equivalent"
SEQUENTIAL_FALLBACK_SCORE_SEMANTICS = "experimental_sequential_fallback"


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
        self.last_batch_metadata = self._metadata(num_grasps=0, num_chunks=0)
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
        start = time.perf_counter()
        num_grasps = len(grasps)
        num_chunks = math.ceil(num_grasps / self.config.nworld) if num_grasps else 0
        self.last_batch_metadata = self._metadata(
            num_grasps=num_grasps,
            num_chunks=num_chunks,
            seconds_total=0.0,
        )
        if not grasps:
            return []

        if self.config.allow_sequential_fallback:
            evaluations = self._evaluate_grasps_sequential_fallback(
                design=design,
                tool_name=tool_name,
                grasps=grasps,
                config=config,
                geometry_config=geometry_config,
                tool_assets_dir=tool_assets_dir,
            )
            seconds_total = time.perf_counter() - start
            self.last_batch_metadata = self._metadata(
                num_grasps=num_grasps,
                num_chunks=num_grasps,
                failure_count=sum(1 for evaluation in evaluations if evaluation.failed),
                seconds_total=seconds_total,
                grasps_per_second=num_grasps / seconds_total if seconds_total > 0 else None,
                score_semantics=SEQUENTIAL_FALLBACK_SCORE_SEMANTICS,
                sequential_fallback=True,
            )
            return evaluations

        try:
            mjw = _import_mujoco_warp()
        except MujocoWarpUnavailableError:
            self.last_batch_metadata = self._metadata(
                num_grasps=num_grasps,
                num_chunks=num_chunks,
                failure_count=num_grasps,
                seconds_total=time.perf_counter() - start,
                failure_reason="mujoco_warp import failed",
            )
            raise

        from handcdo.warp_utils import inspect_warp_batch_capabilities

        capabilities = inspect_warp_batch_capabilities(mjw)
        if not capabilities.supports_true_fixed_grasp_batching:
            self.last_batch_metadata = self._metadata(
                num_grasps=num_grasps,
                num_chunks=num_chunks,
                failure_count=num_grasps,
                seconds_total=time.perf_counter() - start,
                capabilities=capabilities,
                failure_reason=TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE,
            )
            raise NotImplementedError(TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE)

        # PR11-d deliberately does not implement the per-world mutation path
        # until a concrete installed MuJoCo Warp API exposes safe qpos/ctrl
        # writes for every world in one batched data object. The required future
        # sequence is: create/reset one batched data object per chunk, map each
        # grasp to one world, set tool free-joint pose and hand controls per
        # world, close and settle all worlds together, snapshot settled per-world
        # state, restore each world before every wrench disturbance, then reduce
        # scores back to input order.
        self.last_batch_metadata = self._metadata(
            num_grasps=num_grasps,
            num_chunks=num_chunks,
            failure_count=num_grasps,
            seconds_total=time.perf_counter() - start,
            capabilities=capabilities,
            failure_reason=TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE,
        )
        raise NotImplementedError(TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE)

    def _evaluate_grasps_sequential_fallback(
        self,
        design: HandDesign,
        tool_name: str,
        grasps: list[GraspParams],
        config: EvaluationConfig | None,
        geometry_config: GeometryConfig | None,
        tool_assets_dir: str | Path,
    ) -> list[GraspEvaluation]:
        from handcdo.mujoco_eval import evaluate_grasp

        return [
            evaluate_grasp(
                design,
                tool_name,
                grasp,
                config,
                geometry_config=geometry_config,
                tool_assets_dir=tool_assets_dir,
            )
            for grasp in grasps
        ]

    def _metadata(
        self,
        *,
        num_grasps: int,
        num_chunks: int,
        failure_count: int = 0,
        seconds_total: float = 0.0,
        score_semantics: str = EXPERIMENTAL_SCORE_SEMANTICS,
        sequential_fallback: bool = False,
        mjcf_rewrites: list[dict] | None = None,
        grasps_per_second: float | None = None,
        world_steps_per_second: float | None = None,
        capabilities: object | None = None,
        failure_reason: str | None = None,
    ) -> dict:
        from handcdo.warp_utils import warp_batch_metadata

        return warp_batch_metadata(
            nworld=self.config.nworld,
            nconmax=self.config.nconmax,
            naconmax=self.config.naconmax,
            njmax=self.config.njmax,
            num_grasps=num_grasps,
            num_chunks=num_chunks,
            failure_count=failure_count,
            seconds_total=seconds_total,
            score_semantics=score_semantics,
            sequential_fallback=sequential_fallback,
            mjcf_rewrites=mjcf_rewrites,
            grasps_per_second=grasps_per_second,
            world_steps_per_second=world_steps_per_second,
            capabilities=capabilities,
            failure_reason=failure_reason,
        )

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


def _import_mujoco_warp():
    try:
        import mujoco_warp as mjw
    except Exception as exc:
        raise MujocoWarpUnavailableError(
            "MuJoCo Warp backend requires the optional warp extra. Install with:\n"
            f"{INSTALL_HINT}\n"
            f"Import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return mjw
