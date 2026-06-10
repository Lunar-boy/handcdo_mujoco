from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import tempfile
import time

import numpy as np

from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation
from handcdo.tools import ToolSpec


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


class MujocoWarpCapabilityError(NotImplementedError):
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


@dataclass
class WarpSceneBundle:
    mj_model: Any
    mj_data: Any
    warp_model: Any
    warp_data: Any
    tool: ToolSpec
    tool_body_id: int
    tool_qpos_addr: int
    actuator_names: list[str]
    nworld: int
    mjcf_rewrites: list[dict[str, Any]]


@dataclass
class BatchedInitialState:
    qpos_init: np.ndarray
    qvel_init: np.ndarray
    ctrl_init: np.ndarray
    xfrc_zero: np.ndarray


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

    def capabilities(self):
        from handcdo.warp_utils import inspect_warp_batch_capabilities

        try:
            mjw = _import_mujoco_warp()
        except MujocoWarpUnavailableError:
            return inspect_warp_batch_capabilities(None)
        return inspect_warp_batch_capabilities(mjw)

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

        eval_config = config or EvaluationConfig()
        bundle = _build_warp_scene_bundle(
            mjw=mjw,
            design=design,
            tool_name=tool_name,
            geometry_config=geometry_config or GeometryConfig(),
            tool_assets_dir=tool_assets_dir,
            nworld=self.config.nworld,
            nconmax=self.config.nconmax,
            naconmax=self.config.naconmax,
            njmax=self.config.njmax,
        )
        capabilities = inspect_warp_batch_capabilities(
            mjw,
            warp_model=bundle.warp_model,
            warp_data=bundle.warp_data,
            nworld=self.config.nworld,
        )
        if not capabilities.supports_true_fixed_grasp_batching:
            self.last_batch_metadata = self._metadata(
                num_grasps=num_grasps,
                num_chunks=num_chunks,
                failure_count=num_grasps,
                seconds_total=time.perf_counter() - start,
                capabilities=capabilities,
                mjcf_rewrites=bundle.mjcf_rewrites,
                failure_reason=TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE,
            )
            raise MujocoWarpCapabilityError(TRUE_BATCH_INIT_UNAVAILABLE_MESSAGE)

        evaluations: list[GraspEvaluation] = []
        world_steps = 0
        for offset in range(0, num_grasps, self.config.nworld):
            chunk = grasps[offset : offset + self.config.nworld]
            evaluations.extend(
                _evaluate_grasp_chunk_true_warp(
                    mjw=mjw,
                    bundle=bundle,
                    design=design,
                    tool_name=tool_name,
                    grasps=chunk,
                    config=eval_config,
                )
            )
            world_steps += len(chunk) * (
                eval_config.close_steps
                + eval_config.settle_steps
                + len(_wrench_directions()) * eval_config.wrench_steps
            )
        seconds_total = time.perf_counter() - start
        failure_count = sum(1 for evaluation in evaluations if evaluation.failed)
        self.last_batch_metadata = self._metadata(
            num_grasps=num_grasps,
            num_chunks=num_chunks,
            failure_count=failure_count,
            seconds_total=seconds_total,
            capabilities=capabilities,
            mjcf_rewrites=bundle.mjcf_rewrites,
            grasps_per_second=num_grasps / seconds_total if seconds_total > 0 else None,
            world_steps_per_second=world_steps / seconds_total if seconds_total > 0 else None,
        )
        return evaluations

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


def _build_warp_scene_bundle(
    *,
    mjw: Any,
    design: HandDesign,
    tool_name: str,
    geometry_config: GeometryConfig,
    tool_assets_dir: str | Path,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
) -> WarpSceneBundle:
    import mujoco

    from handcdo.hand_model import build_hand_model
    from handcdo.mjcf_generator import build_mjcf_xml
    from handcdo.tools import get_tool
    from handcdo.warp_utils import make_warp_data, prepare_warp_compatible_mjcf

    tool = get_tool(tool_name)
    xml = build_mjcf_xml(
        build_hand_model(design),
        tool=tool,
        geometry_config=geometry_config,
        tool_assets_dir=Path(tool_assets_dir),
    )
    with tempfile.TemporaryDirectory(prefix="handcdo_warp_mjcf_") as tmpdir:
        original_path = Path(tmpdir) / "scene_original.xml"
        warp_path = Path(tmpdir) / "scene_warp.xml"
        original_path.write_text(xml, encoding="utf-8")
        rewrite_info = prepare_warp_compatible_mjcf(original_path, warp_path)
        mj_model = mujoco.MjModel.from_xml_path(str(warp_path))

    mj_data = mujoco.MjData(mj_model)
    warp_model = mjw.put_model(mj_model)
    warp_data = make_warp_data(
        mjw,
        warp_model,
        mj_model,
        mj_data,
        nworld=nworld,
        nconmax=nconmax,
        naconmax=naconmax,
        njmax=njmax,
    )

    tool_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    if tool_body_id < 0:
        raise ValueError("MJCF is missing required body named 'tool'")
    tool_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "tool_free")
    if tool_joint_id < 0:
        raise ValueError("MJCF is missing required free joint named 'tool_free'")
    actuator_names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        for i in range(mj_model.nu)
    ]
    return WarpSceneBundle(
        mj_model=mj_model,
        mj_data=mj_data,
        warp_model=warp_model,
        warp_data=warp_data,
        tool=tool,
        tool_body_id=int(tool_body_id),
        tool_qpos_addr=int(mj_model.jnt_qposadr[tool_joint_id]),
        actuator_names=actuator_names,
        nworld=nworld,
        mjcf_rewrites=list(rewrite_info.get("mjcf_rewrites", [])),
    )


def build_batched_initial_state(
    bundle: WarpSceneBundle,
    grasps: list[GraspParams],
    config: EvaluationConfig,
) -> BatchedInitialState:
    del config
    batch = len(grasps)
    if batch > bundle.nworld:
        raise ValueError(f"batch size {batch} exceeds nworld={bundle.nworld}")
    if bundle.tool_qpos_addr < 0 or bundle.tool_qpos_addr + 7 > int(bundle.mj_model.nq):
        raise ValueError("Invalid tool free-joint qpos address for batched initialization")
    if len(bundle.actuator_names) != int(bundle.mj_model.nu):
        raise ValueError("Actuator metadata does not match MuJoCo model.nu")

    qpos_init = np.zeros((batch, int(bundle.mj_model.nq)), dtype=float)
    qvel_init = np.zeros((batch, int(bundle.mj_model.nv)), dtype=float)
    ctrl_init = np.zeros((batch, int(bundle.mj_model.nu)), dtype=float)
    xfrc_zero = np.zeros((batch, int(bundle.mj_model.nbody), 6), dtype=float)
    base_qpos = np.array(bundle.mj_data.qpos, dtype=float, copy=True)

    for world_index, grasp in enumerate(grasps):
        qpos_init[world_index] = base_qpos
        addr = bundle.tool_qpos_addr
        qpos_init[world_index, addr : addr + 3] = np.array(bundle.tool.reference_pos) + np.array(
            [grasp.dx, grasp.dy, grasp.dz]
        )
        qpos_init[world_index, addr + 3 : addr + 7] = _grasp_quat_xyz(grasp)
        for actuator_index, name in enumerate(bundle.actuator_names):
            base = grasp.thumb_closure if name.startswith("thumb") else grasp.closure
            ctrl_init[world_index, actuator_index] = np.clip(
                base + grasp.spread_bias * (actuator_index % 2),
                -0.25,
                1.3,
            )

    return BatchedInitialState(
        qpos_init=qpos_init,
        qvel_init=qvel_init,
        ctrl_init=ctrl_init,
        xfrc_zero=xfrc_zero,
    )


def _evaluate_grasp_chunk_true_warp(
    *,
    mjw: Any,
    bundle: WarpSceneBundle,
    design: HandDesign,
    tool_name: str,
    grasps: list[GraspParams],
    config: EvaluationConfig,
) -> list[GraspEvaluation]:
    from handcdo.warp_utils import synchronize_warp
    from handcdo.wrench_score import aggregate_wrench_results

    if not grasps:
        return []

    initial = build_batched_initial_state(bundle, grasps, config)
    batch = len(grasps)
    _write_batch_state(mjw, bundle.warp_data, initial.qpos_init, initial.qvel_init, initial.ctrl_init, initial.xfrc_zero)
    for _ in range(config.close_steps + config.settle_steps):
        _warp_step(mjw, bundle.warp_model, bundle.warp_data)
    synchronize_warp()
    settled_qpos = _read_required_field(bundle.warp_data, "qpos", batch)
    settled_qvel = _read_required_field(bundle.warp_data, "qvel", batch)
    settled_ctrl = _read_required_field(bundle.warp_data, "ctrl", batch)

    per_grasp_results = [[] for _ in grasps]
    for direction_name, force_dir, torque_dir in _wrench_directions():
        _write_batch_state(mjw, bundle.warp_data, settled_qpos, settled_qvel, settled_ctrl, initial.xfrc_zero)
        _warp_forward(mjw, bundle.warp_model, bundle.warp_data)
        synchronize_warp()
        start_pos = _read_tool_positions(bundle.warp_data, bundle.tool_body_id, batch)
        start_mat = _read_tool_mats(bundle.warp_data, bundle.tool_body_id, batch)
        stable_steps = np.full(batch, int(config.wrench_steps), dtype=int)
        max_trans = np.zeros(batch, dtype=float)
        max_rot = np.zeros(batch, dtype=float)
        failed = np.zeros(batch, dtype=bool)
        xfrc = np.array(initial.xfrc_zero, copy=True)
        for step in range(config.wrench_steps):
            scale = (step + 1) / max(config.wrench_steps, 1)
            xfrc[:, bundle.tool_body_id, :3] = force_dir * bundle.tool.force_limit * scale
            xfrc[:, bundle.tool_body_id, 3:] = torque_dir * bundle.tool.torque_limit * scale
            _write_required_field(mjw, bundle.warp_data, "xfrc_applied", xfrc)
            _warp_step(mjw, bundle.warp_model, bundle.warp_data)
            synchronize_warp()
            positions = _read_tool_positions(bundle.warp_data, bundle.tool_body_id, batch)
            mats = _read_tool_mats(bundle.warp_data, bundle.tool_body_id, batch)
            trans = np.linalg.norm(positions - start_pos, axis=1)
            rot = np.array([_rotation_error(start_mat[i], mats[i]) for i in range(batch)])
            max_trans = np.maximum(max_trans, trans)
            max_rot = np.maximum(max_rot, rot)
            newly_failed = (~failed) & (
                (trans > config.translation_threshold) | (rot > config.rotation_threshold_rad)
            )
            stable_steps[newly_failed] = step
            failed |= newly_failed
            if bool(np.all(failed)):
                break
        for world_index in range(batch):
            per_grasp_results[world_index].append(
                _wrench_result(
                    direction_name,
                    stable_steps=int(stable_steps[world_index]),
                    total_steps=int(config.wrench_steps),
                    max_translation=float(max_trans[world_index]),
                    max_rotation=float(max_rot[world_index]),
                    failed=bool(failed[world_index]),
                )
            )

    return [
        GraspEvaluation(
            design_id=design.design_id,
            tool=tool_name,
            grasp=grasp.to_dict(),
            score=aggregate_wrench_results(results),
            wrench_results=[result.to_dict() for result in results],
            failed=False,
        )
        for grasp, results in zip(grasps, per_grasp_results, strict=True)
    ]


def _write_batch_state(
    mjw: Any,
    warp_data: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    xfrc_applied: np.ndarray,
) -> None:
    _write_required_field(mjw, warp_data, "qpos", qpos)
    _write_required_field(mjw, warp_data, "qvel", qvel)
    _write_required_field(mjw, warp_data, "ctrl", ctrl)
    _write_required_field(mjw, warp_data, "xfrc_applied", xfrc_applied)


def _write_required_field(mjw: Any, warp_data: Any, field_name: str, value: np.ndarray) -> None:
    from handcdo import warp_utils

    field = getattr(warp_data, field_name, None)
    if field is None:
        raise MujocoWarpCapabilityError(f"MuJoCo Warp data is missing required field {field_name!r}")
    try:
        field[: value.shape[0], ...] = value
        return
    except Exception:
        pass
    errors: list[str] = []
    for world_index in range(int(value.shape[0])):
        wrote, method, reason = warp_utils._try_write_field_per_world(
            field,
            field_name=field_name,
            world_index=world_index,
            value=value[world_index],
            mjw=mjw,
        )
        if not wrote:
            errors.append(f"world {world_index}: {method}: {reason}")
    if errors:
        raise MujocoWarpCapabilityError(
            f"Could not write per-world MuJoCo Warp field {field_name!r}: {'; '.join(errors)}"
        )


def _read_required_field(warp_data: Any, field_name: str, batch: int) -> np.ndarray:
    from handcdo import warp_utils

    field = getattr(warp_data, field_name, None)
    if field is None:
        raise MujocoWarpCapabilityError(f"MuJoCo Warp data is missing required field {field_name!r}")
    array = warp_utils._field_host_array(field)
    if array is None:
        raise MujocoWarpCapabilityError(f"Could not copy MuJoCo Warp field {field_name!r} to host")
    if array.shape[0] < batch:
        raise MujocoWarpCapabilityError(
            f"MuJoCo Warp field {field_name!r} has leading dimension {array.shape[0]}, expected at least {batch}"
        )
    return np.array(array[:batch], dtype=float, copy=True)


def _read_tool_positions(warp_data: Any, tool_body_id: int, batch: int) -> np.ndarray:
    xpos = _read_required_field(warp_data, "xpos", batch)
    return np.array(xpos[:, tool_body_id, :3], dtype=float, copy=True)


def _read_tool_mats(warp_data: Any, tool_body_id: int, batch: int) -> np.ndarray:
    xmat = _read_required_field(warp_data, "xmat", batch)
    mats = xmat[:, tool_body_id]
    return np.array(mats.reshape((batch, 9)), dtype=float, copy=True)


def _warp_step(mjw: Any, warp_model: Any, warp_data: Any) -> None:
    step = getattr(mjw, "step", None)
    if not callable(step):
        raise MujocoWarpCapabilityError("mujoco_warp.step is unavailable")
    for args in ((warp_model, warp_data), (warp_data,), (warp_model, warp_data, None)):
        try:
            step(*args)
            return
        except TypeError:
            continue
    step(warp_model, warp_data)


def _warp_forward(mjw: Any, warp_model: Any, warp_data: Any) -> None:
    for name in ("forward", "mj_forward"):
        forward = getattr(mjw, name, None)
        if not callable(forward):
            continue
        for args in ((warp_model, warp_data), (warp_data,)):
            try:
                forward(*args)
                return
            except TypeError:
                continue
    raise MujocoWarpCapabilityError(
        "mujoco_warp does not expose a forward/kinematics-equivalent update needed "
        "before wrench start-pose measurement"
    )


def _grasp_quat_xyz(grasp: GraspParams) -> np.ndarray:
    import mujoco

    quat = np.zeros(4, dtype=float)
    mujoco.mju_euler2Quat(quat, np.array([grasp.roll, grasp.pitch, grasp.yaw], dtype=float), "XYZ")
    return quat


def _rotation_error(r0: np.ndarray, r1: np.ndarray) -> float:
    from handcdo.wrench_score import rotation_error_from_mats

    return rotation_error_from_mats(np.asarray(r0).reshape(9), np.asarray(r1).reshape(9))


def _wrench_result(
    direction: str,
    *,
    stable_steps: int,
    total_steps: int,
    max_translation: float,
    max_rotation: float,
    failed: bool,
):
    from handcdo.wrench_score import WrenchDirectionResult

    return WrenchDirectionResult(
        direction=direction,
        stable_steps=stable_steps,
        total_steps=total_steps,
        normalized_duration=float(stable_steps / max(total_steps, 1)),
        max_translation=max_translation,
        max_rotation_rad=max_rotation,
        failed=failed,
    )


def _wrench_directions():
    from handcdo.wrench_score import WRENCH_DIRECTIONS

    return WRENCH_DIRECTIONS


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
