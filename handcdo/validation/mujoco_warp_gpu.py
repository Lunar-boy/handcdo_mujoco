from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any

from handcdo.batch_eval import sample_fixed_random_grasps
from handcdo.design_space import DesignSpace
from handcdo.geometry_config import GeometryConfig
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation
from handcdo.utils import ensure_dir, write_json


STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_XFAILED = "xfailed"

XFAIL_REASON_TRUE_BATCHING = (
    "MuJoCo Warp runtime does not support true fixed-grasp batching; "
    "metadata truthfully reports the capability gate"
)


@dataclass(frozen=True)
class RuntimePrerequisites:
    ok: bool
    reason: str | None
    missing_packages: tuple[str, ...]
    import_errors: dict[str, str]
    cuda_available: bool | None
    cuda_reason: str | None
    cuda_devices: tuple[str, ...]


@dataclass(frozen=True)
class ValidationConfig:
    results_dir: Path = Path("outputs/warp_gpu_validation")
    tool: str = "hammer"
    n_grasps: int = 2
    nworld: int = 2
    close_steps: int = 1
    settle_steps: int = 1
    wrench_steps: int = 2
    warmup_steps: int = 1
    readback_interval: int = 1
    nconmax: int | None = 64
    njmax: int = 128
    seed: int = 13
    strict: bool = False
    allow_skip: bool = False
    capture_graph: bool = False


def check_runtime_prerequisites() -> RuntimePrerequisites:
    required = ("mujoco", "mujoco_warp", "warp")
    missing = []
    import_errors: dict[str, str] = {}
    for module_name in required:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"

    cuda_available: bool | None = None
    cuda_reason: str | None = None
    cuda_devices: list[str] = []
    if "warp" not in missing and "warp" not in import_errors:
        cuda_available, cuda_reason, cuda_devices = _probe_warp_cuda()

    reason_parts = []
    if missing:
        reason_parts.append(f"missing required package(s): {', '.join(missing)}")
    if import_errors:
        formatted = ", ".join(f"{name}: {error}" for name, error in sorted(import_errors.items()))
        reason_parts.append(f"required package import failed: {formatted}")
    if cuda_available is False:
        reason_parts.append(cuda_reason or "CUDA device unavailable")

    return RuntimePrerequisites(
        ok=not reason_parts,
        reason="; ".join(reason_parts) if reason_parts else None,
        missing_packages=tuple(missing),
        import_errors=import_errors,
        cuda_available=cuda_available,
        cuda_reason=cuda_reason,
        cuda_devices=tuple(cuda_devices),
    )


def collect_environment_info() -> dict[str, Any]:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "RUN_GPU_TESTS",
        "RUN_STRICT_WARP_INTEGRATION",
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_SUBMIT_DIR",
    ]
    prerequisites = check_runtime_prerequisites()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "executable": sys.executable,
        "environment": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "packages": {
            "mujoco": _module_version("mujoco", "mujoco"),
            "mujoco_warp": _module_version("mujoco_warp", "mujoco-warp"),
            "warp": _module_version("warp", "warp-lang"),
        },
        "cuda": {
            "available": prerequisites.cuda_available,
            "reason": prerequisites.cuda_reason,
            "devices": list(prerequisites.cuda_devices),
        },
        "prerequisites": _prerequisites_payload(prerequisites),
    }


def run_validation(config: ValidationConfig) -> tuple[int, dict[str, Any], Path]:
    started = _safe_timestamp()
    report_path = config.results_dir / f"mujoco_warp_gpu_validation_{started}.json"
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arguments": _config_payload(config),
        "environment": collect_environment_info(),
        "backend_metadata": None,
        "results": [],
        "status": STATUS_FAILED,
        "strict": config.strict,
        "allow_skip": config.allow_skip,
        "exception": None,
        "skip_reason": None,
        "xfail_reason": None,
        "validation": {
            "prerequisites_checked": False,
            "backend_called": False,
            "success_checks_passed": False,
            "xfail_checks_passed": False,
        },
    }

    prerequisites = check_runtime_prerequisites()
    report["validation"]["prerequisites_checked"] = True
    report["environment"]["prerequisites"] = _prerequisites_payload(prerequisites)
    if not prerequisites.ok:
        report["status"] = STATUS_SKIPPED
        report["skip_reason"] = prerequisites.reason
        _write_report(report_path, report)
        return (0 if config.allow_skip else 2), report, report_path

    try:
        evaluations, metadata = evaluate_real_backend_smoke(config)
        report["validation"]["backend_called"] = True
        report["backend_metadata"] = metadata
        report["results"] = [_evaluation_summary(evaluation) for evaluation in evaluations]
        _assert_success_metadata(metadata, evaluations, config)
        report["validation"]["success_checks_passed"] = True
        report["status"] = STATUS_PASSED
        _write_report(report_path, report)
        return 0, report, report_path
    except Exception as exc:
        metadata = getattr(exc, "backend_metadata", None)
        if metadata is not None:
            report["backend_metadata"] = metadata
            report["validation"]["backend_called"] = True
        report["exception"] = _exception_payload(exc)
        if (
            not config.strict
            and report["validation"]["backend_called"]
            and _is_true_batching_capability_failure(exc, metadata)
        ):
            try:
                assert_truthful_capability_failure(metadata, config.n_grasps)
            except Exception as validation_exc:
                report["exception"] = _exception_payload(validation_exc)
                report["validation"]["xfail_checks_passed"] = False
                report["status"] = STATUS_FAILED
                _write_report(report_path, report)
                return 1, report, report_path
            report["validation"]["xfail_checks_passed"] = True
            report["status"] = STATUS_XFAILED
            report["xfail_reason"] = XFAIL_REASON_TRUE_BATCHING
            _write_report(report_path, report)
            return 0, report, report_path
        report["status"] = STATUS_FAILED
        _write_report(report_path, report)
        return 1, report, report_path


def evaluate_real_backend_smoke(config: ValidationConfig) -> tuple[list[GraspEvaluation], dict[str, Any]]:
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    design = DesignSpace().sample(seed=config.seed)
    grasps = sample_fixed_random_grasps(
        n_grasp_trials=config.n_grasps,
        seed=config.seed,
        design_id=design.design_id,
        tool_name=config.tool,
    )
    eval_config = EvaluationConfig(
        close_steps=config.close_steps,
        settle_steps=config.settle_steps,
        wrench_steps=config.wrench_steps,
    )
    geometry_config = GeometryConfig()
    backend = MujocoWarpBackend(
        nworld=config.nworld,
        nconmax=config.nconmax,
        njmax=config.njmax,
        warmup_steps=config.warmup_steps,
        capture_graph=config.capture_graph,
        readback_interval=config.readback_interval,
    )
    try:
        evaluations = backend.evaluate_grasps_batch(
            design,
            config.tool,
            grasps,
            eval_config,
            geometry_config=geometry_config,
        )
    except Exception as exc:
        try:
            setattr(exc, "backend_metadata", backend.last_batch_metadata)
        except Exception:
            pass
        raise
    return evaluations, backend.last_batch_metadata


def is_capability_exception(exc: BaseException) -> bool:
    return _is_mujoco_warp_capability_error(exc)


def assert_truthful_capability_failure(metadata: dict[str, Any] | None, num_grasps: int) -> None:
    _require(metadata is not None, "backend metadata is missing")
    _require(metadata.get("backend") == "mujoco_warp", "metadata backend must be mujoco_warp")
    _require(metadata.get("experimental") is True, "metadata experimental must be true")
    _require(
        metadata.get("score_semantics") == "experimental_non_equivalent",
        "metadata score_semantics must remain experimental_non_equivalent",
    )
    _require(metadata.get("sequential_fallback") is False, "sequential fallback must be false")
    _require(metadata.get("include_in_multifidelity") is False, "include_in_multifidelity must be false")
    _require(metadata.get("true_batched_scoring") is False, "true_batched_scoring must be false")
    if "per_world_state_init" in metadata:
        _require(metadata.get("per_world_state_init") is False, "per_world_state_init must be false")
    _require(metadata.get("failure_count") == num_grasps, "failure_count must match requested grasps")
    _require(bool(metadata.get("failure_reason")), "failure_reason must be non-empty")
    capabilities = metadata.get("warp_capabilities") or {}
    _require(
        capabilities.get("supports_true_fixed_grasp_batching") is False,
        "warp_capabilities must report supports_true_fixed_grasp_batching=false",
    )


def assert_successful_integration(
    evaluations: list[GraspEvaluation],
    metadata: dict[str, Any] | None,
    *,
    tool_name: str,
    num_grasps: int,
    nworld: int,
    warmup_steps: int,
    readback_interval: int,
) -> None:
    _require(len(evaluations) == num_grasps, "evaluation count must match requested grasps")
    _require(metadata is not None, "backend metadata is missing")
    _require(metadata.get("backend") == "mujoco_warp", "metadata backend must be mujoco_warp")
    _require(metadata.get("experimental") is True, "metadata experimental must be true")
    _require(
        metadata.get("score_semantics") == "experimental_non_equivalent",
        "metadata score_semantics must remain experimental_non_equivalent",
    )
    _require(metadata.get("sequential_fallback") is False, "sequential fallback must be false")
    _require(metadata.get("include_in_multifidelity") is False, "include_in_multifidelity must be false")
    _require(metadata.get("failure_count") == 0, "failure_count must be zero")
    _require(metadata.get("failure_reason") is None, "failure_reason must be null on success")
    _require(metadata.get("true_batched_scoring") is True, "true_batched_scoring must be true")
    _require(metadata.get("per_world_state_init") is True, "per_world_state_init must be true")
    _require(metadata.get("num_grasps") == num_grasps, "num_grasps must match requested grasps")
    _require(metadata.get("nworld") == nworld, "nworld must match validation config")
    _require(metadata.get("num_chunks", 0) >= 1, "num_chunks must be at least one")
    _require(metadata.get("warmup_requested_steps") == warmup_steps, "warmup_requested_steps mismatch")
    _require(
        metadata.get("warmup_executed_steps") in (0, warmup_steps),
        "warmup_executed_steps must be zero or the requested warmup steps",
    )
    _require(metadata.get("readback_interval") == readback_interval, "readback_interval mismatch")
    _require(metadata.get("capture_graph_requested") is False, "capture_graph_requested must be false")
    _require(metadata.get("capture_graph_enabled") is False, "capture_graph_enabled must be false")
    for evaluation in evaluations:
        _require(evaluation.tool == tool_name, "evaluation tool mismatch")
        _require(evaluation.failed is False, "evaluation must not be failed")
        _require(evaluation.error is None, "evaluation error must be null")
        _require(isinstance(evaluation.score, float), "evaluation score must be a float")
        _require(len(evaluation.wrench_results) == 12, "evaluation must include 12 wrench results")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate real MuJoCo Warp GPU fixed-grasp batch integration.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/warp_gpu_validation"))
    parser.add_argument("--tool", default="hammer")
    parser.add_argument("--n-grasps", type=int, default=2)
    parser.add_argument("--nworld", type=int, default=2)
    parser.add_argument("--close-steps", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--wrench-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--readback-interval", type=int, default=1)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-skip", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ValidationConfig(
        results_dir=args.results_dir,
        tool=args.tool,
        n_grasps=args.n_grasps,
        nworld=args.nworld,
        close_steps=args.close_steps,
        settle_steps=args.settle_steps,
        wrench_steps=args.wrench_steps,
        warmup_steps=args.warmup_steps,
        readback_interval=args.readback_interval,
        nconmax=args.nconmax,
        njmax=args.njmax,
        seed=args.seed,
        strict=args.strict,
        allow_skip=args.allow_skip,
    )
    _validate_config(config)
    code, report, report_path = run_validation(config)
    print(f"{report['status']}: wrote {report_path}")
    if report.get("skip_reason"):
        print(f"skip_reason: {report['skip_reason']}")
    if report.get("exception"):
        exception = report["exception"]
        print(f"exception: {exception['type']}: {exception['message']}")
    return code


def _assert_success_metadata(
    metadata: dict[str, Any] | None,
    evaluations: list[GraspEvaluation],
    config: ValidationConfig,
) -> None:
    assert_successful_integration(
        evaluations,
        metadata,
        tool_name=config.tool,
        num_grasps=config.n_grasps,
        nworld=config.nworld,
        warmup_steps=config.warmup_steps,
        readback_interval=config.readback_interval,
    )


def _probe_warp_cuda() -> tuple[bool, str | None, list[str]]:
    try:
        warp = importlib.import_module("warp")
    except Exception as exc:
        return False, f"warp import failed during CUDA probe: {type(exc).__name__}: {exc}", []

    devices: list[str] = []
    try:
        if hasattr(warp, "get_cuda_device_count"):
            count = int(warp.get_cuda_device_count())
            if count <= 0:
                return False, "warp reports zero CUDA devices", []
        if hasattr(warp, "get_device"):
            try:
                device = warp.get_device("cuda")
            except TypeError:
                device = warp.get_device()
            if device is None:
                return False, "warp CUDA device lookup returned None", []
            devices.append(str(device))
            if "cuda" not in str(device).lower():
                return False, f"warp device is not CUDA: {device}", devices
        return True, None, devices
    except Exception as exc:
        return False, f"CUDA device probe failed: {type(exc).__name__}: {exc}", devices


def _module_version(module_name: str, distribution_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None
    if module is not None:
        version = getattr(module, "__version__", None)
        if version is not None:
            return str(version)
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _config_payload(config: ValidationConfig) -> dict[str, Any]:
    return {
        "results_dir": str(config.results_dir),
        "tool": config.tool,
        "n_grasps": config.n_grasps,
        "nworld": config.nworld,
        "close_steps": config.close_steps,
        "settle_steps": config.settle_steps,
        "wrench_steps": config.wrench_steps,
        "warmup_steps": config.warmup_steps,
        "readback_interval": config.readback_interval,
        "nconmax": config.nconmax,
        "njmax": config.njmax,
        "seed": config.seed,
        "strict": config.strict,
        "allow_skip": config.allow_skip,
        "capture_graph": config.capture_graph,
    }


def _prerequisites_payload(prerequisites: RuntimePrerequisites) -> dict[str, Any]:
    return {
        "ok": prerequisites.ok,
        "reason": prerequisites.reason,
        "missing_packages": list(prerequisites.missing_packages),
        "import_errors": prerequisites.import_errors,
        "cuda_available": prerequisites.cuda_available,
        "cuda_reason": prerequisites.cuda_reason,
        "cuda_devices": list(prerequisites.cuda_devices),
    }


def _evaluation_summary(evaluation: GraspEvaluation) -> dict[str, Any]:
    return {
        "design_id": evaluation.design_id,
        "tool": evaluation.tool,
        "score": float(evaluation.score),
        "failed": bool(evaluation.failed),
        "error": evaluation.error,
        "wrench_result_count": len(evaluation.wrench_results),
    }


def _exception_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "capability_failure": is_capability_exception(exc),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, report)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_true_batching_capability_failure(exc: BaseException, metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    capabilities = metadata.get("warp_capabilities") or {}
    if capabilities.get("supports_true_fixed_grasp_batching") is not False:
        return False
    if metadata.get("sequential_fallback") is not False:
        return False
    if metadata.get("true_batched_scoring") is not False:
        return False
    if _is_mujoco_warp_capability_error(exc):
        return True
    if isinstance(exc, NotImplementedError):
        failure_reason = str(metadata.get("failure_reason") or "").lower()
        return (
            "true fixed-grasp batching" in failure_reason
            or "true per-world fixed-grasp" in failure_reason
            or "fake batched scores" in failure_reason
        )
    return False


def _is_mujoco_warp_capability_error(exc: BaseException) -> bool:
    try:
        from handcdo.backends.mujoco_warp import MujocoWarpCapabilityError
    except Exception:
        return type(exc).__name__ == "MujocoWarpCapabilityError"
    return isinstance(exc, MujocoWarpCapabilityError)


def _validate_config(config: ValidationConfig) -> None:
    for name in ("n_grasps", "nworld", "close_steps", "settle_steps", "wrench_steps", "warmup_steps", "readback_interval", "njmax", "seed"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if config.n_grasps <= 0:
        raise ValueError("n_grasps must be > 0")
    if config.nworld <= 0:
        raise ValueError("nworld must be > 0")
    if config.n_grasps > config.nworld:
        raise ValueError("n_grasps must be <= nworld for the smoke validation")
    if config.close_steps < 0 or config.settle_steps < 0 or config.warmup_steps < 0:
        raise ValueError("close_steps, settle_steps, and warmup_steps must be >= 0")
    if config.wrench_steps <= 0:
        raise ValueError("wrench_steps must be > 0")
    if config.readback_interval <= 0:
        raise ValueError("readback_interval must be > 0")
    if config.nconmax is not None and config.nconmax <= 0:
        raise ValueError("nconmax must be > 0 when set")
    if config.njmax <= 0:
        raise ValueError("njmax must be > 0")
