from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from handcdo.mujoco_eval import GraspEvaluation
from handcdo.validation import mujoco_warp_gpu
from handcdo.validation.mujoco_warp_gpu import RuntimePrerequisites, ValidationConfig


def _available_prerequisites() -> RuntimePrerequisites:
    return RuntimePrerequisites(
        ok=True,
        reason=None,
        missing_packages=(),
        import_errors={},
        cuda_available=True,
        cuda_reason=None,
        cuda_devices=("cuda:0",),
    )


def _success_metadata(n_grasps: int = 2) -> dict:
    return {
        "backend": "mujoco_warp",
        "experimental": True,
        "score_semantics": "experimental_non_equivalent",
        "sequential_fallback": False,
        "include_in_multifidelity": False,
        "failure_count": 0,
        "failure_reason": None,
        "true_batched_scoring": True,
        "per_world_state_init": True,
        "num_grasps": n_grasps,
        "nworld": 2,
        "num_chunks": 1,
        "warmup_requested_steps": 1,
        "warmup_executed_steps": 1,
        "readback_interval": 1,
        "capture_graph_requested": False,
        "capture_graph_enabled": False,
    }


def _capability_failure_metadata(n_grasps: int = 2) -> dict:
    return {
        "backend": "mujoco_warp",
        "experimental": True,
        "score_semantics": "experimental_non_equivalent",
        "sequential_fallback": False,
        "include_in_multifidelity": False,
        "failure_count": n_grasps,
        "failure_reason": "MujocoWarpCapabilityError: true fixed-grasp batching unavailable",
        "true_batched_scoring": False,
        "per_world_state_init": False,
        "num_grasps": n_grasps,
        "nworld": 2,
        "num_chunks": 1,
        "warp_capabilities": {"supports_true_fixed_grasp_batching": False},
    }


def _evaluations(n_grasps: int = 2) -> list[GraspEvaluation]:
    wrench_results = [
        {
            "direction": f"direction_{index}",
            "stable_steps": 1,
            "total_steps": 1,
            "normalized_duration": 1.0,
            "max_translation": 0.0,
            "max_rotation_rad": 0.0,
            "failed": False,
        }
        for index in range(12)
    ]
    return [
        GraspEvaluation(
            design_id="design",
            tool="hammer",
            grasp={},
            score=1.0,
            wrench_results=wrench_results,
            failed=False,
            error=None,
        )
        for _ in range(n_grasps)
    ]


def _patch_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mujoco_warp_gpu, "check_runtime_prerequisites", _available_prerequisites)
    monkeypatch.setattr(
        mujoco_warp_gpu,
        "collect_environment_info",
        lambda: {"prerequisites": {"ok": True}, "cuda": {"available": True}},
    )


def test_submit_wrapper_precreates_directories_and_invokes_slurm() -> None:
    path = Path("scripts/submit_mujoco_warp_gpu_validation.sh")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "mkdir -p logs outputs/warp_gpu_validation" in text
    assert "sbatch" in text
    assert "slurm/validate_mujoco_warp_gpu.sbatch" in text
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_default_validation_capability_failure_returns_xfailed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from handcdo.backends.mujoco_warp import MujocoWarpCapabilityError

    _patch_prerequisites(monkeypatch)
    metadata = _capability_failure_metadata()

    def fail_with_capability_gate(config: ValidationConfig):
        exc = MujocoWarpCapabilityError("true fixed-grasp batching unavailable")
        exc.backend_metadata = metadata
        raise exc

    monkeypatch.setattr(mujoco_warp_gpu, "evaluate_real_backend_smoke", fail_with_capability_gate)

    code, report, report_path = mujoco_warp_gpu.run_validation(ValidationConfig(results_dir=tmp_path))

    assert code == 0
    assert report["status"] == "xfailed"
    assert report["strict"] is False
    assert report["backend_metadata"] == metadata
    assert report["xfail_reason"]
    assert report["validation"]["xfail_checks_passed"] is True
    assert report_path.exists()


def test_strict_validation_capability_failure_returns_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from handcdo.backends.mujoco_warp import MujocoWarpCapabilityError

    _patch_prerequisites(monkeypatch)
    metadata = _capability_failure_metadata()

    def fail_with_capability_gate(config: ValidationConfig):
        exc = MujocoWarpCapabilityError("true fixed-grasp batching unavailable")
        exc.backend_metadata = metadata
        raise exc

    monkeypatch.setattr(mujoco_warp_gpu, "evaluate_real_backend_smoke", fail_with_capability_gate)

    code, report, _ = mujoco_warp_gpu.run_validation(ValidationConfig(results_dir=tmp_path, strict=True))

    assert code == 1
    assert report["status"] == "failed"
    assert report["strict"] is True
    assert report["xfail_reason"] is None


@pytest.mark.parametrize("strict", [False, True])
def test_unexpected_exception_never_xfails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    strict: bool,
) -> None:
    _patch_prerequisites(monkeypatch)

    def fail_unexpectedly(config: ValidationConfig):
        exc = RuntimeError("scene build exploded")
        exc.backend_metadata = _capability_failure_metadata()
        raise exc

    monkeypatch.setattr(mujoco_warp_gpu, "evaluate_real_backend_smoke", fail_unexpectedly)

    code, report, _ = mujoco_warp_gpu.run_validation(ValidationConfig(results_dir=tmp_path, strict=strict))

    assert code == 1
    assert report["status"] == "failed"
    assert report["xfail_reason"] is None


def test_validation_success_path_requires_success_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_prerequisites(monkeypatch)

    def succeed(config: ValidationConfig):
        return _evaluations(config.n_grasps), _success_metadata(config.n_grasps)

    monkeypatch.setattr(mujoco_warp_gpu, "evaluate_real_backend_smoke", succeed)

    code, report, _ = mujoco_warp_gpu.run_validation(ValidationConfig(results_dir=tmp_path))

    assert code == 0
    assert report["status"] == "passed"
    assert report["validation"]["success_checks_passed"] is True
    assert report["results"][0]["wrench_result_count"] == 12
