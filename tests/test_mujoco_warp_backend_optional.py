from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from handcdo.backends.batched import supports_batched_grasps
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import GraspEvaluation
from handcdo.warp_utils import WarpAvailability


def _grasp(**overrides):
    values = {
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "closure": 0.5,
        "thumb_closure": 0.5,
        "spread_bias": 0.0,
    }
    values.update(overrides)
    return GraspParams(**values)


def test_importing_backends_does_not_import_mujoco_warp_package():
    for module_name in (
        "handcdo.backends",
        "handcdo.backends.registry",
        "handcdo.backends.mujoco_warp",
        "mujoco_warp",
    ):
        sys.modules.pop(module_name, None)

    backends = importlib.import_module("handcdo.backends")

    assert backends.get_backend("mujoco").name == "mujoco_cpu"
    assert "mujoco_warp" not in sys.modules
    assert "handcdo.backends.mujoco_warp" not in sys.modules


def test_get_backend_cpu_aliases_do_not_require_mujoco_warp():
    from handcdo.backends import get_backend

    assert get_backend("mujoco").name == "mujoco_cpu"
    assert get_backend("mujoco_cpu").name == "mujoco_cpu"


def test_get_backend_mujoco_warp_returns_backend_when_dependency_available(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import get_backend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = get_backend("mujoco_warp")

    assert backend.name == "mujoco_warp"


def test_get_backend_mujoco_warp_fails_clearly_when_dependency_absent(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import get_backend
    from handcdo.backends.mujoco_warp import MujocoWarpUnavailableError

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(False, "ModuleNotFoundError: No module named 'mujoco_warp'", "mujoco_warp", None),
    )

    with pytest.raises(MujocoWarpUnavailableError) as exc_info:
        get_backend("mujoco_warp")

    message = str(exc_info.value)
    assert "MuJoCo Warp backend requires the optional warp extra" in message
    assert 'python3 -m pip install -e ".[warp]"' in message


def test_missing_mujoco_warp_does_not_break_cpu_backend_construction(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import get_backend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    assert get_backend("mujoco").name == "mujoco_cpu"
    assert get_backend("mujoco_cpu").name == "mujoco_cpu"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nworld": 0}, "nworld"),
        ({"nconmax": 0}, "nconmax"),
        ({"naconmax": 0}, "naconmax"),
        ({"njmax": 0}, "njmax"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"capture_graph": "yes"}, "capture_graph"),
        ({"allow_sequential_fallback": "no"}, "allow_sequential_fallback"),
    ],
)
def test_invalid_constructor_values_are_rejected_before_optional_import(monkeypatch, kwargs, match):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    def fail_if_called():
        raise AssertionError("optional Warp availability should not be checked for invalid constructor values")

    monkeypatch.setattr(warp_utils, "check_warp_available", fail_if_called)

    with pytest.raises((TypeError, ValueError), match=match):
        MujocoWarpBackend(**kwargs)


def test_supports_batched_grasps_true_for_mujoco_warp_skeleton_when_available(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = MujocoWarpBackend()

    assert callable(getattr(backend, "evaluate_grasps_batch", None))
    assert supports_batched_grasps(backend) is True


def test_mujoco_warp_single_evaluation_is_not_implemented(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = MujocoWarpBackend()

    with pytest.raises(NotImplementedError, match="Single-grasp MuJoCo Warp evaluation"):
        backend.evaluate_grasp(None, "hammer", None, None)


def test_empty_batch_returns_empty_with_conservative_metadata(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    backend = MujocoWarpBackend(nworld=8)

    assert backend.evaluate_grasps_batch(None, "hammer", [], None) == []
    assert backend.last_batch_metadata["backend"] == "mujoco_warp"
    assert backend.last_batch_metadata["experimental"] is True
    assert backend.last_batch_metadata["score_semantics"] == "experimental_non_equivalent"
    assert backend.last_batch_metadata["num_grasps"] == 0
    assert backend.last_batch_metadata["num_chunks"] == 0
    assert backend.last_batch_metadata["sequential_fallback"] is False


def test_batch_refuses_when_true_per_world_initialization_is_unavailable(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco_warp",
        SimpleNamespace(put_model=object(), make_data=object(), step=object()),
    )

    backend = MujocoWarpBackend(nworld=2)
    grasps = [_grasp(), _grasp(dx=0.01), _grasp(dy=0.01)]

    with pytest.raises(NotImplementedError, match="refusing to report fake batched scores"):
        backend.evaluate_grasps_batch(None, "hammer", grasps, None)

    metadata = backend.last_batch_metadata
    assert metadata["score_semantics"] == "experimental_non_equivalent"
    assert metadata["failure_count"] == 3
    assert metadata["num_chunks"] == 2
    assert metadata["sequential_fallback"] is False
    capabilities = metadata["warp_capabilities"]
    assert capabilities["can_put_model"] is True
    assert capabilities["can_put_data"] is False
    assert capabilities["can_make_data"] is True
    assert capabilities["can_step"] is True
    assert capabilities["accepted_data_allocation_kwargs"] == []
    assert capabilities["data_allocation_probe_error"]
    assert capabilities["can_set_per_world_qpos"] is False
    assert capabilities["can_set_per_world_qvel"] is False
    assert capabilities["can_set_per_world_ctrl"] is False
    assert capabilities["can_set_per_world_xfrc"] is False
    assert capabilities["supports_true_fixed_grasp_batching"] is False
    reason = capabilities["true_fixed_grasp_batching_reason"]
    assert reason
    assert "qpos" in reason
    assert "qvel" in reason
    assert "ctrl" in reason
    assert "xfrc" in reason


def test_batch_default_does_not_silently_call_cpu_evaluate_grasp(monkeypatch):
    from handcdo import mujoco_eval, warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco_warp",
        SimpleNamespace(put_model=object(), make_data=object(), step=object()),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("CPU evaluate_grasp must not be called by default")

    monkeypatch.setattr(mujoco_eval, "evaluate_grasp", fail_if_called)

    backend = MujocoWarpBackend()
    with pytest.raises(NotImplementedError):
        backend.evaluate_grasps_batch(None, "hammer", [_grasp()], None)


def test_sequential_fallback_is_explicit_and_labeled(monkeypatch):
    from handcdo import mujoco_eval, warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    def fake_evaluate_grasp(design, tool_name, grasp, config, geometry_config=None, tool_assets_dir="assets/tools"):
        return GraspEvaluation(
            design_id=design.design_id,
            tool=tool_name,
            grasp=grasp.to_dict(),
            score=0.5,
            wrench_results=[],
        )

    monkeypatch.setattr(mujoco_eval, "evaluate_grasp", fake_evaluate_grasp)

    design = SimpleNamespace(design_id="debug-design")
    grasps = [_grasp(), _grasp(dx=0.01)]
    backend = MujocoWarpBackend(allow_sequential_fallback=True)

    evaluations = backend.evaluate_grasps_batch(design, "hammer", grasps, None)

    assert [evaluation.score for evaluation in evaluations] == [0.5, 0.5]
    assert backend.last_batch_metadata["score_semantics"] == "experimental_sequential_fallback"
    assert backend.last_batch_metadata["sequential_fallback"] is True
    assert backend.last_batch_metadata["failure_count"] == 0
