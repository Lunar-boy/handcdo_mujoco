from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from handcdo.backends.batched import supports_batched_grasps
from handcdo.grasp_sampling import GraspParams
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation
from handcdo.tools import ToolSpec
from handcdo.warp_utils import WarpAvailability, WarpBatchCapabilities


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


def _fake_bundle(nworld=2):
    from handcdo.backends.mujoco_warp import WarpSceneBundle

    return WarpSceneBundle(
        mj_model=SimpleNamespace(nq=7, nv=6, nu=2, nbody=3),
        mj_data=SimpleNamespace(qpos=np.zeros(7)),
        warp_model=SimpleNamespace(label="warp_model"),
        warp_data=SimpleNamespace(
            qpos=np.zeros((nworld, 7)),
            qvel=np.zeros((nworld, 6)),
            ctrl=np.zeros((nworld, 2)),
            xfrc_applied=np.zeros((nworld, 3, 6)),
            xpos=np.zeros((nworld, 3, 3)),
            xmat=np.tile(np.eye(3).reshape(1, 1, 9), (nworld, 3, 1)),
        ),
        tool=ToolSpec("hammer", 0.55, (1.1, 0.02, 0.002), (0.11, 0.0, 0.055), (1.0, 0.0, 0.0, 0.0), 18.0, 0.55),
        tool_body_id=1,
        tool_qpos_addr=0,
        actuator_names=["finger1_pos", "thumb1_pos"],
        nworld=nworld,
        mjcf_rewrites=[],
    )


def _capabilities(supports: bool):
    return WarpBatchCapabilities(
        can_put_model=True,
        can_put_data=True,
        can_make_data=False,
        can_step=True,
        accepted_data_allocation_kwargs=[],
        data_allocation_probe_error=None,
        can_set_per_world_qpos=supports,
        can_set_per_world_qvel=supports,
        can_set_per_world_ctrl=supports,
        can_set_per_world_xfrc=supports,
        true_fixed_grasp_batching_reason="verified" if supports else "xfrc unavailable",
        has_qpos=supports,
        has_qvel=supports,
        has_ctrl=supports,
        has_xfrc_applied=supports,
        qpos_is_batched=supports,
        qvel_is_batched=supports,
        ctrl_is_batched=supports,
        xfrc_is_batched=supports,
        qpos_write_tested=supports,
        qvel_write_tested=supports,
        ctrl_write_tested=supports,
        xfrc_write_tested=supports,
    )


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
    assert backend.last_batch_metadata["true_batched_scoring"] is False
    assert backend.last_batch_metadata["per_world_state_init"] is False


def test_batched_initial_state_uses_cpu_pose_and_actuator_semantics(monkeypatch):
    from handcdo.backends import mujoco_warp
    from handcdo.backends.mujoco_warp import build_batched_initial_state

    monkeypatch.setattr(mujoco_warp, "_grasp_quat_xyz", lambda grasp: np.array([1.0, 0.0, 0.0, 0.0]))
    bundle = _fake_bundle(nworld=4)
    grasps = [
        _grasp(dx=0.01, dy=-0.02, dz=0.03, closure=0.4, thumb_closure=0.7, spread_bias=0.1),
        _grasp(dx=-0.01, dy=0.02, dz=-0.03, closure=0.2, thumb_closure=0.9, spread_bias=-0.2),
    ]

    state = build_batched_initial_state(bundle, grasps, EvaluationConfig())

    assert state.qpos_init.shape == (2, 7)
    assert state.qvel_init.shape == (2, 6)
    assert state.ctrl_init.shape == (2, 2)
    assert state.xfrc_zero.shape == (2, 3, 6)
    np.testing.assert_allclose(state.qpos_init[0, :3], [0.12, -0.02, 0.085])
    np.testing.assert_allclose(state.qpos_init[1, :3], [0.10, 0.02, 0.025])
    np.testing.assert_allclose(state.qpos_init[:, 3:7], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(state.ctrl_init[0], [0.4, 0.8])
    np.testing.assert_allclose(state.ctrl_init[1], [0.2, 0.7])


def test_batched_initial_state_rejects_batch_larger_than_nworld(monkeypatch):
    from handcdo.backends import mujoco_warp
    from handcdo.backends.mujoco_warp import build_batched_initial_state

    monkeypatch.setattr(mujoco_warp, "_grasp_quat_xyz", lambda grasp: np.array([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="exceeds nworld"):
        build_batched_initial_state(_fake_bundle(nworld=1), [_grasp(), _grasp()], EvaluationConfig())


def test_true_warp_batch_path_chunks_by_nworld_and_sets_metadata(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import mujoco_warp
    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco_warp",
        SimpleNamespace(put_model=object(), put_data=object(), step=object()),
    )
    monkeypatch.setattr(mujoco_warp, "_build_warp_scene_bundle", lambda **kwargs: _fake_bundle(nworld=2))
    monkeypatch.setattr(warp_utils, "inspect_warp_batch_capabilities", lambda *args, **kwargs: _capabilities(True))
    chunks = []

    def fake_evaluate_chunk(*, grasps, design, tool_name, **kwargs):
        chunks.append([grasp.dx for grasp in grasps])
        return [
            GraspEvaluation(
                design_id=design.design_id,
                tool=tool_name,
                grasp=grasp.to_dict(),
                score=0.25,
                wrench_results=[],
            )
            for grasp in grasps
        ]

    monkeypatch.setattr(mujoco_warp, "_evaluate_grasp_chunk_true_warp", fake_evaluate_chunk)
    design = SimpleNamespace(design_id="design-true-warp")
    grasps = [_grasp(dx=0.0), _grasp(dx=0.1), _grasp(dx=0.2)]
    backend = MujocoWarpBackend(nworld=2)

    evaluations = backend.evaluate_grasps_batch(design, "hammer", grasps, EvaluationConfig(wrench_steps=1))

    assert len(evaluations) == 3
    assert chunks == [[0.0, 0.1], [0.2]]
    metadata = backend.last_batch_metadata
    assert metadata["num_chunks"] == 2
    assert metadata["failure_count"] == 0
    assert metadata["true_batched_scoring"] is True
    assert metadata["per_world_state_init"] is True
    assert metadata["wrench_directions"] == 12
    assert metadata["sequential_fallback"] is False


def test_batch_refuses_when_true_per_world_initialization_is_unavailable(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends import mujoco_warp
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
    bundle = _fake_bundle(nworld=2)
    inspected = {}
    monkeypatch.setattr(mujoco_warp, "_build_warp_scene_bundle", lambda **kwargs: bundle)

    def fake_inspect(mjw, *, warp_model=None, warp_data=None, nworld=None):
        inspected["warp_data"] = warp_data
        inspected["nworld"] = nworld
        return _capabilities(False)

    monkeypatch.setattr(warp_utils, "inspect_warp_batch_capabilities", fake_inspect)

    backend = MujocoWarpBackend(nworld=2)
    grasps = [_grasp(), _grasp(dx=0.01), _grasp(dy=0.01)]

    with pytest.raises(NotImplementedError, match="refusing to report fake batched scores"):
        backend.evaluate_grasps_batch(None, "hammer", grasps, None)

    assert inspected == {"warp_data": bundle.warp_data, "nworld": 2}
    metadata = backend.last_batch_metadata
    assert metadata["score_semantics"] == "experimental_non_equivalent"
    assert metadata["failure_count"] == 3
    assert metadata["num_chunks"] == 2
    assert metadata["sequential_fallback"] is False
    capabilities = metadata["warp_capabilities"]
    assert capabilities["can_put_model"] is True
    assert capabilities["can_put_data"] is True
    assert capabilities["can_make_data"] is False
    assert capabilities["can_step"] is True
    assert capabilities["accepted_data_allocation_kwargs"] == []
    assert capabilities["data_allocation_probe_error"] is None
    assert capabilities["can_set_per_world_qpos"] is False
    assert capabilities["can_set_per_world_qvel"] is False
    assert capabilities["can_set_per_world_ctrl"] is False
    assert capabilities["can_set_per_world_xfrc"] is False
    assert capabilities["supports_true_fixed_grasp_batching"] is False
    reason = capabilities["true_fixed_grasp_batching_reason"]
    assert "xfrc" in reason


def test_batch_default_does_not_silently_call_cpu_evaluate_grasp(monkeypatch):
    from handcdo import mujoco_eval, warp_utils
    from handcdo.backends import mujoco_warp
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
    monkeypatch.setattr(mujoco_warp, "_build_warp_scene_bundle", lambda **kwargs: _fake_bundle())
    monkeypatch.setattr(warp_utils, "inspect_warp_batch_capabilities", lambda *args, **kwargs: _capabilities(False))

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
