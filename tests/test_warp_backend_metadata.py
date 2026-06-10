from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from handcdo.warp_utils import WarpBatchCapabilities, inspect_warp_batch_capabilities, warp_batch_metadata


def test_inspect_warp_batch_capabilities_reports_mandatory_conservative_probe_fields():
    fake_mjw = SimpleNamespace(
        put_model=object(),
        put_data=object(),
        make_data=object(),
        step=object(),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw)

    assert capabilities.can_put_model is True
    assert capabilities.can_put_data is True
    assert capabilities.can_make_data is True
    assert capabilities.can_step is True
    assert capabilities.accepted_data_allocation_kwargs == []
    assert capabilities.data_allocation_probe_error
    assert "not probed" in capabilities.data_allocation_probe_error
    assert capabilities.can_set_per_world_qpos is False
    assert capabilities.can_set_per_world_qvel is False
    assert capabilities.can_set_per_world_ctrl is False
    assert capabilities.can_set_per_world_xfrc is False
    assert capabilities.supports_true_fixed_grasp_batching is False
    reason = capabilities.true_fixed_grasp_batching_reason
    assert reason
    assert "qpos" in reason
    assert "qvel" in reason
    assert "ctrl" in reason
    assert "xfrc" in reason


def test_warp_batch_metadata_includes_required_pr11d_keys():
    metadata = warp_batch_metadata(
        nworld=64,
        nconmax=64,
        naconmax=None,
        njmax=128,
        num_grasps=128,
        num_chunks=2,
        failure_count=0,
        seconds_total=0.0,
    )

    assert metadata == {
        "backend": "mujoco_warp",
        "experimental": True,
        "score_semantics": "experimental_non_equivalent",
        "nworld": 64,
        "nconmax": 64,
        "naconmax": None,
        "njmax": 128,
        "num_grasps": 128,
        "num_chunks": 2,
        "failure_count": 0,
        "sequential_fallback": False,
        "seconds_total": 0.0,
        "grasps_per_second": None,
        "world_steps_per_second": None,
        "mjcf_rewrites": [],
    }


def test_warp_batch_metadata_records_conservative_capabilities_and_failure_reason():
    capabilities = WarpBatchCapabilities(
        can_put_model=True,
        can_put_data=True,
        can_make_data=True,
        can_step=True,
        accepted_data_allocation_kwargs=[],
        data_allocation_probe_error="not probed: no concrete MuJoCo objects",
        can_set_per_world_qpos=False,
        can_set_per_world_qvel=False,
        can_set_per_world_ctrl=False,
        can_set_per_world_xfrc=False,
        true_fixed_grasp_batching_reason="qpos qvel ctrl xfrc not verified",
    )

    metadata = warp_batch_metadata(
        nworld=8,
        nconmax=64,
        naconmax=None,
        njmax=128,
        num_grasps=3,
        num_chunks=1,
        failure_count=3,
        seconds_total=0.01,
        capabilities=capabilities,
        failure_reason="true batching unavailable",
    )

    assert metadata["score_semantics"] == "experimental_non_equivalent"
    assert metadata["failure_reason"] == "true batching unavailable"
    assert metadata["warp_capabilities"] | {
        "can_put_model": True,
        "can_put_data": True,
        "can_make_data": True,
        "can_step": True,
        "accepted_data_allocation_kwargs": [],
        "data_allocation_probe_error": "not probed: no concrete MuJoCo objects",
        "can_set_per_world_qpos": False,
        "can_set_per_world_qvel": False,
        "can_set_per_world_ctrl": False,
        "can_set_per_world_xfrc": False,
        "supports_true_fixed_grasp_batching": False,
        "true_fixed_grasp_batching_reason": "qpos qvel ctrl xfrc not verified",
    } == metadata["warp_capabilities"]


def test_capability_probe_detects_batched_mock_data_and_verified_writes():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object())
    warp_data = SimpleNamespace(
        qpos=np.zeros((2, 3)),
        qvel=np.zeros((2, 2)),
        ctrl=np.zeros((2, 1)),
        xfrc_applied=np.zeros((2, 4, 6)),
    )
    original_qpos = warp_data.qpos.copy()

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.has_qpos is True
    assert capabilities.has_qvel is True
    assert capabilities.has_ctrl is True
    assert capabilities.has_xfrc_applied is True
    assert capabilities.qpos_is_batched is True
    assert capabilities.qvel_is_batched is True
    assert capabilities.ctrl_is_batched is True
    assert capabilities.xfrc_is_batched is True
    assert capabilities.can_set_per_world_qpos is True
    assert capabilities.can_set_per_world_qvel is True
    assert capabilities.can_set_per_world_ctrl is True
    assert capabilities.can_set_per_world_xfrc is True
    assert capabilities.supports_true_fixed_grasp_batching is True
    np.testing.assert_allclose(warp_data.qpos, original_qpos)


def test_capability_probe_rejects_unbatched_mock_data():
    fake_mjw = SimpleNamespace(put_model=object(), make_data=object(), step=object())
    warp_data = SimpleNamespace(
        qpos=np.zeros((3,)),
        qvel=np.zeros((2,)),
        ctrl=np.zeros((1,)),
        xfrc_applied=np.zeros((4, 6)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.has_qpos is True
    assert capabilities.qpos_is_batched is False
    assert capabilities.can_set_per_world_qpos is False
    assert capabilities.supports_true_fixed_grasp_batching is False


def test_shape_only_probe_does_not_report_write_support_for_read_only_fields():
    class ReadOnlyField:
        def __init__(self, shape):
            self.shape = shape

        def numpy(self):
            return np.zeros(self.shape)

    fake_mjw = SimpleNamespace(put_model=object(), make_data=object(), step=object())
    warp_data = SimpleNamespace(
        qpos=ReadOnlyField((2, 3)),
        qvel=ReadOnlyField((2, 2)),
        ctrl=ReadOnlyField((2, 1)),
        xfrc_applied=ReadOnlyField((2, 4, 6)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.qpos_is_batched is True
    assert capabilities.qvel_is_batched is True
    assert capabilities.ctrl_is_batched is True
    assert capabilities.xfrc_is_batched is True
    assert capabilities.can_set_per_world_qpos is False
    assert capabilities.supports_true_fixed_grasp_batching is False


def test_sequential_fallback_metadata_is_explicitly_non_batch_throughput():
    metadata = warp_batch_metadata(
        nworld=4,
        nconmax=64,
        naconmax=None,
        njmax=128,
        num_grasps=2,
        num_chunks=2,
        failure_count=0,
        seconds_total=1.0,
        score_semantics="experimental_sequential_fallback",
        sequential_fallback=True,
        grasps_per_second=2.0,
    )

    assert metadata["score_semantics"] == "experimental_sequential_fallback"
    assert metadata["sequential_fallback"] is True
    assert metadata["world_steps_per_second"] is None
