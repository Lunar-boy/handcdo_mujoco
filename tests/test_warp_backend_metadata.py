from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from handcdo.warp_utils import WarpBatchCapabilities, inspect_warp_batch_capabilities, warp_batch_metadata


class WholeBatchOnlyField:
    def __init__(self, shape, fill=0.0):
        self.data = np.full(shape, fill, dtype=float)
        self.assignments = []

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return "cpu-test"

    def __array__(self, dtype=None, copy=None):
        return np.array(self.data, dtype=dtype, copy=copy if copy is not None else True)

    def __setitem__(self, key, value):
        if key is not Ellipsis:
            raise TypeError("per-world and partial writes rejected")
        array = np.asarray(value, dtype=float)
        if array.shape != self.data.shape:
            raise ValueError(f"expected full shape {self.data.shape}, got {array.shape}")
        self.assignments.append(array.copy())
        self.data[...] = array


class RejectAllWriteField:
    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=float)

    @property
    def shape(self):
        return self.data.shape

    def __array__(self, dtype=None, copy=None):
        return np.array(self.data, dtype=dtype, copy=copy if copy is not None else True)

    def __setitem__(self, key, value):
        raise TypeError("writes rejected")


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
    assert capabilities.can_forward is False
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

    assert metadata | {
        "backend": "mujoco_warp",
        "experimental": True,
        "score_semantics": "experimental_non_equivalent",
        "true_batched_scoring": True,
        "per_world_state_init": True,
        "wrench_directions": 12,
        "include_in_multifidelity": False,
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
    } == metadata


def test_warp_batch_metadata_records_conservative_capabilities_and_failure_reason():
    capabilities = WarpBatchCapabilities(
        can_put_model=True,
        can_put_data=True,
        can_make_data=True,
        can_step=True,
        can_forward=False,
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
    assert metadata["true_batched_scoring"] is False
    assert metadata["per_world_state_init"] is False
    assert metadata["include_in_multifidelity"] is False
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


def test_capability_probe_rejects_missing_xpos_xmat_even_with_state_writes():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object(), forward=lambda *args: None)
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
    assert capabilities.qpos_write_method == "direct_setitem"
    assert capabilities.has_xpos is False
    assert capabilities.has_xmat is False
    assert capabilities.supports_true_fixed_grasp_batching is False
    assert "xpos" in capabilities.true_fixed_grasp_batching_reason
    assert "xmat" in capabilities.true_fixed_grasp_batching_reason
    np.testing.assert_allclose(warp_data.qpos, original_qpos)


def test_capability_probe_detects_full_batched_mock_data_and_verified_writes():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object(), forward=lambda *args: None)
    warp_data = SimpleNamespace(
        qpos=np.zeros((2, 3)),
        qvel=np.zeros((2, 2)),
        ctrl=np.zeros((2, 1)),
        xfrc_applied=np.zeros((2, 4, 6)),
        xpos=np.zeros((2, 4, 3)),
        xmat=np.zeros((2, 4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.can_forward is True
    assert capabilities.kinematics_update_method == "forward"
    assert capabilities.has_xpos is True
    assert capabilities.has_xmat is True
    assert capabilities.xpos_is_batched is True
    assert capabilities.xmat_is_batched is True
    assert capabilities.supports_true_fixed_grasp_batching is True


def test_capability_probe_uses_whole_batch_write_fallback_and_reports_diagnostics():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object(), forward=lambda *args: None)
    warp_data = SimpleNamespace(
        qpos=WholeBatchOnlyField((2, 3)),
        qvel=WholeBatchOnlyField((2, 2)),
        ctrl=WholeBatchOnlyField((2, 1)),
        xfrc_applied=WholeBatchOnlyField((2, 4, 6)),
        xpos=np.zeros((2, 4, 3)),
        xmat=np.zeros((2, 4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.supports_true_fixed_grasp_batching is True
    assert capabilities.can_set_per_world_qpos is True
    assert capabilities.can_set_per_world_qvel is True
    assert capabilities.can_set_per_world_ctrl is True
    assert capabilities.can_set_per_world_xfrc is True
    assert capabilities.qpos_write_method == "whole_batch_setitem"
    qpos_report = capabilities.field_reports["qpos"]
    assert qpos_report["write_tested"] is True
    assert qpos_report["write_roundtrip_verified"] is True
    assert qpos_report["type"] == "WholeBatchOnlyField"
    assert qpos_report["dtype"] == "float64"
    assert qpos_report["device"] == "cpu-test"
    assert any(attempt["success"] is False for attempt in qpos_report["write_attempts"])
    assert any(attempt["method"] == "whole_batch_setitem" for attempt in qpos_report["write_attempts"])


def test_whole_batch_probe_verifies_distinct_worlds_and_restores_original():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object(), forward=lambda *args: None)
    qpos = WholeBatchOnlyField((2, 3))
    qpos.data[0] = [1.0, 2.0, 3.0]
    qpos.data[1] = [4.0, 5.0, 6.0]
    original = qpos.data.copy()
    warp_data = SimpleNamespace(
        qpos=qpos,
        qvel=WholeBatchOnlyField((2, 2)),
        ctrl=WholeBatchOnlyField((2, 1)),
        xfrc_applied=WholeBatchOnlyField((2, 4, 6)),
        xpos=np.zeros((2, 4, 3)),
        xmat=np.zeros((2, 4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.supports_true_fixed_grasp_batching is True
    assert len(qpos.assignments) >= 2
    mutated = qpos.assignments[0]
    assert mutated[0, 0] != original[0, 0]
    assert mutated[1, 0] != original[1, 0]
    assert mutated[0, 0] != mutated[1, 0]
    np.testing.assert_allclose(qpos.data, original)


def test_capability_probe_reports_unsupported_whole_batch_write_truthfully():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object(), forward=lambda *args: None)
    warp_data = SimpleNamespace(
        qpos=RejectAllWriteField((2, 3)),
        qvel=RejectAllWriteField((2, 2)),
        ctrl=RejectAllWriteField((2, 1)),
        xfrc_applied=RejectAllWriteField((2, 4, 6)),
        xpos=np.zeros((2, 4, 3)),
        xmat=np.zeros((2, 4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.supports_true_fixed_grasp_batching is False
    assert capabilities.can_set_per_world_qpos is False
    assert "lacks verified batched write support" in capabilities.true_fixed_grasp_batching_reason
    qpos_report = capabilities.field_reports["qpos"]
    assert qpos_report["write_tested"] is False
    assert "whole-batch assignment failed" in qpos_report["reason"]
    assert qpos_report["write_attempts"]


def test_capability_probe_can_use_field_native_copy_method():
    class CopySlice:
        def __init__(self, parent, world_index):
            self.parent = parent
            self.world_index = world_index

        def copy_(self, value):
            self.parent.data[self.world_index] = value

    class CopyOnlyField:
        def __init__(self, shape):
            self.data = np.zeros(shape)

        @property
        def shape(self):
            return self.data.shape

        def __array__(self, dtype=None, copy=None):
            return np.array(self.data, dtype=dtype, copy=copy if copy is not None else True)

        def __getitem__(self, key):
            world_index = key[0] if isinstance(key, tuple) else key
            return CopySlice(self, world_index)

    fake_mjw = SimpleNamespace(put_model=object(), make_data=object(), step=object(), forward=lambda *args: None)
    warp_data = SimpleNamespace(
        qpos=CopyOnlyField((2, 3)),
        qvel=CopyOnlyField((2, 2)),
        ctrl=CopyOnlyField((2, 1)),
        xfrc_applied=CopyOnlyField((2, 4, 6)),
        xpos=np.zeros((2, 4, 3)),
        xmat=np.zeros((2, 4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.can_set_per_world_qpos is True
    assert capabilities.qpos_write_method == "field.copy_"
    assert capabilities.supports_true_fixed_grasp_batching is True
    np.testing.assert_allclose(warp_data.qpos.data, np.zeros((2, 3)))


def test_capability_probe_rejects_unbatched_mock_data():
    fake_mjw = SimpleNamespace(put_model=object(), make_data=object(), step=object(), forward=lambda *args: None)
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

    fake_mjw = SimpleNamespace(put_model=object(), make_data=object(), step=object(), forward=lambda *args: None)
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


def test_capability_probe_rejects_missing_forward_even_with_full_data():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object())
    warp_data = SimpleNamespace(
        qpos=np.zeros((2, 3)),
        qvel=np.zeros((2, 2)),
        ctrl=np.zeros((2, 1)),
        xfrc_applied=np.zeros((2, 4, 6)),
        xpos=np.zeros((2, 4, 3)),
        xmat=np.zeros((2, 4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.can_forward is False
    assert capabilities.supports_true_fixed_grasp_batching is False
    assert "forward" in capabilities.true_fixed_grasp_batching_reason


def test_capability_probe_rejects_unbatched_xpos_xmat():
    fake_mjw = SimpleNamespace(put_model=object(), put_data=object(), step=object(), forward=lambda *args: None)
    warp_data = SimpleNamespace(
        qpos=np.zeros((2, 3)),
        qvel=np.zeros((2, 2)),
        ctrl=np.zeros((2, 1)),
        xfrc_applied=np.zeros((2, 4, 6)),
        xpos=np.zeros((4, 3)),
        xmat=np.zeros((4, 9)),
    )

    capabilities = inspect_warp_batch_capabilities(fake_mjw, warp_data=warp_data, nworld=2)

    assert capabilities.xpos_is_batched is False
    assert capabilities.xmat_is_batched is False
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

def test_warp_batch_metadata_includes_null_failure_reason_on_success():
    from handcdo.warp_utils import warp_batch_metadata

    metadata = warp_batch_metadata(
        nworld=2,
        nconmax=64,
        naconmax=None,
        njmax=128,
        num_grasps=2,
        num_chunks=1,
        failure_count=0,
        seconds_total=0.1,
        failure_reason=None,
        true_batched_scoring=True,
        per_world_state_init=True,
    )

    assert "failure_reason" in metadata
    assert metadata["failure_reason"] is None
