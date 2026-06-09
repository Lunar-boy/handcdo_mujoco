from __future__ import annotations

from handcdo.backends.batched import BatchedSimulatorBackend, supports_batched_grasps
from handcdo.backends.mujoco_cpu import MujocoCpuBackend


class PlainObject:
    pass


class NonCallableBatchAttribute:
    evaluate_grasps_batch = None


class DummyBatchedBackend:
    name = "dummy_batched"

    def evaluate_grasps_batch(
        self,
        design,
        tool_name,
        grasps,
        config,
        geometry_config=None,
        tool_assets_dir="assets/tools",
    ):
        return []


def test_batched_protocol_imports_without_warp_dependency():
    assert BatchedSimulatorBackend.__name__ == "BatchedSimulatorBackend"


def test_supports_batched_grasps_false_for_plain_objects():
    assert supports_batched_grasps(PlainObject()) is False
    assert supports_batched_grasps(NonCallableBatchAttribute()) is False


def test_supports_batched_grasps_true_for_callable_batch_method():
    backend = DummyBatchedBackend()

    assert supports_batched_grasps(backend) is True
    assert isinstance(backend, BatchedSimulatorBackend)


def test_existing_cpu_backend_imports_and_is_not_forced_to_batch():
    backend = MujocoCpuBackend()

    assert backend.name == "mujoco_cpu"
    assert supports_batched_grasps(backend) is False
