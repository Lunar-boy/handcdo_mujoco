from __future__ import annotations

import os

import pytest

from handcdo.batch_eval import sample_fixed_random_grasps
from handcdo.design_space import DesignSpace
from handcdo.geometry_config import GeometryConfig
from handcdo.mujoco_eval import EvaluationConfig
from handcdo.validation.mujoco_warp_gpu import (
    assert_successful_integration,
    assert_truthful_capability_failure,
    check_runtime_prerequisites,
    is_capability_exception,
)


pytestmark = pytest.mark.gpu


def test_real_mujoco_warp_backend_fixed_grasp_batch_integration() -> None:
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU integration tests.")

    prerequisites = check_runtime_prerequisites()
    if not prerequisites.ok:
        pytest.skip(prerequisites.reason or "MuJoCo Warp GPU runtime prerequisites are unavailable.")

    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    strict = os.environ.get("RUN_STRICT_WARP_INTEGRATION") == "1"
    nworld = 2
    num_grasps = 2
    warmup_steps = 1
    readback_interval = 1
    tool_name = "hammer"
    design = DesignSpace().sample(seed=13)
    grasps = sample_fixed_random_grasps(
        n_grasp_trials=num_grasps,
        seed=13,
        design_id=design.design_id,
        tool_name=tool_name,
    )
    config = EvaluationConfig(close_steps=1, settle_steps=1, wrench_steps=1)
    geometry_config = GeometryConfig()
    backend = MujocoWarpBackend(
        nworld=nworld,
        warmup_steps=warmup_steps,
        capture_graph=False,
        readback_interval=readback_interval,
    )

    try:
        evaluations = backend.evaluate_grasps_batch(
            design,
            tool_name,
            grasps,
            config,
            geometry_config=geometry_config,
        )
    except Exception as exc:
        if is_capability_exception(exc):
            metadata = backend.last_batch_metadata
            assert_truthful_capability_failure(metadata, len(grasps))
            if strict:
                raise
            pytest.xfail(f"MuJoCo Warp runtime lacks true fixed-grasp batching support: {exc}")
        raise

    assert_successful_integration(
        evaluations,
        backend.last_batch_metadata,
        tool_name=tool_name,
        num_grasps=len(grasps),
        nworld=nworld,
        warmup_steps=warmup_steps,
        readback_interval=readback_interval,
    )
