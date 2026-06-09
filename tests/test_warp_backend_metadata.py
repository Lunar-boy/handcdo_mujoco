from __future__ import annotations

from handcdo.warp_utils import WarpBatchCapabilities, warp_batch_metadata


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
        can_make_data=True,
        can_step=True,
        can_set_per_world_qpos=False,
        can_set_per_world_ctrl=False,
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
    assert metadata["warp_capabilities"] == {
        "can_put_model": True,
        "can_make_data": True,
        "can_step": True,
        "can_set_per_world_qpos": False,
        "can_set_per_world_ctrl": False,
        "supports_true_fixed_grasp_batching": False,
    }


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
