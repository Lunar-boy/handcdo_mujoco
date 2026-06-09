from __future__ import annotations

import pytest

from handcdo.batch_eval import (
    EXPERIMENTAL_WARP_METADATA,
    sample_fixed_random_grasps,
    summarize_tool_batch_results,
)
from handcdo.design_space import DesignSpace
from handcdo.mujoco_eval import GraspEvaluation


def test_pr11e_top_level_warp_result_schema_is_explicitly_experimental():
    payload = {
        "design_id": "design-a",
        "parameters": {},
        "hand_score": 0.0,
        "tool_results": [
            {
                "tool": "hammer",
                "best_score": 0.0,
                "best_grasp": {},
                "trials": [],
            }
        ],
        "failed": False,
        "backend": "mujoco_warp",
        "experimental": True,
        "include_in_multifidelity": False,
        "score_semantics": "experimental_non_equivalent",
        "warp_metadata": {
            "nworld": 64,
            "nconmax": 64,
            "naconmax": None,
            "njmax": 128,
            "warmup_steps": 0,
            "capture_graph": False,
            "batch_size": 64,
            "num_grasps": 128,
            "num_chunks": 2,
            "seconds_total": 0.0,
            "grasps_per_second": None,
            "world_steps_per_second": None,
            "failure_count": 0,
            "sequential_fallback": False,
            "mjcf_rewrites": [],
        },
    }

    assert payload["backend"] == "mujoco_warp"
    assert payload["experimental"] is True
    assert payload["include_in_multifidelity"] is False
    assert payload["score_semantics"] == "experimental_non_equivalent"
    assert payload["score_semantics"] != "intended_cpu_equivalent"
    assert payload["warp_metadata"] == {
        "nworld": 64,
        "nconmax": 64,
        "naconmax": None,
        "njmax": 128,
        "warmup_steps": 0,
        "capture_graph": False,
        "batch_size": 64,
        "num_grasps": 128,
        "num_chunks": 2,
        "seconds_total": 0.0,
        "grasps_per_second": None,
        "world_steps_per_second": None,
        "failure_count": 0,
        "sequential_fallback": False,
        "mjcf_rewrites": [],
    }


def test_warp_batch_summary_uses_conservative_experimental_metadata():
    design = DesignSpace().sample(seed=12)
    grasps = sample_fixed_random_grasps(1, seed=8, design_id=design.design_id, tool_name="hammer")
    evaluation = GraspEvaluation(design.design_id, "hammer", grasps[0].to_dict(), 1.5, [], failed=False)

    summary = summarize_tool_batch_results("hammer", grasps, [evaluation])

    for key, value in EXPERIMENTAL_WARP_METADATA.items():
        assert summary[key] == value
    assert summary["tool"] == "hammer"
    assert summary["best_score"] == 1.5
    assert summary["failure_count"] == 0
    assert summary["trials"] == [evaluation.to_dict()]


def test_warp_batch_summary_preserves_failed_trials_without_selecting_them():
    design = DesignSpace().sample(seed=13)
    grasps = sample_fixed_random_grasps(2, seed=8, design_id=design.design_id, tool_name="hammer")
    failed_high_score = GraspEvaluation(
        design.design_id,
        "hammer",
        grasps[0].to_dict(),
        100.0,
        [],
        failed=True,
        error="synthetic failure",
    )
    successful = GraspEvaluation(design.design_id, "hammer", grasps[1].to_dict(), 0.25, [], failed=False)

    summary = summarize_tool_batch_results("hammer", grasps, [failed_high_score, successful])

    assert summary["best_score"] == 0.25
    assert summary["best_grasp"] == successful.to_dict()
    assert summary["failure_count"] == 1
    assert summary["trials"][0] == failed_high_score.to_dict()


def test_warp_batch_summary_all_failed_trials_reports_no_successful_best():
    design = DesignSpace().sample(seed=14)
    grasps = sample_fixed_random_grasps(1, seed=8, design_id=design.design_id, tool_name="hammer")
    failed = GraspEvaluation(design.design_id, "hammer", grasps[0].to_dict(), 100.0, [], failed=True, error="failed")

    summary = summarize_tool_batch_results("hammer", grasps, [failed])

    assert summary["best_score"] == 0.0
    assert summary["best_grasp"]["failed"] is True
    assert summary["best_grasp"]["error"] == "no successful trials completed"
    assert summary["failure_count"] == 1
    assert summary["trials"] == [failed.to_dict()]


def test_warp_batch_summary_validates_grasp_and_evaluation_lengths():
    design = DesignSpace().sample(seed=15)
    grasps = sample_fixed_random_grasps(1, seed=8, design_id=design.design_id, tool_name="hammer")

    with pytest.raises(ValueError, match="len\\(grasps\\)"):
        summarize_tool_batch_results("hammer", grasps, [])


def test_same_design_tool_seed_trial_tuple_reproduces_grasps_without_backend():
    design = DesignSpace().sample(seed=16)

    first = sample_fixed_random_grasps(4, seed=99, design_id=design.design_id, tool_name="hammer")
    second = sample_fixed_random_grasps(4, seed=99, design_id=design.design_id, tool_name="hammer")

    assert first == second
