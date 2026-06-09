from __future__ import annotations

import importlib
import sys

import pytest

from handcdo.batch_eval import evaluate_fixed_grasps_batched, sample_fixed_random_grasps, summarize_tool_batch_results
from handcdo.design_space import DesignSpace
from handcdo.mujoco_eval import GraspEvaluation


def _design():
    return DesignSpace().sample(seed=3)


class DummyBatchedBackend:
    name = "dummy_batched"

    def __init__(self, fail_indices: set[int] | None = None) -> None:
        self.batch_calls = 0
        self.single_calls = 0
        self.seen_grasps = None
        self.fail_indices = fail_indices or set()

    def evaluate_grasp(self, *args, **kwargs):
        self.single_calls += 1
        raise AssertionError("evaluate_grasp must not be called by fixed-grasp orchestration")

    def evaluate_grasps_batch(
        self,
        design,
        tool_name,
        grasps,
        config,
        geometry_config=None,
        tool_assets_dir="assets/tools",
    ):
        self.batch_calls += 1
        self.seen_grasps = grasps
        return [
            GraspEvaluation(
                design_id=design.design_id,
                tool=tool_name,
                grasp=grasp.to_dict(),
                score=0.0 if index in self.fail_indices else float(index + 1),
                wrench_results=[],
                failed=index in self.fail_indices,
                error="synthetic failure" if index in self.fail_indices else None,
            )
            for index, grasp in enumerate(grasps)
        ]


def test_empty_grasp_list_returns_empty_without_backend_call():
    backend = DummyBatchedBackend()

    assert evaluate_fixed_grasps_batched(backend, _design(), "hammer", [], None) == []
    assert backend.batch_calls == 0
    assert backend.single_calls == 0


def test_batch_helper_preserves_length_order_and_uses_one_batch_call():
    design = _design()
    grasps = sample_fixed_random_grasps(4, seed=17, design_id=design.design_id, tool_name="hammer")
    backend = DummyBatchedBackend()

    evaluations = evaluate_fixed_grasps_batched(backend, design, "hammer", grasps, None)

    assert backend.batch_calls == 1
    assert backend.single_calls == 0
    assert backend.seen_grasps == grasps
    assert len(evaluations) == len(grasps)
    assert [evaluation.grasp for evaluation in evaluations] == [grasp.to_dict() for grasp in grasps]
    assert [evaluation.score for evaluation in evaluations] == [1.0, 2.0, 3.0, 4.0]


def test_failed_evaluations_are_not_dropped():
    design = _design()
    grasps = sample_fixed_random_grasps(3, seed=5, design_id=design.design_id, tool_name="hammer")
    backend = DummyBatchedBackend(fail_indices={1})

    evaluations = evaluate_fixed_grasps_batched(backend, design, "hammer", grasps, None)

    assert len(evaluations) == 3
    assert [evaluation.failed for evaluation in evaluations] == [False, True, False]
    assert evaluations[1].error == "synthetic failure"


def test_backend_batch_exception_becomes_structured_failures():
    class RaisingBatchBackend(DummyBatchedBackend):
        def evaluate_grasps_batch(self, *args, **kwargs):
            self.batch_calls += 1
            raise RuntimeError("backend exploded")

    design = _design()
    grasps = sample_fixed_random_grasps(2, seed=9, design_id=design.design_id, tool_name="hammer")
    backend = RaisingBatchBackend()

    evaluations = evaluate_fixed_grasps_batched(backend, design, "hammer", grasps, None)

    assert backend.batch_calls == 1
    assert len(evaluations) == 2
    assert all(evaluation.failed for evaluation in evaluations)
    assert all("RuntimeError: backend exploded" == evaluation.error for evaluation in evaluations)


def test_wrong_batch_result_length_fails_clearly():
    class ShortBatchBackend(DummyBatchedBackend):
        def evaluate_grasps_batch(self, *args, **kwargs):
            self.batch_calls += 1
            return []

    design = _design()
    grasps = sample_fixed_random_grasps(2, seed=10, design_id=design.design_id, tool_name="hammer")

    with pytest.raises(ValueError, match="wrong number of evaluations"):
        evaluate_fixed_grasps_batched(ShortBatchBackend(), design, "hammer", grasps, None)


def test_summary_ignores_failed_trials_for_best_grasp_but_preserves_trials():
    design = _design()
    grasps = sample_fixed_random_grasps(3, seed=11, design_id=design.design_id, tool_name="hammer")
    evaluations = [
        GraspEvaluation(design.design_id, "hammer", grasps[0].to_dict(), 2.0, [], failed=False),
        GraspEvaluation(design.design_id, "hammer", grasps[1].to_dict(), 99.0, [], failed=True, error="failed high score"),
        GraspEvaluation(design.design_id, "hammer", grasps[2].to_dict(), 3.0, [], failed=False),
    ]

    summary = summarize_tool_batch_results("hammer", grasps, evaluations)

    assert summary["failure_count"] == 1
    assert summary["best_score"] == 3.0
    assert summary["best_grasp"]["grasp"] == grasps[2].to_dict()
    assert len(summary["trials"]) == 3
    assert summary["trials"][1]["failed"] is True


def test_fixed_random_grasp_generation_is_deterministic_and_contextual():
    first = sample_fixed_random_grasps(5, seed=42, design_id="design-a", tool_name="hammer")
    second = sample_fixed_random_grasps(5, seed=42, design_id="design-a", tool_name="hammer")
    different_seed = sample_fixed_random_grasps(5, seed=43, design_id="design-a", tool_name="hammer")
    different_tool = sample_fixed_random_grasps(5, seed=42, design_id="design-a", tool_name="spoon")

    assert first == second
    assert first != different_seed
    assert first != different_tool


def test_helper_import_does_not_require_mujoco_warp():
    sys.modules.pop("handcdo.batch_eval", None)
    sys.modules.pop("mujoco_warp", None)

    module = importlib.import_module("handcdo.batch_eval")

    assert module.evaluate_fixed_grasps_batched.__name__ == "evaluate_fixed_grasps_batched"
    assert "mujoco_warp" not in sys.modules
