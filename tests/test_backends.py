from __future__ import annotations

import pytest

from handcdo.backends import get_backend
from handcdo.design_space import DesignSpace
from handcdo.mujoco_eval import GraspEvaluation
from handcdo.optimize_grasp import optimize_grasp_for_tool
from handcdo.optimize_hand import build_parser, evaluate_design


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls = []

    def evaluate_grasp(self, design, tool_name, grasp, config):
        self.calls.append((design, tool_name, grasp, config))
        return GraspEvaluation(
            design_id=design.design_id,
            tool=tool_name,
            grasp=grasp.to_dict(),
            score=float(len(self.calls)),
            wrench_results=[],
        )


def test_get_backend_mujoco_cpu():
    assert get_backend("mujoco_cpu").name == "mujoco_cpu"


def test_get_backend_mujoco_legacy_alias():
    assert get_backend("mujoco").name == get_backend("mujoco_cpu").name


def test_get_backend_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown simulator backend"):
        get_backend("definitely_not_a_backend")


def test_parser_accepts_mujoco_backend_aliases():
    parser = build_parser()
    assert parser.parse_args(["--backend", "mujoco"]).backend == "mujoco"
    assert parser.parse_args(["--backend", "mujoco_cpu"]).backend == "mujoco_cpu"


def test_optimize_grasp_routes_random_search_to_backend():
    design = DesignSpace().sample(seed=0)
    backend = FakeBackend()
    payload = optimize_grasp_for_tool(design, "hammer", n_trials=3, seed=0, sampler="random", backend=backend)

    assert len(backend.calls) == 3
    assert payload["tool"] == "hammer"
    assert payload["best_score"] == 3.0
    assert payload["best_grasp"]["score"] == 3.0
    assert len(payload["trials"]) == 3


def test_optimize_grasp_routes_tpe_search_to_backend():
    pytest.importorskip("optuna")
    design = DesignSpace().sample(seed=1)
    backend = FakeBackend()
    payload = optimize_grasp_for_tool(design, "spoon", n_trials=2, seed=0, sampler="tpe", backend=backend)

    assert len(backend.calls) == 2
    assert payload["tool"] == "spoon"
    assert payload["best_score"] == 2.0
    assert len(payload["trials"]) == 2


def test_evaluate_design_reuses_fake_backend_for_tools(tmp_path):
    design = DesignSpace().sample(seed=2)
    backend = FakeBackend()

    payload = evaluate_design(
        design,
        tools=["hammer", "knife"],
        n_grasp_trials=1,
        output_dir=tmp_path,
        seed=0,
        backend=backend,
    )

    assert len(backend.calls) == 2
    assert payload["design_id"] == design.design_id
    assert payload["parameters"] == design.to_dict()
    assert payload["hand_score"] == 1.5
    assert [result["tool"] for result in payload["tool_results"]] == ["hammer", "knife"]
    assert payload["failed"] is False


def test_get_backend_normalizes_whitespace_and_case():
    assert get_backend(" mujoco_cpu ").name == "mujoco_cpu"
    assert get_backend("MUJOCO").name == "mujoco_cpu"
