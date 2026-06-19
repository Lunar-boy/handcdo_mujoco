import json
import math
from pathlib import Path

from handcdo.design_space import DesignSpace
from handcdo.paper_eval_protocol import (
    DesignEvaluationResult,
    DeterministicSmokeBackend,
    PaperEvaluationConfig,
    ToolEvaluationResult,
    WrenchEvaluationResult,
    aggregate_best_grasp,
    aggregate_tool_score,
    evaluate_design_protocol,
)
from handcdo.wrench_score import canonical_wrench_directions, normalized_stable_time


def test_canonical_wrench_directions_are_unique_and_normalized():
    directions = canonical_wrench_directions()
    assert len(directions) == 12
    assert len({direction.name for direction in directions}) == 12
    for direction in directions:
        force_norm = math.sqrt(sum(value * value for value in direction.force))
        torque_norm = math.sqrt(sum(value * value for value in direction.torque))
        assert force_norm in {0.0, 1.0}
        assert torque_norm in {0.0, 1.0}
        assert (force_norm, torque_norm) in {(1.0, 0.0), (0.0, 1.0)}


def test_normalized_stable_time_is_bounded():
    assert normalized_stable_time(0, 100) == 0.0
    assert normalized_stable_time(50, 100) == 0.5
    assert normalized_stable_time(100, 100) == 1.0
    assert normalized_stable_time(150, 100) == 1.0
    assert normalized_stable_time(-10, 100) == 0.0
    assert normalized_stable_time(1, 0) == 0.0


def test_aggregation_selects_best_grasp_and_averages_wrenches():
    assert aggregate_tool_score([0.0, 0.5, 1.0]) == 0.5
    assert aggregate_best_grasp([[0.0, 0.5], [0.75, 1.0]]) == 0.875


def test_result_schema_json_round_trip(tmp_path):
    result = DesignEvaluationResult(
        design_id="design",
        geometry_config="configs/eval_paper_like.yaml",
        tools=(
            ToolEvaluationResult(
                tool="hammer",
                grasp_id="grasp",
                wrench_results=(WrenchEvaluationResult("+Fx", 0.5, 0.5, "stability_threshold_exceeded"),),
                mean_score=0.5,
            ),
        ),
        aggregate_score=0.5,
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    assert DesignEvaluationResult.from_json(path) == result


def test_paper_protocol_config_loads_required_fields():
    path = Path("configs/eval_paper_protocol.yaml")
    config = PaperEvaluationConfig.from_yaml(path)
    assert config.tools == ("hammer", "spoon", "knife")
    assert config.candidates_per_tool == 8
    assert config.evaluation.force_magnitude == 1.0
    assert config.evaluation.torque_magnitude == 0.05
    assert config.evaluation.wrench_steps == 500
    assert config.geometry_config_path.endswith("configs/eval_paper_like.yaml")


def test_smoke_evaluator_is_deterministic_with_same_seed():
    config = PaperEvaluationConfig.from_yaml("configs/eval_paper_protocol.yaml")
    design = DesignSpace.from_yaml("configs/search_space.yaml").sample(seed=7)
    backend = DeterministicSmokeBackend()
    first = evaluate_design_protocol(design, config, backend=backend)
    second = evaluate_design_protocol(design, config, backend=backend)
    assert first == second
    assert 0.0 <= first.aggregate_score <= 1.0
    assert len(first.tools) == 3
    assert all(len(tool.wrench_results) == 12 for tool in first.tools)
