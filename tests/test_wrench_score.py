from handcdo.wrench_score import WrenchDirectionResult, aggregate_wrench_results


def test_wrench_score_bounds():
    results = [
        WrenchDirectionResult("a", 0, 10, 0.0, 0.1, 0.2, True),
        WrenchDirectionResult("b", 10, 10, 1.0, 0.0, 0.0, False),
    ]
    score = aggregate_wrench_results(results)
    assert 0.0 <= score <= 1.0
    assert score == 0.5
