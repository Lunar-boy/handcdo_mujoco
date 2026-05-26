from handcdo.collect_results import collect_results
from handcdo.utils import write_json


def test_collector_handles_missing_or_failed_payloads(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "ok.json",
        {
            "design_id": "ok",
            "parameters": {"finger_number": 2, "finger_code": "1-1-1"},
            "hand_score": 0.25,
            "tool_results": [{"tool": "hammer", "best_score": 0.3}],
            "failed": False,
        },
    )
    (results / "bad.json").write_text("{not-json", encoding="utf-8")
    rows = collect_results(results, tmp_path / "merged.csv")
    assert len(rows) == 2
    assert (tmp_path / "merged.csv").exists()
    assert any(row["failed"] for row in rows)
