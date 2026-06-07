from __future__ import annotations

import pandas as pd
import pytest

from handcdo.multifidelity import select_top_designs


def test_select_top_designs_excludes_failed_and_ties_by_design_id(tmp_path):
    input_csv = tmp_path / "results.csv"
    output_ids = tmp_path / "ids.txt"
    pd.DataFrame(
        [
            {"design_id": "b", "hand_score": 0.9, "failed": False},
            {"design_id": "a", "hand_score": 0.9, "failed": False},
            {"design_id": "failed", "hand_score": 1.0, "failed": True},
            {"design_id": "nan", "hand_score": "not-a-number", "failed": False},
        ]
    ).to_csv(input_csv, index=False)

    selected = select_top_designs(input_csv, 3, output_ids)

    assert selected == ["a", "b"]
    assert output_ids.read_text(encoding="utf-8") == "a\nb\n"


def test_select_top_designs_can_include_failed_rows(tmp_path):
    input_csv = tmp_path / "results.csv"
    output_ids = tmp_path / "ids.txt"
    pd.DataFrame(
        [
            {"design_id": "ok", "hand_score": 0.1, "failed": False},
            {"design_id": "failed", "hand_score": 0.8, "failed": True},
        ]
    ).to_csv(input_csv, index=False)

    selected = select_top_designs(input_csv, 1, output_ids, include_failed=True)

    assert selected == ["failed"]


def test_select_top_designs_fails_when_required_columns_are_missing(tmp_path):
    input_csv = tmp_path / "results.csv"
    output_ids = tmp_path / "ids.txt"
    pd.DataFrame([{"design_id": "a", "score": 1.0}]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="hand_score"):
        select_top_designs(input_csv, 1, output_ids)
