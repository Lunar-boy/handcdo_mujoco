from __future__ import annotations

import importlib

import pandas as pd
import pytest

from handcdo.compare_benchmarks import compare_benchmarks


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_self_comparison_mean_delta_zero(tmp_path):
    csv_path = tmp_path / "baseline.csv"
    _write_csv(csv_path, [{"design_id": "a", "hand_score": 1.0}, {"design_id": "b", "hand_score": 0.5}])

    summary = compare_benchmarks(csv_path, csv_path, tmp_path / "compare", top_k="1")

    assert summary["scores"]["mean_delta"] == 0.0
    assert (tmp_path / "compare" / "comparison_summary.json").exists()
    assert (tmp_path / "compare" / "joined_scores.csv").exists()


def test_top_k_overlap_and_effective_k(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(
        left,
        [
            {"design_id": "a", "hand_score": 3.0},
            {"design_id": "b", "hand_score": 2.0},
            {"design_id": "c", "hand_score": 1.0},
        ],
    )
    _write_csv(
        right,
        [
            {"design_id": "a", "hand_score": 1.0},
            {"design_id": "b", "hand_score": 3.0},
            {"design_id": "c", "hand_score": 2.0},
        ],
    )

    summary = compare_benchmarks(left, right, tmp_path / "compare", top_k="1,5")

    assert summary["top_k"]["1"]["overlap_count"] == 0
    assert summary["top_k"]["1"]["overlap_ratio"] == 0.0
    assert summary["top_k"]["5"]["effective_k"] == 3
    assert summary["top_k"]["5"]["overlap_count"] == 3


def test_missing_design_id_raises_clear_error(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"id": "a", "hand_score": 1.0}])
    _write_csv(right, [{"design_id": "a", "hand_score": 1.0}])

    with pytest.raises(ValueError, match="design_id"):
        compare_benchmarks(left, right, tmp_path / "compare")


def test_missing_selected_score_column_raises_clear_error(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"design_id": "a", "other_score": 1.0}])
    _write_csv(right, [{"design_id": "a", "hand_score": 1.0}])

    with pytest.raises(ValueError, match="hand_score"):
        compare_benchmarks(left, right, tmp_path / "compare")


def test_no_common_designs_raises_clear_error(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"design_id": "a", "hand_score": 1.0}])
    _write_csv(right, [{"design_id": "b", "hand_score": 1.0}])

    with pytest.raises(ValueError, match="No common"):
        compare_benchmarks(left, right, tmp_path / "compare")


def test_designs_only_in_left_and_right_are_counted(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"design_id": "a", "hand_score": 1.0}, {"design_id": "b", "hand_score": 0.5}])
    _write_csv(right, [{"design_id": "a", "hand_score": 1.0}, {"design_id": "c", "hand_score": 0.5}])

    summary = compare_benchmarks(left, right, tmp_path / "compare")

    assert summary["only_left_count"] == 1
    assert summary["only_right_count"] == 1
    assert summary["only_left"] == ["b"]
    assert summary["only_right"] == ["c"]


def test_scipy_absence_does_not_fail(tmp_path, monkeypatch):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"design_id": "a", "hand_score": 1.0}, {"design_id": "b", "hand_score": 0.5}])
    _write_csv(right, [{"design_id": "a", "hand_score": 1.0}, {"design_id": "b", "hand_score": 0.5}])
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "scipy.stats":
            raise ImportError("no scipy")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    summary = compare_benchmarks(left, right, tmp_path / "compare")

    assert summary["spearman"] is None
    assert summary["warnings"]


def test_shared_tool_best_score_columns_are_compared(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"design_id": "a", "hand_score": 1.0, "hammer_best_score": 0.25}])
    _write_csv(right, [{"design_id": "a", "hand_score": 1.5, "hammer_best_score": 0.5}])

    summary = compare_benchmarks(left, right, tmp_path / "compare")
    joined = pd.read_csv(tmp_path / "compare" / "joined_scores.csv")

    assert summary["tool_scores"]["hammer_best_score"]["mean_delta"] == 0.25
    assert "hammer_best_score_delta" in joined.columns
