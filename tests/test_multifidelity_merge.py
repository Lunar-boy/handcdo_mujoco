from __future__ import annotations

import pandas as pd

from handcdo.multifidelity import merge_multifidelity_results


def test_merge_multifidelity_results_preserves_scores_metadata_and_parameters(tmp_path):
    fast_csv = tmp_path / "fast.csv"
    medium_csv = tmp_path / "medium.csv"
    high_csv = tmp_path / "high.csv"
    output_csv = tmp_path / "merged.csv"

    pd.DataFrame(
        [
            {
                "design_id": "a",
                "hand_score": 0.1,
                "failed": False,
                "backend": "mujoco_cpu",
                "seed": 1,
                "finger_number": 2,
                "hammer_best_score": 0.11,
            },
            {"design_id": "b", "hand_score": 0.2, "failed": False, "finger_number": 3},
        ]
    ).to_csv(fast_csv, index=False)
    pd.DataFrame(
        [
            {
                "design_id": "a",
                "hand_score": 0.3,
                "failed": False,
                "config_path": "configs/eval_medium.yaml",
                "n_grasp_trials": 4,
                "sampler": "tpe",
                "finger_number": 2,
            },
        ]
    ).to_csv(medium_csv, index=False)
    pd.DataFrame(
        [
            {
                "design_id": "c",
                "hand_score": 0.5,
                "failed": False,
                "error": "",
                "finger_number": 4,
            },
        ]
    ).to_csv(high_csv, index=False)

    merged = merge_multifidelity_results(
        {"fast": fast_csv, "medium": medium_csv, "high": high_csv},
        output_csv,
    )

    assert output_csv.exists()
    assert {"hand_score_fast", "hand_score_medium", "hand_score_high"}.issubset(merged.columns)
    assert {"failed_fast", "backend_fast", "config_path_medium", "n_grasp_trials_medium", "sampler_medium"}.issubset(
        merged.columns
    )
    assert "finger_number" in merged.columns
    assert "finger_number_fast" not in merged.columns
    assert "hammer_best_score_fast" in merged.columns

    by_id = merged.set_index("design_id")
    assert by_id.loc["a", "best_available_score"] == 0.3
    assert by_id.loc["b", "best_available_score"] == 0.2
    assert by_id.loc["c", "best_available_score"] == 0.5


def test_merge_multifidelity_results_handles_missing_fidelity_csvs(tmp_path):
    fast_csv = tmp_path / "fast.csv"
    output_csv = tmp_path / "merged.csv"
    pd.DataFrame([{"design_id": "a", "hand_score": 0.4, "finger_code": "1-1-1"}]).to_csv(fast_csv, index=False)

    merged = merge_multifidelity_results({"fast": fast_csv}, output_csv)

    assert list(merged["design_id"]) == ["a"]
    assert merged.loc[0, "best_available_score"] == 0.4
    assert "hand_score_fast" in merged.columns
