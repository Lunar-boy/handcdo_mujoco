from __future__ import annotations

import json

import pandas as pd
import pytest

from handcdo.design_space import DesignSpace
from handcdo.surrogate.features import compute_failure_aware_best_score
from handcdo.surrogate.train import resolve_target_column, train_surrogate


def _training_frame(n_rows: int = 6) -> pd.DataFrame:
    rows = []
    for seed in range(n_rows):
        design = DesignSpace().sample(seed=seed)
        rows.append(
            {
                "design_id": design.design_id,
                "hand_score": float(seed) / 10.0,
                "best_available_score": float(seed),
                "failed": False,
                **design.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def test_train_surrogate_writes_model_and_metadata(tmp_path):
    results_csv = tmp_path / "results.csv"
    _training_frame().to_csv(results_csv, index=False)

    model_path = train_surrogate(results_csv, tmp_path / "model", seed=7)

    assert model_path.exists()
    metadata_path = tmp_path / "model" / "surrogate_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["target"] == "failure_aware_best_available_score"
    assert metadata["target_requested"] == "best_available_score"
    assert metadata["target_used"] == "failure_aware_best_available_score"
    assert metadata["csv_best_available_score_present"] is True
    assert metadata["model_type"] == "random_forest"
    assert metadata["seed"] == 7
    assert metadata["n_rows_total"] == 6
    assert metadata["n_rows_used"] == 6
    assert "finger_code" in metadata["categorical_columns"]
    assert "thumb_angle" in metadata["numeric_columns"]
    assert metadata["feature_columns"] == [spec.name for spec in DesignSpace().specs]
    assert metadata["sklearn_model_class"] == "RandomForestRegressor"
    assert (tmp_path / "model" / "surrogate_diagnostics.json").exists()
    assert metadata["diagnostics_path"] == str(tmp_path / "model" / "surrogate_diagnostics.json")


def test_resolve_target_column_prefers_best_available_score():
    df = pd.DataFrame({"hand_score": [0.1], "best_available_score": [0.2]})

    assert resolve_target_column(df) == "best_available_score"


def test_resolve_target_prefers_failure_aware_best_for_multifidelity_scores():
    df = pd.DataFrame({"hand_score_high": [0.1], "failed_high": [False]})

    assert resolve_target_column(df) == "best_available_score"


def test_explicit_hand_score_target_excludes_failed_rows(tmp_path):
    df = _training_frame()
    df.loc[0, "failed"] = True
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model", target="hand_score", min_rows=5)
    metadata = json.loads((tmp_path / "model" / "surrogate_metadata.json").read_text(encoding="utf-8"))

    assert metadata["target"] == "hand_score"
    assert metadata["target_requested"] == "hand_score"
    assert metadata["target_used"] == "hand_score"
    assert metadata["n_rows_used"] == 5


def test_train_drops_nan_and_non_numeric_targets(tmp_path):
    df = _training_frame()
    df["hand_score"] = df["hand_score"].astype(object)
    df.loc[0, "hand_score"] = None
    df.loc[1, "hand_score"] = "not-a-score"
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model", target="hand_score", min_rows=4)
    metadata = json.loads((tmp_path / "model" / "surrogate_metadata.json").read_text(encoding="utf-8"))

    assert metadata["n_rows_used"] == 4


def test_train_missing_target_fails_clearly(tmp_path):
    results_csv = tmp_path / "results.csv"
    _training_frame().drop(columns=["hand_score", "best_available_score"]).to_csv(results_csv, index=False)

    with pytest.raises(ValueError, match="No supported target column"):
        train_surrogate(results_csv, tmp_path / "model")


def test_train_too_few_valid_rows_fails_clearly(tmp_path):
    results_csv = tmp_path / "results.csv"
    _training_frame(4).to_csv(results_csv, index=False)

    with pytest.raises(ValueError, match="Need at least 5 valid training rows"):
        train_surrogate(results_csv, tmp_path / "model")


def test_requested_missing_target_fails_clearly(tmp_path):
    results_csv = tmp_path / "results.csv"
    _training_frame().to_csv(results_csv, index=False)

    with pytest.raises(ValueError, match="Requested target column"):
        train_surrogate(results_csv, tmp_path / "model", target="missing_score")


def test_failure_aware_best_score_uses_highest_non_failed_fidelity():
    df = pd.DataFrame(
        {
            "hand_score_high": [0.0, 9.0, 0.0, 0.0],
            "failed_high": [True, False, "true", "1"],
            "hand_score_medium": [5.0, 4.0, 0.0, 0.0],
            "failed_medium": [False, False, "yes", 1],
            "hand_score_fast": [2.0, 1.0, 3.0, 0.0],
            "failed_fast": [False, False, 0, "y"],
        }
    )

    scores = compute_failure_aware_best_score(df)

    assert scores.tolist()[:3] == [5.0, 9.0, 3.0]
    assert pd.isna(scores.iloc[3])


def test_training_uses_failure_aware_best_when_csv_best_disagrees(tmp_path):
    design = DesignSpace().sample(seed=0)
    df = pd.DataFrame(
        [
            {
                "design_id": design.design_id,
                "best_available_score": 100.0,
                "hand_score_high": 0.0,
                "failed_high": True,
                "hand_score_medium": 7.0,
                "failed_medium": False,
                **design.to_dict(),
            }
        ]
    )
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model", target="best_available_score", min_rows=1)
    diagnostics = json.loads((tmp_path / "model" / "surrogate_diagnostics.json").read_text(encoding="utf-8"))

    assert diagnostics["target_mean"] == 7.0


def test_explicit_high_score_target_excludes_failed_high_rows(tmp_path):
    df = _training_frame()
    df["hand_score_high"] = range(len(df))
    df["failed_high"] = [True, False, False, False, False, False]
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model", target="hand_score_high", min_rows=5)
    metadata = json.loads((tmp_path / "model" / "surrogate_metadata.json").read_text(encoding="utf-8"))

    assert metadata["n_rows_used"] == 5


def test_diagnostics_small_dataset_disables_cv(tmp_path):
    results_csv = tmp_path / "results.csv"
    _training_frame(6).to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model")
    diagnostics = json.loads((tmp_path / "model" / "surrogate_diagnostics.json").read_text(encoding="utf-8"))

    for key in (
        "n_rows_used",
        "n_features_raw",
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "train_r2",
        "train_mae",
        "cv_enabled",
        "cv_folds",
        "cv_r2_mean",
        "cv_r2_std",
        "cv_mae_mean",
        "cv_mae_std",
        "diagnostic_warnings",
    ):
        assert key in diagnostics
    assert diagnostics["cv_enabled"] is False
    assert diagnostics["cv_r2_mean"] is None
    assert diagnostics["diagnostic_warnings"]


def test_diagnostics_larger_dataset_enables_cv(tmp_path):
    results_csv = tmp_path / "results.csv"
    _training_frame(12).to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model")
    diagnostics = json.loads((tmp_path / "model" / "surrogate_diagnostics.json").read_text(encoding="utf-8"))

    assert diagnostics["cv_enabled"] is True
    assert diagnostics["cv_folds"] == 5
    assert diagnostics["cv_mae_mean"] is not None


def test_diagnostics_constant_target_warns_without_crashing(tmp_path):
    df = _training_frame()
    df["hand_score"] = 1.0
    df["best_available_score"] = 1.0
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model")
    diagnostics = json.loads((tmp_path / "model" / "surrogate_diagnostics.json").read_text(encoding="utf-8"))

    assert diagnostics["train_r2"] is None
    assert any("variance" in warning.lower() for warning in diagnostics["diagnostic_warnings"])
