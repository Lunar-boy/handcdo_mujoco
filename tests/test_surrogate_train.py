from __future__ import annotations

import json

import pandas as pd
import pytest

from handcdo.design_space import DesignSpace
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
    assert metadata["target"] == "best_available_score"
    assert metadata["model_type"] == "random_forest"
    assert metadata["seed"] == 7
    assert metadata["n_rows_total"] == 6
    assert metadata["n_rows_used"] == 6
    assert "finger_code" in metadata["categorical_columns"]
    assert "thumb_angle" in metadata["numeric_columns"]
    assert metadata["feature_columns"] == [spec.name for spec in DesignSpace().specs]
    assert metadata["sklearn_model_class"] == "RandomForestRegressor"


def test_resolve_target_column_prefers_best_available_score():
    df = pd.DataFrame({"hand_score": [0.1], "best_available_score": [0.2]})

    assert resolve_target_column(df) == "best_available_score"


def test_explicit_hand_score_target_excludes_failed_rows(tmp_path):
    df = _training_frame()
    df.loc[0, "failed"] = True
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model", target="hand_score", min_rows=5)
    metadata = json.loads((tmp_path / "model" / "surrogate_metadata.json").read_text(encoding="utf-8"))

    assert metadata["target"] == "hand_score"
    assert metadata["n_rows_used"] == 5


def test_train_drops_nan_and_non_numeric_targets(tmp_path):
    df = _training_frame()
    df["best_available_score"] = df["best_available_score"].astype(object)
    df.loc[0, "best_available_score"] = None
    df.loc[1, "best_available_score"] = "not-a-score"
    results_csv = tmp_path / "results.csv"
    df.to_csv(results_csv, index=False)

    train_surrogate(results_csv, tmp_path / "model", min_rows=4)
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
