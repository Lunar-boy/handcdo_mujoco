from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from handcdo.design_space import DesignSpace


DEFAULT_TARGET_PRIORITY = (
    "best_available_score",
    "hand_score_high",
    "hand_score_medium",
    "hand_score_fast",
    "hand_score",
)

FAILED_COLUMN_BY_TARGET = {
    "hand_score": "failed",
    "hand_score_fast": "failed_fast",
    "hand_score_medium": "failed_medium",
    "hand_score_high": "failed_high",
}


def load_design_space(search_space: str | Path | None) -> DesignSpace:
    if search_space is None:
        return DesignSpace()
    return DesignSpace.from_yaml(search_space)


def design_feature_columns(space: DesignSpace, df: pd.DataFrame) -> list[str]:
    return [spec.name for spec in space.specs if spec.name in df.columns]


def split_feature_columns(space: DesignSpace, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [name for name in feature_columns if space.by_name[name].kind == "categorical"]
    numeric = [name for name in feature_columns if space.by_name[name].kind in {"int", "float"}]
    return categorical, numeric


def prepare_feature_frame(df: pd.DataFrame, feature_columns: list[str], numeric_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {', '.join(missing)}")
    frame = df[feature_columns].copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def is_truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def valid_training_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    selected = df.copy()
    selected[target_column] = pd.to_numeric(selected[target_column], errors="coerce")
    selected = selected.dropna(subset=[target_column])

    failed_column = FAILED_COLUMN_BY_TARGET.get(target_column)
    if failed_column and failed_column in selected.columns:
        selected = selected[~selected[failed_column].map(is_truthy)]
    elif target_column == "best_available_score":
        failed_columns = [column for column in ("failed_high", "failed_medium", "failed_fast", "failed") if column in selected.columns]
        if failed_columns:
            failed_all = selected[failed_columns].apply(lambda row: all(is_truthy(value) for value in row), axis=1)
            selected = selected[~failed_all]

    if selected.empty:
        raise ValueError(f"No valid training rows remain for target column {target_column!r}")
    return selected
