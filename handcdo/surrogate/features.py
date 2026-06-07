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

FIDELITY_SCORE_COLUMNS = (
    ("hand_score_high", "failed_high"),
    ("hand_score_medium", "failed_medium"),
    ("hand_score_fast", "failed_fast"),
    ("hand_score", "failed"),
)

FAILURE_AWARE_BEST_TARGET = "failure_aware_best_available_score"


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
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def has_failure_aware_score_inputs(df: pd.DataFrame) -> bool:
    return any(score_column in df.columns for score_column, _ in FIDELITY_SCORE_COLUMNS)


def has_multifidelity_score_inputs(df: pd.DataFrame) -> bool:
    return any(column in df.columns for column in ("hand_score_high", "hand_score_medium", "hand_score_fast"))


def compute_failure_aware_best_score(df: pd.DataFrame) -> pd.Series:
    if not has_failure_aware_score_inputs(df):
        raise ValueError("Cannot compute failure-aware best score: no hand_score* columns are available")

    result = pd.Series(pd.NA, index=df.index, dtype="Float64")
    unresolved = pd.Series(True, index=df.index)
    for score_column, failed_column in FIDELITY_SCORE_COLUMNS:
        if score_column not in df.columns:
            continue
        scores = pd.to_numeric(df[score_column], errors="coerce")
        failed = df[failed_column].map(is_truthy) if failed_column in df.columns else pd.Series(False, index=df.index)
        valid = unresolved & scores.notna() & ~failed
        result.loc[valid] = scores.loc[valid]
        unresolved = unresolved & ~valid
    return result.astype(float)


def valid_training_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    selected = df.copy()

    if target_column == "best_available_score":
        selected[FAILURE_AWARE_BEST_TARGET] = compute_failure_aware_best_score(selected)
        selected = selected.dropna(subset=[FAILURE_AWARE_BEST_TARGET])
    else:
        selected[target_column] = pd.to_numeric(selected[target_column], errors="coerce")
        selected = selected.dropna(subset=[target_column])

    failed_column = FAILED_COLUMN_BY_TARGET.get(target_column)
    if target_column != "best_available_score" and failed_column and failed_column in selected.columns:
        selected = selected[~selected[failed_column].map(is_truthy)]

    if selected.empty:
        raise ValueError(f"No valid training rows remain for target column {target_column!r}")
    return selected
