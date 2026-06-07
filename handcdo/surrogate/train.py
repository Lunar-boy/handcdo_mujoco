from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from handcdo.utils import ensure_dir, write_json

from .features import (
    DEFAULT_TARGET_PRIORITY,
    FAILURE_AWARE_BEST_TARGET,
    design_feature_columns,
    has_failure_aware_score_inputs,
    has_multifidelity_score_inputs,
    load_design_space,
    prepare_feature_frame,
    split_feature_columns,
    valid_training_rows,
)


def resolve_target_column(df: pd.DataFrame, requested_target: str | None = None) -> str:
    if requested_target is not None:
        if requested_target == "best_available_score" and has_failure_aware_score_inputs(df):
            return requested_target
        if requested_target not in df.columns:
            raise ValueError(f"Requested target column {requested_target!r} is missing from results CSV")
        return requested_target

    if "best_available_score" in df.columns or has_multifidelity_score_inputs(df):
        if has_failure_aware_score_inputs(df):
            return "best_available_score"
    for column in DEFAULT_TARGET_PRIORITY:
        if column in df.columns:
            return column
    raise ValueError(f"No supported target column found. Expected one of: {', '.join(DEFAULT_TARGET_PRIORITY)}")


def train_surrogate(
    results_csv: str | Path,
    output_dir: str | Path,
    search_space: str | Path | None = "configs/search_space.yaml",
    target: str | None = None,
    model_type: str = "random_forest",
    seed: int = 0,
    min_rows: int = 5,
) -> Path:
    results_path = Path(results_csv)
    output_path = ensure_dir(output_dir)
    df = pd.read_csv(results_path)
    space = load_design_space(search_space)
    target_column = resolve_target_column(df, target)
    train_df = valid_training_rows(df, target_column)
    target_used = FAILURE_AWARE_BEST_TARGET if target_column == "best_available_score" else target_column

    feature_columns = design_feature_columns(space, train_df)
    if not feature_columns:
        raise ValueError("No active design-space parameter columns were found in the training CSV")
    categorical_columns, numeric_columns = split_feature_columns(space, feature_columns)

    if len(train_df) < min_rows:
        raise ValueError(f"Need at least {min_rows} valid training rows, found {len(train_df)}")

    model = _make_model(model_type=model_type, seed=seed)
    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_columns,
                        ),
                        (
                            "numeric",
                            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                            numeric_columns,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", model),
        ]
    )

    x_train = prepare_feature_frame(train_df, feature_columns, numeric_columns)
    y_train = pd.to_numeric(train_df[target_used], errors="coerce")
    pipeline.fit(x_train, y_train)
    diagnostics_path = output_path / "surrogate_diagnostics.json"
    diagnostics = compute_surrogate_diagnostics(pipeline, x_train, y_train, seed=seed)

    metadata = {
        "results_csv": str(results_path),
        "target": target_used,
        "target_requested": target_column,
        "target_used": target_used,
        "csv_best_available_score_present": "best_available_score" in df.columns,
        "model_type": model_type,
        "seed": seed,
        "n_rows_total": int(len(df)),
        "n_rows_used": int(len(train_df)),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "search_space": None if search_space is None else str(search_space),
        "sklearn_model_class": type(model).__name__,
        "diagnostics_path": str(diagnostics_path),
    }

    model_path = output_path / "surrogate_model.joblib"
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to serialize sklearn surrogate models") from exc
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    write_json(output_path / "surrogate_metadata.json", metadata)
    write_json(diagnostics_path, diagnostics)
    return model_path


def compute_surrogate_diagnostics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    min_cv_rows: int = 10,
) -> dict[str, Any]:
    y = pd.to_numeric(y, errors="coerce").dropna()
    X = X.loc[y.index]
    warnings: list[str] = []
    predictions = pipeline.predict(X)
    target_std = float(y.std(ddof=0)) if len(y) else 0.0
    if target_std <= 1e-12:
        warnings.append("Target variance is zero or near-zero; R2 diagnostics are not meaningful.")
    train_r2 = _safe_r2(y, predictions)
    train_mae = float(mean_absolute_error(y, predictions)) if len(y) else None

    cv_enabled = False
    cv_folds: int | None = None
    cv_r2_scores: list[float] = []
    cv_mae_scores: list[float] = []
    if len(y) < min_cv_rows:
        warnings.append(f"Cross-validation disabled: need at least {min_cv_rows} rows, found {len(y)}.")
    elif target_std <= 1e-12:
        warnings.append("Cross-validation R2 may be uninformative because target variance is near zero.")
    else:
        cv_folds = min(5, len(y) // 2)
        if cv_folds >= 2:
            cv_enabled = True
            splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            for train_index, test_index in splitter.split(X):
                fold_model = clone(pipeline)
                fold_model.fit(X.iloc[train_index], y.iloc[train_index])
                fold_predictions = fold_model.predict(X.iloc[test_index])
                cv_r2_scores.append(_safe_r2(y.iloc[test_index], fold_predictions))
                cv_mae_scores.append(float(mean_absolute_error(y.iloc[test_index], fold_predictions)))
        else:
            warnings.append("Cross-validation disabled: folds would be too small.")

    return {
        "n_rows_used": int(len(y)),
        "n_features_raw": int(X.shape[1]),
        "target_mean": _float_or_none(y.mean()),
        "target_std": target_std,
        "target_min": _float_or_none(y.min()),
        "target_max": _float_or_none(y.max()),
        "train_r2": train_r2,
        "train_mae": train_mae,
        "cv_enabled": cv_enabled,
        "cv_folds": cv_folds if cv_enabled else None,
        "cv_r2_mean": _mean_or_none(cv_r2_scores),
        "cv_r2_std": _std_or_none(cv_r2_scores),
        "cv_mae_mean": _mean_or_none(cv_mae_scores),
        "cv_mae_std": _std_or_none(cv_mae_scores),
        "diagnostic_warnings": warnings,
    }


def _safe_r2(y_true: pd.Series, y_pred: Any) -> float | None:
    if len(y_true) < 2:
        return None
    if float(y_true.std(ddof=0)) <= 1e-12:
        return None
    return float(r2_score(y_true, y_pred))


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(pd.Series(values).mean())


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(pd.Series(values).std(ddof=0))


def _make_model(model_type: str, seed: int) -> RandomForestRegressor | ExtraTreesRegressor:
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    if model_type == "extra_trees":
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(f"Unsupported model_type {model_type!r}; expected 'random_forest' or 'extra_trees'")
