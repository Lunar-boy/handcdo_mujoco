from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from handcdo.utils import ensure_dir, write_json

from .features import (
    DEFAULT_TARGET_PRIORITY,
    design_feature_columns,
    load_design_space,
    prepare_feature_frame,
    split_feature_columns,
    valid_training_rows,
)


def resolve_target_column(df: pd.DataFrame, requested_target: str | None = None) -> str:
    if requested_target is not None:
        if requested_target not in df.columns:
            raise ValueError(f"Requested target column {requested_target!r} is missing from results CSV")
        return requested_target

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
    y_train = pd.to_numeric(train_df[target_column], errors="coerce")
    pipeline.fit(x_train, y_train)

    metadata = {
        "results_csv": str(results_path),
        "target": target_column,
        "model_type": model_type,
        "seed": seed,
        "n_rows_total": int(len(df)),
        "n_rows_used": int(len(train_df)),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "search_space": None if search_space is None else str(search_space),
        "sklearn_model_class": type(model).__name__,
    }

    model_path = output_path / "surrogate_model.joblib"
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to serialize sklearn surrogate models") from exc
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    write_json(output_path / "surrogate_metadata.json", metadata)
    return model_path


def _make_model(model_type: str, seed: int) -> RandomForestRegressor | ExtraTreesRegressor:
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    if model_type == "extra_trees":
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(f"Unsupported model_type {model_type!r}; expected 'random_forest' or 'extra_trees'")
