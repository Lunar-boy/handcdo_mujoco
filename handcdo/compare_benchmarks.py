from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


def compare_benchmarks(
    left: str | Path,
    right: str | Path,
    output_dir: str | Path,
    *,
    score_column: str = "hand_score",
    top_k: list[int] | str = "5,10",
    fail_on_regression: bool = False,
    max_mean_drop: float | None = None,
    min_spearman: float | None = None,
) -> dict[str, Any]:
    import pandas as pd

    output_dir = ensure_dir(output_dir)
    left_df = pd.read_csv(left)
    right_df = pd.read_csv(right)
    _require_columns(left_df, ["design_id", score_column], "left")
    _require_columns(right_df, ["design_id", score_column], "right")

    left_ids = set(left_df["design_id"].astype(str))
    right_ids = set(right_df["design_id"].astype(str))
    only_left = sorted(left_ids - right_ids)
    only_right = sorted(right_ids - left_ids)

    left_prepared = _select_columns(left_df, score_column)
    right_prepared = _select_columns(right_df, score_column)
    joined = left_prepared.merge(right_prepared, on="design_id", how="inner", suffixes=("_left", "_right"))
    if joined.empty:
        raise ValueError("No common design_id values between benchmark CSV files")

    left_score = f"{score_column}_left"
    right_score = f"{score_column}_right"
    joined["delta"] = joined[right_score] - joined[left_score]

    tool_metrics = {}
    for column in sorted(_shared_tool_columns(left_df, right_df)):
        left_column = f"{column}_left"
        right_column = f"{column}_right"
        delta_column = f"{column}_delta"
        joined[delta_column] = joined[right_column] - joined[left_column]
        tool_metrics[column] = _score_summary(joined[left_column], joined[right_column], joined[delta_column])

    spearman, spearman_warning = _spearman(joined[left_score], joined[right_score])
    pearson = joined[left_score].corr(joined[right_score], method="pearson") if len(joined) > 1 else None
    top_k_values = _parse_top_k(top_k)
    top_k_summary = _top_k_summary(left_df, right_df, score_column, top_k_values, set(joined["design_id"].astype(str)))

    summary = {
        "score_column": score_column,
        "n_left": int(len(left_df)),
        "n_right": int(len(right_df)),
        "n_common": int(len(joined)),
        "only_left_count": len(only_left),
        "only_right_count": len(only_right),
        "only_left": only_left,
        "only_right": only_right,
        "scores": _score_summary(joined[left_score], joined[right_score], joined["delta"]),
        "top_k": top_k_summary,
        "spearman": _json_float(spearman),
        "pearson": _json_float(pearson),
        "warnings": [spearman_warning] if spearman_warning else [],
        "tool_scores": tool_metrics,
    }

    joined.to_csv(output_dir / "joined_scores.csv", index=False)
    write_json(output_dir / "comparison_summary.json", _json_clean(summary))

    failures = _regression_failures(summary, fail_on_regression, max_mean_drop, min_spearman)
    if failures:
        raise RuntimeError("; ".join(failures))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two handcdo benchmark CSV files.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-column", default="hand_score")
    parser.add_argument("--top-k", default="5,10")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--max-mean-drop", type=float)
    parser.add_argument("--min-spearman", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = compare_benchmarks(
            args.left,
            args.right,
            args.output_dir,
            score_column=args.score_column,
            top_k=args.top_k,
            fail_on_regression=args.fail_on_regression,
            max_mean_drop=args.max_mean_drop,
            min_spearman=args.min_spearman,
        )
    except RuntimeError as exc:
        print(f"Regression check failed: {exc}")
        raise SystemExit(1) from exc
    print(f"Compared {summary['n_common']} common designs")


def _require_columns(df: Any, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required column: {missing[0]}")


def _select_columns(df: Any, score_column: str) -> Any:
    columns = ["design_id", score_column]
    columns.extend(sorted(column for column in df.columns if column.endswith("_best_score") and column not in columns))
    for column in ("failed", "error"):
        if column in df.columns:
            columns.append(column)
    return df[columns].copy()


def _shared_tool_columns(left_df: Any, right_df: Any) -> set[str]:
    left = {column for column in left_df.columns if column.endswith("_best_score")}
    right = {column for column in right_df.columns if column.endswith("_best_score")}
    return left & right


def _score_summary(left: Any, right: Any, delta: Any) -> dict[str, float | None]:
    return {
        "mean_left": _json_float(left.mean()),
        "mean_right": _json_float(right.mean()),
        "median_left": _json_float(left.median()),
        "median_right": _json_float(right.median()),
        "mean_delta": _json_float(delta.mean()),
        "median_delta": _json_float(delta.median()),
        "min_delta": _json_float(delta.min()),
        "max_delta": _json_float(delta.max()),
    }


def _top_k_summary(left_df: Any, right_df: Any, score_column: str, top_k: list[int], common_ids: set[str]) -> dict[str, Any]:
    left_common = left_df[left_df["design_id"].astype(str).isin(common_ids)]
    right_common = right_df[right_df["design_id"].astype(str).isin(common_ids)]
    n_common = len(common_ids)
    summary = {}
    for k in top_k:
        effective_k = min(k, n_common)
        left_top = set(left_common.sort_values(score_column, ascending=False).head(effective_k)["design_id"].astype(str))
        right_top = set(right_common.sort_values(score_column, ascending=False).head(effective_k)["design_id"].astype(str))
        overlap = left_top & right_top
        summary[str(k)] = {
            "effective_k": effective_k,
            "overlap_count": len(overlap),
            "overlap_ratio": (len(overlap) / effective_k) if effective_k else None,
        }
    return summary


def _spearman(left: Any, right: Any) -> tuple[float | None, str | None]:
    try:
        stats = importlib.import_module("scipy.stats")
    except Exception:
        return None, "scipy is unavailable; spearman rank correlation was not computed"
    result = stats.spearmanr(left, right)
    statistic = getattr(result, "statistic", result[0])
    return _json_float(statistic), None


def _parse_top_k(value: list[int] | str) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _regression_failures(
    summary: dict[str, Any],
    fail_on_regression: bool,
    max_mean_drop: float | None,
    min_spearman: float | None,
) -> list[str]:
    if not fail_on_regression:
        return []
    failures = []
    mean_delta = summary["scores"]["mean_delta"]
    spearman = summary.get("spearman")
    if max_mean_drop is not None and mean_delta is not None and mean_delta < -max_mean_drop:
        failures.append(f"mean_delta {mean_delta} is below allowed drop {-max_mean_drop}")
    if min_spearman is not None and spearman is not None and spearman < min_spearman:
        failures.append(f"spearman {spearman} is below minimum {min_spearman}")
    return failures


def _json_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        return _json_float(value)
    return value


if __name__ == "__main__":
    main()
