from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .design_space import HandDesign
from .geometry_config import GeometryConfig
from .mujoco_eval import EvaluationConfig
from .optimize_hand import evaluate_design
from .utils import ensure_dir, read_json, read_yaml, write_json


METADATA_FIELDS = ("fidelity", "backend", "config_path", "n_grasp_trials", "sampler", "seed")
MERGED_METADATA_FIELDS = ("failed", "error", "backend", "config_path", "n_grasp_trials", "sampler", "seed")
BASE_COLUMNS = {"design_id", "hand_score", *MERGED_METADATA_FIELDS, "fidelity"}
KNOWN_FIDELITY_PRIORITY = ("high", "medium", "fast")


def select_top_designs(
    input_csv: str | Path,
    top_k: int,
    output_design_ids: str | Path,
    score_column: str = "hand_score",
    include_failed: bool = False,
) -> list[str]:
    if top_k < 0:
        raise ValueError(f"top_k={top_k!r} must be >= 0")

    df = pd.read_csv(input_csv)
    missing = [column for column in ("design_id", score_column) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    selected = df.copy()
    if not include_failed and "failed" in selected.columns:
        selected = selected[~selected["failed"].map(_is_truthy)]

    selected[score_column] = pd.to_numeric(selected[score_column], errors="coerce")
    selected = selected.dropna(subset=[score_column])
    selected["design_id"] = selected["design_id"].astype(str)
    selected = selected.sort_values([score_column, "design_id"], ascending=[False, True], kind="mergesort")

    design_ids = selected["design_id"].head(top_k).tolist()
    output_path = Path(output_design_ids)
    ensure_dir(output_path.parent)
    output_path.write_text("".join(f"{design_id}\n" for design_id in design_ids), encoding="utf-8")
    return design_ids


def load_design_ids(path: str | Path) -> list[str]:
    design_ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    design_ids = [design_id for design_id in design_ids if design_id]
    seen: set[str] = set()
    duplicates: list[str] = []
    for design_id in design_ids:
        if design_id in seen and design_id not in duplicates:
            duplicates.append(design_id)
        seen.add(design_id)
    if duplicates:
        raise ValueError(f"Duplicate design id(s) in {path}: {', '.join(duplicates)}")
    return design_ids


def reevaluate_designs(
    *,
    design_dir: str | Path,
    design_ids: list[str],
    results_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    fidelity: str,
    tools: list[str],
    backend: str = "mujoco_cpu",
    seed: int = 0,
    n_grasp_trials: int | None = None,
    sampler: str | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> list[dict[str, Any]]:
    design_root = Path(design_dir)
    results_root = ensure_dir(results_dir)
    output_root = ensure_dir(output_dir)
    config_data = read_yaml(config_path)
    eval_config = EvaluationConfig.from_dict(config_data)
    geometry_config = GeometryConfig.from_dict(config_data)
    grasp_config = config_data.get("grasp", {})
    resolved_n_grasp_trials = int(n_grasp_trials if n_grasp_trials is not None else grasp_config.get("n_trials", 4))
    resolved_sampler = str(sampler if sampler is not None else grasp_config.get("sampler", "tpe"))

    payloads: list[dict[str, Any]] = []
    for offset, design_id in enumerate(design_ids):
        design_json = design_root / design_id / "design.json"
        if not design_json.exists():
            raise FileNotFoundError(f"Missing design JSON for design id {design_id!r}: {design_json}")

        design = HandDesign.from_json(design_json)
        if design.design_id != design_id:
            raise ValueError(
                f"Design id {design_id!r} does not match loaded design id {design.design_id!r} from {design_json}"
            )

        payload = evaluate_design(
            design,
            tools=tools,
            n_grasp_trials=resolved_n_grasp_trials,
            output_dir=output_root,
            result_dir=results_root,
            seed=seed + offset,
            config=eval_config,
            geometry_config=geometry_config,
            backend_name=backend,
            sampler=resolved_sampler,
            tool_assets_dir=tool_assets_dir,
        )
        if not isinstance(payload, dict):
            payload = read_json(payload)
        payload.update(
            {
                "fidelity": fidelity,
                "backend": backend,
                "config_path": str(config_path),
                "n_grasp_trials": resolved_n_grasp_trials,
                "sampler": resolved_sampler,
                "seed": seed + offset,
            }
        )
        write_json(results_root / f"{design.design_id}.json", payload)
        payloads.append(payload)
    return payloads


def merge_multifidelity_results(inputs: Mapping[str, str | Path], output_csv: str | Path) -> pd.DataFrame:
    if not inputs:
        raise ValueError("At least one input CSV is required")

    frames: dict[str, pd.DataFrame] = {}
    for fidelity, path in sorted(inputs.items()):
        df = pd.read_csv(path)
        if "design_id" not in df.columns:
            raise ValueError(f"Missing required column design_id in {path}")
        frames[str(fidelity)] = df.copy()

    all_design_ids = sorted({str(design_id) for df in frames.values() for design_id in df["design_id"].astype(str)})
    merged = pd.DataFrame({"design_id": all_design_ids})

    for fidelity in sorted(frames):
        df = frames[fidelity].copy()
        df["design_id"] = df["design_id"].astype(str)
        renamed: dict[str, str] = {}
        if "hand_score" in df.columns:
            renamed["hand_score"] = f"hand_score_{fidelity}"
        for field in MERGED_METADATA_FIELDS:
            if field in df.columns:
                renamed[field] = f"{field}_{fidelity}"
        for column in df.columns:
            if column not in BASE_COLUMNS and column.endswith("_best_score"):
                renamed[column] = f"{column}_{fidelity}"
        metric_df = df[["design_id", *renamed]].rename(columns=renamed)
        merged = merged.merge(metric_df, on="design_id", how="left")

    parameter_columns = sorted(
        {
            column
            for df in frames.values()
            for column in df.columns
            if column not in BASE_COLUMNS and not column.endswith("_best_score")
        }
    )
    for column in parameter_columns:
        merged[column] = pd.NA
        for fidelity in _priority_order(frames):
            df = frames[fidelity]
            if column not in df.columns:
                continue
            series = df.assign(design_id=df["design_id"].astype(str)).set_index("design_id")[column]
            merged[column] = merged[column].combine_first(merged["design_id"].map(series))

    merged["best_available_score"] = pd.NA
    for fidelity in _priority_order(frames):
        column = f"hand_score_{fidelity}"
        if column in merged.columns:
            merged["best_available_score"] = merged["best_available_score"].combine_first(
                pd.to_numeric(merged[column], errors="coerce")
            )

    ordered_columns = _ordered_columns(merged.columns)
    merged = merged[ordered_columns]
    ensure_dir(Path(output_csv).parent)
    merged.to_csv(output_csv, index=False)
    return merged


def _is_truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _priority_order(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    known = [fidelity for fidelity in KNOWN_FIDELITY_PRIORITY if fidelity in frames]
    others = sorted(fidelity for fidelity in frames if fidelity not in KNOWN_FIDELITY_PRIORITY)
    return known + others


def _ordered_columns(columns: Any) -> list[str]:
    columns = list(columns)
    first = [column for column in ("design_id", "best_available_score") if column in columns]
    scores = sorted(column for column in columns if column.startswith("hand_score_"))
    metadata = sorted(column for column in columns if any(column.startswith(f"{field}_") for field in MERGED_METADATA_FIELDS))
    remaining = sorted(column for column in columns if column not in set(first + scores + metadata))
    return first + scores + metadata + remaining
