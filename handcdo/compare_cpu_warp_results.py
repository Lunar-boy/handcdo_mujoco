from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .utils import write_json


CPU_REFERENCE_BACKENDS = {"mujoco", "mujoco_cpu"}
WARP_BACKEND = "mujoco_warp"
WARP_SCORE_SEMANTICS = "experimental_non_equivalent"


@dataclass(frozen=True)
class ResultRecord:
    design_id: str
    path: Path
    payload: dict[str, Any]
    score: float | None
    failed: bool
    tool_scores: dict[str, float]
    tool_failures: dict[str, int]


def compare_cpu_warp_results(
    cpu_results_dir: str | Path,
    warp_results_dir: str | Path,
    *,
    out: str | Path | None = None,
    tools: list[str] | str | None = None,
    top_k: int = 20,
    fail_on_missing: bool = False,
    fail_on_rank_drift: float | None = None,
    fail_on_score_diff: float | None = None,
) -> tuple[dict[str, Any], int]:
    selected_tools = _parse_tools(tools)
    warnings: list[str] = []
    cpu_dir = Path(cpu_results_dir)
    warp_dir = Path(warp_results_dir)

    cpu_records = _load_result_dir(cpu_dir, role="cpu", selected_tools=selected_tools, warnings=warnings)
    warp_records = _load_result_dir(warp_dir, role="warp", selected_tools=selected_tools, warnings=warnings)

    cpu_ids = set(cpu_records)
    warp_ids = set(warp_records)
    matched_ids = sorted(cpu_ids & warp_ids)
    missing_in_cpu = sorted(warp_ids - cpu_ids)
    missing_in_warp = sorted(cpu_ids - warp_ids)

    by_design = _compare_designs(cpu_records, warp_records, matched_ids, selected_tools)
    _attach_ranks(by_design)
    overall = _overall_metrics(by_design, top_k=top_k, warnings=warnings)
    by_tool = _tool_metrics(cpu_records, warp_records, matched_ids, selected_tools)

    summary = {
        "cpu_results_dir": str(cpu_dir),
        "warp_results_dir": str(warp_dir),
        "num_cpu_designs": len(cpu_records),
        "num_warp_designs": len(warp_records),
        "num_matched_designs": len(matched_ids),
        "missing_in_cpu": missing_in_cpu,
        "missing_in_warp": missing_in_warp,
        "score_semantics": {
            "cpu": "reference",
            "warp": WARP_SCORE_SEMANTICS,
        },
        "tools": selected_tools,
        "top_k": int(top_k),
        "overall": overall,
        "by_tool": by_tool,
        "by_design": by_design,
        "warnings": warnings,
    }

    for warning in warnings:
        print(f"warning: {warning}")

    exit_code = _exit_code(
        summary,
        fail_on_missing=fail_on_missing,
        fail_on_rank_drift=fail_on_rank_drift,
        fail_on_score_diff=fail_on_score_diff,
    )
    if out is not None:
        write_json(out, _json_clean(summary))
    return _json_clean(summary), exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CPU MuJoCo reference results against experimental MuJoCo Warp JSON results."
    )
    parser.add_argument("--cpu-results-dir", required=True)
    parser.add_argument("--warp-results-dir", required=True)
    parser.add_argument("--out", help="Optional JSON summary output path.")
    parser.add_argument("--tools", help="Optional comma-separated tool subset.")
    parser.add_argument("--top-k", type=_positive_int, default=20)
    parser.add_argument("--fail-on-missing", action="store_true", default=False)
    parser.add_argument(
        "--fail-on-rank-drift",
        type=float,
        help="Exit nonzero when max absolute rank displacement exceeds this threshold.",
    )
    parser.add_argument(
        "--fail-on-score-diff",
        type=float,
        help="Exit nonzero when max absolute score difference exceeds this threshold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, exit_code = compare_cpu_warp_results(
            args.cpu_results_dir,
            args.warp_results_dir,
            out=args.out,
            tools=args.tools,
            top_k=args.top_k,
            fail_on_missing=args.fail_on_missing,
            fail_on_rank_drift=args.fail_on_rank_drift,
            fail_on_score_diff=args.fail_on_score_diff,
        )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 1
    print(
        "Compared "
        f"{summary['num_matched_designs']} matched designs "
        f"({summary['num_cpu_designs']} CPU, {summary['num_warp_designs']} Warp)."
    )
    return exit_code


def _load_result_dir(
    results_dir: Path,
    *,
    role: str,
    selected_tools: list[str] | None,
    warnings: list[str],
) -> dict[str, ResultRecord]:
    if not results_dir.exists():
        warnings.append(f"{role} results directory does not exist: {results_dir}")
        return {}
    records: dict[str, ResultRecord] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not read {role} result JSON {path}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{role} result JSON {path} must contain an object")

        design_id = str(payload.get("design_id") or _design_id_from_path(path))
        if design_id in records:
            warnings.append(f"{role} results contain duplicate design_id {design_id}; using {path}")

        _validate_payload(role, path, payload, warnings)
        records[design_id] = ResultRecord(
            design_id=design_id,
            path=path,
            payload=payload,
            score=_json_float(payload.get("hand_score")),
            failed=bool(payload.get("failed", False)),
            tool_scores=_extract_tool_scores(payload, selected_tools),
            tool_failures=_extract_tool_failures(payload, selected_tools),
        )
    return records


def _validate_payload(role: str, path: Path, payload: dict[str, Any], warnings: list[str]) -> None:
    backend = payload.get("backend")
    score_semantics = payload.get("score_semantics")
    if role == "warp":
        if payload.get("experimental") is not True:
            warnings.append(f"Warp result {path} lacks experimental=true")
        if "score_semantics" not in payload:
            warnings.append(f"Warp result {path} lacks score_semantics")
        elif score_semantics == "intended_cpu_equivalent":
            warnings.append(f"Warp result {path} claims intended_cpu_equivalent")
        elif score_semantics != WARP_SCORE_SEMANTICS:
            warnings.append(f"Warp result {path} has unexpected score_semantics={score_semantics!r}")
        if backend != WARP_BACKEND:
            warnings.append(f"Warp result {path} has backend={backend!r}, expected {WARP_BACKEND!r}")
    else:
        if backend is not None and backend not in CPU_REFERENCE_BACKENDS:
            warnings.append(f"CPU result {path} appears to be non-CPU backend={backend!r}")
        if payload.get("experimental") is True:
            warnings.append(f"CPU result {path} is marked experimental")
        if score_semantics == WARP_SCORE_SEMANTICS:
            warnings.append(f"CPU result {path} uses Warp score_semantics")

    if "hand_score" not in payload:
        warnings.append(f"{role} result {path} lacks hand_score")
    if "tool_results" in payload and not isinstance(payload.get("tool_results"), list):
        warnings.append(f"{role} result {path} has non-list tool_results")


def _compare_designs(
    cpu_records: dict[str, ResultRecord],
    warp_records: dict[str, ResultRecord],
    matched_ids: list[str],
    selected_tools: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design_id in matched_ids:
        cpu = cpu_records[design_id]
        warp = warp_records[design_id]
        tool_names = _tool_names(cpu, warp, selected_tools)
        tool_diffs = {}
        missing_tools_cpu = []
        missing_tools_warp = []
        for tool in tool_names:
            cpu_score = cpu.tool_scores.get(tool)
            warp_score = warp.tool_scores.get(tool)
            if cpu_score is None:
                missing_tools_cpu.append(tool)
            if warp_score is None:
                missing_tools_warp.append(tool)
            if cpu_score is not None and warp_score is not None:
                signed = warp_score - cpu_score
                tool_diffs[tool] = {
                    "cpu_score": cpu_score,
                    "warp_score": warp_score,
                    "signed_score_diff": signed,
                    "abs_score_diff": abs(signed),
                    "cpu_failure_count": cpu.tool_failures.get(tool, 0),
                    "warp_failure_count": warp.tool_failures.get(tool, 0),
                    "failure_count_diff": warp.tool_failures.get(tool, 0) - cpu.tool_failures.get(tool, 0),
                }
        signed_diff = None if cpu.score is None or warp.score is None else warp.score - cpu.score
        rows.append(
            {
                "design_id": design_id,
                "cpu_score": cpu.score,
                "warp_score": warp.score,
                "signed_score_diff": signed_diff,
                "abs_score_diff": abs(signed_diff) if signed_diff is not None else None,
                "cpu_failed": cpu.failed,
                "warp_failed": warp.failed,
                "cpu_failure_count": sum(cpu.tool_failures.values()),
                "warp_failure_count": sum(warp.tool_failures.values()),
                "failure_count_diff": sum(warp.tool_failures.values()) - sum(cpu.tool_failures.values()),
                "missing_tools_cpu": sorted(missing_tools_cpu),
                "missing_tools_warp": sorted(missing_tools_warp),
                "missing_tool_count": len(missing_tools_cpu) + len(missing_tools_warp),
                "tool_diffs": tool_diffs,
                "cpu_rank": None,
                "warp_rank": None,
                "rank_displacement": None,
            }
        )
    return rows


def _attach_ranks(rows: list[dict[str, Any]]) -> None:
    cpu_rank = _rank_rows(rows, "cpu_score")
    warp_rank = _rank_rows(rows, "warp_score")
    for row in rows:
        design_id = row["design_id"]
        row["cpu_rank"] = cpu_rank.get(design_id)
        row["warp_rank"] = warp_rank.get(design_id)
        if row["cpu_rank"] is not None and row["warp_rank"] is not None:
            row["rank_displacement"] = row["warp_rank"] - row["cpu_rank"]


def _overall_metrics(rows: list[dict[str, Any]], *, top_k: int, warnings: list[str]) -> dict[str, Any]:
    diffs = [row["abs_score_diff"] for row in rows if row["abs_score_diff"] is not None]
    rank_pairs = [
        (float(row["cpu_rank"]), float(row["warp_rank"]))
        for row in rows
        if row["cpu_rank"] is not None and row["warp_rank"] is not None
    ]
    rank_displacements = [
        abs(float(row["rank_displacement"]))
        for row in rows
        if row["rank_displacement"] is not None
    ]
    if len(rank_pairs) < 3:
        warnings.append("Fewer than 3 matched ranked designs; rank correlation is not meaningful")
    return {
        "mean_abs_score_diff": _mean(diffs),
        "median_abs_score_diff": _median(diffs),
        "max_abs_score_diff": max(diffs) if diffs else None,
        "rank_spearman": _spearman(rank_pairs),
        "rank_kendall": _kendall(rank_pairs),
        "top_k_overlap": _top_k_overlap(rows, top_k),
        "top_k_cpu_recall_in_warp": _top_k_cpu_recall_in_warp(rows, top_k),
        "rank_sign_flip_count": _rank_sign_flip_count(rows),
        "mean_abs_rank_displacement": _mean(rank_displacements),
        "max_abs_rank_displacement": max(rank_displacements) if rank_displacements else None,
    }


def _tool_metrics(
    cpu_records: dict[str, ResultRecord],
    warp_records: dict[str, ResultRecord],
    matched_ids: list[str],
    selected_tools: list[str] | None,
) -> dict[str, Any]:
    tool_names: set[str] = set(selected_tools or [])
    for design_id in matched_ids:
        tool_names.update(cpu_records[design_id].tool_scores)
        tool_names.update(warp_records[design_id].tool_scores)

    metrics = {}
    for tool in sorted(tool_names):
        signed_diffs = []
        abs_diffs = []
        missing_in_cpu = 0
        missing_in_warp = 0
        cpu_failures = 0
        warp_failures = 0
        for design_id in matched_ids:
            cpu = cpu_records[design_id]
            warp = warp_records[design_id]
            cpu_score = cpu.tool_scores.get(tool)
            warp_score = warp.tool_scores.get(tool)
            cpu_failures += cpu.tool_failures.get(tool, 0)
            warp_failures += warp.tool_failures.get(tool, 0)
            if cpu_score is None:
                missing_in_cpu += 1
            if warp_score is None:
                missing_in_warp += 1
            if cpu_score is not None and warp_score is not None:
                signed = warp_score - cpu_score
                signed_diffs.append(signed)
                abs_diffs.append(abs(signed))
        metrics[tool] = {
            "num_matched_scores": len(abs_diffs),
            "mean_signed_score_diff": _mean(signed_diffs),
            "mean_abs_score_diff": _mean(abs_diffs),
            "median_abs_score_diff": _median(abs_diffs),
            "max_abs_score_diff": max(abs_diffs) if abs_diffs else None,
            "cpu_failure_count": cpu_failures,
            "warp_failure_count": warp_failures,
            "failure_count_diff": warp_failures - cpu_failures,
            "missing_in_cpu_count": missing_in_cpu,
            "missing_in_warp_count": missing_in_warp,
        }
    return metrics


def _exit_code(
    summary: dict[str, Any],
    *,
    fail_on_missing: bool,
    fail_on_rank_drift: float | None,
    fail_on_score_diff: float | None,
) -> int:
    failures = []
    if fail_on_missing and (summary["missing_in_cpu"] or summary["missing_in_warp"]):
        failures.append("missing designs exist in one result directory")
    max_rank_drift = summary["overall"].get("max_abs_rank_displacement")
    if fail_on_rank_drift is not None and max_rank_drift is not None and max_rank_drift > fail_on_rank_drift:
        failures.append(
            "max absolute rank displacement "
            f"{max_rank_drift} exceeds threshold {fail_on_rank_drift}"
        )
    max_score_diff = summary["overall"].get("max_abs_score_diff")
    if fail_on_score_diff is not None and max_score_diff is not None and max_score_diff > fail_on_score_diff:
        failures.append(f"max absolute score difference {max_score_diff} exceeds threshold {fail_on_score_diff}")
    for failure in failures:
        print(f"failure: {failure}")
    return 1 if failures else 0


def _extract_tool_scores(payload: dict[str, Any], selected_tools: list[str] | None) -> dict[str, float]:
    scores = {}
    allowed = set(selected_tools) if selected_tools is not None else None
    for tool_result in payload.get("tool_results") or []:
        if not isinstance(tool_result, dict):
            continue
        tool = tool_result.get("tool")
        if not tool or (allowed is not None and tool not in allowed):
            continue
        score = _json_float(tool_result.get("best_score"))
        if score is not None:
            scores[str(tool)] = score
    return scores


def _extract_tool_failures(payload: dict[str, Any], selected_tools: list[str] | None) -> dict[str, int]:
    failures = {}
    allowed = set(selected_tools) if selected_tools is not None else None
    for tool_result in payload.get("tool_results") or []:
        if not isinstance(tool_result, dict):
            continue
        tool = tool_result.get("tool")
        if not tool or (allowed is not None and tool not in allowed):
            continue
        failures[str(tool)] = _int_or_zero(tool_result.get("failure_count"))
    return failures


def _tool_names(cpu: ResultRecord, warp: ResultRecord, selected_tools: list[str] | None) -> list[str]:
    if selected_tools is not None:
        return selected_tools
    return sorted(set(cpu.tool_scores) | set(warp.tool_scores))


def _rank_rows(rows: list[dict[str, Any]], score_key: str) -> dict[str, int]:
    ranked = [
        row
        for row in rows
        if isinstance(row.get(score_key), (int, float)) and math.isfinite(float(row[score_key]))
    ]
    ranked.sort(key=lambda row: (-float(row[score_key]), str(row["design_id"])))
    return {str(row["design_id"]): index + 1 for index, row in enumerate(ranked)}


def _top_k_overlap(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any] | None:
    ranked_cpu = sorted(
        [row for row in rows if row["cpu_rank"] is not None],
        key=lambda row: (row["cpu_rank"], str(row["design_id"])),
    )
    ranked_warp = sorted(
        [row for row in rows if row["warp_rank"] is not None],
        key=lambda row: (row["warp_rank"], str(row["design_id"])),
    )
    effective_k = min(top_k, len(ranked_cpu), len(ranked_warp))
    if effective_k <= 0:
        return None
    cpu_top = {row["design_id"] for row in ranked_cpu[:effective_k]}
    warp_top = {row["design_id"] for row in ranked_warp[:effective_k]}
    overlap = sorted(cpu_top & warp_top)
    return {
        "k": top_k,
        "effective_k": effective_k,
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / effective_k,
        "design_ids": overlap,
    }


def _top_k_cpu_recall_in_warp(rows: list[dict[str, Any]], top_k: int) -> float | None:
    if top_k <= 0:
        return None
    ranked_cpu = sorted(
        [row for row in rows if row["cpu_rank"] is not None],
        key=lambda row: (row["cpu_rank"], str(row["design_id"])),
    )
    ranked_warp = sorted(
        [row for row in rows if row["warp_rank"] is not None],
        key=lambda row: (row["warp_rank"], str(row["design_id"])),
    )
    cpu_top = {row["design_id"] for row in ranked_cpu[:top_k]}
    warp_top = {row["design_id"] for row in ranked_warp[:top_k]}
    if not cpu_top:
        return None
    return len(cpu_top & warp_top) / len(cpu_top)


def _rank_sign_flip_count(rows: list[dict[str, Any]]) -> int:
    ranked = [
        row
        for row in rows
        if row["cpu_rank"] is not None and row["warp_rank"] is not None
    ]
    flips = 0
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            cpu_delta = ranked[i]["cpu_rank"] - ranked[j]["cpu_rank"]
            warp_delta = ranked[i]["warp_rank"] - ranked[j]["warp_rank"]
            if cpu_delta == 0 or warp_delta == 0:
                continue
            if cpu_delta * warp_delta < 0:
                flips += 1
    return flips


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return _pearson(xs, ys)


def _kendall(pairs: list[tuple[float, float]]) -> float | None:
    n = len(pairs)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x_delta = pairs[i][0] - pairs[j][0]
            y_delta = pairs[i][1] - pairs[j][1]
            product = x_delta * y_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    return numerator / math.sqrt(x_var * y_var)


def _parse_tools(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        tools = [tool.strip() for tool in value if tool.strip()]
    else:
        tools = [tool.strip() for tool in value.split(",") if tool.strip()]
    if not tools:
        raise ValueError("--tools must include at least one tool when provided")
    return tools


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _json_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _design_id_from_path(path: Path) -> str:
    name = path.name
    suffix = ".mujoco_warp.experimental.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value
