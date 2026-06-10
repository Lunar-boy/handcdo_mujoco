#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handcdo.batch_eval import (
    evaluate_fixed_grasps_batched,
    sample_fixed_random_grasps,
    summarize_tool_batch_results,
)
from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.mujoco_eval import EvaluationConfig
from handcdo.utils import ensure_dir, read_yaml, setup_logging, write_json
from handcdo.warp_utils import availability_payload, check_warp_available


BACKEND = "mujoco_warp"
SCORE_SEMANTICS = "experimental_non_equivalent"
INSTALL_HINT = 'python3 -m pip install -e ".[warp]"'
WARP_RESULT_SUFFIX = ".mujoco_warp.experimental.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _optional_positive_int(value: str) -> int:
    return _positive_int(value)


def _parse_tools(value: str) -> list[str]:
    tools = [tool.strip() for tool in value.split(",") if tool.strip()]
    if not tools:
        raise argparse.ArgumentTypeError("at least one tool is required")
    return tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental MuJoCo Warp fixed random-grasp batch evaluator."
    )
    parser.add_argument("--design-dir", default="outputs/designs")
    parser.add_argument("--design-ids", default=None, help="Text file with one design id per line.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", default="configs/eval_fast.yaml")
    parser.add_argument("--tools", type=_parse_tools, default=_parse_tools("hammer,spoon,knife"))
    parser.add_argument("--n-grasp-trials", type=_nonnegative_int, default=64)
    parser.add_argument("--sampler", choices=("random",), default="random")
    parser.add_argument("--nworld", type=_positive_int, default=64)
    parser.add_argument("--nconmax", type=_positive_int, default=64)
    parser.add_argument("--naconmax", type=_optional_positive_int, default=None)
    parser.add_argument("--njmax", type=_positive_int, default=128)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=0)
    parser.add_argument("--capture-graph", action="store_true", default=False)
    parser.add_argument("--readback-interval", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-designs", type=_positive_int, default=None)
    parser.add_argument("--require-warp", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument(
        "--allow-mixed-backend-dir",
        action="store_true",
        default=False,
        help=(
            "Allow writing experimental MuJoCo Warp JSON into a results directory "
            "that appears to contain CPU result JSON files."
        ),
    )
    return parser


def collect_design_files(
    design_dir: str | Path,
    design_ids_path: str | Path | None = None,
    max_designs: int | None = None,
) -> list[Path]:
    root = Path(design_dir)
    if design_ids_path is None:
        design_files = sorted(root.glob("*/design.json"))
    else:
        design_ids = [
            line.strip()
            for line in Path(design_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        design_files = [root / design_id / "design.json" for design_id in design_ids]

    if max_designs is not None:
        design_files = design_files[:max_designs]
    return design_files


def result_path(results_dir: str | Path, design_id: str) -> Path:
    return Path(results_dir) / f"{design_id}{WARP_RESULT_SUFFIX}"


def _find_cpu_style_result_jsons(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.glob("*.json") if not path.name.endswith(WARP_RESULT_SUFFIX))


def _validate_results_dir_for_warp(results_dir: Path, allow_mixed_backend_dir: bool) -> None:
    cpu_like_jsons = _find_cpu_style_result_jsons(results_dir)
    if cpu_like_jsons and not allow_mixed_backend_dir:
        examples = ", ".join(str(path) for path in cpu_like_jsons[:5])
        raise FileExistsError(
            "Refusing to write experimental MuJoCo Warp results into a directory "
            "that appears to contain CPU result JSON files. Experimental MuJoCo "
            "Warp results are intentionally separated because they are not "
            "CPU-equivalent and must not be mixed with CPU/multifidelity result "
            "pools by accident. Pass --allow-mixed-backend-dir to override. "
            f"Examples: {examples}"
        )


def _empty_metadata(args: argparse.Namespace, *, num_grasps: int, failure_count: int = 0) -> dict[str, Any]:
    return {
        "nworld": args.nworld,
        "nconmax": args.nconmax,
        "naconmax": args.naconmax,
        "njmax": args.njmax,
        "warmup_steps": args.warmup_steps,
        "capture_graph": args.capture_graph,
        "batch_size": args.nworld,
        "readback_interval": args.readback_interval,
        "true_batched_scoring": False,
        "per_world_state_init": False,
        "wrench_directions": 12,
        "include_in_multifidelity": False,
        "scene_build_ok": False,
        "capability_probe_ok": False,
        "warmup_completed": False,
        "warmup_requested_steps": args.warmup_steps,
        "warmup_executed_steps": 0,
        "warmup_seconds": 0.0,
        "warmup_reason": "disabled" if args.warmup_steps == 0 else None,
        "capture_graph_requested": args.capture_graph,
        "capture_graph_enabled": False,
        "capture_graph_reason": "disabled" if not args.capture_graph else None,
        "capture_graph_sections": [],
        "capture_graph_replay_count": 0,
        "completed_chunks": 0,
        "failed_chunks": 0,
        "chunk_reset_strategy": "unknown",
        "chunk_reset_count": 0,
        "inactive_worlds_zeroed": False,
        "sync_strategy": "phase_boundary_and_readback_interval",
        "sync_count": None,
        "host_readback_count": None,
        "readback_semantics": "per-step threshold detection",
        "num_grasps": num_grasps,
        "num_chunks": math.ceil(num_grasps / args.nworld) if num_grasps else 0,
        "seconds_total": 0.0,
        "grasps_per_second": None,
        "world_steps_per_second": None,
        "failure_count": failure_count,
        "sequential_fallback": False,
        "mjcf_rewrites": [],
    }


def _normalize_backend_metadata(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = _empty_metadata(
        args,
        num_grasps=int(metadata.get("num_grasps", 0)),
        failure_count=int(metadata.get("failure_count", 0)),
    )
    normalized.update(
        {
            "nworld": metadata.get("nworld", args.nworld),
            "nconmax": metadata.get("nconmax", args.nconmax),
            "naconmax": metadata.get("naconmax", args.naconmax),
            "njmax": metadata.get("njmax", args.njmax),
            "num_chunks": metadata.get("num_chunks", normalized["num_chunks"]),
            "seconds_total": float(metadata.get("seconds_total", 0.0)),
            "grasps_per_second": metadata.get("grasps_per_second"),
            "world_steps_per_second": metadata.get("world_steps_per_second"),
            "sequential_fallback": bool(metadata.get("sequential_fallback", False)),
            "mjcf_rewrites": metadata.get("mjcf_rewrites", []),
            "true_batched_scoring": bool(metadata.get("true_batched_scoring", False)),
            "per_world_state_init": bool(metadata.get("per_world_state_init", False)),
            "wrench_directions": int(metadata.get("wrench_directions", 12)),
            "include_in_multifidelity": bool(metadata.get("include_in_multifidelity", False)),
            "readback_interval": int(metadata.get("readback_interval", args.readback_interval)),
        }
    )
    for key in (
        "scene_build_ok",
        "capability_probe_ok",
        "warmup_completed",
        "warmup_requested_steps",
        "warmup_executed_steps",
        "warmup_seconds",
        "warmup_reason",
        "capture_graph_requested",
        "capture_graph_enabled",
        "capture_graph_reason",
        "capture_graph_sections",
        "capture_graph_replay_count",
        "completed_chunks",
        "failed_chunks",
        "chunk_reset_strategy",
        "chunk_reset_count",
        "inactive_worlds_zeroed",
        "sync_strategy",
        "sync_count",
        "host_readback_count",
        "readback_semantics",
    ):
        if key in metadata:
            normalized[key] = metadata[key]
    for key in ("warp_capabilities", "failure_reason"):
        if key in metadata:
            normalized[key] = metadata[key]
    return normalized


def _aggregate_tool_metadata(args: argparse.Namespace, tool_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    num_grasps = sum(int(item.get("num_grasps", 0)) for item in tool_metadata)
    failure_count = sum(int(item.get("failure_count", 0)) for item in tool_metadata)
    seconds_total = sum(float(item.get("seconds_total", 0.0)) for item in tool_metadata)
    metadata = _empty_metadata(args, num_grasps=num_grasps, failure_count=failure_count)
    metadata["num_chunks"] = sum(int(item.get("num_chunks", 0)) for item in tool_metadata)
    metadata["seconds_total"] = seconds_total
    metadata["grasps_per_second"] = num_grasps / seconds_total if seconds_total > 0 else None
    metadata["sequential_fallback"] = any(bool(item.get("sequential_fallback", False)) for item in tool_metadata)
    metadata["true_batched_scoring"] = any(bool(item.get("true_batched_scoring", False)) for item in tool_metadata)
    metadata["per_world_state_init"] = any(bool(item.get("per_world_state_init", False)) for item in tool_metadata)
    metadata["wrench_directions"] = 12
    metadata["include_in_multifidelity"] = False
    metadata["scene_build_ok"] = all(bool(item.get("scene_build_ok", False)) for item in tool_metadata) if tool_metadata else False
    metadata["capability_probe_ok"] = all(bool(item.get("capability_probe_ok", False)) for item in tool_metadata) if tool_metadata else False
    metadata["warmup_completed"] = all(bool(item.get("warmup_completed", False)) for item in tool_metadata) if tool_metadata else False
    metadata["warmup_executed_steps"] = sum(int(item.get("warmup_executed_steps", 0)) for item in tool_metadata)
    metadata["warmup_seconds"] = sum(float(item.get("warmup_seconds", 0.0)) for item in tool_metadata)
    metadata["capture_graph_requested"] = any(bool(item.get("capture_graph_requested", False)) for item in tool_metadata)
    metadata["capture_graph_enabled"] = any(bool(item.get("capture_graph_enabled", False)) for item in tool_metadata)
    metadata["completed_chunks"] = sum(int(item.get("completed_chunks", 0)) for item in tool_metadata)
    metadata["failed_chunks"] = sum(int(item.get("failed_chunks", 0)) for item in tool_metadata)
    metadata["chunk_reset_count"] = sum(int(item.get("chunk_reset_count", 0)) for item in tool_metadata)
    metadata["inactive_worlds_zeroed"] = all(bool(item.get("inactive_worlds_zeroed", False)) for item in tool_metadata) if tool_metadata else False
    metadata["sync_count"] = sum(int(item.get("sync_count") or 0) for item in tool_metadata)
    metadata["host_readback_count"] = sum(int(item.get("host_readback_count") or 0) for item in tool_metadata)
    metadata["mjcf_rewrites"] = [
        rewrite for item in tool_metadata for rewrite in item.get("mjcf_rewrites", [])
    ]
    for key in ("warp_capabilities", "failure_reason", "capture_graph_reason", "chunk_reset_strategy", "readback_semantics"):
        values = [item[key] for item in tool_metadata if key in item]
        if values:
            metadata[key] = values[-1]
    return metadata


def _skipped_payload(
    design: HandDesign,
    args: argparse.Namespace,
    error: str,
    availability: dict[str, Any],
) -> dict[str, Any]:
    num_grasps = args.n_grasp_trials * len(args.tools)
    return {
        "design_id": design.design_id,
        "parameters": design.to_dict(),
        "hand_score": 0.0,
        "tool_results": [],
        "failed": True,
        "error": error,
        "backend": BACKEND,
        "experimental": True,
        "include_in_multifidelity": False,
        "score_semantics": SCORE_SEMANTICS,
        "warp_metadata": _empty_metadata(args, num_grasps=num_grasps, failure_count=num_grasps),
        "warp_availability": availability,
    }


def evaluate_design_warp(
    design: HandDesign,
    args: argparse.Namespace,
    eval_config: EvaluationConfig,
    geometry_config: GeometryConfig,
) -> dict[str, Any]:
    availability = check_warp_available()
    availability_json = availability_payload(availability)
    if not availability.available:
        reason = availability.reason or "mujoco_warp is unavailable"
        message = (
            "MuJoCo Warp is unavailable. Install the optional dependency with "
            f'{INSTALL_HINT}. Availability check failed: {reason}'
        )
        if args.require_warp:
            raise RuntimeError(message)
        return _skipped_payload(design, args, message, availability_json)

    from handcdo.backends.mujoco_warp import MujocoWarpBackend

    backend = MujocoWarpBackend(
        nworld=args.nworld,
        nconmax=args.nconmax,
        naconmax=args.naconmax,
        njmax=args.njmax,
        warmup_steps=args.warmup_steps,
        capture_graph=args.capture_graph,
        allow_sequential_fallback=False,
        readback_interval=args.readback_interval,
    )

    tool_results: list[dict[str, Any]] = []
    tool_metadata: list[dict[str, Any]] = []
    start = time.perf_counter()
    for tool_name in args.tools:
        grasps = sample_fixed_random_grasps(
            args.n_grasp_trials,
            seed=args.seed,
            design_id=design.design_id,
            tool_name=tool_name,
        )
        evaluations = evaluate_fixed_grasps_batched(
            backend,
            design,
            tool_name,
            grasps,
            eval_config,
            geometry_config=geometry_config,
        )
        summary = summarize_tool_batch_results(tool_name, grasps, evaluations)
        metadata = _normalize_backend_metadata(args, backend.last_batch_metadata)
        summary["warp_metadata"] = metadata
        tool_results.append(summary)
        tool_metadata.append(metadata)
        if args.fail_fast and any(evaluation.failed for evaluation in evaluations):
            break

    successful_tool_scores = [
        float(result["best_score"])
        for result in tool_results
        if result.get("failure_count", 0) < args.n_grasp_trials
    ]
    aggregate_metadata = _aggregate_tool_metadata(args, tool_metadata)
    aggregate_metadata["seconds_total"] = time.perf_counter() - start
    if aggregate_metadata["seconds_total"] > 0 and aggregate_metadata["num_grasps"] > 0:
        aggregate_metadata["grasps_per_second"] = (
            aggregate_metadata["num_grasps"] / aggregate_metadata["seconds_total"]
        )

    failed = not successful_tool_scores or any(result.get("failure_count", 0) for result in tool_results)
    payload: dict[str, Any] = {
        "design_id": design.design_id,
        "parameters": design.to_dict(),
        "hand_score": sum(successful_tool_scores) / len(successful_tool_scores)
        if successful_tool_scores
        else 0.0,
        "tool_results": tool_results,
        "failed": failed,
        "backend": BACKEND,
        "experimental": True,
        "include_in_multifidelity": False,
        "score_semantics": SCORE_SEMANTICS,
        "warp_metadata": aggregate_metadata,
        "warp_availability": availability_json,
    }
    if failed:
        payload["error"] = "one or more experimental MuJoCo Warp grasp evaluations failed"
    return payload


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sampler != "random":
        raise ValueError(
            "MuJoCo Warp batched evaluation currently supports sampler=random only. "
            "TPE is sequential/adaptive and is intentionally out of scope for this PR."
        )

    setup_logging()
    design_files = collect_design_files(args.design_dir, args.design_ids, args.max_designs)
    missing = [path for path in design_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing design JSON files: {', '.join(str(path) for path in missing)}")
    if not design_files:
        raise FileNotFoundError(f"No design JSON files found under {args.design_dir}")

    results_dir = ensure_dir(args.results_dir)
    _validate_results_dir_for_warp(results_dir, args.allow_mixed_backend_dir)
    designs = [HandDesign.from_json(path) for path in design_files]
    existing = [result_path(results_dir, design.design_id) for design in designs if result_path(results_dir, design.design_id).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing MuJoCo Warp result files without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    config_data = read_yaml(args.config)
    eval_config = EvaluationConfig.from_dict(config_data)
    geometry_config = GeometryConfig.from_dict(config_data)

    payloads: list[dict[str, Any]] = []
    for design in designs:
        try:
            payload = evaluate_design_warp(design, args, eval_config, geometry_config)
        except Exception as exc:
            if args.fail_fast or args.require_warp:
                raise
            payload = {
                "design_id": design.design_id,
                "parameters": design.to_dict(),
                "hand_score": 0.0,
                "tool_results": [],
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "backend": BACKEND,
                "experimental": True,
                "include_in_multifidelity": False,
                "score_semantics": SCORE_SEMANTICS,
                "warp_metadata": _empty_metadata(
                    args,
                    num_grasps=args.n_grasp_trials * len(args.tools),
                    failure_count=args.n_grasp_trials * len(args.tools),
                ),
            }
        write_json(result_path(results_dir, design.design_id), payload)
        payloads.append(payload)
    return payloads


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        parser.exit(1, f"error: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
