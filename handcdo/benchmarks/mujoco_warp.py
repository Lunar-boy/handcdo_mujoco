from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import sample_random_grasp
from handcdo.mjcf_generator import write_design_model
from handcdo.mujoco_eval import EvaluationConfig
from handcdo.tools import get_tool
from handcdo.utils import ensure_dir, read_yaml, write_json
from handcdo.warp_utils import (
    WarpAvailability,
    availability_payload,
    check_warp_available,
    make_warp_data,
    prepare_warp_compatible_mjcf,
    synchronize_warp,
    utc_timestamp,
)


CSV_COLUMNS = (
    "backend",
    "available",
    "success",
    "scene_mode",
    "nworld",
    "nconmax",
    "naconmax",
    "njmax",
    "seconds_mean",
    "seconds_std",
    "steps",
    "warmup_steps",
    "repeats",
    "total_sim_steps",
    "total_world_steps",
    "steps_per_second_total",
    "world_steps_per_second",
    "steps_per_second_per_world",
    "failure_count",
    "failure_stage",
    "exception_type",
    "error",
)


@dataclass(frozen=True)
class WarpBenchmarkConfig:
    output_dir: Path
    design_json: Path | None = None
    search_space: Path = Path("configs/search_space.yaml")
    config_path: Path = Path("configs/eval_fast.yaml")
    tool: str = "hammer"
    seed: int = 0
    scene_mode: str = "load_step"
    cpu_repeats: int = 3
    warp_repeats: int = 3
    warmup_steps: int = 10
    steps: int = 100
    nworld: int = 64
    nconmax: int | None = 64
    naconmax: int | None = None
    njmax: int = 128
    sweep_nworld: tuple[int, ...] | None = None
    sweep_nconmax: tuple[int, ...] | None = None
    sweep_njmax: tuple[int, ...] | None = None
    require_warp: bool = False
    skip_cpu: bool = False
    overwrite: bool = False
    no_warp_xml_rewrite: bool = False


def validate_config(config: WarpBenchmarkConfig) -> None:
    if config.scene_mode not in {"load_step", "contact_smoke"}:
        raise ValueError("scene_mode must be one of: load_step, contact_smoke")
    if config.steps <= 0:
        raise ValueError(f"steps={config.steps!r} must be > 0")
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps={config.warmup_steps!r} must be >= 0")
    if config.cpu_repeats <= 0:
        raise ValueError(f"cpu_repeats={config.cpu_repeats!r} must be > 0")
    if config.warp_repeats <= 0:
        raise ValueError(f"warp_repeats={config.warp_repeats!r} must be > 0")
    if config.nworld <= 0:
        raise ValueError(f"nworld={config.nworld!r} must be > 0")
    if config.nconmax is not None and config.nconmax <= 0:
        raise ValueError(f"nconmax={config.nconmax!r} must be > 0")
    if config.naconmax is not None and config.naconmax <= 0:
        raise ValueError(f"naconmax={config.naconmax!r} must be > 0")
    if config.njmax <= 0:
        raise ValueError(f"njmax={config.njmax!r} must be > 0")
    get_tool(config.tool)
    for name, values in (
        ("sweep_nworld", config.sweep_nworld),
        ("sweep_nconmax", config.sweep_nconmax),
        ("sweep_njmax", config.sweep_njmax),
    ):
        if values is not None and any(value <= 0 for value in values):
            raise ValueError(f"{name} values must all be > 0")


def parse_positive_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(f"{value!r} must be a comma-separated list of positive integers")
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} must be a comma-separated list of positive integers") from exc
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{value!r} must contain only positive integers")
    return parsed


def capacity_rows(config: WarpBenchmarkConfig) -> list[dict[str, int | None]]:
    nworld_values = config.sweep_nworld or (config.nworld,)
    nconmax_values = config.sweep_nconmax or (config.nconmax,)
    njmax_values = config.sweep_njmax or (config.njmax,)
    return [
        {"nworld": nworld, "nconmax": nconmax, "naconmax": config.naconmax, "njmax": njmax}
        for nworld in nworld_values
        for nconmax in nconmax_values
        for njmax in njmax_values
    ]


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory {output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)


def _load_eval_and_geometry_config(config_path: Path) -> tuple[EvaluationConfig, GeometryConfig]:
    data = read_yaml(config_path)
    return EvaluationConfig.from_dict(data), GeometryConfig.from_dict(data)


def _select_design(config: WarpBenchmarkConfig) -> HandDesign:
    space = DesignSpace.from_yaml(config.search_space)
    if config.design_json is not None:
        return HandDesign.from_json(config.design_json, space=space)
    return space.sample(seed=config.seed)


def build_fixed_benchmark_model(config: WarpBenchmarkConfig) -> dict[str, Any]:
    evaluation_config, geometry_config = _load_eval_and_geometry_config(config.config_path)
    design = _select_design(config)
    design_dir = ensure_dir(config.output_dir / "design")
    model_dir = ensure_dir(config.output_dir / "model")
    staging_dir = ensure_dir(config.output_dir / "_staging")
    design.to_json(design_dir / "design.json")
    generated_model = write_design_model(
        design,
        staging_dir,
        tool_name=config.tool,
        geometry_config=geometry_config,
    )
    original_model = model_dir / "original_model.xml"
    warp_model = model_dir / "warp_model.xml"
    shutil.copyfile(generated_model, original_model)
    shutil.rmtree(staging_dir, ignore_errors=True)
    rewrite_result = prepare_warp_compatible_mjcf(
        original_model,
        warp_model,
        allow_rewrite=not config.no_warp_xml_rewrite,
    )
    return {
        "design": design,
        "evaluation_config": evaluation_config,
        "geometry_config": geometry_config,
        "original_mjcf_path": original_model,
        "warp_mjcf_path": warp_model,
        "mjcf_rewrites": rewrite_result["mjcf_rewrites"],
        "mjcf_files_differ": rewrite_result["mjcf_files_differ"],
    }


def _failure_row(
    *,
    backend: str,
    scene_mode: str,
    steps: int,
    warmup_steps: int,
    repeats: int,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
    available: bool,
    failure_stage: str,
    exc: BaseException | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return _normalize_row(
        {
            "backend": backend,
            "available": available,
            "success": False,
            "scene_mode": scene_mode,
            "nworld": nworld,
            "nconmax": nconmax,
            "naconmax": naconmax,
            "njmax": njmax,
            "seconds_mean": None,
            "seconds_std": None,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "repeats": repeats,
            "total_sim_steps": 0,
            "total_world_steps": 0,
            "steps_per_second_total": None,
            "world_steps_per_second": None,
            "steps_per_second_per_world": None,
            "failure_count": 1,
            "failure_stage": failure_stage,
            "exception_type": type(exc).__name__ if exc is not None else None,
            "error": error if error is not None else str(exc) if exc is not None else None,
        }
    )


def _timing_row(
    *,
    backend: str,
    scene_mode: str,
    steps: int,
    warmup_steps: int,
    repeats: int,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
    durations: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seconds_mean = float(np.mean(durations)) if durations else 0.0
    seconds_std = float(np.std(durations)) if len(durations) > 1 else 0.0
    total_sim_steps = int(steps * repeats)
    total_world_steps = int(steps * repeats * nworld)
    row = {
        "backend": backend,
        "available": True,
        "success": True,
        "scene_mode": scene_mode,
        "nworld": nworld,
        "nconmax": nconmax,
        "naconmax": naconmax,
        "njmax": njmax,
        "seconds_mean": seconds_mean,
        "seconds_std": seconds_std,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "repeats": repeats,
        "total_sim_steps": total_sim_steps,
        "total_world_steps": total_world_steps,
        "steps_per_second_total": total_sim_steps / seconds_mean if seconds_mean > 0 else None,
        "world_steps_per_second": total_world_steps / seconds_mean if seconds_mean > 0 else None,
        "steps_per_second_per_world": (total_world_steps / seconds_mean / nworld) if seconds_mean > 0 else None,
        "failure_count": 0,
        "failure_stage": None,
        "exception_type": None,
        "error": None,
    }
    if extra:
        row.update(extra)
    return _normalize_row(row)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in CSV_COLUMNS} | {
        key: value for key, value in row.items() if key not in CSV_COLUMNS
    }


def _setup_cpu_scene(
    model: Any,
    data: Any,
    scene_mode: str,
    seed: int,
    evaluation_config: EvaluationConfig,
    tool_name: str,
) -> None:
    import mujoco

    if scene_mode == "contact_smoke":
        from handcdo.mujoco_eval import _close_hand, _set_tool_pose

        grasp = sample_random_grasp(seed=seed)
        _set_tool_pose(model, data, get_tool(tool_name), grasp)
        mujoco.mj_forward(model, data)
        _close_hand(model, data, grasp, max(1, min(evaluation_config.close_steps, 50)))
    else:
        mujoco.mj_forward(model, data)


def run_cpu_mujoco_timing(
    mjcf_path: str | Path,
    steps: int,
    warmup_steps: int,
    repeats: int,
    scene_mode: str,
    seed: int,
    tool_name: str = "hammer",
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    try:
        import mujoco
    except Exception as exc:
        return _failure_row(
            backend="mujoco_cpu",
            scene_mode=scene_mode,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            nworld=1,
            nconmax=None,
            naconmax=None,
            njmax=0,
            available=False,
            failure_stage="import",
            exc=exc,
        )

    evaluation_config = evaluation_config or EvaluationConfig()
    load_start = time.perf_counter()
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as exc:
        return _failure_row(
            backend="mujoco_cpu",
            scene_mode=scene_mode,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            nworld=1,
            nconmax=None,
            naconmax=None,
            njmax=0,
            available=True,
            failure_stage="load_model",
            exc=exc,
        )
    load_seconds = time.perf_counter() - load_start

    durations: list[float] = []
    max_contacts = 0
    max_constraints = 0
    try:
        for repeat in range(repeats):
            data = mujoco.MjData(model)
            _setup_cpu_scene(model, data, scene_mode, seed + repeat, evaluation_config, tool_name)
            for _ in range(warmup_steps):
                mujoco.mj_step(model, data)
                max_contacts = max(max_contacts, int(getattr(data, "ncon", 0)))
                max_constraints = max(max_constraints, int(getattr(data, "nefc", 0)))
            start = time.perf_counter()
            for _ in range(steps):
                mujoco.mj_step(model, data)
                max_contacts = max(max_contacts, int(getattr(data, "ncon", 0)))
                max_constraints = max(max_constraints, int(getattr(data, "nefc", 0)))
            durations.append(time.perf_counter() - start)
    except Exception as exc:
        return _failure_row(
            backend="mujoco_cpu",
            scene_mode=scene_mode,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            nworld=1,
            nconmax=int(getattr(model, "nconmax", 0)) or None,
            naconmax=None,
            njmax=int(getattr(model, "njmax", 0)) if hasattr(model, "njmax") else 0,
            available=True,
            failure_stage="step",
            exc=exc,
        )

    return _timing_row(
        backend="mujoco_cpu",
        scene_mode=scene_mode,
        steps=steps,
        warmup_steps=warmup_steps,
        repeats=repeats,
        nworld=1,
        nconmax=int(getattr(model, "nconmax", 0)) or None,
        naconmax=None,
        njmax=int(getattr(model, "njmax", 0)) if hasattr(model, "njmax") else 0,
        durations=durations,
        extra={
            "load_seconds": load_seconds,
            "max_contacts_observed": max_contacts,
            "max_constraints_observed": max_constraints,
        },
    )


def run_warp_timing(
    mjcf_path: str | Path,
    steps: int,
    warmup_steps: int,
    repeats: int,
    scene_mode: str,
    seed: int,
    nworld: int,
    nconmax: int | None,
    naconmax: int | None,
    njmax: int,
    tool_name: str = "hammer",
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    evaluation_config = evaluation_config or EvaluationConfig()
    import_start = time.perf_counter()
    try:
        import mujoco
        import mujoco_warp as mjw
    except Exception as exc:
        return _failure_row(
            backend="mujoco_warp",
            scene_mode=scene_mode,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            nworld=nworld,
            nconmax=nconmax,
            naconmax=naconmax,
            njmax=njmax,
            available=False,
            failure_stage="import",
            exc=exc,
        )
    import_seconds = time.perf_counter() - import_start

    try:
        load_start = time.perf_counter()
        mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        mj_data = mujoco.MjData(mj_model)
        _setup_cpu_scene(mj_model, mj_data, scene_mode, seed, evaluation_config, tool_name)
        load_seconds = time.perf_counter() - load_start
        transfer_start = time.perf_counter()
        if not hasattr(mjw, "put_model"):
            raise AttributeError("mujoco_warp.put_model is unavailable")
        warp_model = mjw.put_model(mj_model)
        transfer_seconds = time.perf_counter() - transfer_start
        allocation_start = time.perf_counter()
        warp_data = make_warp_data(mjw, warp_model, mj_model, mj_data, nworld, nconmax, naconmax, njmax)
        allocation_seconds = time.perf_counter() - allocation_start
        if not hasattr(mjw, "step"):
            raise AttributeError("mujoco_warp.step is unavailable")
        synchronized, sync_warning = synchronize_warp()
        warmup_start = time.perf_counter()
        for _ in range(warmup_steps):
            mjw.step(warp_model, warp_data)
        synchronized_after_warmup, sync_warning_after = synchronize_warp()
        warmup_seconds = time.perf_counter() - warmup_start
        durations: list[float] = []
        for _ in range(repeats):
            synchronize_warp()
            start = time.perf_counter()
            for _ in range(steps):
                mjw.step(warp_model, warp_data)
            synchronize_warp()
            durations.append(time.perf_counter() - start)
    except Exception as exc:
        return _failure_row(
            backend="mujoco_warp",
            scene_mode=scene_mode,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            nworld=nworld,
            nconmax=nconmax,
            naconmax=naconmax,
            njmax=njmax,
            available=True,
            failure_stage="warp_step",
            exc=exc,
        )

    return _timing_row(
        backend="mujoco_warp",
        scene_mode=scene_mode,
        steps=steps,
        warmup_steps=warmup_steps,
        repeats=repeats,
        nworld=nworld,
        nconmax=nconmax,
        naconmax=naconmax,
        njmax=njmax,
        durations=durations,
        extra={
            "import_seconds": import_seconds,
            "load_seconds": load_seconds,
            "transfer_seconds": transfer_seconds,
            "allocation_seconds": allocation_seconds,
            "warmup_seconds": warmup_seconds,
            "synchronized": bool(synchronized and synchronized_after_warmup),
            "sync_warning": sync_warning or sync_warning_after,
            "capture_graph": False,
            "contact_smoke_setup": scene_mode == "contact_smoke",
        },
    )


def scheduler_profile() -> str | None:
    import os

    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition == "capella":
        return "capella"
    if partition == "alpha":
        return "alpha"
    return None


def write_benchmark_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "availability.json", result["availability"])
    write_json(output_dir / "benchmark_results.json", result)
    with (output_dir / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(row)


def run_benchmark(config: WarpBenchmarkConfig) -> dict[str, Any]:
    validate_config(config)
    _prepare_output_dir(config.output_dir, config.overwrite)
    availability = check_warp_available()
    availability_meta = availability_payload(availability)
    model_payload = build_fixed_benchmark_model(config)
    rows: list[dict[str, Any]] = []

    if not config.skip_cpu:
        rows.append(
            run_cpu_mujoco_timing(
                model_payload["original_mjcf_path"],
                steps=config.steps,
                warmup_steps=config.warmup_steps,
                repeats=config.cpu_repeats,
                scene_mode=config.scene_mode,
                seed=config.seed,
                tool_name=config.tool,
                evaluation_config=model_payload["evaluation_config"],
            )
        )

    warp_skip_reason = None
    if not availability.available:
        warp_skip_reason = availability.reason or "MuJoCo Warp is unavailable"
        if config.require_warp:
            result = _result_payload(config, availability_meta, model_payload, rows, warp_skip_reason)
            write_benchmark_outputs(result, config.output_dir)
            raise RuntimeError(f"MuJoCo Warp is required but unavailable: {warp_skip_reason}")
    else:
        for row in capacity_rows(config):
            rows.append(
                run_warp_timing(
                    model_payload["warp_mjcf_path"],
                    steps=config.steps,
                    warmup_steps=config.warmup_steps,
                    repeats=config.warp_repeats,
                    scene_mode=config.scene_mode,
                    seed=config.seed,
                    nworld=int(row["nworld"] or 1),
                    nconmax=row["nconmax"],
                    naconmax=row["naconmax"],
                    njmax=int(row["njmax"] or config.njmax),
                    tool_name=config.tool,
                    evaluation_config=model_payload["evaluation_config"],
                )
            )

    result = _result_payload(config, availability_meta, model_payload, rows, warp_skip_reason)
    write_benchmark_outputs(result, config.output_dir)
    if config.require_warp and any(row["backend"] == "mujoco_warp" and not row["success"] for row in rows):
        raise RuntimeError("MuJoCo Warp is required but at least one benchmark row failed")
    return result


def _config_to_dict(config: WarpBenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _result_payload(
    config: WarpBenchmarkConfig,
    availability_meta: dict[str, Any],
    model_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    warp_skip_reason: str | None,
) -> dict[str, Any]:
    exceptions = [
        f"{row.get('backend')}:{row.get('failure_stage')}:{row.get('exception_type')}:{row.get('error')}"
        for row in rows
        if row.get("failure_count")
    ]
    return {
        "benchmark_schema_version": 1,
        "timestamp": utc_timestamp(),
        "scheduler_profile": scheduler_profile(),
        "input": _config_to_dict(config),
        "design_id": model_payload["design"].design_id,
        "original_mjcf_path": str(model_payload["original_mjcf_path"]),
        "warp_mjcf_path": str(model_payload["warp_mjcf_path"]),
        "mjcf_rewrites": model_payload["mjcf_rewrites"],
        "mjcf_files_differ": model_payload["mjcf_files_differ"],
        "scene_mode": config.scene_mode,
        "score_semantics": "none_benchmark_only",
        "scene_mode_semantics": (
            "contact_smoke_not_score_equivalent" if config.scene_mode == "contact_smoke" else "load_step_not_score_equivalent"
        ),
        "availability": availability_meta,
        "warp_skip_reason": warp_skip_reason,
        "rows": rows,
        "cpu_timing": next((row for row in rows if row["backend"] == "mujoco_cpu"), None),
        "warp_timing": [row for row in rows if row["backend"] == "mujoco_warp"],
        "failure_count": sum(int(row.get("failure_count") or 0) for row in rows),
        "exceptions": exceptions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark-only MuJoCo Warp compatibility and throughput diagnostics.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--design-json", type=Path)
    parser.add_argument("--search-space", type=Path, default=Path("configs/search_space.yaml"))
    parser.add_argument("--config", dest="config_path", type=Path, default=Path("configs/eval_fast.yaml"))
    parser.add_argument("--tool", default="hammer", choices=("hammer", "spoon", "knife"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene-mode", choices=("load_step", "contact_smoke"), default="load_step")
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--warp-repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--nworld", type=int, default=64)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--naconmax", type=int)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--sweep-nworld", type=parse_positive_int_list)
    parser.add_argument("--sweep-nconmax", type=parse_positive_int_list)
    parser.add_argument("--sweep-njmax", type=parse_positive_int_list)
    parser.add_argument("--require-warp", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-warp-xml-rewrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = WarpBenchmarkConfig(**vars(args))
    try:
        result = run_benchmark(config)
    except Exception as exc:
        print(f"benchmark_mujoco_warp failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result.get("warp_skip_reason"):
        print(f"MuJoCo Warp skipped: {result['warp_skip_reason']}")
    print(f"Wrote benchmark outputs to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
