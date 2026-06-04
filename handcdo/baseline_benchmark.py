from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .collect_results import collect_results
from .design_space import HandDesign
from .geometry_config import GeometryConfig
from .mujoco_eval import EvaluationConfig
from .optimize_hand import evaluate_design
from .slurm_batch import generate_designs
from .utils import ensure_dir, read_yaml, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def parse_tools(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def load_benchmark_designs(design_dir: str | Path, n_designs: int) -> list[HandDesign]:
    design_dir = Path(design_dir)
    design_files: list[Path] = []
    manifest_path = design_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        design_ids = manifest.get("design_ids", [])
        if not isinstance(design_ids, list):
            raise ValueError(f"{manifest_path} must contain a list field named design_ids")
        design_files = [design_dir / str(design_id) / "design.json" for design_id in design_ids]
    else:
        design_files = sorted(design_dir.glob("*/design.json"))

    missing = [path for path in design_files if not path.exists()]
    if missing:
        raise ValueError(f"Design listed in manifest is missing: {missing[0]}")
    if len(design_files) < n_designs:
        raise ValueError(f"Need {n_designs} designs in {design_dir}, found {len(design_files)}")
    return [HandDesign.from_json(path) for path in design_files[:n_designs]]


def get_git_metadata(repo_dir: str | Path | None = None) -> dict[str, Any]:
    cwd = Path(repo_dir) if repo_dir is not None else Path.cwd()

    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        value = completed.stdout.strip()
        return value or None

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = False
    status = run_git(["status", "--porcelain"])
    if status is not None:
        dirty = bool(status)
    return {"commit": commit, "branch": branch, "dirty": dirty}


def get_environment_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "mujoco_version": _module_version("mujoco"),
        "numpy_version": _module_version("numpy"),
        "pandas_version": _module_version("pandas"),
    }


def write_benchmark_metadata(
    output_dir: str | Path,
    *,
    seed: int,
    n_designs: int,
    n_grasp_trials: int,
    tools: list[str],
    backend: str,
    config_path: str | Path,
    search_space_path: str | Path | None,
    design_dir: str | Path,
    results_dir: str | Path,
    results_csv: str | Path,
    successful_designs: int | None = None,
    failed_designs: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metadata = {
        "benchmark_schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_metadata(output_dir.parent),
        "environment": get_environment_metadata(),
        "benchmark": {
            "seed": seed,
            "n_designs": n_designs,
            "n_grasp_trials": n_grasp_trials,
            "tools": tools,
            "backend": backend,
            "config_path": str(config_path),
            "search_space_path": str(search_space_path) if search_space_path is not None else None,
            "design_dir": str(design_dir),
            "results_dir": str(results_dir),
            "results_csv": str(results_csv),
            "config_sha256": _file_sha256(config_path),
            "search_space_sha256": _file_sha256(search_space_path) if search_space_path is not None else None,
            "successful_designs": successful_designs,
            "failed_designs": failed_designs,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def run_baseline_benchmark(
    *,
    n_designs: int = 20,
    n_grasp_trials: int = 4,
    tools: list[str] | str = "hammer,spoon,knife",
    seed: int = 0,
    backend: str = "mujoco_cpu",
    config: str | Path = "configs/default_eval.yaml",
    search_space: str | Path | None = "configs/search_space.yaml",
    output_dir: str | Path = "outputs/baselines/current",
    design_dir: str | Path | None = None,
    reuse_designs: bool = False,
    sampler: str | None = None,
) -> dict[str, Any]:
    setup_logging()
    tools = parse_tools(tools)
    output_dir = ensure_dir(output_dir)
    results_dir = ensure_dir(output_dir / "results")
    results_csv = output_dir / "results.csv"
    benchmark_design_dir = Path(design_dir) if design_dir is not None else output_dir / "designs"

    if design_dir is None:
        if not (reuse_designs and (benchmark_design_dir / "manifest.json").exists()):
            generate_designs(n_designs, benchmark_design_dir, seed=seed, search_space=search_space)
    designs = load_benchmark_designs(benchmark_design_dir, n_designs)

    config_data = read_yaml(config)
    sampler_name = sampler or str(config_data.get("grasp", {}).get("sampler", "tpe"))
    eval_config = EvaluationConfig.from_dict(config_data)
    geometry_config = GeometryConfig.from_dict(config_data)

    payloads: list[dict[str, Any]] = []
    for design_index, design in enumerate(designs):
        try:
            payload = evaluate_design(
                design,
                tools=tools,
                n_grasp_trials=n_grasp_trials,
                output_dir=output_dir,
                result_dir=results_dir,
                seed=seed + design_index,
                config=eval_config,
                geometry_config=geometry_config,
                backend_name=backend,
                sampler=sampler_name,
            )
        except Exception as exc:
            LOGGER.exception("Failed design %s", design.design_id)
            payload = {
                "design_id": design.design_id,
                "parameters": design.to_dict(),
                "hand_score": 0.0,
                "tool_results": [],
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(results_dir / f"{design.design_id}.json", payload)
        payloads.append(payload)

    rows = collect_results(results_dir, results_csv)
    failed_designs = sum(1 for row in rows if bool(row.get("failed")))
    metadata = write_benchmark_metadata(
        output_dir,
        seed=seed,
        n_designs=n_designs,
        n_grasp_trials=n_grasp_trials,
        tools=tools,
        backend=backend,
        config_path=config,
        search_space_path=search_space,
        design_dir=benchmark_design_dir,
        results_dir=results_dir,
        results_csv=results_csv,
        successful_designs=len(rows) - failed_designs,
        failed_designs=failed_designs,
    )
    return {"payloads": payloads, "rows": rows, "metadata": metadata}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible MuJoCo CPU baseline benchmark.")
    parser.add_argument("--n-designs", type=int, default=20)
    parser.add_argument("--n-grasp-trials", type=int, default=4)
    parser.add_argument("--tools", default="hammer,spoon,knife")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="mujoco_cpu", choices=["mujoco", "mujoco_cpu"])
    parser.add_argument("--config", default="configs/default_eval.yaml")
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    parser.add_argument("--output-dir", default="outputs/baselines/current")
    parser.add_argument("--design-dir")
    parser.add_argument("--reuse-designs", action="store_true")
    parser.add_argument("--sampler")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_baseline_benchmark(
        n_designs=args.n_designs,
        n_grasp_trials=args.n_grasp_trials,
        tools=args.tools,
        seed=args.seed,
        backend=args.backend,
        config=args.config,
        search_space=args.search_space,
        output_dir=args.output_dir,
        design_dir=args.design_dir,
        reuse_designs=args.reuse_designs,
        sampler=args.sampler,
    )
    print(f"Wrote {len(result['rows'])} rows to {Path(args.output_dir) / 'results.csv'}")


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
