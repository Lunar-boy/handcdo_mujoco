from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .design_space import DesignSpace, HandDesign
from .geometry_config import GeometryConfig
from .mujoco_eval import EvaluationConfig
from .optimize_hand import evaluate_design
from .utils import ensure_dir, read_yaml, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def generate_designs(n: int, output_dir: str | Path, seed: int = 0, search_space: str | Path | None = None) -> list[str]:
    space = DesignSpace.from_yaml(search_space) if search_space else DesignSpace()
    design_dir = ensure_dir(output_dir)
    ids: list[str] = []
    for i in range(n):
        design = space.sample(seed=seed + i)
        out = design_dir / design.design_id
        ensure_dir(out)
        design.to_json(out / "design.json")
        ids.append(design.design_id)
    write_json(Path(design_dir) / "manifest.json", {"design_ids": ids})
    return ids


def evaluate_task(
    task_id: int,
    designs_per_task: int,
    design_dir: str | Path,
    results_dir: str | Path,
    config_path: str | Path,
    tools: list[str],
    seed: int = 0,
) -> list[dict[str, Any]]:
    setup_logging()
    design_dir = Path(design_dir)
    output_root = Path(results_dir).parent
    ensure_dir(results_dir)
    config_data = read_yaml(config_path)
    eval_config = EvaluationConfig.from_dict(config_data)
    geometry_config = GeometryConfig.from_dict(config_data)
    n_grasp_trials = int(config_data.get("grasp", {}).get("n_trials", 4))
    sampler = str(config_data.get("grasp", {}).get("sampler", "tpe"))
    design_files = sorted(design_dir.glob("*/design.json"))
    start = task_id * designs_per_task
    selected = design_files[start : start + designs_per_task]
    payloads: list[dict[str, Any]] = []
    for offset, design_file in enumerate(selected):
        try:
            design = HandDesign.from_json(design_file)
            payload = evaluate_design(
                design,
                tools=tools,
                n_grasp_trials=n_grasp_trials,
                output_dir=output_root,
                result_dir=results_dir,
                seed=seed + task_id * 10000 + offset,
                config=eval_config,
                geometry_config=geometry_config,
                sampler=sampler,
            )
        except Exception as exc:
            LOGGER.exception("Failed design file %s", design_file)
            payload = {
                "design_id": design_file.parent.name,
                "parameters": {},
                "hand_score": 0.0,
                "tool_results": [],
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(Path(results_dir) / f"{design_file.parent.name}.json", payload)
        payloads.append(payload)
    return payloads


def main_generate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-designs", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/designs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    args = parser.parse_args()
    ids = generate_designs(args.n_designs, args.output_dir, args.seed, args.search_space)
    print("\n".join(ids))


def main_evaluate_batch() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--designs-per-task", type=int, default=1)
    parser.add_argument("--design-dir", default="outputs/designs")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--config", default="configs/default_eval.yaml")
    parser.add_argument("--tools", default="hammer,spoon,knife")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    evaluate_task(args.task_id, args.designs_per_task, args.design_dir, args.results_dir, args.config, tools, args.seed)
