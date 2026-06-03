from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .backends import SimulatorBackend, get_backend
from .design_space import DesignSpace, HandDesign
from .geometry_config import GeometryConfig
from .mjcf_generator import write_design_model
from .mujoco_eval import EvaluationConfig
from .optimize_grasp import optimize_grasp_for_tool
from .tools import tool_names
from .utils import ensure_dir, read_yaml, setup_logging, write_json


def evaluate_design(
    design: HandDesign,
    tools: list[str],
    n_grasp_trials: int,
    output_dir: str | Path,
    result_dir: str | Path | None = None,
    seed: int = 0,
    config: EvaluationConfig | None = None,
    geometry_config: GeometryConfig | None = None,
    backend_name: str = "mujoco_cpu",
    backend: SimulatorBackend | None = None,
) -> dict[str, Any]:
    backend = backend or get_backend(backend_name)
    result_dir = ensure_dir(result_dir if result_dir is not None else Path(output_dir) / "results")
    design_dir = ensure_dir(Path(output_dir) / "designs" / design.design_id)
    design.to_json(design_dir / "design.json")
    write_design_model(design, output_dir, geometry_config=geometry_config)
    tool_results = []
    for i, tool in enumerate(tools):
        tool_results.append(
            optimize_grasp_for_tool(
                design,
                tool,
                n_trials=n_grasp_trials,
                seed=seed + 1009 * (i + 1),
                config=config,
                geometry_config=geometry_config,
                backend=backend,
            )
        )
    score = float(np.mean([r["best_score"] for r in tool_results])) if tool_results else 0.0
    payload = {
        "design_id": design.design_id,
        "parameters": design.to_dict(),
        "hand_score": score,
        "tool_results": tool_results,
        "failed": False,
    }
    write_json(result_dir / f"{design.design_id}.json", payload)
    return payload


def run_optuna(args: argparse.Namespace) -> None:
    import optuna

    setup_logging()
    output_dir = ensure_dir(args.output_dir)
    config_data = read_yaml(args.config) if args.config else {}
    eval_config = EvaluationConfig.from_dict(config_data)
    geometry_config = GeometryConfig.from_dict(config_data)
    space = DesignSpace.from_yaml(args.search_space) if args.search_space else DesignSpace()
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        design = space.optuna_suggest(trial)
        payload = evaluate_design(
            design,
            tools=tools,
            n_grasp_trials=args.n_grasp_trials,
            output_dir=output_dir,
            seed=args.seed + trial.number * 100,
            config=eval_config,
            geometry_config=geometry_config,
            backend_name=args.backend,
        )
        trial.set_user_attr("design_id", design.design_id)
        return float(payload["hand_score"])

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    write_json(output_dir / "best_design.json", {"value": study.best_value, "params": study.best_params})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TPE hand co-design optimization with MuJoCo evaluation.")
    parser.add_argument("--study-name", default="handcdo-mujoco")
    parser.add_argument("--storage", default="sqlite:///outputs/handcdo_optuna.db")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-grasp-trials", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tools", default="hammer,spoon,knife")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "mujoco_cpu"])
    parser.add_argument("--config", default="configs/default_eval.yaml")
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    return parser


def main() -> None:
    run_optuna(build_parser().parse_args())


if __name__ == "__main__":
    main()
