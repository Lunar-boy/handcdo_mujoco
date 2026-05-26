from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .design_space import HandDesign
from .grasp_sampling import optuna_suggest_grasp, sample_random_grasp
from .mujoco_eval import EvaluationConfig, GraspEvaluation, evaluate_grasp

LOGGER = logging.getLogger(__name__)


def optimize_grasp_for_tool(
    design: HandDesign,
    tool_name: str,
    n_trials: int,
    seed: int = 0,
    config: EvaluationConfig | None = None,
    sampler: str = "tpe",
) -> dict[str, Any]:
    best: GraspEvaluation | None = None
    trials: list[dict[str, Any]] = []
    if sampler == "tpe":
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

            def objective(trial: Any) -> float:
                nonlocal best
                grasp = optuna_suggest_grasp(trial)
                result = evaluate_grasp(design, tool_name, grasp, config)
                trials.append(result.to_dict())
                if best is None or result.score > best.score:
                    best = result
                return result.score

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except Exception as exc:
            LOGGER.warning("Falling back to random grasp search because Optuna failed: %s", exc)
            sampler = "random"

    if sampler != "tpe":
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            grasp = sample_random_grasp(rng=rng)
            result = evaluate_grasp(design, tool_name, grasp, config)
            trials.append(result.to_dict())
            if best is None or result.score > best.score:
                best = result

    best_payload = best.to_dict() if best else {"score": 0.0, "failed": True, "error": "no trials completed"}
    return {"tool": tool_name, "best_score": float(best_payload.get("score", 0.0)), "best_grasp": best_payload, "trials": trials}
