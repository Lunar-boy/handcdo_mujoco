from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from handcdo.backends.batched import BatchedSimulatorBackend, supports_batched_grasps
from handcdo.design_space import HandDesign
from handcdo.geometry_config import GeometryConfig
from handcdo.grasp_sampling import GraspParams, sample_random_grasp
from handcdo.mujoco_eval import EvaluationConfig, GraspEvaluation


EXPERIMENTAL_WARP_METADATA = {
    "backend": "mujoco_warp",
    "experimental": True,
    "score_semantics": "experimental_non_equivalent",
}


def evaluate_fixed_grasps_batched(
    backend: BatchedSimulatorBackend,
    design: HandDesign,
    tool_name: str,
    grasps: list[GraspParams],
    config: EvaluationConfig | None,
    geometry_config: GeometryConfig | None = None,
    tool_assets_dir: str | Path = "assets/tools",
) -> list[GraspEvaluation]:
    if not supports_batched_grasps(backend):
        raise TypeError("backend must expose evaluate_grasps_batch for fixed-grasp batch orchestration")
    if not grasps:
        return []

    try:
        evaluations = backend.evaluate_grasps_batch(
            design,
            tool_name,
            grasps,
            config,
            geometry_config=geometry_config,
            tool_assets_dir=tool_assets_dir,
        )
    except Exception as exc:
        return [
            GraspEvaluation(
                design_id=design.design_id,
                tool=tool_name,
                grasp=grasp.to_dict(),
                score=0.0,
                wrench_results=[],
                failed=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            for grasp in grasps
        ]

    if len(evaluations) != len(grasps):
        raise ValueError(
            "Batched backend returned the wrong number of evaluations: "
            f"expected {len(grasps)}, got {len(evaluations)}"
        )
    return evaluations


def sample_fixed_random_grasps(
    n_grasp_trials: int,
    seed: int,
    design_id: str | None = None,
    tool_name: str | None = None,
) -> list[GraspParams]:
    if n_grasp_trials < 0:
        raise ValueError("n_grasp_trials must be >= 0")

    rng = np.random.default_rng(_derive_seed(seed=seed, design_id=design_id, tool_name=tool_name))
    return [sample_random_grasp(rng=rng) for _ in range(n_grasp_trials)]


def summarize_tool_batch_results(
    tool_name: str,
    grasps: list[GraspParams],
    evaluations: list[GraspEvaluation],
) -> dict[str, Any]:
    if len(grasps) != len(evaluations):
        raise ValueError(f"len(grasps) must match len(evaluations): {len(grasps)} != {len(evaluations)}")

    successful = [evaluation for evaluation in evaluations if not evaluation.failed]
    best = max(successful, key=lambda evaluation: evaluation.score, default=None)
    best_payload = (
        best.to_dict()
        if best is not None
        else {"score": 0.0, "failed": True, "error": "no successful trials completed"}
    )

    return {
        "tool": tool_name,
        **EXPERIMENTAL_WARP_METADATA,
        "best_score": float(best_payload.get("score", 0.0)),
        "best_grasp": best_payload,
        "failure_count": sum(1 for evaluation in evaluations if evaluation.failed),
        "trials": [evaluation.to_dict() for evaluation in evaluations],
    }


def _derive_seed(seed: int, design_id: str | None, tool_name: str | None) -> int:
    payload = {"seed": int(seed), "design_id": design_id or "", "tool_name": tool_name or ""}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
