from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from handcdo.design_space import HandDesign
from handcdo.utils import ensure_dir, write_json

from .features import load_design_space, prepare_feature_frame


def propose_candidates(
    model_path: str | Path,
    search_space: str | Path | None,
    n_random: int,
    top_k: int,
    output_dir: str | Path,
    seed: int = 0,
    existing_csv: str | Path | None = None,
    exclude_existing: bool = True,
) -> list[dict[str, Any]]:
    if n_random <= 0:
        raise ValueError("n_random must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k > n_random:
        raise ValueError("top_k must be <= n_random")

    bundle = _load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    metadata = bundle.get("metadata", {})
    feature_columns = list(metadata.get("feature_columns") or [])
    numeric_columns = list(metadata.get("numeric_columns") or [])
    if not feature_columns:
        raise ValueError("Serialized surrogate model is missing feature metadata")

    resolved_search_space = search_space if search_space is not None else metadata.get("search_space")
    space = load_design_space(resolved_search_space)
    output_path = ensure_dir(output_dir)

    existing_design_ids = _load_existing_design_ids(existing_csv, metadata) if exclude_existing else set()
    candidates = _sample_unique_candidates(space, n_random=n_random, seed=seed, existing_design_ids=existing_design_ids)
    if len(candidates) < top_k:
        raise ValueError(
            f"Only {len(candidates)} unique non-existing candidate(s) available after sampling {n_random}; need top_k={top_k}"
        )

    candidate_df = pd.DataFrame([candidate.to_dict() | {"design_id": candidate.design_id} for candidate in candidates])
    x_candidate = prepare_feature_frame(candidate_df, feature_columns, numeric_columns)
    candidate_df["predicted_score"] = pipeline.predict(x_candidate)
    candidate_df = candidate_df.sort_values(
        ["predicted_score", "design_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    selected = candidate_df.head(top_k).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    output_columns = ["rank", "design_id", "predicted_score", *feature_columns]
    selected = selected[output_columns]

    proposed_design_dir = ensure_dir(output_path / "proposed_designs")
    selected_records = selected.to_dict(orient="records")
    by_design_id = {candidate.design_id: candidate for candidate in candidates}
    for row in selected_records:
        design_id = str(row["design_id"])
        design = by_design_id[design_id]
        design.to_json(proposed_design_dir / design_id / "design.json")

    selected.to_csv(output_path / "proposed_candidates.csv", index=False)
    write_json(
        output_path / "manifest.json",
        {
            "model_path": str(model_path),
            "search_space": None if resolved_search_space is None else str(resolved_search_space),
            "seed": seed,
            "n_random": n_random,
            "top_k": top_k,
            "exclude_existing": exclude_existing,
            "selected_design_ids": [str(row["design_id"]) for row in selected_records],
        },
    )
    return selected_records


def _load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load sklearn surrogate models") from exc
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError(f"Invalid surrogate model bundle at {model_path}")
    return payload


def _load_existing_design_ids(existing_csv: str | Path | None, metadata: dict[str, Any]) -> set[str]:
    csv_path = existing_csv if existing_csv is not None else metadata.get("results_csv")
    if not csv_path:
        return set()
    df = pd.read_csv(csv_path)
    if "design_id" not in df.columns:
        return set()
    return set(df["design_id"].dropna().astype(str))


def _sample_unique_candidates(
    space: Any,
    n_random: int,
    seed: int,
    existing_design_ids: set[str],
) -> list[HandDesign]:
    rng = np.random.default_rng(seed)
    candidates: list[HandDesign] = []
    seen: set[str] = set()
    for _ in range(n_random):
        design = space.sample(rng=rng)
        if design.design_id in seen or design.design_id in existing_design_ids:
            continue
        seen.add(design.design_id)
        candidates.append(design)
    return candidates
