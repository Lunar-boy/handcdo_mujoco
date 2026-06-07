from __future__ import annotations

import pandas as pd
import pytest

from handcdo.design_space import DesignSpace, HandDesign
from handcdo.surrogate import propose_candidates, train_surrogate


def _training_frame(n_rows: int = 8, constant_target: bool = False) -> pd.DataFrame:
    rows = []
    for seed in range(n_rows):
        design = DesignSpace().sample(seed=seed)
        rows.append(
            {
                "design_id": design.design_id,
                "best_available_score": 1.0 if constant_target else float(seed),
                "failed": False,
                **design.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _train_model(tmp_path, frame: pd.DataFrame | None = None):
    results_csv = tmp_path / "results.csv"
    (frame if frame is not None else _training_frame()).to_csv(results_csv, index=False)
    model_path = train_surrogate(results_csv, tmp_path / "model", min_rows=5)
    return model_path, results_csv


def test_proposal_writes_top_k_rows_and_loadable_design_json(tmp_path):
    model_path, _ = _train_model(tmp_path)

    rows = propose_candidates(
        model_path=model_path,
        search_space="configs/search_space.yaml",
        n_random=12,
        top_k=4,
        output_dir=tmp_path / "proposal",
        seed=20,
        exclude_existing=False,
    )

    proposal_csv = tmp_path / "proposal" / "proposed_candidates.csv"
    proposal_df = pd.read_csv(proposal_csv)
    assert len(rows) == 4
    assert len(proposal_df) == 4
    for column in ("rank", "design_id", "predicted_score", *[spec.name for spec in DesignSpace().specs]):
        assert column in proposal_df.columns
    for design_id in proposal_df["design_id"].astype(str):
        design_json = tmp_path / "proposal" / "proposed_designs" / design_id / "design.json"
        assert design_json.exists()
        assert HandDesign.from_json(design_json).design_id == design_id


def test_proposal_is_deterministic_for_fixed_seed(tmp_path):
    model_path, _ = _train_model(tmp_path)
    kwargs = dict(
        model_path=model_path,
        search_space="configs/search_space.yaml",
        n_random=15,
        top_k=5,
        seed=30,
        exclude_existing=False,
    )

    propose_candidates(output_dir=tmp_path / "proposal_a", **kwargs)
    propose_candidates(output_dir=tmp_path / "proposal_b", **kwargs)

    assert (tmp_path / "proposal_a" / "proposed_candidates.csv").read_text(encoding="utf-8") == (
        tmp_path / "proposal_b" / "proposed_candidates.csv"
    ).read_text(encoding="utf-8")


def test_proposal_ties_sort_by_design_id(tmp_path):
    model_path, _ = _train_model(tmp_path, _training_frame(constant_target=True))

    propose_candidates(
        model_path=model_path,
        search_space="configs/search_space.yaml",
        n_random=10,
        top_k=6,
        output_dir=tmp_path / "proposal",
        seed=40,
        exclude_existing=False,
    )

    proposal_df = pd.read_csv(tmp_path / "proposal" / "proposed_candidates.csv")
    assert proposal_df["predicted_score"].nunique() == 1
    assert proposal_df["design_id"].tolist() == sorted(proposal_df["design_id"].tolist())


def test_existing_design_ids_are_excluded_when_requested(tmp_path):
    model_path, _ = _train_model(tmp_path)
    rng = __import__("numpy").random.default_rng(50)
    existing_ids = [DesignSpace().sample(rng=rng).design_id for _ in range(5)]
    existing_csv = tmp_path / "existing.csv"
    pd.DataFrame({"design_id": existing_ids}).to_csv(existing_csv, index=False)

    propose_candidates(
        model_path=model_path,
        search_space="configs/search_space.yaml",
        n_random=12,
        top_k=4,
        output_dir=tmp_path / "proposal",
        seed=50,
        existing_csv=existing_csv,
        exclude_existing=True,
    )

    proposal_df = pd.read_csv(tmp_path / "proposal" / "proposed_candidates.csv")
    assert set(proposal_df["design_id"]).isdisjoint(existing_ids)


def test_invalid_top_k_fails_clearly(tmp_path):
    model_path, _ = _train_model(tmp_path)

    with pytest.raises(ValueError, match="top_k must be <= n_random"):
        propose_candidates(
            model_path=model_path,
            search_space="configs/search_space.yaml",
            n_random=2,
            top_k=3,
            output_dir=tmp_path / "proposal",
        )
