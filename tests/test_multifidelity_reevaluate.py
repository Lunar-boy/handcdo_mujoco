from __future__ import annotations

import pytest

from handcdo.design_space import DesignSpace
from handcdo.multifidelity import load_design_ids, reevaluate_designs


def test_load_design_ids_strips_whitespace_and_rejects_duplicates(tmp_path):
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("\n a \n\n b\n", encoding="utf-8")

    assert load_design_ids(ids_path) == ["a", "b"]

    ids_path.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate design id"):
        load_design_ids(ids_path)


def test_reevaluate_designs_missing_design_json_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        reevaluate_designs(
            design_dir=tmp_path / "designs",
            design_ids=["missing"],
            results_dir=tmp_path / "results",
            output_dir=tmp_path / "out",
            config_path="configs/eval_fast.yaml",
            fidelity="fast",
            tools=["hammer"],
        )


def test_reevaluate_designs_adds_metadata_and_rewrites_payload(tmp_path, monkeypatch):
    design = DesignSpace().sample(seed=0)
    design_dir = tmp_path / "designs" / design.design_id
    design_dir.mkdir(parents=True)
    design.to_json(design_dir / "design.json")

    def fake_evaluate_design(loaded_design, **kwargs):
        payload = {
            "design_id": loaded_design.design_id,
            "parameters": loaded_design.to_dict(),
            "hand_score": 0.7,
            "tool_results": [],
            "failed": False,
        }
        from handcdo.utils import write_json

        write_json(kwargs["result_dir"] / f"{loaded_design.design_id}.json", payload)
        return payload

    monkeypatch.setattr("handcdo.multifidelity.evaluate_design", fake_evaluate_design)

    payloads = reevaluate_designs(
        design_dir=tmp_path / "designs",
        design_ids=[design.design_id],
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "out",
        config_path="configs/eval_medium.yaml",
        fidelity="medium",
        tools=["hammer"],
        backend="mujoco_cpu",
        seed=100,
        n_grasp_trials=3,
        sampler="random",
    )

    payload = payloads[0]
    assert payload["fidelity"] == "medium"
    assert payload["backend"] == "mujoco_cpu"
    assert payload["config_path"] == "configs/eval_medium.yaml"
    assert payload["n_grasp_trials"] == 3
    assert payload["sampler"] == "random"
    assert payload["seed"] == 100
    assert '"fidelity": "medium"' in (tmp_path / "results" / f"{design.design_id}.json").read_text(encoding="utf-8")
