from __future__ import annotations

from pathlib import Path

from handcdo import baseline_benchmark
from handcdo.baseline_benchmark import (
    get_git_metadata,
    load_benchmark_designs,
    run_baseline_benchmark,
    write_benchmark_metadata,
)
from handcdo.design_space import DesignSpace
from handcdo.utils import write_json


def test_metadata_writer_includes_required_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("grasp:\n  n_trials: 1\n", encoding="utf-8")

    metadata = write_benchmark_metadata(
        tmp_path,
        seed=7,
        n_designs=2,
        n_grasp_trials=1,
        tools=["hammer"],
        backend="mujoco_cpu",
        config_path=config,
        search_space_path=None,
        design_dir=tmp_path / "designs",
        results_dir=tmp_path / "results",
        results_csv=tmp_path / "results.csv",
    )

    assert metadata["benchmark_schema_version"] == 1
    assert metadata["benchmark"]["seed"] == 7
    assert metadata["benchmark"]["tools"] == ["hammer"]
    assert metadata["benchmark"]["backend"] == "mujoco_cpu"
    assert metadata["benchmark"]["config_path"] == str(config)
    assert metadata["environment"]["python_version"]


def test_git_metadata_fails_gracefully_outside_git(tmp_path):
    metadata = get_git_metadata(tmp_path)

    assert set(metadata) == {"commit", "branch", "dirty"}
    assert metadata["commit"] is None
    assert metadata["branch"] is None
    assert metadata["dirty"] is False


def test_design_loading_from_existing_directory_is_deterministic(tmp_path):
    design_root = tmp_path / "designs"
    design_a = DesignSpace().sample(seed=0)
    design_b = DesignSpace().sample(seed=1)
    for design in (design_b, design_a):
        out = design_root / design.design_id
        out.mkdir(parents=True)
        design.to_json(out / "design.json")
    write_json(design_root / "manifest.json", {"design_ids": [design_a.design_id, design_b.design_id]})

    designs = load_benchmark_designs(design_root, 2)

    assert [design.design_id for design in designs] == [design_a.design_id, design_b.design_id]


def test_benchmark_output_paths_are_created_without_real_simulation(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("grasp:\n  n_trials: 1\n", encoding="utf-8")

    def fake_evaluate_design(
        design,
        tools,
        n_grasp_trials,
        output_dir,
        result_dir=None,
        seed=0,
        config=None,
        geometry_config=None,
        backend_name="mujoco_cpu",
        backend=None,
    ):
        payload = {
            "design_id": design.design_id,
            "parameters": design.to_dict(),
            "hand_score": 1.0,
            "tool_results": [{"tool": tools[0], "best_score": 1.0}],
            "failed": False,
        }
        write_json(Path(result_dir) / f"{design.design_id}.json", payload)
        return payload

    monkeypatch.setattr(baseline_benchmark, "evaluate_design", fake_evaluate_design)
    output_dir = tmp_path / "baseline"

    result = run_baseline_benchmark(
        n_designs=2,
        n_grasp_trials=1,
        tools=["hammer"],
        seed=0,
        backend="mujoco_cpu",
        config=config,
        search_space=None,
        output_dir=output_dir,
    )

    assert (output_dir / "designs" / "manifest.json").exists()
    assert (output_dir / "results").is_dir()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "metadata.json").exists()
    assert len(result["rows"]) == 2
