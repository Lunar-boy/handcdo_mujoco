from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from handcdo.design_space import DesignSpace
from handcdo.mujoco_eval import GraspEvaluation
from handcdo.warp_utils import WarpAvailability


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_design_batch_warp.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_design_batch_warp_test_module", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_design(tmp_path: Path):
    design = DesignSpace().sample(seed=21)
    design_dir = tmp_path / "designs" / design.design_id
    design_dir.mkdir(parents=True)
    design.to_json(design_dir / "design.json")
    return design


def _args(module, tmp_path: Path, *extra: str):
    return module.build_parser().parse_args(
        [
            "--design-dir",
            str(tmp_path / "designs"),
            "--results-dir",
            str(tmp_path / "warp_results"),
            "--config",
            "configs/eval_fast.yaml",
            "--tools",
            "hammer",
            "--n-grasp-trials",
            "2",
            "--nworld",
            "2",
            *extra,
        ]
    )


def test_help_import_does_not_import_mujoco_warp(capsys):
    sys.modules.pop("mujoco_warp", None)
    module = _load_script()

    with pytest.raises(SystemExit) as exc_info:
        module.build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "mujoco_warp" not in sys.modules
    assert "Experimental MuJoCo Warp" in capsys.readouterr().out


def test_sampler_tpe_is_rejected_by_argparse():
    module = _load_script()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--results-dir", "out", "--sampler", "tpe"])


def test_missing_warp_writes_skipped_experimental_result(tmp_path, monkeypatch):
    design = _write_design(tmp_path)
    module = _load_script()
    args = _args(module, tmp_path)
    monkeypatch.setattr(
        module,
        "check_warp_available",
        lambda: WarpAvailability(False, "ModuleNotFoundError: No module named 'mujoco_warp'", "mujoco_warp", None),
    )

    payloads = module.run(args)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["design_id"] == design.design_id
    assert payload["failed"] is True
    assert payload["backend"] == "mujoco_warp"
    assert payload["experimental"] is True
    assert payload["include_in_multifidelity"] is False
    assert payload["score_semantics"] == "experimental_non_equivalent"
    assert "python3 -m pip install -e" in payload["error"]
    assert payload["warp_availability"]["warp_available"] is False
    metadata = payload["warp_metadata"]
    assert metadata["nworld"] == 2
    assert metadata["batch_size"] == 2
    assert metadata["num_grasps"] == 2
    assert metadata["num_chunks"] == 1
    assert metadata["failure_count"] == 2

    written = tmp_path / "warp_results" / f"{design.design_id}.mujoco_warp.experimental.json"
    assert json.loads(written.read_text(encoding="utf-8")) == payload


def test_require_warp_missing_fails_without_writing_result(tmp_path, monkeypatch):
    design = _write_design(tmp_path)
    module = _load_script()
    args = _args(module, tmp_path, "--require-warp")
    monkeypatch.setattr(
        module,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    with pytest.raises(RuntimeError, match="MuJoCo Warp is unavailable"):
        module.run(args)

    assert not (tmp_path / "warp_results" / f"{design.design_id}.mujoco_warp.experimental.json").exists()


def test_refuses_to_overwrite_existing_results_without_flag(tmp_path, monkeypatch):
    design = _write_design(tmp_path)
    module = _load_script()
    results_dir = tmp_path / "warp_results"
    results_dir.mkdir()
    (results_dir / f"{design.design_id}.mujoco_warp.experimental.json").write_text("{}", encoding="utf-8")
    args = _args(module, tmp_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        module.run(args)


def test_cpu_style_json_in_results_dir_fails_by_default_without_writing(tmp_path, monkeypatch):
    design = _write_design(tmp_path)
    module = _load_script()
    results_dir = tmp_path / "warp_results"
    results_dir.mkdir()
    (results_dir / "design_0001.json").write_text("{}", encoding="utf-8")
    args = _args(module, tmp_path)

    with pytest.raises(FileExistsError) as exc_info:
        module.run(args)

    message = str(exc_info.value)
    assert "CPU result JSON" in message
    assert "Experimental MuJoCo Warp results are intentionally separated" in message
    assert "--allow-mixed-backend-dir" in message
    assert not (results_dir / f"{design.design_id}.mujoco_warp.experimental.json").exists()


def test_main_reports_mixed_backend_dir_error_before_writing(tmp_path, capsys):
    design = _write_design(tmp_path)
    module = _load_script()
    results_dir = tmp_path / "warp_results"
    results_dir.mkdir()
    (results_dir / "design_0001.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        module.main(
            [
                "--design-dir",
                str(tmp_path / "designs"),
                "--results-dir",
                str(results_dir),
                "--config",
                "configs/eval_fast.yaml",
                "--tools",
                "hammer",
                "--n-grasp-trials",
                "2",
            ]
        )

    assert exc_info.value.code == 1
    assert "--allow-mixed-backend-dir" in capsys.readouterr().err
    assert not (results_dir / f"{design.design_id}.mujoco_warp.experimental.json").exists()


def test_allow_mixed_backend_dir_writes_skipped_result_next_to_cpu_style_json(tmp_path, monkeypatch):
    design = _write_design(tmp_path)
    module = _load_script()
    results_dir = tmp_path / "warp_results"
    results_dir.mkdir()
    (results_dir / "design_0001.json").write_text("{}", encoding="utf-8")
    args = _args(module, tmp_path, "--allow-mixed-backend-dir")
    monkeypatch.setattr(
        module,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    payload = module.run(args)[0]

    assert payload["failed"] is True
    assert payload["include_in_multifidelity"] is False
    assert (results_dir / f"{design.design_id}.mujoco_warp.experimental.json").exists()


def test_existing_warp_experimental_json_is_not_cpu_style_result(tmp_path):
    module = _load_script()
    results_dir = tmp_path / "warp_results"
    results_dir.mkdir()
    warp_result = results_dir / "other_design.mujoco_warp.experimental.json"
    warp_result.write_text("{}", encoding="utf-8")

    assert module._find_cpu_style_result_jsons(results_dir) == []
    module._validate_results_dir_for_warp(results_dir, allow_mixed_backend_dir=False)


def test_available_warp_path_uses_backend_and_writes_schema(tmp_path, monkeypatch):
    design = _write_design(tmp_path)
    module = _load_script()
    args = _args(module, tmp_path)
    monkeypatch.setattr(
        module,
        "check_warp_available",
        lambda: WarpAvailability(True, None, "mujoco_warp", "test"),
    )

    class DummyWarpBackend:
        name = "mujoco_warp"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.last_batch_metadata = {}

        def evaluate_grasps_batch(self, design, tool_name, grasps, config, geometry_config=None, tool_assets_dir="assets/tools"):
            self.last_batch_metadata = {
                "nworld": self.kwargs["nworld"],
                "nconmax": self.kwargs["nconmax"],
                "naconmax": self.kwargs["naconmax"],
                "njmax": self.kwargs["njmax"],
                "num_grasps": len(grasps),
                "num_chunks": 1,
                "failure_count": 0,
                "seconds_total": 0.5,
                "grasps_per_second": 4.0,
                "world_steps_per_second": None,
                "sequential_fallback": False,
                "mjcf_rewrites": [],
            }
            return [
                GraspEvaluation(
                    design.design_id,
                    tool_name,
                    grasp.to_dict(),
                    float(index + 1),
                    [],
                    failed=False,
                )
                for index, grasp in enumerate(grasps)
            ]

    monkeypatch.setitem(
        sys.modules,
        "handcdo.backends.mujoco_warp",
        type(sys)("handcdo.backends.mujoco_warp"),
    )
    sys.modules["handcdo.backends.mujoco_warp"].MujocoWarpBackend = DummyWarpBackend

    payload = module.run(args)[0]

    assert payload["failed"] is False
    assert payload["include_in_multifidelity"] is False
    assert payload["hand_score"] == 2.0
    assert payload["tool_results"][0]["tool"] == "hammer"
    assert payload["tool_results"][0]["best_score"] == 2.0
    assert payload["tool_results"][0]["failure_count"] == 0
    assert payload["tool_results"][0]["warp_metadata"]["num_grasps"] == 2
    metadata = payload["warp_metadata"]
    assert metadata["nworld"] == 2
    assert metadata["batch_size"] == 2
    assert metadata["num_grasps"] == 2
    assert metadata["num_chunks"] == 1
    assert metadata["failure_count"] == 0
    assert metadata["warmup_steps"] == 0
    assert metadata["capture_graph"] is False
    assert payload["score_semantics"] != "intended_cpu_equivalent"
