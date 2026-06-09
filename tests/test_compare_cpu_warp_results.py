from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from handcdo.compare_cpu_warp_results import compare_cpu_warp_results
from handcdo.utils import write_json


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_cpu_warp_results.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_cpu_warp_results_test_module", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpu_payload(design_id: str, score: float, *, tool: str = "hammer", backend: str = "mujoco_cpu"):
    return {
        "design_id": design_id,
        "hand_score": score,
        "tool_results": [{"tool": tool, "best_score": score + 0.1, "failure_count": 1}],
        "failed": False,
        "backend": backend,
    }


def _warp_payload(
    design_id: str,
    score: float,
    *,
    tool: str = "hammer",
    experimental: bool | None = True,
    score_semantics: str | None = "experimental_non_equivalent",
):
    payload = {
        "design_id": design_id,
        "hand_score": score,
        "tool_results": [{"tool": tool, "best_score": score + 0.1, "failure_count": 2}],
        "failed": False,
        "backend": "mujoco_warp",
        "include_in_multifidelity": False,
    }
    if experimental is not None:
        payload["experimental"] = experimental
    if score_semantics is not None:
        payload["score_semantics"] = score_semantics
    return payload


def test_help_import_does_not_import_mujoco_warp(capsys):
    sys.modules.pop("mujoco_warp", None)
    module = _load_script()

    try:
        module.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "mujoco_warp" not in sys.modules
    assert "Compare CPU MuJoCo reference results" in capsys.readouterr().out


def test_empty_directories_produce_controlled_summary(tmp_path):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()

    summary, exit_code = compare_cpu_warp_results(cpu_dir, warp_dir)

    assert exit_code == 0
    assert summary["num_cpu_designs"] == 0
    assert summary["num_warp_designs"] == 0
    assert summary["num_matched_designs"] == 0
    assert summary["overall"]["mean_abs_score_diff"] is None
    assert summary["overall"]["top_k_overlap"] is None
    assert summary["overall"]["top_k_cpu_recall_in_warp"] is None
    assert summary["overall"]["rank_sign_flip_count"] == 0
    assert any("Fewer than 3 matched" in warning for warning in summary["warnings"])


def test_matched_fake_jsons_produce_expected_score_difference_and_output(tmp_path):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "design_a.json", _cpu_payload("design_a", 1.0))
    write_json(warp_dir / "design_a.mujoco_warp.experimental.json", _warp_payload("design_a", 1.25))
    out = tmp_path / "comparison.json"

    summary, exit_code = compare_cpu_warp_results(cpu_dir, warp_dir, out=out)

    assert exit_code == 0
    assert summary["overall"]["mean_abs_score_diff"] == 0.25
    assert summary["overall"]["max_abs_score_diff"] == 0.25
    assert "top_k_cpu_recall_in_warp" in summary["overall"]
    assert "rank_sign_flip_count" in summary["overall"]
    assert summary["by_design"][0]["signed_score_diff"] == 0.25
    assert summary["by_tool"]["hammer"]["failure_count_diff"] == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "top_k_cpu_recall_in_warp" in payload["overall"]
    assert "rank_sign_flip_count" in payload["overall"]
    assert payload == summary


def test_missing_designs_are_reported_in_each_direction(tmp_path):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "cpu_only.json", _cpu_payload("cpu_only", 1.0))
    write_json(warp_dir / "warp_only.mujoco_warp.experimental.json", _warp_payload("warp_only", 1.0))

    summary, exit_code = compare_cpu_warp_results(cpu_dir, warp_dir)

    assert exit_code == 0
    assert summary["missing_in_cpu"] == ["warp_only"]
    assert summary["missing_in_warp"] == ["cpu_only"]


def test_warning_emitted_when_warp_result_lacks_experimental_true(tmp_path, capsys):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "design_a.json", _cpu_payload("design_a", 1.0))
    write_json(
        warp_dir / "design_a.mujoco_warp.experimental.json",
        _warp_payload("design_a", 1.0, experimental=None),
    )

    summary, _ = compare_cpu_warp_results(cpu_dir, warp_dir)

    assert any("lacks experimental=true" in warning for warning in summary["warnings"])
    assert "lacks experimental=true" in capsys.readouterr().out


def test_warning_emitted_when_warp_claims_intended_cpu_equivalent(tmp_path, capsys):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "design_a.json", _cpu_payload("design_a", 1.0))
    write_json(
        warp_dir / "design_a.mujoco_warp.experimental.json",
        _warp_payload("design_a", 1.0, score_semantics="intended_cpu_equivalent"),
    )

    summary, _ = compare_cpu_warp_results(cpu_dir, warp_dir)

    assert any("claims intended_cpu_equivalent" in warning for warning in summary["warnings"])
    assert "claims intended_cpu_equivalent" in capsys.readouterr().out


def test_fail_on_missing_exits_nonzero_when_designs_are_missing(tmp_path):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "cpu_only.json", _cpu_payload("cpu_only", 1.0))

    summary, exit_code = compare_cpu_warp_results(cpu_dir, warp_dir, fail_on_missing=True)

    assert summary["missing_in_warp"] == ["cpu_only"]
    assert exit_code == 1


def test_fail_thresholds_use_max_score_diff_and_max_rank_displacement(tmp_path):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "a.json", _cpu_payload("a", 3.0))
    write_json(cpu_dir / "b.json", _cpu_payload("b", 2.0))
    write_json(cpu_dir / "c.json", _cpu_payload("c", 1.0))
    write_json(warp_dir / "a.mujoco_warp.experimental.json", _warp_payload("a", 1.0))
    write_json(warp_dir / "b.mujoco_warp.experimental.json", _warp_payload("b", 3.0))
    write_json(warp_dir / "c.mujoco_warp.experimental.json", _warp_payload("c", 2.0))

    summary, exit_code = compare_cpu_warp_results(
        cpu_dir,
        warp_dir,
        fail_on_score_diff=1.5,
        fail_on_rank_drift=1.0,
    )

    assert summary["overall"]["max_abs_score_diff"] == 2.0
    assert summary["overall"]["max_abs_rank_displacement"] == 2.0
    assert exit_code == 1


def test_top_k_cpu_recall_and_rank_sign_flip_count_use_rank_semantics(tmp_path):
    cpu_dir = tmp_path / "cpu"
    warp_dir = tmp_path / "warp"
    cpu_dir.mkdir()
    warp_dir.mkdir()
    write_json(cpu_dir / "a.json", _cpu_payload("a", 0.9))
    write_json(cpu_dir / "b.json", _cpu_payload("b", 0.8))
    write_json(cpu_dir / "c.json", _cpu_payload("c", 0.7))
    write_json(warp_dir / "a.mujoco_warp.experimental.json", _warp_payload("a", 0.10))
    write_json(warp_dir / "b.mujoco_warp.experimental.json", _warp_payload("b", 0.95))
    write_json(warp_dir / "c.mujoco_warp.experimental.json", _warp_payload("c", 0.85))

    summary, exit_code = compare_cpu_warp_results(cpu_dir, warp_dir, top_k=2)

    assert exit_code == 0
    overall = summary["overall"]
    assert overall["top_k_overlap"]["overlap_ratio"] == 0.5
    assert overall["top_k_cpu_recall_in_warp"] == 0.5
    assert overall["rank_sign_flip_count"] == 2


def test_subprocess_help_does_not_require_mujoco_warp():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--cpu-results-dir" in result.stdout
    assert "No module named 'mujoco_warp'" not in result.stderr
