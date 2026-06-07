from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from handcdo.backends.registry import get_backend
from handcdo.benchmarks import mujoco_warp
from handcdo.benchmarks.mujoco_warp import (
    WarpAvailability,
    WarpBenchmarkConfig,
    check_warp_available,
    parse_positive_int_list,
    prepare_warp_compatible_mjcf,
    run_benchmark,
    validate_config,
    write_benchmark_outputs,
)


def test_check_warp_available_does_not_raise():
    availability = check_warp_available()

    assert isinstance(availability, WarpAvailability)
    assert availability.package in {None, "mujoco_warp"}


def test_script_help_works_without_gpu_dependencies():
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_mujoco_warp.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout


def test_missing_warp_without_require_warp_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mujoco_warp,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    result = run_benchmark(WarpBenchmarkConfig(output_dir=tmp_path / "bench", skip_cpu=True))

    assert result["warp_skip_reason"] == "missing for test"
    assert (tmp_path / "bench" / "availability.json").exists()
    assert (tmp_path / "bench" / "benchmark_results.json").exists()
    assert (tmp_path / "bench" / "benchmark_results.csv").exists()


def test_missing_warp_with_require_warp_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mujoco_warp,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    code = mujoco_warp.main(["--output-dir", str(tmp_path / "bench"), "--skip-cpu", "--require-warp"])

    assert code == 1
    assert (tmp_path / "bench" / "availability.json").exists()


def test_validate_config_rejects_invalid_steps(tmp_path):
    with pytest.raises(ValueError, match="steps"):
        validate_config(WarpBenchmarkConfig(output_dir=tmp_path, steps=0))


def test_validate_config_rejects_invalid_nworld(tmp_path):
    with pytest.raises(ValueError, match="nworld"):
        validate_config(WarpBenchmarkConfig(output_dir=tmp_path, nworld=0))


def test_sweep_parser_rejects_invalid_values():
    assert parse_positive_int_list("1,8,32") == (1, 8, 32)
    with pytest.raises(Exception):
        parse_positive_int_list("1,0")
    with pytest.raises(Exception):
        parse_positive_int_list("1,,2")


def test_output_schema_writes_required_csv_columns(tmp_path):
    row = {column: None for column in mujoco_warp.CSV_COLUMNS}
    row.update(
        {
            "backend": "mujoco_cpu",
            "available": False,
            "success": False,
            "failure_count": 1,
        }
    )
    result = {
        "availability": {"warp_available": False},
        "rows": [row],
        "benchmark_schema_version": 1,
    }

    write_benchmark_outputs(result, tmp_path)

    with (tmp_path / "benchmark_results.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert set(mujoco_warp.CSV_COLUMNS).issubset(reader.fieldnames or [])
        rows = list(reader)
    assert rows[0]["backend"] == "mujoco_cpu"


def test_mjcf_rewrite_records_integrator_change(tmp_path):
    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"
    original.write_text(
        '<mujoco model="x"><option integrator="implicitfast" timestep="0.002" /></mujoco>',
        encoding="utf-8",
    )

    result = prepare_warp_compatible_mjcf(original, rewritten)

    assert original.read_text(encoding="utf-8").count("implicitfast") == 1
    assert "Euler" in rewritten.read_text(encoding="utf-8")
    assert result["mjcf_rewrites"] == [
        {
            "field": "option.integrator",
            "old": "implicitfast",
            "new": "Euler",
            "reason": "benchmark-local MuJoCo Warp compatibility",
        }
    ]


def test_mjcf_rewrite_can_be_disabled(tmp_path):
    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"
    text = '<mujoco model="x"><option integrator="implicitfast" timestep="0.002" /></mujoco>'
    original.write_text(text, encoding="utf-8")

    result = prepare_warp_compatible_mjcf(original, rewritten, allow_rewrite=False)

    assert rewritten.read_text(encoding="utf-8") == text
    assert result["mjcf_rewrites"] == []
    assert result["mjcf_files_differ"] is False


def test_backend_registry_does_not_gain_warp_backend():
    with pytest.raises(ValueError, match="Unknown simulator backend"):
        get_backend("mujoco_warp")


def test_cpu_smoke_benchmark_skips_cleanly_if_mujoco_unavailable(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    del mujoco

    result = mujoco_warp.run_cpu_mujoco_timing(
        tmp_path / "missing.xml",
        steps=1,
        warmup_steps=0,
        repeats=1,
        scene_mode="load_step",
        seed=0,
    )

    assert result["backend"] == "mujoco_cpu"
    assert result["success"] is False
    assert result["failure_stage"] == "load_model"


def test_optional_slurm_helpers_contain_expected_scheduler_profiles():
    capella = open("slurm/mujoco_warp_capella_smoke.sbatch", encoding="utf-8").read()
    alpha = open("slurm/mujoco_warp_alpha_sweep.sbatch", encoding="utf-8").read()

    assert "#SBATCH --partition=capella" in capella
    assert "#SBATCH --gres=gpu:4" in capella
    assert "#SBATCH --partition=alpha" in alpha
    assert "#SBATCH --gres=gpu:8" in alpha
