from __future__ import annotations

import csv
import subprocess
import sys

from handcdo.benchmarks import mujoco_warp
from handcdo.warp_utils import (
    WarpAvailability,
    availability_payload,
    check_warp_available,
    prepare_warp_compatible_mjcf,
    synchronize_warp,
)


def test_warp_utils_availability_checks_do_not_raise_without_warp():
    availability = check_warp_available()
    payload = availability_payload(availability)
    synchronized, warning = synchronize_warp()

    assert isinstance(availability, WarpAvailability)
    assert payload["package"] in {None, "mujoco_warp"}
    assert payload["warp_available"] is availability.available
    assert isinstance(synchronized, bool)
    assert warning is None or isinstance(warning, str)


def test_prepare_warp_compatible_mjcf_is_available_from_utils(tmp_path):
    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"
    original.write_text('<mujoco model="x"><option integrator="implicitfast" /></mujoco>', encoding="utf-8")

    result = prepare_warp_compatible_mjcf(original, rewritten)

    assert result["mjcf_rewrites"][0]["field"] == "option.integrator"
    assert 'integrator="Euler"' in rewritten.read_text(encoding="utf-8")


def test_existing_benchmark_help_still_works_without_importing_warp_at_startup():
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_mujoco_warp.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "--require-warp" in result.stdout


def test_pr10_benchmark_csv_columns_remain_unchanged(tmp_path):
    expected_columns = (
        "backend",
        "available",
        "success",
        "scene_mode",
        "nworld",
        "nconmax",
        "naconmax",
        "njmax",
        "seconds_mean",
        "seconds_std",
        "steps",
        "warmup_steps",
        "repeats",
        "total_sim_steps",
        "total_world_steps",
        "steps_per_second_total",
        "world_steps_per_second",
        "steps_per_second_per_world",
        "failure_count",
        "failure_stage",
        "exception_type",
        "error",
    )
    row = {column: None for column in mujoco_warp.CSV_COLUMNS}
    row.update({"backend": "mujoco_warp", "available": False, "success": False})
    result = {
        "availability": {"warp_available": False},
        "rows": [row],
        "benchmark_schema_version": 1,
    }

    mujoco_warp.write_benchmark_outputs(result, tmp_path)

    with (tmp_path / "benchmark_results.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == expected_columns
