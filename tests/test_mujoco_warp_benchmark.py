from __future__ import annotations

import csv
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
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


def test_prepare_warp_compatible_mjcf_rewrites_nonzero_margins(tmp_path):
    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"
    original.write_text(
        """<mujoco model="test">
  <option integrator="implicitfast"/>
  <default>
    <geom margin="0.001" solref="0.012 1"/>
  </default>
  <worldbody>
    <body name="body">
      <geom name="g1" type="box" size="0.1 0.1 0.1" margin="0.002"/>
      <geom name="g2" type="box" size="0.1 0.1 0.1" margin="0"/>
      <geom name="g3" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2" margin="0.003"/>
  </contact>
</mujoco>
""",
        encoding="utf-8",
    )

    result = prepare_warp_compatible_mjcf(original, rewritten, allow_rewrite=True)

    text = rewritten.read_text(encoding="utf-8")
    assert 'integrator="Euler"' in text
    assert 'margin="0.001"' not in text
    assert 'margin="0.002"' not in text
    assert 'margin="0.003"' not in text
    assert 'margin="0"' in text
    assert "margin=" in text

    rewrites = result["mjcf_rewrites"]
    assert any(rewrite["field"] == "option.integrator" for rewrite in rewrites)
    assert any(rewrite["field"] == "geom.margin" and rewrite["old"] == "0.001" for rewrite in rewrites)
    assert any(
        rewrite["field"] == "geom.margin" and rewrite["name"] == "g1" and rewrite["old"] == "0.002"
        for rewrite in rewrites
    )
    assert any(
        rewrite["field"] == "pair.margin"
        and rewrite["geom1"] == "g1"
        and rewrite["geom2"] == "g2"
        and rewrite["old"] == "0.003"
        for rewrite in rewrites
    )


def test_mjcf_rewrite_can_be_disabled(tmp_path):
    original = tmp_path / "original.xml"
    rewritten = tmp_path / "warp.xml"
    text = '<mujoco model="x"><option integrator="implicitfast" timestep="0.002" /><default><geom margin="0.001" /></default></mujoco>'
    original.write_text(text, encoding="utf-8")

    result = prepare_warp_compatible_mjcf(original, rewritten, allow_rewrite=False)

    assert rewritten.read_text(encoding="utf-8") == text
    assert 'margin="0.001"' in rewritten.read_text(encoding="utf-8")
    assert result["mjcf_rewrites"] == []
    assert result["mjcf_files_differ"] is False


def test_backend_registry_exposes_warp_backend_lazily_when_requested(monkeypatch):
    from handcdo import warp_utils
    from handcdo.backends.mujoco_warp import MujocoWarpUnavailableError

    monkeypatch.setattr(
        warp_utils,
        "check_warp_available",
        lambda: WarpAvailability(False, "missing for test", "mujoco_warp", None),
    )

    with pytest.raises(MujocoWarpUnavailableError, match="optional warp extra"):
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


def test_warp_timing_uses_dedicated_smoke_data_for_capability_probe(monkeypatch, tmp_path):
    class FakeModel:
        nconmax = 0
        njmax = 16

        @classmethod
        def from_xml_path(cls, path):
            return cls()

    class FakeData:
        ncon = 0
        nefc = 0

        def __init__(self, model):
            self.model = model

    fake_mujoco = SimpleNamespace(
        MjModel=FakeModel,
        MjData=FakeData,
        mj_forward=lambda model, data: None,
        mj_step=lambda model, data: None,
    )
    fake_mjw = SimpleNamespace(
        put_model=lambda model: SimpleNamespace(kind="warp_model"),
        step=lambda model, data: setattr(data, "stepped", True),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco_warp", fake_mjw)
    monkeypatch.setattr(mujoco_warp, "synchronize_warp", lambda: (True, None))

    data_objects = []

    def fake_make_warp_data(*args, **kwargs):
        data = SimpleNamespace(
            qpos=np.zeros((2, 1)),
            qvel=np.zeros((2, 1)),
            ctrl=np.zeros((2, 1)),
            xfrc_applied=np.zeros((2, 1, 6)),
            stepped=False,
        )
        data_objects.append(data)
        return data

    monkeypatch.setattr(mujoco_warp, "make_warp_data", fake_make_warp_data)

    row = mujoco_warp.run_warp_timing(
        tmp_path / "model.xml",
        steps=1,
        warmup_steps=1,
        repeats=1,
        scene_mode="load_step",
        seed=0,
        nworld=2,
        nconmax=8,
        naconmax=None,
        njmax=16,
    )

    assert row["success"] is True
    assert len(data_objects) == 2
    smoke_data, timing_data = data_objects
    assert smoke_data.stepped is False
    assert timing_data.stepped is True
    assert row["warp_capabilities"]["can_set_per_world_qpos"] is True
    assert row["capability_probe_error"] is None


def test_optional_slurm_helpers_contain_expected_scheduler_profiles():
    capella = open("slurm/mujoco_warp_capella_smoke.sbatch", encoding="utf-8").read()
    #alpha = open("slurm/mujoco_warp_alpha_sweep.sbatch", encoding="utf-8").read()


    # Capella PR10 smoke should be lightweight.
    assert "#SBATCH --partition=capella" in capella
    assert "#SBATCH --nodes=1" in capella
    assert "#SBATCH --ntasks=1" in capella
    assert "#SBATCH --cpus-per-task=8" in capella
    assert "#SBATCH --mem=64000" in capella
    assert "#SBATCH --gres=gpu:1" in capella
    assert "#SBATCH --time=00:30:00" in capella

    # The script must not use system Python for project commands.
    assert "VENV_PATH=" in capella
    assert "${VENV_PATH}/bin/python" in capella or '"${PYTHON}"' in capella
    assert "python3 -m pytest" not in capella
    assert "${PYTHON}\" -m pytest" in capella or '"${PYTHON}" -m pytest' in capella
