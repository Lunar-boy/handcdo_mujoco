from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from handcdo.design_space import DesignSpace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "propose_surrogate_candidates.py"


def _write_training_csv(path: Path, n_rows: int = 8) -> None:
    rows = []
    for seed in range(n_rows):
        design = DesignSpace().sample(seed=seed)
        rows.append(
            {
                "design_id": design.design_id,
                "best_available_score": float(seed),
                "hand_score": float(seed),
                "failed": False,
                **design.to_dict(),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_train_only_writes_model_metadata_diagnostics_and_no_proposals(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_training_csv(results_csv)

    result = _run_cli(
        "--mode",
        "train-only",
        "--results-csv",
        str(results_csv),
        "--output-dir",
        str(tmp_path / "out"),
        "--seed",
        "0",
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "model" / "surrogate_model.joblib").exists()
    assert (tmp_path / "out" / "model" / "surrogate_metadata.json").exists()
    assert (tmp_path / "out" / "model" / "surrogate_diagnostics.json").exists()
    assert not (tmp_path / "out" / "proposed_candidates.csv").exists()
    assert not (tmp_path / "out" / "manifest.json").exists()
    assert not (tmp_path / "out" / "proposed_designs").exists()
    assert "diagnostics" in result.stdout


def test_cli_propose_only_uses_existing_model(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_training_csv(results_csv)
    train_result = _run_cli(
        "--mode",
        "train-only",
        "--results-csv",
        str(results_csv),
        "--output-dir",
        str(tmp_path / "model_out"),
    )
    assert train_result.returncode == 0, train_result.stderr

    result = _run_cli(
        "--mode",
        "propose-only",
        "--model-path",
        str(tmp_path / "model_out" / "model" / "surrogate_model.joblib"),
        "--n-random",
        "12",
        "--top-k",
        "3",
        "--output-dir",
        str(tmp_path / "proposal"),
        "--no-exclude-existing",
    )

    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(tmp_path / "proposal" / "proposed_candidates.csv")) == 3
    assert len(list((tmp_path / "proposal" / "proposed_designs").glob("*/design.json"))) == 3


def test_cli_train_propose_writes_model_and_proposals(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_training_csv(results_csv)

    result = _run_cli(
        "--mode",
        "train-propose",
        "--results-csv",
        str(results_csv),
        "--n-random",
        "12",
        "--top-k",
        "3",
        "--output-dir",
        str(tmp_path / "out"),
        "--no-exclude-existing",
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "model" / "surrogate_model.joblib").exists()
    assert (tmp_path / "out" / "model" / "surrogate_diagnostics.json").exists()
    assert len(pd.read_csv(tmp_path / "out" / "proposed_candidates.csv")) == 3


def test_cli_missing_required_args_fail(tmp_path):
    train_only = _run_cli("--mode", "train-only", "--output-dir", str(tmp_path / "out"))
    propose_only = _run_cli("--mode", "propose-only", "--output-dir", str(tmp_path / "out"))
    bad_top_k = _run_cli(
        "--mode",
        "propose-only",
        "--model-path",
        str(tmp_path / "missing.joblib"),
        "--n-random",
        "2",
        "--top-k",
        "3",
        "--output-dir",
        str(tmp_path / "out"),
    )

    assert train_only.returncode != 0
    assert "--results-csv" in train_only.stderr
    assert propose_only.returncode != 0
    assert "--model-path" in propose_only.stderr
    assert bad_top_k.returncode != 0
    assert "--top-k must be <= --n-random" in bad_top_k.stderr


def test_cli_overwrite_allows_reusing_proposal_output_dir(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_training_csv(results_csv)
    train_result = _run_cli(
        "--mode",
        "train-only",
        "--results-csv",
        str(results_csv),
        "--output-dir",
        str(tmp_path / "model_out"),
    )
    assert train_result.returncode == 0, train_result.stderr
    model_path = tmp_path / "model_out" / "model" / "surrogate_model.joblib"

    base_args = [
        "--mode",
        "propose-only",
        "--model-path",
        str(model_path),
        "--n-random",
        "12",
        "--top-k",
        "3",
        "--output-dir",
        str(tmp_path / "proposal"),
        "--no-exclude-existing",
    ]
    first = _run_cli(*base_args)
    second = _run_cli(*base_args)
    third = _run_cli(*base_args, "--overwrite")

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "already exists" in second.stderr or "non-empty" in second.stderr
    assert third.returncode == 0, third.stderr
