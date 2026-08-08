import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from riskprobe.cli import app


runner = CliRunner()


def _write_config(path: Path, data_path: Path) -> None:
    path.write_text(
        f"""dataset:
  id: synthetic-demo
  path: {data_path}
  read_only: true
columns:
  entity: entity_id
  snapshot: snapshot_date
  segment: institution
  target: target
target:
  positive_value: 1
  positive_meaning: bad_debt
  performance_window_days: null
snapshot:
  meaning: public_relative_reference
features:
  families:
    order: [order_]
    browse: [browse_]
    platform: [multi_platform_]
    embedding: [emb_]
segment_display_name: institution
time_validation_enabled: true
discovery:
  min_support: 0.05
  max_single_rules: 8
  beam_width: 4
  max_pair_rules: 4
  random_seed: 42
validation:
  alpha: 0.05
  min_segment_consistency: 0.6
  max_lift_decay: 0.3
  bootstrap_rounds: 100
  min_group_size: 20
""",
        encoding="utf-8",
    )


def test_synthetic_then_run(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.parquet"
    synthetic = runner.invoke(
        app,
        ["synthetic", "--output", str(data_path), "--rows", "5000", "--seed", "42"],
    )

    assert synthetic.exit_code == 0
    assert data_path.exists()
    synthetic_payload = json.loads(synthetic.stdout)
    assert synthetic_payload == {
        "columns": 16,
        "command": "synthetic",
        "rows": 5000,
        "truth_rule_ids": [
            "hidden_order_cancellation",
            "hidden_night_browsing",
            "hidden_multi_platform_low_order",
        ],
    }
    assert "bank_north" not in synthetic.stdout
    assert "entity_id" not in synthetic.stdout

    config_path = tmp_path / "demo.yaml"
    _write_config(config_path, data_path)
    run = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--runs-dir", str(tmp_path / "runs")],
    )

    assert run.exit_code == 0, run.stdout
    run_payload = json.loads(run.stdout)
    assert run_payload["command"] == "run"
    assert run_payload["metadata_grade"] == "B"
    assert run_payload["artifact_count"] == 6
    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "metadata_report.json").exists()
    assert json.loads((run_dirs[0] / "metadata_report.json").read_text())["metadata_grade"] == "B"


def test_command_parsing_lists_all_supported_commands_and_rejects_missing_options() -> None:
    help_result = runner.invoke(app, ["--help"])
    missing_output = runner.invoke(app, ["synthetic", "--rows", "10", "--seed", "42"])

    assert help_result.exit_code == 0
    for command in ("synthetic", "inspect", "discover", "run"):
        assert command in help_result.stdout
    assert missing_output.exit_code == 2


def test_inspect_and_discover_emit_safe_json_summaries(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.parquet"
    assert runner.invoke(
        app,
        ["synthetic", "--output", str(data_path), "--rows", "5000", "--seed", "42"],
    ).exit_code == 0
    config_path = tmp_path / "demo.yaml"
    _write_config(config_path, data_path)

    inspect = runner.invoke(
        app,
        ["inspect", "--config", str(config_path), "--runs-dir", str(tmp_path / "runs")],
    )
    discover = runner.invoke(
        app,
        ["discover", "--config", str(config_path), "--runs-dir", str(tmp_path / "runs")],
    )

    assert inspect.exit_code == 0, inspect.stdout
    assert discover.exit_code == 0, discover.stdout
    inspect_payload = json.loads(inspect.stdout)
    discover_payload = json.loads(discover.stdout)
    assert inspect_payload == {
        "command": "inspect",
        "feature_count": 12,
        "metadata_grade": "B",
        "row_count": 5000,
        "segment_count": 4,
    }
    assert discover_payload["command"] == "discover"
    assert discover_payload["candidate_rule_count"] == len(discover_payload["rule_ids"])
    assert "bank_north" not in inspect.stdout + discover.stdout
    assert str(data_path) not in inspect.stdout + discover.stdout


def test_command_failure_is_structured_actionable_and_does_not_leak_paths(tmp_path: Path) -> None:
    private_config = tmp_path / "private-client-config.yaml"
    result = runner.invoke(
        app,
        ["inspect", "--config", str(private_config), "--runs-dir", str(tmp_path / "runs")],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload == {
        "error": "configuration_error",
        "message": "Check that --config names a readable, valid local YAML configuration.",
    }
    assert str(private_config) not in result.stdout


def test_synthetic_is_deterministic_and_overwrites_existing_output(tmp_path: Path) -> None:
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first = runner.invoke(
        app,
        ["synthetic", "--output", str(first_path), "--rows", "250", "--seed", "7"],
    )
    first_path.write_bytes(b"not parquet")
    overwrite = runner.invoke(
        app,
        ["synthetic", "--output", str(first_path), "--rows", "250", "--seed", "7"],
    )
    second = runner.invoke(
        app,
        ["synthetic", "--output", str(second_path), "--rows", "250", "--seed", "7"],
    )

    assert first.exit_code == overwrite.exit_code == second.exit_code == 0
    assert pl.read_parquet(first_path).equals(pl.read_parquet(second_path))
    assert first_path.read_bytes() == second_path.read_bytes()


def test_public_example_config_declares_grade_b_synthetic_contract() -> None:
    example = Path("configs/synthetic.example.yaml").read_text(encoding="utf-8")

    assert "data/synthetic/behavior.parquet" in example
    assert "entity: entity_id" in example
    assert "snapshot: snapshot_date" in example
    assert "segment: institution" in example
    assert "target: target" in example
    assert "performance_window_days: null" in example
    for prefix in ("order_", "browse_", "multi_platform_", "emb_"):
        assert prefix in example


def test_parser_failures_are_safe_structured_errors() -> None:
    cases = (
        (["synthetic", "--output", "/tmp/ignored.parquet", "--rows", "NaN", "--seed", "42"], "NaN"),
        (["synthetic", "--output", "/tmp/ignored.parquet", "--rows", "1", "--seed", "42", "--unknown"], "--unknown"),
    )

    for arguments, private_input in cases:
        result = runner.invoke(app, arguments)

        assert result.exit_code == 2
        assert json.loads(result.stdout) == {
            "error": "argument_error",
            "message": "Use --help to review the required command and option values.",
        }
        assert private_input not in result.stdout


def test_unusable_runs_directory_is_a_safe_actionable_error(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.parquet"
    assert runner.invoke(
        app,
        ["synthetic", "--output", str(data_path), "--rows", "100", "--seed", "42"],
    ).exit_code == 0
    config_path = tmp_path / "demo.yaml"
    _write_config(config_path, data_path)
    runs_file = tmp_path / "not-a-directory"
    runs_file.write_text("not a directory", encoding="utf-8")

    result = runner.invoke(
        app,
        ["inspect", "--config", str(config_path), "--runs-dir", str(runs_file)],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "error": "runs_directory_error",
        "message": "Choose a writable local --runs-dir directory.",
    }
    assert str(runs_file) not in result.stdout


def test_evaluate_drift_rejects_repository_runs_dir() -> None:
    result = runner.invoke(
        app,
        [
            "evaluate-drift",
            "--config",
            "configs/synthetic.example.yaml",
            "--runs-dir",
            "runs",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "evaluation_error"
