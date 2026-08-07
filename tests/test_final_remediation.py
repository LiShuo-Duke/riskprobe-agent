import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset
from riskprobe.profiling import profile_dataset
from riskprobe.reporting import render_risk_report
from riskprobe.rules.discovery import discover_rules
from riskprobe.service import RiskProbeService
from riskprobe.synthetic import generate_behavior_dataset
from typer.testing import CliRunner

from riskprobe.cli import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
runner = CliRunner()


def _config(data_path: Path, *, explicit_catalog: Path | None = None) -> ProjectConfig:
    features: dict[str, object] = {
        "families": {
            "order": ["order_"],
            "browse": ["browse_"],
            "platform": ["multi_platform_"],
        }
    }
    if explicit_catalog is not None:
        features["explicit_catalog"] = explicit_catalog
    return ProjectConfig.model_validate(
        {
            "dataset": {"id": "final-remediation", "path": data_path},
            "columns": {
                "entity": "entity_id",
                "snapshot": "snapshot_date",
                "segment": "institution",
                "target": "target",
            },
            "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
            "snapshot": {"meaning": "customer_specified_feature_cutoff"},
            "features": features,
            "time_validation_enabled": False,
            "discovery": {
                "min_support": 0.05,
                "max_single_rules": 20,
                "beam_width": 12,
                "max_pair_rules": 20,
                "random_seed": 42,
            },
            "validation": {
                "bootstrap_rounds": 100,
                "min_group_size": 20,
            },
        }
    )


def _wide_dataset(path: Path) -> None:
    pl.DataFrame(
        {
            "entity_id": [f"private-{index}" for index in range(100)],
            "snapshot_date": ["not-a-date"] * 100,
            "institution": ["private-segment"] * 100,
            "target": [index % 2 for index in range(100)],
            "order_signal": [float(index % 10) for index in range(100)],
            "browse_signal": [float(index % 7) for index in range(100)],
            "undeclared_wide_feature": [float(index) for index in range(100)],
        }
    ).write_parquet(path)


def test_profile_discovery_and_partitions_exclude_undeclared_wide_columns(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "wide.parquet"
    _wide_dataset(data_path)
    config = _config(data_path)
    requested_columns: list[tuple[str, ...]] = []
    original_collect = ParquetDataset.collect

    def capture_collect(self: ParquetDataset, columns: list[str]) -> pl.DataFrame:
        requested_columns.append(tuple(columns))
        return original_collect(self, columns)

    monkeypatch.setattr(ParquetDataset, "collect", capture_collect)
    profile = profile_dataset(ParquetDataset(data_path), config)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")
    service.discover()

    assert profile.feature_count == 2
    assert requested_columns
    assert all("undeclared_wide_feature" not in columns for columns in requested_columns)
    assert {"snapshot_date", "institution", "target"}.issubset(
        set().union(*map(set, requested_columns))
    )


def test_explicit_catalog_is_resolved_relative_to_config_and_overrides_prefixes(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "wide.parquet"
    _wide_dataset(data_path)
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text("features:\n  - browse_signal\n", encoding="utf-8")
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        f"""dataset:
  id: final-remediation
  path: {data_path}
columns:
  entity: entity_id
  snapshot: snapshot_date
  segment: institution
  target: target
target:
  positive_value: 1
  positive_meaning: bad_debt
snapshot:
  meaning: customer_specified_feature_cutoff
features:
  families:
    order: [order_]
    browse: [browse_]
  explicit_catalog: catalog.yaml
time_validation_enabled: false
""",
        encoding="utf-8",
    )

    config = ProjectConfig.from_yaml(config_path)
    features = RiskProbeService(config=config, runs_dir=tmp_path / "runs")._feature_names(
        ParquetDataset(data_path)
    )

    assert config.features.explicit_catalog == catalog_path
    assert features == ["browse_signal"]


def test_empty_or_overlapping_prefix_selection_never_expands_to_wide_columns(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "wide.parquet"
    _wide_dataset(data_path)
    config = _config(data_path)
    overlapping = config.model_copy(
        update={
            "features": config.features.model_copy(
                update={"families": {"first": ("order_",), "second": ("order_",)}}
            )
        }
    )
    empty = config.model_copy(
        update={"features": config.features.model_copy(update={"families": {}})}
    )
    dataset = ParquetDataset(data_path)

    assert RiskProbeService(config=overlapping, runs_dir=tmp_path / "overlap")._feature_names(
        dataset
    ) == ["order_signal"]
    assert RiskProbeService(config=empty, runs_dir=tmp_path / "empty")._feature_names(
        dataset
    ) == []


def test_fixed_service_train_split_discovers_each_synthetic_truth_rule(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "synthetic.parquet"
    frame, truth = generate_behavior_dataset(rows=20_000, seed=42)
    frame.write_parquet(data_path)
    config = _config(data_path)
    config = config.model_copy(
        update={
            "time_validation_enabled": True,
            "discovery": config.discovery.model_copy(
                update={"min_support": 0.03, "beam_width": 120}
            ),
        }
    )
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")
    dataset = service._dataset()
    feature_names = service._feature_names(dataset)
    train, test, holdout, excluded_null_snapshot_rows = service._partitions(
        dataset, feature_names
    )
    rules = service._discover_from_train(train, feature_names)

    def same_direction(left: str, right: str) -> bool:
        return ({left, right} <= {">", ">="}) or ({left, right} <= {"<", "<="})

    assert excluded_null_snapshot_rows == 0
    assert holdout is not None
    assert set(train["snapshot_date"]).isdisjoint(test["snapshot_date"])
    assert set(test["snapshot_date"]).isdisjoint(holdout["snapshot_date"])
    assert all(
        condition.feature in feature_names
        for rule in rules
        for condition in rule.conditions
    )
    for truth_rule in truth.hidden_rules:
        expected_conditions = {condition.feature: condition for condition in truth_rule.conditions}
        assert any(
            len(discovered.conditions) == len(expected_conditions)
            and {condition.feature for condition in discovered.conditions}
            == set(expected_conditions)
            and all(
                same_direction(
                    condition.operator, expected_conditions[condition.feature].operator
                )
                for condition in discovered.conditions
            )
            for discovered in rules
        ), truth_rule.rule_id


def _subprocess_run_payload(work_dir: Path) -> dict[str, object]:
    script = """
import hashlib
import json
from pathlib import Path

from riskprobe.config import ProjectConfig
from riskprobe.service import RiskProbeService
from riskprobe.synthetic import generate_behavior_dataset

work_dir = Path.cwd() / "subprocess-work"
work_dir.mkdir()
data_path = work_dir / "synthetic.parquet"
frame, _ = generate_behavior_dataset(rows=5_000, seed=42)
frame.write_parquet(data_path)
config = ProjectConfig.model_validate({
    "dataset": {"id": "subprocess-synthetic", "path": data_path},
    "columns": {"entity": "entity_id", "snapshot": "snapshot_date", "segment": "institution", "target": "target"},
    "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
    "snapshot": {"meaning": "customer_specified_feature_cutoff"},
    "features": {"families": {"order": ["order_"], "browse": ["browse_"], "platform": ["multi_platform_"]}},
    "time_validation_enabled": False,
    "discovery": {"min_support": 0.03, "max_single_rules": 40, "beam_width": 12, "max_pair_rules": 20, "random_seed": 42},
    "validation": {"bootstrap_rounds": 100, "min_group_size": 20},
})
context = RiskProbeService(config=config, runs_dir=work_dir / "runs").run()
artifacts = sorted(path for path in context.run_dir.iterdir() if path.is_file())
print(json.dumps({
    "run_id": context.run_id,
    "rules": __import__("polars").read_parquet(context.run_dir / "candidate_rules.parquet").to_dicts(),
    "artifact_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
}, sort_keys=True))
"""
    python = VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable)
    environment = os.environ | {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=work_dir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_independent_python_processes_produce_identical_rules_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _subprocess_run_payload(first_dir)
    second = _subprocess_run_payload(second_dir)

    assert set(first["artifact_hashes"]) == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    }
    assert first == second
    assert b"/Users/" not in json.dumps(first, sort_keys=True).encode()


def test_all_cli_success_commands_emit_safe_json_and_each_help_lists_key_options(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "demo.parquet"
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        f"""dataset:
  id: demo
  path: {data_path}
columns: {{entity: entity_id, snapshot: snapshot_date, segment: institution, target: target}}
target: {{positive_value: 1, positive_meaning: bad_debt}}
snapshot: {{meaning: customer_specified_feature_cutoff}}
features: {{families: {{order: [order_], browse: [browse_], platform: [multi_platform_]}}}}
time_validation_enabled: false
validation: {{bootstrap_rounds: 100, min_group_size: 20}}
""",
        encoding="utf-8",
    )
    commands = [
        ["synthetic", "--output", str(data_path), "--rows", "1000", "--seed", "42"],
        ["inspect", "--config", str(config_path), "--runs-dir", str(tmp_path / "runs")],
        ["discover", "--config", str(config_path), "--runs-dir", str(tmp_path / "runs")],
        ["run", "--config", str(config_path), "--runs-dir", str(tmp_path / "runs")],
    ]
    expected_keys = {
        "synthetic": {"command", "rows", "columns", "truth_rule_ids"},
        "inspect": {"command", "feature_count", "metadata_grade", "row_count", "segment_count"},
        "discover": {"command", "candidate_rule_count", "rule_ids"},
        "run": {"command", "run_id", "reused", "metadata_grade", "artifact_count"},
    }
    help_options = {
        "synthetic": ("--output", "--rows", "--seed"),
        "inspect": ("--config", "--runs-dir"),
        "discover": ("--config", "--runs-dir"),
        "run": ("--config", "--runs-dir"),
    }

    for command in commands:
        result = runner.invoke(app, command)
        payload = json.loads(result.stdout)
        assert result.exit_code == 0, result.stdout
        assert set(payload) == expected_keys[command[0]]
        assert payload["command"] == command[0]
        assert str(tmp_path) not in result.stdout
        assert "private-segment" not in result.stdout
        help_result = runner.invoke(app, [command[0], "--help"])
        assert help_result.exit_code == 0
        assert all(option in help_result.stdout for option in help_options[command[0]])


def test_grade_b_report_limits_evidence_to_time_slice_stability(tmp_path: Path) -> None:
    frame, _ = generate_behavior_dataset(rows=100, seed=42)
    data_path = tmp_path / "report-input.parquet"
    frame.write_parquet(data_path)
    config = _config(data_path).model_copy(update={"time_validation_enabled": True})
    profile = profile_dataset(ParquetDataset(data_path), config)

    report = render_risk_report(profile, [])

    assert "evidence reflects stability across time slices, not a known performance window" in report
    assert "严格 OOT" not in report
    assert "可上线" not in report


def test_grade_b_report_discloses_when_time_slice_stability_was_not_evaluated(
    tmp_path: Path,
) -> None:
    frame, _ = generate_behavior_dataset(rows=100, seed=42)
    data_path = tmp_path / "report-input.parquet"
    frame.write_parquet(data_path)
    profile = profile_dataset(ParquetDataset(data_path), _config(data_path))

    report = render_risk_report(profile, [])

    assert "evidence reflects stability across time slices" not in report
    assert "time-slice stability was not evaluated" in report


def test_zero_pair_limit_disables_second_order_rules() -> None:
    values = [-2.0, -1.0, 1.0, 2.0] * 25
    train = pl.DataFrame(
        {
            "left": values,
            "right": [-2.0, 1.0, -1.0, 2.0] * 25,
            "target": [
                int(left > 0 and right > 0)
                for left, right in zip(values, [-2.0, 1.0, -1.0, 2.0] * 25, strict=True)
            ],
        }
    )

    rules = discover_rules(
        train,
        ["left", "right"],
        "target",
        _config(Path("synthetic.parquet")).discovery.model_copy(
            update={"min_support": 0.1, "max_pair_rules": 0}
        ),
    )

    assert all(len(rule.conditions) == 1 for rule in rules)


def test_pair_beam_has_a_global_candidate_budget(monkeypatch) -> None:
    import riskprobe.rules.discovery as discovery_module

    values = [float(index % 2) for index in range(80)]
    train = pl.DataFrame(
        {
            **{f"feature_{index}": values for index in range(8)},
            "target": [int(value) for value in values],
        }
    )
    original_candidate = discovery_module._candidate
    pair_evaluations = 0

    def count_pair_evaluations(*args, **kwargs):
        nonlocal pair_evaluations
        if len(args[2]) == 2:
            pair_evaluations += 1
        return original_candidate(*args, **kwargs)

    monkeypatch.setattr(discovery_module, "_candidate", count_pair_evaluations)
    discover_rules(
        train,
        [f"feature_{index}" for index in range(8)],
        "target",
        _config(Path("synthetic.parquet")).discovery.model_copy(
            update={"min_support": 0.1, "beam_width": 4, "max_pair_rules": 1}
        ),
    )

    assert pair_evaluations <= 6


def test_grade_b_report_does_not_claim_time_stability_for_a_single_snapshot(
    tmp_path: Path,
) -> None:
    frame, _ = generate_behavior_dataset(rows=100, seed=42)
    data_path = tmp_path / "single-snapshot.parquet"
    frame.with_columns(pl.lit("2024-01-01").alias("snapshot_date")).write_parquet(data_path)
    config = _config(data_path).model_copy(update={"time_validation_enabled": True})
    profile = profile_dataset(ParquetDataset(data_path), config)

    report = render_risk_report(profile, [])

    assert "evidence reflects stability across time slices" not in report
    assert "time-slice stability was not evaluated" in report
    assert "Time Decay" not in report
