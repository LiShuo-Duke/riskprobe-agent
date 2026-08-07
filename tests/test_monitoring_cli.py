from typer.testing import CliRunner

from riskprobe.cli import app

runner = CliRunner()


def test_evaluate_drift_reports_recall_and_writes_monitoring_artifacts(tmp_path) -> None:
    data_path = tmp_path / "behavior.parquet"
    generated = runner.invoke(app, ["synthetic", "--output", str(data_path), "--rows", "1000", "--seed", "42"])
    assert generated.exit_code == 0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset: {{id: synthetic_demo, path: {data_path}}}
columns: {{entity: entity_id, snapshot: snapshot_date, segment: institution, target: target}}
target: {{positive_value: 1, positive_meaning: bad_debt, performance_window_days: null}}
snapshot: {{meaning: customer_specified_feature_cutoff}}
features: {{families: {{order: [order_], browse: [browse_], multi: [multi_], embedding: [emb_]}}}}
segment_display_name: institution
time_validation_enabled: true
""".strip(),
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"

    result = runner.invoke(
        app,
        ["evaluate-drift", "--config", str(config_path), "--runs-dir", str(runs_dir), "--seed", "42"],
    )

    assert result.exit_code == 0
    assert "recall" in result.stdout.lower()
    assert '"recall": 1.0' in result.stdout
    evaluation_files = list((runs_dir / "monitoring").rglob("drift_evaluation.json"))
    assert len(evaluation_files) == 1
    assert (evaluation_files[0].parent / "anomaly_alerts.json").is_file()
    assert (evaluation_files[0].parent / "diagnoses.json").is_file()
