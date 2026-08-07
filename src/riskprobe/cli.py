import json
from pathlib import Path
from typing import NoReturn, Sequence

import click
import typer

from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.models import EvidenceCard, RiskRule, RuleMetrics
from riskprobe.monitoring.detection import detect_anomalies
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.injection import DriftScenario, evaluate_alerts, inject_drift
from riskprobe.monitoring.reference import build_reference_snapshot
from typer.core import TyperGroup

from riskprobe.config import ProjectConfig
from riskprobe.service import RiskProbeService
from riskprobe.synthetic import generate_behavior_dataset


class StructuredErrorGroup(TyperGroup):
    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: object,
    ) -> object:
        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except click.ClickException:
            if not standalone_mode:
                raise
            typer.echo(
                json.dumps(
                    {
                        "error": "argument_error",
                        "message": "Use --help to review the required command and option values.",
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(2)
        if standalone_mode and isinstance(result, int) and result:
            raise SystemExit(result)
        return result


app = typer.Typer(
    cls=StructuredErrorGroup,
    help="Run local RiskProbe synthetic demonstrations and analyses.",
)


def _fail(category: str, message: str) -> NoReturn:
    typer.echo(json.dumps({"error": category, "message": message}, sort_keys=True))
    raise typer.Exit(code=2)


def _service(config: Path | None, runs_dir: Path | None) -> RiskProbeService:
    if config is None or runs_dir is None:
        _fail("input_error", "Provide both --config and --runs-dir local paths.")
    try:
        project_config = ProjectConfig.from_yaml(config)
    except Exception:
        _fail(
            "configuration_error",
            "Check that --config names a readable, valid local YAML configuration.",
        )
    try:
        return RiskProbeService(config=project_config, runs_dir=runs_dir)
    except Exception:
        _fail("runs_directory_error", "Choose a writable local --runs-dir directory.")


@app.command()
def synthetic(
    output: Path | None = typer.Option(..., help="Destination Parquet file."),
    rows: int = typer.Option(..., help="Number of synthetic rows to create."),
    seed: int = typer.Option(..., help="Deterministic synthetic-data seed."),
) -> None:
    """Write a deterministic public synthetic Parquet dataset."""
    if output is None:
        _fail("input_error", "Provide a writable local --output Parquet path.")
    try:
        frame, truth = generate_behavior_dataset(rows=rows, seed=seed)
    except (TypeError, ValueError):
        _fail("input_error", "Use positive --rows and a non-negative integer --seed.")
    try:
        frame.write_parquet(output)
    except OSError:
        _fail("output_error", "Choose a writable local --output Parquet path.")
    typer.echo(
        json.dumps(
            {
                "columns": frame.width,
                "command": "synthetic",
                "rows": frame.height,
                "truth_rule_ids": [rule.rule_id for rule in truth.hidden_rules],
            },
            sort_keys=True,
        )
    )


@app.command()
def inspect(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Print a safe JSON dataset-profile summary."""
    service = _service(config, runs_dir)
    try:
        profile = service.inspect()
    except Exception:
        _fail(
            "inspection_error",
            "Check the configured local dataset and column roles, then rerun inspect.",
        )
    typer.echo(
        json.dumps(
            {
                "command": "inspect",
                "feature_count": profile.feature_count,
                "metadata_grade": profile.metadata_grade,
                "row_count": profile.row_count,
                "segment_count": len(profile.segment_counts),
            },
            sort_keys=True,
        )
    )


@app.command()
def discover(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Print safe JSON candidate-rule identifiers."""
    service = _service(config, runs_dir)
    try:
        rule_ids = sorted(rule.rule_id for rule in service.discover())
    except Exception:
        _fail(
            "discovery_error",
            "Check the configured local dataset and feature families, then rerun discover.",
        )
    typer.echo(
        json.dumps(
            {
                "candidate_rule_count": len(rule_ids),
                "command": "discover",
                "rule_ids": rule_ids,
            },
            sort_keys=True,
        )
    )


@app.command()
def run(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Create or reuse a complete immutable local analysis run."""
    service = _service(config, runs_dir)
    try:
        context = service.run()
    except Exception:
        _fail(
            "run_error",
            "Check the configured local dataset and runs directory, then rerun locally.",
        )
    typer.echo(
        json.dumps(
            {
                "artifact_count": 6,
                "command": "run",
                "metadata_grade": service.config.metadata_grade,
                "reused": context.is_existing,
                "run_id": context.run_id,
            },
            sort_keys=True,
        )
    )


def _monitoring_inputs(config: ProjectConfig):
    dataset = ParquetDataset(config.dataset.path)
    profile = RiskProbeService(config=config, runs_dir=Path(".")).inspect()
    roles = (config.columns.entity, config.columns.snapshot, config.columns.segment, config.columns.target)
    features = config.features.select_columns(dataset.schema().names(), roles)
    frame = dataset.collect([config.columns.segment, config.columns.target, *features])
    return frame, profile, FeatureCatalog.from_columns(features, config.features.families)


def _write_monitoring_json(output_dir: Path, name: str, payload: object) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@app.command()
def snapshot(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Create an aggregate-only monitoring reference snapshot."""
    service = _service(config, runs_dir)
    try:
        context, reference = service.monitoring_snapshot()
    except Exception:
        _fail("snapshot_error", "Check the local configuration and source data, then rerun snapshot.")
    typer.echo(json.dumps({"command": "snapshot", "reference_run_id": context.run_id, "snapshot_id": reference.snapshot_id}, sort_keys=True))


@app.command()
def monitor(
    reference_run_id: str = typer.Option(..., help="Existing monitoring reference run ID."),
    current_config: Path = typer.Option(..., help="Local YAML configuration for current data."),
    runs_dir: Path = typer.Option(..., help="Local monitoring runs directory."),
) -> None:
    """Compare a current local dataset to an aggregate reference snapshot."""
    try:
        reference = __import__("riskprobe.monitoring.models", fromlist=["ReferenceSnapshot"]).ReferenceSnapshot.model_validate_json(
            (runs_dir / "monitoring" / reference_run_id / "reference_snapshot.json").read_text(encoding="utf-8")
        )
        current = ProjectConfig.from_yaml(current_config)
        frame, _, catalog = _monitoring_inputs(current)
        alerts = detect_anomalies(reference, frame, (), catalog)
        diagnoses = diagnose_alerts(alerts, reference, frame, catalog, top_k=3)
        output_dir = runs_dir / "monitoring" / reference_run_id / "current"
        _write_monitoring_json(output_dir, "anomaly_alerts.json", [item.model_dump(mode="json") for item in alerts])
        _write_monitoring_json(output_dir, "diagnoses.json", [item.model_dump(mode="json") for item in diagnoses])
    except Exception:
        _fail("monitor_error", "Check the aggregate reference run and current local configuration.")
    typer.echo(json.dumps({"alert_count": len(alerts), "command": "monitor"}, sort_keys=True))


@app.command("evaluate-drift")
def evaluate_drift(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local monitoring runs directory."),
    seed: int = typer.Option(..., help="Deterministic drift injection seed."),
) -> None:
    """Inject six reproducible drift scenarios and report aggregate detection scores."""
    try:
        project_config = ProjectConfig.from_yaml(config)
        frame, profile, catalog = _monitoring_inputs(project_config)
        metrics = RuleMetrics(
            support_count=100,
            coverage=0.10,
            base_bad_rate=0.10,
            hit_bad_rate=0.20,
            non_hit_bad_rate=0.08,
            lift=2.0,
            precision=0.20,
            recall=0.20,
            p_value=0.01,
        )
        baseline_card = EvidenceCard(
            rule=RiskRule(rule_id="synthetic_monitor_rule", conditions=(), origin="evaluation"),
            train=metrics,
            test=metrics,
            slices=(),
            lift_ci=(1.5, 2.5),
            adjusted_p_value=0.01,
            segment_consistency=1.0,
            max_time_decay=0.0,
            grade="Stable",
        )
        reference = build_reference_snapshot(frame, profile, (baseline_card,), catalog, project_config)
        institution = str(frame.get_column(project_config.columns.segment)[0])
        scenarios = (
            DriftScenario(scenario_id="missingness", drift_type="missingness", target="browse_pv_30d", magnitude=0.60),
            DriftScenario(scenario_id="numeric", drift_type="numeric_shift", target="browse_pv_30d", magnitude=5.00),
            DriftScenario(scenario_id="population", drift_type="population_shift", target=project_config.columns.segment, magnitude=0.60, institution=institution),
            DriftScenario(scenario_id="label", drift_type="label_shift", target=project_config.columns.target, magnitude=1.00),
            DriftScenario(scenario_id="schema", drift_type="schema", target="browse_pv_30d", magnitude=0.30),
            DriftScenario(scenario_id="rule-decay", drift_type="rule_decay", target="browse_pv_30d", magnitude=1.00),
        )
        all_alerts = []
        truths = []
        all_diagnoses = []
        for offset, scenario in enumerate(scenarios):
            injected = inject_drift(frame, scenario, seed + offset)
            current_cards = (baseline_card,)
            truth = injected.truth
            if scenario.drift_type == "population_shift":
                truth = truth.model_copy(update={"expected_scope_value": institution})
            elif scenario.drift_type == "label_shift":
                truth = truth.model_copy(update={"expected_scope_value": reference.dataset_id})
            elif scenario.drift_type == "rule_decay":
                current_cards = (
                    baseline_card.model_copy(
                        update={"test": metrics.model_copy(update={"lift": 1.0})}
                    ),
                )
                truth = truth.model_copy(update={"expected_scope_value": baseline_card.rule.rule_id})
            alerts = detect_anomalies(reference, injected.frame, current_cards, catalog)
            all_alerts.extend(alerts)
            truths.append(truth)
            all_diagnoses.extend(diagnose_alerts(alerts, reference, injected.frame, catalog, top_k=3))
        score = evaluate_alerts(all_alerts, truths, top_k=3)
        output_dir = runs_dir / "monitoring" / f"evaluation-{reference.snapshot_id[:16]}"
        _write_monitoring_json(output_dir, "anomaly_alerts.json", [item.model_dump(mode="json") for item in all_alerts])
        _write_monitoring_json(output_dir, "diagnoses.json", [item.model_dump(mode="json") for item in all_diagnoses])
        _write_monitoring_json(output_dir, "drift_evaluation.json", score.model_dump(mode="json"))
    except Exception:
        _fail("evaluation_error", "Check the local configuration, synthetic-compatible features, and --seed.")
    typer.echo(json.dumps({"command": "evaluate-drift", **score.model_dump(mode="json")}, sort_keys=True))
