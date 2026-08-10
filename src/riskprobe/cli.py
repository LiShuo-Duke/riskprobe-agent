import hashlib
import json
import stat
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn, Sequence

import click
import polars as pl
import typer

from riskprobe.adapters.company import preflight_company_dataset
from riskprobe.adapters.home_credit import HomeCreditPaths, prepare_home_credit
from riskprobe.benchmarking import BenchmarkRecord, StageTiming, total_agent_minutes
from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.models import EvidenceCard, RiskRule, RuleMetrics
from riskprobe.monitoring.detection import detect_anomalies
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.injection import DetectionScore, DriftScenario, evaluate_alerts, inject_drift
from riskprobe.monitoring.reference import build_reference_snapshot
from riskprobe.privacy import assert_safe_payload, stable_token
from typer.core import TyperGroup

from riskprobe.config import ProjectConfig
from riskprobe.resume_evidence import aggregate_benchmarks, render_markdown
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


def _safe_alert_payload(alert: object, *, expose_segment_values: bool) -> dict[str, object]:
    payload = alert.model_dump(mode="json")
    if payload.get("scope") == "institution":
        scope_value = str(payload["scope_value"])
        payload["scope_value"] = stable_token(scope_value)
        if expose_segment_values:
            payload["institution_name"] = scope_value
    assert_safe_payload(payload)
    return payload


def _safe_diagnosis_payload(
    diagnosis: object, *, expose_segment_values: bool
) -> dict[str, object]:
    payload = diagnosis.model_dump(mode="json")
    for cause in payload.get("root_causes", []):
        if not isinstance(cause, dict):
            continue
        value = cause.get("value")
        if cause.get("dimension") == "institution" and expose_segment_values:
            cause["institution_name"] = value
        cause["value"] = (
            value
            if cause.get("dimension") == "institution" and expose_segment_values
            else stable_token(value)
        )
    assert_safe_payload(payload)
    return payload


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
        _write_monitoring_json(
            output_dir,
            "anomaly_alerts.json",
            [
                _safe_alert_payload(
                    item, expose_segment_values=current.privacy.expose_segment_values
                )
                for item in alerts
            ],
        )
        _write_monitoring_json(
            output_dir,
            "diagnoses.json",
            [
                _safe_diagnosis_payload(
                    item, expose_segment_values=current.privacy.expose_segment_values
                )
                for item in diagnoses
            ],
        )
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
        runs_dir = _protected_path(runs_dir)
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
        numeric_features = [
            spec.name
            for spec in catalog.features
            if frame.schema[spec.name].is_numeric()
        ]
        if not numeric_features:
            raise ValueError("evaluate-drift requires at least one numeric feature")
        monitor_feature = numeric_features[0]
        segment_column = project_config.columns.segment
        target_column = project_config.columns.target
        institution = str(frame.get_column(segment_column)[0])
        scenarios = (
            DriftScenario(
                scenario_id="missingness",
                drift_type="missingness",
                target=monitor_feature,
                magnitude=0.60,
                target_column=target_column,
                segment_column=segment_column,
            ),
            DriftScenario(
                scenario_id="numeric",
                drift_type="numeric_shift",
                target=monitor_feature,
                magnitude=5.00,
                target_column=target_column,
                segment_column=segment_column,
            ),
            DriftScenario(
                scenario_id="population",
                drift_type="population_shift",
                target=segment_column,
                magnitude=0.60,
                institution=institution,
                target_column=target_column,
                segment_column=segment_column,
            ),
            DriftScenario(
                scenario_id="label",
                drift_type="label_shift",
                target=target_column,
                magnitude=1.00,
                target_column=target_column,
                segment_column=segment_column,
            ),
            DriftScenario(
                scenario_id="schema",
                drift_type="schema",
                target=monitor_feature,
                magnitude=0.30,
                target_column=target_column,
                segment_column=segment_column,
            ),
            DriftScenario(
                scenario_id="rule-decay",
                drift_type="rule_decay",
                target=monitor_feature,
                magnitude=1.00,
                target_column=target_column,
                segment_column=segment_column,
            ),
        )
        all_alerts = []
        truths = []
        all_diagnoses = []
        scenario_scores: dict[str, DetectionScore] = {}
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
            diagnoses = diagnose_alerts(alerts, reference, injected.frame, catalog, top_k=3)
            scenario_scores[scenario.scenario_id] = evaluate_alerts(
                alerts, (truth,), top_k=3, diagnoses=diagnoses
            )
            all_alerts.extend(alerts)
            truths.append(truth)
            all_diagnoses.extend(diagnoses)
        score = DetectionScore(
            precision=sum(item.precision for item in scenario_scores.values()) / len(scenario_scores),
            recall=sum(item.recall for item in scenario_scores.values()) / len(scenario_scores),
            false_positive_rate=None,
            false_discovery_rate=sum(item.false_discovery_rate for item in scenario_scores.values()) / len(scenario_scores),
            top_k_root_cause_hit=sum(item.top_k_root_cause_hit for item in scenario_scores.values()) / len(scenario_scores),
        )
        output_dir = _protected_path(
            runs_dir / "monitoring" / f"evaluation-{reference.snapshot_id[:16]}"
        )
        _write_monitoring_json(
            output_dir,
            "anomaly_alerts.json",
            [
                _safe_alert_payload(
                    item,
                    expose_segment_values=project_config.privacy.expose_segment_values,
                )
                for item in all_alerts
            ],
        )
        _write_monitoring_json(
            output_dir,
            "diagnoses.json",
            [
                _safe_diagnosis_payload(
                    item,
                    expose_segment_values=project_config.privacy.expose_segment_values,
                )
                for item in all_diagnoses
            ],
        )
        _write_monitoring_json(
            output_dir,
            "drift_evaluation.json",
            {**score.model_dump(mode="json"), "scenarios": {
                scenario_id: item.model_dump(mode="json")
                for scenario_id, item in sorted(scenario_scores.items())
            }},
        )
    except Exception:
        _fail("evaluation_error", "Check the local configuration and configured numeric features.")
    typer.echo(json.dumps({"command": "evaluate-drift", **score.model_dump(mode="json")}, sort_keys=True))


def _repository_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / ".git").exists() else Path.cwd().resolve()


def _protected_path(path: Path, *, output: bool = False) -> Path:
    """Allow only owner-private paths outside the repository checkout."""
    resolved = path.expanduser().resolve()
    repo = _repository_root()
    if resolved == repo or repo in resolved.parents:
        raise ValueError("benchmark outputs must not be written inside the repository")
    parent = resolved.parent if output else resolved
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent.is_dir() or stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise ValueError("benchmark output directory must be owner-private")
    return resolved


def _code_version() -> str:
    try:
        return version("riskprobe-agent")
    except PackageNotFoundError:
        return "0.1.0"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@app.command("prepare-home-credit")
def prepare_home_credit_command(
    input_dir: Path = typer.Option(..., help="Directory containing user-downloaded Home Credit CSVs."),
    output: Path = typer.Option(..., help="Local output Parquet path."),
) -> None:
    """Prepare public Home Credit behavior features without reading prediction files."""
    try:
        result = prepare_home_credit(HomeCreditPaths.from_directory(input_dir), output)
    except ValueError as error:
        _fail("home_credit_input_error", str(error))
    except (OSError, pl.exceptions.PolarsError):
        _fail("home_credit_preparation_error", "Check public CSV inputs and writable output path.")
    typer.echo(
        json.dumps(
            {
                "columns": result.columns,
                "command": "prepare-home-credit",
                "feature_families": result.feature_families,
                "rows": result.rows,
                "source_table_count": len(result.source_tables),
            },
            sort_keys=True,
        )
    )


@app.command("preflight-company")
def preflight_company(
    config: Path = typer.Option(..., help="Local company YAML configuration."),
) -> None:
    """Print aggregate-only, read-only Parquet readiness information."""
    try:
        result = preflight_company_dataset(ProjectConfig.from_yaml(config))
    except Exception:
        _fail(
            "company_preflight_error",
            "Check the local configuration, role columns, and read-only Parquet schema.",
        )
    typer.echo(
        json.dumps(
            {
                "batch_count": result.batch_count,
                "command": "preflight-company",
                "feature_count": result.feature_count,
                "feature_family_counts": result.feature_family_counts,
                "label_rate": result.label_rate,
                "limitations": result.limitations,
                "metadata_grade": result.metadata_grade,
                "row_count": result.row_count,
                "segment_count": result.segment_count,
            },
            sort_keys=True,
        )
    )


@app.command()
def benchmark(
    config: Path = typer.Option(..., help="Local company YAML configuration."),
    runs_dir: Path = typer.Option(..., help="Ignored local run directory."),
    baseline_record: Path = typer.Option(..., help="Human-authored local baseline JSON record."),
) -> None:
    """Measure local workflow stages without changing the manual baseline record."""
    try:
        project_config = ProjectConfig.from_yaml(config)
        runs_dir = _protected_path(runs_dir)
        baseline = BenchmarkRecord.model_validate_json(
            baseline_record.read_text(encoding="utf-8")
        )
        service = RiskProbeService(config=project_config, runs_dir=runs_dir)
        config_hash = service.store.config_fingerprint(project_config)
        data_fingerprint = _file_fingerprint(project_config.dataset.path)
        if (
            baseline.dataset_id != project_config.dataset.id
            or baseline.config_hash != config_hash
            or baseline.data_fingerprint != data_fingerprint
        ):
            raise ValueError("baseline record identity does not match current dataset and configuration")
        timings: list[StageTiming] = []

        started = time.perf_counter()
        service.inspect()
        timings.append(StageTiming(stage="inspect", seconds=time.perf_counter() - started))

        started = time.perf_counter()
        service.discover()
        timings.append(StageTiming(stage="discover", seconds=time.perf_counter() - started))

        started = time.perf_counter()
        context = service.run()
        timings.append(StageTiming(stage="validate", seconds=time.perf_counter() - started))

        started = time.perf_counter()
        service.monitoring_snapshot()
        timings.append(StageTiming(stage="monitor", seconds=time.perf_counter() - started))

        started = time.perf_counter()
        (context.run_dir / "risk_report.md").read_bytes()
        timings.append(StageTiming(stage="report", seconds=time.perf_counter() - started))

        record = baseline.model_copy(
            update={
                "agent_minutes": total_agent_minutes(timings),
                "code_version": _code_version(),
                "config_hash": config_hash,
                "data_fingerprint": data_fingerprint,
                "baseline_fingerprint": _file_fingerprint(baseline_record),
                "baseline_task_id": baseline.task_id,
                "dataset_id": project_config.dataset.id,
                "run_id": context.run_id,
                "measured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "stage_timings": tuple(timings),
            }
        ).validate_consistency()
        output = context.run_dir / "benchmark_record.json"
        if output.exists():
            raise FileExistsError("benchmark record already exists and is immutable")
        output.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        output.chmod(0o400)
    except Exception:
        _fail(
            "benchmark_error",
            "Check the local configuration, human baseline record, and ignored runs directory.",
        )
    typer.echo(
        json.dumps(
            {
                "command": "benchmark",
                "run_id": record.run_id,
                "stage_count": len(record.stage_timings),
            },
            sort_keys=True,
        )
    )


@app.command("resume-evidence")
def resume_evidence(
    records_dir: Path = typer.Option(..., help="Ignored local directory containing benchmark records."),
    output: Path = typer.Option(..., help="Ignored internal Markdown output path."),
) -> None:
    """Create internal resume text only from complete measured benchmark records."""
    try:
        records_dir = _protected_path(records_dir)
        output = _protected_path(output, output=True)
        records = [
            BenchmarkRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(records_dir.rglob("benchmark_record.json"))
        ]
        evidence = aggregate_benchmarks(records)
        if output.exists():
            raise FileExistsError("resume evidence output already exists and is immutable")
        output.write_text(render_markdown(evidence), encoding="utf-8")
        output.chmod(0o400)
    except Exception as error:
        _fail("resume_evidence_error", str(error))
    typer.echo(
        json.dumps(
            {
                "command": "resume-evidence",
                "source_run_count": len(evidence.source_run_ids),
                "task_count": evidence.task_count,
            },
            sort_keys=True,
        )
    )
