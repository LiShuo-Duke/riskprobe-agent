from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Sequence

import click
import typer
from typer.core import TyperGroup

from riskprobe import cross_client_cli as _cross_client_cli
from riskprobe.config import ProjectConfig
from riskprobe.evals import (
    EvalObservation,
    EvalObservationV2,
    EvalSuite,
    EvalSuiteV2,
)
from riskprobe.policy import Budget, Principal, Role
from riskprobe.service import RiskProbeService
from riskprobe.synthetic import generate_behavior_dataset

if TYPE_CHECKING:
    from riskprobe.agents.decision_providers import DecisionProviderConfig


class EvalVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class AgentDecisionProviderMode(str, Enum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"


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


def _service(
    config: Path | None,
    runs_dir: Path | None,
    state_dir: Path | None = None,
    decision_provider_config: DecisionProviderConfig | None = None,
) -> RiskProbeService:
    if runs_dir is None:
        _fail("input_error", "Provide a local --runs-dir path.")
    project_config: ProjectConfig | None = None
    if config is not None:
        try:
            project_config = ProjectConfig.from_yaml(config)
        except Exception:
            _fail(
                "configuration_error",
                "Check that --config names a readable, valid local YAML configuration.",
            )
    try:
        return RiskProbeService(
            config=project_config,
            runs_dir=runs_dir,
            state_dir=state_dir,
            decision_provider_config=decision_provider_config,
        )
    except Exception:
        _fail("runs_directory_error", "Choose a writable local --runs-dir directory.")


def _read_json(path: Path, *, category: str, message: str) -> object:
    try:
        encoded = path.read_bytes()
        if len(encoded) > 16 * 1024 * 1024:
            raise ValueError
        return json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except Exception:
        _fail(category, message)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _observations(
    path: Path,
    *,
    model: type[EvalObservation] | type[EvalObservationV2],
) -> dict[str, EvalObservation | EvalObservationV2]:
    payload = _read_json(
        path,
        category="evaluation_error",
        message="Check the frozen local suite and observations JSON, then rerun evaluate.",
    )
    if type(payload) is not dict or set(payload) != {"observations"}:
        _fail(
            "evaluation_error",
            "Check the frozen local suite and observations JSON, then rerun evaluate.",
        )
    items = payload["observations"]
    if type(items) is not list:
        _fail(
            "evaluation_error",
            "Check the frozen local suite and observations JSON, then rerun evaluate.",
        )
    observations: dict[str, EvalObservation | EvalObservationV2] = {}
    try:
        for item in items:
            observation = model.model_validate_json(
                json.dumps(
                    item,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            if observation.case_id in observations:
                raise ValueError
            observations[observation.case_id] = observation
    except Exception:
        _fail(
            "evaluation_error",
            "Check the frozen local suite and observations JSON, then rerun evaluate.",
        )
    return observations


def _write_report(path: Path, payload: object) -> None:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
        path.write_text(encoded, encoding="utf-8")
    except Exception:
        _fail("output_error", "Choose a writable local --output JSON path.")


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


@app.command()
def diagnose(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    run_id: str | None = typer.Option(
        None,
        help="Existing run ID; omit to create or reuse the authoritative local run.",
    ),
    state_dir: Path | None = typer.Option(None, help="Optional mutable sidecar directory."),
) -> None:
    """Persist aggregate diagnostic evidence for one local run."""
    service = _service(config, runs_dir, state_dir)
    try:
        effective_run_id = service.run().run_id if run_id is None else run_id
        response = service.diagnose(run_id=effective_run_id)
    except Exception:
        _fail(
            "diagnosis_error",
            "Check the local run and dataset, then rerun diagnose.",
        )
    typer.echo(
        json.dumps(
            {
                "command": "diagnose",
                "finding_count": response.finding_count,
                "finding_ids": list(response.finding_ids),
                "run_id": effective_run_id,
            },
            sort_keys=True,
        )
    )


@app.command()
def recommend(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    run_id: str = typer.Option(..., help="Existing 16-character local run identifier."),
    evidence_id: list[str] = typer.Option(
        [],
        "--evidence-id",
        help="Diagnostic evidence ID; repeat for multiple findings.",
    ),
    all_current_diagnostics: bool = typer.Option(
        False,
        "--all-current-diagnostics",
        help="Run diagnostics now and recommend from exactly that same-run result.",
    ),
    state_dir: Path | None = typer.Option(None, help="Optional mutable sidecar directory."),
) -> None:
    """Persist evidence-linked, human-gated local recommendations."""
    if all_current_diagnostics == bool(evidence_id):
        _fail(
            "input_error",
            "Provide either --evidence-id values or --all-current-diagnostics.",
        )
    service = _service(config, runs_dir, state_dir)
    try:
        if all_current_diagnostics:
            response = service.recommend(
                run_id=run_id,
                evidence_ids=(),
                all_current_diagnostics=True,
            )
        else:
            response = service.recommend(
                run_id=run_id,
                evidence_ids=tuple(evidence_id),
            )
    except Exception:
        _fail(
            "recommendation_error",
            "Check the local run and diagnostic evidence, then rerun recommend.",
        )
    typer.echo(
        json.dumps(
            {
                "command": "recommend",
                "recommendation_count": response.recommendation_count,
                "recommendation_ids": list(response.recommendation_ids),
                "run_id": run_id,
            },
            sort_keys=True,
        )
    )


@app.command()
def status(
    config: Path | None = typer.Option(
        None,
        help="Optional local YAML configuration; not needed for runtime status.",
    ),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    run_id: str = typer.Option(..., help="Existing 16-character local run identifier."),
) -> None:
    """Print a bounded local runtime status."""
    service = _service(config, runs_dir)
    try:
        response = service.status(run_id=run_id)
    except Exception:
        _fail("runtime_error", "Check the local run identifier, then rerun status.")
    typer.echo(
        json.dumps(
            {"command": "status", **response.model_dump(mode="json")},
            sort_keys=True,
        )
    )


@app.command()
def trace(
    config: Path | None = typer.Option(
        None,
        help="Optional local YAML configuration; not needed for runtime trace.",
    ),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    run_id: str = typer.Option(..., help="Existing 16-character local run identifier."),
    node_id: str | None = typer.Option(None, help="Optional public runtime node ID."),
) -> None:
    """Print a redacted local runtime trace."""
    service = _service(config, runs_dir)
    try:
        response = service.trace(run_id=run_id, node_id=node_id)
    except Exception:
        _fail("runtime_error", "Check the local run identifier, then rerun trace.")
    typer.echo(
        json.dumps(
            {"command": "trace", **response.model_dump(mode="json")},
            sort_keys=True,
        )
    )


@app.command()
def agent(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    state_dir: Path | None = typer.Option(None, help="Optional mutable sidecar directory."),
    principal_id: str = typer.Option("local-analyst", help="Public local principal ID."),
    role: Role = typer.Option(Role.ANALYST, help="Local policy role."),
    max_queries: int = typer.Option(8, help="Bounded local tool-query budget."),
    objective: str = typer.Option("comprehensive", help="Allowlisted public objective code."),
    decision_provider_mode: AgentDecisionProviderMode = typer.Option(
        AgentDecisionProviderMode.DISABLED,
        help="Standalone controlled-decision provider mode.",
    ),
) -> None:
    """Run the deterministic offline comprehensive agent."""
    from riskprobe.agents.decision_providers import (
        DecisionProviderConfig,
        DecisionProviderMode,
    )

    service = _service(
        config,
        runs_dir,
        state_dir,
        decision_provider_config=DecisionProviderConfig(
            mode=DecisionProviderMode(decision_provider_mode.value)
        ),
    )
    try:
        result = service.orchestrate(
            dataset_id=service.config.dataset.id,
            principal=Principal(principal_id=principal_id, role=role),
            budget=Budget(max_queries=max_queries),
            objective=objective,
        )
    except Exception:
        _fail(
            "agent_error",
            "Check the local configuration, policy identity, and sidecar state, then rerun agent.",
        )
    typer.echo(
        json.dumps(
            {
                "approved": result.review.approved,
                "command": "agent",
                "diagnosis_evidence_ids": list(result.diagnosis_evidence_ids),
                "evidence_ids": list(result.evidence_ids),
                "reason_codes": [reason.value for reason in result.review.reason_codes],
                "retry_count": result.retry_count,
                "run_id": result.session_id,
                "status": result.status.value,
                "tool_sequence": list(result.tool_sequence),
            },
            sort_keys=True,
        )
    )


@app.command()
def evaluate(
    eval_version: EvalVersion = typer.Option(..., help="Evaluation schema version."),
    suite: Path = typer.Option(..., help="Frozen local evaluation suite JSON."),
    observations: Path = typer.Option(..., help="Local replay observations JSON."),
    candidate_version: str = typer.Option(..., help="Public candidate version."),
    output: Path | None = typer.Option(None, help="Optional full report JSON output."),
) -> None:
    """Replay frozen offline observations without dynamic code loading."""
    try:
        suite_bytes = suite.read_bytes()
        if len(suite_bytes) > 16 * 1024 * 1024:
            raise ValueError
        if eval_version is EvalVersion.V1:
            frozen_suite = EvalSuite.model_validate_json(suite_bytes)
            if not frozen_suite.verify_integrity():
                raise ValueError
            by_case = _observations(observations, model=EvalObservation)
            expected = {case.case_id for case in frozen_suite.cases}
            if set(by_case) != expected:
                raise ValueError
            report = RiskProbeService.evaluate_v1(
                frozen_suite,
                lambda case, seed: by_case[case.case_id],
                candidate_version=candidate_version,
            )
            summary: dict[str, object] = {
                "candidate_version": report.candidate_version,
                "case_count": len(report.case_results),
                "command": "evaluate",
                "eval_version": "v1",
                "passed": report.passed,
                "report_hash": report.report_hash,
                "suite_hash": report.suite_hash,
                "suite_id": report.suite_id,
            }
        else:
            frozen_suite_v2 = EvalSuiteV2.model_validate_json(suite_bytes)
            if not frozen_suite_v2.verify_integrity():
                raise ValueError
            by_case_v2 = _observations(observations, model=EvalObservationV2)
            expected_v2 = {case.case_id for case in frozen_suite_v2.cases}
            if set(by_case_v2) != expected_v2:
                raise ValueError
            report = RiskProbeService.evaluate_v2(
                frozen_suite_v2,
                lambda case, seed: by_case_v2[case.case_id],
                candidate_version=candidate_version,
            )
            summary = {
                "candidate_version": report.candidate_version,
                "case_count": len(report.case_results),
                "command": "evaluate",
                "eval_version": "v2",
                "report_hash": report.report_hash,
                "suite_hash": report.suite_hash,
                "suite_id": report.suite_id,
            }
    except typer.Exit:
        raise
    except Exception:
        _fail(
            "evaluation_error",
            "Check the frozen local suite and observations JSON, then rerun evaluate.",
        )
    if output is not None:
        _write_report(output, report.model_dump(mode="json"))
    typer.echo(json.dumps(summary, sort_keys=True))


@app.command("rag-build")
def rag_build(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    run_id: str = typer.Option(..., help="Existing 16-character local run identifier."),
    root_id: str = typer.Option(..., help="Registered provider-safe root ID."),
    root: Path = typer.Option(..., help="Manifest-attested provider-safe root."),
    scope_id: str = typer.Option(..., help="Public local citation scope ID."),
    state_dir: Path | None = typer.Option(None, help="Optional mutable sidecar directory."),
    provider_summaries: Path | None = typer.Option(
        None,
        help="Optional exact provider-safe aggregate summaries JSON.",
    ),
) -> None:
    """Build a sealed local citation index from attested content."""
    service = _service(config, runs_dir, state_dir)
    summaries: tuple[dict[str, object], ...] = ()
    if provider_summaries is not None:
        payload = _read_json(
            provider_summaries,
            category="rag_build_error",
            message="Check the provider-safe manifest and aggregate summaries, then rerun rag-build.",
        )
        if (
            type(payload) is not dict
            or set(payload) != {"provider_summaries"}
            or type(payload["provider_summaries"]) is not list
            or any(type(item) is not dict for item in payload["provider_summaries"])
        ):
            _fail(
                "rag_build_error",
                "Check the provider-safe manifest and aggregate summaries, then rerun rag-build.",
            )
        summaries = tuple(payload["provider_summaries"])
    try:
        result = service.build_local_rag(
            run_id=run_id,
            roots={root_id: root},
            root_id=root_id,
            scope_id=scope_id,
            provider_summaries=summaries,
        )
    except Exception:
        _fail(
            "rag_build_error",
            "Check the provider-safe manifest and aggregate summaries, then rerun rag-build.",
        )
    typer.echo(
        json.dumps(
            {"command": "rag-build", "run_id": run_id, **result.model_dump(mode="json")},
            sort_keys=True,
        )
    )


@app.command("rag-query")
def rag_query(
    config: Path | None = typer.Option(
        None,
        help="Optional local YAML configuration; not needed for a sealed index query.",
    ),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
    run_id: str = typer.Option(..., help="Existing 16-character local run identifier."),
    root_id: str | None = typer.Option(
        None,
        help="Optional legacy provider-safe root ID; use together with --root.",
    ),
    root: Path | None = typer.Option(
        None,
        help="Optional legacy provider-safe root; use together with --root-id.",
    ),
    scope_id: str = typer.Option(..., help="Public local citation scope ID."),
    query_id: str = typer.Option(..., help="Public local query ID."),
    query_text: str = typer.Option(..., help="Safe local query text."),
    limit: int = typer.Option(5, help="Maximum citations, from 1 through 100."),
    state_dir: Path | None = typer.Option(None, help="Optional mutable sidecar directory."),
) -> None:
    """Query a sealed local index without returning source text."""
    if (root_id is None) != (root is None):
        _fail("input_error", "Provide both legacy --root-id and --root, or neither.")
    service = _service(config, runs_dir, state_dir)
    try:
        result = service.query_local_rag(
            run_id=run_id,
            roots={} if root_id is None or root is None else {root_id: root},
            scope_id=scope_id,
            query_id=query_id,
            query_text=query_text,
            limit=limit,
        )
    except Exception:
        _fail(
            "rag_query_error",
            "Check the local index scope and safe query, then rerun rag-query.",
        )
    typer.echo(
        json.dumps(
            {"command": "rag-query", "run_id": run_id, **result.model_dump(mode="json")},
            sort_keys=True,
        )
    )


_safe_alert_payload = _cross_client_cli._safe_alert_payload
_cross_client_cli.register_cross_client_commands(app)
