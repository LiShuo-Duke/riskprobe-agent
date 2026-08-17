import json
from pathlib import Path

import polars as pl
import pytest
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
    for command in (
        "synthetic",
        "inspect",
        "discover",
        "run",
        "diagnose",
        "recommend",
        "status",
        "trace",
        "agent",
        "evaluate",
        "rag-build",
        "rag-query",
    ):
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


def test_evaluate_replays_frozen_v1_observations_and_writes_report(tmp_path: Path) -> None:
    from riskprobe.evals import EvalCase, EvalObservation, EvalSuite

    suite = EvalSuite(
        suite_id="cli-suite-v1",
        cases=(
            EvalCase(
                case_id="cli-case",
                objective="comprehensive",
                expected_tool_sequence=("inspect",),
                require_diagnosis=False,
            ),
        ),
    )
    observation = EvalObservation(
        case_id="cli-case",
        task_succeeded=True,
        tool_sequence=("inspect",),
        evidence_ids=(),
        diagnosis_evidence_ids=(),
        policy_violations=0,
        privacy_violations=0,
    )
    suite_path = tmp_path / "suite.json"
    observations_path = tmp_path / "observations.json"
    output_path = tmp_path / "report.json"
    suite_path.write_text(suite.model_dump_json(), encoding="utf-8")
    observations_path.write_text(
        json.dumps(
            {"observations": [observation.model_dump(mode="json")]},
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--eval-version",
            "v1",
            "--suite",
            str(suite_path),
            "--observations",
            str(observations_path),
            "--candidate-version",
            "candidate-v1",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "evaluate"
    assert payload["eval_version"] == "v1"
    assert payload["passed"] is True
    assert payload["report_hash"] == persisted["report_hash"]
    assert str(suite_path) not in result.stdout
    assert str(observations_path) not in result.stdout
    assert str(output_path) not in result.stdout


def test_new_service_commands_emit_safe_json_without_private_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from riskprobe.rag import BuildResult, QueryResult
    from riskprobe.tools import (
        DiagnoseResponse,
        RecommendResponse,
        StatusResponse,
        TraceEvent,
        TraceResponse,
    )

    finding_id = "a" * 64
    recommendation_id = "b" * 64
    run_id = "0123456789abcdef"

    class FakeService:
        config = SimpleNamespace(dataset=SimpleNamespace(id="synthetic-demo"))

        def diagnose(self, *, run_id: str) -> DiagnoseResponse:
            return DiagnoseResponse(dataset_id="synthetic-demo", finding_ids=(finding_id,))

        def recommend(self, *, run_id: str, evidence_ids: tuple[str, ...]) -> RecommendResponse:
            assert evidence_ids == (finding_id,)
            return RecommendResponse(
                dataset_id="synthetic-demo",
                recommendation_ids=(recommendation_id,),
            )

        def status(self, *, run_id: str) -> StatusResponse:
            return StatusResponse(run_id=run_id, status="succeeded")

        def trace(self, *, run_id: str, node_id: str | None = None) -> TraceResponse:
            del node_id
            return TraceResponse(
                run_id=run_id,
                events=(
                    TraceEvent(
                        sequence=1,
                        node_id="profile",
                        event_type="node_succeeded",
                        status="succeeded",
                        attempt=1,
                    ),
                ),
            )

        def orchestrate(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(
                session_id=run_id,
                status=SimpleNamespace(value="succeeded"),
                review=SimpleNamespace(approved=True, reason_codes=()),
                evidence_ids=(finding_id, recommendation_id),
                diagnosis_evidence_ids=(finding_id,),
                retry_count=0,
                tool_sequence=("inspect", "diagnose", "discover", "recommend", "review"),
            )

        def build_local_rag(self, **kwargs: object) -> BuildResult:
            del kwargs
            return BuildResult(
                scope_id="scope-" + "1" * 24,
                document_count=1,
                index_hash="c" * 64,
            )

        def query_local_rag(self, **kwargs: object) -> QueryResult:
            del kwargs
            return QueryResult(scope_id="scope-" + "1" * 24, citations=())

    monkeypatch.setattr("riskprobe.cli._service", lambda *args, **kwargs: FakeService())
    common = ["--config", str(tmp_path / "private.yaml"), "--runs-dir", str(tmp_path / "runs")]
    invocations = (
        ["diagnose", *common, "--run-id", run_id],
        [
            "recommend",
            *common,
            "--run-id",
            run_id,
            "--evidence-id",
            finding_id,
        ],
        ["status", *common, "--run-id", run_id],
        ["trace", *common, "--run-id", run_id],
        ["agent", *common],
        [
            "rag-build",
            *common,
            "--run-id",
            run_id,
            "--root-id",
            "docs-root",
            "--root",
            str(tmp_path / "private-root"),
            "--scope-id",
            "scope-main",
        ],
        [
            "rag-query",
            *common,
            "--run-id",
            run_id,
            "--root-id",
            "docs-root",
            "--root",
            str(tmp_path / "private-root"),
            "--scope-id",
            "scope-main",
            "--query-id",
            "query-main",
            "--query-text",
            "private query text",
        ],
    )

    for arguments in invocations:
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, (arguments[0], result.stdout, result.exception)
        payload = json.loads(result.stdout)
        assert payload["command"] == arguments[0]
        assert str(tmp_path) not in result.stdout
        assert "private query text" not in result.stdout


def test_state_commands_do_not_require_config_or_query_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from riskprobe.rag import QueryResult
    from riskprobe.tools import StatusResponse, TraceResponse

    run_id = "0123456789abcdef"
    calls: list[tuple[object, ...]] = []

    class FakeService:
        def status(self, *, run_id: str) -> StatusResponse:
            calls.append(("status", run_id))
            return StatusResponse(run_id=run_id, status="succeeded")

        def trace(self, *, run_id: str, node_id: str | None = None) -> TraceResponse:
            calls.append(("trace", run_id, node_id))
            return TraceResponse(run_id=run_id, events=())

        def query_local_rag(self, **kwargs: object) -> QueryResult:
            calls.append(("rag-query", kwargs["run_id"], kwargs["roots"]))
            return QueryResult(scope_id="scope-" + "1" * 24, citations=())

    def fake_service(
        config: Path | None,
        runs_dir: Path | None,
        state_dir: Path | None = None,
    ) -> FakeService:
        assert config is None
        assert runs_dir == tmp_path / "runs"
        calls.append(("service", state_dir))
        return FakeService()

    monkeypatch.setattr("riskprobe.cli._service", fake_service)
    common = ["--runs-dir", str(tmp_path / "runs"), "--run-id", run_id]
    invocations = (
        ["status", *common],
        ["trace", *common],
        [
            "rag-query",
            *common,
            "--state-dir",
            str(tmp_path / "state"),
            "--scope-id",
            "scope-main",
            "--query-id",
            "query-main",
            "--query-text",
            "safe query",
        ],
    )

    for arguments in invocations:
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, (arguments[0], result.stdout, result.exception)

    assert ("status", run_id) in calls
    assert ("trace", run_id, None) in calls
    assert ("rag-query", run_id, {}) in calls


def test_diagnose_without_run_id_uses_authoritative_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from riskprobe.tools import DiagnoseResponse

    authoritative_run_id = "0123456789abcdef"
    finding_id = "a" * 64
    calls: list[tuple[str, object]] = []

    class FakeService:
        def run(self) -> object:
            calls.append(("run", None))
            return SimpleNamespace(run_id=authoritative_run_id)

        def diagnose(self, *, run_id: str) -> DiagnoseResponse:
            calls.append(("diagnose", run_id))
            return DiagnoseResponse(
                dataset_id="synthetic-demo",
                finding_ids=(finding_id,),
            )

    monkeypatch.setattr("riskprobe.cli._service", lambda *args, **kwargs: FakeService())
    result = runner.invoke(
        app,
        [
            "diagnose",
            "--config",
            str(tmp_path / "project.yaml"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, (result.stdout, result.exception)
    assert calls == [("run", None), ("diagnose", authoritative_run_id)]
    payload = json.loads(result.stdout)
    assert payload["run_id"] == authoritative_run_id
    assert payload["finding_ids"] == [finding_id]


def test_recommend_all_current_diagnostics_is_explicit_and_exclusive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from riskprobe.tools import RecommendResponse

    run_id = "0123456789abcdef"
    recommendation_id = "b" * 64
    calls: list[tuple[tuple[str, ...], bool]] = []

    class FakeService:
        def recommend(
            self,
            *,
            run_id: str,
            evidence_ids: tuple[str, ...],
            all_current_diagnostics: bool = False,
        ) -> RecommendResponse:
            assert run_id == "0123456789abcdef"
            calls.append((evidence_ids, all_current_diagnostics))
            return RecommendResponse(
                dataset_id="synthetic-demo",
                recommendation_ids=(recommendation_id,),
            )

    monkeypatch.setattr("riskprobe.cli._service", lambda *args, **kwargs: FakeService())
    common = [
        "recommend",
        "--config",
        str(tmp_path / "project.yaml"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--run-id",
        run_id,
    ]

    success = runner.invoke(app, [*common, "--all-current-diagnostics"])
    ambiguous = runner.invoke(
        app,
        [*common, "--all-current-diagnostics", "--evidence-id", "a" * 64],
    )
    missing = runner.invoke(app, common)

    assert success.exit_code == 0, (success.stdout, success.exception)
    assert calls == [((), True)]
    assert json.loads(success.stdout)["recommendation_ids"] == [recommendation_id]
    assert ambiguous.exit_code == 2
    assert json.loads(ambiguous.stdout)["error"] == "input_error"
    assert missing.exit_code == 2
    assert json.loads(missing.stdout)["error"] == "input_error"


@pytest.mark.parametrize(
    ("mode_value", "expected_mode"),
    (("disabled", "disabled"), ("deterministic", "deterministic")),
)
def test_agent_passes_only_standalone_decision_provider_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode_value: str,
    expected_mode: str,
) -> None:
    from types import SimpleNamespace

    captured: dict[str, object] = {}

    class FakeService:
        config = SimpleNamespace(dataset=SimpleNamespace(id="synthetic-demo"))

        def orchestrate(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(
                session_id="0123456789abcdef",
                status=SimpleNamespace(value="succeeded"),
                review=SimpleNamespace(approved=True, reason_codes=()),
                evidence_ids=("a" * 64,),
                diagnosis_evidence_ids=("a" * 64,),
                retry_count=0,
                tool_sequence=(
                    "inspect",
                    "diagnose",
                    "discover",
                    "recommend",
                    "review",
                ),
            )

    def fake_service(*args: object, **kwargs: object) -> FakeService:
        del args
        captured.update(kwargs)
        return FakeService()

    monkeypatch.setattr("riskprobe.cli._service", fake_service)
    common = [
        "--config",
        str(tmp_path / "project.yaml"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--decision-provider-mode",
    ]

    result = runner.invoke(app, ["agent", *common, mode_value])

    assert result.exit_code == 0, result.stdout
    provider_config = captured["decision_provider_config"]
    assert provider_config.mode.value == expected_mode

    external = runner.invoke(app, ["agent", *common, "external_host"])
    assert external.exit_code == 2
    assert json.loads(external.stdout)["error"] == "argument_error"
