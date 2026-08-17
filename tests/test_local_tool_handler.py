from pathlib import Path
from types import SimpleNamespace

import pytest

from riskprobe.evidence import EvidenceRecord, EvidenceStore
from riskprobe.execution.models import RunStatus
from riskprobe.models import RiskRule
from riskprobe.monitoring.models import (
    DiagnosticReport,
    FindingKind,
    FindingSeverity,
    RiskFinding,
    SafeProfile,
)
from riskprobe.policy import Budget, PolicyEngine, Principal, Role
from riskprobe.profiling import DatasetProfile
from riskprobe.registry import DatasetRegistry
from riskprobe.service import RiskProbeService as DomainService
from riskprobe.tools import (
    DiagnoseRequest,
    DiscoverRequest,
    EvidenceLookupRequest,
    HandlerToolGateway,
    InspectRequest,
    LocalRiskProbeToolHandler,
    RecommendRequest,
    RunRequest,
    StatusRequest,
    ToolContractError,
    TraceRequest,
)

_RUN_ID = "0123456789abcdef"


def _profile(dataset_id: str) -> SafeProfile:
    return SafeProfile(
        dataset_id=dataset_id,
        row_count=100,
        feature_count=2,
        positive_rate=0.2,
        segment_count=2,
        min_segment_size=40,
        max_segment_size=60,
        snapshot_min="2024-01-01",
        snapshot_max="2024-02-01",
        metadata_grade="B",
        issue_codes=("LABEL_PERFORMANCE_WINDOW_UNKNOWN",),
        issue_count=1,
    )


def _report(dataset_id: str) -> DiagnosticReport:
    findings = (
        RiskFinding(
            kind=FindingKind.DATA_QUALITY,
            severity=FindingSeverity.WARNING,
            code="missing_values",
            metrics={"affected_count": 10, "affected_rate": 0.1},
        ),
        RiskFinding(
            kind=FindingKind.FEATURE_DRIFT,
            severity=FindingSeverity.WARNING,
            code="feature_psi",
            feature="order_count",
            metrics={"psi": 0.25},
        ),
    )
    return DiagnosticReport(profile=_profile(dataset_id), findings=findings)


def _install_local_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeService:
        def __init__(self, *, config: object, runs_dir: Path) -> None:
            self.config = config
            self.runs_dir = runs_dir

        def inspect(self) -> DatasetProfile:
            return DatasetProfile(
                dataset_id=self.config.dataset.id,
                row_count=100,
                feature_count=2,
                positive_rate=0.2,
                segment_counts={"redacted-a": 40, "redacted-b": 60},
                snapshot_min=None,
                snapshot_max=None,
                metadata_grade=self.config.metadata_grade,
                issues=(),
            )

        def discover(self) -> list[RiskRule]:
            return [RiskRule(rule_id="rule-1", conditions=(), origin="single")]

        def _diagnose_with_store(
            self,
            run_id: str,
            evidence_store: EvidenceStore,
            *,
            dataset_id: str | None = None,
            run_context: object | None = None,
        ) -> object:
            return DomainService(
                config=self.config,
                runs_dir=self.runs_dir,
            )._diagnose_with_store(
                run_id,
                evidence_store,
                dataset_id=dataset_id,
                run_context=run_context,
            )

        def _recommend_with_store(
            self,
            run_id: str,
            evidence_ids: tuple[str, ...],
            evidence_store: EvidenceStore,
            *,
            dataset_id: str | None = None,
            safe_profile: SafeProfile | None = None,
            decision_controller: object | None = None,
            decision_result_evidence_id: str | None = None,
        ) -> object:
            return DomainService(
                config=self.config,
                runs_dir=self.runs_dir,
            )._recommend_with_store(
                run_id,
                evidence_ids,
                evidence_store,
                dataset_id=dataset_id,
                safe_profile=safe_profile,
                decision_controller=decision_controller,
                decision_result_evidence_id=decision_result_evidence_id,
            )

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(run_id=_RUN_ID, is_existing=False)

    class FakeRuntime:
        def __init__(self, runs_dir: Path, run_id: str) -> None:
            assert run_id == _RUN_ID
            self.runs_dir = runs_dir

        def run_status(self) -> RunStatus:
            return RunStatus.SUCCEEDED

        def trace(self) -> list[dict[str, object]]:
            return [
                {
                    "sequence": 1,
                    "node_id": "profile",
                    "event_type": "node_succeeded",
                    "status": "succeeded",
                    "attempt": 1,
                    "input_fingerprint": "/private/not-public",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ]

    def fake_diagnose(dataset: object, config: object) -> DiagnosticReport:
        del dataset
        return _report(config.dataset.id)

    monkeypatch.setattr("riskprobe.tools.local.RiskProbeService", FakeService)
    monkeypatch.setattr("riskprobe.tools.local.RunRuntime", FakeRuntime)
    monkeypatch.setattr("riskprobe.service.diagnose_dataset", fake_diagnose)


def _gateway(
    *,
    synthetic_config: object,
    handler: LocalRiskProbeToolHandler,
) -> HandlerToolGateway:
    return HandlerToolGateway(
        registry=DatasetRegistry.from_mapping({"synthetic_demo": synthetic_config}),
        policy=PolicyEngine(),
        handler=handler,
    )


def _invoke(gateway: HandlerToolGateway, request: object) -> object:
    return gateway.invoke(
        Principal(principal_id="local-operator", role=Role.OPERATOR),
        request,
        Budget(max_queries=1),
    )


def test_local_handler_covers_all_contracts_with_closed_evidence_and_safe_lookup(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_local_fakes(monkeypatch)
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    handler = LocalRiskProbeToolHandler(
        run_id=_RUN_ID,
        runs_dir=tmp_path / "runs",
        evidence_store=store,
    )
    gateway = _gateway(synthetic_config=synthetic_config, handler=handler)

    inspect = _invoke(gateway, InspectRequest(dataset_id="synthetic_demo"))
    discover = _invoke(gateway, DiscoverRequest(dataset_id="synthetic_demo"))
    diagnose = _invoke(gateway, DiagnoseRequest(dataset_id="synthetic_demo"))
    recommend = _invoke(
        gateway,
        RecommendRequest(
            dataset_id="synthetic_demo",
            evidence_ids=diagnose.finding_ids,
        ),
    )
    run = _invoke(gateway, RunRequest(dataset_id="synthetic_demo"))
    status = _invoke(gateway, StatusRequest(run_id=_RUN_ID))
    trace = _invoke(gateway, TraceRequest(run_id=_RUN_ID))
    lookup = _invoke(
        gateway,
        EvidenceLookupRequest(evidence_id=diagnose.finding_ids[0]),
    )

    assert inspect.metadata_grade == "B"
    assert discover.rule_ids == ("rule-1",)
    assert run.run_id == _RUN_ID
    assert status.status == "succeeded"
    assert trace.events[0].model_dump(mode="json") == {
        "sequence": 1,
        "node_id": "profile",
        "event_type": "node_succeeded",
        "status": "succeeded",
        "attempt": 1,
        "error_class": None,
    }

    finding_records = {evidence_id: store.get(evidence_id) for evidence_id in diagnose.finding_ids}
    assert all(record is not None for record in finding_records.values())
    finding_id_to_evidence_id = {
        str(record.payload["finding_id"]): evidence_id
        for evidence_id, record in finding_records.items()
        if record is not None
    }
    for evidence_id, record in finding_records.items():
        assert record is not None
        assert evidence_id == store.content_id(record)
        assert record.run_id == _RUN_ID
        assert record.kind == "diagnostic.finding"
        assert record.parent_ids == ()

    assert recommend.recommendation_ids
    for evidence_id in recommend.recommendation_ids:
        record = store.get(evidence_id)
        assert record is not None
        assert evidence_id == store.content_id(record)
        assert record.run_id == _RUN_ID
        assert record.kind == "recommendation"
        assert record.payload["decision_eligibility"] == "analysis_only"
        assert record.payload["human_approval_required"] is True
        assert record.parent_ids == tuple(
            sorted(
                finding_id_to_evidence_id[finding_id]
                for finding_id in record.payload["finding_ids"]
            )
        )

    assert lookup.evidence_id == diagnose.finding_ids[0]
    assert lookup.run_id == _RUN_ID
    assert lookup.payload == finding_records[diagnose.finding_ids[0]].payload
    serialized = "".join(
        response.model_dump_json()
        for response in (inspect, discover, diagnose, recommend, run, status, trace, lookup)
    )
    assert str(synthetic_config.dataset.path) not in serialized
    assert "not-public" not in serialized
    assert "raw_rows" not in serialized
    assert "secret" not in serialized


def test_local_handler_lookup_and_run_mismatch_fail_without_private_details(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_local_fakes(monkeypatch)
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    foreign_id = store.append(
        EvidenceRecord(
            run_id="fedcba9876543210",
            kind="diagnostic.finding",
            payload={"finding_id": "a" * 64},
            producer_version="local-test/1",
        )
    )
    handler = LocalRiskProbeToolHandler(
        run_id=_RUN_ID,
        runs_dir=tmp_path / "runs",
        evidence_store=store,
    )
    gateway = _gateway(synthetic_config=synthetic_config, handler=handler)

    for evidence_id in (foreign_id, "f" * 64):
        with pytest.raises(ToolContractError) as exc_info:
            _invoke(gateway, EvidenceLookupRequest(evidence_id=evidence_id))
        assert str(exc_info.value) == "tool handler failed"
        assert str(synthetic_config.dataset.path) not in str(exc_info.value)

    class WrongRunService:
        def __init__(self, *, config: object, runs_dir: Path) -> None:
            del config, runs_dir

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(run_id="fedcba9876543210", is_existing=False)

    monkeypatch.setattr("riskprobe.tools.local.RiskProbeService", WrongRunService)
    with pytest.raises(ToolContractError) as exc_info:
        _invoke(gateway, RunRequest(dataset_id="synthetic_demo"))
    assert str(exc_info.value) == "tool handler failed"
