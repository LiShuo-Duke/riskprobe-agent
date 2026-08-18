from pathlib import Path

import pytest

from riskprobe.runtime import RunRuntime
from riskprobe.service import RiskProbeService

_RUN_ONE = "0123456789abcdef"
_RUN_TWO = "fedcba9876543210"
_RUN_NESTED = "1111111111111111"
_RUN_RUNTIME = "2222222222222222"


def test_service_evidence_workflow_uses_sidecars_and_closes_grade_b(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=runs_dir,
        state_dir=state_dir,
    )

    diagnosis = service.diagnose(run_id=_RUN_ONE)
    recommendations = service.recommend(
        run_id=_RUN_ONE,
        evidence_ids=diagnosis.finding_ids,
    )

    assert diagnosis.finding_ids
    assert recommendations.recommendation_ids
    for evidence_id in recommendations.recommendation_ids:
        record = service.evidence(run_id=_RUN_ONE, evidence_id=evidence_id)
        assert record.kind == "recommendation"
        assert record.parent_ids
        assert set(record.parent_ids).issubset(diagnosis.finding_ids)
        assert record.payload["decision_eligibility"] == "analysis_only"
        assert record.payload["human_approval_required"] is True

    assert list(state_dir.glob(f".{_RUN_ONE}.evidence.sqlite3"))
    assert not (runs_dir / _RUN_ONE).exists()


def test_service_evidence_workflow_rejects_cross_run_access(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
    )
    diagnosis = service.diagnose(run_id=_RUN_ONE)

    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        service.evidence(run_id=_RUN_TWO, evidence_id=diagnosis.finding_ids[0])
    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        service.recommend(run_id=_RUN_TWO, evidence_ids=diagnosis.finding_ids)


def test_service_rejects_state_directory_inside_run_artifact_directory(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    runs_dir = tmp_path / "runs"
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=runs_dir,
        state_dir=runs_dir / _RUN_NESTED / "state",
    )

    with pytest.raises(ValueError, match="state directory"):
        service.diagnose(run_id=_RUN_NESTED)


def test_service_status_and_trace_are_safe_projections(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runtime = RunRuntime(runs_dir, _RUN_RUNTIME)
    runtime.start_node("profile", input_fingerprint="secret-input-fingerprint")
    runtime.succeed_node(
        "profile",
        input_fingerprint="secret-input-fingerprint",
        output={"private": "must-not-escape"},
    )
    service = RiskProbeService(config=None, runs_dir=runs_dir)

    status = service.status(run_id=_RUN_RUNTIME)
    trace = service.trace(run_id=_RUN_RUNTIME)

    assert status.run_id == _RUN_RUNTIME
    assert status.status == "running"
    assert trace.run_id == _RUN_RUNTIME
    assert trace.events
    serialized = trace.model_dump_json()
    assert "secret-input-fingerprint" not in serialized
    assert "must-not-escape" not in serialized
    assert "timestamp" not in serialized


def test_service_orchestrate_assembles_closed_local_agent(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    from types import SimpleNamespace

    from riskprobe.monitoring.models import (
        DiagnosticReport,
        FindingKind,
        FindingSeverity,
        RiskFinding,
        SafeProfile,
    )
    from riskprobe.policy import Budget, Principal, Role
    from riskprobe.profiling import DatasetProfile

    profile = DatasetProfile(
        dataset_id=synthetic_config.dataset.id,
        row_count=100,
        feature_count=2,
        positive_rate=0.2,
        segment_counts={"redacted-a": 40, "redacted-b": 60},
        snapshot_min=None,
        snapshot_max=None,
        metadata_grade="B",
        issues=(),
    )
    finding = RiskFinding(
        kind=FindingKind.DATA_QUALITY,
        severity=FindingSeverity.WARNING,
        code="missing_values",
        metrics={"affected_count": 10, "affected_rate": 0.1},
    )
    report = DiagnosticReport(
        profile=SafeProfile.from_profile(profile),
        findings=(finding,),
    )

    monkeypatch.setattr(
        RiskProbeService,
        "run",
        lambda self: SimpleNamespace(run_id=_RUN_ONE, is_existing=False),
    )
    monkeypatch.setattr(RiskProbeService, "inspect", lambda self: profile)
    monkeypatch.setattr(
        RiskProbeService,
        "discover",
        lambda self: [],
    )
    monkeypatch.setattr("riskprobe.service.diagnose_dataset", lambda dataset, config: report)

    def network_forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", network_forbidden)
    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", network_forbidden)
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
    )

    result = service.orchestrate(
        dataset_id=synthetic_config.dataset.id,
        principal=Principal(principal_id="local-analyst", role=Role.ANALYST),
        budget=Budget(max_queries=8),
    )

    assert result.session_id == _RUN_ONE
    assert result.status.value == "succeeded"
    session_path = tmp_path / "state" / f".{_RUN_ONE}.sessions.sqlite3"
    evidence_path = tmp_path / "state" / f".{_RUN_ONE}.evidence.sqlite3"
    assert session_path.is_file()
    assert evidence_path.is_file()
    private_path = str(synthetic_config.dataset.path).encode("utf-8")
    assert private_path not in session_path.read_bytes()
    assert private_path not in evidence_path.read_bytes()
    assert not (tmp_path / "runs" / _RUN_ONE).exists()


def test_service_local_rag_uses_sidecars_and_returns_citations_only(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    import hashlib
    import json

    root = tmp_path / "provider-safe-root"
    root.mkdir()
    document = root / "guide.md"
    content = b"Aggregate risk monitoring guidance"
    document.write_bytes(content)
    manifest = {
        "format_version": 1,
        "documents": [
            {
                "path": "guide.md",
                "content_hash": hashlib.sha256(content).hexdigest(),
                "privacy_class": "provider_safe",
            }
        ],
    }
    (root / ".riskprobe-rag-manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
    )

    built = service.build_local_rag(
        run_id=_RUN_ONE,
        roots={"docs-root": root},
        root_id="docs-root",
        scope_id="scope-main",
    )
    query_service = RiskProbeService(
        config=None,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
    )
    queried = query_service.query_local_rag(
        run_id=_RUN_ONE,
        scope_id="scope-main",
        query_id="query-main",
        query_text="risk monitoring",
    )

    assert built.document_count == 1
    assert queried.citations
    serialized = queried.model_dump_json()
    assert str(root) not in serialized
    assert "Aggregate risk monitoring guidance" not in serialized
    index_name = f"riskprobe_{_RUN_ONE}_rag_index.json"
    index_path = state_dir / index_name
    assert index_path.is_file()
    assert (state_dir / f"{index_name}.key").is_file()
    stored = index_path.read_bytes()
    assert content not in stored
    assert str(root).encode("utf-8") not in stored
    assert b"risk monitoring" not in stored
    assert not (tmp_path / "runs" / _RUN_ONE).exists()


def test_service_eval_wrappers_preserve_v1_and_v2_contracts(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    from riskprobe.evals import (
        EvalCase,
        EvalCaseV2,
        EvalObservation,
        EvalObservationV2,
        EvalSuite,
        EvalSuiteV2,
    )

    base_case = EvalCase(
        case_id="service-eval",
        objective="comprehensive",
        expected_tool_sequence=("inspect",),
    )
    base_observation = EvalObservation(
        case_id="service-eval",
        task_succeeded=True,
        tool_sequence=("inspect",),
        evidence_ids=(),
        diagnosis_evidence_ids=(),
        policy_violations=0,
        privacy_violations=0,
    )
    service = RiskProbeService(config=synthetic_config, runs_dir=tmp_path / "runs")
    v1_suite = EvalSuite(suite_id="service-v1", cases=(base_case,))
    v1_report = service.evaluate_v1(
        v1_suite,
        lambda case, seed: base_observation,
        candidate_version="candidate-v1",
    )

    v2_case = EvalCaseV2(
        base_case=base_case,
        expected_rule_ids=(),
        drift_universe_ids=(),
        drift_ground_truth_ids=(),
        diagnosis_relevant_ids=(),
        diagnosis_k=1,
        expected_recommendation_ids=(),
    )
    v2_observation = EvalObservationV2(
        base_observation=base_observation,
        recovered_rule_ids=(),
        detected_drift_ids=(),
        diagnosis_ranked_ids=(),
        recommendation_ids=(),
    )
    v2_suite = EvalSuiteV2(suite_id="service-v2", cases=(v2_case,))
    v2_report = service.evaluate_v2(
        v2_suite,
        lambda case, seed: v2_observation,
        candidate_version="candidate-v2",
    )

    assert v1_report.verify_integrity() is True
    assert v2_report.verify_integrity() is True


def test_service_recommend_all_current_uses_only_immediate_diagnosis(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.tools import DiagnoseResponse, RecommendResponse

    finding_ids = ("a" * 64, "b" * 64)
    recommendation_id = "c" * 64
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
    )
    calls: list[tuple[str, str, tuple[str, ...], object]] = []

    def fake_diagnose(
        run_id: str,
        evidence_store: object,
        *,
        dataset_id: str | None = None,
    ) -> DiagnoseResponse:
        calls.append(("diagnose", run_id, (), evidence_store))
        return DiagnoseResponse(
            dataset_id=dataset_id or synthetic_config.dataset.id,
            finding_ids=finding_ids,
        )

    def fake_recommend(
        run_id: str,
        evidence_ids: tuple[str, ...],
        evidence_store: object,
        *,
        dataset_id: str | None = None,
        safe_profile: object | None = None,
    ) -> RecommendResponse:
        del safe_profile
        assert dataset_id == synthetic_config.dataset.id
        calls.append(("recommend", run_id, evidence_ids, evidence_store))
        return RecommendResponse(
            dataset_id=dataset_id,
            recommendation_ids=(recommendation_id,),
        )

    monkeypatch.setattr(service, "_diagnose_with_store", fake_diagnose)
    monkeypatch.setattr(service, "_recommend_with_store", fake_recommend)

    response = service.recommend(
        run_id=_RUN_ONE,
        evidence_ids=(),
        all_current_diagnostics=True,
    )

    assert response.recommendation_ids == (recommendation_id,)
    assert calls[0][:3] == ("diagnose", _RUN_ONE, ())
    assert calls[1][:3] == ("recommend", _RUN_ONE, finding_ids)
    assert calls[0][3] is calls[1][3]

    with pytest.raises(ValueError, match="either evidence_ids"):
        service.recommend(run_id=_RUN_ONE, evidence_ids=())
    with pytest.raises(ValueError, match="either evidence_ids"):
        service.recommend(
            run_id=_RUN_ONE,
            evidence_ids=finding_ids,
            all_current_diagnostics=True,
        )

    def empty_diagnose(
        run_id: str,
        evidence_store: object,
        *,
        dataset_id: str | None = None,
    ) -> DiagnoseResponse:
        del run_id, evidence_store
        return DiagnoseResponse(
            dataset_id=dataset_id or synthetic_config.dataset.id,
            finding_ids=(),
        )

    monkeypatch.setattr(service, "_diagnose_with_store", empty_diagnose)
    with pytest.raises(ValueError, match="current diagnostics are unavailable"):
        service.recommend(
            run_id=_RUN_ONE,
            evidence_ids=(),
            all_current_diagnostics=True,
        )
