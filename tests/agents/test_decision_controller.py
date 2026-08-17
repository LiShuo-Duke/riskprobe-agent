import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionProposal,
    DecisionReason,
    DecisionSource,
    DecisionStatus,
)
from riskprobe.agents.decision_controller import (
    DecisionController,
    DecisionControllerError,
    _DecisionUnavailableReason,
)
from riskprobe.agents.decision_providers import (
    DecisionProviderMode,
    _DecisionProviderBinding,
    _DecisionProviderIdentity,
    _DecisionProviderRole,
)
from riskprobe.evidence import (
    EvidenceParentError,
    EvidenceRecord,
    EvidenceStore,
)
from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
from riskprobe.recommendations.policy import ActionCode
from riskprobe.tools import DiscoverResponse, InspectResponse

_RUN_ID = "0123456789abcdef"
_DATASET_ID = "synthetic_demo"
_ANCHOR_ID = "f" * 64
_ISSUED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_SUBMITTED_AT = _ISSUED_AT + timedelta(seconds=5)


def _finding(kind: FindingKind, code: str) -> RiskFinding:
    return RiskFinding(
        kind=kind,
        severity=FindingSeverity.WARNING,
        code=code,
        metrics={"affected_rate": 0.25},
    )


def _append_finding(
    store: EvidenceStore,
    finding: RiskFinding,
    *,
    run_id: str = _RUN_ID,
    kind: str = "diagnostic.finding",
    parent_ids: tuple[str, ...] = (),
) -> str:
    return store.append(
        EvidenceRecord(
            run_id=run_id,
            kind=kind,
            payload={
                **finding.model_dump(mode="json"),
                "dataset_id": _DATASET_ID,
            },
            parent_ids=parent_ids,
            producer_version="diagnostics-v1",
        )
    )


def _responses() -> tuple[InspectResponse, DiscoverResponse]:
    return (
        InspectResponse(
            dataset_id=_DATASET_ID,
            row_count=100,
            feature_count=8,
            metadata_grade="A",
            issue_codes=("LABEL_PERFORMANCE_WINDOW_UNKNOWN",),
        ),
        DiscoverResponse(
            dataset_id=_DATASET_ID,
            rule_ids=("rule-b", "rule-a"),
        ),
    )


def _prepare(
    store: EvidenceStore,
    diagnosis_ids: tuple[str, ...],
    *,
    now: datetime = _ISSUED_AT,
):
    inspect, discover = _responses()
    controller = DecisionController(store, clock=lambda: now)
    preparation = controller.prepare(
        session_id=_RUN_ID,
        attempt=0,
        anchor_node_id=_ANCHOR_ID,
        diagnosis_evidence_ids=diagnosis_ids,
        inspect_response=inspect,
        discover_response=discover,
        orchestrator_version="orchestrator-v1",
        planner_version="planner-v1",
    )
    return controller, preparation


def _proposal(context: DecisionContext) -> DecisionProposal:
    return DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )


def _provider_binding() -> _DecisionProviderBinding:
    primary = _DecisionProviderIdentity(
        provider_id="kiro-host",
        mode=DecisionProviderMode.EXTERNAL_HOST,
        version="kiro-host-v1",
    )
    fallback = _DecisionProviderIdentity(
        provider_id="deterministic",
        mode=DecisionProviderMode.DETERMINISTIC,
        version="deterministic-decision-provider-v1",
    )
    return _DecisionProviderBinding(
        primary=primary,
        fallback=fallback,
        selected=primary,
        selected_role=_DecisionProviderRole.PRIMARY,
    )


def test_prepare_and_fresh_controller_submit_persist_exact_parent_graph(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    data_quality_id = _append_finding(
        store,
        _finding(FindingKind.DATA_QUALITY, "missing_values"),
    )
    drift_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    diagnosis_ids = tuple(sorted((data_quality_id, drift_id)))

    _, preparation = _prepare(store, diagnosis_ids)

    assert preparation.context.diagnosis_evidence_ids == diagnosis_ids
    assert tuple(
        item.evidence_id for item in preparation.context.findings
    ) == diagnosis_ids
    assert preparation.context.rule_ids == ("rule-a", "rule-b")
    context_record = store.get(preparation.context_evidence_id)
    assert context_record is not None
    assert context_record.kind == "decision.context"
    assert context_record.run_id == _RUN_ID
    assert context_record.parent_ids == diagnosis_ids
    assert (
        DecisionContext.model_validate_json(
            json.dumps(dict(context_record.payload), sort_keys=True)
        )
        == preparation.context
    )

    fresh = DecisionController(store, clock=lambda: _SUBMITTED_AT)
    submission = fresh.submit(
        context_evidence_id=preparation.context_evidence_id,
        proposal=_proposal(preparation.context),
        provider_binding=_provider_binding(),
    )

    assert submission.result.status is DecisionStatus.ACCEPTED
    assert submission.submitted_at == _SUBMITTED_AT
    proposal_record = store.get(submission.proposal_evidence_id)
    result_record = store.get(submission.result_evidence_id)
    assert proposal_record is not None
    assert result_record is not None
    assert proposal_record.kind == "decision.proposal"
    assert proposal_record.parent_ids == (preparation.context_evidence_id,)
    assert proposal_record.payload["submitted_at"] == "2026-01-01T12:00:05Z"
    assert proposal_record.payload["proposal"] == submission.proposal.model_dump(mode="json")
    assert proposal_record.payload["provider_binding"] == _provider_binding().model_dump(
        mode="json"
    )
    assert submission.provider_binding == _provider_binding()
    assert result_record.kind == "decision.result"
    assert result_record.parent_ids == (
        preparation.context_evidence_id,
        submission.proposal_evidence_id,
    )
    assert result_record.payload == submission.result.model_dump(mode="json")

    replayed = DecisionController(
        store,
        clock=lambda: preparation.context.expires_at + timedelta(days=1),
    ).replay(
        result_evidence_id=submission.result_evidence_id,
        expected_run_id=_RUN_ID,
    )
    assert replayed == submission


def test_submit_records_exact_expiry_as_authoritative_rejection(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))

    rejected = DecisionController(
        store,
        clock=lambda: preparation.context.expires_at,
    ).submit(
        context_evidence_id=preparation.context_evidence_id,
        proposal=_proposal(preparation.context),
        provider_binding=_provider_binding(),
    )

    assert rejected.result.status is DecisionStatus.REJECTED
    assert rejected.result.reason_codes == (DecisionReason.CONTEXT_EXPIRED,)
    assert rejected.result.action_codes == ()
    assert DecisionController(
        store,
        clock=lambda: preparation.context.issued_at,
    ).replay(
        result_evidence_id=rejected.result_evidence_id,
        expected_run_id=_RUN_ID,
    ) == rejected


def test_prepare_requires_the_complete_run_diagnosis_set(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    selected_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _append_finding(
        store,
        _finding(FindingKind.DATA_QUALITY, "missing_values"),
    )

    with pytest.raises(
        DecisionControllerError,
        match="^decision context is unavailable$",
    ):
        _prepare(store, (selected_id,))


def test_submit_rejects_unknown_wrong_kind_parent_and_canonical_tamper(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))
    proposal = _proposal(preparation.context)

    with pytest.raises(
        DecisionControllerError,
        match="^decision context is unavailable$",
    ):
        DecisionController(store, clock=lambda: _SUBMITTED_AT).submit(
            context_evidence_id="0" * 64,
            proposal=proposal,
            provider_binding=_provider_binding(),
        )

    wrong_kind_id = store.append(
        EvidenceRecord(
            run_id=_RUN_ID,
            kind="decision.proposal",
            payload=preparation.context.model_dump(mode="json"),
            parent_ids=(finding_id,),
            producer_version="decision-controller-v1",
        )
    )
    wrong_parent_id = store.append(
        EvidenceRecord(
            run_id=_RUN_ID,
            kind="decision.context",
            payload=preparation.context.model_dump(mode="json"),
            producer_version="decision-controller-v1",
        )
    )
    tampered_payload = preparation.context.model_dump(mode="json")
    tampered_payload["context_id"] = "0" * 64
    tampered_id = store.append(
        EvidenceRecord(
            run_id=_RUN_ID,
            kind="decision.context",
            payload=tampered_payload,
            parent_ids=(finding_id,),
            producer_version="decision-controller-v1",
        )
    )

    for context_evidence_id in (wrong_kind_id, wrong_parent_id, tampered_id):
        with pytest.raises(
            DecisionControllerError,
            match="^decision context is unavailable$",
        ):
            DecisionController(store, clock=lambda: _SUBMITTED_AT).submit(
                context_evidence_id=context_evidence_id,
                proposal=proposal,
                provider_binding=_provider_binding(),
            )


def test_submit_rejects_tampered_proposal_and_duplicate_submission(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))
    proposal = _proposal(preparation.context)
    controller = DecisionController(store, clock=lambda: _SUBMITTED_AT)
    tampered = DecisionProposal.model_construct(
        **{
            **proposal.model_dump(mode="python"),
            "proposal_id": "0" * 64,
        }
    )

    with pytest.raises(
        DecisionControllerError,
        match="^decision proposal is unavailable$",
    ):
        controller.submit(
            context_evidence_id=preparation.context_evidence_id,
            proposal=tampered,
            provider_binding=_provider_binding(),
        )

    controller.submit(
        context_evidence_id=preparation.context_evidence_id,
        proposal=proposal,
        provider_binding=_provider_binding(),
    )
    with pytest.raises(
        DecisionControllerError,
        match="^decision submission already exists$",
    ):
        DecisionController(store, clock=lambda: _SUBMITTED_AT).submit(
            context_evidence_id=preparation.context_evidence_id,
            proposal=proposal,
            provider_binding=_provider_binding(),
        )


def test_replay_rejects_malformed_competing_unavailable_record(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))
    controller = DecisionController(store, clock=lambda: _SUBMITTED_AT)
    submission = controller.submit(
        context_evidence_id=preparation.context_evidence_id,
        proposal=_proposal(preparation.context),
        provider_binding=_provider_binding(),
    )
    store.append(
        EvidenceRecord(
            run_id=_RUN_ID,
            kind="decision.unavailable",
            payload={
                "context_id": preparation.context.context_id,
                "reason": "provider_error",
            },
            parent_ids=(preparation.context_evidence_id, finding_id),
            producer_version=controller.version,
        )
    )

    with pytest.raises(
        DecisionControllerError,
        match="^decision audit is unavailable$",
    ):
        controller.replay(
            result_evidence_id=submission.result_evidence_id,
            expected_run_id=_RUN_ID,
        )


def test_replay_unavailable_rejects_malformed_competing_proposal(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    controller, preparation = _prepare(store, (finding_id,))
    outcome = controller.record_unavailable(
        context_evidence_id=preparation.context_evidence_id,
        reason=_DecisionUnavailableReason.PROVIDER_ERROR,
        provider_binding=_provider_binding(),
    )
    store.append(
        EvidenceRecord(
            run_id=_RUN_ID,
            kind="decision.proposal",
            payload={"context_id": preparation.context.context_id},
            parent_ids=(preparation.context_evidence_id, finding_id),
            producer_version=controller.version,
        )
    )

    with pytest.raises(
        DecisionControllerError,
        match="^decision audit is unavailable$",
    ):
        controller.replay_unavailable(
            outcome_evidence_id=outcome.outcome_evidence_id,
            expected_run_id=_RUN_ID,
        )


def test_submit_uses_one_atomic_batch_for_proposal_and_result(
    tmp_path: Path,
) -> None:
    class TrackingEvidenceStore(EvidenceStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.appended_kinds: list[str] = []
            self.batch_sizes: list[int] = []

        def append(self, record: EvidenceRecord) -> str:
            self.appended_kinds.append(record.kind)
            return super().append(record)

        def append_many(
            self,
            records: Sequence[EvidenceRecord],
        ) -> tuple[str, ...]:
            batch = tuple(records)
            self.batch_sizes.append(len(batch))
            return super().append_many(batch)

    store = TrackingEvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))
    store.appended_kinds.clear()
    store.batch_sizes.clear()

    DecisionController(store, clock=lambda: _SUBMITTED_AT).submit(
        context_evidence_id=preparation.context_evidence_id,
        proposal=_proposal(preparation.context),
        provider_binding=_provider_binding(),
    )

    assert store.appended_kinds == []
    assert store.batch_sizes == [2]


def test_submit_batch_failure_does_not_leave_an_orphan_proposal(
    tmp_path: Path,
) -> None:
    class FailingSubmissionStore(EvidenceStore):
        fail_submission = False

        def append_many(
            self,
            records: Sequence[EvidenceRecord],
        ) -> tuple[str, ...]:
            batch = tuple(records)
            if self.fail_submission:
                if len(batch) == 1 and batch[0].kind == "decision.result":
                    raise EvidenceParentError("parent evidence is unavailable")
                if len(batch) == 2:
                    result = batch[1].model_copy(
                        update={
                            "parent_ids": (
                                batch[1].parent_ids[0],
                                "0" * 64,
                            )
                        }
                    )
                    return super().append_many((batch[0], result))
            return super().append_many(batch)

    store = FailingSubmissionStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))
    store.fail_submission = True

    with pytest.raises(
        DecisionControllerError,
        match="^decision submission is unavailable$",
    ):
        DecisionController(store, clock=lambda: _SUBMITTED_AT).submit(
            context_evidence_id=preparation.context_evidence_id,
            proposal=_proposal(preparation.context),
            provider_binding=_provider_binding(),
        )

    assert not {
        record.kind for record in store.list_run(_RUN_ID)
    }.intersection({"decision.proposal", "decision.result"})


def test_concurrent_fresh_controllers_allow_only_one_submission(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding_id = _append_finding(
        store,
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
    )
    _, preparation = _prepare(store, (finding_id,))
    proposal = _proposal(preparation.context)
    barrier = Barrier(2)

    def submit() -> object:
        def clock() -> datetime:
            barrier.wait(timeout=5)
            return _SUBMITTED_AT

        try:
            return DecisionController(store, clock=clock).submit(
                context_evidence_id=preparation.context_evidence_id,
                proposal=proposal,
                provider_binding=_provider_binding(),
            )
        except DecisionControllerError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: submit(), range(2)))

    submissions = [
        outcome for outcome in outcomes if not isinstance(outcome, Exception)
    ]
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(submissions) == 1
    assert len(errors) == 1
    assert str(errors[0]) == "decision submission already exists"
