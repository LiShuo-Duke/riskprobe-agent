import builtins
import hashlib
import json
import sqlite3
import socket
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.evals import EvalCase, EvalHarness, EvalObservation, EvalReport, EvalSuite
from riskprobe.evolution import (
    CandidateVersion,
    ContentKind,
    EvaluationGate,
    EvolutionIntegrityError,
    EvolutionRegistry,
    HumanApproval,
)

_EVIDENCE_A = "a" * 64
_SEQUENCE = ("inspect", "diagnose", "discover", "recommend", "review")
_APPROVER = "approver.alice"


def _suite(
    *,
    suite_id: str = "frozen-suite-v1",
    seed: int = 42,
    objective: str = "comprehensive",
) -> EvalSuite:
    return EvalSuite(
        suite_id=suite_id,
        seed=seed,
        cases=(
            EvalCase(
                case_id="case-1",
                objective=objective,
                expected_tool_sequence=_SEQUENCE,
                required_evidence_ids=(_EVIDENCE_A,),
            ),
        ),
    )


def _report(
    *,
    passed: bool,
    version: str,
    suite: EvalSuite | None = None,
) -> EvalReport:
    frozen_suite = suite or _suite()

    def runner(case: EvalCase, seed: int) -> EvalObservation:
        del case, seed
        return EvalObservation(
            case_id="case-1",
            task_succeeded=passed,
            tool_sequence=_SEQUENCE,
            evidence_ids=(_EVIDENCE_A,),
            diagnosis_evidence_ids=(_EVIDENCE_A,),
            policy_violations=0,
            privacy_violations=0,
        )

    return EvalHarness(seed=frozen_suite.seed).evaluate(
        frozen_suite,
        runner,
        candidate_version=version,
    )


def _registry(path: Path, *suites: EvalSuite) -> EvolutionRegistry:
    trusted = suites or (_suite(),)
    return EvolutionRegistry(
        path,
        trusted_suites=trusted,
        allowed_approver_ids=(_APPROVER,),
    )


def _approval(
    candidate_version_id: str,
    *,
    approver_id: str = _APPROVER,
    action: str = "promote",
) -> HumanApproval:
    return HumanApproval.attest(
        action=action,
        candidate_version_id=candidate_version_id,
        approver_id=approver_id,
    )


def _hash(character: str) -> str:
    return character * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _create_v1_fixture(path: Path, suite: EvalSuite) -> tuple[CandidateVersion, str]:
    report = _report(passed=True, version="candidate-v1", suite=suite)
    candidate = CandidateVersion(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_gate=EvaluationGate.from_report(report),
    )
    old_report_payload = {
        "action": "promote",
        "active_version_id": candidate.version_id,
        "candidate_version_id": candidate.version_id,
        "eval_passed": True,
        "human_approved": True,
        "previous_active_version_id": None,
        "promoted": True,
        "reason_codes": [],
    }
    old_report_hash = _payload_hash(old_report_payload)
    old_report_json = _canonical_json(
        {**old_report_payload, "report_hash": old_report_hash}
    )
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE candidate_versions (
                version_id TEXT PRIMARY KEY,
                parent_version_id TEXT REFERENCES candidate_versions(version_id),
                content_hashes_json TEXT NOT NULL,
                eval_gate_json TEXT NOT NULL,
                candidate_json TEXT NOT NULL
            );
            CREATE TABLE evolution_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                active_version_id TEXT REFERENCES candidate_versions(version_id)
            );
            CREATE TABLE promotion_history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                candidate_version_id TEXT NOT NULL REFERENCES candidate_versions(version_id),
                previous_active_version_id TEXT,
                active_version_id TEXT,
                promoted INTEGER NOT NULL,
                report_json TEXT NOT NULL
            );
            CREATE TRIGGER candidate_versions_no_update
            BEFORE UPDATE ON candidate_versions
            BEGIN
                SELECT RAISE(ABORT, 'candidate versions are immutable');
            END;
            CREATE TRIGGER candidate_versions_no_delete
            BEFORE DELETE ON candidate_versions
            BEGIN
                SELECT RAISE(ABORT, 'candidate versions are immutable');
            END;
            CREATE TRIGGER promotion_history_no_update
            BEFORE UPDATE ON promotion_history
            BEGIN
                SELECT RAISE(ABORT, 'promotion history is append-only');
            END;
            CREATE TRIGGER promotion_history_no_delete
            BEFORE DELETE ON promotion_history
            BEGIN
                SELECT RAISE(ABORT, 'promotion history is append-only');
            END;
            PRAGMA user_version = 1;
            """
        )
        candidate_payload = candidate.model_dump(mode="json")
        connection.execute(
            """
            INSERT INTO candidate_versions (
                version_id, parent_version_id, content_hashes_json,
                eval_gate_json, candidate_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                candidate.version_id,
                None,
                _canonical_json(candidate_payload["content_hashes"]),
                _canonical_json(candidate.eval_gate.model_dump(mode="json")),
                _canonical_json(candidate_payload),
            ),
        )
        connection.execute(
            "INSERT INTO evolution_state (singleton, active_version_id) VALUES (1, ?)",
            (candidate.version_id,),
        )
        connection.execute(
            """
            INSERT INTO promotion_history (
                action, candidate_version_id, previous_active_version_id,
                active_version_id, promoted, report_json
            ) VALUES ('promote', ?, NULL, ?, 1, ?)
            """,
            (candidate.version_id, candidate.version_id, old_report_json),
        )
        connection.commit()
    finally:
        connection.close()
    return candidate, old_report_hash


def test_registry_versions_only_allowlisted_content_hashes_and_not_content(
    tmp_path: Path,
) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    candidate = registry.register_candidate(
        content_hashes={
            ContentKind.PROMPT: _hash("1"),
            ContentKind.TEMPLATE: _hash("2"),
            ContentKind.POLICY: _hash("3"),
            ContentKind.CONFIG: _hash("4"),
        },
        eval_report=_report(passed=True, version="candidate-v1", suite=suite),
    )

    assert registry.get(candidate.version_id) == candidate
    assert set(candidate.content_hashes) == set(ContentKind)
    database_bytes = registry.path.read_bytes()
    assert b"raw prompt body" not in database_bytes
    assert len(candidate.version_id) == 64


def test_candidate_dto_rejects_code_path_executable_and_nonhash_fields() -> None:
    with pytest.raises(ValidationError):
        CandidateVersion.model_validate(
            {
                "version_id": "a" * 64,
                "content_hashes": {"prompt": "b" * 64},
                "eval_gate": {
                    "suite_id": "suite-v1",
                    "suite_hash": "c" * 64,
                    "report_hash": "d" * 64,
                    "seed": 42,
                    "passed": True,
                },
                "code": "import os",
            }
        )
    with pytest.raises(ValidationError):
        CandidateVersion.model_validate(
            {
                "version_id": "a" * 64,
                "content_hashes": {"path": "/tmp/candidate.py"},
                "eval_gate": {
                    "suite_id": "suite-v1",
                    "suite_hash": "c" * 64,
                    "report_hash": "d" * 64,
                    "seed": 42,
                    "passed": True,
                },
            }
        )


def test_candidate_versions_are_immutable_in_dto_and_sqlite(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    candidate = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=_report(passed=True, version="candidate-v1", suite=suite),
    )

    with pytest.raises(ValidationError):
        candidate.content_hashes = {ContentKind.PROMPT: _hash("2")}
    connection = sqlite3.connect(registry.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE candidate_versions SET content_hashes_json = '{}' WHERE version_id = ?",
                (candidate.version_id,),
            )
    finally:
        connection.close()


def test_trusted_suite_requires_exact_id_hash_seed_and_real_v1_report(
    tmp_path: Path,
) -> None:
    trusted = _suite()
    same_id_other_hash = _suite(objective="alternate")
    same_id_other_seed = _suite(seed=7)
    registry = _registry(tmp_path / "evolution.sqlite3", trusted)

    for index, suite in enumerate((same_id_other_hash, same_id_other_seed), start=1):
        with pytest.raises(ValueError, match="trusted suite"):
            registry.register_candidate(
                content_hashes={ContentKind.PROMPT: str(index) * 64},
                eval_report=_report(passed=True, version=f"candidate-{index}", suite=suite),
            )

    valid_report = _report(passed=True, version="candidate-valid", suite=trusted)
    forged_report = valid_report.model_copy(update={"report_hash": _hash("f")})
    with pytest.raises(ValueError, match="integrity"):
        registry.register_candidate(
            content_hashes={ContentKind.PROMPT: _hash("3")},
            eval_report=forged_report,
        )

    class ReportLike:
        def verify_integrity(self) -> bool:
            return True

    with pytest.raises(TypeError, match="EvalReport"):
        registry.register_candidate(
            content_hashes={ContentKind.PROMPT: _hash("4")},
            eval_report=ReportLike(),  # type: ignore[arg-type]
        )


def test_registry_rejects_invalid_or_conflicting_trusted_suite_configuration(
    tmp_path: Path,
) -> None:
    trusted = _suite()
    forged = trusted.model_copy(update={"suite_hash": _hash("f")})
    with pytest.raises(ValueError, match="integrity"):
        _registry(tmp_path / "forged.sqlite3", forged)
    with pytest.raises(ValueError, match="conflicting"):
        _registry(tmp_path / "conflicting.sqlite3", trusted, _suite(objective="alternate"))


def test_bootstrap_is_explicit_and_cannot_carry_a_forged_comparison(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    report = _report(passed=True, version="candidate-v1", suite=suite)
    candidate = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=report,
    )
    approval = _approval(candidate.version_id)

    with pytest.raises(ValueError, match="bootstrap"):
        registry.promote(candidate.version_id, approval=approval)
    self_comparison = EvalHarness().compare(report, report)
    with pytest.raises(ValueError, match="bootstrap"):
        registry.promote(
            candidate.version_id,
            approval=approval,
            bootstrap=True,
            baseline_report=report,
            comparison=self_comparison,
        )
    assert registry.promotion_history() == ()
    assert registry.audit_history() == ()

    promoted = registry.promote(
        candidate.version_id,
        approval=approval,
        bootstrap=True,
    )
    assert promoted.promoted is True
    assert registry.active_version() == candidate


def test_active_promotion_requires_current_baseline_and_bound_comparison(
    tmp_path: Path,
) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    harness = EvalHarness()
    first_report = _report(passed=True, version="candidate-v1", suite=suite)
    first = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=first_report,
    )
    registry.promote(
        first.version_id,
        approval=_approval(first.version_id),
        bootstrap=True,
    )
    second_report = _report(passed=True, version="candidate-v2", suite=suite)
    second = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("2")},
        eval_report=second_report,
        parent_version_id=first.version_id,
    )

    with pytest.raises(ValueError, match="bootstrap"):
        registry.promote(
            second.version_id,
            approval=_approval(second.version_id),
            bootstrap=True,
        )
    with pytest.raises(ValueError, match="baseline"):
        registry.promote(second.version_id, approval=_approval(second.version_id))

    stale_report = _report(passed=True, version="candidate-stale", suite=suite)
    with pytest.raises(ValueError, match="current active"):
        registry.promote(
            second.version_id,
            approval=_approval(second.version_id),
            baseline_report=stale_report,
            comparison=harness.compare(stale_report, second_report),
        )

    wrong_challenger = _report(passed=True, version="candidate-other", suite=suite)
    with pytest.raises(ValueError, match="challenger"):
        registry.promote(
            second.version_id,
            approval=_approval(second.version_id),
            baseline_report=first_report,
            comparison=harness.compare(first_report, wrong_challenger),
        )

    with pytest.raises(ValueError, match="self comparison"):
        registry.promote(
            first.version_id,
            approval=_approval(first.version_id),
            baseline_report=first_report,
            comparison=harness.compare(first_report, first_report),
        )

    assert len(registry.promotion_history()) == 1
    assert len(registry.audit_history()) == 1
    promoted = registry.promote(
        second.version_id,
        approval=_approval(second.version_id),
        baseline_report=first_report,
        comparison=harness.compare(first_report, second_report),
    )
    assert promoted.promoted is True
    assert registry.active_version() == second


def test_active_promotion_rejects_reused_self_report(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    report = _report(passed=True, version="candidate-v1", suite=suite)
    first = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=report,
    )
    registry.promote(
        first.version_id,
        approval=_approval(first.version_id),
        bootstrap=True,
    )
    challenger = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("2")},
        eval_report=report,
        parent_version_id=first.version_id,
    )

    with pytest.raises(ValueError, match="self comparison"):
        registry.promote(
            challenger.version_id,
            approval=_approval(challenger.version_id),
            baseline_report=report,
            comparison=EvalHarness().compare(report, report),
        )
    assert registry.active_version() == first
    assert len(registry.audit_history()) == 1


def test_active_promotion_rejects_cross_suite_comparison(tmp_path: Path) -> None:
    first_suite = _suite(suite_id="suite-a")
    second_suite = _suite(suite_id="suite-b")
    registry = _registry(tmp_path / "evolution.sqlite3", first_suite, second_suite)
    first_report = _report(passed=True, version="candidate-v1", suite=first_suite)
    first = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=first_report,
    )
    registry.promote(
        first.version_id,
        approval=_approval(first.version_id),
        bootstrap=True,
    )
    second_report = _report(passed=True, version="candidate-v2", suite=second_suite)
    second = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("2")},
        eval_report=second_report,
        parent_version_id=first.version_id,
    )

    with pytest.raises(ValueError, match="same trusted suite"):
        registry.promote(
            second.version_id,
            approval=_approval(second.version_id),
            baseline_report=first_report,
            comparison=EvalHarness().compare(first_report, second_report),
        )
    assert registry.active_version() == first
    assert len(registry.audit_history()) == 1


def test_promotion_uses_approver_allowlist_and_persists_attestation(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    report = _report(passed=True, version="candidate-v1", suite=suite)
    candidate = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=report,
    )

    with pytest.raises(ValidationError):
        HumanApproval(
            candidate_version_id=candidate.version_id,
            approver_id=_APPROVER,
            approved=True,
        )
    with pytest.raises(ValidationError):
        _approval(candidate.version_id, approver_id="not public/id")

    denied_bool = registry.promote(
        candidate.version_id,
        approval=True,  # type: ignore[arg-type]
        bootstrap=True,
    )
    assert denied_bool.promoted is False
    assert registry.active_version() is None

    untrusted = _approval(candidate.version_id, approver_id="approver.mallory")
    denied_untrusted = registry.promote(
        candidate.version_id,
        approval=untrusted,
        bootstrap=True,
    )
    assert denied_untrusted.promoted is False
    assert "untrusted_approver" in denied_untrusted.reason_codes
    assert registry.audit_history() == ()

    approval = _approval(candidate.version_id)
    assert approval == _approval(candidate.version_id)
    promoted = registry.promote(
        candidate.version_id,
        approval=approval,
        bootstrap=True,
    )
    assert promoted.promoted is True
    assert promoted.approver_id == _APPROVER
    assert promoted.approval_attestation_hash == approval.attestation_hash
    assert registry.promotion_history()[-1] == promoted

    connection = sqlite3.connect(registry.path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT approver_id, approval_attestation_hash
            FROM promotion_history WHERE promoted = 1
            """
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert row["approver_id"] == _APPROVER
    assert row["approval_attestation_hash"] == approval.attestation_hash


def test_regression_denial_and_successful_rollback_are_audited(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    harness = EvalHarness()
    first_report = _report(passed=True, version="candidate-v1", suite=suite)
    first = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=first_report,
    )
    registry.promote(
        first.version_id,
        approval=_approval(first.version_id),
        bootstrap=True,
    )

    failing_report = _report(passed=False, version="candidate-v2", suite=suite)
    failing = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("2")},
        eval_report=failing_report,
        comparison=harness.compare(first_report, failing_report),
        parent_version_id=first.version_id,
    )
    denied = registry.promote(
        failing.version_id,
        approval=_approval(failing.version_id),
        baseline_report=first_report,
        comparison=harness.compare(first_report, failing_report),
    )
    assert denied.promoted is False
    assert "metric_regression" in denied.reason_codes
    assert registry.active_version() == first

    third_report = _report(passed=True, version="candidate-v3", suite=suite)
    third = registry.register_candidate(
        content_hashes={ContentKind.CONFIG: _hash("3")},
        eval_report=third_report,
        parent_version_id=first.version_id,
    )
    registry.promote(
        third.version_id,
        approval=_approval(third.version_id),
        baseline_report=first_report,
        comparison=harness.compare(first_report, third_report),
    )
    rollback_approval = _approval(first.version_id, action="rollback")
    rollback = registry.rollback(first.version_id, approval=rollback_approval)

    assert rollback.promoted is True
    assert rollback.action == "rollback"
    assert rollback.approver_id == _APPROVER
    assert rollback.approval_attestation_hash == rollback_approval.attestation_hash
    assert registry.active_version() == first
    assert registry.get(third.version_id) == third

    audit = registry.audit_history()
    assert [event.sequence for event in audit] == [1, 2, 3]
    assert [event.event_type for event in audit] == ["promote", "promote", "rollback"]
    assert audit[0].previous_event_hash is None
    assert audit[1].previous_event_hash == audit[0].event_hash
    assert audit[2].previous_event_hash == audit[1].event_hash
    assert all(event.suite_id == suite.suite_id for event in audit)
    assert all(event.suite_hash == suite.suite_hash for event in audit)
    assert all(event.seed == suite.seed for event in audit)


def test_rollback_requires_trusted_actor_and_historical_candidate(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    first_report = _report(passed=True, version="candidate-v1", suite=suite)
    first = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=first_report,
    )
    registry.promote(
        first.version_id,
        approval=_approval(first.version_id),
        bootstrap=True,
    )
    second_report = _report(passed=True, version="candidate-v2", suite=suite)
    second = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("2")},
        eval_report=second_report,
        parent_version_id=first.version_id,
    )

    with pytest.raises(ValueError, match="never promoted"):
        registry.rollback(second.version_id, approval=_approval(second.version_id, action="rollback"))
    with pytest.raises(ValueError, match="trusted approver"):
        registry.rollback(
            first.version_id,
            approval=_approval(
                first.version_id,
                approver_id="approver.mallory",
                action="rollback",
            ),
        )
    with pytest.raises(ValueError, match="rollback attestation"):
        registry.rollback(first.version_id, approval=_approval(first.version_id))
    assert registry.active_version() == first
    assert len(registry.audit_history()) == 1


def test_state_history_and_audit_rollback_together_on_audit_failure(tmp_path: Path) -> None:
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)
    candidate = registry.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=_report(passed=True, version="candidate-v1", suite=suite),
    )
    connection = sqlite3.connect(registry.path)
    try:
        connection.execute(
            """
            CREATE TRIGGER force_audit_failure
            BEFORE INSERT ON evolution_audit
            BEGIN
                SELECT RAISE(ABORT, 'forced audit failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(EvolutionIntegrityError):
        registry.promote(
            candidate.version_id,
            approval=_approval(candidate.version_id),
            bootstrap=True,
        )
    assert registry.active_version() is None
    assert registry.promotion_history() == ()
    assert registry.audit_history() == ()


def test_v1_database_migrates_transactionally_and_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "evolution.sqlite3"
    suite = _suite()
    candidate, old_report_hash = _create_v1_fixture(path, suite)

    registry = _registry(path, suite)
    assert registry.get(candidate.version_id) == candidate
    assert registry.active_version() == candidate
    assert registry.promotion_history()[0].report_hash == old_report_hash
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(promotion_history)")
        }
        assert {
            "approver_id",
            "approval_attestation_hash",
            "suite_id",
            "suite_hash",
            "seed",
        }.issubset(columns)
        migrated = connection.execute(
            """
            SELECT approver_id, approval_attestation_hash, suite_id, suite_hash, seed
            FROM promotion_history
            """
        ).fetchone()
        assert migrated is not None
        assert migrated["approver_id"] == "legacy-v1-unverified"
        assert len(migrated["approval_attestation_hash"]) == 64
        assert (migrated["suite_id"], migrated["suite_hash"], migrated["seed"]) == (
            suite.suite_id,
            suite.suite_hash,
            suite.seed,
        )
    finally:
        connection.close()

    audit_before = registry.audit_history()
    assert len(audit_before) == 1
    assert audit_before[0].event_type == "promote"
    assert audit_before[0].candidate_version_id == candidate.version_id
    reopened = _registry(path, suite)
    assert reopened.list_candidates() == (candidate,)
    assert reopened.promotion_history()[0].report_hash == old_report_hash
    assert reopened.audit_history() == audit_before


def test_unknown_future_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version = 99")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(EvolutionIntegrityError):
        _registry(path, _suite())
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall() == []
    finally:
        connection.close()


def test_v1_candidate_and_eval_report_hashes_are_unchanged() -> None:
    report = _report(passed=True, version="candidate-v1")
    candidate = CandidateVersion(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_gate=EvaluationGate.from_report(report),
    )

    assert report.report_hash == "1b98fcce0cecdce1a2ed13c817d2ccf3be583a89292393b5e8101f21fd87165c"
    assert candidate.version_id == "e0280d472442d2984ad98dc88142eec269dea3c48f101fccee5dc2f7315eb6ae"


def test_public_read_apis_work_without_write_trust_configuration(tmp_path: Path) -> None:
    path = tmp_path / "evolution.sqlite3"
    suite = _suite()
    configured = _registry(path, suite)
    report = _report(passed=True, version="candidate-v1", suite=suite)
    candidate = configured.register_candidate(
        content_hashes={ContentKind.PROMPT: _hash("1")},
        eval_report=report,
    )
    configured.promote(
        candidate.version_id,
        approval=_approval(candidate.version_id),
        bootstrap=True,
    )

    read_only_configuration = EvolutionRegistry(path)
    assert read_only_configuration.get_candidate(candidate.version_id) == candidate
    assert read_only_configuration.list_candidates() == (candidate,)
    assert read_only_configuration.active_version() == candidate
    assert len(read_only_configuration.promotion_history()) == 1
    assert len(read_only_configuration.audit_history()) == 1
    with pytest.raises(ValueError, match="trusted suite"):
        read_only_configuration.register_candidate(
            content_hashes={ContentKind.PROMPT: _hash("2")},
            eval_report=_report(passed=True, version="candidate-v2", suite=suite),
        )


def test_evolution_is_offline_and_never_uses_arbitrary_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network or arbitrary execution is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    suite = _suite()
    registry = _registry(tmp_path / "evolution.sqlite3", suite)

    candidate = registry.register_candidate(
        content_hashes={ContentKind.POLICY: _hash("a")},
        eval_report=_report(passed=True, version="candidate-offline", suite=suite),
    )

    assert registry.get(candidate.version_id) == candidate
