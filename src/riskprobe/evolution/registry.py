"""SQLite hash-only evolution registry with trusted suites and atomic audit."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import ValidationError

from riskprobe.evals import EvalComparison, EvalReport, EvalSuite
from riskprobe.evolution.models import (
    AuditEvent,
    CandidateVersion,
    ContentKind,
    EvaluationGate,
    HumanApproval,
    PromotionReport,
)
from riskprobe.privacy import canonical_payload_hash

_SCHEMA_VERSION = 2
_LEGACY_APPROVER_ID = "legacy-v1-unverified"
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_METRIC_NAMES = (
    "task_success",
    "tool_sequence",
    "evidence_completeness",
    "policy_compliance",
    "privacy_compliance",
    "replay_determinism",
)

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS candidate_versions (
        version_id TEXT PRIMARY KEY,
        parent_version_id TEXT REFERENCES candidate_versions(version_id),
        content_hashes_json TEXT NOT NULL,
        eval_gate_json TEXT NOT NULL,
        candidate_json TEXT NOT NULL,
        eval_report_json TEXT,
        comparison_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evolution_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        active_version_id TEXT REFERENCES candidate_versions(version_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS promotion_history (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        candidate_version_id TEXT NOT NULL REFERENCES candidate_versions(version_id),
        previous_active_version_id TEXT,
        active_version_id TEXT,
        promoted INTEGER NOT NULL,
        report_json TEXT NOT NULL,
        approver_id TEXT,
        approval_attestation_hash TEXT,
        suite_id TEXT,
        suite_hash TEXT,
        seed INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evolution_audit (
        sequence INTEGER PRIMARY KEY,
        event_type TEXT NOT NULL CHECK (event_type IN ('promote', 'rollback')),
        candidate_version_id TEXT NOT NULL REFERENCES candidate_versions(version_id),
        previous_active_version_id TEXT REFERENCES candidate_versions(version_id),
        suite_id TEXT NOT NULL,
        suite_hash TEXT NOT NULL,
        seed INTEGER NOT NULL,
        approver_id TEXT NOT NULL,
        approval_attestation_hash TEXT NOT NULL,
        promotion_report_hash TEXT NOT NULL,
        previous_event_hash TEXT,
        event_hash TEXT NOT NULL UNIQUE,
        event_json TEXT NOT NULL
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS candidate_versions_no_update
    BEFORE UPDATE ON candidate_versions
    BEGIN
        SELECT RAISE(ABORT, 'candidate versions are immutable');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS candidate_versions_no_delete
    BEFORE DELETE ON candidate_versions
    BEGIN
        SELECT RAISE(ABORT, 'candidate versions are immutable');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS promotion_history_no_update
    BEFORE UPDATE ON promotion_history
    BEGIN
        SELECT RAISE(ABORT, 'promotion history is append-only');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS promotion_history_no_delete
    BEFORE DELETE ON promotion_history
    BEGIN
        SELECT RAISE(ABORT, 'promotion history is append-only');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS evolution_audit_no_update
    BEFORE UPDATE ON evolution_audit
    BEGIN
        SELECT RAISE(ABORT, 'evolution audit is append-only');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS evolution_audit_no_delete
    BEFORE DELETE ON evolution_audit
    BEGIN
        SELECT RAISE(ABORT, 'evolution audit is append-only');
    END
    """,
)

_REQUIRED_COLUMNS = {
    "candidate_versions": {
        "version_id",
        "parent_version_id",
        "content_hashes_json",
        "eval_gate_json",
        "candidate_json",
        "eval_report_json",
        "comparison_json",
    },
    "evolution_state": {"singleton", "active_version_id"},
    "promotion_history": {
        "sequence",
        "action",
        "candidate_version_id",
        "previous_active_version_id",
        "active_version_id",
        "promoted",
        "report_json",
        "approver_id",
        "approval_attestation_hash",
        "suite_id",
        "suite_hash",
        "seed",
    },
    "evolution_audit": {
        "sequence",
        "event_type",
        "candidate_version_id",
        "previous_active_version_id",
        "suite_id",
        "suite_hash",
        "seed",
        "approver_id",
        "approval_attestation_hash",
        "promotion_report_hash",
        "previous_event_hash",
        "event_hash",
        "event_json",
    },
}


class EvolutionIntegrityError(RuntimeError):
    """Raised when immutable registry data fails strict reconstruction."""


class EvolutionRegistry:
    """Store immutable candidates and activate only trusted, attested evaluations."""

    def __init__(
        self,
        path: Path,
        *,
        trusted_suites: Iterable[EvalSuite] = (),
        allowed_approver_ids: Iterable[str] = (),
    ) -> None:
        self.path = Path(path)
        self._trusted_suites = self._normalize_trusted_suites(trusted_suites)
        self._allowed_approver_ids = self._normalize_approvers(allowed_approver_ids)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_candidate(
        self,
        *,
        content_hashes: Mapping[ContentKind | str, str],
        eval_report: EvalReport,
        comparison: EvalComparison | None = None,
        parent_version_id: str | None = None,
    ) -> CandidateVersion:
        report = self._validate_trusted_report(eval_report)
        if comparison is not None and type(comparison) is not EvalComparison:
            raise TypeError("comparison must be an EvalComparison instance")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            parent: CandidateVersion | None = None
            parent_report: EvalReport | None = None
            if parent_version_id is not None:
                parent_row = self._candidate_row(connection, parent_version_id)
                if parent_row is None:
                    raise KeyError("parent candidate is unavailable")
                parent = self._candidate_from_row(parent_row)
                parent_report = self._report_from_row(parent_row, parent, required=False)
            if comparison is not None:
                if parent is None or parent_report is None:
                    raise ValueError("comparison requires a verified parent baseline report")
                self._validate_comparison_binding(
                    baseline=parent_report,
                    challenger=report,
                    comparison=comparison,
                    baseline_label="parent baseline",
                )

            gate = EvaluationGate.from_report(report, comparison)
            candidate = CandidateVersion(
                content_hashes=content_hashes,
                eval_gate=gate,
                parent_version_id=parent_version_id,
            )
            candidate_json = _canonical_json(candidate.model_dump(mode="json"))
            report_json = _canonical_json(report.model_dump(mode="json"))
            comparison_json = (
                None
                if comparison is None
                else _canonical_json(comparison.model_dump(mode="json"))
            )
            existing = self._candidate_row(connection, candidate.version_id)
            if existing is not None:
                existing_candidate = self._candidate_from_row(existing)
                if (
                    existing_candidate != candidate
                    or existing["eval_report_json"] != report_json
                    or existing["comparison_json"] != comparison_json
                ):
                    raise EvolutionIntegrityError("evolution integrity check failed")
                connection.commit()
                return candidate

            connection.execute(
                """
                INSERT INTO candidate_versions (
                    version_id, parent_version_id, content_hashes_json,
                    eval_gate_json, candidate_json, eval_report_json, comparison_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.version_id,
                    candidate.parent_version_id,
                    _canonical_json(candidate.model_dump(mode="json")["content_hashes"]),
                    _canonical_json(candidate.eval_gate.model_dump(mode="json")),
                    candidate_json,
                    report_json,
                    comparison_json,
                ),
            )
            connection.commit()
            return candidate
        except (EvolutionIntegrityError, KeyError, TypeError, ValueError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    create_candidate = register_candidate

    def get(self, version_id: str) -> CandidateVersion | None:
        connection = self._connect()
        try:
            row = self._candidate_row(connection, version_id)
            return None if row is None else self._candidate_from_row(row)
        except sqlite3.Error as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    get_candidate = get

    def list_candidates(self) -> tuple[CandidateVersion, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM candidate_versions ORDER BY rowid"
            ).fetchall()
            return tuple(self._candidate_from_row(row) for row in rows)
        except sqlite3.Error as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    def active_version(self) -> CandidateVersion | None:
        connection = self._connect()
        try:
            active_id = self._active_id(connection)
            if active_id is None:
                return None
            row = self._candidate_row(connection, active_id)
            if row is None:
                raise EvolutionIntegrityError("evolution integrity check failed")
            return self._candidate_from_row(row)
        except sqlite3.Error as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    def promote(
        self,
        version_id: str,
        *,
        approval: HumanApproval | None = None,
        bootstrap: bool = False,
        baseline_report: EvalReport | None = None,
        comparison: EvalComparison | None = None,
    ) -> PromotionReport:
        if type(bootstrap) is not bool:
            raise TypeError("bootstrap must be a bool")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            candidate_row = self._candidate_row(connection, version_id)
            if candidate_row is None:
                raise KeyError("candidate version is unavailable")
            candidate = self._candidate_from_row(candidate_row)
            challenger_report = self._report_from_row(candidate_row, candidate, required=True)
            assert challenger_report is not None
            self._assert_trusted_binding(
                challenger_report.suite_id,
                challenger_report.suite_hash,
                challenger_report.seed,
            )
            previous_id = self._active_id(connection)
            active_comparison: EvalComparison | None = None

            if previous_id is None:
                if not bootstrap:
                    raise ValueError("initial activation requires explicit bootstrap=True")
                if baseline_report is not None or comparison is not None:
                    raise ValueError("bootstrap promotion cannot include a baseline or comparison")
                if candidate.parent_version_id is not None:
                    raise ValueError("bootstrap candidate cannot declare a baseline parent")
            else:
                if bootstrap:
                    raise ValueError("bootstrap must be false when an active baseline exists")
                if baseline_report is None or comparison is None:
                    raise ValueError("active promotion requires baseline report and comparison")
                if version_id == previous_id:
                    raise ValueError("self comparison cannot promote the active candidate")
                if candidate.parent_version_id != previous_id:
                    raise ValueError("challenger parent does not match current active candidate")

                active_row = self._candidate_row(connection, previous_id)
                if active_row is None:
                    raise EvolutionIntegrityError("evolution integrity check failed")
                active_candidate = self._candidate_from_row(active_row)
                stored_baseline = self._report_from_row(
                    active_row,
                    active_candidate,
                    required=True,
                )
                assert stored_baseline is not None
                supplied_baseline = self._validate_trusted_report(baseline_report)
                if (
                    supplied_baseline.report_hash != stored_baseline.report_hash
                    or supplied_baseline.candidate_version != stored_baseline.candidate_version
                ):
                    raise ValueError("baseline report does not match current active candidate")
                self._validate_comparison_binding(
                    baseline=stored_baseline,
                    challenger=challenger_report,
                    comparison=comparison,
                    baseline_label="current active",
                )
                active_comparison = comparison

            approved, approval_reason, trusted_approval = self._promotion_approval(
                approval,
                version_id,
            )
            regressions = set(candidate.eval_gate.regressed_metrics)
            eval_passed = candidate.eval_gate.passed and challenger_report.passed
            if active_comparison is not None:
                regressions.update(active_comparison.regressed_metrics)
                eval_passed = (
                    eval_passed
                    and active_comparison.compatible
                    and active_comparison.candidate_passed
                    and not active_comparison.regressed_metrics
                )

            reasons: list[str] = []
            if not eval_passed:
                reasons.append("eval_gate_failed")
            if regressions:
                reasons.append("metric_regression")
            if not approved:
                reasons.append(approval_reason)
            promoted = not reasons
            active_id = previous_id
            if promoted:
                self._set_active(connection, version_id)
                active_id = version_id

            report = PromotionReport(
                action="promote",
                candidate_version_id=version_id,
                previous_active_version_id=previous_id,
                active_version_id=active_id,
                promoted=promoted,
                eval_passed=eval_passed,
                human_approved=approved,
                reason_codes=tuple(reasons),
                suite_id=candidate.eval_gate.suite_id,
                suite_hash=candidate.eval_gate.suite_hash,
                seed=candidate.eval_gate.seed,
                approver_id=(
                    None if trusted_approval is None else trusted_approval.approver_id
                ),
                approval_attestation_hash=(
                    None if trusted_approval is None else trusted_approval.attestation_hash
                ),
            )
            self._append_history(connection, report)
            if promoted:
                assert trusted_approval is not None
                self._append_audit(
                    connection,
                    event_type="promote",
                    candidate=candidate,
                    previous_active_version_id=previous_id,
                    approver_id=trusted_approval.approver_id,
                    approval_attestation_hash=trusted_approval.attestation_hash,
                    promotion_report_hash=report.report_hash,
                )
            connection.commit()
            return report
        except (EvolutionIntegrityError, KeyError, TypeError, ValueError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    def rollback(
        self,
        version_id: str,
        *,
        approval: HumanApproval,
    ) -> PromotionReport:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            candidate_row = self._candidate_row(connection, version_id)
            if candidate_row is None:
                raise KeyError("candidate version is unavailable")
            candidate = self._candidate_from_row(candidate_row)
            self._assert_trusted_binding(
                candidate.eval_gate.suite_id,
                candidate.eval_gate.suite_hash,
                candidate.eval_gate.seed,
            )
            stored_report = self._report_from_row(candidate_row, candidate, required=False)
            if stored_report is not None:
                self._assert_trusted_binding(
                    stored_report.suite_id,
                    stored_report.suite_hash,
                    stored_report.seed,
                )

            prior_promotion = connection.execute(
                """
                SELECT 1 FROM evolution_audit
                WHERE candidate_version_id = ? AND event_type = 'promote'
                LIMIT 1
                """,
                (version_id,),
            ).fetchone()
            if prior_promotion is None:
                raise ValueError("rollback target was never promoted")
            trusted_approval = self._rollback_approval(approval, version_id)
            previous_id = self._active_id(connection)
            if previous_id == version_id:
                raise ValueError("rollback target is already active")

            self._set_active(connection, version_id)
            report = PromotionReport(
                action="rollback",
                candidate_version_id=version_id,
                previous_active_version_id=previous_id,
                active_version_id=version_id,
                promoted=True,
                eval_passed=candidate.eval_gate.passed,
                human_approved=True,
                suite_id=candidate.eval_gate.suite_id,
                suite_hash=candidate.eval_gate.suite_hash,
                seed=candidate.eval_gate.seed,
                approver_id=trusted_approval.approver_id,
                approval_attestation_hash=trusted_approval.attestation_hash,
            )
            self._append_history(connection, report)
            self._append_audit(
                connection,
                event_type="rollback",
                candidate=candidate,
                previous_active_version_id=previous_id,
                approver_id=trusted_approval.approver_id,
                approval_attestation_hash=trusted_approval.attestation_hash,
                promotion_report_hash=report.report_hash,
            )
            connection.commit()
            return report
        except (EvolutionIntegrityError, KeyError, TypeError, ValueError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    def promotion_history(self) -> tuple[PromotionReport, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM promotion_history ORDER BY sequence"
            ).fetchall()
            return tuple(self._promotion_from_row(connection, row) for row in rows)
        except sqlite3.Error as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    def audit_history(self) -> tuple[AuditEvent, ...]:
        connection = self._connect()
        try:
            return self._audit_history_in_connection(connection)
        except sqlite3.Error as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()

    @staticmethod
    def _normalize_trusted_suites(
        suites: Iterable[EvalSuite],
    ) -> dict[str, tuple[str, int]]:
        normalized: dict[str, tuple[str, int]] = {}
        for suite in suites:
            if type(suite) is not EvalSuite:
                raise TypeError("trusted suites must be frozen EvalSuite v1 instances")
            if not suite.verify_integrity():
                raise ValueError("trusted suite integrity check failed")
            binding = (suite.suite_hash, suite.seed)
            existing = normalized.get(suite.suite_id)
            if existing is not None and existing != binding:
                raise ValueError("conflicting trusted suite binding for suite ID")
            normalized[suite.suite_id] = binding
        return normalized

    @staticmethod
    def _normalize_approvers(approver_ids: Iterable[str]) -> frozenset[str]:
        normalized: set[str] = set()
        for approver_id in approver_ids:
            if type(approver_id) is not str or _PUBLIC_ID.fullmatch(approver_id) is None:
                raise ValueError("allowed approver IDs must be public identifiers")
            normalized.add(approver_id)
        return frozenset(normalized)

    def _assert_trusted_binding(self, suite_id: str, suite_hash: str, seed: int) -> None:
        if self._trusted_suites.get(suite_id) != (suite_hash, seed):
            raise ValueError("evaluation does not match a trusted suite binding")

    def _validate_trusted_report(self, report: EvalReport) -> EvalReport:
        if type(report) is not EvalReport:
            raise TypeError("eval report must be an EvalReport v1 instance")
        if not report.verify_integrity():
            raise ValueError("eval report integrity check failed")
        self._assert_trusted_binding(report.suite_id, report.suite_hash, report.seed)
        return report

    def _validate_comparison_binding(
        self,
        *,
        baseline: EvalReport,
        challenger: EvalReport,
        comparison: EvalComparison,
        baseline_label: str,
    ) -> None:
        if type(comparison) is not EvalComparison:
            raise TypeError("comparison must be an EvalComparison instance")
        if (
            baseline.report_hash == challenger.report_hash
            or baseline.candidate_version == challenger.candidate_version
        ):
            raise ValueError("self comparison cannot establish a challenger baseline")
        baseline_binding = (baseline.suite_id, baseline.suite_hash, baseline.seed)
        challenger_binding = (challenger.suite_id, challenger.suite_hash, challenger.seed)
        if baseline_binding != challenger_binding:
            raise ValueError("baseline and challenger must use the same trusted suite")
        self._assert_trusted_binding(*baseline_binding)
        if (
            comparison.baseline_report_hash != baseline.report_hash
            or comparison.baseline_version != baseline.candidate_version
        ):
            raise ValueError(f"comparison does not bind {baseline_label} report")
        if (
            comparison.candidate_report_hash != challenger.report_hash
            or comparison.candidate_version != challenger.candidate_version
        ):
            raise ValueError("comparison does not bind challenger report")

        expected_deltas = {
            name: getattr(challenger.metrics, name) - getattr(baseline.metrics, name)
            for name in _METRIC_NAMES
        }
        if any(
            abs(comparison.deltas[name] - expected_deltas[name]) > 1e-12
            for name in _METRIC_NAMES
        ):
            raise ValueError("comparison metric deltas do not match bound reports")
        expected_regressions = tuple(
            name for name, delta in expected_deltas.items() if delta < -1e-12
        )
        expected_passed = challenger.passed and not expected_regressions
        if (
            not comparison.compatible
            or comparison.regressed_metrics != expected_regressions
            or comparison.candidate_passed != expected_passed
        ):
            raise ValueError("comparison gates do not match bound reports")

    def _promotion_approval(
        self,
        approval: HumanApproval | object | None,
        version_id: str,
    ) -> tuple[bool, str, HumanApproval | None]:
        if type(approval) is not HumanApproval:
            return False, "human_approval_required", None
        try:
            validated = HumanApproval.model_validate_json(approval.model_dump_json())
        except (ValidationError, ValueError):
            return False, "human_approval_required", None
        if (
            validated.action != "promote"
            or validated.candidate_version_id != version_id
            or validated.approved is not True
        ):
            return False, "human_approval_required", None
        if validated.approver_id not in self._allowed_approver_ids:
            return False, "untrusted_approver", None
        return True, "", validated

    def _rollback_approval(
        self,
        approval: HumanApproval | object,
        version_id: str,
    ) -> HumanApproval:
        if type(approval) is not HumanApproval:
            raise ValueError("rollback attestation is required")
        try:
            validated = HumanApproval.model_validate_json(approval.model_dump_json())
        except (ValidationError, ValueError) as error:
            raise ValueError("rollback attestation is invalid") from error
        if (
            validated.action != "rollback"
            or validated.candidate_version_id != version_id
            or validated.approved is not True
        ):
            raise ValueError("rollback attestation does not bind the target")
        if validated.approver_id not in self._allowed_approver_ids:
            raise ValueError("rollback requires a trusted approver")
        return validated

    @staticmethod
    def _candidate_from_json(value: str) -> CandidateVersion:
        try:
            return CandidateVersion.model_validate_json(value)
        except (ValidationError, ValueError) as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error

    def _candidate_from_row(self, row: sqlite3.Row) -> CandidateVersion:
        candidate = self._candidate_from_json(str(row["candidate_json"]))
        try:
            content_hashes = _canonical_json(json.loads(str(row["content_hashes_json"])))
            eval_gate = _canonical_json(json.loads(str(row["eval_gate_json"])))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        if (
            str(row["version_id"]) != candidate.version_id
            or row["parent_version_id"] != candidate.parent_version_id
            or content_hashes
            != _canonical_json(candidate.model_dump(mode="json")["content_hashes"])
            or eval_gate != _canonical_json(candidate.eval_gate.model_dump(mode="json"))
        ):
            raise EvolutionIntegrityError("evolution integrity check failed")
        return candidate

    def _report_from_row(
        self,
        row: sqlite3.Row,
        candidate: CandidateVersion,
        *,
        required: bool,
    ) -> EvalReport | None:
        value = row["eval_report_json"]
        if value is None:
            if required:
                raise EvolutionIntegrityError(
                    "candidate does not contain a verified evaluation report"
                )
            return None
        try:
            report = EvalReport.model_validate_json(str(value))
        except (ValidationError, ValueError) as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        if (
            not report.verify_integrity()
            or report.suite_id != candidate.eval_gate.suite_id
            or report.suite_hash != candidate.eval_gate.suite_hash
            or report.seed != candidate.eval_gate.seed
            or report.report_hash != candidate.eval_gate.report_hash
        ):
            raise EvolutionIntegrityError("evolution integrity check failed")
        return report

    @staticmethod
    def _candidate_row(
        connection: sqlite3.Connection,
        version_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM candidate_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()

    @staticmethod
    def _active_id(connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            "SELECT active_version_id FROM evolution_state WHERE singleton = 1"
        ).fetchone()
        return None if row is None or row["active_version_id"] is None else str(row[0])

    @staticmethod
    def _set_active(connection: sqlite3.Connection, version_id: str) -> None:
        connection.execute(
            """
            INSERT INTO evolution_state (singleton, active_version_id) VALUES (1, ?)
            ON CONFLICT(singleton) DO UPDATE SET active_version_id = excluded.active_version_id
            """,
            (version_id,),
        )

    @staticmethod
    def _append_history(connection: sqlite3.Connection, report: PromotionReport) -> None:
        connection.execute(
            """
            INSERT INTO promotion_history (
                action, candidate_version_id, previous_active_version_id,
                active_version_id, promoted, report_json, approver_id,
                approval_attestation_hash, suite_id, suite_hash, seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.action,
                report.candidate_version_id,
                report.previous_active_version_id,
                report.active_version_id,
                int(report.promoted),
                _canonical_json(report.model_dump(mode="json")),
                report.approver_id,
                report.approval_attestation_hash,
                report.suite_id,
                report.suite_hash,
                report.seed,
            ),
        )

    @staticmethod
    def _append_audit(
        connection: sqlite3.Connection,
        *,
        event_type: str,
        candidate: CandidateVersion,
        previous_active_version_id: str | None,
        approver_id: str,
        approval_attestation_hash: str,
        promotion_report_hash: str,
    ) -> AuditEvent:
        previous = connection.execute(
            "SELECT sequence, event_hash FROM evolution_audit ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if previous is None else int(previous["sequence"]) + 1
        previous_event_hash = None if previous is None else str(previous["event_hash"])
        event = AuditEvent(
            sequence=sequence,
            event_type=event_type,
            candidate_version_id=candidate.version_id,
            previous_active_version_id=previous_active_version_id,
            suite_id=candidate.eval_gate.suite_id,
            suite_hash=candidate.eval_gate.suite_hash,
            seed=candidate.eval_gate.seed,
            approver_id=approver_id,
            approval_attestation_hash=approval_attestation_hash,
            promotion_report_hash=promotion_report_hash,
            previous_event_hash=previous_event_hash,
        )
        connection.execute(
            """
            INSERT INTO evolution_audit (
                sequence, event_type, candidate_version_id,
                previous_active_version_id, suite_id, suite_hash, seed,
                approver_id, approval_attestation_hash, promotion_report_hash,
                previous_event_hash, event_hash, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.sequence,
                event.event_type,
                event.candidate_version_id,
                event.previous_active_version_id,
                event.suite_id,
                event.suite_hash,
                event.seed,
                event.approver_id,
                event.approval_attestation_hash,
                event.promotion_report_hash,
                event.previous_event_hash,
                event.event_hash,
                _canonical_json(event.model_dump(mode="json")),
            ),
        )
        return event

    def _promotion_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> PromotionReport:
        try:
            report = PromotionReport.model_validate_json(str(row["report_json"]))
        except (ValidationError, ValueError) as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        if (
            str(row["action"]) != report.action
            or str(row["candidate_version_id"]) != report.candidate_version_id
            or row["previous_active_version_id"] != report.previous_active_version_id
            or row["active_version_id"] != report.active_version_id
            or bool(row["promoted"]) is not report.promoted
        ):
            raise EvolutionIntegrityError("evolution integrity check failed")
        candidate_row = self._candidate_row(connection, report.candidate_version_id)
        if candidate_row is None:
            raise EvolutionIntegrityError("evolution integrity check failed")
        candidate = self._candidate_from_row(candidate_row)
        stored_suite = (str(row["suite_id"]), str(row["suite_hash"]), int(row["seed"]))
        candidate_suite = (
            candidate.eval_gate.suite_id,
            candidate.eval_gate.suite_hash,
            candidate.eval_gate.seed,
        )
        if stored_suite != candidate_suite:
            raise EvolutionIntegrityError("evolution integrity check failed")
        if report.suite_id is None:
            if (
                row["approver_id"] != _LEGACY_APPROVER_ID
                or row["approval_attestation_hash"] is None
            ):
                raise EvolutionIntegrityError("evolution integrity check failed")
        elif (
            (report.suite_id, report.suite_hash, report.seed) != stored_suite
            or report.approver_id != row["approver_id"]
            or report.approval_attestation_hash != row["approval_attestation_hash"]
        ):
            raise EvolutionIntegrityError("evolution integrity check failed")
        return report

    @staticmethod
    def _audit_event_from_row(row: sqlite3.Row) -> AuditEvent:
        try:
            event = AuditEvent.model_validate_json(str(row["event_json"]))
        except (ValidationError, ValueError) as error:
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        fields = (
            "sequence",
            "event_type",
            "candidate_version_id",
            "previous_active_version_id",
            "suite_id",
            "suite_hash",
            "seed",
            "approver_id",
            "approval_attestation_hash",
            "promotion_report_hash",
            "previous_event_hash",
            "event_hash",
        )
        if any(row[field] != getattr(event, field) for field in fields):
            raise EvolutionIntegrityError("evolution integrity check failed")
        return event

    def _audit_history_in_connection(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[AuditEvent, ...]:
        rows = connection.execute("SELECT * FROM evolution_audit ORDER BY sequence").fetchall()
        events: list[AuditEvent] = []
        previous_hash: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            event = self._audit_event_from_row(row)
            if (
                event.sequence != expected_sequence
                or event.previous_event_hash != previous_hash
            ):
                raise EvolutionIntegrityError("evolution audit hash chain is invalid")
            events.append(event)
            previous_hash = event.event_hash
        return tuple(events)

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN EXCLUSIVE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1, _SCHEMA_VERSION}:
                raise EvolutionIntegrityError("unsupported evolution schema version")
            if version == 0:
                existing = connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name IN (
                        'candidate_versions', 'evolution_state',
                        'promotion_history', 'evolution_audit'
                    )
                    """
                ).fetchall()
                if existing:
                    raise EvolutionIntegrityError("unversioned evolution schema is unsafe")
                self._create_v2_schema(connection)
            elif version == 1:
                self._migrate_v1(connection)
            else:
                self._create_v2_schema(connection)
            self._validate_v2_schema(connection)
            self._validate_database(connection)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except EvolutionIntegrityError:
            connection.rollback()
            raise
        except (ValidationError, ValueError, json.JSONDecodeError, sqlite3.Error) as error:
            connection.rollback()
            raise EvolutionIntegrityError("evolution integrity check failed") from error
        finally:
            connection.close()
        try:
            os.chmod(self.path, 0o600)
        except OSError as error:
            raise EvolutionIntegrityError("unable to secure evolution database") from error

    @staticmethod
    def _create_v2_schema(connection: sqlite3.Connection) -> None:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        v1_required = {
            "candidate_versions": {
                "version_id",
                "parent_version_id",
                "content_hashes_json",
                "eval_gate_json",
                "candidate_json",
            },
            "evolution_state": {"singleton", "active_version_id"},
            "promotion_history": {
                "sequence",
                "action",
                "candidate_version_id",
                "previous_active_version_id",
                "active_version_id",
                "promoted",
                "report_json",
            },
        }
        for table, required in v1_required.items():
            columns = self._table_columns(connection, table)
            if not required.issubset(columns):
                raise EvolutionIntegrityError("invalid v1 evolution schema")

        connection.execute("DROP TRIGGER IF EXISTS promotion_history_no_update")
        connection.execute("DROP TRIGGER IF EXISTS promotion_history_no_delete")
        self._add_column(connection, "candidate_versions", "eval_report_json TEXT")
        self._add_column(connection, "candidate_versions", "comparison_json TEXT")
        self._add_column(connection, "promotion_history", "approver_id TEXT")
        self._add_column(
            connection,
            "promotion_history",
            "approval_attestation_hash TEXT",
        )
        self._add_column(connection, "promotion_history", "suite_id TEXT")
        self._add_column(connection, "promotion_history", "suite_hash TEXT")
        self._add_column(connection, "promotion_history", "seed INTEGER")
        for statement in _SCHEMA_STATEMENTS:
            if "promotion_history_no_" not in statement:
                connection.execute(statement)

        if connection.execute("SELECT 1 FROM evolution_audit LIMIT 1").fetchone() is not None:
            raise EvolutionIntegrityError("v1 database contains unexpected audit data")
        rows = connection.execute(
            "SELECT * FROM promotion_history ORDER BY sequence"
        ).fetchall()
        for row in rows:
            candidate_row = self._candidate_row(connection, str(row["candidate_version_id"]))
            if candidate_row is None:
                raise EvolutionIntegrityError("evolution integrity check failed")
            candidate = self._candidate_from_row(candidate_row)
            try:
                report = PromotionReport.model_validate_json(str(row["report_json"]))
            except (ValidationError, ValueError) as error:
                raise EvolutionIntegrityError("evolution integrity check failed") from error
            if (
                report.suite_id is not None
                or report.action != row["action"]
                or report.candidate_version_id != row["candidate_version_id"]
                or report.previous_active_version_id != row["previous_active_version_id"]
                or report.active_version_id != row["active_version_id"]
                or report.promoted != bool(row["promoted"])
                or (report.promoted and not report.human_approved)
            ):
                raise EvolutionIntegrityError("evolution integrity check failed")
            attestation_hash = canonical_payload_hash(
                {
                    "candidate_version_id": candidate.version_id,
                    "legacy_report_hash": report.report_hash,
                    "source_schema_version": 1,
                }
            )
            connection.execute(
                """
                UPDATE promotion_history
                SET approver_id = ?, approval_attestation_hash = ?,
                    suite_id = ?, suite_hash = ?, seed = ?
                WHERE sequence = ?
                """,
                (
                    _LEGACY_APPROVER_ID,
                    attestation_hash,
                    candidate.eval_gate.suite_id,
                    candidate.eval_gate.suite_hash,
                    candidate.eval_gate.seed,
                    row["sequence"],
                ),
            )
            if report.promoted:
                self._append_audit(
                    connection,
                    event_type=report.action,
                    candidate=candidate,
                    previous_active_version_id=report.previous_active_version_id,
                    approver_id=_LEGACY_APPROVER_ID,
                    approval_attestation_hash=attestation_hash,
                    promotion_report_hash=report.report_hash,
                )

        for statement in _SCHEMA_STATEMENTS:
            if "CREATE TRIGGER" in statement:
                connection.execute(statement)

    @staticmethod
    def _add_column(connection: sqlite3.Connection, table: str, definition: str) -> None:
        column = definition.split()[0]
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _validate_v2_schema(self, connection: sqlite3.Connection) -> None:
        for table, required in _REQUIRED_COLUMNS.items():
            if not required.issubset(self._table_columns(connection, table)):
                raise EvolutionIntegrityError("invalid evolution v2 schema")

    def _validate_database(self, connection: sqlite3.Connection) -> None:
        candidate_rows = connection.execute("SELECT * FROM candidate_versions").fetchall()
        for row in candidate_rows:
            candidate = self._candidate_from_row(row)
            self._report_from_row(row, candidate, required=False)
        active_id = self._active_id(connection)
        if active_id is not None and self._candidate_row(connection, active_id) is None:
            raise EvolutionIntegrityError("evolution integrity check failed")
        history_rows = connection.execute(
            "SELECT * FROM promotion_history ORDER BY sequence"
        ).fetchall()
        for row in history_rows:
            self._promotion_from_row(connection, row)
        self._audit_history_in_connection(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = ["EvolutionIntegrityError", "EvolutionRegistry"]
