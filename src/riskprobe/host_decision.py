"""Host-driven two-phase decision coordination without model callbacks."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Condition, Thread, local
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from riskprobe.analysis_contracts import (
    AnalysisSummary,
    DecisionSummary,
    DiagnosticsSummary,
    RecommendationSummary,
    StageName,
    StageStatus,
    StageSummary,
)
from riskprobe.agents.contracts import AgentResult, AgentStatus, ReviewReason
from riskprobe.agents.results import AgentResultStore
from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionFinding,
    DecisionProposal,
    DecisionReason,
    DecisionResult,
    DecisionSource,
    DecisionStatus,
)
from riskprobe.agents.decision_providers import (
    DecisionDisposition,
    DecisionProviderConfig,
    DecisionProviderMode,
    DecisionProviderResolution,
)
from riskprobe.recommendations.policy import ActionCode
from riskprobe.terminal_reports import TerminalReportSubject

if TYPE_CHECKING:
    from riskprobe.agents.decision_controller import DecisionSubmission
    from riskprobe.evidence import EvidenceRecord, EvidenceStore

_PUBLIC_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_WAIT_SECONDS = 300.0
_PROTOCOL_VERSION = "riskprobe.host-decision.v1"
_SESSION_FORMAT = "riskprobe.host-sessions.v1"
_SESSION_FILE = ".riskprobe-host-sessions.json"
_SESSION_LOCK = ".riskprobe-host-sessions.lock"
_MAX_SESSION_BYTES = 10 * 1024 * 1024
_LEGACY_NORMAL_OUTCOME_FIELDS = frozenset(
    {
        "protocol_version",
        "phase",
        "context_id",
        "agent_result",
        "decision_status",
        "reason_codes",
        "action_codes",
        "context_evidence_id",
        "proposal_evidence_id",
        "result_evidence_id",
        "expires_at",
    }
)
_NORMAL_OUTCOME_FIELDS = _LEGACY_NORMAL_OUTCOME_FIELDS | {
    "analysis_summary",
    "decision_summary",
}
_LEGACY_NO_ACTION_OUTCOME_FIELDS = frozenset(
    {
        "protocol_version",
        "phase",
        "terminal_reason",
        "action_codes",
        "agent_result",
    }
)
_NO_ACTION_OUTCOME_FIELDS = _LEGACY_NO_ACTION_OUTCOME_FIELDS | {
    "analysis_summary",
    "decision_summary",
}
_LEGACY_REVIEW_FIELDS = frozenset(
    {"approved", "reason_codes", "evidence_ids", "retry_allowed"}
)
_CURRENT_REVIEW_FIELDS = _LEGACY_REVIEW_FIELDS | {"no_action_required"}
_CURRENT_AGENT_RESULT_FIELDS = frozenset(AgentResult.model_fields)
_HOST_FAILURE_CODES = frozenset(
    {
        "profile_contract_failed",
        "partition_failed",
        "discover_failed",
        "scorecard_failed",
        "scorecard_terms_failed",
        "scorecard_features_failed",
        "scorecard_metrics_failed",
        "scorecard_refs_failed",
        "scorecard_metadata_failed",
        "scorecard_refs_contract_failed",
        "scorecard_content_failed",
        "scorecard_splits_contract_failed",
        "scorecard_contract_failed",
        "artifact_runtime_failed",
        "analysis_summary_failed",
        "artifact_finalize_failed",
        "agent_session_failed",
        "agent_state_unavailable",
        "agent_state_incomplete",
        "agent_orchestration_failed",
        "context_timeout",
        "session_state_unavailable",
    }
)


class HostDecisionError(RuntimeError):
    """Raised with a fixed message when a host decision cannot safely continue."""


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class HostDecisionFailure(_StrictDTO):
    phase: Literal["context"] = "context"
    error_code: Literal[
        "profile_contract_failed",
        "partition_failed",
        "discover_failed",
        "scorecard_failed",
        "scorecard_terms_failed",
        "scorecard_features_failed",
        "scorecard_metrics_failed",
        "scorecard_refs_failed",
        "scorecard_metadata_failed",
        "scorecard_refs_contract_failed",
        "scorecard_content_failed",
        "scorecard_splits_contract_failed",
        "scorecard_contract_failed",
        "artifact_runtime_failed",
        "analysis_summary_failed",
        "artifact_finalize_failed",
        "agent_session_failed",
        "agent_state_unavailable",
        "agent_state_incomplete",
        "agent_orchestration_failed",
        "context_timeout",
        "session_state_unavailable",
    ]


class HostDecisionContext(_StrictDTO):
    protocol_version: Literal["riskprobe.host-decision.v1"] = _PROTOCOL_VERSION
    phase: Literal["awaiting_proposal"] = "awaiting_proposal"
    provider_id: str
    provider_version: str
    context: DecisionContext

    @field_validator("provider_id", "provider_version")
    @classmethod
    def validate_provider_identity(cls, value: str) -> str:
        if _PUBLIC_TOKEN.fullmatch(value) is None:
            raise ValueError("provider identity must use public tokens")
        return value


def _canonical_outcome_payload(
    payload: object,
    *,
    outer_fields: frozenset[str],
    error_message: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(payload) is not dict or frozenset(payload) != outer_fields:
        raise ValueError(error_message)
    agent_result = payload.get("agent_result")
    if (
        type(agent_result) is not dict
        or frozenset(agent_result) != _CURRENT_AGENT_RESULT_FIELDS
        or type(agent_result.get("review")) is not dict
        or frozenset(agent_result["review"]) != _CURRENT_REVIEW_FIELDS
    ):
        raise ValueError(error_message)
    return payload, agent_result


def _canonical_summaries(
    payload: dict[str, object],
) -> tuple[AnalysisSummary, DecisionSummary]:
    raw_analysis = payload.get("analysis_summary")
    raw_decision = payload.get("decision_summary")
    if type(raw_analysis) is not dict or type(raw_decision) is not dict:
        raise ValueError("invalid current summary payload")
    analysis = AnalysisSummary.model_validate_json(
        json.dumps(
            raw_analysis,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    decision = DecisionSummary.model_validate_json(
        json.dumps(
            raw_decision,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    if (
        analysis.model_dump(mode="json") != raw_analysis
        or decision.model_dump(mode="json") != raw_decision
    ):
        raise ValueError("non-canonical current summary payload")
    return analysis, decision


class _HostDecisionOutcomeBase(_StrictDTO):
    protocol_version: Literal["riskprobe.host-decision.v1"] = _PROTOCOL_VERSION
    phase: Literal["terminal"] = "terminal"
    context_id: str
    agent_result: AgentResult
    decision_status: DecisionStatus
    reason_codes: tuple[DecisionReason, ...] = ()
    action_codes: tuple[ActionCode, ...] = ()
    context_evidence_id: str | None = None
    proposal_evidence_id: str | None = None
    result_evidence_id: str | None = None
    expires_at: datetime

    @field_validator(
        "context_evidence_id",
        "proposal_evidence_id",
        "result_evidence_id",
    )
    @classmethod
    def validate_optional_evidence_id(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("evidence ID must be a SHA-256 identifier")
        return value

    @field_validator("expires_at")
    @classmethod
    def normalize_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiry must be timezone-aware")
        return value.astimezone(UTC)


class HostDecisionOutcome(_HostDecisionOutcomeBase):
    analysis_summary: AnalysisSummary
    decision_summary: DecisionSummary

    @model_validator(mode="after")
    def validate_agent_analysis_summary(self) -> HostDecisionOutcome:
        if self.agent_result.analysis_summary is None:
            raise ValueError("current outcome requires agent analysis summary")
        if self.agent_result.review.no_action_required:
            raise ValueError("current outcome requires a normal agent result")
        return self

    @model_serializer(mode="wrap")
    def serialize_current(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload, agent_result = _canonical_outcome_payload(
            handler(self),
            outer_fields=_NORMAL_OUTCOME_FIELDS,
            error_message="current outcome requires canonical payload",
        )
        if (
            type(payload["analysis_summary"]) is not dict
            or type(payload["decision_summary"]) is not dict
            or type(agent_result["analysis_summary"]) is not dict
        ):
            raise ValueError("current outcome requires canonical payload")
        return payload


class _LegacyHostDecisionOutcome(_HostDecisionOutcomeBase):
    @model_validator(mode="after")
    def validate_legacy_result(self) -> _LegacyHostDecisionOutcome:
        if (
            self.agent_result.analysis_summary is not None
            or not _normal_terminal_state_matches(
                self.agent_result,
                decision_status=self.decision_status,
                reason_codes=self.reason_codes,
                action_codes=self.action_codes,
            )
        ):
            raise ValueError("legacy outcome requires a normal agent result")
        return self

    @model_serializer(mode="wrap")
    def serialize_legacy(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload, agent_result = _canonical_outcome_payload(
            handler(self),
            outer_fields=_LEGACY_NORMAL_OUTCOME_FIELDS,
            error_message="legacy outcome requires canonical payload",
        )
        agent_result.pop("analysis_summary")
        return payload


class _LegacyHostDecisionOutcomeV0(_LegacyHostDecisionOutcome):
    @model_serializer(mode="wrap")
    def serialize_legacy_v0(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload, agent_result = _canonical_outcome_payload(
            handler(self),
            outer_fields=_LEGACY_NORMAL_OUTCOME_FIELDS,
            error_message="legacy outcome requires canonical payload",
        )
        agent_result.pop("analysis_summary")
        agent_result["review"].pop("no_action_required")
        return payload


def _is_clean_no_action_result(
    result: AgentResult,
    action_codes: tuple[ActionCode, ...],
) -> bool:
    return (
        result.status is AgentStatus.SUCCEEDED
        and result.review.approved
        and result.review.no_action_required
        and not result.review.reason_codes
        and not result.review.evidence_ids
        and not result.review.retry_allowed
        and not result.evidence_ids
        and not result.diagnosis_evidence_ids
        and not action_codes
        and result.tool_sequence
        == ("inspect", "diagnose", "discover", "recommend", "review")
    )


class HostDecisionNoActionOutcome(_StrictDTO):
    protocol_version: Literal["riskprobe.host-decision.v1"] = _PROTOCOL_VERSION
    phase: Literal["terminal"] = "terminal"
    terminal_reason: Literal["no_actionable_diagnosis"] = "no_actionable_diagnosis"
    action_codes: tuple[ActionCode, ...] = ()
    agent_result: AgentResult
    analysis_summary: AnalysisSummary
    decision_summary: DecisionSummary

    @model_validator(mode="after")
    def validate_clean_no_action_result(self) -> HostDecisionNoActionOutcome:
        result = self.agent_result
        expected_analysis = _terminal_analysis_summary(
            result.analysis_summary,
            findings=(),
            result=result,
            decision_context_succeeded=False,
        )
        if (
            not _is_clean_no_action_result(result, self.action_codes)
            or result.analysis_summary is None
            or expected_analysis is None
            or self.analysis_summary != expected_analysis
            or self.decision_summary != _no_action_decision_summary(result)
        ):
            raise ValueError("no-action outcome requires a clean terminal result")
        return self

    @model_serializer(mode="wrap")
    def serialize_current(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload, agent_result = _canonical_outcome_payload(
            handler(self),
            outer_fields=_NO_ACTION_OUTCOME_FIELDS,
            error_message="current no-action outcome requires canonical payload",
        )
        if (
            type(payload["analysis_summary"]) is not dict
            or type(payload["decision_summary"]) is not dict
            or type(agent_result["analysis_summary"]) is not dict
        ):
            raise ValueError("current no-action outcome requires canonical payload")
        return payload


class _LegacyHostDecisionNoActionOutcome(_StrictDTO):
    protocol_version: Literal["riskprobe.host-decision.v1"] = _PROTOCOL_VERSION
    phase: Literal["terminal"] = "terminal"
    terminal_reason: Literal["no_actionable_diagnosis"] = "no_actionable_diagnosis"
    action_codes: tuple[ActionCode, ...] = ()
    agent_result: AgentResult

    @model_validator(mode="after")
    def validate_legacy_no_action_result(
        self,
    ) -> _LegacyHostDecisionNoActionOutcome:
        if (
            not _is_clean_no_action_result(self.agent_result, self.action_codes)
            or self.agent_result.analysis_summary is not None
        ):
            raise ValueError("legacy no-action outcome is invalid")
        return self

    @model_serializer
    def serialize_legacy(self) -> dict[str, object]:
        result = self.agent_result.model_dump(mode="json")
        result.pop("analysis_summary", None)
        return {
            "protocol_version": self.protocol_version,
            "phase": self.phase,
            "terminal_reason": self.terminal_reason,
            "action_codes": tuple(action.value for action in self.action_codes),
            "agent_result": result,
        }


HostDecisionTerminal = (
    HostDecisionOutcome
    | _LegacyHostDecisionOutcome
    | _LegacyHostDecisionOutcomeV0
    | HostDecisionNoActionOutcome
    | _LegacyHostDecisionNoActionOutcome
)


class _StoredHostSession(_StrictDTO):
    format_version: Literal["riskprobe.host-sessions.v1"] = _SESSION_FORMAT
    key: str
    provider_id: str
    provider_version: str
    lifecycle: Literal["awaiting_proposal", "terminal", "failed"]
    report_run_id: str | None = None
    context: HostDecisionContext | None = None
    proposal: DecisionProposal | None = None
    outcome: dict[str, object] | None = None
    failure: HostDecisionFailure | None = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return HostDecisionCoordinator._validated_key(value)

    @field_validator("provider_id", "provider_version")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if _PUBLIC_TOKEN.fullmatch(value) is None:
            raise ValueError("provider identity must use public tokens")
        return value

    @field_validator("report_run_id")
    @classmethod
    def validate_report_run_id(cls, value: str | None) -> str | None:
        if value is not None and _PUBLIC_TOKEN.fullmatch(value) is None:
            raise ValueError("report run ID must use a public token")
        return value

    @model_validator(mode="after")
    def validate_lifecycle_payload(self) -> _StoredHostSession:
        if self.lifecycle == "terminal":
            outcome_fields = frozenset(self.outcome or {})
            is_no_action = outcome_fields in {
                _LEGACY_NO_ACTION_OUTCOME_FIELDS,
                _NO_ACTION_OUTCOME_FIELDS,
            }
            valid = (
                self.outcome is not None
                and self.failure is None
                and outcome_fields
                in {
                    _LEGACY_NORMAL_OUTCOME_FIELDS,
                    _NORMAL_OUTCOME_FIELDS,
                    _LEGACY_NO_ACTION_OUTCOME_FIELDS,
                    _NO_ACTION_OUTCOME_FIELDS,
                }
                and (not is_no_action or (self.context is None and self.proposal is None))
            )
        elif self.lifecycle == "failed":
            valid = self.outcome is None and self.failure is not None
        else:
            valid = self.outcome is None and self.failure is None
        if not valid:
            raise ValueError("host session lifecycle payload is invalid")
        return self


HostDecisionRunner = Callable[[], AgentResult]


@dataclass
class _SessionState:
    key: str
    report_run_id: str | None = None
    context: HostDecisionContext | None = None
    proposal: DecisionProposal | None = None
    outcome: HostDecisionTerminal | None = None
    failure: HostDecisionFailure | None = None
    failed: bool = False
    done: bool = False
    runner_started: bool = False


class _HostSessionStore:
    """Private atomic sidecar for safe Host session projections only."""

    def __init__(self, state_dir: Path | None) -> None:
        self.directory = None if state_dir is None else Path(state_dir).expanduser()
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.directory, 0o700)
            details = self.directory.lstat()
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) != 0o700
            ):
                raise OSError("host state directory is not private")
        except OSError as error:
            raise HostDecisionError("host decision is unavailable") from error
        self.path = self.directory / _SESSION_FILE
        self.lock_path = self.directory / _SESSION_LOCK

    def load(self) -> dict[str, _StoredHostSession]:
        if self.directory is None:
            return {}
        try:
            details = self.path.lstat()
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise HostDecisionError("host decision is unavailable") from error
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size > _MAX_SESSION_BYTES
        ):
            raise HostDecisionError("host decision is unavailable")
        try:
            content = self.path.read_bytes()
            envelope = json.loads(content)
            if (
                type(envelope) is not dict
                or set(envelope) != {"format_version", "sessions"}
                or envelope["format_version"] != _SESSION_FORMAT
                or type(envelope["sessions"]) is not dict
            ):
                raise ValueError("invalid host session envelope")
            sessions: dict[str, _StoredHostSession] = {}
            for key, payload in envelope["sessions"].items():
                if not isinstance(key, str) or _IDEMPOTENCY_KEY.fullmatch(key) is None:
                    raise ValueError("invalid host session key")
                try:
                    if type(payload) is not dict:
                        raise ValueError("invalid host session")
                    session = _StoredHostSession.model_validate_json(
                        json.dumps(
                            payload,
                            allow_nan=False,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
                    if session.key != key:
                        raise ValueError("host session key mismatch")
                except Exception:
                    provider_id = (
                        payload.get("provider_id")
                        if type(payload) is dict
                        and isinstance(payload.get("provider_id"), str)
                        and _PUBLIC_TOKEN.fullmatch(payload["provider_id"]) is not None
                        else "unavailable"
                    )
                    provider_version = (
                        payload.get("provider_version")
                        if type(payload) is dict
                        and isinstance(payload.get("provider_version"), str)
                        and _PUBLIC_TOKEN.fullmatch(payload["provider_version"])
                        is not None
                        else "unavailable"
                    )
                    session = _StoredHostSession(
                        key=key,
                        provider_id=provider_id,
                        provider_version=provider_version,
                        lifecycle="failed",
                        failure=HostDecisionFailure(
                            error_code="session_state_unavailable"
                        ),
                    )
                sessions[key] = session
            return sessions
        except HostDecisionError:
            raise
        except Exception as error:
            raise HostDecisionError("host decision is unavailable") from error

    def save(self, session: _StoredHostSession) -> None:
        if self.directory is None:
            return
        try:
            with self._locked():
                sessions = self.load()
                sessions[session.key] = session
                envelope = {
                    "format_version": _SESSION_FORMAT,
                    "sessions": {
                        key: sessions[key].model_dump(mode="json")
                        for key in sorted(sessions)
                    },
                }
                content = (
                    json.dumps(
                        envelope,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                if len(content) > _MAX_SESSION_BYTES:
                    raise HostDecisionError("host decision is unavailable")
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=self.directory,
                        prefix=f".{self.path.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                        os.fchmod(handle.fileno(), 0o600)
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.path)
                    temporary = None
                    directory_fd = os.open(self.directory, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        except HostDecisionError:
            raise
        except OSError as error:
            raise HostDecisionError("host decision is unavailable") from error

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(
            self.lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid():
                raise HostDecisionError("host decision is unavailable")
            with os.fdopen(descriptor, "r+b") as handle:
                descriptor = -1
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except HostDecisionError:
            raise
        except OSError as error:
            raise HostDecisionError("host decision is unavailable") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)


class HostDecisionCoordinator:
    """Bridge independent Host proposals into the existing synchronous agent flow."""

    mode = DecisionProviderMode.EXTERNAL_HOST

    def __init__(
        self,
        *,
        provider_id: str,
        version: str,
        state_dir: Path | None = None,
    ) -> None:
        DecisionProviderConfig(
            mode=DecisionProviderMode.EXTERNAL_HOST,
            provider_id=provider_id,
            provider_version=version,
        )
        self.provider_id = provider_id
        self.version = version
        self._condition = Condition()
        self._sessions: dict[str, _SessionState] = {}
        self._store = _HostSessionStore(state_dir)
        self._thread_key = local()
        for key, stored in self._store.load().items():
            if (
                stored.provider_id != self.provider_id
                or stored.provider_version != self.version
            ):
                self._sessions[key] = self._unavailable_session(key)
                continue
            try:
                self._sessions[key] = self._session_from_stored(stored)
            except Exception:
                self._sessions[key] = self._unavailable_session(key)

    def get_context(
        self,
        *,
        idempotency_key: str,
        runner: Callable[[], AgentResult],
    ) -> HostDecisionContext | HostDecisionFailure | HostDecisionNoActionOutcome:
        key = self._validated_key(idempotency_key)
        if not callable(runner):
            raise TypeError("runner must be callable")
        thread: Thread | None = None
        with self._condition:
            session = self._sessions.get(key)
            if session is None:
                session = _SessionState(key=key)
                self._sessions[key] = session
            if session.failure is not None:
                return session.failure
            if isinstance(
                session.outcome,
                (HostDecisionNoActionOutcome, _LegacyHostDecisionNoActionOutcome),
            ):
                return session.outcome
            if not session.done and not session.runner_started:
                session.runner_started = True
                thread = Thread(
                    target=self._execute,
                    args=(key, runner),
                    name=f"riskprobe-host-decision-{key}",
                    daemon=True,
                )
            elif session.context is not None:
                return session.context
        if thread is not None:
            thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._sessions[key].context is not None
                or self._sessions[key].done,
                timeout=_CONTEXT_WAIT_SECONDS,
            )
            session = self._sessions[key]
            if session.failure is not None:
                return session.failure
            if isinstance(
                session.outcome,
                (HostDecisionNoActionOutcome, _LegacyHostDecisionNoActionOutcome),
            ):
                return session.outcome
            if not ready:
                return HostDecisionFailure(error_code="context_timeout")
            if session.context is None:
                return HostDecisionFailure(error_code="session_state_unavailable")
            return session.context

    def get_failure(self, *, idempotency_key: str) -> HostDecisionFailure | None:
        """Return an already-recorded safe failure without starting a session."""

        key = self._validated_key(idempotency_key)
        with self._condition:
            session = self._sessions.get(key)
            if session is None:
                try:
                    session = self._refresh_session(key)
                except HostDecisionError:
                    return HostDecisionFailure(error_code="session_state_unavailable")
            return None if session is None else session.failure

    def report_subject(
        self,
        *,
        idempotency_key: str,
    ) -> TerminalReportSubject | None:
        """Project one fully persisted terminal session into a report subject."""

        key = self._validated_key(idempotency_key)
        with self._condition:
            try:
                session = self._sessions.get(key)
                if session is None:
                    session = self._refresh_session(key)
                if session is None or not session.done:
                    return None
                stored = self._store.load().get(key)
                if stored is None:
                    if session.report_run_id is None:
                        return None
                    raise HostDecisionError("host decision is unavailable")
                if stored.report_run_id is None:
                    return None
                if (
                    stored.provider_id != self.provider_id
                    or stored.provider_version != self.version
                ):
                    raise HostDecisionError("host decision is unavailable")
                persisted = self._session_from_stored(stored)
                if not persisted.done:
                    raise HostDecisionError("host decision is unavailable")
                return self._report_subject_from_session(persisted)
            except HostDecisionError:
                raise
            except Exception as error:
                raise HostDecisionError("host decision is unavailable") from error

    def _report_subject_from_session(
        self,
        session: _SessionState,
    ) -> TerminalReportSubject | None:
        run_id = session.report_run_id
        if run_id is None or not session.done:
            return None
        try:
            context = session.context
            proposal = session.proposal
            outcome = session.outcome
            if session.failed:
                if session.failure is None or outcome is not None:
                    raise HostDecisionError("host decision is unavailable")
                if context is None:
                    if proposal is not None:
                        raise HostDecisionError("host decision is unavailable")
                    context_id = None
                    findings = ()
                    diagnosis_evidence_ids = ()
                    analysis_summary = None
                    proposal_action_codes = ()
                else:
                    decision_context = context.context
                    if (
                        context.provider_id != self.provider_id
                        or context.provider_version != self.version
                        or decision_context.session_id != run_id
                    ):
                        raise HostDecisionError("host decision is unavailable")
                    context_id = decision_context.context_id
                    findings = decision_context.findings
                    diagnosis_evidence_ids = decision_context.diagnosis_evidence_ids
                    analysis_summary = decision_context.analysis_summary
                    if proposal is None:
                        proposal_action_codes = ()
                    else:
                        if (
                            proposal.context_id != decision_context.context_id
                            or proposal.diagnosis_evidence_ids
                            != decision_context.diagnosis_evidence_ids
                            or proposal.source is not DecisionSource.EXTERNAL_HOST
                            or proposal.source_version != self.version
                        ):
                            raise HostDecisionError("host decision is unavailable")
                        proposal_action_codes = tuple(
                            action.value for action in proposal.action_codes
                        )
                return TerminalReportSubject(
                    idempotency_key=session.key,
                    run_id=run_id,
                    context_id=context_id,
                    findings=findings,
                    proposal_action_codes=proposal_action_codes,
                    diagnosis_evidence_ids=diagnosis_evidence_ids,
                    agent_result=None,
                    analysis_summary=analysis_summary,
                    decision_summary=None,
                    terminal_status="failed",
                    error_code=session.failure.error_code,
                )
            if type(outcome) in {
                _LegacyHostDecisionOutcome,
                _LegacyHostDecisionOutcomeV0,
                _LegacyHostDecisionNoActionOutcome,
            }:
                return None
            if type(outcome) in {
                HostDecisionOutcome,
                HostDecisionNoActionOutcome,
            }:
                if self._store.directory is None:
                    raise HostDecisionError("host decision is unavailable")
                authoritative = AgentResultStore(
                    self._store.directory / f".{run_id}.agent-result.json"
                ).load()
                if authoritative is None or outcome.agent_result != authoritative:
                    raise HostDecisionError("host decision is unavailable")
            if type(outcome) is HostDecisionOutcome and proposal is None:
                if (
                    session.failure is not None
                    or context is None
                    or context.provider_id != self.provider_id
                    or context.provider_version != self.version
                    or context.context.session_id != run_id
                    or outcome.context_id != context.context.context_id
                    or outcome.expires_at != context.context.expires_at
                ):
                    raise HostDecisionError("host decision is unavailable")
                return None
            if type(outcome) is HostDecisionNoActionOutcome:
                if (
                    session.failure is not None
                    or context is not None
                    or proposal is not None
                    or outcome.agent_result.session_id != run_id
                ):
                    raise HostDecisionError("host decision is unavailable")
                return TerminalReportSubject(
                    idempotency_key=session.key,
                    run_id=run_id,
                    context_id=None,
                    findings=(),
                    proposal_action_codes=(),
                    diagnosis_evidence_ids=(),
                    agent_result=outcome.agent_result,
                    analysis_summary=outcome.analysis_summary,
                    decision_summary=outcome.decision_summary,
                    terminal_status="no_action",
                )
            if (
                session.failure is not None
                or type(outcome) is not HostDecisionOutcome
                or context is None
                or proposal is None
            ):
                raise HostDecisionError("host decision is unavailable")
            decision_context = context.context
            result = outcome.agent_result
            if (
                context.provider_id != self.provider_id
                or context.provider_version != self.version
                or decision_context.session_id != run_id
                or result.session_id != run_id
                or outcome.context_id != decision_context.context_id
                or outcome.expires_at != decision_context.expires_at
                or proposal.context_id != decision_context.context_id
                or proposal.diagnosis_evidence_ids
                != decision_context.diagnosis_evidence_ids
                or proposal.source is not DecisionSource.EXTERNAL_HOST
                or proposal.source_version != self.version
                or outcome.context_evidence_id is None
                or outcome.proposal_evidence_id is None
                or outcome.result_evidence_id is None
            ):
                raise HostDecisionError("host decision is unavailable")
            submission, recommendations = self._decision_evidence_projection(
                context=decision_context,
                proposal=proposal,
                result=result,
                result_evidence_id=outcome.result_evidence_id,
            )
            decision = submission.result
            if (
                submission.context_evidence_id != outcome.context_evidence_id
                or submission.proposal_evidence_id
                != outcome.proposal_evidence_id
                or decision.status is not outcome.decision_status
                or decision.reason_codes != outcome.reason_codes
                or decision.action_codes != outcome.action_codes
            ):
                raise HostDecisionError("host decision is unavailable")
            expected_analysis_summary = _terminal_analysis_summary(
                result.analysis_summary or decision_context.analysis_summary,
                findings=decision_context.findings,
                result=result,
                decision_context_succeeded=True,
            )
            expected_decision_summary = _decision_summary(
                proposal=proposal,
                result=result,
                recommendations=recommendations,
                decision_status=decision.status,
                selected_action_codes=decision.action_codes,
            )
            if (
                expected_analysis_summary is None
                or outcome.analysis_summary != expected_analysis_summary
                or outcome.decision_summary != expected_decision_summary
            ):
                raise HostDecisionError("host decision is unavailable")
            decision_summary = outcome.decision_summary
            return TerminalReportSubject(
                idempotency_key=session.key,
                run_id=run_id,
                context_id=decision_context.context_id,
                findings=decision_context.findings,
                proposal_action_codes=tuple(
                    action.value for action in proposal.action_codes
                ),
                diagnosis_evidence_ids=decision_context.diagnosis_evidence_ids,
                agent_result=result,
                analysis_summary=outcome.analysis_summary,
                decision_summary=decision_summary,
                terminal_status=outcome.decision_status.value,
                decision_reason_codes=tuple(
                    reason.value for reason in outcome.reason_codes
                ),
            )
        except HostDecisionError:
            raise
        except Exception as error:
            raise HostDecisionError("host decision is unavailable") from error

    def submit_proposal(
        self,
        *,
        idempotency_key: str,
        proposal: DecisionProposal | None = None,
    ) -> HostDecisionTerminal:
        key = self._validated_key(idempotency_key)
        canonical: DecisionProposal | None = None
        if proposal is not None:
            try:
                if type(proposal) is not DecisionProposal:
                    raise TypeError("proposal must be a DecisionProposal")
                canonical = DecisionProposal.model_validate(
                    proposal.model_dump(mode="python")
                )
            except Exception as error:
                raise HostDecisionError("host decision is unavailable") from error
        with self._condition:
            session = self._sessions.get(key)
            if session is None:
                session = self._refresh_session(key)
            if session is not None and isinstance(
                session.outcome,
                (HostDecisionNoActionOutcome, _LegacyHostDecisionNoActionOutcome),
            ):
                return session.outcome
            context = None if session is None else session.context
            if (
                canonical is None
                or session is None
                or context is None
                or canonical.context_id != context.context.context_id
                or canonical.diagnosis_evidence_ids
                != context.context.diagnosis_evidence_ids
                or canonical.source is not DecisionSource.EXTERNAL_HOST
                or canonical.source_version != self.version
            ):
                raise HostDecisionError("host decision is unavailable")
            if type(session.outcome) is HostDecisionOutcome and session.proposal is None:
                if not self._validated_terminal_replay(
                    context,
                    canonical,
                    session.outcome,
                ):
                    raise HostDecisionError("host decision is unavailable")
                session.proposal = canonical
                try:
                    self._persist(session)
                except Exception:
                    session.proposal = None
                    raise
                return session.outcome
            if type(session.outcome) in {
                _LegacyHostDecisionOutcome,
                _LegacyHostDecisionOutcomeV0,
            }:
                if session.proposal != canonical or not self._validated_legacy_terminal_replay(
                    context,
                    canonical,
                    session.outcome,
                ):
                    raise HostDecisionError("host decision is unavailable")
                return session.outcome
            if session.proposal is None:
                if self._remaining(context.context) <= 0 or session.done:
                    raise HostDecisionError("host decision is unavailable")
                session.proposal = canonical
                self._persist(session)
                self._condition.notify_all()
            elif session.proposal != canonical:
                raise HostDecisionError("host decision is unavailable")
            if session.outcome is not None:
                return session.outcome
            remaining = self._remaining(context.context)
            finished = self._condition.wait_for(
                lambda: session.outcome is not None or session.done,
                timeout=remaining,
            )
            if not finished or session.failed or session.outcome is None:
                raise HostDecisionError("host decision is unavailable")
            return session.outcome

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        if type(context) is not DecisionContext:
            raise TypeError("context must be a DecisionContext")
        key = getattr(self._thread_key, "key", None)
        with self._condition:
            if key is None:
                candidates = [
                    session
                    for session in self._sessions.values()
                    if session.context is None and not session.done
                ]
                if len(candidates) != 1:
                    raise HostDecisionError("host decision is unavailable")
                session = candidates[0]
            else:
                session = self._sessions.get(key)
                if session is None:
                    raise HostDecisionError("host decision is unavailable")
            if session.context is not None and session.context.context != context:
                raise HostDecisionError("host decision is unavailable")
            binding_changed = session.report_run_id is None
            if not self._bind_report_run_id(session, context.session_id):
                raise HostDecisionError("host decision is unavailable")
            context_changed = session.context is None
            if context_changed:
                session.context = HostDecisionContext(
                    provider_id=self.provider_id,
                    provider_version=self.version,
                    context=context,
                )
            if context_changed or binding_changed:
                self._persist(session)
                self._condition.notify_all()
            while session.proposal is None:
                remaining = self._remaining(context)
                if remaining <= 0:
                    return DecisionProviderResolution(
                        disposition=DecisionDisposition.PENDING
                    )
                self._condition.wait(timeout=remaining)
            return DecisionProviderResolution(
                disposition=DecisionDisposition.PROPOSAL,
                proposal=session.proposal,
            )

    @staticmethod
    def _failure_for_exception(error: BaseException) -> HostDecisionFailure:
        code = getattr(error, "host_failure_code", None)
        if isinstance(code, str) and code in _HOST_FAILURE_CODES:
            return HostDecisionFailure(error_code=code)
        if isinstance(error, HostDecisionError):
            return HostDecisionFailure(error_code="session_state_unavailable")
        return HostDecisionFailure(error_code="agent_session_failed")

    @staticmethod
    def _bind_report_run_id(session: _SessionState, run_id: str) -> bool:
        if _PUBLIC_TOKEN.fullmatch(run_id) is None:
            return False
        if session.report_run_id is not None and session.report_run_id != run_id:
            return False
        session.report_run_id = run_id
        return True

    @staticmethod
    def _report_run_id_for_exception(error: BaseException) -> str | None:
        code = getattr(error, "host_failure_code", None)
        run_id = getattr(error, "report_run_id", None)
        if (
            not isinstance(code, str)
            or code not in _HOST_FAILURE_CODES
            or not isinstance(run_id, str)
            or _PUBLIC_TOKEN.fullmatch(run_id) is None
        ):
            return None
        return run_id

    def _execute(self, key: str, runner: Callable[[], AgentResult]) -> None:
        self._thread_key.key = key
        try:
            result = runner()
            if type(result) is not AgentResult:
                raise TypeError("runner returned an invalid result")
        except Exception as error:
            with self._condition:
                session = self._sessions.get(key)
                if session is not None:
                    session.failure = self._failure_for_exception(error)
                    report_run_id = self._report_run_id_for_exception(error)
                    if report_run_id is not None and not self._bind_report_run_id(
                        session,
                        report_run_id,
                    ):
                        session.report_run_id = None
                        session.failure = HostDecisionFailure(
                            error_code="session_state_unavailable"
                        )
                    session.failed = True
                    session.done = True
                    try:
                        self._persist(session)
                    except HostDecisionError:
                        session.failure = HostDecisionFailure(
                            error_code="session_state_unavailable"
                        )
                    self._condition.notify_all()
            return
        finally:
            self._thread_key.key = None
        with self._condition:
            session = self._sessions.get(key)
            if session is None:
                return
            if not self._bind_report_run_id(session, result.session_id):
                session.report_run_id = None
                session.failure = HostDecisionFailure(
                    error_code="session_state_unavailable"
                )
                session.failed = True
                session.done = True
            elif session.context is None:
                replay = self._terminal_replay_for(result)
                if replay is not None:
                    session.context, session.outcome = replay
                else:
                    try:
                        analysis_summary = _terminal_analysis_summary(
                            result.analysis_summary,
                            findings=(),
                            result=result,
                            decision_context_succeeded=False,
                        )
                        if analysis_summary is None:
                            raise ValueError("analysis summary is unavailable")
                        session.outcome = HostDecisionNoActionOutcome(
                            agent_result=result,
                            analysis_summary=analysis_summary,
                            decision_summary=_no_action_decision_summary(result),
                        )
                    except ValueError:
                        session.failure = HostDecisionFailure(
                            error_code=(
                                "agent_orchestration_failed"
                                if result.status is AgentStatus.REJECTED
                                or ReviewReason.TOOL_FAILURE
                                in result.review.reason_codes
                                else "agent_state_incomplete"
                            )
                        )
                        session.failed = True
                session.done = True
            elif session.proposal is None:
                session.failure = HostDecisionFailure(error_code="agent_session_failed")
                session.failed = True
                session.done = True
            else:
                try:
                    session.outcome = self._build_outcome(
                        session.context.context,
                        session.proposal,
                        result,
                    )
                    session.done = True
                except Exception as error:
                    session.failure = self._failure_for_exception(error)
                    session.failed = True
                    session.done = True
            try:
                self._persist(session)
            except HostDecisionError:
                session.failure = HostDecisionFailure(
                    error_code="session_state_unavailable"
                )
                session.failed = True
                session.done = True
            self._condition.notify_all()

    def _terminal_replay_for(
        self,
        result: AgentResult,
    ) -> tuple[HostDecisionContext, HostDecisionOutcome] | None:
        in_flight = [
            candidate
            for candidate in self._sessions.values()
            if (
                not candidate.failed
                and not candidate.done
                and candidate.context is not None
                and candidate.proposal is not None
                and candidate.context.context.session_id == result.session_id
            )
        ]
        if in_flight:
            self._condition.wait_for(
                lambda: all(candidate.done for candidate in in_flight),
                timeout=_CONTEXT_WAIT_SECONDS,
            )
        replay: tuple[HostDecisionContext, HostDecisionOutcome] | None = None
        for candidate in self._sessions.values():
            if (
                candidate.failed
                or not candidate.done
                or candidate.context is None
                or candidate.proposal is None
                or type(candidate.outcome) is not HostDecisionOutcome
                or candidate.outcome.agent_result != result
            ):
                continue
            if not self._validated_terminal_replay(
                candidate.context,
                candidate.proposal,
                candidate.outcome,
            ):
                return None
            current = (candidate.context, candidate.outcome)
            if replay is not None and replay != current:
                return None
            replay = current
        return replay

    def _decision_evidence_projection(
        self,
        *,
        context: DecisionContext,
        proposal: DecisionProposal,
        result: AgentResult,
        result_evidence_id: str | None = None,
    ) -> tuple[DecisionSubmission, tuple[RecommendationSummary, ...]]:
        if (
            self._store.directory is None
            or _PUBLIC_TOKEN.fullmatch(context.session_id) is None
        ):
            raise HostDecisionError("host decision is unavailable")
        evidence_path = (
            self._store.directory / f".{context.session_id}.evidence.sqlite3"
        )
        details = evidence_path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
        ):
            raise HostDecisionError("host decision is unavailable")

        from riskprobe.agents.decision_controller import DecisionController
        from riskprobe.evidence import EvidenceStore

        evidence_store = EvidenceStore(evidence_path)
        records = evidence_store.list_run(context.session_id)
        matching_result_ids = tuple(
            EvidenceStore.content_id(record)
            for record in records
            if record.kind == "decision.result"
            and record.payload.get("proposal_id") == proposal.proposal_id
        )
        if result_evidence_id is None:
            if len(matching_result_ids) != 1:
                raise HostDecisionError("host decision is unavailable")
            result_evidence_id = matching_result_ids[0]
        elif matching_result_ids != (result_evidence_id,):
            raise HostDecisionError("host decision is unavailable")
        submission = DecisionController(evidence_store).replay(
            result_evidence_id=result_evidence_id,
            expected_run_id=context.session_id,
        )
        selected = submission.provider_binding.selected
        if (
            submission.context != context
            or submission.proposal != proposal
            or result.session_id != context.session_id
            or result.diagnosis_evidence_ids != context.diagnosis_evidence_ids
            or not _terminal_decision_matches(result, submission.result)
            or selected.provider_id != self.provider_id
            or selected.mode is not DecisionProviderMode.EXTERNAL_HOST
            or selected.version != self.version
        ):
            raise HostDecisionError("host decision is unavailable")
        recommendations = _recommendation_summaries_from_evidence(
            evidence_store=evidence_store,
            records=records,
            context=context,
            decision=submission.result,
            result=result,
        )
        return submission, recommendations

    def _validated_legacy_terminal_replay(
        self,
        context: HostDecisionContext,
        proposal: DecisionProposal,
        outcome: _LegacyHostDecisionOutcome,
    ) -> bool:
        if (
            outcome.context_evidence_id is None
            or outcome.proposal_evidence_id is None
            or outcome.result_evidence_id is None
        ):
            return False
        try:
            submission, _ = self._decision_evidence_projection(
                context=context.context,
                proposal=proposal,
                result=outcome.agent_result,
                result_evidence_id=outcome.result_evidence_id,
            )
            return (
                context.provider_id == self.provider_id
                and context.provider_version == self.version
                and submission.context_evidence_id == outcome.context_evidence_id
                and submission.proposal_evidence_id == outcome.proposal_evidence_id
                and submission.result_evidence_id == outcome.result_evidence_id
                and submission.result.status is outcome.decision_status
                and submission.result.reason_codes == outcome.reason_codes
                and submission.result.action_codes == outcome.action_codes
                and outcome.context_id == context.context.context_id
                and outcome.expires_at == context.context.expires_at
            )
        except Exception:
            return False

    def _validated_terminal_replay(
        self,
        context: HostDecisionContext,
        proposal: DecisionProposal,
        outcome: HostDecisionOutcome,
    ) -> bool:
        if (
            outcome.context_evidence_id is None
            or outcome.proposal_evidence_id is None
            or outcome.result_evidence_id is None
        ):
            return False
        try:
            submission, recommendations = self._decision_evidence_projection(
                context=context.context,
                proposal=proposal,
                result=outcome.agent_result,
                result_evidence_id=outcome.result_evidence_id,
            )
            expected_analysis_summary = _terminal_analysis_summary(
                outcome.agent_result.analysis_summary
                or context.context.analysis_summary,
                findings=context.context.findings,
                result=outcome.agent_result,
                decision_context_succeeded=True,
            )
            expected_decision_summary = _decision_summary(
                proposal=proposal,
                result=outcome.agent_result,
                recommendations=recommendations,
                decision_status=submission.result.status,
                selected_action_codes=submission.result.action_codes,
            )
            return (
                context.provider_id == self.provider_id
                and context.provider_version == self.version
                and submission.context_evidence_id == outcome.context_evidence_id
                and submission.proposal_evidence_id == outcome.proposal_evidence_id
                and submission.result_evidence_id == outcome.result_evidence_id
                and submission.result.status is outcome.decision_status
                and submission.result.reason_codes == outcome.reason_codes
                and submission.result.action_codes == outcome.action_codes
                and outcome.context_id == context.context.context_id
                and outcome.expires_at == context.context.expires_at
                and expected_analysis_summary is not None
                and outcome.analysis_summary == expected_analysis_summary
                and outcome.decision_summary == expected_decision_summary
            )
        except Exception:
            return False

    def _build_outcome(
        self,
        context: DecisionContext,
        proposal: DecisionProposal,
        result: AgentResult,
    ) -> HostDecisionOutcome:
        try:
            submission, recommendation_summaries = (
                self._decision_evidence_projection(
                    context=context,
                    proposal=proposal,
                    result=result,
                )
            )
            decision = submission.result
            return HostDecisionOutcome(
                context_id=context.context_id,
                agent_result=result,
                decision_status=decision.status,
                reason_codes=decision.reason_codes,
                action_codes=decision.action_codes,
                context_evidence_id=submission.context_evidence_id,
                proposal_evidence_id=submission.proposal_evidence_id,
                result_evidence_id=submission.result_evidence_id,
                expires_at=context.expires_at,
                analysis_summary=_terminal_analysis_summary(
                    result.analysis_summary or context.analysis_summary,
                    findings=context.findings,
                    result=result,
                    decision_context_succeeded=True,
                ),
                decision_summary=_decision_summary(
                    proposal=proposal,
                    result=result,
                    recommendations=recommendation_summaries,
                    decision_status=decision.status,
                    selected_action_codes=decision.action_codes,
                ),
            )
        except HostDecisionError:
            raise
        except Exception as error:
            raise HostDecisionError("host decision is unavailable") from error

    def _refresh_session(self, key: str) -> _SessionState | None:
        stored = self._store.load().get(key)
        if stored is None:
            return None
        if (
            stored.provider_id != self.provider_id
            or stored.provider_version != self.version
        ):
            session = self._unavailable_session(key)
        else:
            try:
                session = self._session_from_stored(stored)
            except Exception:
                session = self._unavailable_session(key)
        self._sessions[key] = session
        return session

    def _persist(self, session: _SessionState) -> None:
        lifecycle = (
            "failed"
            if session.failed
            else "terminal"
            if session.outcome is not None
            else "awaiting_proposal"
        )
        self._store.save(
            _StoredHostSession(
                key=session.key,
                provider_id=self.provider_id,
                provider_version=self.version,
                lifecycle=lifecycle,
                report_run_id=session.report_run_id,
                context=session.context,
                proposal=session.proposal,
                outcome=(
                    None
                    if session.outcome is None
                    else session.outcome.model_dump(mode="json")
                ),
                failure=session.failure,
            )
        )

    @staticmethod
    def _unavailable_session(key: str) -> _SessionState:
        return _SessionState(
            key=key,
            failure=HostDecisionFailure(error_code="session_state_unavailable"),
            failed=True,
            done=True,
        )

    @staticmethod
    def _session_from_stored(stored: _StoredHostSession) -> _SessionState:
        outcome = (
            None
            if stored.outcome is None
            else HostDecisionCoordinator._outcome_from_payload(stored.outcome)
        )
        failure = (
            stored.failure
            if stored.lifecycle == "failed" and stored.failure is not None
            else HostDecisionFailure(error_code="session_state_unavailable")
            if stored.lifecycle == "failed"
            else None
        )
        report_run_id = stored.report_run_id
        if report_run_id is not None:
            if (
                stored.context is not None
                and stored.context.context.session_id != report_run_id
            ) or (
                outcome is not None
                and outcome.agent_result.session_id != report_run_id
            ):
                raise ValueError("host session report binding is inconsistent")
        return _SessionState(
            key=stored.key,
            report_run_id=report_run_id,
            context=stored.context,
            proposal=stored.proposal,
            outcome=outcome,
            failure=failure,
            failed=stored.lifecycle == "failed",
            done=stored.lifecycle in {"terminal", "failed"},
            runner_started=False,
        )

    @staticmethod
    def _outcome_from_payload(payload: dict[str, object]) -> HostDecisionTerminal:
        payload_fields = frozenset(payload)
        if (
            payload_fields
            not in {
                _LEGACY_NORMAL_OUTCOME_FIELDS,
                _NORMAL_OUTCOME_FIELDS,
                _LEGACY_NO_ACTION_OUTCOME_FIELDS,
                _NO_ACTION_OUTCOME_FIELDS,
            }
            or type(payload.get("agent_result")) is not dict
        ):
            raise HostDecisionError("host decision is unavailable")
        try:
            from riskprobe.agents.results import _reconstruct_result

            agent_result_payload = payload["agent_result"]
            agent_result = _reconstruct_result(agent_result_payload)
            raw_actions = payload["action_codes"]
            if type(raw_actions) is not list:
                raise ValueError("invalid action codes")
            action_codes = tuple(ActionCode(action) for action in raw_actions)
            agent_result_is_current = "analysis_summary" in agent_result_payload
            review_payload = agent_result_payload.get("review")
            if type(review_payload) is not dict:
                raise ValueError("invalid review generation")
            review_fields = frozenset(review_payload)
            review_is_current = review_fields == _CURRENT_REVIEW_FIELDS
            review_is_legacy = review_fields == _LEGACY_REVIEW_FIELDS
            if not review_is_current and not review_is_legacy:
                raise ValueError("invalid review generation")
            if payload_fields == _LEGACY_NO_ACTION_OUTCOME_FIELDS:
                if agent_result_is_current or not review_is_current:
                    raise ValueError("mixed legacy no-action payload")
                return _LegacyHostDecisionNoActionOutcome(
                    protocol_version=payload["protocol_version"],
                    phase=payload["phase"],
                    terminal_reason=payload["terminal_reason"],
                    action_codes=action_codes,
                    agent_result=agent_result,
                )
            if payload_fields == _NO_ACTION_OUTCOME_FIELDS:
                if (
                    not agent_result_is_current
                    or not review_is_current
                    or type(payload["analysis_summary"]) is not dict
                    or type(payload["decision_summary"]) is not dict
                ):
                    raise ValueError("invalid current no-action payload")
                analysis_summary, decision_summary = _canonical_summaries(payload)
                return HostDecisionNoActionOutcome(
                    protocol_version=payload["protocol_version"],
                    phase=payload["phase"],
                    terminal_reason=payload["terminal_reason"],
                    action_codes=action_codes,
                    agent_result=agent_result,
                    analysis_summary=analysis_summary,
                    decision_summary=decision_summary,
                )
            raw_reasons = payload["reason_codes"]
            if type(raw_reasons) is not list:
                raise ValueError("invalid decision fields")
            outcome_fields = {
                "protocol_version": payload["protocol_version"],
                "phase": payload["phase"],
                "context_id": payload["context_id"],
                "agent_result": agent_result,
                "decision_status": DecisionStatus(payload["decision_status"]),
                "reason_codes": tuple(
                    DecisionReason(reason) for reason in raw_reasons
                ),
                "action_codes": action_codes,
                "context_evidence_id": payload["context_evidence_id"],
                "proposal_evidence_id": payload["proposal_evidence_id"],
                "result_evidence_id": payload["result_evidence_id"],
                "expires_at": datetime.fromisoformat(
                    str(payload["expires_at"]).replace("Z", "+00:00")
                ),
            }
            if payload_fields == _LEGACY_NORMAL_OUTCOME_FIELDS:
                if agent_result_is_current:
                    raise ValueError("mixed legacy outcome payload")
                legacy_type = (
                    _LegacyHostDecisionOutcomeV0
                    if review_is_legacy
                    else _LegacyHostDecisionOutcome
                )
                return legacy_type(**outcome_fields)
            if payload_fields != _NORMAL_OUTCOME_FIELDS:
                raise ValueError("invalid outcome generation")
            if (
                not agent_result_is_current
                or not review_is_current
                or type(payload["analysis_summary"]) is not dict
                or type(payload["decision_summary"]) is not dict
            ):
                raise ValueError("invalid current outcome payload")
            analysis_summary, decision_summary = _canonical_summaries(payload)
            return HostDecisionOutcome(
                **outcome_fields,
                analysis_summary=analysis_summary,
                decision_summary=decision_summary,
            )
        except HostDecisionError:
            raise
        except Exception as error:
            raise HostDecisionError("host decision is unavailable") from error

    @staticmethod
    def _remaining(context: DecisionContext) -> float:
        remaining = (context.expires_at - datetime.now(UTC)).total_seconds()
        return remaining if math.isfinite(remaining) else 0.0

    @staticmethod
    def _validated_key(value: str) -> str:
        if not isinstance(value, str) or _IDEMPOTENCY_KEY.fullmatch(value) is None:
            raise ValueError("idempotency_key must be a public token")
        return value


__all__ = [
    "HostDecisionContext",
    "HostDecisionCoordinator",
    "HostDecisionError",
    "HostDecisionFailure",
    "HostDecisionNoActionOutcome",
    "HostDecisionOutcome",
    "HostDecisionRunner",
    "HostDecisionTerminal",
]


def _terminal_analysis_summary(
    summary: AnalysisSummary | None,
    *,
    findings: tuple[DecisionFinding, ...],
    result: AgentResult,
    decision_context_succeeded: bool,
) -> AnalysisSummary | None:
    if summary is None:
        return None
    updates = {stage.name: stage for stage in summary.stages}

    def succeeded(name: StageName) -> StageSummary:
        return StageSummary(
            name=name,
            status=StageStatus.SUCCEEDED,
            enabled=True,
            output_available=True,
        )

    if "inspect" in result.tool_sequence:
        updates[StageName.INSPECT] = succeeded(StageName.INSPECT)
    no_action = result.review.no_action_required
    if no_action:
        for name in (StageName.DECISION_CONTEXT, StageName.RECOMMEND):
            updates[name] = StageSummary(
                name=name,
                status=StageStatus.SKIPPED,
                enabled=True,
                output_available=False,
                reason_code="no_action_required",
            )
    else:
        if decision_context_succeeded:
            updates[StageName.DECISION_CONTEXT] = succeeded(
                StageName.DECISION_CONTEXT
            )
        if "recommend" in result.tool_sequence:
            updates[StageName.RECOMMEND] = succeeded(StageName.RECOMMEND)
    review_status = (
        StageStatus.SUCCEEDED if result.review.approved else StageStatus.FAILED
    )
    review_reason = None if result.review.approved else "review_rejected"
    updates[StageName.REVIEW] = StageSummary(
        name=StageName.REVIEW,
        status=review_status,
        enabled=True,
        output_available=result.review.approved,
        reason_code=review_reason,
    )
    updates[StageName.TERMINAL] = StageSummary(
        name=StageName.TERMINAL,
        status=review_status,
        enabled=True,
        output_available=result.review.approved,
        reason_code=review_reason,
    )
    payload = summary.model_dump(mode="python")
    finding_counts_by_kind: dict[str, int] = {}
    finding_counts_by_severity: dict[str, int] = {}
    finding_counts_by_code: dict[str, int] = {}
    for item in findings:
        finding = item.finding
        for counts, value in (
            (finding_counts_by_kind, finding.kind.value),
            (finding_counts_by_severity, finding.severity.value),
            (finding_counts_by_code, finding.code),
        ):
            counts[value] = counts.get(value, 0) + 1
    payload["diagnostics"] = DiagnosticsSummary(
        diagnostic_time_enabled=bool(
            summary.partition is not None
            and summary.partition.applied_time_validation
        ),
        finding_counts_by_kind=finding_counts_by_kind,
        finding_counts_by_severity=finding_counts_by_severity,
        finding_counts_by_code=finding_counts_by_code,
    )
    payload["stages"] = tuple(updates[name] for name in StageName)
    return AnalysisSummary.model_validate(payload)


def _normal_terminal_state_matches(
    result: AgentResult,
    *,
    decision_status: DecisionStatus,
    reason_codes: tuple[DecisionReason, ...],
    action_codes: tuple[ActionCode, ...],
) -> bool:
    if (
        result.review.no_action_required
        or result.review.evidence_ids != result.evidence_ids
    ):
        return False
    if decision_status is DecisionStatus.ACCEPTED:
        return (
            not reason_codes
            and bool(action_codes)
            and result.status is AgentStatus.SUCCEEDED
            and result.review.approved
            and not result.review.reason_codes
            and not result.review.retry_allowed
            and result.tool_sequence
            == ("inspect", "diagnose", "discover", "recommend", "review")
        )
    return (
        bool(reason_codes)
        and not action_codes
        and result.status is AgentStatus.REJECTED
        and not result.review.approved
        and result.review.reason_codes == (ReviewReason.TOOL_FAILURE,)
        and not result.review.retry_allowed
        and result.tool_sequence == ("inspect", "diagnose", "discover", "review")
    )


def _terminal_decision_matches(
    result: AgentResult,
    decision: DecisionResult,
) -> bool:
    return (
        result.diagnosis_evidence_ids == decision.diagnosis_evidence_ids
        and _normal_terminal_state_matches(
            result,
            decision_status=decision.status,
            reason_codes=decision.reason_codes,
            action_codes=decision.action_codes,
        )
    )


def _recommendation_summaries_from_evidence(
    *,
    evidence_store: EvidenceStore,
    records: tuple[EvidenceRecord, ...],
    context: DecisionContext,
    decision: DecisionResult,
    result: AgentResult,
) -> tuple[RecommendationSummary, ...]:
    from riskprobe.recommendations.models import (
        DecisionEligibility,
        Recommendation,
    )

    records_by_id = {
        evidence_store.content_id(record): record for record in records
    }
    result_evidence_ids = set(result.evidence_ids)
    if not result_evidence_ids.issubset(records_by_id):
        raise ValueError("result evidence is unavailable")
    recommendation_records = tuple(
        (evidence_id, records_by_id[evidence_id])
        for evidence_id in sorted(result_evidence_ids)
        if records_by_id[evidence_id].kind == "recommendation"
    )
    recommendation_evidence_ids = {
        evidence_id for evidence_id, _ in recommendation_records
    }
    if result_evidence_ids != (
        set(context.diagnosis_evidence_ids) | recommendation_evidence_ids
    ):
        raise ValueError("result evidence set is invalid")
    if decision.status is DecisionStatus.REJECTED and recommendation_records:
        raise ValueError("rejected decisions cannot contain recommendations")

    finding_evidence = {
        item.finding.finding_id: item.evidence_id for item in context.findings
    }
    expected_payload_fields = frozenset(Recommendation.model_fields) | {
        "dataset_id"
    }
    actions: list[ActionCode] = []
    summaries: list[RecommendationSummary] = []
    for evidence_id, record in recommendation_records:
        if record.run_id != context.session_id:
            raise ValueError("recommendation run binding is invalid")
        payload = dict(record.payload)
        if (
            frozenset(payload) != expected_payload_fields
            or payload.pop("dataset_id") != context.dataset_id
        ):
            raise ValueError("recommendation payload is invalid")
        recommendation = Recommendation.model_validate_json(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        try:
            action = ActionCode(recommendation.action_code)
            expected_parents = tuple(
                sorted(
                    finding_evidence[finding_id]
                    for finding_id in recommendation.finding_ids
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("recommendation binding is invalid") from error
        if (
            record.parent_ids != expected_parents
            or recommendation.human_approval_required is not True
            or (
                context.metadata_grade == "B"
                and recommendation.decision_eligibility
                is not DecisionEligibility.ANALYSIS_ONLY
            )
            or action in actions
        ):
            raise ValueError("recommendation binding is invalid")
        actions.append(action)
        summaries.append(
            RecommendationSummary(
                action_code=recommendation.action_code,
                parent_finding_ids=recommendation.finding_ids,
                evidence_id=evidence_id,
                analysis_only=(
                    recommendation.decision_eligibility
                    is DecisionEligibility.ANALYSIS_ONLY
                ),
                limitations=(recommendation.rationale_code,),
            )
        )
    if tuple(sorted(actions, key=lambda action: action.value)) != decision.action_codes:
        raise ValueError("recommendation actions are invalid")
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (summary.action_code, summary.evidence_id),
        )
    )


def _no_action_decision_summary(result: AgentResult) -> DecisionSummary:
    return DecisionSummary(
        selected_action_codes=(),
        recommendations=(),
        recommendation_status=StageStatus.SKIPPED,
        review_approved=True,
        review_reason_codes=(),
        no_action_required=True,
        retry_count=result.retry_count,
        tool_sequence=result.tool_sequence,
        evidence_complete=True,
        decision_status="no_action",
        final_status="succeeded",
    )


def _decision_summary(
    *,
    proposal: DecisionProposal,
    result: AgentResult,
    recommendations: tuple[RecommendationSummary, ...],
    decision_status: DecisionStatus,
    selected_action_codes: tuple[ActionCode, ...],
) -> DecisionSummary:
    no_action = result.review.no_action_required
    return DecisionSummary(
        selected_action_codes=tuple(action.value for action in selected_action_codes),
        recommendations=recommendations,
        recommendation_status=(
            StageStatus.SKIPPED if no_action else StageStatus.SUCCEEDED
            if result.review.approved else StageStatus.FAILED
        ),
        review_approved=result.review.approved,
        review_reason_codes=tuple(reason.value for reason in result.review.reason_codes),
        no_action_required=no_action,
        retry_count=result.retry_count,
        tool_sequence=result.tool_sequence,
        evidence_complete=(
            tuple(sorted(result.diagnosis_evidence_ids))
            == tuple(sorted(proposal.diagnosis_evidence_ids))
        ),
        decision_status=(
            "accepted" if decision_status is DecisionStatus.ACCEPTED else "rejected"
        ),
        final_status=("succeeded" if result.review.approved else "rejected"),
    )
