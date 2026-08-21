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
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from riskprobe.agents.contracts import AgentResult
from riskprobe.agents.decision_contracts import (
    DecisionContext,
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

_PUBLIC_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_WAIT_SECONDS = 300.0
_PROTOCOL_VERSION = "riskprobe.host-decision.v1"
_SESSION_FORMAT = "riskprobe.host-sessions.v1"
_SESSION_FILE = ".riskprobe-host-sessions.json"
_SESSION_LOCK = ".riskprobe-host-sessions.lock"
_MAX_SESSION_BYTES = 10 * 1024 * 1024


class HostDecisionError(RuntimeError):
    """Raised with a fixed message when a host decision cannot safely continue."""


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


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


class HostDecisionOutcome(_StrictDTO):
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


class _StoredHostSession(_StrictDTO):
    format_version: Literal["riskprobe.host-sessions.v1"] = _SESSION_FORMAT
    key: str
    provider_id: str
    provider_version: str
    lifecycle: Literal["awaiting_proposal", "terminal", "failed"]
    context: HostDecisionContext | None = None
    proposal: DecisionProposal | None = None
    outcome: dict[str, object] | None = None

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


HostDecisionRunner = Callable[[], AgentResult]


@dataclass
class _SessionState:
    key: str
    context: HostDecisionContext | None = None
    proposal: DecisionProposal | None = None
    outcome: HostDecisionOutcome | None = None
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
                if not isinstance(key, str) or type(payload) is not dict:
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
                raise HostDecisionError("host decision is unavailable")
            self._sessions[key] = self._session_from_stored(stored)

    def get_context(
        self,
        *,
        idempotency_key: str,
        runner: Callable[[], AgentResult],
    ) -> HostDecisionContext:
        key = self._validated_key(idempotency_key)
        if not callable(runner):
            raise TypeError("runner must be callable")
        thread: Thread | None = None
        with self._condition:
            session = self._sessions.get(key)
            if session is None:
                session = _SessionState(key=key)
                self._sessions[key] = session
            if session.context is None and not session.runner_started:
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
            context = self._sessions[key].context
            if not ready or context is None:
                raise HostDecisionError("host decision is unavailable")
            return context

    def submit_proposal(
        self,
        *,
        idempotency_key: str,
        proposal: DecisionProposal,
    ) -> HostDecisionOutcome:
        key = self._validated_key(idempotency_key)
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
            context = None if session is None else session.context
            if (
                session is None
                or context is None
                or canonical.context_id != context.context.context_id
                or canonical.diagnosis_evidence_ids
                != context.context.diagnosis_evidence_ids
                or canonical.source is not DecisionSource.EXTERNAL_HOST
                or canonical.source_version != self.version
            ):
                raise HostDecisionError("host decision is unavailable")
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
            if session.context is None:
                session.context = HostDecisionContext(
                    provider_id=self.provider_id,
                    provider_version=self.version,
                    context=context,
                )
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

    def _execute(self, key: str, runner: Callable[[], AgentResult]) -> None:
        self._thread_key.key = key
        try:
            result = runner()
            if type(result) is not AgentResult:
                raise TypeError("runner returned an invalid result")
        except Exception:
            with self._condition:
                session = self._sessions.get(key)
                if session is not None:
                    session.failed = True
                    session.done = True
                    self._persist(session)
                    self._condition.notify_all()
            return
        finally:
            self._thread_key.key = None
        with self._condition:
            session = self._sessions.get(key)
            if session is None:
                return
            if session.context is None or session.proposal is None:
                session.failed = True
                session.done = True
            else:
                session.outcome = self._build_outcome(
                    session.context.context,
                    session.proposal,
                    result,
                )
                session.done = True
            self._persist(session)
            self._condition.notify_all()

    def _build_outcome(
        self,
        context: DecisionContext,
        proposal: DecisionProposal,
        result: AgentResult,
    ) -> HostDecisionOutcome:
        decision_status = (
            DecisionStatus.ACCEPTED
            if result.review.approved
            else DecisionStatus.REJECTED
        )
        reason_codes: tuple[DecisionReason, ...] = ()
        action_codes = proposal.action_codes if result.review.approved else ()
        context_evidence_id = proposal_evidence_id = result_evidence_id = None
        evidence_path = None
        if self._store.directory is not None and _PUBLIC_TOKEN.fullmatch(context.session_id):
            candidate = self._store.directory / f".{context.session_id}.evidence.sqlite3"
            if candidate.is_file():
                evidence_path = candidate
        if evidence_path is not None:
            try:
                from riskprobe.evidence import EvidenceStore

                records = EvidenceStore(evidence_path).list_run(context.session_id)
                context_record = next(
                    (
                        record
                        for record in records
                        if record.kind == "decision.context"
                        and record.payload.get("context_id") == context.context_id
                    ),
                    None,
                )
                proposal_record = next(
                    (
                        record
                        for record in records
                        if record.kind == "decision.proposal"
                        and isinstance(record.payload.get("proposal"), dict)
                        and record.payload["proposal"].get("proposal_id")
                        == proposal.proposal_id
                    ),
                    None,
                )
                result_record = next(
                    (
                        record
                        for record in records
                        if record.kind == "decision.result"
                        and record.payload.get("proposal_id")
                        == proposal.proposal_id
                    ),
                    None,
                )
                if context_record and proposal_record and result_record:
                    decision = DecisionResult.model_validate_json(
                        json.dumps(
                            dict(result_record.payload),
                            allow_nan=False,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
                    decision_status = decision.status
                    reason_codes = decision.reason_codes
                    action_codes = decision.action_codes
                    context_evidence_id = EvidenceStore.content_id(context_record)
                    proposal_evidence_id = EvidenceStore.content_id(proposal_record)
                    result_evidence_id = EvidenceStore.content_id(result_record)
            except Exception:
                pass
        return HostDecisionOutcome(
            context_id=context.context_id,
            agent_result=result,
            decision_status=decision_status,
            reason_codes=reason_codes,
            action_codes=action_codes,
            context_evidence_id=context_evidence_id,
            proposal_evidence_id=proposal_evidence_id,
            result_evidence_id=result_evidence_id,
            expires_at=context.expires_at,
        )

    def _refresh_session(self, key: str) -> _SessionState | None:
        stored = self._store.load().get(key)
        if stored is None:
            return None
        if (
            stored.provider_id != self.provider_id
            or stored.provider_version != self.version
        ):
            raise HostDecisionError("host decision is unavailable")
        session = self._session_from_stored(stored)
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
                context=session.context,
                proposal=session.proposal,
                outcome=(
                    None
                    if session.outcome is None
                    else session.outcome.model_dump(mode="json")
                ),
            )
        )

    @staticmethod
    def _session_from_stored(stored: _StoredHostSession) -> _SessionState:
        outcome = (
            None
            if stored.outcome is None
            else HostDecisionCoordinator._outcome_from_payload(stored.outcome)
        )
        return _SessionState(
            key=stored.key,
            context=stored.context,
            proposal=stored.proposal,
            outcome=outcome,
            failed=stored.lifecycle == "failed",
            done=stored.lifecycle in {"terminal", "failed"},
            runner_started=stored.context is not None,
        )

    @staticmethod
    def _outcome_from_payload(payload: dict[str, object]) -> HostDecisionOutcome:
        expected = {
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
        if set(payload) != expected or type(payload["agent_result"]) is not dict:
            raise HostDecisionError("host decision is unavailable")
        try:
            from riskprobe.agents.results import _reconstruct_result

            agent_result = _reconstruct_result(payload["agent_result"])
            raw_reasons = payload["reason_codes"]
            raw_actions = payload["action_codes"]
            if type(raw_reasons) is not list or type(raw_actions) is not list:
                raise ValueError("invalid decision fields")
            expires_at = datetime.fromisoformat(str(payload["expires_at"]).replace("Z", "+00:00"))
            return HostDecisionOutcome(
                protocol_version=payload["protocol_version"],
                phase=payload["phase"],
                context_id=payload["context_id"],
                agent_result=agent_result,
                decision_status=DecisionStatus(payload["decision_status"]),
                reason_codes=tuple(DecisionReason(reason) for reason in raw_reasons),
                action_codes=tuple(ActionCode(action) for action in raw_actions),
                context_evidence_id=payload["context_evidence_id"],
                proposal_evidence_id=payload["proposal_evidence_id"],
                result_evidence_id=payload["result_evidence_id"],
                expires_at=expires_at,
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
    "HostDecisionOutcome",
    "HostDecisionRunner",
]
