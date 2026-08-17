"""Host-driven two-phase decision coordination without model callbacks."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Condition, Thread
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from riskprobe.agents.contracts import AgentResult
from riskprobe.agents.decision_contracts import DecisionContext, DecisionProposal, DecisionSource
from riskprobe.agents.decision_providers import (
    DecisionDisposition,
    DecisionProviderConfig,
    DecisionProviderMode,
    DecisionProviderResolution,
)

_PUBLIC_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTEXT_WAIT_SECONDS = 300.0
_TERMINAL_WAIT_SECONDS = 300.0
_PROTOCOL_VERSION = "riskprobe.host-decision.v1"


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


HostDecisionRunner = Callable[[], AgentResult]


class HostDecisionCoordinator:
    """Bridge one active Host proposal into the existing synchronous agent flow."""

    mode = DecisionProviderMode.EXTERNAL_HOST

    def __init__(self, *, provider_id: str, version: str) -> None:
        DecisionProviderConfig(
            mode=DecisionProviderMode.EXTERNAL_HOST,
            provider_id=provider_id,
            provider_version=version,
        )
        self.provider_id = provider_id
        self.version = version
        self._condition = Condition()
        self._idempotency_key: str | None = None
        self._context: DecisionContext | None = None
        self._proposal: DecisionProposal | None = None
        self._result: AgentResult | None = None
        self._failed = False
        self._done = False

    def get_context(
        self,
        *,
        idempotency_key: str,
        runner: HostDecisionRunner,
    ) -> HostDecisionContext:
        key = self._validated_key(idempotency_key)
        if not callable(runner):
            raise TypeError("runner must be callable")
        thread: Thread | None = None
        with self._condition:
            if self._idempotency_key is None:
                self._idempotency_key = key
                thread = Thread(
                    target=self._execute,
                    args=(runner,),
                    name="riskprobe-host-decision",
                    daemon=True,
                )
            elif self._idempotency_key != key:
                raise HostDecisionError("host decision is unavailable")
        if thread is not None:
            thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._context is not None or self._done,
                timeout=_CONTEXT_WAIT_SECONDS,
            )
            if not ready or self._context is None:
                raise HostDecisionError("host decision is unavailable")
            return HostDecisionContext(
                provider_id=self.provider_id,
                provider_version=self.version,
                context=self._context,
            )

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
            context = self._context
            if (
                self._idempotency_key != key
                or context is None
                or canonical.context_id != context.context_id
                or canonical.diagnosis_evidence_ids
                != context.diagnosis_evidence_ids
                or canonical.source is not DecisionSource.EXTERNAL_HOST
                or canonical.source_version != self.version
            ):
                raise HostDecisionError("host decision is unavailable")
            if self._proposal is None:
                if self._done:
                    raise HostDecisionError("host decision is unavailable")
                self._proposal = canonical
                self._condition.notify_all()
            elif self._proposal != canonical:
                raise HostDecisionError("host decision is unavailable")
            finished = self._condition.wait_for(
                lambda: self._done,
                timeout=_TERMINAL_WAIT_SECONDS,
            )
            if not finished or self._failed or self._result is None:
                raise HostDecisionError("host decision is unavailable")
            return HostDecisionOutcome(
                context_id=context.context_id,
                agent_result=self._result,
            )

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        if type(context) is not DecisionContext:
            raise TypeError("context must be a DecisionContext")
        with self._condition:
            if self._context is not None and self._context != context:
                raise HostDecisionError("host decision is unavailable")
            self._context = context
            self._condition.notify_all()
            while self._proposal is None:
                remaining = (
                    context.expires_at - datetime.now(UTC)
                ).total_seconds()
                if not math.isfinite(remaining) or remaining <= 0:
                    return DecisionProviderResolution(
                        disposition=DecisionDisposition.PENDING
                    )
                self._condition.wait(timeout=remaining)
            return DecisionProviderResolution(
                disposition=DecisionDisposition.PROPOSAL,
                proposal=self._proposal,
            )

    def _execute(self, runner: HostDecisionRunner) -> None:
        try:
            result = runner()
            if type(result) is not AgentResult:
                raise TypeError("runner returned an invalid result")
        except Exception:
            with self._condition:
                self._failed = True
                self._done = True
                self._condition.notify_all()
            return
        with self._condition:
            self._result = result
            self._done = True
            self._condition.notify_all()

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
