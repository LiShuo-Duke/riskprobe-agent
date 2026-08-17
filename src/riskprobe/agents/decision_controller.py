"""Transport-independent sidecar controller for bounded decision submissions."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionFinding,
    DecisionPolicy,
    DecisionProposal,
    DecisionResult,
    DecisionSource,
    ProposalValidator,
    default_decision_policy,
)
from riskprobe.agents.decision_providers import (
    DecisionProviderMode,
    _DecisionProviderBinding,
)
from riskprobe.evidence import (
    EvidenceRecord,
    EvidenceStore,
    PrivacyClass,
    assert_safe_payload as assert_safe_evidence_payload,
)
from riskprobe.monitoring.models import RiskFinding
from riskprobe.recommendations.policy import RECOMMENDATION_POLICY_VERSION
from riskprobe.tools import DiscoverResponse, InspectResponse

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_CONTEXT_KIND = "decision.context"
_PROPOSAL_KIND = "decision.proposal"
_RESULT_KIND = "decision.result"
_UNAVAILABLE_KIND = "decision.unavailable"
_DIAGNOSTIC_KIND = "diagnostic.finding"
_STORE_LOCKS: dict[str, threading.Lock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


class DecisionControllerError(RuntimeError):
    """Raised with a fixed safe message when decision state cannot be trusted."""


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class DecisionPreparation(_StrictDTO):
    """Canonical context plus its content-addressed sidecar identifier."""

    context_evidence_id: str
    context: DecisionContext

    @field_validator("context_evidence_id")
    @classmethod
    def validate_context_evidence_id(cls, value: str) -> str:
        return _validated_sha(value, "context_evidence_id")


class _DecisionUnavailableReason(StrEnum):
    PROVIDER_PENDING = "provider_pending"
    PROVIDER_ERROR = "provider_error"


class _DecisionUnavailableOutcome(_StrictDTO):
    context_evidence_id: str
    outcome_evidence_id: str
    context: DecisionContext
    reason: _DecisionUnavailableReason
    provider_binding: _DecisionProviderBinding

    @field_validator("context_evidence_id", "outcome_evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str, info: object) -> str:
        return _validated_sha(value, getattr(info, "field_name", "evidence_id"))

    @model_validator(mode="after")
    def validate_domain_binding(self) -> _DecisionUnavailableOutcome:
        if self.context_evidence_id == self.outcome_evidence_id:
            raise ValueError("decision unavailable outcome binding is invalid")
        return self


class DecisionSubmission(_StrictDTO):
    """Canonical persisted proposal/result binding returned by submit and replay."""

    context_evidence_id: str
    proposal_evidence_id: str
    result_evidence_id: str
    context: DecisionContext
    proposal: DecisionProposal
    result: DecisionResult
    provider_binding: _DecisionProviderBinding
    submitted_at: datetime

    @field_validator(
        "context_evidence_id",
        "proposal_evidence_id",
        "result_evidence_id",
    )
    @classmethod
    def validate_evidence_id(cls, value: str, info: object) -> str:
        return _validated_sha(value, getattr(info, "field_name", "evidence_id"))

    @field_validator("submitted_at")
    @classmethod
    def normalize_submitted_at(cls, value: datetime) -> datetime:
        return _normalized_utc(value)

    @model_validator(mode="after")
    def validate_domain_binding(self) -> DecisionSubmission:
        if (
            len(
                {
                    self.context_evidence_id,
                    self.proposal_evidence_id,
                    self.result_evidence_id,
                }
            )
            != 3
            or self.proposal.context_id != self.context.context_id
            or self.result.context_id != self.context.context_id
            or self.result.proposal_id != self.proposal.proposal_id
            or self.result.policy_id != self.context.policy.policy_id
            or self.result.diagnosis_evidence_ids
            != self.context.diagnosis_evidence_ids
            or self.result.source is not self.proposal.source
            or self.result.source_version != self.proposal.source_version
            or not _provider_binding_matches_proposal(
                self.provider_binding,
                self.proposal,
            )
        ):
            raise ValueError("decision submission binding is invalid")
        return self


class _StoredProposal(_StrictDTO):
    schema_version: Literal["riskprobe.stored-decision-proposal.v2"] = (
        "riskprobe.stored-decision-proposal.v2"
    )
    proposal: DecisionProposal
    provider_binding: _DecisionProviderBinding
    submitted_at: datetime

    @field_validator("submitted_at")
    @classmethod
    def normalize_submitted_at(cls, value: datetime) -> datetime:
        return _normalized_utc(value)

    @model_validator(mode="after")
    def validate_provider_binding(self) -> _StoredProposal:
        if not _provider_binding_matches_proposal(
            self.provider_binding,
            self.proposal,
        ):
            raise ValueError("stored proposal provider binding is invalid")
        return self


class _StoredUnavailable(_StrictDTO):
    schema_version: Literal["riskprobe.stored-decision-unavailable.v1"] = (
        "riskprobe.stored-decision-unavailable.v1"
    )
    context_id: str
    diagnosis_evidence_ids: tuple[str, ...]
    reason: _DecisionUnavailableReason
    provider_binding: _DecisionProviderBinding

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str) -> str:
        return _validated_sha(value, "context_id")

    @field_validator("diagnosis_evidence_ids")
    @classmethod
    def normalize_diagnosis_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_sha_sequence(
            value,
            "diagnosis_evidence_ids",
            require=True,
        )


class DecisionController:
    """Prepare authoritative contexts and persist one strict proposal per context."""

    def __init__(
        self,
        evidence_store: EvidenceStore,
        *,
        policy: DecisionPolicy | None = None,
        validator: ProposalValidator | None = None,
        clock: Callable[[], datetime] | None = None,
        version: str = "decision-controller-v1",
    ) -> None:
        if not isinstance(evidence_store, EvidenceStore):
            raise TypeError("evidence_store must be an EvidenceStore")
        selected_policy = default_decision_policy() if policy is None else policy
        selected_validator = ProposalValidator() if validator is None else validator
        if type(selected_policy) is not DecisionPolicy:
            raise TypeError("policy must be a DecisionPolicy")
        if type(selected_validator) is not ProposalValidator:
            raise TypeError("validator must be a ProposalValidator")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
            raise ValueError("version must be a public version token")
        self._evidence_store = evidence_store
        self._policy = selected_policy
        self._validator = selected_validator
        self._clock = clock or (lambda: datetime.now(UTC))
        self.version = version

    @property
    def validator_version(self) -> str:
        return self._validator.version

    def prepare(
        self,
        *,
        session_id: str,
        attempt: int,
        anchor_node_id: str,
        diagnosis_evidence_ids: Sequence[str],
        inspect_response: InspectResponse,
        discover_response: DiscoverResponse,
        orchestrator_version: str,
        planner_version: str,
    ) -> DecisionPreparation:
        """Persist one canonical context after rereading the complete diagnosis set."""

        try:
            inspect = _revalidate_model(InspectResponse, inspect_response)
            discover = _revalidate_model(DiscoverResponse, discover_response)
            if inspect.dataset_id != discover.dataset_id:
                raise ValueError("response datasets do not match")
            diagnosis_ids = _validated_sha_sequence(
                diagnosis_evidence_ids,
                "diagnosis_evidence_ids",
                require=True,
            )
            findings, diagnostics_version = self._diagnosis_for_run(
                run_id=session_id,
                dataset_id=inspect.dataset_id,
                diagnosis_evidence_ids=diagnosis_ids,
            )
            if _VERSION.fullmatch(orchestrator_version) is None or _VERSION.fullmatch(
                planner_version
            ) is None:
                raise ValueError("component versions are invalid")
            issued_at = self._now()
            context = DecisionContext(
                session_id=session_id,
                attempt=attempt,
                anchor_node_id=anchor_node_id,
                dataset_id=inspect.dataset_id,
                objective="comprehensive",
                metadata_grade=inspect.metadata_grade,
                row_count=inspect.row_count,
                feature_count=inspect.feature_count,
                issue_codes=inspect.issue_codes,
                rule_ids=discover.rule_ids,
                diagnosis_evidence_ids=diagnosis_ids,
                findings=findings,
                policy=self._policy,
                issued_at=issued_at,
                expires_at=issued_at
                + timedelta(seconds=self._policy.context_ttl_seconds),
                component_versions={
                    "diagnostics": diagnostics_version,
                    "orchestrator": orchestrator_version,
                    "planner": planner_version,
                    "recommendations": RECOMMENDATION_POLICY_VERSION,
                },
            )
            with self._exclusive_store_lock():
                self._require_new_context(context)
                context_evidence_id = self._evidence_store.append(
                    EvidenceRecord(
                        run_id=session_id,
                        kind=_CONTEXT_KIND,
                        payload=context.model_dump(mode="json"),
                        parent_ids=context.diagnosis_evidence_ids,
                        producer_version=self.version,
                    )
                )
            return DecisionPreparation(
                context_evidence_id=context_evidence_id,
                context=context,
            )
        except DecisionControllerError:
            raise
        except Exception as error:
            raise DecisionControllerError("decision context is unavailable") from error

    def submit(
        self,
        *,
        context_evidence_id: str,
        proposal: DecisionProposal,
        provider_binding: _DecisionProviderBinding,
    ) -> DecisionSubmission:
        """Validate against an authoritative stored context and persist the result."""

        context = self._load_context(context_evidence_id)
        try:
            canonical_proposal = _revalidate_model(DecisionProposal, proposal)
            canonical_binding = _revalidate_model(
                _DecisionProviderBinding,
                provider_binding,
            )
            if not _provider_binding_matches_proposal(
                canonical_binding,
                canonical_proposal,
            ):
                raise ValueError("proposal provider binding is invalid")
        except Exception as error:
            raise DecisionControllerError("decision proposal is unavailable") from error
        try:
            submitted_at = self._now()
            result = self._validator.validate(
                context,
                canonical_proposal,
                now=submitted_at,
            )
            stored_proposal = _StoredProposal(
                proposal=canonical_proposal,
                provider_binding=canonical_binding,
                submitted_at=submitted_at,
            )
            with self._exclusive_store_lock():
                self._require_no_submission(
                    context_evidence_id=context_evidence_id,
                    context_id=context.context_id,
                    run_id=context.session_id,
                )
                proposal_record = EvidenceRecord(
                    run_id=context.session_id,
                    kind=_PROPOSAL_KIND,
                    payload=stored_proposal.model_dump(mode="json"),
                    parent_ids=(context_evidence_id,),
                    producer_version=self.version,
                )
                proposal_evidence_id = EvidenceStore.content_id(proposal_record)
                result_record = EvidenceRecord(
                    run_id=context.session_id,
                    kind=_RESULT_KIND,
                    payload=result.model_dump(mode="json"),
                    parent_ids=(context_evidence_id, proposal_evidence_id),
                    producer_version=self.version,
                )
                persisted_ids = self._evidence_store.append_many(
                    (proposal_record, result_record)
                )
                if persisted_ids[0] != proposal_evidence_id:
                    raise ValueError("proposal evidence identity is invalid")
                proposal_evidence_id, result_evidence_id = persisted_ids
            return DecisionSubmission(
                context_evidence_id=context_evidence_id,
                proposal_evidence_id=proposal_evidence_id,
                result_evidence_id=result_evidence_id,
                context=context,
                proposal=canonical_proposal,
                result=result,
                provider_binding=canonical_binding,
                submitted_at=submitted_at,
            )
        except DecisionControllerError:
            raise
        except Exception as error:
            raise DecisionControllerError("decision submission is unavailable") from error

    def record_unavailable(
        self,
        *,
        context_evidence_id: str,
        reason: _DecisionUnavailableReason,
        provider_binding: _DecisionProviderBinding,
    ) -> _DecisionUnavailableOutcome:
        """Persist a provider-unavailable outcome without fabricating a proposal."""

        context = self._load_context(context_evidence_id)
        try:
            if type(reason) is not _DecisionUnavailableReason:
                raise TypeError("reason must be a decision unavailable reason")
            canonical_binding = _revalidate_model(
                _DecisionProviderBinding,
                provider_binding,
            )
            stored = _StoredUnavailable(
                context_id=context.context_id,
                diagnosis_evidence_ids=context.diagnosis_evidence_ids,
                reason=reason,
                provider_binding=canonical_binding,
            )
            with self._exclusive_store_lock():
                self._require_no_submission(
                    context_evidence_id=context_evidence_id,
                    context_id=context.context_id,
                    run_id=context.session_id,
                )
                outcome_evidence_id = self._evidence_store.append(
                    EvidenceRecord(
                        run_id=context.session_id,
                        kind=_UNAVAILABLE_KIND,
                        payload=stored.model_dump(mode="json"),
                        parent_ids=(context_evidence_id,),
                        producer_version=self.version,
                    )
                )
            return _DecisionUnavailableOutcome(
                context_evidence_id=context_evidence_id,
                outcome_evidence_id=outcome_evidence_id,
                context=context,
                reason=reason,
                provider_binding=canonical_binding,
            )
        except DecisionControllerError:
            raise
        except Exception as error:
            raise DecisionControllerError(
                "decision unavailable outcome is unavailable"
            ) from error

    def replay_unavailable(
        self,
        *,
        outcome_evidence_id: str,
        expected_run_id: str | None = None,
    ) -> _DecisionUnavailableOutcome:
        """Strictly replay one persisted provider-unavailable outcome."""

        try:
            outcome_record = self._record(
                outcome_evidence_id,
                kind=_UNAVAILABLE_KIND,
                expected_run_id=expected_run_id,
            )
            if len(outcome_record.parent_ids) != 1:
                raise ValueError("unavailable outcome parent graph is invalid")
            context_evidence_id = outcome_record.parent_ids[0]
            context = self._load_context(
                context_evidence_id,
                expected_run_id=outcome_record.run_id,
            )
            stored = _model_from_payload(
                _StoredUnavailable,
                outcome_record.payload,
            )
            if (
                stored.context_id != context.context_id
                or stored.diagnosis_evidence_ids
                != context.diagnosis_evidence_ids
            ):
                raise ValueError("unavailable outcome binding is invalid")
            self._require_exact_unavailable(
                context_evidence_id=context_evidence_id,
                context_id=context.context_id,
                outcome_evidence_id=outcome_evidence_id,
                run_id=outcome_record.run_id,
            )
            return _DecisionUnavailableOutcome(
                context_evidence_id=context_evidence_id,
                outcome_evidence_id=outcome_evidence_id,
                context=context,
                reason=stored.reason,
                provider_binding=stored.provider_binding,
            )
        except DecisionControllerError:
            raise
        except Exception as error:
            raise DecisionControllerError("decision audit is unavailable") from error

    def replay(
        self,
        *,
        result_evidence_id: str,
        expected_run_id: str | None = None,
    ) -> DecisionSubmission:
        """Revalidate persisted binding at authoritative submission time, never wall time."""

        try:
            result_record = self._record(
                result_evidence_id,
                kind=_RESULT_KIND,
                expected_run_id=expected_run_id,
            )
            if len(result_record.parent_ids) != 2:
                raise ValueError("result parent graph is invalid")
            context_evidence_id, proposal_evidence_id = result_record.parent_ids
            context = self._load_context(
                context_evidence_id,
                expected_run_id=result_record.run_id,
            )
            proposal_record = self._record(
                proposal_evidence_id,
                kind=_PROPOSAL_KIND,
                expected_run_id=result_record.run_id,
            )
            if proposal_record.parent_ids != (context_evidence_id,):
                raise ValueError("proposal parent graph is invalid")
            stored_proposal = _model_from_payload(
                _StoredProposal,
                proposal_record.payload,
            )
            result = _model_from_payload(DecisionResult, result_record.payload)
            expected_result = self._validator.validate(
                context,
                stored_proposal.proposal,
                now=stored_proposal.submitted_at,
            )
            if result != expected_result:
                raise ValueError("persisted result is not canonical")
            self._require_exact_submission(
                context_evidence_id=context_evidence_id,
                context_id=context.context_id,
                proposal_evidence_id=proposal_evidence_id,
                result_evidence_id=result_evidence_id,
                run_id=result_record.run_id,
            )
            return DecisionSubmission(
                context_evidence_id=context_evidence_id,
                proposal_evidence_id=proposal_evidence_id,
                result_evidence_id=result_evidence_id,
                context=context,
                proposal=stored_proposal.proposal,
                result=result,
                provider_binding=stored_proposal.provider_binding,
                submitted_at=stored_proposal.submitted_at,
            )
        except DecisionControllerError:
            raise
        except Exception as error:
            raise DecisionControllerError("decision audit is unavailable") from error

    def _load_context(
        self,
        context_evidence_id: str,
        *,
        expected_run_id: str | None = None,
    ) -> DecisionContext:
        try:
            record = self._record(
                context_evidence_id,
                kind=_CONTEXT_KIND,
                expected_run_id=expected_run_id,
            )
            context = _model_from_payload(DecisionContext, record.payload)
            if (
                record.run_id != context.session_id
                or record.parent_ids != context.diagnosis_evidence_ids
                or context.policy != self._policy
                or context.component_versions["recommendations"]
                != RECOMMENDATION_POLICY_VERSION
            ):
                raise ValueError("context record binding is invalid")
            findings, diagnostics_version = self._diagnosis_for_run(
                run_id=context.session_id,
                dataset_id=context.dataset_id,
                diagnosis_evidence_ids=context.diagnosis_evidence_ids,
            )
            if (
                findings != context.findings
                or context.component_versions["diagnostics"] != diagnostics_version
            ):
                raise ValueError("context diagnosis binding is invalid")
            return context
        except DecisionControllerError:
            raise
        except Exception as error:
            raise DecisionControllerError("decision context is unavailable") from error

    def _record(
        self,
        evidence_id: str,
        *,
        kind: str,
        expected_run_id: str | None,
    ) -> EvidenceRecord:
        _validated_sha(evidence_id, "evidence_id")
        record = self._evidence_store.get(evidence_id)
        if (
            not isinstance(record, EvidenceRecord)
            or EvidenceStore.content_id(record) != evidence_id
            or record.kind != kind
            or record.producer_version != self.version
            or (expected_run_id is not None and record.run_id != expected_run_id)
        ):
            raise ValueError("evidence record binding is invalid")
        return record

    def _diagnosis_for_run(
        self,
        *,
        run_id: str,
        dataset_id: str,
        diagnosis_evidence_ids: tuple[str, ...],
    ) -> tuple[tuple[DecisionFinding, ...], str]:
        if self._evidence_store.verify_chain(run_id) is not True:
            raise ValueError("diagnosis evidence chain is invalid")
        records = self._evidence_store.list_run(run_id)
        diagnostic_records = {
            EvidenceStore.content_id(record): record
            for record in records
            if record.kind == _DIAGNOSTIC_KIND
        }
        if tuple(sorted(diagnostic_records)) != diagnosis_evidence_ids:
            raise ValueError("diagnosis evidence set is incomplete")

        findings: list[DecisionFinding] = []
        finding_ids: set[str] = set()
        producer_versions: set[str] = set()
        for evidence_id in diagnosis_evidence_ids:
            record = self._evidence_store.get(evidence_id)
            if (
                not isinstance(record, EvidenceRecord)
                or EvidenceStore.content_id(record) != evidence_id
                or record.run_id != run_id
                or record.kind != _DIAGNOSTIC_KIND
                or record.privacy_class is not PrivacyClass.AGGREGATE
                or record.parent_ids
                or record.payload.get("dataset_id") != dataset_id
                or _VERSION.fullmatch(record.producer_version) is None
            ):
                raise ValueError("diagnosis evidence binding is invalid")
            payload = dict(record.payload)
            assert_safe_evidence_payload(payload)
            payload.pop("dataset_id", None)
            finding = _model_from_payload(RiskFinding, payload)
            if finding.finding_id in finding_ids:
                raise ValueError("diagnosis finding is duplicated")
            finding_ids.add(finding.finding_id)
            producer_versions.add(record.producer_version)
            findings.append(
                DecisionFinding(
                    evidence_id=evidence_id,
                    finding=finding,
                )
            )
        if len(producer_versions) != 1:
            raise ValueError("diagnosis producer version is ambiguous")
        return (
            tuple(sorted(findings, key=lambda item: item.evidence_id)),
            next(iter(producer_versions)),
        )

    def _require_new_context(self, context: DecisionContext) -> None:
        for record in self._evidence_store.list_run(context.session_id):
            if record.kind != _CONTEXT_KIND:
                continue
            stored = _model_from_payload(DecisionContext, record.payload)
            if (
                stored.session_id == context.session_id
                and stored.attempt == context.attempt
                and stored.anchor_node_id == context.anchor_node_id
            ):
                raise DecisionControllerError("decision context already exists")

    def _decision_records_for_context(
        self,
        *,
        context_evidence_id: str,
        context_id: str,
        run_id: str,
    ) -> tuple[tuple[str, EvidenceRecord], ...]:
        records: list[tuple[str, EvidenceRecord]] = []
        for record in self._evidence_store.list_run(run_id):
            if _record_targets_context(
                record,
                context_evidence_id=context_evidence_id,
                context_id=context_id,
            ):
                records.append((EvidenceStore.content_id(record), record))
        return tuple(records)

    def _require_no_submission(
        self,
        *,
        context_evidence_id: str,
        context_id: str,
        run_id: str,
    ) -> None:
        if self._decision_records_for_context(
            context_evidence_id=context_evidence_id,
            context_id=context_id,
            run_id=run_id,
        ):
            raise DecisionControllerError("decision submission already exists")

    def _require_exact_submission(
        self,
        *,
        context_evidence_id: str,
        context_id: str,
        proposal_evidence_id: str,
        result_evidence_id: str,
        run_id: str,
    ) -> None:
        records = self._decision_records_for_context(
            context_evidence_id=context_evidence_id,
            context_id=context_id,
            run_id=run_id,
        )
        proposal_ids = [
            evidence_id
            for evidence_id, record in records
            if record.kind == _PROPOSAL_KIND
        ]
        result_ids = [
            evidence_id
            for evidence_id, record in records
            if record.kind == _RESULT_KIND
        ]
        unavailable_ids = [
            evidence_id
            for evidence_id, record in records
            if record.kind == _UNAVAILABLE_KIND
        ]
        if (
            proposal_ids != [proposal_evidence_id]
            or result_ids != [result_evidence_id]
            or unavailable_ids
        ):
            raise ValueError("decision submission is duplicated")

    def _require_exact_unavailable(
        self,
        *,
        context_evidence_id: str,
        context_id: str,
        outcome_evidence_id: str,
        run_id: str,
    ) -> None:
        records = self._decision_records_for_context(
            context_evidence_id=context_evidence_id,
            context_id=context_id,
            run_id=run_id,
        )
        proposal_ids = [
            evidence_id
            for evidence_id, record in records
            if record.kind == _PROPOSAL_KIND
        ]
        result_ids = [
            evidence_id
            for evidence_id, record in records
            if record.kind == _RESULT_KIND
        ]
        unavailable_ids = [
            evidence_id
            for evidence_id, record in records
            if record.kind == _UNAVAILABLE_KIND
        ]
        if (
            proposal_ids
            or result_ids
            or unavailable_ids != [outcome_evidence_id]
        ):
            raise ValueError("decision unavailable outcome is duplicated")

    @contextmanager
    def _exclusive_store_lock(self) -> Iterator[None]:
        store_path = self._evidence_store.path.absolute()
        lock_path = store_path.with_name(f".{store_path.name}.decision.lock")
        lock_key = str(lock_path)
        with _STORE_LOCKS_GUARD:
            process_lock = _STORE_LOCKS.setdefault(lock_key, threading.Lock())
        with process_lock:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.fchmod(descriptor, 0o600)
                details = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_uid != os.geteuid()
                ):
                    raise DecisionControllerError("decision state is unavailable")
                with os.fdopen(descriptor, "r+b") as handle:
                    descriptor = -1
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    def _now(self) -> datetime:
        return _normalized_utc(self._clock())


def _revalidate_model(model_type: type[BaseModel], value: object):
    if type(value) is not model_type:
        raise TypeError(f"value must be a {model_type.__name__}")
    return model_type.model_validate(value.model_dump(mode="python"))


def _model_from_payload(model_type: type[BaseModel], payload: object):
    jsonable = dict(payload) if isinstance(payload, Mapping) else payload
    return model_type.model_validate_json(
        json.dumps(
            jsonable,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _record_targets_context(
    record: EvidenceRecord,
    *,
    context_evidence_id: str,
    context_id: str,
) -> bool:
    if record.kind not in {
        _PROPOSAL_KIND,
        _RESULT_KIND,
        _UNAVAILABLE_KIND,
    }:
        return False
    if context_evidence_id in record.parent_ids:
        return True
    if record.payload.get("context_id") == context_id:
        return True
    proposal = record.payload.get("proposal")
    return (
        isinstance(proposal, Mapping)
        and proposal.get("context_id") == context_id
    )


def _provider_binding_matches_proposal(
    binding: _DecisionProviderBinding,
    proposal: DecisionProposal,
) -> bool:
    expected_source = (
        DecisionSource.DETERMINISTIC
        if binding.selected.mode is DecisionProviderMode.DETERMINISTIC
        else DecisionSource.EXTERNAL_HOST
    )
    return (
        binding.selected.mode is not DecisionProviderMode.DISABLED
        and proposal.source is expected_source
        and proposal.source_version == binding.selected.version
    )


def _normalized_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("decision time must be timezone-aware")
    return value.astimezone(UTC)


def _validated_sha(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a SHA-256 identifier")
    return value


def _validated_sha_sequence(
    values: Sequence[str],
    field_name: str,
    *,
    require: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence")
    normalized = tuple(values)
    if (require and not normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique SHA-256 identifiers")
    for item in normalized:
        _validated_sha(item, field_name)
    return tuple(sorted(normalized))


__all__ = [
    "DecisionController",
    "DecisionControllerError",
    "DecisionPreparation",
    "DecisionSubmission",
]
