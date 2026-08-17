"""Owner-only atomic persistence for terminal agent results."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from riskprobe.agents.contracts import (
    AgentResult,
    AgentState,
    AgentStatus,
    ExecutionPlan,
    PlanStep,
    ReviewDecision,
    ReviewReason,
)
from riskprobe.tools.models import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
)

_FORMAT_VERSION = "riskprobe.agent-result.v1"
_MAX_RESULT_BYTES = 1_048_576
_REQUEST_TYPES = {
    "inspect": InspectRequest,
    "diagnose": DiagnoseRequest,
    "discover": DiscoverRequest,
    "recommend": RecommendRequest,
}


class AgentResultIntegrityError(RuntimeError):
    """Raised when a persisted terminal result is missing required integrity."""


class AgentResultStore:
    """Persist one canonical terminal result without mutating session history."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except OSError as error:
            raise AgentResultIntegrityError("agent result is unavailable") from error
        handle = os.fdopen(descriptor, "r+b")
        try:
            details = os.fstat(handle.fileno())
            if not _is_private_regular(details):
                raise AgentResultIntegrityError("agent result is unavailable")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if not handle.closed:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    def load(self) -> AgentResult | None:
        try:
            path_details = self.path.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise AgentResultIntegrityError("agent result is unavailable") from error
        if not _is_private_regular(path_details):
            raise AgentResultIntegrityError("agent result is unavailable")

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
            with os.fdopen(descriptor, "rb") as handle:
                descriptor_details = os.fstat(handle.fileno())
                if (
                    not _is_private_regular(descriptor_details)
                    or (path_details.st_dev, path_details.st_ino)
                    != (descriptor_details.st_dev, descriptor_details.st_ino)
                    or descriptor_details.st_size > _MAX_RESULT_BYTES
                ):
                    raise AgentResultIntegrityError("agent result is unavailable")
                content = handle.read(_MAX_RESULT_BYTES + 1)
        except AgentResultIntegrityError:
            raise
        except OSError as error:
            raise AgentResultIntegrityError("agent result is unavailable") from error
        if len(content) > _MAX_RESULT_BYTES:
            raise AgentResultIntegrityError("agent result is unavailable")
        return _decode_result(content)

    def publish(self, result: AgentResult) -> None:
        if type(result) is not AgentResult:
            raise TypeError("result must be an AgentResult")
        existing = self.load()
        if existing is not None:
            if existing != result:
                raise AgentResultIntegrityError("agent result is unavailable")
            return

        result_payload = result.model_dump(mode="json")
        result_json = _canonical_json(result_payload)
        envelope = {
            "format_version": _FORMAT_VERSION,
            "result": result_payload,
            "result_sha256": hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
        }
        content = f"{_canonical_json(envelope)}\n".encode("utf-8")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, self.path, follow_symlinks=False)
            except FileExistsError:
                existing = self.load()
                if existing != result:
                    raise AgentResultIntegrityError("agent result is unavailable")
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except AgentResultIntegrityError:
            raise
        except OSError as error:
            raise AgentResultIntegrityError("agent result is unavailable") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _reconstruct_result(payload: dict[str, object]) -> AgentResult:
    expected_result_fields = {
        "session_id",
        "status",
        "plan",
        "review",
        "tool_sequence",
        "evidence_ids",
        "diagnosis_evidence_ids",
        "retry_count",
        "state_history",
        "leaf_node_id",
        "redacted_summary",
    }
    if set(payload) != expected_result_fields:
        raise ValueError("invalid result fields")
    plan_payload = payload["plan"]
    if type(plan_payload) is not dict or set(plan_payload) != {
        "objective",
        "dataset_id",
        "steps",
        "component_versions",
    }:
        raise ValueError("invalid plan fields")
    raw_steps = plan_payload["steps"]
    if type(raw_steps) is not list:
        raise ValueError("invalid plan steps")
    steps: list[PlanStep] = []
    for raw_step in raw_steps:
        if type(raw_step) is not dict or set(raw_step) != {
            "step_id",
            "tool_name",
            "request",
            "requires_evidence",
            "production_action",
        }:
            raise ValueError("invalid plan step fields")
        tool_name = raw_step["tool_name"]
        request_payload = raw_step["request"]
        if tool_name == "review":
            if request_payload is not None:
                raise ValueError("invalid review request")
            request = None
        else:
            request_type = _REQUEST_TYPES.get(tool_name)
            if request_type is None or type(request_payload) is not dict:
                raise ValueError("invalid tool request")
            request = request_type.model_validate_json(
                _canonical_json(request_payload)
            )
        steps.append(
            PlanStep(
                step_id=raw_step["step_id"],
                tool_name=tool_name,
                request=request,
                requires_evidence=raw_step["requires_evidence"],
                production_action=raw_step["production_action"],
            )
        )
    plan = ExecutionPlan(
        objective=plan_payload["objective"],
        dataset_id=plan_payload["dataset_id"],
        steps=tuple(steps),
        component_versions=plan_payload["component_versions"],
    )

    review_payload = payload["review"]
    if type(review_payload) is not dict or set(review_payload) != {
        "approved",
        "reason_codes",
        "evidence_ids",
        "retry_allowed",
    }:
        raise ValueError("invalid review fields")
    reason_codes = review_payload["reason_codes"]
    review_evidence = review_payload["evidence_ids"]
    if type(reason_codes) is not list or type(review_evidence) is not list:
        raise ValueError("invalid review sequences")
    review = ReviewDecision(
        approved=review_payload["approved"],
        reason_codes=tuple(ReviewReason(reason) for reason in reason_codes),
        evidence_ids=tuple(review_evidence),
        retry_allowed=review_payload["retry_allowed"],
    )

    tool_sequence = payload["tool_sequence"]
    evidence_ids = payload["evidence_ids"]
    diagnosis_ids = payload["diagnosis_evidence_ids"]
    state_history = payload["state_history"]
    if any(
        type(value) is not list
        for value in (tool_sequence, evidence_ids, diagnosis_ids, state_history)
    ):
        raise ValueError("invalid result sequences")
    return AgentResult(
        session_id=payload["session_id"],
        status=AgentStatus(payload["status"]),
        plan=plan,
        review=review,
        tool_sequence=tuple(tool_sequence),
        evidence_ids=tuple(evidence_ids),
        diagnosis_evidence_ids=tuple(diagnosis_ids),
        retry_count=payload["retry_count"],
        state_history=tuple(AgentState(state) for state in state_history),
        leaf_node_id=payload["leaf_node_id"],
        redacted_summary=payload["redacted_summary"],
    )


def _decode_result(content: bytes) -> AgentResult:
    try:
        envelope = json.loads(content)
        if (
            type(envelope) is not dict
            or set(envelope) != {"format_version", "result", "result_sha256"}
            or envelope["format_version"] != _FORMAT_VERSION
            or type(envelope["result"]) is not dict
            or not isinstance(envelope["result_sha256"], str)
        ):
            raise ValueError("invalid result envelope")
        if content != f"{_canonical_json(envelope)}\n".encode("utf-8"):
            raise ValueError("non-canonical result envelope")
        result_json = _canonical_json(envelope["result"])
        if hashlib.sha256(result_json.encode("utf-8")).hexdigest() != envelope[
            "result_sha256"
        ]:
            raise ValueError("result digest mismatch")
        result = _reconstruct_result(envelope["result"])
        if result.model_dump(mode="json") != envelope["result"]:
            raise ValueError("result projection mismatch")
        return result
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        raise AgentResultIntegrityError("agent result is unavailable") from error


def _is_private_regular(details: os.stat_result) -> bool:
    return (
        stat.S_ISREG(details.st_mode)
        and details.st_uid == os.geteuid()
        and stat.S_IMODE(details.st_mode) == 0o600
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _jsonable(value: object) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


__all__ = ["AgentResultIntegrityError", "AgentResultStore"]
