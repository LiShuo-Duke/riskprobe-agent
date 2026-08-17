"""In-process delivery application facade with no HTTP or network dependency."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from riskprobe.delivery.queue import (
    ErrorClass,
    JobRequest,
    JobStatus,
    JobSummary,
    SQLiteQueue,
    _StrictDTO,
    _require_job_id,
)
from riskprobe.delivery.telemetry import (
    LocalTelemetrySink,
    TelemetryEvent,
    TelemetryEventName,
    TelemetryIntegrityError,
    TelemetryStorageError,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class JobView(_StrictDTO):
    """Safe facade projection containing no queued request or result payload."""

    job_id: str
    status: JobStatus
    attempt: int = Field(ge=0)
    request_hash: str
    result_hash: str | None = None
    error_class: ErrorClass | None = None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return _require_job_id(value)

    @field_validator("request_hash", "result_hash")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("content hash must be SHA-256")
        return value


class ApplicationResult(_StrictDTO):
    """Successful result projection exposing only its content hash."""

    job_id: str
    status: JobStatus
    attempt: int = Field(ge=1)
    content_hash: str

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return _require_job_id(value)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: JobStatus) -> JobStatus:
        if value is not JobStatus.SUCCEEDED:
            raise ValueError("application result must be succeeded")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("content_hash must be SHA-256")
        return value


class DeliveryApplication:
    """Synchronous local queue facade with best-effort telemetry observability."""

    def __init__(
        self,
        queue: SQLiteQueue,
        *,
        telemetry: LocalTelemetrySink | None = None,
    ) -> None:
        if not isinstance(queue, SQLiteQueue):
            raise TypeError("queue must be a SQLiteQueue")
        if telemetry is not None and not isinstance(telemetry, LocalTelemetrySink):
            raise TypeError("telemetry must be a LocalTelemetrySink")
        self._queue = queue
        self._telemetry = telemetry

    def submit(
        self,
        request: JobRequest,
        *,
        idempotency_key: str,
        max_attempts: int = 3,
    ) -> JobView:
        summary = self._queue.enqueue(
            request,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
        self._emit(TelemetryEventName.JOB_SUBMITTED, summary, summary.request_hash)
        return self._view(summary)

    def status(self, job_id: str) -> JobView:
        summary = self._queue.get(job_id)
        self._emit(TelemetryEventName.JOB_STATUS, summary, summary.result_hash)
        return self._view(summary)

    def result(self, job_id: str) -> ApplicationResult | None:
        completed = self._queue.completed(job_id)
        if completed is None:
            return None
        summary = completed.summary
        content_hash = completed.result.content_hash
        self._emit(TelemetryEventName.JOB_RESULT, summary, content_hash)
        return ApplicationResult.safe_validate(
            {
                "job_id": summary.job_id,
                "status": summary.status,
                "attempt": summary.attempt,
                "content_hash": content_hash,
            }
        )

    def cancel(self, job_id: str) -> JobView:
        summary = self._queue.cancel(job_id)
        self._emit(TelemetryEventName.JOB_CANCELLED, summary, summary.request_hash)
        return self._view(summary)

    def _emit(
        self,
        event_name: TelemetryEventName,
        summary: JobSummary,
        content_hash: str | None,
    ) -> None:
        if self._telemetry is None:
            return
        event = TelemetryEvent.safe_validate(
            {
                "event_name": event_name,
                "job_id": summary.job_id,
                "status": summary.status,
                "attempt": summary.attempt,
                "error_class": summary.error_class,
                "content_hash": content_hash,
            }
        )
        try:
            self._telemetry.append(event)
        except (TelemetryStorageError, TelemetryIntegrityError):
            # Queue mutations are already committed; observability is best effort.
            return

    @staticmethod
    def _view(summary: JobSummary) -> JobView:
        return JobView.safe_validate(
            {
                "job_id": summary.job_id,
                "status": summary.status,
                "attempt": summary.attempt,
                "request_hash": summary.request_hash,
                "result_hash": summary.result_hash,
                "error_class": summary.error_class,
            }
        )


__all__ = ["ApplicationResult", "DeliveryApplication", "JobView"]
