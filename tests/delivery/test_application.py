import socket
import threading
from pathlib import Path

import pytest

from riskprobe.delivery import (
    DeliveryApplication,
    IdempotencyConflictError,
    JobNotFoundError,
    JobRequest,
    JobResult,
    JobStatus,
    LocalTelemetrySink,
    SQLiteQueue,
    TelemetryIntegrityError,
    TelemetryStorageError,
)


def _request(*, kind: str = "profile", count: int = 4) -> JobRequest:
    return JobRequest(
        kind=kind,
        payload={"aggregate_count": count, "dataset_hash": "b" * 64},
    )


def test_application_duplicate_submit_returns_same_safe_job_view(tmp_path: Path) -> None:
    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    application = DeliveryApplication(queue)
    request = _request()

    first = application.submit(request, idempotency_key="application-001")
    duplicate = application.submit(request, idempotency_key="application-001")

    assert duplicate == first
    assert first.status is JobStatus.PENDING
    assert set(first.model_dump(mode="json")) == {
        "attempt",
        "error_class",
        "job_id",
        "request_hash",
        "result_hash",
        "status",
    }
    assert "payload" not in first.model_dump_json()
    assert "aggregate_count" not in first.model_dump_json()


def test_application_key_collision_and_unknown_job_errors_are_fixed(tmp_path: Path) -> None:
    application = DeliveryApplication(SQLiteQueue(tmp_path / "queue.sqlite3"))
    application.submit(_request(), idempotency_key="application-collision")

    with pytest.raises(IdempotencyConflictError) as collision:
        application.submit(
            _request(kind="report"),
            idempotency_key="application-collision",
        )
    assert str(collision.value) == "idempotency key conflicts with existing request"

    unknown_id = "job-" + "0" * 32
    with pytest.raises(JobNotFoundError) as unavailable:
        application.status(unknown_id)
    assert str(unavailable.value) == "job is unavailable"
    assert unknown_id not in str(unavailable.value)


def test_application_status_result_and_cancel_never_return_request_payload(
    tmp_path: Path,
) -> None:
    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    application = DeliveryApplication(queue)
    submitted = application.submit(_request(), idempotency_key="application-result")

    assert application.result(submitted.job_id) is None
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    result = JobResult(summary={"aggregate_count": 4, "approval_rate": 0.8})
    queue.succeed(claimed.lease, result)

    status = application.status(submitted.job_id)
    application_result = application.result(submitted.job_id)

    assert status.status is JobStatus.SUCCEEDED
    assert status.result_hash == result.content_hash
    assert application_result is not None
    assert application_result.job_id == submitted.job_id
    assert application_result.status is JobStatus.SUCCEEDED
    assert application_result.attempt == 1
    assert application_result.content_hash == result.content_hash
    assert set(application_result.model_dump(mode="json")) == {
        "attempt",
        "content_hash",
        "job_id",
        "status",
    }
    assert "aggregate_count" not in application_result.model_dump_json()

    cancellable = application.submit(_request(count=5), idempotency_key="application-cancel")
    cancelled = application.cancel(cancellable.job_id)
    assert cancelled.status is JobStatus.CANCELLED
    assert application.status(cancellable.job_id) == cancelled
    assert application.result(cancellable.job_id) is None


def test_complete_local_workflow_creates_no_network_connection_or_background_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", blocked_network)
    monkeypatch.setattr(socket.socket, "connect", blocked_network)
    before_threads = {(thread.ident, thread.name) for thread in threading.enumerate()}

    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    telemetry = LocalTelemetrySink(tmp_path / "telemetry.sqlite3")
    application = DeliveryApplication(queue, telemetry=telemetry)
    submitted = application.submit(_request(), idempotency_key="offline-001")
    claimed = queue.claim(owner="offline-worker", lease_seconds=30)
    assert claimed is not None
    renewed = queue.heartbeat(claimed.lease, lease_seconds=30)
    queue.succeed(renewed, JobResult(summary={"aggregate_count": 4}))

    assert application.status(submitted.job_id).status is JobStatus.SUCCEEDED
    assert application.result(submitted.job_id) is not None
    assert telemetry.list()
    assert {(thread.ident, thread.name) for thread in threading.enumerate()} == before_threads


def test_application_result_uses_one_atomic_queue_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    application = DeliveryApplication(queue)
    submitted = application.submit(_request(), idempotency_key="atomic-result")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    expected = JobResult(summary={"aggregate_count": 4})
    queue.succeed(claimed.lease, expected)

    def forbid_separate_status_read(job_id: str) -> None:
        raise AssertionError("result must not perform a separate status read")

    monkeypatch.setattr(queue, "get", forbid_separate_status_read)

    result = application.result(submitted.job_id)
    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.content_hash == expected.content_hash


@pytest.mark.parametrize(
    "telemetry_error_type",
    [
        pytest.param(TelemetryStorageError, id="storage"),
        pytest.param(TelemetryIntegrityError, id="integrity"),
    ],
)
def test_application_telemetry_failures_are_best_effort_after_queue_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry_error_type: type[RuntimeError],
) -> None:
    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    telemetry = LocalTelemetrySink(tmp_path / "telemetry.sqlite3")
    application = DeliveryApplication(queue, telemetry=telemetry)

    def unavailable_telemetry(event: object) -> None:
        raise telemetry_error_type("telemetry fixed failure")

    monkeypatch.setattr(telemetry, "append", unavailable_telemetry)

    submitted = application.submit(
        _request(count=1),
        idempotency_key="best-effort-cancel",
    )
    assert queue.get(submitted.job_id).status is JobStatus.PENDING
    assert application.status(submitted.job_id).status is JobStatus.PENDING
    cancelled = application.cancel(submitted.job_id)
    assert cancelled.status is JobStatus.CANCELLED
    assert queue.get(submitted.job_id).status is JobStatus.CANCELLED

    result_job = application.submit(
        _request(count=2),
        idempotency_key="best-effort-result",
    )
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    assert claimed.job_id == result_job.job_id
    expected = JobResult(summary={"aggregate_count": 2})
    queue.succeed(claimed.lease, expected)

    result = application.result(result_job.job_id)
    assert result is not None
    assert result.content_hash == expected.content_hash
    assert queue.get(result_job.job_id).status is JobStatus.SUCCEEDED


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_application_does_not_swallow_unexpected_telemetry_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    telemetry = LocalTelemetrySink(tmp_path / "telemetry.sqlite3")
    application = DeliveryApplication(queue, telemetry=telemetry)
    request = _request()

    def broken_telemetry(event: object) -> None:
        raise failure_type("telemetry programming failure")

    monkeypatch.setattr(telemetry, "append", broken_telemetry)

    with pytest.raises(failure_type, match="telemetry programming failure"):
        application.submit(request, idempotency_key="unexpected-telemetry")

    persisted = queue.enqueue(request, idempotency_key="unexpected-telemetry")
    assert persisted.status is JobStatus.PENDING


def test_application_maps_invalid_persisted_result_projection_to_integrity_error(
    tmp_path: Path,
) -> None:
    import sqlite3
    import traceback

    from riskprobe.delivery import QueueIntegrityError

    queue = SQLiteQueue(tmp_path / "queue.sqlite3")
    application = DeliveryApplication(queue)
    submitted = application.submit(_request(), idempotency_key="invalid-projection")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    queue.succeed(claimed.lease, JobResult(summary={"aggregate_count": 4}))
    with sqlite3.connect(queue.path) as connection:
        connection.execute(
            "UPDATE jobs SET attempt = ? WHERE job_id = ?",
            (0, submitted.job_id),
        )

    with pytest.raises(QueueIntegrityError) as rejected:
        application.result(submitted.job_id)

    formatted = "".join(
        traceback.format_exception(
            type(rejected.value),
            rejected.value,
            rejected.value.__traceback__,
        )
    )
    assert str(rejected.value) == "queue integrity check failed"
    assert rejected.value.__cause__ is None
    assert "greater than or equal to 1" not in formatted
