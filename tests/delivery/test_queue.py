import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from riskprobe.delivery import (
    ErrorClass,
    IdempotencyConflictError,
    InvalidJobTransitionError,
    JobFailure,
    JobLease,
    JobRequest,
    JobResult,
    JobStatus,
    LeaseConflictError,
    QueueStorageError,
    SQLiteQueue,
)


class MutableClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def _request(*, count: int = 3, kind: str = "profile") -> JobRequest:
    return JobRequest(
        kind=kind,
        payload={"aggregate_count": count, "dataset_hash": "a" * 64},
    )


def _queue(tmp_path: Path, *, clock: MutableClock | None = None) -> SQLiteQueue:
    return SQLiteQueue(tmp_path / "delivery.sqlite3", clock=clock or MutableClock())


def test_job_request_is_strict_deeply_immutable_and_privacy_safe() -> None:
    source = {
        "aggregate_count": 3,
        "metrics": {"approved_count": 2},
    }
    request = JobRequest(kind="profile", payload=source)
    source["aggregate_count"] = 99
    source["metrics"]["approved_count"] = 99

    assert request.payload["aggregate_count"] == 3
    assert request.payload["metrics"] == {"approved_count": 2}
    with pytest.raises(TypeError):
        request.payload["aggregate_count"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        request.payload["metrics"]["approved_count"] = 4  # type: ignore[index]
    with pytest.raises(ValidationError):
        request.kind = "report"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        JobRequest(kind="profile", payload={}, raw_rows=())  # type: ignore[call-arg]

    unsafe_path = "/private/customers/data.parquet"
    with pytest.raises(ValidationError) as path_error:
        JobRequest(kind="profile", payload={"source": unsafe_path})
    assert unsafe_path not in str(path_error.value)

    with pytest.raises(ValidationError):
        JobRequest(kind="profile", payload={"segment": "premium customers"})
    with pytest.raises(ValidationError):
        JobRequest(
            kind="profile",
            payload={"raw_rows": [{"customer_id": "customer-123456"}]},
        )


def test_enqueue_is_idempotent_and_key_collision_fails_closed(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    request = _request()

    first = queue.enqueue(request, idempotency_key="submission-001", max_attempts=2)
    duplicate = queue.enqueue(request, idempotency_key="submission-001", max_attempts=2)

    assert duplicate == first
    assert first.status is JobStatus.PENDING
    assert first.attempt == 0
    assert "payload" not in first.model_dump(mode="json")

    with pytest.raises(IdempotencyConflictError) as collision:
        queue.enqueue(_request(count=4), idempotency_key="submission-001", max_attempts=2)
    assert str(collision.value) == "idempotency key conflicts with existing request"

    with pytest.raises(IdempotencyConflictError):
        queue.enqueue(request, idempotency_key="submission-001", max_attempts=3)


def test_claim_is_fifo_and_concurrent_claim_never_duplicates_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    submitted = tuple(
        queue.enqueue(
            _request(count=index),
            idempotency_key=f"fifo-{index}",
        )
        for index in range(3)
    )

    claimed = tuple(
        queue.claim(owner=f"fifo-worker-{index}", lease_seconds=60)
        for index in range(3)
    )

    assert tuple(item.job_id for item in claimed if item is not None) == tuple(
        item.job_id for item in submitted
    )

    second_queue = SQLiteQueue(tmp_path / "concurrent.sqlite3")
    single = second_queue.enqueue(_request(), idempotency_key="concurrent-001")
    barrier = Barrier(8)

    def claim(index: int):
        barrier.wait()
        return second_queue.claim(owner=f"worker-{index}", lease_seconds=60)

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = tuple(executor.map(claim, range(8)))

    winners = tuple(outcome for outcome in outcomes if outcome is not None)
    assert len(winners) == 1
    assert winners[0].job_id == single.job_id
    assert second_queue.get(single.job_id).attempt == 1


def test_expired_lease_is_reclaimed_after_crash_with_a_new_token(tmp_path: Path) -> None:
    clock = MutableClock()
    path = tmp_path / "delivery.sqlite3"
    first_process = SQLiteQueue(path, clock=clock)
    submitted = first_process.enqueue(
        _request(),
        idempotency_key="crash-001",
        max_attempts=3,
    )
    first_claim = first_process.claim(owner="worker-a", lease_seconds=10)
    assert first_claim is not None

    clock.advance(11)
    recovered_process = SQLiteQueue(path, clock=clock)
    recovered = recovered_process.claim(owner="worker-b", lease_seconds=10)

    assert recovered is not None
    assert recovered.job_id == submitted.job_id
    assert recovered.attempt == 2
    assert recovered.lease.owner == "worker-b"
    assert recovered.lease.token != first_claim.lease.token
    with pytest.raises(LeaseConflictError, match="job lease is unavailable"):
        recovered_process.succeed(first_claim.lease, JobResult(summary={"aggregate_count": 3}))


def test_expired_final_attempt_moves_job_to_dead_letter(tmp_path: Path) -> None:
    clock = MutableClock()
    queue = _queue(tmp_path, clock=clock)
    submitted = queue.enqueue(
        _request(),
        idempotency_key="crash-final-001",
        max_attempts=1,
    )
    assert queue.claim(owner="worker-a", lease_seconds=5) is not None

    clock.advance(6)

    assert queue.claim(owner="worker-b", lease_seconds=5) is None
    dead = queue.get(submitted.job_id)
    assert dead.status is JobStatus.DEAD_LETTERED
    assert dead.error_class is ErrorClass.LEASE_EXPIRED
    assert queue.dead_letters() == (dead,)


def test_heartbeat_requires_current_owner_and_token_and_extends_lease(tmp_path: Path) -> None:
    clock = MutableClock()
    queue = _queue(tmp_path, clock=clock)
    queue.enqueue(_request(), idempotency_key="heartbeat-001")
    claimed = queue.claim(owner="worker-a", lease_seconds=10)
    assert claimed is not None

    wrong_owner = JobLease(
        job_id=claimed.job_id,
        owner="worker-b",
        token=claimed.lease.token,
        expires_at=claimed.lease.expires_at,
    )
    wrong_token = JobLease(
        job_id=claimed.job_id,
        owner=claimed.lease.owner,
        token="lease-" + "f" * 32,
        expires_at=claimed.lease.expires_at,
    )
    with pytest.raises(LeaseConflictError, match="job lease is unavailable"):
        queue.heartbeat(wrong_owner, lease_seconds=20)
    with pytest.raises(LeaseConflictError, match="job lease is unavailable"):
        queue.heartbeat(wrong_token, lease_seconds=20)

    clock.advance(5)
    extended = queue.heartbeat(claimed.lease, lease_seconds=20)
    assert extended.token == claimed.lease.token
    assert extended.expires_at > claimed.lease.expires_at

    clock.advance(6)
    succeeded = queue.succeed(extended, JobResult(summary={"aggregate_count": 3}))
    assert succeeded.status is JobStatus.SUCCEEDED


def test_retry_requeues_then_exhaustion_moves_to_dlq(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    submitted = queue.enqueue(
        _request(),
        idempotency_key="retry-001",
        max_attempts=2,
    )
    first = queue.claim(owner="worker-a", lease_seconds=30)
    assert first is not None

    pending = queue.retry(
        first.lease,
        JobFailure(error_class=ErrorClass.TIMEOUT),
    )
    assert pending.status is JobStatus.PENDING
    assert pending.attempt == 1
    assert pending.error_class is ErrorClass.TIMEOUT

    second = queue.claim(owner="worker-b", lease_seconds=30)
    assert second is not None
    assert second.job_id == submitted.job_id
    assert second.attempt == 2
    dead = queue.retry(
        second.lease,
        JobFailure(error_class=ErrorClass.RUNTIME),
    )

    assert dead.status is JobStatus.DEAD_LETTERED
    assert dead.error_class is ErrorClass.RUNTIME
    assert queue.claim(owner="worker-c", lease_seconds=30) is None
    assert queue.dead_letters() == (dead,)


def test_success_persists_only_safe_result_and_terminal_job_is_not_reclaimed(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    submitted = queue.enqueue(_request(), idempotency_key="success-001")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    result = JobResult(summary={"aggregate_count": 3, "approved_rate": 0.75})

    succeeded = queue.succeed(claimed.lease, result)

    assert succeeded.job_id == submitted.job_id
    assert succeeded.status is JobStatus.SUCCEEDED
    assert succeeded.result_hash == result.content_hash
    assert queue.result(submitted.job_id) == result
    assert queue.claim(owner="worker-b", lease_seconds=30) is None

    with pytest.raises(ValidationError):
        JobResult(summary={"source": "/private/customer/result.json"})


def test_pending_and_running_jobs_can_be_cancelled_but_terminal_jobs_cannot(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    pending = queue.enqueue(_request(count=1), idempotency_key="cancel-pending")
    cancelled_pending = queue.cancel(pending.job_id)
    assert cancelled_pending.status is JobStatus.CANCELLED

    running = queue.enqueue(_request(count=2), idempotency_key="cancel-running")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    assert claimed.job_id == running.job_id
    cancelled_running = queue.cancel(running.job_id)
    assert cancelled_running.status is JobStatus.CANCELLED
    with pytest.raises(LeaseConflictError, match="job lease is unavailable"):
        queue.succeed(claimed.lease, JobResult(summary={"aggregate_count": 2}))

    completed = queue.enqueue(_request(count=3), idempotency_key="cancel-completed")
    completed_claim = queue.claim(owner="worker-b", lease_seconds=30)
    assert completed_claim is not None
    queue.succeed(completed_claim.lease, JobResult(summary={"aggregate_count": 3}))
    with pytest.raises(InvalidJobTransitionError, match="job transition is unavailable"):
        queue.cancel(completed.job_id)

    assert queue.claim(owner="worker-c", lease_seconds=30) is None


def test_database_is_0600_and_unsafe_existing_files_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "delivery.sqlite3"
    SQLiteQueue(path)
    assert path.stat().st_mode & 0o777 == 0o600

    path.chmod(0o644)
    with pytest.raises(QueueStorageError) as broad_permissions:
        SQLiteQueue(path)
    assert str(broad_permissions.value) == "queue storage is unavailable"

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"not a database")
    target.chmod(0o600)
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(target)
    with pytest.raises(QueueStorageError, match="queue storage is unavailable"):
        SQLiteQueue(link)

    directory = tmp_path / "directory.sqlite3"
    directory.mkdir()
    with pytest.raises(QueueStorageError, match="queue storage is unavailable"):
        SQLiteQueue(directory)

    assert not os.path.samefile(link, path)


def test_expired_lease_is_rejected_after_waiting_for_database_write_lock(
    tmp_path: Path,
) -> None:
    import sqlite3
    from threading import Event

    class CoordinatedClock(MutableClock):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.observed = Event()
            self.release = Event()

        def __call__(self) -> datetime:
            value = self.current
            if self.armed:
                self.observed.set()
                assert self.release.wait(timeout=2)
            return value

    clock = CoordinatedClock()
    queue = _queue(tmp_path, clock=clock)
    queue.enqueue(_request(), idempotency_key="lock-expiry-001")
    claimed = queue.claim(owner="worker-a", lease_seconds=10)
    assert claimed is not None

    blocker = sqlite3.connect(queue.path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    clock.armed = True
    operation_started = Event()

    def acknowledge():
        operation_started.set()
        return queue.succeed(
            claimed.lease,
            JobResult(summary={"aggregate_count": 3}),
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(acknowledge)
            assert operation_started.wait(timeout=1)
            clock.observed.wait(timeout=1)
            clock.advance(11)
            clock.release.set()
            blocker.commit()
            with pytest.raises(LeaseConflictError, match="job lease is unavailable"):
                future.result(timeout=2)
    finally:
        clock.release.set()
        blocker.close()


def test_heartbeat_never_shortens_an_existing_lease(tmp_path: Path) -> None:
    clock = MutableClock()
    queue = _queue(tmp_path, clock=clock)
    queue.enqueue(_request(), idempotency_key="heartbeat-no-shorten")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None

    renewed = queue.heartbeat(claimed.lease, lease_seconds=2)

    assert renewed.expires_at == claimed.lease.expires_at


def test_claim_rejects_safe_request_json_that_no_longer_matches_hash(tmp_path: Path) -> None:
    import sqlite3

    from riskprobe.delivery import QueueIntegrityError

    queue = _queue(tmp_path)
    submitted = queue.enqueue(_request(count=3), idempotency_key="tampered-request")
    replacement = _request(count=99)
    with sqlite3.connect(queue.path) as connection:
        connection.execute(
            "UPDATE jobs SET request_json = ? WHERE job_id = ?",
            (replacement.model_dump_json(), submitted.job_id),
        )

    with pytest.raises(QueueIntegrityError) as tampered:
        queue.claim(owner="worker-a", lease_seconds=30)
    assert str(tampered.value) == "queue integrity check failed"


def test_database_creation_os_errors_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import riskprobe.delivery.queue as queue_module

    def deny_permissions(*args: object, **kwargs: object) -> None:
        raise PermissionError("secret filesystem detail")

    monkeypatch.setattr(queue_module.os, "fchmod", deny_permissions)

    with pytest.raises(QueueStorageError) as unavailable:
        SQLiteQueue(tmp_path / "blocked.sqlite3")
    assert str(unavailable.value) == "queue storage is unavailable"
    assert "secret filesystem detail" not in str(unavailable.value)


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    pending = [error]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return tuple(chain)


def _assert_sanitized_delivery_error(
    error: BaseException,
    *forbidden: str,
) -> None:
    import traceback

    chain = _exception_chain(error)
    assert chain == (error,)
    for current in chain:
        formatted = "".join(
            traceback.format_exception(type(current), current, current.__traceback__)
        )
        for value in forbidden:
            assert value not in str(current)
            assert value not in repr(current)
            assert value not in formatted


def test_untrusted_job_validation_factories_raise_fixed_errors() -> None:
    from riskprobe.delivery import DeliveryValidationError

    secret = "/private/customer-secret/input.parquet"
    cases = (
        (JobRequest, {"kind": "profile", "payload": {"source": secret}}),
        (JobResult, {"summary": {"source": secret}}),
    )

    for dto_type, payload in cases:
        with pytest.raises(DeliveryValidationError) as rejected:
            dto_type.safe_validate(payload)

        assert str(rejected.value) == "delivery input is invalid"
        _assert_sanitized_delivery_error(rejected.value, secret)


@pytest.mark.parametrize("unsafe_parent_kind", ["group_writable", "symlink"])
def test_queue_rejects_unsafe_direct_parent(
    tmp_path: Path,
    unsafe_parent_kind: str,
) -> None:
    if unsafe_parent_kind == "group_writable":
        parent = tmp_path / "unsafe-parent"
        parent.mkdir(mode=0o700)
        parent.chmod(0o770)
    else:
        target = tmp_path / "owned-parent"
        target.mkdir(mode=0o700)
        parent = tmp_path / "linked-parent"
        parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(QueueStorageError) as rejected:
        SQLiteQueue(parent / "queue.sqlite3")

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, str(parent))


@pytest.mark.parametrize(
    "unsafe_ancestor_kind",
    ["group_writable", "world_writable", "symlink"],
)
def test_queue_rejects_unsafe_intermediate_ancestor(
    tmp_path: Path,
    unsafe_ancestor_kind: str,
) -> None:
    ancestor = tmp_path / "unsafe-ancestor"
    if unsafe_ancestor_kind == "symlink":
        target = tmp_path / "owned-ancestor"
        target.mkdir(mode=0o700)
        (target / "safe-parent").mkdir(mode=0o700)
        ancestor.symlink_to(target, target_is_directory=True)
    else:
        ancestor.mkdir(mode=0o700)
        (ancestor / "safe-parent").mkdir(mode=0o700)
        ancestor.chmod(0o770 if unsafe_ancestor_kind == "group_writable" else 0o702)

    with pytest.raises(QueueStorageError) as rejected:
        SQLiteQueue(ancestor / "safe-parent" / "queue.sqlite3")

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, str(ancestor))


def test_queue_rejects_ancestor_replacement_while_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    import riskprobe.delivery.queue as queue_module

    storage_root = tmp_path / "storage-root"
    direct_parent = storage_root / "direct-parent"
    direct_parent.mkdir(parents=True, mode=0o700)
    expected_path = direct_parent / "queue.sqlite3"
    SQLiteQueue(expected_path)

    displaced_root = tmp_path / "displaced-root"
    real_connect = sqlite3.connect
    replaced = False

    def replacing_connect(database: object, *args: object, **kwargs: object):
        nonlocal replaced
        connection = real_connect(database, *args, **kwargs)
        if Path(database) == expected_path and not replaced:
            replaced = True
            os.replace(storage_root, displaced_root)
            storage_root.mkdir(mode=0o700)
            direct_parent.mkdir(mode=0o700)
            os.link(displaced_root / "direct-parent" / "queue.sqlite3", expected_path)
        return connection

    monkeypatch.setattr(queue_module.sqlite3, "connect", replacing_connect)

    with pytest.raises(QueueStorageError) as rejected:
        SQLiteQueue(expected_path)

    assert replaced
    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, str(storage_root))


def test_queue_rejects_path_replacement_while_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    import riskprobe.delivery.queue as queue_module

    expected_path = tmp_path / "queue.sqlite3"
    displaced_path = tmp_path / "opened.sqlite3"
    real_connect = sqlite3.connect
    replaced = False

    def replacing_connect(database: object, *args: object, **kwargs: object):
        nonlocal replaced
        connection = real_connect(database, *args, **kwargs)
        if Path(database) == expected_path and not replaced:
            replaced = True
            os.replace(expected_path, displaced_path)
            replacement = real_connect(expected_path)
            replacement.close()
            expected_path.chmod(0o600)
        return connection

    monkeypatch.setattr(queue_module.sqlite3, "connect", replacing_connect)

    with pytest.raises(QueueStorageError) as rejected:
        SQLiteQueue(expected_path)

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, str(expected_path))


class _FaultyQueueConnection:
    def __init__(
        self,
        *,
        begin_error: BaseException | None = None,
        commit_error: BaseException | None = None,
        rollback_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.begin_error = begin_error
        self.commit_error = commit_error
        self.rollback_error = rollback_error
        self.close_error = close_error

    def execute(self, statement: str, parameters: object = ()) -> None:
        if statement == "BEGIN IMMEDIATE" and self.begin_error is not None:
            raise self.begin_error

    def commit(self) -> None:
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error


@pytest.mark.parametrize("failure_phase", ["begin", "body", "commit", "close"])
def test_queue_transaction_faults_are_fixed_and_cleanup_cannot_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    import sqlite3

    queue = _queue(tmp_path)
    secrets = {
        "begin": "secret begin /private/queue.sqlite3",
        "body": "secret body /private/queue.sqlite3",
        "commit": "secret commit /private/queue.sqlite3",
        "rollback": "secret rollback /private/queue.sqlite3",
        "close": "secret close /private/queue.sqlite3",
    }
    connection = _FaultyQueueConnection(
        begin_error=(
            sqlite3.OperationalError(secrets["begin"])
            if failure_phase == "begin"
            else None
        ),
        commit_error=(
            sqlite3.OperationalError(secrets["commit"])
            if failure_phase == "commit"
            else None
        ),
        rollback_error=sqlite3.OperationalError(secrets["rollback"]),
        close_error=(
            sqlite3.OperationalError(secrets["close"])
            if failure_phase in {"begin", "body", "commit", "close"}
            else None
        ),
    )
    monkeypatch.setattr(queue, "_connect", lambda: connection)

    with pytest.raises(QueueStorageError) as rejected:
        with queue._transaction():
            if failure_phase == "body":
                raise sqlite3.OperationalError(secrets["body"])

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, *secrets.values())


@pytest.mark.parametrize(
    "primary",
    [
        LeaseConflictError("job lease is unavailable"),
        KeyboardInterrupt("stop queue operation"),
        SystemExit("stop queue process"),
    ],
)
def test_queue_transaction_preserves_primary_base_exception_over_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException,
) -> None:
    import sqlite3

    queue = _queue(tmp_path)
    rollback_secret = "secret rollback /private/queue.sqlite3"
    close_secret = "secret close /private/queue.sqlite3"
    connection = _FaultyQueueConnection(
        rollback_error=sqlite3.OperationalError(rollback_secret),
        close_error=sqlite3.OperationalError(close_secret),
    )
    monkeypatch.setattr(queue, "_connect", lambda: connection)

    with pytest.raises(type(primary)) as raised:
        with queue._transaction():
            raise primary

    assert raised.value is primary
    _assert_sanitized_delivery_error(raised.value, rollback_secret, close_secret)


def test_queue_read_connection_close_failure_is_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    queue = _queue(tmp_path)
    secret = "secret close /private/queue.sqlite3"
    connection = _FaultyQueueConnection(close_error=sqlite3.OperationalError(secret))
    monkeypatch.setattr(queue, "_connect", lambda: connection)

    with pytest.raises(QueueStorageError) as rejected:
        with queue._connection():
            pass

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, secret)


@pytest.mark.parametrize("clock_error_type", [RuntimeError, OSError])
def test_queue_clock_failure_traceback_is_sanitized(
    tmp_path: Path,
    clock_error_type: type[Exception],
) -> None:
    secret = "secret clock /private/queue.sqlite3"

    def broken_clock() -> datetime:
        raise clock_error_type(secret)

    queue = SQLiteQueue(tmp_path / "queue.sqlite3", clock=broken_clock)

    with pytest.raises(QueueStorageError) as rejected:
        queue.enqueue(_request(), idempotency_key="clock-failure")

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(rejected.value, secret)


def test_lease_is_expired_at_exact_deadline_for_all_mutations_and_claim(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    queue = _queue(tmp_path, clock=clock)
    submitted = queue.enqueue(
        _request(),
        idempotency_key="exact-expiry",
        max_attempts=2,
    )
    claimed = queue.claim(owner="worker-a", lease_seconds=10)
    assert claimed is not None
    clock.advance(10)

    operations = (
        lambda: queue.heartbeat(claimed.lease, lease_seconds=10),
        lambda: queue.retry(
            claimed.lease,
            JobFailure(error_class=ErrorClass.TIMEOUT),
        ),
        lambda: queue.succeed(
            claimed.lease,
            JobResult(summary={"aggregate_count": 3}),
        ),
    )
    for operation in operations:
        with pytest.raises(LeaseConflictError, match="job lease is unavailable"):
            operation()

    reclaimed = queue.claim(owner="worker-b", lease_seconds=10)
    assert reclaimed is not None
    assert reclaimed.job_id == submitted.job_id
    assert reclaimed.attempt == 2
    assert reclaimed.lease.token != claimed.lease.token


def test_succeeded_result_tampering_raises_fixed_integrity_error(
    tmp_path: Path,
) -> None:
    import sqlite3

    from riskprobe.delivery import QueueIntegrityError

    queue = _queue(tmp_path)
    submitted = queue.enqueue(_request(), idempotency_key="tampered-result")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    queue.succeed(claimed.lease, JobResult(summary={"aggregate_count": 3}))
    with sqlite3.connect(queue.path) as connection:
        connection.execute(
            "UPDATE jobs SET result_hash = ? WHERE job_id = ?",
            ("f" * 64, submitted.job_id),
        )

    with pytest.raises(QueueIntegrityError) as rejected:
        queue.result(submitted.job_id)

    assert str(rejected.value) == "queue integrity check failed"
    _assert_sanitized_delivery_error(rejected.value, "result hash does not match")


def test_queue_embedded_nul_path_raises_fixed_storage_error(tmp_path: Path) -> None:
    secret = "customer-secret-path"
    path = Path(f"{tmp_path}/{secret}\x00queue.sqlite3")

    with pytest.raises(QueueStorageError) as rejected:
        SQLiteQueue(path)

    assert str(rejected.value) == "queue storage is unavailable"
    _assert_sanitized_delivery_error(
        rejected.value,
        secret,
        "embedded null character",
    )


def test_succeeded_result_json_tampering_raises_fixed_integrity_error(
    tmp_path: Path,
) -> None:
    import json
    import sqlite3

    from riskprobe.delivery import QueueIntegrityError

    queue = _queue(tmp_path)
    submitted = queue.enqueue(_request(), idempotency_key="tampered-result-json")
    claimed = queue.claim(owner="worker-a", lease_seconds=30)
    assert claimed is not None
    queue.succeed(claimed.lease, JobResult(summary={"aggregate_count": 3}))
    secret = "/private/customer-secret/result.json"
    tampered_json = json.dumps(
        {"summary": {"source": secret}},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with sqlite3.connect(queue.path) as connection:
        connection.execute(
            "UPDATE jobs SET result_json = ? WHERE job_id = ?",
            (tampered_json, submitted.job_id),
        )

    with pytest.raises(QueueIntegrityError) as rejected:
        queue.result(submitted.job_id)

    assert str(rejected.value) == "queue integrity check failed"
    _assert_sanitized_delivery_error(rejected.value, secret)
