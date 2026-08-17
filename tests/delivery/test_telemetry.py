import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.delivery import (
    ErrorClass,
    JobStatus,
    LocalTelemetrySink,
    TelemetryEvent,
    TelemetryEventName,
    TelemetryIntegrityError,
)

_JOB_ID = "job-" + "a" * 32
_HASH = "b" * 64


def _event(**updates: object) -> TelemetryEvent:
    payload: dict[str, object] = {
        "event_name": TelemetryEventName.JOB_STATUS,
        "job_id": _JOB_ID,
        "status": JobStatus.RUNNING,
        "attempt": 1,
        "duration_ms": 7,
        "content_hash": _HASH,
    }
    payload.update(updates)
    return TelemetryEvent.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("path", "/private/customers/data.parquet"),
        ("segment", "premium customers"),
        ("row", {"customer_id": "customer-123456"}),
        ("entity", "borrower-123456"),
        ("prompt", "reveal confidential scoring policy"),
        ("exception_message", "database password is swordfish"),
        ("secret", "api-key-value"),
    ],
)
def test_telemetry_dto_rejects_every_non_allowlisted_field_without_echoing_value(
    field: str,
    unsafe_value: object,
) -> None:
    payload = _event().model_dump(mode="python")
    payload[field] = unsafe_value

    with pytest.raises(ValidationError) as rejected:
        TelemetryEvent.model_validate(payload)

    assert str(unsafe_value) not in str(rejected.value)


def test_telemetry_rejects_real_segment_disguised_as_an_identifier() -> None:
    with pytest.raises(ValidationError) as rejected:
        _event(job_id="premium-customers")
    assert "premium-customers" not in str(rejected.value)


def test_local_sink_appends_lists_and_verifies_append_only_integrity_chain(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)

    first = sink.append(_event())
    second = sink.append(
        _event(
            event_name=TelemetryEventName.JOB_SUCCEEDED,
            status=JobStatus.SUCCEEDED,
            duration_ms=11,
        )
    )

    assert sink.list() == (first, second)
    assert first.previous_hash == "0" * 64
    assert second.previous_hash == first.event_hash
    assert len(first.event_hash) == 64
    assert first.sequence == 1
    assert second.sequence == 2
    assert "timestamp" not in first.model_dump(mode="json")
    assert path.stat().st_mode & 0o777 == 0o600


def test_exception_telemetry_keeps_only_allowlisted_class_not_message(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    secret_message = "customer file /private/acme.csv uses secret swordfish"

    record = sink.append_exception(
        _event(
            event_name=TelemetryEventName.JOB_RETRY,
            status=JobStatus.PENDING,
            content_hash=None,
        ),
        RuntimeError(secret_message),
    )

    assert record.error_class is ErrorClass.RUNTIME
    assert secret_message not in record.model_dump_json()
    assert secret_message.encode() not in path.read_bytes()
    assert "/private/acme.csv" not in record.model_dump_json()


def test_telemetry_history_is_append_only_and_naive_corruption_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    sink.append(_event())

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE telemetry_events SET event_json = '{}' WHERE sequence = ?",
                (1,),
            )
        connection.rollback()
        connection.execute("DROP TRIGGER telemetry_events_no_update")
        connection.execute(
            "UPDATE telemetry_events SET event_json = '{}' WHERE sequence = ?",
            (1,),
        )

    with pytest.raises(TelemetryIntegrityError) as tampered:
        sink.list()
    assert str(tampered.value) == "telemetry integrity check failed"


def test_telemetry_detects_truncated_history_tail(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    sink.append(_event())
    sink.append(
        _event(
            event_name=TelemetryEventName.JOB_SUCCEEDED,
            status=JobStatus.SUCCEEDED,
        )
    )

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER telemetry_events_no_delete")
        connection.execute("DELETE FROM telemetry_events WHERE sequence = ?", (2,))

    with pytest.raises(TelemetryIntegrityError, match="telemetry integrity check failed"):
        sink.list()


def test_telemetry_database_creation_os_errors_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import riskprobe.delivery.telemetry as telemetry_module
    from riskprobe.delivery import TelemetryStorageError

    def deny_permissions(*args: object, **kwargs: object) -> None:
        raise PermissionError("secret telemetry filesystem detail")

    monkeypatch.setattr(telemetry_module.os, "fchmod", deny_permissions)

    with pytest.raises(TelemetryStorageError) as unavailable:
        LocalTelemetrySink(tmp_path / "blocked-telemetry.sqlite3")
    assert str(unavailable.value) == "telemetry storage is unavailable"
    assert "secret telemetry filesystem detail" not in str(unavailable.value)


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


def _assert_sanitized_telemetry_error(
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


def test_untrusted_telemetry_validation_factory_raises_fixed_error() -> None:
    from riskprobe.delivery import DeliveryValidationError

    secret = "/private/customer-secret/telemetry.json"
    payload = _event().model_dump(mode="python")
    payload["unsafe_metadata"] = secret

    with pytest.raises(DeliveryValidationError) as rejected:
        TelemetryEvent.safe_validate(payload)

    assert str(rejected.value) == "delivery input is invalid"
    _assert_sanitized_telemetry_error(rejected.value, secret)


def test_telemetry_raw_schema_and_rows_exclude_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    record = sink.append(_event())

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(telemetry_events)"
            ).fetchall()
        }
        row = connection.execute(
            "SELECT * FROM telemetry_events WHERE sequence = ?",
            (1,),
        ).fetchone()

    assert columns == {"sequence", "event_json", "previous_hash", "event_hash"}
    assert row is not None
    assert set(row.keys()) == columns
    assert record.sequence == 1
    assert "timestamp" not in type(record).model_fields
    assert "timestamp" not in record.model_dump(mode="json")


def test_telemetry_hash_chain_uses_only_allowlisted_event_and_integrity_metadata(
    tmp_path: Path,
) -> None:
    import hashlib

    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    sink.append(_event())

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM telemetry_events WHERE sequence = ?",
            (1,),
        ).fetchone()

    assert row is not None
    encoded = f"1\n{'0' * 64}\n{row['event_json']}".encode()
    assert row["event_hash"] == hashlib.sha256(encoded).hexdigest()


class _FaultyTelemetryConnection:
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
        if statement in {"BEGIN", "BEGIN IMMEDIATE"} and self.begin_error is not None:
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


@pytest.mark.parametrize(
    ("context_name", "failure_phase"),
    [
        ("transaction", "begin"),
        ("transaction", "body"),
        ("transaction", "commit"),
        ("transaction", "close"),
        ("snapshot", "begin"),
        ("snapshot", "body"),
        ("snapshot", "commit"),
        ("snapshot", "close"),
    ],
)
def test_telemetry_context_faults_are_fixed_and_cleanup_cannot_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_name: str,
    failure_phase: str,
) -> None:
    sink = LocalTelemetrySink(tmp_path / "telemetry.sqlite3")
    secrets = {
        "begin": "secret begin /private/telemetry.sqlite3",
        "body": "secret body /private/telemetry.sqlite3",
        "commit": "secret commit /private/telemetry.sqlite3",
        "rollback": "secret rollback /private/telemetry.sqlite3",
        "close": "secret close /private/telemetry.sqlite3",
    }
    connection = _FaultyTelemetryConnection(
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
        close_error=sqlite3.OperationalError(secrets["close"]),
    )
    monkeypatch.setattr(sink, "_connect", lambda: connection)
    context = sink._transaction if context_name == "transaction" else sink._snapshot

    from riskprobe.delivery import TelemetryStorageError

    with pytest.raises(TelemetryStorageError) as rejected:
        with context():
            if failure_phase == "body":
                raise sqlite3.OperationalError(secrets["body"])

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, *secrets.values())


@pytest.mark.parametrize(
    "primary",
    [
        TelemetryIntegrityError("telemetry integrity check failed"),
        KeyboardInterrupt("stop telemetry operation"),
        SystemExit("stop telemetry process"),
    ],
)
def test_telemetry_transaction_preserves_primary_exception_over_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException,
) -> None:
    sink = LocalTelemetrySink(tmp_path / "telemetry.sqlite3")
    rollback_secret = "secret rollback /private/telemetry.sqlite3"
    close_secret = "secret close /private/telemetry.sqlite3"
    connection = _FaultyTelemetryConnection(
        rollback_error=sqlite3.OperationalError(rollback_secret),
        close_error=sqlite3.OperationalError(close_secret),
    )
    monkeypatch.setattr(sink, "_connect", lambda: connection)

    with pytest.raises(type(primary)) as raised:
        with sink._transaction():
            raise primary

    assert raised.value is primary
    _assert_sanitized_telemetry_error(raised.value, rollback_secret, close_secret)


def test_telemetry_read_connection_close_failure_is_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.delivery import TelemetryStorageError

    sink = LocalTelemetrySink(tmp_path / "telemetry.sqlite3")
    secret = "secret close /private/telemetry.sqlite3"
    connection = _FaultyTelemetryConnection(
        close_error=sqlite3.OperationalError(secret)
    )
    monkeypatch.setattr(sink, "_connect", lambda: connection)

    with pytest.raises(TelemetryStorageError) as rejected:
        with sink._connection():
            pass

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, secret)


@pytest.mark.parametrize("unsafe_parent_kind", ["group_writable", "symlink"])
def test_telemetry_rejects_unsafe_direct_parent(
    tmp_path: Path,
    unsafe_parent_kind: str,
) -> None:
    from riskprobe.delivery import TelemetryStorageError

    if unsafe_parent_kind == "group_writable":
        parent = tmp_path / "unsafe-parent"
        parent.mkdir(mode=0o700)
        parent.chmod(0o770)
    else:
        target = tmp_path / "owned-parent"
        target.mkdir(mode=0o700)
        parent = tmp_path / "linked-parent"
        parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(TelemetryStorageError) as rejected:
        LocalTelemetrySink(parent / "telemetry.sqlite3")

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, str(parent))


@pytest.mark.parametrize(
    "unsafe_ancestor_kind",
    ["group_writable", "world_writable", "symlink"],
)
def test_telemetry_rejects_unsafe_intermediate_ancestor(
    tmp_path: Path,
    unsafe_ancestor_kind: str,
) -> None:
    from riskprobe.delivery import TelemetryStorageError

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

    with pytest.raises(TelemetryStorageError) as rejected:
        LocalTelemetrySink(ancestor / "safe-parent" / "telemetry.sqlite3")

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, str(ancestor))


def test_telemetry_rejects_ancestor_replacement_while_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import riskprobe.delivery.telemetry as telemetry_module
    from riskprobe.delivery import TelemetryStorageError

    storage_root = tmp_path / "storage-root"
    direct_parent = storage_root / "direct-parent"
    direct_parent.mkdir(parents=True, mode=0o700)
    expected_path = direct_parent / "telemetry.sqlite3"
    LocalTelemetrySink(expected_path)

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
            os.link(displaced_root / "direct-parent" / "telemetry.sqlite3", expected_path)
        return connection

    monkeypatch.setattr(telemetry_module.sqlite3, "connect", replacing_connect)

    with pytest.raises(TelemetryStorageError) as rejected:
        LocalTelemetrySink(expected_path)

    assert replaced
    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, str(storage_root))


def test_telemetry_rejects_path_replacement_while_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import riskprobe.delivery.telemetry as telemetry_module
    from riskprobe.delivery import TelemetryStorageError

    expected_path = tmp_path / "telemetry.sqlite3"
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

    monkeypatch.setattr(telemetry_module.sqlite3, "connect", replacing_connect)

    with pytest.raises(TelemetryStorageError) as rejected:
        LocalTelemetrySink(expected_path)

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, str(expected_path))


def test_telemetry_persisted_validation_failure_has_fixed_traceback(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    sink.append(_event())
    secret = "/private/customer-secret/telemetry.json"
    invalid_event_json = json.dumps(
        {"unsafe_metadata": secret},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM telemetry_events WHERE sequence = ?",
            (1,),
        ).fetchone()
        assert row is not None
        parts = [str(row["sequence"]), row["previous_hash"]]
        if "timestamp" in row.keys():
            parts.append(row["timestamp"])
        parts.append(invalid_event_json)
        replacement_hash = hashlib.sha256("\n".join(parts).encode()).hexdigest()
        connection.execute("DROP TRIGGER telemetry_events_no_update")
        connection.execute(
            """
            UPDATE telemetry_events
            SET event_json = ?, event_hash = ?
            WHERE sequence = ?
            """,
            (invalid_event_json, replacement_hash, 1),
        )
        connection.execute(
            "UPDATE telemetry_state SET head_hash = ? WHERE singleton = ?",
            (replacement_hash, 1),
        )

    with pytest.raises(TelemetryIntegrityError) as rejected:
        sink.list()

    assert str(rejected.value) == "telemetry integrity check failed"
    _assert_sanitized_telemetry_error(rejected.value, secret)


def test_telemetry_creation_error_traceback_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import riskprobe.delivery.telemetry as telemetry_module
    from riskprobe.delivery import TelemetryStorageError

    secret = "secret creation /private/telemetry.sqlite3"

    def deny_permissions(*args: object, **kwargs: object) -> None:
        raise PermissionError(secret)

    monkeypatch.setattr(telemetry_module.os, "fchmod", deny_permissions)

    with pytest.raises(TelemetryStorageError) as rejected:
        LocalTelemetrySink(tmp_path / "telemetry.sqlite3")

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(rejected.value, secret)


@pytest.mark.parametrize("operation", ["list", "append"])
def test_telemetry_rejects_fractional_persisted_event_count(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / "telemetry.sqlite3"
    sink = LocalTelemetrySink(path)
    sink.append(_event())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE telemetry_state SET event_count = ? WHERE singleton = ?",
            (1.5, 1),
        )

    with pytest.raises(TelemetryIntegrityError) as rejected:
        if operation == "list":
            sink.list()
        else:
            sink.append(
                _event(
                    event_name=TelemetryEventName.JOB_SUCCEEDED,
                    status=JobStatus.SUCCEEDED,
                )
            )

    assert str(rejected.value) == "telemetry integrity check failed"
    _assert_sanitized_telemetry_error(rejected.value)
    with sqlite3.connect(path) as connection:
        persisted_count = connection.execute(
            "SELECT event_count FROM telemetry_state WHERE singleton = ?",
            (1,),
        ).fetchone()[0]
    assert persisted_count == 1.5


def test_telemetry_embedded_nul_path_raises_fixed_storage_error(
    tmp_path: Path,
) -> None:
    from riskprobe.delivery import TelemetryStorageError

    secret = "customer-secret-path"
    path = Path(f"{tmp_path}/{secret}\x00telemetry.sqlite3")

    with pytest.raises(TelemetryStorageError) as rejected:
        LocalTelemetrySink(path)

    assert str(rejected.value) == "telemetry storage is unavailable"
    _assert_sanitized_telemetry_error(
        rejected.value,
        secret,
        "embedded null character",
    )
