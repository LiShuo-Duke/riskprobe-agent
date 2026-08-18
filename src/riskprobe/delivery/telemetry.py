"""Local allowlisted telemetry with append-only corruption detection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

from pydantic import Field, ValidationError, field_validator

from riskprobe.delivery.queue import (
    Clock,
    ErrorClass,
    JobStatus,
    _AncestorSnapshot,
    _StorageIdentity,
    _StrictDTO,
    _UnsafeStoragePath,
    _raise_unlinked,
    _require_job_id,
    _storage_ancestor_snapshot,
)

_SCHEMA_VERSION = 2
_ROOT_HASH = "0" * 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STORAGE_ERROR = "telemetry storage is unavailable"
_INTEGRITY_ERROR = "telemetry integrity check failed"
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS telemetry_schema (
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS telemetry_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        event_count INTEGER NOT NULL CHECK (event_count >= 0),
        head_hash TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS telemetry_events (
        sequence INTEGER PRIMARY KEY,
        event_json TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        event_hash TEXT NOT NULL UNIQUE
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS telemetry_events_no_update
    BEFORE UPDATE ON telemetry_events
    BEGIN
        SELECT RAISE(ABORT, 'telemetry history is append-only');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS telemetry_events_no_delete
    BEFORE DELETE ON telemetry_events
    BEGIN
        SELECT RAISE(ABORT, 'telemetry history is append-only');
    END
    """,
)


class TelemetryIntegrityError(RuntimeError):
    """Raised when local telemetry does not verify against its integrity chain."""


class TelemetryStorageError(RuntimeError):
    """Raised with a fixed message when telemetry storage is unavailable."""


class TelemetryEventName(StrEnum):
    JOB_CANCELLED = "job_cancelled"
    JOB_CLAIMED = "job_claimed"
    JOB_DEAD_LETTERED = "job_dead_lettered"
    JOB_HEARTBEAT = "job_heartbeat"
    JOB_RESULT = "job_result"
    JOB_RETRY = "job_retry"
    JOB_STATUS = "job_status"
    JOB_SUBMITTED = "job_submitted"
    JOB_SUCCEEDED = "job_succeeded"


class TelemetryEvent(_StrictDTO):
    """The telemetry allowlist; use safe_validate for untrusted mappings."""

    event_name: TelemetryEventName
    job_id: str
    status: JobStatus
    attempt: int = Field(ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    error_class: ErrorClass | None = None
    content_hash: str | None = None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return _require_job_id(value)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("content_hash must be SHA-256")
        return value


class TelemetryRecord(TelemetryEvent):
    """One verified event plus append-only integrity-chain metadata."""

    sequence: int = Field(ge=1)
    previous_hash: str
    event_hash: str

    @field_validator("previous_hash", "event_hash")
    @classmethod
    def validate_chain_hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("integrity hash must be SHA-256")
        return value


class LocalTelemetrySink:
    """Synchronous local append-only observability with corruption detection.

    The unkeyed chain detects accidental and naive corruption, not a same-user
    database writer that can rewrite rows and state. Ancestor and file identity
    checks narrow pathname races, but stdlib SQLite does not expose the opened
    database file descriptor. Pure stdlib cannot fully eliminate races by a
    same-UID actor able to rewrite any path component; these checks are not a
    descriptor-relative no-follow guarantee.
    """

    def __init__(self, path: Path, *, clock: Clock | None = None) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        try:
            self.path = Path(path).absolute()
        except (OSError, TypeError, ValueError):
            _raise_telemetry_storage_error()
        self._prepare_parent_directory()
        self._prepare_database_file()
        self._initialize_schema()

    def append(self, event: TelemetryEvent) -> TelemetryRecord:
        if not isinstance(event, TelemetryEvent):
            raise TypeError("event must be a TelemetryEvent")
        event_json = _canonical_json(event.model_dump(mode="json"))
        with self._transaction() as connection:
            count, previous_hash = self._verified_head(connection)
            sequence = count + 1
            event_hash = _event_hash(sequence, previous_hash, event_json)
            record = self._record(event, sequence, previous_hash, event_hash)
            connection.execute(
                """
                INSERT INTO telemetry_events(
                    sequence, event_json, previous_hash, event_hash
                ) VALUES (?, ?, ?, ?)
                """,
                (sequence, event_json, previous_hash, event_hash),
            )
            connection.execute(
                """
                UPDATE telemetry_state
                SET event_count = ?, head_hash = ?
                WHERE singleton = ?
                """,
                (sequence, event_hash, 1),
            )
        return record

    def append_exception(
        self,
        event: TelemetryEvent,
        error: BaseException,
    ) -> TelemetryRecord:
        if not isinstance(event, TelemetryEvent):
            raise TypeError("event must be a TelemetryEvent")
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        sanitized = TelemetryEvent(
            event_name=event.event_name,
            job_id=event.job_id,
            status=event.status,
            attempt=event.attempt,
            duration_ms=event.duration_ms,
            error_class=_classify_exception(error),
            content_hash=event.content_hash,
        )
        return self.append(sanitized)

    def list(self) -> tuple[TelemetryRecord, ...]:
        with self._snapshot() as connection:
            state = connection.execute(
                "SELECT event_count, head_hash FROM telemetry_state WHERE singleton = ?",
                (1,),
            ).fetchone()
            rows = connection.execute(
                "SELECT * FROM telemetry_events ORDER BY sequence"
            ).fetchall()

        if state is None:
            _raise_telemetry_integrity_error()
        try:
            expected_count = _require_integrity_integer(
                state["event_count"],
                minimum=0,
            )
            expected_head = _require_integrity_hash(state["head_hash"])
        except (IndexError, KeyError, TypeError, ValueError):
            _raise_telemetry_integrity_error()

        records: list[TelemetryRecord] = []
        expected_previous = _ROOT_HASH
        for expected_sequence, row in enumerate(rows, start=1):
            try:
                sequence = _require_integrity_integer(
                    row["sequence"],
                    minimum=1,
                )
                event_json = _require_integrity_text(row["event_json"])
                previous_hash = _require_integrity_hash(row["previous_hash"])
                event_hash = _require_integrity_hash(row["event_hash"])
                if (
                    sequence != expected_sequence
                    or previous_hash != expected_previous
                    or event_hash != _event_hash(sequence, previous_hash, event_json)
                ):
                    _raise_telemetry_integrity_error()
                event = TelemetryEvent.model_validate_json(event_json)
                record = self._record(event, sequence, previous_hash, event_hash)
            except TelemetryIntegrityError:
                raise
            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ):
                _raise_telemetry_integrity_error()
            records.append(record)
            expected_previous = event_hash

        if len(records) != expected_count or expected_previous != expected_head:
            _raise_telemetry_integrity_error()
        return tuple(records)

    events = list

    def _verified_head(self, connection: sqlite3.Connection) -> tuple[int, str]:
        state = connection.execute(
            "SELECT event_count, head_hash FROM telemetry_state WHERE singleton = ?",
            (1,),
        ).fetchone()
        last = connection.execute(
            "SELECT sequence, event_hash FROM telemetry_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if state is None:
            _raise_telemetry_integrity_error()
        try:
            count = _require_integrity_integer(state["event_count"], minimum=0)
            head_hash = _require_integrity_hash(state["head_hash"])
            last_sequence = (
                _require_integrity_integer(last["sequence"], minimum=1)
                if last is not None
                else None
            )
            last_hash = (
                _require_integrity_hash(last["event_hash"])
                if last is not None
                else None
            )
        except (IndexError, KeyError, TypeError, ValueError):
            _raise_telemetry_integrity_error()
        if count == 0:
            if last is not None or head_hash != _ROOT_HASH:
                _raise_telemetry_integrity_error()
        elif last is None or last_sequence != count or last_hash != head_hash:
            _raise_telemetry_integrity_error()
        return count, head_hash

    def _prepare_parent_directory(self) -> None:
        try:
            expected = _storage_ancestor_snapshot(
                self.path.parent,
                allow_missing=True,
            )
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = _storage_ancestor_snapshot(
                self.path.parent,
                allow_missing=False,
            )
            if current[: len(expected)] != expected:
                raise _UnsafeStoragePath
        except (OSError, ValueError, _UnsafeStoragePath):
            _raise_telemetry_storage_error()

    def _validate_parent_directory(self) -> _AncestorSnapshot:
        try:
            return _storage_ancestor_snapshot(
                self.path.parent,
                allow_missing=False,
            )
        except (OSError, ValueError, _UnsafeStoragePath):
            _raise_telemetry_storage_error()

    def _require_ancestor_snapshot(self, expected: _AncestorSnapshot) -> None:
        if self._validate_parent_directory() != expected:
            _raise_telemetry_storage_error()

    def _prepare_database_file(self) -> None:
        expected_ancestors = self._validate_parent_directory()
        try:
            self.path.lstat()
        except FileNotFoundError:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor: int | None = None
            try:
                descriptor = os.open(self.path, flags, 0o600)
                os.fchmod(descriptor, 0o600)
                details = os.fstat(descriptor)
                mode = stat.S_IMODE(details.st_mode)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_uid != os.geteuid()
                    or mode != 0o600
                ):
                    _raise_telemetry_storage_error()
                descriptor_identity = (
                    details.st_dev,
                    details.st_ino,
                    details.st_uid,
                    mode,
                )
                if self._validate_database_file() != descriptor_identity:
                    _raise_telemetry_storage_error()
            except (OSError, ValueError):
                if descriptor is not None:
                    _close_descriptor_after_failure(descriptor)
                _raise_telemetry_storage_error()
            except BaseException:
                if descriptor is not None:
                    _close_descriptor_after_failure(descriptor)
                raise
            else:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except (OSError, ValueError):
                        _raise_telemetry_storage_error()
        except (OSError, ValueError):
            _raise_telemetry_storage_error()
        self._require_ancestor_snapshot(expected_ancestors)
        self._validate_database_file()

    def _validate_database_file(self) -> _StorageIdentity:
        try:
            details = self.path.lstat()
        except (OSError, ValueError):
            _raise_telemetry_storage_error()
        mode = stat.S_IMODE(details.st_mode)
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or mode != 0o600
        ):
            _raise_telemetry_storage_error()
        return (details.st_dev, details.st_ino, details.st_uid, mode)

    def _validate_connected_database(self, connection: sqlite3.Connection) -> None:
        try:
            rows = connection.execute("PRAGMA database_list").fetchall()
            main_rows = tuple(row for row in rows if row["name"] == "main")
            if len(main_rows) != 1:
                _raise_telemetry_storage_error()
            database_path = main_rows[0]["file"]
            if not isinstance(database_path, str) or not database_path:
                _raise_telemetry_storage_error()
            connected_path = Path(database_path).resolve(strict=True)
            expected_path = self.path.resolve(strict=True)
        except TelemetryStorageError:
            raise
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError):
            _raise_telemetry_storage_error()
        if connected_path != expected_path:
            _raise_telemetry_storage_error()

    def _connect(self) -> sqlite3.Connection:
        expected_ancestors = self._validate_parent_directory()
        expected_identity = self._validate_database_file()
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=5.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            self._validate_connected_database(connection)
            self._require_ancestor_snapshot(expected_ancestors)
            if self._validate_database_file() != expected_identity:
                _raise_telemetry_storage_error()
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            self._require_ancestor_snapshot(expected_ancestors)
            if self._validate_database_file() != expected_identity:
                _raise_telemetry_storage_error()
            return connection
        except TelemetryStorageError:
            if connection is not None:
                _close_connection_after_failure(connection)
            raise
        except (sqlite3.Error, OSError, ValueError):
            if connection is not None:
                _close_connection_after_failure(connection)
            _raise_telemetry_storage_error()
        except BaseException:
            if connection is not None:
                _close_connection_after_failure(connection)
            raise

    def _initialize_schema(self) -> None:
        with self._transaction() as connection:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            versions = connection.execute("SELECT version FROM telemetry_schema").fetchall()
            if not versions:
                connection.execute(
                    "INSERT INTO telemetry_schema(version) VALUES (?)",
                    (_SCHEMA_VERSION,),
                )
            elif len(versions) != 1 or versions[0]["version"] != _SCHEMA_VERSION:
                _raise_telemetry_storage_error()
            states = connection.execute(
                "SELECT event_count, head_hash FROM telemetry_state WHERE singleton = ?",
                (1,),
            ).fetchall()
            if not states:
                event_count = connection.execute(
                    "SELECT COUNT(*) FROM telemetry_events"
                ).fetchone()[0]
                if event_count != 0:
                    _raise_telemetry_storage_error()
                connection.execute(
                    """
                    INSERT INTO telemetry_state(singleton, event_count, head_hash)
                    VALUES (?, ?, ?)
                    """,
                    (1, 0, _ROOT_HASH),
                )
            elif len(states) != 1:
                _raise_telemetry_storage_error()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        except (sqlite3.Error, OSError):
            _close_connection_after_failure(connection)
            _raise_telemetry_storage_error()
        except BaseException:
            _close_connection_after_failure(connection)
            raise
        else:
            try:
                connection.close()
            except (sqlite3.Error, OSError):
                _raise_telemetry_storage_error()

    @contextmanager
    def _snapshot(self) -> Iterator[sqlite3.Connection]:
        with self._sqlite_context("BEGIN") as connection:
            yield connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._sqlite_context("BEGIN IMMEDIATE") as connection:
            yield connection

    @contextmanager
    def _sqlite_context(self, begin_statement: str) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute(begin_statement)
        except (sqlite3.Error, OSError):
            _close_connection_after_failure(connection)
            _raise_telemetry_storage_error()
        except BaseException:
            _close_connection_after_failure(connection)
            raise

        try:
            yield connection
        except (sqlite3.Error, OSError):
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            _raise_telemetry_storage_error()
        except BaseException:
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            raise

        try:
            connection.commit()
        except (sqlite3.Error, OSError):
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            _raise_telemetry_storage_error()
        except BaseException:
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            raise

        try:
            connection.close()
        except (sqlite3.Error, OSError):
            _raise_telemetry_storage_error()

    @staticmethod
    def _record(
        event: TelemetryEvent,
        sequence: int,
        previous_hash: str,
        event_hash: str,
    ) -> TelemetryRecord:
        try:
            return TelemetryRecord(
                **event.model_dump(mode="python"),
                sequence=sequence,
                previous_hash=previous_hash,
                event_hash=event_hash,
            )
        except (TypeError, ValueError, ValidationError):
            _raise_telemetry_integrity_error()


def _raise_telemetry_integrity_error() -> NoReturn:
    _raise_unlinked(TelemetryIntegrityError(_INTEGRITY_ERROR))


def _raise_telemetry_storage_error() -> NoReturn:
    _raise_unlinked(TelemetryStorageError(_STORAGE_ERROR))


def _require_integrity_integer(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("integrity integer is invalid")
    return value


def _require_integrity_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("integrity text is invalid")
    return value


def _require_integrity_hash(value: object) -> str:
    text = _require_integrity_text(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError("integrity hash is invalid")
    return text


def _close_descriptor_after_failure(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except BaseException:
        pass


def _rollback_after_failure(connection: sqlite3.Connection) -> None:
    try:
        connection.rollback()
    except BaseException:
        pass


def _close_connection_after_failure(connection: sqlite3.Connection) -> None:
    try:
        connection.close()
    except BaseException:
        pass


def _classify_exception(error: BaseException) -> ErrorClass:
    if isinstance(error, TimeoutError):
        return ErrorClass.TIMEOUT
    if isinstance(error, ValidationError):
        return ErrorClass.VALIDATION
    if isinstance(error, sqlite3.IntegrityError):
        return ErrorClass.INTEGRITY
    if isinstance(error, RuntimeError):
        return ErrorClass.RUNTIME
    return ErrorClass.INTERNAL


def _event_hash(
    sequence: int,
    previous_hash: str,
    event_json: str,
) -> str:
    encoded = f"{sequence}\n{previous_hash}\n{event_json}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "LocalTelemetrySink",
    "TelemetryEvent",
    "TelemetryEventName",
    "TelemetryIntegrityError",
    "TelemetryRecord",
    "TelemetryStorageError",
]
