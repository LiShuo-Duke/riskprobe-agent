"""Transactional local SQLite queue with strict privacy-safe contracts."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import sqlite3
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from riskprobe.privacy import assert_safe_payload, canonical_payload_hash

_SCHEMA_VERSION = 1
_CODE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PUBLIC_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_JOB_ID = re.compile(r"^job-[0-9a-f]{32}$")
_LEASE_TOKEN = re.compile(r"^lease-[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATTEMPTS = 100
_MAX_LEASE_SECONDS = 86_400.0
_STORAGE_ERROR = "queue storage is unavailable"
_INTEGRITY_ERROR = "queue integrity check failed"

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS queue_schema (
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL UNIQUE,
        idempotency_key TEXT NOT NULL UNIQUE,
        request_hash TEXT NOT NULL,
        request_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'running', 'succeeded', 'dead_lettered', 'cancelled')
        ),
        attempt INTEGER NOT NULL CHECK (attempt >= 0),
        max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
        lease_owner TEXT,
        lease_token TEXT,
        lease_expires_at REAL,
        result_json TEXT,
        result_hash TEXT,
        error_class TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        CHECK (
            (status = 'running' AND lease_owner IS NOT NULL
                AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR
            (status != 'running' AND lease_owner IS NULL
                AND lease_token IS NULL AND lease_expires_at IS NULL)
        ),
        CHECK (
            (status = 'succeeded' AND result_json IS NOT NULL AND result_hash IS NOT NULL)
            OR
            (status != 'succeeded' AND result_json IS NULL AND result_hash IS NULL)
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_claimable
    ON jobs(status, sequence)
    """,
)

Clock = Callable[[], datetime]
_StorageIdentity = tuple[int, int, int, int]
_AncestorSnapshot = tuple[tuple[Path, _StorageIdentity], ...]


class _UnsafeStoragePath(RuntimeError):
    """Internal marker for a storage path that fails the trust policy."""


class DeliveryError(RuntimeError):
    """Base class for fixed-message delivery failures."""


class DeliveryValidationError(DeliveryError):
    """Fixed-message rejection from an untrusted DTO validation boundary."""


class IdempotencyConflictError(DeliveryError):
    """Raised when one idempotency key is reused for a different submission."""


class InvalidJobTransitionError(DeliveryError):
    """Raised when a terminal or otherwise invalid state transition is requested."""


class JobNotFoundError(DeliveryError):
    """Raised without echoing an unavailable job identifier."""


class LeaseConflictError(DeliveryError):
    """Raised when lease ownership, token, state, or expiry is not current."""


class QueueIntegrityError(DeliveryError):
    """Raised when persisted queue content violates strict contracts."""


class QueueStorageError(DeliveryError):
    """Raised with a fixed message for unsafe or unavailable queue storage."""


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEAD_LETTERED = "dead_lettered"
    CANCELLED = "cancelled"


class ErrorClass(StrEnum):
    """The complete allowlist of persistable error classifications."""

    INTERNAL = "InternalError"
    INTEGRITY = "IntegrityError"
    LEASE_EXPIRED = "LeaseExpired"
    RUNTIME = "RuntimeError"
    TIMEOUT = "TimeoutError"
    VALIDATION = "ValidationError"


class _StrictDTO(BaseModel):
    """Strict DTO base; direct construction is for trusted Python values."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    @classmethod
    def safe_validate(cls, value: Mapping[str, object]) -> Self:
        """Validate an untrusted mapping without exposing its rejected content."""

        try:
            return cls.model_validate(value)
        except Exception:
            _raise_delivery_validation_error()


class JobRequest(_StrictDTO):
    """A privacy-checked request; use safe_validate for untrusted mappings."""

    kind: str
    payload: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("kind must be a public code")
        return value

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        normalized = _copy_string_mapping(value, message="request payload is not safe")
        try:
            assert_safe_payload({"payload": normalized})
        except Exception:
            raise ValueError("request payload is not safe") from None
        return MappingProxyType(_freeze_mapping(normalized))

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, object]) -> dict[str, object]:
        return _jsonable_mapping(value)

    @property
    def content_hash(self) -> str:
        return canonical_payload_hash({"kind": self.kind, "payload": self.payload})


class JobResult(_StrictDTO):
    """A safe aggregate result; use safe_validate for untrusted mappings."""

    summary: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        normalized = _copy_string_mapping(value, message="job result is not safe")
        try:
            assert_safe_payload({"summary": normalized})
        except Exception:
            raise ValueError("job result is not safe") from None
        return MappingProxyType(_freeze_mapping(normalized))

    @field_serializer("summary")
    def serialize_summary(self, value: Mapping[str, object]) -> dict[str, object]:
        return _jsonable_mapping(value)

    @property
    def content_hash(self) -> str:
        return canonical_payload_hash({"summary": self.summary})


class JobFailure(_StrictDTO):
    """Failure input that intentionally has no message field."""

    error_class: ErrorClass


class JobLease(_StrictDTO):
    """Opaque owner-bound capability for one currently running attempt."""

    job_id: str
    owner: str
    token: str
    expires_at: datetime

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return _require_job_id(value)

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, value: str) -> str:
        return _require_public_token(value, name="owner")

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        if _LEASE_TOKEN.fullmatch(value) is None:
            raise ValueError("lease token must be opaque")
        return value

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lease expiry must be timezone-aware")
        return value.astimezone(timezone.utc)


class JobSummary(_StrictDTO):
    """Payload-free state projection safe for application callers."""

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


class ClaimedJob(JobSummary):
    """Safe worker projection containing the validated request and lease."""

    request: JobRequest
    lease: JobLease


class CompletedJob(_StrictDTO):
    """One atomic successful queue snapshot and its verified safe result."""

    summary: JobSummary
    result: JobResult


class SQLiteQueue:
    """A finite-attempt FIFO queue using immediate SQLite transactions.

    Ancestor and file identity checks narrow pathname races, but stdlib
    SQLite does not expose the opened database file descriptor. Pure stdlib
    cannot fully eliminate races by a same-UID actor able to rewrite any path
    component; these checks are not a descriptor-relative no-follow guarantee.
    """

    def __init__(self, path: Path, *, clock: Clock | None = None) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        try:
            self.path = Path(path).absolute()
        except (OSError, TypeError, ValueError):
            _raise_queue_storage_error()
        self._clock = clock or _utc_now
        self._prepare_parent_directory()
        self._prepare_database_file()
        self._initialize_schema()

    def enqueue(
        self,
        request: JobRequest,
        *,
        idempotency_key: str,
        max_attempts: int = 3,
    ) -> JobSummary:
        if not isinstance(request, JobRequest):
            raise TypeError("request must be a JobRequest")
        key = _require_public_token(idempotency_key, name="idempotency_key")
        attempts = _require_max_attempts(max_attempts)
        request_hash = request.content_hash
        request_json = _canonical_json(request.model_dump(mode="json"))

        with self._transaction() as connection:
            now = self._now_timestamp()
            current = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if current is not None:
                if (
                    current["request_hash"] != request_hash
                    or current["request_json"] != request_json
                    or current["max_attempts"] != attempts
                ):
                    raise IdempotencyConflictError(
                        "idempotency key conflicts with existing request"
                    )
                return self._summary_from_row(current)

            job_id = f"job-{secrets.token_hex(16)}"
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, idempotency_key, request_hash, request_json,
                    status, attempt, max_attempts, lease_owner, lease_token,
                    lease_expires_at, result_json, result_hash, error_class,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)
                """,
                (
                    job_id,
                    key,
                    request_hash,
                    request_json,
                    JobStatus.PENDING.value,
                    0,
                    attempts,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                _raise_queue_integrity_error()
            return self._summary_from_row(row)

    def claim(self, *, owner: str, lease_seconds: float = 30.0) -> ClaimedJob | None:
        public_owner = _require_public_token(owner, name="owner")
        duration = _require_lease_seconds(lease_seconds)

        with self._transaction() as connection:
            now = self._now_timestamp()
            while True:
                row = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = ?
                       OR (status = ? AND lease_expires_at <= ?)
                    ORDER BY sequence
                    LIMIT 1
                    """,
                    (JobStatus.PENDING.value, JobStatus.RUNNING.value, now),
                ).fetchone()
                if row is None:
                    return None

                attempt = int(row["attempt"])
                max_attempts = int(row["max_attempts"])
                if attempt >= max_attempts:
                    error_class = (
                        ErrorClass.LEASE_EXPIRED
                        if row["status"] == JobStatus.RUNNING.value
                        else ErrorClass.INTERNAL
                    )
                    connection.execute(
                        """
                        UPDATE jobs
                        SET status = ?, lease_owner = NULL, lease_token = NULL,
                            lease_expires_at = NULL, error_class = ?, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (
                            JobStatus.DEAD_LETTERED.value,
                            error_class.value,
                            now,
                            row["job_id"],
                        ),
                    )
                    continue

                token = f"lease-{secrets.token_hex(16)}"
                expiry = now + duration
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, attempt = ?, lease_owner = ?, lease_token = ?,
                        lease_expires_at = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        JobStatus.RUNNING.value,
                        attempt + 1,
                        public_owner,
                        token,
                        expiry,
                        now,
                        row["job_id"],
                    ),
                )
                claimed = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (row["job_id"],),
                ).fetchone()
                if claimed is None:
                    _raise_queue_integrity_error()
                return self._claimed_from_row(claimed)

    def heartbeat(
        self,
        lease: JobLease,
        *,
        lease_seconds: float = 30.0,
    ) -> JobLease:
        if not isinstance(lease, JobLease):
            raise TypeError("lease must be a JobLease")
        duration = _require_lease_seconds(lease_seconds)
        with self._transaction() as connection:
            now = self._now_timestamp()
            row = self._require_active_lease(connection, lease, now=now)
            current_expiry = float(row["lease_expires_at"])
            expiry = max(current_expiry, now + duration)
            connection.execute(
                """
                UPDATE jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (expiry, now, row["job_id"]),
            )
        return JobLease(
            job_id=lease.job_id,
            owner=lease.owner,
            token=lease.token,
            expires_at=_from_timestamp(expiry),
        )

    def succeed(self, lease: JobLease, result: JobResult) -> JobSummary:
        if not isinstance(lease, JobLease):
            raise TypeError("lease must be a JobLease")
        if not isinstance(result, JobResult):
            raise TypeError("result must be a JobResult")
        result_json = _canonical_json(result.model_dump(mode="json"))
        result_hash = result.content_hash
        with self._transaction() as connection:
            now = self._now_timestamp()
            row = self._require_active_lease(connection, lease, now=now)
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, result_json = ?, result_hash = ?,
                    error_class = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    result_json,
                    result_hash,
                    now,
                    row["job_id"],
                ),
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (row["job_id"],),
            ).fetchone()
            if updated is None:
                _raise_queue_integrity_error()
            return self._summary_from_row(updated)

    ack = succeed

    def retry(self, lease: JobLease, failure: JobFailure) -> JobSummary:
        if not isinstance(lease, JobLease):
            raise TypeError("lease must be a JobLease")
        if not isinstance(failure, JobFailure):
            raise TypeError("failure must be a JobFailure")
        with self._transaction() as connection:
            now = self._now_timestamp()
            row = self._require_active_lease(connection, lease, now=now)
            status = (
                JobStatus.DEAD_LETTERED
                if int(row["attempt"]) >= int(row["max_attempts"])
                else JobStatus.PENDING
            )
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, error_class = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status.value,
                    failure.error_class.value,
                    now,
                    row["job_id"],
                ),
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (row["job_id"],),
            ).fetchone()
            if updated is None:
                _raise_queue_integrity_error()
            return self._summary_from_row(updated)

    def cancel(self, job_id: str) -> JobSummary:
        public_job_id = _require_job_id(job_id)
        with self._transaction() as connection:
            now = self._now_timestamp()
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (public_job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFoundError("job is unavailable")
            status = self._status_from_value(row["status"])
            if status is JobStatus.CANCELLED:
                return self._summary_from_row(row)
            if status not in {JobStatus.PENDING, JobStatus.RUNNING}:
                raise InvalidJobTransitionError("job transition is unavailable")
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (JobStatus.CANCELLED.value, now, public_job_id),
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (public_job_id,),
            ).fetchone()
            if updated is None:
                _raise_queue_integrity_error()
            return self._summary_from_row(updated)

    def get(self, job_id: str) -> JobSummary:
        public_job_id = _require_job_id(job_id)
        with self._connection() as connection:
            row = self._job_row(connection, public_job_id)
            return self._summary_from_row(row)

    status = get

    def completed(self, job_id: str) -> CompletedJob | None:
        """Return summary and verified result from one SQLite read snapshot."""

        public_job_id = _require_job_id(job_id)
        with self._connection() as connection:
            row = self._job_row(connection, public_job_id)
            summary = self._summary_from_row(row)
            if summary.status is not JobStatus.SUCCEEDED:
                return None
            return CompletedJob(summary=summary, result=self._result_from_row(row))

    def result(self, job_id: str) -> JobResult | None:
        completed = self.completed(job_id)
        return completed.result if completed is not None else None

    def dead_letters(self) -> tuple[JobSummary, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY sequence",
                (JobStatus.DEAD_LETTERED.value,),
            ).fetchall()
            return tuple(self._summary_from_row(row) for row in rows)

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
            _raise_queue_storage_error()

    def _validate_parent_directory(self) -> _AncestorSnapshot:
        try:
            return _storage_ancestor_snapshot(
                self.path.parent,
                allow_missing=False,
            )
        except (OSError, ValueError, _UnsafeStoragePath):
            _raise_queue_storage_error()

    def _require_ancestor_snapshot(self, expected: _AncestorSnapshot) -> None:
        if self._validate_parent_directory() != expected:
            _raise_queue_storage_error()

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
                    _raise_queue_storage_error()
                descriptor_identity = (
                    details.st_dev,
                    details.st_ino,
                    details.st_uid,
                    mode,
                )
                if self._validate_database_file() != descriptor_identity:
                    _raise_queue_storage_error()
            except (OSError, ValueError):
                if descriptor is not None:
                    _close_descriptor_after_failure(descriptor)
                _raise_queue_storage_error()
            except BaseException:
                if descriptor is not None:
                    _close_descriptor_after_failure(descriptor)
                raise
            else:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except (OSError, ValueError):
                        _raise_queue_storage_error()
        except (OSError, ValueError):
            _raise_queue_storage_error()
        self._require_ancestor_snapshot(expected_ancestors)
        self._validate_database_file()

    def _validate_database_file(self) -> _StorageIdentity:
        try:
            details = self.path.lstat()
        except (OSError, ValueError):
            _raise_queue_storage_error()
        mode = stat.S_IMODE(details.st_mode)
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or mode != 0o600
        ):
            _raise_queue_storage_error()
        return (details.st_dev, details.st_ino, details.st_uid, mode)

    def _validate_connected_database(self, connection: sqlite3.Connection) -> None:
        try:
            rows = connection.execute("PRAGMA database_list").fetchall()
            main_rows = tuple(row for row in rows if row["name"] == "main")
            if len(main_rows) != 1:
                _raise_queue_storage_error()
            database_path = main_rows[0]["file"]
            if not isinstance(database_path, str) or not database_path:
                _raise_queue_storage_error()
            connected_path = Path(database_path).resolve(strict=True)
            expected_path = self.path.resolve(strict=True)
        except QueueStorageError:
            raise
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError):
            _raise_queue_storage_error()
        if connected_path != expected_path:
            _raise_queue_storage_error()

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
                _raise_queue_storage_error()
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            self._require_ancestor_snapshot(expected_ancestors)
            if self._validate_database_file() != expected_identity:
                _raise_queue_storage_error()
            return connection
        except QueueStorageError:
            if connection is not None:
                _close_connection_after_failure(connection)
            raise
        except (sqlite3.Error, OSError, ValueError):
            if connection is not None:
                _close_connection_after_failure(connection)
            _raise_queue_storage_error()
        except BaseException:
            if connection is not None:
                _close_connection_after_failure(connection)
            raise

    def _initialize_schema(self) -> None:
        with self._transaction() as connection:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            versions = connection.execute("SELECT version FROM queue_schema").fetchall()
            if not versions:
                connection.execute(
                    "INSERT INTO queue_schema(version) VALUES (?)",
                    (_SCHEMA_VERSION,),
                )
            elif len(versions) != 1 or versions[0]["version"] != _SCHEMA_VERSION:
                _raise_queue_storage_error()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        except (sqlite3.Error, OSError):
            _close_connection_after_failure(connection)
            _raise_queue_storage_error()
        except BaseException:
            _close_connection_after_failure(connection)
            raise
        else:
            try:
                connection.close()
            except (sqlite3.Error, OSError):
                _raise_queue_storage_error()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
        except (sqlite3.Error, OSError):
            _close_connection_after_failure(connection)
            _raise_queue_storage_error()
        except BaseException:
            _close_connection_after_failure(connection)
            raise

        try:
            yield connection
        except (sqlite3.Error, OSError):
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            _raise_queue_storage_error()
        except BaseException:
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            raise

        try:
            connection.commit()
        except (sqlite3.Error, OSError):
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            _raise_queue_storage_error()
        except BaseException:
            _rollback_after_failure(connection)
            _close_connection_after_failure(connection)
            raise

        try:
            connection.close()
        except (sqlite3.Error, OSError):
            _raise_queue_storage_error()

    @staticmethod
    def _job_row(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise JobNotFoundError("job is unavailable")
        return row

    def _require_active_lease(
        self,
        connection: sqlite3.Connection,
        lease: JobLease,
        *,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (lease.job_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != JobStatus.RUNNING.value
            or row["lease_owner"] != lease.owner
            or row["lease_token"] != lease.token
            or not isinstance(row["lease_expires_at"], (int, float))
            or not math.isfinite(float(row["lease_expires_at"]))
            or float(row["lease_expires_at"]) <= now
        ):
            raise LeaseConflictError("job lease is unavailable")
        return row

    def _claimed_from_row(self, row: sqlite3.Row) -> ClaimedJob:
        summary = self._summary_from_row(row)
        try:
            request_json = row["request_json"]
            if not isinstance(request_json, str):
                raise ValueError("request JSON is invalid")
            request = JobRequest.model_validate_json(request_json)
            if (
                request.content_hash != summary.request_hash
                or _canonical_json(request.model_dump(mode="json")) != request_json
            ):
                raise ValueError("request hash does not match")
            lease = JobLease(
                job_id=summary.job_id,
                owner=str(row["lease_owner"]),
                token=str(row["lease_token"]),
                expires_at=_from_timestamp(float(row["lease_expires_at"])),
            )
            return ClaimedJob(
                **summary.model_dump(mode="python"),
                request=request,
                lease=lease,
            )
        except (TypeError, ValueError, ValidationError):
            _raise_queue_integrity_error()

    def _result_from_row(self, row: sqlite3.Row) -> JobResult:
        result_json = row["result_json"]
        result_hash = row["result_hash"]
        if not isinstance(result_json, str) or not isinstance(result_hash, str):
            _raise_queue_integrity_error()
        try:
            result = JobResult.model_validate_json(result_json)
            if (
                result.content_hash != result_hash
                or _canonical_json(result.model_dump(mode="json")) != result_json
            ):
                raise ValueError("result hash does not match")
            return result
        except (TypeError, ValueError, ValidationError):
            _raise_queue_integrity_error()

    def _summary_from_row(self, row: sqlite3.Row) -> JobSummary:
        try:
            status = self._status_from_value(row["status"])
            attempt = int(row["attempt"])
            if (
                status
                in {
                    JobStatus.RUNNING,
                    JobStatus.SUCCEEDED,
                    JobStatus.DEAD_LETTERED,
                }
                and attempt < 1
            ):
                raise ValueError("active and completed jobs require an attempt")
            error_class = (
                ErrorClass(str(row["error_class"]))
                if row["error_class"] is not None
                else None
            )
            return JobSummary(
                job_id=str(row["job_id"]),
                status=status,
                attempt=attempt,
                request_hash=str(row["request_hash"]),
                result_hash=(
                    str(row["result_hash"]) if row["result_hash"] is not None else None
                ),
                error_class=error_class,
            )
        except (TypeError, ValueError, ValidationError):
            _raise_queue_integrity_error()

    @staticmethod
    def _status_from_value(value: object) -> JobStatus:
        try:
            return JobStatus(str(value))
        except ValueError:
            _raise_queue_integrity_error()

    def _now_timestamp(self) -> float:
        try:
            value = self._clock()
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                _raise_queue_storage_error()
            timestamp = value.astimezone(timezone.utc).timestamp()
            if not math.isfinite(timestamp):
                _raise_queue_storage_error()
            return timestamp
        except DeliveryError:
            raise
        except Exception:
            _raise_queue_storage_error()


def _raise_unlinked(error: BaseException) -> NoReturn:
    """Raise a public exception without retaining the active exception graph."""

    try:
        raise error from None
    finally:
        error.__cause__ = None
        error.__context__ = None


def _raise_delivery_validation_error() -> NoReturn:
    _raise_unlinked(DeliveryValidationError("delivery input is invalid"))


def _raise_queue_integrity_error() -> NoReturn:
    _raise_unlinked(QueueIntegrityError(_INTEGRITY_ERROR))


def _raise_queue_storage_error() -> NoReturn:
    _raise_unlinked(QueueStorageError(_STORAGE_ERROR))


def _storage_ancestor_snapshot(
    directory: Path,
    *,
    allow_missing: bool,
) -> _AncestorSnapshot:
    paths = tuple(reversed((directory, *directory.parents)))
    snapshot: list[tuple[Path, _StorageIdentity]] = []
    missing_component = False
    effective_uid = os.geteuid()

    for index, component in enumerate(paths):
        try:
            details = component.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise _UnsafeStoragePath
            missing_component = True
            continue

        if missing_component:
            raise _UnsafeStoragePath

        mode = stat.S_IMODE(details.st_mode)
        is_direct_parent = index == len(paths) - 1
        owner_is_trusted = (
            details.st_uid == effective_uid
            if is_direct_parent
            else details.st_uid in {0, effective_uid}
        )
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISDIR(details.st_mode)
            or not owner_is_trusted
            or mode & 0o022
        ):
            raise _UnsafeStoragePath
        identity = (details.st_dev, details.st_ino, details.st_uid, mode)
        snapshot.append((component, identity))

    return tuple(snapshot)


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


def _require_public_token(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _PUBLIC_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a public identifier")
    return value


def _require_job_id(value: object) -> str:
    if not isinstance(value, str) or _JOB_ID.fullmatch(value) is None:
        raise ValueError("job_id must be a public identifier")
    return value


def _require_max_attempts(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_ATTEMPTS
    ):
        raise ValueError("max_attempts must be a bounded positive integer")
    return value


def _require_lease_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lease_seconds must be a bounded positive number")
    duration = float(value)
    if not math.isfinite(duration) or not 0 < duration <= _MAX_LEASE_SECONDS:
        raise ValueError("lease_seconds must be a bounded positive number")
    return duration


def _copy_string_mapping(
    value: Mapping[str, object],
    *,
    message: str,
) -> dict[str, object]:
    try:
        normalized = dict(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if any(not isinstance(key, str) for key in normalized):
        raise ValueError(message)
    return normalized


def _freeze_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _freeze_value(item) for key, item in sorted(value.items())}


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        normalized = _copy_string_mapping(value, message="payload is not safe")
        return MappingProxyType(_freeze_mapping(normalized))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _jsonable_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _jsonable(item) for key, item in value.items()}


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _from_timestamp(value: float) -> datetime:
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "ClaimedJob",
    "CompletedJob",
    "DeliveryError",
    "DeliveryValidationError",
    "ErrorClass",
    "IdempotencyConflictError",
    "InvalidJobTransitionError",
    "JobFailure",
    "JobLease",
    "JobNotFoundError",
    "JobRequest",
    "JobResult",
    "JobStatus",
    "JobSummary",
    "LeaseConflictError",
    "QueueIntegrityError",
    "QueueStorageError",
    "SQLiteQueue",
]
