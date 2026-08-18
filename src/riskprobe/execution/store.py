"""SQLite-backed transactional execution state and event storage."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from riskprobe.execution.models import (
    ArtifactRef,
    NodeCheckpoint,
    NodeStatus,
    RetryPolicy,
    RunBudget,
    RunIdentity,
    RunStatus,
)

_SCHEMA_VERSION = 1


class ExecutionStore:
    """Store node state, events, and checkpoint refs in one SQLite transaction."""

    def __init__(
        self,
        runs_dir: Path,
        run_id: str,
        *,
        retry_policy: RetryPolicy | None = None,
        budget: RunBudget | None = None,
    ) -> None:
        self.identity = RunIdentity(run_id)
        self.run_id = run_id
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.runs_dir / f".{run_id}.runtime.sqlite3"
        self.retry_policy = retry_policy or RetryPolicy()
        self.budget = budget or RunBudget()
        self._prepare_database_file()
        self._initialize_schema()
        self._recover_stale_running_nodes()

    @staticmethod
    def database_path_for(runs_dir: Path, run_id: str) -> Path:
        RunIdentity(run_id)
        return Path(runs_dir).resolve() / f".{run_id}.runtime.sqlite3"

    @staticmethod
    def is_secure_database(path: Path) -> bool:
        try:
            ExecutionStore._validate_database_file(Path(path))
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _validate_database_file(path: Path) -> None:
        try:
            details = path.lstat()
        except OSError as error:
            raise RuntimeError("runtime database must be a secure regular file") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise RuntimeError("runtime database must be a secure regular file")
        if details.st_uid != os.geteuid():
            raise RuntimeError("runtime database owner does not match current user")
        if stat.S_IMODE(details.st_mode) != 0o600:
            raise RuntimeError("runtime database permissions must be 0600")

    def _prepare_database_file(self) -> None:
        try:
            self.database_path.lstat()
        except FileNotFoundError:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.database_path, flags, 0o600)
            except OSError as error:
                raise RuntimeError(
                    "runtime database must be a secure regular file"
                ) from error
            try:
                os.fchmod(descriptor, 0o600)
                details = os.fstat(descriptor)
                if not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid():
                    raise RuntimeError(
                        "runtime database must be a secure regular file"
                    )
            finally:
                os.close(descriptor)
        self._validate_database_file(self.database_path)

    def _connect(self) -> sqlite3.Connection:
        self._validate_database_file(self.database_path)
        try:
            connection = sqlite3.connect(
                self.database_path,
                timeout=5.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
        except sqlite3.DatabaseError as error:
            try:
                connection.close()
            except UnboundLocalError:
                pass
            raise RuntimeError(
                f"runtime database for run {self.run_id} is invalid"
            ) from error
        self._validate_database_file(self.database_path)
        return connection

    def _initialize_schema(self) -> None:
        try:
            with self._transaction() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_schema (
                        version INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS nodes (
                        run_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        input_fingerprint TEXT NOT NULL,
                        output_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, node_id),
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        node_id TEXT,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        input_fingerprint TEXT NOT NULL,
                        error_class TEXT,
                        metadata_json TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        PRIMARY KEY (run_id, sequence),
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    );
                    CREATE TABLE IF NOT EXISTS checkpoint_artifacts (
                        run_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        filename TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        schema_version TEXT NOT NULL,
                        PRIMARY KEY (run_id, node_id, filename),
                        FOREIGN KEY (run_id, node_id)
                            REFERENCES nodes(run_id, node_id) ON DELETE CASCADE
                    );
                    """
                )
                versions = connection.execute(
                    "SELECT version FROM runtime_schema"
                ).fetchall()
                if not versions:
                    connection.execute(
                        "INSERT INTO runtime_schema(version) VALUES (?)",
                        (_SCHEMA_VERSION,),
                    )
                elif len(versions) != 1 or versions[0]["version"] != _SCHEMA_VERSION:
                    raise RuntimeError("runtime database schema version is unsupported")
                run_ids = [
                    row["run_id"]
                    for row in connection.execute("SELECT run_id FROM runs").fetchall()
                ]
                now = self._timestamp()
                if not run_ids:
                    connection.execute(
                        """
                        INSERT INTO runs(run_id, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (self.run_id, RunStatus.PENDING.value, now, now),
                    )
                elif run_ids != [self.run_id]:
                    raise RuntimeError(
                        "runtime database belongs to a different run_id"
                    )
        except RuntimeError:
            raise
        except sqlite3.DatabaseError as error:
            raise RuntimeError(
                f"runtime database for run {self.run_id} is invalid"
            ) from error

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _canonical_json(payload: Any) -> str:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _validate_node_id(node_id: str) -> None:
        if (
            not isinstance(node_id, str)
            or not node_id
            or node_id.startswith(".")
            or "/" in node_id
            or "\\" in node_id
        ):
            raise ValueError("node_id must be a non-empty public name")

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        node_id: str | None,
        event_type: str,
        status: NodeStatus | RunStatus,
        attempt: int,
        input_fingerprint: str,
        error_class: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO events(
                run_id, sequence, node_id, event_type, status, attempt,
                input_fingerprint, error_class, metadata_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                sequence,
                node_id,
                event_type,
                status.value,
                attempt,
                input_fingerprint,
                error_class,
                self._canonical_json(dict(metadata or {})),
                self._timestamp(),
            ),
        )

    def _recover_stale_running_nodes(self) -> None:
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT node_id, attempt, input_fingerprint
                FROM nodes
                WHERE run_id = ? AND status = ?
                ORDER BY node_id
                """,
                (self.run_id, NodeStatus.RUNNING.value),
            ).fetchall()
            if not rows:
                return
            now = self._timestamp()
            for row in rows:
                self._append_event(
                    connection,
                    node_id=row["node_id"],
                    event_type="node_interrupted",
                    status=NodeStatus.INTERRUPTED,
                    attempt=row["attempt"],
                    input_fingerprint=row["input_fingerprint"],
                )
                connection.execute(
                    """
                    UPDATE nodes SET status = ?, updated_at = ?
                    WHERE run_id = ? AND node_id = ?
                    """,
                    (
                        NodeStatus.INTERRUPTED.value,
                        now,
                        self.run_id,
                        row["node_id"],
                    ),
                )
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.INTERRUPTED.value, now, self.run_id),
            )

    def start_node(self, node_id: str, *, input_fingerprint: str) -> int:
        self._validate_node_id(node_id)
        if not isinstance(input_fingerprint, str) or not input_fingerprint:
            raise ValueError("input_fingerprint must be non-empty")
        self.budget.ensure_before_deadline()
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (self.run_id,)
            ).fetchone()
            if run is None:
                raise RuntimeError("runtime run identity is missing")
            if run["status"] == RunStatus.CANCELLED.value:
                raise RuntimeError("run is cancelled")
            current = connection.execute(
                "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            ).fetchone()
            if current is None:
                node_count = connection.execute(
                    "SELECT COUNT(*) FROM nodes WHERE run_id = ?",
                    (self.run_id,),
                ).fetchone()[0]
                if self.budget.max_nodes is not None and node_count >= self.budget.max_nodes:
                    raise RuntimeError("run node budget has been exceeded")
            total_attempts = connection.execute(
                "SELECT COALESCE(SUM(attempt), 0) FROM nodes WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()[0]
            if (
                self.budget.max_attempts is not None
                and total_attempts >= self.budget.max_attempts
            ):
                raise RuntimeError("run attempt budget has been exceeded")
            if current is not None and current["status"] == NodeStatus.RUNNING.value:
                raise ValueError(f"node {node_id} is already running")
            previous_attempt = current["attempt"] if current is not None else 0
            attempt = previous_attempt + 1
            if attempt > self.retry_policy.max_attempts:
                raise RuntimeError(f"node {node_id} exhausted retry attempts")
            metadata: dict[str, Any] = {}
            if attempt > 1:
                metadata["backoff_seconds"] = self.retry_policy.backoff_for_attempt(
                    attempt
                )
            self._append_event(
                connection,
                node_id=node_id,
                event_type="node_started",
                status=NodeStatus.RUNNING,
                attempt=attempt,
                input_fingerprint=input_fingerprint,
                metadata=metadata,
            )
            now = self._timestamp()
            connection.execute(
                """
                INSERT INTO nodes(
                    run_id, node_id, status, attempt, input_fingerprint,
                    output_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                    status = excluded.status,
                    attempt = excluded.attempt,
                    input_fingerprint = excluded.input_fingerprint,
                    output_json = excluded.output_json,
                    updated_at = excluded.updated_at
                """,
                (
                    self.run_id,
                    node_id,
                    NodeStatus.RUNNING.value,
                    attempt,
                    input_fingerprint,
                    "{}",
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM checkpoint_artifacts WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            )
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.RUNNING.value, now, self.run_id),
            )
            return attempt

    def succeed_node(
        self,
        node_id: str,
        *,
        input_fingerprint: str,
        output: Mapping[str, Any],
        artifact_refs: Iterable[ArtifactRef] = (),
    ) -> NodeCheckpoint:
        self._validate_node_id(node_id)
        if not isinstance(output, Mapping):
            raise TypeError("checkpoint output must be a mapping")
        rendered_output = self._canonical_json(dict(output))
        references = tuple(artifact_refs)
        if not all(isinstance(reference, ArtifactRef) for reference in references):
            raise TypeError("artifact_refs must contain ArtifactRef values")
        if len({reference.filename for reference in references}) != len(references):
            raise ValueError("artifact_refs must have unique filenames")
        with self._transaction() as connection:
            current = self._require_running(
                connection, node_id, input_fingerprint=input_fingerprint
            )
            attempt = current["attempt"]
            self._append_event(
                connection,
                node_id=node_id,
                event_type="node_succeeded",
                status=NodeStatus.SUCCEEDED,
                attempt=attempt,
                input_fingerprint=input_fingerprint,
            )
            now = self._timestamp()
            connection.execute(
                """
                UPDATE nodes
                SET status = ?, output_json = ?, updated_at = ?
                WHERE run_id = ? AND node_id = ?
                """,
                (
                    NodeStatus.SUCCEEDED.value,
                    rendered_output,
                    now,
                    self.run_id,
                    node_id,
                ),
            )
            connection.execute(
                "DELETE FROM checkpoint_artifacts WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            )
            connection.executemany(
                """
                INSERT INTO checkpoint_artifacts(
                    run_id, node_id, attempt, filename, sha256, size, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        self.run_id,
                        node_id,
                        attempt,
                        reference.filename,
                        reference.sha256,
                        reference.size,
                        reference.schema_version,
                    )
                    for reference in references
                ],
            )
            run_status = (
                RunStatus.SUCCEEDED
                if node_id == "finalize"
                else RunStatus.RUNNING
            )
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (run_status.value, now, self.run_id),
            )
        checkpoint = self.checkpoint(node_id, input_fingerprint=input_fingerprint)
        if checkpoint is None:
            raise RuntimeError("checkpoint was not persisted")
        return checkpoint

    def fail_node(
        self,
        node_id: str,
        *,
        input_fingerprint: str,
        error_class: str,
    ) -> None:
        self._validate_node_id(node_id)
        if not isinstance(error_class, str) or not error_class:
            raise ValueError("error_class must be non-empty")
        with self._transaction() as connection:
            current = self._require_running(
                connection, node_id, input_fingerprint=input_fingerprint
            )
            self._append_event(
                connection,
                node_id=node_id,
                event_type="node_failed",
                status=NodeStatus.FAILED,
                attempt=current["attempt"],
                input_fingerprint=input_fingerprint,
                error_class=error_class,
            )
            now = self._timestamp()
            connection.execute(
                """
                UPDATE nodes SET status = ?, updated_at = ?
                WHERE run_id = ? AND node_id = ?
                """,
                (NodeStatus.FAILED.value, now, self.run_id, node_id),
            )
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.FAILED.value, now, self.run_id),
            )

    def _require_running(
        self,
        connection: sqlite3.Connection,
        node_id: str,
        *,
        input_fingerprint: str,
    ) -> sqlite3.Row:
        current = connection.execute(
            "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
            (self.run_id, node_id),
        ).fetchone()
        if current is None or current["status"] != NodeStatus.RUNNING.value:
            raise ValueError(f"node {node_id} must be running")
        if current["input_fingerprint"] != input_fingerprint:
            raise ValueError(f"node {node_id} input fingerprint changed while running")
        return current

    def checkpoint(
        self, node_id: str, *, input_fingerprint: str
    ) -> NodeCheckpoint | None:
        self._validate_node_id(node_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            ).fetchone()
            if (
                row is None
                or row["status"] != NodeStatus.SUCCEEDED.value
                or row["input_fingerprint"] != input_fingerprint
            ):
                return None
            return self._checkpoint_from_row(connection, row)

    def load_verified_checkpoint(
        self,
        node_id: str,
        *,
        input_fingerprint: str,
        run_dir: Path,
        expected_artifacts: Mapping[str, str] | None = None,
    ) -> NodeCheckpoint | None:
        checkpoint = self.checkpoint(
            node_id, input_fingerprint=input_fingerprint
        )
        if checkpoint is None:
            return None
        expected = dict(expected_artifacts) if expected_artifacts is not None else None
        if expected is not None:
            actual = {
                reference.filename: reference.schema_version
                for reference in checkpoint.artifact_refs
            }
            if actual != expected:
                return None
        for reference in checkpoint.artifact_refs:
            expected_schema = (
                expected.get(reference.filename) if expected is not None else None
            )
            if not reference.verify(
                run_dir, expected_schema_version=expected_schema
            ):
                return None
        return checkpoint

    verified_checkpoint = load_verified_checkpoint

    def _checkpoint_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> NodeCheckpoint:
        try:
            output = json.loads(row["output_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("runtime checkpoint output is invalid") from error
        if not isinstance(output, dict):
            raise RuntimeError("runtime checkpoint output is invalid")
        artifact_rows = connection.execute(
            """
            SELECT filename, sha256, size, schema_version
            FROM checkpoint_artifacts
            WHERE run_id = ? AND node_id = ?
            ORDER BY filename
            """,
            (self.run_id, row["node_id"]),
        ).fetchall()
        try:
            references = tuple(
                ArtifactRef(
                    filename=artifact["filename"],
                    sha256=artifact["sha256"],
                    size=artifact["size"],
                    schema_version=artifact["schema_version"],
                )
                for artifact in artifact_rows
            )
            return NodeCheckpoint(
                node_id=row["node_id"],
                status=NodeStatus(row["status"]),
                attempt=row["attempt"],
                input_fingerprint=row["input_fingerprint"],
                output=output,
                updated_at=row["updated_at"],
                artifact_refs=references,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("runtime checkpoint is invalid") from error

    def node_status(self, node_id: str) -> NodeStatus:
        self._validate_node_id(node_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM nodes WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            ).fetchone()
        if row is None:
            return NodeStatus.PENDING
        try:
            return NodeStatus(row["status"])
        except ValueError as error:
            raise RuntimeError(f"runtime node {node_id} has an invalid status") from error

    def run_status(self) -> RunStatus:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (self.run_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError("runtime run identity is missing")
        try:
            return RunStatus(row["status"])
        except ValueError as error:
            raise RuntimeError("runtime run status is invalid") from error

    def events(self, node_id: str | None = None) -> list[dict[str, Any]]:
        if node_id is not None:
            self._validate_node_id(node_id)
        query = "SELECT * FROM events WHERE run_id = ?"
        parameters: list[Any] = [self.run_id]
        if node_id is not None:
            query += " AND node_id = ?"
            parameters.append(node_id)
        query += " ORDER BY sequence"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError("runtime event metadata is invalid") from error
            event: dict[str, Any] = {
                "attempt": row["attempt"],
                "event_type": row["event_type"],
                "input_fingerprint": row["input_fingerprint"],
                "node_id": row["node_id"],
                "run_id": row["run_id"],
                "sequence": row["sequence"],
                "status": row["status"],
                "timestamp": row["timestamp"],
            }
            if row["error_class"] is not None:
                event["error_class"] = row["error_class"]
            if metadata:
                event["metadata"] = metadata
            events.append(event)
        return events

    trace = events

    def invalidate_from(
        self, node_id: str, downstream: Iterable[str] = ()
    ) -> None:
        node_ids = tuple(dict.fromkeys((node_id, *tuple(downstream))))
        for candidate in node_ids:
            self._validate_node_id(candidate)
        with self._transaction() as connection:
            now = self._timestamp()
            for candidate in node_ids:
                row = connection.execute(
                    "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
                    (self.run_id, candidate),
                ).fetchone()
                if row is None:
                    continue
                self._append_event(
                    connection,
                    node_id=candidate,
                    event_type="node_invalidated",
                    status=NodeStatus.INVALIDATED,
                    attempt=row["attempt"],
                    input_fingerprint=row["input_fingerprint"],
                )
                connection.execute(
                    """
                    UPDATE nodes
                    SET status = ?, output_json = ?, updated_at = ?
                    WHERE run_id = ? AND node_id = ?
                    """,
                    (
                        NodeStatus.INVALIDATED.value,
                        "{}",
                        now,
                        self.run_id,
                        candidate,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM checkpoint_artifacts
                    WHERE run_id = ? AND node_id = ?
                    """,
                    (self.run_id, candidate),
                )
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.PENDING.value, now, self.run_id),
            )

    def cancel(self, node_id: str | None = None) -> None:
        if node_id is not None:
            self._validate_node_id(node_id)
        with self._transaction() as connection:
            now = self._timestamp()
            if node_id is None:
                rows = connection.execute(
                    """
                    SELECT * FROM nodes
                    WHERE run_id = ? AND status = ?
                    ORDER BY node_id
                    """,
                    (self.run_id, NodeStatus.RUNNING.value),
                ).fetchall()
                for row in rows:
                    self._cancel_node(connection, row, now)
                self._append_event(
                    connection,
                    node_id=None,
                    event_type="run_cancelled",
                    status=RunStatus.CANCELLED,
                    attempt=0,
                    input_fingerprint="",
                )
                connection.execute(
                    "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                    (RunStatus.CANCELLED.value, now, self.run_id),
                )
                return
            row = connection.execute(
                "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"node {node_id} does not exist")
            self._cancel_node(connection, row, now)

    def _cancel_node(
        self, connection: sqlite3.Connection, row: sqlite3.Row, now: str
    ) -> None:
        self._append_event(
            connection,
            node_id=row["node_id"],
            event_type="node_cancelled",
            status=NodeStatus.CANCELLED,
            attempt=row["attempt"],
            input_fingerprint=row["input_fingerprint"],
        )
        connection.execute(
            """
            UPDATE nodes SET status = ?, updated_at = ?
            WHERE run_id = ? AND node_id = ?
            """,
            (
                NodeStatus.CANCELLED.value,
                now,
                self.run_id,
                row["node_id"],
            ),
        )
        connection.execute(
            "DELETE FROM checkpoint_artifacts WHERE run_id = ? AND node_id = ?",
            (self.run_id, row["node_id"]),
        )

    def reconcile_published(
        self,
        *,
        node_id: str,
        input_fingerprint: str,
        output: Mapping[str, Any],
        artifact_refs: Iterable[ArtifactRef] = (),
    ) -> NodeCheckpoint:
        self._validate_node_id(node_id)
        references = tuple(artifact_refs)
        rendered_output = self._canonical_json(dict(output))
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            ).fetchone()
            if current is not None and current["status"] == NodeStatus.SUCCEEDED.value:
                connection.execute(
                    "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                    (RunStatus.SUCCEEDED.value, self._timestamp(), self.run_id),
                )
                checkpoint = self._checkpoint_from_row(connection, current)
                return checkpoint
            attempt = max(1, current["attempt"] if current is not None else 0)
            self._append_event(
                connection,
                node_id=node_id,
                event_type="run_reconciled",
                status=NodeStatus.SUCCEEDED,
                attempt=attempt,
                input_fingerprint=input_fingerprint,
            )
            now = self._timestamp()
            connection.execute(
                """
                INSERT INTO nodes(
                    run_id, node_id, status, attempt, input_fingerprint,
                    output_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                    status = excluded.status,
                    attempt = excluded.attempt,
                    input_fingerprint = excluded.input_fingerprint,
                    output_json = excluded.output_json,
                    updated_at = excluded.updated_at
                """,
                (
                    self.run_id,
                    node_id,
                    NodeStatus.SUCCEEDED.value,
                    attempt,
                    input_fingerprint,
                    rendered_output,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM checkpoint_artifacts WHERE run_id = ? AND node_id = ?",
                (self.run_id, node_id),
            )
            connection.executemany(
                """
                INSERT INTO checkpoint_artifacts(
                    run_id, node_id, attempt, filename, sha256, size, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        self.run_id,
                        node_id,
                        attempt,
                        reference.filename,
                        reference.sha256,
                        reference.size,
                        reference.schema_version,
                    )
                    for reference in references
                ],
            )
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.SUCCEEDED.value, now, self.run_id),
            )
        checkpoint = self.checkpoint(node_id, input_fingerprint=input_fingerprint)
        if checkpoint is None:
            raise RuntimeError("published run reconciliation failed")
        return checkpoint
