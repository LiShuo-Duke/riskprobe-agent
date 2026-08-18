"""SQLite-backed append-only evidence storage with per-run hash chains."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from riskprobe.evidence.models import (
    EvidenceIntegrityError,
    EvidenceParentError,
    EvidenceRecord,
    PrivacyClass,
    UnsafeEvidenceError,
    assert_safe_payload,
)


_ROOT_HASH = "0" * 64
_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    parent_ids_json TEXT NOT NULL,
    artifact_hashes_json TEXT NOT NULL,
    privacy_class TEXT NOT NULL,
    producer_version TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    chain_hash TEXT NOT NULL,
    content_json TEXT NOT NULL,
    UNIQUE (run_id, sequence)
);
CREATE INDEX IF NOT EXISTS evidence_records_run_sequence
    ON evidence_records (run_id, sequence);
"""

SafePayloadHook = Callable[[object], None]


class EvidenceStore:
    """Persist evidence through an append-only API and verify it before returning it."""

    def __init__(
        self,
        path: Path,
        *,
        safe_payload_hook: SafePayloadHook = assert_safe_payload,
    ) -> None:
        if not callable(safe_payload_hook):
            raise TypeError("safe_payload_hook must be callable")
        self.path = Path(path)
        self._safe_payload_hook = safe_payload_hook
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def content_id(record: EvidenceRecord) -> str:
        """Return the canonical lowercase SHA-256 identity for an evidence record."""

        if not isinstance(record, EvidenceRecord):
            raise TypeError("record must be an EvidenceRecord")
        content_json = _canonical_json(_record_payload(record))
        return hashlib.sha256(content_json.encode("utf-8")).hexdigest()

    def append(self, record: EvidenceRecord) -> str:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("record must be an EvidenceRecord")
        return self.append_many((record,))[0]

    def append_many(
        self,
        records: Sequence[EvidenceRecord],
    ) -> tuple[str, ...]:
        """Atomically append an ordered batch of evidence records."""

        if isinstance(records, (str, bytes, bytearray)) or not isinstance(
            records,
            Sequence,
        ):
            raise TypeError("records must be a sequence of EvidenceRecord values")
        batch = tuple(records)
        prepared: list[tuple[EvidenceRecord, str, str]] = []
        for record in batch:
            if not isinstance(record, EvidenceRecord):
                raise TypeError("records must contain EvidenceRecord values")
            self._check_payload(record.payload)
            content_json = _canonical_json(_record_payload(record))
            prepared.append((record, self.content_id(record), content_json))
        if not prepared:
            return ()

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run_ids = tuple(sorted({record.run_id for record, _, _ in prepared}))
            if any(
                not self._verify_chain_connection(connection, run_id)
                for run_id in run_ids
            ):
                raise EvidenceIntegrityError("evidence integrity check failed")

            evidence_ids: list[str] = []
            for record, evidence_id, content_json in prepared:
                existing = connection.execute(
                    "SELECT * FROM evidence_records WHERE evidence_id = ?",
                    (evidence_id,),
                ).fetchone()
                if existing is not None:
                    if existing["content_json"] != content_json:
                        raise EvidenceIntegrityError(
                            "evidence integrity check failed"
                        )
                    evidence_ids.append(evidence_id)
                    continue

                for parent_id in record.parent_ids:
                    parent = connection.execute(
                        "SELECT run_id FROM evidence_records WHERE evidence_id = ?",
                        (parent_id,),
                    ).fetchone()
                    if parent is None or parent["run_id"] != record.run_id:
                        raise EvidenceParentError(
                            "parent evidence is unavailable"
                        )

                previous = connection.execute(
                    """
                    SELECT sequence, chain_hash
                    FROM evidence_records
                    WHERE run_id = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                    (record.run_id,),
                ).fetchone()
                sequence = 1 if previous is None else int(previous["sequence"]) + 1
                previous_hash = (
                    _ROOT_HASH
                    if previous is None
                    else str(previous["chain_hash"])
                )
                chain_hash = _chain_hash(
                    evidence_id=evidence_id,
                    previous_hash=previous_hash,
                    run_id=record.run_id,
                    sequence=sequence,
                )
                connection.execute(
                    """
                    INSERT INTO evidence_records (
                        evidence_id, run_id, sequence, kind, payload_json,
                        parent_ids_json, artifact_hashes_json, privacy_class,
                        producer_version, previous_hash, chain_hash, content_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        record.run_id,
                        sequence,
                        record.kind,
                        _canonical_json(record.payload),
                        _canonical_json(record.parent_ids),
                        _canonical_json(record.artifact_hashes),
                        record.privacy_class.value,
                        record.producer_version,
                        previous_hash,
                        chain_hash,
                        content_json,
                    ),
                )
                evidence_ids.append(evidence_id)

            if any(
                not self._verify_chain_connection(connection, run_id)
                for run_id in run_ids
            ):
                raise EvidenceIntegrityError("evidence integrity check failed")
            connection.commit()
            return tuple(evidence_ids)
        except (EvidenceIntegrityError, EvidenceParentError, UnsafeEvidenceError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise EvidenceIntegrityError("evidence integrity check failed") from error
        finally:
            connection.close()

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM evidence_records WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if row is None:
                return None
            if not self._verify_chain_connection(connection, str(row["run_id"])):
                raise EvidenceIntegrityError("evidence integrity check failed")
            return self._record_from_row(row)
        except sqlite3.Error as error:
            raise EvidenceIntegrityError("evidence integrity check failed") from error
        finally:
            connection.close()

    def list_run(self, run_id: str) -> tuple[EvidenceRecord, ...]:
        connection = self._connect()
        try:
            if not self._verify_chain_connection(connection, run_id):
                raise EvidenceIntegrityError("evidence integrity check failed")
            rows = connection.execute(
                "SELECT * FROM evidence_records WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
            return tuple(self._record_from_row(row) for row in rows)
        except sqlite3.Error as error:
            raise EvidenceIntegrityError("evidence integrity check failed") from error
        finally:
            connection.close()

    def verify_chain(self, run_id: str) -> bool:
        connection = self._connect()
        try:
            return self._verify_chain_connection(connection, run_id)
        except (sqlite3.Error, ValueError, TypeError):
            return False
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, _SCHEMA_VERSION}:
                raise EvidenceIntegrityError("evidence integrity check failed")
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.Error as error:
            raise EvidenceIntegrityError("evidence integrity check failed") from error
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _check_payload(self, payload: object) -> None:
        try:
            result = self._safe_payload_hook(payload)
            if result is False:
                raise ValueError("payload hook rejected value")
        except Exception as error:
            raise UnsafeEvidenceError("evidence payload is not safe") from error

    def _record_from_row(self, row: sqlite3.Row) -> EvidenceRecord:
        try:
            payload = json.loads(str(row["payload_json"]))
            parent_ids = json.loads(str(row["parent_ids_json"]))
            artifact_hashes = json.loads(str(row["artifact_hashes_json"]))
            record = EvidenceRecord(
                run_id=str(row["run_id"]),
                kind=str(row["kind"]),
                payload=payload,
                parent_ids=tuple(parent_ids),
                artifact_hashes=artifact_hashes,
                privacy_class=PrivacyClass(str(row["privacy_class"])),
                producer_version=str(row["producer_version"]),
            )
            self._check_payload(record.payload)
            return record
        except (json.JSONDecodeError, TypeError, ValidationError, UnsafeEvidenceError) as error:
            raise EvidenceIntegrityError("evidence integrity check failed") from error

    def _verify_chain_connection(self, connection: sqlite3.Connection, run_id: str) -> bool:
        try:
            rows = connection.execute(
                "SELECT * FROM evidence_records WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
            expected_previous = _ROOT_HASH
            seen_ids: set[str] = set()
            for expected_sequence, row in enumerate(rows, start=1):
                if int(row["sequence"]) != expected_sequence:
                    return False
                if str(row["previous_hash"]) != expected_previous:
                    return False
                record = self._record_from_row(row)
                if record.run_id != run_id or any(
                    parent_id not in seen_ids for parent_id in record.parent_ids
                ):
                    return False
                if str(row["payload_json"]) != _canonical_json(record.payload):
                    return False
                if str(row["parent_ids_json"]) != _canonical_json(record.parent_ids):
                    return False
                if str(row["artifact_hashes_json"]) != _canonical_json(record.artifact_hashes):
                    return False
                content_json = _canonical_json(_record_payload(record))
                evidence_id = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
                if str(row["content_json"]) != content_json:
                    return False
                if str(row["evidence_id"]) != evidence_id:
                    return False
                computed_chain = _chain_hash(
                    evidence_id=evidence_id,
                    previous_hash=expected_previous,
                    run_id=run_id,
                    sequence=expected_sequence,
                )
                if str(row["chain_hash"]) != computed_chain:
                    return False
                seen_ids.add(evidence_id)
                expected_previous = computed_chain
            return True
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
            EvidenceIntegrityError,
        ):
            return False


def _record_payload(record: EvidenceRecord) -> dict[str, object]:
    return {
        "artifact_hashes": record.artifact_hashes,
        "kind": record.kind,
        "parent_ids": record.parent_ids,
        "payload": record.payload,
        "privacy_class": record.privacy_class.value,
        "producer_version": record.producer_version,
        "run_id": record.run_id,
    }


def _chain_hash(*, evidence_id: str, previous_hash: str, run_id: str, sequence: int) -> str:
    payload = _canonical_json(
        {
            "evidence_id": evidence_id,
            "previous_hash": previous_hash,
            "run_id": run_id,
            "sequence": sequence,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _jsonable(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value
