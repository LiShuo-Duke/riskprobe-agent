"""SQLite append-only session trees containing only privacy-safe projections."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from riskprobe.privacy import assert_safe_payload

_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CODE = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_VERSION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_ROOT_HASH = "0" * 64
_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_nodes (
    node_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    kind TEXT NOT NULL,
    parent_node_id TEXT REFERENCES session_nodes(node_id),
    branch_id TEXT NOT NULL,
    retry_of_node_id TEXT REFERENCES session_nodes(node_id),
    goal TEXT NOT NULL,
    tool_name TEXT,
    tool_arguments_json TEXT,
    evidence_ids_json TEXT NOT NULL,
    redacted_summary TEXT,
    component_versions_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    chain_hash TEXT NOT NULL,
    content_json TEXT NOT NULL,
    UNIQUE (session_id, sequence)
);
CREATE INDEX IF NOT EXISTS session_nodes_parent ON session_nodes(parent_node_id, sequence);
CREATE INDEX IF NOT EXISTS session_nodes_session ON session_nodes(session_id, sequence);
CREATE TRIGGER IF NOT EXISTS session_nodes_no_update
BEFORE UPDATE ON session_nodes
BEGIN
    SELECT RAISE(ABORT, 'session history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS session_nodes_no_delete
BEFORE DELETE ON session_nodes
BEGIN
    SELECT RAISE(ABORT, 'session history is append-only');
END;
"""

SafePayloadHook = Callable[[object], None]


class SessionIntegrityError(RuntimeError):
    """Raised when persisted session history fails deterministic integrity checks."""


class UnsafeSessionPayloadError(ValueError):
    """Raised with a fixed message when session content fails the privacy gate."""


class SessionNodeKind(StrEnum):
    ROOT = "root"
    CHILD = "child"
    FORK = "fork"
    RETRY = "retry"


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class SessionToolCall(_StrictDTO):
    """Safe persisted projection of a typed tool request."""

    tool_name: str
    arguments: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("tool_name must be a public code")
        return value

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        normalized = dict(value)
        for key, item in normalized.items():
            if not isinstance(key, str):
                raise ValueError("tool arguments must use string keys")
            if key.endswith("_id") and not isinstance(item, str):
                raise ValueError("identifier arguments must be strings")
        try:
            assert_safe_payload({"arguments": normalized})
        except Exception as error:
            raise ValueError("tool arguments are not safe") from error
        return MappingProxyType(_freeze_mapping(normalized))

    @field_serializer("arguments")
    def serialize_arguments(self, value: Mapping[str, object]) -> dict[str, object]:
        return _jsonable(value)


class SessionNode(_StrictDTO):
    """Strict immutable DTO for one append-only session-tree node."""

    session_id: str
    node_id: str
    sequence: int = Field(ge=1)
    kind: SessionNodeKind
    parent_node_id: str | None = None
    branch_id: str
    retry_of_node_id: str | None = None
    goal: str
    tool_call: SessionToolCall | None = None
    evidence_ids: tuple[str, ...] = ()
    redacted_summary: str | None = None
    component_versions: Mapping[str, str]

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("session_id must be a public identifier")
        return value

    @field_validator("node_id", "branch_id", "retry_of_node_id")
    @classmethod
    def validate_hash_ids(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("session node links must be SHA-256 identifiers")
        return value

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        if not value or len(value) > 512:
            raise ValueError("goal must be bounded and non-empty")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("evidence IDs must be unique SHA-256 identifiers")
        return tuple(sorted(value))

    @field_validator("redacted_summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 2_048):
            raise ValueError("redacted summary must be bounded and non-empty")
        return value

    @field_validator("component_versions")
    @classmethod
    def validate_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        normalized = dict(value)
        if not normalized or any(
            _VERSION_KEY.fullmatch(key) is None or _VERSION_VALUE.fullmatch(version) is None
            for key, version in normalized.items()
        ):
            raise ValueError("component_versions must contain public version tokens")
        return MappingProxyType(dict(sorted(normalized.items())))

    @field_serializer("component_versions")
    def serialize_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def validate_structure_and_privacy(self) -> SessionNode:
        if self.kind is SessionNodeKind.ROOT:
            if self.parent_node_id is not None or self.retry_of_node_id is not None:
                raise ValueError("root session node cannot have parent or retry links")
            if self.sequence != 1:
                raise ValueError("root session node must have sequence one")
        elif self.parent_node_id is None:
            raise ValueError("non-root session node requires a parent")
        if self.kind is SessionNodeKind.RETRY:
            if self.retry_of_node_id != self.parent_node_id:
                raise ValueError("retry must link to the retried parent")
        elif self.retry_of_node_id is not None:
            raise ValueError("only retry nodes can contain retry links")
        try:
            assert_safe_payload(_safe_node_payload(self))
        except Exception as error:
            raise ValueError("session payload is not safe") from error
        return self


class SessionStore:
    """Append-only SQLite session tree with a per-session integrity chain."""

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

    def create_session(
        self,
        *,
        goal: str,
        component_versions: Mapping[str, str],
        session_id: str | None = None,
        redacted_summary: str | None = None,
    ) -> SessionNode:
        public_session_id = session_id or f"session-{uuid.uuid4().hex}"
        branch_id = _digest({"branch": "root", "session_id": public_session_id})
        return self._insert(
            session_id=public_session_id,
            kind=SessionNodeKind.ROOT,
            parent=None,
            branch_id=branch_id,
            goal=goal,
            tool_call=None,
            evidence_ids=(),
            redacted_summary=redacted_summary,
            component_versions=component_versions,
            retry_of_node_id=None,
        )

    create_root = create_session

    def append_child(
        self,
        parent_node_id: str,
        *,
        goal: str | None = None,
        tool_call: SessionToolCall | None = None,
        evidence_ids: Sequence[str] = (),
        redacted_summary: str | None = None,
        component_versions: Mapping[str, str] | None = None,
    ) -> SessionNode:
        return self._insert_relative(
            parent_node_id,
            kind=SessionNodeKind.CHILD,
            goal=goal,
            tool_call=tool_call,
            evidence_ids=evidence_ids,
            redacted_summary=redacted_summary,
            component_versions=component_versions,
            new_branch=False,
        )

    append = append_child

    def fork(
        self,
        parent_node_id: str,
        *,
        goal: str | None = None,
        tool_call: SessionToolCall | None = None,
        evidence_ids: Sequence[str] = (),
        redacted_summary: str | None = None,
        component_versions: Mapping[str, str] | None = None,
    ) -> SessionNode:
        return self._insert_relative(
            parent_node_id,
            kind=SessionNodeKind.FORK,
            goal=goal,
            tool_call=tool_call,
            evidence_ids=evidence_ids,
            redacted_summary=redacted_summary,
            component_versions=component_versions,
            new_branch=True,
        )

    def retry(
        self,
        parent_node_id: str,
        *,
        goal: str | None = None,
        tool_call: SessionToolCall | None = None,
        evidence_ids: Sequence[str] = (),
        redacted_summary: str | None = None,
        component_versions: Mapping[str, str] | None = None,
    ) -> SessionNode:
        return self._insert_relative(
            parent_node_id,
            kind=SessionNodeKind.RETRY,
            goal=goal,
            tool_call=tool_call,
            evidence_ids=evidence_ids,
            redacted_summary=redacted_summary,
            component_versions=component_versions,
            new_branch=False,
        )

    def get(self, node_id: str) -> SessionNode | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM session_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                return None
            self._require_valid_session(connection, str(row["session_id"]))
            return self._node_from_row(row)
        except sqlite3.Error as error:
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()

    get_node = get

    def children(self, node_id: str) -> tuple[SessionNode, ...]:
        connection = self._connect()
        try:
            parent = connection.execute(
                "SELECT session_id FROM session_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if parent is None:
                raise KeyError("session node is unavailable")
            self._require_valid_session(connection, str(parent["session_id"]))
            rows = connection.execute(
                "SELECT * FROM session_nodes WHERE parent_node_id = ? ORDER BY sequence",
                (node_id,),
            ).fetchall()
            return tuple(self._node_from_row(row) for row in rows)
        except sqlite3.Error as error:
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()

    list_children = children

    def branch(self, node_id: str) -> tuple[SessionNode, ...]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM session_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                raise KeyError("session node is unavailable")
            self._require_valid_session(connection, str(row["session_id"]))
            branch: list[SessionNode] = []
            while row is not None:
                node = self._node_from_row(row)
                branch.append(node)
                if node.parent_node_id is None:
                    break
                row = connection.execute(
                    "SELECT * FROM session_nodes WHERE node_id = ?", (node.parent_node_id,)
                ).fetchone()
                if row is None:
                    raise SessionIntegrityError("session integrity check failed")
            return tuple(reversed(branch))
        except sqlite3.Error as error:
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()

    def replay(
        self,
        session_id: str,
        *,
        branch_id: str | None = None,
        leaf_node_id: str | None = None,
    ) -> tuple[SessionNode, ...]:
        if leaf_node_id is not None:
            path = self.branch(leaf_node_id)
            if not path or path[0].session_id != session_id:
                raise KeyError("session branch is unavailable")
            return path
        connection = self._connect()
        try:
            self._require_valid_session(connection, session_id)
            if branch_id is None:
                rows = connection.execute(
                    "SELECT * FROM session_nodes WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM session_nodes
                    WHERE session_id = ? AND branch_id = ?
                    ORDER BY sequence
                    """,
                    (session_id, branch_id),
                ).fetchall()
            return tuple(self._node_from_row(row) for row in rows)
        except sqlite3.Error as error:
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()

    def _insert_relative(
        self,
        parent_node_id: str,
        *,
        kind: SessionNodeKind,
        goal: str | None,
        tool_call: SessionToolCall | None,
        evidence_ids: Sequence[str],
        redacted_summary: str | None,
        component_versions: Mapping[str, str] | None,
        new_branch: bool,
    ) -> SessionNode:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            parent_row = connection.execute(
                "SELECT * FROM session_nodes WHERE node_id = ?", (parent_node_id,)
            ).fetchone()
            if parent_row is None:
                raise KeyError("parent session node is unavailable")
            session_id = str(parent_row["session_id"])
            self._require_valid_session(connection, session_id)
            parent = self._node_from_row(parent_row)
            sequence = self._next_sequence(connection, session_id)
            branch_id = (
                _digest(
                    {
                        "fork_parent": parent_node_id,
                        "sequence": sequence,
                        "session_id": session_id,
                    }
                )
                if new_branch
                else parent.branch_id
            )
            node = self._build_node(
                session_id=session_id,
                sequence=sequence,
                kind=kind,
                parent=parent,
                branch_id=branch_id,
                goal=parent.goal if goal is None else goal,
                tool_call=tool_call,
                evidence_ids=evidence_ids,
                redacted_summary=redacted_summary,
                component_versions=(
                    parent.component_versions
                    if component_versions is None
                    else component_versions
                ),
                retry_of_node_id=parent_node_id if kind is SessionNodeKind.RETRY else None,
            )
            self._persist_node(connection, node)
            connection.commit()
            return node
        except (KeyError, UnsafeSessionPayloadError, ValidationError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()

    def _insert(
        self,
        *,
        session_id: str,
        kind: SessionNodeKind,
        parent: SessionNode | None,
        branch_id: str,
        goal: str,
        tool_call: SessionToolCall | None,
        evidence_ids: Sequence[str],
        redacted_summary: str | None,
        component_versions: Mapping[str, str],
        retry_of_node_id: str | None,
    ) -> SessionNode:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM session_nodes WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing is not None:
                raise ValueError("session_id already exists")
            node = self._build_node(
                session_id=session_id,
                sequence=1,
                kind=kind,
                parent=parent,
                branch_id=branch_id,
                goal=goal,
                tool_call=tool_call,
                evidence_ids=evidence_ids,
                redacted_summary=redacted_summary,
                component_versions=component_versions,
                retry_of_node_id=retry_of_node_id,
            )
            self._persist_node(connection, node)
            connection.commit()
            return node
        except (UnsafeSessionPayloadError, ValidationError, ValueError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()

    def _build_node(
        self,
        *,
        session_id: str,
        sequence: int,
        kind: SessionNodeKind,
        parent: SessionNode | None,
        branch_id: str,
        goal: str,
        tool_call: SessionToolCall | None,
        evidence_ids: Sequence[str],
        redacted_summary: str | None,
        component_versions: Mapping[str, str],
        retry_of_node_id: str | None,
    ) -> SessionNode:
        normalized_evidence = tuple(sorted(evidence_ids))
        payload: dict[str, object] = {
            "branch_id": branch_id,
            "component_versions": dict(component_versions),
            "evidence_ids": normalized_evidence,
            "goal": goal,
            "kind": kind.value,
            "parent_node_id": parent.node_id if parent is not None else None,
            "redacted_summary": redacted_summary,
            "retry_of_node_id": retry_of_node_id,
            "sequence": sequence,
            "session_id": session_id,
            "tool_call": tool_call.model_dump(mode="json") if tool_call is not None else None,
        }
        self._check_payload(payload)
        node_id = _digest(payload)
        return SessionNode(
            session_id=session_id,
            node_id=node_id,
            sequence=sequence,
            kind=kind,
            parent_node_id=parent.node_id if parent is not None else None,
            branch_id=branch_id,
            retry_of_node_id=retry_of_node_id,
            goal=goal,
            tool_call=tool_call,
            evidence_ids=normalized_evidence,
            redacted_summary=redacted_summary,
            component_versions=component_versions,
        )

    def _persist_node(self, connection: sqlite3.Connection, node: SessionNode) -> None:
        content_json = _canonical_json(_node_content(node))
        previous = connection.execute(
            """
            SELECT chain_hash FROM session_nodes
            WHERE session_id = ? ORDER BY sequence DESC LIMIT 1
            """,
            (node.session_id,),
        ).fetchone()
        previous_hash = _ROOT_HASH if previous is None else str(previous["chain_hash"])
        chain_hash = _digest(
            {
                "node_id": node.node_id,
                "previous_hash": previous_hash,
                "sequence": node.sequence,
                "session_id": node.session_id,
            }
        )
        connection.execute(
            """
            INSERT INTO session_nodes (
                node_id, session_id, sequence, kind, parent_node_id, branch_id,
                retry_of_node_id, goal, tool_name, tool_arguments_json,
                evidence_ids_json, redacted_summary, component_versions_json,
                previous_hash, chain_hash, content_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.node_id,
                node.session_id,
                node.sequence,
                node.kind.value,
                node.parent_node_id,
                node.branch_id,
                node.retry_of_node_id,
                node.goal,
                node.tool_call.tool_name if node.tool_call is not None else None,
                _canonical_json(node.tool_call.arguments) if node.tool_call is not None else None,
                _canonical_json(node.evidence_ids),
                node.redacted_summary,
                _canonical_json(node.component_versions),
                previous_hash,
                chain_hash,
                content_json,
            ),
        )

    def _require_valid_session(self, connection: sqlite3.Connection, session_id: str) -> None:
        rows = connection.execute(
            "SELECT * FROM session_nodes WHERE session_id = ? ORDER BY sequence", (session_id,)
        ).fetchall()
        if not rows:
            raise KeyError("session is unavailable")
        previous_hash = _ROOT_HASH
        seen: set[str] = set()
        for expected_sequence, row in enumerate(rows, start=1):
            try:
                node = self._node_from_row(row)
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise SessionIntegrityError("session integrity check failed") from error
            if node.sequence != expected_sequence or str(row["previous_hash"]) != previous_hash:
                raise SessionIntegrityError("session integrity check failed")
            if node.parent_node_id is not None and node.parent_node_id not in seen:
                raise SessionIntegrityError("session integrity check failed")
            content_json = _canonical_json(_node_content(node))
            if str(row["content_json"]) != content_json or _digest(_node_content(node)) != node.node_id:
                raise SessionIntegrityError("session integrity check failed")
            expected_chain = _digest(
                {
                    "node_id": node.node_id,
                    "previous_hash": previous_hash,
                    "sequence": node.sequence,
                    "session_id": node.session_id,
                }
            )
            if str(row["chain_hash"]) != expected_chain:
                raise SessionIntegrityError("session integrity check failed")
            seen.add(node.node_id)
            previous_hash = expected_chain

    def _node_from_row(self, row: sqlite3.Row) -> SessionNode:
        tool_name = row["tool_name"]
        tool_call = (
            SessionToolCall(
                tool_name=str(tool_name),
                arguments=json.loads(str(row["tool_arguments_json"])),
            )
            if tool_name is not None
            else None
        )
        node = SessionNode(
            session_id=str(row["session_id"]),
            node_id=str(row["node_id"]),
            sequence=int(row["sequence"]),
            kind=SessionNodeKind(str(row["kind"])),
            parent_node_id=(
                str(row["parent_node_id"]) if row["parent_node_id"] is not None else None
            ),
            branch_id=str(row["branch_id"]),
            retry_of_node_id=(
                str(row["retry_of_node_id"]) if row["retry_of_node_id"] is not None else None
            ),
            goal=str(row["goal"]),
            tool_call=tool_call,
            evidence_ids=tuple(json.loads(str(row["evidence_ids_json"]))),
            redacted_summary=(
                str(row["redacted_summary"]) if row["redacted_summary"] is not None else None
            ),
            component_versions=json.loads(str(row["component_versions_json"])),
        )
        self._check_payload(_safe_node_payload(node))
        return node

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM session_nodes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["sequence"]) + 1

    def _check_payload(self, payload: object) -> None:
        try:
            result = self._safe_payload_hook(payload)
            if result is False:
                raise ValueError("privacy hook rejected payload")
        except Exception as error:
            raise UnsafeSessionPayloadError("session payload is not safe") from error

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, _SCHEMA_VERSION}:
                raise SessionIntegrityError("session integrity check failed")
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.Error as error:
            raise SessionIntegrityError("session integrity check failed") from error
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


def _safe_node_payload(node: SessionNode) -> dict[str, object]:
    return {
        "branch_id": node.branch_id,
        "component_versions": dict(node.component_versions),
        "evidence_ids": node.evidence_ids,
        "goal": node.goal,
        "kind": node.kind.value,
        "node_id": node.node_id,
        "parent_node_id": node.parent_node_id,
        "redacted_summary": node.redacted_summary,
        "retry_of_node_id": node.retry_of_node_id,
        "sequence": node.sequence,
        "session_id": node.session_id,
        "tool_call": node.tool_call.model_dump(mode="json") if node.tool_call is not None else None,
    }


def _node_content(node: SessionNode) -> dict[str, object]:
    payload = _safe_node_payload(node)
    payload.pop("node_id")
    return payload


def _freeze_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _freeze_value(item) for key, item in value.items()}


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(_freeze_mapping(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    return value


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


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "SafePayloadHook",
    "SessionIntegrityError",
    "SessionNode",
    "SessionNodeKind",
    "SessionStore",
    "SessionToolCall",
    "UnsafeSessionPayloadError",
]
