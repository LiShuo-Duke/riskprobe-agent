"""Privacy-safe aggregate payload validation and explicit redaction tokens."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import PurePath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

_MAX_DEPTH = 16
_MAX_ITEMS = 10_000
_MAX_STRING_LENGTH = 16_384
_TOKEN_PATTERN = re.compile(r"^segment-[0-9a-f]{24}$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
_ENTITY_PREFIX = re.compile(
    r"^(?:entity|customer|account|borrower|member|person|user|loan)[_-][A-Za-z0-9_-]*\d{3,}$",
    re.IGNORECASE,
)
_IDENTIFIER_LIKE = re.compile(
    r"\b(?:entity|sample|customer|account|user|row|record|phone|email|path|file)(?:[_=-]|\s+id\s*=)",
    re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"^\d{8,}$")

_FORBIDDEN_KEYS = frozenset(
    {
        "account_id",
        "borrower_id",
        "config_path",
        "customer_id",
        "dataset_path",
        "entity",
        "entity_id",
        "entity_ids",
        "file_path",
        "file_uri",
        "member_id",
        "path",
        "paths",
        "person_id",
        "raw",
        "raw_record",
        "raw_records",
        "raw_row",
        "raw_rows",
        "row",
        "rows",
        "source_path",
        "target",
        "user_id",
    }
)
_SEGMENT_KEYS = frozenset(
    {
        "group_value",
        "segment",
        "segment_label",
        "segment_name",
        "segment_token",
        "segment_value",
        "segment_values",
    }
)
_SAFE_OBJECT_ARRAY_KEYS = frozenset({"findings", "recommendations"})


class UnsafePayloadError(ValueError):
    """Raised when a payload can expose row-level or operational data."""


class SegmentToken(BaseModel):
    """An explicit marker proving a segment label was irreversibly tokenized."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    kind: Literal["segment"] = "segment"
    redacted: Literal[True] = True
    token: str

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        if _TOKEN_PATTERN.fullmatch(value) is None:
            raise ValueError("token must be a canonical segment token")
        return value


def tokenize_segment(value: object, *, namespace: str = "") -> SegmentToken:
    """Return a deterministic, namespaced token without retaining the source label."""

    canonical = json.dumps(
        {"namespace": namespace, "value": _token_source(value)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:24]
    return SegmentToken(token=f"segment-{digest}")


def stable_token(value: object, *, namespace: str = "value") -> str:
    """Return a deterministic opaque compatibility token."""

    digest = hashlib.sha256(f"{namespace}:{value!s}".encode()).hexdigest()[:16]
    return f"tok_{digest}"


def redact_payload(payload: object) -> Any:
    """Recursively replace string values with deterministic opaque tokens."""

    if isinstance(payload, Mapping):
        return {str(key): redact_payload(value) for key, value in payload.items()}
    if isinstance(payload, (tuple, list)):
        return [redact_payload(value) for value in payload]
    if isinstance(payload, str):
        return stable_token(payload)
    return payload


def suppress_small_groups(
    records: Iterable[Mapping[str, Any]], count_key: str, min_group_size: int
) -> list[dict[str, Any]]:
    """Keep only aggregate records meeting the shared minimum group threshold."""

    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    return [
        dict(record)
        for record in records
        if isinstance(record.get(count_key), int)
        and not isinstance(record[count_key], bool)
        and record[count_key] >= min_group_size
    ]


def assert_safe_payload(payload: object) -> None:
    """Recursively reject values that are unsafe outside the trusted data process."""

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, Mapping):
        _unsafe()

    item_count = 0

    def visit(value: object, *, depth: int, parent_key: str | None = None) -> None:
        nonlocal item_count
        item_count += 1
        if depth > _MAX_DEPTH or item_count > _MAX_ITEMS:
            _unsafe()

        if isinstance(value, SegmentToken):
            return
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                _unsafe()
            return
        if isinstance(value, str):
            if (
                len(value) > _MAX_STRING_LENGTH
                or _looks_like_path(value)
                or _looks_like_entity(value)
                or _TOKEN_PATTERN.fullmatch(value) is not None
            ):
                _unsafe()
            return
        if isinstance(value, (PurePath, bytes, bytearray, memoryview)):
            _unsafe()
        if isinstance(value, (date, datetime)):
            return
        if isinstance(value, Mapping):
            if _is_explicit_segment_token(value):
                return
            for key, item in value.items():
                if not isinstance(key, str):
                    _unsafe()
                normalized = _normalize_key(key)
                if normalized in _SEGMENT_KEYS:
                    if item is not None and not _is_explicit_segment_token(item):
                        _unsafe()
                    continue
                if _is_forbidden_key(normalized):
                    _unsafe()
                visit(item, depth=depth + 1, parent_key=normalized)
            return
        if isinstance(value, Sequence):
            if isinstance(value, (str, bytes, bytearray, memoryview)):
                _unsafe()
            if _looks_like_row_array(value, parent_key=parent_key):
                _unsafe()
            for item in value:
                visit(item, depth=depth + 1, parent_key=parent_key)
            return
        _unsafe()

    visit(payload, depth=0)


def canonical_payload_hash(payload: Mapping[str, object]) -> str:
    """Hash a safe payload using deterministic JSON independent of mapping order."""

    assert_safe_payload(payload)
    encoded = json.dumps(
        _canonical_value(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_source(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"type": type(value).__name__, "value": "non-finite"}
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return {"type": type(value).__name__, "value": str(value)}


def _canonical_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and value == 0:
        return 0.0
    return value


def _is_explicit_segment_token(value: object) -> bool:
    if isinstance(value, SegmentToken):
        return True
    if not isinstance(value, Mapping):
        return False
    if set(value) != {"kind", "redacted", "token"}:
        return False
    try:
        SegmentToken.model_validate(value)
    except ValidationError:
        return False
    return True


def _is_forbidden_key(value: str) -> bool:
    sensitive_prefixes = (
        "account",
        "borrower",
        "customer",
        "entity",
        "loan",
        "member",
        "person",
        "user",
    )
    aggregate_suffixes = ("_count", "_rate", "_share", "_total")
    has_sensitive_prefix = any(
        value == prefix or value.startswith(f"{prefix}_")
        for prefix in sensitive_prefixes
    )
    return (
        value in _FORBIDDEN_KEYS
        or value in {"record", "records"}
        or value.endswith("_path")
        or value.endswith("_uri")
        or value.startswith("raw_")
        or (has_sensitive_prefix and not value.endswith(aggregate_suffixes))
    )


def _looks_like_path(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    return (
        stripped.startswith(("/", "\\"))
        or _WINDOWS_DRIVE_PATH.match(stripped) is not None
        or _URI.match(stripped) is not None
        or lowered in {".", ".."}
        or lowered.startswith(("../", "..\\"))
        or "/../" in lowered
        or "\\..\\" in lowered
    )


def _looks_like_entity(value: str) -> bool:
    stripped = value.strip()
    return (
        _EMAIL.fullmatch(stripped) is not None
        or _UUID.fullmatch(stripped) is not None
        or _ENTITY_PREFIX.fullmatch(stripped) is not None
        or _IDENTIFIER_LIKE.search(stripped) is not None
        or _LONG_NUMBER.fullmatch(stripped) is not None
    )


def _looks_like_row_array(value: Sequence[object], *, parent_key: str | None) -> bool:
    if parent_key in _SAFE_OBJECT_ARRAY_KEYS or not value:
        return False
    if not all(isinstance(item, Mapping) for item in value):
        return False
    key_sets = [frozenset(item) for item in value if isinstance(item, Mapping)]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        return False
    return all(
        item is None or isinstance(item, (bool, int, float, str))
        for row in value
        if isinstance(row, Mapping)
        for item in row.values()
    )


def _normalize_key(value: str) -> str:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    return re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")


def _unsafe() -> None:
    raise UnsafePayloadError("payload is not safe")


__all__ = [
    "SegmentToken",
    "UnsafePayloadError",
    "assert_safe_payload",
    "canonical_payload_hash",
    "redact_payload",
    "stable_token",
    "suppress_small_groups",
    "tokenize_segment",
]
