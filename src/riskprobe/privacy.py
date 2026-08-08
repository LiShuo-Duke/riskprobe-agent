"""Privacy gates for local aggregate-only tool payloads."""

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any


_DEFAULT_FORBIDDEN_FIELDS = frozenset(
    {"entity_id", "md5_phone", "rows", "records", "raw_data", "file_path"}
)
_PATH_LIKE = re.compile(
    r"(?:^|\s)(?:file://|/|~\/|[A-Za-z]:[\\/])|(?:\.parquet|\.csv|\.json)(?:$|\s)",
    re.IGNORECASE,
)
_IDENTIFIER_LIKE = re.compile(
    r"\b(?:entity|sample|customer|account|user|row|record|phone|email|path|file)(?:[_=-]|\s+id\s*=)",
    re.IGNORECASE,
)


class UnsafePayloadError(ValueError):
    """Raised when a tool payload could expose prohibited detail."""


def assert_safe_payload(
    payload: object, forbidden_fields: Iterable[str] = _DEFAULT_FORBIDDEN_FIELDS
) -> None:
    """Reject forbidden keys and detail-bearing strings at every nesting level."""
    forbidden = frozenset(str(field).lower() for field in forbidden_fields)
    _assert_safe(payload, forbidden)


def stable_token(value: object, *, namespace: str = "value") -> str:
    """Return a deterministic opaque token without retaining the source value."""
    digest = hashlib.sha256(f"{namespace}:{value!s}".encode("utf-8")).hexdigest()[:16]
    return f"tok_{digest}"


def redact_payload(payload: object) -> Any:
    """Recursively redact values before they cross an aggregate-only boundary.

    Numeric aggregates remain numeric; every string value becomes a stable opaque token.
    This deliberately loses detail rather than guessing whether a string is identifying,
    including strings that collide with protocol enum names.
    """
    if isinstance(payload, Mapping):
        return {str(key): redact_payload(value) for key, value in payload.items()}
    if isinstance(payload, (tuple, list)):
        return [redact_payload(value) for value in payload]
    if isinstance(payload, str):
        # A string can be a data value even when it happens to equal a protocol
        # enum (for example a real segment named ``institution``).  Tokenize all
        # string values; mapping keys remain structural and are not data values.
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


def _assert_safe(payload: object, forbidden: frozenset[str]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in forbidden:
                raise UnsafePayloadError(f"forbidden payload field: {key}")
            _assert_safe(value, forbidden)
    elif isinstance(payload, (tuple, list)):
        for value in payload:
            _assert_safe(value, forbidden)
    elif isinstance(payload, str) and (_PATH_LIKE.search(payload) or _IDENTIFIER_LIKE.search(payload)):
        raise UnsafePayloadError("detail-bearing string is not allowed in a tool payload")
