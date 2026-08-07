"""Privacy gates for local aggregate-only tool payloads."""

from collections.abc import Iterable, Mapping
from typing import Any


_DEFAULT_FORBIDDEN_FIELDS = frozenset(
    {"entity_id", "md5_phone", "rows", "records", "raw_data", "file_path"}
)


class UnsafePayloadError(ValueError):
    """Raised when a tool payload could expose prohibited detail."""


def assert_safe_payload(
    payload: object, forbidden_fields: Iterable[str] = _DEFAULT_FORBIDDEN_FIELDS
) -> None:
    """Reject forbidden keys at every nesting level before a payload is returned."""
    forbidden = frozenset(forbidden_fields)
    _assert_safe(payload, forbidden)


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
