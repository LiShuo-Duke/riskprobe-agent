"""Strict evidence records and aggregate-payload safety checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_KIND = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_DENIED_PAYLOAD_KEYS = frozenset(
    {
        "api_key",
        "auth_token",
        "config",
        "config_path",
        "credentials",
        "dataset",
        "dataset_path",
        "entity",
        "entity_id",
        "entity_ids",
        "file_path",
        "password",
        "path",
        "paths",
        "raw",
        "raw_row",
        "raw_rows",
        "row",
        "rows",
        "secret",
        "segment",
        "segment_value",
        "segment_values",
    }
)
_MAX_PAYLOAD_DEPTH = 16
_MAX_PAYLOAD_ITEMS = 10_000
_MAX_STRING_LENGTH = 16_384


class EvidenceIntegrityError(RuntimeError):
    """Raised when persisted evidence no longer matches its hash chain."""


class EvidenceParentError(ValueError):
    """Raised when evidence references an unavailable parent."""


class UnsafeEvidenceError(ValueError):
    """Raised when evidence could expose non-aggregate or private data."""


class PrivacyClass(StrEnum):
    """Privacy classification persisted with each evidence record."""

    AGGREGATE = "aggregate"
    RESTRICTED = "restricted"


class EvidenceRecord(BaseModel):
    """Content-addressed aggregate evidence supplied by a deterministic producer."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    run_id: str
    kind: str
    payload: Mapping[str, object]
    parent_ids: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, str] = Field(default_factory=dict)
    privacy_class: PrivacyClass = PrivacyClass.AGGREGATE
    producer_version: str

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("run_id must be a public identifier")
        return value

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if _KIND.fullmatch(value) is None:
            raise ValueError("kind must be a public identifier")
        return value

    @field_validator("payload")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(value))

    @field_validator("parent_ids")
    @classmethod
    def validate_parent_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("parent_ids must contain unique SHA-256 identifiers")
        return value

    @field_validator("artifact_hashes")
    @classmethod
    def validate_artifact_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        normalized = dict(value)
        for name, digest in normalized.items():
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError("artifact_hashes must map public names to SHA-256 values")
        return MappingProxyType(normalized)

    @field_serializer("payload", "artifact_hashes")
    def serialize_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @field_validator("producer_version")
    @classmethod
    def validate_producer_version(cls, value: str) -> str:
        if not value or len(value) > 128 or any(character.isspace() for character in value):
            raise ValueError("producer_version must be a non-empty version token")
        return value

    @model_validator(mode="after")
    def require_aggregate_privacy(self) -> EvidenceRecord:
        if self.privacy_class is not PrivacyClass.AGGREGATE:
            raise ValueError("only aggregate evidence can be persisted")
        return self


def assert_safe_payload(payload: object) -> None:
    """Reject paths, row-level identifiers, and values outside bounded JSON aggregates."""

    item_count = 0

    def visit(value: object, depth: int) -> None:
        nonlocal item_count
        item_count += 1
        if depth > _MAX_PAYLOAD_DEPTH or item_count > _MAX_PAYLOAD_ITEMS:
            raise UnsafeEvidenceError("evidence payload is not safe")
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise UnsafeEvidenceError("evidence payload is not safe")
            return
        if isinstance(value, str):
            if len(value) > _MAX_STRING_LENGTH or _looks_like_path(value):
                raise UnsafeEvidenceError("evidence payload is not safe")
            return
        if isinstance(value, Path) or isinstance(value, (bytes, bytearray, memoryview)):
            raise UnsafeEvidenceError("evidence payload is not safe")
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or _normalise_key(key) in _DENIED_PAYLOAD_KEYS:
                    raise UnsafeEvidenceError("evidence payload is not safe")
                visit(item, depth + 1)
            return
        if isinstance(value, Sequence):
            for item in value:
                visit(item, depth + 1)
            return
        raise UnsafeEvidenceError("evidence payload is not safe")

    if not isinstance(payload, Mapping):
        raise UnsafeEvidenceError("evidence payload is not safe")
    visit(payload, 0)


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _looks_like_path(value: str) -> bool:
    lowered = value.strip().lower()
    return (
        lowered.startswith(("/", "\\\\", "file://", "s3://", "gs://", "http://", "https://"))
        or _WINDOWS_PATH.match(value.strip()) is not None
        or lowered in {"..", "."}
        or lowered.startswith(("../", "..\\"))
        or "/../" in lowered
        or "\\..\\" in lowered
    )
