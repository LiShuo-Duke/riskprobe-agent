"""Strict privacy-safe models for the local citation index.

Direct Pydantic construction and ``model_validate`` are trusted-code APIs. Call each
public model's ``safe_validate`` factory at an untrusted boundary; it revalidates a
copy and converts every validation failure to a fixed, unlinked domain exception.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, NoReturn, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from riskprobe.privacy import assert_safe_payload

_OPAQUE_SCOPE_ID = re.compile(r"^scope-[0-9a-f]{24}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URI = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>()]+")
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[A-Za-z]{2,}\b")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/][^\s]+")
_UNC_PATH = re.compile(r"\\\\[^\\\s]+\\[^\s]+")
_UNIX_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:[A-Za-z0-9._~-]+/)*[A-Za-z0-9._~%-]+"
)
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_ENTITY_ID = re.compile(
    r"\b(?:entity|customer|account|borrower|member|person|user|loan)"
    r"[_-][A-Za-z0-9_-]*\d{3,}\b",
    re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"\b\d{8,}\b")
_SEGMENT_FIELD = re.compile(
    r"\b(?:segment|cohort|group)(?:[ _-](?:label|name|value))?"
    r"(?:\s*[:=]\s*|\s+)[\"']?[^\s,;}]+",
    re.IGNORECASE,
)
_SECRET_REFERENCE = re.compile(
    r"[\"']?(?:password|passwd|secret|api[ _-]?key|access[ _-]?token|"
    r"auth[ _-]?token|bearer)[\"']?"
    r"\s*(?::|=|\bis\b|\bwas\b|\bof\b|\bvalue\b|\s)\s*"
    r"[\"']?[^\s,.;}]+",
    re.IGNORECASE,
)
_SECRET_MATERIAL = re.compile(
    r"\b(?:sk-(?:live|test)-?[A-Za-z0-9_-]{6,}|"
    r"gh[pousr]_[A-Za-z0-9]{8,}|AKIA[A-Z0-9]{12,}|"
    r"Bearer\s+[A-Za-z0-9._~+/-]{8,})\b",
    re.IGNORECASE,
)
_CODE_LINE = re.compile(
    r"(?m)^\s*(?:def\s+\w+\s*\(|class\s+\w+\s*[:({]|"
    r"from\s+\S+\s+import\s+|import\s+\S+|function\s+\w+\s*\(|"
    r"(?:const|let|var)\s+\w+\s*=|#include\s*[<\"]|SELECT\s+.+\s+FROM\s+)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?<![=!<>])\b[A-Za-z_][A-Za-z0-9_.-]{1,63}\s*=\s*(?!=)\S+"
)
_JSON_ROW = re.compile(r"(?:\{|\[\s*\{)\s*\"[^\"\r\n]{1,128}\"\s*:", re.DOTALL)
_MARKDOWN_TABLE_ROW = re.compile(r"(?m)^\s*\|.*\|\s*$")
_MARKDOWN_TABLE_SEPARATOR = re.compile(
    r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_RAW_ROW_FIELD = re.compile(
    r"\b(?:raw[ _-]?data|raw[ _-]?rows?|records?|table[ _-]?dump)\s*[:=]",
    re.IGNORECASE,
)
_SECRET_ALIAS_ASSIGNMENT = re.compile(
    r"\b(?:client[ _-]?secret|credential(?:s)?|pwd|token)\b"
    r"\s*(?::|=|\bis\b|\bwas\b|\bof\b|\bvalue\b|\bnamed\b)\s*[\"']?\S+",
    re.IGNORECASE,
)
_INLINE_CODE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_.]*\s*\([^()\r\n]{0,128}\)",
    re.IGNORECASE,
)
_DELIMITED_ROW = re.compile(r"(?m)^\s*\S[^\r\n]*(?:\t|\|)[^\r\n]*\S\s*$")
_LIST_COMPREHENSION = re.compile(
    r"\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s+in\b",
    re.IGNORECASE,
)
_PROVIDER_BOUNDARY_KEYS = frozenset(
    {"aggregate_count", "metric_code", "operation", "status_code"}
)
_PROVIDER_METRICS = {
    "approval_rate": "approval rate",
    "control_coverage": "control coverage",
    "delinquency_rate": "delinquency rate",
    "finding_count": "finding count",
    "liquidity_rate": "liquidity rate",
}
_PROVIDER_STATUSES = frozenset({"declined", "improved", "stable", "unknown", "warning"})
_MAX_SAFE_TEXT_LENGTH = 262_144
_MAX_TITLE_LENGTH = 256
_MAX_PROVIDER_OPERATION_LENGTH = 32
_MAX_PROVIDER_METRIC_LENGTH = 32
_MAX_PROVIDER_STATUS_LENGTH = 16
_SHA256_LENGTH = 64
_OPAQUE_SCOPE_ID_LENGTH = len("scope-") + 24


class UnsafeIndexRequestError(ValueError):
    """Raised when a root, scope, query ID, or request shape is unsafe."""


class UnsafeContentError(ValueError):
    """Raised when document, summary, or query content fails the privacy gate."""


class IndexIntegrityError(RuntimeError):
    """Raised when persisted citation index integrity cannot be established."""


def _raise_unlinked(error_type: type[Exception], message: str) -> NoReturn:
    """Raise a fixed domain exception with no retained cause or context graph."""

    error = error_type(message)
    try:
        raise error from None
    finally:
        error.__cause__ = None
        error.__context__ = None


class _StrictModel(BaseModel):
    """Base for trusted direct construction plus a sanitized untrusted factory."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    @classmethod
    def safe_validate(cls, value: object) -> Self:
        """Copy and fully validate an exact built-in mapping."""

        try:
            return _safe_model_from_boundary(cls, value)  # type: ignore[return-value]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            value = None
            _raise_unlinked(
                UnsafeIndexRequestError,
                "citation index request is not safe",
            )


def _exact_string_keys(value: dict[object, object]) -> frozenset[str]:
    keys: list[str] = []
    for key in value:
        if type(key) is not str:
            raise TypeError
        keys.append(key)
    return frozenset(keys)


def _safe_model_from_boundary(
    model_type: type[_StrictModel],
    value: object,
) -> _StrictModel:
    fields = model_type.model_fields
    if type(value) is not dict or len(value) > len(fields):
        raise TypeError
    keys = _exact_string_keys(value)
    allowed_keys = frozenset(fields)
    required_keys = frozenset(
        field_name for field_name, field in fields.items() if field.is_required()
    )
    if not required_keys.issubset(keys) or not keys.issubset(allowed_keys):
        raise ValueError
    _preflight_model_strings(model_type, value, keys)
    candidate = dict(value)
    if "citations" in fields and "citations" in keys:
        citations = value["citations"]
        if type(citations) is not tuple or len(citations) > 100:
            raise TypeError
        validated_citations: list[Citation] = []
        for item in citations:
            if type(item) is not dict or len(item) > len(Citation.model_fields):
                raise TypeError
            validated_citations.append(Citation.safe_validate(item))
        candidate["citations"] = tuple(validated_citations)
    return model_type.model_validate(candidate)


def _bounded_exact_string(
    value: object,
    *,
    max_length: int,
    exact_length: int | None = None,
) -> str:
    if type(value) is not str:
        raise TypeError
    length = len(value)
    if length > max_length or (exact_length is not None and length != exact_length):
        raise ValueError
    return value


def _preflight_model_strings(
    model_type: type[_StrictModel],
    value: dict[object, object],
    keys: frozenset[str],
) -> None:
    if model_type in {BuildResult, QueryResult}:
        _bounded_exact_string(
            value["scope_id"],
            max_length=_OPAQUE_SCOPE_ID_LENGTH,
            exact_length=_OPAQUE_SCOPE_ID_LENGTH,
        )
    if model_type is BuildResult:
        _bounded_exact_string(
            value["index_hash"],
            max_length=_SHA256_LENGTH,
            exact_length=_SHA256_LENGTH,
        )
        document_count = value["document_count"]
        if type(document_count) is not int or not 0 <= document_count <= 8_192:
            raise ValueError
    if model_type is Citation:
        for field_name in ("citation_id", "document_id", "content_hash"):
            _bounded_exact_string(
                value[field_name],
                max_length=_SHA256_LENGTH,
                exact_length=_SHA256_LENGTH,
            )
        rank = value["rank"]
        score = value["score"]
        if type(rank) is not int or not 1 <= rank <= 100:
            raise ValueError
        if type(score) not in {int, float} or not 0 <= score <= 1:
            raise ValueError
        if type(score) is float and not math.isfinite(score):
            raise ValueError
        if "title" in keys and value["title"] is not None:
            _bounded_exact_string(value["title"], max_length=_MAX_TITLE_LENGTH)


class ProviderSafeSummary(_StrictModel):
    """Trusted internal DTO rendered from a strict aggregate boundary operation.

    Direct construction remains a trusted-code API for internal representation.
    ``safe_validate`` never treats a model instance as boundary authority; only an
    exact ``aggregate_status`` dict can create a boundary-approved summary.
    """

    text: str
    aggregate_count: int = Field(ge=0, le=1_000_000)
    content_hash: str = ""

    @classmethod
    def safe_validate(cls, value: object) -> Self:
        """Render only an exact, bounded aggregate operation."""

        try:
            return _provider_summary_from_boundary(cls, value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            value = None
            _raise_unlinked(UnsafeContentError, "citation content is not safe")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_safe_text(value, allow_empty=False)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if type(value) is not str:
            raise ValueError("content_hash must be a SHA-256 identifier")
        length = len(value)
        if value and (
            length != _SHA256_LENGTH or _SHA256.fullmatch(value) is None
        ):
            raise ValueError("content_hash must be a SHA-256 identifier")
        return value

    @model_validator(mode="after")
    def derive_content_hash(self) -> ProviderSafeSummary:
        payload = {
            "aggregate_count": self.aggregate_count,
            "text": self.text,
        }
        try:
            assert_safe_payload(payload)
        except Exception:
            raise ValueError("citation content is not safe") from None
        expected = hashlib.sha256(b"provider-safe\0" + _canonical_bytes(payload)).hexdigest()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("content_hash does not match provider summary")
        object.__setattr__(self, "content_hash", expected)
        return self


def _provider_summary_from_boundary(
    model_type: type[ProviderSafeSummary],
    value: object,
) -> ProviderSafeSummary:
    if type(value) is not dict or len(value) != 4:
        raise TypeError
    if _exact_string_keys(value) != _PROVIDER_BOUNDARY_KEYS:
        raise ValueError
    operation = _bounded_exact_string(
        value["operation"],
        max_length=_MAX_PROVIDER_OPERATION_LENGTH,
    )
    metric_code = _bounded_exact_string(
        value["metric_code"],
        max_length=_MAX_PROVIDER_METRIC_LENGTH,
    )
    status_code = _bounded_exact_string(
        value["status_code"],
        max_length=_MAX_PROVIDER_STATUS_LENGTH,
    )
    aggregate_count = value["aggregate_count"]
    if (
        operation != "aggregate_status"
        or metric_code not in _PROVIDER_METRICS
        or status_code not in _PROVIDER_STATUSES
        or type(aggregate_count) is not int
        or not 0 <= aggregate_count <= 1_000_000
    ):
        raise ValueError
    text = f"Aggregate {_PROVIDER_METRICS[metric_code]} status is {status_code}."
    return model_type(text=text, aggregate_count=aggregate_count)


class BuildResult(_StrictModel):
    """Minimal trusted result returned after replacing one scope's local index."""

    scope_id: str
    document_count: int = Field(ge=0, le=8_192)
    index_hash: str

    @field_validator("scope_id")
    @classmethod
    def validate_scope_id(cls, value: str) -> str:
        return _validated_identifier(value, "scope_id")

    @field_validator("index_hash")
    @classmethod
    def validate_index_hash(cls, value: str) -> str:
        return _validated_hash(value, "index_hash")


class Citation(_StrictModel):
    """Content-addressed metadata; direct construction is trusted-code only."""

    rank: int = Field(ge=1, le=100)
    citation_id: str
    document_id: str
    content_hash: str
    score: float = Field(ge=0, le=1)
    title: str | None = None

    @field_validator("citation_id", "document_id", "content_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validated_hash(value, info.field_name)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("citation score must be finite")
        return value

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if type(value) is not str or len(value) > _MAX_TITLE_LENGTH:
            raise ValueError("citation title is not safe")
        normalized = _validate_safe_text(value, allow_empty=False)
        if "\n" in normalized:
            raise ValueError("citation title is not safe")
        return normalized


class QueryResult(_StrictModel):
    """Ordered citations for exactly one isolated scope; direct input is trusted."""

    scope_id: str
    citations: tuple[Citation, ...] = Field(default=(), max_length=100)

    @field_validator("citations", mode="before")
    @classmethod
    def preflight_citations(cls, value: object) -> object:
        if type(value) is not tuple or len(value) > 100:  # type: ignore[arg-type]
            raise ValueError("citations must be a bounded exact tuple")
        return value

    @field_validator("scope_id")
    @classmethod
    def validate_scope_id(cls, value: str) -> str:
        return _validated_identifier(value, "scope_id")

    @model_validator(mode="after")
    def validate_citation_order(self) -> QueryResult:
        if tuple(item.rank for item in self.citations) != tuple(
            range(1, len(self.citations) + 1)
        ):
            raise ValueError("citation ranks must be contiguous")
        if len({item.citation_id for item in self.citations}) != len(self.citations):
            raise ValueError("citation identifiers must be unique")
        order = tuple((-item.score, item.citation_id) for item in self.citations)
        if order != tuple(sorted(order)):
            raise ValueError("citations must use deterministic score order")
        return self


def _validated_identifier(value: str, field_name: str) -> str:
    try:
        if type(value) is not str:
            raise ValueError
        length = len(value)
        if (
            length != _OPAQUE_SCOPE_ID_LENGTH
            or _OPAQUE_SCOPE_ID.fullmatch(value) is None
        ):
            raise ValueError
        assert_safe_payload({"identifier": value})
        return value
    except Exception:
        raise ValueError(f"{field_name} must be a safe identifier") from None


def _validated_hash(value: str, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a SHA-256 identifier")
    length = len(value)
    if length != _SHA256_LENGTH or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a SHA-256 identifier")
    return value


def _contains_list_comprehension(value: str) -> bool:
    for line in value.splitlines():
        depth = 0
        cursor = 0
        for marker in _LIST_COMPREHENSION.finditer(line):
            while cursor < marker.start():
                character = line[cursor]
                if character == "[":
                    depth += 1
                elif character == "]" and depth:
                    depth -= 1
                cursor += 1
            if depth:
                return True
            cursor = marker.end()
    return False


def _normalize_safe_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validate_safe_text(value: str, *, allow_empty: bool) -> str:
    if type(value) is not str:
        raise ValueError("citation content is not safe")
    length = len(value)
    if length > _MAX_SAFE_TEXT_LENGTH or "\t" in value or "|" in value:
        raise ValueError("citation content is not safe")
    normalized = _normalize_safe_text(value)
    try:
        if not allow_empty and not normalized:
            raise ValueError
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise ValueError
        assert_safe_payload({"text": normalized})
        if "\t" in normalized or "|" in normalized:
            raise ValueError
        if _contains_list_comprehension(normalized):
            raise ValueError
        if any(
            pattern.search(normalized) is not None
            for pattern in (
                _URI,
                _EMAIL,
                _WINDOWS_PATH,
                _UNC_PATH,
                _UNIX_PATH,
                _UUID,
                _ENTITY_ID,
                _LONG_NUMBER,
                _SEGMENT_FIELD,
                _SECRET_REFERENCE,
                _SECRET_MATERIAL,
                _CODE_LINE,
                _ASSIGNMENT,
                _JSON_ROW,
                _RAW_ROW_FIELD,
                _SECRET_ALIAS_ASSIGNMENT,
                _INLINE_CODE,
                _DELIMITED_ROW,
            )
        ):
            raise ValueError
        if "`" in normalized or "~~~" in normalized:
            raise ValueError
        if _MARKDOWN_TABLE_ROW.search(normalized) and _MARKDOWN_TABLE_SEPARATOR.search(
            normalized
        ):
            raise ValueError
        comma_lines = [
            line for line in normalized.splitlines() if line.count(",") >= 1 and line.strip()
        ]
        if comma_lines:
            raise ValueError
    except Exception:
        raise ValueError("citation content is not safe") from None
    return normalized


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "BuildResult",
    "Citation",
    "IndexIntegrityError",
    "ProviderSafeSummary",
    "QueryResult",
    "UnsafeContentError",
    "UnsafeIndexRequestError",
]
