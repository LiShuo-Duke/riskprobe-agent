"""Strict authorization subjects, calls, capabilities, and query budgets."""

from __future__ import annotations

import re
from enum import StrEnum
from threading import Lock

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


_PRINCIPAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_DATASET_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicyDeniedError(PermissionError):
    """Raised when a capability is absent from the selected profile."""


class QueryBudgetExceededError(PolicyDeniedError):
    """Raised when an otherwise authorized call exceeds its shared budget."""


class Role(StrEnum):
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    OPERATOR = "operator"


class Capability(StrEnum):
    INSPECT = "inspect"
    DISCOVER = "discover"
    DIAGNOSE = "diagnose"
    RECOMMEND = "recommend"
    RUN = "run"
    STATUS = "status"
    TRACE = "trace"
    EVIDENCE_LOOKUP = "evidence_lookup"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class Principal(_StrictFrozenModel):
    principal_id: str
    role: Role

    @field_validator("principal_id")
    @classmethod
    def validate_principal_id(cls, value: str) -> str:
        if _PRINCIPAL_ID.fullmatch(value) is None:
            raise ValueError("principal_id must be a public identifier")
        return value


class ToolCall(_StrictFrozenModel):
    capability: Capability
    dataset_id: str | None = None
    run_id: str | None = None
    evidence_id: str | None = None

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str | None) -> str | None:
        if value is not None and _DATASET_ID.fullmatch(value) is None:
            raise ValueError("dataset_id must be a registered identifier")
        return value

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is not None and _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("run_id must be a public identifier")
        return value

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("evidence_id must be a SHA-256 identifier")
        return value


class Budget(BaseModel):
    """Mutable, bounded counter deliberately shared across authorized calls."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )

    max_queries: int = Field(
        ge=0,
        validation_alias=AliasChoices("max_queries", "query_limit", "limit"),
    )
    used_queries: int = Field(default=0, ge=0)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    @model_validator(mode="after")
    def require_usage_within_limit(self) -> Budget:
        if self.used_queries > self.max_queries:
            raise ValueError("used_queries cannot exceed max_queries")
        return self

    @property
    def remaining_queries(self) -> int:
        return self.max_queries - self.used_queries

    @property
    def query_limit(self) -> int:
        return self.max_queries

    def consume(self, amount: int = 1) -> bool:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("query cost must be a non-negative integer")
        with self._lock:
            if self.used_queries + amount > self.max_queries:
                return False
            self.used_queries += amount
            return True
