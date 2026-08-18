"""Strict execution contracts for resumable RiskProbe runs."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str

    def __post_init__(self) -> None:
        if _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError(
                "run_id must be a 16-character lowercase hexadecimal value"
            )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content-addressed reference to one plain file in a run directory."""

    filename: str
    sha256: str
    size: int
    schema_version: str

    def __post_init__(self) -> None:
        if not _is_plain_filename(self.filename):
            raise ValueError("artifact filename must be a plain file name")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("artifact sha256 must be lowercase hexadecimal")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("artifact size must be a non-negative integer")
        if (
            not isinstance(self.schema_version, str)
            or not self.schema_version
            or len(self.schema_version) > 128
            or any(character.isspace() for character in self.schema_version)
        ):
            raise ValueError("schema_version must be a non-empty token")

    @classmethod
    def from_path(cls, path: Path, schema_version: str) -> ArtifactRef:
        candidate = Path(path)
        if not _is_plain_filename(candidate.name):
            raise ValueError("artifact filename must be a plain file name")
        try:
            digest, size = _secure_file_digest(candidate)
        except (OSError, ValueError) as error:
            raise ValueError("artifact must be a regular non-symlink file") from error
        return cls(
            filename=candidate.name,
            sha256=digest,
            size=size,
            schema_version=schema_version,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactRef:
        try:
            return cls(
                filename=payload["filename"],
                sha256=payload["sha256"],
                size=payload["size"],
                schema_version=payload["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("artifact reference is invalid") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size": self.size,
        }

    def verify(
        self,
        run_dir: Path,
        *,
        expected_schema_version: str | None = None,
    ) -> bool:
        if (
            expected_schema_version is not None
            and self.schema_version != expected_schema_version
        ):
            return False
        try:
            digest, size = _secure_file_digest(Path(run_dir) / self.filename)
        except (OSError, ValueError):
            return False
        return digest == self.sha256 and size == self.size


@dataclass(frozen=True, slots=True)
class NodeCheckpoint:
    node_id: str
    status: NodeStatus
    attempt: int
    input_fingerprint: str
    output: Mapping[str, Any]
    updated_at: str
    artifact_refs: tuple[ArtifactRef, ...] = ()

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        return self.artifact_refs


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Finite retry contract. Backoff values are trace metadata, never sleeps."""

    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        values = tuple(float(value) for value in self.backoff_seconds)
        if len(values) > self.max_attempts - 1:
            raise ValueError("backoff_seconds cannot exceed the retry count")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("backoff_seconds must contain finite non-negative values")
        object.__setattr__(self, "backoff_seconds", values)

    def backoff_for_attempt(self, attempt: int) -> float:
        if attempt <= 1:
            return 0.0
        index = attempt - 2
        if index >= len(self.backoff_seconds):
            return 0.0
        return self.backoff_seconds[index]


@dataclass(frozen=True, slots=True)
class RunBudget:
    max_nodes: int | None = 64
    max_attempts: int | None = 256
    deadline: datetime | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_nodes", self.max_nodes),
            ("max_attempts", self.max_attempts),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.deadline is not None:
            if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
                raise ValueError("deadline must be timezone-aware")
            object.__setattr__(
                self,
                "deadline",
                self.deadline.astimezone(timezone.utc),
            )

    def ensure_before_deadline(self) -> None:
        if self.deadline is not None and datetime.now(timezone.utc) >= self.deadline:
            raise RuntimeError("run deadline has been exceeded")


def _is_plain_filename(filename: str) -> bool:
    return (
        isinstance(filename, str)
        and bool(filename)
        and filename not in {".", ".."}
        and Path(filename).name == filename
        and "/" not in filename
        and "\\" not in filename
        and "\x00" not in filename
    )


def _secure_file_digest(path: Path) -> tuple[str, int]:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise ValueError("artifact is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (details.st_dev, details.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("artifact changed while opening")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or size != after.st_size
        ):
            raise ValueError("artifact changed while hashing")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)
