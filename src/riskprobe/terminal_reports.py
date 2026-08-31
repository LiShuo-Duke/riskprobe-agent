from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from riskprobe.agents.contracts import AgentResult
from riskprobe.agents.decision_contracts import DecisionFinding
from riskprobe.analysis_contracts import AnalysisSummary, DecisionSummary
from riskprobe.models import FrozenModel
from riskprobe.privacy import assert_safe_payload
from riskprobe.report_models import ReportModel
from riskprobe.reporting import render_report_markdown
from riskprobe.reporting_docx import render_report_docx

_REPORT_SCHEMA = "riskprobe.terminal-report-manifest.v1"
_REPORT_ROOT_NAME = ".riskprobe-terminal-reports"
_MARKDOWN_NAME = "final_risk_report.md"
_DOCX_NAME = "final_risk_report.docx"
_MANIFEST_NAME = "final_report_manifest.json"
_ALLOWED_PARTIAL_FILES = frozenset({_MARKDOWN_NAME, _DOCX_NAME})
_ALLOWED_COMPLETE_FILES = frozenset({_MARKDOWN_NAME, _DOCX_NAME, _MANIFEST_NAME})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_REPORT_BYTES = 64 * 1024 * 1024


class TerminalReportError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class TerminalReportSubject:
    idempotency_key: str
    run_id: str
    context_id: str | None
    findings: tuple[DecisionFinding, ...]
    proposal_action_codes: tuple[str, ...]
    diagnosis_evidence_ids: tuple[str, ...]
    agent_result: AgentResult | None
    analysis_summary: AnalysisSummary | None
    decision_summary: DecisionSummary | None
    terminal_status: Literal["accepted", "rejected", "no_action", "failed"]
    decision_reason_codes: tuple[str, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not self.idempotency_key or len(self.idempotency_key) > 256:
            raise ValueError("idempotency key is invalid")
        if _PUBLIC_ID.fullmatch(self.run_id) is None:
            raise ValueError("run ID is invalid")
        if self.context_id is not None and _SHA256.fullmatch(self.context_id) is None:
            raise ValueError("context ID is invalid")
        if len(self.proposal_action_codes) != len(set(self.proposal_action_codes)):
            raise ValueError("proposal actions must be unique")
        if len(self.decision_reason_codes) != len(set(self.decision_reason_codes)):
            raise ValueError("decision reasons must be unique")
        assert_safe_payload({"decision_reason_codes": self.decision_reason_codes})
        if len(self.diagnosis_evidence_ids) != len(set(self.diagnosis_evidence_ids)) or any(
            _SHA256.fullmatch(item) is None for item in self.diagnosis_evidence_ids
        ):
            raise ValueError("diagnosis evidence IDs are invalid")
        if self.terminal_status == "failed" and self.error_code is None:
            raise ValueError("failed reports require an error code")
        if self.terminal_status != "failed" and self.error_code is not None:
            raise ValueError("successful terminal reports cannot contain an error code")


class TerminalReportFile(FrozenModel):
    name: Literal["final_risk_report.md", "final_risk_report.docx"]
    sha256: str
    size: int = Field(ge=0, le=_MAX_REPORT_BYTES)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("file hash must be SHA-256")
        return value


class TerminalReportManifest(FrozenModel):
    schema_version: Literal["riskprobe.terminal-report-manifest.v1"] = _REPORT_SCHEMA
    report_id: str
    run_id: str
    context_id: str | None = None
    session_id: str | None = None
    terminal_status: Literal["accepted", "rejected", "no_action", "failed"]
    model_sha256: str
    markdown: TerminalReportFile
    docx: TerminalReportFile

    @field_validator("report_id", "model_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("manifest hashes must be SHA-256")
        return value

    @field_validator("run_id", "session_id")
    @classmethod
    def validate_ids(cls, value: str | None) -> str | None:
        if value is not None and _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("manifest identity is invalid")
        return value

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("context ID is invalid")
        return value

    @model_validator(mode="after")
    def validate_files(self) -> TerminalReportManifest:
        if self.markdown.name != _MARKDOWN_NAME or self.docx.name != _DOCX_NAME:
            raise ValueError("terminal report filenames are invalid")
        return self


class TerminalReportStore:
    def __init__(self, state_dir: Path) -> None:
        self._state_dir = Path(state_dir)
        self._root = self._state_dir / _REPORT_ROOT_NAME
        _ensure_private_directory(self._state_dir, create=False)
        _ensure_private_directory(self._root, create=True)

    def ensure_published(
        self,
        *,
        subject: TerminalReportSubject,
        model: ReportModel,
    ) -> TerminalReportManifest:
        expected_session_id = (
            subject.agent_result.session_id
            if subject.agent_result is not None
            else None
        )
        if (
            model.scope != "terminal"
            or model.run_id != subject.run_id
            or model.terminal_status != subject.terminal_status
            or model.session_id != expected_session_id
        ):
            raise TerminalReportError("terminal_report_integrity_failed")
        report_id = derive_report_id(subject)
        model_content = _canonical_json_bytes(model.model_dump(mode="json"))
        model_sha256 = hashlib.sha256(model_content).hexdigest()
        expected_identity = {
            "report_id": report_id,
            "run_id": subject.run_id,
            "context_id": subject.context_id,
            "session_id": (
                subject.agent_result.session_id
                if subject.agent_result is not None
                else None
            ),
            "terminal_status": subject.terminal_status,
            "model_sha256": model_sha256,
        }
        with self._report_lock(report_id):
            report_dir = self._root / report_id
            _ensure_private_directory(report_dir, create=True)
            _remove_orphan_temporaries(report_dir)
            existing = self._load_existing(report_dir)
            if existing is not None:
                self._validate_identity(existing, expected_identity)
                self._verify_bundle(report_dir, existing)
                return existing
            self._validate_partial_directory(report_dir)
            try:
                markdown_content = render_report_markdown(model).encode("utf-8")
                docx_content = render_report_docx(model)
            except BaseException as error:
                raise TerminalReportError("terminal_report_render_failed") from error
            if len(markdown_content) > _MAX_REPORT_BYTES or len(docx_content) > _MAX_REPORT_BYTES:
                raise TerminalReportError("terminal_report_size_exceeded")
            markdown_record = _file_record(_MARKDOWN_NAME, markdown_content)
            docx_record = _file_record(_DOCX_NAME, docx_content)
            manifest = TerminalReportManifest(
                **expected_identity,
                markdown=markdown_record,
                docx=docx_record,
            )
            try:
                _atomic_write(report_dir / _MARKDOWN_NAME, markdown_content)
                _atomic_write(report_dir / _DOCX_NAME, docx_content)
                _atomic_write(
                    report_dir / _MANIFEST_NAME,
                    _canonical_json_bytes(manifest.model_dump(mode="json")),
                )
                _fsync_directory(report_dir)
            except BaseException as error:
                raise TerminalReportError("terminal_report_publish_failed") from error
            self._verify_bundle(report_dir, manifest)
            return manifest

    @contextmanager
    def _report_lock(self, report_id: str) -> Iterator[None]:
        lock_path = self._root / f".{report_id}.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise TerminalReportError("terminal_report_lock_failed") from error
        try:
            os.fchmod(descriptor, 0o600)
            _validate_private_stat(os.fstat(descriptor), directory=False)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except TerminalReportError:
            raise
        except BaseException as error:
            raise TerminalReportError("terminal_report_lock_failed") from error
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _load_existing(self, report_dir: Path) -> TerminalReportManifest | None:
        manifest_path = report_dir / _MANIFEST_NAME
        if not manifest_path.exists():
            return None
        try:
            content = _read_private_file(manifest_path)
            payload = json.loads(content)
            manifest = TerminalReportManifest.model_validate(payload)
        except BaseException as error:
            raise TerminalReportError("terminal_report_integrity_failed") from error
        if content != _canonical_json_bytes(manifest.model_dump(mode="json")):
            raise TerminalReportError("terminal_report_integrity_failed")
        return manifest

    @staticmethod
    def _validate_identity(
        manifest: TerminalReportManifest,
        expected: dict[str, object],
    ) -> None:
        actual = {
            "report_id": manifest.report_id,
            "run_id": manifest.run_id,
            "context_id": manifest.context_id,
            "session_id": manifest.session_id,
            "terminal_status": manifest.terminal_status,
            "model_sha256": manifest.model_sha256,
        }
        if actual != expected:
            raise TerminalReportError("terminal_report_integrity_failed")

    @staticmethod
    def _validate_partial_directory(report_dir: Path) -> None:
        try:
            names = {item.name for item in report_dir.iterdir()}
        except OSError as error:
            raise TerminalReportError("terminal_report_integrity_failed") from error
        if not names.issubset(_ALLOWED_PARTIAL_FILES):
            raise TerminalReportError("terminal_report_integrity_failed")

    @staticmethod
    def _verify_bundle(
        report_dir: Path,
        manifest: TerminalReportManifest,
    ) -> None:
        try:
            names = {item.name for item in report_dir.iterdir()}
        except OSError as error:
            raise TerminalReportError("terminal_report_integrity_failed") from error
        if names != _ALLOWED_COMPLETE_FILES:
            raise TerminalReportError("terminal_report_integrity_failed")
        for record in (manifest.markdown, manifest.docx):
            content = _read_private_file(report_dir / record.name)
            if len(content) != record.size or hashlib.sha256(content).hexdigest() != record.sha256:
                raise TerminalReportError("terminal_report_integrity_failed")


def derive_report_id(subject: TerminalReportSubject) -> str:
    payload = "\n".join(
        (
            subject.idempotency_key,
            subject.run_id,
            subject.context_id or "no-context",
            (
                subject.agent_result.leaf_node_id
                if subject.agent_result is not None
                else "no-result"
            ),
            subject.terminal_status,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(name: str, content: bytes) -> TerminalReportFile:
    return TerminalReportFile(
        name=name,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _ensure_private_directory(path: Path, *, create: bool) -> None:
    try:
        if create:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = path.stat(follow_symlinks=False)
        _validate_private_stat(details, directory=True)
        if stat.S_IMODE(details.st_mode) != 0o700:
            path.chmod(0o700)
            details = path.stat(follow_symlinks=False)
            _validate_private_stat(details, directory=True)
    except BaseException as error:
        raise TerminalReportError("terminal_report_storage_unavailable") from error


def _validate_private_stat(details: os.stat_result, *, directory: bool) -> None:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected_type(details.st_mode)
        or details.st_uid != os.getuid()
        or (not directory and details.st_nlink != 1)
    ):
        raise TerminalReportError("terminal_report_integrity_failed")


def _remove_orphan_temporaries(report_dir: Path) -> None:
    prefixes = tuple(f".{name}." for name in _ALLOWED_COMPLETE_FILES)
    try:
        entries = tuple(report_dir.iterdir())
    except OSError as error:
        raise TerminalReportError("terminal_report_integrity_failed") from error
    for entry in entries:
        if not entry.name.endswith(".tmp") or not entry.name.startswith(prefixes):
            continue
        try:
            details = entry.stat(follow_symlinks=False)
            _validate_private_stat(details, directory=False)
            entry.unlink()
        except BaseException as error:
            raise TerminalReportError("terminal_report_integrity_failed") from error


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        _fsync_directory(path.parent)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _read_private_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TerminalReportError("terminal_report_integrity_failed") from error
    try:
        details = os.fstat(descriptor)
        _validate_private_stat(details, directory=False)
        if stat.S_IMODE(details.st_mode) != 0o600 or details.st_size > _MAX_REPORT_BYTES:
            raise TerminalReportError("terminal_report_integrity_failed")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(_MAX_REPORT_BYTES + 1)
        if len(content) > _MAX_REPORT_BYTES:
            raise TerminalReportError("terminal_report_integrity_failed")
        return content
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "TerminalReportError",
    "TerminalReportManifest",
    "TerminalReportStore",
    "TerminalReportSubject",
    "derive_report_id",
]
