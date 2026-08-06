import dataclasses
import fcntl
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO

import polars as pl
from pydantic import BaseModel


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__class__.model_fields
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON values must be finite")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


_REQUIRED_ARTIFACTS = (
    "manifest.json",
    "metadata_report.json",
    "data_profile.json",
    "candidate_rules.parquet",
    "evidence_cards.json",
    "risk_report.md",
)
_INTEGRITY_ARTIFACTS = _REQUIRED_ARTIFACTS[1:]


def _file_integrity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _is_complete_run(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.json"
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return False
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    if manifest_bytes != f"{_canonical_json(manifest)}\n".encode("utf-8"):
        return False
    if manifest.get("artifacts") != list(_REQUIRED_ARTIFACTS):
        return False
    integrity = manifest.get("artifact_integrity")
    if not isinstance(integrity, dict) or set(integrity) != set(_INTEGRITY_ARTIFACTS):
        return False
    try:
        directory_entries = {entry.name for entry in run_dir.iterdir()}
    except OSError:
        return False
    allowed_entries = set(_REQUIRED_ARTIFACTS)
    if (run_dir / ".incomplete").is_file():
        allowed_entries.add(".incomplete")
    if directory_entries != allowed_entries:
        return False
    for name in _INTEGRITY_ARTIFACTS:
        path = run_dir / name
        record = integrity.get(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(record, dict)
            or set(record) != {"sha256", "size"}
            or not isinstance(record.get("sha256"), str)
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] < 0
        ):
            return False
        try:
            if _file_integrity(path) != record:
                return False
        except OSError:
            return False
    return True


def _release_lock(handle: BinaryIO | None) -> None:
    if handle is not None and not handle.closed:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class RunContext:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        *,
        is_existing: bool,
        lock_handle: BinaryIO | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.is_existing = is_existing
        self._lock_handle = lock_handle
        self._writable = not is_existing

    def _target(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", "..", ".incomplete"}:
            raise ValueError("artifact name must be a plain file name")
        return self.run_dir / name

    def _ensure_writable(self) -> None:
        if not self._writable:
            raise FileExistsError(f"run {self.run_id} is immutable")

    def _atomic_bytes(self, name: str, content: bytes) -> None:
        self._ensure_writable()
        target = self._target(name)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.run_dir,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                temporary = Path(handle.name)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write_json(self, name: str, payload: Any) -> None:
        rendered = json.dumps(
            _jsonable(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._atomic_bytes(name, f"{rendered}\n".encode("utf-8"))

    def write_canonical_json(self, name: str, payload: Any) -> None:
        self._atomic_bytes(name, f"{_canonical_json(payload)}\n".encode("utf-8"))

    def write_text(self, name: str, content: str) -> None:
        self._atomic_bytes(name, content.encode("utf-8"))

    def write_parquet(self, name: str, frame: pl.DataFrame) -> None:
        self._ensure_writable()
        target = self._target(name)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.run_dir,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def finalize(self) -> None:
        self._ensure_writable()
        if not _is_complete_run(self.run_dir):
            raise RuntimeError(f"run {self.run_id} is not complete")
        (self.run_dir / ".incomplete").unlink()
        self._writable = False
        _release_lock(self._lock_handle)
        self._lock_handle = None

    def cleanup(self) -> None:
        if self._writable and (self.run_dir / ".incomplete").exists():
            shutil.rmtree(self.run_dir)
        self._writable = False
        _release_lock(self._lock_handle)
        self._lock_handle = None


class RunStore:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def compute_run_id(
        self,
        config: Any,
        data_fingerprint: str,
        code_version: str,
    ) -> str:
        payload = f"{_canonical_json(config)}{data_fingerprint}{code_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def create(
        self,
        config: Any,
        data_fingerprint: str,
        code_version: str,
    ) -> RunContext:
        run_id = self.compute_run_id(config, data_fingerprint, code_version)
        run_dir = self.runs_dir / run_id
        incomplete = run_dir / ".incomplete"
        lock_handle = (self.runs_dir / f".{run_id}.lock").open("a+b")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_handle.close()
            raise RuntimeError(f"run {run_id} is active") from error

        try:
            if run_dir.exists():
                if incomplete.is_file():
                    shutil.rmtree(run_dir)
                elif run_dir.is_dir() and _is_complete_run(run_dir):
                    _release_lock(lock_handle)
                    return RunContext(run_id, run_dir, is_existing=True)
                elif run_dir.is_dir():
                    raise RuntimeError(f"run {run_id} is not complete")
                else:
                    raise FileExistsError(run_dir)
            run_dir.mkdir()
            incomplete.write_bytes(b"")
            return RunContext(
                run_id,
                run_dir,
                is_existing=False,
                lock_handle=lock_handle,
            )
        except BaseException:
            _release_lock(lock_handle)
            raise
