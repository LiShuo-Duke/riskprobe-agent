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
_MANIFEST_IDENTITY_FIELDS = (
    "run_id",
    "config_fingerprint",
    "data_fingerprint",
    "code_version",
    "dataset_id",
    "time_validation_enabled",
)
_MANIFEST_FIELDS = {
    "artifacts",
    "artifact_integrity",
    *_MANIFEST_IDENTITY_FIELDS,
}


def _file_integrity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _write_canonical_json(path: Path, payload: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(f"{_canonical_json(payload)}\n".encode("utf-8"))
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(path)
        path.chmod(0o400)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _is_complete_run(
    run_dir: Path,
    expected_identity: Mapping[str, Any],
    integrity_anchor: Path | None = None,
) -> bool:
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
    if set(manifest) != _MANIFEST_FIELDS:
        return False
    if manifest.get("artifacts") != list(_REQUIRED_ARTIFACTS):
        return False
    identity = {name: manifest.get(name) for name in _MANIFEST_IDENTITY_FIELDS}
    if identity != dict(expected_identity):
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
    if integrity_anchor is None:
        return True
    try:
        if integrity_anchor.is_symlink() or not integrity_anchor.is_file():
            return False
        anchor_bytes = integrity_anchor.read_bytes()
        anchor = json.loads(anchor_bytes)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    expected_anchor = {"artifact_integrity": integrity, "identity": identity}
    return (
        isinstance(anchor, dict)
        and anchor_bytes == f"{_canonical_json(anchor)}\n".encode("utf-8")
        and anchor == expected_anchor
    )


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
        expected_identity: Mapping[str, Any],
        integrity_anchor: Path,
        lock_handle: BinaryIO | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.is_existing = is_existing
        self._expected_identity = dict(expected_identity)
        self._integrity_anchor = integrity_anchor
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
        if not _is_complete_run(self.run_dir, self._expected_identity):
            raise RuntimeError(f"run {self.run_id} is not complete")
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        _write_canonical_json(
            self._integrity_anchor,
            {
                "artifact_integrity": manifest["artifact_integrity"],
                "identity": self._expected_identity,
            },
        )
        if not _is_complete_run(
            self.run_dir, self._expected_identity, self._integrity_anchor
        ):
            self._integrity_anchor.unlink(missing_ok=True)
            raise RuntimeError(f"run {self.run_id} is not complete")
        (self.run_dir / ".incomplete").unlink()
        self._writable = False
        _release_lock(self._lock_handle)
        self._lock_handle = None

    def cleanup(self) -> None:
        if self._writable:
            self._integrity_anchor.unlink(missing_ok=True)
            if (self.run_dir / ".incomplete").exists():
                shutil.rmtree(self.run_dir)
        self._writable = False
        _release_lock(self._lock_handle)
        self._lock_handle = None


class RunStore:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _identity_config(config: Any) -> Any:
        payload = _jsonable(config)
        if not isinstance(payload, dict):
            return payload
        identity = dict(payload)
        dataset = identity.get("dataset")
        if isinstance(dataset, dict) and "path" in dataset:
            identity["dataset"] = {**dataset, "path": "local-parquet-input"}
        features = identity.get("features")
        if isinstance(features, dict) and features.get("explicit_catalog") is not None:
            catalog = Path(str(features["explicit_catalog"]))
            try:
                catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
            except OSError:
                catalog_digest = "unreadable"
            identity["features"] = {
                **features,
                "explicit_catalog": {"content_sha256": catalog_digest},
            }
        return identity

    @classmethod
    def config_fingerprint(cls, config: Any) -> str:
        return hashlib.sha256(
            f"{_canonical_json(cls._identity_config(config))}\n".encode("utf-8")
        ).hexdigest()

    def compute_run_id(
        self,
        config: Any,
        data_fingerprint: str,
        code_version: str,
    ) -> str:
        payload = f"{_canonical_json(self._identity_config(config))}{data_fingerprint}{code_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def create(
        self,
        config: Any,
        data_fingerprint: str,
        code_version: str,
        *,
        dataset_id: str | None = None,
        time_validation_enabled: bool | None = None,
    ) -> RunContext:
        run_id = self.compute_run_id(config, data_fingerprint, code_version)
        expected_identity = {
            "run_id": run_id,
            "config_fingerprint": self.config_fingerprint(config),
            "data_fingerprint": data_fingerprint,
            "code_version": code_version,
            "dataset_id": dataset_id,
            "time_validation_enabled": time_validation_enabled,
        }
        run_dir = self.runs_dir / run_id
        incomplete = run_dir / ".incomplete"
        integrity_anchor = self.runs_dir / f".{run_id}.integrity.json"
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
                    integrity_anchor.unlink(missing_ok=True)
                elif run_dir.is_dir() and _is_complete_run(
                    run_dir, expected_identity, integrity_anchor
                ):
                    _release_lock(lock_handle)
                    return RunContext(
                        run_id,
                        run_dir,
                        is_existing=True,
                        expected_identity=expected_identity,
                        integrity_anchor=integrity_anchor,
                    )
                elif run_dir.is_dir():
                    raise RuntimeError(f"run {run_id} is not complete")
                else:
                    raise FileExistsError(run_dir)
            else:
                integrity_anchor.unlink(missing_ok=True)
            run_dir.mkdir()
            incomplete.write_bytes(b"")
            return RunContext(
                run_id,
                run_dir,
                is_existing=False,
                expected_identity=expected_identity,
                integrity_anchor=integrity_anchor,
                lock_handle=lock_handle,
            )
        except BaseException:
            _release_lock(lock_handle)
            raise
