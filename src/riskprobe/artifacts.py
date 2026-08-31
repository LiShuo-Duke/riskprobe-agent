import dataclasses
import fcntl
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO

import polars as pl
from pydantic import BaseModel

from riskprobe.execution.store import ExecutionStore


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


_LEGACY_REQUIRED_ARTIFACTS = (
    "manifest.json",
    "metadata_report.json",
    "data_profile.json",
    "candidate_rules.parquet",
    "evidence_cards.json",
    "risk_report.md",
)
_LEGACY_SCORECARD_ARTIFACTS = (*_LEGACY_REQUIRED_ARTIFACTS, "scorecard.json")
_REQUIRED_ARTIFACTS = (*_LEGACY_REQUIRED_ARTIFACTS, "analysis_summary.json")
_SCORECARD_ARTIFACTS = (*_REQUIRED_ARTIFACTS, "scorecard.json")
_LEGACY_ARTIFACT_SETS = frozenset(
    (_LEGACY_REQUIRED_ARTIFACTS, _LEGACY_SCORECARD_ARTIFACTS)
)
_CURRENT_ARTIFACT_SETS = frozenset((_REQUIRED_ARTIFACTS, _SCORECARD_ARTIFACTS))
_RUN_ID = re.compile(r"^[0-9a-f]{16}$")
_LEGACY_MANIFEST_IDENTITY_FIELDS = (
    "run_id",
    "config_fingerprint",
    "data_fingerprint",
    "code_version",
    "dataset_id",
    "time_validation_enabled",
)
_MANIFEST_IDENTITY_FIELDS = (
    *_LEGACY_MANIFEST_IDENTITY_FIELDS,
    "time_validation_mode",
    "time_validation_applied",
)
_MANIFEST_FIELDS = {
    "schema_version",
    "artifacts",
    "artifact_integrity",
    *_MANIFEST_IDENTITY_FIELDS,
}
_LEGACY_MANIFEST_FIELDS = {
    "artifacts",
    "artifact_integrity",
    *_LEGACY_MANIFEST_IDENTITY_FIELDS,
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
    manifest_fields = set(manifest)
    if manifest_fields == _MANIFEST_FIELDS:
        if manifest.get("schema_version") != "riskprobe.manifest.v1":
            return False
        allowed_artifact_sets = _CURRENT_ARTIFACT_SETS
        identity_fields = _MANIFEST_IDENTITY_FIELDS
    elif manifest_fields == _LEGACY_MANIFEST_FIELDS:
        allowed_artifact_sets = _LEGACY_ARTIFACT_SETS
        identity_fields = _LEGACY_MANIFEST_IDENTITY_FIELDS
    else:
        return False
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or tuple(artifacts) not in allowed_artifact_sets
    ):
        return False
    artifact_names = tuple(artifacts)
    integrity_artifacts = artifact_names[1:]
    identity = {name: manifest[name] for name in identity_fields}
    expected = {name: expected_identity.get(name) for name in identity_fields}
    if identity != expected:
        return False
    integrity = manifest.get("artifact_integrity")
    if not isinstance(integrity, dict) or set(integrity) != set(integrity_artifacts):
        return False
    try:
        directory_entries = {entry.name for entry in run_dir.iterdir()}
    except OSError:
        return False
    allowed_entries = set(artifact_names)
    if (run_dir / ".incomplete").is_file():
        allowed_entries.add(".incomplete")
    if directory_entries != allowed_entries:
        return False
    for name in integrity_artifacts:
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

    def require_binding(
        self,
        *,
        config_fingerprint: str,
        dataset_id: str | None,
        time_validation_enabled: bool | None,
        time_validation_mode: str | None,
        time_validation_applied: bool | None,
    ) -> None:
        """Require this context to be a complete run for the supplied public binding."""

        expected = {
            "config_fingerprint": config_fingerprint,
            "dataset_id": dataset_id,
            "time_validation_enabled": time_validation_enabled,
            "time_validation_mode": time_validation_mode,
            "time_validation_applied": time_validation_applied,
        }
        if self._writable or any(
            self._expected_identity.get(name) != value
            for name, value in expected.items()
        ) or not _is_complete_run(
            self.run_dir,
            self._expected_identity,
            self._integrity_anchor,
        ):
            raise RuntimeError(f"run {self.run_id} is not complete")

    def read_verified_artifact(self, name: str) -> bytes:
        """Read one immutable artifact while enforcing its anchored integrity."""

        if name not in _SCORECARD_ARTIFACTS[1:]:
            raise ValueError("artifact is not readable through the integrity view")
        self.require_binding(
            config_fingerprint=str(self._expected_identity["config_fingerprint"]),
            dataset_id=self._expected_identity["dataset_id"],
            time_validation_enabled=self._expected_identity["time_validation_enabled"],
            time_validation_mode=self._expected_identity["time_validation_mode"],
            time_validation_applied=self._expected_identity["time_validation_applied"],
        )
        try:
            manifest = json.loads((self.run_dir / "manifest.json").read_bytes())
            artifacts = manifest["artifacts"]
            if not isinstance(artifacts, list) or name not in artifacts:
                raise ValueError("artifact is not declared by the manifest")
            expected_integrity = manifest["artifact_integrity"][name]
            content = self._target(name).read_bytes()
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"run {self.run_id} is not complete") from error
        actual_integrity = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        if actual_integrity != expected_integrity or not _is_complete_run(
            self.run_dir,
            self._expected_identity,
            self._integrity_anchor,
        ):
            raise RuntimeError(f"run {self.run_id} is not complete")
        return content

    def finalize(self) -> None:
        self._ensure_writable()
        if not _is_complete_run(self.run_dir, self._expected_identity):
            raise RuntimeError(f"run {self.run_id} is not complete")
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        identity_fields = (
            _MANIFEST_IDENTITY_FIELDS
            if "schema_version" in manifest
            else _LEGACY_MANIFEST_IDENTITY_FIELDS
        )
        _write_canonical_json(
            self._integrity_anchor,
            {
                "artifact_integrity": manifest["artifact_integrity"],
                "identity": {
                    name: manifest[name] for name in identity_fields
                },
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

    def release(self) -> None:
        """Release the writer lock while intentionally preserving an incomplete run."""
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

    def open_verified(self, run_id: str) -> RunContext:
        """Open an existing finalized run through its anchored read-only view."""

        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a 16-character lowercase hexadecimal value")
        run_dir = self.runs_dir / run_id
        integrity_anchor = self.runs_dir / f".{run_id}.integrity.json"
        if (
            run_dir.is_symlink()
            or not run_dir.is_dir()
            or (run_dir / ".incomplete").exists()
        ):
            raise RuntimeError(f"run {run_id} is not complete")
        try:
            manifest = json.loads((run_dir / "manifest.json").read_bytes())
            if not isinstance(manifest, dict):
                raise TypeError("manifest must be an object")
            manifest_fields = set(manifest)
            if manifest_fields == _MANIFEST_FIELDS:
                identity_fields = _MANIFEST_IDENTITY_FIELDS
            elif manifest_fields == _LEGACY_MANIFEST_FIELDS:
                identity_fields = _LEGACY_MANIFEST_IDENTITY_FIELDS
            else:
                raise TypeError("manifest fields are invalid")
            expected_identity = {
                name: manifest[name] for name in identity_fields
            }
            if identity_fields == _LEGACY_MANIFEST_IDENTITY_FIELDS:
                enabled = expected_identity["time_validation_enabled"]
                if enabled is not None and type(enabled) is not bool:
                    raise TypeError("legacy time validation identity is invalid")
                expected_identity["time_validation_mode"] = (
                    None if enabled is None else "strict" if enabled else "disabled"
                )
                expected_identity["time_validation_applied"] = enabled
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"run {run_id} is not complete") from error
        if expected_identity["run_id"] != run_id or not _is_complete_run(
            run_dir,
            expected_identity,
            integrity_anchor,
        ):
            raise RuntimeError(f"run {run_id} is not complete")
        return RunContext(
            run_id,
            run_dir,
            is_existing=True,
            expected_identity=expected_identity,
            integrity_anchor=integrity_anchor,
        )

    def create(
        self,
        config: Any,
        data_fingerprint: str,
        code_version: str,
        *,
        dataset_id: str | None = None,
        time_validation_enabled: bool | None = None,
        time_validation_mode: str | None = None,
        time_validation_applied: bool | None = None,
    ) -> RunContext:
        run_id = self.compute_run_id(config, data_fingerprint, code_version)
        expected_identity = {
            "run_id": run_id,
            "config_fingerprint": self.config_fingerprint(config),
            "data_fingerprint": data_fingerprint,
            "code_version": code_version,
            "dataset_id": dataset_id,
            "time_validation_enabled": time_validation_enabled,
            "time_validation_mode": time_validation_mode,
            "time_validation_applied": time_validation_applied,
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
                    runtime_database = ExecutionStore.database_path_for(
                        self.runs_dir, run_id
                    )
                    try:
                        runtime_database.lstat()
                    except FileNotFoundError:
                        shutil.rmtree(run_dir)
                        integrity_anchor.unlink(missing_ok=True)
                    else:
                        if not ExecutionStore.is_secure_database(runtime_database):
                            raise RuntimeError(
                                f"run {run_id} does not have a secure runtime database"
                            )
                        return RunContext(
                            run_id,
                            run_dir,
                            is_existing=False,
                            expected_identity=expected_identity,
                            integrity_anchor=integrity_anchor,
                            lock_handle=lock_handle,
                        )
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
