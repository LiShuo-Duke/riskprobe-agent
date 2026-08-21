from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from riskprobe.artifacts import RunContext, RunStore
from riskprobe.config import ProjectConfig
from riskprobe.dates import normalize_date_series
from riskprobe.evidence import EvidenceRecord, EvidenceStore, PrivacyClass
from riskprobe.execution.models import ArtifactRef
from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.institutions import discover_local_rules
from riskprobe.models import EvidenceCard, RiskRule, SliceMetrics
from riskprobe.monitoring.models import (
    DiagnosticReport,
    ReferenceSnapshot,
    RiskFinding,
    SafeProfile,
)
from riskprobe.monitoring.reference import build_reference_snapshot
from riskprobe.monitoring.service import diagnose_dataset
from riskprobe.profiling import DatasetProfile, profile_dataset
from riskprobe.recommendations import build_recommendations
from riskprobe.reporting import (
    evidence_sort_key,
    redact_limitation,
    redact_segment_value,
    render_risk_report,
    safe_dataset_id,
)
from riskprobe.rules.discovery import DiscoveryResult, discover_rules, discover_with_metrics
from riskprobe.rules.validation import validate_rules
from riskprobe.runtime import RunRuntime

if TYPE_CHECKING:
    from riskprobe.agents import AgentResult
    from riskprobe.agents.decision_controller import DecisionController
    from riskprobe.agents.decision_providers import (
        DecisionProvider,
        DecisionProviderConfig,
    )
    from riskprobe.evals import (
        EvalReport,
        EvalReportV2,
        EvalRunner,
        EvalRunnerV2,
        EvalSuite,
        EvalSuiteV2,
        RunnerCallable,
        RunnerCallableV2,
    )
    from riskprobe.policy import Budget, Principal
    from riskprobe.rag import BuildResult, QueryResult
    from riskprobe.tools.models import (
        DiagnoseResponse,
        EvidenceLookupResponse,
        RecommendResponse,
        StatusResponse,
        TraceResponse,
    )

_DISCOVERY_SAMPLE_LIMIT = 50_000
_ARTIFACT_NAMES = (
    "manifest.json",
    "metadata_report.json",
    "data_profile.json",
    "candidate_rules.parquet",
    "evidence_cards.json",
    "risk_report.md",
)
_ARTIFACT_SCHEMAS = {
    "manifest.json": "riskprobe.manifest.v1",
    "metadata_report.json": "riskprobe.metadata-report.v1",
    "data_profile.json": "riskprobe.data-profile.v1",
    "candidate_rules.parquet": "riskprobe.candidate-rules.v1",
    "evidence_cards.json": "riskprobe.evidence-cards.v1",
    "risk_report.md": "riskprobe.risk-report.v1",
}
_NODE_ORDER = ("profile", "partition", "discover", "validate", "report", "finalize")
_SLICE_ORDER = {"dataset": 0, "segment": 1, "time": 2}
_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}
_SNAPSHOT_ROOT_PREFIX = "riskprobe-input-snapshots-"
_SNAPSHOT_DIR_PREFIX = "snapshot-"
_SNAPSHOT_MARKER_NAME = ".riskprobe-snapshot"
_SNAPSHOT_LOCK_NAME = ".lock"
_SNAPSHOT_FILE_NAME = "input.parquet"
_SNAPSHOT_MARKER = b"riskprobe raw snapshot v1\n"
_RUN_ID = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIAGNOSTIC_KIND = "diagnostic.finding"
_RECOMMENDATION_KIND = "recommendation"
_STATUS_MAP = {
    "pending": "pending",
    "running": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "interrupted": "failed",
    "invalidated": "failed",
    "cancelled": "cancelled",
}
_SERVICE_PRODUCER_VERSION = "riskprobe-service-v1"


@dataclass(frozen=True, slots=True)
class AgentCitationResult:
    """Completed agent result with metadata-only local citations kept separate."""

    agent_result: AgentResult
    citations: QueryResult


def _restore_json_tuples(value: object) -> object:
    """Restore tuple-shaped model fields after JSON list serialization."""
    if isinstance(value, list):
        return tuple(_restore_json_tuples(item) for item in value)
    if isinstance(value, dict):
        restored = {key: _restore_json_tuples(item) for key, item in value.items()}
        if {"rule", "train", "test", "grade"}.issubset(restored) and "max_time_decay" not in restored:
            restored["max_time_decay"] = 0.0
        return restored
    return value


def _package_version() -> str:
    try:
        return version("riskprobe-agent")
    except PackageNotFoundError:
        return "0.1.0"


def _code_identity(source_root: Path | None = None) -> str:
    """Return a deterministic identity for the package version and source contents."""
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    if root.is_dir():
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(part == "__pycache__" for part in path.parts)
            and path.suffix != ".pyc"
        )
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return f"{_package_version()}+src-{digest.hexdigest()[:16]}"


def _node_input_fingerprint(run_id: str, node_id: str) -> str:
    return hashlib.sha256(f"riskprobe-node-v1:{run_id}:{node_id}".encode("utf-8")).hexdigest()


def _safe_dataset_id(dataset_id: str) -> str:
    return safe_dataset_id(dataset_id)


def _is_owned_private_directory(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(details.st_mode)
        and details.st_uid == os.geteuid()
        and stat.S_IMODE(details.st_mode) == 0o700
    )


def _is_owned_regular_file(path: Path, modes: set[int]) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(details.st_mode)
        and details.st_uid == os.geteuid()
        and stat.S_IMODE(details.st_mode) in modes
    )


def _snapshot_root() -> Path:
    root = Path(tempfile.gettempdir()) / f"{_SNAPSHOT_ROOT_PREFIX}{os.geteuid()}"
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    if not _is_owned_private_directory(root):
        raise RuntimeError("riskprobe snapshot root is not private")
    return root


def _snapshot_name_is_safe(name: str) -> bool:
    token = name.removeprefix(_SNAPSHOT_DIR_PREFIX)
    return (
        name.startswith(_SNAPSHOT_DIR_PREFIX)
        and len(token) == 32
        and all(character in "0123456789abcdef" for character in token)
    )


def _write_private_marker(snapshot_dir: Path) -> None:
    marker = snapshot_dir / _SNAPSHOT_MARKER_NAME
    with marker.open("xb") as handle:
        handle.write(_SNAPSHOT_MARKER)
    marker.chmod(0o400)


def _create_snapshot_lock(snapshot_dir: Path) -> BinaryIO:
    lock_path = snapshot_dir / _SNAPSHOT_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_snapshot_lock(handle: BinaryIO | None) -> None:
    if handle is not None and not handle.closed:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _recovery_lock(snapshot_dir: Path) -> BinaryIO | None:
    lock_path = snapshot_dir / _SNAPSHOT_LOCK_NAME
    if not _is_owned_private_directory(snapshot_dir) or not _is_owned_regular_file(
        lock_path, {0o600}
    ):
        return None
    try:
        descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        handle = os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        lock_details = lock_path.lstat()
        descriptor_details = os.fstat(handle.fileno())
        if (
            not _is_owned_regular_file(lock_path, {0o600})
            or (lock_details.st_dev, lock_details.st_ino)
            != (descriptor_details.st_dev, descriptor_details.st_ino)
        ):
            _release_snapshot_lock(handle)
            return None
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        _release_snapshot_lock(handle)
        return None
    return handle


def _is_recoverable_snapshot(snapshot_dir: Path) -> bool:
    try:
        entries = {entry.name for entry in snapshot_dir.iterdir()}
    except OSError:
        return False
    expected = {
        _SNAPSHOT_MARKER_NAME,
        _SNAPSHOT_LOCK_NAME,
        _SNAPSHOT_FILE_NAME,
    }
    if entries != expected:
        return False
    marker = snapshot_dir / _SNAPSHOT_MARKER_NAME
    snapshot = snapshot_dir / _SNAPSHOT_FILE_NAME
    return (
        _is_owned_regular_file(marker, {0o400})
        and marker.read_bytes() == _SNAPSHOT_MARKER
        and _is_owned_regular_file(snapshot, {0o400, 0o600})
    )


def _recover_stale_dataset_snapshots(root: Path) -> None:
    if not _is_owned_private_directory(root):
        return
    try:
        candidates = list(root.iterdir())
    except OSError:
        return
    for candidate in candidates:
        if not _snapshot_name_is_safe(candidate.name):
            continue
        lock_handle = _recovery_lock(candidate)
        if lock_handle is None:
            continue
        try:
            if _is_recoverable_snapshot(candidate):
                shutil.rmtree(candidate)
        finally:
            _release_snapshot_lock(lock_handle)


@contextmanager
def _stable_dataset_snapshot(
    source: Path, _legacy_runs_dir: Path | None = None
) -> Iterator[Path]:
    root = _snapshot_root()
    _recover_stale_dataset_snapshots(root)
    while True:
        snapshot_dir = root / f"{_SNAPSHOT_DIR_PREFIX}{secrets.token_hex(16)}"
        try:
            snapshot_dir.mkdir(mode=0o700)
        except FileExistsError:
            continue
        break
    lock_handle: BinaryIO | None = None
    try:
        _write_private_marker(snapshot_dir)
        lock_handle = _create_snapshot_lock(snapshot_dir)
        snapshot_path = snapshot_dir / _SNAPSHOT_FILE_NAME
        with source.open("rb") as source_handle:
            descriptor = os.open(
                snapshot_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as snapshot_handle:
                os.fchmod(snapshot_handle.fileno(), 0o600)
                shutil.copyfileobj(source_handle, snapshot_handle)
                snapshot_handle.flush()
        snapshot_path.chmod(0o400)
        yield snapshot_path
    finally:
        _release_snapshot_lock(lock_handle)
        shutil.rmtree(snapshot_dir, ignore_errors=True)


def _parquet_metadata_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _time_split(
    frame: pl.DataFrame, snapshot_col: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, int]:
    order_column = "__riskprobe_snapshot_order"
    parsed = normalize_date_series(frame.get_column(snapshot_col))
    with_order = frame.with_columns(parsed.alias(order_column))
    excluded_null_snapshot_rows = with_order.get_column(order_column).null_count()
    ordered = with_order.filter(pl.col(order_column).is_not_null()).sort(
        order_column, maintain_order=True
    )
    groups = ordered.partition_by(order_column, maintain_order=True)
    group_count = len(groups)

    def closest_boundary(target: float, minimum: int, maximum: int) -> int:
        candidates = range(minimum, maximum + 1)
        return min(
            candidates,
            key=lambda index: (
                abs(sum(group.height for group in groups[:index]) - target),
                index,
            ),
        )

    if group_count >= 3:
        train_group_end = closest_boundary(ordered.height * 0.6, 1, group_count - 2)
        test_group_end = closest_boundary(
            ordered.height * 0.8,
            train_group_end + 1,
            group_count - 1,
        )
    elif group_count == 2:
        train_group_end, test_group_end = 1, 2
    else:
        train_group_end = test_group_end = group_count

    def combine(start: int, end: int) -> pl.DataFrame:
        if start == end:
            return ordered.clear().drop(order_column)
        return pl.concat(groups[start:end]).drop(order_column)

    return (
        combine(0, train_group_end),
        combine(train_group_end, test_group_end),
        combine(test_group_end, group_count),
        excluded_null_snapshot_rows,
    )


def _stratified_labels(
    frame: pl.DataFrame, target_col: str, segment_col: str
) -> np.ndarray:
    return np.asarray(
        [
            json.dumps(
                [segment, target],
                default=str,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            for segment, target in frame.select([segment_col, target_col]).rows()
        ],
        dtype=object,
    )


def _target_key(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=True, separators=(",", ":"))


def _constrained_composite_split(
    frame: pl.DataFrame, target_col: str, segment_col: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Allocate every composite group to both partitions when feasible."""
    indices = np.arange(frame.height)
    targets = frame.get_column(target_col).to_list()
    target_keys = [_target_key(value) for value in targets]
    desired_train, _ = train_test_split(
        indices,
        train_size=0.7,
        random_state=42,
        shuffle=True,
        stratify=frame.get_column(target_col).to_numpy(),
    )
    desired_counts: dict[str, int] = {}
    for index in desired_train:
        key = target_keys[int(index)]
        desired_counts[key] = desired_counts.get(key, 0) + 1

    groups: dict[str, list[int]] = {}
    group_targets: dict[str, str] = {}
    for index, (segment, target) in enumerate(
        frame.select([segment_col, target_col]).rows()
    ):
        group_key = json.dumps(
            [segment, target],
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        groups.setdefault(group_key, []).append(index)
        group_targets[group_key] = target_keys[index]

    allocations: dict[str, int] = {}
    for target_key in sorted(desired_counts):
        target_groups = [
            group_key
            for group_key in sorted(groups)
            if group_targets[group_key] == target_key
        ]
        minimum = len(target_groups)
        maximum = sum(len(groups[group_key]) - 1 for group_key in target_groups)
        desired = desired_counts[target_key]
        if desired < minimum or desired > maximum:
            return None
        remaining = desired - minimum
        allocations.update({group_key: 1 for group_key in target_groups})
        while remaining:
            progressed = False
            for group_key in target_groups:
                capacity = len(groups[group_key]) - 1
                if allocations[group_key] >= capacity:
                    continue
                allocations[group_key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
            if not progressed:
                return None

    train_indices: list[int] = []
    for group_key in sorted(groups):
        selected, _ = train_test_split(
            np.asarray(groups[group_key]),
            train_size=allocations[group_key],
            random_state=42,
            shuffle=True,
        )
        train_indices.extend(int(index) for index in selected)
    train_set = set(train_indices)
    test_indices = [index for index in indices if int(index) not in train_set]
    return np.asarray(sorted(train_set)), np.asarray(test_indices)


def _stratified_split_with_limitations(
    frame: pl.DataFrame, target_col: str, segment_col: str
) -> tuple[pl.DataFrame, pl.DataFrame, None, tuple[str, ...]]:
    composite_labels = _stratified_labels(frame, target_col, segment_col)
    composite_counts = np.unique(composite_labels, return_counts=True)[1]
    use_composite = bool(composite_counts.size) and int(composite_counts.min()) >= 2
    limitations: tuple[str, ...] = ()
    split_indices = (
        _constrained_composite_split(frame, target_col, segment_col)
        if use_composite
        else None
    )
    if split_indices is None:
        indices = np.arange(frame.height)
        train_indices, test_indices = train_test_split(
            indices,
            train_size=0.7,
            random_state=42,
            shuffle=True,
            stratify=frame.get_column(target_col).to_numpy(),
        )
        limitations = (
            "Institution × target stratification unavailable; fell back to target-only stratification",
        )
    else:
        train_indices, test_indices = split_indices
    train = frame[sorted(int(index) for index in train_indices)]
    test = frame[sorted(int(index) for index in test_indices)]
    return train, test, None, limitations


def _stratified_split(
    frame: pl.DataFrame, target_col: str, segment_col: str | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, None]:
    """Compatibility wrapper for deterministic stratified splitting."""
    segment = segment_col or target_col
    train, test, holdout, _ = _stratified_split_with_limitations(
        frame, target_col, segment
    )
    return train, test, holdout


def _candidate_frame(rules: list[RiskRule]) -> pl.DataFrame:
    rows = [
        {
            "rule_id": rule.rule_id,
            "origin": rule.origin,
            "conditions_json": json.dumps(
                [condition.model_dump(mode="json") for condition in rule.conditions],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        for rule in sorted(rules, key=lambda item: item.rule_id)
    ]
    if rows:
        return pl.DataFrame(rows)
    return pl.DataFrame(
        schema={
            "rule_id": pl.String,
            "origin": pl.String,
            "conditions_json": pl.String,
        }
    )


def _sorted_card(
    card: EvidenceCard, *, already_redacted: bool = False
) -> EvidenceCard:
    slices = tuple(
        sorted(
            (
                item
                if already_redacted or item.slice_type != "segment"
                else item.model_copy(
                    update={"slice_value": redact_segment_value(item.slice_value)}
                )
                for item in card.slices
            ),
            key=lambda item: (
                _SLICE_ORDER[item.slice_type],
                item.slice_value,
            ),
        )
    )
    return card.model_copy(
        update={
            "slices": slices,
            "limitations": tuple(
                sorted(
                    {
                        redact_limitation(
                            item,
                            already_redacted=already_redacted,
                        )
                        for item in card.limitations
                    }
                )
            ),
        }
    )


def _evidence_payload(
    cards: list[EvidenceCard], *, time_validation_enabled: bool
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for card in sorted((_sorted_card(card) for card in cards), key=evidence_sort_key):
        item = card.model_dump(mode="json")
        if not time_validation_enabled:
            item.pop("max_time_decay", None)
            item["slices"] = [
                slice_item
                for slice_item in item["slices"]
                if slice_item["slice_type"] != "time"
            ]
        payload.append(item)
    return payload


def _restore_rules(source: Path | bytes) -> list[RiskRule]:
    try:
        frame = pl.read_parquet(BytesIO(source) if isinstance(source, bytes) else source)
        if frame.columns != ["rule_id", "origin", "conditions_json"]:
            raise ValueError("checkpoint rule columns are invalid")
        restored: list[RiskRule] = []
        for item in frame.iter_rows(named=True):
            conditions = json.loads(item["conditions_json"])
            if not isinstance(conditions, list):
                raise ValueError("checkpoint rule conditions are invalid")
            restored.append(
                RiskRule.model_validate(
                    {
                        "rule_id": item["rule_id"],
                        "origin": item["origin"],
                        "conditions": tuple(conditions),
                    }
                )
            )
        canonical = _candidate_frame(restored)
        if frame.schema != canonical.schema or frame.to_dicts() != canonical.to_dicts():
            raise ValueError("checkpoint rules are not canonical")
        return restored
    except Exception as error:
        raise ValueError("checkpoint rules are invalid") from error


def _restore_cards(path: Path) -> list[EvidenceCard]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("checkpoint evidence must be a list")
        restored: list[EvidenceCard] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("checkpoint evidence item is invalid")
            rule = dict(item["rule"])
            rule["conditions"] = tuple(rule["conditions"])
            restored.append(
                EvidenceCard.model_validate(
                    {
                        **item,
                        "rule": rule,
                        "slices": tuple(item["slices"]),
                        "lift_ci": tuple(item["lift_ci"]),
                        "limitations": tuple(item["limitations"]),
                        "max_time_decay": item.get("max_time_decay", 0.0),
                    }
                )
            )
        return restored
    except Exception as error:
        raise ValueError("checkpoint evidence is invalid") from error


def _issue_payload(issue: Any) -> dict[str, Any]:
    payload = asdict(issue)
    if issue.code == "SINGLE_CLASS_SLICE":
        payload["message"] = "single-class slice detected"
    return payload


def _profile_payload(
    profile: DatasetProfile, *, excluded_null_snapshot_rows: int
) -> dict[str, Any]:
    segment_sizes = list(profile.segment_counts.values())
    return {
        "dataset_id": profile.dataset_id,
        "row_count": profile.row_count,
        "feature_count": profile.feature_count,
        "positive_rate": profile.positive_rate,
        "segment_count": len(segment_sizes),
        "segment_size_min": min(segment_sizes, default=0),
        "segment_size_max": max(segment_sizes, default=0),
        "snapshot_min": profile.snapshot_min,
        "snapshot_max": profile.snapshot_max,
        "excluded_null_snapshot_rows": excluded_null_snapshot_rows,
        "metadata_grade": profile.metadata_grade,
        "issues": [
            _issue_payload(issue)
            for issue in sorted(
                profile.issues,
                key=lambda item: (item.severity, item.code, item.family, item.message),
            )
        ],
    }


def _restore_safe_profile(content: bytes, *, expected_dataset_id: str) -> SafeProfile:
    try:
        payload = json.loads(content)
        expected_fields = {
            "dataset_id",
            "row_count",
            "feature_count",
            "positive_rate",
            "segment_count",
            "segment_size_min",
            "segment_size_max",
            "snapshot_min",
            "snapshot_max",
            "excluded_null_snapshot_rows",
            "metadata_grade",
            "issues",
        }
        if type(payload) is not dict or set(payload) != expected_fields:
            raise ValueError("run profile fields are invalid")
        if payload["dataset_id"] != expected_dataset_id:
            raise ValueError("run profile dataset is invalid")
        integer_fields = (
            "row_count",
            "feature_count",
            "segment_count",
            "segment_size_min",
            "segment_size_max",
            "excluded_null_snapshot_rows",
        )
        if any(
            type(payload[name]) is not int or payload[name] < 0
            for name in integer_fields
        ):
            raise ValueError("run profile counts are invalid")
        if payload["excluded_null_snapshot_rows"] > payload["row_count"]:
            raise ValueError("run profile excluded row count is invalid")
        segment_count = payload["segment_count"]
        segment_min = payload["segment_size_min"]
        segment_max = payload["segment_size_max"]
        if (segment_count == 0 and (segment_min != 0 or segment_max != 0)) or (
            segment_count > 0 and segment_min > segment_max
        ):
            raise ValueError("run profile segment summary is invalid")
        positive_rate = payload["positive_rate"]
        if positive_rate is not None and (
            isinstance(positive_rate, bool)
            or not isinstance(positive_rate, (int, float))
            or not np.isfinite(positive_rate)
            or not 0 <= positive_rate <= 1
        ):
            raise ValueError("run profile positive rate is invalid")
        if payload["metadata_grade"] not in {"A", "B"}:
            raise ValueError("run profile metadata grade is invalid")
        if any(
            value is not None and not isinstance(value, str)
            for value in (payload["snapshot_min"], payload["snapshot_max"])
        ):
            raise ValueError("run profile snapshots are invalid")
        issues = payload["issues"]
        if type(issues) is not list:
            raise ValueError("run profile issues are invalid")
        issue_codes: list[str] = []
        issue_keys: list[tuple[str, str, str, str]] = []
        issue_fields = {
            "code",
            "severity",
            "family",
            "features",
            "affected_rows",
            "message",
        }
        for issue in issues:
            if type(issue) is not dict or set(issue) != issue_fields:
                raise ValueError("run profile issue fields are invalid")
            if (
                not isinstance(issue["code"], str)
                or not issue["code"]
                or len(issue["code"]) > 128
                or issue["severity"] not in {"warning", "error"}
                or not isinstance(issue["family"], str)
                or not isinstance(issue["message"], str)
                or type(issue["features"]) is not list
                or any(not isinstance(feature, str) for feature in issue["features"])
                or type(issue["affected_rows"]) is not int
                or issue["affected_rows"] < 0
            ):
                raise ValueError("run profile issue is invalid")
            issue_codes.append(issue["code"])
            issue_keys.append(
                (
                    issue["severity"],
                    issue["code"],
                    issue["family"],
                    issue["message"],
                )
            )
        if issue_keys != sorted(issue_keys):
            raise ValueError("run profile issues are not canonical")
        return SafeProfile(
            dataset_id=expected_dataset_id,
            row_count=payload["row_count"],
            feature_count=payload["feature_count"],
            positive_rate=(
                None if positive_rate is None else float(positive_rate)
            ),
            segment_count=segment_count,
            min_segment_size=None if segment_count == 0 else segment_min,
            max_segment_size=None if segment_count == 0 else segment_max,
            snapshot_min=payload["snapshot_min"],
            snapshot_max=payload["snapshot_max"],
            metadata_grade=payload["metadata_grade"],
            issue_codes=tuple(sorted(set(issue_codes))),
            issue_count=len(issues),
        )
    except Exception as error:
        raise ValueError("run profile is invalid") from error


def _render_service_report(
    profile: DatasetProfile,
    cards: list[EvidenceCard],
    validation_limitations: tuple[str, ...],
    institution_analysis: dict[str, Any] | None = None,
    *,
    cards_are_redacted: bool = False,
    expose_segment_values: bool = False,
) -> str:
    report = render_risk_report(
        profile,
        cards,
        institution_analysis=institution_analysis,
        expose_segment_values=expose_segment_values,
        segments_are_redacted=cards_are_redacted,
    )
    if validation_limitations and not cards:
        replacement = "\n".join(
            f"- {redact_limitation(limitation, already_redacted=cards_are_redacted)}"
            for limitation in sorted(validation_limitations)
        )
        report = report.replace("- None identified by configured checks", replacement)
    return report


def _with_limitation(
    cards: list[EvidenceCard], limitation: str, *, downgrade: bool
) -> list[EvidenceCard]:
    return [
        card.model_copy(
            update={
                "grade": "Suspicious" if downgrade else card.grade,
                "limitations": tuple(sorted({*card.limitations, limitation})),
            }
        )
        for card in cards
    ]


def _attach_holdout(
    primary: list[EvidenceCard], holdout: list[EvidenceCard]
) -> list[EvidenceCard]:
    holdout_by_id = {card.rule.rule_id: card for card in holdout}
    combined: list[EvidenceCard] = []
    for card in primary:
        holdout_card = holdout_by_id.get(card.rule.rule_id)
        if holdout_card is None:
            combined.extend(
                _with_limitation(
                    [card],
                    "Holdout evidence is missing for this rule",
                    downgrade=True,
                )
            )
            continue
        holdout_slice = SliceMetrics(
            slice_type="dataset",
            slice_value="Holdout",
            metrics=holdout_card.test,
        )
        holdout_limitations = tuple(
            f"holdout: {limitation}" for limitation in holdout_card.limitations
        )
        grade = max(
            (card.grade, holdout_card.grade),
            key=lambda item: _GRADE_ORDER[item],
        )
        combined.append(
            card.model_copy(
                update={
                    "slices": card.slices + (holdout_slice,),
                    "limitations": card.limitations + holdout_limitations,
                    "grade": grade,
                    "max_time_decay": max(
                        card.max_time_decay, holdout_card.max_time_decay
                    ),
                    "segment_consistency": min(
                        card.segment_consistency,
                        holdout_card.segment_consistency,
                    ),
                }
            )
        )
    return combined


def _artifact_integrity(run_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in _ARTIFACT_NAMES[1:]:
        content = (run_dir / name).read_bytes()
        records[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    return records


class RiskProbeService:
    def __init__(
        self,
        *,
        config: ProjectConfig | Path | None,
        runs_dir: Path,
        state_dir: Path | None = None,
        decision_provider_config: DecisionProviderConfig | None = None,
        decision_provider: DecisionProvider | None = None,
    ) -> None:
        from riskprobe.agents.decision_providers import bind_decision_providers

        self._config = (
            ProjectConfig.from_yaml(config) if isinstance(config, Path) else config
        )
        self.store = RunStore(runs_dir)
        self.state_dir = Path(state_dir) if state_dir is not None else self.store.runs_dir
        (
            self._decision_provider,
            self._decision_fallback,
        ) = bind_decision_providers(
            decision_provider_config,
            external_provider=decision_provider,
        )
        self._split_limitations: tuple[str, ...] = ()

    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            raise RuntimeError("project configuration is unavailable")
        return self._config

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a 16-character lowercase hexadecimal value")
        return run_id

    def _state_directory(self, run_id: str) -> Path:
        public_run_id = self._validate_run_id(run_id)
        state_dir = self.state_dir.expanduser().resolve(strict=False)
        run_dir = (self.store.runs_dir / public_run_id).resolve(strict=False)
        if state_dir == run_dir or run_dir in state_dir.parents:
            raise ValueError("state directory must be outside the run artifact directory")
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def _sidecar_path(self, run_id: str, suffix: str) -> Path:
        public_run_id = self._validate_run_id(run_id)
        return self._state_directory(public_run_id) / f".{public_run_id}.{suffix}"

    def _evidence_store(self, run_id: str) -> EvidenceStore:
        return EvidenceStore(self._sidecar_path(run_id, "evidence.sqlite3"))

    def _dataset(self) -> ParquetDataset:
        return ParquetDataset(self.config.dataset.path)

    @contextmanager
    def _snapshot_dataset(self) -> Iterator[ParquetDataset]:
        with _stable_dataset_snapshot(self.config.dataset.path) as snapshot_path:
            yield ParquetDataset(snapshot_path)

    def _feature_names(self, dataset: ParquetDataset) -> list[str]:
        roles = (
            self.config.columns.entity,
            self.config.columns.snapshot,
            self.config.columns.segment,
            self.config.columns.target,
        )
        return self.config.features.select_columns(dataset.schema().names(), roles)

    def _partitions(
        self, dataset: ParquetDataset, feature_names: list[str]
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, int]:
        columns = [
            self.config.columns.snapshot,
            self.config.columns.segment,
            self.config.columns.target,
            *feature_names,
        ]
        frame = dataset.collect(columns)
        if self.config.time_validation_enabled:
            self._split_limitations = ()
            return _time_split(frame, self.config.columns.snapshot)
        train, test, holdout, limitations = _stratified_split_with_limitations(
            frame,
            self.config.columns.target,
            self.config.columns.segment,
        )
        self._split_limitations = limitations
        return train, test, holdout, 0

    def _discover_from_train(
        self, train: pl.DataFrame, feature_names: list[str]
    ) -> list[RiskRule]:
        discovery_columns = [*feature_names, self.config.columns.target]
        sample = train.select(discovery_columns)
        if sample.height > _DISCOVERY_SAMPLE_LIMIT:
            sample = sample.sample(
                n=_DISCOVERY_SAMPLE_LIMIT,
                shuffle=True,
                seed=self.config.discovery.random_seed,
            )
        return discover_rules(
            sample,
            feature_names,
            self.config.columns.target,
            self.config.discovery,
        )

    def _discovery_result_from_train(
        self, train: pl.DataFrame, feature_names: list[str]
    ) -> DiscoveryResult:
        discovery_columns = [*feature_names, self.config.columns.target]
        sample = train.select(discovery_columns)
        if sample.height > _DISCOVERY_SAMPLE_LIMIT:
            sample = sample.sample(
                n=_DISCOVERY_SAMPLE_LIMIT,
                shuffle=True,
                seed=self.config.discovery.random_seed,
            )
        return discover_with_metrics(
            sample,
            feature_names,
            self.config.columns.target,
            self.config.discovery,
        )

    def _validate(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        holdout: pl.DataFrame | None,
        rules: list[RiskRule],
    ) -> tuple[list[EvidenceCard], tuple[str, ...]]:
        involved_features = sorted(
            {
                condition.feature
                for rule in rules
                for condition in rule.conditions
            }
        )
        validation_columns = [
            *involved_features,
            self.config.columns.target,
            self.config.columns.segment,
        ]
        if self.config.time_validation_enabled:
            validation_columns.append(self.config.columns.snapshot)
        train_projection = train.select(validation_columns)
        test_projection = test.select(validation_columns)
        kwargs = {
            "target_col": self.config.columns.target,
            "segment_col": self.config.columns.segment,
            "snapshot_col": self.config.columns.snapshot,
            "segment_display_name": self.config.segment_display_name,
            "time_validation_enabled": self.config.time_validation_enabled,
            "config": self.config.validation,
            "metadata_grade": self.config.metadata_grade,
        }
        validation_limitations: list[str] = []
        holdout_limitation: str | None = None
        if self.config.time_validation_enabled:
            if holdout is None or holdout.is_empty():
                holdout_limitation = (
                    "Holdout partition is empty; validation unavailable"
                )
            elif (
                holdout.get_column(self.config.columns.target).n_unique() < 2
            ):
                holdout_limitation = (
                    "Holdout partition has a single target class; "
                    "validation unavailable"
                )
            if holdout_limitation is not None:
                validation_limitations.append(holdout_limitation)

        if train_projection.is_empty() or not (
            train_projection.get_column(self.config.columns.target) == 1
        ).any():
            validation_limitations.append(
                "Train partition has no positive target; validation unavailable"
            )
            return [], tuple(validation_limitations)
        if test_projection.is_empty() or not (
            test_projection.get_column(self.config.columns.target) == 1
        ).any():
            validation_limitations.append(
                "Test partition has no positive target; validation unavailable"
            )
            return [], tuple(validation_limitations)

        cards = validate_rules(train_projection, test_projection, rules, **kwargs)
        if not self.config.time_validation_enabled:
            return cards, tuple(validation_limitations)
        if holdout_limitation is not None:
            return (
                _with_limitation(cards, holdout_limitation, downgrade=True),
                tuple(validation_limitations),
            )
        if not rules or holdout is None:
            return cards, tuple(validation_limitations)

        holdout_projection = holdout.select(validation_columns)
        try:
            holdout_cards = validate_rules(
                train_projection,
                holdout_projection,
                rules,
                **kwargs,
            )
        except Exception:
            limitation = "Holdout validation could not be computed"
            validation_limitations.append(limitation)
            return (
                _with_limitation(cards, limitation, downgrade=True),
                tuple(validation_limitations),
            )
        return _attach_holdout(cards, holdout_cards), tuple(validation_limitations)

    def _verified_run_context(self, context: RunContext) -> RunContext:
        if type(context) is not RunContext:
            raise TypeError("context must be a RunContext")
        if context.run_dir != self.store.runs_dir / context.run_id:
            raise RuntimeError("run artifacts are unavailable")
        context.require_binding(
            config_fingerprint=self.store.config_fingerprint(self.config),
            dataset_id=_safe_dataset_id(self.config.dataset.id),
            time_validation_enabled=self.config.time_validation_enabled,
        )
        return context

    def _profile_from_run(self, context: RunContext) -> SafeProfile:
        verified = self._verified_run_context(context)
        return _restore_safe_profile(
            verified.read_verified_artifact("data_profile.json"),
            expected_dataset_id=_safe_dataset_id(self.config.dataset.id),
        )

    def _rules_from_run(self, context: RunContext) -> list[RiskRule]:
        verified = self._verified_run_context(context)
        return _restore_rules(
            verified.read_verified_artifact("candidate_rules.parquet")
        )

    def _data_fingerprint_from_run(self, context: RunContext) -> str:
        verified = self._verified_run_context(context)
        try:
            manifest = json.loads(
                (verified.run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            fingerprint = manifest["data_fingerprint"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("run artifacts are unavailable") from error
        self._verified_run_context(verified)
        if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
            raise RuntimeError("run artifacts are unavailable")
        return fingerprint

    def inspect(self) -> DatasetProfile:
        with self._snapshot_dataset() as dataset:
            return profile_dataset(dataset, self.config)

    def _assert_rule_conclusion_allowed(self, profile: DatasetProfile) -> None:
        if profile.metadata_grade not in {"A", "B"}:
            raise ValueError("metadata grade below B blocks rule conclusions")

    def discover_with_metrics(self) -> DiscoveryResult:
        profile = self.inspect()
        self._assert_rule_conclusion_allowed(profile)
        dataset = self._dataset()
        feature_names = self._feature_names(dataset)
        train, _, _, _ = self._partitions(dataset, feature_names)
        return self._discovery_result_from_train(train, feature_names)

    def discover(self) -> list[RiskRule]:
        with self._snapshot_dataset() as dataset:
            profile = profile_dataset(dataset, self.config)
            self._assert_rule_conclusion_allowed(profile)
            feature_names = self._feature_names(dataset)
            train, _, _, _ = self._partitions(dataset, feature_names)
            return self._discover_from_train(train, feature_names)

    def diagnose(self, *, run_id: str) -> DiagnoseResponse:
        """Persist aggregate diagnostics as content-addressed sidecar evidence."""

        return self._diagnose_with_store(run_id, self._evidence_store(run_id))

    def _diagnose_with_store(
        self,
        run_id: str,
        evidence_store: EvidenceStore,
        *,
        dataset_id: str | None = None,
        run_context: RunContext | None = None,
    ) -> DiagnoseResponse:
        from riskprobe.tools.models import DiagnoseResponse

        public_run_id = self._validate_run_id(run_id)
        public_dataset_id = dataset_id or self.config.dataset.id
        if not isinstance(evidence_store, EvidenceStore):
            raise TypeError("evidence_store must be an EvidenceStore")
        if run_context is not None and type(run_context) is not RunContext:
            raise TypeError("run_context must be a RunContext")
        expected_fingerprint = (
            None
            if run_context is None
            else self._data_fingerprint_from_run(run_context)
        )
        with _stable_dataset_snapshot(self.config.dataset.path) as snapshot_path:
            current_fingerprint = _parquet_metadata_fingerprint(snapshot_path)
            if (
                expected_fingerprint is not None
                and current_fingerprint != expected_fingerprint
            ):
                raise RuntimeError("run input is unavailable")
            report = diagnose_dataset(ParquetDataset(snapshot_path), self.config)
        artifact_hashes = (
            {}
            if expected_fingerprint is None
            else {"input.parquet": expected_fingerprint}
        )
        evidence_ids = tuple(
            evidence_store.append(
                EvidenceRecord(
                    run_id=public_run_id,
                    kind=_DIAGNOSTIC_KIND,
                    payload={
                        **finding.model_dump(mode="json"),
                        "dataset_id": public_dataset_id,
                    },
                    artifact_hashes=artifact_hashes,
                    producer_version=_SERVICE_PRODUCER_VERSION,
                )
            )
            for finding in report.findings
        )
        return DiagnoseResponse(
            dataset_id=public_dataset_id,
            finding_ids=evidence_ids,
        )

    def recommend(
        self,
        *,
        run_id: str,
        evidence_ids: Sequence[str] = (),
        all_current_diagnostics: bool = False,
    ) -> RecommendResponse:
        """Build human-gated recommendations from one explicit diagnostic selection."""

        if type(all_current_diagnostics) is not bool:
            raise TypeError("all_current_diagnostics must be a boolean")
        public_run_id = self._validate_run_id(run_id)
        normalized_ids = self._validated_evidence_ids(evidence_ids)
        if all_current_diagnostics == bool(normalized_ids):
            raise ValueError(
                "provide either evidence_ids or all_current_diagnostics"
            )
        evidence_store = self._evidence_store(public_run_id)
        public_dataset_id = self.config.dataset.id
        if all_current_diagnostics:
            diagnosis = self._diagnose_with_store(
                public_run_id,
                evidence_store,
                dataset_id=public_dataset_id,
            )
            normalized_ids = diagnosis.finding_ids
            if not normalized_ids:
                raise ValueError("current diagnostics are unavailable")
        return self._recommend_with_store(
            public_run_id,
            normalized_ids,
            evidence_store,
            dataset_id=public_dataset_id,
        )

    def _recommend_with_store(
        self,
        run_id: str,
        evidence_ids: Sequence[str],
        evidence_store: EvidenceStore,
        *,
        dataset_id: str | None = None,
        safe_profile: SafeProfile | None = None,
        decision_controller: DecisionController | None = None,
        decision_result_evidence_id: str | None = None,
    ) -> RecommendResponse:
        from riskprobe.tools.models import RecommendResponse

        public_run_id = self._validate_run_id(run_id)
        public_dataset_id = dataset_id or self.config.dataset.id
        normalized_ids = self._validated_evidence_ids(evidence_ids)
        if not isinstance(evidence_store, EvidenceStore):
            raise TypeError("evidence_store must be an EvidenceStore")
        if safe_profile is not None and (
            type(safe_profile) is not SafeProfile
            or safe_profile.dataset_id != public_dataset_id
        ):
            raise ValueError("safe_profile must match the requested dataset")
        if (decision_controller is None) != (decision_result_evidence_id is None):
            raise ValueError("controlled recommendation binding is incomplete")

        accepted_actions = None
        if decision_controller is not None:
            from riskprobe.agents.decision_contracts import DecisionStatus
            from riskprobe.agents.decision_controller import DecisionController

            if type(decision_controller) is not DecisionController:
                raise TypeError("decision_controller must be a DecisionController")
            submission = decision_controller.replay(
                result_evidence_id=decision_result_evidence_id,
                expected_run_id=public_run_id,
            )
            if (
                submission.result.status is not DecisionStatus.ACCEPTED
                or submission.result.diagnosis_evidence_ids
                != tuple(sorted(normalized_ids))
            ):
                raise RuntimeError("decision is unavailable")
            accepted_actions = frozenset(submission.result.action_codes)

        finding_evidence: dict[str, str] = {}
        findings: list[RiskFinding] = []
        for evidence_id in normalized_ids:
            record = self._current_evidence(
                public_run_id,
                evidence_id,
                evidence_store,
            )
            if (
                record.kind != _DIAGNOSTIC_KIND
                or record.parent_ids
                or record.payload.get("dataset_id") != public_dataset_id
            ):
                raise RuntimeError("evidence is unavailable")
            payload = dict(record.payload)
            payload.pop("dataset_id", None)
            finding = RiskFinding.model_validate_json(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            if finding.finding_id in finding_evidence:
                raise RuntimeError("evidence is unavailable")
            finding_evidence[finding.finding_id] = evidence_id
            findings.append(finding)

        profile = safe_profile or SafeProfile.from_profile(self.inspect())
        report = DiagnosticReport(profile=profile, findings=tuple(findings))
        recommendations = build_recommendations(
            report,
            metadata_grade=report.metadata_grade,
        )
        if accepted_actions is not None:
            from riskprobe.recommendations.policy import ActionCode

            try:
                available_actions = {
                    ActionCode(recommendation.action_code)
                    for recommendation in recommendations
                }
            except (TypeError, ValueError) as error:
                raise RuntimeError("recommendations are unavailable") from error
            if not accepted_actions or not accepted_actions.issubset(
                available_actions
            ):
                raise RuntimeError("decision is unavailable")
            recommendations = tuple(
                recommendation
                for recommendation in recommendations
                if ActionCode(recommendation.action_code) in accepted_actions
            )
            final_actions = tuple(
                sorted(
                    {
                        ActionCode(recommendation.action_code)
                        for recommendation in recommendations
                    },
                    key=lambda action: action.value,
                )
            )
            if final_actions != tuple(
                sorted(accepted_actions, key=lambda action: action.value)
            ) or len(final_actions) != len(recommendations):
                raise RuntimeError("decision is unavailable")
        recommendation_ids: list[str] = []
        for recommendation in recommendations:
            try:
                parent_ids = tuple(
                    sorted(
                        finding_evidence[finding_id]
                        for finding_id in recommendation.finding_ids
                    )
                )
            except KeyError as error:
                raise RuntimeError("evidence is unavailable") from error
            recommendation_ids.append(
                evidence_store.append(
                    EvidenceRecord(
                        run_id=public_run_id,
                        kind=_RECOMMENDATION_KIND,
                        payload={
                            **recommendation.model_dump(mode="json"),
                            "dataset_id": public_dataset_id,
                        },
                        parent_ids=parent_ids,
                        producer_version=_SERVICE_PRODUCER_VERSION,
                    )
                )
            )
        return RecommendResponse(
            dataset_id=public_dataset_id,
            recommendation_ids=tuple(recommendation_ids),
        )

    def evidence(
        self,
        *,
        run_id: str,
        evidence_id: str,
    ) -> EvidenceLookupResponse:
        """Return one aggregate evidence record only when it belongs to the run."""

        return self._evidence_with_store(
            run_id,
            evidence_id,
            self._evidence_store(run_id),
        )

    def _evidence_with_store(
        self,
        run_id: str,
        evidence_id: str,
        evidence_store: EvidenceStore,
    ) -> EvidenceLookupResponse:
        from riskprobe.tools.models import EvidenceLookupResponse

        public_run_id = self._validate_run_id(run_id)
        if not isinstance(evidence_store, EvidenceStore):
            raise TypeError("evidence_store must be an EvidenceStore")
        record = self._current_evidence(
            public_run_id,
            evidence_id,
            evidence_store,
        )
        return EvidenceLookupResponse(
            evidence_id=evidence_id,
            run_id=record.run_id,
            kind=record.kind,
            payload=record.payload,
            parent_ids=record.parent_ids,
            artifact_hashes=record.artifact_hashes,
            privacy_class=record.privacy_class.value,
            producer_version=record.producer_version,
        )

    def status(self, *, run_id: str) -> StatusResponse:
        """Return the bounded public runtime status for an existing run."""

        from riskprobe.tools.models import StatusResponse

        runtime = self._runtime(run_id)
        return StatusResponse(
            run_id=runtime.run_id,
            status=self._tool_status(runtime.run_status().value),
        )

    def trace(
        self,
        *,
        run_id: str,
        node_id: str | None = None,
    ) -> TraceResponse:
        """Project runtime events without timestamps, fingerprints, or outputs."""

        from riskprobe.tools.models import TraceEvent, TraceResponse

        runtime = self._runtime(run_id)
        events: list[TraceEvent] = []
        for event in runtime.trace(node_id):
            sequence = event.get("sequence")
            attempt = event.get("attempt")
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise RuntimeError("runtime trace is unavailable")
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                attempt = 1
            events.append(
                TraceEvent(
                    sequence=sequence,
                    node_id=event.get("node_id") or "run",
                    event_type=event["event_type"],
                    status=self._tool_status(event["status"]),
                    attempt=attempt,
                    error_class=event.get("error_class"),
                )
            )
        return TraceResponse(run_id=runtime.run_id, events=tuple(events))

    @staticmethod
    def _validated_evidence_ids(evidence_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(evidence_ids, (str, bytes, bytearray)):
            raise TypeError("evidence_ids must be a sequence of SHA-256 identifiers")
        normalized = tuple(evidence_ids)
        if len(normalized) != len(set(normalized)) or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in normalized
        ):
            raise ValueError("evidence_ids must contain unique SHA-256 identifiers")
        return normalized

    @staticmethod
    def _current_evidence(
        run_id: str,
        evidence_id: str,
        evidence_store: EvidenceStore,
    ) -> EvidenceRecord:
        if not isinstance(evidence_id, str) or _SHA256.fullmatch(evidence_id) is None:
            raise RuntimeError("evidence is unavailable")
        record = evidence_store.get(evidence_id)
        if (
            record is None
            or record.run_id != run_id
            or record.privacy_class is not PrivacyClass.AGGREGATE
            or EvidenceStore.content_id(record) != evidence_id
        ):
            raise RuntimeError("evidence is unavailable")
        return record

    def _runtime(self, run_id: str) -> RunRuntime:
        public_run_id = self._validate_run_id(run_id)
        database_path = self.store.runs_dir / f".{public_run_id}.runtime.sqlite3"
        if not _is_owned_regular_file(database_path, {0o600}):
            raise RuntimeError("runtime is unavailable")
        return RunRuntime(self.store.runs_dir, public_run_id)

    @staticmethod
    def _tool_status(status: object) -> str:
        if not isinstance(status, str) or status not in _STATUS_MAP:
            raise RuntimeError("runtime status is unavailable")
        return _STATUS_MAP[status]

    def orchestrate(
        self,
        *,
        dataset_id: str,
        principal: Principal,
        budget: Budget,
        objective: str = "comprehensive",
    ) -> AgentResult:
        """Run the immutable pipeline, then execute the closed local agent graph."""

        from riskprobe.agents import AgentOrchestrator, Planner, Reviewer, SessionStore
        from riskprobe.agents.decision_controller import DecisionController
        from riskprobe.agents.results import AgentResultStore
        from riskprobe.policy import (
            Budget,
            Capability,
            PolicyDeniedError,
            PolicyEngine,
            Principal,
        )
        from riskprobe.registry import DatasetRegistry
        from riskprobe.tools import (
            DiagnoseRequest,
            DiscoverRequest,
            HandlerToolGateway,
            InspectRequest,
            LocalRiskProbeToolHandler,
            RecommendRequest,
        )

        if type(principal) is not Principal:
            raise TypeError("principal must be a Principal")
        if type(budget) is not Budget:
            raise TypeError("budget must be a Budget")
        if dataset_id != self.config.dataset.id:
            raise ValueError("dataset ID is not registered")
        context = self.run()
        run_id = self._validate_run_id(context.run_id)
        evidence_store = self._evidence_store(run_id)
        session_path = self._sidecar_path(run_id, "sessions.sqlite3")
        try:
            session_path.lstat()
        except FileNotFoundError:
            session_sidecar_existed = False
        except OSError as error:
            raise RuntimeError("agent state is unavailable") from error
        else:
            session_sidecar_existed = True
            if not _is_owned_regular_file(session_path, {0o600}):
                raise RuntimeError("agent state is unavailable")
        sessions = SessionStore(session_path)
        decision_controller = DecisionController(evidence_store)
        handler = LocalRiskProbeToolHandler(
            run_id=run_id,
            runs_dir=self.store.runs_dir,
            evidence_store=evidence_store,
            run_context=context if type(context) is RunContext else None,
            decision_controller=decision_controller,
        )
        policy = PolicyEngine()
        gateway = HandlerToolGateway(
            registry=DatasetRegistry.from_mapping({dataset_id: self.config}),
            policy=policy,
            handler=handler,
        )
        planner = Planner(
            allowed_tools={
                "inspect": InspectRequest,
                "diagnose": DiagnoseRequest,
                "discover": DiscoverRequest,
                "recommend": RecommendRequest,
            }
        )
        orchestrator = AgentOrchestrator(
            planner=planner,
            reviewer=Reviewer(),
            gateway=gateway,
            sessions=sessions,
            evidence_resolver=evidence_store,
            decision_controller=decision_controller,
            decision_provider=self._decision_provider,
            decision_fallback=self._decision_fallback,
        )
        if type(context) is not RunContext:
            return orchestrator.run(
                objective=objective,
                dataset_id=dataset_id,
                principal=principal,
                budget=budget,
                session_id=run_id,
            )

        metadata_grade = self._profile_from_run(context).metadata_grade
        result_store = AgentResultStore(
            self._sidecar_path(run_id, "agent-result.json")
        )
        with result_store.locked():
            cached = result_store.load()
            if cached is not None:
                required_capabilities = {
                    Capability.INSPECT,
                    Capability.DIAGNOSE,
                    Capability.DISCOVER,
                    Capability.RECOMMEND,
                }
                if not required_capabilities.issubset(
                    policy.capabilities_for(principal.role)
                ):
                    raise PolicyDeniedError("capability is not authorized")
                return orchestrator.validate_terminal_result(
                    cached,
                    objective=objective,
                    dataset_id=dataset_id,
                    session_id=run_id,
                    metadata_grade=metadata_grade,
                )
            if session_sidecar_existed:
                raise RuntimeError("agent result is unavailable")
            result = orchestrator.run(
                objective=objective,
                dataset_id=dataset_id,
                principal=principal,
                budget=budget,
                session_id=run_id,
            )
            validated = orchestrator.validate_terminal_result(
                result,
                objective=objective,
                dataset_id=dataset_id,
                session_id=run_id,
                metadata_grade=metadata_grade,
            )
            result_store.publish(validated)
            return validated

    def orchestrate_with_citations(
        self,
        *,
        dataset_id: str,
        principal: Principal,
        budget: Budget,
        objective: str = "comprehensive",
        roots: Mapping[str, Path],
        scope_id: str,
        query_id: str,
        query_text: str,
        limit: int = 5,
    ) -> AgentCitationResult:
        """Run the agent first, then attach metadata-only local RAG citations."""

        agent_result = self.orchestrate(
            dataset_id=dataset_id,
            principal=principal,
            budget=budget,
            objective=objective,
        )
        citations = self.query_local_rag(
            run_id=agent_result.session_id,
            roots=roots,
            scope_id=scope_id,
            query_id=query_id,
            query_text=query_text,
            limit=limit,
        )
        return AgentCitationResult(agent_result=agent_result, citations=citations)

    def build_local_rag(
        self,
        *,
        run_id: str,
        roots: Mapping[str, Path],
        root_id: str,
        scope_id: str,
        provider_summaries: Sequence[Mapping[str, object]] = (),
    ) -> BuildResult:
        """Build a provider-safe local citation index in mutable sidecar state."""

        from riskprobe.rag import LocalCitationIndex, ProviderSafeSummary

        if type(provider_summaries) not in {list, tuple}:
            raise TypeError("provider_summaries must be a list or tuple")
        normalized: list[dict[str, object]] = []
        for summary in provider_summaries:
            if type(summary) is not dict:
                raise TypeError("provider summaries must be exact boundary mappings")
            ProviderSafeSummary.safe_validate(summary)
            normalized.append(dict(summary))
        public_run_id = self._validate_run_id(run_id)
        index = LocalCitationIndex(
            index_path=(
                self._state_directory(public_run_id)
                / f"riskprobe_{public_run_id}_rag_index.json"
            ),
            roots=roots,
        )
        return index.build(
            root_id=root_id,
            scope_id=scope_id,
            provider_summaries=tuple(normalized),
        )

    def query_local_rag(
        self,
        *,
        run_id: str,
        roots: Mapping[str, Path] | None = None,
        scope_id: str,
        query_id: str,
        query_text: str,
        limit: int = 5,
    ) -> QueryResult:
        """Query the sealed local index and return citation metadata only."""

        from riskprobe.rag import LocalCitationIndex

        public_run_id = self._validate_run_id(run_id)
        index = LocalCitationIndex(
            index_path=(
                self._state_directory(public_run_id)
                / f"riskprobe_{public_run_id}_rag_index.json"
            ),
            roots={} if roots is None else roots,
        )
        return index.query(
            scope_id=scope_id,
            query_id=query_id,
            query_text=query_text,
            limit=limit,
        )

    @staticmethod
    def evaluate_v1(
        suite: EvalSuite,
        runner: EvalRunner | RunnerCallable | object,
        *,
        candidate_version: str,
    ) -> EvalReport:
        """Evaluate an exact frozen v1 suite without dynamic code loading."""

        from riskprobe.evals import EvalHarness, EvalSuite

        if type(suite) is not EvalSuite or not suite.verify_integrity():
            raise TypeError("suite must be an integrity-checked EvalSuite v1")
        return EvalHarness(seed=suite.seed).evaluate(
            suite,
            runner,
            candidate_version=candidate_version,
        )

    @staticmethod
    def evaluate_v2(
        suite: EvalSuiteV2,
        runner: EvalRunnerV2 | RunnerCallableV2 | object,
        *,
        candidate_version: str,
    ) -> EvalReportV2:
        """Evaluate an exact frozen v2 suite without dynamic code loading."""

        from riskprobe.evals import EvalHarnessV2, EvalSuiteV2

        if type(suite) is not EvalSuiteV2 or not suite.verify_integrity():
            raise TypeError("suite must be an integrity-checked EvalSuiteV2")
        return EvalHarnessV2(seed=suite.seed).evaluate(
            suite,
            runner,
            candidate_version=candidate_version,
        )

    def run(self) -> RunContext:
        with _stable_dataset_snapshot(self.config.dataset.path) as snapshot_path:
            dataset = ParquetDataset(snapshot_path)
            data_fingerprint = _parquet_metadata_fingerprint(snapshot_path)
            code_version = _code_identity()
            expected_dataset_id = _safe_dataset_id(self.config.dataset.id)
            context = self.store.create(
                self.config,
                data_fingerprint,
                code_version,
                dataset_id=expected_dataset_id,
                time_validation_enabled=self.config.time_validation_enabled,
            )
            finalize_fingerprint = _node_input_fingerprint(
                context.run_id, "finalize"
            )
            if context.is_existing:
                try:
                    runtime = RunRuntime(self.store.runs_dir, context.run_id)
                    runtime.reconcile_published(
                        input_fingerprint=finalize_fingerprint,
                        output={"artifact": "manifest.json"},
                        artifact_refs=(
                            ArtifactRef.from_path(
                                context.run_dir / "manifest.json",
                                _ARTIFACT_SCHEMAS["manifest.json"],
                            ),
                        ),
                    )
                except Exception:
                    # Immutable artifacts are the success fact. Runtime repair is best effort.
                    pass
                return context

            try:
                runtime = RunRuntime(self.store.runs_dir, context.run_id)

                def downstream(node_id: str) -> tuple[str, ...]:
                    index = _NODE_ORDER.index(node_id)
                    return _NODE_ORDER[index + 1 :]

                def verified_checkpoint(
                    node_id: str,
                    expected_artifacts: Mapping[str, str],
                ) -> Any:
                    input_fingerprint = _node_input_fingerprint(
                        context.run_id, node_id
                    )
                    checkpoint = runtime.checkpoint(
                        node_id, input_fingerprint=input_fingerprint
                    )
                    if checkpoint is None:
                        return None
                    verified = runtime.verified_checkpoint(
                        node_id,
                        input_fingerprint=input_fingerprint,
                        run_dir=context.run_dir,
                        expected_artifacts=expected_artifacts,
                    )
                    if verified is None:
                        runtime.invalidate_from(node_id, downstream(node_id))
                    return verified

                def mark_failed(
                    node_id: str,
                    input_fingerprint: str,
                    error: BaseException,
                ) -> None:
                    try:
                        runtime.fail_node(
                            node_id,
                            input_fingerprint=input_fingerprint,
                            error_class=error.__class__.__name__,
                        )
                    except Exception:
                        pass

                def execute_artifact_node(
                    node_id: str,
                    expected_artifacts: Mapping[str, str],
                    action: Any,
                    restore: Any,
                ) -> tuple[Any, bool]:
                    checkpoint = verified_checkpoint(node_id, expected_artifacts)
                    if checkpoint is not None:
                        try:
                            return restore(checkpoint.output), True
                        except Exception:
                            runtime.invalidate_from(node_id, downstream(node_id))
                    input_fingerprint = _node_input_fingerprint(
                        context.run_id, node_id
                    )
                    runtime.start_node(
                        node_id, input_fingerprint=input_fingerprint
                    )
                    try:
                        result, output = action()
                        references = tuple(
                            ArtifactRef.from_path(
                                context.run_dir / filename,
                                schema_version,
                            )
                            for filename, schema_version in expected_artifacts.items()
                        )
                        runtime.succeed_node(
                            node_id,
                            input_fingerprint=input_fingerprint,
                            output=output,
                            artifact_refs=references,
                        )
                    except BaseException as error:
                        mark_failed(node_id, input_fingerprint, error)
                        raise
                    return result, False

                profile = profile_dataset(dataset, self.config)
                self._assert_rule_conclusion_allowed(profile)
                artifact_profile = replace(
                    profile,
                    dataset_id=expected_dataset_id,
                )
                feature_names = self._feature_names(dataset)
                profile_fingerprint = _node_input_fingerprint(
                    context.run_id, "profile"
                )
                if runtime.checkpoint(
                    "profile", input_fingerprint=profile_fingerprint
                ) is None:
                    runtime.start_node(
                        "profile", input_fingerprint=profile_fingerprint
                    )
                    try:
                        runtime.succeed_node(
                            "profile",
                            input_fingerprint=profile_fingerprint,
                            output={"feature_count": len(feature_names)},
                        )
                    except BaseException as error:
                        mark_failed("profile", profile_fingerprint, error)
                        raise

                partition_artifacts = {
                    "data_profile.json": _ARTIFACT_SCHEMAS["data_profile.json"]
                }
                partition_checkpoint = verified_checkpoint(
                    "partition", partition_artifacts
                )
                partition_fingerprint = _node_input_fingerprint(
                    context.run_id, "partition"
                )
                if partition_checkpoint is None:
                    runtime.start_node(
                        "partition", input_fingerprint=partition_fingerprint
                    )
                    try:
                        (
                            train,
                            test,
                            holdout,
                            excluded_null_snapshot_rows,
                        ) = self._partitions(dataset, feature_names)
                        context.write_json(
                            "data_profile.json",
                            _profile_payload(
                                artifact_profile,
                                excluded_null_snapshot_rows=excluded_null_snapshot_rows,
                            ),
                        )
                        runtime.succeed_node(
                            "partition",
                            input_fingerprint=partition_fingerprint,
                            output={
                                "excluded_null_snapshot_rows": (
                                    excluded_null_snapshot_rows
                                ),
                                "holdout_rows": (
                                    holdout.height if holdout is not None else 0
                                ),
                                "test_rows": test.height,
                                "train_rows": train.height,
                            },
                            artifact_refs=(
                                ArtifactRef.from_path(
                                    context.run_dir / "data_profile.json",
                                    partition_artifacts["data_profile.json"],
                                ),
                            ),
                        )
                    except BaseException as error:
                        mark_failed("partition", partition_fingerprint, error)
                        raise
                else:
                    (
                        train,
                        test,
                        holdout,
                        excluded_null_snapshot_rows,
                    ) = self._partitions(dataset, feature_names)

                discover_artifacts = {
                    "candidate_rules.parquet": _ARTIFACT_SCHEMAS[
                        "candidate_rules.parquet"
                    ]
                }

                def discover_action() -> tuple[list[RiskRule], dict[str, Any]]:
                    rules = self._discover_from_train(train, feature_names)
                    context.write_parquet(
                        "candidate_rules.parquet", _candidate_frame(rules)
                    )
                    return rules, {"rule_count": len(rules)}

                def restore_rules(_output: Mapping[str, Any]) -> list[RiskRule]:
                    return _restore_rules(
                        context.run_dir / "candidate_rules.parquet"
                    )

                rules, _ = execute_artifact_node(
                    "discover",
                    discover_artifacts,
                    discover_action,
                    restore_rules,
                )

                validate_artifacts = {
                    "evidence_cards.json": _ARTIFACT_SCHEMAS[
                        "evidence_cards.json"
                    ]
                }

                def validate_action() -> tuple[
                    tuple[
                        list[EvidenceCard],
                        tuple[str, ...],
                        dict[str, Any],
                    ],
                    dict[str, Any],
                ]:
                    cards, validation_limitations = self._validate(
                        train, test, holdout, rules
                    )
                    if excluded_null_snapshot_rows:
                        null_snapshot_limitation = (
                            "Time validation excluded "
                            f"{excluded_null_snapshot_rows} rows with null snapshot values"
                        )
                        cards = _with_limitation(
                            cards,
                            null_snapshot_limitation,
                            downgrade=False,
                        )
                        validation_limitations = tuple(
                            sorted(
                                {
                                    *validation_limitations,
                                    null_snapshot_limitation,
                                }
                            )
                        )
                    institution_analysis = discover_local_rules(
                        train,
                        test,
                        cards,
                        feature_names,
                        target_col=self.config.columns.target,
                        segment_col=self.config.columns.segment,
                        snapshot_col=self.config.columns.snapshot,
                        time_validation_enabled=(
                            self.config.time_validation_enabled
                        ),
                        discovery_config=self.config.discovery,
                        validation_config=self.config.validation,
                        confirmed_features=frozenset(feature_names),
                        segment_display_name=self.config.segment_display_name,
                        metadata_grade=self.config.metadata_grade,
                        holdout=holdout,
                        expose_segment_values=(
                            self.config.privacy.expose_segment_values
                        ),
                    )
                    context.write_json(
                        "evidence_cards.json",
                        _evidence_payload(
                            cards,
                            time_validation_enabled=(
                                self.config.time_validation_enabled
                            ),
                        ),
                    )
                    return (
                        cards,
                        validation_limitations,
                        institution_analysis,
                    ), {
                        "evidence_count": len(cards),
                        "institution_analysis": institution_analysis,
                        "validation_limitations": list(validation_limitations),
                    }

                def restore_validation(
                    output: Mapping[str, Any],
                ) -> tuple[
                    list[EvidenceCard],
                    tuple[str, ...],
                    dict[str, Any],
                ]:
                    limitations = output.get("validation_limitations", [])
                    institution_analysis = output.get("institution_analysis")
                    if not isinstance(limitations, list) or not all(
                        isinstance(item, str) for item in limitations
                    ):
                        raise RuntimeError(
                            "validate checkpoint limitations are invalid"
                        )
                    if type(institution_analysis) is not dict:
                        raise RuntimeError(
                            "validate checkpoint institution analysis is invalid"
                        )
                    return (
                        _restore_cards(
                            context.run_dir / "evidence_cards.json"
                        ),
                        tuple(limitations),
                        institution_analysis,
                    )

                validation_result, cards_are_redacted = execute_artifact_node(
                    "validate",
                    validate_artifacts,
                    validate_action,
                    restore_validation,
                )
                cards, validation_limitations, institution_analysis = (
                    validation_result
                )
                split_rows = {
                    "train": train.height,
                    "test": test.height,
                    "holdout": holdout.height if holdout is not None else 0,
                }
                limitations = sorted(
                    {
                        *validation_limitations,
                        *(
                            redact_limitation(
                                limitation,
                                already_redacted=cards_are_redacted,
                            )
                            for card in cards
                            for limitation in card.limitations
                        ),
                    }
                )
                if profile.metadata_grade == "B":
                    limitations = sorted(
                        {"label performance window unknown", *limitations}
                    )
                validation_limitations = tuple(
                    sorted({*validation_limitations, *self._split_limitations})
                )
                limitations = sorted({*limitations, *self._split_limitations})

                report_artifacts = {
                    "metadata_report.json": _ARTIFACT_SCHEMAS[
                        "metadata_report.json"
                    ],
                    "risk_report.md": _ARTIFACT_SCHEMAS["risk_report.md"],
                }

                def report_action() -> tuple[None, dict[str, Any]]:
                    context.write_json(
                        "metadata_report.json",
                        {
                            "metadata_grade": profile.metadata_grade,
                            "limitations": limitations,
                            "split_rows": split_rows,
                            "split_strategy": (
                                "time_group_split"
                                if self.config.time_validation_enabled
                                else (
                                    "institution_target_stratified"
                                    if not self._split_limitations
                                    else "target_stratified_fallback"
                                )
                            ),
                            "time_validation_enabled": (
                                self.config.time_validation_enabled
                            ),
                            "institution_analysis": institution_analysis,
                        },
                    )
                    context.write_text(
                        "risk_report.md",
                        _render_service_report(
                            artifact_profile,
                            cards,
                            validation_limitations,
                            institution_analysis,
                            cards_are_redacted=cards_are_redacted,
                            expose_segment_values=(
                                self.config.privacy.expose_segment_values
                            ),
                        ),
                    )
                    return None, {"report_count": 2}

                def restore_report(_output: Mapping[str, Any]) -> None:
                    return None

                execute_artifact_node(
                    "report",
                    report_artifacts,
                    report_action,
                    restore_report,
                )

                existing_finalize = runtime.checkpoint(
                    "finalize", input_fingerprint=finalize_fingerprint
                )
                if existing_finalize is not None:
                    runtime.invalidate_from("finalize")
                runtime.start_node(
                    "finalize", input_fingerprint=finalize_fingerprint
                )
                try:
                    context.write_canonical_json(
                        "manifest.json",
                        {
                            "artifact_integrity": _artifact_integrity(
                                context.run_dir
                            ),
                            "artifacts": list(_ARTIFACT_NAMES),
                            "code_version": code_version,
                            "config_fingerprint": self.store.config_fingerprint(
                                self.config
                            ),
                            "data_fingerprint": data_fingerprint,
                            "dataset_id": expected_dataset_id,
                            "run_id": context.run_id,
                            "time_validation_enabled": (
                                self.config.time_validation_enabled
                            ),
                        },
                    )
                    manifest_reference = ArtifactRef.from_path(
                        context.run_dir / "manifest.json",
                        _ARTIFACT_SCHEMAS["manifest.json"],
                    )
                    context.finalize()
                except BaseException as error:
                    mark_failed("finalize", finalize_fingerprint, error)
                    raise

                try:
                    runtime.succeed_node(
                        "finalize",
                        input_fingerprint=finalize_fingerprint,
                        output={"artifact": "manifest.json"},
                        artifact_refs=(manifest_reference,),
                    )
                except Exception:
                    # Publication is immutable and already verified; trace can reconcile later.
                    return context
                return context
            except BaseException:
                context.release()
                raise

    def monitoring_snapshot(self) -> tuple[RunContext, ReferenceSnapshot]:
        """Build and persist an aggregate reference snapshot beside an immutable run."""
        context = self.run()
        cards = tuple(
            EvidenceCard.model_validate(_restore_json_tuples(item))
            for item in json.loads((context.run_dir / "evidence_cards.json").read_text(encoding="utf-8"))
        )
        manifest = json.loads(
            (context.run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        expected_fingerprint = manifest["data_fingerprint"]
        with _stable_dataset_snapshot(self.config.dataset.path) as snapshot_path:
            snapshot_fingerprint = _parquet_metadata_fingerprint(snapshot_path)
            if snapshot_fingerprint != expected_fingerprint:
                raise ValueError(
                    "monitoring reference snapshot does not match run data"
                )
            dataset = ParquetDataset(snapshot_path)
            profile = profile_dataset(dataset, self.config)
            feature_names = self._feature_names(dataset)
            monitoring_columns = tuple(
                dict.fromkeys(
                    (
                        self.config.columns.entity,
                        self.config.columns.snapshot,
                        self.config.columns.segment,
                        self.config.columns.target,
                        *feature_names,
                    )
                )
            )
            frame = dataset.collect(monitoring_columns)
            catalog = FeatureCatalog.from_columns(
                feature_names, self.config.features.families
            )
            snapshot = build_reference_snapshot(
                frame, profile, cards, catalog, self.config
            )
        output_dir = self.store.runs_dir / "monitoring" / context.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reference_snapshot.json").write_text(
            snapshot.model_dump_json(indent=2), encoding="utf-8"
        )
        return context, snapshot
