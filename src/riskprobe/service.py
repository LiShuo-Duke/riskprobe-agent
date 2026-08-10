import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from riskprobe.artifacts import RunContext, RunStore
from riskprobe.config import ProjectConfig
from riskprobe.dates import normalize_date_series
from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.institutions import discover_local_rules
from riskprobe.models import EvidenceCard, RiskRule, SliceMetrics
from riskprobe.monitoring.models import ReferenceSnapshot
from riskprobe.monitoring.reference import build_reference_snapshot
from riskprobe.profiling import DatasetProfile, profile_dataset
from riskprobe.reporting import (
    evidence_sort_key,
    redact_limitation,
    redact_segment_value,
    render_risk_report,
    safe_dataset_id,
)
from riskprobe.rules.discovery import DiscoveryResult, discover_rules, discover_with_metrics
from riskprobe.rules.validation import validate_rules

_DISCOVERY_SAMPLE_LIMIT = 50_000
_ARTIFACT_NAMES = (
    "manifest.json",
    "metadata_report.json",
    "data_profile.json",
    "candidate_rules.parquet",
    "evidence_cards.json",
    "risk_report.md",
)
_SLICE_ORDER = {"dataset": 0, "segment": 1, "time": 2}
_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}
_SNAPSHOT_ROOT_PREFIX = "riskprobe-input-snapshots-"
_SNAPSHOT_DIR_PREFIX = "snapshot-"
_SNAPSHOT_MARKER_NAME = ".riskprobe-snapshot"
_SNAPSHOT_LOCK_NAME = ".lock"
_SNAPSHOT_FILE_NAME = "input.parquet"
_SNAPSHOT_MARKER = b"riskprobe raw snapshot v1\n"


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


def _sorted_card(card: EvidenceCard) -> EvidenceCard:
    slices = tuple(
        sorted(
            (
                item.model_copy(
                    update={"slice_value": redact_segment_value(item.slice_value)}
                )
                if item.slice_type == "segment"
                else item
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
                sorted({redact_limitation(item) for item in card.limitations})
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


def _render_service_report(
    profile: DatasetProfile,
    cards: list[EvidenceCard],
    validation_limitations: tuple[str, ...],
    institution_analysis: dict[str, Any] | None = None,
    *,
    expose_segment_values: bool = False,
) -> str:
    report = render_risk_report(
        profile,
        cards,
        institution_analysis=institution_analysis,
        expose_segment_values=expose_segment_values,
    )
    if validation_limitations and not cards:
        replacement = "\n".join(
            f"- {limitation}" for limitation in sorted(validation_limitations)
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
    def __init__(self, *, config: ProjectConfig | Path, runs_dir: Path) -> None:
        self.config = (
            ProjectConfig.from_yaml(config) if isinstance(config, Path) else config
        )
        self.store = RunStore(runs_dir)
        self._split_limitations: tuple[str, ...] = ()

    def _dataset(self) -> ParquetDataset:
        return ParquetDataset(self.config.dataset.path)

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

    def inspect(self) -> DatasetProfile:
        return profile_dataset(self._dataset(), self.config)

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
        profile = self.inspect()
        self._assert_rule_conclusion_allowed(profile)
        dataset = self._dataset()
        feature_names = self._feature_names(dataset)
        train, _, _, _ = self._partitions(dataset, feature_names)
        return self._discover_from_train(train, feature_names)

    def run(self) -> RunContext:
        with _stable_dataset_snapshot(self.config.dataset.path) as snapshot_path:
            dataset = ParquetDataset(snapshot_path)
            data_fingerprint = _parquet_metadata_fingerprint(snapshot_path)
            code_version = _package_version()
            expected_dataset_id = _safe_dataset_id(self.config.dataset.id)
            context = self.store.create(
                self.config,
                data_fingerprint,
                code_version,
                dataset_id=expected_dataset_id,
                time_validation_enabled=self.config.time_validation_enabled,
            )
            if context.is_existing:
                return context

            try:
                profile = profile_dataset(dataset, self.config)
                self._assert_rule_conclusion_allowed(profile)
                artifact_profile = replace(
                    profile,
                    dataset_id=expected_dataset_id,
                )
                feature_names = self._feature_names(dataset)
                train, test, holdout, excluded_null_snapshot_rows = self._partitions(
                    dataset, feature_names
                )
                rules = self._discover_from_train(train, feature_names)
                cards, validation_limitations = self._validate(
                    train, test, holdout, rules
                )
                validation_limitations = tuple(
                    sorted({*validation_limitations, *self._split_limitations})
                )
                institution_analysis = discover_local_rules(
                    train,
                    test,
                    cards,
                    feature_names,
                    target_col=self.config.columns.target,
                    segment_col=self.config.columns.segment,
                    snapshot_col=self.config.columns.snapshot,
                    time_validation_enabled=self.config.time_validation_enabled,
                    discovery_config=self.config.discovery,
                    validation_config=self.config.validation,
                    confirmed_features=frozenset(feature_names),
                    segment_display_name=self.config.segment_display_name,
                    metadata_grade=self.config.metadata_grade,
                    holdout=holdout,
                    expose_segment_values=self.config.privacy.expose_segment_values,
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
                split_rows = {
                    "train": train.height,
                    "test": test.height,
                    "holdout": holdout.height if holdout is not None else 0,
                }
                limitations = sorted(
                    {
                        *validation_limitations,
                        *(
                            redact_limitation(limitation)
                            for card in cards
                            for limitation in card.limitations
                        ),
                    }
                )
                if profile.metadata_grade == "B":
                    limitations = sorted(
                        {"label performance window unknown", *limitations}
                    )
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
                        "time_validation_enabled": self.config.time_validation_enabled,
                        "institution_analysis": institution_analysis,
                    },
                )
                context.write_json(
                    "data_profile.json",
                    _profile_payload(
                        artifact_profile,
                        excluded_null_snapshot_rows=excluded_null_snapshot_rows,
                    ),
                )
                context.write_parquet(
                    "candidate_rules.parquet", _candidate_frame(rules)
                )
                context.write_json(
                    "evidence_cards.json",
                    _evidence_payload(
                        cards,
                        time_validation_enabled=self.config.time_validation_enabled,
                    ),
                )
                context.write_text(
                    "risk_report.md",
                    _render_service_report(
                        artifact_profile,
                        cards,
                        validation_limitations,
                        institution_analysis,
                        expose_segment_values=self.config.privacy.expose_segment_values,
                    ),
                )
                context.write_canonical_json(
                    "manifest.json",
                    {
                        "artifact_integrity": _artifact_integrity(context.run_dir),
                        "artifacts": list(_ARTIFACT_NAMES),
                        "code_version": code_version,
                        "config_fingerprint": self.store.config_fingerprint(self.config),
                        "data_fingerprint": data_fingerprint,
                        "dataset_id": expected_dataset_id,
                        "run_id": context.run_id,
                        "time_validation_enabled": self.config.time_validation_enabled,
                    },
                )
                context.finalize()
                return context
            except BaseException:
                context.cleanup()
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
