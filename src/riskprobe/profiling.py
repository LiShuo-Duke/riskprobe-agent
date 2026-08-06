from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.features.catalog import FeatureCatalog, QualityIssue, check_window_invariants
from riskprobe.io.parquet import ParquetDataset


class DataContractError(ValueError):
    """Raised when a dataset cannot satisfy its configured column contract."""


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    dataset_id: str
    row_count: int
    feature_count: int
    positive_rate: float | None
    segment_counts: Mapping[str, int]
    snapshot_min: date | None
    snapshot_max: date | None
    metadata_grade: str
    issues: tuple[QualityIssue, ...]


def profile_dataset(dataset: ParquetDataset, config: ProjectConfig) -> DatasetProfile:
    schema = dataset.schema()
    role_columns = (
        config.columns.entity,
        config.columns.snapshot,
        config.columns.segment,
        config.columns.target,
    )
    missing_roles = tuple(column for column in role_columns if column not in schema)
    if missing_roles:
        raise DataContractError(f"missing required role columns: {', '.join(missing_roles)}")

    feature_columns = tuple(column for column in schema.names() if column not in role_columns)
    selected_columns = (
        config.columns.snapshot,
        config.columns.segment,
        config.columns.target,
        *feature_columns,
    )
    frame = dataset.collect(selected_columns)
    row_count = frame.height

    snapshot_min, snapshot_max, parsed_snapshots = _snapshot_range(frame, config)
    positive_rate = _positive_rate(frame, config)
    segment_counts = _segment_counts(frame, config.columns.segment)

    catalog = FeatureCatalog.from_columns(feature_columns, config.features.families)
    issues = list(check_window_invariants(frame, catalog))
    if config.target.performance_window_days is None:
        issues.append(
            QualityIssue(
                code="LABEL_PERFORMANCE_WINDOW_UNKNOWN",
                severity="warning",
                family="target",
                features=(),
                affected_rows=row_count,
                message="target performance window is not configured",
            )
        )

    issues.extend(
        _single_class_issues(
            frame,
            group_column=config.columns.segment,
            target_column=config.columns.target,
        )
    )
    if parsed_snapshots is not None:
        frame_with_dates = frame.with_columns(parsed_snapshots.alias(config.columns.snapshot))
        issues.extend(
            _single_class_issues(
                frame_with_dates,
                group_column=config.columns.snapshot,
                target_column=config.columns.target,
            )
        )

    return DatasetProfile(
        dataset_id=config.dataset.id,
        row_count=row_count,
        feature_count=len(feature_columns),
        positive_rate=positive_rate,
        segment_counts=MappingProxyType(segment_counts),
        snapshot_min=snapshot_min,
        snapshot_max=snapshot_max,
        metadata_grade=config.metadata_grade,
        issues=tuple(issues),
    )


def _snapshot_range(
    frame: pl.DataFrame,
    config: ProjectConfig,
) -> tuple[date | None, date | None, pl.Series | None]:
    snapshot_column = config.columns.snapshot
    snapshots = frame.get_column(snapshot_column)
    if snapshots.null_count() == len(snapshots):
        raise DataContractError(f"snapshot column {snapshot_column} must not be all null")
    if not config.time_validation_enabled:
        return None, None, None

    if snapshots.dtype == pl.Date:
        parsed = snapshots
    elif isinstance(snapshots.dtype, pl.Datetime):
        parsed = snapshots.cast(pl.Date)
    else:
        parsed = snapshots.cast(pl.String).str.to_date(strict=False)
    if parsed.null_count() != snapshots.null_count():
        raise DataContractError(f"snapshot column {snapshot_column} contains invalid dates")
    return parsed.min(), parsed.max(), parsed


def _positive_rate(frame: pl.DataFrame, config: ProjectConfig) -> float | None:
    if frame.height == 0:
        return None
    positives = frame.get_column(config.columns.target) == config.target.positive_value
    return float(positives.fill_null(False).sum() / frame.height)


def _segment_counts(frame: pl.DataFrame, segment_column: str) -> dict[str, int]:
    counts = frame.group_by(segment_column).len()
    return {
        str(segment): int(count)
        for segment, count in counts.iter_rows()
    }


def _single_class_issues(
    frame: pl.DataFrame,
    *,
    group_column: str,
    target_column: str,
) -> tuple[QualityIssue, ...]:
    grouped = frame.group_by(group_column).agg(
        pl.col(target_column).drop_nulls().n_unique().alias("class_count"),
        pl.len().alias("row_count"),
    )
    issues = []
    for group_value, class_count, group_rows in grouped.iter_rows():
        if class_count <= 1:
            issues.append(
                QualityIssue(
                    code="SINGLE_CLASS_SLICE",
                    severity="warning",
                    family=group_column,
                    features=(),
                    affected_rows=int(group_rows),
                    message=f"{group_column} slice {group_value!r} contains a single target class",
                )
            )
    return tuple(issues)
