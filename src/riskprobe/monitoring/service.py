"""Deterministic orchestration for comprehensive aggregate diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset
from riskprobe.monitoring.drift import diagnose_drift
from riskprobe.monitoring.models import DiagnosticReport, RiskFinding, SafeProfile
from riskprobe.monitoring.quality import NumericRange, diagnose_quality
from riskprobe.monitoring.segments import diagnose_segments
from riskprobe.monitoring.time import diagnose_time
from riskprobe.profiling import profile_dataset


def diagnose_dataset(
    dataset: ParquetDataset,
    config: ProjectConfig,
    *,
    numeric_ranges: Mapping[str, NumericRange] | None = None,
    drift_bin_count: int = 10,
) -> DiagnosticReport:
    """Run all aggregate diagnostics without serializing rows, paths, or segment labels."""

    profile = profile_dataset(dataset, config)
    safe_profile = SafeProfile.from_profile(profile)
    schema = dataset.schema()
    role_columns = (
        config.columns.entity,
        config.columns.snapshot,
        config.columns.segment,
        config.columns.target,
    )
    features = tuple(config.features.select_columns(schema.names(), role_columns))
    selected_columns = tuple(dict.fromkeys((*role_columns, *features)))
    frame = dataset.collect(selected_columns)

    findings: list[RiskFinding] = list(
        diagnose_quality(
            frame,
            entity_column=config.columns.entity,
            snapshot_column=config.columns.snapshot,
            target_column=config.columns.target,
            feature_columns=features,
            numeric_ranges=numeric_ranges,
        )
    )

    reference, current = _reference_current_frames(
        frame,
        snapshot_column=config.columns.snapshot,
        time_enabled=config.time_validation_enabled,
    )
    if reference.height and current.height:
        findings.extend(
            diagnose_drift(
                reference,
                current,
                feature_columns=features,
                target_column=config.columns.target,
                positive_value=config.target.positive_value,
                bin_count=drift_bin_count,
            )
        )

    findings.extend(
        diagnose_segments(
            frame,
            segment_column=config.columns.segment,
            target_column=config.columns.target,
            min_group_size=config.validation.min_group_size,
            positive_value=config.target.positive_value,
            token_namespace=config.dataset.id,
        )
    )

    limitations: list[str] = []
    if config.time_validation_enabled:
        findings.extend(
            diagnose_time(
                frame,
                snapshot_column=config.columns.snapshot,
                target_column=config.columns.target,
                positive_value=config.target.positive_value,
            )
        )
    else:
        limitations.append("time_monitoring_disabled")
    if not reference.height or not current.height:
        limitations.append("insufficient_drift_reference")
    if config.metadata_grade == "B":
        limitations.append("label_performance_window_unknown")

    return DiagnosticReport(
        profile=safe_profile,
        findings=tuple(findings),
        limitations=tuple(limitations),
    )


def _reference_current_frames(
    frame: pl.DataFrame,
    *,
    snapshot_column: str,
    time_enabled: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if frame.height < 2:
        return frame, frame.head(0)
    if time_enabled:
        dates = _date_series(frame.get_column(snapshot_column))
        unique_dates = sorted(date for date in dates.drop_nulls().unique().to_list())
        if len(unique_dates) >= 2:
            cutoff = max(1, min(len(unique_dates) - 1, math.ceil(len(unique_dates) * 0.6)))
            reference_dates = unique_dates[:cutoff]
            reference = frame.filter(dates.is_in(reference_dates))
            current = frame.filter(dates.is_not_null() & ~dates.is_in(reference_dates))
            if reference.height and current.height:
                return reference, current
    cutoff = max(1, min(frame.height - 1, math.floor(frame.height * 0.7)))
    return frame.head(cutoff), frame.slice(cutoff)


def _date_series(series: pl.Series) -> pl.Series:
    if series.dtype == pl.Date:
        return series
    if isinstance(series.dtype, pl.Datetime):
        return series.cast(pl.Date)
    if series.dtype == pl.Null:
        return pl.Series(series.name, [None] * len(series), dtype=pl.Date)
    return series.cast(pl.String, strict=False).str.to_date(strict=False)


__all__ = ["diagnose_dataset"]
