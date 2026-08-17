"""Monthly aggregate target-rate and sample-volume diagnostics."""

from __future__ import annotations

import polars as pl

from riskprobe.monitoring.models import (
    FindingKind,
    FindingSeverity,
    RiskFinding,
    finding_sort_key,
)


def diagnose_time(
    frame: pl.DataFrame,
    *,
    snapshot_column: str,
    target_column: str,
    positive_value: int = 1,
    min_months: int = 3,
    min_month_size: int = 1,
    target_warning: float = 0.10,
    target_critical: float = 0.25,
    sample_drop_warning: float = 0.25,
    sample_drop_critical: float = 0.50,
) -> tuple[RiskFinding, ...]:
    """Aggregate by month and detect target shifts, abrupt drops, and sparse history."""

    if min_months < 1 or min_month_size < 1:
        raise ValueError("time diagnostic minimums must be positive")
    if snapshot_column not in frame.columns or target_column not in frame.columns:
        raise ValueError("time diagnostics require configured columns")

    monthly = _monthly_aggregates(
        frame,
        snapshot_column=snapshot_column,
        target_column=target_column,
        positive_value=positive_value,
    )
    if len(monthly) < min_months:
        return (
            RiskFinding(
                kind=FindingKind.TIME_INSTABILITY,
                severity=FindingSeverity.WARNING,
                code="insufficient_time_data",
                metrics={"month_count": len(monthly), "sample_count": frame.height},
            ),
        )

    findings: list[RiskFinding] = []
    sparse_month_count = sum(count < min_month_size for _, count, _ in monthly)
    if sparse_month_count:
        findings.append(
            RiskFinding(
                kind=FindingKind.TIME_INSTABILITY,
                severity=FindingSeverity.WARNING,
                code="insufficient_month_sample",
                metrics={
                    "month_count": len(monthly),
                    "sample_count": frame.height,
                    "affected_count": sparse_month_count,
                },
            )
        )

    for previous, current in zip(monthly, monthly[1:], strict=False):
        _, previous_count, previous_rate = previous
        current_period, current_count, current_rate = current
        rate_shift = abs(current_rate - previous_rate)
        if rate_shift >= target_warning:
            findings.append(
                RiskFinding(
                    kind=FindingKind.TIME_INSTABILITY,
                    severity=_threshold_severity(
                        rate_shift,
                        warning=target_warning,
                        critical=target_critical,
                    ),
                    code="monthly_target_rate_change",
                    period=current_period,
                    metrics={
                        "previous_rate": previous_rate,
                        "current_rate": current_rate,
                        "rate_shift": float(rate_shift),
                    },
                )
            )
        drop_rate = (
            (previous_count - current_count) / previous_count
            if previous_count > 0 and current_count < previous_count
            else 0.0
        )
        if drop_rate >= sample_drop_warning:
            findings.append(
                RiskFinding(
                    kind=FindingKind.TIME_INSTABILITY,
                    severity=_threshold_severity(
                        drop_rate,
                        warning=sample_drop_warning,
                        critical=sample_drop_critical,
                    ),
                    code="monthly_sample_drop",
                    period=current_period,
                    metrics={
                        "previous_count": previous_count,
                        "current_count": current_count,
                        "drop_rate": float(drop_rate),
                    },
                )
            )
    return tuple(sorted(findings, key=finding_sort_key))


def _monthly_aggregates(
    frame: pl.DataFrame,
    *,
    snapshot_column: str,
    target_column: str,
    positive_value: int,
) -> tuple[tuple[str, int, float], ...]:
    if frame.height == 0:
        return ()
    dates = _date_series(frame.get_column(snapshot_column))
    monthly_frame = pl.DataFrame(
        {
            "month": dates.dt.strftime("%Y-%m"),
            "target": frame.get_column(target_column),
        }
    ).filter(pl.col("month").is_not_null())
    if monthly_frame.height == 0:
        return ()
    grouped = (
        monthly_frame.group_by("month")
        .agg(
            pl.len().alias("sample_count"),
            (pl.col("target") == positive_value).sum().alias("positive_count"),
        )
        .sort("month")
    )
    return tuple(
        (str(month), int(count), float(int(positive_count) / int(count)))
        for month, count, positive_count in grouped.iter_rows()
    )


def _date_series(series: pl.Series) -> pl.Series:
    if series.dtype == pl.Date:
        return series
    if isinstance(series.dtype, pl.Datetime):
        return series.cast(pl.Date)
    if series.dtype == pl.Null:
        return pl.Series(series.name, [None] * len(series), dtype=pl.Date)
    return series.cast(pl.String, strict=False).str.to_date(strict=False)


def _threshold_severity(
    value: float,
    *,
    warning: float,
    critical: float,
) -> FindingSeverity:
    if not 0 <= warning <= critical:
        raise ValueError("time diagnostic thresholds are invalid")
    return FindingSeverity.CRITICAL if value >= critical else FindingSeverity.WARNING


analyze_time = diagnose_time
time_findings = diagnose_time

__all__ = ["analyze_time", "diagnose_time", "time_findings"]
