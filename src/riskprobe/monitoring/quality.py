"""Aggregate-only data quality diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from riskprobe.monitoring.models import (
    FindingKind,
    FindingSeverity,
    RiskFinding,
    finding_sort_key,
)

NumericRange = tuple[float | None, float | None]


def diagnose_quality(
    frame: pl.DataFrame,
    *,
    entity_column: str,
    snapshot_column: str,
    feature_columns: Sequence[str] | None = None,
    target_column: str | None = None,
    numeric_ranges: Mapping[str, NumericRange] | None = None,
) -> tuple[RiskFinding, ...]:
    """Diagnose missing, constant, duplicate, range, and non-finite values."""

    if frame.height == 0:
        return ()
    features = tuple(feature_columns or ())
    _require_columns(
        frame,
        tuple(
            dict.fromkeys(
                (entity_column, snapshot_column, *((target_column,) if target_column else ()), *features)
            )
        ),
    )
    ranges = dict(numeric_ranges or {})
    _validate_ranges(ranges)
    findings: list[RiskFinding] = []

    missing_columns = tuple(
        dict.fromkeys(
            (entity_column, snapshot_column, *((target_column,) if target_column else ()), *features)
        )
    )
    for column in missing_columns:
        missing_count = frame.get_column(column).null_count()
        if missing_count:
            findings.append(
                _affected_finding(
                    code="missing_values",
                    feature=column,
                    count=missing_count,
                    total=frame.height,
                )
            )

    for feature in features:
        series = frame.get_column(feature)
        non_null_count = frame.height - series.null_count()
        if non_null_count and series.drop_nulls().n_unique() == 1:
            findings.append(
                RiskFinding(
                    kind=FindingKind.DATA_QUALITY,
                    severity=FindingSeverity.WARNING,
                    code="constant_feature",
                    feature=feature,
                    metrics={"affected_count": frame.height, "affected_rate": 1.0},
                )
            )

        non_finite_count = _non_finite_count(series)
        if non_finite_count:
            findings.append(
                _affected_finding(
                    code="non_finite_values",
                    feature=feature,
                    count=non_finite_count,
                    total=frame.height,
                )
            )

        if feature in ranges:
            out_of_range_count = _out_of_range_count(series, ranges[feature])
            if out_of_range_count:
                findings.append(
                    _affected_finding(
                        code="numeric_out_of_range",
                        feature=feature,
                        count=out_of_range_count,
                        total=frame.height,
                    )
                )

    duplicate_count = _duplicate_key_count(frame, entity_column, snapshot_column)
    if duplicate_count:
        findings.append(
            _affected_finding(
                code="duplicate_entity_snapshot",
                feature=None,
                count=duplicate_count,
                total=frame.height,
            )
        )

    return tuple(sorted(findings, key=finding_sort_key))


def _affected_finding(
    *,
    code: str,
    feature: str | None,
    count: int,
    total: int,
) -> RiskFinding:
    rate = count / total if total else 0.0
    return RiskFinding(
        kind=FindingKind.DATA_QUALITY,
        severity=_rate_severity(rate),
        code=code,
        feature=feature,
        metrics={"affected_count": count, "affected_rate": float(rate)},
    )


def _rate_severity(rate: float) -> FindingSeverity:
    return FindingSeverity.CRITICAL if rate >= 0.25 else FindingSeverity.WARNING


def _duplicate_key_count(
    frame: pl.DataFrame,
    entity_column: str,
    snapshot_column: str,
) -> int:
    unique_count = int(
        frame.select(pl.struct(entity_column, snapshot_column).n_unique()).item()
    )
    return max(0, frame.height - unique_count)


def _non_finite_count(series: pl.Series) -> int:
    if not series.dtype.is_numeric():
        return 0
    return sum(
        1
        for value in series.drop_nulls().to_list()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    )


def _out_of_range_count(series: pl.Series, bounds: NumericRange) -> int:
    if not series.dtype.is_numeric():
        return 0
    lower, upper = bounds
    count = 0
    for value in series.drop_nulls().to_list():
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        if (lower is not None and numeric < lower) or (
            upper is not None and numeric > upper
        ):
            count += 1
    return count


def _validate_ranges(ranges: Mapping[str, NumericRange]) -> None:
    for bounds in ranges.values():
        if len(bounds) != 2:
            raise ValueError("numeric ranges require lower and upper bounds")
        lower, upper = bounds
        if lower is not None and not math.isfinite(lower):
            raise ValueError("numeric range bounds must be finite")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("numeric range bounds must be finite")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("numeric range lower bound exceeds upper bound")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = tuple(column for column in columns if column not in frame.columns)
    if missing:
        raise ValueError("quality diagnostics require configured columns")


analyze_quality = diagnose_quality
quality_findings = diagnose_quality

__all__ = ["NumericRange", "analyze_quality", "diagnose_quality", "quality_findings"]
