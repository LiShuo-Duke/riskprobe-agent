"""Small-cell-suppressed segment risk diagnostics."""

from __future__ import annotations

import math

import polars as pl

from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
from riskprobe.privacy import tokenize_segment


def diagnose_segments(
    frame: pl.DataFrame,
    *,
    segment_column: str,
    target_column: str,
    min_group_size: int,
    positive_value: int = 1,
    token_namespace: str = "",
) -> tuple[RiskFinding, ...]:
    """Return aggregate segment rates only for groups meeting minimum size."""

    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    if segment_column not in frame.columns or target_column not in frame.columns:
        raise ValueError("segment diagnostics require configured columns")
    if frame.height == 0:
        return ()

    positive_count = int((frame.get_column(target_column) == positive_value).sum())
    base_rate = positive_count / frame.height
    grouped = frame.group_by(segment_column).agg(
        pl.len().alias("sample_count"),
        (pl.col(target_column) == positive_value).sum().alias("positive_count"),
    )
    findings: list[RiskFinding] = []
    for segment_value, sample_count, group_positive_count in grouped.iter_rows():
        count = int(sample_count)
        if count < min_group_size:
            continue
        target_rate = int(group_positive_count) / count
        lift = target_rate / base_rate if base_rate > 0 else 0.0
        population_share = count / frame.height
        findings.append(
            RiskFinding(
                kind=FindingKind.SEGMENT_RISK,
                severity=_segment_severity(lift),
                code="segment_target_rate",
                segment_token=tokenize_segment(
                    segment_value,
                    namespace=token_namespace,
                ),
                metrics={
                    "sample_count": count,
                    "target_rate": float(target_rate),
                    "lift": float(lift),
                    "population_share": float(population_share),
                },
            )
        )
    return tuple(sorted(findings, key=lambda finding: finding.finding_id))


def _segment_severity(lift: float) -> FindingSeverity:
    if not math.isfinite(lift):
        return FindingSeverity.CRITICAL
    if lift >= 2.0:
        return FindingSeverity.CRITICAL
    if lift >= 1.25:
        return FindingSeverity.WARNING
    return FindingSeverity.INFO


analyze_segments = diagnose_segments
segment_findings = diagnose_segments

__all__ = ["analyze_segments", "diagnose_segments", "segment_findings"]
