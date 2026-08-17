import polars as pl
import pytest

from riskprobe.monitoring.models import FindingSeverity
from riskprobe.monitoring.time import diagnose_time
from riskprobe.privacy import assert_safe_payload


def _find(findings: tuple[object, ...], code: str):
    return next(finding for finding in findings if finding.code == code)


def test_time_diagnostics_detect_monthly_target_rate_change() -> None:
    frame = pl.DataFrame(
        {
            "snapshot": ["2024-01-01"] * 10 + ["2024-02-01"] * 10,
            "target": [1] + [0] * 9 + [1] * 4 + [0] * 6,
        }
    )

    findings = diagnose_time(
        frame,
        snapshot_column="snapshot",
        target_column="target",
        min_months=2,
    )

    target = _find(findings, "monthly_target_rate_change")
    assert target.severity is FindingSeverity.CRITICAL
    assert target.period == "2024-02"
    assert target.metrics["previous_rate"] == pytest.approx(0.1)
    assert target.metrics["current_rate"] == pytest.approx(0.4)
    assert target.metrics["rate_shift"] == pytest.approx(0.3)
    assert_safe_payload(target.model_dump(mode="json"))


def test_time_diagnostics_detect_abrupt_sample_drop() -> None:
    frame = pl.DataFrame(
        {
            "snapshot": ["2024-01-01"] * 20 + ["2024-02-01"] * 5,
            "target": [0] * 25,
        }
    )

    finding = _find(
        diagnose_time(
            frame,
            snapshot_column="snapshot",
            target_column="target",
            min_months=2,
        ),
        "monthly_sample_drop",
    )

    assert finding.severity is FindingSeverity.CRITICAL
    assert finding.metrics == {
        "previous_count": 20,
        "current_count": 5,
        "drop_rate": 0.75,
    }


def test_time_diagnostics_report_insufficient_months_without_raw_dates() -> None:
    frame = pl.DataFrame(
        {
            "snapshot": ["2024-01-03", "2024-01-19"],
            "target": [0, 1],
        }
    )

    findings = diagnose_time(
        frame,
        snapshot_column="snapshot",
        target_column="target",
        min_months=3,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.code == "insufficient_time_data"
    assert finding.metrics == {"month_count": 1, "sample_count": 2}
    assert "2024-01-03" not in str(finding.model_dump(mode="json"))


def test_time_diagnostics_handle_all_null_dates_as_insufficient() -> None:
    frame = pl.DataFrame({"snapshot": [None, None], "target": [0, 1]})

    finding = diagnose_time(
        frame,
        snapshot_column="snapshot",
        target_column="target",
        min_months=2,
    )[0]

    assert finding.code == "insufficient_time_data"
    assert finding.metrics == {"month_count": 0, "sample_count": 2}
