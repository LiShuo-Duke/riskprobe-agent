from collections import Counter
from collections.abc import Sequence
import hashlib
from pathlib import Path, PureWindowsPath
import re
from urllib.parse import unquote, urlsplit

from riskprobe.models import EvidenceCard
from riskprobe.profiling import DatasetProfile

_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}
_EMBEDDED_POSIX_PATH = re.compile(
    r"(?:^|[=:\s|;,])/(?:[^/\s]+/)+[^/\s]+"
)
_EMBEDDED_WINDOWS_PATH = re.compile(
    r"(?:^|[=:\s|;,])(?:[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])+[^\\/\s]+|\\\\[^\\/\s]+[\\/][^\s]+)"
)


def safe_dataset_id(dataset_id: str) -> str:
    decoded = unquote(dataset_id)
    parsed = urlsplit(decoded)
    is_file_uri = parsed.scheme.lower() == "file"
    is_path = (
        is_file_uri
        or Path(decoded).is_absolute()
        or PureWindowsPath(decoded).is_absolute()
        or _EMBEDDED_POSIX_PATH.search(decoded) is not None
        or _EMBEDDED_WINDOWS_PATH.search(decoded) is not None
    )
    if not is_path:
        return dataset_id
    digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8]
    return f"dataset-{digest}"


def redact_segment_value(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"segment-{digest}"


def redact_limitation(limitation: str) -> str:
    prefix = "holdout: " if limitation.startswith("holdout: ") else ""
    body = limitation[len(prefix) :]
    descriptor, separator, value = body.partition(": ")
    if separator and descriptor.startswith("single-class ") and descriptor != "single-class time":
        return f"{prefix}{descriptor}: {redact_segment_value(value)}"
    return limitation


def _issue_message(code: str, message: str) -> str:
    return "single-class slice detected" if code == "SINGLE_CLASS_SLICE" else message


def evidence_sort_key(card: EvidenceCard) -> tuple[int, float, str]:
    return _GRADE_ORDER[card.grade], -card.test.lift, card.rule.rule_id


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def render_risk_report(
    profile: DatasetProfile,
    evidence_cards: Sequence[EvidenceCard],
) -> str:
    cards = sorted(evidence_cards, key=evidence_sort_key)
    counts = Counter(card.grade for card in cards)
    time_validation_enabled = (
        profile.snapshot_min is not None
        and profile.snapshot_max is not None
        and profile.snapshot_min != profile.snapshot_max
    )
    has_holdout = any(
        item.slice_type == "dataset" and item.slice_value == "Holdout"
        for card in cards
        for item in card.slices
    )
    lines = [
        "# RiskProbe Risk Report",
        "",
        f"**Metadata Grade: {profile.metadata_grade}**",
    ]
    if profile.metadata_grade == "B":
        limitation = (
            "label performance window unknown; evidence reflects stability across "
            "time slices, not a known performance window; conclusions require "
            "additional validation."
        )
        if not time_validation_enabled:
            limitation = (
                "label performance window unknown; time-slice stability was not "
                "evaluated; conclusions require additional validation."
            )
        lines.extend(["", f"> Limitation: {limitation}"])
    lines.extend(
        [
            "",
            "## Sample Overview",
            "",
            f"- Dataset: `{safe_dataset_id(profile.dataset_id)}`",
            f"- Rows: {profile.row_count}",
            f"- Features: {profile.feature_count}",
            f"- Positive rate: {_number(profile.positive_rate)}",
            f"- Segment count: {len(profile.segment_counts)}",
        ]
    )
    if time_validation_enabled:
        lines.extend(
            [
                f"- Snapshot range: {profile.snapshot_min.isoformat()} to "
                f"{profile.snapshot_max.isoformat()}",
            ]
        )
    lines.extend(
        [
            "",
            "## Quality Issues",
            "",
        ]
    )
    if profile.issues:
        lines.extend(
            f"- [{issue.severity}] {issue.code}: {_issue_message(issue.code, issue.message)} "
            f"(affected rows: {issue.affected_rows})"
            for issue in sorted(
                profile.issues,
                key=lambda item: (item.severity, item.code, item.family, item.message),
            )
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            "| Stable | Local | Unstable | Suspicious |",
            "| ---: | ---: | ---: | ---: |",
            f"| {counts['Stable']} | {counts['Local']} | {counts['Unstable']} | "
            f"{counts['Suspicious']} |",
            "",
            "## Top Rule Evidence",
            "",
        ]
    )
    headers = [
        "Rule ID",
        "Grade",
        "Test Lift",
        "Test Coverage",
        "Adjusted p-value",
        "Lift CI",
        "Segment Consistency",
    ]
    if has_holdout:
        headers.insert(3, "Holdout Lift")
    if time_validation_enabled:
        headers.append("Time Decay")
    lines.extend(
        [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(['---'] + ['---:' for _ in headers[1:]])} |",
        ]
    )
    for card in cards:
        row = [
            card.rule.rule_id,
            card.grade,
            _number(card.test.lift),
            _number(card.test.coverage),
            _number(card.adjusted_p_value),
            f"{_number(card.lift_ci[0])}–{_number(card.lift_ci[1])}",
            _number(card.segment_consistency),
        ]
        if has_holdout:
            holdout = next(
                (
                    item
                    for item in card.slices
                    if item.slice_type == "dataset" and item.slice_value == "Holdout"
                ),
                None,
            )
            row.insert(3, _number(holdout.metrics.lift if holdout is not None else None))
        if time_validation_enabled:
            row.append(_number(card.max_time_decay))
        lines.append(f"| {' | '.join(row)} |")
    if not cards:
        lines.append(f"| {' | '.join(['None'] + ['—' for _ in headers[1:]])} |")

    limitations = sorted(
        {
            redact_limitation(limitation)
            for card in cards
            for limitation in card.limitations
        }
    )
    if profile.metadata_grade == "B":
        limitations = sorted({"label performance window unknown", *limitations})
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in limitations)
    if not limitations:
        lines.append("- None identified by configured checks")
    return "\n".join(lines) + "\n"
