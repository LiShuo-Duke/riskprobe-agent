from collections import Counter
from collections.abc import Sequence
import hashlib
from pathlib import Path, PureWindowsPath
import re
from urllib.parse import unquote, urlsplit

from riskprobe.models import EvidenceCard
from riskprobe.privacy import stable_token
from riskprobe.profiling import DatasetProfile

_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}
_EMBEDDED_POSIX_PATH = re.compile(r"(?:^|[=:\s|;,])/(?:[^/\s]+/)+[^/\s]+")
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
    """Return a deterministic redacted segment code."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"segment-{digest}"


def redact_limitation(
    limitation: str,
    *,
    already_redacted: bool = False,
    expose_segment_values: bool = False,
) -> str:
    """Keep public limitations readable while honoring segment display policy."""
    if already_redacted or expose_segment_values:
        return limitation
    match = re.match(r"^((?:holdout:\s*)?single-class [^:]+): (.+)$", limitation)
    if match:
        return f"{match.group(1)}: {redact_segment_value(match.group(2))}"
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
    institution_analysis: dict[str, object] | None = None,
    *,
    expose_segment_values: bool = True,
    segments_are_redacted: bool = False,
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
        lines.append(
            f"- Snapshot range: {profile.snapshot_min.isoformat()} to "
            f"{profile.snapshot_max.isoformat()}"
        )
    lines.extend(["", "## Quality Issues", ""])
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
        "Test KS (signed)",
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
            _number(card.test.ks_signed),
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

    institution_rows = [
        (card.rule.rule_id, item)
        for card in cards
        for item in card.slices
        if item.slice_type == "segment"
    ]
    if institution_rows:
        institution_headers = [
            "Rule ID",
            "Institution Token",
            *(["Institution Name"] if expose_segment_values else []),
            "Support",
            "Coverage",
            "Hit Bad Rate",
            "Lift",
            "Direction",
        ]
        lines.extend(
            [
                "",
                "## Institution Evidence",
                "",
                f"| {' | '.join(institution_headers)} |",
                f"| {' | '.join(['---', '---'] + (['---'] if expose_segment_values else []) + ['---:'] * 4 + ['---'])} |",
            ]
        )
        for rule_id, item in institution_rows:
            metrics = item.metrics
            token = stable_token(item.slice_value, namespace="institution")
            row = [
                rule_id,
                token,
                *([item.slice_value] if expose_segment_values else []),
                str(metrics.support_count),
                _number(metrics.coverage),
                _number(metrics.hit_bad_rate),
                _number(metrics.lift),
                "positive" if metrics.lift > 1.0 else "non-positive",
            ]
            lines.append(f"| {' | '.join(row)} |")

    if institution_analysis is not None:
        institution_reports = institution_analysis.get("institution_reports", [])
        reports = institution_reports if isinstance(institution_reports, list) else []
        lines.extend(
            [
                "",
                "## Institution Analysis",
                "",
                "- 分析顺序：先合并全机构发现总体规则，再按机构验证；仅对 Local 且样本充足的机构进行局部发现。",
                f"- Eligible institutions: {institution_analysis.get('eligible_institution_count', 0)}",
                f"- Triggered local discovery: {institution_analysis.get('triggered_institution_count', 0)}",
                f"- Blocked local discovery: {institution_analysis.get('blocked_institution_count', 0)}",
                f"- Interpretation: {institution_analysis.get('interpretation', '机构级结果仅用于验证和人工复核。')}",
            ]
        )
        if reports:
            analysis_headers = [
                "Institution Token",
                *(["Institution Name"] if expose_segment_values else []),
                "Status",
                "Train Rows",
                "Test Rows",
                "Local Rules",
                "Reason",
            ]
            lines.extend(
                [
                    "",
                    f"| {' | '.join(analysis_headers)} |",
                    f"| {' | '.join(['---'] * len(analysis_headers))} |",
                ]
            )
            for item in reports:
                if not isinstance(item, dict):
                    continue
                row = [
                    str(item.get("institution_token", "—")),
                    *(
                        [str(item.get("institution_name", "—"))]
                        if expose_segment_values
                        else []
                    ),
                    str(item.get("status", "—")),
                    str(item.get("train_row_count", "—")),
                    str(item.get("test_row_count", "—")),
                    str(item.get("rule_count", "—")),
                    str(item.get("reason", "—")),
                ]
                lines.append(f"| {' | '.join(row)} |")

            top_rows: list[tuple[str, str, dict[str, object]]] = []
            for item in reports:
                if not isinstance(item, dict) or item.get("status") != "completed":
                    continue
                validation = item.get("validation_report", {})
                if not isinstance(validation, dict):
                    continue
                top_rules = validation.get("top_rules", [])
                if not isinstance(top_rules, list):
                    continue
                token = str(item.get("institution_token", "—"))
                name = str(item.get("institution_name", "—"))
                top_rows.extend(
                    (token, name, rule)
                    for rule in top_rules[:5]
                    if isinstance(rule, dict)
                )
            if top_rows:
                top_headers = [
                    "Institution Token",
                    *(["Institution Name"] if expose_segment_values else []),
                    "Rule ID",
                    "Conditions",
                    "Grade",
                    "Test Lift",
                    "Coverage",
                    "Interpretation",
                ]
                lines.extend(
                    [
                        "",
                        "### Institution TOP5",
                        "",
                        f"| {' | '.join(top_headers)} |",
                        f"| {' | '.join(['---'] * len(top_headers))} |",
                    ]
                )
                for token, name, rule in top_rows:
                    conditions = rule.get("conditions", [])
                    condition_text = " AND ".join(
                        f"{condition.get('feature', 'feature')} {condition.get('operator', '?')} {condition.get('value', '—')}"
                        for condition in conditions
                        if isinstance(condition, dict)
                    )
                    test = rule.get("test", {})
                    row = [
                        token,
                        *([name] if expose_segment_values else []),
                        str(rule.get("rule_id", "—")),
                        condition_text or "—",
                        str(rule.get("grade", "—")),
                        _number(test.get("lift") if isinstance(test, dict) else None),
                        _number(test.get("coverage") if isinstance(test, dict) else None),
                        str(rule.get("interpretation", "机构规则仅供人工复核")),
                    ]
                    lines.append(f"| {' | '.join(row)} |")

    limitations = sorted(
        {
            redact_limitation(
                limitation,
                already_redacted=segments_are_redacted,
                expose_segment_values=expose_segment_values,
            )
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
