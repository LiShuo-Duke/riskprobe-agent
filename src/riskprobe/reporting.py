from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from riskprobe.analysis_contracts import AnalysisSummary, MetricDistribution
from riskprobe.models import EvidenceCard
from riskprobe.profiling import DatasetProfile
from riskprobe.report_models import (
    ReportCondition,
    ReportModel,
    ReportRule,
    build_analysis_report_model,
    evidence_sort_key,
    safe_dataset_id,
)


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


def _issue_message(code: str) -> str:
    messages = {
        "LABEL_PERFORMANCE_WINDOW_UNKNOWN": "标签表现窗口未配置",
        "SINGLE_CLASS_SLICE": "检测到单类别切片",
    }
    return messages.get(code, "检测到配置化数据质量问题")


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _integer(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def _cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _condition_text(condition: ReportCondition) -> str:
    if condition.operator == "is_null":
        return f"{condition.feature.value} IS NULL"
    value = (
        repr(condition.numeric_value)
        if condition.numeric_value is not None
        else condition.value_token or "N/A"
    )
    return f"{condition.feature.value} {condition.operator} {value}"


def _rule_text(rule: ReportRule) -> str:
    return " AND ".join(_condition_text(condition) for condition in rule.conditions) or "N/A"


def _append_table(lines: list[str], headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    lines.extend(
        [
            f"| {' | '.join(_cell(item) for item in headers)} |",
            f"| {' | '.join('---' if index == 0 else '---:' for index in range(len(headers)))} |",
        ]
    )
    if rows:
        lines.extend(
            f"| {' | '.join(_cell(item) for item in row)} |"
            for row in rows
        )
    else:
        lines.append(f"| {' | '.join(['None', *['—' for _ in headers[1:]]])} |")


def _distribution_text(distribution: MetricDistribution) -> str:
    if distribution.count == 0:
        return "N/A"
    return (
        f"count={distribution.count}, min={_number(distribution.minimum)}, "
        f"median={_number(distribution.median)}, max={_number(distribution.maximum)}"
    )


def render_report_markdown(model: ReportModel) -> str:
    lines = [
        "# RiskProbe Risk Report",
        "",
        "## RiskProbe 风险分析报告",
        "",
    ]
    _append_cover(lines, model)
    _append_executive_summary(lines, model)
    _append_stage_summary(lines, model)
    _append_profile_partition(lines, model)
    _append_discovery(lines, model)
    _append_top_rules(lines, model)
    _append_scorecard(lines, model)
    _append_institutions(lines, model)
    _append_decision_review(lines, model)
    _append_limitations(lines, model)
    _append_rule_appendix(lines, model)
    return "\n".join(lines) + "\n"


def _append_cover(lines: list[str], model: ReportModel) -> None:
    profile = model.profile
    summary = model.analysis_summary
    metadata_grade = (
        profile.metadata_grade
        if profile is not None
        else summary.profile.metadata_grade
        if summary is not None and summary.profile is not None
        else "N/A"
    )
    dataset_id = (
        profile.dataset_id
        if profile is not None
        else summary.input.dataset_id
        if summary is not None and summary.input is not None
        else "N/A"
    )
    lines.extend(
        [
            "### 执行结论",
            "",
            f"- Dataset: `{dataset_id}`",
            f"- Run ID: `{model.run_id or 'N/A'}`",
            f"- Session ID: `{model.session_id or 'N/A'}`",
            f"- Scope: `{model.scope}`",
            f"- Terminal status: `{model.terminal_status or 'N/A'}`",
            f"- Error code: `{model.error_code or 'N/A'}`",
            f"- **Metadata Grade: {metadata_grade}**",
            (
                "- 时间切片稳定性：已评估"
                if model.time_validation_applied
                else "- 时间切片稳定性：未评估"
            ),
        ]
    )
    if metadata_grade == "B":
        lines.extend(
            [
                "",
                "> 限制：表现窗口未知；Stable 不代表严格 OOT、生产就绪或自动上线。",
            ]
        )
    if model.time_validation_applied:
        lines.append(
            "> 时间切片评估不等于已知表现窗口、严格 OOT 或生产就绪。"
            if metadata_grade == "B"
            else "> 时间切片评估不等于已知表现窗口或生产就绪。"
        )


def _append_executive_summary(lines: list[str], model: ReportModel) -> None:
    summary = model.analysis_summary
    profile = model.profile
    analysis_profile = summary.profile if summary is not None else None
    discovery = summary.discovery if summary is not None else None
    scorecard = summary.scorecard if summary is not None else None
    row_count = profile.row_count if profile is not None else analysis_profile.row_count if analysis_profile is not None else None
    feature_count = profile.feature_count if profile is not None else analysis_profile.feature_count if analysis_profile is not None else None
    positive_rate = profile.positive_rate if profile is not None else analysis_profile.positive_rate if analysis_profile is not None else None
    lines.extend(
        [
            "",
            "## 1. 管理摘要",
            "",
            f"- 样本量: {_integer(row_count)}",
            f"- 特征数: {_integer(feature_count)}",
            f"- 正样本率: {_number(positive_rate)}",
            f"- 候选规则数: {_integer(discovery.candidate_rule_count if discovery is not None else None)}",
            f"- 入选规则数: {len(model.all_rules)}",
            f"- Stable 规则数: {model.grade_counts.get('Stable', 0)}",
            f"- 评分卡状态: `{scorecard.status.value if scorecard is not None else 'N/A'}`",
            f"- 最终决策: `{model.decision_summary.decision_status if model.decision_summary is not None else model.terminal_status or 'N/A'}`",
        ]
    )


def _append_stage_summary(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 2. 全流程阶段摘要", ""])
    summary = model.analysis_summary
    if summary is None:
        lines.append("- 未提供结构化阶段摘要。")
        return
    rows = [
        (
            stage.name.value,
            stage.status.value,
            "是" if stage.enabled else "否",
            "是" if stage.output_available else "否",
            stage.reason_code or "N/A",
            _integer(stage.duration_ms),
            ", ".join(stage.limitations) or "N/A",
        )
        for stage in summary.stages
    ]
    _append_table(
        lines,
        ("Stage", "Status", "Enabled", "Output", "Reason", "Duration ms", "Limitations"),
        rows,
    )


def _append_profile_partition(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 3. 数据画像、质量与切分", "", "**Sample Overview**", ""])
    profile = model.profile
    summary = model.analysis_summary
    analysis_profile = summary.profile if summary is not None else None
    if profile is not None:
        lines.extend(
            [
                f"- Dataset: `{profile.dataset_id}`",
                f"- Rows: {profile.row_count}",
                f"- Features: {profile.feature_count}",
                f"- Positive rate: {_number(profile.positive_rate)}",
                f"- Segment count: {profile.segment_count}",
                f"- Snapshot range: {profile.snapshot_min.isoformat() if profile.snapshot_min else 'N/A'} to {profile.snapshot_max.isoformat() if profile.snapshot_max else 'N/A'}",
                "",
                "### Quality Issues",
                "",
            ]
        )
        if profile.issues:
            lines.extend(
                f"- [{issue.severity}] {issue.code}: {_issue_message(issue.code)} (affected rows: {issue.affected_rows})"
                for issue in profile.issues
            )
        else:
            lines.append("- None")
    elif analysis_profile is not None:
        lines.extend(
            [
                f"- Rows: {analysis_profile.row_count}",
                f"- Features: {analysis_profile.feature_count}",
                f"- Numeric features: {analysis_profile.numeric_feature_count}",
                f"- Positive rate: {_number(analysis_profile.positive_rate)}",
                f"- Segment count: {analysis_profile.segment_count}",
                f"- Issue codes: {', '.join(analysis_profile.issue_codes) or 'None'}",
            ]
        )
    else:
        lines.append("- 未提供数据画像。")
    partition = summary.partition if summary is not None else None
    lines.extend(["", "### 数据切分", ""])
    if partition is None:
        lines.append("- 未提供切分摘要。")
        return
    _append_table(
        lines,
        ("Strategy", "Mode", "Time Requested", "Time Applied", "Train", "Test", "Holdout", "Excluded Null", "Fallback"),
        (
            (
                partition.strategy,
                partition.mode,
                partition.requested_time_validation,
                partition.applied_time_validation,
                partition.train_rows,
                partition.test_rows,
                partition.holdout_rows,
                partition.excluded_null_snapshot_rows,
                partition.fallback_reason_code or "N/A",
            ),
        ),
    )


def _append_discovery(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 4. 规则发现与筛选", "", "**Evidence Summary**", ""])
    _append_table(
        lines,
        ("Stable", "Local", "Unstable", "Suspicious"),
        (
            (
                model.grade_counts.get("Stable", 0),
                model.grade_counts.get("Local", 0),
                model.grade_counts.get("Unstable", 0),
                model.grade_counts.get("Suspicious", 0),
            ),
        ),
    )
    summary = model.analysis_summary
    discovery = summary.discovery if summary is not None else None
    if discovery is None:
        lines.extend(["", f"- 入选规则: {len(model.all_rules)}"])
        return
    lines.extend(
        [
            "",
            f"- Input features: {discovery.input_feature_count}",
            f"- Eligible features: {discovery.eligible_feature_count}",
            f"- Candidate rules: {discovery.candidate_rule_count}",
            f"- Selected rules: {discovery.selected_rule_count}",
            f"- Single candidates/selected: {discovery.single_candidate_count}/{discovery.single_rule_count}",
            f"- Pair candidates/selected: {discovery.pair_candidate_count}/{discovery.pair_rule_count}",
            f"- Lift: {_distribution_text(discovery.lift)}",
            f"- Coverage: {_distribution_text(discovery.support)}",
            f"- Precision: {_distribution_text(discovery.precision)}",
        ]
    )
    if discovery.skipped_feature_reason_counts:
        lines.append(
            "- Skipped features: "
            + ", ".join(
                f"{code}={count}"
                for code, count in discovery.skipped_feature_reason_counts.items()
            )
        )


def _append_top_rules(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 5. TOP10 规则", "", "**Top Rule Evidence**", ""])
    headers = (
        "Rank",
        "Rule ID",
        "规则条件",
        "Origin",
        "Grade",
        "Support",
        "Test Lift",
        "Holdout Lift",
        "Coverage",
        "Hit Bad Rate",
        "signed KS",
        "Adjusted p-value",
        "Lift CI",
        "Segment Consistency",
    )
    rows = tuple(
        (
            rule.rank,
            rule.rule_id,
            _rule_text(rule),
            rule.origin,
            rule.grade,
            rule.test.support_count,
            _number(rule.test.lift),
            _number(rule.holdout.lift if rule.holdout is not None else None),
            _number(rule.test.coverage),
            _number(rule.test.hit_bad_rate),
            _number(rule.test.signed_ks),
            _number(rule.adjusted_p_value),
            f"{_number(rule.lift_ci[0])}–{_number(rule.lift_ci[1])}",
            _number(rule.segment_consistency),
        )
        + ((_number(rule.time_decay),) if model.time_validation_applied else ())
        for rule in model.top_rules
    )
    _append_table(
        lines,
        headers + (("Time Decay",) if model.time_validation_applied else ()),
        rows,
    )
    for rule in model.top_rules:
        details = [
            "",
            f"### #{rule.rank} `{rule.rule_id}`",
            "",
            f"- 条件: `{_rule_text(rule)}`",
            f"- Grade / Origin: `{rule.grade}` / `{rule.origin}`",
            f"- Train Lift / Test Lift / Holdout Lift: {_number(rule.train.lift)} / {_number(rule.test.lift)} / {_number(rule.holdout.lift if rule.holdout is not None else None)}",
        ]
        if model.time_validation_applied:
            details.extend(
                [
                    f"- Segment slices / Time slices: {rule.segment_slice_count} / {rule.time_slice_count}",
                    f"- Segment Consistency / Time Decay: {_number(rule.segment_consistency)} / {_number(rule.time_decay)}",
                ]
            )
        else:
            details.extend(
                [
                    f"- Segment slices: {rule.segment_slice_count}",
                    f"- Segment Consistency: {_number(rule.segment_consistency)}",
                ]
            )
        details.append(f"- Limitations: {', '.join(rule.limitations) or 'None'}")
        lines.extend(details)


def _append_scorecard(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 6. 评分卡", "", "**WOE Binning and Scorecard**", ""])
    summary = model.analysis_summary
    scorecard = summary.scorecard if summary is not None else None
    if scorecard is None:
        lines.append("- 未提供评分卡摘要。")
        return
    lines.extend(
        [
            f"- Enabled: {scorecard.enabled}",
            f"- Status: `{scorecard.status.value}`",
            f"- Reason: `{scorecard.reason_code or 'N/A'}`",
            f"- Model: `{scorecard.model_type or 'N/A'}`",
            f"- Calibrated: {scorecard.calibrated if scorecard.calibrated is not None else 'N/A'}",
            f"- Imbalance strategy: `{scorecard.imbalance_strategy or 'N/A'}`",
            f"- Intercept: {_number(scorecard.intercept)}",
        ]
    )
    _append_table(
        lines,
        ("Split", "Status", "Samples", "Positive Rate", "AUC", "KS", "Gini", "Mean Score", "Reason"),
        tuple(
            (
                split.split,
                split.status.value,
                split.sample_count,
                _number(split.positive_rate),
                _number(split.auc),
                _number(split.ks),
                _number(split.gini),
                _number(split.mean_score),
                split.reason_code or "N/A",
            )
            for split in scorecard.splits
        ),
    )
    lines.extend(["", "### WOE / IV 特征", ""])
    _append_table(
        lines,
        ("Feature", "IV", "Bins", "Monotonic", "Missing Bin"),
        tuple(
            (
                item.feature.value,
                _number(item.iv),
                item.bin_count,
                item.monotonic,
                item.has_missing_bin,
            )
            for item in scorecard.feature_summaries
        ),
    )
    lines.extend(["", "### 模型系数", ""])
    _append_table(
        lines,
        ("Term", "Kind", "Coefficient"),
        tuple(
            (term.term.value, term.kind, _number(term.coefficient))
            for term in scorecard.terms
        ),
    )


def _append_institutions(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 7. 机构与稳定性分析", "", "**Institution Evidence**", ""])
    _append_table(
        lines,
        ("Rule ID", "Institution Token", "Support", "Coverage", "Hit Bad Rate", "Lift", "Direction"),
        tuple(
            (
                row.rule_id,
                row.institution_token,
                row.support_count,
                _number(row.coverage),
                _number(row.hit_bad_rate),
                _number(row.lift),
                row.direction,
            )
            for row in model.institution_evidence
        ),
    )
    lines.extend(["", "**Institution Analysis**", ""])
    if model.institution_summary is None:
        lines.append("- 未提供机构分析摘要。")
        return
    lines.extend(
        [
            f"- Eligible institutions: {model.institution_summary.eligible_count}",
            f"- Triggered local discovery: {model.institution_summary.triggered_count}",
            f"- Blocked local discovery: {model.institution_summary.blocked_count}",
            f"- Interpretation: {model.institution_summary.interpretation}",
        ]
    )


def _append_decision_review(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 8. 诊断、建议与审核", ""])
    summary = model.analysis_summary
    diagnostics = summary.diagnostics if summary is not None else None
    if diagnostics is not None:
        lines.extend(
            [
                f"- Diagnostic time enabled: {diagnostics.diagnostic_time_enabled}",
                "- Finding counts by kind: " + (", ".join(f"{key}={value}" for key, value in diagnostics.finding_counts_by_kind.items()) or "None"),
                "- Finding counts by severity: " + (", ".join(f"{key}={value}" for key, value in diagnostics.finding_counts_by_severity.items()) or "None"),
            ]
        )
    lines.extend(
        [
            f"- Proposal actions: {', '.join(model.proposal_action_codes) or 'None'}",
            f"- Host decision reasons: {', '.join(model.decision_reason_codes) or 'None'}",
            f"- Diagnosis evidence IDs: {', '.join(model.diagnosis_evidence_ids) or 'None'}",
            f"- Agent tool sequence: {' → '.join(model.agent_tool_sequence) or 'None'}",
            f"- Agent state history: {' → '.join(model.agent_state_history) or 'None'}",
            f"- Agent summary: {model.agent_summary or 'N/A'}",
        ]
    )
    if model.findings:
        lines.extend(["", "### Diagnosis Findings", ""])
        _append_table(
            lines,
            ("Evidence ID", "Kind", "Severity", "Summary", "Limitations"),
            tuple(
                (
                    finding.evidence_id,
                    finding.kind,
                    finding.severity,
                    finding.summary,
                    ", ".join(finding.limitations) or "None",
                )
                for finding in model.findings
            ),
        )
    decision = model.decision_summary
    if decision is None:
        lines.append(
            "- 无需 Host proposal/无可执行动作。"
            if model.terminal_status == "no_action"
            else "- Host 决策尚未执行。"
        )
    else:
        lines.extend(
            [
                f"- Decision status: `{decision.decision_status}`",
                f"- Final status: `{decision.final_status}`",
                f"- Selected actions: {', '.join(decision.selected_action_codes) or 'None'}",
                f"- Recommendation status: `{decision.recommendation_status.value}`",
                f"- Review approved: {decision.review_approved}",
                f"- Review reasons: {', '.join(decision.review_reason_codes) or 'None'}",
                f"- Evidence complete: {decision.evidence_complete}",
                f"- Retry count: {decision.retry_count}",
                f"- No action required: {decision.no_action_required}",
                f"- Tool sequence: {' → '.join(decision.tool_sequence)}",
            ]
        )
        if decision.recommendations:
            lines.extend(["", "### Recommendations", ""])
            _append_table(
                lines,
                ("Action", "Evidence ID", "Parent Findings", "Human Approval", "Analysis Only", "Limitations"),
                tuple(
                    (
                        item.action_code,
                        item.evidence_id,
                        ", ".join(item.parent_finding_ids),
                        item.human_approval_required,
                        item.analysis_only,
                        ", ".join(item.limitations) or "None",
                    )
                    for item in decision.recommendations
                ),
            )
    if model.review is not None:
        lines.extend(
            [
                "",
                "### Review",
                "",
                f"- Approved: {model.review.approved}",
                f"- Reasons: {', '.join(model.review.reason_codes) or 'None'}",
                f"- Evidence IDs: {', '.join(model.review.evidence_ids) or 'None'}",
                f"- Retry allowed: {model.review.retry_allowed}",
                f"- No action required: {model.review.no_action_required}",
            ]
        )


def _append_limitations(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 9. 限制", "", "**Limitations**", ""])
    limitations = set(model.limitations)
    limitations.update(
        limitation
        for rule in model.top_rules
        for limitation in rule.limitations
    )
    limitations.add(
        "时间切片稳定性：已评估"
        if model.time_validation_applied
        else "时间切片稳定性：未评估"
    )
    metadata_grade = (
        model.profile.metadata_grade
        if model.profile is not None
        else model.analysis_summary.profile.metadata_grade
        if model.analysis_summary is not None
        and model.analysis_summary.profile is not None
        else None
    )
    if metadata_grade == "B":
        limitations.add("label performance window unknown")
    if model.time_validation_applied:
        limitations.add(
            "时间切片评估不等于已知表现窗口、严格 OOT 或生产就绪"
            if metadata_grade == "B"
            else "时间切片评估不等于已知表现窗口或生产就绪"
        )
    lines.extend(f"- {item}" for item in sorted(limitations))
    definitions = [
        "- Lift：命中样本坏样本率相对总体坏样本率的倍数。",
        "- signed KS：命中坏样本率减命中好样本率，保留风险方向。",
        "- Adjusted p-value：多重检验校正后的显著性指标。",
        "- Segment Consistency：规则跨机构/分群方向一致性。",
    ]
    if model.time_validation_applied:
        definitions.append("- Time Decay：规则跨时间窗口的最大 Lift 衰减。")
    lines.extend(["", "### 指标定义", "", *definitions])


def _append_rule_appendix(lines: list[str], model: ReportModel) -> None:
    lines.extend(["", "## 10. 附录：全部入选规则简表", ""])
    _append_table(
        lines,
        ("Rule ID", "规则条件", "Origin", "Grade", "Test Lift", "Holdout Lift", "Coverage", "signed KS"),
        tuple(
            (
                rule.rule_id,
                _rule_text(rule),
                rule.origin,
                rule.grade,
                _number(rule.test.lift),
                _number(rule.holdout.lift if rule.holdout is not None else None),
                _number(rule.test.coverage),
                _number(rule.test.signed_ks),
            )
            for rule in model.all_rules
        ),
    )


def render_risk_report(
    profile: DatasetProfile,
    evidence_cards: Sequence[EvidenceCard],
    institution_analysis: dict[str, object] | None = None,
    *,
    expose_segment_values: bool = False,
    segments_are_redacted: bool = False,
    time_validation_applied: bool | None = None,
    run_limitations: Sequence[str] = (),
    analysis_summary: AnalysisSummary | None = None,
    run_id: str | None = None,
) -> str:
    del expose_segment_values
    model = build_analysis_report_model(
        evidence_cards=evidence_cards,
        profile=profile,
        analysis_summary=analysis_summary,
        institution_analysis=institution_analysis,
        time_validation_applied=time_validation_applied,
        segments_are_redacted=segments_are_redacted,
        run_limitations=run_limitations,
        run_id=run_id,
    )
    return render_report_markdown(model)


__all__ = [
    "evidence_sort_key",
    "redact_limitation",
    "redact_segment_value",
    "render_report_markdown",
    "render_risk_report",
    "safe_dataset_id",
]
