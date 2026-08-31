from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, field_validator, model_validator

from riskprobe.analysis_contracts import (
    AnalysisSummary,
    DecisionSummary,
    FeatureRef,
    StageName,
    StageStatus,
    StageSummary,
)
from riskprobe.models import (
    Condition,
    EvidenceCard,
    EvidenceGrade,
    FrozenModel,
    Operator,
    RuleMetrics,
)
from riskprobe.privacy import assert_safe_payload, stable_token
from riskprobe.profiling import DatasetProfile

if TYPE_CHECKING:
    from riskprobe.agents.decision_contracts import DecisionFinding
    from riskprobe.terminal_reports import TerminalReportSubject

_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}
_EMBEDDED_POSIX_PATH = re.compile(r"(?:^|[=:\s|;,])/(?:[^/\s]+/)+[^/\s]+")
_EMBEDDED_WINDOWS_PATH = re.compile(
    r"(?:^|[=:\s|;,])(?:[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])+[^\\/\s]+|\\\\[^\\/\s]+[\\/][^\s]+)"
)
_SAFE_FILE_DATASET_ID = re.compile(
    r"^file-(?!id(?:[-_]|$))[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*$"
)
_REDACTED_SEGMENT = re.compile(r"^segment-[0-9a-f]{8}$")


class ReportCondition(FrozenModel):
    feature: FeatureRef
    operator: Operator
    numeric_value: int | float | None = None
    value_token: str | None = None

    @model_validator(mode="after")
    def validate_value_shape(self) -> ReportCondition:
        has_number = self.numeric_value is not None
        has_token = self.value_token is not None
        if self.operator == "is_null":
            if has_number or has_token:
                raise ValueError("is_null must not carry a value")
            return self
        if has_number == has_token:
            raise ValueError("condition must carry exactly one safe value")
        return self


class ReportRuleMetrics(FrozenModel):
    split: Literal["Train", "Test", "Holdout"]
    support_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    hit_bad_rate: float = Field(ge=0.0, le=1.0)
    lift: float
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    signed_ks: float | None = Field(default=None, ge=-1.0, le=1.0)


class ReportRule(FrozenModel):
    rank: int = Field(ge=1)
    rule_id: str
    origin: str
    conditions: tuple[ReportCondition, ...]
    grade: EvidenceGrade
    train: ReportRuleMetrics
    test: ReportRuleMetrics
    holdout: ReportRuleMetrics | None = None
    lift_ci: tuple[float, float]
    adjusted_p_value: float = Field(ge=0.0, le=1.0)
    segment_consistency: float = Field(ge=0.0, le=1.0)
    time_decay: float | None = Field(default=None, ge=0.0)
    segment_slice_count: int = Field(default=0, ge=0)
    time_slice_count: int = Field(default=0, ge=0)
    limitations: tuple[str, ...] = ()

    @field_validator("rule_id", "origin")
    @classmethod
    def validate_codes(cls, value: str) -> str:
        if not value:
            raise ValueError("rule identity must be non-empty")
        assert_safe_payload({"value": value})
        return value

    @model_validator(mode="after")
    def validate_metrics(self) -> ReportRule:
        if self.train.split != "Train" or self.test.split != "Test":
            raise ValueError("rule train/test metrics are mislabelled")
        if self.holdout is not None and self.holdout.split != "Holdout":
            raise ValueError("rule holdout metrics are mislabelled")
        if self.lift_ci[0] > self.lift_ci[1]:
            raise ValueError("lift interval is invalid")
        return self


class ReportIssue(FrozenModel):
    severity: str
    code: str
    affected_rows: int = Field(ge=0)


class ReportProfile(FrozenModel):
    dataset_id: str
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    segment_count: int = Field(ge=0)
    snapshot_min: date | None = None
    snapshot_max: date | None = None
    metadata_grade: Literal["A", "B"]
    issues: tuple[ReportIssue, ...] = ()


class ReportInstitutionEvidence(FrozenModel):
    rule_id: str
    institution_token: str
    support_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    hit_bad_rate: float = Field(ge=0.0, le=1.0)
    lift: float
    direction: Literal["positive", "non-positive"]


class ReportInstitutionSummary(FrozenModel):
    eligible_count: int = Field(default=0, ge=0)
    triggered_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    interpretation: str = "机构级结果仅用于稳定性验证和人工复核"


class ReportFinding(FrozenModel):
    evidence_id: str
    kind: str
    severity: str
    summary: str
    limitations: tuple[str, ...] = ()


class ReportReview(FrozenModel):
    approved: bool
    reason_codes: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    retry_allowed: bool
    no_action_required: bool


class ReportModel(FrozenModel):
    schema_version: Literal["riskprobe.final-report.v1"] = "riskprobe.final-report.v1"
    scope: Literal["analysis", "terminal"]
    title: str = "RiskProbe 风险分析报告"
    run_id: str | None = None
    session_id: str | None = None
    terminal_status: Literal["accepted", "rejected", "no_action", "failed"] | None = None
    error_code: str | None = None
    profile: ReportProfile | None = None
    analysis_summary: AnalysisSummary | None = None
    time_validation_applied: bool = False
    top_rules: tuple[ReportRule, ...] = ()
    all_rules: tuple[ReportRule, ...] = ()
    grade_counts: Mapping[str, int] = Field(default_factory=dict)
    institution_evidence: tuple[ReportInstitutionEvidence, ...] = ()
    institution_summary: ReportInstitutionSummary | None = None
    findings: tuple[ReportFinding, ...] = ()
    proposal_action_codes: tuple[str, ...] = ()
    decision_reason_codes: tuple[str, ...] = ()
    diagnosis_evidence_ids: tuple[str, ...] = ()
    agent_tool_sequence: tuple[str, ...] = ()
    agent_state_history: tuple[str, ...] = ()
    agent_summary: str | None = None
    decision_summary: DecisionSummary | None = None
    review: ReportReview | None = None
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> ReportModel:
        if len(self.top_rules) > 10:
            raise ValueError("top rule count exceeds ten")
        if tuple(rule.rank for rule in self.top_rules) != tuple(
            range(1, len(self.top_rules) + 1)
        ):
            raise ValueError("top rule ranks are not contiguous")
        if self.scope == "terminal":
            if self.terminal_status is None:
                raise ValueError("terminal report requires terminal status")
            if (self.terminal_status == "failed") != (self.error_code is not None):
                raise ValueError("terminal failure status and error code disagree")
            if (
                self.decision_summary is not None
                and self.terminal_status != "failed"
                and self.decision_summary.decision_status != self.terminal_status
            ):
                raise ValueError("terminal status and decision summary disagree")
        if len(self.proposal_action_codes) != len(set(self.proposal_action_codes)):
            raise ValueError("proposal action codes must be unique")
        if len(self.decision_reason_codes) != len(set(self.decision_reason_codes)):
            raise ValueError("decision reason codes must be unique")
        if len(self.diagnosis_evidence_ids) != len(set(self.diagnosis_evidence_ids)):
            raise ValueError("diagnosis evidence IDs must be unique")
        _assert_report_safe(self)
        return self


def safe_dataset_id(dataset_id: str) -> str:
    decoded = unquote(dataset_id)
    parsed = urlsplit(decoded)
    is_path = (
        parsed.scheme.lower() == "file"
        or Path(decoded).is_absolute()
        or PureWindowsPath(decoded).is_absolute()
        or _EMBEDDED_POSIX_PATH.search(decoded) is not None
        or _EMBEDDED_WINDOWS_PATH.search(decoded) is not None
    )
    if not is_path:
        return dataset_id
    digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8]
    return f"dataset-{digest}"


def _assert_safe_dataset_id(dataset_id: str) -> None:
    if safe_dataset_id(dataset_id) != dataset_id:
        raise ValueError("report dataset ID is unsafe")
    try:
        assert_safe_payload({"value": dataset_id})
    except ValueError:
        if _SAFE_FILE_DATASET_ID.fullmatch(dataset_id) is None:
            raise


def evidence_sort_key(card: EvidenceCard) -> tuple[int, float, str]:
    return _GRADE_ORDER[card.grade], -card.test.lift, card.rule.rule_id


def build_analysis_report_model(
    *,
    evidence_cards: Sequence[EvidenceCard],
    profile: DatasetProfile | None = None,
    analysis_summary: AnalysisSummary | None = None,
    institution_analysis: Mapping[str, object] | None = None,
    time_validation_applied: bool | None = None,
    segments_are_redacted: bool = False,
    run_limitations: Sequence[str] = (),
    run_id: str | None = None,
) -> ReportModel:
    cards = tuple(sorted(evidence_cards, key=evidence_sort_key))
    effective_time_validation = (
        time_validation_applied
        if time_validation_applied is not None
        else bool(
            analysis_summary is not None
            and analysis_summary.partition is not None
            and analysis_summary.partition.applied_time_validation
        )
    )
    segment_tokens = _segment_tokens(
        cards,
        already_redacted=segments_are_redacted,
    )
    projected = tuple(
        _project_card(
            rank=index,
            card=card,
            time_validation_applied=effective_time_validation,
            segment_tokens=segment_tokens,
        )
        for index, card in enumerate(cards, start=1)
    )
    top_rules = tuple(
        rule.model_copy(update={"rank": index})
        for index, rule in enumerate(projected[:10], start=1)
    )
    limitations = tuple(
        sorted(
            {
                *(
                    _project_limitation(
                        item,
                        segment_tokens=segment_tokens,
                        namespace="report-limitation",
                    )
                    for item in run_limitations
                ),
                *(
                    _project_limitation(
                        item,
                        segment_tokens=segment_tokens,
                        namespace="rule-limitation",
                    )
                    for card in cards
                    for item in card.limitations
                ),
            }
        )
    )
    return ReportModel(
        scope="analysis",
        run_id=run_id,
        profile=_project_profile(profile) if profile is not None else None,
        analysis_summary=analysis_summary,
        time_validation_applied=effective_time_validation,
        top_rules=top_rules,
        all_rules=projected,
        grade_counts=dict(sorted(Counter(card.grade for card in cards).items())),
        institution_evidence=_project_institution_evidence(
            cards,
            segment_tokens=segment_tokens,
        ),
        institution_summary=_project_institution_summary(institution_analysis),
        limitations=limitations,
    )


def build_final_report_model(
    *,
    subject: TerminalReportSubject,
    evidence_cards: Sequence[EvidenceCard],
    artifact_analysis_summary: AnalysisSummary | None,
    institution_analysis: Mapping[str, object] | None = None,
    time_validation_applied: bool | None = None,
    run_limitations: Sequence[str] = (),
) -> ReportModel:
    """Build the validated terminal projection from verified aggregate inputs."""

    analysis_summary = _terminal_analysis_summary(
        subject=subject,
        artifact_analysis_summary=artifact_analysis_summary,
    )
    base = build_analysis_report_model(
        evidence_cards=evidence_cards,
        analysis_summary=analysis_summary,
        institution_analysis=institution_analysis,
        time_validation_applied=time_validation_applied,
        segments_are_redacted=True,
        run_limitations=run_limitations,
        run_id=subject.run_id,
    )
    agent_result = subject.agent_result
    review = (
        None
        if agent_result is None
        else ReportReview(
            approved=agent_result.review.approved,
            reason_codes=tuple(reason.value for reason in agent_result.review.reason_codes),
            evidence_ids=agent_result.review.evidence_ids,
            retry_allowed=agent_result.review.retry_allowed,
            no_action_required=agent_result.review.no_action_required,
        )
    )
    payload = {
        name: getattr(base, name)
        for name in ReportModel.model_fields
    }
    payload.update(
        {
            "scope": "terminal",
            "session_id": agent_result.session_id if agent_result is not None else None,
            "terminal_status": subject.terminal_status,
            "error_code": subject.error_code,
            "analysis_summary": analysis_summary,
            "findings": tuple(_project_finding(item) for item in subject.findings),
            "proposal_action_codes": tuple(str(item) for item in subject.proposal_action_codes),
            "decision_reason_codes": subject.decision_reason_codes,
            "diagnosis_evidence_ids": subject.diagnosis_evidence_ids,
            "agent_tool_sequence": agent_result.tool_sequence if agent_result is not None else (),
            "agent_state_history": (
                tuple(state.value for state in agent_result.state_history)
                if agent_result is not None
                else ()
            ),
            "agent_summary": (
                agent_result.redacted_summary if agent_result is not None else None
            ),
            "decision_summary": subject.decision_summary,
            "review": review,
        }
    )
    return ReportModel.model_validate(payload)


def _terminal_analysis_summary(
    *,
    subject: TerminalReportSubject,
    artifact_analysis_summary: AnalysisSummary | None,
) -> AnalysisSummary:
    if subject.terminal_status == "no_action":
        if subject.agent_result is None:
            raise ValueError("no-action report requires an agent result")
        if subject.analysis_summary is None:
            raise ValueError("no-action report requires a terminal analysis summary")
        if subject.decision_summary is None:
            raise ValueError("no-action report requires a decision summary")
        return subject.analysis_summary
    summary = subject.analysis_summary or artifact_analysis_summary
    if summary is None:
        raise ValueError("terminal report requires an analysis summary")
    if subject.terminal_status != "failed":
        return summary
    stages = tuple(
        StageSummary(
            name=stage.name,
            status=StageStatus.FAILED,
            enabled=True,
            output_available=False,
            reason_code=subject.error_code,
            duration_ms=stage.duration_ms,
            limitations=stage.limitations,
        )
        if stage.name is StageName.TERMINAL
        else stage
        for stage in summary.stages
    )
    payload = {
        name: getattr(summary, name)
        for name in AnalysisSummary.model_fields
    }
    payload["stages"] = stages
    return AnalysisSummary.model_validate(payload)


def _project_finding(item: DecisionFinding) -> ReportFinding:
    evidence_id = item.evidence_id
    finding = item.finding
    parts = [f"code={finding.code}"]
    if finding.feature is not None:
        parts.append(f"feature={FeatureRef.from_name(finding.feature).value}")
    if finding.period is not None:
        parts.append(f"period={finding.period}")
    if finding.segment_token is not None:
        parts.append(f"segment={finding.segment_token.token}")
    if finding.metrics:
        parts.append(
            "metrics="
            + ",".join(
                f"{name}={value!r}"
                for name, value in sorted(finding.metrics.items())
            )
        )
    return ReportFinding(
        evidence_id=evidence_id,
        kind=finding.kind.value,
        severity=finding.severity.value,
        summary=_safe_text(
            "; ".join(parts),
            namespace="finding-summary",
        ),
        limitations=finding.limitations,
    )


def _project_profile(profile: DatasetProfile) -> ReportProfile:
    return ReportProfile(
        dataset_id=safe_dataset_id(profile.dataset_id),
        row_count=profile.row_count,
        feature_count=profile.feature_count,
        positive_rate=profile.positive_rate,
        segment_count=len(profile.segment_counts),
        snapshot_min=profile.snapshot_min,
        snapshot_max=profile.snapshot_max,
        metadata_grade=profile.metadata_grade,
        issues=tuple(
            ReportIssue(
                severity=_safe_text(issue.severity, namespace="issue-severity"),
                code=_safe_text(issue.code, namespace="issue-code"),
                affected_rows=issue.affected_rows,
            )
            for issue in sorted(
                profile.issues,
                key=lambda item: (item.severity, item.code, item.affected_rows),
            )
        ),
    )


def _project_card(
    *,
    rank: int,
    card: EvidenceCard,
    time_validation_applied: bool,
    segment_tokens: Mapping[str, str],
) -> ReportRule:
    holdout_slice = next(
        (
            item
            for item in card.slices
            if item.slice_type == "dataset" and item.slice_value == "Holdout"
        ),
        None,
    )
    conditions = tuple(
        sorted(
            (_project_condition(condition) for condition in card.rule.conditions),
            key=_condition_sort_key,
        )
    )
    return ReportRule(
        rank=rank,
        rule_id=card.rule.rule_id,
        origin=_safe_text(card.rule.origin, namespace="rule-origin"),
        conditions=conditions,
        grade=card.grade,
        train=_project_metrics("Train", card.train),
        test=_project_metrics("Test", card.test),
        holdout=(
            None
            if holdout_slice is None
            else _project_metrics("Holdout", holdout_slice.metrics)
        ),
        lift_ci=card.lift_ci,
        adjusted_p_value=card.adjusted_p_value,
        segment_consistency=card.segment_consistency,
        time_decay=(None if not time_validation_applied else card.max_time_decay),
        segment_slice_count=sum(item.slice_type == "segment" for item in card.slices),
        time_slice_count=sum(item.slice_type == "time" for item in card.slices),
        limitations=tuple(
            sorted(
                _project_limitation(
                    item,
                    segment_tokens=segment_tokens,
                    namespace="rule-limitation",
                )
                for item in card.limitations
            )
        ),
    )


def _project_condition(condition: Condition) -> ReportCondition:
    feature = FeatureRef.from_name(condition.feature)
    if condition.operator == "is_null":
        if condition.value is not None:
            raise ValueError("is_null condition contains a value")
        return ReportCondition(feature=feature, operator=condition.operator)
    if isinstance(condition.value, bool) or condition.value is None:
        raise ValueError("condition contains an unsupported value")
    if isinstance(condition.value, (int, float)):
        return ReportCondition(
            feature=feature,
            operator=condition.operator,
            numeric_value=condition.value,
        )
    return ReportCondition(
        feature=feature,
        operator=condition.operator,
        value_token=stable_token(
            condition.value,
            namespace=f"rule-condition:{feature.value}",
        ),
    )


def _condition_sort_key(condition: ReportCondition) -> tuple[str, str, str]:
    value = (
        repr(condition.numeric_value)
        if condition.numeric_value is not None
        else condition.value_token or ""
    )
    return condition.feature.value, condition.operator, value


def _project_metrics(
    split: Literal["Train", "Test", "Holdout"],
    metrics: RuleMetrics,
) -> ReportRuleMetrics:
    return ReportRuleMetrics(
        split=split,
        support_count=metrics.support_count,
        coverage=metrics.coverage,
        hit_bad_rate=metrics.hit_bad_rate,
        lift=metrics.lift,
        precision=metrics.precision,
        recall=metrics.recall,
        signed_ks=metrics.ks_signed,
    )


def _segment_tokens(
    cards: Sequence[EvidenceCard],
    *,
    already_redacted: bool,
) -> dict[str, str]:
    values = {
        item.slice_value
        for card in cards
        for item in card.slices
        if item.slice_type == "segment"
    }
    if already_redacted:
        if any(_REDACTED_SEGMENT.fullmatch(value) is None for value in values):
            raise ValueError("redacted segment value is invalid")
        return {value: value for value in values}
    return {
        value: stable_token(value, namespace="institution")
        for value in values
    }


def _project_limitation(
    value: object,
    *,
    segment_tokens: Mapping[str, str],
    namespace: str,
) -> str:
    text = str(value)
    for segment, token in sorted(
        segment_tokens.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        text = text.replace(segment, token)
    return _safe_text(text, namespace=namespace)


def _project_institution_evidence(
    cards: Sequence[EvidenceCard],
    *,
    segment_tokens: Mapping[str, str],
) -> tuple[ReportInstitutionEvidence, ...]:
    rows = [
        ReportInstitutionEvidence(
            rule_id=card.rule.rule_id,
            institution_token=segment_tokens[item.slice_value],
            support_count=item.metrics.support_count,
            coverage=item.metrics.coverage,
            hit_bad_rate=item.metrics.hit_bad_rate,
            lift=item.metrics.lift,
            direction="positive" if item.metrics.lift > 1.0 else "non-positive",
        )
        for card in cards
        for item in card.slices
        if item.slice_type == "segment"
    ]
    return tuple(
        sorted(rows, key=lambda item: (item.rule_id, item.institution_token))
    )


def _project_institution_summary(
    analysis: Mapping[str, object] | None,
) -> ReportInstitutionSummary | None:
    if analysis is None:
        return None
    return ReportInstitutionSummary(
        eligible_count=_non_negative_int(analysis.get("eligible_institution_count")),
        triggered_count=_non_negative_int(analysis.get("triggered_institution_count")),
        blocked_count=_non_negative_int(analysis.get("blocked_institution_count")),
    )


def _non_negative_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _safe_text(value: object, *, namespace: str) -> str:
    text = str(value).strip()
    if not text:
        return stable_token("empty", namespace=namespace)
    try:
        assert_safe_payload({"value": text})
    except ValueError:
        return stable_token(text, namespace=namespace)
    return text


def _assert_report_safe(model: ReportModel) -> None:
    assert_safe_payload(
        {
            "schema_version": model.schema_version,
            "scope": model.scope,
            "title": model.title,
            "run_id": model.run_id,
            "session_id": model.session_id,
            "terminal_status": model.terminal_status,
            "error_code": model.error_code,
            "time_validation_applied": model.time_validation_applied,
            "grade_counts": model.grade_counts,
            "proposal_action_codes": model.proposal_action_codes,
            "decision_reason_codes": model.decision_reason_codes,
            "diagnosis_evidence_ids": model.diagnosis_evidence_ids,
            "agent_tool_sequence": model.agent_tool_sequence,
            "agent_state_history": model.agent_state_history,
            "agent_summary": model.agent_summary,
            "limitations": model.limitations,
        }
    )
    if model.profile is not None:
        profile = model.profile.model_dump(mode="json")
        issues = profile.pop("issues")
        dataset_id = profile.pop("dataset_id")
        _assert_safe_dataset_id(dataset_id)
        assert_safe_payload(profile)
        for issue in issues:
            assert_safe_payload(issue)
    if model.analysis_summary is not None:
        analysis_summary = model.analysis_summary.model_dump(mode="json")
        input_summary = analysis_summary.get("input")
        if isinstance(input_summary, dict):
            dataset_id = input_summary.pop("dataset_id", None)
            if dataset_id is not None:
                _assert_safe_dataset_id(dataset_id)
        assert_safe_payload(analysis_summary)
    for rule in (*model.top_rules, *model.all_rules):
        payload = rule.model_dump(mode="json")
        conditions = payload.pop("conditions")
        assert_safe_payload(payload)
        for condition in conditions:
            assert_safe_payload(condition)
    for row in model.institution_evidence:
        assert_safe_payload(row)
    if model.institution_summary is not None:
        assert_safe_payload(model.institution_summary)
    for finding in model.findings:
        assert_safe_payload(finding)
    if model.decision_summary is not None:
        assert_safe_payload(model.decision_summary)
    if model.review is not None:
        assert_safe_payload(model.review)


__all__ = [
    "ReportCondition",
    "ReportFinding",
    "ReportInstitutionEvidence",
    "ReportInstitutionSummary",
    "ReportModel",
    "ReportProfile",
    "ReportReview",
    "ReportRule",
    "ReportRuleMetrics",
    "build_analysis_report_model",
    "build_final_report_model",
    "evidence_sort_key",
    "safe_dataset_id",
]
