from collections.abc import Sequence
from typing import Any

from riskprobe.config import ValidationConfig
from riskprobe.models import EvidenceCard, RiskRule, RuleMetrics
from riskprobe.privacy import stable_token
from riskprobe.rules.discovery import DiscoveryResult

_GRADE_NAMES = ("Stable", "Local", "Unstable", "Suspicious")
_METRIC_FIELDS = (
    "support_count",
    "coverage",
    "base_bad_rate",
    "hit_bad_rate",
    "non_hit_bad_rate",
    "lift",
    "precision",
    "recall",
    "p_value",
)


def _metric_payload(metrics: RuleMetrics) -> dict[str, float | int]:
    return {field: getattr(metrics, field) for field in _METRIC_FIELDS}


def summarize_conditions(
    rule: RiskRule, confirmed_features: frozenset[str]
) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    for condition in rule.conditions:
        if condition.feature not in confirmed_features:
            raise ValueError("rule condition must use a confirmed feature")
        if not isinstance(condition.value, (int, float)) or isinstance(condition.value, bool):
            raise ValueError("rule condition value must be numeric")
        conditions.append(
            {
                "feature": condition.feature,
                "operator": condition.operator,
                "value": condition.value,
            }
        )
    return conditions


def _rule_type(rule: RiskRule) -> str:
    if len(rule.conditions) == 1:
        return "single"
    if len(rule.conditions) == 2:
        return "two_condition"
    return "multi_condition"


def _candidate_item(
    rule: RiskRule,
    metrics: RuleMetrics,
    confirmed_features: frozenset[str],
    rank: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "rule_id": rule.rule_id,
        "rule_type": _rule_type(rule),
        "condition_count": len(rule.conditions),
        "conditions": summarize_conditions(rule, confirmed_features),
        "origin": rule.origin,
        "train": _metric_payload(metrics),
    }


def summarize_candidate_rules(
    result: DiscoveryResult,
    confirmed_features: frozenset[str],
    limit: int = 5,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("summary limit must be positive")
    ordered = sorted(
        result.rules,
        key=lambda rule: (
            -result.train_metrics[rule.rule_id].lift,
            -result.train_metrics[rule.rule_id].support_count,
            rule.rule_id,
        ),
    )
    top_rules = [
        _candidate_item(
            rule,
            result.train_metrics[rule.rule_id],
            confirmed_features,
            rank,
        )
        for rank, rule in enumerate(ordered[:limit], start=1)
    ]
    pairs = [rule for rule in ordered if len(rule.conditions) == 2][:limit]
    top_pairs = [
        _candidate_item(
            rule,
            result.train_metrics[rule.rule_id],
            confirmed_features,
            rank,
        )
        for rank, rule in enumerate(pairs, start=1)
    ]
    return {
        "candidate_rule_count": len(result.rules),
        "single_rule_count": result.single_rules_selected,
        "two_condition_rule_count": result.pair_rules_selected,
        "single_candidates_before_cap": result.single_candidates_before_cap,
        "two_condition_candidates_before_diversity": result.pair_candidates_before_diversity,
        "top_rules": top_rules,
        "top_two_condition_rules": top_pairs,
    }


def stable_reasons(
    card: EvidenceCard,
    validation_config: ValidationConfig,
    *,
    time_validation_enabled: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if card.adjusted_p_value > validation_config.alpha:
        reasons.append("adjusted_p_value_above_alpha")
    if card.lift_ci[0] <= 1.0:
        reasons.append("lift_ci_lower_not_above_one")
    has_segment_slices = any(item.slice_type == "segment" for item in card.slices)
    has_single_class_limitation = any(
        limitation.startswith("single-class") for limitation in card.limitations
    )
    if (
        card.test.support_count < validation_config.min_group_size
        or not has_segment_slices
        or has_single_class_limitation
        or (
            time_validation_enabled
            and not any(item.slice_type == "time" for item in card.slices)
        )
    ):
        reasons.append("insufficient_samples")
    if card.max_time_decay > validation_config.max_lift_decay:
        reasons.append("time_decay_above_limit")
    stable_segment_exists = any(
        item.slice_type == "segment" and item.metrics.lift > 1.0
        for item in card.slices
    )
    if (
        card.segment_consistency < validation_config.min_segment_consistency
        and stable_segment_exists
    ):
        reasons.append("low_segment_consistency")
    return reasons


def _institution_results(
    card: EvidenceCard, *, expose_segment_values: bool = True
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in card.slices:
        if item.slice_type != "segment":
            continue
        result: dict[str, Any] = {
            "institution_token": stable_token(
                item.slice_value, namespace="institution"
            ),
            "metrics": _metric_payload(item.metrics),
            "direction": "positive" if item.metrics.lift > 1.0 else "non_positive",
        }
        if expose_segment_values:
            result["institution_name"] = item.slice_value
        results.append(result)
    return results


def _grade_interpretation(grade: str) -> str:
    return {
        "Stable": "总体规则在满足样本门槛的机构中方向较一致，统计证据支持继续人工复核。",
        "Local": "总体效果主要集中在少数机构，不能直接当作全机构规则。",
        "Unstable": "规则在时间或验证切片中衰减明显，暂不适合作为稳定结论。",
        "Suspicious": "样本量或统计证据不足，当前结果应视为待验证而不是有效规则。",
    }[grade]


def _limitation_codes(card: EvidenceCard) -> list[str]:
    codes: set[str] = set()
    for limitation in card.limitations:
        normalized = limitation.lower()
        if "label performance window unknown" in normalized:
            codes.add("label_performance_window_unknown")
        if "holdout" in normalized:
            codes.add("holdout_evidence_unavailable")
        if "single-class" in normalized:
            codes.add("single_class_slice")
        if "missing time" in normalized or "snapshot" in normalized:
            codes.add("time_values_unavailable")
    return sorted(codes)


def _evidence_item(
    card: EvidenceCard,
    confirmed_features: frozenset[str],
    validation_config: ValidationConfig,
    rank: int,
    *,
    time_validation_enabled: bool,
    expose_segment_values: bool,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "rule_id": card.rule.rule_id,
        "rule_type": _rule_type(card.rule),
        "condition_count": len(card.rule.conditions),
        "conditions": summarize_conditions(card.rule, confirmed_features),
        "train": _metric_payload(card.train),
        "test": _metric_payload(card.test),
        "lift_ci": {"lower": card.lift_ci[0], "upper": card.lift_ci[1]},
        "adjusted_p_value": card.adjusted_p_value,
        "segment_consistency": card.segment_consistency,
        "max_time_decay": card.max_time_decay,
        "grade": card.grade,
        "reason_codes": stable_reasons(
            card,
            validation_config,
            time_validation_enabled=time_validation_enabled,
        ),
        "limitation_codes": _limitation_codes(card),
        "institution_results": _institution_results(
            card, expose_segment_values=expose_segment_values
        ),
        "interpretation": _grade_interpretation(card.grade),
    }


def summarize_evidence_cards(
    cards: Sequence[EvidenceCard],
    confirmed_features: frozenset[str],
    validation_config: ValidationConfig,
    limit: int = 5,
    *,
    time_validation_enabled: bool = False,
    expose_segment_values: bool = True,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("summary limit must be positive")
    ordered = sorted(
        cards,
        key=lambda card: (-card.test.lift, -card.test.support_count, card.rule.rule_id),
    )
    top_rules = [
        _evidence_item(
            card,
            confirmed_features,
            validation_config,
            rank,
            time_validation_enabled=time_validation_enabled,
            expose_segment_values=expose_segment_values,
        )
        for rank, card in enumerate(ordered[:limit], start=1)
    ]
    pairs = [card for card in ordered if len(card.rule.conditions) == 2][:limit]
    top_pairs = [
        _evidence_item(
            card,
            confirmed_features,
            validation_config,
            rank,
            time_validation_enabled=time_validation_enabled,
            expose_segment_values=expose_segment_values,
        )
        for rank, card in enumerate(pairs, start=1)
    ]
    stable_cards = [card for card in ordered if card.grade == "Stable"][:limit]
    stable_top = [
        _evidence_item(
            card,
            confirmed_features,
            validation_config,
            rank,
            time_validation_enabled=time_validation_enabled,
            expose_segment_values=expose_segment_values,
        )
        for rank, card in enumerate(stable_cards, start=1)
    ]
    grade_counts = {grade: sum(card.grade == grade for card in cards) for grade in _GRADE_NAMES}
    institution_tokens = {
        stable_token(item.slice_value, namespace="institution")
        for card in cards
        for item in card.slices
        if item.slice_type == "segment"
    }
    if len(institution_tokens) <= 1:
        institution_interpretation = (
            "当前只有一个或没有可用机构切片，结果按全局规则解释，未形成跨机构稳定性比较。"
        )
    else:
        institution_interpretation = (
            "总体规则先在全机构合并数据中发现，再按机构比较 Support、命中率和 Lift；"
            "Local 仅表示效果集中于少数机构。"
        )
    limitation_codes = sorted(
        {
            code
            for card in cards
            for code in _limitation_codes(card)
        }
    )
    institution_names = sorted(
        {
            item.slice_value
            for card in cards
            for item in card.slices
            if item.slice_type == "segment"
        }
    )
    institution_summary: dict[str, Any] = {
        "institution_count": len(institution_tokens),
        "rules_with_institution_metrics": sum(
            any(item.slice_type == "segment" for item in card.slices)
            for card in cards
        ),
        "interpretation": institution_interpretation,
    }
    if expose_segment_values:
        institution_summary["institution_names"] = institution_names
    return {
        "grade_counts": grade_counts,
        "top_rules": top_rules,
        "top_two_condition_rules": top_pairs,
        "stable_top_rules": stable_top,
        "limitation_codes": limitation_codes,
        "institution_summary": institution_summary,
    }


_ALERT_TYPES = (
    "schema",
    "missingness",
    "distribution",
    "population",
    "label",
    "rule_decay",
)


def _safe_string(value: object) -> str:
    from riskprobe.privacy import stable_token

    return stable_token(value)


def _safe_evidence(evidence: dict[str, float | int | str]) -> dict[str, float | int | str]:
    return {
        key: value if isinstance(value, (int, float)) and not isinstance(value, bool) else _safe_string(value)
        for key, value in evidence.items()
    }


def summarize_alerts(
    alerts: Sequence[Any],
    *,
    reference_row_count: int,
    reference_positive_rate: float,
    reference_feature_count: int,
    current_row_count: int,
    current_positive_rate: float,
    current_feature_count: int,
    expose_segment_values: bool = True,
) -> dict[str, Any]:
    counts = {alert_type: 0 for alert_type in _ALERT_TYPES}
    summaries: list[dict[str, Any]] = []
    for alert in alerts:
        counts[alert.alert_type] += 1
        summary: dict[str, Any] = {
            "alert_id": _safe_string(alert.alert_id),
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "scope": alert.scope,
            "scope_value": _safe_string(alert.scope_value),
            "metric": alert.metric,
            "reference_value": (
                alert.reference_value
                if isinstance(alert.reference_value, (int, float))
                else _safe_string(alert.reference_value)
                if alert.reference_value is not None
                else None
            ),
            "current_value": (
                alert.current_value
                if isinstance(alert.current_value, (int, float))
                else _safe_string(alert.current_value)
                if alert.current_value is not None
                else None
            ),
            "delta": alert.delta,
            "evidence": _safe_evidence(alert.evidence),
        }
        if expose_segment_values and alert.scope == "institution":
            summary["institution_name"] = alert.scope_value
        summaries.append(summary)
    global_alerts = [item for item in summaries if item["scope"] != "institution"]
    institution_alerts = [item for item in summaries if item["scope"] == "institution"]
    if institution_alerts and global_alerts:
        interpretation = (
            "同时存在全局和机构级告警；应先确认总体变化，再定位对总体结论贡献最大的机构。"
        )
    elif institution_alerts:
        interpretation = (
            "当前告警主要集中在机构层面，不能直接解释为全局规则失效。"
        )
    elif global_alerts:
        interpretation = (
            "当前告警主要位于全局层面，尚未发现单独机构告警；仍需结合机构稳定性报告复核。"
        )
    else:
        interpretation = "当前没有触发全局或机构级告警，诊断阶段没有新增根因。"
    return {
        "alert_counts": counts,
        "alerts": summaries,
        "global_alerts": global_alerts,
        "institution_alerts": institution_alerts,
        "interpretation": interpretation,
        "overview": {
            "reference_row_count": reference_row_count,
            "current_row_count": current_row_count,
            "reference_positive_rate": reference_positive_rate,
            "current_positive_rate": current_positive_rate,
            "reference_feature_count": reference_feature_count,
            "current_feature_count": current_feature_count,
        },
    }


def summarize_diagnoses(diagnoses: Sequence[Any]) -> dict[str, Any]:
    summaries = []
    for diagnosis in diagnoses:
        alert = diagnosis.alerts[0] if diagnosis.alerts else None
        summaries.append(
            {
                "alert_id": _safe_string(alert.alert_id) if alert is not None else None,
                "root_causes": [
                    {
                        "dimension": cause.dimension,
                        "value": _safe_string(cause.value),
                        "contribution": cause.contribution,
                        "rank": cause.rank,
                        "evidence": _safe_evidence(cause.evidence),
                    }
                    for cause in diagnosis.root_causes[:3]
                ],
            }
        )
    return {
        "alert_count": len(diagnoses),
        "diagnoses": summaries,
        "empty": not diagnoses,
    }
