from collections.abc import Sequence
from typing import Any

import polars as pl

from riskprobe.config import DiscoveryConfig, ValidationConfig
from riskprobe.explainability import summarize_candidate_rules, summarize_evidence_cards
from riskprobe.models import EvidenceCard, SliceMetrics
from riskprobe.privacy import stable_token
from riskprobe.rules.discovery import discover_with_metrics
from riskprobe.rules.validation import validate_rules

_MAX_LOCAL_INSTITUTIONS = 5
_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}


def _local_candidates(cards: Sequence[EvidenceCard]) -> dict[str, tuple[float, int]]:
    candidates: dict[str, tuple[float, int]] = {}
    for card in cards:
        if card.grade != "Local":
            continue
        for item in card.slices:
            if item.slice_type != "segment" or item.metrics.lift <= 1.0:
                continue
            current = candidates.get(item.slice_value)
            score = (item.metrics.lift, item.metrics.support_count)
            if current is None or score > current:
                candidates[item.slice_value] = score
    return candidates


def _segment_frame(frame: pl.DataFrame, segment_col: str, segment_value: str) -> pl.DataFrame:
    return frame.filter(pl.col(segment_col).cast(pl.String) == pl.lit(segment_value))


def _institution_fields(
    institution_value: str, *, expose_segment_values: bool
) -> dict[str, str]:
    fields = {
        "institution_token": stable_token(institution_value, namespace="institution")
    }
    if expose_segment_values:
        fields["institution_name"] = institution_value
    return fields


def _blocked(
    institution_value: str,
    reason: str,
    *,
    score: tuple[float, int] | None = None,
    expose_segment_values: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **_institution_fields(
            institution_value, expose_segment_values=expose_segment_values
        ),
        "status": "blocked",
        "reason": reason,
    }
    if score is not None:
        payload["global_lift"] = score[0]
        payload["global_support_count"] = score[1]
    return payload


def _with_holdout_limitation(
    cards: Sequence[EvidenceCard], limitation: str
) -> list[EvidenceCard]:
    return [
        card.model_copy(
            update={
                "grade": "Suspicious",
                "limitations": tuple(sorted({*card.limitations, limitation})),
            }
        )
        for card in cards
    ]


def _attach_holdout(
    primary: Sequence[EvidenceCard], holdout: Sequence[EvidenceCard]
) -> list[EvidenceCard]:
    holdout_by_id = {card.rule.rule_id: card for card in holdout}
    combined: list[EvidenceCard] = []
    for card in primary:
        holdout_card = holdout_by_id.get(card.rule.rule_id)
        if holdout_card is None:
            combined.append(
                card.model_copy(
                    update={
                        "grade": "Suspicious",
                        "limitations": tuple(
                            sorted(
                                {
                                    *card.limitations,
                                    "Holdout evidence is missing for this rule",
                                }
                            )
                        ),
                    }
                )
            )
            continue
        combined.append(
            card.model_copy(
                update={
                    "slices": card.slices
                    + (
                        SliceMetrics(
                            slice_type="dataset",
                            slice_value="Holdout",
                            metrics=holdout_card.test,
                        ),
                    ),
                    "limitations": tuple(
                        sorted(
                            {
                                *card.limitations,
                                *(f"holdout: {item}" for item in holdout_card.limitations),
                            }
                        )
                    ),
                    "grade": max(
                        (card.grade, holdout_card.grade),
                        key=lambda grade: _GRADE_ORDER[grade],
                    ),
                    "max_time_decay": max(
                        card.max_time_decay, holdout_card.max_time_decay
                    ),
                    "segment_consistency": min(
                        card.segment_consistency,
                        holdout_card.segment_consistency,
                    ),
                }
            )
        )
    return combined


def discover_local_rules(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cards: Sequence[EvidenceCard],
    feature_names: list[str],
    *,
    target_col: str,
    segment_col: str,
    snapshot_col: str,
    time_validation_enabled: bool,
    discovery_config: DiscoveryConfig,
    validation_config: ValidationConfig,
    confirmed_features: frozenset[str],
    segment_display_name: str = "institution",
    metadata_grade: str = "B",
    holdout: pl.DataFrame | None = None,
    runs_dir: object | None = None,
    expose_segment_values: bool = True,
) -> dict[str, Any]:
    """Discover rules only for sufficiently supported Local institutions.

    This function is called after global validation. It never runs the
    discovery engine when no global card is Local, returns tokenized institution
    reports, and propagates local validation failures instead of hiding them.
    """
    del runs_dir
    candidates = _local_candidates(cards)
    ordered_candidates = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1][0],
            -item[1][1],
            stable_token(item[0], namespace="institution"),
        ),
    )
    reports: list[dict[str, Any]] = []
    completed = 0

    for index, (institution_value, score) in enumerate(ordered_candidates):
        if index >= _MAX_LOCAL_INSTITUTIONS:
            reports.append(
                _blocked(
                    institution_value,
                    "local institution discovery limit reached",
                    score=score,
                    expose_segment_values=expose_segment_values,
                )
            )
            continue

        local_train = _segment_frame(train, segment_col, institution_value)
        local_test = _segment_frame(test, segment_col, institution_value)
        if (
            local_train.height < validation_config.min_group_size
            or local_test.height < validation_config.min_group_size
        ):
            reports.append(
                _blocked(
                    institution_value,
                    "institution Train/Test sample is below the minimum discovery size",
                    score=score,
                    expose_segment_values=expose_segment_values,
                )
            )
            continue
        if (
            local_train.get_column(target_col).drop_nulls().n_unique() < 2
            or local_test.get_column(target_col).drop_nulls().n_unique() < 2
        ):
            reports.append(
                _blocked(
                    institution_value,
                    "institution Train/Test contains a single target class",
                    score=score,
                    expose_segment_values=expose_segment_values,
                )
            )
            continue

        result = discover_with_metrics(
            local_train.select([*feature_names, target_col]),
            feature_names,
            target_col,
            discovery_config,
        )
        validation_columns = [*feature_names, target_col, segment_col]
        if time_validation_enabled:
            validation_columns.append(snapshot_col)
        local_cards = validate_rules(
            local_train.select(validation_columns),
            local_test.select(validation_columns),
            result.rules,
            target_col=target_col,
            segment_col=segment_col,
            snapshot_col=snapshot_col,
            segment_display_name=segment_display_name,
            time_validation_enabled=time_validation_enabled,
            config=validation_config,
            metadata_grade=metadata_grade,
        )
        holdout_validated = False
        if time_validation_enabled:
            local_holdout = (
                _segment_frame(holdout, segment_col, institution_value)
                if holdout is not None
                else None
            )
            if local_holdout is None or local_holdout.is_empty():
                local_cards = _with_holdout_limitation(
                    local_cards,
                    "Holdout partition is empty; local validation unavailable",
                )
            elif local_holdout.get_column(target_col).drop_nulls().n_unique() < 2:
                local_cards = _with_holdout_limitation(
                    local_cards,
                    "Holdout partition has a single target class; local validation unavailable",
                )
            else:
                holdout_cards = validate_rules(
                    local_train.select(validation_columns),
                    local_holdout.select(validation_columns),
                    result.rules,
                    target_col=target_col,
                    segment_col=segment_col,
                    snapshot_col=snapshot_col,
                    segment_display_name=segment_display_name,
                    time_validation_enabled=time_validation_enabled,
                    config=validation_config,
                    metadata_grade=metadata_grade,
                )
                local_cards = _attach_holdout(local_cards, holdout_cards)
                holdout_validated = True

        report = {
            **_institution_fields(
                institution_value, expose_segment_values=expose_segment_values
            ),
            "status": "completed",
            "train_row_count": local_train.height,
            "test_row_count": local_test.height,
            "rule_count": len(result.rules),
            "global_lift": score[0],
            "global_support_count": score[1],
            "holdout_validated": holdout_validated,
            "discovery_report": summarize_candidate_rules(result, confirmed_features),
            "validation_report": summarize_evidence_cards(
                local_cards,
                confirmed_features,
                validation_config,
                time_validation_enabled=time_validation_enabled,
                expose_segment_values=expose_segment_values,
            ),
            "interpretation": (
                "该机构满足局部规则发现门槛；以下规则仅代表机构内证据，"
                "不得自动视为全局规则或直接上线。"
                + (
                    "局部结果包含 Holdout 验证。"
                    if holdout_validated
                    else "局部结果未获得可用 Holdout 证据。"
                )
                + (
                    "当前为 B 级数据，表现窗口未知。"
                    if metadata_grade == "B"
                    else ""
                )
            ),
        }
        reports.append(report)
        completed += 1

    blocked_count = sum(report["status"] == "blocked" for report in reports)
    if not candidates:
        interpretation = (
            "总体规则没有触发 Local 条件，因此未执行机构内规则发现；"
            "当前结论优先采用全机构合并规则。"
        )
    elif completed:
        interpretation = (
            f"共有 {completed} 个机构满足 Local 且样本充足条件，已生成机构内候选；"
            "机构规则与全局规则分开治理。"
        )
    else:
        interpretation = (
            "发现了潜在 Local 机构，但均未通过机构内样本或标签门槛，"
            "因此没有生成机构内规则。"
        )
    return {
        "analysis_mode": "global_first_conditional_local",
        "eligible_institution_count": len(candidates),
        "triggered_institution_count": completed,
        "blocked_institution_count": blocked_count,
        "institution_reports": reports,
        "interpretation": interpretation,
    }
