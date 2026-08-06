from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from riskprobe.config import ValidationConfig
from riskprobe.metrics import adjust_pvalues, bootstrap_lift_ci, compute_rule_metrics
from riskprobe.models import EvidenceCard, RiskRule, RuleMetrics, SliceMetrics
from riskprobe.rules.expression import evaluate_rule


@dataclass(frozen=True)
class _RuleValidation:
    rule: RiskRule
    train: RuleMetrics
    test: RuleMetrics
    slices: tuple[SliceMetrics, ...]
    lift_ci: tuple[float, float]
    segment_consistency: float
    max_time_decay: float
    insufficient_samples: bool


def _metrics(frame: pl.DataFrame, rule: RiskRule, target_col: str) -> RuleMetrics:
    return compute_rule_metrics(
        evaluate_rule(frame, rule).to_numpy(),
        frame.get_column(target_col).to_numpy(),
        positive_value=1,
    )


def _group_slices(
    frame: pl.DataFrame,
    rule: RiskRule,
    *,
    target_col: str,
    group_col: str,
    slice_type: Literal["segment", "time"],
    min_group_size: int,
) -> tuple[tuple[SliceMetrics, ...], bool]:
    slices: list[SliceMetrics] = []
    has_uncomputable_group = False
    for group in frame.partition_by(group_col, maintain_order=True):
        if group.height < min_group_size:
            continue
        try:
            metrics = _metrics(group, rule, target_col)
        except ValueError as error:
            if str(error) == "target has no positive samples":
                has_uncomputable_group = True
                continue
            raise
        slices.append(
            SliceMetrics(
                slice_type=slice_type,
                slice_value=str(group.get_column(group_col)[0]),
                metrics=metrics,
            )
        )
    return tuple(slices), has_uncomputable_group


def _with_time_bucket(frame: pl.DataFrame, snapshot_col: str) -> pl.DataFrame:
    dtype = frame.schema[snapshot_col]
    if dtype == pl.String:
        parsed = pl.col(snapshot_col).str.to_date(
            format="%Y-%m-%d", strict=False
        )
    elif dtype == pl.Date:
        parsed = pl.col(snapshot_col)
    elif isinstance(dtype, pl.Datetime):
        parsed = pl.col(snapshot_col).dt.date()
    else:
        raise ValueError("snapshot column contains invalid dates")

    bucketed = frame.with_columns(parsed.dt.strftime("%Y-%m").alias("__time_bucket"))
    if bucketed.get_column("__time_bucket").null_count():
        raise ValueError("snapshot column contains invalid dates")
    return bucketed


def _validate_rule(
    train: pl.DataFrame,
    test: pl.DataFrame,
    rule: RiskRule,
    *,
    target_col: str,
    segment_col: str,
    snapshot_col: str,
    time_validation_enabled: bool,
    config: ValidationConfig,
) -> _RuleValidation:
    train_metrics = _metrics(train, rule, target_col)
    test_mask = evaluate_rule(test, rule).to_numpy()
    test_target = test.get_column(target_col).to_numpy()
    test_metrics = compute_rule_metrics(test_mask, test_target, positive_value=1)
    lift_ci = bootstrap_lift_ci(
        test_mask,
        test_target,
        positive_value=1,
        rounds=config.bootstrap_rounds,
        random_seed=42,
    )

    segment_slices, has_uncomputable_segment = _group_slices(
        test,
        rule,
        target_col=target_col,
        group_col=segment_col,
        slice_type="segment",
        min_group_size=config.min_group_size,
    )
    stable_segment_count = sum(
        item.metrics.lift > 1.0 for item in segment_slices
    )
    segment_consistency = (
        stable_segment_count / len(segment_slices) if segment_slices else 0.0
    )

    time_slices: tuple[SliceMetrics, ...] = ()
    has_uncomputable_time = False
    max_time_decay = 0.0
    if time_validation_enabled:
        time_frame = _with_time_bucket(test, snapshot_col)
        time_slices, has_uncomputable_time = _group_slices(
            time_frame,
            rule,
            target_col=target_col,
            group_col="__time_bucket",
            slice_type="time",
            min_group_size=config.min_group_size,
        )
        if time_slices and train_metrics.lift > 0.0:
            minimum_time_lift = min(item.metrics.lift for item in time_slices)
            max_time_decay = max(
                0.0,
                (train_metrics.lift - minimum_time_lift) / train_metrics.lift,
            )

    insufficient_samples = (
        train.height < config.min_group_size
        or test.height < config.min_group_size
        or not segment_slices
        or has_uncomputable_segment
        or (
            time_validation_enabled
            and (not time_slices or has_uncomputable_time)
        )
    )
    return _RuleValidation(
        rule=rule,
        train=train_metrics,
        test=test_metrics,
        slices=segment_slices + time_slices,
        lift_ci=lift_ci,
        segment_consistency=segment_consistency,
        max_time_decay=max_time_decay,
        insufficient_samples=insufficient_samples,
    )


def _grade(
    validation: _RuleValidation,
    adjusted_p_value: float,
    config: ValidationConfig,
) -> Literal["Stable", "Local", "Unstable", "Suspicious"]:
    if (
        adjusted_p_value > config.alpha
        or validation.lift_ci[0] <= 1.0
        or validation.insufficient_samples
    ):
        return "Suspicious"
    if validation.max_time_decay > config.max_lift_decay:
        return "Unstable"
    stable_segment_exists = any(
        item.slice_type == "segment" and item.metrics.lift > 1.0
        for item in validation.slices
    )
    if (
        validation.segment_consistency < config.min_segment_consistency
        and stable_segment_exists
    ):
        return "Local"
    return "Stable"


def validate_rules(
    train: pl.DataFrame,
    test: pl.DataFrame,
    rules: Sequence[RiskRule],
    *,
    target_col: str,
    segment_col: str,
    snapshot_col: str,
    segment_display_name: str,
    time_validation_enabled: bool,
    config: ValidationConfig,
    metadata_grade: str,
) -> list[EvidenceCard]:
    if not rules:
        return []

    validations = [
        _validate_rule(
            train,
            test,
            rule,
            target_col=target_col,
            segment_col=segment_col,
            snapshot_col=snapshot_col,
            time_validation_enabled=time_validation_enabled,
            config=config,
        )
        for rule in rules
    ]
    adjusted_p_values = adjust_pvalues(
        [validation.train.p_value for validation in validations]
    )
    limitations = (
        ("label performance window unknown",) if metadata_grade == "B" else ()
    )

    return [
        EvidenceCard(
            rule=validation.rule,
            train=validation.train,
            test=validation.test,
            slices=validation.slices,
            lift_ci=validation.lift_ci,
            adjusted_p_value=adjusted_p_value,
            segment_consistency=validation.segment_consistency,
            max_time_decay=validation.max_time_decay,
            grade=_grade(validation, adjusted_p_value, config),
            limitations=limitations,
        )
        for validation, adjusted_p_value in zip(
            validations, adjusted_p_values, strict=True
        )
    ]
