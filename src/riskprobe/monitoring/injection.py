"""Reproducible local drift injection and alert scoring."""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import polars as pl

from riskprobe.models import Condition, FrozenModel

from .models import Alert, Diagnosis


DriftType = Literal[
    "missingness", "numeric_shift", "population_shift", "label_shift", "schema", "rule_decay"
]


class DriftScenario(FrozenModel):
    scenario_id: str
    drift_type: DriftType
    target: str
    magnitude: float
    institution: str | None = None
    target_column: str = "target"
    segment_column: str = "institution"
    rule_id: str | None = None
    rule_conditions: tuple[Condition, ...] = ()
    rule_hit_mask: tuple[bool, ...] | None = None


class DriftTruth(FrozenModel):
    scenario_id: str
    expected_alert_type: Literal[
        "schema", "missingness", "distribution", "population", "label", "rule_decay"
    ]
    expected_scope_value: str
    scene_key: str
    root_cause_dimension: str
    root_cause_value: str


class DetectionScore(FrozenModel):
    precision: float
    recall: float
    # TN is not available from aggregate alert streams. Keep this explicitly
    # nullable rather than presenting alert-level FDR as a population FPR.
    false_positive_rate: float | None
    false_discovery_rate: float
    top_k_root_cause_hit: float


@dataclass(frozen=True, slots=True)
class InjectedDrift:
    frame: pl.DataFrame
    truth: DriftTruth


def inject_drift(frame: pl.DataFrame, scenario: DriftScenario, seed: int) -> InjectedDrift:
    """Return a modified copy and aggregate-compatible ground truth, never mutating input."""
    if scenario.target not in frame.columns and scenario.drift_type != "rule_decay":
        raise ValueError(f"drift target is not a frame column: {scenario.target}")
    if scenario.drift_type == "rule_decay" and scenario.target_column not in frame.columns:
        raise ValueError(f"rule decay target column is not a frame column: {scenario.target_column}")
    if scenario.magnitude < 0:
        raise ValueError("drift magnitude must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if scenario.rule_hit_mask is not None and len(scenario.rule_hit_mask) != frame.height:
        raise ValueError("rule_hit_mask must have one entry per frame row")

    rng = np.random.default_rng(seed)
    candidates = _candidate_indices(frame, scenario.institution, scenario.segment_column)
    selected = _select_indices(candidates, scenario.magnitude, rng)
    if scenario.drift_type == "missingness":
        changed = _replace_selected(frame, scenario.target, selected, None)
    elif scenario.drift_type == "numeric_shift":
        changed = _numeric_shift(frame, scenario.target, selected, scenario.magnitude)
    elif scenario.drift_type == "population_shift":
        changed = _oversample(frame, candidates, scenario.magnitude, rng)
    elif scenario.drift_type == "label_shift":
        changed = _flip_labels(frame, selected, scenario.target_column, old=0, new=1)
    elif scenario.drift_type == "schema":
        changed = frame.drop(scenario.target)
    else:
        changed = _rule_decay(frame, scenario, selected)
    return InjectedDrift(frame=changed, truth=_truth(scenario))


def evaluate_alerts(
    alerts: Iterable[Alert],
    truth: Iterable[DriftTruth],
    top_k: int = 3,
    diagnoses: Iterable[Diagnosis] = (),
) -> DetectionScore:
    """Score one scenario's alerts; report FDR separately because TN is unknown."""
    alert_list = list(alerts)
    truth_list = list(truth)
    truth_keys = {(item.expected_alert_type, item.expected_scope_value) for item in truth_list}
    matched_alerts = [
        alert for alert in alert_list if (alert.alert_type, alert.scope_value) in truth_keys
    ]
    alert_keys = {(alert.alert_type, alert.scope_value) for alert in alert_list}
    matched_truths = sum(
        (item.expected_alert_type, item.expected_scope_value) in alert_keys for item in truth_list
    )
    precision = len(matched_alerts) / len(alert_list) if alert_list else 0.0
    recall = matched_truths / len(truth_list) if truth_list else 0.0
    false_discovery_rate = (
        (len(alert_list) - len(matched_alerts)) / len(alert_list) if alert_list else 0.0
    )
    false_positive_rate = None
    cause_keys = {
        (cause.dimension, cause.value)
        for diagnosis in diagnoses
        for cause in diagnosis.root_causes[:top_k]
    }
    top_k_hit = (
        sum(
            (item.root_cause_dimension, item.root_cause_value) in cause_keys
            for item in truth_list
        )
        / len(truth_list)
        if truth_list
        else 0.0
    )
    return DetectionScore(
        precision=precision,
        recall=recall,
        false_positive_rate=false_positive_rate,
        false_discovery_rate=false_discovery_rate,
        top_k_root_cause_hit=top_k_hit,
    )


def _truth(scenario: DriftScenario) -> DriftTruth:
    alert_type = {
        "missingness": "missingness", "numeric_shift": "distribution",
        "population_shift": "population", "label_shift": "label",
        "schema": "schema", "rule_decay": "rule_decay",
    }[scenario.drift_type]
    scope_value = scenario.rule_id or scenario.target if scenario.drift_type == "rule_decay" else scenario.target
    root_dimension = {
        "missingness": "feature", "numeric_shift": "feature", "population_shift": "segment",
        "label_shift": "target", "schema": "schema", "rule_decay": "rule",
    }[scenario.drift_type]
    root_value = {
        "missingness": scenario.target, "numeric_shift": scenario.target,
        "population_shift": scenario.institution or scenario.target,
        "label_shift": scenario.target_column, "schema": scenario.target,
        "rule_decay": scenario.rule_id or scenario.target,
    }[scenario.drift_type]
    return DriftTruth(
        scenario_id=scenario.scenario_id,
        expected_alert_type=alert_type,
        expected_scope_value=scope_value,
        scene_key=scenario.rule_id or scenario.target,
        root_cause_dimension=root_dimension,
        root_cause_value=root_value,
    )


def _candidate_indices(frame: pl.DataFrame, institution: str | None, segment_column: str) -> list[int]:
    if institution is None or segment_column not in frame.columns:
        return list(range(frame.height))
    return [
        index for index, value in enumerate(frame.get_column(segment_column).to_list())
        if str(value) == institution
    ]


def _select_indices(candidates: list[int], magnitude: float, rng: np.random.Generator) -> list[int]:
    if not candidates or magnitude == 0:
        return []
    count = min(len(candidates), max(1, round(len(candidates) * magnitude)))
    return sorted(int(index) for index in rng.choice(candidates, size=count, replace=False))


def _replace_selected(frame: pl.DataFrame, column: str, selected: list[int], value: object) -> pl.DataFrame:
    if not selected:
        return frame.clone()
    return (
        frame.with_row_index("__riskprobe_injection_index")
        .with_columns(
            pl.when(pl.col("__riskprobe_injection_index").is_in(selected))
            .then(pl.lit(value, dtype=frame.schema[column]))
            .otherwise(pl.col(column)).alias(column)
        )
        .drop("__riskprobe_injection_index")
    )


def _numeric_shift(frame: pl.DataFrame, column: str, selected: list[int], magnitude: float) -> pl.DataFrame:
    if not frame.schema[column].is_numeric():
        raise ValueError("numeric_shift requires a numeric target")
    if not selected:
        return frame.clone()
    standard_deviation = frame.get_column(column).drop_nulls().std() or 0.0
    return (
        frame.with_row_index("__riskprobe_injection_index")
        .with_columns(
            pl.when(pl.col("__riskprobe_injection_index").is_in(selected))
            .then(pl.col(column) + magnitude * standard_deviation)
            .otherwise(pl.col(column)).cast(frame.schema[column]).alias(column)
        )
        .drop("__riskprobe_injection_index")
    )


def _oversample(frame: pl.DataFrame, candidates: list[int], magnitude: float, rng: np.random.Generator) -> pl.DataFrame:
    if not candidates or magnitude == 0:
        return frame.clone()
    count = max(1, round(frame.height * magnitude))
    selected = [int(index) for index in rng.choice(candidates, size=count, replace=True)]
    return pl.concat((frame, frame[selected]), how="vertical")


def _flip_labels(frame: pl.DataFrame, selected: list[int], column: str, old: int, new: int) -> pl.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"label drift requires a target column named '{column}'")
    eligible = [index for index in selected if frame.get_column(column)[index] == old]
    return _replace_selected(frame, column, eligible, new)


def _rule_decay(frame: pl.DataFrame, scenario: DriftScenario, selected: list[int]) -> pl.DataFrame:
    if scenario.rule_hit_mask is not None:
        hits = [index for index, hit in enumerate(scenario.rule_hit_mask) if hit]
    elif scenario.rule_conditions:
        hits = _condition_indices(frame, scenario.rule_conditions)
    else:
        # Legacy callers have no rule card. Use positive target rows as the
        # explicit default rule mask; never infer a rule from feature medians.
        hits = [
            index for index, value in enumerate(frame.get_column(scenario.target_column).to_list())
            if value == 1
        ]
    eligible = sorted(set(selected).intersection(hits))
    return _flip_labels(frame, eligible, scenario.target_column, old=1, new=0)


def _condition_indices(frame: pl.DataFrame, conditions: tuple[Condition, ...]) -> list[int]:
    mask = pl.lit(True)
    for condition in conditions:
        column = pl.col(condition.feature)
        if condition.operator == "is_null":
            mask &= column.is_null()
        elif condition.operator == "==":
            mask &= column == condition.value
        elif condition.operator == "!=":
            mask &= column != condition.value
        elif condition.operator == ">":
            mask &= column > condition.value
        elif condition.operator == ">=":
            mask &= column >= condition.value
        elif condition.operator == "<":
            mask &= column < condition.value
        else:
            mask &= column <= condition.value
    return frame.with_row_index("__riskprobe_rule_index").filter(mask)["__riskprobe_rule_index"].to_list()
