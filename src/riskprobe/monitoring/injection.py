"""Reproducible local drift injection and alert scoring."""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import polars as pl

from riskprobe.models import FrozenModel

from .models import Alert


DriftType = Literal[
    "missingness",
    "numeric_shift",
    "population_shift",
    "label_shift",
    "schema",
    "rule_decay",
]


class DriftScenario(FrozenModel):
    scenario_id: str
    drift_type: DriftType
    target: str
    magnitude: float
    institution: str | None = None


class DriftTruth(FrozenModel):
    scenario_id: str
    expected_alert_type: Literal[
        "schema", "missingness", "distribution", "population", "label", "rule_decay"
    ]
    expected_scope_value: str


class DetectionScore(FrozenModel):
    precision: float
    recall: float
    false_positive_rate: float
    top_k_root_cause_hit: float


@dataclass(frozen=True, slots=True)
class InjectedDrift:
    frame: pl.DataFrame
    truth: DriftTruth


def inject_drift(frame: pl.DataFrame, scenario: DriftScenario, seed: int) -> InjectedDrift:
    """Return a modified copy and aggregate-compatible ground truth, never mutating input."""
    if scenario.target not in frame.columns:
        raise ValueError(f"drift target is not a frame column: {scenario.target}")
    if scenario.magnitude < 0:
        raise ValueError("drift magnitude must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    candidates = _candidate_indices(frame, scenario.institution)
    selected = _select_indices(candidates, scenario.magnitude, rng)
    if scenario.drift_type == "missingness":
        changed = _replace_selected(frame, scenario.target, selected, None)
    elif scenario.drift_type == "numeric_shift":
        changed = _numeric_shift(frame, scenario.target, selected, scenario.magnitude)
    elif scenario.drift_type == "population_shift":
        changed = _oversample(frame, candidates, scenario.magnitude, rng)
    elif scenario.drift_type == "label_shift":
        changed = _flip_labels(frame, selected, old=0, new=1)
    elif scenario.drift_type == "schema":
        changed = frame.drop(scenario.target)
    else:
        changed = _rule_decay(frame, scenario.target, selected)
    return InjectedDrift(frame=changed, truth=_truth(scenario))


def evaluate_alerts(
    alerts: Iterable[Alert], truth: Iterable[DriftTruth], top_k: int = 3
) -> DetectionScore:
    """Score alert matching solely by alert type and stable aggregate scope code."""
    alert_list = list(alerts)
    truth_list = list(truth)
    truth_keys = {(item.expected_alert_type, item.expected_scope_value) for item in truth_list}
    matched_alerts = [
        alert
        for alert in alert_list
        if (alert.alert_type, alert.scope_value) in truth_keys
    ]
    alert_keys = {(alert.alert_type, alert.scope_value) for alert in alert_list}
    matched_truths = sum(
        (item.expected_alert_type, item.expected_scope_value) in alert_keys
        for item in truth_list
    )
    precision = len(matched_alerts) / len(alert_list) if alert_list else 0.0
    recall = matched_truths / len(truth_list) if truth_list else 0.0
    false_positive_rate = (
        (len(alert_list) - len(matched_alerts)) / len(alert_list) if alert_list else 0.0
    )
    top_keys = {(alert.alert_type, alert.scope_value) for alert in alert_list[:top_k]}
    top_k_hit = (
        sum((item.expected_alert_type, item.expected_scope_value) in top_keys for item in truth_list)
        / len(truth_list)
        if truth_list
        else 0.0
    )
    return DetectionScore(
        precision=precision,
        recall=recall,
        false_positive_rate=false_positive_rate,
        top_k_root_cause_hit=top_k_hit,
    )


def _truth(scenario: DriftScenario) -> DriftTruth:
    alert_type = {
        "missingness": "missingness",
        "numeric_shift": "distribution",
        "population_shift": "population",
        "label_shift": "label",
        "schema": "schema",
        "rule_decay": "rule_decay",
    }[scenario.drift_type]
    return DriftTruth(
        scenario_id=scenario.scenario_id,
        expected_alert_type=alert_type,
        expected_scope_value=scenario.target,
    )


def _candidate_indices(frame: pl.DataFrame, institution: str | None) -> list[int]:
    if institution is None or "institution" not in frame.columns:
        return list(range(frame.height))
    return [
        index
        for index, value in enumerate(frame.get_column("institution").to_list())
        if str(value) == institution
    ]


def _select_indices(candidates: list[int], magnitude: float, rng: np.random.Generator) -> list[int]:
    if not candidates or magnitude == 0:
        return []
    count = min(len(candidates), max(1, round(len(candidates) * magnitude)))
    return sorted(int(index) for index in rng.choice(candidates, size=count, replace=False))


def _replace_selected(
    frame: pl.DataFrame, column: str, selected: list[int], value: object
) -> pl.DataFrame:
    if not selected:
        return frame.clone()
    return (
        frame.with_row_index("__riskprobe_injection_index")
        .with_columns(
            pl.when(pl.col("__riskprobe_injection_index").is_in(selected))
            .then(pl.lit(value, dtype=frame.schema[column]))
            .otherwise(pl.col(column))
            .alias(column)
        )
        .drop("__riskprobe_injection_index")
    )


def _numeric_shift(
    frame: pl.DataFrame, column: str, selected: list[int], magnitude: float
) -> pl.DataFrame:
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
            .otherwise(pl.col(column))
            .cast(frame.schema[column])
            .alias(column)
        )
        .drop("__riskprobe_injection_index")
    )


def _oversample(
    frame: pl.DataFrame, candidates: list[int], magnitude: float, rng: np.random.Generator
) -> pl.DataFrame:
    if not candidates or magnitude == 0:
        return frame.clone()
    count = max(1, round(frame.height * magnitude))
    selected = [int(index) for index in rng.choice(candidates, size=count, replace=True)]
    return pl.concat((frame, frame[selected]), how="vertical")


def _flip_labels(frame: pl.DataFrame, selected: list[int], old: int, new: int) -> pl.DataFrame:
    if "target" not in frame.columns:
        raise ValueError("label drift requires a target column named 'target'")
    eligible = [
        index
        for index in selected
        if frame.get_column("target")[index] == old
    ]
    return _replace_selected(frame, "target", eligible, new)


def _rule_decay(frame: pl.DataFrame, target: str, selected: list[int]) -> pl.DataFrame:
    if "target" not in frame.columns:
        raise ValueError("rule decay requires a target column named 'target'")
    values = frame.get_column(target)
    if values.dtype.is_numeric():
        median = values.drop_nulls().median()
        selected = [index for index in selected if values[index] is not None and values[index] >= median]
    return _flip_labels(frame, selected, old=1, new=0)
