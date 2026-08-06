import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

_WINDOW_PATTERN = re.compile(r"_(7|30|90|180|365)d(?:_|$)")
_RATIO_MARKERS = ("rate", "ratio", "pct", "percent", "share")
_AMOUNT_MARKERS = ("amount", "amt")
_COUNT_MARKERS = ("cnt", "count", "num", "pv", "days")
_NON_CUMULATIVE_MARKERS = ("avg", "mean", "median", "min", "max", "latest", "last")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    family: str
    window_days: int | None
    aggregation: str
    value_type: str


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: Literal["warning", "error"]
    family: str
    features: tuple[str, ...]
    affected_rows: int
    message: str


@dataclass(frozen=True, slots=True)
class FeatureCatalog:
    features: tuple[FeatureSpec, ...]

    @classmethod
    def from_columns(
        cls,
        columns: Iterable[str],
        family_prefixes: Mapping[str, Sequence[str]],
    ) -> "FeatureCatalog":
        specs = tuple(_feature_spec(column, family_prefixes) for column in columns)
        return cls(features=specs)


def _feature_spec(
    column: str,
    family_prefixes: Mapping[str, Sequence[str]],
) -> FeatureSpec:
    family = "unknown"
    metric_name = column
    for candidate_family, prefixes in family_prefixes.items():
        matching_prefix = next((prefix for prefix in prefixes if column.startswith(prefix)), None)
        if matching_prefix is not None:
            family = candidate_family
            metric_name = column[len(matching_prefix) :]
            break

    window_match = _WINDOW_PATTERN.search(column)
    window_days = int(window_match.group(1)) if window_match is not None else None
    aggregation = _WINDOW_PATTERN.sub("_", f"_{metric_name}").strip("_")
    value_type = _value_type(aggregation)
    return FeatureSpec(
        name=column,
        family=family,
        window_days=window_days,
        aggregation=aggregation,
        value_type=value_type,
    )


def _value_type(aggregation: str) -> str:
    tokens = set(aggregation.lower().split("_"))
    if tokens.intersection(_RATIO_MARKERS):
        return "ratio"
    if tokens.intersection(_AMOUNT_MARKERS):
        return "amount"
    if tokens.intersection(_COUNT_MARKERS):
        return "count"
    return "numeric"


def _is_comparable_cumulative(spec: FeatureSpec) -> bool:
    tokens = set(spec.aggregation.lower().split("_"))
    return spec.value_type == "count" and not tokens.intersection(_NON_CUMULATIVE_MARKERS)


def check_window_invariants(
    frame: pl.DataFrame,
    catalog: FeatureCatalog,
) -> tuple[QualityIssue, ...]:
    window_groups: defaultdict[tuple[str, str, int], list[FeatureSpec]] = defaultdict(list)
    for spec in catalog.features:
        if spec.window_days is not None and _is_comparable_cumulative(spec):
            window_groups[(spec.family, spec.aggregation, spec.window_days)].append(spec)

    groups: defaultdict[tuple[str, str], dict[int, list[FeatureSpec]]] = defaultdict(dict)
    for (family, aggregation, window_days), specs in window_groups.items():
        groups[(family, aggregation)][window_days] = specs

    issues: list[QualityIssue] = []
    for (family, aggregation), windows in groups.items():
        ordered_windows = sorted(windows.items())
        if len(ordered_windows) < 2:
            continue
        inversion = pl.lit(False)
        for (_, shorter_specs), (_, longer_specs) in zip(ordered_windows, ordered_windows[1:]):
            for shorter in shorter_specs:
                for longer in longer_specs:
                    inversion = inversion | (pl.col(shorter.name) > pl.col(longer.name)).fill_null(
                        False
                    )
        affected_rows = frame.select(inversion.sum()).item()
        if affected_rows:
            features = tuple(spec.name for _, specs in ordered_windows for spec in specs)
            issues.append(
                QualityIssue(
                    code="WINDOW_INVERSION",
                    severity="warning",
                    family=family,
                    features=features,
                    affected_rows=int(affected_rows),
                    message=(
                        f"{affected_rows} rows decrease across cumulative windows "
                        f"for {family}.{aggregation}"
                    ),
                )
            )
    return tuple(issues)
