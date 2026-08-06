import hashlib
import json
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from riskprobe.artifacts import RunContext, RunStore
from riskprobe.config import ProjectConfig
from riskprobe.dates import normalize_date_series
from riskprobe.io.parquet import ParquetDataset
from riskprobe.models import EvidenceCard, RiskRule, SliceMetrics
from riskprobe.profiling import DatasetProfile, profile_dataset
from riskprobe.reporting import (
    evidence_sort_key,
    redact_limitation,
    redact_segment_value,
    render_risk_report,
)
from riskprobe.rules.discovery import discover_rules
from riskprobe.rules.validation import validate_rules

_DISCOVERY_SAMPLE_LIMIT = 50_000
_ARTIFACT_NAMES = (
    "manifest.json",
    "metadata_report.json",
    "data_profile.json",
    "candidate_rules.parquet",
    "evidence_cards.json",
    "risk_report.md",
)
_SLICE_ORDER = {"dataset": 0, "segment": 1, "time": 2}
_GRADE_ORDER = {"Stable": 0, "Local": 1, "Unstable": 2, "Suspicious": 3}


def _package_version() -> str:
    try:
        return version("riskprobe-agent")
    except PackageNotFoundError:
        return "0.1.0"


def _safe_dataset_id(dataset_id: str) -> str:
    if Path(dataset_id).is_absolute() or PureWindowsPath(dataset_id).is_absolute():
        digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8]
        return f"dataset-{digest}"
    return dataset_id


def _parquet_metadata_fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size < 12:
            raise ValueError("invalid Parquet file")
        handle.seek(-8, 2)
        footer_size = int.from_bytes(handle.read(4), byteorder="little")
        if handle.read(4) != b"PAR1" or footer_size > size - 12:
            raise ValueError("invalid Parquet footer")
        handle.seek(-(footer_size + 8), 2)
        metadata = handle.read(footer_size)
    return hashlib.sha256(metadata).hexdigest()


def _time_split(frame: pl.DataFrame, snapshot_col: str) -> tuple[pl.DataFrame, ...]:
    order_column = "__riskprobe_snapshot_order"
    parsed = normalize_date_series(frame.get_column(snapshot_col))
    ordered = frame.with_columns(parsed.alias(order_column)).sort(
        order_column, nulls_last=True, maintain_order=True
    )
    groups = ordered.partition_by(order_column, maintain_order=True)
    group_count = len(groups)

    def closest_boundary(target: float, minimum: int, maximum: int) -> int:
        candidates = range(minimum, maximum + 1)
        return min(
            candidates,
            key=lambda index: (
                abs(sum(group.height for group in groups[:index]) - target),
                index,
            ),
        )

    if group_count >= 3:
        train_group_end = closest_boundary(ordered.height * 0.6, 1, group_count - 2)
        test_group_end = closest_boundary(
            ordered.height * 0.8,
            train_group_end + 1,
            group_count - 1,
        )
    elif group_count == 2:
        train_group_end, test_group_end = 1, 2
    else:
        train_group_end = test_group_end = group_count

    def combine(start: int, end: int) -> pl.DataFrame:
        if start == end:
            return ordered.clear().drop(order_column)
        return pl.concat(groups[start:end]).drop(order_column)

    return (
        combine(0, train_group_end),
        combine(train_group_end, test_group_end),
        combine(test_group_end, group_count),
    )


def _stratified_split(
    frame: pl.DataFrame, target_col: str
) -> tuple[pl.DataFrame, pl.DataFrame, None]:
    indices = np.arange(frame.height)
    train_indices, test_indices = train_test_split(
        indices,
        train_size=0.7,
        random_state=42,
        shuffle=True,
        stratify=frame.get_column(target_col).to_numpy(),
    )
    train = frame[sorted(int(index) for index in train_indices)]
    test = frame[sorted(int(index) for index in test_indices)]
    return train, test, None


def _candidate_frame(rules: list[RiskRule]) -> pl.DataFrame:
    rows = [
        {
            "rule_id": rule.rule_id,
            "origin": rule.origin,
            "conditions_json": json.dumps(
                [condition.model_dump(mode="json") for condition in rule.conditions],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        for rule in sorted(rules, key=lambda item: item.rule_id)
    ]
    if rows:
        return pl.DataFrame(rows)
    return pl.DataFrame(
        schema={
            "rule_id": pl.String,
            "origin": pl.String,
            "conditions_json": pl.String,
        }
    )


def _sorted_card(card: EvidenceCard) -> EvidenceCard:
    slices = tuple(
        sorted(
            (
                item.model_copy(
                    update={"slice_value": redact_segment_value(item.slice_value)}
                )
                if item.slice_type == "segment"
                else item
                for item in card.slices
            ),
            key=lambda item: (
                _SLICE_ORDER[item.slice_type],
                item.slice_value,
            ),
        )
    )
    return card.model_copy(
        update={
            "slices": slices,
            "limitations": tuple(
                sorted({redact_limitation(item) for item in card.limitations})
            ),
        }
    )


def _evidence_payload(
    cards: list[EvidenceCard], *, time_validation_enabled: bool
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for card in sorted((_sorted_card(card) for card in cards), key=evidence_sort_key):
        item = card.model_dump(mode="json")
        if not time_validation_enabled:
            item.pop("max_time_decay", None)
            item["slices"] = [
                slice_item
                for slice_item in item["slices"]
                if slice_item["slice_type"] != "time"
            ]
        payload.append(item)
    return payload


def _issue_payload(issue: Any) -> dict[str, Any]:
    payload = asdict(issue)
    if issue.code == "SINGLE_CLASS_SLICE":
        payload["message"] = "single-class slice detected"
    return payload


def _profile_payload(profile: DatasetProfile) -> dict[str, Any]:
    segment_sizes = list(profile.segment_counts.values())
    return {
        "dataset_id": profile.dataset_id,
        "row_count": profile.row_count,
        "feature_count": profile.feature_count,
        "positive_rate": profile.positive_rate,
        "segment_count": len(segment_sizes),
        "segment_size_min": min(segment_sizes, default=0),
        "segment_size_max": max(segment_sizes, default=0),
        "snapshot_min": profile.snapshot_min,
        "snapshot_max": profile.snapshot_max,
        "metadata_grade": profile.metadata_grade,
        "issues": [
            _issue_payload(issue)
            for issue in sorted(
                profile.issues,
                key=lambda item: (item.severity, item.code, item.family, item.message),
            )
        ],
    }


def _render_service_report(
    profile: DatasetProfile,
    cards: list[EvidenceCard],
    validation_limitations: tuple[str, ...],
) -> str:
    report = render_risk_report(profile, cards)
    if validation_limitations and not cards:
        replacement = "\n".join(
            f"- {limitation}" for limitation in sorted(validation_limitations)
        )
        report = report.replace("- None identified by configured checks", replacement)
    return report


def _attach_holdout(
    primary: list[EvidenceCard], holdout: list[EvidenceCard]
) -> list[EvidenceCard]:
    holdout_by_id = {card.rule.rule_id: card for card in holdout}
    combined: list[EvidenceCard] = []
    for card in primary:
        holdout_card = holdout_by_id.get(card.rule.rule_id)
        if holdout_card is None:
            combined.append(card)
            continue
        holdout_slice = SliceMetrics(
            slice_type="dataset",
            slice_value="Holdout",
            metrics=holdout_card.test,
        )
        holdout_limitations = tuple(
            f"holdout: {limitation}" for limitation in holdout_card.limitations
        )
        grade = max(
            (card.grade, holdout_card.grade),
            key=lambda item: _GRADE_ORDER[item],
        )
        combined.append(
            card.model_copy(
                update={
                    "slices": card.slices + (holdout_slice,),
                    "limitations": card.limitations + holdout_limitations,
                    "grade": grade,
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


class RiskProbeService:
    def __init__(self, *, config: ProjectConfig | Path, runs_dir: Path) -> None:
        self.config = (
            ProjectConfig.from_yaml(config) if isinstance(config, Path) else config
        )
        self.store = RunStore(runs_dir)

    def _dataset(self) -> ParquetDataset:
        return ParquetDataset(self.config.dataset.path)

    def _feature_names(self, dataset: ParquetDataset) -> list[str]:
        roles = {
            self.config.columns.entity,
            self.config.columns.snapshot,
            self.config.columns.segment,
            self.config.columns.target,
        }
        return sorted(name for name in dataset.schema().names() if name not in roles)

    def _partitions(
        self, dataset: ParquetDataset, feature_names: list[str]
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        columns = [
            self.config.columns.snapshot,
            self.config.columns.segment,
            self.config.columns.target,
            *feature_names,
        ]
        frame = dataset.collect(columns)
        if self.config.time_validation_enabled:
            return _time_split(frame, self.config.columns.snapshot)
        return _stratified_split(frame, self.config.columns.target)

    def _discover_from_train(
        self, train: pl.DataFrame, feature_names: list[str]
    ) -> list[RiskRule]:
        discovery_columns = [*feature_names, self.config.columns.target]
        sample = train.select(discovery_columns)
        if sample.height > _DISCOVERY_SAMPLE_LIMIT:
            sample = sample.sample(
                n=_DISCOVERY_SAMPLE_LIMIT,
                shuffle=True,
                seed=self.config.discovery.random_seed,
            )
        return discover_rules(
            sample,
            feature_names,
            self.config.columns.target,
            self.config.discovery,
        )

    def _validate(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        holdout: pl.DataFrame | None,
        rules: list[RiskRule],
    ) -> tuple[list[EvidenceCard], tuple[str, ...]]:
        involved_features = sorted(
            {
                condition.feature
                for rule in rules
                for condition in rule.conditions
            }
        )
        validation_columns = [
            *involved_features,
            self.config.columns.target,
            self.config.columns.segment,
        ]
        if self.config.time_validation_enabled:
            validation_columns.append(self.config.columns.snapshot)
        train_projection = train.select(validation_columns)
        test_projection = test.select(validation_columns)
        kwargs = {
            "target_col": self.config.columns.target,
            "segment_col": self.config.columns.segment,
            "snapshot_col": self.config.columns.snapshot,
            "segment_display_name": self.config.segment_display_name,
            "time_validation_enabled": self.config.time_validation_enabled,
            "config": self.config.validation,
            "metadata_grade": self.config.metadata_grade,
        }
        validation_limitations: list[str] = []
        if train_projection.is_empty() or not (
            train_projection.get_column(self.config.columns.target) == 1
        ).any():
            return [], ("Train partition has no positive target; validation unavailable",)
        if test_projection.is_empty() or not (
            test_projection.get_column(self.config.columns.target) == 1
        ).any():
            return [], ("Test partition has no positive target; validation unavailable",)

        cards = validate_rules(train_projection, test_projection, rules, **kwargs)
        if holdout is not None and not holdout.is_empty() and rules:
            holdout_projection = holdout.select(validation_columns)
            if (holdout_projection.get_column(self.config.columns.target) == 1).any():
                holdout_cards = validate_rules(
                    train_projection,
                    holdout_projection,
                    rules,
                    **kwargs,
                )
                cards = _attach_holdout(cards, holdout_cards)
            else:
                limitation = "Holdout partition has no positive target; validation unavailable"
                validation_limitations.append(limitation)
                cards = [
                    card.model_copy(
                        update={"limitations": card.limitations + (limitation,)}
                    )
                    for card in cards
                ]
        return cards, tuple(validation_limitations)

    def inspect(self) -> DatasetProfile:
        return profile_dataset(self._dataset(), self.config)

    def discover(self) -> list[RiskRule]:
        dataset = self._dataset()
        feature_names = self._feature_names(dataset)
        train, _, _ = self._partitions(dataset, feature_names)
        return self._discover_from_train(train, feature_names)

    def run(self) -> RunContext:
        dataset = self._dataset()
        data_fingerprint = _parquet_metadata_fingerprint(self.config.dataset.path)
        code_version = _package_version()
        context = self.store.create(
            self.config,
            data_fingerprint,
            code_version,
        )
        if context.is_existing:
            return context

        try:
            profile = profile_dataset(dataset, self.config)
            artifact_profile = replace(
                profile,
                dataset_id=_safe_dataset_id(profile.dataset_id),
            )
            feature_names = self._feature_names(dataset)
            train, test, holdout = self._partitions(dataset, feature_names)
            rules = self._discover_from_train(train, feature_names)
            cards, validation_limitations = self._validate(train, test, holdout, rules)
            split_rows = {
                "train": train.height,
                "test": test.height,
                "holdout": holdout.height if holdout is not None else 0,
            }
            limitations = sorted(
                {
                    *validation_limitations,
                    *(
                        redact_limitation(limitation)
                        for card in cards
                        for limitation in card.limitations
                    ),
                }
            )
            if profile.metadata_grade == "B":
                limitations = sorted({"label performance window unknown", *limitations})
            context.write_json(
                "manifest.json",
                {
                    "artifacts": list(_ARTIFACT_NAMES),
                    "code_version": code_version,
                    "data_fingerprint": data_fingerprint,
                    "dataset_id": artifact_profile.dataset_id,
                    "run_id": context.run_id,
                    "time_validation_enabled": self.config.time_validation_enabled,
                },
            )
            context.write_json(
                "metadata_report.json",
                {
                    "metadata_grade": profile.metadata_grade,
                    "limitations": limitations,
                    "split_rows": split_rows,
                    "time_validation_enabled": self.config.time_validation_enabled,
                },
            )
            context.write_json("data_profile.json", _profile_payload(artifact_profile))
            context.write_parquet("candidate_rules.parquet", _candidate_frame(rules))
            context.write_json(
                "evidence_cards.json",
                _evidence_payload(
                    cards,
                    time_validation_enabled=self.config.time_validation_enabled,
                ),
            )
            context.write_text(
                "risk_report.md",
                _render_service_report(
                    artifact_profile,
                    cards,
                    validation_limitations,
                ),
            )
            context.finalize()
            return context
        except BaseException:
            context.cleanup()
            raise
