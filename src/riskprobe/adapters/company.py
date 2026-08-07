"""Read-only schema and metadata checks for local company Parquet datasets."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from riskprobe.batching import FeatureBatch, plan_feature_batches
from riskprobe.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class CompanyPreflight:
    row_count: int
    feature_count: int
    feature_family_counts: dict[str, int]
    batches: tuple[FeatureBatch, ...]
    label_rate: float
    snapshot_min: str | None
    snapshot_max: str | None
    segment_count: int
    metadata_grade: str
    limitations: tuple[str, ...]

    @property
    def batch_count(self) -> int:
        return len(self.batches)


def _require_role_types(schema: pl.Schema, config: ProjectConfig) -> None:
    roles = config.columns
    missing = [name for name in (roles.entity, roles.snapshot, roles.segment, roles.target) if name not in schema]
    if missing:
        raise ValueError("Parquet schema is missing required role columns")
    snapshot_type = str(schema[roles.snapshot])
    target_type = str(schema[roles.target])
    entity_type = str(schema[roles.entity])
    segment_type = str(schema[roles.segment])
    if not snapshot_type.startswith(("Date", "Datetime")):
        raise ValueError("snapshot column must have a date or datetime type")
    if not target_type.startswith(("Int", "UInt", "Float", "Decimal")):
        raise ValueError("target column must have a numeric type")
    if not entity_type.startswith(("String", "Categorical", "Enum")):
        raise ValueError("entity column must have a string-like type")
    if not segment_type.startswith(("String", "Categorical", "Enum")):
        raise ValueError("segment column must have a string-like type")


def _family_counts(columns: list[str], config: ProjectConfig) -> dict[str, int]:
    return {
        family: sum(
            column.startswith(prefix)
            for column in columns
            for prefix in prefixes
        )
        for family, prefixes in config.features.families.items()
    }


def preflight_company_dataset(config: ProjectConfig) -> CompanyPreflight:
    """Inspect only Parquet metadata and role columns; never write the source file."""
    path = Path(config.dataset.path)
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    _require_role_types(schema, config)
    role_columns = (
        config.columns.entity,
        config.columns.snapshot,
        config.columns.segment,
        config.columns.target,
    )
    features = config.features.select_columns(schema.names(), role_columns)
    if not features:
        raise ValueError("Parquet schema has no configured feature columns")
    aggregate = (
        lazy.select(
            pl.len().alias("row_count"),
            pl.col(config.columns.target).cast(pl.Float64).mean().alias("label_rate"),
            pl.col(config.columns.snapshot).min().cast(pl.String).alias("snapshot_min"),
            pl.col(config.columns.snapshot).max().cast(pl.String).alias("snapshot_max"),
            pl.col(config.columns.segment).n_unique().alias("segment_count"),
        )
        .collect()
        .row(0, named=True)
    )
    batches = plan_feature_batches(
        features,
        batch_size=64,
        role_columns=role_columns,
    )
    limitations = (
        "label performance window unknown",
        "customer feature cutoff is not a strict production OOT claim",
    ) if config.metadata_grade == "B" else ()
    return CompanyPreflight(
        row_count=int(aggregate["row_count"]),
        feature_count=len(features),
        feature_family_counts=_family_counts(features, config),
        batches=batches,
        label_rate=float(aggregate["label_rate"] or 0.0),
        snapshot_min=aggregate["snapshot_min"],
        snapshot_max=aggregate["snapshot_max"],
        segment_count=int(aggregate["segment_count"]),
        metadata_grade=config.metadata_grade,
        limitations=limitations,
    )
