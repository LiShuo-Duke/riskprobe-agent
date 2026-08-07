"""Stable, role-aware feature batches for wide local Parquet datasets."""

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from riskprobe.features.catalog import FeatureCatalog


_DEFAULT_ROLE_COLUMNS = ("entity_id", "snapshot_date", "segment", "target")


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    index: int
    features: tuple[str, ...]
    required_columns: tuple[str, ...]


def _feature_names(
    schema: Iterable[str] | pl.Schema,
    catalog: FeatureCatalog | None,
) -> tuple[str, ...]:
    if catalog is not None:
        return tuple(spec.name for spec in catalog.features)
    if isinstance(schema, pl.Schema):
        return tuple(schema.names())
    return tuple(schema)


def plan_feature_batches(
    schema: Iterable[str] | pl.Schema,
    catalog: FeatureCatalog | None = None,
    *,
    batch_size: int = 64,
    role_columns: tuple[str, ...] = _DEFAULT_ROLE_COLUMNS,
) -> tuple[FeatureBatch, ...]:
    """Split selected feature names into ordered batches without role columns."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    feature_names = tuple(name for name in _feature_names(schema, catalog) if name not in role_columns)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must be unique")
    return tuple(
        FeatureBatch(
            index=index,
            features=feature_names[start : start + batch_size],
            required_columns=role_columns,
        )
        for index, start in enumerate(range(0, len(feature_names), batch_size), start=1)
    )
