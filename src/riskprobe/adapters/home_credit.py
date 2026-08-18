"""Local Home Credit CSV aggregation using Polars lazy frames only."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl


_APPLICATION_COLUMNS = (
    "SK_ID_CURR",
    "TARGET",
    "NAME_INCOME_TYPE",
    "DAYS_BIRTH",
    "AMT_INCOME_TOTAL",
)
_HISTORY_COLUMNS = {
    "previous_application": (
        "SK_ID_CURR",
        "DAYS_DECISION",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "NAME_CONTRACT_STATUS",
    ),
    "installments_payments": (
        "SK_ID_CURR",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ),
    "POS_CASH_balance": ("SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"),
    "credit_card_balance": (
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_PAYMENT_CURRENT",
        "SK_DPD",
    ),
    "bureau": (
        "SK_ID_CURR",
        "DAYS_CREDIT",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "CREDIT_ACTIVE",
    ),
}


@dataclass(frozen=True, slots=True)
class HomeCreditPaths:
    application_train: Path
    history_tables: tuple[tuple[str, Path], ...]

    @classmethod
    def from_directory(cls, directory: Path) -> "HomeCreditPaths":
        directory = Path(directory)
        application_train = directory / "application_train.csv"
        if not application_train.is_file():
            raise ValueError("Home Credit input must include application_train.csv")
        history_tables = tuple(
            (name, directory / f"{name}.csv")
            for name in _HISTORY_COLUMNS
            if (directory / f"{name}.csv").is_file()
        )
        if not history_tables:
            raise ValueError("Home Credit input must include at least one history table")
        return cls(application_train=application_train, history_tables=history_tables)


@dataclass(frozen=True, slots=True)
class HomeCreditPreparationResult:
    rows: int
    columns: int
    source_tables: tuple[str, ...]
    feature_families: tuple[str, ...]


def _lazy_csv(path: Path, columns: tuple[str, ...]) -> pl.LazyFrame:
    return pl.scan_csv(path).select(columns)


def _previous_aggregates(history: pl.LazyFrame) -> list[pl.LazyFrame]:
    aggregates: list[pl.LazyFrame] = []
    for window in (30, 90, 365):
        aggregates.append(
            history.filter(pl.col("DAYS_DECISION") >= -window)
            .group_by("SK_ID_CURR")
            .agg(
                pl.len().alias(f"prev_application_cnt_{window}d"),
                pl.col("AMT_APPLICATION").sum().alias(f"prev_application_amt_{window}d"),
                pl.col("AMT_CREDIT").sum().alias(f"prev_credit_amt_{window}d"),
            )
        )
    aggregates.append(
        history.group_by("SK_ID_CURR").agg(
            (pl.col("NAME_CONTRACT_STATUS") == "Refused")
            .mean()
            .alias("prev_refused_rate_all")
        )
    )
    return aggregates


def _installment_aggregates(history: pl.LazyFrame) -> list[pl.LazyFrame]:
    aggregates: list[pl.LazyFrame] = []
    for window in (30, 90, 365):
        aggregates.append(
            history.filter(pl.col("DAYS_INSTALMENT") >= -window)
            .group_by("SK_ID_CURR")
            .agg(
                pl.len().alias(f"inst_record_cnt_{window}d"),
                (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT"))
                .clip(lower_bound=0)
                .mean()
                .alias(f"inst_overdue_days_mean_{window}d"),
                (pl.col("AMT_PAYMENT") < pl.col("AMT_INSTALMENT"))
                .mean()
                .alias(f"inst_underpayment_rate_{window}d"),
            )
        )
    return aggregates


def _pos_aggregates(history: pl.LazyFrame) -> list[pl.LazyFrame]:
    aggregates: list[pl.LazyFrame] = []
    for months in (3, 6, 12):
        aggregates.append(
            history.filter(pl.col("MONTHS_BALANCE") >= -months)
            .group_by("SK_ID_CURR")
            .agg(
                pl.col("MONTHS_BALANCE").n_unique().alias(f"pos_active_month_cnt_{months}m"),
                pl.col("SK_DPD").mean().alias(f"pos_dpd_mean_{months}m"),
                pl.col("SK_DPD").max().alias(f"pos_dpd_max_{months}m"),
            )
        )
    return aggregates


def _credit_card_aggregates(history: pl.LazyFrame) -> list[pl.LazyFrame]:
    aggregates: list[pl.LazyFrame] = []
    for months in (3, 6, 12):
        aggregates.append(
            history.filter(pl.col("MONTHS_BALANCE") >= -months)
            .group_by("SK_ID_CURR")
            .agg(
                pl.col("AMT_BALANCE").mean().alias(f"cc_balance_mean_{months}m"),
                pl.when(pl.col("AMT_BALANCE").sum() > 0)
                .then(pl.col("AMT_PAYMENT_CURRENT").sum() / pl.col("AMT_BALANCE").sum())
                .otherwise(None)
                .alias(f"cc_payment_balance_ratio_{months}m"),
                pl.col("SK_DPD").max().alias(f"cc_dpd_max_{months}m"),
            )
        )
    return aggregates


def _bureau_aggregates(history: pl.LazyFrame) -> list[pl.LazyFrame]:
    aggregates: list[pl.LazyFrame] = []
    for window in (90, 365):
        aggregates.append(
            history.filter(pl.col("DAYS_CREDIT") >= -window)
            .group_by("SK_ID_CURR")
            .agg(
                pl.len().alias(f"bureau_query_cnt_{window}d"),
                (pl.col("CREDIT_ACTIVE") == "Active")
                .sum()
                .alias(f"bureau_active_credit_cnt_{window}d"),
                pl.when(pl.col("AMT_CREDIT_SUM").sum() > 0)
                .then(pl.col("AMT_CREDIT_SUM_DEBT").sum() / pl.col("AMT_CREDIT_SUM").sum())
                .otherwise(None)
                .alias(f"bureau_debt_credit_ratio_{window}d"),
            )
        )
    return aggregates


def _history_aggregates(name: str, history: pl.LazyFrame) -> list[pl.LazyFrame]:
    aggregators = {
        "previous_application": _previous_aggregates,
        "installments_payments": _installment_aggregates,
        "POS_CASH_balance": _pos_aggregates,
        "credit_card_balance": _credit_card_aggregates,
        "bureau": _bureau_aggregates,
    }
    return aggregators[name](history)


def prepare_home_credit(
    paths: HomeCreditPaths, output_path: Path, seed: int = 42
) -> HomeCreditPreparationResult:
    """Aggregate whitelisted public CSV columns into the RiskProbe Parquet contract."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    base = (
        _lazy_csv(paths.application_train, _APPLICATION_COLUMNS)
        .select(
            pl.col("SK_ID_CURR").alias("entity_id"),
            pl.col("TARGET").alias("target"),
            pl.col("NAME_INCOME_TYPE").alias("customer_segment"),
        )
        .with_columns(pl.lit("public_relative_reference").alias("snapshot_date"))
    )
    prepared = base
    feature_families: list[str] = []
    for name, path in paths.history_tables:
        history = _lazy_csv(path, _HISTORY_COLUMNS[name])
        for aggregate in _history_aggregates(name, history):
            prepared = prepared.join(aggregate, left_on="entity_id", right_on="SK_ID_CURR", how="left")
        feature_families.append(name)
    frame = prepared.sort("entity_id").collect()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="zstd", statistics=True)
    return HomeCreditPreparationResult(
        rows=frame.height,
        columns=frame.width,
        source_tables=("application_train", *(name for name, _ in paths.history_tables)),
        feature_families=tuple(feature_families),
    )
