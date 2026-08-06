from datetime import date
from pathlib import Path

import polars as pl
import pytest

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset
from riskprobe.profiling import DataContractError, profile_dataset


def _write_dataset(tmp_path: Path, data: dict[str, list[object]]) -> ParquetDataset:
    path = tmp_path / "sample.parquet"
    pl.DataFrame(data).write_parquet(path)
    return ParquetDataset(path)


def _config_with(
    config: ProjectConfig,
    *,
    performance_window_days: int | None = None,
    time_validation_enabled: bool | None = None,
) -> ProjectConfig:
    updates: dict[str, object] = {
        "target": config.target.model_copy(
            update={"performance_window_days": performance_window_days}
        )
    }
    if time_validation_enabled is not None:
        updates["time_validation_enabled"] = time_validation_enabled
    return config.model_copy(update=updates)


def test_profile_is_grade_b_when_performance_window_is_unknown(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b", "c", "d"],
            "snapshot_date": ["2026-01-01"] * 4,
            "institution": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
            "order_cnt_7d": [0, 1, 2, 3],
            "order_cnt_30d": [1, 2, 3, 4],
        },
    )

    profile = profile_dataset(dataset, synthetic_config)

    assert profile.metadata_grade == "B"
    assert profile.row_count == 4
    assert profile.snapshot_min == date(2026, 1, 1)
    assert profile.snapshot_max == date(2026, 1, 1)
    assert "LABEL_PERFORMANCE_WINDOW_UNKNOWN" in {issue.code for issue in profile.issues}


def test_enabled_time_validation_rejects_invalid_snapshot(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b"],
            "snapshot_date": ["2026-01-01", "not-a-date"],
            "institution": ["A", "B"],
            "target": [0, 1],
            "order_cnt_7d": [0, 1],
        },
    )

    with pytest.raises(DataContractError, match="invalid dates"):
        profile_dataset(dataset, synthetic_config)


def test_profile_accepts_categorical_snapshots_with_original_nulls(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    path = tmp_path / "categorical-snapshots.parquet"
    pl.DataFrame(
        {
            "entity_id": ["a", "b", "c"],
            "snapshot_date": ["2026-01-01", None, "2026-02-01"],
            "institution": ["A", "A", "B"],
            "target": [0, 1, 0],
            "order_cnt_7d": [0, 1, 2],
        }
    ).with_columns(pl.col("snapshot_date").cast(pl.Categorical)).write_parquet(path)
    dataset = ParquetDataset(path)

    profile = profile_dataset(dataset, synthetic_config)

    assert profile.snapshot_min == date(2026, 1, 1)
    assert profile.snapshot_max == date(2026, 2, 1)


@pytest.mark.parametrize("missing_role", ["entity_id", "snapshot_date", "institution", "target"])
def test_missing_role_column_raises_data_contract_error(
    tmp_path: Path,
    synthetic_config: ProjectConfig,
    missing_role: str,
) -> None:
    data: dict[str, list[object]] = {
        "entity_id": ["a", "b"],
        "snapshot_date": ["2026-01-01", "2026-01-02"],
        "institution": ["A", "B"],
        "target": [0, 1],
        "order_cnt_7d": [0, 1],
    }
    del data[missing_role]
    dataset = _write_dataset(tmp_path, data)

    with pytest.raises(DataContractError, match=missing_role):
        profile_dataset(dataset, synthetic_config)


def test_single_class_segment_is_warned_without_blocking_profile(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b", "c", "d"],
            "snapshot_date": ["2026-01-01"] * 4,
            "institution": ["A", "A", "B", "B"],
            "target": [0, 0, 0, 1],
            "order_cnt_7d": [0, 1, 2, 3],
        },
    )

    profile = profile_dataset(dataset, _config_with(synthetic_config, performance_window_days=30))

    issue = next(issue for issue in profile.issues if issue.code == "SINGLE_CLASS_SLICE")
    assert issue.severity == "warning"
    assert issue.family == "institution"
    assert issue.affected_rows == 2
    assert profile.row_count == 4


def test_disabled_time_validation_accepts_unparsed_non_null_snapshots(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b"],
            "snapshot_date": ["not-a-date", "still-not-a-date"],
            "institution": ["A", "B"],
            "target": [0, 1],
            "order_cnt_7d": [0, 1],
        },
    )
    config = _config_with(
        synthetic_config,
        performance_window_days=30,
        time_validation_enabled=False,
    )

    profile = profile_dataset(dataset, config)

    assert profile.snapshot_min is None
    assert profile.snapshot_max is None
    assert all(issue.family != "snapshot_date" for issue in profile.issues)


def test_disabled_time_validation_rejects_all_null_snapshots(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b"],
            "snapshot_date": [None, None],
            "institution": ["A", "B"],
            "target": [0, 1],
            "order_cnt_7d": [0, 1],
        },
    )
    config = _config_with(synthetic_config, time_validation_enabled=False)

    with pytest.raises(DataContractError, match="snapshot_date"):
        profile_dataset(dataset, config)


@pytest.mark.parametrize("target_values", [[0, None], [None, None]])
def test_null_target_values_raise_data_contract_error(
    tmp_path: Path,
    synthetic_config: ProjectConfig,
    target_values: list[int | None],
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b"],
            "snapshot_date": ["2026-01-01", "2026-01-02"],
            "institution": ["A", "A"],
            "target": target_values,
            "order_cnt_7d": [0, 1],
        },
    )

    with pytest.raises(DataContractError, match=r"target.*null"):
        profile_dataset(dataset, synthetic_config)


def test_non_binary_target_values_raise_data_contract_error(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b", "c"],
            "snapshot_date": ["2026-01-01"] * 3,
            "institution": ["A"] * 3,
            "target": [0, 1, 2],
            "order_cnt_7d": [0, 1, 2],
        },
    )

    with pytest.raises(DataContractError, match=r"target.*0 and 1"):
        profile_dataset(dataset, synthetic_config)


@pytest.mark.parametrize("target_value", [0, 1])
def test_single_class_zero_or_one_segment_is_warned_without_blocking(
    tmp_path: Path,
    synthetic_config: ProjectConfig,
    target_value: int,
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["a", "b"],
            "snapshot_date": ["2026-01-01", "2026-01-02"],
            "institution": ["A", "A"],
            "target": [target_value, target_value],
            "order_cnt_7d": [0, 1],
        },
    )

    profile = profile_dataset(
        dataset,
        _config_with(synthetic_config, performance_window_days=30),
    )

    segment_issues = [
        issue
        for issue in profile.issues
        if issue.code == "SINGLE_CLASS_SLICE" and issue.family == "institution"
    ]
    assert len(segment_issues) == 1
    assert segment_issues[0].affected_rows == 2
    assert profile.positive_rate == float(target_value)


def test_profile_contains_aggregates_not_entity_values(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {
            "entity_id": ["sensitive-a", "sensitive-b"],
            "snapshot_date": ["2026-01-01", "2026-02-01"],
            "institution": ["A", "B"],
            "target": [0, 1],
            "order_cnt_7d": [0, 1],
        },
    )

    profile = profile_dataset(dataset, _config_with(synthetic_config, performance_window_days=30))

    assert profile.feature_count == 1
    assert profile.positive_rate == 0.5
    assert profile.segment_counts == {"A": 1, "B": 1}
    assert "sensitive-a" not in repr(profile)
    assert "sensitive-b" not in repr(profile)
