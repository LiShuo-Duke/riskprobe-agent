from pathlib import Path

import polars as pl
import pytest

from riskprobe.io.parquet import MissingColumnsError, ParquetDataset


def test_collect_reads_only_requested_columns(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"id": [1, 2], "target": [0, 1], "unused": [9, 9]}).write_parquet(path)
    dataset = ParquetDataset(path)

    result = dataset.collect(["id", "target"])

    assert result.columns == ["id", "target"]
    assert result.to_dict(as_series=False) == {"id": [1, 2], "target": [0, 1]}


def test_schema_reports_parquet_columns_without_collecting_data(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"id": [1], "score": [0.5]}).write_parquet(path)

    schema = ParquetDataset(path).schema()

    assert schema.names() == ["id", "score"]


def test_scan_returns_lazy_frame_with_requested_column_order(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"id": [1], "score": [0.5]}).write_parquet(path)

    result = ParquetDataset(path).scan(["score", "id"])

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().columns == ["score", "id"]


def test_missing_columns_raise_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(path)

    with pytest.raises(MissingColumnsError) as error:
        ParquetDataset(path).collect(["target", "score"])

    assert error.value.missing == ("target", "score")
    assert str(error.value) == "missing Parquet columns: target, score"


def test_dataset_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        ParquetDataset(tmp_path / "missing.parquet")


def test_dataset_rejects_non_parquet_path(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.parquet"):
        ParquetDataset(path)


def test_scan_rejects_duplicate_requested_columns(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(path)

    with pytest.raises(ValueError, match="duplicate requested columns: id"):
        ParquetDataset(path).scan(["id", "id"])
