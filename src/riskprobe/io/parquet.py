from collections.abc import Iterable
from pathlib import Path

import polars as pl


class MissingColumnsError(ValueError):
    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(f"missing Parquet columns: {', '.join(self.missing)}")


class ParquetDataset:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if self._path.suffix.lower() != ".parquet":
            raise ValueError("Parquet dataset path must have a .parquet extension")
        if not self._path.exists():
            raise FileNotFoundError(self._path)

    def schema(self) -> pl.Schema:
        return pl.scan_parquet(self._path).collect_schema()

    def scan(self, columns: Iterable[str]) -> pl.LazyFrame:
        requested = tuple(columns)
        seen: set[str] = set()
        duplicate_names: set[str] = set()
        duplicates: list[str] = []
        for column in requested:
            if column in seen and column not in duplicate_names:
                duplicates.append(column)
                duplicate_names.add(column)
            seen.add(column)
        if duplicates:
            raise ValueError(f"duplicate requested columns: {', '.join(duplicates)}")

        source = pl.scan_parquet(self._path)
        available = source.collect_schema().names()
        missing = tuple(column for column in requested if column not in available)
        if missing:
            raise MissingColumnsError(missing)
        return source.select(requested)

    def collect(self, columns: Iterable[str]) -> pl.DataFrame:
        return self.scan(columns).collect()
