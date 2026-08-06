import polars as pl


def normalize_date_series(values: pl.Series) -> pl.Series:
    """Normalize date-like values while preserving original nulls."""
    if values.dtype == pl.Date:
        parsed = values
    elif isinstance(values.dtype, pl.Datetime):
        parsed = values.cast(pl.Date)
    else:
        try:
            parsed = values.cast(pl.String).str.to_date(strict=False, exact=True)
        except pl.exceptions.ComputeError as error:
            raise ValueError("date values contain invalid dates") from error

    if parsed.null_count() != values.null_count():
        raise ValueError("date values contain invalid dates")
    return parsed
