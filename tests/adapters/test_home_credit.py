import polars as pl

from riskprobe.adapters.home_credit import HomeCreditPaths, prepare_home_credit


def test_prepare_home_credit_aggregates_history_without_post_target_fields(tmp_path) -> None:
    pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [1, 0],
            "NAME_INCOME_TYPE": ["Working", "Pensioner"],
            "DAYS_BIRTH": [-12000, -20000],
            "AMT_INCOME_TOTAL": [100000.0, 80000.0],
        }
    ).write_csv(tmp_path / "application_train.csv")
    pl.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "DAYS_DECISION": [-10, -100, -20],
            "AMT_APPLICATION": [1000.0, 2000.0, 500.0],
            "AMT_CREDIT": [900.0, 1800.0, 500.0],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "NOT_WHITELISTED": ["private", "private", "private"],
        }
    ).write_csv(tmp_path / "previous_application.csv")
    output = tmp_path / "home_credit.parquet"

    result = prepare_home_credit(HomeCreditPaths.from_directory(tmp_path), output)
    frame = pl.read_parquet(output)

    assert result.rows == 2
    assert result.source_tables == ("application_train", "previous_application")
    assert "prev_application_cnt_30d" in frame.columns
    assert "prev_refused_rate_all" in frame.columns
    assert "TARGET" not in frame.columns
    assert "NOT_WHITELISTED" not in frame.columns
    assert frame.columns[:4] == [
        "entity_id",
        "target",
        "customer_segment",
        "snapshot_date",
    ]
    assert frame.get_column("snapshot_date").unique().to_list() == [
        "public_relative_reference"
    ]


def test_home_credit_paths_requires_application_and_one_history_table(tmp_path) -> None:
    pl.DataFrame({"SK_ID_CURR": [1], "TARGET": [0]}).write_csv(
        tmp_path / "application_train.csv"
    )

    try:
        HomeCreditPaths.from_directory(tmp_path)
    except ValueError as error:
        assert "history" in str(error)
    else:
        raise AssertionError("expected a missing history table error")
