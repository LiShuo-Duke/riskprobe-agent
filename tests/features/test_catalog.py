import polars as pl

from riskprobe.features.catalog import FeatureCatalog, check_window_invariants


def test_window_inversion_is_reported_at_feature_family_level() -> None:
    frame = pl.DataFrame({"order_cnt_7d": [5, 1], "order_cnt_30d": [3, 4]})
    catalog = FeatureCatalog.from_columns(
        frame.columns,
        {"order": ["order_"], "browse": ["browse_"]},
    )

    issues = check_window_invariants(frame, catalog)

    assert issues[0].code == "WINDOW_INVERSION"
    assert issues[0].family == "order"
    assert issues[0].affected_rows == 1


def test_unknown_feature_prefix_is_cataloged_as_unknown() -> None:
    catalog = FeatureCatalog.from_columns(
        ["order_cnt_7d", "mystery_cnt_30d"],
        {"order": ("order_",)},
    )

    unknown = next(spec for spec in catalog.features if spec.name == "mystery_cnt_30d")

    assert unknown.family == "unknown"
    assert unknown.window_days == 30
    assert unknown.aggregation == "mystery_cnt"
    assert unknown.value_type == "count"


def test_non_cumulative_ratio_and_amount_fields_do_not_report_inversions() -> None:
    frame = pl.DataFrame(
        {
            "order_cancel_rate_7d": [0.8],
            "order_cancel_rate_30d": [0.2],
            "order_amount_7d": [100.0],
            "order_amount_30d": [50.0],
        }
    )
    catalog = FeatureCatalog.from_columns(frame.columns, {"order": ("order_",)})

    issues = check_window_invariants(frame, catalog)

    assert issues == ()


def test_only_matching_family_and_aggregation_are_compared() -> None:
    frame = pl.DataFrame(
        {
            "order_paid_cnt_7d": [5],
            "order_refund_cnt_30d": [3],
            "browse_paid_cnt_30d": [3],
        }
    )
    catalog = FeatureCatalog.from_columns(
        frame.columns,
        {"order": ("order_",), "browse": ("browse_",)},
    )

    issues = check_window_invariants(frame, catalog)

    assert issues == ()


def test_non_cumulative_average_count_does_not_report_inversion() -> None:
    frame = pl.DataFrame(
        {
            "order_avg_cnt_7d": [5.0],
            "order_avg_cnt_30d": [3.0],
        }
    )
    catalog = FeatureCatalog.from_columns(frame.columns, {"order": ("order_",)})

    issues = check_window_invariants(frame, catalog)

    assert issues == ()


def test_window_before_aggregation_is_compared() -> None:
    frame = pl.DataFrame(
        {
            "order_7d_cnt": [5],
            "order_30d_cnt": [3],
        }
    )
    catalog = FeatureCatalog.from_columns(frame.columns, {"order": ("order_",)})

    issues = check_window_invariants(frame, catalog)

    assert len(issues) == 1
    assert issues[0].affected_rows == 1


def test_same_window_aliases_are_never_compared() -> None:
    frame = pl.DataFrame(
        {
            "order_cnt_7d": [5],
            "order_7d_cnt": [1],
        }
    )
    catalog = FeatureCatalog.from_columns(frame.columns, {"order": ("order_",)})

    issues = check_window_invariants(frame, catalog)

    assert issues == ()


def test_every_alias_is_compared_with_aliases_in_the_next_unique_window() -> None:
    frame = pl.DataFrame(
        {
            "order_cnt_7d": [5],
            "order_7d_cnt": [None],
            "order_cnt_30d": [None],
            "order_30d_cnt": [3],
        }
    )
    catalog = FeatureCatalog.from_columns(frame.columns, {"order": ("order_",)})

    issues = check_window_invariants(frame, catalog)

    assert len(issues) == 1
    assert issues[0].affected_rows == 1
