from riskprobe.batching import plan_feature_batches


def test_977_features_are_split_without_role_columns() -> None:
    features = [f"order_f_{index:04d}" for index in range(977)]
    batches = plan_feature_batches(features, batch_size=64)
    flattened = [name for batch in batches for name in batch.features]

    assert len(batches) == 16
    assert flattened == features
    assert all(len(batch.features) <= 64 for batch in batches)
    assert all(
        batch.required_columns == ("entity_id", "snapshot_date", "segment", "target")
        for batch in batches
    )


def test_feature_batches_exclude_role_columns_from_features() -> None:
    batches = plan_feature_batches(
        ["entity_id", "snapshot_date", "segment", "target", "order_f_0001"],
        role_columns=("entity_id", "snapshot_date", "segment", "target"),
    )

    assert batches[0].features == ("order_f_0001",)
