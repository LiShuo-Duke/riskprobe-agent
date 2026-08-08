import pytest

from riskprobe.privacy import UnsafePayloadError, assert_safe_payload, redact_payload, suppress_small_groups


def test_small_group_is_removed_from_tool_output() -> None:
    records = [{"institution": "A", "count": 99}, {"institution": "B", "count": 101}]
    assert suppress_small_groups(records, "count", 100) == [{"institution": "B", "count": 101}]




def test_redact_payload_tokenizes_protocol_enum_data_values() -> None:
    payload = redact_payload(
        {
            "dataset_id": "dataset",
            "segment": "institution",
            "feature": "feature",
            "rule_id": "rule",
        }
    )

    assert payload["dataset_id"] != "dataset"
    assert payload["segment"] != "institution"
    assert payload["feature"] != "feature"
    assert payload["rule_id"] != "rule"
    assert all(value.startswith("tok_") for value in payload.values())
    with pytest.raises(UnsafePayloadError):
        assert_safe_payload({"result": {"entity_id": "never-return"}})


def test_payload_gate_rejects_path_like_and_identifier_strings() -> None:
    with pytest.raises(UnsafePayloadError):
        assert_safe_payload({"message": "/Users/test/private.parquet"})
    with pytest.raises(UnsafePayloadError):
        assert_safe_payload({"message": "entity_id=customer-123"})
