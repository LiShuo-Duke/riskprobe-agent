import pytest

from riskprobe.privacy import UnsafePayloadError, assert_safe_payload, suppress_small_groups


def test_small_group_is_removed_from_tool_output() -> None:
    records = [{"institution": "A", "count": 99}, {"institution": "B", "count": 101}]
    assert suppress_small_groups(records, "count", 100) == [{"institution": "B", "count": 101}]


def test_payload_gate_rejects_nested_raw_detail_field() -> None:
    with pytest.raises(UnsafePayloadError):
        assert_safe_payload({"result": {"entity_id": "never-return"}})
