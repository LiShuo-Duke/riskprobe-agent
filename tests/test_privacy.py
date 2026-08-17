import math
from pathlib import Path

import pytest

from riskprobe.privacy import (
    SegmentToken,
    UnsafePayloadError,
    assert_safe_payload,
    tokenize_segment,
)


def test_safe_aggregate_payload_and_explicit_segment_token_are_allowed() -> None:
    token = tokenize_segment("private-segment", namespace="dataset-1")
    payload = {
        "finding_count": 2,
        "rates": (0.1, 0.2),
        "segment": token,
        "nested": {"count": 10, "enabled": True},
    }

    assert_safe_payload(payload)
    assert isinstance(token, SegmentToken)
    assert token.token.startswith("segment-")
    assert "private-segment" not in token.token


def test_segment_tokens_are_stable_and_namespaced() -> None:
    first = tokenize_segment("real-label", namespace="dataset-a")
    repeated = tokenize_segment("real-label", namespace="dataset-a")
    other_namespace = tokenize_segment("real-label", namespace="dataset-b")

    assert first == repeated
    assert first != other_namespace


@pytest.mark.parametrize(
    "payload",
    [
        {"entity": "borrower-001"},
        {"outer": {"customer_id": "borrower-001"}},
        {"raw_rows": [{"value": 1}]},
        {"records": [{"score": 1}]},
        {"records": [{"score": 1}, {"score": 2}]},
        {"entityValue": "private-entity"},
        {"segmentName": "real-segment"},
        {"segment": "real-segment"},
        {"segment_label": "real-segment"},
    ],
)
def test_recursive_gate_rejects_high_risk_keys_rows_and_unredacted_segments(
    payload: dict[str, object],
) -> None:
    with pytest.raises(UnsafePayloadError, match="payload is not safe"):
        assert_safe_payload(payload)


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "/private/customer/data.parquet",
        "file:///private/customer/data.parquet",
        r"C:\\private\\customer\\data.parquet",
        "D:/private/customer/data.parquet",
        r"\\server\share\data.parquet",
        Path("relative.parquet"),
    ],
)
def test_gate_rejects_path_values(unsafe_value: object) -> None:
    with pytest.raises(UnsafePayloadError, match="payload is not safe"):
        assert_safe_payload({"summary": unsafe_value})


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "entity-123456",
        "customer_987654",
        "person@example.test",
        "123456789012",
    ],
)
def test_gate_rejects_entity_like_values_under_neutral_keys(unsafe_value: str) -> None:
    with pytest.raises(UnsafePayloadError, match="payload is not safe"):
        assert_safe_payload({"value": unsafe_value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_gate_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(UnsafePayloadError, match="payload is not safe"):
        assert_safe_payload({"rate": value})


def test_gate_error_never_echoes_private_input() -> None:
    secret = "/private/acme/customer-data.parquet"

    with pytest.raises(UnsafePayloadError) as exc_info:
        assert_safe_payload({"summary": secret})

    assert str(exc_info.value) == "payload is not safe"
    assert secret not in str(exc_info.value)
