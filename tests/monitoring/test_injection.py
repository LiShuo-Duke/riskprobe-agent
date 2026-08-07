import pytest

from riskprobe.monitoring.injection import DriftScenario, evaluate_alerts, inject_drift
from riskprobe.synthetic import generate_behavior_dataset


@pytest.mark.parametrize(
    ("drift_type", "target"),
    [
        ("missingness", "browse_pv_30d"),
        ("numeric_shift", "browse_pv_30d"),
        ("population_shift", "institution"),
        ("label_shift", "target"),
        ("schema", "browse_pv_30d"),
        ("rule_decay", "browse_pv_30d"),
    ],
)
def test_all_drift_scenarios_are_reproducible_and_record_machine_readable_truth(
    drift_type: str, target: str
) -> None:
    frame, _ = generate_behavior_dataset(1_000, seed=42)
    scenario = DriftScenario(
        scenario_id=f"{drift_type}-browse",
        drift_type=drift_type,
        target=target,
        magnitude=0.30,
        institution="B",
    )

    first = inject_drift(frame, scenario, seed=7)
    second = inject_drift(frame, scenario, seed=7)

    assert first.frame.equals(second.frame)
    assert frame.equals(generate_behavior_dataset(1_000, seed=42)[0])
    assert first.truth.expected_alert_type == drift_type.replace("numeric_shift", "distribution").replace(
        "population_shift", "population"
    ).replace("label_shift", "label")
    assert first.truth.expected_scope_value == target


def test_alert_evaluation_matches_type_and_scope_value() -> None:
    frame, _ = generate_behavior_dataset(1_000, seed=42)
    injected = inject_drift(
        frame,
        DriftScenario(
            scenario_id="missing-browse",
            drift_type="missingness",
            target="browse_pv_30d",
            magnitude=0.30,
            institution="B",
        ),
        seed=7,
    )

    score = evaluate_alerts((), (injected.truth,))

    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.false_positive_rate == 0.0
    assert score.top_k_root_cause_hit == 0.0


def test_numeric_shift_preserves_existing_missingness() -> None:
    frame, _ = generate_behavior_dataset(1_000, seed=42)

    injected = inject_drift(
        frame,
        DriftScenario(
            scenario_id="numeric-browse",
            drift_type="numeric_shift",
            target="browse_pv_30d",
            magnitude=0.30,
        ),
        seed=7,
    )

    assert injected.frame.get_column("browse_pv_30d").null_count() == 0
    assert injected.frame.get_column("browse_pv_30d").mean() > frame.get_column("browse_pv_30d").mean()
