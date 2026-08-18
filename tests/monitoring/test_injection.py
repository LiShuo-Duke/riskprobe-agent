import pytest

from riskprobe.monitoring.detection import detect_anomalies
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.injection import DriftScenario, evaluate_alerts, inject_drift
from riskprobe.monitoring.reference import build_reference_snapshot
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
    assert score.false_positive_rate is None
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


def test_rule_decay_uses_explicit_rule_condition_and_role_columns() -> None:
    import polars as pl
    from riskprobe.models import Condition

    frame = pl.DataFrame(
        {"cohort": ["x", "x", "y", "y"], "outcome": [1, 0, 1, 0], "score": [9.0, 8.0, 1.0, 0.0]}
    )
    injected = inject_drift(
        frame,
        DriftScenario(
            scenario_id="custom-rule",
            drift_type="rule_decay",
            target="rule-7",
            magnitude=1.0,
            institution="x",
            target_column="outcome",
            segment_column="cohort",
            rule_conditions=(Condition(feature="score", operator=">", value=5.0),),
        ),
        seed=1,
    )

    assert injected.frame.get_column("outcome").to_list() == [0, 0, 1, 0]
    assert injected.truth.scene_key == "rule-7"


def test_evaluate_alerts_does_not_call_fdr_false_positive_rate() -> None:
    from riskprobe.monitoring.models import Alert

    alert = Alert(
        alert_id="a1", alert_type="missingness", severity="warning", scope="feature",
        scope_value="feature-token", metric="missing_rate", reference_value=0.0,
        current_value=0.5, delta=0.5, evidence={}
    )
    truth = DriftScenario(
        scenario_id="s1", drift_type="missingness", target="feature-token", magnitude=0.2
    )
    score = evaluate_alerts(
        (alert,),
        (inject_drift(__import__("polars").DataFrame({"feature-token": [1.0]}), truth, seed=1).truth,),
    )
    assert score.false_positive_rate is None
    assert score.false_discovery_rate == 0.0


def test_top_k_uses_diagnosis_root_cause_not_alert_order() -> None:
    from riskprobe.monitoring.models import Alert, Diagnosis, RootCause

    alert = Alert(
        alert_id="a1", alert_type="missingness", severity="warning", scope="feature",
        scope_value="feature-token", metric="missing_rate", reference_value=0.0,
        current_value=0.5, delta=0.5, evidence={}
    )
    truth = inject_drift(
        __import__("polars").DataFrame({"feature-token": [1.0]}),
        DriftScenario(scenario_id="s1", drift_type="missingness", target="feature-token", magnitude=0.2),
        seed=1,
    ).truth
    diagnosis = Diagnosis(
        snapshot_id="s", alerts=(alert,),
        root_causes=(RootCause(dimension="feature", value="feature-token", contribution=1.0, rank=1, evidence={}),),
        created_at="1970-01-01T00:00:00Z",
    )
    score = evaluate_alerts((alert,), (truth,), diagnoses=(diagnosis,))
    assert score.top_k_root_cause_hit == 1.0


@pytest.mark.parametrize(
    ("drift_type", "magnitude"),
    [("missingness", 0.80), ("numeric_shift", 5.00)],
)
def test_detect_diagnose_evaluate_chain_matches_feature_truth(
    reference_fixture, drift_type: str, magnitude: float
) -> None:
    reference = build_reference_snapshot(**reference_fixture)
    segment = reference_fixture["frame"].get_column(reference.segment_column)[0]
    injected = inject_drift(
        reference_fixture["frame"],
        DriftScenario(
            scenario_id="chain-drift",
            drift_type=drift_type,
            target="browse_pv_30d",
            magnitude=magnitude,
            institution=None if drift_type == "numeric_shift" else str(segment),
            segment_column=reference.segment_column,
        ),
        seed=17,
    )

    alerts = detect_anomalies(
        reference, injected.frame, (), reference_fixture["catalog"]
    )
    diagnoses = diagnose_alerts(
        alerts, reference, injected.frame, reference_fixture["catalog"], top_k=3
    )
    score = evaluate_alerts(alerts, (injected.truth,), diagnoses=diagnoses)

    assert score.recall == 1.0
    assert score.top_k_root_cause_hit == 1.0
    feature_diagnosis = next(
        diagnosis
        for diagnosis in diagnoses
        if diagnosis.alerts[0].scope_value == injected.truth.root_cause_value
    )
    assert any(
        cause.dimension == injected.truth.root_cause_dimension
        and cause.value == injected.truth.root_cause_value
        for cause in feature_diagnosis.root_causes
    )
