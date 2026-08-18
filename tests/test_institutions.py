from pathlib import Path

import polars as pl

from riskprobe.config import DiscoveryConfig, ValidationConfig
from riskprobe.models import Condition, EvidenceCard, RiskRule, RuleMetrics, SliceMetrics
from riskprobe.institutions import discover_local_rules


def _metrics(lift: float, support_count: int = 40) -> RuleMetrics:
    return RuleMetrics(
        support_count=support_count,
        coverage=0.2,
        base_bad_rate=0.2,
        hit_bad_rate=0.4,
        non_hit_bad_rate=0.1,
        lift=lift,
        precision=0.4,
        recall=0.2,
        p_value=0.01,
    )


def _card(grade: str, institution: str, support_count: int = 40) -> EvidenceCard:
    rule = RiskRule(
        rule_id="global-rule",
        conditions=(Condition(feature="feature_a", operator=">", value=0.5),),
        origin="test",
    )
    return EvidenceCard(
        rule=rule,
        train=_metrics(2.0),
        test=_metrics(2.0),
        slices=(
            SliceMetrics(
                slice_type="segment",
                slice_value=institution,
                metrics=_metrics(2.5, support_count),
            ),
        ),
        lift_ci=(1.2, 3.0),
        adjusted_p_value=0.01,
        segment_consistency=0.4,
        max_time_decay=0.0,
        grade=grade,  # type: ignore[arg-type]
    )


def _frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.DataFrame(
        {
            "institution": ["A"] * 40 + ["B"] * 10,
            "target": [0, 1] * 20 + [0, 1] * 5,
            "feature_a": [float(index % 4) for index in range(50)],
        }
    )
    return train, train.clone()


def test_local_discovery_triggers_only_for_local_well_supported_institutions(
    tmp_path: Path,
) -> None:
    train, test = _frames()
    report = discover_local_rules(
        train,
        test,
        [_card("Local", "A", 40), _card("Stable", "B", 10)],
        ["feature_a"],
        target_col="target",
        segment_col="institution",
        snapshot_col="snapshot_date",
        time_validation_enabled=False,
        discovery_config=DiscoveryConfig(max_single_rules=2, max_pair_rules=0),
        validation_config=ValidationConfig(min_group_size=20),
        confirmed_features=frozenset({"feature_a"}),
        runs_dir=tmp_path,
        expose_segment_values=True,
    )

    assert report["triggered_institution_count"] == 1
    assert report["institution_reports"][0]["status"] == "completed"
    assert report["institution_reports"][0]["institution_token"].startswith("tok_")
    assert report["institution_reports"][0]["institution_name"] == "A"
    assert report["institution_reports"][0]["rule_count"] >= 0


def test_local_discovery_blocks_small_institutions_without_running_discovery(
    tmp_path: Path,
) -> None:
    train, test = _frames()
    report = discover_local_rules(
        train,
        test,
        [_card("Local", "B", 10)],
        ["feature_a"],
        target_col="target",
        segment_col="institution",
        snapshot_col="snapshot_date",
        time_validation_enabled=False,
        discovery_config=DiscoveryConfig(max_single_rules=2, max_pair_rules=0),
        validation_config=ValidationConfig(min_group_size=20),
        confirmed_features=frozenset({"feature_a"}),
        runs_dir=tmp_path,
    )

    assert report["triggered_institution_count"] == 0
    blocked = report["institution_reports"][0]
    assert blocked["status"] == "blocked"
    assert "sample" in blocked["reason"]
