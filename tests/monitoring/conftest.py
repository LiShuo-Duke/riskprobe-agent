import polars as pl
import pytest

from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.models import EvidenceCard, RiskRule, RuleMetrics
from riskprobe.profiling import profile_dataset
from riskprobe.synthetic import generate_behavior_dataset


@pytest.fixture
def reference_fixture(tmp_path, synthetic_config):
    frame, _ = generate_behavior_dataset(5_000, seed=42)
    frame = frame.with_columns(
        [
            pl.lit("user_0001").alias("entity_id"),
            pl.lit("private-file-marker").alias("undeclared_text"),
        ]
    )
    frame.write_parquet(synthetic_config.dataset.path)
    catalog = FeatureCatalog.from_columns(frame.columns, synthetic_config.features.families)
    profile = profile_dataset(ParquetDataset(synthetic_config.dataset.path), synthetic_config)
    metrics = RuleMetrics(
        support_count=100,
        coverage=0.02,
        base_bad_rate=0.1,
        hit_bad_rate=0.3,
        non_hit_bad_rate=0.09,
        lift=3.0,
        precision=0.3,
        recall=0.06,
        p_value=0.01,
    )
    evidence_card = EvidenceCard(
        rule=RiskRule(rule_id="6c1469285066", conditions=(), origin="test"),
        train=metrics,
        test=metrics,
        slices=(),
        lift_ci=(2.0, 4.0),
        adjusted_p_value=0.01,
        segment_consistency=1.0,
        max_time_decay=0.0,
        grade="Stable",
    )
    return {
        "frame": frame,
        "profile": profile,
        "evidence_cards": (evidence_card,),
        "catalog": catalog,
        "config": synthetic_config,
    }


@pytest.fixture
def catalog(reference_fixture):
    return reference_fixture["catalog"]
