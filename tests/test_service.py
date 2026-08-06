import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from riskprobe.config import ProjectConfig
from riskprobe.features.catalog import QualityIssue
from riskprobe.models import Condition, EvidenceCard, RiskRule, RuleMetrics, SliceMetrics
from riskprobe.profiling import DatasetProfile
from riskprobe.reporting import render_risk_report
from riskprobe.service import RiskProbeService


def _small_config(
    tmp_path: Path,
    *,
    rows: int = 200,
    time_validation_enabled: bool = False,
    metadata_grade: str = "B",
) -> ProjectConfig:
    snapshots: list[object]
    if time_validation_enabled:
        snapshots = [date(2024, 1, 1) + timedelta(days=index) for index in range(rows)]
    else:
        snapshots = ["not-a-date"] * rows
    frame = pl.DataFrame(
        {
            "entity_id": [f"private-{index}" for index in range(rows)],
            "snapshot_date": snapshots,
            "institution": ["A" if index % 4 < 2 else "B" for index in range(rows)],
            "target": [index % 2 for index in range(rows)],
            "feature_a": [float(index % 10) for index in range(rows)],
            "unused_feature": [float(index) for index in range(rows)],
        }
    )
    data_path = tmp_path / "input.parquet"
    frame.write_parquet(data_path)
    target: dict[str, Any] = {
        "positive_value": 1,
        "positive_meaning": "bad_debt",
    }
    if metadata_grade == "A":
        target["performance_window_days"] = 30
    return ProjectConfig.model_validate(
        {
            "dataset": {"id": "small", "path": data_path},
            "columns": {
                "entity": "entity_id",
                "snapshot": "snapshot_date",
                "segment": "institution",
                "target": "target",
            },
            "target": target,
            "snapshot": {"meaning": "customer_specified_feature_cutoff"},
            "features": {"families": {"feature": ["feature_"]}},
            "time_validation_enabled": time_validation_enabled,
            "discovery": {
                "min_support": 0.05,
                "max_single_rules": 2,
                "beam_width": 2,
                "max_pair_rules": 0,
                "random_seed": 42,
            },
            "validation": {
                "alpha": 0.05,
                "min_segment_consistency": 0.6,
                "max_lift_decay": 0.3,
                "bootstrap_rounds": 100,
                "min_group_size": 20,
            },
        }
    )


def _metrics(lift: float) -> RuleMetrics:
    return RuleMetrics(
        support_count=20,
        coverage=0.2,
        base_bad_rate=0.1,
        hit_bad_rate=0.2,
        non_hit_bad_rate=0.075,
        lift=lift,
        precision=0.2,
        recall=0.4,
        p_value=0.01,
    )


def _rule(rule_id: str = "rule-a", feature: str = "feature_a") -> RiskRule:
    return RiskRule(
        rule_id=rule_id,
        conditions=(Condition(feature=feature, operator=">", value=5.0),),
        origin="test",
    )


def _card(
    rule_id: str = "rule-a",
    *,
    grade: str = "Stable",
    test_lift: float = 2.0,
    slices: tuple[SliceMetrics, ...] = (),
    limitations: tuple[str, ...] = (),
) -> EvidenceCard:
    return EvidenceCard(
        rule=_rule(rule_id),
        train=_metrics(2.1),
        test=_metrics(test_lift),
        slices=slices,
        lift_ci=(1.1, 2.5),
        adjusted_p_value=0.02,
        segment_consistency=1.0,
        max_time_decay=0.0,
        grade=grade,  # type: ignore[arg-type]
        limitations=limitations,
    )


def test_service_run_writes_required_artifacts(tmp_path, synthetic_config) -> None:
    service = RiskProbeService(config=synthetic_config, runs_dir=tmp_path / "runs")
    result = service.run()
    names = {path.name for path in result.run_dir.iterdir()}
    assert {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    } <= names


def test_inspect_and_discover_return_existing_domain_models(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    profile = service.inspect()
    rules = service.discover()

    assert isinstance(profile, DatasetProfile)
    assert all(isinstance(rule, RiskRule) for rule in rules)


def test_same_input_produces_byte_for_byte_identical_artifacts(tmp_path: Path) -> None:
    config = _small_config(tmp_path, rows=400)
    first = RiskProbeService(config=config, runs_dir=tmp_path / "runs-a").run()
    second = RiskProbeService(config=config, runs_dir=tmp_path / "runs-b").run()

    first_bytes = {path.name: path.read_bytes() for path in first.run_dir.iterdir()}
    second_bytes = {path.name: path.read_bytes() for path in second.run_dir.iterdir()}

    assert first.run_id == second.run_id
    assert first_bytes == second_bytes


def test_service_does_not_overwrite_complete_duplicate_run(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")
    first = service.run()
    report = first.run_dir / "risk_report.md"
    report.write_text("immutable", encoding="utf-8")

    second = service.run()

    assert second.is_existing is True
    assert report.read_text(encoding="utf-8") == "immutable"


def test_service_failure_removes_incomplete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    def fail_report(*args: object, **kwargs: object) -> str:
        raise RuntimeError("simulated rendering failure")

    monkeypatch.setattr("riskprobe.service.render_risk_report", fail_report)
    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        service.run()

    assert [path for path in (tmp_path / "runs").iterdir() if path.is_dir()] == []


def test_disabled_time_split_is_stratified_projected_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100, time_validation_enabled=False)
    original = config.dataset.path.read_bytes()
    captured: dict[str, object] = {}

    def fake_discover(
        train: pl.DataFrame,
        feature_names: list[str],
        target_col: str,
        config: object,
    ) -> list[RiskRule]:
        captured["discovery"] = train
        captured["feature_names"] = feature_names
        return [_rule()]

    def fake_validate(
        train: pl.DataFrame, test: pl.DataFrame, rules: object, **kwargs: object
    ) -> list[EvidenceCard]:
        captured["validation"] = (train, test, kwargs)
        return [_card()]

    monkeypatch.setattr("riskprobe.service.discover_rules", fake_discover)
    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    discovery = captured["discovery"]
    train, test, kwargs = captured["validation"]  # type: ignore[misc]
    assert isinstance(discovery, pl.DataFrame)
    assert discovery.columns == ["feature_a", "unused_feature", "target"]
    assert captured["feature_names"] == ["feature_a", "unused_feature"]
    assert train.columns == ["feature_a", "target", "institution"]
    assert test.columns == ["feature_a", "target", "institution"]
    assert (train.height, test.height) == (70, 30)
    assert train.get_column("target").value_counts().sort("target")["count"].to_list() == [35, 35]
    assert test.get_column("target").value_counts().sort("target")["count"].to_list() == [15, 15]
    assert kwargs["time_validation_enabled"] is False
    assert config.dataset.path.read_bytes() == original
    assert not config.dataset.path.with_suffix(".tmp").exists()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    report = (result.run_dir / "risk_report.md").read_text()
    assert "max_time_decay" not in json.dumps(evidence)
    assert "time decay" not in report.lower()
    assert "时间衰减" not in report


def test_enabled_time_split_is_sorted_60_20_20_and_validates_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(
        tmp_path,
        rows=100,
        time_validation_enabled=True,
        metadata_grade="A",
    )
    calls: list[tuple[pl.DataFrame, pl.DataFrame]] = []

    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])

    def fake_validate(
        train: pl.DataFrame, test: pl.DataFrame, rules: object, **kwargs: object
    ) -> list[EvidenceCard]:
        calls.append((train, test))
        return [_card()]

    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    assert [(train.height, test.height) for train, test in calls] == [(60, 20), (60, 20)]
    train, test = calls[0]
    _, holdout = calls[1]
    assert train["snapshot_date"].max() <= test["snapshot_date"].min()
    assert test["snapshot_date"].max() <= holdout["snapshot_date"].min()
    payload = json.loads((result.run_dir / "evidence_cards.json").read_text())
    holdout_slices = [
        item
        for item in payload[0]["slices"]
        if item["slice_type"] == "dataset" and item["slice_value"] == "Holdout"
    ]
    assert len(holdout_slices) == 1


def test_artifact_rules_and_slices_have_stable_sorting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    rules = [_rule("z-rule"), _rule("a-rule")]
    unsorted_slices = (
        SliceMetrics(slice_type="segment", slice_value="Z", metrics=_metrics(1.2)),
        SliceMetrics(slice_type="dataset", slice_value="Holdout", metrics=_metrics(1.4)),
        SliceMetrics(slice_type="segment", slice_value="A", metrics=_metrics(1.3)),
    )
    cards = [
        _card("z-rule", grade="Suspicious", test_lift=1.1, slices=unsorted_slices),
        _card("a-rule", grade="Stable", test_lift=1.8),
    ]
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: rules)
    monkeypatch.setattr("riskprobe.service.validate_rules", lambda *args, **kwargs: cards)

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    candidate_ids = pl.read_parquet(result.run_dir / "candidate_rules.parquet")["rule_id"].to_list()
    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    assert candidate_ids == ["a-rule", "z-rule"]
    assert [item["rule"]["rule_id"] for item in evidence] == ["a-rule", "z-rule"]
    slices = [
        (item["slice_type"], item["slice_value"])
        for item in evidence[1]["slices"]
    ]
    assert slices[0] == ("dataset", "Holdout")
    assert [value for slice_type, value in slices[1:] if slice_type == "segment"] == sorted(
        value for slice_type, value in slices[1:] if slice_type == "segment"
    )
    assert all(value.startswith("segment-") for _, value in slices[1:])


def test_report_is_sorted_formatted_and_grade_b_leads_with_limitations() -> None:
    profile = DatasetProfile(
        dataset_id="safe-dataset-id",
        row_count=100,
        feature_count=2,
        positive_rate=0.123456,
        segment_counts={"B": 40, "A": 60},
        snapshot_min=date(2024, 1, 1),
        snapshot_max=date(2024, 2, 1),
        metadata_grade="B",
        issues=(
            QualityIssue(
                code="LABEL_PERFORMANCE_WINDOW_UNKNOWN",
                severity="warning",
                family="target",
                features=(),
                affected_rows=100,
                message="target performance window is not configured",
            ),
        ),
    )
    cards = [
        _card("later", grade="Suspicious", test_lift=9.0, limitations=("lim-z",)),
        _card("b-rule", grade="Stable", test_lift=1.5),
        _card("a-rule", grade="Stable", test_lift=2.0, limitations=("lim-a",)),
    ]

    report = render_risk_report(profile, cards)

    assert any("Metadata Grade: B" in line for line in report.splitlines()[:8])
    assert any(
        "label performance window unknown" in line for line in report.splitlines()[:12]
    )
    assert report.index("a-rule") < report.index("b-rule") < report.index("later")
    assert "0.1235" in report
    assert "2.0000" in report
    assert "严格 OOT" not in report
    assert "可上线" not in report
    assert "/Users/" not in report


def test_outputs_redact_segment_values_and_absolute_input_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    frame = pl.read_parquet(config.dataset.path).with_columns(
        pl.when(pl.col("institution") == "A")
        .then(pl.lit("SECRET_CLIENT_ALPHA"))
        .otherwise(pl.lit("SECRET_CLIENT_BETA"))
        .alias("institution")
    )
    frame.write_parquet(config.dataset.path)
    card = _card(
        slices=(
            SliceMetrics(
                slice_type="segment",
                slice_value="SECRET_CLIENT_ALPHA",
                metrics=_metrics(1.5),
            ),
        ),
        limitations=("single-class institution: SECRET_CLIENT_ALPHA",),
    )
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr("riskprobe.service.validate_rules", lambda *args, **kwargs: [card])

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    combined = b"\n".join(path.read_bytes() for path in result.run_dir.iterdir())
    assert b"private-" not in combined
    assert b"SECRET_CLIENT_ALPHA" not in combined
    assert b"SECRET_CLIENT_BETA" not in combined
    assert str(config.dataset.path).encode() not in combined


def test_holdout_failure_conservatively_downgrades_grade_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(
        tmp_path,
        rows=100,
        time_validation_enabled=True,
        metadata_grade="A",
    )
    responses = [
        [_card(grade="Stable", test_lift=2.0)],
        [_card(grade="Suspicious", test_lift=0.5)],
    ]
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr(
        "riskprobe.service.validate_rules", lambda *args, **kwargs: responses.pop(0)
    )

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    report = (result.run_dir / "risk_report.md").read_text()
    assert evidence[0]["grade"] == "Suspicious"
    assert "Holdout Lift" in report
    assert "0.5000" in report


def test_time_split_never_places_one_snapshot_in_multiple_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(
        tmp_path,
        rows=100,
        time_validation_enabled=True,
        metadata_grade="A",
    )
    frame = pl.read_parquet(config.dataset.path).with_columns(
        pl.Series(
            "snapshot_date",
            [date(2024, 1, 1)] * 65
            + [date(2024, 2, 1)] * 20
            + [date(2024, 3, 1)] * 15,
        )
    )
    frame.write_parquet(config.dataset.path)
    calls: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])

    def fake_validate(
        train: pl.DataFrame, test: pl.DataFrame, rules: object, **kwargs: object
    ) -> list[EvidenceCard]:
        calls.append((train, test))
        return [_card()]

    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    train_dates = set(calls[0][0]["snapshot_date"].to_list())
    test_dates = set(calls[0][1]["snapshot_date"].to_list())
    holdout_dates = set(calls[1][1]["snapshot_date"].to_list())
    assert train_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(holdout_dates)
    assert test_dates.isdisjoint(holdout_dates)


def test_time_partition_without_positives_produces_auditable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(
        tmp_path,
        rows=100,
        time_validation_enabled=True,
        metadata_grade="A",
    )
    frame = pl.read_parquet(config.dataset.path).with_columns(
        pl.Series("target", [index % 2 for index in range(60)] + [0] * 40)
    )
    frame.write_parquet(config.dataset.path)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    metadata = json.loads((result.run_dir / "metadata_report.json").read_text())
    assert evidence == []
    assert "Test partition has no positive target; validation unavailable" in metadata[
        "limitations"
    ]


def test_path_like_dataset_id_is_not_written_to_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    private_id = "/company/private/customer-a/input.parquet"
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"id": private_id})}
    )
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    combined = b"\n".join(path.read_bytes() for path in result.run_dir.iterdir())
    assert private_id.encode() not in combined
    assert b"dataset-" in combined
