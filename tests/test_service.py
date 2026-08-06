import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import riskprobe.service as service_module

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
    assert names == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    }
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["artifact_integrity"]) == names - {"manifest.json"}
    for name, integrity in manifest["artifact_integrity"].items():
        content = (result.run_dir / name).read_bytes()
        assert integrity == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    assert manifest_path.read_text() == json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


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


def test_service_rejects_tampered_complete_run(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")
    first = service.run()
    report = first.run_dir / "risk_report.md"
    report.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not complete"):
        service.run()

    assert report.read_text(encoding="utf-8") == "tampered"


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
    assert list((tmp_path / "runs").glob("*.parquet")) == []


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

    text_artifacts = b"\n".join(
        path.read_bytes()
        for path in result.run_dir.iterdir()
        if path.suffix != ".parquet"
    )
    candidate_rows = pl.read_parquet(
        result.run_dir / "candidate_rules.parquet"
    ).rows(named=True)
    logical_parquet = json.dumps(candidate_rows, sort_keys=True).encode()
    combined = text_artifacts + b"\n" + logical_parquet
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


def test_run_analyzes_same_snapshot_used_for_fingerprint_after_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=200)
    original_fingerprint = service_module._parquet_metadata_fingerprint(
        config.dataset.path
    )
    replacement = tmp_path / "replacement.parquet"
    pl.read_parquet(config.dataset.path).head(80).write_parquet(replacement)
    real_fingerprint = service_module._parquet_metadata_fingerprint
    replaced = False

    def replace_source_after_fingerprint(path: Path) -> str:
        nonlocal replaced
        fingerprint = real_fingerprint(path)
        if not replaced:
            replacement.replace(config.dataset.path)
            replaced = True
        return fingerprint

    monkeypatch.setattr(
        service_module,
        "_parquet_metadata_fingerprint",
        replace_source_after_fingerprint,
    )
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    profile = json.loads((result.run_dir / "data_profile.json").read_text())
    assert manifest["data_fingerprint"] == original_fingerprint
    assert profile["row_count"] == 200
    assert pl.read_parquet(config.dataset.path).height == 80
    assert list((tmp_path / "runs").glob("*.parquet")) == []


def test_empty_holdout_downgrades_each_card_and_reports_limitation(
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
            [date(2024, 1, 1)] * 70 + [date(2024, 2, 1)] * 30,
        )
    )
    frame.write_parquet(config.dataset.path)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr(
        "riskprobe.service.validate_rules", lambda *args, **kwargs: [_card()]
    )

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    report = (result.run_dir / "risk_report.md").read_text()
    limitation = "Holdout partition is empty; validation unavailable"
    assert evidence[0]["grade"] == "Suspicious"
    assert limitation in evidence[0]["limitations"]
    assert limitation in report


def test_single_class_holdout_downgrades_each_card_and_reports_limitation(
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
            [date(2024, 1, 1)] * 60
            + [date(2024, 2, 1)] * 20
            + [date(2024, 3, 1)] * 20,
        ),
        pl.Series("target", [index % 2 for index in range(80)] + [0] * 20),
    )
    frame.write_parquet(config.dataset.path)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr(
        "riskprobe.service.validate_rules", lambda *args, **kwargs: [_card()]
    )

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    metadata = json.loads((result.run_dir / "metadata_report.json").read_text())
    limitation = "Holdout partition has a single target class; validation unavailable"
    assert evidence[0]["grade"] == "Suspicious"
    assert limitation in evidence[0]["limitations"]
    assert limitation in metadata["limitations"]


@pytest.mark.parametrize(
    "error",
    [ValueError("unstable implementation detail"), RuntimeError("backend failure")],
)
def test_holdout_validation_exception_downgrades_each_card_instead_of_failing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    config = _small_config(
        tmp_path,
        rows=100,
        time_validation_enabled=True,
        metadata_grade="A",
    )
    responses = iter([[_card()], error])

    def fake_validate(*args: object, **kwargs: object) -> list[EvidenceCard]:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    limitation = "Holdout validation could not be computed"
    assert evidence[0]["grade"] == "Suspicious"
    assert limitation in evidence[0]["limitations"]
    assert str(error) not in json.dumps(evidence)


def test_missing_holdout_rule_downgrades_only_missing_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(
        tmp_path,
        rows=100,
        time_validation_enabled=True,
        metadata_grade="A",
    )
    responses = [[_card()], []]
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr(
        "riskprobe.service.validate_rules", lambda *args, **kwargs: responses.pop(0)
    )

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    limitation = "Holdout evidence is missing for this rule"
    assert evidence[0]["grade"] == "Suspicious"
    assert limitation in evidence[0]["limitations"]


def test_null_snapshots_are_excluded_and_audited_not_treated_as_holdout(
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
            [date(2024, 1, 1)] * 48
            + [date(2024, 2, 1)] * 16
            + [date(2024, 3, 1)] * 16
            + [None] * 20,
            dtype=pl.Date,
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

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    profile = json.loads((result.run_dir / "data_profile.json").read_text())
    evidence = json.loads((result.run_dir / "evidence_cards.json").read_text())
    metadata = json.loads((result.run_dir / "metadata_report.json").read_text())
    report = (result.run_dir / "risk_report.md").read_text()
    limitation = "Time validation excluded 20 rows with null snapshot values"
    assert all(
        partition.get_column("snapshot_date").null_count() == 0
        for call in calls
        for partition in call
    )
    assert profile["excluded_null_snapshot_rows"] == 20
    assert sum(metadata["split_rows"].values()) == 80
    assert limitation in evidence[0]["limitations"]
    assert limitation in report


@pytest.mark.parametrize(
    "private_id",
    [
        "file:///Users/alice/private/input.parquet",
        "file:///Users/alice/private%20folder/input.parquet",
        "file%3A%2F%2F%2FUsers%2Falice%2Fprivate%2Finput.parquet",
        "file:///C:/Users/Alice/private/input.parquet",
        "source=/Users/alice/private/input.parquet",
        r"source=C:\Users\Alice\private\input.parquet",
    ],
)
def test_file_uri_and_prefixed_path_dataset_ids_are_redacted_everywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    private_id: str,
) -> None:
    config = _small_config(tmp_path)
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"id": private_id})}
    )
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    text = b"\n".join(
        path.read_bytes()
        for path in result.run_dir.iterdir()
        if path.suffix != ".parquet"
    ).decode()
    assert private_id not in text
    assert "dataset-" in text


def test_renderer_redacts_path_dataset_id_without_service_boundary() -> None:
    profile = DatasetProfile(
        dataset_id="file:///Users/alice/private/input.parquet",
        row_count=1,
        feature_count=0,
        positive_rate=0.0,
        segment_counts={},
        snapshot_min=None,
        snapshot_max=None,
        metadata_grade="A",
        issues=(),
    )

    report = render_risk_report(profile, [])

    assert "file:///Users/alice" not in report
    assert "dataset-" in report


@pytest.mark.parametrize(
    "business_id",
    ["portfolio/retail-2024", "customer:premium", "file-processing-2024"],
)
def test_renderer_preserves_ordinary_business_dataset_ids(business_id: str) -> None:
    profile = DatasetProfile(
        dataset_id=business_id,
        row_count=1,
        feature_count=0,
        positive_rate=0.0,
        segment_counts={},
        snapshot_min=None,
        snapshot_max=None,
        metadata_grade="A",
        issues=(),
    )

    report = render_risk_report(profile, [])

    assert f"`{business_id}`" in report


def test_stable_snapshot_is_private_to_os_temp_and_removed_after_use(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    with service_module._stable_dataset_snapshot(
        config.dataset.path, runs_dir
    ) as snapshot_path:
        snapshot_dir = snapshot_path.parent
        assert snapshot_dir != runs_dir
        assert snapshot_path.read_bytes() == config.dataset.path.read_bytes()
        assert snapshot_path.stat().st_mode & 0o777 == 0o400
        assert snapshot_dir.stat().st_mode & 0o777 == 0o700
        assert not list(runs_dir.glob(".riskprobe-input-*.parquet"))

    assert not snapshot_dir.exists()


def test_snapshot_copy_failure_removes_temporary_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("simulated snapshot copy failure")

    monkeypatch.setattr(service_module.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="simulated snapshot copy failure"):
        RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    assert not list((tmp_path / "runs").glob(".riskprobe-input-*.parquet"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "other-run"),
        ("config_fingerprint", "0" * 64),
        ("data_fingerprint", "0" * 64),
        ("code_version", "other-version"),
        ("dataset_id", "other-dataset"),
        ("time_validation_enabled", True),
    ],
)
def test_reuse_rejects_canonical_manifest_identity_mutation(
    tmp_path: Path, field: str, replacement: object
) -> None:
    config = _small_config(tmp_path)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")
    first = service.run()

    first.run_dir.chmod(0o755)
    manifest_path = first.run_dir / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not complete"):
        service.run()
