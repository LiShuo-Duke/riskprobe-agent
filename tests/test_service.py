import fcntl
import hashlib
import json
import uuid
from contextlib import contextmanager
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
from riskprobe.runtime import NodeStatus, RunRuntime
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


def test_local_handler_reads_inspect_and_discover_from_verified_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, PolicyEngine, Principal, Role
    from riskprobe.registry import DatasetRegistry
    from riskprobe.tools import (
        DiscoverRequest,
        HandlerToolGateway,
        InspectRequest,
        LocalRiskProbeToolHandler,
    )

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    context = RiskProbeService(config=config, runs_dir=runs_dir).run()
    profile_payload = json.loads(
        (context.run_dir / "data_profile.json").read_text(encoding="utf-8")
    )
    expected_rule_ids = tuple(
        pl.read_parquet(context.run_dir / "candidate_rules.parquet")
        .get_column("rule_id")
        .to_list()
    )

    def reject_recomputation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("completed run artifacts must be reused")

    monkeypatch.setattr(RiskProbeService, "inspect", reject_recomputation)
    monkeypatch.setattr(RiskProbeService, "discover", reject_recomputation)
    handler = LocalRiskProbeToolHandler(
        run_id=context.run_id,
        runs_dir=runs_dir,
        evidence_store=EvidenceStore(tmp_path / "evidence.sqlite3"),
        run_context=context,
    )
    gateway = HandlerToolGateway(
        registry=DatasetRegistry.from_mapping({config.dataset.id: config}),
        policy=PolicyEngine(),
        handler=handler,
    )
    principal = Principal(principal_id="artifact-reader", role=Role.OPERATOR)

    inspect = gateway.invoke(
        principal,
        InspectRequest(dataset_id=config.dataset.id),
        Budget(max_queries=1),
    )
    discover = gateway.invoke(
        principal,
        DiscoverRequest(dataset_id=config.dataset.id),
        Budget(max_queries=1),
    )

    assert inspect.model_dump(mode="json") == {
        "dataset_id": config.dataset.id,
        "row_count": profile_payload["row_count"],
        "feature_count": profile_payload["feature_count"],
        "metadata_grade": profile_payload["metadata_grade"],
        "issue_codes": sorted(
            {issue["code"] for issue in profile_payload["issues"]}
        ),
    }
    assert discover.rule_ids == expected_rule_ids


def test_local_handler_recommend_uses_verified_profile_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.evidence import EvidenceRecord, EvidenceStore
    from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
    from riskprobe.policy import Budget, PolicyEngine, Principal, Role
    from riskprobe.registry import DatasetRegistry
    from riskprobe.tools import HandlerToolGateway, LocalRiskProbeToolHandler, RecommendRequest

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    context = RiskProbeService(config=config, runs_dir=runs_dir).run()
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    finding = RiskFinding(
        kind=FindingKind.DATA_QUALITY,
        severity=FindingSeverity.WARNING,
        code="missing_values",
        metrics={"affected_count": 10, "affected_rate": 0.05},
    )
    finding_id = store.append(
        EvidenceRecord(
            run_id=context.run_id,
            kind="diagnostic.finding",
            payload={
                **finding.model_dump(mode="json"),
                "dataset_id": config.dataset.id,
            },
            producer_version="artifact-fast-path-test-v1",
        )
    )
    recomputed = False

    def reject_recomputation(*args: object, **kwargs: object) -> object:
        nonlocal recomputed
        del args, kwargs
        recomputed = True
        raise AssertionError("completed profile artifact must be reused")

    monkeypatch.setattr(RiskProbeService, "inspect", reject_recomputation)
    handler = LocalRiskProbeToolHandler(
        run_id=context.run_id,
        runs_dir=runs_dir,
        evidence_store=store,
        run_context=context,
    )
    gateway = HandlerToolGateway(
        registry=DatasetRegistry.from_mapping({config.dataset.id: config}),
        policy=PolicyEngine(),
        handler=handler,
    )

    response = gateway.invoke(
        Principal(principal_id="artifact-reader", role=Role.OPERATOR),
        RecommendRequest(
            dataset_id=config.dataset.id,
            evidence_ids=(finding_id,),
        ),
        Budget(max_queries=1),
    )

    assert response.recommendation_ids
    assert recomputed is False


def test_orchestrate_reuses_verified_terminal_result_without_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.agents import SessionStore
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, Principal, Role
    from riskprobe.tools import LocalRiskProbeToolHandler

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=config,
        runs_dir=runs_dir,
        state_dir=state_dir,
    )
    principal = Principal(principal_id="cache-reader", role=Role.ANALYST)
    first = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )
    session_path = state_dir / f".{first.session_id}.sessions.sqlite3"
    evidence_path = state_dir / f".{first.session_id}.evidence.sqlite3"
    result_path = state_dir / f".{first.session_id}.agent-result.json"
    session_count = len(SessionStore(session_path).replay(first.session_id))
    evidence_count = len(EvidenceStore(evidence_path).list_run(first.session_id))
    tool_calls = 0

    def reject_tool_call(*args: object, **kwargs: object) -> object:
        nonlocal tool_calls
        del args, kwargs
        tool_calls += 1
        raise AssertionError("terminal result reuse must not invoke tools")

    monkeypatch.setattr(LocalRiskProbeToolHandler, "handle", reject_tool_call)
    second = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )

    assert second == first
    assert tool_calls == 0
    assert result_path.is_file()
    assert result_path.stat().st_mode & 0o777 == 0o600
    assert len(SessionStore(session_path).replay(first.session_id)) == session_count
    assert len(EvidenceStore(evidence_path).list_run(first.session_id)) == evidence_count


def test_orchestrate_fails_closed_when_terminal_result_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.agents import SessionStore
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, Principal, Role
    from riskprobe.tools import LocalRiskProbeToolHandler

    config = _small_config(tmp_path)
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
    )
    principal = Principal(principal_id="cache-reader", role=Role.ANALYST)
    first = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )
    session_path = state_dir / f".{first.session_id}.sessions.sqlite3"
    evidence_path = state_dir / f".{first.session_id}.evidence.sqlite3"
    result_path = state_dir / f".{first.session_id}.agent-result.json"
    session_count = len(SessionStore(session_path).replay(first.session_id))
    evidence_count = len(EvidenceStore(evidence_path).list_run(first.session_id))
    result_path.unlink()
    tool_calls = 0

    def reject_tool_call(*args: object, **kwargs: object) -> object:
        nonlocal tool_calls
        del args, kwargs
        tool_calls += 1
        raise AssertionError("incomplete terminal state must not rerun tools")

    monkeypatch.setattr(LocalRiskProbeToolHandler, "handle", reject_tool_call)
    with pytest.raises(RuntimeError, match="agent result is unavailable"):
        service.orchestrate(
            dataset_id=config.dataset.id,
            principal=principal,
            budget=Budget(max_queries=16),
        )

    assert tool_calls == 0
    assert not result_path.exists()
    assert len(SessionStore(session_path).replay(first.session_id)) == session_count
    assert len(EvidenceStore(evidence_path).list_run(first.session_id)) == evidence_count


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


def test_service_failure_preserves_incomplete_run_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    def fail_report(*args: object, **kwargs: object) -> str:
        raise RuntimeError("simulated rendering failure")

    monkeypatch.setattr("riskprobe.service.render_risk_report", fail_report)
    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        service.run()

    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / ".incomplete").is_file()
    runtime_databases = list((tmp_path / "runs").glob(".*.runtime.sqlite3"))
    assert len(runtime_databases) == 1
    assert runtime_databases[0].stat().st_mode & 0o777 == 0o600
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
    assert discovery.columns == ["feature_a", "target"]
    assert captured["feature_names"] == ["feature_a"]
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


_SNAPSHOT_MARKER = b"riskprobe raw snapshot v1\n"


def _write_recoverable_snapshot(root: Path, source: Path) -> Path:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    snapshot_dir = root / f"snapshot-{uuid.uuid4().hex}"
    snapshot_dir.mkdir(mode=0o700)
    marker = snapshot_dir / ".riskprobe-snapshot"
    marker.write_bytes(_SNAPSHOT_MARKER)
    marker.chmod(0o400)
    lock = snapshot_dir / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    snapshot = snapshot_dir / "input.parquet"
    snapshot.write_bytes(source.read_bytes())
    snapshot.chmod(0o400)
    return snapshot_dir


def test_stable_snapshot_uses_private_service_root_and_never_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    snapshot_root = tmp_path / "riskprobe-private-snapshots"
    snapshot_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        service_module, "_snapshot_root", lambda: snapshot_root, raising=False
    )

    with service_module._stable_dataset_snapshot(
        config.dataset.path, runs_dir
    ) as snapshot_path:
        snapshot_dir = snapshot_path.parent
        assert snapshot_dir.parent == snapshot_root
        assert snapshot_path.read_bytes() == config.dataset.path.read_bytes()
        assert snapshot_path.stat().st_mode & 0o777 == 0o400
        assert snapshot_dir.stat().st_mode & 0o777 == 0o700
        assert snapshot_root.stat().st_mode & 0o777 == 0o700
        assert list(runs_dir.iterdir()) == []

    assert not snapshot_dir.exists()
    assert list(snapshot_root.iterdir()) == []


def test_service_recovers_only_unlocked_safe_snapshots_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    snapshot_root = tmp_path / "riskprobe-private-snapshots"
    stale = _write_recoverable_snapshot(snapshot_root, config.dataset.path)
    active = _write_recoverable_snapshot(snapshot_root, config.dataset.path)
    insecure = _write_recoverable_snapshot(snapshot_root, config.dataset.path)
    insecure.chmod(0o755)
    unknown = snapshot_root / "unknown-entry"
    unknown.write_text("leave me alone", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = snapshot_root / f"snapshot-{uuid.uuid4().hex}"
    linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        service_module, "_snapshot_root", lambda: snapshot_root, raising=False
    )
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])

    with (active / ".lock").open("r+b") as active_lock:
        fcntl.flock(active_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

        assert not stale.exists()
        assert active.is_dir()
        assert insecure.is_dir()
        assert unknown.read_text(encoding="utf-8") == "leave me alone"
        assert linked.is_symlink()
        assert outside.is_dir()


def test_snapshot_copy_failure_cleans_recovered_and_new_private_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    snapshot_root = tmp_path / "riskprobe-private-snapshots"
    _write_recoverable_snapshot(snapshot_root, config.dataset.path)
    monkeypatch.setattr(
        service_module, "_snapshot_root", lambda: snapshot_root, raising=False
    )

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("simulated snapshot copy failure")

    monkeypatch.setattr(service_module.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="simulated snapshot copy failure"):
        RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    assert list(snapshot_root.iterdir()) == []
    assert list((tmp_path / "runs").iterdir()) == []


def test_snapshot_cleanup_failure_is_reported_and_recovered_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    snapshot_root = tmp_path / "riskprobe-private-snapshots"
    snapshot_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        service_module, "_snapshot_root", lambda: snapshot_root, raising=False
    )
    real_rmtree = service_module.shutil.rmtree

    def fail_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if Path(path).parent == snapshot_root:
            raise OSError("simulated snapshot cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(service_module.shutil, "rmtree", fail_cleanup)
    with pytest.raises(OSError, match="simulated snapshot cleanup failure"):
        with service_module._stable_dataset_snapshot(config.dataset.path) as snapshot:
            stale_snapshot_dir = snapshot.parent

    monkeypatch.setattr(service_module.shutil, "rmtree", real_rmtree)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])
    RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()

    assert not stale_snapshot_dir.exists()


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


def test_distinct_same_footer_inputs_do_not_reuse_a_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def footer_fingerprint(path: Path) -> str:
        with path.open("rb") as handle:
            handle.seek(-8, 2)
            footer_size = int.from_bytes(handle.read(4), byteorder="little")
            assert handle.read(4) == b"PAR1"
            handle.seek(-(footer_size + 8), 2)
            return hashlib.sha256(handle.read(footer_size)).hexdigest()

    first_config = _small_config(tmp_path, rows=200)
    second_path = tmp_path / "same-footer-different-values.parquet"
    first_frame = pl.read_parquet(first_config.dataset.path)
    first_frame.with_columns(
        first_frame.get_column("feature_a").reverse().alias("feature_a")
    ).write_parquet(second_path)
    second_config = first_config.model_copy(
        update={
            "dataset": first_config.dataset.model_copy(update={"path": second_path})
        }
    )
    assert footer_fingerprint(first_config.dataset.path) == footer_fingerprint(second_path)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])

    first = RiskProbeService(config=first_config, runs_dir=tmp_path / "runs").run()
    second = RiskProbeService(config=second_config, runs_dir=tmp_path / "runs").run()

    assert first.run_id != second.run_id
    assert second.is_existing is False


def test_inspect_and_discover_both_use_stable_input_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path)
    calls: list[Path] = []
    real_snapshot = service_module._stable_dataset_snapshot

    @contextmanager
    def tracked_snapshot(source: Path, *args: object, **kwargs: object):
        calls.append(source)
        with real_snapshot(source, *args, **kwargs) as snapshot:
            yield snapshot

    monkeypatch.setattr(service_module, "_stable_dataset_snapshot", tracked_snapshot)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    service.inspect()
    service.discover()

    assert calls == [config.dataset.path, config.dataset.path]


def test_code_identity_changes_when_source_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "riskprobe"
    source_root.mkdir()
    source_file = source_root / "engine.py"
    source_file.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(service_module, "_package_version", lambda: "0.1.0")

    first = service_module._code_identity(source_root)
    source_file.write_text("VERSION = 2\n", encoding="utf-8")
    second = service_module._code_identity(source_root)

    assert first.startswith("0.1.0+src-")
    assert first != second


def test_report_failure_preserves_checkpoints_and_resumes_completed_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100)
    calls = {"discover": 0, "validate": 0, "render": 0}
    original_render = service_module.render_risk_report

    def fake_discover(*args: object, **kwargs: object) -> list[RiskRule]:
        calls["discover"] += 1
        return [_rule()]

    def fake_validate(*args: object, **kwargs: object) -> list[EvidenceCard]:
        calls["validate"] += 1
        return [_card()]

    def fail_first_render(*args: object, **kwargs: object) -> str:
        calls["render"] += 1
        if calls["render"] == 1:
            raise RuntimeError("simulated rendering failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr("riskprobe.service.discover_rules", fake_discover)
    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    monkeypatch.setattr("riskprobe.service.render_risk_report", fail_first_render)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        service.run()

    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / ".incomplete").is_file()
    runtime = RunRuntime(tmp_path / "runs", run_dir.name)
    assert runtime.node_status("discover") is NodeStatus.SUCCEEDED
    assert runtime.node_status("validate") is NodeStatus.SUCCEEDED
    assert runtime.node_status("report") is NodeStatus.FAILED

    service.run()

    assert calls == {"discover": 1, "validate": 1, "render": 2}
    assert runtime.node_status("finalize") is NodeStatus.SUCCEEDED
    assert not (run_dir / ".incomplete").exists()


def test_tampered_rule_checkpoint_invalidates_it_and_downstream_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100)
    calls = {"discover": 0, "validate": 0, "render": 0}
    original_render = service_module.render_risk_report

    def fake_discover(*args: object, **kwargs: object) -> list[RiskRule]:
        calls["discover"] += 1
        return [_rule()]

    def fake_validate(*args: object, **kwargs: object) -> list[EvidenceCard]:
        calls["validate"] += 1
        return [_card()]

    def fail_first_render(*args: object, **kwargs: object) -> str:
        calls["render"] += 1
        if calls["render"] == 1:
            raise RuntimeError("simulated rendering failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr("riskprobe.service.discover_rules", fake_discover)
    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    monkeypatch.setattr("riskprobe.service.render_risk_report", fail_first_render)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        service.run()

    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    (run_dir / "candidate_rules.parquet").write_bytes(b"tampered checkpoint")

    service.run()

    assert calls == {"discover": 2, "validate": 2, "render": 2}
    runtime = RunRuntime(tmp_path / "runs", run_dir.name)
    invalidated = [
        event["node_id"]
        for event in runtime.trace()
        if event["event_type"] == "node_invalidated"
    ]
    assert invalidated == ["discover", "validate", "report"]
    assert not (run_dir / ".incomplete").exists()


def test_missing_evidence_checkpoint_recomputes_validate_and_report_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100)
    calls = {"discover": 0, "validate": 0, "render": 0}
    original_render = service_module.render_risk_report

    def fake_discover(*args: object, **kwargs: object) -> list[RiskRule]:
        calls["discover"] += 1
        return [_rule()]

    def fake_validate(*args: object, **kwargs: object) -> list[EvidenceCard]:
        calls["validate"] += 1
        return [_card()]

    def fail_first_render(*args: object, **kwargs: object) -> str:
        calls["render"] += 1
        if calls["render"] == 1:
            raise RuntimeError("simulated rendering failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr("riskprobe.service.discover_rules", fake_discover)
    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    monkeypatch.setattr("riskprobe.service.render_risk_report", fail_first_render)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        service.run()

    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    (run_dir / "evidence_cards.json").unlink()

    service.run()

    assert calls == {"discover": 1, "validate": 2, "render": 2}
    runtime = RunRuntime(tmp_path / "runs", run_dir.name)
    invalidated = [
        event["node_id"]
        for event in runtime.trace()
        if event["event_type"] == "node_invalidated"
    ]
    assert invalidated == ["validate", "report"]


def test_artifact_producing_checkpoints_record_verified_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr("riskprobe.service.validate_rules", lambda *args, **kwargs: [_card()])

    result = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()
    runtime = RunRuntime(tmp_path / "runs", result.run_id)

    expected = {
        "partition": {"data_profile.json"},
        "discover": {"candidate_rules.parquet"},
        "validate": {"evidence_cards.json"},
        "report": {"metadata_report.json", "risk_report.md"},
        "finalize": {"manifest.json"},
    }
    for node_id, filenames in expected.items():
        checkpoint = runtime.checkpoint(
            node_id,
            input_fingerprint=service_module._node_input_fingerprint(result.run_id, node_id),
        )
        assert checkpoint is not None
        assert {reference.filename for reference in checkpoint.artifact_refs} == filenames
        assert all(reference.schema_version for reference in checkpoint.artifact_refs)


def test_immutable_publish_remains_success_when_runtime_success_recording_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100)
    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [])
    original_succeed = RunRuntime.succeed_node
    failed = False

    def fail_finalize_once(
        self: RunRuntime,
        node_id: str,
        **kwargs: object,
    ):
        nonlocal failed
        if node_id == "finalize" and not failed:
            failed = True
            raise OSError("simulated runtime write failure after publish")
        return original_succeed(self, node_id, **kwargs)

    monkeypatch.setattr(RunRuntime, "succeed_node", fail_finalize_once)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    published = service.run()

    assert failed is True
    assert not (published.run_dir / ".incomplete").exists()
    assert {path.name for path in published.run_dir.iterdir()} == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    }

    reused = service.run()
    runtime = RunRuntime(tmp_path / "runs", published.run_id)
    assert reused.is_existing is True
    assert runtime.node_status("finalize") is NodeStatus.SUCCEEDED
    assert any(event["event_type"] == "run_reconciled" for event in runtime.trace())


def test_real_token_shaped_segment_is_redacted_once_across_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, rows=100)
    real_segment = "segment-deadbeef"
    token = f"segment-{hashlib.sha256(real_segment.encode()).hexdigest()[:8]}"
    double_token = f"segment-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
    card = _card(
        slices=(
            SliceMetrics(
                slice_type="segment",
                slice_value=real_segment,
                metrics=_metrics(1.5),
            ),
        ),
        limitations=(f"single-class institution: {real_segment}",),
    )
    calls = {"validate": 0, "render": 0}
    original_render = service_module.render_risk_report

    def fake_validate(*args: object, **kwargs: object) -> list[EvidenceCard]:
        calls["validate"] += 1
        return [card]

    def fail_first_render(*args: object, **kwargs: object) -> str:
        calls["render"] += 1
        if calls["render"] == 1:
            raise RuntimeError("simulated rendering failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr("riskprobe.service.discover_rules", lambda *args, **kwargs: [_rule()])
    monkeypatch.setattr("riskprobe.service.validate_rules", fake_validate)
    monkeypatch.setattr("riskprobe.service.render_risk_report", fail_first_render)
    service = RiskProbeService(config=config, runs_dir=tmp_path / "runs")

    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        service.run()

    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    first_evidence = (run_dir / "evidence_cards.json").read_bytes()
    assert real_segment.encode() not in first_evidence
    assert token.encode() in first_evidence

    service.run()

    final_evidence = (run_dir / "evidence_cards.json").read_bytes()
    report = (run_dir / "risk_report.md").read_text(encoding="utf-8")
    assert calls == {"validate": 1, "render": 2}
    assert final_evidence == first_evidence
    assert real_segment not in report
    assert token in report
    assert double_token not in report


def test_local_handler_filters_private_controlled_recommendation_before_persist(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from riskprobe.agents.decision_contracts import DecisionProposal, DecisionSource
    from riskprobe.agents.decision_controller import DecisionController
    from riskprobe.agents.decision_providers import (
        DecisionProviderMode,
        _DecisionProviderBinding,
        _DecisionProviderIdentity,
        _DecisionProviderRole,
    )
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, PolicyEngine, Principal, Role
    from riskprobe.recommendations.policy import applicable_action_codes
    from riskprobe.registry import DatasetRegistry
    from riskprobe.tools import HandlerToolGateway, LocalRiskProbeToolHandler
    from riskprobe.tools.models import (
        DiscoverResponse,
        InspectResponse,
        _ControlledRecommendRequest,
    )

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    service = RiskProbeService(config=config, runs_dir=runs_dir)
    context = service.run()
    store = EvidenceStore(tmp_path / "controlled-evidence.sqlite3")
    diagnosis = service._diagnose_with_store(
        context.run_id,
        store,
        dataset_id=config.dataset.id,
    )
    profile = service._profile_from_run(context)
    rules = service._rules_from_run(context)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    controller = DecisionController(store, clock=lambda: now)
    preparation = controller.prepare(
        session_id=context.run_id,
        attempt=0,
        anchor_node_id="a" * 64,
        diagnosis_evidence_ids=diagnosis.finding_ids,
        inspect_response=InspectResponse(
            dataset_id=config.dataset.id,
            row_count=profile.row_count,
            feature_count=profile.feature_count,
            metadata_grade=profile.metadata_grade,
            issue_codes=profile.issue_codes,
        ),
        discover_response=DiscoverResponse(
            dataset_id=config.dataset.id,
            rule_ids=tuple(sorted(rule.rule_id for rule in rules)),
        ),
        orchestrator_version="orchestrator-v1",
        planner_version="planner-v1",
    )
    selected_action = applicable_action_codes(
        item.finding for item in preparation.context.findings
    )[0]
    primary_provider = _DecisionProviderIdentity(
        provider_id="test-host",
        mode=DecisionProviderMode.EXTERNAL_HOST,
        version="test-host-v1",
    )
    provider_binding = _DecisionProviderBinding(
        primary=primary_provider,
        fallback=_DecisionProviderIdentity(
            provider_id="deterministic",
            mode=DecisionProviderMode.DETERMINISTIC,
            version="deterministic-decision-provider-v1",
        ),
        selected=primary_provider,
        selected_role=_DecisionProviderRole.PRIMARY,
    )
    submission = controller.submit(
        context_evidence_id=preparation.context_evidence_id,
        proposal=DecisionProposal(
            context_id=preparation.context.context_id,
            diagnosis_evidence_ids=preparation.context.diagnosis_evidence_ids,
            action_codes=(selected_action,),
            source=DecisionSource.EXTERNAL_HOST,
            source_version="test-host-v1",
        ),
        provider_binding=provider_binding,
    )
    handler = LocalRiskProbeToolHandler(
        run_id=context.run_id,
        runs_dir=runs_dir,
        evidence_store=store,
        run_context=context,
        decision_controller=controller,
    )
    gateway = HandlerToolGateway(
        registry=DatasetRegistry.from_mapping({config.dataset.id: config}),
        policy=PolicyEngine(),
        handler=handler,
    )

    response = gateway.invoke(
        Principal(principal_id="controlled-reader", role=Role.OPERATOR),
        _ControlledRecommendRequest(
            dataset_id=config.dataset.id,
            evidence_ids=diagnosis.finding_ids,
            decision_result_evidence_id=submission.result_evidence_id,
        ),
        Budget(max_queries=1),
    )

    assert response.recommendation_ids
    persisted_actions = {
        store.get(evidence_id).payload["action_code"]  # type: ignore[union-attr]
        for evidence_id in response.recommendation_ids
    }
    assert persisted_actions == {selected_action.value}


def test_orchestrate_runtime_provider_config_preserves_v1_run_identity(
    tmp_path: Path,
) -> None:
    from riskprobe.agents.decision_providers import (
        DecisionProviderConfig,
        DecisionProviderMode,
    )
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, Principal, Role

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    baseline = RiskProbeService(config=config, runs_dir=runs_dir).run()
    service = RiskProbeService(
        config=config,
        runs_dir=runs_dir,
        state_dir=tmp_path / "deterministic-state",
        decision_provider_config=DecisionProviderConfig(
            mode=DecisionProviderMode.DETERMINISTIC
        ),
    )

    result = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=Principal(
            principal_id="deterministic-runtime-reader",
            role=Role.ANALYST,
        ),
        budget=Budget(max_queries=16),
    )

    assert result.session_id == baseline.run_id
    assert result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert {path.name for path in baseline.run_dir.iterdir()} == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    }
    records = EvidenceStore(
        tmp_path
        / "deterministic-state"
        / f".{result.session_id}.evidence.sqlite3"
    ).list_run(result.session_id)
    proposal = next(record for record in records if record.kind == "decision.proposal")
    binding = proposal.payload["provider_binding"]
    assert binding["selected_role"] == "primary"
    assert binding["primary"]["mode"] == "deterministic"
    assert binding["fallback"]["mode"] == "deterministic"


def test_orchestrate_uses_strict_injected_external_host_proposal(
    tmp_path: Path,
) -> None:
    from riskprobe.agents.decision_contracts import (
        DecisionContext,
        DecisionProposal,
        DecisionSource,
    )
    from riskprobe.agents.decision_providers import (
        DecisionDisposition,
        DecisionProviderConfig,
        DecisionProviderMode,
        DecisionProviderResolution,
        DeterministicDecisionProvider,
    )
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, Principal, Role

    class ExternalHost:
        mode = DecisionProviderMode.EXTERNAL_HOST
        provider_id = "service-external-host"
        version = "service-external-host-v1"

        def __init__(self) -> None:
            self.calls = 0

        def resolve(
            self,
            *,
            context: DecisionContext,
        ) -> DecisionProviderResolution:
            assert type(context) is DecisionContext
            self.calls += 1
            deterministic = DeterministicDecisionProvider().resolve(
                context=context
            ).proposal
            assert deterministic is not None
            return DecisionProviderResolution(
                disposition=DecisionDisposition.PROPOSAL,
                proposal=DecisionProposal(
                    context_id=context.context_id,
                    diagnosis_evidence_ids=context.diagnosis_evidence_ids,
                    action_codes=deterministic.action_codes,
                    source=DecisionSource.EXTERNAL_HOST,
                    source_version=self.version,
                ),
            )

    provider = ExternalHost()
    config = _small_config(tmp_path)
    state_dir = tmp_path / "external-state"
    service = RiskProbeService(
        config=config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
        decision_provider_config=DecisionProviderConfig(
            mode=DecisionProviderMode.EXTERNAL_HOST,
            provider_id=provider.provider_id,
            provider_version=provider.version,
        ),
        decision_provider=provider,
    )

    result = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=Principal(
            principal_id="external-runtime-reader",
            role=Role.ANALYST,
        ),
        budget=Budget(max_queries=16),
    )

    assert result.review.approved is True
    assert provider.calls == 1
    records = EvidenceStore(
        state_dir / f".{result.session_id}.evidence.sqlite3"
    ).list_run(result.session_id)
    proposal = next(record for record in records if record.kind == "decision.proposal")
    binding = proposal.payload["provider_binding"]
    assert binding["selected_role"] == "primary"
    assert binding["selected"] == {
        "provider_id": provider.provider_id,
        "mode": "external_host",
        "version": provider.version,
    }


@pytest.mark.parametrize("provider_behavior", ("pending", "error"))
def test_orchestrate_caches_provider_unavailable_without_reinvoking_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_behavior: str,
) -> None:
    from riskprobe.agents.contracts import AgentStatus, ReviewReason
    from riskprobe.agents.decision_providers import (
        DecisionDisposition,
        DecisionProviderError,
        DecisionProviderMode,
        DecisionProviderResolution,
        DisabledDecisionProvider,
    )
    from riskprobe.agents.sessions import SessionStore
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, Principal, Role

    calls = 0

    def unavailable(
        self: object,
        *,
        context: object,
    ) -> DecisionProviderResolution:
        nonlocal calls
        del self, context
        calls += 1
        if provider_behavior == "pending":
            return DecisionProviderResolution(
                disposition=DecisionDisposition.PENDING
            )
        raise DecisionProviderError("private service provider failure")

    monkeypatch.setattr(
        DisabledDecisionProvider,
        "mode",
        DecisionProviderMode.EXTERNAL_HOST,
    )
    monkeypatch.setattr(
        DisabledDecisionProvider,
        "provider_id",
        f"service-{provider_behavior}-host",
    )
    monkeypatch.setattr(
        DisabledDecisionProvider,
        "version",
        "service-unavailable-host-v1",
    )
    monkeypatch.setattr(DisabledDecisionProvider, "resolve", unavailable)
    config = _small_config(tmp_path)
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
    )
    principal = Principal(
        principal_id="unavailable-cache-reader",
        role=Role.ANALYST,
    )

    first = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )

    assert first.status is AgentStatus.REJECTED
    assert ReviewReason.TOOL_FAILURE in first.review.reason_codes
    assert calls == 1
    result_path = state_dir / f".{first.session_id}.agent-result.json"
    assert result_path.is_file()
    evidence_store = EvidenceStore(
        state_dir / f".{first.session_id}.evidence.sqlite3"
    )
    unavailable_record = next(
        record
        for record in evidence_store.list_run(first.session_id)
        if record.kind == "decision.unavailable"
    )
    assert unavailable_record.payload["reason"] == f"provider_{provider_behavior}"
    assert "private service provider failure" not in unavailable_record.model_dump_json()
    session_nodes = SessionStore(
        state_dir / f".{first.session_id}.sessions.sqlite3"
    ).replay(first.session_id)
    assert not any(
        node.tool_call is not None and node.tool_call.tool_name == "recommend"
        for node in session_nodes
    )

    replay_calls = 0

    def reject_replay(*args: object, **kwargs: object) -> object:
        nonlocal replay_calls
        del args, kwargs
        replay_calls += 1
        raise AssertionError("cache replay must not invoke provider")

    monkeypatch.setattr(DisabledDecisionProvider, "resolve", reject_replay)

    second = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )

    assert second == first
    assert replay_calls == 0


def test_orchestrate_default_fallback_is_sidecar_only_and_cache_skips_decision_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.agents.decision_controller import DecisionController
    from riskprobe.agents.decision_providers import (
        DeterministicDecisionProvider,
        DisabledDecisionProvider,
    )
    from riskprobe.agents.sessions import SessionStore
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, Principal, Role
    from riskprobe.tools import LocalRiskProbeToolHandler

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=config,
        runs_dir=runs_dir,
        state_dir=state_dir,
    )
    principal = Principal(principal_id="decision-cache-reader", role=Role.ANALYST)
    first = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )
    evidence_path = state_dir / f".{first.session_id}.evidence.sqlite3"
    session_path = state_dir / f".{first.session_id}.sessions.sqlite3"
    evidence_store = EvidenceStore(evidence_path)
    records = evidence_store.list_run(first.session_id)
    assert [record.kind for record in records].count("decision.context") == 1
    assert [record.kind for record in records].count("decision.proposal") == 1
    assert [record.kind for record in records].count("decision.result") == 1
    assert all(
        EvidenceStore.content_id(record) not in first.evidence_ids
        for record in records
        if record.kind.startswith("decision.")
    )
    run_dir = runs_dir / first.session_id
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    }
    session_count = len(SessionStore(session_path).replay(first.session_id))
    evidence_count = len(records)
    calls = {"prepare": 0, "submit": 0, "disabled": 0, "fallback": 0, "tool": 0}

    def reject(name: str):
        def fail(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls[name] += 1
            raise AssertionError(f"cache hit must not call {name}")

        return fail

    monkeypatch.setattr(DecisionController, "prepare", reject("prepare"))
    monkeypatch.setattr(DecisionController, "submit", reject("submit"))
    monkeypatch.setattr(DisabledDecisionProvider, "resolve", reject("disabled"))
    monkeypatch.setattr(
        DeterministicDecisionProvider,
        "resolve",
        reject("fallback"),
    )
    monkeypatch.setattr(LocalRiskProbeToolHandler, "handle", reject("tool"))

    second = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )

    assert second == first
    assert calls == {"prepare": 0, "submit": 0, "disabled": 0, "fallback": 0, "tool": 0}
    assert len(SessionStore(session_path).replay(first.session_id)) == session_count
    assert len(EvidenceStore(evidence_path).list_run(first.session_id)) == evidence_count


def test_orchestrate_cache_fails_closed_when_decision_evidence_is_tampered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    from riskprobe.policy import Budget, Principal, Role
    from riskprobe.tools import LocalRiskProbeToolHandler

    config = _small_config(tmp_path)
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
    )
    principal = Principal(principal_id="tamper-reader", role=Role.ANALYST)
    first = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=principal,
        budget=Budget(max_queries=16),
    )
    evidence_path = state_dir / f".{first.session_id}.evidence.sqlite3"
    with sqlite3.connect(evidence_path) as connection:
        connection.execute(
            "UPDATE evidence_records SET kind = ? WHERE kind = ?",
            ("decision.proposal", "decision.result"),
        )
        connection.commit()
    tool_calls = 0

    def reject_tool_call(*args: object, **kwargs: object) -> object:
        nonlocal tool_calls
        del args, kwargs
        tool_calls += 1
        raise AssertionError("tampered cache must not invoke tools")

    monkeypatch.setattr(LocalRiskProbeToolHandler, "handle", reject_tool_call)

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        service.orchestrate(
            dataset_id=config.dataset.id,
            principal=principal,
            budget=Budget(max_queries=16),
        )

    assert tool_calls == 0


def test_public_recommendation_v1_signatures_remain_exact() -> None:
    import inspect

    from riskprobe.recommendations import build_recommendations
    from riskprobe.tools import RecommendRequest, RecommendResponse

    assert set(RecommendRequest.model_fields) == {"dataset_id", "evidence_ids"}
    assert set(RecommendResponse.model_fields) == {
        "dataset_id",
        "recommendation_ids",
    }
    assert list(inspect.signature(RiskProbeService.recommend).parameters) == [
        "self",
        "run_id",
        "evidence_ids",
        "all_current_diagnostics",
    ]
    assert list(inspect.signature(build_recommendations).parameters) == [
        "report",
        "metadata_grade",
    ]


def test_local_handler_rejects_diagnostics_from_snapshot_outside_bound_run(
    tmp_path: Path,
) -> None:
    from riskprobe.evidence import EvidenceStore
    from riskprobe.policy import Budget, PolicyEngine, Principal, Role
    from riskprobe.registry import DatasetRegistry
    from riskprobe.tools import (
        DiagnoseRequest,
        HandlerToolGateway,
        LocalRiskProbeToolHandler,
        ToolContractError,
    )

    config = _small_config(tmp_path)
    runs_dir = tmp_path / "runs"
    context = RiskProbeService(config=config, runs_dir=runs_dir).run()
    pl.read_parquet(config.dataset.path).head(80).write_parquet(config.dataset.path)
    store = EvidenceStore(tmp_path / "bound-diagnosis.sqlite3")
    handler = LocalRiskProbeToolHandler(
        run_id=context.run_id,
        runs_dir=runs_dir,
        evidence_store=store,
        run_context=context,
    )
    gateway = HandlerToolGateway(
        registry=DatasetRegistry.from_mapping({config.dataset.id: config}),
        policy=PolicyEngine(),
        handler=handler,
    )

    with pytest.raises(ToolContractError, match="^tool handler failed$"):
        gateway.invoke(
            Principal(principal_id="snapshot-reader", role=Role.ANALYST),
            DiagnoseRequest(dataset_id=config.dataset.id),
            Budget(max_queries=1),
        )

    assert store.list_run(context.run_id) == ()
