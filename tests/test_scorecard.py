from __future__ import annotations

import numpy as np
import polars as pl

from riskprobe.rules.scorecard import fit_woe_binning


def test_woe_binning_is_train_only_and_handles_missing_values() -> None:
    train = pl.DataFrame(
        {
            "income": [10, 20, 30, 40, 50, 60, None, 80],
            "target": [1, 1, 1, 0, 0, 0, 1, 0],
        }
    )
    model = fit_woe_binning(
        train,
        feature="income",
        target_col="target",
        max_bins=4,
        min_bin_fraction=0.1,
    )

    transformed = model.transform(pl.Series("income", [5, 35, 100, None]))

    assert model.iv > 0
    assert len(model.edges) <= 3
    assert len(transformed) == 4
    assert np.isfinite(transformed).all()
    assert transformed[-1] == model.missing_woe


def test_monotonic_binning_merges_bad_rate_violations() -> None:
    values = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    targets = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    model = fit_woe_binning(
        pl.DataFrame({"x": values, "target": targets}),
        feature="x",
        target_col="target",
        max_bins=6,
        min_bin_fraction=0.05,
        monotonic="increasing",
    )

    assert model.monotonic == "increasing"
    assert all(
        left <= right
        for left, right in zip(model.bad_rates, model.bad_rates[1:], strict=False)
    )


def test_transform_uses_frozen_training_edges() -> None:
    train = pl.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "target": [0, 0, 0, 1, 1, 1]}
    )
    model = fit_woe_binning(
        train,
        feature="x",
        target_col="target",
        max_bins=3,
        min_bin_fraction=0.1,
    )

    before = model.edges
    transformed = model.transform(pl.Series("x", [0, 100]))

    assert model.edges == before
    assert transformed.shape == (2,)
    assert np.isfinite(transformed).all()


def test_scorecard_fuses_woe_features_and_rule_hits() -> None:
    from riskprobe.models import Condition, RiskRule
    from riskprobe.rules.scorecard import fit_scorecard

    train = pl.DataFrame(
        {
            "x": list(range(1, 13)),
            "target": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )
    rule = RiskRule(
        rule_id="high_x",
        conditions=(Condition(feature="x", operator=">", value=8.0),),
        origin="test",
    )

    model = fit_scorecard(
        train,
        feature_names=("x",),
        target_col="target",
        rules=(rule,),
        monotonic="increasing",
    )
    prediction = model.predict(
        pl.DataFrame({"x": [2, 10, 12]}),
        top_reason_codes=2,
    )

    assert len(model.coefficients) == 2
    assert prediction.probabilities[2] > prediction.probabilities[0]
    assert prediction.risk_scores[2] > prediction.risk_scores[0]
    assert prediction.risk_levels[2] in {"high", "critical"}
    assert prediction.reason_codes[2]


def test_scorecard_transform_is_frozen_and_validates_required_columns() -> None:
    import pytest
    from riskprobe.rules.scorecard import fit_scorecard

    train = pl.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "target": [0, 0, 0, 1, 1, 1]}
    )
    model = fit_scorecard(train, feature_names=("x",), target_col="target")
    edges = model.binning_models[0].edges

    prediction = model.predict(pl.DataFrame({"x": [0, 10]}))

    assert model.binning_models[0].edges == edges
    assert len(prediction.probabilities) == 2
    with pytest.raises(ValueError, match="feature columns"):
        model.predict(pl.DataFrame({"other": [1]}))
