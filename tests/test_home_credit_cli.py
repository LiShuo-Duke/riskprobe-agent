import json

import polars as pl
from typer.testing import CliRunner

from riskprobe.cli import app


runner = CliRunner()


def test_prepare_home_credit_rejects_missing_application_table(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "prepare-home-credit",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "x.parquet"),
        ],
    )

    assert result.exit_code == 2
    assert "application_train.csv" in result.stdout


def test_prepare_home_credit_prints_only_aggregate_summary(tmp_path) -> None:
    pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "NAME_INCOME_TYPE": ["Working", "Pensioner"],
            "DAYS_BIRTH": [-10000, -20000],
            "AMT_INCOME_TOTAL": [100.0, 200.0],
        }
    ).write_csv(tmp_path / "application_train.csv")
    pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "DAYS_DECISION": [-10, -20],
            "AMT_APPLICATION": [100.0, 200.0],
            "AMT_CREDIT": [90.0, 180.0],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
        }
    ).write_csv(tmp_path / "previous_application.csv")
    output = tmp_path / "prepared.parquet"

    result = runner.invoke(
        app,
        ["prepare-home-credit", "--input-dir", str(tmp_path), "--output", str(output)],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["command"] == "prepare-home-credit"
    assert payload["rows"] == 2
    assert payload["source_table_count"] == 2
    assert output.exists()
    assert "Working" not in result.stdout
