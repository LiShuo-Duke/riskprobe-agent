import json
from pathlib import Path
from typing import NoReturn, Sequence

import click
import typer
from typer.core import TyperGroup

from riskprobe.config import ProjectConfig
from riskprobe.service import RiskProbeService
from riskprobe.synthetic import generate_behavior_dataset


class StructuredErrorGroup(TyperGroup):
    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: object,
    ) -> object:
        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except click.ClickException:
            if not standalone_mode:
                raise
            typer.echo(
                json.dumps(
                    {
                        "error": "argument_error",
                        "message": "Use --help to review the required command and option values.",
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(2)
        if standalone_mode and isinstance(result, int) and result:
            raise SystemExit(result)
        return result


app = typer.Typer(
    cls=StructuredErrorGroup,
    help="Run local RiskProbe synthetic demonstrations and analyses.",
)


def _fail(category: str, message: str) -> NoReturn:
    typer.echo(json.dumps({"error": category, "message": message}, sort_keys=True))
    raise typer.Exit(code=2)


def _service(config: Path | None, runs_dir: Path | None) -> RiskProbeService:
    if config is None or runs_dir is None:
        _fail("input_error", "Provide both --config and --runs-dir local paths.")
    try:
        project_config = ProjectConfig.from_yaml(config)
    except Exception:
        _fail(
            "configuration_error",
            "Check that --config names a readable, valid local YAML configuration.",
        )
    try:
        return RiskProbeService(config=project_config, runs_dir=runs_dir)
    except Exception:
        _fail("runs_directory_error", "Choose a writable local --runs-dir directory.")


@app.command()
def synthetic(
    output: Path | None = typer.Option(..., help="Destination Parquet file."),
    rows: int = typer.Option(..., help="Number of synthetic rows to create."),
    seed: int = typer.Option(..., help="Deterministic synthetic-data seed."),
) -> None:
    """Write a deterministic public synthetic Parquet dataset."""
    if output is None:
        _fail("input_error", "Provide a writable local --output Parquet path.")
    try:
        frame, truth = generate_behavior_dataset(rows=rows, seed=seed)
    except (TypeError, ValueError):
        _fail("input_error", "Use positive --rows and a non-negative integer --seed.")
    try:
        frame.write_parquet(output)
    except OSError:
        _fail("output_error", "Choose a writable local --output Parquet path.")
    typer.echo(
        json.dumps(
            {
                "columns": frame.width,
                "command": "synthetic",
                "rows": frame.height,
                "truth_rule_ids": [rule.rule_id for rule in truth.hidden_rules],
            },
            sort_keys=True,
        )
    )


@app.command()
def inspect(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Print a safe JSON dataset-profile summary."""
    service = _service(config, runs_dir)
    try:
        profile = service.inspect()
    except Exception:
        _fail(
            "inspection_error",
            "Check the configured local dataset and column roles, then rerun inspect.",
        )
    typer.echo(
        json.dumps(
            {
                "command": "inspect",
                "feature_count": profile.feature_count,
                "metadata_grade": profile.metadata_grade,
                "row_count": profile.row_count,
                "segment_count": len(profile.segment_counts),
            },
            sort_keys=True,
        )
    )


@app.command()
def discover(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Print safe JSON candidate-rule identifiers."""
    service = _service(config, runs_dir)
    try:
        rule_ids = sorted(rule.rule_id for rule in service.discover())
    except Exception:
        _fail(
            "discovery_error",
            "Check the configured local dataset and feature families, then rerun discover.",
        )
    typer.echo(
        json.dumps(
            {
                "candidate_rule_count": len(rule_ids),
                "command": "discover",
                "rule_ids": rule_ids,
            },
            sort_keys=True,
        )
    )


@app.command()
def run(
    config: Path = typer.Option(..., help="Local YAML project configuration."),
    runs_dir: Path = typer.Option(..., help="Local directory for immutable runs."),
) -> None:
    """Create or reuse a complete immutable local analysis run."""
    service = _service(config, runs_dir)
    try:
        context = service.run()
    except Exception:
        _fail(
            "run_error",
            "Check the configured local dataset and runs directory, then rerun locally.",
        )
    typer.echo(
        json.dumps(
            {
                "artifact_count": 6,
                "command": "run",
                "metadata_grade": service.config.metadata_grade,
                "reused": context.is_existing,
                "run_id": context.run_id,
            },
            sort_keys=True,
        )
    )
