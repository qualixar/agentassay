"""CLI command: ``agentassay run`` -- run agent assay trials.

Loads agent configuration and test scenario from YAML files, validates
them, and displays a Rich-formatted summary of the configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from agentassay.cli.helpers import console, load_yaml, write_json


@click.command("run")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="YAML config file with AssayConfig parameters.",
)
@click.option(
    "--scenario", "-s",
    type=click.Path(exists=True),
    help="YAML scenario file with TestScenario definition.",
)
@click.option(
    "--n", "-n",
    type=int,
    default=None,
    help="Number of trials (overrides config).",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output JSON file for results.",
)
def run_command(
    config: str | None,
    scenario: str | None,
    n: int | None,
    output: str | None,
) -> None:
    """Run agent assay trials.

    Loads the agent configuration and test scenario from YAML files,
    executes N stochastic trials, and outputs results as JSON with
    a Rich-formatted summary.

    Note: This command requires an agent callable to be configured.
    For full trial execution, use the Python API. This CLI command
    demonstrates the workflow and validates configuration files.

    \b
    Examples:
        agentassay run --config assay.yaml --scenario qa.yaml -n 50
        agentassay run -c config.yaml -s scenario.yaml -o results.json
    """
    console.print(Panel.fit(
        "[bold]AgentAssay[/bold] -- Stochastic Trial Runner",
        border_style="blue",
    ))

    # Load configuration
    assay_kwargs: dict[str, Any] = {}
    if config:
        cfg_data = load_yaml(config, "config")
        # Map YAML keys to AssayConfig fields
        key_map = {
            "num_trials": "num_trials",
            "n": "num_trials",
            "trials": "num_trials",
            "alpha": "significance_level",
            "significance_level": "significance_level",
            "power": "power",
            "effect_size_threshold": "effect_size_threshold",
            "confidence_method": "confidence_method",
            "regression_test": "regression_test",
            "use_sprt": "use_sprt",
            "seed": "seed",
            "timeout": "timeout_seconds",
            "timeout_seconds": "timeout_seconds",
            "max_cost": "max_cost_usd",
            "max_cost_usd": "max_cost_usd",
            "parallel": "parallel_trials",
            "parallel_trials": "parallel_trials",
        }
        for yaml_key, model_key in key_map.items():
            if yaml_key in cfg_data:
                assay_kwargs[model_key] = cfg_data[yaml_key]

    # Override num_trials from CLI if provided
    if n is not None:
        assay_kwargs["num_trials"] = n

    try:
        from agentassay.core.models import AssayConfig as AC
        assay_cfg = AC(**assay_kwargs)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    # Display config
    config_table = Table(title="Assay Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Trials", str(assay_cfg.num_trials))
    config_table.add_row("Significance (alpha)", f"{assay_cfg.significance_level:.4f}")
    config_table.add_row("Power (1-beta)", f"{assay_cfg.power:.4f}")
    config_table.add_row("Confidence Method", assay_cfg.confidence_method)
    config_table.add_row("Regression Test", assay_cfg.regression_test)
    config_table.add_row("SPRT Enabled", str(assay_cfg.use_sprt))
    config_table.add_row("Timeout (s)", f"{assay_cfg.timeout_seconds:.0f}")
    config_table.add_row("Max Cost ($)", f"{assay_cfg.max_cost_usd:.2f}")
    config_table.add_row("Parallel Trials", str(assay_cfg.parallel_trials))
    if assay_cfg.seed is not None:
        config_table.add_row("Seed", str(assay_cfg.seed))
    console.print(config_table)

    # Load scenario
    if scenario:
        scenario_data = load_yaml(scenario, "scenario")
        console.print(f"\n[cyan]Scenario:[/cyan] {scenario_data.get('name', 'unnamed')}")
        console.print(
            f"[dim]{scenario_data.get('description', 'No description')}[/dim]"
        )

    console.print()
    console.print(
        "[yellow]Note:[/yellow] Full trial execution requires an agent callable. "
        "Use the Python API (TrialRunner) for actual trial runs."
    )
    console.print(
        "[dim]This command validates config/scenario files and displays parameters.[/dim]"
    )

    # If output requested, write the config as JSON for reference
    if output:
        output_data = {
            "command": "run",
            "config": assay_cfg.model_dump(mode="json"),
            "scenario_file": scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "config_validated",
            "note": "Trial execution requires Python API. See agentassay.core.runner.TrialRunner.",
        }
        write_json(output_data, output, "run output")
