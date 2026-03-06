# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""CLI command: ``agentassay coverage`` -- coverage analysis.

Computes and displays the 5-dimensional coverage vector from trial
result JSON files containing execution traces.
"""

from __future__ import annotations

import click
from rich.panel import Panel
from rich.table import Table

from agentassay.cli.helpers import console, load_json


@click.command("coverage")
@click.option(
    "--results",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with trial results containing execution traces.",
)
@click.option(
    "--tools",
    type=str,
    default=None,
    help="Comma-separated list of known tool names for the agent.",
)
@click.option(
    "--models",
    type=str,
    default=None,
    help="Comma-separated list of known model names for the agent.",
)
def coverage_command(
    results: str,
    tools: str | None,
    models: str | None,
) -> None:
    """Show coverage metrics for agent test results.

    Computes the 5-dimensional coverage vector:
    C = (C_tool, C_path, C_state, C_boundary, C_model)

    Each dimension measures how thoroughly the test suite exercises
    a specific aspect of the agent's behavior.

    \b
    Examples:
        agentassay coverage --results trials.json --tools search,calculate,write
        agentassay coverage -r results.json --tools search --models gpt-4o,claude-opus-4-6
    """
    console.print(
        Panel.fit(
            "[bold]AgentAssay[/bold] -- Coverage Analysis",
            border_style="blue",
        )
    )

    # Parse known tools and models
    known_tools: set[str] = set()
    if tools:
        known_tools = {t.strip() for t in tools.split(",") if t.strip()}

    known_models: set[str] = set()
    if models:
        known_models = {m.strip() for m in models.split(",") if m.strip()}

    # Load results
    results_data = load_json(results, "results")

    # Extract traces and compute coverage
    from agentassay.coverage import AgentCoverageCollector

    collector = AgentCoverageCollector(
        known_tools=known_tools,
        known_models=known_models if known_models else None,
    )

    # Try to reconstruct traces from results JSON
    if isinstance(results_data, list):
        items = results_data
    elif isinstance(results_data, dict):
        items = results_data.get("results", results_data.get("trials", []))
    else:
        items = []

    trace_count = 0
    tools_observed: set[str] = set()
    models_observed: set[str] = set()
    paths_observed: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        trace_data = item.get("trace")
        if trace_data is None:
            continue

        try:
            from agentassay.core.models import ExecutionTrace

            trace = ExecutionTrace(**trace_data)
            collector.update(trace)
            trace_count += 1
            tools_observed.update(trace.tools_used)
            models_observed.add(trace.model)
            paths_observed.add(" -> ".join(trace.decision_path))
        except Exception:
            # Skip malformed traces
            continue

    if trace_count == 0:
        console.print("[yellow]Warning:[/yellow] No valid execution traces found in results file.")
        console.print("[dim]Coverage requires TrialResult objects with ExecutionTrace data.[/dim]")

        # Show empty coverage
        cov_table = Table(title="Coverage Vector (no data)", show_header=True)
        cov_table.add_column("Dimension", style="cyan")
        cov_table.add_column("Score", justify="right")
        cov_table.add_column("Status", justify="center")
        for dim in ["Tool", "Path", "State", "Boundary", "Model"]:
            cov_table.add_row(dim, "0.00%", "[dim]N/A[/dim]")
        cov_table.add_row("Overall", "0.00%", "[dim]N/A[/dim]", style="bold")
        console.print(cov_table)
        return

    # Get coverage snapshot
    snapshot = collector.snapshot()

    # Display coverage table
    cov_table = Table(title="Coverage Vector (5 Dimensions)", show_header=True)
    cov_table.add_column("Dimension", style="cyan")
    cov_table.add_column("Score", justify="right")
    cov_table.add_column("Bar", min_width=20)
    cov_table.add_column("Status", justify="center")

    def _bar(value: float) -> str:
        """Create a visual bar for coverage percentage."""
        filled = int(value * 20)
        return "[green]" + "#" * filled + "[/green]" + "[dim]" + "-" * (20 - filled) + "[/dim]"

    def _status(value: float) -> str:
        if value >= 0.80:
            return "[green]GOOD[/green]"
        elif value >= 0.50:
            return "[yellow]MODERATE[/yellow]"
        else:
            return "[red]LOW[/red]"

    for dim_name, dim_value in snapshot.dimensions.items():
        cov_table.add_row(
            dim_name.capitalize(),
            f"{dim_value:.2%}",
            _bar(dim_value),
            _status(dim_value),
        )

    cov_table.add_row(
        "Overall (geo. mean)",
        f"{snapshot.overall:.2%}",
        _bar(snapshot.overall),
        _status(snapshot.overall),
        style="bold",
    )

    console.print(cov_table)

    # Weakest dimension
    weakest_name, weakest_val = snapshot.weakest
    console.print(f"\n[yellow]Weakest dimension:[/yellow] {weakest_name} ({weakest_val:.2%})")

    # Summary stats
    stats_table = Table(title="Analysis Summary", show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Traces analyzed", str(trace_count))
    stats_table.add_row(
        "Known tools", str(known_tools) if known_tools else "[dim]auto-detected[/dim]"
    )
    stats_table.add_row("Tools observed", ", ".join(sorted(tools_observed)) or "[dim]none[/dim]")
    stats_table.add_row(
        "Known models", str(known_models) if known_models else "[dim]auto-detected[/dim]"
    )
    stats_table.add_row("Models observed", ", ".join(sorted(models_observed)) or "[dim]none[/dim]")
    stats_table.add_row("Unique paths", str(len(paths_observed)))
    console.print(stats_table)
