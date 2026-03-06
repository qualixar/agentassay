"""CLI command: ``agentassay mutate`` -- mutation testing.

Lists available mutation operators, validates configuration, and
displays the mutation score interpretation guide.
"""

from __future__ import annotations

from datetime import datetime, timezone

import click
from rich.panel import Panel
from rich.table import Table

from agentassay.cli.helpers import console, load_yaml, write_json


@click.command("mutate")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="YAML config file with AgentConfig parameters.",
)
@click.option(
    "--scenario", "-s",
    type=click.Path(exists=True),
    help="YAML scenario file with TestScenario definition.",
)
@click.option(
    "--operators",
    type=str,
    default=None,
    help="Comma-separated list of operator categories: prompt,tool,model,context",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output JSON file for mutation results.",
)
def mutate_command(
    config: str | None,
    scenario: str | None,
    operators: str | None,
    output: str | None,
) -> None:
    """Run mutation testing on an agent.

    Applies mutation operators (prompt, tool, model, context perturbations)
    and measures how many the test suite detects. A high mutation score
    means the tests are sensitive to changes.

    \b
    Operator categories:
        prompt   - Synonym, order, noise, and drop mutations
        tool     - Removal, reorder, and noise mutations
        model    - Swap and version mutations
        context  - Truncation, noise, and permutation mutations

    \b
    Examples:
        agentassay mutate --config agent.yaml --scenario qa.yaml
        agentassay mutate -c agent.yaml -s qa.yaml --operators prompt,tool
    """
    console.print(Panel.fit(
        "[bold]AgentAssay[/bold] -- Mutation Testing",
        border_style="blue",
    ))

    # Parse operator categories
    selected_categories: list[str] | None = None
    if operators:
        selected_categories = [cat.strip().lower() for cat in operators.split(",")]
        valid_cats = {"prompt", "tool", "model", "context"}
        invalid = set(selected_categories) - valid_cats
        if invalid:
            raise click.ClickException(
                f"Unknown operator categories: {invalid}. "
                f"Valid: {sorted(valid_cats)}"
            )

    # Display selected operators
    from agentassay.mutation import DEFAULT_OPERATORS

    ops = DEFAULT_OPERATORS
    if selected_categories:
        ops = [op for op in DEFAULT_OPERATORS if op.category in selected_categories]

    op_table = Table(title="Mutation Operators", show_header=True)
    op_table.add_column("#", style="dim", justify="right")
    op_table.add_column("Name", style="cyan")
    op_table.add_column("Category", style="yellow")

    for i, op in enumerate(ops, 1):
        op_table.add_row(str(i), op.name, op.category)

    console.print(op_table)
    console.print(f"\n[cyan]Total operators:[/cyan] {len(ops)}")

    if config:
        cfg_data = load_yaml(config, "agent config")
        console.print(f"[cyan]Agent:[/cyan] {cfg_data.get('name', 'unnamed')}")

    if scenario:
        sc_data = load_yaml(scenario, "scenario")
        console.print(f"[cyan]Scenario:[/cyan] {sc_data.get('name', 'unnamed')}")

    console.print()
    console.print(
        "[yellow]Note:[/yellow] Full mutation execution requires an agent callable. "
        "Use the Python API (MutationRunner) for actual mutation runs."
    )

    # Mutation score explanation
    console.print(Panel(
        "[bold]Mutation Score Interpretation[/bold]\n\n"
        "  [green]>= 0.80[/green]  Strong test suite (detects most perturbations)\n"
        "  [yellow]0.50 - 0.80[/yellow]  Moderate (some blind spots)\n"
        "  [red]< 0.50[/red]  Weak (tests pass regardless of mutations)",
        border_style="dim",
    ))

    if output:
        output_data = {
            "command": "mutate",
            "operators": [
                {"name": op.name, "category": op.category}
                for op in ops
            ],
            "operator_count": len(ops),
            "selected_categories": selected_categories,
            "config_file": config,
            "scenario_file": scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "operators_listed",
            "note": "Execution requires Python API. See agentassay.mutation.MutationRunner.",
        }
        write_json(output_data, output, "mutation config")
