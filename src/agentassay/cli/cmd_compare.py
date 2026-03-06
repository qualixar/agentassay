"""CLI command: ``agentassay compare`` -- regression comparison.

Loads baseline and current result JSON files, runs Fisher's exact test,
and reports whether a statistically significant regression occurred.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import click
from rich.panel import Panel
from rich.table import Table

from agentassay.cli.helpers import (
    console,
    extract_passed_list,
    load_json,
    write_json,
)
from agentassay.statistics.confidence import wilson_interval
from agentassay.statistics.hypothesis import fisher_exact_regression


@click.command("compare")
@click.option(
    "--baseline", "-b",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with baseline trial results.",
)
@click.option(
    "--current", "-c",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with current trial results.",
)
@click.option(
    "--alpha", "-a",
    type=float,
    default=0.05,
    show_default=True,
    help="Significance level for the regression test.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output JSON file for comparison results.",
)
def compare_command(
    baseline: str,
    current: str,
    alpha: float,
    output: str | None,
) -> None:
    """Compare baseline vs. current results for regression.

    Loads two JSON result files, extracts pass/fail outcomes, and runs
    Fisher's exact test to determine whether a statistically significant
    regression occurred.

    \b
    Result JSON format (either works):
        [{"passed": true}, {"passed": false}, ...]
        {"results": [{"passed": true}, ...]}

    \b
    Examples:
        agentassay compare --baseline v1.json --current v2.json
        agentassay compare -b baseline.json -c current.json --alpha 0.01
    """
    console.print(Panel.fit(
        "[bold]AgentAssay[/bold] -- Regression Comparison",
        border_style="blue",
    ))

    # Load results
    baseline_data = load_json(baseline, "baseline")
    current_data = load_json(current, "current")

    baseline_passed = extract_passed_list(baseline_data)
    current_passed = extract_passed_list(current_data)

    baseline_n = len(baseline_passed)
    current_n = len(current_passed)
    baseline_k = sum(baseline_passed)
    current_k = sum(current_passed)

    # Compute CIs
    baseline_ci = wilson_interval(baseline_k, baseline_n)
    current_ci = wilson_interval(current_k, current_n)

    # Run regression test
    result = fisher_exact_regression(
        baseline_passes=baseline_k,
        baseline_n=baseline_n,
        current_passes=current_k,
        current_n=current_n,
        alpha=alpha,
    )

    # Display summary table
    summary = Table(title="Regression Comparison", show_header=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Baseline", style="white", justify="right")
    summary.add_column("Current", style="white", justify="right")
    summary.add_row("Trials", str(baseline_n), str(current_n))
    summary.add_row("Passed", str(baseline_k), str(current_k))
    summary.add_row(
        "Pass Rate",
        f"{result.baseline_rate:.2%}",
        f"{result.current_rate:.2%}",
    )
    summary.add_row(
        "95% CI",
        f"[{baseline_ci.lower:.4f}, {baseline_ci.upper:.4f}]",
        f"[{current_ci.lower:.4f}, {current_ci.upper:.4f}]",
    )
    console.print(summary)

    # Display test results
    test_table = Table(title="Statistical Test", show_header=True)
    test_table.add_column("Field", style="cyan")
    test_table.add_column("Value", style="white")
    test_table.add_row("Test", result.test_name)
    test_table.add_row("Statistic", f"{result.statistic:.4f}")
    test_table.add_row("p-value", f"{result.p_value:.6f}")
    test_table.add_row("Alpha", f"{alpha:.4f}")
    test_table.add_row("Effect Size", f"{result.effect_size_name} = {result.effect_size:+.4f}")
    test_table.add_row(
        "Significant",
        "[red]YES[/red]" if result.significant else "[green]NO[/green]",
    )
    console.print(test_table)

    # Verdict
    if result.significant:
        console.print(Panel(
            f"[bold red]REGRESSION DETECTED[/bold red]\n\n{result.interpretation}",
            border_style="red",
            title="Verdict",
        ))
    else:
        console.print(Panel(
            f"[bold green]NO REGRESSION[/bold green]\n\n{result.interpretation}",
            border_style="green",
            title="Verdict",
        ))

    # Write output
    if output:
        output_data = {
            "command": "compare",
            "baseline_file": baseline,
            "current_file": current,
            "baseline_n": baseline_n,
            "baseline_passed": baseline_k,
            "baseline_rate": result.baseline_rate,
            "current_n": current_n,
            "current_passed": current_k,
            "current_rate": result.current_rate,
            "test_name": result.test_name,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "alpha": alpha,
            "effect_size": result.effect_size,
            "effect_size_name": result.effect_size_name,
            "significant": result.significant,
            "interpretation": result.interpretation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_data, output, "comparison results")

    # Exit code reflects verdict
    if result.significant:
        sys.exit(1)
