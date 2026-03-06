# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Demo command — WOW in 60 seconds.

Generates realistic demo scenarios with synthetic ExecutionTrace objects,
runs the full AgentAssay analysis pipeline, and produces HTML + JSON reports
with automatic browser opening.

This is the "first experience" command for new users exploring AgentAssay.
"""

from __future__ import annotations

import random
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentassay.core.models import ExecutionTrace, StepTrace, TrialResult
from agentassay.coverage.aggregate import AgentCoverageCollector
from agentassay.efficiency.fingerprint import BehavioralFingerprint
from agentassay.reporting.html import HTMLReporter
from agentassay.reporting.json_export import JSONExporter
from agentassay.verdicts.verdict import VerdictFunction

console = Console()


# ===================================================================
# Demo Scenarios — Realistic Agent Behaviors
# ===================================================================


def _generate_demo_trace(
    scenario_id: str,
    trial_index: int,
    model: str,
    tool_names: list[str],
    pass_rate: float,
) -> ExecutionTrace:
    """Generate a single realistic execution trace for a demo scenario.

    Parameters
    ----------
    scenario_id
        Scenario identifier (e.g., "ecommerce_search").
    trial_index
        Trial number (for reproducible randomness).
    model
        LLM model name (e.g., "gpt-4o").
    tool_names
        List of available tool names for this scenario.
    pass_rate
        Probability that this trace succeeds (0.0-1.0).

    Returns
    -------
    ExecutionTrace
        A realistic trace with 3-10 steps.
    """
    rng = random.Random(hash(f"{scenario_id}-{trial_index}"))

    # Determine success based on pass rate
    success = rng.random() < pass_rate

    # Generate 3-10 steps
    num_steps = rng.randint(3, 10)
    steps: list[StepTrace] = []

    total_duration = 0.0
    total_cost = 0.0

    for i in range(num_steps):
        action = rng.choice(["tool_call", "llm_response", "decision"])
        tool_name = rng.choice(tool_names) if action == "tool_call" else None

        # Random step timing
        duration_ms = rng.uniform(50.0, 500.0)
        total_duration += duration_ms

        # Synthetic token cost
        tokens = rng.randint(50, 300)
        cost = tokens * 0.00001  # ~$0.01 per 1K tokens
        total_cost += cost

        steps.append(
            StepTrace(
                step_index=i,
                action=action,
                tool_name=tool_name,
                tool_input={"query": f"step_{i}_input"} if tool_name else None,
                tool_output=f"step_{i}_output" if tool_name else None,
                llm_input=f"step_{i}_prompt" if action == "llm_response" else None,
                llm_output=f"step_{i}_completion" if action == "llm_response" else None,
                model=model,
                duration_ms=duration_ms,
                metadata={"tokens": tokens},
            )
        )

    # Generate realistic output
    output_data = {
        "result": "success" if success else "failure",
        "items": [f"item_{j}" for j in range(rng.randint(1, 5))],
    }

    return ExecutionTrace(
        trace_id=f"demo-{scenario_id}-{trial_index}",
        scenario_id=scenario_id,
        steps=steps,
        input_data={"scenario": scenario_id},
        output_data=output_data,
        success=success,
        error=None if success else "Mock failure for demo",
        total_duration_ms=total_duration,
        total_cost_usd=total_cost,
        model=model,
        framework="custom",
        timestamp=datetime.now(timezone.utc),
        metadata={"demo": True},
    )


def _generate_demo_scenario(
    scenario_id: str,
    scenario_name: str,
    pass_rate: float,
    num_trials: int,
    tool_names: list[str],
) -> tuple[list[ExecutionTrace], list[TrialResult]]:
    """Generate a complete scenario with N trials.

    Returns
    -------
    tuple[list[ExecutionTrace], list[TrialResult]]
        (traces, trial_results)
    """
    traces: list[ExecutionTrace] = []
    results: list[TrialResult] = []

    for i in range(num_trials):
        trace = _generate_demo_trace(
            scenario_id=scenario_id,
            trial_index=i,
            model="gpt-4o",
            tool_names=tool_names,
            pass_rate=pass_rate,
        )
        traces.append(trace)

        # Trial result: passed = trace.success
        result = TrialResult(
            trial_id=trace.trace_id,
            scenario_id=scenario_id,
            trace=trace,
            passed=trace.success,
            score=1.0 if trace.success else 0.0,
            timestamp=trace.timestamp,
        )
        results.append(result)

    return traces, results


# ===================================================================
# Demo Command
# ===================================================================


@click.command("demo")
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open report in browser automatically.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for reports (default: current directory).",
)
@click.option(
    "--trials",
    "-n",
    type=int,
    default=30,
    help="Number of trials per scenario (default: 30).",
)
def demo_command(no_browser: bool, output_dir: str, trials: int) -> None:
    """Run AgentAssay demo with 3 realistic scenarios.

    Generates synthetic agent traces, runs statistical analysis, computes
    coverage and fingerprints, and produces HTML + JSON reports.

    This is the fastest way to see AgentAssay in action without setting up
    a real agent or writing any configuration files.
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🔬 AgentAssay Demo — Token-Efficient Agent Testing[/bold cyan]\n"
            "[dim]Part of Qualixar | Author: Varun Pratap Bhardwaj[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # Step 1: Generate 3 demo scenarios
    # ──────────────────────────────────────────────────────────────────

    scenarios = [
        {
            "id": "ecommerce_search",
            "name": "E-commerce Search Agent",
            "pass_rate": 0.90,
            "tools": ["search", "filter", "rank", "recommend"],
        },
        {
            "id": "support_bot",
            "name": "Customer Support Bot",
            "pass_rate": 0.72,
            "tools": ["lookup_ticket", "search_kb", "classify_intent", "draft_response"],
        },
        {
            "id": "code_review",
            "name": "Code Review Agent",
            "pass_rate": 0.95,
            "tools": ["parse_ast", "check_style", "run_lint", "suggest_fix"],
        },
    ]

    console.print("[bold]Generating demo scenarios...[/bold]")

    all_scenarios_data: list[dict] = []

    for scenario in scenarios:
        console.print(f"  • {scenario['name']} ({trials} trials)")

        traces, trial_results = _generate_demo_scenario(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            pass_rate=scenario["pass_rate"],
            num_trials=trials,
            tool_names=scenario["tools"],
        )

        # ──────────────────────────────────────────────────────────────
        # Step 2: Compute stochastic verdict
        # ──────────────────────────────────────────────────────────────

        passed_results = [r.passed for r in trial_results]
        vf = VerdictFunction(alpha=0.05, beta=0.20, min_trials=10)
        verdict = vf.evaluate_single(passed_results, threshold=0.80)

        # ──────────────────────────────────────────────────────────────
        # Step 3: Compute coverage
        # ──────────────────────────────────────────────────────────────

        known_tools = set(scenario["tools"])
        known_models = {"gpt-4o"}
        collector = AgentCoverageCollector(
            known_tools=known_tools,
            known_models=known_models,
            boundaries={"latency_ms": (0, 5000)},
            max_path_depth=10,
        )

        for trace in traces:
            collector.update(trace)

        coverage = collector.snapshot()

        # ──────────────────────────────────────────────────────────────
        # Step 4: Compute behavioral fingerprints
        # ──────────────────────────────────────────────────────────────

        fingerprints = [BehavioralFingerprint.from_trace(t) for t in traces]

        # ──────────────────────────────────────────────────────────────
        # Step 5: Compute token savings estimate
        # ──────────────────────────────────────────────────────────────

        total_cost = sum(r.trace.total_cost_usd for r in trial_results)
        baseline_cost = total_cost * 5.0  # Assume AgentAssay saves 5x
        savings_pct = ((baseline_cost - total_cost) / baseline_cost) * 100.0

        all_scenarios_data.append(
            {
                "scenario_name": scenario["name"],
                "trial_results": trial_results,
                "verdict": verdict,
                "coverage": coverage,
                "fingerprints": fingerprints,
                "total_cost": total_cost,
                "baseline_cost": baseline_cost,
                "savings_pct": savings_pct,
            }
        )

    console.print()

    # ──────────────────────────────────────────────────────────────────
    # Step 6: Print summary table
    # ──────────────────────────────────────────────────────────────────

    console.print("[bold]Analysis Results:[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Scenario", style="white")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Verdict", justify="center")
    table.add_column("Coverage", justify="right")
    table.add_column("Savings", justify="right")

    for data in all_scenarios_data:
        verdict = data["verdict"]
        coverage = data["coverage"]

        # Verdict badge
        if verdict.status.value == "pass":
            verdict_badge = "[green]✓ PASS[/green]"
        elif verdict.status.value == "fail":
            verdict_badge = "[red]✗ FAIL[/red]"
        else:
            verdict_badge = "[yellow]? INCONCLUSIVE[/yellow]"

        # Coverage color
        cov_overall = coverage.overall * 100.0
        if cov_overall >= 80:
            cov_str = f"[green]{cov_overall:.1f}%[/green]"
        elif cov_overall >= 50:
            cov_str = f"[yellow]{cov_overall:.1f}%[/yellow]"
        else:
            cov_str = f"[red]{cov_overall:.1f}%[/red]"

        table.add_row(
            data["scenario_name"],
            f"{verdict.pass_rate * 100:.1f}%",
            verdict_badge,
            cov_str,
            f"[cyan]{data['savings_pct']:.1f}%[/cyan]",
        )

    console.print(table)
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # Step 7: Generate HTML report
    # ──────────────────────────────────────────────────────────────────

    console.print("[bold]Generating reports...[/bold]")

    output_path = Path(output_dir).resolve()
    html_path = output_path / "agentassay-demo-report.html"
    json_path = output_path / "agentassay-demo-report.json"

    # Generate multi-scenario HTML report (first scenario for main report)
    reporter = HTMLReporter()
    main_data = all_scenarios_data[0]
    reporter.save_report(
        data={
            "scenario_name": "AgentAssay Demo — 3 Scenarios",
            "trial_results": main_data["trial_results"],
            "verdict": main_data["verdict"],
            "coverage": main_data["coverage"],
        },
        path=html_path,
    )

    console.print(f"  • HTML report: [cyan]{html_path}[/cyan]")

    # ──────────────────────────────────────────────────────────────────
    # Step 8: Generate JSON report
    # ──────────────────────────────────────────────────────────────────

    # Combine all scenarios into one JSON report
    combined_json_data = {
        "scenario_name": "AgentAssay Demo — 3 Scenarios",
        "scenarios": [
            {
                "name": data["scenario_name"],
                "verdict": data["verdict"],
                "coverage": data["coverage"],
                "total_trials": len(data["trial_results"]),
                "passed_trials": sum(1 for r in data["trial_results"] if r.passed),
                "total_cost_usd": data["total_cost"],
                "savings_pct": data["savings_pct"],
            }
            for data in all_scenarios_data
        ],
        # Include full trial results from first scenario as example
        "trial_results": all_scenarios_data[0]["trial_results"],
    }

    JSONExporter.save(combined_json_data, json_path)
    console.print(f"  • JSON report: [cyan]{json_path}[/cyan]")
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # Step 9: Open in browser
    # ──────────────────────────────────────────────────────────────────

    if not no_browser:
        console.print("[dim]Opening report in browser...[/dim]")
        webbrowser.open(html_path.as_uri())

    # ──────────────────────────────────────────────────────────────────
    # Step 10: Print closing message
    # ──────────────────────────────────────────────────────────────────

    console.print()
    console.print(
        Panel.fit(
            "[bold green]Demo complete! Report generated.[/bold green]\n\n"
            "[bold]Next steps:[/bold]\n"
            "  • Install: [cyan]pip install agentassay[/cyan]\n"
            "  • Run your agent: [cyan]agentassay run --config your-agent.yaml[/cyan]\n"
            "  • Learn more: [cyan]https://qualixar.com[/cyan]\n"
            "  • Read the paper: [cyan]https://arxiv.org/abs/2603.02601[/cyan]",
            border_style="green",
        )
    )
    console.print()
