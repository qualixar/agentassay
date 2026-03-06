"""CLI command: ``agentassay report`` -- HTML report generation.

Generates a self-contained HTML report from trial results JSON with
pass rates, confidence intervals, and visual verdict display.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

import agentassay
from agentassay.cli.helpers import (
    console,
    extract_passed_list,
    load_json,
)
from agentassay.statistics.confidence import wilson_interval


@click.command("report")
@click.option(
    "--results", "-r",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with trial results.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="agentassay-report.html",
    show_default=True,
    help="Output HTML file path.",
)
def report_command(results: str, output: str) -> None:
    """Generate an HTML report from trial results.

    Creates a self-contained HTML report with pass rates, confidence
    intervals, and test verdicts. The report can be opened in any
    browser and is suitable for archiving and sharing.

    \b
    Examples:
        agentassay report --results trials.json
        agentassay report -r results.json -o my-report.html
    """
    console.print(Panel.fit(
        "[bold]AgentAssay[/bold] -- HTML Report Generator",
        border_style="blue",
    ))

    # Load results
    results_data = load_json(results, "results")

    # Extract pass/fail
    try:
        passed_list = extract_passed_list(results_data)
    except click.ClickException:
        passed_list = []

    n = len(passed_list)
    k = sum(passed_list)
    rate = k / n if n > 0 else 0.0

    # Compute CI if we have data
    ci_lower = 0.0
    ci_upper = 1.0
    if n > 0:
        ci = wilson_interval(k, n)
        ci_lower = ci.lower
        ci_upper = ci.upper

    # Generate HTML report
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = _build_html(n, k, rate, ci_lower, ci_upper, timestamp)

    # Write HTML
    out_path = Path(output)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        console.print(f"[green]Report written to {output}[/green]")
    except OSError as exc:
        raise click.ClickException(f"Cannot write report to {output}: {exc}") from exc

    # Print summary
    summary = Table(title="Report Summary", show_header=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Trials", str(n))
    summary.add_row("Pass Rate", f"{rate:.2%}")
    summary.add_row("95% CI", f"[{ci_lower:.4f}, {ci_upper:.4f}]")
    summary.add_row("Output", output)
    console.print(summary)


def _build_html(
    n: int,
    k: int,
    rate: float,
    ci_lower: float,
    ci_upper: float,
    timestamp: str,
) -> str:
    """Build the self-contained HTML report string."""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentAssay Report</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #e6edf3;
            --text-dim: #8b949e;
            --green: #3fb950;
            --red: #f85149;
            --yellow: #d29922;
            --blue: #58a6ff;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.3rem; margin: 1.5rem 0 0.75rem; color: var(--blue); }}
        .subtitle {{ color: var(--text-dim); margin-bottom: 2rem; }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        .verdict {{
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
        }}
        .verdict.pass {{ color: var(--green); }}
        .verdict.fail {{ color: var(--red); }}
        .verdict.inconclusive {{ color: var(--yellow); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0.5rem 0;
        }}
        th, td {{
            padding: 0.6rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-dim); font-weight: 600; }}
        .bar-container {{
            width: 100%;
            height: 24px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: var(--green);
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .bar-fill.warn {{ background: var(--yellow); }}
        .bar-fill.fail {{ background: var(--red); }}
        footer {{
            margin-top: 3rem;
            text-align: center;
            color: var(--text-dim);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <h1>AgentAssay Report</h1>
    <p class="subtitle">Generated: {timestamp}</p>

    <div class="card">
        <div class="verdict {'pass' if rate >= 0.80 else 'fail' if rate < 0.50 else 'inconclusive'}">
            {'PASS' if n > 0 and ci_lower >= 0.80 else 'FAIL' if n > 0 and ci_upper < 0.80 else 'INCONCLUSIVE' if n > 0 else 'NO DATA'}
        </div>
    </div>

    <h2>Summary</h2>
    <div class="card">
        <table>
            <tr><th>Total Trials</th><td>{n}</td></tr>
            <tr><th>Passed</th><td>{k}</td></tr>
            <tr><th>Failed</th><td>{n - k}</td></tr>
            <tr><th>Pass Rate</th><td>{rate:.2%}</td></tr>
            <tr><th>95% CI (Wilson)</th><td>[{ci_lower:.4f}, {ci_upper:.4f}]</td></tr>
        </table>
    </div>

    <h2>Pass Rate</h2>
    <div class="card">
        <div class="bar-container">
            <div class="bar-fill {'warn' if rate < 0.80 else ''} {'fail' if rate < 0.50 else ''}"
                 style="width: {rate * 100:.1f}%"></div>
        </div>
        <p style="text-align: center; margin-top: 0.5rem; color: var(--text-dim);">
            {rate:.1%} ({k}/{n})
        </p>
    </div>

    <h2>Methodology</h2>
    <div class="card">
        <table>
            <tr><th>Framework</th><td>AgentAssay v{agentassay.__version__}</td></tr>
            <tr><th>Confidence Interval</th><td>Wilson score (1927)</td></tr>
            <tr><th>Regression Test</th><td>Fisher's exact test</td></tr>
            <tr><th>Verdict Semantics</th><td>Stochastic: (alpha, beta, n)-triple</td></tr>
        </table>
    </div>

    <footer>
        <p>Generated by AgentAssay v{agentassay.__version__}</p>
        <p>Formal regression testing for non-deterministic AI agent workflows</p>
    </footer>
</body>
</html>
"""
