# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Test report command — run pytest and generate HTML report.

Executes the full AgentAssay test suite programmatically and produces
a beautiful HTML report showing test results with pass rates, coverage
breakdowns, and Qualixar attribution.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console

console = Console()


def _generate_test_report_html(
    total: int,
    passed: int,
    failed: int,
    duration: float,
    module_breakdown: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    """Generate a self-contained HTML test report.

    Parameters
    ----------
    total
        Total number of tests.
    passed
        Number of tests passed.
    failed
        Number of tests failed.
    duration
        Total test duration in seconds.
    module_breakdown
        Dict mapping module names to {"passed": int, "failed": int}.
    output_path
        Path to write the HTML report.
    """
    pass_rate = (passed / total * 100.0) if total > 0 else 0.0

    # Compute Wilson confidence interval for pass rate
    from agentassay.statistics.confidence import wilson_interval

    ci = wilson_interval(successes=passed, n=total, confidence=0.95)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentAssay Test Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    line-height: 1.6;
    padding: 2rem;
    max-width: 1100px;
    margin: 0 auto;
  }}
  a {{ color: #58a6ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  .report-header {{
    border-bottom: 1px solid #30363d;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
  }}
  .report-header h1 {{
    font-size: 1.6rem;
    color: #f0f6fc;
    margin-bottom: 0.25rem;
  }}
  .report-header .meta {{
    font-size: 0.85rem;
    color: #8b949e;
  }}

  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .card h2 {{
    font-size: 1.1rem;
    color: #f0f6fc;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
  }}

  .kpi-row {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }}
  .kpi {{
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 0.75rem;
    background: #0d1117;
    border-radius: 6px;
    border: 1px solid #21262d;
  }}
  .kpi .label {{ font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }}
  .kpi .value {{ font-size: 1.4rem; font-weight: 700; color: #f0f6fc; }}

  .bar-container {{
    background: #21262d;
    border-radius: 4px;
    height: 24px;
    position: relative;
    overflow: hidden;
    margin: 0.25rem 0;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }}
  .bar-label {{
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.7rem;
    font-weight: 600;
    color: #f0f6fc;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }}
  .fill-green {{ background: linear-gradient(90deg, #238636, #2ea043); }}
  .fill-yellow {{ background: linear-gradient(90deg, #9e6a03, #bb8009); }}
  .fill-red {{ background: linear-gradient(90deg, #da3633, #f85149); }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid #30363d;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
  }}
  td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #21262d;
  }}
  tr:hover td {{ background: rgba(56, 139, 253, 0.05); }}

  .footer {{
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #30363d;
    text-align: center;
    font-size: 0.75rem;
    color: #484f58;
  }}
  .qualixar-attribution {{
    margin-top: 1rem;
    padding-top: 0.75rem;
    border-top: 1px solid #21262d;
    font-size: 0.7rem;
    color: #6e7681;
  }}
  .qualixar-attribution a {{
    color: #58a6ff;
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>AgentAssay Test Report</h1>
  <div class="meta">
    Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
  </div>
</div>

<div class="card">
  <h2>Test Summary</h2>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Total Tests</div>
      <div class="value">{total}</div>
    </div>
    <div class="kpi">
      <div class="label">Passed</div>
      <div class="value" style="color: #3fb950;">{passed}</div>
    </div>
    <div class="kpi">
      <div class="label">Failed</div>
      <div class="value" style="color: #f85149;">{failed}</div>
    </div>
    <div class="kpi">
      <div class="label">Pass Rate</div>
      <div class="value">{pass_rate:.1f}%</div>
    </div>
    <div class="kpi">
      <div class="label">Duration</div>
      <div class="value">{duration:.1f}s</div>
    </div>
  </div>

  <div class="bar-container">
    <div class="bar-fill {'fill-green' if pass_rate >= 80 else 'fill-yellow' if pass_rate >= 50 else 'fill-red'}"
         style="width: {pass_rate}%;"></div>
    <div class="bar-label">{pass_rate:.1f}%</div>
  </div>

  <div style="margin-top: 0.75rem; font-size: 0.8rem; color: #8b949e;">
    Wilson 95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]
  </div>
</div>

<div class="card">
  <h2>Module Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>Module</th>
        <th style="text-align: right;">Passed</th>
        <th style="text-align: right;">Failed</th>
        <th style="text-align: right;">Total</th>
        <th style="text-align: right;">Pass Rate</th>
      </tr>
    </thead>
    <tbody>
"""

    for module_name in sorted(module_breakdown.keys()):
        stats = module_breakdown[module_name]
        mod_passed = stats["passed"]
        mod_failed = stats["failed"]
        mod_total = mod_passed + mod_failed
        mod_rate = (mod_passed / mod_total * 100.0) if mod_total > 0 else 0.0

        color = "#3fb950" if mod_rate >= 80 else "#bb8009" if mod_rate >= 50 else "#f85149"

        html += f"""      <tr>
        <td>{module_name}</td>
        <td style="text-align: right; color: #3fb950;">{mod_passed}</td>
        <td style="text-align: right; color: #f85149;">{mod_failed}</td>
        <td style="text-align: right;">{mod_total}</td>
        <td style="text-align: right; color: {color};">{mod_rate:.1f}%</td>
      </tr>
"""

    html += """    </tbody>
  </table>
</div>

<div class="footer">
  Generated by AgentAssay
  <div class="qualixar-attribution">
    <p>Part of Qualixar | Author: Varun Pratap Bhardwaj</p>
    <p><a href="https://qualixar.com" target="_blank">qualixar.com</a> | <a href="https://varunpratap.com" target="_blank">varunpratap.com</a></p>
    <p>License: Apache-2.0</p>
  </div>
</div>

</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


@click.command("test-report")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="agentassay-test-report.html",
    help="Output path for HTML report.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open report in browser automatically.",
)
def test_report_command(output: str, no_browser: bool) -> None:
    """Run AgentAssay test suite and generate HTML report.

    Executes pytest programmatically, parses the results, and produces
    a beautiful HTML report with pass rates, module breakdown, and
    Wilson confidence intervals.
    """
    console.print()
    console.print("[bold]Running AgentAssay test suite...[/bold]")
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # Run pytest with JUnit XML output for structured parsing
    # ──────────────────────────────────────────────────────────────────

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tf:
        junit_xml_path = Path(tf.name)

    try:
        # Run pytest
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-q",
                "--tb=short",
                "--no-header",
                f"--junit-xml={junit_xml_path}",
            ],
            capture_output=True,
            text=True,
        )

        # Show pytest output
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr, style="dim")

        # ──────────────────────────────────────────────────────────────
        # Parse JUnit XML
        # ──────────────────────────────────────────────────────────────

        import xml.etree.ElementTree as ET

        tree = ET.parse(junit_xml_path)
        root = tree.getroot()

        # Extract summary stats
        testsuite = root.find("testsuite")
        if testsuite is None:
            testsuite = root  # Sometimes the root is the testsuite

        total = int(testsuite.get("tests", "0"))
        failures = int(testsuite.get("failures", "0"))
        errors = int(testsuite.get("errors", "0"))
        duration = float(testsuite.get("time", "0.0"))

        failed = failures + errors
        passed = total - failed

        # Module breakdown
        module_breakdown: dict[str, dict[str, int]] = {}

        for testcase in testsuite.findall(".//testcase"):
            classname = testcase.get("classname", "unknown")
            # Extract module name from classname (e.g., "tests.core.test_models" -> "core")
            parts = classname.split(".")
            if len(parts) >= 2 and parts[0] == "tests":
                module_name = parts[1]
            else:
                module_name = "other"

            if module_name not in module_breakdown:
                module_breakdown[module_name] = {"passed": 0, "failed": 0}

            # Check if test failed
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                module_breakdown[module_name]["failed"] += 1
            else:
                module_breakdown[module_name]["passed"] += 1

        # ──────────────────────────────────────────────────────────────
        # Generate HTML report
        # ──────────────────────────────────────────────────────────────

        console.print()
        console.print("[bold]Generating HTML report...[/bold]")

        output_path = Path(output).resolve()
        _generate_test_report_html(
            total=total,
            passed=passed,
            failed=failed,
            duration=duration,
            module_breakdown=module_breakdown,
            output_path=output_path,
        )

        console.print(f"  • Report saved: [cyan]{output_path}[/cyan]")
        console.print()

        # Summary
        if failed == 0:
            console.print(f"[green]✓ All {total} tests passed![/green]")
        else:
            console.print(f"[yellow]⚠ {passed}/{total} tests passed, {failed} failed[/yellow]")

        # ──────────────────────────────────────────────────────────────
        # Open in browser
        # ──────────────────────────────────────────────────────────────

        if not no_browser:
            console.print()
            console.print("[dim]Opening report in browser...[/dim]")
            webbrowser.open(output_path.as_uri())

    finally:
        # Clean up temp XML file
        if junit_xml_path.exists():
            junit_xml_path.unlink()

    console.print()
