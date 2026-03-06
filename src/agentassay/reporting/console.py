"""Rich-based console reporter for AgentAssay terminal output.

Renders stochastic test results, verdicts, coverage vectors, mutation
reports, gate decisions, and metamorphic reports as styled Rich output
suitable for terminal display and CI log files.

Uses Rich >= 13.0 components: Console, Table, Panel, Columns, and
manual progress-bar rendering via bar characters.

Design note: Every ``print_*`` method handles ``None`` / missing data
gracefully — a reporter should never crash the pipeline.
"""

from __future__ import annotations

from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentassay.core.models import TrialResult
from agentassay.coverage.aggregate import CoverageTuple
from agentassay.mutation.runner import MutationSuiteResult
from agentassay.metamorphic.runner import MetamorphicTestResult
from agentassay.verdicts.gate import GateDecision, GateReport
from agentassay.verdicts.verdict import StochasticVerdict, VerdictStatus


# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

_VERDICT_STYLES: dict[VerdictStatus, str] = {
    VerdictStatus.PASS: "bold green",
    VerdictStatus.FAIL: "bold red",
    VerdictStatus.INCONCLUSIVE: "bold yellow",
}

_GATE_STYLES: dict[GateDecision, str] = {
    GateDecision.DEPLOY: "bold green",
    GateDecision.BLOCK: "bold red",
    GateDecision.WARN: "bold yellow",
    GateDecision.SKIP: "bold dim",
}

_GATE_ICONS: dict[GateDecision, str] = {
    GateDecision.DEPLOY: "[green]DEPLOY[/green]",
    GateDecision.BLOCK: "[red]BLOCK[/red]",
    GateDecision.WARN: "[yellow]WARN[/yellow]",
    GateDecision.SKIP: "[dim]SKIP[/dim]",
}


# ---------------------------------------------------------------------------
# Helper: text progress bar
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """Render a value in [0, 1] as a text-based progress bar.

    Uses Unicode block characters for smooth rendering.
    Returns something like: ``[####............]  0.42``
    """
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    empty = width - filled

    if clamped >= 0.8:
        colour = "green"
    elif clamped >= 0.5:
        colour = "yellow"
    else:
        colour = "red"

    bar_str = f"[{colour}]{'█' * filled}{'░' * empty}[/{colour}]"
    return f"{bar_str}  {clamped:.1%}"


# ===================================================================
# ConsoleReporter
# ===================================================================


class ConsoleReporter:
    """Rich-based console reporter for AgentAssay terminal output.

    Produces styled tables, panels, and columns for every major
    report type in the AgentAssay pipeline.

    Parameters
    ----------
    verbose
        When ``True``, includes additional detail rows (per-trial
        breakdowns, per-operator mutation results, etc.).
    console
        Optional Rich ``Console`` instance. If ``None``, creates a
        new one writing to stdout.
    """

    __slots__ = ("_verbose", "_console")

    def __init__(
        self,
        verbose: bool = False,
        console: Console | None = None,
    ) -> None:
        self._verbose = verbose
        self._console = console or Console()

    # -- Trial summary -------------------------------------------------------

    def print_trial_summary(
        self,
        results: list[TrialResult],
        scenario_name: str,
    ) -> None:
        """Print a summary table for a batch of trial results.

        Displays pass/fail counts, pass rate, mean score, total cost,
        and total duration. In verbose mode, adds per-trial rows.

        Parameters
        ----------
        results
            List of ``TrialResult`` objects from running N trials.
        scenario_name
            Human-readable scenario name for the header.
        """
        if not results:
            self._console.print(
                Panel(
                    "[dim]No trial results to display.[/dim]",
                    title=f"Trial Summary: {scenario_name}",
                    border_style="dim",
                )
            )
            return

        n = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = n - passed
        pass_rate = passed / n if n > 0 else 0.0
        mean_score = sum(r.score for r in results) / n if n > 0 else 0.0
        total_cost = sum(r.trace.total_cost_usd for r in results)
        total_duration = sum(r.trace.total_duration_ms for r in results)

        # Summary table
        table = Table(
            title=f"Trial Summary: {scenario_name}",
            show_header=True,
            header_style="bold cyan",
            border_style="bright_blue",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        pass_style = "green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.5 else "red"

        table.add_row("Total Trials", str(n))
        table.add_row("Passed", f"[green]{passed}[/green]")
        table.add_row("Failed", f"[red]{failed}[/red]")
        table.add_row("Pass Rate", f"[{pass_style}]{pass_rate:.1%}[/{pass_style}]")
        table.add_row("Mean Score", f"{mean_score:.4f}")
        table.add_row("Total Cost", f"${total_cost:.4f}")
        table.add_row("Total Duration", f"{total_duration:.0f} ms")

        self._console.print(table)

        # Verbose: per-trial detail
        if self._verbose and results:
            detail = Table(
                title="Per-Trial Detail",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            detail.add_column("#", justify="right", style="dim")
            detail.add_column("Passed")
            detail.add_column("Score", justify="right")
            detail.add_column("Cost", justify="right")
            detail.add_column("Duration (ms)", justify="right")
            detail.add_column("Steps", justify="right")

            for i, r in enumerate(results, 1):
                status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
                detail.add_row(
                    str(i),
                    status,
                    f"{r.score:.4f}",
                    f"${r.trace.total_cost_usd:.4f}",
                    f"{r.trace.total_duration_ms:.0f}",
                    str(r.trace.step_count),
                )

            self._console.print(detail)

    # -- Verdict display -----------------------------------------------------

    def print_verdict(self, verdict: StochasticVerdict) -> None:
        """Print a stochastic verdict as a coloured panel.

        Shows status, pass rate with confidence interval, p-value,
        effect size, and the verdict reason from details.

        Parameters
        ----------
        verdict
            The ``StochasticVerdict`` to display.
        """
        if verdict is None:
            self._console.print("[dim]No verdict available.[/dim]")
            return

        style = _VERDICT_STYLES.get(verdict.status, "bold white")
        ci_lo, ci_hi = verdict.pass_rate_ci

        lines: list[str] = [
            f"[bold]Status:[/bold]       [{style}]{verdict.status.value.upper()}[/{style}]",
            f"[bold]Pass Rate:[/bold]    {verdict.pass_rate:.4f}  "
            f"[dim]CI: [{ci_lo:.4f}, {ci_hi:.4f}][/dim]",
            f"[bold]Trials:[/bold]       {verdict.num_passed}/{verdict.num_trials} passed",
            f"[bold]Confidence:[/bold]   {verdict.confidence:.0%}",
        ]

        if verdict.p_value is not None:
            lines.append(f"[bold]p-value:[/bold]     {verdict.p_value:.6f}")

        if verdict.effect_size is not None:
            interp = verdict.effect_size_interpretation or "unknown"
            lines.append(
                f"[bold]Effect Size:[/bold]  {verdict.effect_size:.4f} ({interp})"
            )

        if verdict.regression_detected:
            lines.append("[bold red]Regression Detected[/bold red]")

        reason = verdict.details.get("reason", "")
        if reason:
            lines.append(f"[dim]{reason}[/dim]")

        content = "\n".join(lines)

        panel_border = {
            VerdictStatus.PASS: "green",
            VerdictStatus.FAIL: "red",
            VerdictStatus.INCONCLUSIVE: "yellow",
        }.get(verdict.status, "white")

        self._console.print(
            Panel(
                content,
                title="Stochastic Verdict",
                border_style=panel_border,
                padding=(1, 2),
            )
        )

    # -- Regression comparison -----------------------------------------------

    def print_regression_comparison(
        self,
        baseline_rate: float,
        current_rate: float,
        regression_result: Any = None,
    ) -> None:
        """Print a side-by-side regression comparison.

        Shows baseline vs. current pass rates with bar visualizations,
        and the delta between them.

        Parameters
        ----------
        baseline_rate
            Baseline version pass rate in [0, 1].
        current_rate
            Current version pass rate in [0, 1].
        regression_result
            Optional object with ``p_value``, ``effect_size``,
            ``regression_detected`` attributes. If ``None``, only
            shows rates.
        """
        delta = current_rate - baseline_rate
        delta_style = "green" if delta >= 0 else "red"
        delta_str = f"{delta:+.4f}"

        baseline_panel = Panel(
            f"{_bar(baseline_rate)}\n\n"
            f"[bold]Pass Rate:[/bold] {baseline_rate:.4f}",
            title="[bold]Baseline[/bold]",
            border_style="blue",
        )

        current_panel = Panel(
            f"{_bar(current_rate)}\n\n"
            f"[bold]Pass Rate:[/bold] {current_rate:.4f}",
            title="[bold]Current[/bold]",
            border_style="cyan",
        )

        self._console.print(
            Panel(
                Columns([baseline_panel, current_panel], equal=True, expand=True),
                title="Regression Comparison",
                border_style="bright_blue",
            )
        )

        # Delta summary
        parts = [f"[bold]Delta:[/bold] [{delta_style}]{delta_str}[/{delta_style}]"]

        if regression_result is not None:
            p_val = getattr(regression_result, "p_value", None)
            eff = getattr(regression_result, "effect_size", None)
            detected = getattr(regression_result, "regression_detected", None)

            if p_val is not None:
                parts.append(f"[bold]p-value:[/bold] {p_val:.6f}")
            if eff is not None:
                parts.append(f"[bold]Effect:[/bold] {eff:.4f}")
            if detected is not None:
                if detected:
                    parts.append("[bold red]REGRESSION DETECTED[/bold red]")
                else:
                    parts.append("[green]No regression[/green]")

        self._console.print("  ".join(parts))

    # -- Coverage display ----------------------------------------------------

    def print_coverage(self, coverage: CoverageTuple) -> None:
        """Print the 5-dimensional coverage vector with progress bars.

        Shows each coverage dimension with a visual bar, the numeric
        value, and the overall geometric mean score.

        Parameters
        ----------
        coverage
            The ``CoverageTuple`` to display.
        """
        if coverage is None:
            self._console.print("[dim]No coverage data available.[/dim]")
            return

        table = Table(
            title="Agent Coverage Vector (5D)",
            show_header=True,
            header_style="bold cyan",
            border_style="bright_blue",
        )
        table.add_column("Dimension", style="bold", min_width=12)
        table.add_column("Score", justify="right", min_width=8)
        table.add_column("Bar", min_width=30)

        dimensions = [
            ("Tool", coverage.tool),
            ("Path", coverage.path),
            ("State", coverage.state),
            ("Boundary", coverage.boundary),
            ("Model", coverage.model),
        ]

        for name, value in dimensions:
            table.add_row(name, f"{value:.4f}", _bar(value))

        # Overall row with emphasis
        overall = coverage.overall
        table.add_section()
        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{overall:.4f}[/bold]",
            _bar(overall),
        )

        # Weakest dimension callout
        weakest_name, weakest_val = coverage.weakest
        table.add_section()
        table.add_row(
            "[dim]Weakest[/dim]",
            f"[dim]{weakest_val:.4f}[/dim]",
            f"[dim]{weakest_name}[/dim]",
        )

        self._console.print(table)

    # -- Mutation report -----------------------------------------------------

    def print_mutation_report(
        self,
        suite_result: MutationSuiteResult | None,
    ) -> None:
        """Print mutation testing results with per-category breakdown.

        Shows the overall mutation score, killed/survived/errored
        counts, and a per-category breakdown table.

        Parameters
        ----------
        suite_result
            The ``MutationSuiteResult`` to display. If ``None``,
            prints a placeholder message.
        """
        if suite_result is None:
            self._console.print("[dim]No mutation report available.[/dim]")
            return

        score = suite_result.mutation_score
        score_style = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"

        # Header panel
        header_lines = [
            f"[bold]Mutation Score:[/bold]  [{score_style}]{score:.1%}[/{score_style}]",
            f"[bold]Total Mutants:[/bold]   {suite_result.total_mutants}",
            f"[bold]Killed:[/bold]          [green]{suite_result.killed_mutants}[/green]",
            f"[bold]Survived:[/bold]        [red]{suite_result.survived_mutants}[/red]",
            f"[bold]Errored:[/bold]         [yellow]{suite_result.errored_mutants}[/yellow]",
            f"[bold]Duration:[/bold]        {suite_result.total_duration_ms:.0f} ms",
        ]

        self._console.print(
            Panel(
                "\n".join(header_lines),
                title="Mutation Testing Report",
                border_style=score_style,
            )
        )

        # Per-category breakdown
        if suite_result.per_category:
            cat_table = Table(
                title="Per-Category Breakdown",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            cat_table.add_column("Category", style="bold")
            cat_table.add_column("Score", justify="right")
            cat_table.add_column("Bar", min_width=25)

            for category, cat_score in sorted(suite_result.per_category.items()):
                cat_table.add_row(
                    category.title(),
                    f"{cat_score:.1%}",
                    _bar(cat_score),
                )

            self._console.print(cat_table)

        # Verbose: per-operator detail
        if self._verbose and suite_result.results:
            detail = Table(
                title="Per-Operator Detail",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            detail.add_column("Operator", style="bold")
            detail.add_column("Category")
            detail.add_column("Killed")
            detail.add_column("Original", justify="right")
            detail.add_column("Mutant", justify="right")
            detail.add_column("Delta", justify="right")

            for r in suite_result.results:
                killed_str = (
                    "[green]KILLED[/green]" if r.killed
                    else "[red]SURVIVED[/red]"
                )
                if r.error:
                    killed_str = "[yellow]ERROR[/yellow]"

                delta_style = "green" if r.score_delta <= 0 else "red"
                detail.add_row(
                    r.operator_name,
                    r.operator_category,
                    killed_str,
                    f"{r.original_score:.2f}",
                    f"{r.mutant_score:.2f}",
                    f"[{delta_style}]{r.score_delta:+.3f}[/{delta_style}]",
                )

            self._console.print(detail)

    # -- Gate decision -------------------------------------------------------

    def print_gate_decision(self, gate_report: GateReport | None) -> None:
        """Print the deployment gate decision with per-scenario breakdown.

        Shows the overall decision with a colour-coded indicator, the
        exit code, and a per-scenario table with individual decisions
        and reasons.

        Parameters
        ----------
        gate_report
            The ``GateReport`` to display. If ``None``, prints a
            placeholder message.
        """
        if gate_report is None:
            self._console.print("[dim]No gate report available.[/dim]")
            return

        decision = gate_report.overall_decision
        style = _GATE_STYLES.get(decision, "bold white")
        icon = _GATE_ICONS.get(decision, str(decision))

        header_lines = [
            f"[bold]Decision:[/bold]      {icon}",
            f"[bold]Exit Code:[/bold]     {gate_report.exit_code}",
            f"[bold]Scenarios:[/bold]     {gate_report.total_scenarios} total",
            f"  [green]Deployed:[/green]    {gate_report.passed_scenarios}",
            f"  [red]Blocked:[/red]     {gate_report.blocked_scenarios}",
            f"  [yellow]Warned:[/yellow]      {gate_report.warned_scenarios}",
            f"  [dim]Skipped:[/dim]     {gate_report.skipped_scenarios}",
        ]

        self._console.print(
            Panel(
                "\n".join(header_lines),
                title="Deployment Gate Decision",
                border_style=style.replace("bold ", ""),
            )
        )

        # Per-scenario table
        if gate_report.scenario_decisions:
            scen_table = Table(
                title="Per-Scenario Decisions",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            scen_table.add_column("Scenario", style="bold")
            scen_table.add_column("Decision")
            scen_table.add_column("Reason", max_width=60)

            for name, scen_decision in gate_report.scenario_decisions.items():
                reason = gate_report.scenario_reasons.get(name, "")
                scen_icon = _GATE_ICONS.get(scen_decision, str(scen_decision))
                scen_table.add_row(name, scen_icon, reason)

            self._console.print(scen_table)

    # -- Metamorphic report --------------------------------------------------

    def print_metamorphic_report(
        self,
        mr_result: MetamorphicTestResult | None,
    ) -> None:
        """Print metamorphic testing results with per-family breakdown.

        Shows the overall violation rate, per-family violation rates,
        and in verbose mode, per-relation detail.

        Parameters
        ----------
        mr_result
            The ``MetamorphicTestResult`` to display. If ``None``,
            prints a placeholder message.
        """
        if mr_result is None:
            self._console.print("[dim]No metamorphic report available.[/dim]")
            return

        vr = mr_result.violation_rate
        vr_style = "green" if vr <= 0.1 else "yellow" if vr <= 0.3 else "red"

        header_lines = [
            f"[bold]Violation Rate:[/bold]    [{vr_style}]{vr:.1%}[/{vr_style}]",
            f"[bold]Total Relations:[/bold]   {mr_result.total_relations}",
            f"[bold]Violations:[/bold]        [red]{mr_result.violations}[/red]",
            f"[bold]Held:[/bold]              [green]{mr_result.total_relations - mr_result.violations}[/green]",
        ]

        self._console.print(
            Panel(
                "\n".join(header_lines),
                title="Metamorphic Testing Report",
                border_style=vr_style,
            )
        )

        # Per-family breakdown
        if mr_result.per_family:
            fam_table = Table(
                title="Per-Family Breakdown",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            fam_table.add_column("Family", style="bold")
            fam_table.add_column("Tested", justify="right")
            fam_table.add_column("Violations", justify="right")
            fam_table.add_column("Rate", justify="right")

            for family, info in sorted(mr_result.per_family.items()):
                tested = info.get("tested", 0)
                violations = info.get("violations", 0)
                rate = info.get("rate", 0.0)
                rate_style = (
                    "green" if rate <= 0.1
                    else "yellow" if rate <= 0.3
                    else "red"
                )
                fam_table.add_row(
                    family.title(),
                    str(tested),
                    str(violations),
                    f"[{rate_style}]{rate:.1%}[/{rate_style}]",
                )

            self._console.print(fam_table)

        # Verbose: per-relation detail
        if self._verbose and mr_result.results:
            detail = Table(
                title="Per-Relation Detail",
                show_header=True,
                header_style="bold",
                border_style="dim",
            )
            detail.add_column("Relation", style="bold")
            detail.add_column("Family")
            detail.add_column("Holds")
            detail.add_column("Similarity", justify="right")

            for r in mr_result.results:
                holds_str = (
                    "[green]HOLDS[/green]" if r.holds
                    else "[red]VIOLATED[/red]"
                )
                detail.add_row(
                    r.relation_name,
                    r.relation_family,
                    holds_str,
                    f"{r.similarity_score:.4f}",
                )

            self._console.print(detail)

    # -- Utility -------------------------------------------------------------

    def print_separator(self, char: str = "─", width: int = 60) -> None:
        """Print a horizontal separator line.

        Parameters
        ----------
        char
            Character to repeat.
        width
            Total width of the separator.
        """
        self._console.print(f"[dim]{char * width}[/dim]")

    def print_header(self, text: str) -> None:
        """Print a section header.

        Parameters
        ----------
        text
            Header text.
        """
        self._console.print(f"\n[bold bright_white]{text}[/bold bright_white]")
        self.print_separator()
