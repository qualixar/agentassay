"""AgentAssay CLI -- command-line interface for agent regression testing.

Provides the ``agentassay`` command with subcommands for running assays,
comparing results, mutation testing, coverage reporting, and HTML report
generation.

Entry point registered in pyproject.toml::

    [project.scripts]
    agentassay = "agentassay.cli.main:cli"

Commands:
    agentassay run       Run agent assay trials against scenarios.
    agentassay compare   Compare baseline vs. current results for regression.
    agentassay mutate    Run mutation testing suite.
    agentassay coverage  Compute and display coverage metrics.
    agentassay report    Generate an HTML report from results.

All commands use Rich for formatted terminal output and Click for
argument parsing. Each command is implemented in its own module under
``agentassay.cli.cmd_*``.

Backward compatibility: The helper functions ``_load_json``,
``_write_json``, ``_load_yaml``, ``_extract_passed_list``, and
``_verdict_style`` are re-exported here so that any code importing
them from ``agentassay.cli.main`` continues to work.
"""

from __future__ import annotations

import click

import agentassay
from agentassay.cli.cmd_compare import compare_command
from agentassay.cli.cmd_coverage import coverage_command
from agentassay.cli.cmd_dashboard import dashboard_command
from agentassay.cli.cmd_mutate import mutate_command
from agentassay.cli.cmd_report import report_command
from agentassay.cli.cmd_run import run_command

# Re-export helpers for backward compatibility
from agentassay.cli.helpers import (  # noqa: F401
    console,
    error_console,
    extract_passed_list as _extract_passed_list,
    load_json as _load_json,
    load_yaml as _load_yaml,
    verdict_style as _verdict_style,
    write_json as _write_json,
)


# ===================================================================
# CLI Group
# ===================================================================


@click.group()
@click.version_option(version=agentassay.__version__, prog_name="agentassay")
def cli() -> None:
    """AgentAssay: Formal regression testing for AI agents.

    Run stochastic trials, detect regressions with statistical rigor,
    and gate deployments with confidence intervals.
    """
    pass


# Register subcommands
cli.add_command(run_command, "run")
cli.add_command(compare_command, "compare")
cli.add_command(mutate_command, "mutate")
cli.add_command(coverage_command, "coverage")
cli.add_command(report_command, "report")
cli.add_command(dashboard_command, "dashboard")


# ===================================================================
# Entry point guard
# ===================================================================


if __name__ == "__main__":
    cli()
