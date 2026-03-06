# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""``agentassay dashboard`` — launch the interactive dashboard.

Opens a browser-based dashboard for visualizing test results, historical
trends, behavioral fingerprints, and deployment gate decisions.

The dashboard reads from the local results database and provides
real-time monitoring of running test suites.
"""

from __future__ import annotations

import sys

import click

from agentassay.cli.helpers import console, error_console


@click.command("dashboard")
@click.option(
    "--port",
    "-p",
    type=int,
    default=8501,
    show_default=True,
    help="Port to run the dashboard on.",
)
@click.option(
    "--host",
    type=str,
    default="localhost",
    show_default=True,
    help="Host to bind the dashboard to.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't auto-open the browser.",
)
@click.option(
    "--theme",
    type=click.Choice(["dark", "light"], case_sensitive=False),
    default="dark",
    show_default=True,
    help="Color theme for the dashboard.",
)
def dashboard_command(
    port: int,
    host: str,
    no_browser: bool,
    theme: str,
) -> None:
    """Launch the interactive AgentAssay dashboard [EXPERIMENTAL].

    Opens a browser-based dashboard for visualizing test results,
    trends, behavioral fingerprints, and gate decisions.

    WARNING: This feature is experimental and may change without notice.
    Not recommended for production use.

    Requires: pip install agentassay[dashboard]
    """
    # Security: Validate host parameter
    import re
    import ipaddress

    if host not in ("localhost", "127.0.0.1", "0.0.0.0"):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if not re.match(r'^[a-zA-Z0-9.-]+$', host):
                raise click.ClickException(f"Invalid host: {host}")

    # Security: Validate port range
    if not (1024 <= port <= 65535):
        raise click.ClickException(f"Port must be between 1024 and 65535, got {port}")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        error_console.print(
            "[bold red]Dashboard requires streamlit.[/bold red]\n"
            "Install with: [bold]pip install agentassay[dashboard][/bold]"
        )
        sys.exit(1)

    import subprocess

    # Locate the dashboard app entry point
    from importlib.resources import files

    try:
        dashboard_pkg = files("agentassay.dashboard")
        app_path = str(dashboard_pkg.joinpath("app.py"))
    except (ModuleNotFoundError, TypeError):
        # Fallback: construct path relative to this file
        from pathlib import Path

        app_path = str(
            Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
        )

    console.print(
        f"[bold green]Starting AgentAssay Dashboard[/bold green] "
        f"on [bold]{host}:{port}[/bold]"
    )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--theme.base",
        theme,
    ]

    if no_browser:
        cmd.extend(["--server.headless", "true"])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[bold]Dashboard stopped.[/bold]")
    except subprocess.CalledProcessError as exc:
        error_console.print(f"[bold red]Dashboard exited with code {exc.returncode}[/bold red]")
        sys.exit(exc.returncode)
