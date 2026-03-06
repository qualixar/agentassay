# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Shared CLI helpers for AgentAssay commands.

Provides JSON/YAML file I/O, result parsing, and Rich formatting
utilities used across all CLI subcommands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

console = Console()
error_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_json(path: str, label: str) -> dict[str, Any] | list[Any]:
    """Load and parse a JSON file with user-friendly error handling.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    label : str
        Human-readable label for error messages (e.g. "baseline results").

    Returns
    -------
    dict or list
        Parsed JSON content.

    Raises
    ------
    click.ClickException
        If the file is missing, unreadable, or contains invalid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"{label} file not found: {path}")
    if not p.is_file():
        raise click.ClickException(f"{label} path is not a file: {path}")
    try:
        text = p.read_text(encoding="utf-8")
        return json.loads(text)
    except json.JSONDecodeError:
        raise click.ClickException(f"Invalid JSON syntax in {label} file")
    except OSError:
        raise click.ClickException(f"Cannot read {label} file. Check permissions.")


def write_json(data: Any, path: str, label: str) -> None:
    """Write data as formatted JSON with error handling.

    Parameters
    ----------
    data : Any
        JSON-serializable data.
    path : str
        Output file path.
    label : str
        Human-readable label for error messages.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]Wrote {label} to {path}[/green]")
    except OSError:
        raise click.ClickException(f"Cannot write {label}. Check permissions.")


def load_yaml(path: str, label: str) -> dict[str, Any]:
    """Load and parse a YAML file with user-friendly error handling.

    Parameters
    ----------
    path : str
        Path to the YAML file.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    try:
        import yaml
    except ImportError:
        raise click.ClickException(
            "PyYAML is required for YAML config files. Install with: pip install pyyaml"
        )

    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"{label} file not found: {path}")

    try:
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise click.ClickException(
                f"{label} file must contain a YAML mapping, got {type(data).__name__}"
            )
        return data
    except yaml.YAMLError:
        raise click.ClickException(f"Invalid YAML syntax in {label} file")
    except OSError:
        raise click.ClickException(f"Cannot read {label} file. Check permissions.")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def extract_passed_list(results_data: dict[str, Any] | list[Any]) -> list[bool]:
    """Extract a list of boolean pass/fail values from results JSON.

    Supports two formats:
    1. List of objects with a ``passed`` field: ``[{"passed": true}, ...]``
    2. Dictionary with a ``results`` key containing the list.

    Parameters
    ----------
    results_data : dict or list
        Parsed JSON results.

    Returns
    -------
    list[bool]
        Boolean pass/fail values.
    """
    if isinstance(results_data, list):
        items = results_data
    elif isinstance(results_data, dict) and "results" in results_data:
        items = results_data["results"]
    elif isinstance(results_data, dict) and "trials" in results_data:
        items = results_data["trials"]
    else:
        raise click.ClickException(
            "Results JSON must be a list of trial objects or a dict "
            "with a 'results' or 'trials' key."
        )

    passed: list[bool] = []
    for i, item in enumerate(items):
        if isinstance(item, bool):
            passed.append(item)
        elif isinstance(item, dict):
            if "passed" in item:
                passed.append(bool(item["passed"]))
            elif "success" in item:
                passed.append(bool(item["success"]))
            else:
                raise click.ClickException(f"Trial {i} has no 'passed' or 'success' field: {item}")
        else:
            raise click.ClickException(f"Trial {i} is not a dict or bool: {type(item).__name__}")

    return passed


# ---------------------------------------------------------------------------
# Rich formatting
# ---------------------------------------------------------------------------


def verdict_style(status: str) -> str:
    """Map verdict status to Rich color tag."""
    status_upper = status.upper()
    if status_upper == "PASS":
        return "bold green"
    elif status_upper == "FAIL":
        return "bold red"
    elif status_upper == "INCONCLUSIVE":
        return "bold yellow"
    return "white"
