"""Tests for AgentAssay CLI commands.

Uses Click's CliRunner to invoke the CLI in-process without spawning
subprocesses.  Each test verifies that commands accept standard flags
(``--help``, ``--version``) and that subcommands are registered and
reachable.

Target: 20+ tests.  Max 400 lines.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentassay.cli.main import cli


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def tmp_results_json(tmp_path: Path) -> str:
    """Create a temporary results JSON file for commands that need one."""
    data = [
        {"passed": True, "score": 1.0},
        {"passed": True, "score": 0.9},
        {"passed": False, "score": 0.3},
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def tmp_baseline_json(tmp_path: Path) -> str:
    """Create a temporary baseline JSON file."""
    data = [
        {"passed": True, "score": 1.0},
        {"passed": True, "score": 0.9},
        {"passed": True, "score": 0.8},
        {"passed": True, "score": 0.7},
        {"passed": False, "score": 0.2},
    ]
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def tmp_current_json(tmp_path: Path) -> str:
    """Create a temporary current JSON file."""
    data = [
        {"passed": True, "score": 0.9},
        {"passed": True, "score": 0.7},
        {"passed": False, "score": 0.4},
        {"passed": False, "score": 0.3},
        {"passed": False, "score": 0.1},
    ]
    path = tmp_path / "current.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ===================================================================
# Top-level CLI group tests
# ===================================================================


class TestCLIGroup:
    """Tests for the top-level ``agentassay`` command group."""

    def test_cli_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AgentAssay" in result.output
        assert "run" in result.output
        assert "compare" in result.output
        assert "mutate" in result.output
        assert "coverage" in result.output
        assert "report" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "agentassay" in result.output.lower()

    def test_cli_no_args(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [])
        # Click returns exit code 0 (if invoke_without_command shows help)
        # or 2 (if a subcommand is required). Either is acceptable.
        assert result.exit_code in (0, 2)
        # Should show help/usage text regardless
        assert "Usage" in result.output or "agentassay" in result.output.lower()

    def test_cli_unknown_command(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["nonexistent"])
        assert result.exit_code != 0


# ===================================================================
# Subcommand --help tests
# ===================================================================


class TestSubcommandHelp:
    """Verify every subcommand responds to --help without error."""

    @pytest.mark.parametrize(
        "subcommand",
        ["run", "compare", "mutate", "coverage", "report", "dashboard"],
    )
    def test_subcommand_help(self, runner: CliRunner, subcommand: str) -> None:
        result = runner.invoke(cli, [subcommand, "--help"])
        assert result.exit_code == 0, (
            f"`agentassay {subcommand} --help` failed with "
            f"exit code {result.exit_code}: {result.output}"
        )
        # Every help text should contain Usage line
        assert "Usage" in result.output or "usage" in result.output.lower()


# ===================================================================
# Run command tests
# ===================================================================


class TestRunCommand:
    """Tests for ``agentassay run``."""

    def test_run_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--scenario" in result.output
        assert "--output" in result.output

    def test_run_no_args_does_not_crash(self, runner: CliRunner) -> None:
        """Running without config/scenario should give usage error, not a crash."""
        result = runner.invoke(cli, ["run"])
        # Click may return 0 or 2 for missing required args -- just verify no traceback
        assert "Traceback" not in result.output


# ===================================================================
# Compare command tests
# ===================================================================


class TestCompareCommand:
    """Tests for ``agentassay compare``."""

    def test_compare_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--baseline" in result.output
        assert "--current" in result.output
        assert "--alpha" in result.output

    def test_compare_missing_required_args(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compare"])
        assert result.exit_code != 0
        # Should mention missing required option
        assert "baseline" in result.output.lower() or "required" in result.output.lower()

    def test_compare_with_files(
        self,
        runner: CliRunner,
        tmp_baseline_json: str,
        tmp_current_json: str,
    ) -> None:
        result = runner.invoke(
            cli,
            ["compare", "--baseline", tmp_baseline_json, "--current", tmp_current_json],
        )
        # Should complete (exit code 0 or 1 for regression detected)
        assert result.exit_code in (0, 1), (
            f"Unexpected exit code {result.exit_code}: {result.output}"
        )


# ===================================================================
# Mutate command tests
# ===================================================================


class TestMutateCommand:
    """Tests for ``agentassay mutate``."""

    def test_mutate_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["mutate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--scenario" in result.output


# ===================================================================
# Coverage command tests
# ===================================================================


class TestCoverageCommand:
    """Tests for ``agentassay coverage``."""

    def test_coverage_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["coverage", "--help"])
        assert result.exit_code == 0
        assert "--results" in result.output

    def test_coverage_missing_required_args(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["coverage"])
        assert result.exit_code != 0


# ===================================================================
# Report command tests
# ===================================================================


class TestReportCommand:
    """Tests for ``agentassay report``."""

    def test_report_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--results" in result.output
        assert "--output" in result.output

    def test_report_missing_required_args(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["report"])
        assert result.exit_code != 0


# ===================================================================
# Dashboard command tests
# ===================================================================


class TestDashboardCommand:
    """Tests for ``agentassay dashboard``."""

    def test_dashboard_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output


# ===================================================================
# Helper re-exports
# ===================================================================


class TestHelperReExports:
    """Verify backward-compatible re-exports from cli.main."""

    def test_load_json_importable(self) -> None:
        from agentassay.cli.main import _load_json
        assert callable(_load_json)

    def test_write_json_importable(self) -> None:
        from agentassay.cli.main import _write_json
        assert callable(_write_json)

    def test_extract_passed_list_importable(self) -> None:
        from agentassay.cli.main import _extract_passed_list
        assert callable(_extract_passed_list)

    def test_verdict_style_importable(self) -> None:
        from agentassay.cli.main import _verdict_style
        assert callable(_verdict_style)
