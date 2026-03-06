"""Tests for AgentAssay reporting module.

Covers ConsoleReporter, HTMLReporter, and JSONExporter with realistic
test data from the conftest factories.

Target: 15+ tests.  Max 300 lines.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from agentassay.core.models import TrialResult
from agentassay.reporting.console import ConsoleReporter
from agentassay.reporting.html import HTMLReporter
from agentassay.reporting.json_export import JSONExporter

# Import helpers from conftest (available via pytest path)
from tests.conftest import make_trace, make_scenario


# ===================================================================
# Helpers
# ===================================================================


def _make_trial_results(n: int = 10, pass_rate: float = 0.8) -> list[TrialResult]:
    """Build a list of mock TrialResult objects."""
    from datetime import datetime, timezone

    results: list[TrialResult] = []
    for i in range(n):
        passed = i < int(n * pass_rate)
        trace = make_trace(
            steps=2,
            success=passed,
            cost=0.01,
        )
        results.append(
            TrialResult(
                trial_id=f"trial-{i}",
                scenario_id="scenario-1",
                trace=trace,
                passed=passed,
                score=1.0 if passed else 0.0,
                evaluation_details={},
                timestamp=datetime.now(timezone.utc),
            )
        )
    return results


# ===================================================================
# ConsoleReporter tests
# ===================================================================


class TestConsoleReporter:
    """Tests for the Rich-based console reporter."""

    def test_create_reporter(self) -> None:
        reporter = ConsoleReporter()
        assert reporter is not None

    def test_create_verbose_reporter(self) -> None:
        reporter = ConsoleReporter(verbose=True)
        assert reporter is not None

    def test_print_trial_summary_produces_output(self) -> None:
        buf = StringIO()
        console = Console(file=buf, width=120)
        reporter = ConsoleReporter(verbose=False, console=console)
        results = _make_trial_results(5)
        reporter.print_trial_summary(results, "test-scenario")
        output = buf.getvalue()
        assert len(output) > 0
        assert "test-scenario" in output

    def test_print_trial_summary_empty_results(self) -> None:
        buf = StringIO()
        console = Console(file=buf, width=120)
        reporter = ConsoleReporter(console=console)
        reporter.print_trial_summary([], "empty-scenario")
        output = buf.getvalue()
        assert "No trial results" in output

    def test_print_header(self) -> None:
        buf = StringIO()
        console = Console(file=buf, width=120)
        reporter = ConsoleReporter(console=console)
        reporter.print_header("Test Header")
        output = buf.getvalue()
        assert "Test Header" in output

    def test_print_separator(self) -> None:
        buf = StringIO()
        console = Console(file=buf, width=120)
        reporter = ConsoleReporter(console=console)
        reporter.print_separator()
        output = buf.getvalue()
        assert len(output.strip()) > 0


# ===================================================================
# HTMLReporter tests
# ===================================================================


class TestHTMLReporter:
    """Tests for the self-contained HTML reporter."""

    def test_create_reporter(self) -> None:
        reporter = HTMLReporter()
        assert reporter is not None

    def test_generate_empty_report(self) -> None:
        reporter = HTMLReporter()
        html = reporter.generate_report({"scenario_name": "test"})
        assert "<!DOCTYPE html>" in html
        assert "AgentAssay Report" in html
        assert "test" in html

    def test_generate_report_with_trials(self) -> None:
        reporter = HTMLReporter()
        results = _make_trial_results(3)
        html = reporter.generate_report({
            "scenario_name": "trial-test",
            "trial_results": results,
        })
        assert "<!DOCTYPE html>" in html
        assert "trial-test" in html
        # Should contain trial results section
        assert "Trial Results" in html

    def test_save_report(self, tmp_path) -> None:
        reporter = HTMLReporter()
        path = tmp_path / "report.html"
        reporter.save_report({"scenario_name": "saved"}, str(path))
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "saved" in content

    def test_html_contains_valid_structure(self) -> None:
        reporter = HTMLReporter()
        html = reporter.generate_report({"scenario_name": "structure"})
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</body>" in html


# ===================================================================
# JSONExporter tests
# ===================================================================


class TestJSONExporter:
    """Tests for the JSON exporter."""

    def test_export_empty_results(self) -> None:
        json_str = JSONExporter.export_results([])
        assert json_str == "[]"

    def test_export_results_produces_valid_json(self) -> None:
        results = _make_trial_results(3)
        json_str = JSONExporter.export_results(results)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_export_full_report_produces_valid_json(self) -> None:
        results = _make_trial_results(5)
        json_str = JSONExporter.export_full_report({
            "scenario_name": "json-test",
            "trial_results": results,
        })
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "scenario_name" in parsed
        assert parsed["scenario_name"] == "json-test"
        assert "summary" in parsed
        assert parsed["summary"]["total_trials"] == 5
        assert "format_version" in parsed

    def test_export_full_report_empty_data(self) -> None:
        json_str = JSONExporter.export_full_report({})
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "generated_at" in parsed
        assert "format_version" in parsed

    def test_save_creates_file(self, tmp_path) -> None:
        path = tmp_path / "output.json"
        JSONExporter.save({"scenario_name": "save-test"}, str(path))
        assert path.exists()
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["scenario_name"] == "save-test"
