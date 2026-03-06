"""JSON exporter for AgentAssay reports.

Serialises trial results, stochastic verdicts, and full reports to
JSON strings for programmatic consumption. All Pydantic models use
``.model_dump(mode="json")`` which handles datetime objects, enums,
tuples, and custom types.

The exported JSON is deterministic (sorted keys, no-indent by default)
so that two exports of the same data produce byte-identical output --
useful for diffing and caching.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentassay.core.models import TrialResult
from agentassay.coverage.aggregate import CoverageTuple
from agentassay.mutation.runner import MutationSuiteResult
from agentassay.metamorphic.runner import MetamorphicTestResult
from agentassay.verdicts.gate import GateReport
from agentassay.verdicts.verdict import StochasticVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_model(obj: Any) -> Any:
    """Recursively serialize Pydantic models and known types.

    Falls back to ``str()`` for anything not JSON-serialisable.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_serialize_model(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_model(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


def _to_json(obj: Any, *, indent: int | None = None) -> str:
    """Convert a serialisable object to a JSON string.

    Parameters
    ----------
    obj
        Any JSON-serialisable object (or Pydantic model).
    indent
        JSON indentation level. ``None`` for compact output,
        ``2`` for human-readable output.

    Returns
    -------
    str
        JSON string.
    """
    serialised = _serialize_model(obj)
    return json.dumps(serialised, sort_keys=True, indent=indent, default=str)


# ===================================================================
# JSONExporter
# ===================================================================


class JSONExporter:
    """Export AgentAssay data structures to JSON format.

    All methods are static -- no instance state is required. The
    class exists purely for namespacing and discoverability.

    Example
    -------
    >>> from agentassay.reporting import JSONExporter
    >>> json_str = JSONExporter.export_verdict(my_verdict)
    >>> JSONExporter.save(full_data, "report.json")
    """

    @staticmethod
    def export_results(
        results: list[TrialResult],
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize a list of trial results to a JSON string.

        Parameters
        ----------
        results
            List of ``TrialResult`` objects.
        indent
            JSON indentation. Default 2 for readability.

        Returns
        -------
        str
            JSON array of serialised trial results.
        """
        if not results:
            return "[]"
        serialised = [r.model_dump(mode="json") for r in results]
        return json.dumps(serialised, sort_keys=True, indent=indent, default=str)

    @staticmethod
    def export_verdict(
        verdict: StochasticVerdict,
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize a stochastic verdict to a JSON string.

        Parameters
        ----------
        verdict
            The ``StochasticVerdict`` to export.
        indent
            JSON indentation. Default 2 for readability.

        Returns
        -------
        str
            JSON object of the verdict.
        """
        serialised = verdict.model_dump(mode="json")
        return json.dumps(serialised, sort_keys=True, indent=indent, default=str)

    @staticmethod
    def export_full_report(
        data: dict[str, Any],
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize a complete report dictionary to JSON.

        The ``data`` dict may contain any combination of:
            - ``scenario_name``: str
            - ``trial_results``: list[TrialResult]
            - ``verdict``: StochasticVerdict
            - ``coverage``: CoverageTuple
            - ``mutation_report``: MutationSuiteResult
            - ``gate_decision``: GateReport
            - ``metamorphic_report``: MetamorphicTestResult

        Each Pydantic model is serialised via ``.model_dump(mode="json")``.
        Non-model values are passed through as-is (must be JSON-
        serialisable).

        Parameters
        ----------
        data
            Report data dictionary.
        indent
            JSON indentation. Default 2 for readability.

        Returns
        -------
        str
            JSON object of the full report.
        """
        output: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0.0",
        }

        # Scenario name
        if "scenario_name" in data:
            output["scenario_name"] = data["scenario_name"]

        # Trial results
        trial_results: list[TrialResult] | None = data.get("trial_results")
        if trial_results is not None:
            output["trial_results"] = [
                r.model_dump(mode="json") for r in trial_results
            ]
            n = len(trial_results)
            passed = sum(1 for r in trial_results if r.passed)
            output["summary"] = {
                "total_trials": n,
                "passed": passed,
                "failed": n - passed,
                "pass_rate": passed / n if n > 0 else 0.0,
                "mean_score": (
                    sum(r.score for r in trial_results) / n if n > 0 else 0.0
                ),
                "total_cost_usd": sum(
                    r.trace.total_cost_usd for r in trial_results
                ),
            }

        # Verdict
        verdict: StochasticVerdict | None = data.get("verdict")
        if verdict is not None:
            output["verdict"] = verdict.model_dump(mode="json")

        # Coverage
        coverage: CoverageTuple | None = data.get("coverage")
        if coverage is not None:
            cov_dump = coverage.model_dump(mode="json")
            cov_dump["overall"] = coverage.overall
            cov_dump["weakest_dimension"] = coverage.weakest[0]
            cov_dump["weakest_score"] = coverage.weakest[1]
            output["coverage"] = cov_dump

        # Mutation report
        mutation_report: MutationSuiteResult | None = data.get("mutation_report")
        if mutation_report is not None:
            output["mutation_report"] = mutation_report.model_dump(mode="json")

        # Gate decision
        gate_decision: GateReport | None = data.get("gate_decision")
        if gate_decision is not None:
            gate_dump = gate_decision.model_dump(mode="json")
            gate_dump["exit_code"] = gate_decision.exit_code
            output["gate_decision"] = gate_dump

        # Metamorphic report
        metamorphic_report: MetamorphicTestResult | None = data.get(
            "metamorphic_report"
        )
        if metamorphic_report is not None:
            output["metamorphic_report"] = metamorphic_report.model_dump(
                mode="json"
            )

        return json.dumps(output, sort_keys=True, indent=indent, default=str)

    @staticmethod
    def save(
        data: dict[str, Any],
        path: str | Path,
        *,
        indent: int | None = 2,
    ) -> None:
        """Export a full report to a JSON file.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        data
            Report data dictionary (same format as
            ``export_full_report``).
        path
            File path to write the JSON report to.
        indent
            JSON indentation. Default 2 for readability.
        """
        json_str = JSONExporter.export_full_report(data, indent=indent)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
