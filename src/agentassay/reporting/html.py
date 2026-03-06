"""HTML report generator for AgentAssay.

Produces a self-contained HTML file with embedded CSS (dark theme,
modern design). Zero external dependencies at render time -- the
generated HTML works offline without any CDN links.

Uses Jinja2 template strings (not external files) so the reporter
is fully portable as a single Python module.

The ``data`` dict expected by ``generate_report`` may contain any
subset of the following keys:

    - ``scenario_name``: str
    - ``trial_results``: list[TrialResult]
    - ``verdict``: StochasticVerdict
    - ``coverage``: CoverageTuple
    - ``mutation_report``: MutationSuiteResult
    - ``gate_decision``: GateReport
    - ``metamorphic_report``: MetamorphicTestResult

Missing keys are handled gracefully -- the corresponding section is
simply omitted from the output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

from agentassay.core.models import TrialResult
from agentassay.coverage.aggregate import CoverageTuple
from agentassay.mutation.runner import MutationSuiteResult
from agentassay.metamorphic.runner import MetamorphicTestResult
from agentassay.verdicts.gate import GateReport
from agentassay.verdicts.verdict import StochasticVerdict


# ---------------------------------------------------------------------------
# Jinja2 template (self-contained HTML with embedded CSS)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentAssay Report — {{ scenario_name }}</title>
<style>
  /* ── Reset & base ──────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    line-height: 1.6;
    padding: 2rem;
    max-width: 1100px;
    margin: 0 auto;
  }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ── Header ────────────────────────────────────────────── */
  .report-header {
    border-bottom: 1px solid #30363d;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
  }
  .report-header h1 {
    font-size: 1.6rem;
    color: #f0f6fc;
    margin-bottom: 0.25rem;
  }
  .report-header .meta {
    font-size: 0.85rem;
    color: #8b949e;
  }

  /* ── Cards ─────────────────────────────────────────────── */
  .card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
  }
  .card h2 {
    font-size: 1.1rem;
    color: #f0f6fc;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
  }

  /* ── Verdict badge ─────────────────────────────────────── */
  .badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .badge-pass { background: #238636; color: #fff; }
  .badge-fail { background: #da3633; color: #fff; }
  .badge-inconclusive { background: #9e6a03; color: #fff; }
  .badge-deploy { background: #238636; color: #fff; }
  .badge-block { background: #da3633; color: #fff; }
  .badge-warn { background: #9e6a03; color: #fff; }
  .badge-skip { background: #484f58; color: #c9d1d9; }

  /* ── KPI row ───────────────────────────────────────────── */
  .kpi-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }
  .kpi {
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 0.75rem;
    background: #0d1117;
    border-radius: 6px;
    border: 1px solid #21262d;
  }
  .kpi .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }
  .kpi .value { font-size: 1.4rem; font-weight: 700; color: #f0f6fc; }

  /* ── Bar charts (CSS-only) ─────────────────────────────── */
  .bar-container {
    background: #21262d;
    border-radius: 4px;
    height: 20px;
    position: relative;
    overflow: hidden;
    margin: 0.25rem 0;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }
  .bar-label {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.7rem;
    font-weight: 600;
    color: #f0f6fc;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }
  .fill-green { background: linear-gradient(90deg, #238636, #2ea043); }
  .fill-yellow { background: linear-gradient(90deg, #9e6a03, #bb8009); }
  .fill-red { background: linear-gradient(90deg, #da3633, #f85149); }
  .fill-blue { background: linear-gradient(90deg, #1f6feb, #388bfd); }

  /* ── CI visualization ──────────────────────────────────── */
  .ci-viz {
    position: relative;
    height: 30px;
    background: #21262d;
    border-radius: 4px;
    margin: 0.5rem 0;
  }
  .ci-range {
    position: absolute;
    height: 100%;
    background: rgba(56, 139, 253, 0.3);
    border-left: 2px solid #388bfd;
    border-right: 2px solid #388bfd;
    border-radius: 4px;
  }
  .ci-point {
    position: absolute;
    width: 10px;
    height: 10px;
    background: #f0f6fc;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
  }
  .ci-threshold {
    position: absolute;
    width: 2px;
    height: 100%;
    background: #da3633;
    top: 0;
  }

  /* ── Coverage heatmap ──────────────────────────────────── */
  .heatmap-row {
    display: flex;
    align-items: center;
    margin: 0.35rem 0;
    gap: 0.75rem;
  }
  .heatmap-label {
    width: 80px;
    text-align: right;
    font-size: 0.85rem;
    font-weight: 500;
  }
  .heatmap-cell {
    width: 60px;
    height: 28px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
  }

  /* ── Tables ────────────────────────────────────────────── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid #30363d;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
  }
  td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #21262d;
  }
  tr:hover td { background: rgba(56, 139, 253, 0.05); }

  /* ── Footer ────────────────────────────────────────────── */
  .footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #30363d;
    text-align: center;
    font-size: 0.75rem;
    color: #484f58;
  }
</style>
</head>
<body>

<!-- ═══ Header ════════════════════════════════════════════ -->
<div class="report-header">
  <h1>AgentAssay Report</h1>
  <div class="meta">
    Scenario: <strong>{{ scenario_name }}</strong> &middot;
    Generated: {{ timestamp }}
  </div>
</div>

<!-- ═══ Verdict ═══════════════════════════════════════════ -->
{% if verdict %}
<div class="card">
  <h2>Stochastic Verdict</h2>
  <div style="margin-bottom: 0.75rem;">
    <span class="badge badge-{{ verdict.status.value }}">{{ verdict.status.value | upper }}</span>
  </div>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Pass Rate</div>
      <div class="value">{{ "%.1f" | format(verdict.pass_rate * 100) }}%</div>
    </div>
    <div class="kpi">
      <div class="label">Trials</div>
      <div class="value">{{ verdict.num_passed }}/{{ verdict.num_trials }}</div>
    </div>
    <div class="kpi">
      <div class="label">Confidence</div>
      <div class="value">{{ "%.0f" | format(verdict.confidence * 100) }}%</div>
    </div>
    {% if verdict.p_value is not none %}
    <div class="kpi">
      <div class="label">p-value</div>
      <div class="value">{{ "%.4f" | format(verdict.p_value) }}</div>
    </div>
    {% endif %}
    {% if verdict.effect_size is not none %}
    <div class="kpi">
      <div class="label">Effect Size</div>
      <div class="value">{{ "%.4f" | format(verdict.effect_size) }}</div>
    </div>
    {% endif %}
  </div>

  <!-- CI visualization -->
  <div style="margin-top: 0.75rem;">
    <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 0.25rem;">
      Confidence Interval: [{{ "%.4f" | format(verdict.pass_rate_ci[0]) }},
      {{ "%.4f" | format(verdict.pass_rate_ci[1]) }}]
    </div>
    <div class="ci-viz">
      <div class="ci-range"
           style="left: {{ (verdict.pass_rate_ci[0] * 100) }}%;
                  width: {{ ((verdict.pass_rate_ci[1] - verdict.pass_rate_ci[0]) * 100) }}%;">
      </div>
      <div class="ci-point" style="left: {{ (verdict.pass_rate * 100) }}%;"></div>
      {% if verdict_threshold is not none %}
      <div class="ci-threshold" style="left: {{ (verdict_threshold * 100) }}%;"></div>
      {% endif %}
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #484f58;">
      <span>0%</span><span>50%</span><span>100%</span>
    </div>
  </div>
</div>
{% endif %}

<!-- ═══ Trial Results ═════════════════════════════════════ -->
{% if trial_results %}
<div class="card">
  <h2>Trial Results</h2>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Total</div>
      <div class="value">{{ trial_results | length }}</div>
    </div>
    <div class="kpi">
      <div class="label">Passed</div>
      <div class="value" style="color: #3fb950;">{{ passed_count }}</div>
    </div>
    <div class="kpi">
      <div class="label">Failed</div>
      <div class="value" style="color: #f85149;">{{ failed_count }}</div>
    </div>
    <div class="kpi">
      <div class="label">Pass Rate</div>
      <div class="value">{{ "%.1f" | format(pass_rate * 100) }}%</div>
    </div>
    <div class="kpi">
      <div class="label">Mean Score</div>
      <div class="value">{{ "%.3f" | format(mean_score) }}</div>
    </div>
    <div class="kpi">
      <div class="label">Total Cost</div>
      <div class="value">${{ "%.4f" | format(total_cost) }}</div>
    </div>
  </div>

  <!-- Pass rate bar -->
  <div class="bar-container" style="height: 24px;">
    {% set bar_color = "fill-green" if pass_rate >= 0.8 else "fill-yellow" if pass_rate >= 0.5 else "fill-red" %}
    <div class="bar-fill {{ bar_color }}" style="width: {{ (pass_rate * 100) }}%;"></div>
    <div class="bar-label">{{ "%.1f" | format(pass_rate * 100) }}%</div>
  </div>

  <!-- Detail table -->
  <div style="max-height: 400px; overflow-y: auto; margin-top: 1rem;">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Status</th>
          <th>Score</th>
          <th>Cost</th>
          <th>Duration</th>
          <th>Steps</th>
        </tr>
      </thead>
      <tbody>
        {% for r in trial_results %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>
            {% if r.passed %}
            <span style="color: #3fb950;">PASS</span>
            {% else %}
            <span style="color: #f85149;">FAIL</span>
            {% endif %}
          </td>
          <td>{{ "%.4f" | format(r.score) }}</td>
          <td>${{ "%.4f" | format(r.trace.total_cost_usd) }}</td>
          <td>{{ "%.0f" | format(r.trace.total_duration_ms) }} ms</td>
          <td>{{ r.trace.step_count }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

<!-- ═══ Coverage ══════════════════════════════════════════ -->
{% if coverage %}
<div class="card">
  <h2>Agent Coverage Vector (5D)</h2>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Overall</div>
      <div class="value">{{ "%.1f" | format(coverage.overall * 100) }}%</div>
    </div>
  </div>

  {% for dim_name, dim_val in coverage_dims %}
  <div class="heatmap-row">
    <div class="heatmap-label">{{ dim_name }}</div>
    <div class="bar-container" style="flex: 1; height: 24px;">
      {% set c = "fill-green" if dim_val >= 0.8 else "fill-yellow" if dim_val >= 0.5 else "fill-red" %}
      <div class="bar-fill {{ c }}" style="width: {{ (dim_val * 100) }}%;"></div>
      <div class="bar-label">{{ "%.1f" | format(dim_val * 100) }}%</div>
    </div>
    <div class="heatmap-cell"
         style="background: {{ _coverage_color(dim_val) }};">
      {{ "%.0f" | format(dim_val * 100) }}
    </div>
  </div>
  {% endfor %}
</div>
{% endif %}

<!-- ═══ Mutation ══════════════════════════════════════════ -->
{% if mutation_report %}
<div class="card">
  <h2>Mutation Testing</h2>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Mutation Score</div>
      <div class="value"
           style="color: {{ '#3fb950' if mutation_report.mutation_score >= 0.8 else '#bb8009' if mutation_report.mutation_score >= 0.5 else '#f85149' }};">
        {{ "%.1f" | format(mutation_report.mutation_score * 100) }}%
      </div>
    </div>
    <div class="kpi">
      <div class="label">Total</div>
      <div class="value">{{ mutation_report.total_mutants }}</div>
    </div>
    <div class="kpi">
      <div class="label">Killed</div>
      <div class="value" style="color: #3fb950;">{{ mutation_report.killed_mutants }}</div>
    </div>
    <div class="kpi">
      <div class="label">Survived</div>
      <div class="value" style="color: #f85149;">{{ mutation_report.survived_mutants }}</div>
    </div>
    <div class="kpi">
      <div class="label">Errored</div>
      <div class="value" style="color: #bb8009;">{{ mutation_report.errored_mutants }}</div>
    </div>
  </div>

  {% if mutation_report.per_category %}
  <h3 style="font-size: 0.9rem; margin: 1rem 0 0.5rem; color: #8b949e;">Per-Category Breakdown</h3>
  <table>
    <thead><tr><th>Category</th><th>Score</th><th>Bar</th></tr></thead>
    <tbody>
      {% for cat, score in mutation_report.per_category.items() | sort %}
      <tr>
        <td>{{ cat | title }}</td>
        <td>{{ "%.1f" | format(score * 100) }}%</td>
        <td>
          <div class="bar-container" style="height: 16px;">
            {% set c = "fill-green" if score >= 0.8 else "fill-yellow" if score >= 0.5 else "fill-red" %}
            <div class="bar-fill {{ c }}" style="width: {{ (score * 100) }}%;"></div>
          </div>
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endif %}

<!-- ═══ Gate Decision ═════════════════════════════════════ -->
{% if gate_decision %}
<div class="card">
  <h2>Deployment Gate</h2>
  <div style="margin-bottom: 0.75rem;">
    <span class="badge badge-{{ gate_decision.overall_decision.value }}">
      {{ gate_decision.overall_decision.value | upper }}
    </span>
    <span style="margin-left: 0.5rem; font-size: 0.8rem; color: #8b949e;">
      Exit code: {{ gate_decision.exit_code }}
    </span>
  </div>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Deployed</div>
      <div class="value" style="color: #3fb950;">{{ gate_decision.passed_scenarios }}</div>
    </div>
    <div class="kpi">
      <div class="label">Blocked</div>
      <div class="value" style="color: #f85149;">{{ gate_decision.blocked_scenarios }}</div>
    </div>
    <div class="kpi">
      <div class="label">Warned</div>
      <div class="value" style="color: #bb8009;">{{ gate_decision.warned_scenarios }}</div>
    </div>
    <div class="kpi">
      <div class="label">Skipped</div>
      <div class="value">{{ gate_decision.skipped_scenarios }}</div>
    </div>
  </div>

  {% if gate_decision.scenario_decisions %}
  <table>
    <thead><tr><th>Scenario</th><th>Decision</th><th>Reason</th></tr></thead>
    <tbody>
      {% for name, dec in gate_decision.scenario_decisions.items() %}
      <tr>
        <td>{{ name }}</td>
        <td><span class="badge badge-{{ dec.value }}">{{ dec.value | upper }}</span></td>
        <td style="font-size: 0.8rem; color: #8b949e;">{{ gate_decision.scenario_reasons.get(name, '') }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endif %}

<!-- ═══ Metamorphic ═══════════════════════════════════════ -->
{% if metamorphic_report %}
<div class="card">
  <h2>Metamorphic Testing</h2>
  <div class="kpi-row">
    <div class="kpi">
      <div class="label">Violation Rate</div>
      <div class="value"
           style="color: {{ '#3fb950' if metamorphic_report.violation_rate <= 0.1 else '#bb8009' if metamorphic_report.violation_rate <= 0.3 else '#f85149' }};">
        {{ "%.1f" | format(metamorphic_report.violation_rate * 100) }}%
      </div>
    </div>
    <div class="kpi">
      <div class="label">Total</div>
      <div class="value">{{ metamorphic_report.total_relations }}</div>
    </div>
    <div class="kpi">
      <div class="label">Violations</div>
      <div class="value" style="color: #f85149;">{{ metamorphic_report.violations }}</div>
    </div>
    <div class="kpi">
      <div class="label">Held</div>
      <div class="value" style="color: #3fb950;">{{ metamorphic_report.total_relations - metamorphic_report.violations }}</div>
    </div>
  </div>

  {% if metamorphic_report.per_family %}
  <table>
    <thead><tr><th>Family</th><th>Tested</th><th>Violations</th><th>Rate</th></tr></thead>
    <tbody>
      {% for fam, info in metamorphic_report.per_family.items() | sort %}
      <tr>
        <td>{{ fam | title }}</td>
        <td>{{ info.get('tested', 0) }}</td>
        <td>{{ info.get('violations', 0) }}</td>
        <td>{{ "%.1f" | format(info.get('rate', 0) * 100) }}%</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  {% if metamorphic_report.results %}
  <h3 style="font-size: 0.9rem; margin: 1rem 0 0.5rem; color: #8b949e;">Per-Relation Detail</h3>
  <table>
    <thead><tr><th>Relation</th><th>Family</th><th>Holds</th><th>Similarity</th></tr></thead>
    <tbody>
      {% for r in metamorphic_report.results %}
      <tr>
        <td>{{ r.relation_name }}</td>
        <td>{{ r.relation_family }}</td>
        <td>
          {% if r.holds %}
          <span style="color: #3fb950;">HOLDS</span>
          {% else %}
          <span style="color: #f85149;">VIOLATED</span>
          {% endif %}
        </td>
        <td>{{ "%.4f" | format(r.similarity_score) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endif %}

<!-- ═══ Footer ════════════════════════════════════════════ -->
<div class="footer">
  Generated by AgentAssay &middot; {{ timestamp }}
</div>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Coverage colour helper (exposed to Jinja2 as a callable)
# ---------------------------------------------------------------------------

def _coverage_color(value: float) -> str:
    """Map a [0, 1] coverage value to a hex background colour.

    Uses a green-yellow-red gradient:
        >= 0.8  -> green (#238636)
        >= 0.5  -> amber (#9e6a03)
        < 0.5   -> red   (#da3633)
    """
    if value >= 0.8:
        return "#238636"
    if value >= 0.5:
        return "#9e6a03"
    return "#da3633"


# ===================================================================
# HTMLReporter
# ===================================================================


class HTMLReporter:
    """Generate self-contained HTML reports from AgentAssay test data.

    The output is a single HTML file with all CSS embedded inline.
    No external resources (scripts, stylesheets, fonts) are required.

    Example
    -------
    >>> reporter = HTMLReporter()
    >>> html = reporter.generate_report({
    ...     "scenario_name": "code-gen-basic",
    ...     "verdict": my_verdict,
    ...     "trial_results": my_results,
    ... })
    >>> reporter.save_report(data, "report.html")
    """

    def __init__(self) -> None:
        self._env = Environment(loader=BaseLoader(), autoescape=True)
        # Register the coverage colour helper as a global function
        self._env.globals["_coverage_color"] = _coverage_color
        self._template = self._env.from_string(_HTML_TEMPLATE)

    def generate_report(self, data: dict[str, Any]) -> str:
        """Generate a self-contained HTML report string.

        Parameters
        ----------
        data
            Dictionary that may contain any of: ``scenario_name``,
            ``trial_results``, ``verdict``, ``coverage``,
            ``mutation_report``, ``gate_decision``,
            ``metamorphic_report``.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        context = self._build_context(data)
        return self._template.render(**context)

    def save_report(self, data: dict[str, Any], path: str | Path) -> None:
        """Generate an HTML report and write it to a file.

        Parameters
        ----------
        data
            Same as ``generate_report``.
        path
            File path to write the HTML report to. Parent
            directories are created if they do not exist.
        """
        html = self.generate_report(data)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    # -- Internal: build template context ------------------------------------

    def _build_context(self, data: dict[str, Any]) -> dict[str, Any]:
        """Transform raw data dict into the Jinja2 template context.

        Computes derived values (pass counts, mean scores, etc.) so
        the template itself stays logic-light.
        """
        scenario_name = data.get("scenario_name", "Unnamed Scenario")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        ctx: dict[str, Any] = {
            "scenario_name": scenario_name,
            "timestamp": timestamp,
            "verdict": None,
            "verdict_threshold": None,
            "trial_results": None,
            "passed_count": 0,
            "failed_count": 0,
            "pass_rate": 0.0,
            "mean_score": 0.0,
            "total_cost": 0.0,
            "coverage": None,
            "coverage_dims": [],
            "mutation_report": None,
            "gate_decision": None,
            "metamorphic_report": None,
        }

        # Verdict
        verdict: StochasticVerdict | None = data.get("verdict")
        if verdict is not None:
            ctx["verdict"] = verdict
            ctx["verdict_threshold"] = verdict.details.get("threshold")

        # Trial results
        trial_results: list[TrialResult] | None = data.get("trial_results")
        if trial_results:
            ctx["trial_results"] = trial_results
            n = len(trial_results)
            passed = sum(1 for r in trial_results if r.passed)
            ctx["passed_count"] = passed
            ctx["failed_count"] = n - passed
            ctx["pass_rate"] = passed / n if n > 0 else 0.0
            ctx["mean_score"] = (
                sum(r.score for r in trial_results) / n if n > 0 else 0.0
            )
            ctx["total_cost"] = sum(
                r.trace.total_cost_usd for r in trial_results
            )

        # Coverage
        coverage: CoverageTuple | None = data.get("coverage")
        if coverage is not None:
            ctx["coverage"] = coverage
            ctx["coverage_dims"] = [
                ("Tool", coverage.tool),
                ("Path", coverage.path),
                ("State", coverage.state),
                ("Boundary", coverage.boundary),
                ("Model", coverage.model),
            ]

        # Mutation report
        mutation_report: MutationSuiteResult | None = data.get("mutation_report")
        if mutation_report is not None:
            ctx["mutation_report"] = mutation_report

        # Gate decision
        gate_decision: GateReport | None = data.get("gate_decision")
        if gate_decision is not None:
            ctx["gate_decision"] = gate_decision

        # Metamorphic report
        metamorphic_report: MetamorphicTestResult | None = data.get(
            "metamorphic_report"
        )
        if metamorphic_report is not None:
            ctx["metamorphic_report"] = metamorphic_report

        return ctx
