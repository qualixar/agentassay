# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Test Run view — drill into a single test run's details.

Renders:
    * Run selector (recent 20 runs)
    * Run header with agent info and status badge
    * Per-scenario verdict table with pass rates and confidence intervals
    * Cost breakdown pie chart by model
    * Trial timeline table (last 10 trials)
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agentassay.dashboard.helpers import (
    create_pie_chart,
    empty_state,
    format_cost,
    format_duration,
    format_pass_rate,
    metric_card,
    status_badge,
)
from agentassay.persistence import QueryAPI
from agentassay.persistence.storage import ResultStore


def render_test_run(query_api: QueryAPI, store: ResultStore) -> None:
    """Render the Test Run detail page.

    Parameters
    ----------
    query_api : QueryAPI
        Read-only analytical query interface.
    store : ResultStore
        Direct store access for low-level reads.
    """
    st.header("Test Run Detail")

    runs = query_api.list_runs(limit=20)
    if not runs:
        empty_state("No test runs yet. Run <code>agentassay run</code> to get started.")
        return

    # ── Run selector ───────────────────────────────────────────────────

    run_labels = [
        f"{r['agent_name']} / {r['model']} — {_short_ts(r.get('started_at'))}" for r in runs
    ]
    selected_idx = st.selectbox(
        "Select a run",
        range(len(run_labels)),
        format_func=lambda i: run_labels[i],
    )
    if selected_idx is None:
        return

    run = runs[selected_idx]
    run_id: str = run["id"]

    # ── Run header ─────────────────────────────────────────────────────

    _render_run_header(run)

    st.markdown("---")

    # ── Per-scenario verdict table ─────────────────────────────────────

    _render_verdict_table(store, run_id)

    st.markdown("---")

    # ── Cost breakdown + Trial timeline side-by-side ───────────────────

    left, right = st.columns([1, 2])

    with left:
        _render_cost_breakdown(store, run_id)

    with right:
        _render_trial_timeline(store, run_id)


# ── Private section renderers ──────────────────────────────────────────


def _render_run_header(run: dict[str, Any]) -> None:
    """Display the run's key metadata as KPI cards."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Agent", run.get("agent_name", "--"))
    with c2:
        metric_card("Model", run.get("model", "--"))
    with c3:
        metric_card("Framework", run.get("framework", "--"))
    with c4:
        st.markdown("**Status**")
        st.markdown(status_badge(run.get("status")))
    with c5:
        metric_card("Cost", format_cost(run.get("total_cost")))

    started = run.get("started_at", "--")
    completed = run.get("completed_at")
    ts_text = f"Started: {_short_ts(started)}"
    if completed:
        ts_text += f"  |  Completed: {_short_ts(completed)}"
    st.caption(ts_text)


def _render_verdict_table(store: ResultStore, run_id: str) -> None:
    """Per-scenario verdicts with pass rate and confidence intervals."""
    st.subheader("Scenario Verdicts")

    verdicts = store.get_verdicts(run_id)
    if not verdicts:
        empty_state("No verdict data for this run.")
        return

    rows: list[dict[str, Any]] = []
    for v in verdicts:
        rows.append(
            {
                "Scenario": v.get("scenario_id", "--"),
                "Status": status_badge(v.get("status")),
                "Trials": v.get("n_trials", 0),
                "Pass Rate": format_pass_rate(v.get("pass_rate")),
                "CI": _format_ci(v.get("ci_lower"), v.get("ci_upper")),
                "p-value": _format_pvalue(v.get("p_value")),
                "Effect Size": _format_float(v.get("effect_size")),
                "Method": v.get("method", "--"),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


def _render_cost_breakdown(store: ResultStore, run_id: str) -> None:
    """Pie chart showing per-model cost split."""
    st.subheader("Cost Breakdown")

    costs = store.get_costs(run_id)
    if not costs:
        empty_state("No cost data for this run.")
        return

    labels = [c.get("model", "unknown") for c in costs]
    values = [c.get("total_cost", 0.0) for c in costs]

    fig = create_pie_chart(labels, values, "Cost by Model", height=280)
    st.plotly_chart(fig, width="stretch")

    # Summary table underneath
    total = sum(values)
    summary_rows: list[dict[str, str]] = []
    for c in costs:
        summary_rows.append(
            {
                "Model": c.get("model", "unknown"),
                "Tokens (in)": f"{c.get('input_tokens', 0):,}",
                "Tokens (out)": f"{c.get('output_tokens', 0):,}",
                "Cost": format_cost(c.get("total_cost", 0.0)),
            }
        )
    summary_rows.append(
        {
            "Model": "TOTAL",
            "Tokens (in)": "",
            "Tokens (out)": "",
            "Cost": format_cost(total),
        }
    )
    st.dataframe(summary_rows, width="stretch", hide_index=True)


def _render_trial_timeline(store: ResultStore, run_id: str) -> None:
    """Table of the most recent trials within this run."""
    st.subheader("Trial Timeline")

    trials = store.get_trials(run_id)
    if not trials:
        empty_state("No trial data for this run.")
        return

    # Show last 10 trials (most recent first)
    recent = list(reversed(trials[-10:]))
    rows: list[dict[str, Any]] = []
    for t in recent:
        rows.append(
            {
                "Trial #": t.get("trial_num", "--"),
                "Scenario": t.get("scenario_id", "--"),
                "Result": status_badge("PASS" if t.get("success") else "FAIL"),
                "Latency": format_duration(t.get("latency_ms")),
                "Cost": format_cost(t.get("cost")),
                "Steps": t.get("step_count", 0),
                "Tokens": f"{t.get('token_count', 0):,}",
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


# ── Formatting utilities ───────────────────────────────────────────────


def _short_ts(ts: str | None) -> str:
    """Shorten an ISO timestamp to ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return "--"
    return ts[:16].replace("T", " ")


def _format_ci(lower: float | None, upper: float | None) -> str:
    """Format a confidence interval as ``[lower, upper]``."""
    if lower is None or upper is None:
        return "--"
    return f"[{lower:.3f}, {upper:.3f}]"


def _format_pvalue(pval: float | None) -> str:
    """Format a p-value, highlighting significant results."""
    if pval is None:
        return "--"
    if pval < 0.001:
        return "<0.001"
    return f"{pval:.4f}"


def _format_float(val: float | None) -> str:
    """Format a float to 3 decimal places."""
    if val is None:
        return "--"
    return f"{val:.3f}"
