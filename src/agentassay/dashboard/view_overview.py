# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Overview view — landing page with KPI cards, trends, and gate summary.

Renders:
    * 4 summary metric cards (total runs, pass rate, coverage, cost)
    * Recent runs table (last 10)
    * 7-day pass-rate trend chart
    * 7-day cost trend chart
    * Gate decision summary (DEPLOY / BLOCK / WARN counts)
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agentassay.dashboard.helpers import (
    create_trend_chart,
    empty_state,
    format_cost,
    format_pass_rate,
    metric_card,
    status_badge,
)
from agentassay.persistence import QueryAPI
from agentassay.persistence.storage import ResultStore


def render_overview(query_api: QueryAPI, store: ResultStore) -> None:
    """Render the Overview dashboard page.

    Parameters
    ----------
    query_api : QueryAPI
        Read-only analytical query interface.
    store : ResultStore
        Direct store access for low-level reads.
    """
    st.header("Overview")

    summary = query_api.get_run_summary()

    # Guard: no data at all
    if summary["total_runs"] == 0:
        empty_state(
            "No test runs yet. Run <code>agentassay run</code> to get started."
        )
        return

    # ── KPI cards ──────────────────────────────────────────────────────

    _render_kpi_cards(summary, query_api)

    st.markdown("---")

    # ── Recent runs table ──────────────────────────────────────────────

    _render_recent_runs(query_api)

    st.markdown("---")

    # ── Trend charts (side-by-side) ────────────────────────────────────

    _render_trend_charts(query_api)

    st.markdown("---")

    # ── Gate decision summary ──────────────────────────────────────────

    _render_gate_summary(query_api)


# ── Private section renderers ──────────────────────────────────────────


def _render_kpi_cards(summary: dict[str, Any], query_api: QueryAPI) -> None:
    """Four KPI metric cards in a single row."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Runs", str(summary["total_runs"]))

    with c2:
        rate = summary.get("avg_pass_rate")
        # Compute delta vs previous 7-day window
        delta = _pass_rate_delta(query_api)
        metric_card(
            "Avg Pass Rate",
            format_pass_rate(rate),
            delta=delta,
        )

    with c3:
        coverage_data = query_api.get_coverage_trend(days=7)
        if coverage_data:
            avg_cov = sum(r["avg_score"] for r in coverage_data) / len(
                coverage_data
            )
            metric_card("Coverage", format_pass_rate(avg_cov))
        else:
            metric_card("Coverage", "--")

    with c4:
        metric_card(
            "Total Cost",
            format_cost(summary["total_cost"]),
            delta_color="inverse",
        )


def _pass_rate_delta(query_api: QueryAPI) -> str | None:
    """Compute the pass-rate delta between the last 7 days and the prior 7.

    Returns
    -------
    str | None
        e.g. ``"+2.1%"`` or ``"-0.8%"``, or ``None`` if insufficient data.
    """
    recent = query_api.get_pass_rate_trend(days=7)
    prior = query_api.get_pass_rate_trend(days=14)

    if not recent:
        return None

    recent_dates = {row["date"] for row in recent}
    prior_only = [row for row in prior if row["date"] not in recent_dates]

    if not prior_only:
        return None

    avg_recent = sum(r["pass_rate"] for r in recent) / len(recent)
    avg_prior = sum(r["pass_rate"] for r in prior_only) / len(prior_only)
    diff = (avg_recent - avg_prior) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}%"


def _render_recent_runs(query_api: QueryAPI) -> None:
    """Table showing the 10 most recent runs."""
    st.subheader("Recent Runs")

    runs = query_api.list_runs(limit=10)
    if not runs:
        empty_state("No runs recorded yet.")
        return

    rows: list[dict[str, str]] = []
    for run in runs:
        rows.append(
            {
                "Agent": run.get("agent_name", "--"),
                "Model": run.get("model", "--"),
                "Framework": run.get("framework", "--"),
                "Status": status_badge(run.get("status")),
                "Cost": format_cost(run.get("total_cost")),
                "Started": _short_ts(run.get("started_at")),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


def _render_trend_charts(query_api: QueryAPI) -> None:
    """Side-by-side 7-day pass rate and cost trend charts."""
    left, right = st.columns(2)

    with left:
        pr_data = query_api.get_pass_rate_trend(days=7)
        if pr_data:
            fig = create_trend_chart(
                pr_data,
                x_col="date",
                y_col="pass_rate",
                title="7-Day Pass Rate",
                y_format=".0%",
                color="#00CC96",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            empty_state("Not enough data for a 7-day pass-rate trend.")

    with right:
        cost_data = query_api.get_cost_trend(days=7)
        if cost_data:
            fig = create_trend_chart(
                cost_data,
                x_col="date",
                y_col="total_cost",
                title="7-Day Cost Trend",
                y_format="$.2f",
                color="#EF553B",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            empty_state("Not enough data for a 7-day cost trend.")


def _render_gate_summary(query_api: QueryAPI) -> None:
    """Counts of DEPLOY / BLOCK / WARN gate decisions in the last 7 days."""
    st.subheader("Gate Decisions (Last 7 Days)")

    gates = query_api.get_gate_history(days=7)
    if not gates:
        empty_state("No deployment gate decisions recorded.")
        return

    counts: dict[str, int] = {"DEPLOY": 0, "BLOCK": 0, "WARN": 0}
    for g in gates:
        decision = g.get("decision", "").upper()
        if decision in counts:
            counts[decision] += 1
        else:
            counts[decision] = counts.get(decision, 0) + 1

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("\U0001f680 DEPLOY", str(counts.get("DEPLOY", 0)))
    with c2:
        metric_card("\U0001f6d1 BLOCK", str(counts.get("BLOCK", 0)))
    with c3:
        metric_card("\u26a0\ufe0f WARN", str(counts.get("WARN", 0)))


# ── Utility ────────────────────────────────────────────────────────────


def _short_ts(ts: str | None) -> str:
    """Shorten an ISO-8601 timestamp to ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return "--"
    return ts[:16].replace("T", " ")
