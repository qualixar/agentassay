"""History view — trend analysis across multiple test runs.

Renders:
    * Filter controls (agent name, model, date range)
    * Pass rate over time (with regression markers)
    * Cost per run over time
    * Coverage over time by dimension
    * Filterable run history table
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import streamlit as st

from agentassay.dashboard.helpers import (
    create_trend_chart,
    empty_state,
    format_cost,
    format_pass_rate,
    status_badge,
)
from agentassay.persistence import QueryAPI
from agentassay.persistence.storage import ResultStore


def render_history(query_api: QueryAPI, store: ResultStore) -> None:
    """Render the History trend-analysis page.

    Parameters
    ----------
    query_api : QueryAPI
        Read-only analytical query interface.
    store : ResultStore
        Direct store access for low-level reads.
    """
    st.header("History & Trends")

    # ── Filters ────────────────────────────────────────────────────────

    agent_name, model, days = _render_filters(query_api)

    # ── Pass rate trend ────────────────────────────────────────────────

    _render_pass_rate_trend(query_api, agent_name, model, days)

    st.markdown("---")

    # ── Cost trend ─────────────────────────────────────────────────────

    _render_cost_trend(query_api, agent_name, model, days)

    st.markdown("---")

    # ── Coverage trend ─────────────────────────────────────────────────

    _render_coverage_trend(query_api, days)

    st.markdown("---")

    # ── Run history table ──────────────────────────────────────────────

    _render_run_table(query_api, agent_name, model)


# ── Private section renderers ──────────────────────────────────────────


def _render_filters(
    query_api: QueryAPI,
) -> tuple[str | None, str | None, int]:
    """Render the filter bar and return current selections.

    Returns
    -------
    tuple[str | None, str | None, int]
        ``(agent_name, model, days)`` — any can be None meaning "all".
    """
    # Fetch available agents and models for filter dropdowns
    runs = query_api.list_runs(limit=200)
    agents = sorted({r["agent_name"] for r in runs if r.get("agent_name")})
    models = sorted({r["model"] for r in runs if r.get("model")})

    c1, c2, c3 = st.columns(3)

    with c1:
        agent_options = ["All Agents"] + agents
        agent_sel = st.selectbox("Agent", agent_options)
        agent_name = None if agent_sel == "All Agents" else agent_sel

    with c2:
        model_options = ["All Models"] + models
        model_sel = st.selectbox("Model", model_options)
        model = None if model_sel == "All Models" else model_sel

    with c3:
        today = date.today()
        default_start = today - timedelta(days=30)
        date_range = st.date_input(
            "Date Range",
            value=(default_start, today),
            max_value=today,
        )
        # Calculate days from the date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            days = (date_range[1] - date_range[0]).days
        else:
            days = 30

    return agent_name, model, max(days, 1)


def _render_pass_rate_trend(
    query_api: QueryAPI,
    agent_name: str | None,
    model: str | None,
    days: int,
) -> None:
    """Pass rate line chart with regression points highlighted in red."""
    st.subheader("Pass Rate Over Time")

    data = query_api.get_pass_rate_trend(
        agent_name=agent_name, model=model, days=days
    )
    if not data:
        empty_state("Not enough data for pass-rate trend.")
        return

    import plotly.graph_objects as go

    from agentassay.dashboard.helpers import _get_theme_layout

    dates = [row["date"] for row in data]
    rates = [row["pass_rate"] for row in data]

    fig = go.Figure()

    # Main line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rates,
            mode="lines+markers",
            line={"color": "#00CC96", "width": 2},
            marker={"size": 6},
            name="Pass Rate",
        )
    )

    # Highlight regression points (pass rate dropped from previous day)
    reg_x, reg_y = _find_regressions(dates, rates)
    if reg_x:
        fig.add_trace(
            go.Scatter(
                x=reg_x,
                y=reg_y,
                mode="markers",
                marker={"color": "#EF553B", "size": 10, "symbol": "x"},
                name="Regression",
            )
        )

    base = _get_theme_layout()
    merged = {**base}
    merged["title"] = {"text": "Pass Rate Over Time", "font": {"size": 14}}
    merged["yaxis"] = {**base.get("yaxis", {}), "tickformat": ".0%", "range": [0, 1.05]}
    merged["height"] = 350
    merged["showlegend"] = True
    fig.update_layout(**merged)
    st.plotly_chart(fig, width="stretch")


def _find_regressions(
    dates: list[str], rates: list[float]
) -> tuple[list[str], list[float]]:
    """Identify points where pass rate dropped from the previous day.

    Returns
    -------
    tuple[list[str], list[float]]
        ``(x_values, y_values)`` of regression points.
    """
    reg_x: list[str] = []
    reg_y: list[float] = []
    for i in range(1, len(rates)):
        if rates[i] is not None and rates[i - 1] is not None:
            if rates[i] < rates[i - 1]:
                reg_x.append(dates[i])
                reg_y.append(rates[i])
    return reg_x, reg_y


def _render_cost_trend(
    query_api: QueryAPI,
    agent_name: str | None,
    model: str | None,
    days: int,
) -> None:
    """Cost per run over time."""
    st.subheader("Cost Per Run Over Time")

    data = query_api.get_cost_trend(
        agent_name=agent_name, model=model, days=days
    )
    if not data:
        empty_state("Not enough data for cost trend.")
        return

    fig = create_trend_chart(
        data,
        x_col="date",
        y_col="total_cost",
        title="Daily Testing Cost",
        y_format="$.2f",
        color="#EF553B",
        height=300,
    )
    st.plotly_chart(fig, width="stretch")


def _render_coverage_trend(query_api: QueryAPI, days: int) -> None:
    """Coverage by dimension as a grouped bar chart."""
    st.subheader("Coverage by Dimension")

    data = query_api.get_coverage_trend(days=days)
    if not data:
        empty_state("No coverage data available.")
        return

    import plotly.graph_objects as go

    from agentassay.dashboard.helpers import _get_theme_layout

    dimensions = [row["dimension"] for row in data]
    scores = [row["avg_score"] for row in data]

    color_map = {
        "tool": "#636EFA",
        "path": "#00CC96",
        "state": "#AB63FA",
        "boundary": "#FFA15A",
        "model": "#EF553B",
    }
    colors = [color_map.get(d.lower(), "#636EFA") for d in dimensions]

    fig = go.Figure(
        go.Bar(
            x=dimensions,
            y=scores,
            marker_color=colors,
            text=[f"{s:.0%}" for s in scores],
            textposition="outside",
        )
    )
    base = _get_theme_layout()
    merged = {**base}
    merged["title"] = {"text": "Average Coverage Score", "font": {"size": 14}}
    merged["yaxis"] = {**base.get("yaxis", {}), "tickformat": ".0%", "range": [0, 1.1]}
    merged["height"] = 300
    fig.update_layout(**merged)
    st.plotly_chart(fig, width="stretch")


def _render_run_table(
    query_api: QueryAPI,
    agent_name: str | None,
    model: str | None,
) -> None:
    """Full run history table."""
    st.subheader("Run History")

    runs = query_api.list_runs(
        agent_name=agent_name, model=model, limit=50
    )
    if not runs:
        empty_state("No runs match the current filters.")
        return

    rows: list[dict[str, Any]] = []
    for r in runs:
        rows.append(
            {
                "Agent": r.get("agent_name", "--"),
                "Version": r.get("agent_version", "--") or "--",
                "Model": r.get("model", "--"),
                "Framework": r.get("framework", "--"),
                "Status": status_badge(r.get("status")),
                "Trials": r.get("total_trials") or "--",
                "Cost": format_cost(r.get("total_cost")),
                "Started": _short_ts(r.get("started_at")),
                "Completed": _short_ts(r.get("completed_at")),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


# ── Utility ────────────────────────────────────────────────────────────


def _short_ts(ts: str | None) -> str:
    """Shorten an ISO timestamp to ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return "--"
    return ts[:16].replace("T", " ")
