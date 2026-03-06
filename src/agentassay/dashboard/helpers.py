# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Shared formatting utilities and chart factories for the AgentAssay dashboard.

Every helper in this module is framework-agnostic at import time.  Streamlit
and Plotly are imported lazily inside functions that need them, so this file
can be imported safely in non-dashboard contexts (e.g. tests, CLI).
"""

from __future__ import annotations

from typing import Any

# ── Formatting helpers ─────────────────────────────────────────────────


def format_pass_rate(rate: float | None) -> str:
    """Format a pass rate as a coloured percentage string.

    Parameters
    ----------
    rate : float | None
        Value in [0, 1].  ``None`` yields ``"--"``.

    Returns
    -------
    str
        Human-readable percentage, e.g. ``"87.3%"``.
    """
    if rate is None:
        return "--"
    return f"{rate * 100:.1f}%"


def format_cost(cost: float | None) -> str:
    """Format a USD cost value.

    Parameters
    ----------
    cost : float | None
        Dollar amount.  ``None`` yields ``"--"``.

    Returns
    -------
    str
        e.g. ``"$1.23"`` or ``"$0.00"``.
    """
    if cost is None:
        return "--"
    return f"${cost:,.2f}"


def format_duration(ms: float | None) -> str:
    """Format milliseconds into a human-readable duration string.

    Parameters
    ----------
    ms : float | None
        Duration in milliseconds.

    Returns
    -------
    str
        e.g. ``"1.2s"``, ``"2m 15s"``, ``"--"`` for ``None``.
    """
    if ms is None:
        return "--"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}m {remaining:.0f}s"


def status_badge(status: str | None) -> str:
    """Return an emoji + label string for a verdict / run status.

    Parameters
    ----------
    status : str | None
        One of ``"PASS"``, ``"FAIL"``, ``"INCONCLUSIVE"``, ``"running"``,
        ``"completed"``, ``"failed"``, ``"DEPLOY"``, ``"BLOCK"``,
        ``"WARN"``.  Case-insensitive.

    Returns
    -------
    str
        Coloured emoji badge, e.g. ``"PASS"``, ``"FAIL"``.
    """
    if status is None:
        return "--"
    mapping: dict[str, str] = {
        "pass": "\u2705 PASS",
        "fail": "\u274c FAIL",
        "inconclusive": "\u2753 INCONCLUSIVE",
        "running": "\u23f3 RUNNING",
        "completed": "\u2705 COMPLETED",
        "failed": "\u274c FAILED",
        "deploy": "\U0001f680 DEPLOY",
        "block": "\U0001f6d1 BLOCK",
        "warn": "\u26a0\ufe0f WARN",
    }
    return mapping.get(status.lower(), status)


# ── Reusable Streamlit components ──────────────────────────────────────


def empty_state(message: str) -> None:
    """Render a centred empty-state message inside a Streamlit container.

    Parameters
    ----------
    message : str
        The guidance text to display.
    """
    import streamlit as st

    st.markdown(
        f"""
        <div style="text-align:center; padding:4rem 2rem; opacity:0.6;">
            <p style="font-size:1.3rem;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: str = "normal",
) -> None:
    """Render a KPI metric card via ``st.metric``.

    Parameters
    ----------
    label : str
        Metric title.
    value : str
        Primary display value.
    delta : str | None
        Delta annotation (e.g. ``"+2.1%"``).
    delta_color : str
        ``"normal"`` (green-up / red-down), ``"inverse"``, or ``"off"``.
    """
    import streamlit as st

    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ── Chart factories ────────────────────────────────────────────────────

_DARK_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#fafafa"},
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.08)",
        "zerolinecolor": "rgba(255,255,255,0.12)",
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.08)",
        "zerolinecolor": "rgba(255,255,255,0.12)",
    },
    "margin": {"l": 40, "r": 20, "t": 40, "b": 30},
}

_LIGHT_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#1a1a2e"},
    "xaxis": {
        "gridcolor": "rgba(0,0,0,0.08)",
        "zerolinecolor": "rgba(0,0,0,0.12)",
    },
    "yaxis": {
        "gridcolor": "rgba(0,0,0,0.08)",
        "zerolinecolor": "rgba(0,0,0,0.12)",
    },
    "margin": {"l": 40, "r": 20, "t": 40, "b": 30},
}


def _get_theme_layout() -> dict[str, Any]:
    """Return the Plotly layout dict matching Streamlit's active theme.

    Detects the current Streamlit theme (dark or light) and returns
    the appropriate chart layout. Falls back to dark if detection fails.
    """
    try:
        import streamlit as st

        theme = st.get_option("theme.base")
        if theme == "light":
            return _LIGHT_LAYOUT
    except Exception:
        pass
    return _DARK_LAYOUT


def create_trend_chart(
    data: list[dict[str, Any]],
    x_col: str,
    y_col: str,
    title: str,
    y_format: str | None = None,
    color: str = "#636EFA",
    height: int = 300,
) -> Any:
    """Build a Plotly line chart from a list of row dicts.

    Parameters
    ----------
    data : list[dict]
        Rows with at least ``x_col`` and ``y_col`` keys.
    x_col : str
        Column to map to the X axis.
    y_col : str
        Column to map to the Y axis.
    title : str
        Chart title.
    y_format : str | None
        Plotly tick format string for the Y axis (e.g. ``".0%"``).
    color : str
        Hex colour for the line.
    height : int
        Chart height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Ready-to-render figure.
    """
    import plotly.graph_objects as go

    x_vals = [row.get(x_col) for row in data]
    y_vals = [row.get(y_col) for row in data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 5},
            name=y_col,
        )
    )
    base = _get_theme_layout()
    merged_layout = {**base}
    merged_layout["title"] = {"text": title, "font": {"size": 14}}
    merged_layout["height"] = height
    if y_format:
        merged_layout["yaxis"] = {
            **merged_layout.get("yaxis", {}),
            "tickformat": y_format,
        }

    fig.update_layout(**merged_layout)
    return fig


def create_pie_chart(
    labels: list[str],
    values: list[float],
    title: str,
    height: int = 300,
) -> Any:
    """Build a Plotly pie/donut chart.

    Parameters
    ----------
    labels : list[str]
        Category names.
    values : list[float]
        Corresponding numeric values.
    title : str
        Chart title.
    height : int
        Chart height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            marker={"line": {"color": "#1e1e1e", "width": 1}},
        )
    )
    fig.update_layout(
        **_get_theme_layout(),
        title={"text": title, "font": {"size": 14}},
        height=height,
        showlegend=True,
        legend={"font": {"size": 11}},
    )
    return fig
