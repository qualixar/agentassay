"""AgentAssay Dashboard — Streamlit entry point.

Launch::

    streamlit run src/agentassay/dashboard/app.py

Environment variables::

    AGENTASSAY_DB_PATH  Path to the SQLite results database.
                        Defaults to ``~/.agentassay/results.db``.
"""

from __future__ import annotations

import os

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────

st.set_page_config(
    page_title="AgentAssay Dashboard",
    page_icon="\U0001f9ea",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme-aware CSS ───────────────────────────────────────────────────

_DARK_CSS = """
<style>
    [data-testid="stSidebar"] { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="stDataFrame"] thead th { background-color: rgba(255,255,255,0.05); }
</style>
"""

_LIGHT_CSS = """
<style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    [data-testid="stMetric"] {
        background-color: rgba(0,0,0,0.02);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="stDataFrame"] thead th { background-color: rgba(0,0,0,0.04); }
</style>
"""

_current_theme = st.get_option("theme.base") or "dark"
st.markdown(_LIGHT_CSS if _current_theme == "light" else _DARK_CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AgentAssay")
    st.caption("Test More. Spend Less. Ship Confident.")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Test Run", "History", "Fingerprints"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    from agentassay import __version__

    st.caption(f"v{__version__}")

# ── Persistence layer ──────────────────────────────────────────────────

from agentassay.persistence import QueryAPI, ResultStore

db_path = os.environ.get(
    "AGENTASSAY_DB_PATH",
    os.path.expanduser("~/.agentassay/results.db"),
)
store = ResultStore(db_path=db_path)
query_api = QueryAPI(store)

# ── View routing ───────────────────────────────────────────────────────

if page == "Overview":
    from agentassay.dashboard.view_overview import render_overview

    render_overview(query_api, store)

elif page == "Test Run":
    from agentassay.dashboard.view_test_run import render_test_run

    render_test_run(query_api, store)

elif page == "History":
    from agentassay.dashboard.view_history import render_history

    render_history(query_api, store)

elif page == "Fingerprints":
    from agentassay.dashboard.view_fingerprints import render_fingerprints

    render_fingerprints(query_api, store)
