# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Fingerprints view — behavioral fingerprint comparison and drift analysis.

Renders:
    * Dual run selector (baseline vs. candidate)
    * Single-run fingerprint heatmap
    * Side-by-side comparison heatmap
    * Drift summary statistics table
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from agentassay.dashboard.helpers import (
    empty_state,
    format_pass_rate,
)
from agentassay.persistence import QueryAPI
from agentassay.persistence.storage import ResultStore

# ── Color scale (green → yellow → orange → red) ───────────────────────

_HEATMAP_COLORSCALE: list[list[Any]] = [
    [0.0, "#d32f2f"],  # red     (< 0.5)
    [0.5, "#ff9800"],  # orange  (0.5)
    [0.7, "#fdd835"],  # yellow  (0.7)
    [0.9, "#66bb6a"],  # green   (0.9)
    [1.0, "#2e7d32"],  # dark green (1.0)
]


def render_fingerprints(query_api: QueryAPI, store: ResultStore) -> None:
    """Render the Fingerprints comparison page.

    Parameters
    ----------
    query_api : QueryAPI
        Read-only analytical query interface.
    store : ResultStore
        Direct store access for low-level reads.
    """
    st.header("Behavioral Fingerprints")

    runs = query_api.list_runs(limit=30)
    if not runs:
        empty_state("No test runs yet. Run <code>agentassay run</code> to get started.")
        return

    # Check if any run has fingerprint data
    has_fp = False
    for r in runs:
        fps = store.get_fingerprints(r["id"])
        if fps:
            has_fp = True
            break

    if not has_fp:
        empty_state(
            "No fingerprint data yet. Fingerprints are generated during "
            "test runs with behavioral fingerprinting enabled."
        )
        return

    # ── Run selectors ──────────────────────────────────────────────────

    run_labels = [
        f"{r['agent_name']} / {r['model']} — {_short_ts(r.get('started_at'))}" for r in runs
    ]

    c1, c2 = st.columns(2)
    with c1:
        baseline_idx = st.selectbox(
            "Baseline Run",
            range(len(run_labels)),
            format_func=lambda i: run_labels[i],
            key="fp_baseline",
        )
    with c2:
        candidate_idx = st.selectbox(
            "Candidate Run",
            range(len(run_labels)),
            index=min(1, len(run_labels) - 1),
            format_func=lambda i: run_labels[i],
            key="fp_candidate",
        )

    if baseline_idx is None or candidate_idx is None:
        return

    baseline_run = runs[baseline_idx]
    candidate_run = runs[candidate_idx]

    st.markdown("---")

    # ── Single-run heatmaps (side-by-side) ─────────────────────────────

    _render_dual_heatmaps(store, baseline_run, candidate_run)

    st.markdown("---")

    # ── Comparison heatmap ─────────────────────────────────────────────

    _render_comparison(query_api, baseline_run, candidate_run)

    st.markdown("---")

    # ── Drift summary table ────────────────────────────────────────────

    _render_drift_summary(query_api, baseline_run, candidate_run)


# ── Private section renderers ──────────────────────────────────────────


def _render_dual_heatmaps(
    store: ResultStore,
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
) -> None:
    """Side-by-side heatmaps for baseline and candidate fingerprints."""
    left, right = st.columns(2)

    with left:
        st.subheader("Baseline Fingerprint")
        fps = store.get_fingerprints(baseline_run["id"])
        if fps:
            fig = _build_heatmap(fps, f"Baseline: {baseline_run['agent_name']}")
            st.plotly_chart(fig, width="stretch")
        else:
            empty_state("No fingerprint data for the baseline run.")

    with right:
        st.subheader("Candidate Fingerprint")
        fps = store.get_fingerprints(candidate_run["id"])
        if fps:
            fig = _build_heatmap(fps, f"Candidate: {candidate_run['agent_name']}")
            st.plotly_chart(fig, width="stretch")
        else:
            empty_state("No fingerprint data for the candidate run.")


def _render_comparison(
    query_api: QueryAPI,
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
) -> None:
    """Difference heatmap: candidate minus baseline."""
    st.subheader("Fingerprint Comparison")

    comparison = query_api.get_fingerprint_comparison(baseline_run["id"], candidate_run["id"])

    baseline_fps = comparison.get("baseline", [])
    candidate_fps = comparison.get("candidate", [])

    if not baseline_fps or not candidate_fps:
        empty_state("Both runs must have fingerprint data for comparison.")
        return

    # Build scenario-indexed dictionaries
    base_by_scenario = _index_by_scenario(baseline_fps)
    cand_by_scenario = _index_by_scenario(candidate_fps)

    # Find common scenarios
    common = sorted(set(base_by_scenario.keys()) & set(cand_by_scenario.keys()))
    if not common:
        empty_state("No overlapping scenarios between the two runs.")
        return

    # Compute difference vectors
    diff_fps = _compute_diff_vectors(base_by_scenario, cand_by_scenario, common)

    if diff_fps:
        fig = _build_diff_heatmap(diff_fps, common)
        st.plotly_chart(fig, width="stretch")
    else:
        empty_state("Could not compute fingerprint differences.")


def _render_drift_summary(
    query_api: QueryAPI,
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
) -> None:
    """Summary statistics of behavioral drift between runs."""
    st.subheader("Drift Summary")

    comparison = query_api.get_fingerprint_comparison(baseline_run["id"], candidate_run["id"])

    baseline_fps = comparison.get("baseline", [])
    candidate_fps = comparison.get("candidate", [])

    if not baseline_fps or not candidate_fps:
        empty_state("Insufficient data for drift summary.")
        return

    base_by_scenario = _index_by_scenario(baseline_fps)
    cand_by_scenario = _index_by_scenario(candidate_fps)
    common = sorted(set(base_by_scenario.keys()) & set(cand_by_scenario.keys()))

    if not common:
        empty_state("No overlapping scenarios for drift analysis.")
        return

    rows: list[dict[str, str]] = []
    for scenario in common:
        base_vec = base_by_scenario[scenario]
        cand_vec = cand_by_scenario[scenario]

        # Compute L2 distance and cosine similarity
        l2 = _l2_distance(base_vec, cand_vec)
        cosine = _cosine_similarity(base_vec, cand_vec)
        max_delta = _max_abs_delta(base_vec, cand_vec)

        rows.append(
            {
                "Scenario": scenario,
                "Dimensions": str(len(base_vec)),
                "L2 Distance": f"{l2:.4f}",
                "Cosine Similarity": format_pass_rate(cosine),
                "Max Delta": f"{max_delta:.4f}",
                "Drift Signal": _drift_signal(l2),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


# ── Heatmap builders ───────────────────────────────────────────────────


def _build_heatmap(fingerprints: list[dict[str, Any]], title: str) -> Any:
    """Build a Plotly heatmap from fingerprint records.

    Rows = behavioral dimensions, Columns = scenarios.
    """
    import plotly.graph_objects as go

    from agentassay.dashboard.helpers import _get_theme_layout

    scenarios: list[str] = []
    matrix: list[list[float]] = []

    for fp in fingerprints:
        scenarios.append(fp.get("scenario_id", "unknown"))
        vector = _parse_vector(fp.get("vector_json", "[]"))
        matrix.append(vector)

    if not matrix:
        return go.Figure()

    # Transpose: rows = dimensions, cols = scenarios
    n_dims = max(len(v) for v in matrix)
    z_transposed: list[list[float]] = []
    dim_labels: list[str] = [f"dim_{i}" for i in range(n_dims)]

    for dim_idx in range(n_dims):
        row: list[float] = []
        for vec in matrix:
            val = vec[dim_idx] if dim_idx < len(vec) else 0.0
            row.append(val)
        z_transposed.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z_transposed,
            x=scenarios,
            y=dim_labels,
            colorscale=_HEATMAP_COLORSCALE,
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "Score"},
        )
    )
    fig.update_layout(
        **_get_theme_layout(),
        title={"text": title, "font": {"size": 14}},
        height=max(250, n_dims * 25 + 100),
        xaxis_title="Scenarios",
        yaxis_title="Dimensions",
    )
    return fig


def _build_diff_heatmap(
    diff_vectors: dict[str, list[float]],
    scenarios: list[str],
) -> Any:
    """Build a diverging heatmap for fingerprint differences."""
    import plotly.graph_objects as go

    from agentassay.dashboard.helpers import _get_theme_layout

    if not scenarios or not diff_vectors:
        return go.Figure()

    n_dims = max(len(v) for v in diff_vectors.values())
    dim_labels = [f"dim_{i}" for i in range(n_dims)]

    z: list[list[float]] = []
    for dim_idx in range(n_dims):
        row: list[float] = []
        for sc in scenarios:
            vec = diff_vectors.get(sc, [])
            val = vec[dim_idx] if dim_idx < len(vec) else 0.0
            row.append(val)
        z.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=scenarios,
            y=dim_labels,
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar={"title": "Delta"},
        )
    )
    fig.update_layout(
        **_get_theme_layout(),
        title={
            "text": "Fingerprint Difference (Candidate - Baseline)",
            "font": {"size": 14},
        },
        height=max(250, n_dims * 25 + 100),
        xaxis_title="Scenarios",
        yaxis_title="Dimensions",
    )
    return fig


# ── Vector math helpers ────────────────────────────────────────────────


def _parse_vector(json_str: str) -> list[float]:
    """Safely parse a JSON-encoded float vector."""
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _index_by_scenario(
    fps: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Map scenario_id -> parsed vector."""
    result: dict[str, list[float]] = {}
    for fp in fps:
        sid = fp.get("scenario_id", "unknown")
        result[sid] = _parse_vector(fp.get("vector_json", "[]"))
    return result


def _compute_diff_vectors(
    base: dict[str, list[float]],
    cand: dict[str, list[float]],
    common: list[str],
) -> dict[str, list[float]]:
    """Compute element-wise difference (candidate - baseline)."""
    result: dict[str, list[float]] = {}
    for sc in common:
        bv = base[sc]
        cv = cand[sc]
        n = max(len(bv), len(cv))
        diff: list[float] = []
        for i in range(n):
            b = bv[i] if i < len(bv) else 0.0
            c = cv[i] if i < len(cv) else 0.0
            diff.append(c - b)
        result[sc] = diff
    return result


def _l2_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    n = max(len(a), len(b))
    total = 0.0
    for i in range(n):
        va = a[i] if i < len(a) else 0.0
        vb = b[i] if i < len(b) else 0.0
        total += (va - vb) ** 2
    return total**0.5


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 on degenerate input."""
    n = max(len(a), len(b))
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for i in range(n):
        va = a[i] if i < len(a) else 0.0
        vb = b[i] if i < len(b) else 0.0
        dot += va * vb
        mag_a += va * va
        mag_b += vb * vb

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a**0.5 * mag_b**0.5)


def _max_abs_delta(a: list[float], b: list[float]) -> float:
    """Maximum absolute element-wise difference."""
    n = max(len(a), len(b))
    max_d = 0.0
    for i in range(n):
        va = a[i] if i < len(a) else 0.0
        vb = b[i] if i < len(b) else 0.0
        max_d = max(max_d, abs(va - vb))
    return max_d


def _drift_signal(l2: float) -> str:
    """Classify drift severity from L2 distance."""
    if l2 < 0.1:
        return "\u2705 Stable"
    if l2 < 0.3:
        return "\u26a0\ufe0f Minor Drift"
    if l2 < 0.6:
        return "\U0001f7e0 Moderate Drift"
    return "\U0001f534 Significant Drift"


# ── Utility ────────────────────────────────────────────────────────────


def _short_ts(ts: str | None) -> str:
    """Shorten an ISO timestamp to ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return "--"
    return ts[:16].replace("T", " ")
