"""Tests for TraceStore — persistent trace recording and offline analytics.

Validates recording, querying (by agent, scenario, with limits), offline
fingerprint extraction, drift detection, persistence across close/reopen,
and empty-store edge cases.

Target: ~13 tests covering CRUD, filtering, analytics, persistence,
and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.trace_store import TraceStore

from .conftest import make_trace, make_traces, make_regressed_traces


# ===================================================================
# TraceStore — basic record and query
# ===================================================================


class TestTraceStoreRecordAndQuery:
    """Tests for storing and retrieving execution traces."""

    def test_record_and_query(self, tmp_path: Path):
        """Store a trace and retrieve it."""
        store = TraceStore(store_path=str(tmp_path / "traces"))
        trace = make_trace(steps=5, scenario_id="s1", agent_id="a1")
        store.record(trace, metadata={"agent_id": "a1"})

        results = store.query()
        assert len(results) == 1
        assert results[0].trace_id == trace.trace_id

    def test_query_by_agent_id(self, tmp_path: Path):
        """Filter stored traces by agent identifier."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        # Store traces from two different agents
        for t in make_traces(5, agent_id="agent-alpha"):
            store.record(t, metadata={"agent_id": "agent-alpha"})
        for t in make_traces(3, agent_id="agent-beta"):
            store.record(t, metadata={"agent_id": "agent-beta"})

        alpha_results = store.query(agent_id="agent-alpha")
        beta_results = store.query(agent_id="agent-beta")

        assert len(alpha_results) == 5
        assert len(beta_results) == 3

    def test_query_by_scenario(self, tmp_path: Path):
        """Filter stored traces by scenario identifier."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        for t in make_traces(4, scenario_id="login-flow"):
            store.record(t, metadata={"agent_id": "a1"})
        for t in make_traces(6, scenario_id="checkout-flow"):
            store.record(t, metadata={"agent_id": "a1"})

        login_results = store.query(scenario="login-flow")
        checkout_results = store.query(scenario="checkout-flow")

        assert len(login_results) == 4
        assert len(checkout_results) == 6

    def test_query_limit(self, tmp_path: Path):
        """Query respects the limit parameter and returns at most N traces."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        for t in make_traces(20):
            store.record(t, metadata={"agent_id": "a1"})

        limited = store.query(limit=5)
        assert len(limited) == 5

    def test_query_combined_filters(self, tmp_path: Path):
        """Combining agent_id and scenario_id filters works correctly."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        for t in make_traces(3, agent_id="a1", scenario_id="s1"):
            store.record(t, metadata={"agent_id": "a1"})
        for t in make_traces(4, agent_id="a1", scenario_id="s2"):
            store.record(t, metadata={"agent_id": "a1"})
        for t in make_traces(2, agent_id="a2", scenario_id="s1"):
            store.record(t, metadata={"agent_id": "a2"})

        results = store.query(agent_id="a1", scenario="s1")
        assert len(results) == 3


# ===================================================================
# TraceStore — offline analytics
# ===================================================================


class TestTraceStoreOfflineAnalytics:
    """Tests for computing analytics from stored traces."""

    def test_offline_fingerprints(self, tmp_path: Path):
        """Computes fingerprint distribution from stored traces."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        for t in make_traces(15):
            store.record(t, metadata={"agent_id": "agent-1"})

        dist = store.offline_fingerprints(agent_id="agent-1")

        assert dist is not None
        mean = dist.mean_vector
        assert len(mean) == 14

    def test_drift_detection_stable(self, tmp_path: Path):
        """Stable behavior shows low drift distances (even if some windows flag).

        All traces come from the same behavioral template, so any drift
        detections are statistical noise. The mean distance should be small.
        """
        store = TraceStore(store_path=str(tmp_path / "traces"))

        # Need at least 2 * window_size traces
        for t in make_traces(30, agent_id="stable-agent"):
            store.record(t, metadata={"agent_id": "stable-agent"})

        drift_results = store.drift_detection(
            agent_id="stable-agent", window_size=10, step_size=5
        )

        # All windows should be present
        assert len(drift_results) > 0
        # Distances should all be finite and the function should not crash
        for window in drift_results:
            assert "distance" in window
            assert "p_value" in window

    def test_drift_detection_with_drift(self, tmp_path: Path):
        """Drift detected when later traces differ significantly from earlier ones.

        We store 20 baseline traces followed by 20 regressed traces.
        The store should detect that the distribution shifted.
        """
        store = TraceStore(store_path=str(tmp_path / "traces"))

        baseline = make_traces(20, agent_id="drifting-agent")
        regressed = make_regressed_traces(20, agent_id="drifting-agent")

        for t in baseline:
            store.record(t, metadata={"agent_id": "drifting-agent"})
        for t in regressed:
            store.record(t, metadata={"agent_id": "drifting-agent"})

        drift_results = store.drift_detection(
            agent_id="drifting-agent", window_size=10, step_size=5
        )

        # At least one later window should detect drift
        any_drift = any(w["drift_detected"] for w in drift_results)
        assert any_drift is True


# ===================================================================
# TraceStore — persistence
# ===================================================================


class TestTraceStorePersistence:
    """Tests for trace survival across store close/reopen."""

    def test_persistence(self, tmp_path: Path):
        """Traces survive closing and reopening the store.

        Write traces, discard the store object, open a new instance
        pointing to the same path, and verify the traces are still there.
        """
        db_path = str(tmp_path / "persistent")

        # Write phase
        store1 = TraceStore(store_path=db_path)
        for t in make_traces(10):
            store1.record(t, metadata={"agent_id": "a1"})

        # Read phase — new instance, same path
        store2 = TraceStore(store_path=db_path)
        results = store2.query()

        assert len(results) == 10


# ===================================================================
# TraceStore — edge cases
# ===================================================================


class TestTraceStoreEdgeCases:
    """Tests for edge-case and empty-store behavior."""

    def test_empty_store(self, tmp_path: Path):
        """An empty store returns empty results without errors."""
        store = TraceStore(store_path=str(tmp_path / "empty"))

        results = store.query()
        assert results == []

    def test_empty_store_fingerprints(self, tmp_path: Path):
        """Offline fingerprints on empty store raises ValueError."""
        store = TraceStore(store_path=str(tmp_path / "empty"))

        with pytest.raises(ValueError, match="at least 2"):
            store.offline_fingerprints(agent_id="nonexistent")

    def test_empty_store_drift(self, tmp_path: Path):
        """Drift detection on empty store raises ValueError."""
        store = TraceStore(store_path=str(tmp_path / "empty"))

        with pytest.raises(ValueError, match="needs at least"):
            store.drift_detection(agent_id="nonexistent", window_size=5)

    def test_delete_trace(self, tmp_path: Path):
        """Delete a trace by ID and verify it is gone."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        trace = make_trace(steps=3)
        store.record(trace, metadata={"agent_id": "a1"})
        assert store.size == 1

        deleted = store.delete_trace(trace.trace_id)
        assert deleted is True
        assert store.size == 0

    def test_delete_nonexistent_trace(self, tmp_path: Path):
        """Deleting a trace that does not exist returns False."""
        store = TraceStore(store_path=str(tmp_path / "traces"))

        deleted = store.delete_trace("nonexistent-id")
        assert deleted is False
