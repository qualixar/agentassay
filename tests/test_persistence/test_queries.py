"""Tests for QueryAPI — analytical queries over AgentAssay results.

Validates list_runs filtering, pass_rate_trend, cost_trend, coverage_trend,
fingerprint_comparison, gate_history, run_summary, and pagination.

Target: 18+ tests covering all query methods plus edge cases.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agentassay.persistence.queries import QueryAPI
from agentassay.persistence.storage import ResultStore


# ===================================================================
# Helpers
# ===================================================================


def _make_store(tmp_path: Path) -> ResultStore:
    """Create a fresh ResultStore at the given path."""
    return ResultStore(tmp_path / "test.db")


def _save_run(
    store: ResultStore,
    agent_name: str = "agent-a",
    model: str = "gpt-4o",
    framework: str = "langgraph",
    status: str = "completed",
    total_cost: float = 1.0,
    started_at: str | None = None,
    project_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Insert a run with convenient defaults."""
    ts = started_at or datetime.now(timezone.utc).isoformat()
    return store.save_run(
        project_id=project_id,
        agent_name=agent_name,
        model=model,
        framework=framework,
        config_json=json.dumps({"num_trials": 30}),
        started_at=ts,
        status=status,
        total_cost=total_cost,
        run_id=run_id,
    )


def _save_verdict(
    store: ResultStore,
    run_id: str,
    scenario_id: str = "s1",
    pass_rate: float = 0.9,
    status: str = "PASS",
    n_trials: int = 30,
) -> str:
    """Insert a verdict with convenient defaults."""
    return store.save_verdict(
        run_id=run_id,
        scenario_id=scenario_id,
        status=status,
        pass_rate=pass_rate,
        ci_lower=pass_rate - 0.1,
        ci_upper=min(pass_rate + 0.1, 1.0),
        n_trials=n_trials,
        method="wilson",
    )


# ===================================================================
# list_runs
# ===================================================================


class TestListRuns:
    """Tests for QueryAPI.list_runs."""

    def test_list_all_runs(self, tmp_path: Path) -> None:
        """Returns all runs when no filter is applied."""
        store = _make_store(tmp_path)
        for i in range(3):
            _save_run(store, agent_name=f"agent-{i}")
        api = QueryAPI(store)
        runs = api.list_runs()
        assert len(runs) == 3

    def test_filter_by_agent_name(self, tmp_path: Path) -> None:
        """Only runs matching agent_name are returned."""
        store = _make_store(tmp_path)
        _save_run(store, agent_name="alpha")
        _save_run(store, agent_name="beta")
        _save_run(store, agent_name="alpha")
        api = QueryAPI(store)
        runs = api.list_runs(agent_name="alpha")
        assert len(runs) == 2
        assert all(r["agent_name"] == "alpha" for r in runs)

    def test_filter_by_model(self, tmp_path: Path) -> None:
        """Only runs matching model are returned."""
        store = _make_store(tmp_path)
        _save_run(store, model="gpt-4o")
        _save_run(store, model="claude-4")
        api = QueryAPI(store)
        runs = api.list_runs(model="claude-4")
        assert len(runs) == 1
        assert runs[0]["model"] == "claude-4"

    def test_filter_by_framework(self, tmp_path: Path) -> None:
        """Only runs matching framework are returned."""
        store = _make_store(tmp_path)
        _save_run(store, framework="langgraph")
        _save_run(store, framework="crewai")
        api = QueryAPI(store)
        runs = api.list_runs(framework="crewai")
        assert len(runs) == 1

    def test_filter_by_status(self, tmp_path: Path) -> None:
        """Only runs matching status are returned."""
        store = _make_store(tmp_path)
        _save_run(store, status="completed")
        _save_run(store, status="failed")
        api = QueryAPI(store)
        runs = api.list_runs(status="failed")
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"

    def test_filter_by_project_id(self, tmp_path: Path) -> None:
        """Only runs in the given project are returned."""
        store = _make_store(tmp_path)
        pid = store.save_project("P1")
        _save_run(store, project_id=pid)
        _save_run(store)  # No project
        api = QueryAPI(store)
        runs = api.list_runs(project_id=pid)
        assert len(runs) == 1

    def test_pagination_limit(self, tmp_path: Path) -> None:
        """limit restricts result count."""
        store = _make_store(tmp_path)
        for i in range(10):
            _save_run(store, agent_name=f"a-{i}")
        api = QueryAPI(store)
        runs = api.list_runs(limit=3)
        assert len(runs) == 3

    def test_pagination_offset(self, tmp_path: Path) -> None:
        """offset skips the first N results."""
        store = _make_store(tmp_path)
        for i in range(5):
            _save_run(store, agent_name=f"a-{i}")
        api = QueryAPI(store)
        all_runs = api.list_runs(limit=100)
        offset_runs = api.list_runs(limit=100, offset=2)
        assert len(offset_runs) == len(all_runs) - 2

    def test_combined_filters(self, tmp_path: Path) -> None:
        """Multiple filters are AND-ed together."""
        store = _make_store(tmp_path)
        _save_run(store, agent_name="a", model="gpt-4o", status="completed")
        _save_run(store, agent_name="a", model="gpt-4o", status="failed")
        _save_run(store, agent_name="b", model="gpt-4o", status="completed")
        api = QueryAPI(store)
        runs = api.list_runs(agent_name="a", status="completed")
        assert len(runs) == 1


# ===================================================================
# Pass rate trend
# ===================================================================


class TestPassRateTrend:
    """Tests for QueryAPI.get_pass_rate_trend."""

    def test_pass_rate_trend_basic(self, tmp_path: Path) -> None:
        """Returns aggregated pass rates grouped by date."""
        store = _make_store(tmp_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        rid = _save_run(store, started_at=now_iso)
        _save_verdict(store, rid, pass_rate=0.8)
        _save_verdict(store, rid, scenario_id="s2", pass_rate=0.6)

        api = QueryAPI(store)
        trend = api.get_pass_rate_trend(days=30)
        assert len(trend) >= 1
        assert trend[0]["pass_rate"] == pytest.approx(0.7)

    def test_pass_rate_trend_filtered_by_agent(self, tmp_path: Path) -> None:
        """Trend only includes runs for the given agent."""
        store = _make_store(tmp_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        r1 = _save_run(store, agent_name="alpha", started_at=now_iso)
        r2 = _save_run(store, agent_name="beta", started_at=now_iso)
        _save_verdict(store, r1, pass_rate=1.0)
        _save_verdict(store, r2, pass_rate=0.0)

        api = QueryAPI(store)
        trend = api.get_pass_rate_trend(agent_name="alpha", days=30)
        assert len(trend) >= 1
        assert trend[0]["pass_rate"] == pytest.approx(1.0)


# ===================================================================
# Cost trend
# ===================================================================


class TestCostTrend:
    """Tests for QueryAPI.get_cost_trend."""

    def test_cost_trend_basic(self, tmp_path: Path) -> None:
        """Returns aggregated costs grouped by date."""
        store = _make_store(tmp_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        _save_run(store, total_cost=2.5, started_at=now_iso)
        _save_run(store, total_cost=1.5, started_at=now_iso)

        api = QueryAPI(store)
        trend = api.get_cost_trend(days=30)
        assert len(trend) >= 1
        assert trend[0]["total_cost"] == pytest.approx(4.0)
        assert trend[0]["run_count"] == 2


# ===================================================================
# Coverage trend
# ===================================================================


class TestCoverageTrend:
    """Tests for QueryAPI.get_coverage_trend."""

    def test_coverage_trend_by_run_ids(self, tmp_path: Path) -> None:
        """Trend for specific run IDs groups by dimension."""
        store = _make_store(tmp_path)
        r1 = _save_run(store)
        store.save_coverage(run_id=r1, dimension="tool", score=0.8)
        store.save_coverage(run_id=r1, dimension="path", score=0.6)

        api = QueryAPI(store)
        trend = api.get_coverage_trend(run_ids=[r1])
        dims = {row["dimension"] for row in trend}
        assert dims == {"tool", "path"}

    def test_coverage_trend_by_days(self, tmp_path: Path) -> None:
        """Date-based look-back returns coverage dimensions."""
        store = _make_store(tmp_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        r1 = _save_run(store, started_at=now_iso)
        store.save_coverage(run_id=r1, dimension="state", score=0.9)

        api = QueryAPI(store)
        trend = api.get_coverage_trend(days=30)
        assert len(trend) >= 1
        assert trend[0]["dimension"] == "state"


# ===================================================================
# Fingerprint comparison
# ===================================================================


class TestFingerprintComparison:
    """Tests for QueryAPI.get_fingerprint_comparison."""

    def test_fingerprint_comparison(self, tmp_path: Path) -> None:
        """Returns fingerprints for baseline and candidate side-by-side."""
        store = _make_store(tmp_path)
        r1 = _save_run(store, run_id="run-base")
        r2 = _save_run(store, run_id="run-cand")
        store.save_fingerprint(
            run_id=r1, scenario_id="s1",
            vector_json=json.dumps([0.1, 0.2]),
        )
        store.save_fingerprint(
            run_id=r2, scenario_id="s1",
            vector_json=json.dumps([0.3, 0.4]),
        )

        api = QueryAPI(store)
        result = api.get_fingerprint_comparison("run-base", "run-cand")
        assert result["baseline_run_id"] == "run-base"
        assert result["candidate_run_id"] == "run-cand"
        assert len(result["baseline"]) == 1
        assert len(result["candidate"]) == 1

    def test_fingerprint_comparison_empty(self, tmp_path: Path) -> None:
        """Returns empty lists when no fingerprints exist."""
        store = _make_store(tmp_path)
        api = QueryAPI(store)
        result = api.get_fingerprint_comparison("none-1", "none-2")
        assert result["baseline"] == []
        assert result["candidate"] == []


# ===================================================================
# Gate history
# ===================================================================


class TestGateHistory:
    """Tests for QueryAPI.get_gate_history."""

    def test_gate_history_basic(self, tmp_path: Path) -> None:
        """Returns recent gate decisions."""
        store = _make_store(tmp_path)
        rid = _save_run(store)
        store.save_gate_decision(
            run_id=rid, pipeline="main", decision="DEPLOY",
            rules_json=json.dumps({"min_pass": 0.8}),
        )
        api = QueryAPI(store)
        history = api.get_gate_history(days=30)
        assert len(history) >= 1
        assert history[0]["decision"] == "DEPLOY"

    def test_gate_history_filtered_by_pipeline(self, tmp_path: Path) -> None:
        """Only decisions for the given pipeline are returned."""
        store = _make_store(tmp_path)
        rid = _save_run(store)
        store.save_gate_decision(
            run_id=rid, pipeline="main", decision="DEPLOY",
            rules_json="{}",
        )
        store.save_gate_decision(
            run_id=rid, pipeline="staging", decision="BLOCK",
            rules_json="{}",
        )
        api = QueryAPI(store)
        history = api.get_gate_history(pipeline="staging", days=30)
        assert len(history) == 1
        assert history[0]["decision"] == "BLOCK"


# ===================================================================
# Run summary
# ===================================================================


class TestRunSummary:
    """Tests for QueryAPI.get_run_summary."""

    def test_run_summary_empty_db(self, tmp_path: Path) -> None:
        """Summary on an empty database returns zeroed values."""
        store = _make_store(tmp_path)
        api = QueryAPI(store)
        summary = api.get_run_summary()
        assert summary["total_runs"] == 0
        assert summary["total_cost"] == pytest.approx(0.0)
        assert summary["avg_pass_rate"] is None

    def test_run_summary_with_data(self, tmp_path: Path) -> None:
        """Summary aggregates across all runs and verdicts."""
        store = _make_store(tmp_path)
        r1 = _save_run(store, total_cost=2.0)
        r2 = _save_run(store, total_cost=3.0)
        _save_verdict(store, r1, pass_rate=0.8)
        _save_verdict(store, r2, pass_rate=1.0)

        api = QueryAPI(store)
        summary = api.get_run_summary()
        assert summary["total_runs"] == 2
        assert summary["total_cost"] == pytest.approx(5.0)
        assert summary["avg_pass_rate"] == pytest.approx(0.9)
        assert summary["last_run_date"] is not None
