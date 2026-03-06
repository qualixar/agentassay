"""Tests for ResultStore — SQLite-backed persistence engine.

Validates schema creation, CRUD for all 8 tables, update semantics,
retrieval, context manager support, thread safety, and edge cases.

Target: 25+ tests across schema, writes, reads, updates, and edge cases.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest

from agentassay.persistence.storage import ResultStore


# ===================================================================
# Helpers
# ===================================================================


def _make_run_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build keyword arguments for ``save_run`` with sensible defaults."""
    defaults: dict[str, Any] = {
        "agent_name": "test-agent",
        "agent_version": "1.0.0",
        "model": "gpt-4o",
        "framework": "langgraph",
        "config_json": json.dumps({"num_trials": 30}),
        "status": "running",
    }
    defaults.update(overrides)
    return defaults


def _make_trial_kwargs(run_id: str, **overrides: Any) -> dict[str, Any]:
    """Build keyword arguments for ``save_trial`` with sensible defaults."""
    defaults: dict[str, Any] = {
        "run_id": run_id,
        "scenario_id": "scenario-1",
        "trial_num": 1,
        "success": True,
        "latency_ms": 123.4,
        "cost": 0.02,
        "token_count": 500,
        "step_count": 3,
        "trace_json": json.dumps({"steps": []}),
    }
    defaults.update(overrides)
    return defaults


# ===================================================================
# Schema creation
# ===================================================================


class TestSchemaCreation:
    """Verify that the database schema is correctly initialized."""

    def test_creates_all_tables(self, tmp_path: Path) -> None:
        """All 8 expected tables exist after initialization."""
        store = ResultStore(tmp_path / "test.db")
        expected_tables = [
            "projects", "runs", "trials", "verdicts",
            "coverage", "fingerprints", "gate_decisions", "costs",
        ]
        for table in expected_tables:
            assert store.table_exists(table), f"Table '{table}' not found"

    def test_creates_indexes(self, tmp_path: Path) -> None:
        """All expected indexes exist after initialization."""
        store = ResultStore(tmp_path / "test.db")
        conn = store._connect()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name LIKE 'idx_%'"
            ).fetchall()
            index_names = {r["name"] for r in rows}
        finally:
            conn.close()

        expected = {
            "idx_runs_project", "idx_trials_run", "idx_verdicts_run",
            "idx_coverage_run", "idx_fingerprints_run",
            "idx_gate_decisions_pipeline", "idx_costs_run",
        }
        assert expected.issubset(index_names)

    def test_idempotent_schema_init(self, tmp_path: Path) -> None:
        """Re-creating a store on the same DB does not raise."""
        db = tmp_path / "test.db"
        ResultStore(db)
        ResultStore(db)  # Should not raise.

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Nested directories are created automatically."""
        deep = tmp_path / "a" / "b" / "c" / "test.db"
        store = ResultStore(deep)
        assert Path(store.db_path).exists()


# ===================================================================
# Projects CRUD
# ===================================================================


class TestProjectsCrud:
    """Tests for the projects table."""

    def test_save_project(self, tmp_path: Path) -> None:
        """Insert a project and verify it exists."""
        store = ResultStore(tmp_path / "test.db")
        pid = store.save_project("My Project")
        assert pid is not None
        assert len(pid) == 36  # UUID4 format

    def test_save_project_explicit_id(self, tmp_path: Path) -> None:
        """Explicit project ID is respected."""
        store = ResultStore(tmp_path / "test.db")
        pid = store.save_project("P1", project_id="proj-001")
        assert pid == "proj-001"


# ===================================================================
# Runs CRUD
# ===================================================================


class TestRunsCrud:
    """Tests for the runs table."""

    def test_save_run_returns_id(self, tmp_path: Path) -> None:
        """save_run returns a valid UUID."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        assert len(rid) == 36

    def test_save_run_explicit_id(self, tmp_path: Path) -> None:
        """Explicit run_id is respected."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs(run_id="run-001"))
        assert rid == "run-001"

    def test_get_run(self, tmp_path: Path) -> None:
        """Retrieve a saved run by its ID."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs(agent_name="alpha"))
        run = store.get_run(rid)
        assert run is not None
        assert run["agent_name"] == "alpha"
        assert run["model"] == "gpt-4o"
        assert run["status"] == "running"

    def test_get_run_not_found(self, tmp_path: Path) -> None:
        """Querying a non-existent run returns None."""
        store = ResultStore(tmp_path / "test.db")
        assert store.get_run("nonexistent") is None

    def test_save_run_with_project(self, tmp_path: Path) -> None:
        """Run can reference a project via foreign key."""
        store = ResultStore(tmp_path / "test.db")
        pid = store.save_project("Proj")
        rid = store.save_run(**_make_run_kwargs(project_id=pid))
        run = store.get_run(rid)
        assert run is not None
        assert run["project_id"] == pid


# ===================================================================
# Trials CRUD
# ===================================================================


class TestTrialsCrud:
    """Tests for the trials table."""

    def test_save_and_get_trials(self, tmp_path: Path) -> None:
        """Save multiple trials and retrieve them by run_id."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())

        for i in range(5):
            store.save_trial(**_make_trial_kwargs(rid, trial_num=i + 1))

        trials = store.get_trials(rid)
        assert len(trials) == 5
        assert trials[0]["trial_num"] == 1
        assert trials[4]["trial_num"] == 5

    def test_trial_fields_stored(self, tmp_path: Path) -> None:
        """All trial fields are persisted and retrievable."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        tid = store.save_trial(**_make_trial_kwargs(
            rid,
            success=False,
            latency_ms=999.9,
            cost=0.05,
            token_count=1200,
            step_count=7,
            error_msg="timeout",
        ))

        trials = store.get_trials(rid)
        assert len(trials) == 1
        t = trials[0]
        assert t["id"] == tid
        assert t["success"] == 0  # SQLite stores bool as int
        assert t["latency_ms"] == pytest.approx(999.9)
        assert t["cost"] == pytest.approx(0.05)
        assert t["token_count"] == 1200
        assert t["step_count"] == 7
        assert t["error_msg"] == "timeout"

    def test_get_trials_empty(self, tmp_path: Path) -> None:
        """Querying trials for a run with none returns empty list."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        assert store.get_trials(rid) == []


# ===================================================================
# Verdicts CRUD
# ===================================================================


class TestVerdictsCrud:
    """Tests for the verdicts table."""

    def test_save_and_get_verdicts(self, tmp_path: Path) -> None:
        """Save a verdict and retrieve it."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        vid = store.save_verdict(
            run_id=rid,
            scenario_id="s1",
            status="PASS",
            pass_rate=0.9,
            ci_lower=0.75,
            ci_upper=0.98,
            p_value=0.03,
            effect_size=0.12,
            n_trials=30,
            method="fisher",
        )
        verdicts = store.get_verdicts(rid)
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v["id"] == vid
        assert v["status"] == "PASS"
        assert v["pass_rate"] == pytest.approx(0.9)


# ===================================================================
# Coverage CRUD
# ===================================================================


class TestCoverageCrud:
    """Tests for the coverage table."""

    def test_save_and_get_coverage(self, tmp_path: Path) -> None:
        """Save coverage dimensions and retrieve them."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        store.save_coverage(
            run_id=rid, dimension="tool", score=0.85,
            details_json=json.dumps({"tools_covered": 5}),
        )
        store.save_coverage(
            run_id=rid, dimension="path", score=0.70,
        )
        records = store.get_coverage(rid)
        assert len(records) == 2
        dims = {r["dimension"] for r in records}
        assert dims == {"tool", "path"}


# ===================================================================
# Fingerprints CRUD
# ===================================================================


class TestFingerprintsCrud:
    """Tests for the fingerprints table."""

    def test_save_and_get_fingerprints(self, tmp_path: Path) -> None:
        """Save fingerprints and retrieve them."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        vec = [0.1, 0.5, 0.9, 0.3]
        fid = store.save_fingerprint(
            run_id=rid,
            scenario_id="s1",
            vector_json=json.dumps(vec),
        )
        fps = store.get_fingerprints(rid)
        assert len(fps) == 1
        assert fps[0]["id"] == fid
        assert json.loads(fps[0]["vector_json"]) == vec


# ===================================================================
# Gate decisions CRUD
# ===================================================================


class TestGateDecisionsCrud:
    """Tests for the gate_decisions table."""

    def test_save_and_get_gate_decisions(self, tmp_path: Path) -> None:
        """Save a gate decision and retrieve it."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        gid = store.save_gate_decision(
            run_id=rid,
            pipeline="main",
            decision="DEPLOY",
            reason="All pass",
            rules_json=json.dumps({"min_pass_rate": 0.8}),
            commit_sha="abc123",
            pr_number=42,
        )
        gates = store.get_gate_decisions(rid)
        assert len(gates) == 1
        g = gates[0]
        assert g["id"] == gid
        assert g["decision"] == "DEPLOY"
        assert g["pr_number"] == 42


# ===================================================================
# Costs CRUD
# ===================================================================


class TestCostsCrud:
    """Tests for the costs table."""

    def test_save_and_get_costs(self, tmp_path: Path) -> None:
        """Save cost records and retrieve them."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        store.save_cost(
            run_id=rid, model="gpt-4o",
            input_tokens=5000, output_tokens=1000,
            total_cost=0.08, trial_count=10,
        )
        costs = store.get_costs(rid)
        assert len(costs) == 1
        assert costs[0]["total_cost"] == pytest.approx(0.08)


# ===================================================================
# Update operations
# ===================================================================


class TestUpdateOperations:
    """Tests for update_run_status."""

    def test_update_run_status(self, tmp_path: Path) -> None:
        """Update a run's status and cost."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())

        store.update_run_status(rid, "completed", total_cost=1.23)
        run = store.get_run(rid)
        assert run is not None
        assert run["status"] == "completed"
        assert run["completed_at"] is not None
        assert run["total_cost"] == pytest.approx(1.23)

    def test_update_run_status_without_cost(self, tmp_path: Path) -> None:
        """Update status only — cost stays at original value."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs(total_cost=0.5))

        store.update_run_status(rid, "failed")
        run = store.get_run(rid)
        assert run is not None
        assert run["status"] == "failed"
        assert run["total_cost"] == pytest.approx(0.5)


# ===================================================================
# Context manager
# ===================================================================


class TestContextManager:
    """Tests for with-block support."""

    def test_context_manager_basic(self, tmp_path: Path) -> None:
        """Store works inside a with block."""
        with ResultStore(tmp_path / "test.db") as store:
            rid = store.save_run(**_make_run_kwargs())
            assert store.get_run(rid) is not None

    def test_context_manager_exit_no_error(self, tmp_path: Path) -> None:
        """Exiting the context manager does not raise."""
        store = ResultStore(tmp_path / "test.db")
        store.__enter__()
        store.__exit__(None, None, None)  # Should not raise.


# ===================================================================
# Thread safety
# ===================================================================


class TestThreadSafety:
    """Basic thread-safety checks."""

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        """Multiple threads writing trials concurrently do not corrupt data."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        num_threads = 10
        trials_per_thread = 5
        errors: list[Exception] = []

        def _write_trials(thread_idx: int) -> None:
            try:
                for j in range(trials_per_thread):
                    store.save_trial(**_make_trial_kwargs(
                        rid,
                        trial_num=thread_idx * 100 + j,
                        scenario_id=f"scenario-{thread_idx}",
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_write_trials, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        trials = store.get_trials(rid)
        assert len(trials) == num_threads * trials_per_thread


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_duplicate_run_id_raises(self, tmp_path: Path) -> None:
        """Inserting a duplicate primary key raises IntegrityError."""
        store = ResultStore(tmp_path / "test.db")
        store.save_run(**_make_run_kwargs(run_id="dup-1"))
        with pytest.raises(sqlite3.IntegrityError):
            store.save_run(**_make_run_kwargs(run_id="dup-1"))

    def test_duplicate_trial_id_raises(self, tmp_path: Path) -> None:
        """Inserting a duplicate trial PK raises IntegrityError."""
        store = ResultStore(tmp_path / "test.db")
        rid = store.save_run(**_make_run_kwargs())
        store.save_trial(**_make_trial_kwargs(rid, trial_id="t-dup"))
        with pytest.raises(sqlite3.IntegrityError):
            store.save_trial(**_make_trial_kwargs(rid, trial_id="t-dup", trial_num=2))

    def test_count_known_table(self, tmp_path: Path) -> None:
        """count() returns accurate row count."""
        store = ResultStore(tmp_path / "test.db")
        assert store.count("runs") == 0
        store.save_run(**_make_run_kwargs())
        assert store.count("runs") == 1

    def test_count_unknown_table_raises(self, tmp_path: Path) -> None:
        """count() rejects unknown table names to prevent injection."""
        store = ResultStore(tmp_path / "test.db")
        with pytest.raises(ValueError, match="Unknown table"):
            store.count("bobby_tables")

    def test_db_path_property(self, tmp_path: Path) -> None:
        """db_path returns the resolved path string."""
        store = ResultStore(tmp_path / "test.db")
        assert store.db_path.endswith("test.db")
        assert Path(store.db_path).is_absolute()
