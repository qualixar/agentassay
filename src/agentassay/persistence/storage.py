"""SQLite storage engine for AgentAssay test results.

Implements a normalized 8-table schema (projects, runs, trials, verdicts,
coverage, fingerprints, gate_decisions, costs) with proper indexes for
dashboard query patterns.  Uses the ``sqlite3`` stdlib module exclusively --
zero external dependencies.

Thread safety
-------------
Each public method acquires its own connection via ``_connect()``, which
sets ``check_same_thread=False``.  This allows the store to be shared
across threads (e.g. parallel trial runners writing concurrently) without
external locking.  SQLite's internal WAL mode handles the concurrency.

Context manager
---------------
``ResultStore`` supports ``with`` blocks for deterministic cleanup::

    with ResultStore("/tmp/test.db") as store:
        store.save_run(...)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agentassay.persistence.readers import ResultStoreReaderMixin
from agentassay.persistence.schema import (
    INDEX_SQL,
    SCHEMA_SQL,
    _now_iso,
    _uuid,
)


# ===================================================================
# ResultStore
# ===================================================================


class ResultStore(ResultStoreReaderMixin):
    """SQLite-backed storage for AgentAssay test results.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.  Parent directories are created
        automatically.  Defaults to ``~/.agentassay/results.db``.

    Examples
    --------
    >>> store = ResultStore("/tmp/test.db")
    >>> run_id = store.save_run(
    ...     project_id="p1", agent_name="search-agent",
    ...     model="gpt-4o", framework="langgraph",
    ...     config_json="{}", started_at="2026-02-28T00:00:00Z",
    ...     status="running",
    ... )
    >>> store.get_run(run_id)
    {'id': '...', 'agent_name': 'search-agent', ...}
    """

    def __init__(self, db_path: str | Path = "~/.agentassay/results.db") -> None:
        resolved = Path(db_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(resolved)
        self._init_schema()

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        # Nothing to close -- connections are per-call.
        pass

    # -- Internal -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with recommended pragmas.

        Returns
        -------
        sqlite3.Connection
            A fresh connection configured for WAL mode and foreign keys.
        """
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.executescript(INDEX_SQL)
            conn.commit()
        finally:
            conn.close()

    @property
    def db_path(self) -> str:
        """Return the resolved database file path."""
        return self._db_path

    # ===================================================================
    # Write operations
    # ===================================================================

    def save_project(
        self,
        name: str,
        project_id: str | None = None,
    ) -> str:
        """Insert a new project.

        Parameters
        ----------
        name : str
            Human-readable project name.
        project_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The project's unique identifier.
        """
        pid = project_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO projects (id, name) VALUES (?, ?)",
                (pid, name),
            )
            conn.commit()
        finally:
            conn.close()
        return pid

    def save_run(
        self,
        *,
        project_id: str | None = None,
        agent_name: str,
        agent_version: str | None = None,
        model: str,
        framework: str,
        config_json: str,
        started_at: str | None = None,
        status: str = "running",
        total_trials: int | None = None,
        total_cost: float = 0.0,
        run_id: str | None = None,
    ) -> str:
        """Insert a new test run.

        Parameters
        ----------
        project_id : str | None
            FK to projects table. May be None for ad-hoc runs.
        agent_name : str
            Name of the agent under test.
        agent_version : str | None
            Semantic version of the agent.
        model : str
            LLM model identifier (e.g. ``"gpt-4o"``).
        framework : str
            Agent framework (e.g. ``"langgraph"``).
        config_json : str
            JSON-encoded AssayConfig.
        started_at : str | None
            ISO-8601 timestamp.  Defaults to now (UTC).
        status : str
            Run status (``"running"``, ``"completed"``, ``"failed"``).
        total_trials : int | None
            Expected trial count.
        total_cost : float
            Cumulative cost so far.
        run_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The run's unique identifier.
        """
        rid = run_id or _uuid()
        ts = started_at or _now_iso()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO runs
                    (id, project_id, agent_name, agent_version, model,
                     framework, config_json, started_at, status,
                     total_trials, total_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid, project_id, agent_name, agent_version, model,
                    framework, config_json, ts, status,
                    total_trials, total_cost,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return rid

    def save_trial(
        self,
        *,
        run_id: str,
        scenario_id: str,
        trial_num: int,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
        token_count: int = 0,
        step_count: int,
        error_msg: str | None = None,
        trace_json: str,
        trial_id: str | None = None,
    ) -> str:
        """Insert a single trial result.

        Parameters
        ----------
        run_id : str
            FK to the parent run.
        scenario_id : str
            Which scenario was executed.
        trial_num : int
            1-based trial ordinal within the run.
        success : bool
            Whether the trial passed the oracle.
        latency_ms : float
            Wall-clock duration in milliseconds.
        cost : float
            Token cost in USD for this trial.
        token_count : int
            Total tokens consumed.
        step_count : int
            Number of steps in the execution trace.
        error_msg : str | None
            Error text if the trial failed.
        trace_json : str
            JSON-serialized execution trace.
        trial_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The trial's unique identifier.
        """
        tid = trial_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trials
                    (id, run_id, scenario_id, trial_num, success,
                     latency_ms, cost, token_count, step_count,
                     error_msg, trace_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid, run_id, scenario_id, trial_num, success,
                    latency_ms, cost, token_count, step_count,
                    error_msg, trace_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return tid

    def save_verdict(
        self,
        *,
        run_id: str,
        scenario_id: str,
        status: str,
        pass_rate: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float | None = None,
        effect_size: float | None = None,
        n_trials: int,
        method: str,
        verdict_id: str | None = None,
    ) -> str:
        """Insert a statistical verdict.

        Parameters
        ----------
        run_id : str
            FK to the parent run.
        scenario_id : str
            Which scenario was evaluated.
        status : str
            Verdict status (``"PASS"``, ``"FAIL"``, ``"INCONCLUSIVE"``).
        pass_rate : float
            Observed pass rate in [0, 1].
        ci_lower : float
            Lower bound of the confidence interval.
        ci_upper : float
            Upper bound of the confidence interval.
        p_value : float | None
            p-value from the regression test.
        effect_size : float | None
            Cohen's h effect size.
        n_trials : int
            Number of trials in the evaluation.
        method : str
            Statistical method used (e.g. ``"fisher"``, ``"wilson"``).
        verdict_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The verdict's unique identifier.
        """
        vid = verdict_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO verdicts
                    (id, run_id, scenario_id, status, pass_rate,
                     ci_lower, ci_upper, p_value, effect_size,
                     n_trials, method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vid, run_id, scenario_id, status, pass_rate,
                    ci_lower, ci_upper, p_value, effect_size,
                    n_trials, method,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return vid

    def save_coverage(
        self,
        *,
        run_id: str,
        dimension: str,
        score: float,
        details_json: str | None = None,
        coverage_id: str | None = None,
    ) -> str:
        """Insert a coverage measurement.

        Parameters
        ----------
        run_id : str
            FK to the parent run.
        dimension : str
            Coverage dimension (``"tool"``, ``"path"``, ``"state"``,
            ``"boundary"``, ``"model"``).
        score : float
            Coverage score in [0, 1].
        details_json : str | None
            JSON-encoded detail map.
        coverage_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The coverage record's unique identifier.
        """
        cid = coverage_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO coverage
                    (id, run_id, dimension, score, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, run_id, dimension, score, details_json),
            )
            conn.commit()
        finally:
            conn.close()
        return cid

    def save_fingerprint(
        self,
        *,
        run_id: str,
        scenario_id: str,
        vector_json: str,
        fingerprint_id: str | None = None,
    ) -> str:
        """Insert a behavioral fingerprint vector.

        Parameters
        ----------
        run_id : str
            FK to the parent run.
        scenario_id : str
            Which scenario the fingerprint belongs to.
        vector_json : str
            JSON-encoded fingerprint vector.
        fingerprint_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The fingerprint record's unique identifier.
        """
        fid = fingerprint_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO fingerprints
                    (id, run_id, scenario_id, vector_json)
                VALUES (?, ?, ?, ?)
                """,
                (fid, run_id, scenario_id, vector_json),
            )
            conn.commit()
        finally:
            conn.close()
        return fid

    def save_gate_decision(
        self,
        *,
        run_id: str,
        pipeline: str,
        decision: str,
        reason: str | None = None,
        rules_json: str,
        commit_sha: str | None = None,
        pr_number: int | None = None,
        gate_id: str | None = None,
    ) -> str:
        """Insert a deployment gate decision.

        Parameters
        ----------
        run_id : str
            FK to the parent run.
        pipeline : str
            CI/CD pipeline identifier (e.g. ``"main"``, ``"staging"``).
        decision : str
            Gate outcome (``"DEPLOY"``, ``"BLOCK"``, ``"WARN"``).
        reason : str | None
            Human-readable explanation.
        rules_json : str
            JSON-encoded gate rules that were evaluated.
        commit_sha : str | None
            Git commit SHA that triggered the gate.
        pr_number : int | None
            Pull-request number (if applicable).
        gate_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The gate decision's unique identifier.
        """
        gid = gate_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO gate_decisions
                    (id, run_id, pipeline, decision, reason,
                     rules_json, commit_sha, pr_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gid, run_id, pipeline, decision, reason,
                    rules_json, commit_sha, pr_number,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return gid

    def save_cost(
        self,
        *,
        run_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_cost: float = 0.0,
        trial_count: int = 0,
        cost_id: str | None = None,
    ) -> str:
        """Insert a cost accounting record.

        Parameters
        ----------
        run_id : str
            FK to the parent run.
        model : str
            LLM model identifier.
        input_tokens : int
            Prompt tokens consumed.
        output_tokens : int
            Completion tokens consumed.
        total_cost : float
            Cost in USD.
        trial_count : int
            Number of trials this cost covers.
        cost_id : str | None
            Optional explicit ID.  Generated if omitted.

        Returns
        -------
        str
            The cost record's unique identifier.
        """
        cid = cost_id or _uuid()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO costs
                    (id, run_id, model, input_tokens, output_tokens,
                     total_cost, trial_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid, run_id, model, input_tokens, output_tokens,
                    total_cost, trial_count,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return cid

    # ===================================================================
    # Update operations
    # ===================================================================

    def update_run_status(
        self,
        run_id: str,
        status: str,
        completed_at: str | None = None,
        total_cost: float | None = None,
    ) -> None:
        """Update a run's status, completion time, and cost.

        Parameters
        ----------
        run_id : str
            The run to update.
        status : str
            New status (``"completed"``, ``"failed"``, etc.).
        completed_at : str | None
            ISO-8601 completion timestamp.  Defaults to now (UTC).
        total_cost : float | None
            Updated cumulative cost.  ``None`` leaves the value unchanged.
        """
        ts = completed_at or _now_iso()
        conn = self._connect()
        try:
            if total_cost is not None:
                conn.execute(
                    """
                    UPDATE runs
                       SET status = ?, completed_at = ?, total_cost = ?
                     WHERE id = ?
                    """,
                    (status, ts, total_cost, run_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE runs
                       SET status = ?, completed_at = ?
                     WHERE id = ?
                    """,
                    (status, ts, run_id),
                )
            conn.commit()
        finally:
            conn.close()
