"""Analytical query API for AgentAssay result data.

Provides read-only trend aggregations, cross-run comparisons, and summary
statistics intended for dashboard rendering and CI/CD reporting.  All
queries use parameterized SQL — no string interpolation of user input.

Design notes
------------
* Every method accepts optional filter kwargs and returns plain
  ``list[dict]`` or ``dict`` — no ORM, no Pydantic overhead on the read
  path.  Serialization to JSON is trivial from these return types.
* Date-range filters use SQLite's ``date()`` function against ISO-8601
  stored timestamps, so no Python-side date arithmetic is needed.
* The class is stateless beyond its ``ResultStore`` reference; safe to
  share across threads.
"""

from __future__ import annotations

from typing import Any

from agentassay.persistence.storage import ResultStore


class QueryAPI:
    """Read-only analytical queries over the AgentAssay result database.

    Parameters
    ----------
    store : ResultStore
        The storage engine to query against.

    Examples
    --------
    >>> api = QueryAPI(store)
    >>> api.get_run_summary()
    {'total_runs': 42, 'total_cost': 12.34, ...}
    """

    def __init__(self, store: ResultStore) -> None:
        self._store = store

    # -- Internal -----------------------------------------------------------

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only query and return rows as dicts.

        Parameters
        ----------
        sql : str
            Parameterized SQL statement.
        params : tuple
            Bind parameters for the query.

        Returns
        -------
        list[dict[str, Any]]
            Each row as an ``{column: value}`` dictionary.
        """
        conn = self._store._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Execute a query expected to return at most one row.

        Parameters
        ----------
        sql : str
            Parameterized SQL statement.
        params : tuple
            Bind parameters.

        Returns
        -------
        dict[str, Any] | None
            The row as a dict, or ``None`` if no match.
        """
        conn = self._store._connect()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ===================================================================
    # Run listing & filtering
    # ===================================================================

    def list_runs(
        self,
        project_id: str | None = None,
        agent_name: str | None = None,
        model: str | None = None,
        framework: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List runs with optional filters, newest first.

        Parameters
        ----------
        project_id : str | None
            Filter by project.
        agent_name : str | None
            Filter by agent name (exact match).
        model : str | None
            Filter by LLM model.
        framework : str | None
            Filter by agent framework.
        status : str | None
            Filter by run status.
        limit : int
            Maximum rows to return.  Default 50.
        offset : int
            Pagination offset.  Default 0.

        Returns
        -------
        list[dict[str, Any]]
            Run records ordered by ``started_at DESC``.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if framework is not None:
            clauses.append("framework = ?")
            params.append(framework)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT * FROM runs WHERE {where} "
            "ORDER BY started_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        return self._query(sql, tuple(params))

    # ===================================================================
    # Trend queries
    # ===================================================================

    def get_pass_rate_trend(
        self,
        agent_name: str | None = None,
        model: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Aggregate daily pass rates over recent runs.

        Parameters
        ----------
        agent_name : str | None
            Filter by agent.
        model : str | None
            Filter by LLM model.
        days : int
            Look-back window in days.  Default 30.

        Returns
        -------
        list[dict[str, Any]]
            ``[{date, pass_rate, run_count}, ...]`` ordered by date ASC.
        """
        clauses: list[str] = [
            "r.started_at >= date('now', ?)"
        ]
        params: list[Any] = [f"-{days} days"]

        if agent_name is not None:
            clauses.append("r.agent_name = ?")
            params.append(agent_name)
        if model is not None:
            clauses.append("r.model = ?")
            params.append(model)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT
                date(r.started_at) AS date,
                AVG(v.pass_rate)   AS pass_rate,
                COUNT(DISTINCT r.id) AS run_count
            FROM runs r
            JOIN verdicts v ON v.run_id = r.id
            WHERE {where}
            GROUP BY date(r.started_at)
            ORDER BY date(r.started_at) ASC
        """
        return self._query(sql, tuple(params))

    def get_cost_trend(
        self,
        agent_name: str | None = None,
        model: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Aggregate daily costs over recent runs.

        Parameters
        ----------
        agent_name : str | None
            Filter by agent.
        model : str | None
            Filter by LLM model.
        days : int
            Look-back window in days.  Default 30.

        Returns
        -------
        list[dict[str, Any]]
            ``[{date, total_cost, run_count}, ...]`` ordered by date ASC.
        """
        clauses: list[str] = [
            "started_at >= date('now', ?)"
        ]
        params: list[Any] = [f"-{days} days"]

        if agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT
                date(started_at)  AS date,
                SUM(total_cost)   AS total_cost,
                COUNT(*)          AS run_count
            FROM runs
            WHERE {where}
            GROUP BY date(started_at)
            ORDER BY date(started_at) ASC
        """
        return self._query(sql, tuple(params))

    def get_coverage_trend(
        self,
        run_ids: list[str] | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Aggregate coverage scores over recent runs by dimension.

        Parameters
        ----------
        run_ids : list[str] | None
            Explicit set of run IDs to include.  If ``None``, uses a
            date-based look-back window.
        days : int
            Look-back window in days (ignored when ``run_ids`` is given).

        Returns
        -------
        list[dict[str, Any]]
            ``[{dimension, avg_score, measurement_count}, ...]``.
        """
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            sql = f"""
                SELECT
                    dimension,
                    AVG(score)  AS avg_score,
                    COUNT(*)    AS measurement_count
                FROM coverage
                WHERE run_id IN ({placeholders})
                GROUP BY dimension
                ORDER BY dimension
            """
            return self._query(sql, tuple(run_ids))

        sql = """
            SELECT
                c.dimension,
                AVG(c.score)  AS avg_score,
                COUNT(*)      AS measurement_count
            FROM coverage c
            JOIN runs r ON r.id = c.run_id
            WHERE r.started_at >= date('now', ?)
            GROUP BY c.dimension
            ORDER BY c.dimension
        """
        return self._query(sql, (f"-{days} days",))

    # ===================================================================
    # Comparison queries
    # ===================================================================

    def get_fingerprint_comparison(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> dict[str, Any]:
        """Retrieve fingerprints for two runs side-by-side.

        Parameters
        ----------
        baseline_run_id : str
            The reference (known-good) run.
        candidate_run_id : str
            The candidate (new) run to compare against.

        Returns
        -------
        dict[str, Any]
            ``{"baseline": [...], "candidate": [...], "baseline_run_id": ...,
            "candidate_run_id": ...}``.
        """
        baseline = self._query(
            "SELECT * FROM fingerprints WHERE run_id = ? ORDER BY scenario_id",
            (baseline_run_id,),
        )
        candidate = self._query(
            "SELECT * FROM fingerprints WHERE run_id = ? ORDER BY scenario_id",
            (candidate_run_id,),
        )
        return {
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
            "baseline": baseline,
            "candidate": candidate,
        }

    # ===================================================================
    # Gate history
    # ===================================================================

    def get_gate_history(
        self,
        pipeline: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Retrieve deployment gate decisions for a pipeline.

        Parameters
        ----------
        pipeline : str | None
            Filter by pipeline name.  ``None`` returns all pipelines.
        days : int
            Look-back window in days.  Default 30.

        Returns
        -------
        list[dict[str, Any]]
            Gate decisions ordered by ``created_at DESC``.
        """
        clauses: list[str] = [
            "created_at >= date('now', ?)"
        ]
        params: list[Any] = [f"-{days} days"]

        if pipeline is not None:
            clauses.append("pipeline = ?")
            params.append(pipeline)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT * FROM gate_decisions
            WHERE {where}
            ORDER BY created_at DESC
        """
        return self._query(sql, tuple(params))

    # ===================================================================
    # Summary
    # ===================================================================

    def get_run_summary(self) -> dict[str, Any]:
        """Compute a high-level summary across all stored runs.

        Returns
        -------
        dict[str, Any]
            ``{"total_runs": int, "total_cost": float,
            "avg_pass_rate": float | None, "last_run_date": str | None}``.
        """
        run_stats = self._query_one(
            """
            SELECT
                COUNT(*)       AS total_runs,
                COALESCE(SUM(total_cost), 0.0)  AS total_cost,
                MAX(started_at) AS last_run_date
            FROM runs
            """
        )
        if run_stats is None:
            return {
                "total_runs": 0,
                "total_cost": 0.0,
                "avg_pass_rate": None,
                "last_run_date": None,
            }

        avg_row = self._query_one(
            "SELECT AVG(pass_rate) AS avg_pass_rate FROM verdicts"
        )
        avg_pass = avg_row["avg_pass_rate"] if avg_row else None

        return {
            "total_runs": run_stats["total_runs"],
            "total_cost": run_stats["total_cost"],
            "avg_pass_rate": avg_pass,
            "last_run_date": run_stats["last_run_date"],
        }
