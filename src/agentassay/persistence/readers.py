"""Read operations mixin for the AgentAssay ResultStore.

Provides all ``get_*`` query methods and utility methods (``table_exists``,
``count``) as a mixin class.  ``ResultStore`` inherits from this mixin so
that read and write responsibilities live in separate files while exposing
a single unified class to consumers.

All methods acquire their own SQLite connection via ``self._connect()``,
which is defined in the main ``storage`` module.
"""

from __future__ import annotations

from typing import Any

from agentassay.persistence.schema import ALLOWED_TABLES


class ResultStoreReaderMixin:
    """Mixin providing read-only query methods for ResultStore.

    Requires the host class to implement ``_connect()`` returning a
    ``sqlite3.Connection`` with ``row_factory = sqlite3.Row``.
    """

    # ===================================================================
    # Read operations
    # ===================================================================

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a single run by ID.

        Parameters
        ----------
        run_id : str
            The run's unique identifier.

        Returns
        -------
        dict[str, Any] | None
            Run data as a dictionary, or ``None`` if not found.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_trials(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all trials for a given run.

        Parameters
        ----------
        run_id : str
            The parent run's ID.

        Returns
        -------
        list[dict[str, Any]]
            List of trial dictionaries ordered by ``trial_num``.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            rows = conn.execute(
                "SELECT * FROM trials WHERE run_id = ? ORDER BY trial_num",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_verdicts(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all verdicts for a given run.

        Parameters
        ----------
        run_id : str
            The parent run's ID.

        Returns
        -------
        list[dict[str, Any]]
            List of verdict dictionaries.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            rows = conn.execute(
                "SELECT * FROM verdicts WHERE run_id = ?", (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_coverage(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all coverage records for a given run.

        Parameters
        ----------
        run_id : str
            The parent run's ID.

        Returns
        -------
        list[dict[str, Any]]
            List of coverage dictionaries.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            rows = conn.execute(
                "SELECT * FROM coverage WHERE run_id = ?", (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_fingerprints(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all fingerprints for a given run.

        Parameters
        ----------
        run_id : str
            The parent run's ID.

        Returns
        -------
        list[dict[str, Any]]
            List of fingerprint dictionaries.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            rows = conn.execute(
                "SELECT * FROM fingerprints WHERE run_id = ?", (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_gate_decisions(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all gate decisions for a given run.

        Parameters
        ----------
        run_id : str
            The parent run's ID.

        Returns
        -------
        list[dict[str, Any]]
            List of gate decision dictionaries.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            rows = conn.execute(
                "SELECT * FROM gate_decisions WHERE run_id = ?", (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_costs(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all cost records for a given run.

        Parameters
        ----------
        run_id : str
            The parent run's ID.

        Returns
        -------
        list[dict[str, Any]]
            List of cost dictionaries.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            rows = conn.execute(
                "SELECT * FROM costs WHERE run_id = ?", (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ===================================================================
    # Utility
    # ===================================================================

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database.

        Parameters
        ----------
        table_name : str
            Name of the table to check.

        Returns
        -------
        bool
            ``True`` if the table exists.
        """
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            return row[0] > 0 if row else False
        finally:
            conn.close()

    def count(self, table_name: str) -> int:
        """Return the row count for a table.

        Parameters
        ----------
        table_name : str
            Name of the table.  Must be one of the known tables to prevent
            SQL injection (the table name is validated before interpolation).

        Returns
        -------
        int
            Number of rows.

        Raises
        ------
        ValueError
            If ``table_name`` is not a known schema table.
        """
        if table_name not in ALLOWED_TABLES:
            raise ValueError(
                f"Unknown table '{table_name}'. "
                f"Allowed: {sorted(ALLOWED_TABLES)}"
            )
        conn = self._connect()  # type: ignore[attr-defined]
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608 — validated above
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
