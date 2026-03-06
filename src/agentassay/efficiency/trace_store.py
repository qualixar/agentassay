# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Trace-First Testing — persistent trace storage for offline analysis.

THE INSIGHT (paper Section 7.3):

    Every production agent invocation generates an execution trace as a
    byproduct. These traces are typically discarded or sent to an
    observability platform (LangSmith, Langfuse) where they serve only
    as debugging artifacts.

    AgentAssay treats these traces as FIRST-CLASS TEST DATA. By storing
    them persistently:

    1. **Offline coverage:** Compute coverage metrics from stored traces
       at ZERO additional token cost. No need to re-run the agent.

    2. **Offline fingerprinting:** Build fingerprint distributions from
       production data. Compare production behavior to test-time behavior.

    3. **Drift detection:** Analyze how agent behavior evolves over time
       using sliding-window fingerprint distributions. Detect subtle
       behavioral drift before it becomes a regression.

    4. **Trace replay:** Re-analyze old traces with new contracts,
       metamorphic relations, or coverage criteria — again at zero cost.

    The trace store uses gzipped JSON files (one per trace) with a
    lightweight SQLite-free index for fast queries. This keeps the
    dependency footprint minimal and avoids the complexity of a
    database dependency.

STORAGE FORMAT:

    .agentassay/traces/
        index.json          # metadata index for fast lookups
        traces/
            {trace_id}.json.gz   # individual gzipped trace files

    The index maps trace_id -> {agent_id, scenario_id, timestamp, model,
    framework, success, cost_usd} for fast filtered queries without
    decompressing trace files.

References:
    - Humble, J. & Farley, D. (2010). Continuous Delivery. Addison-Wesley.
      Chapter 10: Testing Non-Functional Requirements.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.fingerprint import (
    BehavioralFingerprint,
)
from agentassay.efficiency.distribution import (
    FingerprintDistribution,
)


# ===================================================================
# Custom JSON encoder for Pydantic/datetime serialization
# ===================================================================


class _TraceEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and Pydantic models."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        return super().default(obj)


# ===================================================================
# TraceStore
# ===================================================================


class TraceStore:
    """Persistent store for agent execution traces.

    Enables Trace-First Testing: analyze production traces offline
    at ZERO additional token cost. Coverage, contracts, and metamorphic
    analysis can all run on stored traces.

    The store is file-system based (no database dependency):
    - Each trace is stored as a gzipped JSON file
    - A lightweight JSON index enables fast filtered queries
    - The index is updated atomically on each write

    Parameters
    ----------
    store_path : str
        Directory for trace storage. Created if it does not exist.
        Default: ".agentassay/traces"

    Example
    -------
    >>> store = TraceStore()
    >>> store.record(trace, metadata={"commit": "abc123"})
    >>> traces = store.query(agent_id="search-agent", limit=50)
    >>> fps = store.offline_fingerprints("search-agent", last_n=50)
    """

    __slots__ = ("_store_path", "_traces_dir", "_index_path", "_index", "_max_index_size")

    def __init__(
        self,
        store_path: str = ".agentassay/traces",
        max_index_size: int = 10000,
    ) -> None:
        self._store_path = Path(store_path)
        self._traces_dir = self._store_path / "traces"
        self._index_path = self._store_path / "index.json"
        self._max_index_size = max_index_size

        # Ensure directories exist
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        # Load or create the index — uses OrderedDict for LRU eviction
        self._index: OrderedDict[str, dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------
    # Recording traces
    # ------------------------------------------------------------------

    def record(
        self,
        trace: ExecutionTrace,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a trace from production or testing.

        Writes the trace as a gzipped JSON file and updates the index.

        Parameters
        ----------
        trace : ExecutionTrace
            The execution trace to store.
        metadata : dict | None
            Additional metadata (e.g., commit hash, environment).
            Stored alongside the trace for later querying.

        Returns
        -------
        str
            The trace_id of the stored trace.
        """
        trace_id = trace.trace_id
        extra_meta = metadata or {}

        # Serialize trace to JSON
        trace_data = trace.model_dump(mode="json")
        trace_data["_store_metadata"] = extra_meta

        # Write gzipped JSON
        trace_path = self._traces_dir / f"{trace_id}.json.gz"
        json_bytes = json.dumps(trace_data, cls=_TraceEncoder).encode("utf-8")

        with gzip.open(trace_path, "wb") as f:
            f.write(json_bytes)

        # Update index
        agent_id = extra_meta.get("agent_id", trace.metadata.get("agent_id", "unknown"))
        self._index[trace_id] = {
            "agent_id": agent_id,
            "scenario_id": trace.scenario_id,
            "timestamp": trace.timestamp.isoformat(),
            "model": trace.model,
            "framework": trace.framework,
            "success": trace.success,
            "cost_usd": trace.total_cost_usd,
            "step_count": trace.step_count,
            "file": f"{trace_id}.json.gz",
        }

        # LRU eviction: if the in-memory index exceeds max_index_size,
        # evict the oldest (least-recently-used) entries.
        while len(self._index) > self._max_index_size:
            self._index.popitem(last=False)

        self._save_index()
        return trace_id

    # ------------------------------------------------------------------
    # Querying traces
    # ------------------------------------------------------------------

    def query(
        self,
        agent_id: str | None = None,
        scenario: str | None = None,
        model: str | None = None,
        success: bool | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
    ) -> list[ExecutionTrace]:
        """Query stored traces with filters.

        Filters are ANDed together. All filters are optional; with no
        filters, returns the most recent traces up to ``limit``.

        Parameters
        ----------
        agent_id : str | None
            Filter by agent identifier.
        scenario : str | None
            Filter by scenario_id.
        model : str | None
            Filter by model name.
        success : bool | None
            Filter by success status.
        time_range : tuple[datetime, datetime] | None
            Filter by timestamp range (inclusive).
        limit : int
            Maximum number of traces to return. Default 100.

        Returns
        -------
        list[ExecutionTrace]
            Matching traces, sorted by timestamp descending (newest first).
        """
        # Filter the index first (fast — no decompression)
        matching_ids = self._filter_index(
            agent_id=agent_id,
            scenario=scenario,
            model=model,
            success=success,
            time_range=time_range,
        )

        # Sort by timestamp descending
        matching_ids.sort(
            key=lambda tid: self._index[tid].get("timestamp", ""),
            reverse=True,
        )

        # Load traces up to limit — touch each in the LRU index
        traces: list[ExecutionTrace] = []
        for trace_id in matching_ids[:limit]:
            trace = self._load_trace(trace_id)
            if trace is not None:
                traces.append(trace)
                # LRU touch: move accessed entry to end of OrderedDict
                if trace_id in self._index:
                    self._index.move_to_end(trace_id)

        return traces

    def count(
        self,
        agent_id: str | None = None,
        scenario: str | None = None,
    ) -> int:
        """Count traces matching the given filters without loading them.

        Parameters
        ----------
        agent_id : str | None
            Filter by agent identifier.
        scenario : str | None
            Filter by scenario_id.

        Returns
        -------
        int
            Number of matching traces.
        """
        return len(self._filter_index(agent_id=agent_id, scenario=scenario))

    # ------------------------------------------------------------------
    # Offline analysis (ZERO additional token cost)
    # ------------------------------------------------------------------

    def offline_fingerprints(
        self,
        agent_id: str | None = None,
        scenario: str | None = None,
        last_n: int = 50,
    ) -> FingerprintDistribution:
        """Compute fingerprint distribution from stored traces.

        This is the key Trace-First Testing operation: analyze behavioral
        patterns from historical data at ZERO additional cost.

        Parameters
        ----------
        agent_id : str | None
            Filter by agent identifier.
        scenario : str | None
            Filter by scenario_id.
        last_n : int
            Number of most recent traces to analyze. Default 50.

        Returns
        -------
        FingerprintDistribution
            Distribution computed from stored traces.

        Raises
        ------
        ValueError
            If fewer than 2 matching traces are found.
        """
        traces = self.query(agent_id=agent_id, scenario=scenario, limit=last_n)

        if len(traces) < 2:
            raise ValueError(
                f"Need at least 2 traces for fingerprint distribution, "
                f"found {len(traces)} matching traces."
            )

        fingerprints = [BehavioralFingerprint.from_trace(t) for t in traces]
        return FingerprintDistribution(fingerprints)

    def drift_detection(
        self,
        agent_id: str | None = None,
        scenario: str | None = None,
        window_size: int = 50,
        step_size: int = 25,
        alpha: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Detect behavioral drift over time from stored traces.

        Computes fingerprint distributions for sliding windows over
        time-sorted traces. Each window is compared to the first window
        (reference baseline). If a window is significantly different,
        that time period is flagged as a drift point.

        Parameters
        ----------
        agent_id : str | None
            Filter by agent identifier.
        scenario : str | None
            Filter by scenario_id.
        window_size : int
            Number of traces per window. Default 50.
        step_size : int
            How many traces to advance between windows. Default 25.
        alpha : float
            Significance level for drift detection. Default 0.05.

        Returns
        -------
        list[dict]
            One entry per window with:
            - window_index (int): 0-based window number
            - start_time (str): ISO timestamp of first trace in window
            - end_time (str): ISO timestamp of last trace in window
            - drift_detected (bool): whether this window differs from baseline
            - p_value (float): p-value of the comparison
            - distance (float): Mahalanobis distance from baseline
            - behavioral_variance (float): variance within this window

        Raises
        ------
        ValueError
            If not enough traces for at least 2 windows.
        """
        # Load all matching traces sorted by time (oldest first)
        all_traces = self.query(
            agent_id=agent_id,
            scenario=scenario,
            limit=10_000,  # get everything
        )
        # query returns newest-first; reverse for chronological
        all_traces.reverse()

        if len(all_traces) < 2 * window_size:
            raise ValueError(
                f"Drift detection needs at least {2 * window_size} traces "
                f"(2 windows of size {window_size}), found {len(all_traces)}."
            )

        # Build reference baseline from the first window
        baseline_traces = all_traces[:window_size]
        baseline_fps = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        baseline_dist = FingerprintDistribution(baseline_fps)

        # Slide windows and compare each to baseline
        results: list[dict[str, Any]] = []
        window_idx = 0
        pos = 0

        while pos + window_size <= len(all_traces):
            window_traces = all_traces[pos: pos + window_size]
            window_fps = [BehavioralFingerprint.from_trace(t) for t in window_traces]
            window_dist = FingerprintDistribution(window_fps)

            # Compare to baseline
            test_result = baseline_dist.regression_test(window_dist, alpha=alpha)

            start_ts = window_traces[0].timestamp.isoformat()
            end_ts = window_traces[-1].timestamp.isoformat()

            results.append({
                "window_index": window_idx,
                "start_time": start_ts,
                "end_time": end_ts,
                "drift_detected": test_result["regression_detected"],
                "p_value": test_result["p_value"],
                "distance": test_result["distance"],
                "behavioral_variance": window_dist.behavioral_variance,
                "changed_dimensions": test_result.get("changed_dimensions", []),
            })

            pos += step_size
            window_idx += 1

        return results

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def list_agents(self) -> list[str]:
        """List all unique agent_ids in the store.

        Returns
        -------
        list[str]
            Unique agent identifiers.
        """
        return sorted({
            entry.get("agent_id", "unknown") for entry in self._index.values()
        })

    def list_scenarios(self, agent_id: str | None = None) -> list[str]:
        """List all unique scenario_ids, optionally filtered by agent.

        Parameters
        ----------
        agent_id : str | None
            Filter by agent identifier.

        Returns
        -------
        list[str]
            Unique scenario identifiers.
        """
        scenarios: set[str] = set()
        for entry in self._index.values():
            if agent_id and entry.get("agent_id") != agent_id:
                continue
            scenarios.add(entry.get("scenario_id", "unknown"))
        return sorted(scenarios)

    def delete_trace(self, trace_id: str) -> bool:
        """Delete a single trace by ID.

        Parameters
        ----------
        trace_id : str
            The trace_id to delete.

        Returns
        -------
        bool
            True if the trace was found and deleted, False otherwise.
        """
        if trace_id not in self._index:
            return False

        # Delete the file
        filename = self._index[trace_id].get("file", f"{trace_id}.json.gz")
        filepath = self._traces_dir / filename
        if filepath.exists():
            filepath.unlink()

        # Remove from index
        del self._index[trace_id]
        self._save_index()

        return True

    def prune(self, max_age_days: int = 90) -> int:
        """Delete traces older than max_age_days.

        Parameters
        ----------
        max_age_days : int
            Maximum age in days. Traces older than this are deleted.

        Returns
        -------
        int
            Number of traces deleted.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
        to_delete: list[str] = []

        for trace_id, entry in self._index.items():
            ts_str = entry.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.timestamp() < cutoff:
                    to_delete.append(trace_id)
            except (ValueError, TypeError):
                continue

        for tid in to_delete:
            self.delete_trace(tid)

        return len(to_delete)

    @property
    def size(self) -> int:
        """Total number of traces in the store."""
        return len(self._index)

    @property
    def store_path(self) -> Path:
        """Path to the trace store directory."""
        return self._store_path

    # ------------------------------------------------------------------
    # Internal: index management
    # ------------------------------------------------------------------

    def _load_index(self) -> OrderedDict[str, dict[str, Any]]:
        """Load the index from disk, or return empty OrderedDict if not found."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f, object_pairs_hook=OrderedDict)
                if isinstance(data, dict):
                    # Ensure we always return an OrderedDict for LRU semantics
                    if not isinstance(data, OrderedDict):
                        return OrderedDict(data)
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return OrderedDict()

    def _save_index(self) -> None:
        """Atomically write the index to disk.

        Uses write-to-temp + shutil.move for atomicity and cross-
        filesystem portability.  ``Path.rename()`` can fail when source
        and destination are on different filesystems; ``shutil.move()``
        handles this transparently.
        """
        tmp_path = self._index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._index, f, indent=2)
        shutil.move(str(tmp_path), str(self._index_path))

    def _filter_index(
        self,
        agent_id: str | None = None,
        scenario: str | None = None,
        model: str | None = None,
        success: bool | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[str]:
        """Filter the index and return matching trace_ids.

        All filters are optional and ANDed together.

        Returns
        -------
        list[str]
            Matching trace_ids.
        """
        matching: list[str] = []

        for trace_id, entry in self._index.items():
            if agent_id and entry.get("agent_id") != agent_id:
                continue
            if scenario and entry.get("scenario_id") != scenario:
                continue
            if model and entry.get("model") != model:
                continue
            if success is not None and entry.get("success") != success:
                continue
            if time_range:
                ts_str = entry.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if not (time_range[0] <= ts <= time_range[1]):
                        continue
                except (ValueError, TypeError):
                    continue

            matching.append(trace_id)

        return matching

    def _load_trace(self, trace_id: str) -> ExecutionTrace | None:
        """Load a single trace from its gzipped file.

        Parameters
        ----------
        trace_id : str
            Trace identifier.

        Returns
        -------
        ExecutionTrace | None
            The loaded trace, or None if the file is missing/corrupt.
        """
        entry = self._index.get(trace_id)
        if entry is None:
            return None

        filename = entry.get("file", f"{trace_id}.json.gz")
        filepath = self._traces_dir / filename

        if not filepath.exists():
            return None

        try:
            with gzip.open(filepath, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))

            # Remove store metadata before deserializing
            data.pop("_store_metadata", None)

            return ExecutionTrace.model_validate(data)
        except (json.JSONDecodeError, gzip.BadGzipFile, Exception):
            return None
