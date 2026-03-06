"""Boundary condition coverage tracker for AgentAssay.

Implements the boundary coverage dimension of the 5-dimensional
coverage vector (paper Definition 4.1). Tracks whether the agent
has been tested near the edges of its operational boundaries.

Boundaries are named numeric ranges — e.g., "response_time_ms" with
range (0, 5000), or "token_count" with range (1, 4096). The tracker
checks whether observed values from trace metadata have exercised the
region near the lower and upper bounds of each boundary.

"Near" is defined as within 10% of the range span from each boundary
edge. For a range (0, 100), "near lower" means a value in [0, 10]
and "near upper" means a value in [90, 100].

Coverage ratio = (boundaries with both ends tested) / (total boundaries).
"""

from __future__ import annotations

import math

from agentassay.core.models import ExecutionTrace

# Fraction of range span considered "near" a boundary edge.
_NEAR_FRACTION: float = 0.10


class BoundaryCoverageTracker:
    """Tracks boundary condition exercise across execution traces.

    Parameters
    ----------
    boundaries : dict[str, tuple[float, float]]
        Named boundaries with (min, max) ranges. The keys are
        metadata field names that should appear in trace or step
        metadata. Example::

            {"response_time_ms": (0.0, 5000.0),
             "token_count": (1.0, 4096.0)}

        An empty dict is permitted — coverage_ratio returns 1.0
        (vacuous truth: all zero boundaries are covered).

    Raises
    ------
    ValueError
        If any boundary has min >= max or contains non-finite values.
    """

    __slots__ = ("_boundaries", "_observations")

    def __init__(
        self, boundaries: dict[str, tuple[float, float]] | None = None
    ) -> None:
        boundaries = boundaries or {}
        for name, (lo, hi) in boundaries.items():
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(
                    f"Boundary '{name}' has non-finite values: ({lo}, {hi})"
                )
            if lo >= hi:
                raise ValueError(
                    f"Boundary '{name}' has min >= max: ({lo}, {hi})"
                )
        # Store as (min, max) tuples
        self._boundaries: dict[str, tuple[float, float]] = dict(boundaries)
        # Per-boundary: list of all observed numeric values
        self._observations: dict[str, list[float]] = {
            name: [] for name in self._boundaries
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_values(self, metadata: dict) -> dict[str, list[float]]:
        """Pull boundary-relevant numeric values from a metadata dict.

        Looks for keys matching boundary names. Accepts int/float values.
        Returns a mapping from boundary name to list of extracted values
        (usually 0 or 1 value per metadata dict, but we handle lists too).
        """
        found: dict[str, list[float]] = {}
        for name in self._boundaries:
            if name in metadata:
                val = metadata[name]
                if isinstance(val, (int, float)) and math.isfinite(val):
                    found.setdefault(name, []).append(float(val))
                elif isinstance(val, (list, tuple)):
                    for v in val:
                        if isinstance(v, (int, float)) and math.isfinite(v):
                            found.setdefault(name, []).append(float(v))
        return found

    def _is_near_lower(self, name: str, value: float) -> bool:
        """Check if a value is near the lower boundary."""
        lo, hi = self._boundaries[name]
        span = hi - lo
        threshold = lo + span * _NEAR_FRACTION
        return value <= threshold

    def _is_near_upper(self, name: str, value: float) -> bool:
        """Check if a value is near the upper boundary."""
        lo, hi = self._boundaries[name]
        span = hi - lo
        threshold = hi - span * _NEAR_FRACTION
        return value >= threshold

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trace: ExecutionTrace) -> None:
        """Extract boundary-relevant values from trace and step metadata.

        Scans both ``trace.metadata`` and each ``step.metadata`` for
        keys matching the configured boundary names.

        Parameters
        ----------
        trace : ExecutionTrace
            A completed agent execution trace.
        """
        # Trace-level metadata
        for name, values in self._extract_values(trace.metadata).items():
            self._observations[name].extend(values)

        # Step-level metadata
        for step in trace.steps:
            for name, values in self._extract_values(step.metadata).items():
                self._observations[name].extend(values)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def coverage_report(self) -> dict[str, dict]:
        """Per-boundary coverage details.

        Returns
        -------
        dict[str, dict]
            For each boundary name::

                {
                    "tested": bool,        # any observation recorded
                    "min_seen": float | None,
                    "max_seen": float | None,
                    "near_lower": bool,    # at least one value near lower bound
                    "near_upper": bool,    # at least one value near upper bound
                    "observations": int,   # total values recorded
                }
        """
        report: dict[str, dict] = {}
        for name in self._boundaries:
            obs = self._observations[name]
            if not obs:
                report[name] = {
                    "tested": False,
                    "min_seen": None,
                    "max_seen": None,
                    "near_lower": False,
                    "near_upper": False,
                    "observations": 0,
                }
            else:
                min_seen = min(obs)
                max_seen = max(obs)
                near_lower = any(self._is_near_lower(name, v) for v in obs)
                near_upper = any(self._is_near_upper(name, v) for v in obs)
                report[name] = {
                    "tested": True,
                    "min_seen": min_seen,
                    "max_seen": max_seen,
                    "near_lower": near_lower,
                    "near_upper": near_upper,
                    "observations": len(obs),
                }
        return report

    def coverage_ratio(self) -> float:
        """Fraction of boundaries tested near both ends.

        A boundary counts as "covered" if at least one observation
        falls near its lower bound AND at least one falls near its
        upper bound. This ensures the agent has been stressed at
        the extremes of each operational dimension.

        Returns 1.0 if no boundaries are configured (vacuous truth).

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        if not self._boundaries:
            return 1.0

        covered = 0
        report = self.coverage_report()
        for info in report.values():
            if info["near_lower"] and info["near_upper"]:
                covered += 1

        return covered / len(self._boundaries)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded observations. Boundary definitions are preserved."""
        for name in self._observations:
            self._observations[name].clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ratio = self.coverage_ratio()
        return (
            f"BoundaryCoverageTracker("
            f"coverage={ratio:.2%}, "
            f"boundaries={len(self._boundaries)})"
        )
