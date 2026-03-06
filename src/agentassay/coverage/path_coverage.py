"""Decision path coverage tracker for AgentAssay.

Implements the path coverage dimension of the 5-dimensional coverage
vector (paper Definition 4.1). Tracks unique sequences of actions
(the "decision path") taken by the agent across stochastic trials.

The decision path is the ordered list of ``step.action`` values from
an execution trace, truncated to ``max_path_depth``. Two traces with
identical action sequences (up to depth) are considered to have taken
the same path.

Coverage ratio uses a heuristic estimate for the total path space:
    estimated_total = |unique_actions|^max_path_depth
This is an upper bound — the actual reachable path space is typically
smaller due to domain constraints. The ratio therefore provides a
conservative (lower-bound) coverage estimate.
"""

from __future__ import annotations

from collections import Counter

from agentassay.core.models import ExecutionTrace


class PathCoverageTracker:
    """Tracks unique decision paths across execution traces.

    Parameters
    ----------
    max_path_depth : int
        Maximum number of actions to consider in a path. Traces
        longer than this are truncated. Must be >= 1.

    Raises
    ------
    ValueError
        If ``max_path_depth`` < 1.
    """

    __slots__ = ("_max_depth", "_paths", "_frequency", "_unique_actions")

    def __init__(self, max_path_depth: int = 10) -> None:
        if max_path_depth < 1:
            raise ValueError(
                f"max_path_depth must be >= 1, got {max_path_depth}"
            )
        self._max_depth: int = max_path_depth
        self._paths: set[tuple[str, ...]] = set()
        self._frequency: Counter[tuple[str, ...]] = Counter()
        self._unique_actions: set[str] = set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_path(self, trace: ExecutionTrace) -> tuple[str, ...]:
        """Extract the truncated decision path from a trace.

        Uses ``trace.decision_path`` (which returns the ordered list
        of ``step.action`` values) and truncates to ``max_path_depth``.
        """
        raw = trace.decision_path
        truncated = raw[: self._max_depth]
        return tuple(truncated)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trace: ExecutionTrace) -> None:
        """Record the decision path from a single execution trace.

        Parameters
        ----------
        trace : ExecutionTrace
            A completed agent execution trace.
        """
        path = self._extract_path(trace)
        if not path:
            # Empty trace — nothing to record
            return
        self._paths.add(path)
        self._frequency[path] += 1
        self._unique_actions.update(path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def coverage_ratio(self) -> float:
        """Fraction of estimated path space that has been observed.

        Uses the heuristic:
            estimated_total = |unique_actions|^min(max_path_depth, max_observed_length)

        Returns 0.0 if no paths observed. Capped at 1.0 since the
        heuristic is an upper bound.

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        if not self._paths:
            return 0.0

        n_actions = len(self._unique_actions)
        if n_actions == 0:
            return 0.0

        # Use the max observed path length (capped at max_depth) as the
        # exponent — more realistic than always using max_depth.
        max_observed_len = max(len(p) for p in self._paths)
        exponent = min(max_observed_len, self._max_depth)

        estimated_total = n_actions ** exponent
        if estimated_total == 0:
            return 0.0

        ratio = len(self._paths) / estimated_total
        return min(ratio, 1.0)

    def unique_paths(self) -> set[tuple[str, ...]]:
        """All unique decision paths observed so far.

        Returns
        -------
        set[tuple[str, ...]]
            Each element is a tuple of action strings.
        """
        return set(self._paths)

    def path_frequency(self) -> dict[tuple[str, ...], int]:
        """How many times each unique path has been observed.

        Returns
        -------
        dict[tuple[str, ...], int]
            Mapping from path tuple to observation count.
        """
        return dict(self._frequency)

    def most_common_path(self) -> tuple[str, ...] | None:
        """The path observed most frequently, or None if no paths recorded.

        Returns
        -------
        tuple[str, ...] | None
        """
        if not self._frequency:
            return None
        return self._frequency.most_common(1)[0][0]

    def rarest_path(self) -> tuple[str, ...] | None:
        """The path observed least frequently, or None if no paths recorded.

        When multiple paths share the minimum frequency, returns one
        arbitrarily (the last in ``most_common`` order).

        Returns
        -------
        tuple[str, ...] | None
        """
        if not self._frequency:
            return None
        # most_common() returns descending order; last element is rarest
        return self._frequency.most_common()[-1][0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded observations."""
        self._paths.clear()
        self._frequency.clear()
        self._unique_actions.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ratio = self.coverage_ratio()
        return (
            f"PathCoverageTracker("
            f"coverage={ratio:.2%}, "
            f"unique_paths={len(self._paths)}, "
            f"max_depth={self._max_depth})"
        )
