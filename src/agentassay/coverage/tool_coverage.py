"""Tool coverage tracker for AgentAssay.

Implements the tool coverage dimension of the 5-dimensional coverage
vector (paper Definition 4.1). Measures which tools in the agent's
toolbox have been exercised across stochastic trials.

Coverage ratio = |used_tools| / |known_tools|

A ratio of 1.0 means every tool the agent CAN call has been called
at least once across all observed traces. Low tool coverage indicates
the test suite has not exercised the agent's full capabilities.
"""

from __future__ import annotations

from collections import Counter

from agentassay.core.models import ExecutionTrace


class ToolCoverageTracker:
    """Tracks which tools have been exercised across execution traces.

    Parameters
    ----------
    known_tools : set[str]
        The complete set of tools the agent is configured to use.
        This is the denominator for coverage ratio calculation.
        Must not be empty — an agent with no tools has trivial
        coverage (always 1.0), but that should be expressed at
        the aggregate level, not here.

    Raises
    ------
    ValueError
        If ``known_tools`` is empty.
    """

    __slots__ = ("_known_tools", "_used_tools", "_frequency")

    def __init__(self, known_tools: set[str]) -> None:
        if not known_tools:
            raise ValueError(
                "known_tools must be non-empty. An agent with no tools "
                "cannot have meaningful tool coverage."
            )
        self._known_tools: frozenset[str] = frozenset(known_tools)
        self._used_tools: set[str] = set()
        self._frequency: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trace: ExecutionTrace) -> None:
        """Record tool usage from a single execution trace.

        Extracts every tool_name from the trace's steps and updates
        both the used-tools set and the per-tool call frequency.

        Parameters
        ----------
        trace : ExecutionTrace
            A completed agent execution trace.
        """
        for step in trace.steps:
            if step.tool_name is not None:
                self._used_tools.add(step.tool_name)
                self._frequency[step.tool_name] += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def coverage_ratio(self) -> float:
        """Fraction of known tools that have been exercised.

        Returns
        -------
        float
            Value in [0.0, 1.0]. 1.0 means every known tool has been
            called at least once across all observed traces.
        """
        covered = self._used_tools & self._known_tools
        return len(covered) / len(self._known_tools)

    def uncovered_tools(self) -> set[str]:
        """Tools in the known set that have never been called.

        Returns
        -------
        set[str]
            The difference ``known_tools - used_tools``.
        """
        return set(self._known_tools - self._used_tools)

    def tool_frequency(self) -> dict[str, int]:
        """Per-tool invocation count across all observed traces.

        Returns
        -------
        dict[str, int]
            Mapping from tool name to total call count. Includes
            tools outside ``known_tools`` if the agent called them
            (useful for detecting undocumented tool usage).
        """
        return dict(self._frequency)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded observations. ``known_tools`` is preserved."""
        self._used_tools.clear()
        self._frequency.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ratio = self.coverage_ratio()
        return (
            f"ToolCoverageTracker("
            f"coverage={ratio:.2%}, "
            f"used={len(self._used_tools)}/{len(self._known_tools)})"
        )
