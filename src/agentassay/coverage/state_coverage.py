"""State space coverage tracker for AgentAssay.

Implements the state coverage dimension of the 5-dimensional coverage
vector (paper Definition 4.1). Tracks unique states the agent visits
during execution.

A *state* is an observable snapshot at a single step, defined as the
tuple:
    (action, tool_name_or_None, output_type)

Where ``output_type`` classifies the step's output:
    - For tool calls: type(tool_output).__name__  (e.g. "dict", "str", "NoneType")
    - For LLM steps: "llm_text" if llm_output is non-empty, else "llm_empty"
    - For other actions: "no_output"

This captures the behavioral state space — not internal hidden state
(which is unobservable), but the externally visible action-output
profile at each step. Higher state coverage means the test suite has
explored more of the agent's observable behavior.
"""

from __future__ import annotations

from collections import Counter

from agentassay.core.models import ExecutionTrace, StepTrace


def _classify_output(step: StepTrace) -> str:
    """Classify the output type of a step for state fingerprinting.

    Returns a stable string label for the kind of output produced.
    This keeps state tuples human-readable and hashable.
    """
    # Tool call output
    if step.tool_output is not None:
        return type(step.tool_output).__name__

    # LLM output
    if step.llm_output is not None:
        return "llm_text" if step.llm_output.strip() else "llm_empty"

    return "no_output"


def _step_to_state(step: StepTrace) -> tuple[str, str, str]:
    """Convert a StepTrace into its state tuple representation.

    Returns
    -------
    tuple[str, str, str]
        (action, tool_name_or_"None", output_type)
    """
    tool_label = step.tool_name if step.tool_name is not None else "None"
    output_type = _classify_output(step)
    return (step.action, tool_label, output_type)


class StateCoverageTracker:
    """Tracks unique observable states across execution traces.

    Each step in a trace produces a state tuple. The tracker
    accumulates the set of all unique states and their frequencies.
    """

    __slots__ = ("_states", "_frequency")

    def __init__(self) -> None:
        self._states: set[tuple[str, ...]] = set()
        self._frequency: Counter[tuple[str, ...]] = Counter()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trace: ExecutionTrace) -> None:
        """Extract and record states from every step in the trace.

        Parameters
        ----------
        trace : ExecutionTrace
            A completed agent execution trace.
        """
        for step in trace.steps:
            state = _step_to_state(step)
            self._states.add(state)
            self._frequency[state] += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def coverage_count(self) -> int:
        """Number of unique states observed so far.

        Unlike tool or model coverage, state space has no fixed
        denominator (the total state space is unbounded in theory).
        The raw count is therefore the primary metric. Normalization
        to [0, 1] is handled at the aggregate level.

        Returns
        -------
        int
        """
        return len(self._states)

    def unique_states(self) -> set[tuple[str, ...]]:
        """All unique state tuples observed.

        Returns
        -------
        set[tuple[str, ...]]
            Each element is a (action, tool_name, output_type) tuple.
        """
        return set(self._states)

    def state_frequency(self) -> dict[tuple[str, ...], int]:
        """Observation count for each unique state.

        Returns
        -------
        dict[tuple[str, ...], int]
        """
        return dict(self._frequency)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded observations."""
        self._states.clear()
        self._frequency.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        total_observations = sum(self._frequency.values())
        return (
            f"StateCoverageTracker("
            f"unique_states={len(self._states)}, "
            f"total_observations={total_observations})"
        )
