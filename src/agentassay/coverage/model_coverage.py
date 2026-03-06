# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Model coverage tracker for AgentAssay.

Implements the model coverage dimension of the 5-dimensional coverage
vector (paper Definition 4.1). Tracks which LLM models have been
exercised across stochastic trials.

In production, agents may be deployed against multiple models (e.g.,
GPT-4o, Claude Opus, Gemini Pro) either for fallback, cost
optimization, or A/B testing. Regression testing must cover all
models the agent may encounter. This tracker measures that coverage.

If ``known_models`` is not provided (empty set), the tracker operates
in discovery mode — it records all models seen and reports 1.0 coverage
(since the denominator is undefined). Once ``known_models`` is set,
coverage becomes |tested ∩ known| / |known|.
"""

from __future__ import annotations

from collections import Counter

from agentassay.core.models import ExecutionTrace


class ModelCoverageTracker:
    """Tracks which LLM models have been exercised.

    Parameters
    ----------
    known_models : set[str]
        The set of models the agent may use in production. If empty,
        coverage_ratio returns 1.0 (discovery mode — all observed
        models are "covered" by definition).
    """

    __slots__ = ("_known_models", "_tested_models", "_frequency")

    def __init__(self, known_models: set[str] | None = None) -> None:
        self._known_models: frozenset[str] = frozenset(
            known_models or set()
        )
        self._tested_models: set[str] = set()
        self._frequency: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trace: ExecutionTrace) -> None:
        """Record model usage from a single execution trace.

        Extracts the trace-level ``model`` field and also scans each
        step for per-step ``model`` overrides (e.g., an agent that
        uses a cheaper model for retrieval and a stronger model for
        reasoning).

        Parameters
        ----------
        trace : ExecutionTrace
            A completed agent execution trace.
        """
        # Trace-level model (always present — it's a required field)
        self._tested_models.add(trace.model)
        self._frequency[trace.model] += 1

        # Step-level model overrides
        for step in trace.steps:
            if step.model is not None:
                self._tested_models.add(step.model)
                self._frequency[step.model] += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def coverage_ratio(self) -> float:
        """Fraction of known models that have been tested.

        Returns 1.0 if ``known_models`` was empty (discovery mode).

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        if not self._known_models:
            return 1.0
        covered = self._tested_models & self._known_models
        return len(covered) / len(self._known_models)

    def tested_models(self) -> set[str]:
        """All models observed across traces.

        Returns
        -------
        set[str]
            May include models NOT in ``known_models`` (useful for
            detecting unexpected model usage).
        """
        return set(self._tested_models)

    def untested_models(self) -> set[str]:
        """Known models that have not been tested.

        Returns
        -------
        set[str]
            ``known_models - tested_models``. Empty if discovery mode.
        """
        return set(self._known_models - self._tested_models)

    def model_frequency(self) -> dict[str, int]:
        """Per-model usage count across all traces.

        Returns
        -------
        dict[str, int]
        """
        return dict(self._frequency)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded observations. ``known_models`` is preserved."""
        self._tested_models.clear()
        self._frequency.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ratio = self.coverage_ratio()
        mode = "discovery" if not self._known_models else "tracking"
        return (
            f"ModelCoverageTracker("
            f"coverage={ratio:.2%}, "
            f"tested={len(self._tested_models)}, "
            f"mode={mode})"
        )
