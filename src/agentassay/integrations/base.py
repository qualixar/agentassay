# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Abstract base class for framework adapters.

Every AI agent framework (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK,
smolagents, or a custom callable) gets its own adapter that knows how to:

1. Invoke the framework-specific agent entry point.
2. Capture the execution as an ``ExecutionTrace`` with per-step timing.
3. Expose a ``to_callable()`` that returns a TrialRunner-compatible function.

Design pattern: **Adapter pattern** (GoF) — each concrete adapter translates
a framework-specific interface into the uniform ExecutionTrace contract that
the TrialRunner, statistics engine, and coverage tracker all depend on.

All framework imports are **lazy** — no adapter adds a hard dependency. If a
framework is not installed, the adapter raises a clear ``ImportError`` with
installation instructions (``pip install agentassay[langgraph]`` etc.).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from agentassay.core.models import AgentConfig, ExecutionTrace


class AdapterError(Exception):
    """Raised when an adapter encounters a non-recoverable error."""


class FrameworkNotInstalledError(ImportError):
    """Raised when the required framework package is not installed."""


class AgentAdapter(ABC):
    """Abstract base class for AI agent framework adapters.

    Every adapter wraps a framework-specific agent object and exposes
    two things:

    1. ``run(input_data)`` — execute the agent and return an ``ExecutionTrace``.
    2. ``to_callable()`` — return a ``Callable[[dict], ExecutionTrace]``
       compatible with ``TrialRunner``.

    Subclasses MUST set the ``framework`` class-level attribute to one of
    the framework literals defined in ``AgentConfig.framework``.

    Parameters
    ----------
    model
        The LLM model identifier used by the agent (e.g. ``"gpt-4o"``).
    agent_name
        A human-readable name for this agent (used in traces and reports).
    metadata
        Arbitrary metadata attached to every ``ExecutionTrace`` produced.
    """

    #: Framework identifier — must match AgentConfig.framework literal.
    framework: str

    def __init__(
        self,
        *,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._agent_name = agent_name or f"{self.framework}-agent"
        self._metadata = metadata or {}

    # -- Abstract interface ---------------------------------------------------

    @abstractmethod
    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Execute the agent on the given input and return a full trace.

        Implementations MUST:
        - Capture each discrete step as a ``StepTrace``.
        - Record per-step timing via ``time.perf_counter()``.
        - Handle all exceptions gracefully: return an ``ExecutionTrace``
          with ``success=False`` and the error message rather than raising.
        - Set ``scenario_id`` to ``input_data.get("scenario_id", "default")``.
        """

    @abstractmethod
    def to_callable(self) -> Callable[[dict[str, Any]], ExecutionTrace]:
        """Return a callable compatible with ``TrialRunner``.

        The returned callable has the signature::

            (input_data: dict[str, Any]) -> ExecutionTrace

        This is the function that gets passed to ``TrialRunner.__init__``.
        """

    # -- Concrete helpers -----------------------------------------------------

    def get_config(self) -> AgentConfig:
        """Build an ``AgentConfig`` describing this adapter's agent.

        Returns
        -------
        AgentConfig
            Configuration with the adapter's framework, model, and metadata.
        """
        return AgentConfig(
            agent_id=str(uuid.uuid4()),
            name=self._agent_name,
            framework=self.framework,
            model=self._model,
            metadata=self._metadata,
        )

    @property
    def model(self) -> str:
        """The LLM model identifier for this agent."""
        return self._model

    @property
    def agent_name(self) -> str:
        """Human-readable agent name."""
        return self._agent_name

    # -- Representation -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}("
            f"framework={self.framework!r}, "
            f"model={self._model!r}, "
            f"agent={self._agent_name!r})>"
        )
