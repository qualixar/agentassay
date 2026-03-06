# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""OpenAI Agents SDK adapter for AgentAssay.

Wraps an OpenAI Agents SDK ``Agent`` and captures its execution as an
``ExecutionTrace``. The OpenAI Agents SDK (released Jan 2025, GA Feb 2025)
uses a ``Runner.run()`` pattern with streaming support and built-in tool
execution.

All ``openai`` and ``agents`` imports are **lazy** — this module can be
imported even when the SDK is not installed.

Usage
-----
>>> from agentassay.integrations import OpenAIAgentsAdapter
>>> adapter = OpenAIAgentsAdapter(agent=my_agent, model="gpt-4o")
>>> trace = adapter.run({"query": "Summarize this document"})
>>> runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from agentassay.core.models import ExecutionTrace, StepTrace
from agentassay.integrations.base import (
    AgentAdapter,
    FrameworkNotInstalledError,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "OpenAI Agents adapter requires the openai-agents SDK. "
    "Install with: pip install agentassay[openai]"
)


def _check_openai_agents_installed() -> None:
    """Verify that the OpenAI Agents SDK is available."""
    try:
        import agents  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class OpenAIAgentsAdapter(AgentAdapter):
    """Adapter for OpenAI Agents SDK agents.

    Parameters
    ----------
    agent
        An OpenAI Agents SDK ``Agent`` instance.
    model
        LLM model identifier (e.g. ``"gpt-4o"``). If ``None``, attempts
        to read from ``agent.model``.
    agent_name
        Human-readable name. Defaults to ``agent.name`` or ``"openai-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "openai"

    def __init__(
        self,
        agent: Any,
        *,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Try to pull model/name from the agent object
        resolved_model = model
        if model == "unknown":
            resolved_model = getattr(agent, "model", "unknown") or "unknown"

        resolved_name = agent_name
        if resolved_name is None:
            resolved_name = getattr(agent, "name", None) or "openai-agent"

        super().__init__(model=resolved_model, agent_name=resolved_name, metadata=metadata)
        self._agent = agent

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the OpenAI agent and capture an ExecutionTrace.

        Uses ``Runner.run_sync()`` to execute the agent synchronously
        and then extracts steps from the ``RunResult`` response items.

        Parameters
        ----------
        input_data
            The scenario input. Must contain a ``"query"`` or ``"input"``
            key with the user message, or the entire dict is serialized
            as the prompt.

        Returns
        -------
        ExecutionTrace
            Full trace with tool calls, LLM responses, and timing.
        """
        _check_openai_agents_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            from agents import Runner

            # Build the user prompt from input_data
            user_input = self._build_user_input(input_data)

            run_start = time.perf_counter()
            result = Runner.run_sync(self._agent, user_input)
            run_duration_ms = (time.perf_counter() - run_start) * 1000.0

            steps = self._extract_steps(result, run_duration_ms)
            output = self._extract_output(result)
            total_ms = (time.perf_counter() - overall_start) * 1000.0

            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=steps,
                input_data=input_data,
                output_data=output,
                success=True,
                error=None,
                total_duration_ms=total_ms,
                total_cost_usd=0.0,
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

        except FrameworkNotInstalledError:
            raise

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "OpenAI Agents adapter failed: %s\n%s",
                error_msg,
                traceback.format_exc(),
            )
            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=[],
                input_data=input_data,
                output_data=None,
                success=False,
                error=error_msg,
                total_duration_ms=total_ms,
                total_cost_usd=0.0,
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

    def to_callable(self) -> Callable[[dict[str, Any]], ExecutionTrace]:
        """Return a TrialRunner-compatible callable."""
        _check_openai_agents_installed()
        return self.run

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _build_user_input(input_data: dict[str, Any]) -> str:
        """Extract or construct the user prompt from input_data.

        Checks keys in priority order: ``query``, ``input``, ``prompt``,
        ``message``. Falls back to serializing the entire dict.
        """
        for key in ("query", "input", "prompt", "message"):
            if key in input_data:
                return str(input_data[key])

        # Filter out scenario_id and metadata for the prompt
        filtered = {k: v for k, v in input_data.items() if k not in ("scenario_id", "metadata")}
        if len(filtered) == 1:
            return str(next(iter(filtered.values())))

        import json

        return json.dumps(filtered, default=str)

    def _extract_steps(self, result: Any, total_run_ms: float) -> list[StepTrace]:
        """Extract StepTrace objects from a RunResult.

        The OpenAI Agents SDK ``RunResult`` has:
        - ``new_items``: list of response items (messages, tool calls, etc.)
        - ``final_output``: the final agent output
        - ``raw_responses``: list of raw ModelResponse objects

        We iterate ``new_items`` and classify each into a step.
        """
        steps: list[StepTrace] = []

        new_items = getattr(result, "new_items", None)
        if not new_items:
            # Fallback: single-step trace
            steps.append(
                StepTrace(
                    step_index=0,
                    action="llm_response",
                    llm_output=self._extract_output(result),
                    duration_ms=total_run_ms,
                    model=self._model,
                    metadata={"openai_fallback": True},
                )
            )
            return steps

        num_items = len(new_items)
        per_item_ms = total_run_ms / max(num_items, 1)

        for idx, item in enumerate(new_items):
            action, kwargs = self._classify_item(item)
            steps.append(
                StepTrace(
                    step_index=idx,
                    action=action,
                    duration_ms=per_item_ms,
                    model=self._model,
                    metadata={
                        "openai_item_type": type(item).__name__,
                        "openai_item_index": idx,
                    },
                    **kwargs,
                )
            )

        return steps

    @staticmethod
    def _classify_item(item: Any) -> tuple[str, dict[str, Any]]:
        """Classify an OpenAI Agents SDK response item into a step action.

        Common item types:
        - ``MessageOutputItem``: LLM text response
        - ``ToolCallItem``: tool invocation
        - ``ToolCallOutputItem``: tool result
        - ``HandoffOutputItem``: agent handoff
        """
        extra: dict[str, Any] = {}
        item_type = type(item).__name__.lower()

        # Tool call items
        if "toolcall" in item_type and "output" not in item_type:
            raw_item = getattr(item, "raw_item", item)
            extra["tool_name"] = getattr(raw_item, "name", getattr(item, "name", "unknown_tool"))
            extra["tool_input"] = getattr(raw_item, "arguments", getattr(item, "arguments", {}))
            # Parse arguments if string
            if isinstance(extra["tool_input"], str):
                try:
                    import json

                    extra["tool_input"] = json.loads(extra["tool_input"])
                except (json.JSONDecodeError, TypeError):
                    extra["tool_input"] = {"raw": extra["tool_input"]}
            return "tool_call", extra

        # Tool output items
        if "toolcalloutput" in item_type or "tool_output" in item_type:
            extra["tool_name"] = getattr(item, "tool_name", "unknown_tool")
            extra["tool_input"] = {"source": "tool_response"}
            extra["tool_output"] = getattr(item, "output", str(item))
            return "tool_call", extra

        # Handoff items
        if "handoff" in item_type:
            extra["llm_output"] = f"Handoff to: {getattr(item, 'target_agent', 'unknown')}"
            return "decision", extra

        # Message items (default)
        raw_item = getattr(item, "raw_item", item)
        content = getattr(raw_item, "content", None)
        if content and isinstance(content, list):
            text_parts = [
                getattr(c, "text", str(c))
                for c in content
                if hasattr(c, "text") or isinstance(c, str)
            ]
            extra["llm_output"] = "\n".join(text_parts) if text_parts else str(content)
        elif isinstance(content, str):
            extra["llm_output"] = content
        else:
            extra["llm_output"] = str(item)

        return "llm_response", extra

    @staticmethod
    def _extract_output(result: Any) -> Any:
        """Extract the final output from a RunResult."""
        final = getattr(result, "final_output", None)
        if final is not None:
            return final
        # Fallback to string representation
        return str(result) if result is not None else None
