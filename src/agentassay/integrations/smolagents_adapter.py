# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""HuggingFace smolagents framework adapter for AgentAssay.

Wraps a ``smolagents`` agent (``CodeAgent``, ``ToolCallingAgent``, or
``MultiStepAgent``) and captures each reasoning step as a ``StepTrace``.
smolagents is HuggingFace's lightweight agent framework (GA December 2024)
that supports both code-generation and tool-calling patterns.

All ``smolagents`` imports are **lazy** — this module can be imported
even when smolagents is not installed.

Usage
-----
>>> from agentassay.integrations import SmolAgentsAdapter
>>> adapter = SmolAgentsAdapter(agent=my_agent, model="Qwen/Qwen2.5-72B")
>>> trace = adapter.run({"task": "Find the population of Tokyo"})
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
    "smolagents adapter requires smolagents. Install with: pip install agentassay[smolagents]"
)


def _check_smolagents_installed() -> None:
    """Verify that smolagents is available."""
    try:
        import smolagents  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class SmolAgentsAdapter(AgentAdapter):
    """Adapter for HuggingFace smolagents agents.

    Parameters
    ----------
    agent
        A smolagents agent instance (``CodeAgent``, ``ToolCallingAgent``,
        or ``MultiStepAgent``).
    model
        LLM model identifier. If ``"unknown"``, tries to infer from the
        agent's ``model`` attribute.
    agent_name
        Human-readable name. Defaults to ``"smolagents-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "smolagents"

    def __init__(
        self,
        agent: Any,
        *,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        resolved_model = model
        if model == "unknown":
            agent_model = getattr(agent, "model", None)
            if agent_model is not None:
                # smolagents model objects have a model_id attribute
                resolved_model = getattr(agent_model, "model_id", str(agent_model)) or "unknown"

        super().__init__(model=resolved_model, agent_name=agent_name, metadata=metadata)
        self._agent = agent

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the smolagents agent and capture an ExecutionTrace.

        Calls ``agent.run(task=...)`` and inspects the agent's step log
        (``agent.logs`` or ``agent.memory``) to build per-step traces.

        Parameters
        ----------
        input_data
            The scenario input. Expects a ``"task"`` or ``"query"`` key
            with the task description.

        Returns
        -------
        ExecutionTrace
            Full trace with reasoning steps, tool calls, and timing.
        """
        _check_smolagents_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            task = self._build_task(input_data)

            run_start = time.perf_counter()
            result = self._agent.run(task)
            run_duration_ms = (time.perf_counter() - run_start) * 1000.0

            steps = self._extract_steps(run_duration_ms)
            total_ms = (time.perf_counter() - overall_start) * 1000.0

            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=steps,
                input_data=input_data,
                output_data=result,
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
                "smolagents adapter failed: %s\n%s",
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
        _check_smolagents_installed()
        return self.run

    # -- Internal: step extraction --------------------------------------------

    def _extract_steps(self, total_run_ms: float) -> list[StepTrace]:
        """Extract StepTrace objects from the smolagents agent's log.

        smolagents agents maintain step logs in different attributes
        depending on the version:
        - ``agent.logs``: list of step dicts (common)
        - ``agent.memory``: memory of past steps
        - ``agent.step_logs``: newer API

        Each step log entry typically contains:
        - ``step_number``: int
        - ``tool_calls``: list of tool call dicts
        - ``model_output`` or ``llm_output``: LLM reasoning
        - ``observations``: tool outputs
        - ``error``: any error in the step
        """
        steps: list[StepTrace] = []

        # Try different log attributes
        log_entries = self._get_log_entries()

        if not log_entries:
            # Fallback: single-step trace
            steps.append(
                StepTrace(
                    step_index=0,
                    action="llm_response",
                    llm_output="(step details not available)",
                    duration_ms=total_run_ms,
                    model=self._model,
                    metadata={"smolagents_fallback": True},
                )
            )
            return steps

        num_entries = len(log_entries)
        per_entry_ms = total_run_ms / max(num_entries, 1)
        step_index = 0

        for entry in log_entries:
            entry_steps = self._classify_log_entry(entry, step_index, per_entry_ms)
            steps.extend(entry_steps)
            step_index += len(entry_steps)

        return steps

    def _get_log_entries(self) -> list[Any]:
        """Retrieve step log entries from the agent."""
        # Try smolagents v1.x+: agent.logs
        logs = getattr(self._agent, "logs", None)
        if logs and isinstance(logs, (list, tuple)):
            return list(logs)

        # Try agent.step_logs
        step_logs = getattr(self._agent, "step_logs", None)
        if step_logs and isinstance(step_logs, (list, tuple)):
            return list(step_logs)

        # Try agent.memory (some versions)
        memory = getattr(self._agent, "memory", None)
        if memory and isinstance(memory, (list, tuple)):
            return list(memory)

        return []

    def _classify_log_entry(
        self, entry: Any, start_index: int, per_entry_ms: float
    ) -> list[StepTrace]:
        """Convert a single smolagents log entry into StepTrace(s).

        A single log entry can produce multiple steps if it contains
        both an LLM reasoning step and one or more tool calls.
        """
        steps: list[StepTrace] = []
        current_index = start_index

        # Handle dict-style entries
        if isinstance(entry, dict):
            return self._classify_dict_entry(entry, start_index, per_entry_ms)

        # Handle object-style entries (ActionStep, etc.)
        entry_type = type(entry).__name__.lower()

        # Extract LLM reasoning/thought
        llm_output = (
            getattr(entry, "model_output", None)
            or getattr(entry, "llm_output", None)
            or getattr(entry, "thought", None)
        )

        if llm_output:
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="llm_response",
                    llm_output=str(llm_output)[:2000],
                    duration_ms=per_entry_ms / 2,
                    model=self._model,
                    metadata={
                        "smolagents_step_type": entry_type,
                        "smolagents_step_number": getattr(entry, "step_number", current_index),
                    },
                )
            )
            current_index += 1

        # Extract tool calls
        tool_calls = getattr(entry, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, (list, tuple)):
            for tc in tool_calls:
                tc_name = getattr(tc, "name", None) or (
                    tc.get("name") if isinstance(tc, dict) else "unknown"
                )
                tc_args = getattr(tc, "arguments", None) or (
                    tc.get("arguments", {}) if isinstance(tc, dict) else {}
                )

                steps.append(
                    StepTrace(
                        step_index=current_index,
                        action="tool_call",
                        tool_name=str(tc_name),
                        tool_input=tc_args if isinstance(tc_args, dict) else {"raw": tc_args},
                        tool_output=None,
                        duration_ms=per_entry_ms / 4,
                        model=self._model,
                        metadata={"smolagents_step_type": "tool_call"},
                    )
                )
                current_index += 1

        # Extract observations (tool outputs)
        observations = getattr(entry, "observations", None)
        if observations and not tool_calls:
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="observation",
                    llm_output=str(observations)[:2000],
                    duration_ms=per_entry_ms / 4,
                    model=self._model,
                    metadata={"smolagents_step_type": "observation"},
                )
            )
            current_index += 1

        # If nothing was extracted, create a generic step
        if not steps:
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="observation",
                    llm_output=str(entry)[:500],
                    duration_ms=per_entry_ms,
                    model=self._model,
                    metadata={"smolagents_step_type": "unknown"},
                )
            )

        return steps

    def _classify_dict_entry(
        self, entry: dict[str, Any], start_index: int, per_entry_ms: float
    ) -> list[StepTrace]:
        """Classify a dict-style log entry."""
        steps: list[StepTrace] = []
        current_index = start_index

        # LLM reasoning
        llm_output = entry.get("model_output") or entry.get("llm_output")
        if llm_output:
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="llm_response",
                    llm_output=str(llm_output)[:2000],
                    duration_ms=per_entry_ms / 2,
                    model=self._model,
                    metadata={"smolagents_step_type": "reasoning"},
                )
            )
            current_index += 1

        # Tool calls
        tool_calls = entry.get("tool_calls", [])
        for tc in tool_calls:
            tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else "unknown"
            tc_args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="tool_call",
                    tool_name=str(tc_name),
                    tool_input=tc_args if isinstance(tc_args, dict) else {"raw": tc_args},
                    duration_ms=per_entry_ms / 4,
                    model=self._model,
                    metadata={"smolagents_step_type": "tool_call"},
                )
            )
            current_index += 1

        if not steps:
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="observation",
                    llm_output=str(entry)[:500],
                    duration_ms=per_entry_ms,
                    model=self._model,
                    metadata={"smolagents_step_type": "dict_fallback"},
                )
            )

        return steps

    @staticmethod
    def _build_task(input_data: dict[str, Any]) -> str:
        """Extract or construct the task from input_data."""
        for key in ("task", "query", "input", "prompt", "message"):
            if key in input_data:
                return str(input_data[key])

        filtered = {k: v for k, v in input_data.items() if k not in ("scenario_id", "metadata")}
        if len(filtered) == 1:
            return str(next(iter(filtered.values())))

        import json

        return json.dumps(filtered, default=str)
