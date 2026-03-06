"""Custom callable adapter for AgentAssay.

The escape hatch: wraps any Python callable into an ``AgentAdapter`` so
that any agent — regardless of framework — can be tested with AgentAssay.

Three return-type modes are supported:
1. **ExecutionTrace** — the callable already returns a full trace. Used as-is.
2. **dict** — the callable returns a dict. Wrapped into an ExecutionTrace.
3. **str** — the callable returns a string. Wrapped as a single-step trace.

This is the recommended adapter when:
- Using a framework we don't have a dedicated adapter for.
- The agent is a simple function or pipeline.
- You need full control over trace construction.

Usage
-----
>>> from agentassay.integrations import CustomAdapter
>>>
>>> def my_agent(input_data: dict) -> str:
...     return "The answer is 42"
>>>
>>> adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")
>>> trace = adapter.run({"query": "What is the meaning of life?"})
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
from agentassay.integrations.base import AgentAdapter

logger = logging.getLogger(__name__)


class CustomAdapter(AgentAdapter):
    """Adapter that wraps any callable into an AgentAdapter.

    Parameters
    ----------
    callable_fn
        A function with signature ``(dict[str, Any]) -> T`` where ``T``
        is one of:

        - ``ExecutionTrace``: returned directly (zero overhead).
        - ``dict``: wrapped into an ``ExecutionTrace``. Expected keys:

          - ``"output"`` or ``"result"``: the agent output (required).
          - ``"steps"``: optional list of step dicts.
          - ``"cost"``: optional float cost in USD.
          - ``"success"``: optional bool (default ``True``).
          - ``"error"``: optional error string.

        - ``str``: wrapped as a single-step ``llm_response`` trace.
        - Anything else: wrapped via ``str(result)`` as a single step.

    framework
        Framework identifier. Defaults to ``"custom"``.
    model
        LLM model identifier. Defaults to ``"unknown"``.
    agent_name
        Human-readable name. Defaults to ``"custom-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "custom"

    def __init__(
        self,
        callable_fn: Callable[[dict[str, Any]], Any],
        *,
        framework: str = "custom",
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Override the class-level framework with the instance-level one
        self.framework = framework
        super().__init__(
            model=model,
            agent_name=agent_name or "custom-agent",
            metadata=metadata,
        )
        self._callable_fn = callable_fn

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the wrapped callable and return an ExecutionTrace.

        The callable is invoked with ``input_data``. The return value
        is normalized into an ``ExecutionTrace`` based on its type.

        Parameters
        ----------
        input_data
            The scenario input dictionary passed directly to the callable.

        Returns
        -------
        ExecutionTrace
            A trace wrapping the callable's output. On failure,
            ``success=False`` with the error message.
        """
        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            step_start = time.perf_counter()
            result = self._callable_fn(input_data)
            call_duration_ms = (time.perf_counter() - step_start) * 1000.0

            trace = self._normalize_result(
                result=result,
                input_data=input_data,
                scenario_id=scenario_id,
                trace_id=trace_id,
                call_duration_ms=call_duration_ms,
            )

            # If _normalize_result produced a trace with default duration,
            # update to the actual total
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            if trace.total_duration_ms == 0.0:
                # Frozen model — rebuild with correct duration
                trace = ExecutionTrace(
                    trace_id=trace.trace_id,
                    scenario_id=trace.scenario_id,
                    steps=trace.steps,
                    input_data=trace.input_data,
                    output_data=trace.output_data,
                    success=trace.success,
                    error=trace.error,
                    total_duration_ms=total_ms,
                    total_cost_usd=trace.total_cost_usd,
                    model=trace.model,
                    framework=trace.framework,
                    metadata=trace.metadata,
                )

            return trace

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Custom adapter failed: %s\n%s",
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
        return self.run

    # -- Internal: result normalization ---------------------------------------

    def _normalize_result(
        self,
        result: Any,
        input_data: dict[str, Any],
        scenario_id: str,
        trace_id: str,
        call_duration_ms: float,
    ) -> ExecutionTrace:
        """Convert the callable's return value into an ExecutionTrace.

        Dispatches based on return type:
        - ExecutionTrace: returned as-is.
        - dict: fields extracted and mapped.
        - str: wrapped as a single-step trace.
        - other: str(result) wrapped as a single-step trace.
        """
        # Case 1: Already an ExecutionTrace
        if isinstance(result, ExecutionTrace):
            return result

        # Case 2: Dict result
        if isinstance(result, dict):
            return self._from_dict(
                result, input_data, scenario_id, trace_id, call_duration_ms
            )

        # Case 3: String result
        if isinstance(result, str):
            return self._from_string(
                result, input_data, scenario_id, trace_id, call_duration_ms
            )

        # Case 4: Any other type — stringify
        return self._from_string(
            str(result) if result is not None else "",
            input_data,
            scenario_id,
            trace_id,
            call_duration_ms,
        )

    def _from_dict(
        self,
        result: dict[str, Any],
        input_data: dict[str, Any],
        scenario_id: str,
        trace_id: str,
        call_duration_ms: float,
    ) -> ExecutionTrace:
        """Convert a dict return value into an ExecutionTrace.

        Expected dict shape::

            {
                "output": Any,          # required
                "steps": [...],         # optional list of step dicts
                "cost": float,          # optional
                "success": bool,        # optional, default True
                "error": str | None,    # optional
                "metadata": dict,       # optional
            }

        Each step dict can have::

            {
                "action": str,          # default "llm_response"
                "tool_name": str,
                "tool_input": dict,
                "tool_output": Any,
                "llm_input": str,
                "llm_output": str,
                "duration_ms": float,
                "metadata": dict,
            }
        """
        output = result.get("output") or result.get("result")
        success = result.get("success", True)
        error = result.get("error")
        cost = float(result.get("cost", 0.0))
        extra_metadata = result.get("metadata", {})

        # Build steps
        raw_steps = result.get("steps", [])
        steps: list[StepTrace] = []
        if raw_steps and isinstance(raw_steps, list):
            for idx, step_dict in enumerate(raw_steps):
                if isinstance(step_dict, StepTrace):
                    steps.append(step_dict)
                elif isinstance(step_dict, dict):
                    steps.append(self._dict_to_step(step_dict, idx))
        else:
            # No steps provided: create a single step from output
            steps.append(
                StepTrace(
                    step_index=0,
                    action="llm_response",
                    llm_output=str(output)[:2000] if output is not None else None,
                    duration_ms=call_duration_ms,
                    model=self._model,
                )
            )

        combined_metadata = {**self._metadata, **extra_metadata}

        return ExecutionTrace(
            trace_id=trace_id,
            scenario_id=scenario_id,
            steps=steps,
            input_data=input_data,
            output_data=output,
            success=success,
            error=error,
            total_duration_ms=call_duration_ms,
            total_cost_usd=cost,
            model=self._model,
            framework=self.framework,
            metadata=combined_metadata,
        )

    def _from_string(
        self,
        result: str,
        input_data: dict[str, Any],
        scenario_id: str,
        trace_id: str,
        call_duration_ms: float,
    ) -> ExecutionTrace:
        """Convert a string return value into a single-step ExecutionTrace."""
        steps = [
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=result[:2000] if result else None,
                duration_ms=call_duration_ms,
                model=self._model,
            )
        ]

        return ExecutionTrace(
            trace_id=trace_id,
            scenario_id=scenario_id,
            steps=steps,
            input_data=input_data,
            output_data=result,
            success=True,
            error=None,
            total_duration_ms=call_duration_ms,
            total_cost_usd=0.0,
            model=self._model,
            framework=self.framework,
            metadata=self._metadata,
        )

    def _dict_to_step(
        self, step_dict: dict[str, Any], index: int
    ) -> StepTrace:
        """Convert a raw step dict into a StepTrace."""
        action = step_dict.get("action", "llm_response")
        duration = float(step_dict.get("duration_ms", 0.0))
        step_metadata = step_dict.get("metadata", {})

        kwargs: dict[str, Any] = {}

        # Tool fields
        if step_dict.get("tool_name"):
            kwargs["tool_name"] = step_dict["tool_name"]
        if step_dict.get("tool_input"):
            kwargs["tool_input"] = step_dict["tool_input"]
        if "tool_output" in step_dict:
            kwargs["tool_output"] = step_dict["tool_output"]

        # LLM fields
        if step_dict.get("llm_input"):
            kwargs["llm_input"] = step_dict["llm_input"]
        if step_dict.get("llm_output"):
            kwargs["llm_output"] = step_dict["llm_output"]

        return StepTrace(
            step_index=index,
            action=action,
            duration_ms=duration,
            model=step_dict.get("model", self._model),
            metadata=step_metadata,
            **kwargs,
        )
