"""CrewAI framework adapter for AgentAssay.

Wraps a CrewAI ``Crew`` instance and captures each task execution as a
``StepTrace``. CrewAI organizes work as a sequence of tasks assigned to
agents within a crew — each task result becomes one step in the trace.

All ``crewai`` imports are **lazy** — this module can be imported even
when CrewAI is not installed. The ``ImportError`` is raised only when
``run()`` or ``to_callable()`` is actually called.

Usage
-----
>>> from agentassay.integrations import CrewAIAdapter
>>> adapter = CrewAIAdapter(crew=my_crew, model="gpt-4o")
>>> trace = adapter.run({"topic": "quantum computing trends"})
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
    "CrewAI adapter requires crewai. "
    "Install with: pip install agentassay[crewai]"
)


def _check_crewai_installed() -> None:
    """Verify that crewai is available, raise clear error if not."""
    try:
        import crewai  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class CrewAIAdapter(AgentAdapter):
    """Adapter for CrewAI Crew instances.

    Parameters
    ----------
    crew
        A CrewAI ``Crew`` instance with tasks and agents configured.
    model
        LLM model identifier (e.g. ``"gpt-4o"``).
    agent_name
        Human-readable name for this crew. Defaults to ``"crewai-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "crewai"

    def __init__(
        self,
        crew: Any,
        *,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, agent_name=agent_name, metadata=metadata)
        self._crew = crew

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the CrewAI crew and capture an ExecutionTrace.

        Calls ``crew.kickoff(inputs=input_data)`` and extracts task results
        as individual steps. Each task in the crew maps to one ``StepTrace``.

        Parameters
        ----------
        input_data
            The scenario input dictionary passed to ``crew.kickoff()``.

        Returns
        -------
        ExecutionTrace
            A complete trace with per-task steps, timing, and final output.
            On failure, ``success=False`` with the error message.
        """
        _check_crewai_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            kickoff_start = time.perf_counter()
            result = self._crew.kickoff(inputs=input_data)
            kickoff_duration_ms = (time.perf_counter() - kickoff_start) * 1000.0

            steps = self._extract_steps(result, kickoff_duration_ms)
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
                total_cost_usd=self._extract_cost(result),
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
                "CrewAI adapter failed: %s\n%s",
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
        _check_crewai_installed()
        return self.run

    # -- Internal: step extraction --------------------------------------------

    def _extract_steps(
        self, result: Any, total_kickoff_ms: float
    ) -> list[StepTrace]:
        """Extract StepTrace objects from a CrewAI kickoff result.

        CrewAI's ``CrewOutput`` (v0.80+) exposes ``tasks_output`` — a list
        of ``TaskOutput`` objects, each with agent info, description, and
        raw output. We map each to a step.

        If the result structure is different (older CrewAI), we fall back
        to a single-step trace.
        """
        steps: list[StepTrace] = []

        # CrewAI v0.80+: result has tasks_output attribute
        tasks_output = getattr(result, "tasks_output", None)
        if tasks_output and isinstance(tasks_output, (list, tuple)):
            num_tasks = len(tasks_output)
            # Distribute total time evenly across tasks as approximation
            per_task_ms = total_kickoff_ms / max(num_tasks, 1)

            for idx, task_out in enumerate(tasks_output):
                action, step_kwargs = self._classify_task_output(task_out, idx)
                steps.append(
                    StepTrace(
                        step_index=idx,
                        action=action,
                        duration_ms=per_task_ms,
                        model=self._model,
                        metadata={
                            "crewai_task_index": idx,
                            "crewai_agent": getattr(
                                task_out, "agent", "unknown"
                            ),
                        },
                        **step_kwargs,
                    )
                )
            return steps

        # Fallback: single-step trace from the raw result
        steps.append(
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=str(result) if result is not None else None,
                duration_ms=total_kickoff_ms,
                model=self._model,
                metadata={"crewai_fallback": True},
            )
        )
        return steps

    @staticmethod
    def _classify_task_output(
        task_out: Any, index: int
    ) -> tuple[str, dict[str, Any]]:
        """Classify a CrewAI TaskOutput into a StepTrace action."""
        extra: dict[str, Any] = {}

        description = getattr(task_out, "description", "")
        raw_output = getattr(task_out, "raw", None)
        pydantic_output = getattr(task_out, "pydantic", None)

        # Check for tool usage in the task output
        # CrewAI tasks with tools will have tool_calls in their execution log
        tools_used = getattr(task_out, "tools_used", None)
        if tools_used:
            extra["tool_name"] = (
                tools_used[0] if isinstance(tools_used, list) else str(tools_used)
            )
            extra["tool_input"] = {"description": str(description)[:200]}
            extra["tool_output"] = str(raw_output)[:500] if raw_output else None
            return "tool_call", extra

        # Standard task output — treat as LLM response
        output_str = str(pydantic_output or raw_output or "")
        extra["llm_input"] = str(description)[:500] if description else None
        extra["llm_output"] = output_str[:2000] if output_str else None
        return "llm_response", extra

    @staticmethod
    def _extract_output(result: Any) -> Any:
        """Extract the final output from a CrewAI result."""
        # CrewAI v0.80+: result.raw or result.pydantic
        if hasattr(result, "raw"):
            return result.raw
        if hasattr(result, "pydantic"):
            return result.pydantic
        return str(result) if result is not None else None

    @staticmethod
    def _extract_cost(result: Any) -> float:
        """Extract total cost from a CrewAI result if available."""
        # CrewAI tracks usage via result.token_usage
        usage = getattr(result, "token_usage", None)
        if usage and hasattr(usage, "total_cost"):
            cost = getattr(usage, "total_cost", 0.0)
            return float(cost) if cost else 0.0
        return 0.0
