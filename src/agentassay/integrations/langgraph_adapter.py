# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""LangGraph framework adapter for AgentAssay.

Wraps a LangGraph ``CompiledGraph`` and captures each node execution as a
``StepTrace``. Uses the streaming interface (``graph.stream()``) to get
per-node granularity, falling back to ``graph.invoke()`` if streaming is
not available.

All ``langgraph`` imports are **lazy** — this module can be imported even
when LangGraph is not installed. The ``ImportError`` is raised only when
``run()`` or ``to_callable()`` is actually called.

Usage
-----
>>> from agentassay.integrations import LangGraphAdapter
>>> adapter = LangGraphAdapter(graph=my_compiled_graph, model="gpt-4o")
>>> trace = adapter.run({"query": "What is the capital of France?"})
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
    "LangGraph adapter requires langgraph. Install with: pip install agentassay[langgraph]"
)


def _check_langgraph_installed() -> None:
    """Verify that langgraph is available, raise clear error if not."""
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class LangGraphAdapter(AgentAdapter):
    """Adapter for LangGraph CompiledGraph instances.

    Parameters
    ----------
    graph
        A LangGraph ``CompiledGraph`` instance (the result of
        ``StateGraph(...).compile()``).
    model
        LLM model identifier (e.g. ``"gpt-4o"``, ``"claude-3.5-sonnet"``).
    config
        Optional LangGraph invocation config (passed to ``graph.invoke()``
        or ``graph.stream()``). Can include ``recursion_limit``,
        ``configurable``, callbacks, etc.
    use_stream
        If ``True`` (default), use ``graph.stream()`` for per-node step
        capture. If ``False``, use ``graph.invoke()`` and produce a
        single-step trace.
    agent_name
        Human-readable name for the agent. Defaults to ``"langgraph-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "langgraph"

    def __init__(
        self,
        graph: Any,
        *,
        model: str = "unknown",
        config: dict[str, Any] | None = None,
        use_stream: bool = True,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, agent_name=agent_name, metadata=metadata)
        self._graph = graph
        self._config = config or {}
        self._use_stream = use_stream

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the LangGraph graph and capture an ExecutionTrace.

        When ``use_stream=True``, each streamed chunk maps to a ``StepTrace``
        representing the node that produced it. When ``use_stream=False``,
        the full invocation is captured as a single step.

        Parameters
        ----------
        input_data
            The scenario input dictionary passed as graph state.

        Returns
        -------
        ExecutionTrace
            A complete trace with per-node steps, timing, and output.
            On failure, ``success=False`` with the error message.
        """
        _check_langgraph_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            if self._use_stream:
                steps, output = self._run_stream(input_data)
            else:
                steps, output = self._run_invoke(input_data)

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
                total_cost_usd=0.0,  # LangGraph doesn't expose cost natively
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

        except FrameworkNotInstalledError:
            raise  # Do not swallow install errors

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "LangGraph adapter failed: %s\n%s",
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
        """Return a TrialRunner-compatible callable.

        Returns
        -------
        Callable[[dict[str, Any]], ExecutionTrace]
            A closure that calls ``self.run(input_data)``.
        """
        _check_langgraph_installed()
        return self.run

    # -- Internal: streaming execution ----------------------------------------

    def _run_stream(self, input_data: dict[str, Any]) -> tuple[list[StepTrace], Any]:
        """Execute via ``graph.stream()`` and collect per-node steps.

        Each streamed event is a dict mapping node-name to node-output.
        We create one ``StepTrace`` per node event.
        """
        steps: list[StepTrace] = []
        last_output: Any = None
        step_index = 0

        for event in self._graph.stream(input_data, config=self._config):
            step_start = time.perf_counter()

            # LangGraph stream events: {node_name: node_output}
            if isinstance(event, dict):
                for node_name, node_output in event.items():
                    duration_ms = (time.perf_counter() - step_start) * 1000.0
                    last_output = node_output

                    # Determine step action from node content
                    action, step_kwargs = self._classify_node_output(node_name, node_output)

                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action=action,
                            duration_ms=duration_ms,
                            model=self._model,
                            metadata={"langgraph_node": node_name},
                            **step_kwargs,
                        )
                    )
                    step_index += 1
            else:
                # Non-dict event: wrap as observation
                duration_ms = (time.perf_counter() - step_start) * 1000.0
                steps.append(
                    StepTrace(
                        step_index=step_index,
                        action="observation",
                        llm_output=str(event) if event is not None else None,
                        duration_ms=duration_ms,
                        model=self._model,
                    )
                )
                last_output = event
                step_index += 1

        return steps, last_output

    # -- Internal: invoke execution -------------------------------------------

    def _run_invoke(self, input_data: dict[str, Any]) -> tuple[list[StepTrace], Any]:
        """Execute via ``graph.invoke()`` — single step, no streaming."""
        step_start = time.perf_counter()
        result = self._graph.invoke(input_data, config=self._config)
        duration_ms = (time.perf_counter() - step_start) * 1000.0

        steps = [
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=str(result) if result is not None else None,
                duration_ms=duration_ms,
                model=self._model,
                metadata={"mode": "invoke"},
            )
        ]
        return steps, result

    # -- Internal: heuristic node classification ------------------------------

    @staticmethod
    def _classify_node_output(node_name: str, node_output: Any) -> tuple[str, dict[str, Any]]:
        """Classify a LangGraph node output into a StepTrace action type.

        Applies heuristics:
        - If the output contains ``tool_calls`` or the node name contains
          "tool", classify as ``tool_call``.
        - If the output contains ``content`` (like an AIMessage), classify
          as ``llm_response``.
        - Otherwise, classify as ``observation``.

        Returns
        -------
        tuple[str, dict]
            (action_type, extra_kwargs_for_StepTrace)
        """
        extra: dict[str, Any] = {}

        # Handle dict outputs (most common in LangGraph)
        if isinstance(node_output, dict):
            # Check for messages in the output (common LangGraph pattern)
            messages = node_output.get("messages", [])
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                return _classify_message(last_msg, node_name, extra)

            # Check for tool-related keys
            if "tool_calls" in node_output or "tool" in node_name.lower():
                extra["tool_name"] = node_output.get("tool", node_name)
                extra["tool_input"] = node_output.get("input", node_output.get("args"))
                extra["tool_output"] = node_output.get("output")
                return "tool_call", extra

            # Generic dict output — treat as observation
            extra["llm_output"] = str(node_output)
            return "observation", extra

        # Handle message-like objects (AIMessage, HumanMessage, ToolMessage)
        if hasattr(node_output, "content"):
            return _classify_message(node_output, node_name, extra)

        # Fallback
        extra["llm_output"] = str(node_output) if node_output is not None else None
        return "observation", extra


def _classify_message(
    msg: Any, node_name: str, extra: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Classify a LangChain-style message object into a step action."""
    msg_type = type(msg).__name__.lower()
    content = getattr(msg, "content", str(msg))

    # ToolMessage → tool_call
    if "tool" in msg_type:
        extra["tool_name"] = getattr(msg, "name", node_name)
        extra["tool_output"] = content
        # Provide a dummy tool_input for ToolMessages from tool responses
        extra["tool_input"] = {"source": node_name}
        return "tool_call", extra

    # AIMessage with tool_calls → tool_call (the agent requesting a tool)
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        first_call = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
        if isinstance(first_call, dict):
            extra["tool_name"] = first_call.get("name", node_name)
            extra["tool_input"] = first_call.get("args", {})
        else:
            extra["tool_name"] = getattr(first_call, "name", node_name)
            extra["tool_input"] = getattr(first_call, "args", {})
        return "tool_call", extra

    # AIMessage without tool_calls → llm_response
    if "ai" in msg_type or "assistant" in msg_type:
        extra["llm_output"] = content
        return "llm_response", extra

    # HumanMessage or other → observation
    extra["llm_input"] = content
    return "observation", extra
