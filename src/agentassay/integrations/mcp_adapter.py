# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""MCP (Model Context Protocol) Tools adapter for AgentAssay.

Wraps an MCP client or an Anthropic client with MCP tools and captures
each tool invocation and resource read as a ``StepTrace``.  MCP defines
a JSON-RPC protocol between clients and servers -- tool calls are
``tools/call`` requests and resource accesses are ``resources/read``
requests.

All ``mcp`` and ``anthropic`` imports are **lazy** -- this module can be
imported even when the MCP SDK is not installed.

Usage
-----
>>> from agentassay.integrations.mcp_adapter import MCPToolsAdapter
>>> adapter = MCPToolsAdapter(client=my_mcp_client, model="claude-sonnet-4-20250514")
>>> trace = adapter.run({"query": "Summarize the latest report"})

With an Anthropic client + MCP tools::

>>> adapter = MCPToolsAdapter.from_anthropic_client(
...     client=anthropic_client,
...     tools=mcp_tool_list,
...     model="claude-sonnet-4-20250514",
... )
>>> trace = adapter.run({"query": "What files are in the project?"})
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from agentassay.core.models import AgentConfig, ExecutionTrace, StepTrace
from agentassay.integrations.base import (
    AgentAdapter,
    FrameworkNotInstalledError,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "MCP Tools adapter requires the MCP Python SDK. Install with: pip install agentassay[mcp]"
)


def _check_mcp_installed() -> None:
    """Verify that the MCP SDK is available, raise clear error if not.

    Raises
    ------
    FrameworkNotInstalledError
        If ``mcp`` is not installed.
    """
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class MCPToolsAdapter(AgentAdapter):
    """Adapter for MCP (Model Context Protocol) tool-using agents.

    Wraps an MCP client instance and instruments its tool calls and
    resource reads.  Each ``tools/call`` invocation becomes a
    ``StepTrace`` with ``action="tool_call"`` and each ``resources/read``
    becomes ``action="retrieval"``.

    There are two usage modes:

    1. **Direct MCP client**: Pass an MCP ``Client`` instance.  The
       adapter intercepts ``call_tool`` and ``read_resource`` calls.
    2. **Anthropic client + MCP tools**: Use the ``from_anthropic_client``
       classmethod to wrap an Anthropic ``Client`` together with a list
       of MCP tool definitions.

    Parameters
    ----------
    client
        An MCP ``Client`` instance, an Anthropic ``Client`` instance, or
        any object with a ``call_tool()`` method.
    tools
        Optional list of MCP tool definitions (dicts with ``name``,
        ``description``, ``inputSchema``).  Required when using the
        Anthropic client mode.
    model
        LLM model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    agent_name
        Human-readable name.  Defaults to ``"mcp-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "mcp"

    def __init__(
        self,
        client: Any,
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, agent_name=agent_name, metadata=metadata)
        self._client = client
        self._tools = tools or []
        self._mode = self._detect_mode(client)

    # -- Factory classmethods -------------------------------------------------

    @classmethod
    def from_anthropic_client(
        cls,
        client: Any,
        tools: list[dict[str, Any]],
        *,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MCPToolsAdapter:
        """Create an adapter from an Anthropic client with MCP tools.

        This classmethod sets up the adapter in Anthropic mode where tool
        calls are executed via the Anthropic messages API with MCP tool
        definitions converted to Anthropic's tool format.

        Parameters
        ----------
        client
            An Anthropic ``Client`` or ``AsyncClient`` instance.
        tools
            List of MCP tool definitions (dicts with ``name``,
            ``description``, ``inputSchema`` keys).
        model
            LLM model identifier.
        agent_name
            Human-readable name.
        metadata
            Arbitrary metadata.

        Returns
        -------
        MCPToolsAdapter
            A configured adapter instance in Anthropic mode.
        """
        meta = metadata or {}
        meta["mcp_mode"] = "anthropic"
        return cls(
            client=client,
            tools=tools,
            model=model,
            agent_name=agent_name or "mcp-anthropic-agent",
            metadata=meta,
        )

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the MCP agent and capture an ExecutionTrace.

        Depending on the mode (direct MCP or Anthropic), dispatches to the
        appropriate execution path.  All tool calls and resource reads are
        captured as individual ``StepTrace`` entries.

        Parameters
        ----------
        input_data
            The scenario input dictionary.  Must contain a ``"query"`` or
            ``"input"`` key with the user message, or the entire dict is
            serialized as the prompt.

        Returns
        -------
        ExecutionTrace
            A complete trace with per-tool-call timing.  On failure,
            ``success=False`` with the error message.
        """
        _check_mcp_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            if self._mode == "anthropic":
                steps, output = self._run_anthropic(input_data)
            else:
                steps, output = self._run_mcp_client(input_data)

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
                metadata={
                    **self._metadata,
                    "mcp_mode": self._mode,
                    "mcp_tools_count": len(self._tools),
                },
            )

        except FrameworkNotInstalledError:
            raise

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "MCP Tools adapter failed: %s\n%s",
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
        _check_mcp_installed()
        return self.run

    def get_config(self) -> AgentConfig:
        """Build an ``AgentConfig`` describing this adapter's agent.

        Uses ``"custom"`` as the ``AgentConfig.framework`` value because
        ``"mcp"`` is not yet in the ``AgentConfig`` framework literal.
        The actual framework identifier is stored in ``metadata["framework"]``.

        Returns
        -------
        AgentConfig
            Configuration with framework, model, and metadata.
        """
        import uuid as _uuid

        return AgentConfig(
            agent_id=str(_uuid.uuid4()),
            name=self._agent_name,
            framework="mcp",
            model=self._model,
            metadata={
                **self._metadata,
                "mcp_mode": self._mode,
                "mcp_tools_count": len(self._tools),
            },
        )

    # -- Internal: mode detection ---------------------------------------------

    @staticmethod
    def _detect_mode(client: Any) -> str:
        """Detect whether the client is an MCP client or Anthropic client.

        Parameters
        ----------
        client
            The client object passed to the constructor.

        Returns
        -------
        str
            ``"mcp"`` for MCP clients, ``"anthropic"`` for Anthropic clients.
        """
        client_type = type(client).__name__.lower()

        # Check for Anthropic client markers
        if "anthropic" in client_type or hasattr(client, "messages"):
            return "anthropic"

        # Default to MCP client mode
        return "mcp"

    # -- Internal: user input -------------------------------------------------

    @staticmethod
    def _build_user_input(input_data: dict[str, Any]) -> str:
        """Extract or construct the user prompt from input_data.

        Parameters
        ----------
        input_data
            The raw scenario input dictionary.

        Returns
        -------
        str
            The user prompt string.
        """
        for key in ("query", "input", "prompt", "message"):
            if key in input_data:
                return str(input_data[key])

        filtered = {k: v for k, v in input_data.items() if k not in ("scenario_id", "metadata")}
        if len(filtered) == 1:
            return str(next(iter(filtered.values())))

        return json.dumps(filtered, default=str)

    # -- Internal: MCP client execution ---------------------------------------

    def _run_mcp_client(self, input_data: dict[str, Any]) -> tuple[list[StepTrace], Any]:
        """Execute using a direct MCP client.

        Calls ``call_tool`` for each tool invocation and ``read_resource``
        for resource reads.  The client is expected to have these methods.

        Parameters
        ----------
        input_data
            The scenario input dictionary.

        Returns
        -------
        tuple[list[StepTrace], Any]
            (ordered list of steps, final output).
        """
        steps: list[StepTrace] = []
        step_index = 0
        user_input = self._build_user_input(input_data)

        # Check if client has a run/invoke method for agentic execution
        run_fn = getattr(self._client, "run", None) or getattr(self._client, "invoke", None)

        if run_fn is not None:
            # Client supports an agentic run method -- instrument it
            step_start = time.perf_counter()
            result = run_fn(user_input)
            duration_ms = (time.perf_counter() - step_start) * 1000.0

            # Try to extract tool calls from the result
            tool_calls = self._extract_tool_calls_from_result(result)
            if tool_calls:
                for tc in tool_calls:
                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action="tool_call",
                            tool_name=tc.get("name", "unknown"),
                            tool_input=tc.get("arguments", {}),
                            tool_output=tc.get("result"),
                            duration_ms=duration_ms / max(len(tool_calls), 1),
                            model=self._model,
                            metadata={"mcp_method": "call_tool"},
                        )
                    )
                    step_index += 1
            else:
                steps.append(
                    StepTrace(
                        step_index=step_index,
                        action="llm_response",
                        llm_input=user_input,
                        llm_output=str(result) if result is not None else None,
                        duration_ms=duration_ms,
                        model=self._model,
                        metadata={"mcp_method": "run"},
                    )
                )
                step_index += 1

            return steps, result

        # Fallback: iterate over available tools and call each one
        # that matches the input pattern
        if self._tools:
            for tool_def in self._tools:
                tool_name = tool_def.get("name", "unknown")
                call_tool_fn = getattr(self._client, "call_tool", None)
                if call_tool_fn is None:
                    break

                step_start = time.perf_counter()
                try:
                    result = call_tool_fn(tool_name, arguments={"input": user_input})
                    duration_ms = (time.perf_counter() - step_start) * 1000.0
                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action="tool_call",
                            tool_name=tool_name,
                            tool_input={"input": user_input},
                            tool_output=self._safe_serialize(result),
                            duration_ms=duration_ms,
                            model=self._model,
                            metadata={"mcp_method": "call_tool"},
                        )
                    )
                except Exception as tool_exc:
                    duration_ms = (time.perf_counter() - step_start) * 1000.0
                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action="tool_call",
                            tool_name=tool_name,
                            tool_input={"input": user_input},
                            tool_output=f"Error: {tool_exc}",
                            duration_ms=duration_ms,
                            model=self._model,
                            metadata={"mcp_method": "call_tool", "error": True},
                        )
                    )
                step_index += 1

        # Check for resource reads
        read_resource_fn = getattr(self._client, "read_resource", None)
        resources = input_data.get("resources", [])
        if read_resource_fn and resources:
            for resource_uri in resources:
                step_start = time.perf_counter()
                try:
                    result = read_resource_fn(resource_uri)
                    duration_ms = (time.perf_counter() - step_start) * 1000.0
                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action="retrieval",
                            tool_name=f"resource:{resource_uri}",
                            tool_input={"uri": resource_uri},
                            tool_output=self._safe_serialize(result),
                            duration_ms=duration_ms,
                            model=self._model,
                            metadata={"mcp_method": "read_resource"},
                        )
                    )
                except Exception as res_exc:
                    duration_ms = (time.perf_counter() - step_start) * 1000.0
                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action="retrieval",
                            tool_name=f"resource:{resource_uri}",
                            tool_input={"uri": resource_uri},
                            tool_output=f"Error: {res_exc}",
                            duration_ms=duration_ms,
                            model=self._model,
                            metadata={
                                "mcp_method": "read_resource",
                                "error": True,
                            },
                        )
                    )
                step_index += 1

        output = steps[-1].tool_output if steps else None
        return steps, output

    # -- Internal: Anthropic client execution ---------------------------------

    def _run_anthropic(self, input_data: dict[str, Any]) -> tuple[list[StepTrace], Any]:
        """Execute using an Anthropic client with MCP tool definitions.

        Delegates to ``mcp_anthropic.run_anthropic()`` which handles the
        full agentic loop (messages API, tool_use blocks, iteration).

        Parameters
        ----------
        input_data
            The scenario input dictionary.

        Returns
        -------
        tuple[list[StepTrace], Any]
            (ordered list of steps, final output text).
        """
        from agentassay.integrations.mcp_anthropic import run_anthropic

        return run_anthropic(
            client=self._client,
            tools=self._tools,
            model=self._model,
            input_data=input_data,
            user_input=self._build_user_input(input_data),
            execute_tool_fn=self._execute_tool,
            safe_serialize_fn=self._safe_serialize,
        )

    # -- Internal: tool execution ---------------------------------------------

    def _execute_tool(self, tool_name: str, tool_input: Any) -> tuple[Any, float]:
        """Execute a tool call, trying the MCP client first.

        Parameters
        ----------
        tool_name
            Name of the tool to call.
        tool_input
            Arguments to pass to the tool.

        Returns
        -------
        tuple[Any, float]
            (tool output, duration in milliseconds).
        """
        # Try MCP client's call_tool method
        call_tool_fn = getattr(self._client, "call_tool", None)
        if call_tool_fn is not None:
            start = time.perf_counter()
            try:
                result = call_tool_fn(tool_name, arguments=tool_input)
                duration = (time.perf_counter() - start) * 1000.0
                return self._safe_serialize(result), duration
            except Exception as exc:
                duration = (time.perf_counter() - start) * 1000.0
                return f"Error: {exc}", duration

        return {"status": "no_executor"}, 0.0

    # -- Internal: helpers ----------------------------------------------------

    @staticmethod
    def _convert_mcp_tools_to_anthropic(
        mcp_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert MCP tool definitions to Anthropic tool format.

        Backward-compatibility shim -- delegates to the extracted module.

        Parameters
        ----------
        mcp_tools
            List of MCP tool definition dicts.

        Returns
        -------
        list[dict[str, Any]]
            Tools in Anthropic API format.
        """
        from agentassay.integrations.mcp_anthropic import (
            convert_mcp_tools_to_anthropic,
        )

        return convert_mcp_tools_to_anthropic(mcp_tools)

    @staticmethod
    def _extract_tool_calls_from_result(result: Any) -> list[dict[str, Any]]:
        """Try to extract tool call information from a run result.

        Parameters
        ----------
        result
            The result from an MCP client ``run()`` or ``invoke()`` call.

        Returns
        -------
        list[dict[str, Any]]
            List of tool call dicts with ``name``, ``arguments``, ``result``.
        """
        # Check for tool_calls attribute
        tool_calls = getattr(result, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            return [
                {
                    "name": getattr(
                        tc, "name", tc.get("name", "unknown") if isinstance(tc, dict) else "unknown"
                    ),
                    "arguments": getattr(
                        tc, "arguments", tc.get("arguments", {}) if isinstance(tc, dict) else {}
                    ),
                    "result": getattr(
                        tc, "result", tc.get("result") if isinstance(tc, dict) else None
                    ),
                }
                for tc in tool_calls
            ]

        # Check for steps/events attribute
        events = getattr(result, "steps", None) or getattr(result, "events", None)
        if events and isinstance(events, list):
            calls = []
            for ev in events:
                if hasattr(ev, "tool_name") or (isinstance(ev, dict) and "tool_name" in ev):
                    name = (
                        getattr(ev, "tool_name", ev.get("tool_name"))
                        if isinstance(ev, dict)
                        else getattr(ev, "tool_name", "unknown")
                    )
                    args = (
                        getattr(ev, "arguments", ev.get("arguments", {}))
                        if isinstance(ev, dict)
                        else getattr(ev, "arguments", {})
                    )
                    res = (
                        getattr(ev, "result", ev.get("result"))
                        if isinstance(ev, dict)
                        else getattr(ev, "result", None)
                    )
                    calls.append({"name": name, "arguments": args, "result": res})
            if calls:
                return calls

        return []

    @staticmethod
    def _safe_serialize(value: Any) -> Any:
        """Safely serialize a value for storage in StepTrace fields.

        Parameters
        ----------
        value
            Any value to serialize.

        Returns
        -------
        Any
            The value if it's JSON-serializable, otherwise its string
            representation.
        """
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (dict, list)):
            try:
                json.dumps(value, default=str)
                return value
            except (TypeError, ValueError):
                return str(value)
        return str(value)
