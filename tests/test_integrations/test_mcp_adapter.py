"""Tests for the MCP (Model Context Protocol) Tools adapter.

Validates adapter creation, MCP install checks, tool call capture,
resource read capture, Anthropic client mode, from_anthropic_client
classmethod, error handling, and TrialRunner compatibility.

All tests use ``unittest.mock`` -- mcp and anthropic are NEVER imported
for real.

Target: 20+ tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentassay.core.models import ExecutionTrace, StepTrace
from agentassay.integrations.base import FrameworkNotInstalledError


# ===================================================================
# Helpers
# ===================================================================


def _make_adapter(**kwargs: Any):
    """Create an MCPToolsAdapter with sensible defaults."""
    from agentassay.integrations.mcp_adapter import MCPToolsAdapter

    defaults: dict[str, Any] = {
        "client": MagicMock(),
        "model": "claude-sonnet-4-20250514",
        "agent_name": "test-mcp-agent",
    }
    defaults.update(kwargs)
    return MCPToolsAdapter(**defaults)


def _make_mcp_tool(
    name: str = "get_weather",
    description: str = "Get weather for a location",
    input_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an MCP tool definition dict."""
    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema or {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    }


def _make_anthropic_response(
    content_blocks: list[dict[str, Any]],
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a mocked Anthropic messages.create response."""
    response = MagicMock()

    blocks = []
    for block_def in content_blocks:
        block = MagicMock()
        block.type = block_def["type"]
        if block_def["type"] == "text":
            block.text = block_def["text"]
        elif block_def["type"] == "tool_use":
            block.name = block_def["name"]
            block.input = block_def.get("input", {})
            block.id = block_def.get("id", "toolu_test123")
        blocks.append(block)

    response.content = blocks
    response.stop_reason = stop_reason
    return response


# ===================================================================
# TestMCPToolsAdapter
# ===================================================================


class TestMCPToolsAdapter:
    """Tests for MCPToolsAdapter."""

    # -- Construction -------------------------------------------------------

    def test_create_adapter(self):
        """Adapter can be created without mcp installed."""
        adapter = _make_adapter()
        assert adapter.framework == "mcp"
        assert adapter.model == "claude-sonnet-4-20250514"
        assert adapter.agent_name == "test-mcp-agent"

    def test_default_agent_name(self):
        """Default agent name uses framework prefix."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        adapter = MCPToolsAdapter(client=MagicMock())
        assert adapter.agent_name == "mcp-agent"

    def test_create_with_tools(self):
        """Adapter stores MCP tool definitions."""
        tools = [_make_mcp_tool("search"), _make_mcp_tool("calculate")]
        adapter = _make_adapter(tools=tools)
        assert len(adapter._tools) == 2

    def test_repr(self):
        adapter = _make_adapter()
        r = repr(adapter)
        assert "MCPToolsAdapter" in r
        assert "mcp" in r

    # -- Install check ------------------------------------------------------

    def test_check_mcp_not_installed(self):
        """_check_mcp_installed raises FrameworkNotInstalledError."""
        from agentassay.integrations.mcp_adapter import _check_mcp_installed

        with patch.dict("sys.modules", {"mcp": None}):
            with pytest.raises(FrameworkNotInstalledError, match="MCP"):
                _check_mcp_installed()

    def test_check_mcp_installed_ok(self):
        """_check_mcp_installed succeeds when mcp is importable."""
        from agentassay.integrations.mcp_adapter import _check_mcp_installed

        mock_mcp = MagicMock()
        with patch.dict("sys.modules", {"mcp": mock_mcp}):
            _check_mcp_installed()  # Should not raise

    # -- Mode detection -----------------------------------------------------

    def test_detect_mode_mcp_client(self):
        """MCP client is detected as 'mcp' mode."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        # Use spec=[] to prevent MagicMock from auto-creating .messages
        client = MagicMock(spec=["call_tool", "read_resource"])
        client.__class__.__name__ = "MCPClient"
        assert MCPToolsAdapter._detect_mode(client) == "mcp"

    def test_detect_mode_anthropic_client(self):
        """Anthropic client is detected as 'anthropic' mode."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        client = MagicMock()
        client.__class__.__name__ = "AnthropicClient"
        assert MCPToolsAdapter._detect_mode(client) == "anthropic"

    def test_detect_mode_via_messages_attr(self):
        """Client with .messages attribute is detected as 'anthropic'."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        client = MagicMock(spec=["messages"])
        client.__class__.__name__ = "CustomClient"
        assert MCPToolsAdapter._detect_mode(client) == "anthropic"

    # -- from_anthropic_client classmethod -----------------------------------

    def test_from_anthropic_client(self):
        """from_anthropic_client creates adapter in anthropic mode."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        client = MagicMock()
        tools = [_make_mcp_tool("search")]
        adapter = MCPToolsAdapter.from_anthropic_client(
            client=client,
            tools=tools,
            model="claude-sonnet-4-20250514",
            agent_name="my-agent",
        )
        assert adapter.framework == "mcp"
        assert adapter.agent_name == "my-agent"
        assert adapter._tools == tools
        assert adapter._metadata.get("mcp_mode") == "anthropic"

    def test_from_anthropic_client_default_name(self):
        """from_anthropic_client defaults agent_name to 'mcp-anthropic-agent'."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        adapter = MCPToolsAdapter.from_anthropic_client(
            client=MagicMock(), tools=[]
        )
        assert adapter.agent_name == "mcp-anthropic-agent"

    # -- run(): MCP client with run() method --------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_mcp_client_with_run_method(self, mock_check):
        """run() uses client.run() when available."""
        client = MagicMock()
        client.__class__.__name__ = "MCPClient"
        # Return a simple result from run()
        mock_result = MagicMock()
        mock_result.tool_calls = None
        mock_result.steps = None
        mock_result.events = None
        del mock_result.tool_calls
        del mock_result.steps
        del mock_result.events
        client.run.return_value = "The weather is sunny."

        adapter = _make_adapter(client=client)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "What's the weather?"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True
        assert trace.framework == "mcp"
        assert len(trace.steps) >= 1
        client.run.assert_called_once()

    # -- run(): MCP client with tool calls in result ------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_mcp_client_with_tool_calls_result(self, mock_check):
        """Tool calls in run() result are extracted as steps."""
        client = MagicMock()
        client.__class__.__name__ = "MCPClient"

        # Result with tool_calls attribute
        mock_tc = MagicMock()
        mock_tc.name = "get_weather"
        mock_tc.arguments = {"location": "Paris"}
        mock_tc.result = "Sunny, 22C"

        mock_result = MagicMock()
        mock_result.tool_calls = [mock_tc]
        client.run.return_value = mock_result

        adapter = _make_adapter(client=client)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "Weather in Paris?"})

        assert trace.success is True
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) == 1
        assert tool_steps[0].tool_name == "get_weather"

    # -- run(): MCP client with call_tool ----------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_mcp_client_call_tool(self, mock_check):
        """Tools are called via client.call_tool() when no run() method."""
        client = MagicMock(spec=["call_tool"])
        client.__class__.__name__ = "MCPClient"
        client.call_tool.return_value = {"result": "42"}

        tools = [_make_mcp_tool("calculate")]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "2+2?"})

        assert trace.success is True
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) == 1
        assert tool_steps[0].tool_name == "calculate"
        client.call_tool.assert_called_once()

    # -- run(): MCP client with resource reads ------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_mcp_client_resource_reads(self, mock_check):
        """Resource reads are captured as retrieval steps."""
        client = MagicMock(spec=["read_resource"])
        client.__class__.__name__ = "MCPClient"
        client.read_resource.return_value = {"content": "File contents here."}

        adapter = _make_adapter(client=client, tools=[])
        adapter._mode = "mcp"

        trace = adapter.run({
            "query": "Read the file",
            "resources": ["file:///path/to/document.md"],
        })

        assert trace.success is True
        retrieval_steps = [s for s in trace.steps if s.action == "retrieval"]
        assert len(retrieval_steps) == 1
        assert "document.md" in retrieval_steps[0].tool_name
        client.read_resource.assert_called_once()

    # -- run(): Anthropic client mode ---------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_anthropic_mode_text_only(self, mock_check):
        """Anthropic mode with text-only response produces llm_response step."""
        client = MagicMock()
        client.__class__.__name__ = "AnthropicClient"
        client.messages.create.return_value = _make_anthropic_response([
            {"type": "text", "text": "Hello! How can I help?"},
        ])

        tools = [_make_mcp_tool("search")]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "anthropic"

        trace = adapter.run({"query": "Hello"})

        assert trace.success is True
        assert len(trace.steps) >= 1
        llm_steps = [s for s in trace.steps if s.action == "llm_response"]
        assert len(llm_steps) >= 1
        assert "Hello" in llm_steps[0].llm_output

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_anthropic_mode_with_tool_use(self, mock_check):
        """Anthropic mode with tool_use block produces tool_call step."""
        client = MagicMock()
        client.__class__.__name__ = "AnthropicClient"

        # First call: tool_use response
        tool_response = _make_anthropic_response(
            [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                    "id": "toolu_abc123",
                },
            ],
            stop_reason="tool_use",
        )

        # Second call: final text response after tool result
        final_response = _make_anthropic_response([
            {"type": "text", "text": "It's sunny in Tokyo, 25C."},
        ])

        client.messages.create.side_effect = [tool_response, final_response]

        # Mock call_tool on client for tool execution
        client.call_tool.return_value = {"temperature": "25C", "condition": "sunny"}

        tools = [_make_mcp_tool("get_weather")]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "anthropic"

        trace = adapter.run({"query": "Weather in Tokyo?"})

        assert trace.success is True
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) >= 1
        assert tool_steps[0].tool_name == "get_weather"

    # -- run(): error handling ----------------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_exception_returns_failed_trace(self, mock_check):
        """Exceptions during execution produce success=False traces."""
        client = MagicMock()
        client.__class__.__name__ = "MCPClient"
        client.run.side_effect = ConnectionError("Server unreachable")

        adapter = _make_adapter(client=client)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "test"})

        assert trace.success is False
        assert "ConnectionError" in trace.error
        assert "Server unreachable" in trace.error
        assert trace.steps == []

    def test_run_without_mcp_raises(self):
        """run() raises FrameworkNotInstalledError when mcp missing."""
        adapter = _make_adapter()

        with patch.dict("sys.modules", {"mcp": None}):
            with pytest.raises(FrameworkNotInstalledError, match="MCP"):
                adapter.run({"query": "test"})

    # -- run(): tool call error handling ------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_tool_call_error_captured(self, mock_check):
        """Errors during tool calls are captured, not raised."""
        client = MagicMock(spec=["call_tool"])
        client.__class__.__name__ = "MCPClient"
        client.call_tool.side_effect = RuntimeError("Tool crashed")

        tools = [_make_mcp_tool("broken_tool")]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "test"})

        assert trace.success is True  # Trace succeeds, tool error is captured
        assert len(trace.steps) == 1
        assert "Error" in str(trace.steps[0].tool_output)

    # -- run(): empty tools list --------------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_empty_tools_list(self, mock_check):
        """Empty tool list with no run method produces empty trace."""
        client = MagicMock(spec=[])
        client.__class__.__name__ = "MCPClient"

        adapter = _make_adapter(client=client, tools=[])
        adapter._mode = "mcp"

        trace = adapter.run({"query": "Hello"})

        assert trace.success is True
        assert trace.output_data is None

    # -- run(): multiple tools in sequence ----------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_multiple_tools_sequence(self, mock_check):
        """Multiple tools are called in sequence."""
        client = MagicMock(spec=["call_tool"])
        client.__class__.__name__ = "MCPClient"
        client.call_tool.side_effect = [
            {"temp": "22C"},
            {"result": "3.14"},
        ]

        tools = [
            _make_mcp_tool("get_weather"),
            _make_mcp_tool("calculate"),
        ]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "weather and math"})

        assert trace.success is True
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) == 2
        assert tool_steps[0].tool_name == "get_weather"
        assert tool_steps[1].tool_name == "calculate"

    # -- to_callable --------------------------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_to_callable_returns_run(self, mock_check):
        """to_callable returns self.run."""
        adapter = _make_adapter()
        fn = adapter.to_callable()
        assert callable(fn)
        assert fn == adapter.run

    def test_to_callable_without_mcp_raises(self):
        """to_callable raises when mcp is missing."""
        adapter = _make_adapter()

        with patch.dict("sys.modules", {"mcp": None}):
            with pytest.raises(FrameworkNotInstalledError, match="MCP"):
                adapter.to_callable()

    # -- get_config ---------------------------------------------------------

    def test_get_config(self):
        """get_config returns an AgentConfig with correct values.

        Uses 'custom' as framework since 'mcp' is not in the
        AgentConfig Literal.  The real framework is in metadata.
        """
        adapter = _make_adapter()
        config = adapter.get_config()
        assert config.framework == "mcp"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.name == "test-mcp-agent"

    # -- _build_user_input --------------------------------------------------

    def test_build_user_input_query_key(self):
        """_build_user_input uses 'query' key first."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        result = MCPToolsAdapter._build_user_input(
            {"query": "Hello", "input": "other"}
        )
        assert result == "Hello"

    def test_build_user_input_fallback_to_json(self):
        """_build_user_input serializes to JSON when no known key."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        result = MCPToolsAdapter._build_user_input({"a": 1, "b": 2})
        assert "1" in result
        assert "2" in result

    # -- _convert_mcp_tools_to_anthropic ------------------------------------

    def test_convert_mcp_tools_to_anthropic(self):
        """MCP tool defs are converted to Anthropic format."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        mcp_tools = [_make_mcp_tool("search", "Search the web")]
        result = MCPToolsAdapter._convert_mcp_tools_to_anthropic(mcp_tools)

        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"

    def test_convert_empty_tools(self):
        """Empty tool list converts to empty list."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        result = MCPToolsAdapter._convert_mcp_tools_to_anthropic([])
        assert result == []

    # -- _safe_serialize ----------------------------------------------------

    def test_safe_serialize_primitives(self):
        """Primitives are returned as-is."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        assert MCPToolsAdapter._safe_serialize(None) is None
        assert MCPToolsAdapter._safe_serialize("hello") == "hello"
        assert MCPToolsAdapter._safe_serialize(42) == 42
        assert MCPToolsAdapter._safe_serialize(3.14) == 3.14
        assert MCPToolsAdapter._safe_serialize(True) is True

    def test_safe_serialize_dict(self):
        """Dicts are returned as-is when JSON-serializable."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        data = {"key": "value", "num": 42}
        assert MCPToolsAdapter._safe_serialize(data) == data

    def test_safe_serialize_non_serializable(self):
        """Non-serializable objects are converted to str."""
        from agentassay.integrations.mcp_adapter import MCPToolsAdapter

        result = MCPToolsAdapter._safe_serialize(object())
        assert isinstance(result, str)

    # -- scenario_id --------------------------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_uses_scenario_id(self, mock_check):
        """scenario_id is extracted from input_data."""
        client = MagicMock(spec=[])
        client.__class__.__name__ = "MCPClient"
        adapter = _make_adapter(client=client, tools=[])
        adapter._mode = "mcp"

        trace = adapter.run({"query": "test", "scenario_id": "sc-42"})

        assert trace.scenario_id == "sc-42"

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_run_default_scenario_id(self, mock_check):
        """Default scenario_id is 'default'."""
        client = MagicMock(spec=[])
        client.__class__.__name__ = "MCPClient"
        adapter = _make_adapter(client=client, tools=[])
        adapter._mode = "mcp"

        trace = adapter.run({"query": "test"})

        assert trace.scenario_id == "default"

    # -- Step indices monotonic ---------------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_step_indices_monotonic(self, mock_check):
        """Step indices are monotonically increasing."""
        client = MagicMock(spec=["call_tool", "read_resource"])
        client.__class__.__name__ = "MCPClient"
        client.call_tool.return_value = {"ok": True}
        client.read_resource.return_value = {"content": "data"}

        tools = [_make_mcp_tool("t1"), _make_mcp_tool("t2")]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "mcp"

        trace = adapter.run({
            "query": "test",
            "resources": ["file:///a.txt"],
        })

        indices = [s.step_index for s in trace.steps]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    # -- Metadata -----------------------------------------------------------

    @patch("agentassay.integrations.mcp_adapter._check_mcp_installed")
    def test_metadata_includes_mcp_info(self, mock_check):
        """Trace metadata includes MCP mode and tools count."""
        client = MagicMock(spec=[])
        client.__class__.__name__ = "MCPClient"

        tools = [_make_mcp_tool("a"), _make_mcp_tool("b")]
        adapter = _make_adapter(client=client, tools=tools)
        adapter._mode = "mcp"

        trace = adapter.run({"query": "test"})

        assert trace.metadata["mcp_mode"] == "mcp"
        assert trace.metadata["mcp_tools_count"] == 2
