"""Tests for the AutoGen framework adapter.

Validates adapter creation, run() execution, initiate_chat() pattern,
generate_reply() fallback, message classification, and model inference.

All tests use mock objects — ``autogen`` / ``autogen_agentchat`` NOT required.

Target: 20+ tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.integrations.autogen_adapter import (
    AutoGenAdapter,
    _check_autogen_installed,
)
from agentassay.integrations.base import FrameworkNotInstalledError


def _make_mock_agent(
    llm_config: dict[str, Any] | None = None,
    name: str = "assistant",
    has_run: bool = False,
    has_generate_reply: bool = True,
) -> MagicMock:
    """Create a mock AutoGen agent."""
    agent = MagicMock()
    agent.name = name
    agent.llm_config = llm_config or {}

    if not has_run:
        del agent.run
    if not has_generate_reply:
        del agent.generate_reply

    return agent


def _patch_autogen():
    return patch("agentassay.integrations.autogen_adapter._check_autogen_installed")


class TestInstallCheck:
    def test_raises_error_when_autogen_missing(self):
        with patch.dict("sys.modules", {"autogen_agentchat": None, "autogen": None}):
            with pytest.raises(FrameworkNotInstalledError, match="autogen"):
                _check_autogen_installed()

    def test_no_error_when_autogen_agentchat_installed(self):
        with patch.dict("sys.modules", {"autogen_agentchat": MagicMock()}):
            _check_autogen_installed()

    def test_no_error_when_legacy_autogen_installed(self):
        with patch.dict("sys.modules", {"autogen": MagicMock()}):
            _check_autogen_installed()


class TestAdapterCreation:
    def test_basic_creation(self):
        agent = _make_mock_agent()
        adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
        assert adapter.model == "gpt-4o"
        assert adapter.framework == "autogen"

    def test_infers_model_from_llm_config_list(self):
        agent = _make_mock_agent(llm_config={"config_list": [{"model": "gpt-4o-mini"}]})
        adapter = AutoGenAdapter(agent=agent)
        assert adapter.model == "gpt-4o-mini"

    def test_infers_model_from_llm_config_direct(self):
        agent = _make_mock_agent(llm_config={"model": "claude-3"})
        adapter = AutoGenAdapter(agent=agent)
        assert adapter.model == "claude-3"

    def test_uses_agent_name_from_agent_object(self):
        agent = _make_mock_agent(name="my_assistant")
        adapter = AutoGenAdapter(agent=agent)
        assert adapter.agent_name == "my_assistant"

    def test_default_agent_name_when_not_available(self):
        agent = _make_mock_agent()
        del agent.name
        adapter = AutoGenAdapter(agent=agent)
        assert adapter.agent_name == "autogen-agent"


class TestRunExecution:
    def test_generate_reply_strategy_produces_trace(self):
        agent = _make_mock_agent(has_run=False, has_generate_reply=True)
        agent.generate_reply.return_value = "This is my reply"

        with _patch_autogen():
            adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
            trace = adapter.run({"message": "Hello"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].action == "llm_response"

    def test_run_strategy_with_messages_produces_multi_step_trace(self):
        agent = _make_mock_agent(has_run=True)
        result = MagicMock()
        result.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result.output = "Hi there"
        agent.run_sync.return_value = result

        with _patch_autogen():
            adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
            trace = adapter.run({"message": "Hello"})

        assert trace.success is True
        assert len(trace.steps) == 2

    def test_initiate_chat_strategy_with_user_proxy(self):
        agent = _make_mock_agent(has_run=False)
        user_proxy = MagicMock()
        chat_result = MagicMock()
        chat_result.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        chat_result.summary = "Hi"
        user_proxy.initiate_chat.return_value = chat_result

        with _patch_autogen():
            adapter = AutoGenAdapter(agent=agent, user_proxy=user_proxy, model="gpt-4o")
            trace = adapter.run({"message": "Hello"})

        assert trace.success is True
        assert len(trace.steps) == 2

    def test_fallback_single_step_when_no_messages(self):
        agent = _make_mock_agent(has_run=True)
        result = MagicMock()
        result.messages = None
        result.output = "final output"
        agent.run_sync.return_value = result

        with _patch_autogen():
            adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
            trace = adapter.run({"message": "test"})

        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].metadata.get("autogen_fallback") is True


class TestMessageClassification:
    def test_dict_with_tool_calls_classified_as_tool_call(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "test"}'}}],
        }

        action, kwargs = AutoGenAdapter._classify_autogen_message(msg)

        assert action == "tool_call"
        assert kwargs["tool_name"] == "search"
        assert isinstance(kwargs["tool_input"], dict)

    def test_dict_with_tool_role_classified_as_tool_call(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = {
            "role": "tool",
            "name": "search_result",
            "content": "Found 10 results",
        }

        action, kwargs = AutoGenAdapter._classify_autogen_message(msg)

        assert action == "tool_call"
        assert kwargs["tool_name"] == "search_result"

    def test_dict_with_assistant_role_classified_as_llm_response(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = {"role": "assistant", "content": "Here is my answer"}

        action, kwargs = AutoGenAdapter._classify_autogen_message(msg)

        assert action == "llm_response"
        assert kwargs["llm_output"] == "Here is my answer"

    def test_object_with_toolcall_in_type_classified_as_tool_call(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = MagicMock()
        msg.name = "calculator"
        msg.arguments = {"x": 5, "y": 10}
        msg.__class__.__name__ = "ToolCallMessage"

        action, kwargs = AutoGenAdapter._classify_autogen_message(msg)

        assert action == "tool_call"
        assert kwargs["tool_name"] == "calculator"


class TestErrorHandling:
    def test_agent_exception_returns_failed_trace(self):
        agent = _make_mock_agent()
        agent.generate_reply.side_effect = RuntimeError("Agent crashed")

        with _patch_autogen():
            adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
            trace = adapter.run({"message": "test"})

        assert trace.success is False
        assert "RuntimeError" in trace.error

    def test_framework_not_installed_error_not_swallowed(self):
        agent = _make_mock_agent()

        with patch.dict("sys.modules", {"autogen_agentchat": None, "autogen": None}):
            adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
            with pytest.raises(FrameworkNotInstalledError):
                adapter.run({"message": "test"})


class TestMessageBuilding:
    def test_extracts_message_key(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = AutoGenAdapter._build_message({"message": "Hello world"})
        assert msg == "Hello world"

    def test_extracts_query_key(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = AutoGenAdapter._build_message({"query": "What is AI?"})
        assert msg == "What is AI?"

    def test_serializes_dict_when_no_known_keys(self):
        from agentassay.integrations.autogen_adapter import AutoGenAdapter

        msg = AutoGenAdapter._build_message({"x": 1, "y": 2})
        assert isinstance(msg, str)
        assert "x" in msg or "y" in msg


class TestToCallable:
    def test_returns_callable(self):
        agent = _make_mock_agent()

        with _patch_autogen():
            adapter = AutoGenAdapter(agent=agent, model="gpt-4o")
            callable_func = adapter.to_callable()

        assert callable(callable_func)
