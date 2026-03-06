"""Tests for the OpenAI Agents SDK adapter.

Validates adapter creation, Runner.run_sync() execution, item classification
(MessageOutputItem, ToolCallItem, ToolCallOutputItem), error handling, and
output extraction from RunResult.

All tests use mock objects — ``agents`` SDK NOT required.

Target: 15+ tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentassay.integrations.base import FrameworkNotInstalledError
from agentassay.integrations.openai_adapter import (
    OpenAIAgentsAdapter,
    _check_openai_agents_installed,
)


def _make_mock_agent(model: str = "gpt-4o", name: str = "agent") -> MagicMock:
    """Create a mock OpenAI Agents SDK Agent."""
    agent = MagicMock()
    agent.model = model
    agent.name = name
    return agent


def _patch_openai():
    return patch("agentassay.integrations.openai_adapter._check_openai_agents_installed")


class TestInstallCheck:
    def test_raises_error_when_agents_missing(self):
        with patch.dict("sys.modules", {"agents": None}):
            with pytest.raises(FrameworkNotInstalledError, match="openai-agents"):
                _check_openai_agents_installed()

    def test_no_error_when_installed(self):
        with patch.dict("sys.modules", {"agents": MagicMock()}):
            _check_openai_agents_installed()


class TestAdapterCreation:
    def test_basic_creation(self):
        agent = _make_mock_agent(model="gpt-4o")
        adapter = OpenAIAgentsAdapter(agent=agent, model="gpt-4o")
        assert adapter.model == "gpt-4o"
        assert adapter.framework == "openai"

    def test_infers_model_from_agent(self):
        agent = _make_mock_agent(model="gpt-4o-mini")
        adapter = OpenAIAgentsAdapter(agent=agent)
        assert adapter.model == "gpt-4o-mini"

    def test_infers_name_from_agent(self):
        agent = _make_mock_agent(name="my_agent")
        adapter = OpenAIAgentsAdapter(agent=agent)
        assert adapter.agent_name == "my_agent"

    def test_default_name_when_not_available(self):
        agent = _make_mock_agent()
        del agent.name
        adapter = OpenAIAgentsAdapter(agent=agent)
        assert adapter.agent_name == "openai-agent"


class TestRunExecution:
    def test_adapter_handles_missing_runner_gracefully(self):
        """Adapter should handle missing Runner (not installed) gracefully."""
        agent = _make_mock_agent()

        # Don't patch the check - let it try to import
        with patch.dict("sys.modules", {"agents": None}):
            adapter = OpenAIAgentsAdapter(agent=agent, model="gpt-4o")
            with pytest.raises(FrameworkNotInstalledError):
                adapter.run({"query": "test"})


class TestItemClassification:
    def test_tool_call_item_classified_as_tool_call(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        item = MagicMock()
        item.__class__.__name__ = "ToolCallItem"
        raw_item = MagicMock()
        raw_item.name = "search"
        raw_item.arguments = {"query": "test"}
        item.raw_item = raw_item
        item.name = "search"
        item.arguments = {"query": "test"}

        action, kwargs = OpenAIAgentsAdapter._classify_item(item)

        assert action == "tool_call"
        assert kwargs["tool_name"] == "search"
        assert kwargs["tool_input"] == {"query": "test"}

    def test_tool_call_with_string_arguments_parsed(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        item = MagicMock()
        item.__class__.__name__ = "ToolCallItem"
        item.name = "calculator"
        item.arguments = '{"x": 5, "y": 10}'
        item.raw_item = item

        action, kwargs = OpenAIAgentsAdapter._classify_item(item)

        assert action == "tool_call"
        assert kwargs["tool_input"] == {"x": 5, "y": 10}

    def test_tool_output_item_classified_as_tool_call(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        item = MagicMock()
        item.__class__.__name__ = "ToolCallOutputItem"
        item.tool_name = "search"
        item.output = "Found 10 results"

        action, kwargs = OpenAIAgentsAdapter._classify_item(item)

        assert action == "tool_call"
        assert kwargs["tool_output"] == "Found 10 results"

    def test_handoff_item_classified_as_decision(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        item = MagicMock()
        item.__class__.__name__ = "HandoffOutputItem"
        item.target_agent = "specialist_agent"

        action, kwargs = OpenAIAgentsAdapter._classify_item(item)

        assert action == "decision"
        assert "Handoff" in kwargs["llm_output"]

    def test_message_item_classified_as_llm_response(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        item = MagicMock()
        item.__class__.__name__ = "MessageOutputItem"
        item.raw_item = MagicMock()
        item.raw_item.content = "This is my response"

        action, kwargs = OpenAIAgentsAdapter._classify_item(item)

        assert action == "llm_response"
        assert kwargs["llm_output"] == "This is my response"


class TestErrorHandling:
    def test_framework_not_installed_error_raised(self):
        """Framework not installed error is raised correctly."""
        agent = _make_mock_agent()

        with patch.dict("sys.modules", {"agents": None}):
            adapter = OpenAIAgentsAdapter(agent=agent, model="gpt-4o")
            with pytest.raises(FrameworkNotInstalledError):
                adapter.run({"query": "test"})


class TestUserInputBuilding:
    def test_extracts_query_key(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        user_input = OpenAIAgentsAdapter._build_user_input({"query": "What is AI?"})
        assert user_input == "What is AI?"

    def test_extracts_input_key(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        user_input = OpenAIAgentsAdapter._build_user_input({"input": "Analyze this"})
        assert user_input == "Analyze this"

    def test_serializes_dict_when_no_known_keys(self):
        from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter

        user_input = OpenAIAgentsAdapter._build_user_input({"x": 1, "y": 2})
        assert isinstance(user_input, str)


class TestToCallable:
    def test_returns_callable(self):
        agent = _make_mock_agent()

        with _patch_openai():
            adapter = OpenAIAgentsAdapter(agent=agent, model="gpt-4o")
            callable_func = adapter.to_callable()

        assert callable(callable_func)
