"""Tests for the HuggingFace smolagents framework adapter.

Validates adapter creation, agent.run() execution, step log extraction,
tool call classification, reasoning step detection, and error handling.

All tests use mock objects — ``smolagents`` NOT required.

Target: 15+ tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.integrations.base import FrameworkNotInstalledError
from agentassay.integrations.smolagents_adapter import (
    SmolAgentsAdapter,
    _check_smolagents_installed,
)


def _make_mock_agent(
    model_id: str = "Qwen/Qwen2.5-72B",
    has_logs: bool = True,
) -> MagicMock:
    """Create a mock smolagents agent."""
    agent = MagicMock()
    agent_model = MagicMock()
    agent_model.model_id = model_id
    agent.model = agent_model
    agent.run.return_value = "Agent output"

    if has_logs:
        agent.logs = []
    else:
        del agent.logs

    return agent


def _patch_smolagents():
    return patch("agentassay.integrations.smolagents_adapter._check_smolagents_installed")


class TestInstallCheck:
    def test_raises_error_when_smolagents_missing(self):
        with patch.dict("sys.modules", {"smolagents": None}):
            with pytest.raises(FrameworkNotInstalledError, match="smolagents"):
                _check_smolagents_installed()

    def test_no_error_when_installed(self):
        with patch.dict("sys.modules", {"smolagents": MagicMock()}):
            _check_smolagents_installed()


class TestAdapterCreation:
    def test_basic_creation(self):
        agent = _make_mock_agent(model_id="Qwen/Qwen2.5-72B")
        adapter = SmolAgentsAdapter(agent=agent, model="test-model")
        assert adapter.model == "test-model"
        assert adapter.framework == "smolagents"

    def test_infers_model_from_agent(self):
        agent = _make_mock_agent(model_id="Qwen/Qwen2.5-Coder")
        adapter = SmolAgentsAdapter(agent=agent)
        assert adapter.model == "Qwen/Qwen2.5-Coder"

    def test_default_agent_name(self):
        agent = _make_mock_agent()
        adapter = SmolAgentsAdapter(agent=agent)
        assert adapter.agent_name == "smolagents-agent"


class TestRunExecution:
    def test_successful_run_returns_execution_trace(self):
        agent = _make_mock_agent()
        agent.logs = [
            {"step_number": 0, "model_output": "Thinking step 1"},
            {"step_number": 1, "model_output": "Thinking step 2"},
        ]
        agent.run.return_value = "Final answer"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "solve problem"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True
        assert len(trace.steps) == 2

    def test_extracts_tool_calls_from_log_entry(self):
        agent = _make_mock_agent()
        agent.logs = [
            {
                "step_number": 0,
                "tool_calls": [{"name": "search", "arguments": {"query": "test"}}],
            }
        ]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        # Should have at least one tool_call step
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) >= 1
        assert tool_steps[0].tool_name == "search"

    def test_fallback_when_no_logs(self):
        agent = _make_mock_agent(has_logs=False)
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].metadata.get("smolagents_fallback") is True

    def test_object_style_log_entry_with_attributes(self):
        agent = _make_mock_agent()
        log_entry = MagicMock()
        log_entry.step_number = 0
        log_entry.model_output = "Reasoning"
        log_entry.tool_calls = None
        log_entry.observations = None
        agent.logs = [log_entry]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        assert trace.success is True
        assert len(trace.steps) >= 1


class TestLogExtraction:
    def test_tries_agent_logs_attribute(self):
        agent = _make_mock_agent()
        agent.logs = [{"model_output": "step1"}]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        assert len(trace.steps) >= 1

    def test_tries_agent_step_logs_attribute(self):
        agent = _make_mock_agent(has_logs=False)
        agent.step_logs = [{"model_output": "step1"}]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        assert len(trace.steps) >= 1

    def test_tries_agent_memory_attribute(self):
        agent = _make_mock_agent(has_logs=False)
        agent.memory = [{"model_output": "step1"}]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        assert len(trace.steps) >= 1


class TestStepClassification:
    def test_dict_with_tool_calls_creates_tool_steps(self):
        agent = _make_mock_agent()
        agent.logs = [
            {
                "tool_calls": [
                    {"name": "calculator", "arguments": {"x": 5}},
                    {"name": "search", "arguments": {"q": "test"}},
                ]
            }
        ]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) == 2

    def test_dict_with_llm_output_creates_llm_step(self):
        agent = _make_mock_agent()
        agent.logs = [{"llm_output": "Agent is thinking..."}]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        llm_steps = [s for s in trace.steps if s.action == "llm_response"]
        assert len(llm_steps) >= 1

    def test_observations_without_tool_calls_creates_observation_step(self):
        agent = _make_mock_agent()
        log_entry = MagicMock()
        log_entry.observations = "Observation data"
        log_entry.tool_calls = None
        log_entry.model_output = None
        agent.logs = [log_entry]
        agent.run.return_value = "result"

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        obs_steps = [s for s in trace.steps if s.action == "observation"]
        assert len(obs_steps) >= 1


class TestErrorHandling:
    def test_agent_run_exception_returns_failed_trace(self):
        agent = _make_mock_agent()
        agent.run.side_effect = RuntimeError("Agent crashed")

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            trace = adapter.run({"task": "test"})

        assert trace.success is False
        assert "RuntimeError" in trace.error


class TestTaskBuilding:
    def test_extracts_task_key(self):
        from agentassay.integrations.smolagents_adapter import SmolAgentsAdapter

        task = SmolAgentsAdapter._build_task({"task": "Find population"})
        assert task == "Find population"

    def test_extracts_query_key(self):
        from agentassay.integrations.smolagents_adapter import SmolAgentsAdapter

        task = SmolAgentsAdapter._build_task({"query": "What is AI?"})
        assert task == "What is AI?"

    def test_serializes_dict_when_no_known_keys(self):
        from agentassay.integrations.smolagents_adapter import SmolAgentsAdapter

        task = SmolAgentsAdapter._build_task({"x": 1, "y": 2})
        assert isinstance(task, str)


class TestToCallable:
    def test_returns_callable(self):
        agent = _make_mock_agent()

        with _patch_smolagents():
            adapter = SmolAgentsAdapter(agent=agent, model="test-model")
            callable_func = adapter.to_callable()

        assert callable(callable_func)
