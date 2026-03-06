"""Tests for the CrewAI framework adapter.

Validates adapter creation, kickoff execution, task output extraction,
error handling, and cost estimation from CrewAI result objects.

All tests use mock objects — ``crewai`` does NOT need to be installed.

Target: 15+ tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.integrations.base import FrameworkNotInstalledError
from agentassay.integrations.crewai_adapter import (
    CrewAIAdapter,
    _check_crewai_installed,
)


def _make_mock_crew(kickoff_result: Any = None) -> MagicMock:
    """Create a mock CrewAI Crew."""
    crew = MagicMock()
    crew.kickoff.return_value = kickoff_result or MagicMock()
    return crew


def _make_mock_task_output(description: str, raw: str, agent: str = "agent1") -> MagicMock:
    """Create a mock TaskOutput."""
    task_out = MagicMock()
    task_out.description = description
    task_out.raw = raw
    task_out.agent = agent
    task_out.pydantic = None
    task_out.tools_used = None
    return task_out


def _patch_crewai():
    return patch("agentassay.integrations.crewai_adapter._check_crewai_installed")


class TestInstallCheck:
    def test_raises_error_when_crewai_missing(self):
        with patch.dict("sys.modules", {"crewai": None}):
            with pytest.raises(FrameworkNotInstalledError, match="crewai"):
                _check_crewai_installed()

    def test_no_error_when_installed(self):
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            _check_crewai_installed()


class TestAdapterCreation:
    def test_basic_creation(self):
        crew = _make_mock_crew()
        adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
        assert adapter.model == "gpt-4o"
        assert adapter.framework == "crewai"

    def test_default_agent_name(self):
        crew = _make_mock_crew()
        adapter = CrewAIAdapter(crew=crew)
        assert adapter.agent_name == "crewai-agent"


class TestRunExecution:
    def test_successful_kickoff_returns_execution_trace(self):
        task1 = _make_mock_task_output("Task 1", "Result 1")
        task2 = _make_mock_task_output("Task 2", "Result 2")
        result = MagicMock()
        result.tasks_output = [task1, task2]
        result.raw = "Final result"
        crew = _make_mock_crew(kickoff_result=result)

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True
        assert len(trace.steps) == 2

    def test_extracts_agent_from_task_output(self):
        task1 = _make_mock_task_output("Task", "Result", agent="researcher")
        result = MagicMock()
        result.tasks_output = [task1]
        result.raw = "Final"
        crew = _make_mock_crew(kickoff_result=result)

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert trace.steps[0].metadata["crewai_agent"] == "researcher"

    def test_fallback_to_single_step_when_no_tasks_output(self):
        result = MagicMock(spec=[])  # No tasks_output attribute
        result.configure_mock(return_value="result string")
        crew = _make_mock_crew(kickoff_result=result)

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].metadata.get("crewai_fallback") is True

    def test_tool_usage_classified_as_tool_call(self):
        task = _make_mock_task_output("Task", "Result")
        task.tools_used = ["search", "calculate"]
        result = MagicMock()
        result.tasks_output = [task]
        result.raw = "Final"
        crew = _make_mock_crew(kickoff_result=result)

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert trace.steps[0].action == "tool_call"
        assert trace.steps[0].tool_name == "search"


class TestErrorHandling:
    def test_kickoff_exception_returns_failed_trace(self):
        crew = _make_mock_crew()
        crew.kickoff.side_effect = RuntimeError("Crew failed")

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert trace.success is False
        assert "RuntimeError" in trace.error


class TestCostExtraction:
    def test_extracts_cost_from_result_token_usage(self):
        result = MagicMock()
        result.tasks_output = []
        result.raw = "result"
        result.token_usage = MagicMock()
        result.token_usage.total_cost = 0.05
        crew = _make_mock_crew(kickoff_result=result)

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert trace.total_cost_usd == 0.05

    def test_cost_defaults_to_zero_when_not_available(self):
        result = MagicMock(spec=[])  # No token_usage
        crew = _make_mock_crew(kickoff_result=result)

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            trace = adapter.run({"topic": "test"})

        assert trace.total_cost_usd == 0.0


class TestToCallable:
    def test_returns_callable(self):
        crew = _make_mock_crew()

        with _patch_crewai():
            adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
            callable_func = adapter.to_callable()

        assert callable(callable_func)
