"""Tests for the TrialRunner.

Tests trial execution, evaluation logic, error handling, cost tracking,
and batch execution with mock agents.

Target: ~20 tests.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.core.runner import (
    CostBudgetExceededError,
    TrialRunner,
)
from tests.conftest import (
    failing_agent,
    make_agent_config,
    make_assay_config,
    make_scenario,
    make_trace,
    passing_agent,
    raising_agent,
)


def _slow_agent(input_data: dict[str, Any]) -> ExecutionTrace:
    """Agent that sleeps briefly to simulate work."""
    time.sleep(0.01)
    return make_trace(success=True, input_data=input_data)


class TestTrialRunner:
    """Tests for TrialRunner."""

    def _make_runner(self, agent_fn=None, **assay_kw):
        agent = agent_fn or passing_agent
        config = make_assay_config(**assay_kw)
        agent_config = make_agent_config()
        return TrialRunner(agent, config, agent_config)

    def test_run_trial_passing_agent(self):
        runner = self._make_runner()
        scenario = make_scenario()
        result = runner.run_trial(scenario)
        assert result.trace.success is True
        assert result.passed is True

    def test_run_trial_failing_agent(self):
        runner = self._make_runner(failing_agent)
        scenario = make_scenario()
        result = runner.run_trial(scenario)
        assert result.trace.success is False
        assert result.passed is False

    def test_run_trial_raising_agent(self):
        runner = self._make_runner(raising_agent)
        scenario = make_scenario()
        result = runner.run_trial(scenario)
        assert result.passed is False
        assert result.trace.error is not None
        assert "RuntimeError" in result.trace.error

    def test_run_trials_returns_n_results(self):
        runner = self._make_runner(num_trials=10)
        scenario = make_scenario()
        results = runner.run_trials(scenario, n=5)
        assert len(results) == 5

    def test_run_trials_defaults_to_config_num(self):
        runner = self._make_runner(num_trials=12)
        scenario = make_scenario()
        results = runner.run_trials(scenario)
        assert len(results) == 12

    def test_cost_tracking(self):
        runner = self._make_runner()
        scenario = make_scenario()
        runner.run_trial(scenario)
        assert runner.cumulative_cost_usd > 0

    def test_cost_budget_exceeded(self):
        runner = self._make_runner(max_cost_usd=0.005)
        scenario = make_scenario()
        # First trial costs 0.01, exceeds 0.005 budget after accumulation
        runner.run_trial(scenario)  # accumulates cost
        with pytest.raises(CostBudgetExceededError):
            runner.run_trial(scenario)

    def test_evaluate_max_steps_pass(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"max_steps": 10})
        result = runner.run_trial(scenario)
        assert result.passed is True

    def test_evaluate_max_steps_fail(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"max_steps": 1})
        result = runner.run_trial(scenario)
        # passing_agent produces 3 steps > 1
        assert result.passed is False

    def test_evaluate_must_use_tools(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"must_use_tools": ["search"]})
        result = runner.run_trial(scenario)
        assert result.passed is True

    def test_evaluate_must_use_tools_fail(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"must_use_tools": ["nonexistent_tool"]})
        result = runner.run_trial(scenario)
        assert result.passed is False

    def test_evaluate_must_not_use_tools(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"must_not_use_tools": ["dangerous_tool"]})
        result = runner.run_trial(scenario)
        assert result.passed is True

    def test_evaluate_output_contains(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"output_contains": "success"})
        result = runner.run_trial(scenario)
        assert result.passed is True

    def test_evaluate_output_contains_fail(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"output_contains": "nonexistent_text"})
        result = runner.run_trial(scenario)
        assert result.passed is False

    def test_evaluate_max_cost(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={"max_cost_usd": 10.0})
        result = runner.run_trial(scenario)
        assert result.passed is True

    def test_evaluate_no_properties_agent_success(self):
        runner = self._make_runner()
        scenario = make_scenario(expected_properties={})
        result = runner.run_trial(scenario)
        assert result.passed is True
        assert result.score == 1.0

    def test_evaluate_no_properties_agent_failure(self):
        runner = self._make_runner(failing_agent)
        scenario = make_scenario(expected_properties={})
        result = runner.run_trial(scenario)
        assert result.passed is False

    def test_properties_exposed(self):
        runner = self._make_runner()
        assert runner.config.num_trials == 30
        assert runner.agent_config.framework == "custom"

    def test_parallel_execution(self):
        runner = self._make_runner(num_trials=10, parallel_trials=4)
        scenario = make_scenario()
        results = runner.run_trials(scenario, n=10)
        assert len(results) == 10

    def test_score_is_fraction_of_checks(self):
        runner = self._make_runner()
        scenario = make_scenario(
            expected_properties={
                "max_steps": 10,  # pass (3 <= 10)
                "must_use_tools": ["nonexistent"],  # fail
            }
        )
        result = runner.run_trial(scenario)
        assert result.score == 0.5  # 1 out of 2 checks passed
