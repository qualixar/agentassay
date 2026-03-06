"""Tests for core data models.

Tests all 6 Pydantic models: StepTrace, ExecutionTrace, TestScenario,
TrialResult, AgentConfig, AssayConfig. Validates creation, immutability,
computed properties, validators, and edge cases.

Target: ~40 tests.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentassay.core.models import (
    AgentConfig,
    AssayConfig,
    ExecutionTrace,
    StepTrace,
    TestScenario,
    TrialResult,
)

from tests.conftest import make_step, make_trace, make_scenario


# ===================================================================
# StepTrace
# ===================================================================


class TestStepTrace:
    """Tests for the StepTrace model."""

    def test_create_tool_call_step(self):
        step = make_step(index=0, action="tool_call", tool_name="search")
        assert step.step_index == 0
        assert step.action == "tool_call"
        assert step.tool_name == "search"
        assert step.duration_ms >= 0.0

    def test_create_llm_step(self):
        step = StepTrace(
            step_index=1,
            action="llm_response",
            llm_input="What is AI?",
            llm_output="AI is...",
            model="gpt-4o",
            duration_ms=100.0,
        )
        assert step.action == "llm_response"
        assert step.llm_output == "AI is..."

    def test_action_lowercased(self):
        step = StepTrace(step_index=0, action="  TOOL_CALL  ", tool_name="s", duration_ms=1.0)
        assert step.action == "tool_call"

    def test_frozen_immutable(self):
        step = make_step()
        with pytest.raises(ValidationError):
            step.action = "new_action"

    def test_tool_call_requires_tool_name(self):
        with pytest.raises(ValidationError, match="tool_name is required"):
            StepTrace(step_index=0, action="tool_call", duration_ms=1.0)

    def test_non_tool_action_no_tool_name_ok(self):
        step = StepTrace(step_index=0, action="llm_response", duration_ms=1.0)
        assert step.tool_name is None

    def test_negative_step_index_rejected(self):
        with pytest.raises(ValidationError):
            StepTrace(step_index=-1, action="llm_response", duration_ms=1.0)

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            StepTrace(step_index=0, action="llm_response", duration_ms=-5.0)

    def test_empty_action_rejected(self):
        with pytest.raises(ValidationError):
            StepTrace(step_index=0, action="", duration_ms=1.0)

    def test_step_id_auto_generated(self):
        s1 = make_step(index=0)
        s2 = make_step(index=1)
        assert s1.step_id != s2.step_id

    def test_metadata_default_empty(self):
        step = StepTrace(step_index=0, action="llm_response", duration_ms=1.0)
        assert step.metadata == {}

    def test_timestamp_auto_generated(self):
        step = make_step()
        assert isinstance(step.timestamp, datetime)


# ===================================================================
# ExecutionTrace
# ===================================================================


class TestExecutionTrace:
    """Tests for the ExecutionTrace model."""

    def test_create_basic_trace(self, sample_trace):
        assert sample_trace.success is True
        assert sample_trace.step_count == 3

    def test_tools_used_property(self):
        trace = make_trace(steps=3, tools=["search", "calculate"])
        # 3 steps, alternating: search, calculate, search
        assert "search" in trace.tools_used
        assert "calculate" in trace.tools_used

    def test_tools_used_empty_steps(self):
        trace = make_trace(steps=0)
        assert trace.tools_used == set()

    def test_step_count_property(self):
        for n in [0, 1, 5, 10]:
            trace = make_trace(steps=n)
            assert trace.step_count == n

    def test_decision_path_property(self):
        trace = make_trace(steps=3, tools=["search", "calc"])
        path = trace.decision_path
        assert len(path) == 3
        assert all(isinstance(a, str) for a in path)

    def test_frozen_immutable(self, sample_trace):
        with pytest.raises(ValidationError):
            sample_trace.success = False

    def test_steps_ordered_validator(self):
        """Steps with non-monotonic indices should be rejected."""
        s0 = make_step(index=0)
        s2 = make_step(index=2, tool_name="calc")
        s1 = make_step(index=1, tool_name="write")
        with pytest.raises(ValidationError, match="monotonically increasing"):
            ExecutionTrace(
                scenario_id="s1",
                steps=[s0, s2, s1],  # wrong order
                model="m",
                framework="f",
            )

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            ExecutionTrace(
                scenario_id="s1",
                model="m",
                framework="f",
                total_cost_usd=-1.0,
            )

    def test_empty_scenario_id_rejected(self):
        with pytest.raises(ValidationError):
            ExecutionTrace(
                scenario_id="",
                model="m",
                framework="f",
            )

    def test_zero_cost_trace(self):
        trace = make_trace(cost=0.0)
        assert trace.total_cost_usd == 0.0

    def test_trace_with_error(self):
        trace = make_trace(success=False, error="something broke")
        assert trace.error == "something broke"
        assert trace.success is False


# ===================================================================
# TestScenario
# ===================================================================


class TestTestScenario:
    """Tests for the TestScenario model."""

    def test_create_basic_scenario(self, sample_scenario):
        assert sample_scenario.scenario_id == "scenario-1"
        assert sample_scenario.name == "Test Scenario"

    def test_frozen_immutable(self, sample_scenario):
        with pytest.raises(ValidationError):
            sample_scenario.name = "changed"

    def test_empty_scenario_id_rejected(self):
        with pytest.raises(ValidationError):
            TestScenario(scenario_id="", name="test")

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            TestScenario(scenario_id="s1", name="")

    def test_default_timeout(self):
        s = make_scenario()
        assert s.timeout_seconds == 300.0

    def test_custom_expected_properties(self):
        s = make_scenario(expected_properties={"max_steps": 5, "must_use_tools": ["search"]})
        assert s.expected_properties["max_steps"] == 5

    def test_tags_default_empty(self):
        s = make_scenario()
        assert s.tags == []

    def test_evaluator_none_by_default(self):
        s = make_scenario()
        assert s.evaluator is None

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValidationError):
            TestScenario(scenario_id="s", name="n", timeout_seconds=-1.0)


# ===================================================================
# TrialResult
# ===================================================================


class TestTrialResult:
    """Tests for the TrialResult model."""

    def test_create_basic_result(self, sample_trace):
        result = TrialResult(
            scenario_id="s1",
            trace=sample_trace,
            passed=True,
            score=0.9,
        )
        assert result.passed is True
        assert result.score == 0.9

    def test_frozen_immutable(self, sample_trace):
        result = TrialResult(scenario_id="s1", trace=sample_trace)
        with pytest.raises(ValidationError):
            result.passed = True

    def test_score_bounds_enforced(self, sample_trace):
        with pytest.raises(ValidationError):
            TrialResult(scenario_id="s1", trace=sample_trace, score=1.5)
        with pytest.raises(ValidationError):
            TrialResult(scenario_id="s1", trace=sample_trace, score=-0.1)

    def test_default_score_zero(self, sample_trace):
        result = TrialResult(scenario_id="s1", trace=sample_trace)
        assert result.score == 0.0

    def test_default_passed_false(self, sample_trace):
        result = TrialResult(scenario_id="s1", trace=sample_trace)
        assert result.passed is False


# ===================================================================
# AgentConfig
# ===================================================================


class TestAgentConfig:
    """Tests for the AgentConfig model."""

    def test_create_basic_config(self, sample_agent_config):
        assert sample_agent_config.framework == "custom"
        assert sample_agent_config.model == "test-model"

    def test_mutable_config(self, sample_agent_config):
        sample_agent_config.version = "2.0.0"
        assert sample_agent_config.version == "2.0.0"

    @pytest.mark.parametrize(
        "framework",
        ["langgraph", "crewai", "autogen", "openai", "smolagents", "custom"],
    )
    def test_valid_frameworks(self, framework):
        cfg = AgentConfig(agent_id="a", name="n", framework=framework, model="m")
        assert cfg.framework == framework

    def test_invalid_framework_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(agent_id="a", name="n", framework="unknown", model="m")

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(agent_id="", name="n", framework="custom", model="m")


# ===================================================================
# AssayConfig
# ===================================================================


class TestAssayConfig:
    """Tests for the AssayConfig model."""

    def test_create_default_config(self):
        cfg = AssayConfig()
        assert cfg.num_trials == 30
        assert cfg.significance_level == 0.05
        assert cfg.power == 0.80
        assert cfg.confidence_method == "wilson"

    def test_mutable_config(self):
        cfg = AssayConfig()
        cfg.num_trials = 100
        assert cfg.num_trials == 100

    def test_low_trials_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AssayConfig(num_trials=5)
            assert len(w) >= 1
            assert "very low" in str(w[0].message).lower()

    def test_num_trials_bounds(self):
        with pytest.raises(ValidationError):
            AssayConfig(num_trials=0)
        with pytest.raises(ValidationError):
            AssayConfig(num_trials=10_001)

    def test_significance_level_bounds(self):
        with pytest.raises(ValidationError):
            AssayConfig(significance_level=0.0)
        with pytest.raises(ValidationError):
            AssayConfig(significance_level=1.0)

    def test_sprt_consistency_validator(self):
        """SPRT requires alpha + beta < 1."""
        with pytest.raises(ValidationError, match="alpha \\+ beta"):
            AssayConfig(
                use_sprt=True,
                significance_level=0.6,
                power=0.3,  # beta = 0.7, alpha + beta = 1.3 >= 1
            )

    def test_sprt_valid_config(self):
        cfg = AssayConfig(
            use_sprt=True,
            significance_level=0.05,
            power=0.80,
        )
        assert cfg.use_sprt is True

    @pytest.mark.parametrize(
        "method", ["wilson", "clopper-pearson", "normal"]
    )
    def test_valid_confidence_methods(self, method):
        cfg = AssayConfig(confidence_method=method)
        assert cfg.confidence_method == method

    @pytest.mark.parametrize(
        "test", ["fisher", "chi2", "ks", "mann-whitney"]
    )
    def test_valid_regression_tests(self, test):
        cfg = AssayConfig(regression_test=test)
        assert cfg.regression_test == test

    def test_seed_none_by_default(self):
        cfg = AssayConfig()
        assert cfg.seed is None

    def test_parallel_trials_bounds(self):
        with pytest.raises(ValidationError):
            AssayConfig(parallel_trials=0)
        with pytest.raises(ValidationError):
            AssayConfig(parallel_trials=257)
