"""Shared test fixtures for AgentAssay test suite.

Provides reusable factories and fixtures for creating test data:
- StepTrace, ExecutionTrace, TestScenario, TrialResult builders
- Mock agent callables (passing, failing, flaky)
- Configuration fixtures (AgentConfig, AssayConfig)
- Contract dict fixtures
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from agentassay.core.models import (
    AgentConfig,
    AssayConfig,
    ExecutionTrace,
    StepTrace,
    TestScenario,
    TrialResult,
)


# ===================================================================
# Helper factories
# ===================================================================


def make_step(
    index: int = 0,
    action: str = "tool_call",
    tool_name: str | None = "search",
    tool_input: dict[str, Any] | None = None,
    tool_output: Any = None,
    llm_input: str | None = None,
    llm_output: str | None = None,
    model: str | None = None,
    duration_ms: float = 50.0,
    metadata: dict[str, Any] | None = None,
) -> StepTrace:
    """Build a StepTrace with sensible defaults."""
    return StepTrace(
        step_index=index,
        action=action,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_output=tool_output,
        llm_input=llm_input,
        llm_output=llm_output,
        model=model,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def make_trace(
    steps: int = 3,
    tools: list[str] | None = None,
    success: bool = True,
    cost: float = 0.01,
    model: str = "test-model",
    framework: str = "custom",
    output_data: Any = "test output",
    scenario_id: str = "scenario-1",
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> ExecutionTrace:
    """Build an ExecutionTrace with sensible defaults.

    Parameters
    ----------
    steps : int
        Number of steps to generate.
    tools : list[str] | None
        Tool names to cycle through. If None, uses ["search", "calculate"].
    success : bool
        Whether the trace reports success.
    cost : float
        Total cost in USD.
    model : str
        Model name.
    framework : str
        Framework name.
    output_data : Any
        The trace output.
    scenario_id : str
        Scenario identifier.
    input_data : dict | None
        Input data dict.
    metadata : dict | None
        Trace-level metadata.
    error : str | None
        Error message (if any).
    """
    if tools is None:
        tools = ["search", "calculate"]

    step_list: list[StepTrace] = []
    for i in range(steps):
        tool = tools[i % len(tools)] if tools else None
        action = "tool_call" if tool else "llm_response"
        step_list.append(
            make_step(
                index=i,
                action=action,
                tool_name=tool,
                tool_output={"result": f"step-{i}"},
                duration_ms=20.0 + i * 5,
                metadata=metadata or {},
            )
        )

    return ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        steps=step_list,
        input_data=input_data or {"query": "test query"},
        output_data=output_data,
        success=success,
        error=error,
        total_duration_ms=sum(s.duration_ms for s in step_list),
        total_cost_usd=cost,
        model=model,
        framework=framework,
        metadata=metadata or {},
    )


def make_scenario(
    scenario_id: str = "scenario-1",
    name: str = "Test Scenario",
    input_data: dict[str, Any] | None = None,
    expected_properties: dict[str, Any] | None = None,
    evaluator: str | None = None,
    tags: list[str] | None = None,
    timeout_seconds: float = 300.0,
) -> TestScenario:
    """Build a TestScenario with sensible defaults."""
    return TestScenario(
        scenario_id=scenario_id,
        name=name,
        input_data=input_data or {"query": "test query"},
        expected_properties=expected_properties or {},
        evaluator=evaluator,
        tags=tags or [],
        timeout_seconds=timeout_seconds,
    )


def make_agent_config(
    agent_id: str = "test-agent",
    name: str = "Test Agent",
    framework: str = "custom",
    model: str = "test-model",
    version: str = "1.0.0",
    parameters: dict[str, Any] | None = None,
) -> AgentConfig:
    """Build an AgentConfig with sensible defaults."""
    return AgentConfig(
        agent_id=agent_id,
        name=name,
        framework=framework,
        model=model,
        version=version,
        parameters=parameters or {},
    )


def make_assay_config(
    num_trials: int = 30,
    significance_level: float = 0.05,
    power: float = 0.80,
    confidence_method: str = "wilson",
    regression_test: str = "fisher",
    use_sprt: bool = False,
    seed: int | None = None,
    max_cost_usd: float = 50.0,
    timeout_seconds: float = 600.0,
    parallel_trials: int = 1,
) -> AssayConfig:
    """Build an AssayConfig with sensible defaults."""
    return AssayConfig(
        num_trials=num_trials,
        significance_level=significance_level,
        power=power,
        confidence_method=confidence_method,
        regression_test=regression_test,
        use_sprt=use_sprt,
        seed=seed,
        max_cost_usd=max_cost_usd,
        timeout_seconds=timeout_seconds,
        parallel_trials=parallel_trials,
    )


def make_contract_dict(
    name: str = "test_contract",
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a contract dict suitable for ContractOracle."""
    if constraints is None:
        constraints = [
            {
                "name": "max_steps",
                "type": "guardrail",
                "severity": "hard",
                "condition": "step_count <= 10",
            },
            {
                "name": "cost_limit",
                "type": "guardrail",
                "severity": "hard",
                "condition": "total_cost_usd <= 1.0",
            },
            {
                "name": "must_succeed",
                "type": "postcondition",
                "severity": "soft",
                "condition": "success",
            },
        ]
    return {"contract": {"name": name, "constraints": constraints}}


# ===================================================================
# Mock agent callables
# ===================================================================


def passing_agent(input_data: dict[str, Any]) -> ExecutionTrace:
    """A mock agent that always succeeds."""
    return make_trace(
        steps=3,
        success=True,
        cost=0.01,
        input_data=input_data,
        output_data="success",
    )


def failing_agent(input_data: dict[str, Any]) -> ExecutionTrace:
    """A mock agent that always fails."""
    return make_trace(
        steps=1,
        success=False,
        cost=0.005,
        input_data=input_data,
        output_data=None,
        error="Agent failed intentionally",
    )


def flaky_agent(
    input_data: dict[str, Any],
    pass_rate: float = 0.5,
) -> ExecutionTrace:
    """A mock agent that passes with the given probability."""
    if random.random() < pass_rate:
        return make_trace(steps=3, success=True, input_data=input_data)
    return make_trace(
        steps=1, success=False, input_data=input_data, error="Flaky failure"
    )


def raising_agent(input_data: dict[str, Any]) -> ExecutionTrace:
    """A mock agent that always raises an exception."""
    raise RuntimeError("Agent crashed")


def mutation_agent(
    config: AgentConfig, input_data: dict[str, Any]
) -> ExecutionTrace:
    """A mock agent for mutation runner (accepts config + input_data)."""
    return make_trace(
        steps=2,
        success=True,
        cost=0.01,
        model=config.model,
        framework=config.framework,
        input_data=input_data,
        output_data="mutation test output",
    )


def mutation_agent_sensitive(
    config: AgentConfig, input_data: dict[str, Any]
) -> ExecutionTrace:
    """A mock agent that fails when mutated (model is not 'test-model')."""
    is_original = config.model == "test-model"
    return make_trace(
        steps=2,
        success=is_original,
        cost=0.01,
        model=config.model,
        framework=config.framework,
        input_data=input_data,
        output_data="original output" if is_original else "mutant output",
    )


# ===================================================================
# Pytest fixtures
# ===================================================================


@pytest.fixture
def sample_step() -> StepTrace:
    """A single tool_call StepTrace."""
    return make_step()


@pytest.fixture
def sample_trace() -> ExecutionTrace:
    """A 3-step successful ExecutionTrace."""
    return make_trace()


@pytest.fixture
def failed_trace() -> ExecutionTrace:
    """A 1-step failed ExecutionTrace."""
    return make_trace(steps=1, success=False, error="test error")


@pytest.fixture
def sample_scenario() -> TestScenario:
    """A basic TestScenario."""
    return make_scenario()


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """A basic AgentConfig."""
    return make_agent_config()


@pytest.fixture
def sample_assay_config() -> AssayConfig:
    """A basic AssayConfig with 30 trials."""
    return make_assay_config()


@pytest.fixture
def sample_contract_dict() -> dict[str, Any]:
    """A basic contract dict with 3 constraints."""
    return make_contract_dict()
