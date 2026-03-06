"""Shared test fixtures and factories for the efficiency module tests.

Provides helpers for creating execution traces with controlled behavioral
properties — step counts, tool distributions, error rates, token usage,
and cost — suitable for testing fingerprinting, budgeting, trace storage,
multi-fidelity testing, and warm-start SPRT.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

import numpy as np
import pytest

from agentassay.core.models import (
    AgentConfig,
    ExecutionTrace,
    StepTrace,
)


# ===================================================================
# Trace factory helpers
# ===================================================================


def _make_step(
    index: int,
    action: str = "tool_call",
    tool_name: str | None = "search",
    duration_ms: float = 25.0,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StepTrace:
    """Build a single StepTrace with controlled properties."""
    return StepTrace(
        step_index=index,
        action=action,
        tool_name=tool_name if action == "tool_call" else None,
        tool_input={"q": f"query-{index}"} if action == "tool_call" else None,
        tool_output={"result": f"result-{index}"} if action == "tool_call" else None,
        llm_input=f"prompt-{index}" if action == "llm_response" else None,
        llm_output=f"response-{index}" if action == "llm_response" else None,
        model=model,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def make_trace(
    steps: int = 5,
    tools: list[str] | None = None,
    passed: bool = True,
    tokens: int = 500,
    cost: float = 0.01,
    model: str = "test-model",
    framework: str = "custom",
    scenario_id: str = "scenario-1",
    agent_id: str = "agent-1",
    error: str | None = None,
    seed: int | None = None,
) -> ExecutionTrace:
    """Create a test execution trace with controlled properties.

    Parameters
    ----------
    steps : int
        Number of steps to generate.
    tools : list[str] | None
        Tool names to cycle through for tool_call steps.
        If None, uses ["search", "calculate"].
    passed : bool
        Whether the trace records success.
    tokens : int
        Approximate total token count stored in metadata.
    cost : float
        Total cost in USD.
    model : str
        Model identifier.
    framework : str
        Framework identifier.
    scenario_id : str
        Scenario identifier.
    agent_id : str
        Agent identifier stored in metadata.
    error : str | None
        Error message if the trace failed.
    seed : int | None
        Optional seed for reproducible variation in step durations.
    """
    if tools is None:
        tools = ["search", "calculate"]

    rng = random.Random(seed)

    step_list: list[StepTrace] = []
    for i in range(steps):
        # Alternate between tool_call and llm_response
        if i % 3 == 2:
            action = "llm_response"
            tool = None
        else:
            action = "tool_call"
            tool = tools[i % len(tools)]

        duration = 20.0 + rng.uniform(-5.0, 15.0)
        step_list.append(
            _make_step(
                index=i,
                action=action,
                tool_name=tool,
                duration_ms=duration,
                model=model if action == "llm_response" else None,
            )
        )

    total_duration = sum(s.duration_ms for s in step_list)

    return ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        steps=step_list,
        input_data={"query": "test query", "agent_id": agent_id},
        output_data="success output" if passed else None,
        success=passed,
        error=error,
        total_duration_ms=total_duration,
        total_cost_usd=cost,
        model=model,
        framework=framework,
        metadata={
            "agent_id": agent_id,
            "total_tokens": tokens,
        },
    )


def make_traces(
    n: int,
    *,
    steps: int = 5,
    tools: list[str] | None = None,
    passed: bool = True,
    tokens: int = 500,
    cost: float = 0.01,
    model: str = "test-model",
    scenario_id: str = "scenario-1",
    agent_id: str = "agent-1",
) -> list[ExecutionTrace]:
    """Create multiple similar traces with slight random variation.

    Each trace gets a different seed so step durations differ slightly,
    but the overall behavioral structure (tool sequence, step count,
    success/failure) is the same.
    """
    return [
        make_trace(
            steps=steps,
            tools=tools,
            passed=passed,
            tokens=tokens + i * 10,  # slight token variation
            cost=cost + i * 0.001,
            model=model,
            scenario_id=scenario_id,
            agent_id=agent_id,
            seed=1000 + i,
        )
        for i in range(n)
    ]


def make_regressed_traces(
    n: int,
    *,
    steps: int = 8,
    tools: list[str] | None = None,
    pass_rate: float = 0.4,
    tokens: int = 1200,
    cost: float = 0.05,
    model: str = "test-model-v2",
    scenario_id: str = "scenario-1",
    agent_id: str = "agent-1",
) -> list[ExecutionTrace]:
    """Create traces that exhibit regression (different behavior).

    These traces differ from the default ``make_traces`` output in:
    - More steps (8 vs 5)
    - Different tools (["search", "write", "delete"] vs ["search", "calculate"])
    - Lower pass rate (40% vs 100%)
    - Higher token usage (1200 vs 500)
    - Higher cost ($0.05 vs $0.01)
    - Different model identifier
    """
    if tools is None:
        tools = ["search", "write", "delete"]

    rng = random.Random(42)
    traces = []
    for i in range(n):
        did_pass = rng.random() < pass_rate
        traces.append(
            make_trace(
                steps=steps,
                tools=tools,
                passed=did_pass,
                tokens=tokens + i * 20,
                cost=cost + i * 0.002,
                model=model,
                scenario_id=scenario_id,
                agent_id=agent_id,
                error=None if did_pass else "Regressed agent failure",
                seed=2000 + i,
            )
        )
    return traces


# ===================================================================
# Pytest fixtures
# ===================================================================


@pytest.fixture
def baseline_traces() -> list[ExecutionTrace]:
    """30 baseline traces from a stable agent."""
    return make_traces(30)


@pytest.fixture
def regressed_traces() -> list[ExecutionTrace]:
    """30 traces from a regressed agent."""
    return make_regressed_traces(30)


@pytest.fixture
def single_trace() -> ExecutionTrace:
    """A single 5-step successful trace."""
    return make_trace()


@pytest.fixture
def error_trace() -> ExecutionTrace:
    """A single trace with an error."""
    return make_trace(steps=2, passed=False, error="Agent crashed")


@pytest.fixture
def tool_heavy_trace() -> ExecutionTrace:
    """A trace with many different tools."""
    return make_trace(
        steps=10,
        tools=["search", "calculate", "write", "read", "delete"],
        tokens=2000,
    )


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """A basic AgentConfig for efficiency tests."""
    return AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        framework="custom",
        model="test-model",
        version="1.0.0",
    )
