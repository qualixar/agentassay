"""Shared fixtures for experiment integration tests.

Provides:
- MockAzureFoundryClient with realistic async mocks
- Factory functions for trial results with tool-call steps
- E7 config fixtures
- Scoring fixtures
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from experiments.runner.daemon import ALL_MODELS


# -------------------------------------------------------------------
# Mock step / trial result factories
# -------------------------------------------------------------------


def make_agent_step(
    step_index: int = 0,
    action: str = "tool_call",
    tool_name: str = "search_products",
    tool_input: dict[str, Any] | None = None,
    tool_output: str | None = None,
    llm_output: str | None = None,
    model: str = "gpt-5.2-chat",
    duration_ms: float = 120.0,
) -> dict[str, Any]:
    """Build a realistic agent step dict (as emitted by run_agent)."""
    step: dict[str, Any] = {
        "step_index": step_index,
        "action": action,
        "model": model,
        "duration_ms": duration_ms,
    }
    if action == "tool_call":
        step["tool_name"] = tool_name
        step["tool_input"] = tool_input or {"query": "laptop"}
        step["tool_output"] = tool_output or json.dumps(
            {"status": "ok", "result": f"Result for {tool_name}"}
        )
    else:
        step["llm_output"] = llm_output or "Task completed successfully."
        step["usage"] = {
            "prompt_tokens": 150,
            "completion_tokens": 80,
            "total_tokens": 230,
        }
    return step


def make_mock_agent_result(
    model: str = "gpt-5.2-chat",
    success: bool = True,
    num_tool_calls: int = 2,
    total_cost: float = 0.005,
    total_tokens: int = 500,
) -> dict[str, Any]:
    """Build a realistic run_agent result dict."""
    steps: list[dict[str, Any]] = []
    tool_names = ["search_products", "add_to_cart", "checkout", "get_price"]
    for i in range(num_tool_calls):
        steps.append(
            make_agent_step(
                step_index=i,
                action="tool_call",
                tool_name=tool_names[i % len(tool_names)],
                model=model,
                duration_ms=80.0 + i * 20.0,
            )
        )
    # Final LLM response step
    steps.append(
        make_agent_step(
            step_index=num_tool_calls,
            action="llm_response",
            llm_output="Order placed successfully for laptop. Total: $999.99",
            model=model,
            duration_ms=150.0,
        )
    )

    return {
        "steps": steps,
        "final_output": "Order placed successfully for laptop. Total: $999.99",
        "success": success,
        "error": None if success else "Agent failed",
        "total_duration_ms": sum(s["duration_ms"] for s in steps),
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "model": model,
        "step_count": len(steps),
    }


def make_trial_result(
    passed: bool = True,
    model: str = "gpt-5.2-chat",
    scenario_id: str = "ecommerce",
    cost_usd: float = 0.005,
    tokens: int = 500,
    trial_index: int = 0,
    include_steps: bool = True,
) -> dict[str, Any]:
    """Build a realistic trial result dict (as emitted by run_single_trial)."""
    steps: list[dict[str, Any]] = []
    if include_steps:
        steps = [
            make_agent_step(step_index=0, action="tool_call", tool_name="search"),
            make_agent_step(step_index=1, action="tool_call", tool_name="add_to_cart"),
            make_agent_step(step_index=2, action="llm_response", llm_output="Done."),
        ]
    return {
        "trial_id": str(uuid.uuid4()),
        "trial_index": trial_index,
        "scenario_id": scenario_id,
        "model": model,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "scoring_details": {},
        "success": True,
        "error": None,
        "duration_ms": 350.0,
        "cost_usd": cost_usd,
        "tokens": tokens,
        "step_count": len(steps),
        "_steps": steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def make_baseline_results(
    n: int = 20,
    pass_rate: float = 0.8,
    model: str = "gpt-5.2-chat",
) -> list[dict[str, Any]]:
    """Build a list of baseline trial results with a given pass rate."""
    results = []
    n_pass = int(n * pass_rate)
    for i in range(n):
        results.append(
            make_trial_result(
                passed=(i < n_pass),
                model=model,
                trial_index=i,
                cost_usd=0.005,
                tokens=500,
            )
        )
    return results


# -------------------------------------------------------------------
# Mock Azure client
# -------------------------------------------------------------------


class MockAzureFoundryClient:
    """A mock AzureFoundryClient that returns deterministic results.

    Does NOT call any real Azure APIs. All methods return canned data.
    """

    def __init__(
        self,
        default_success: bool = True,
        default_model: str = "gpt-5.2-chat",
    ) -> None:
        self._default_success = default_success
        self._default_model = default_model
        self._call_count = 0

    async def run_agent(
        self,
        model: str = "",
        system_prompt: str = "",
        user_input: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_steps: int = 10,
        temperature: float = 0.7,
        tool_executor: Any | None = None,
    ) -> dict[str, Any]:
        """Return a canned agent result."""
        self._call_count += 1
        return make_mock_agent_result(
            model=model or self._default_model,
            success=self._default_success,
            total_cost=0.005,
            total_tokens=500,
        )

    async def chat_completion(
        self,
        model: str = "",
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a canned chat completion result."""
        self._call_count += 1
        return {
            "content": "Test response",
            "tool_calls": [],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 80,
                "total_tokens": 230,
            },
            "model": model or self._default_model,
            "finish_reason": "stop",
            "estimated_cost_usd": 0.002,
        }

    def available_models(self) -> list[str]:
        """Return the full model list."""
        return list(ALL_MODELS)

    async def close(self) -> None:
        """No-op close."""
        pass

    @property
    def call_count(self) -> int:
        return self._call_count


# -------------------------------------------------------------------
# Scoring fixture
# -------------------------------------------------------------------


def dummy_scorer(trace: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """A trivial scorer that always passes."""
    return {
        "passed": True,
        "score": 1.0,
        "scoring_details": {"reason": "Mock scorer passes always"},
    }


def mixed_scorer_factory(pass_rate: float = 0.6):
    """Return a scorer that passes a fraction of trials."""
    _counter = {"n": 0}

    def _score(trace: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
        _counter["n"] += 1
        passed = (_counter["n"] % int(1.0 / (1.0 - pass_rate + 1e-9))) != 0
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "scoring_details": {},
        }

    return _score


# -------------------------------------------------------------------
# E7 config fixtures
# -------------------------------------------------------------------


@pytest.fixture
def e7_approaches() -> list[dict[str, Any]]:
    """The 5 E7 approach configs."""
    return [
        {"name": "fixed_n", "trials": 5},
        {"name": "sprt_only", "max_trials": 20, "use_sprt": True},
        {
            "name": "sprt_fingerprint",
            "max_trials": 20,
            "use_sprt": True,
            "use_fingerprinting": True,
        },
        {
            "name": "sprt_fp_budget",
            "calibration_size": 5,
            "use_sprt": True,
            "use_fingerprinting": True,
            "use_adaptive_budget": True,
        },
        {
            "name": "full_system",
            "calibration_size": 5,
            "use_sprt": True,
            "use_fingerprinting": True,
            "use_adaptive_budget": True,
            "use_trace_first": True,
        },
    ]


@pytest.fixture
def e7_params() -> dict[str, Any]:
    """E7 statistical parameters."""
    return {
        "alpha": 0.05,
        "beta": 0.10,
        "delta": 0.10,
        "repetitions": 1,
    }


@pytest.fixture
def e7_experiment_config(e7_approaches, e7_params) -> dict[str, Any]:
    """Minimal E7 experiment config dict."""
    return {
        "experiment_id": "e7_efficiency",
        "experiment": {"id": "e7_efficiency", "name": "E7 Test"},
        "models": [{"name": "gpt-5.2-chat", "subscription": 1}],
        "scenarios": ["ecommerce"],
        "parameters": {
            "approaches": e7_approaches,
            **e7_params,
        },
        "temperature": 0.7,
        "max_steps": 5,
    }


@pytest.fixture
def mock_client() -> MockAzureFoundryClient:
    """A fresh mock Azure client."""
    return MockAzureFoundryClient()


@pytest.fixture
def e7_deps():
    """Lazy-load the real E7 dependencies from agentassay modules."""
    from experiments.runner.daemon import _load_e7_dependencies

    return _load_e7_dependencies()


@pytest.fixture
def baseline_results_20() -> list[dict[str, Any]]:
    """20 baseline results with 80% pass rate."""
    return make_baseline_results(n=20, pass_rate=0.8)


@pytest.fixture
def baseline_results_high() -> list[dict[str, Any]]:
    """20 baseline results with 100% pass rate."""
    return make_baseline_results(n=20, pass_rate=1.0)


@pytest.fixture
def baseline_results_low() -> list[dict[str, Any]]:
    """20 baseline results with 0% pass rate (all fail)."""
    return make_baseline_results(n=20, pass_rate=0.0)


@pytest.fixture
def scenario_dict() -> dict[str, Any]:
    """A minimal scenario dict for E7 approach runners."""
    return {
        "scenario_id": "ecommerce",
        "name": "ecommerce",
        "system_prompt": "You are a helpful ecommerce agent.",
        "user_input": "Complete the ecommerce task.",
        "tools": [],
        "tool_responses": {},
        "expected": {},
    }
