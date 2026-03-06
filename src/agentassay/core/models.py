# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Core data models for AgentAssay.

Foundational types for formal regression testing of non-deterministic
AI agent workflows. All trace/result models are frozen (immutable) to
guarantee reproducibility. Configuration models are mutable.

Uses Pydantic v2 throughout with Python 3.10+ type syntax.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """UTC-aware timestamp."""
    return datetime.now(timezone.utc)


# ===================================================================
# Trace Models (frozen — immutable after creation)
# ===================================================================


class StepTrace(BaseModel):
    """A single step in an agent's execution trace.

    Captures exactly one atomic action: a tool call, an LLM generation,
    a decision branch, or any custom action type. Steps are ordered by
    ``step_index`` within their parent ``ExecutionTrace``.
    """

    model_config = ConfigDict(frozen=True)

    step_id: str = Field(default_factory=_uuid)
    step_index: int = Field(ge=0)
    action: str = Field(
        min_length=1,
        description=(
            "Atomic action type: 'tool_call', 'llm_response', "
            "'decision', 'retrieval', 'observation', etc."
        ),
    )

    # Tool-call fields (populated when action == "tool_call")
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None

    # LLM fields (populated when action involves an LLM)
    llm_input: str | None = None
    llm_output: str | None = None
    model: str | None = None

    # Timing
    timestamp: datetime = Field(default_factory=_now)
    duration_ms: float = Field(ge=0.0)

    # Extensible
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- Validators ----------------------------------------------------------

    @field_validator("action")
    @classmethod
    def _action_lowercase(cls, v: str) -> str:
        return v.strip().lower()

    @model_validator(mode="after")
    def _tool_fields_consistency(self) -> StepTrace:
        """If action is 'tool_call', tool_name must be set."""
        if self.action == "tool_call" and not self.tool_name:
            raise ValueError("tool_name is required when action is 'tool_call'")
        return self


class ExecutionTrace(BaseModel):
    """A complete agent execution for a single trial.

    Represents one invocation of the agent-under-test on a given input.
    Contains the ordered sequence of ``StepTrace`` objects, the final
    output, cost accounting, and success/failure status.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(default_factory=_uuid)
    scenario_id: str = Field(min_length=1)

    steps: list[StepTrace] = Field(default_factory=list)

    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: Any = None

    success: bool = False
    error: str | None = None

    total_duration_ms: float = Field(ge=0.0, default=0.0)
    total_cost_usd: float = Field(ge=0.0, default=0.0)

    model: str = Field(min_length=1)
    framework: str = Field(min_length=1)

    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- Computed properties -------------------------------------------------

    @property
    def tools_used(self) -> set[str]:
        """All unique tool names invoked during this execution."""
        return {s.tool_name for s in self.steps if s.tool_name is not None}

    @property
    def step_count(self) -> int:
        """Total number of steps in the trace."""
        return len(self.steps)

    @property
    def decision_path(self) -> list[str]:
        """Ordered sequence of actions taken (the behavioral fingerprint)."""
        return [s.action for s in self.steps]

    # -- Validators ----------------------------------------------------------

    @model_validator(mode="after")
    def _steps_ordered(self) -> ExecutionTrace:
        """Ensure step indices are monotonically increasing."""
        indices = [s.step_index for s in self.steps]
        if indices != sorted(indices):
            raise ValueError(
                f"StepTrace.step_index values must be monotonically increasing; got {indices}"
            )
        return self


# ===================================================================
# Test Specification Models
# ===================================================================


class TestScenario(BaseModel):
    """Defines a single test case for an agent.

    A scenario is a *template*: it specifies the input, the expected
    properties (the oracle), and optionally a named evaluator function.
    It does NOT contain results -- those live in ``TrialResult``.
    """

    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""

    input_data: dict[str, Any] = Field(default_factory=dict)

    # Declarative expected-property bag.  Examples:
    #   {"max_steps": 10, "must_use_tools": ["search", "calculate"]}
    #   {"output_contains": "Paris", "max_cost_usd": 0.05}
    expected_properties: dict[str, Any] = Field(default_factory=dict)

    # Name of a registered evaluator function (resolved at runtime).
    evaluator: str | None = None

    tags: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(gt=0.0, default=300.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ===================================================================
# Result Models (frozen)
# ===================================================================


class TrialResult(BaseModel):
    """Result of running one trial: execution trace + evaluation verdict.

    ``passed`` is the boolean verdict for THIS individual trial.
    ``score`` is a continuous quality measure in [0, 1].
    """

    model_config = ConfigDict(frozen=True)

    trial_id: str = Field(default_factory=_uuid)
    scenario_id: str = Field(min_length=1)

    trace: ExecutionTrace
    passed: bool = False
    score: float = Field(ge=0.0, le=1.0, default=0.0)

    evaluation_details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)


# ===================================================================
# Configuration Models (mutable)
# ===================================================================


class AgentConfig(BaseModel):
    """Configuration describing the agent under test.

    ``framework`` is constrained to known agent frameworks so that
    framework-specific adapters can be selected automatically.
    """

    model_config = ConfigDict(frozen=False)

    agent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)

    framework: Literal[
        "langgraph",
        "crewai",
        "autogen",
        "openai",
        "smolagents",
        "semantic_kernel",
        "bedrock",
        "mcp",
        "vertex",
        "custom",
    ]

    model: str = Field(min_length=1)
    version: str = "0.0.0"
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssayConfig(BaseModel):
    """Master configuration for a test run (an *assay*).

    Controls the statistical parameters, resource limits, and execution
    strategy for running N stochastic trials of an agent.

    Terminology:
        - ``num_trials``: how many times to invoke the agent per scenario.
        - ``significance_level`` (alpha): Type-I error rate.
        - ``power`` (1 - beta): probability of detecting a true regression.
        - ``effect_size_threshold``: minimum pass-rate drop to flag as regression.
        - ``confidence_method``: interval estimator for pass rate.
        - ``regression_test``: statistical test for comparing two trial sets.
        - ``use_sprt``: enable Wald's Sequential Probability Ratio Test for
          early stopping (saves cost when the answer is clear early).
    """

    model_config = ConfigDict(frozen=False)

    # --- Statistical parameters ---
    num_trials: int = Field(default=30, ge=1, le=10_000)
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    power: float = Field(default=0.80, gt=0.0, lt=1.0)
    effect_size_threshold: float = Field(default=0.10, ge=0.0, le=1.0)

    confidence_method: Literal["wilson", "clopper-pearson", "normal"] = "wilson"

    regression_test: Literal["fisher", "chi2", "ks", "mann-whitney"] = "fisher"

    # --- Sequential testing (SPRT) ---
    use_sprt: bool = False
    sprt_strength: float = Field(
        default=0.0,
        ge=0.0,
        description=("Wald log-likelihood ratio threshold. 0 means auto-derive from alpha/beta."),
    )

    # --- Reproducibility ---
    seed: int | None = None

    # --- Resource limits ---
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    max_cost_usd: float = Field(default=50.0, ge=0.0)
    parallel_trials: int = Field(default=1, ge=1, le=256)

    # --- Extensible ---
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- Validators ----------------------------------------------------------

    @field_validator("num_trials")
    @classmethod
    def _num_trials_warning(cls, v: int) -> int:
        """Stochastic testing with < 10 trials is unreliable."""
        if v < 10:
            import warnings

            warnings.warn(
                f"num_trials={v} is very low for stochastic testing; "
                "results will have wide confidence intervals. "
                "Consider >= 30 trials for meaningful statistics.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def _sprt_consistency(self) -> AssayConfig:
        """If SPRT is enabled but strength is 0, that's fine (auto-derive).
        But alpha + beta must be < 1 for SPRT to make sense."""
        if self.use_sprt:
            beta = 1.0 - self.power
            if self.significance_level + beta >= 1.0:
                raise ValueError(
                    "SPRT requires alpha + beta < 1.0; got "
                    f"alpha={self.significance_level}, beta={beta}"
                )
        return self
