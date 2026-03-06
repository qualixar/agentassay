# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Contract evaluation data models for AgentAssay.

Defines the frozen Pydantic v2 models that represent the output of
contract evaluation: ``ContractViolation`` (a single constraint violation)
and ``ContractEvaluation`` (the aggregate result with score and verdict).

These models are immutable after creation to guarantee that evaluation
results are never silently mutated downstream.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractViolation(BaseModel):
    """A single constraint violation detected during contract evaluation.

    Each violation records which constraint was violated, on which step,
    what was expected vs. what was observed, and the severity.

    Severity semantics (from ABC):
        - **hard**: The constraint MUST be satisfied. A hard violation means
          the agent is definitively non-compliant. Score impact: full penalty.
        - **soft**: The constraint SHOULD be satisfied. A soft violation is
          a warning -- the agent is suboptimal but not broken. Score impact:
          partial penalty.
    """

    model_config = ConfigDict(frozen=True)

    contract_name: str = Field(
        min_length=1,
        description="Name of the parent contract.",
    )
    constraint_name: str = Field(
        min_length=1,
        description="Name of the specific constraint that was violated.",
    )
    constraint_type: Literal["precondition", "postcondition", "invariant", "guardrail"] = Field(
        description="Category of the violated constraint.",
    )
    violated_at_step: int | None = Field(
        default=None,
        description=(
            "Step index where the violation was detected, or None for "
            "trace-level constraints (postconditions, guardrails)."
        ),
    )
    expected: str = Field(
        min_length=1,
        description="Human-readable description of what was expected.",
    )
    actual: str = Field(
        min_length=1,
        description="Human-readable description of what was observed.",
    )
    severity: Literal["hard", "soft"] = Field(
        description="Constraint severity from the contract definition.",
    )


class ContractEvaluation(BaseModel):
    """Result of evaluating one execution trace against a behavioral contract.

    This is the primary output of the ``ContractOracle``. It aggregates
    all violations found and computes a normalized score.

    Score semantics:
        - 1.0: No violations -- the trace fully satisfies the contract.
        - 0.0: Maximum violations (all hard constraints violated).
        - Between: Proportional reduction per violation.
          Hard violations reduce the score by ``1.0 / total_constraints``.
          Soft violations reduce the score by ``0.5 / total_constraints``.
    """

    model_config = ConfigDict(frozen=True)

    contract_name: str = Field(
        min_length=1,
        description="Name of the evaluated contract.",
    )
    passed: bool = Field(
        description=(
            "True if NO hard violations were detected. Soft violations alone do not cause failure."
        ),
    )
    violations: list[ContractViolation] = Field(
        default_factory=list,
        description="All violations detected (both hard and soft).",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized compliance score in [0.0, 1.0].",
    )
    trace_id: str = Field(
        min_length=1,
        description="ID of the evaluated execution trace.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evaluation was computed (UTC).",
    )
