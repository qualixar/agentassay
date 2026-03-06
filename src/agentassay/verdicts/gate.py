"""Deployment gate semantics for CI/CD integration.

This module translates stochastic test verdicts into CI/CD deployment
decisions. It bridges the statistical world of agent testing with the
operational world of deployment pipelines.

Concept:
    A deployment gate is a checkpoint in a CI/CD pipeline that decides
    whether an agent build should proceed to production. Unlike traditional
    gates that check binary test results, agent deployment gates must
    handle three-valued logic (PASS/FAIL/INCONCLUSIVE) and aggregate
    multiple scenario verdicts into a single deploy/block decision.

Gate Semantics:
    The gate maps StochasticVerdicts to GateDecisions:

        PASS verdict         -> DEPLOY  (safe to deploy)
        FAIL verdict         -> BLOCK   (regression detected, block deployment)
        INCONCLUSIVE verdict -> WARN    (proceed with caution)
                             -> BLOCK   (if block_on_inconclusive=True)

    Suite-level aggregation (all scenarios combined):
        DEPLOY only if ALL required scenarios are DEPLOY
        BLOCK  if ANY required scenario is BLOCK
        WARN   if any scenario is WARN but none are BLOCK

Integration:
    Designed to integrate with:
    - GitHub Actions (exit codes: 0=deploy, 1=block, 2=warn)
    - GitLab CI/CD (allow_failure semantics)
    - Jenkins (build result mapping)
    - Any CI system with pass/fail/unstable states

Example:
    >>> from agentassay.verdicts.gate import DeploymentGate, GateConfig
    >>> gate = DeploymentGate(GateConfig(min_pass_rate=0.85))
    >>> overall, per_scenario = gate.evaluate_suite(scenario_verdicts)
    >>> if overall == GateDecision.BLOCK:
    ...     sys.exit(1)  # Fail the CI job
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentassay.verdicts.verdict import StochasticVerdict, VerdictStatus


class GateDecision(StrEnum):
    """CI/CD deployment gate decision.

    Maps to standard CI exit codes and status indicators.
    """

    DEPLOY = "deploy"
    """Safe to deploy. All checks passed with statistical confidence."""

    BLOCK = "block"
    """Regression detected. Deployment must be blocked."""

    WARN = "warn"
    """Inconclusive results. Proceed with caution.

    Human review recommended. The statistical evidence was not strong
    enough to determine pass or fail definitively.
    """

    SKIP = "skip"
    """Insufficient data to make any decision.

    Typically occurs when no trials were run or all scenarios were
    excluded from evaluation.
    """


class GateConfig(BaseModel):
    """Configuration for deployment gate behavior.

    These parameters control how strict the gate is and how it handles
    edge cases like inconclusive results and missing scenarios.

    Attributes:
        min_pass_rate: Minimum acceptable pass rate for single-version tests.
            A scenario's observed pass rate must exceed this threshold
            (with statistical confidence) to pass the gate.
        max_regression_pct: Maximum acceptable drop in pass rate between
            versions. A regression of 5% means if baseline was 0.90,
            current must be >= 0.855 (0.90 * (1 - 0.05)).
        require_all_scenarios: If True, ALL scenarios must pass for the
            gate to open. If False, only scenarios marked as required
            must pass.
        block_on_inconclusive: If True, INCONCLUSIVE verdicts are treated
            as BLOCK (conservative). If False, treated as WARN (permissive).
        min_trials_per_scenario: Minimum trials required per scenario.
            Scenarios with fewer trials produce SKIP decisions.
    """

    min_pass_rate: float = Field(default=0.80, ge=0.0, le=1.0)
    max_regression_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    require_all_scenarios: bool = True
    block_on_inconclusive: bool = False
    min_trials_per_scenario: int = Field(default=30, ge=1)

    @field_validator("max_regression_pct")
    @classmethod
    def _validate_regression_pct(cls, v: float) -> float:
        if v > 0.5:
            import warnings

            warnings.warn(
                f"max_regression_pct={v} is very permissive (>50% drop allowed). "
                "Consider using a stricter threshold.",
                UserWarning,
                stacklevel=2,
            )
        return v


class GateReport(BaseModel):
    """Detailed report from a deployment gate evaluation.

    Contains the overall decision, per-scenario decisions, and metadata
    useful for CI/CD reporting and audit trails.

    Attributes:
        overall_decision: The aggregate gate decision.
        scenario_decisions: Mapping of scenario name to gate decision.
        scenario_reasons: Mapping of scenario name to human-readable reason.
        total_scenarios: Number of scenarios evaluated.
        passed_scenarios: Number of scenarios that passed.
        blocked_scenarios: Number of scenarios that blocked.
        warned_scenarios: Number of scenarios with warnings.
        skipped_scenarios: Number of scenarios skipped.
        config: The gate configuration used.
        timestamp: When the evaluation was performed (UTC).
    """

    model_config = {"frozen": True}

    overall_decision: GateDecision
    scenario_decisions: dict[str, GateDecision]
    scenario_reasons: dict[str, str] = Field(default_factory=dict)
    total_scenarios: int = Field(ge=0)
    passed_scenarios: int = Field(ge=0)
    blocked_scenarios: int = Field(ge=0)
    warned_scenarios: int = Field(ge=0)
    skipped_scenarios: int = Field(ge=0)
    config: GateConfig
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def exit_code(self) -> int:
        """Return a CI-compatible exit code.

        Returns:
            0 for DEPLOY, 1 for BLOCK, 2 for WARN, 3 for SKIP.
        """
        return {
            GateDecision.DEPLOY: 0,
            GateDecision.BLOCK: 1,
            GateDecision.WARN: 2,
            GateDecision.SKIP: 3,
        }[self.overall_decision]


class DeploymentGate:
    """Deployment gate that maps stochastic verdicts to CI/CD decisions.

    The gate is the bridge between statistical agent testing and operational
    deployment pipelines. It aggregates per-scenario verdicts into a single
    deploy/block decision with configurable strictness.

    Example:
        >>> config = GateConfig(min_pass_rate=0.85, block_on_inconclusive=True)
        >>> gate = DeploymentGate(config)
        >>> decision = gate.evaluate_scenario(some_verdict)
        >>> overall, per_scenario = gate.evaluate_suite(all_verdicts)

    Args:
        config: Gate configuration. Uses defaults if not provided.
    """

    __slots__ = ("_config",)

    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config or GateConfig()

    @property
    def config(self) -> GateConfig:
        """The gate configuration."""
        return self._config

    def evaluate_scenario(
        self,
        verdict: StochasticVerdict,
    ) -> tuple[GateDecision, str]:
        """Map a single scenario's verdict to a gate decision.

        Decision mapping:
            1. If trials < min_trials_per_scenario -> SKIP
            2. If verdict is FAIL -> BLOCK
            3. If verdict is PASS:
               a. If regression detected (from details) -> BLOCK
               b. If pass_rate < min_pass_rate AND CI upper < min_pass_rate -> BLOCK
               c. Otherwise -> DEPLOY
            4. If verdict is INCONCLUSIVE:
               a. If block_on_inconclusive -> BLOCK
               b. Otherwise -> WARN

        Args:
            verdict: The StochasticVerdict for this scenario.

        Returns:
            Tuple of (GateDecision, reason_string).
        """
        # Check minimum trial count
        if verdict.num_trials < self._config.min_trials_per_scenario:
            reason = (
                f"Insufficient trials: {verdict.num_trials} < "
                f"{self._config.min_trials_per_scenario} minimum"
            )
            return (GateDecision.SKIP, reason)

        if verdict.status == VerdictStatus.FAIL:
            reason = self._build_fail_reason(verdict)
            return (GateDecision.BLOCK, reason)

        if verdict.status == VerdictStatus.PASS:
            # Double-check: even if the verdict says PASS, verify pass rate
            # is above our gate's minimum threshold
            if verdict.pass_rate_ci[1] < self._config.min_pass_rate:
                reason = (
                    f"Pass rate CI upper ({verdict.pass_rate_ci[1]:.4f}) "
                    f"below gate minimum ({self._config.min_pass_rate:.4f})"
                )
                return (GateDecision.BLOCK, reason)

            # Check regression magnitude if this is a regression verdict
            if self._is_regression_beyond_tolerance(verdict):
                baseline_rate = verdict.details.get("baseline_pass_rate")
                reason = (
                    f"Pass rate dropped beyond tolerance: "
                    f"baseline={baseline_rate:.4f}, "
                    f"current={verdict.pass_rate:.4f}, "
                    f"max_drop={self._config.max_regression_pct:.2%}"
                )
                return (GateDecision.BLOCK, reason)

            reason = (
                f"Pass: rate={verdict.pass_rate:.4f}, "
                f"CI=[{verdict.pass_rate_ci[0]:.4f}, {verdict.pass_rate_ci[1]:.4f}]"
            )
            return (GateDecision.DEPLOY, reason)

        # INCONCLUSIVE
        if self._config.block_on_inconclusive:
            reason = (
                f"Inconclusive verdict blocked by policy "
                f"(block_on_inconclusive=True). "
                f"Rate={verdict.pass_rate:.4f}, "
                f"CI=[{verdict.pass_rate_ci[0]:.4f}, {verdict.pass_rate_ci[1]:.4f}]"
            )
            return (GateDecision.BLOCK, reason)

        reason = (
            f"Inconclusive: rate={verdict.pass_rate:.4f}, "
            f"CI=[{verdict.pass_rate_ci[0]:.4f}, {verdict.pass_rate_ci[1]:.4f}]. "
            f"Consider running more trials."
        )
        return (GateDecision.WARN, reason)

    def evaluate_suite(
        self,
        verdicts: dict[str, StochasticVerdict],
    ) -> tuple[GateDecision, dict[str, GateDecision]]:
        """Evaluate all scenarios and produce an overall gate decision.

        Aggregation logic:
            1. Evaluate each scenario independently.
            2. If require_all_scenarios=True:
               - Overall DEPLOY only if ALL scenarios are DEPLOY.
               - Overall BLOCK if ANY scenario is BLOCK.
               - Overall WARN if ANY scenario is WARN (and none BLOCK).
               - Overall SKIP if ALL scenarios are SKIP.
            3. If require_all_scenarios=False:
               - Same logic, but SKIP scenarios are ignored.

        Args:
            verdicts: Mapping of scenario name to StochasticVerdict.

        Returns:
            Tuple of (overall_decision, per_scenario_decisions).
        """
        if not verdicts:
            return (GateDecision.SKIP, {})

        per_scenario: dict[str, GateDecision] = {}
        per_reason: dict[str, str] = {}

        for name, verdict in verdicts.items():
            decision, reason = self.evaluate_scenario(verdict)
            per_scenario[name] = decision
            per_reason[name] = reason

        overall = self._aggregate_decisions(per_scenario)

        return (overall, per_scenario)

    def evaluate_suite_detailed(
        self,
        verdicts: dict[str, StochasticVerdict],
    ) -> GateReport:
        """Evaluate all scenarios and produce a detailed gate report.

        Like evaluate_suite but returns a full GateReport with counts
        and reasons, suitable for CI reporting and audit trails.

        Args:
            verdicts: Mapping of scenario name to StochasticVerdict.

        Returns:
            GateReport with full evaluation details.
        """
        if not verdicts:
            return GateReport(
                overall_decision=GateDecision.SKIP,
                scenario_decisions={},
                scenario_reasons={},
                total_scenarios=0,
                passed_scenarios=0,
                blocked_scenarios=0,
                warned_scenarios=0,
                skipped_scenarios=0,
                config=self._config,
            )

        per_scenario: dict[str, GateDecision] = {}
        per_reason: dict[str, str] = {}

        for name, verdict in verdicts.items():
            decision, reason = self.evaluate_scenario(verdict)
            per_scenario[name] = decision
            per_reason[name] = reason

        overall = self._aggregate_decisions(per_scenario)
        decisions = list(per_scenario.values())

        return GateReport(
            overall_decision=overall,
            scenario_decisions=per_scenario,
            scenario_reasons=per_reason,
            total_scenarios=len(decisions),
            passed_scenarios=decisions.count(GateDecision.DEPLOY),
            blocked_scenarios=decisions.count(GateDecision.BLOCK),
            warned_scenarios=decisions.count(GateDecision.WARN),
            skipped_scenarios=decisions.count(GateDecision.SKIP),
            config=self._config,
        )

    def _aggregate_decisions(
        self, per_scenario: dict[str, GateDecision]
    ) -> GateDecision:
        """Aggregate per-scenario decisions into an overall decision.

        Args:
            per_scenario: Mapping of scenario name to gate decision.

        Returns:
            The overall gate decision.
        """
        decisions = list(per_scenario.values())

        if not decisions:
            return GateDecision.SKIP

        # Filter out SKIP if not requiring all scenarios
        active_decisions = decisions
        if not self._config.require_all_scenarios:
            active_decisions = [d for d in decisions if d != GateDecision.SKIP]
            if not active_decisions:
                return GateDecision.SKIP

        # BLOCK dominates everything
        if GateDecision.BLOCK in active_decisions:
            return GateDecision.BLOCK

        # WARN next priority
        if GateDecision.WARN in active_decisions:
            return GateDecision.WARN

        # All remaining scenarios must be DEPLOY or SKIP
        if all(
            d in (GateDecision.DEPLOY, GateDecision.SKIP) for d in active_decisions
        ):
            # If requiring all and some are SKIP, that means insufficient data
            if self._config.require_all_scenarios and GateDecision.SKIP in decisions:
                return GateDecision.SKIP
            return GateDecision.DEPLOY

        return GateDecision.SKIP

    def _is_regression_beyond_tolerance(
        self, verdict: StochasticVerdict
    ) -> bool:
        """Check if a regression exceeds the configured tolerance.

        Args:
            verdict: The verdict to check.

        Returns:
            True if regression exceeds max_regression_pct.
        """
        baseline_rate = verdict.details.get("baseline_pass_rate")
        if baseline_rate is None:
            return False

        if not isinstance(baseline_rate, (int, float)):
            return False

        if baseline_rate == 0.0:
            return False

        # Calculate the drop percentage
        drop = (baseline_rate - verdict.pass_rate) / baseline_rate
        return drop > self._config.max_regression_pct

    def _build_fail_reason(self, verdict: StochasticVerdict) -> str:
        """Build a human-readable failure reason from a verdict.

        Args:
            verdict: The failed verdict.

        Returns:
            Descriptive reason string.
        """
        parts = [f"Regression detected: rate={verdict.pass_rate:.4f}"]

        if verdict.p_value is not None:
            parts.append(f"p={verdict.p_value:.6f}")

        if verdict.effect_size is not None:
            parts.append(
                f"effect={verdict.effect_size:.4f}"
                f" ({verdict.effect_size_interpretation or 'unknown'})"
            )

        baseline_rate = verdict.details.get("baseline_pass_rate")
        if baseline_rate is not None:
            parts.append(f"baseline={baseline_rate:.4f}")

        reason_from_verdict = verdict.details.get("reason")
        if reason_from_verdict:
            parts.append(f"[{reason_from_verdict}]")

        return ", ".join(parts)

    def __repr__(self) -> str:
        return (
            f"DeploymentGate(min_pass_rate={self._config.min_pass_rate}, "
            f"max_regression_pct={self._config.max_regression_pct}, "
            f"block_on_inconclusive={self._config.block_on_inconclusive})"
        )
