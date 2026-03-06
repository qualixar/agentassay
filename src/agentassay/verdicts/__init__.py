# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Verdicts module — stochastic test semantics and deployment gates.

This module is the decision-making core of AgentAssay. It transforms
raw statistical results from trial executions into actionable verdicts
(PASS/FAIL/INCONCLUSIVE) and deployment gate decisions (DEPLOY/BLOCK/WARN).

Key Components:
    StochasticVerdict: Statistically-backed test verdict with CI and p-value.
    VerdictStatus: Three-valued verdict enum (PASS, FAIL, INCONCLUSIVE).
    VerdictFunction: The (alpha, beta, n)-test triple from Definition 3.2.
    DeploymentGate: CI/CD integration that maps verdicts to deploy decisions.
    GateDecision: Four-valued gate enum (DEPLOY, BLOCK, WARN, SKIP).
    GateConfig: Configuration for deployment gate behavior.
    GateReport: Detailed evaluation report for CI/CD reporting.

Usage:
    >>> from agentassay.verdicts import VerdictFunction, DeploymentGate, GateConfig
    >>>
    >>> # Single-version threshold test
    >>> vf = VerdictFunction(alpha=0.05, min_trials=30)
    >>> results = [True] * 27 + [False] * 3
    >>> verdict = vf.evaluate_single(results, threshold=0.80)
    >>> print(verdict.status)  # VerdictStatus.PASS
    >>>
    >>> # Cross-version regression test
    >>> baseline = [True] * 90 + [False] * 10
    >>> current = [True] * 75 + [False] * 25
    >>> verdict = vf.evaluate_regression(baseline, current)
    >>> print(verdict.regression_detected)  # True
    >>>
    >>> # CI/CD deployment gate
    >>> gate = DeploymentGate(GateConfig(min_pass_rate=0.85))
    >>> overall, per_scenario = gate.evaluate_suite({"qa": verdict})
    >>> print(overall)  # GateDecision.BLOCK
"""

from agentassay.verdicts.gate import (
    DeploymentGate,
    GateConfig,
    GateDecision,
    GateReport,
)
from agentassay.verdicts.verdict import (
    StochasticVerdict,
    VerdictFunction,
    VerdictStatus,
)

__all__ = [
    "DeploymentGate",
    "GateConfig",
    "GateDecision",
    "GateReport",
    "StochasticVerdict",
    "VerdictFunction",
    "VerdictStatus",
]
