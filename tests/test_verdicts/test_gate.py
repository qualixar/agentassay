"""Tests for deployment gate module.

Tests DeploymentGate, GateConfig, GateReport, and suite evaluation.

Target: ~15 tests.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from agentassay.verdicts.gate import (
    DeploymentGate,
    GateConfig,
    GateDecision,
    GateReport,
)
from agentassay.verdicts.verdict import StochasticVerdict, VerdictStatus


def _make_verdict(
    status: VerdictStatus = VerdictStatus.PASS,
    pass_rate: float = 0.9,
    ci: tuple[float, float] = (0.85, 0.95),
    num_trials: int = 100,
    num_passed: int = 90,
    p_value: float | None = None,
    regression_detected: bool = False,
    details: dict | None = None,
) -> StochasticVerdict:
    return StochasticVerdict(
        status=status,
        confidence=0.95,
        pass_rate=pass_rate,
        pass_rate_ci=ci,
        num_trials=num_trials,
        num_passed=num_passed,
        p_value=p_value,
        regression_detected=regression_detected,
        details=details or {},
    )


class TestGateConfig:
    """Tests for GateConfig validation."""

    def test_default_values(self):
        cfg = GateConfig()
        assert cfg.min_pass_rate == 0.80
        assert cfg.block_on_inconclusive is False

    def test_high_regression_pct_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GateConfig(max_regression_pct=0.6)
            assert len(w) >= 1
            assert "permissive" in str(w[0].message).lower()

    def test_min_pass_rate_bounds(self):
        with pytest.raises(ValidationError):
            GateConfig(min_pass_rate=-0.1)
        with pytest.raises(ValidationError):
            GateConfig(min_pass_rate=1.5)

    def test_min_trials_must_be_positive(self):
        with pytest.raises(ValidationError):
            GateConfig(min_trials_per_scenario=0)


class TestGateDecision:
    """Tests for GateDecision enum."""

    def test_deploy_value(self):
        assert GateDecision.DEPLOY == "deploy"

    def test_block_value(self):
        assert GateDecision.BLOCK == "block"

    def test_warn_value(self):
        assert GateDecision.WARN == "warn"

    def test_skip_value(self):
        assert GateDecision.SKIP == "skip"


class TestDeploymentGate:
    """Tests for DeploymentGate evaluation."""

    def test_deploy_on_pass(self):
        gate = DeploymentGate()
        verdict = _make_verdict(status=VerdictStatus.PASS)
        decision, reason = gate.evaluate_scenario(verdict)
        assert decision == GateDecision.DEPLOY

    def test_block_on_fail(self):
        gate = DeploymentGate()
        verdict = _make_verdict(
            status=VerdictStatus.FAIL,
            pass_rate=0.3,
            ci=(0.2, 0.4),
            num_passed=30,
        )
        decision, reason = gate.evaluate_scenario(verdict)
        assert decision == GateDecision.BLOCK

    def test_warn_on_inconclusive(self):
        gate = DeploymentGate(GateConfig(block_on_inconclusive=False))
        verdict = _make_verdict(
            status=VerdictStatus.INCONCLUSIVE,
            pass_rate=0.7,
            ci=(0.6, 0.8),
            num_passed=70,
        )
        decision, reason = gate.evaluate_scenario(verdict)
        assert decision == GateDecision.WARN

    def test_block_on_inconclusive_when_configured(self):
        gate = DeploymentGate(GateConfig(block_on_inconclusive=True))
        verdict = _make_verdict(
            status=VerdictStatus.INCONCLUSIVE,
            pass_rate=0.7,
            ci=(0.6, 0.8),
            num_passed=70,
        )
        decision, reason = gate.evaluate_scenario(verdict)
        assert decision == GateDecision.BLOCK

    def test_skip_on_insufficient_trials(self):
        gate = DeploymentGate(GateConfig(min_trials_per_scenario=100))
        verdict = _make_verdict(num_trials=10, num_passed=9)
        decision, reason = gate.evaluate_scenario(verdict)
        assert decision == GateDecision.SKIP

    def test_block_on_low_ci_upper(self):
        gate = DeploymentGate(GateConfig(min_pass_rate=0.90))
        verdict = _make_verdict(
            status=VerdictStatus.PASS,
            pass_rate=0.7,
            ci=(0.6, 0.8),
            num_passed=70,
        )
        decision, _ = gate.evaluate_scenario(verdict)
        assert decision == GateDecision.BLOCK


class TestEvaluateSuite:
    """Tests for suite-level evaluation."""

    def test_all_pass_deploys(self):
        gate = DeploymentGate()
        verdicts = {
            "s1": _make_verdict(status=VerdictStatus.PASS),
            "s2": _make_verdict(status=VerdictStatus.PASS),
        }
        overall, per = gate.evaluate_suite(verdicts)
        assert overall == GateDecision.DEPLOY

    def test_one_fail_blocks_all(self):
        gate = DeploymentGate()
        verdicts = {
            "s1": _make_verdict(status=VerdictStatus.PASS),
            "s2": _make_verdict(
                status=VerdictStatus.FAIL,
                pass_rate=0.3,
                ci=(0.2, 0.4),
                num_passed=30,
            ),
        }
        overall, per = gate.evaluate_suite(verdicts)
        assert overall == GateDecision.BLOCK

    def test_empty_verdicts_skip(self):
        gate = DeploymentGate()
        overall, per = gate.evaluate_suite({})
        assert overall == GateDecision.SKIP

    def test_mixed_warn_and_deploy(self):
        gate = DeploymentGate(GateConfig(block_on_inconclusive=False))
        verdicts = {
            "s1": _make_verdict(status=VerdictStatus.PASS),
            "s2": _make_verdict(
                status=VerdictStatus.INCONCLUSIVE,
                pass_rate=0.7,
                ci=(0.6, 0.8),
                num_passed=70,
            ),
        }
        overall, per = gate.evaluate_suite(verdicts)
        assert overall == GateDecision.WARN


class TestGateReport:
    """Tests for GateReport."""

    def test_exit_code_deploy(self):
        gate = DeploymentGate()
        verdicts = {
            "s1": _make_verdict(status=VerdictStatus.PASS),
        }
        report = gate.evaluate_suite_detailed(verdicts)
        assert isinstance(report, GateReport)
        assert report.exit_code == 0

    def test_exit_code_block(self):
        gate = DeploymentGate()
        verdicts = {
            "s1": _make_verdict(
                status=VerdictStatus.FAIL,
                pass_rate=0.3,
                ci=(0.2, 0.4),
                num_passed=30,
            ),
        }
        report = gate.evaluate_suite_detailed(verdicts)
        assert report.exit_code == 1

    def test_counts_correct(self):
        gate = DeploymentGate()
        verdicts = {
            "s1": _make_verdict(status=VerdictStatus.PASS),
            "s2": _make_verdict(status=VerdictStatus.PASS),
        }
        report = gate.evaluate_suite_detailed(verdicts)
        assert report.total_scenarios == 2
        assert report.passed_scenarios == 2
        assert report.blocked_scenarios == 0

    def test_empty_report(self):
        gate = DeploymentGate()
        report = gate.evaluate_suite_detailed({})
        assert report.total_scenarios == 0
        assert report.exit_code == 3  # SKIP
