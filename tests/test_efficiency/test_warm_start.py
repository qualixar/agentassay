"""Tests for Warm-Start SPRT — leveraging historical data for faster decisions.

Validates that WarmStartSPRT uses prior data to reach decisions faster than
cold-start SPRT, still makes correct accept/reject decisions, degrades to
standard SPRT when prior is empty, and computes meaningful savings estimates.

Target: ~8 tests covering speed, correctness, degradation, strong priors,
and savings computation.
"""

from __future__ import annotations

import pytest

from agentassay.efficiency.warm_start import WarmStartSPRT
from agentassay.statistics.sprt import SPRTRunner

from .conftest import make_traces, make_regressed_traces


# ===================================================================
# Helpers
# ===================================================================


def _run_cold_sprt(
    p0: float, p1: float, results: list[bool], alpha: float = 0.05, beta: float = 0.20
) -> int:
    """Run a standard (cold-start) SPRT and return trials used."""
    runner = SPRTRunner(p0=p0, p1=p1, alpha=alpha, beta=beta)
    trials_used = 0
    for passed in results:
        result = runner.update(passed=passed)
        trials_used = result.trials_used
        if result.decision != "continue":
            break
    return trials_used


def _extract_pass_sequence(traces) -> list[bool]:
    """Extract the pass/fail sequence from a list of ExecutionTrace objects."""
    return [t.success for t in traces]


# ===================================================================
# WarmStartSPRT — speed advantage
# ===================================================================


class TestWarmStartSpeed:
    """Tests for warm start requiring fewer trials than cold start."""

    def test_warm_start_fewer_trials(self):
        """Warm start with valid prior data uses fewer trials than cold start.

        We give the warm-start SPRT 10 historical baseline successes as prior,
        then feed 30 new baseline traces. The warm start should decide
        earlier because it starts with accumulated evidence.
        """
        new_data = make_traces(30, passed=True)

        p0 = 0.90
        delta = 0.30  # p1 = 0.60

        # Cold SPRT: no prior
        cold_sequence = _extract_pass_sequence(new_data)
        cold_trials = _run_cold_sprt(p0, p0 - delta, cold_sequence)

        # Warm SPRT: has 10 historical passes as prior (moderate, not overwhelming)
        warm = WarmStartSPRT(
            theta_0=p0, delta=delta, alpha=0.05, beta=0.20,
            prior_successes=10, prior_trials=10,
        )

        warm_trials = 0
        if not warm.is_decided:
            for passed in cold_sequence:
                decision = warm.update(passed)
                warm_trials = warm.trials_used
                if decision != "continue":
                    break

        # Warm start should need fewer new trials (or at most the same)
        assert warm_trials <= cold_trials


# ===================================================================
# WarmStartSPRT — decision correctness
# ===================================================================


class TestWarmStartCorrectness:
    """Tests for warm-start SPRT making correct statistical decisions."""

    def test_warm_start_correct_decision_accept(self):
        """Warm start correctly accepts H0 (no regression) for a passing agent.

        Historical data shows the agent passes consistently. New data also passes.
        The SPRT should accept H0 (baseline maintained).
        """
        new_data = make_traces(50, passed=True)

        warm = WarmStartSPRT(
            theta_0=0.90, delta=0.30, alpha=0.05, beta=0.20,
            prior_successes=10, prior_trials=10,
        )

        decision = warm.decision  # may already be decided from prior
        if not warm.is_decided:
            for passed in _extract_pass_sequence(new_data):
                decision = warm.update(passed)
                if decision != "continue":
                    break

        assert decision == "accept_h0"

    def test_warm_start_correct_decision_reject(self):
        """Warm start correctly accepts H1 (regression detected) for a failing agent.

        Historical data shows the agent was passing. New data shows failures.
        The SPRT should accept H1 (regression occurred).
        """
        new_data = make_regressed_traces(50, pass_rate=0.3)

        warm = WarmStartSPRT(
            theta_0=0.90, delta=0.40, alpha=0.05, beta=0.20,
            prior_successes=0, prior_trials=0,  # cold start, but with failures
        )

        decision = "continue"
        for passed in _extract_pass_sequence(new_data):
            decision = warm.update(passed)
            if decision != "continue":
                break

        assert decision == "accept_h1"


# ===================================================================
# WarmStartSPRT — degenerate cases
# ===================================================================


class TestWarmStartDegenerate:
    """Tests for edge cases: no prior and very strong prior."""

    def test_warm_start_with_zero_prior(self):
        """With no prior data, warm start degrades to a standard cold SPRT.

        The number of trials used should be the same as cold SPRT since
        there is no accumulated evidence.
        """
        new_data = make_traces(50, passed=True)
        sequence = _extract_pass_sequence(new_data)

        p0, delta = 0.90, 0.30

        cold_trials = _run_cold_sprt(p0, p0 - delta, sequence)

        warm = WarmStartSPRT(
            theta_0=p0, delta=delta, alpha=0.05, beta=0.20,
            prior_successes=0, prior_trials=0,
        )

        warm_trials = 0
        for passed in sequence:
            decision = warm.update(passed)
            warm_trials = warm.trials_used
            if decision != "continue":
                break

        # Without prior, warm SPRT should behave identically to cold SPRT
        assert warm_trials == cold_trials

    def test_warm_start_strong_prior(self):
        """With a very strong prior (100 historical passes), very few new trials needed.

        If historical data overwhelmingly supports H0, the prior alone may
        be sufficient to cross the boundary — or at most 1-2 new trials.
        """
        warm = WarmStartSPRT(
            theta_0=0.90, delta=0.40, alpha=0.05, beta=0.20,
            prior_successes=100, prior_trials=100,
        )

        decision = warm.decision
        trials = warm.trials_used

        if not warm.is_decided:
            new_data = make_traces(10, passed=True)
            for passed in _extract_pass_sequence(new_data):
                decision = warm.update(passed)
                trials = warm.trials_used
                if decision != "continue":
                    break

        # Should decide very quickly with 100 historical passes
        # (may even decide from prior alone => 0 new trials)
        assert trials <= 5
        assert decision == "accept_h0"


# ===================================================================
# WarmStartSPRT — savings estimation
# ===================================================================


class TestWarmStartSavings:
    """Tests for savings metrics computation."""

    def test_expected_savings(self):
        """Savings estimate is a reasonable percentage (between 0% and 100%)."""
        warm = WarmStartSPRT(
            theta_0=0.90, delta=0.30, alpha=0.05, beta=0.20,
            prior_successes=20, prior_trials=20,
        )

        savings = warm.expected_savings()

        assert isinstance(savings, float)
        assert 0.0 <= savings <= 100.0

    def test_expected_savings_zero_with_no_prior(self):
        """With no prior, savings should be 0% (no advantage over cold start)."""
        warm = WarmStartSPRT(
            theta_0=0.90, delta=0.30, alpha=0.05, beta=0.20,
            prior_successes=0, prior_trials=0,
        )

        savings = warm.expected_savings()

        assert savings == pytest.approx(0.0, abs=0.1)
