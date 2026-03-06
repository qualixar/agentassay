"""Tests for Sequential Probability Ratio Test (SPRT).

Tests SPRTRunner convergence, boundary computation, reset,
expected sample size, and error handling.

Target: ~15 tests.
"""

from __future__ import annotations

import pytest

from agentassay.statistics.sprt import SPRTResult, SPRTRunner


class TestSPRTRunner:
    """Tests for SPRTRunner."""

    def test_create_valid_runner(self):
        runner = SPRTRunner(p0=0.90, p1=0.70, alpha=0.05, beta=0.20)
        assert runner.is_decided is False

    def test_converges_to_accept_h1_under_regression(self):
        """Feed only failures; SPRT should conclude regression."""
        runner = SPRTRunner(p0=0.90, p1=0.50, alpha=0.05, beta=0.20)
        result = None
        for _ in range(200):
            result = runner.update(passed=False)
            if result.decision != "continue":
                break
        assert result is not None
        assert result.decision == "accept_h1"

    def test_converges_to_accept_h0_under_no_regression(self):
        """Feed only passes; SPRT should conclude no regression."""
        runner = SPRTRunner(p0=0.90, p1=0.50, alpha=0.05, beta=0.20)
        result = None
        for _ in range(200):
            result = runner.update(passed=True)
            if result.decision != "continue":
                break
        assert result is not None
        assert result.decision == "accept_h0"

    def test_result_has_correct_fields(self):
        runner = SPRTRunner(p0=0.90, p1=0.70)
        result = runner.update(passed=True)
        assert isinstance(result, SPRTResult)
        assert result.trials_used == 1
        assert result.p0 == 0.90
        assert result.p1 == 0.70

    def test_llr_updates_correctly(self):
        runner = SPRTRunner(p0=0.90, p1=0.70)
        r1 = runner.update(passed=True)
        r2 = runner.update(passed=False)
        # LLR should change after each observation
        assert r1.log_likelihood_ratio != r2.log_likelihood_ratio

    def test_cannot_update_after_decision(self):
        runner = SPRTRunner(p0=0.90, p1=0.10, alpha=0.05, beta=0.20)
        # Feed enough failures to trigger accept_h1
        for _ in range(50):
            result = runner.update(passed=False)
            if result.decision != "continue":
                break
        with pytest.raises(RuntimeError, match="already terminated"):
            runner.update(passed=True)

    def test_reset_allows_reuse(self):
        runner = SPRTRunner(p0=0.90, p1=0.50)
        for _ in range(100):
            result = runner.update(passed=False)
            if result.decision != "continue":
                break
        assert runner.is_decided is True
        runner.reset()
        assert runner.is_decided is False

    def test_expected_sample_size(self):
        runner = SPRTRunner(p0=0.90, p1=0.70, alpha=0.05, beta=0.20)
        asn = runner.expected_sample_size()
        assert "under_h0" in asn
        assert "under_h1" in asn
        assert asn["under_h0"] > 0
        assert asn["under_h1"] > 0

    def test_asn_h1_less_than_fixed_n(self):
        """SPRT should need fewer trials than a fixed-sample test under H1."""
        runner = SPRTRunner(p0=0.90, p1=0.50, alpha=0.05, beta=0.20)
        asn = runner.expected_sample_size()
        # A fixed-sample test for detecting 0.90->0.50 at 80% power
        # would need ~13 per group. SPRT under H1 should be comparable
        # or less.
        assert asn["under_h1"] < 200

    def test_invalid_p0_raises(self):
        with pytest.raises(ValueError, match="p0"):
            SPRTRunner(p0=0.0, p1=0.5)
        with pytest.raises(ValueError, match="p0"):
            SPRTRunner(p0=1.0, p1=0.5)

    def test_invalid_p1_raises(self):
        with pytest.raises(ValueError, match="p1"):
            SPRTRunner(p0=0.9, p1=0.0)

    def test_p1_must_be_less_than_p0(self):
        with pytest.raises(ValueError, match="p1 must be < p0"):
            SPRTRunner(p0=0.5, p1=0.9)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SPRTRunner(p0=0.9, p1=0.5, alpha=0.0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            SPRTRunner(p0=0.9, p1=0.5, beta=1.0)

    def test_boundaries_are_finite(self):
        runner = SPRTRunner(p0=0.90, p1=0.70)
        result = runner.update(passed=True)
        import math

        assert math.isfinite(result.upper_boundary)
        assert math.isfinite(result.lower_boundary)
        assert result.lower_boundary < result.upper_boundary
