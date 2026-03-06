"""Tests for hypothesis testing module.

Tests Fisher exact, Chi-squared, KS, and Mann-Whitney regression tests
from both the structured (Pydantic) and legacy APIs.

Target: ~25 tests.
"""

from __future__ import annotations

import pytest

from agentassay.statistics.hypothesis import (
    RegressionTestResult,
    chi2_regression,
    fisher_exact_regression,
    ks_regression,
    mann_whitney_regression,
)
from agentassay.statistics.hypothesis_legacy import (
    HypothesisResult,
    test_binary_regression as binary_regression_fn,
    test_score_regression as score_regression_fn,
)


# ===================================================================
# Fisher exact (structured API)
# ===================================================================


class TestFisherExactRegression:
    """Tests for fisher_exact_regression."""

    def test_no_regression_equal_rates(self):
        result = fisher_exact_regression(90, 100, 90, 100)
        assert isinstance(result, RegressionTestResult)
        assert result.significant is False
        assert result.p_value >= 0.05

    def test_detects_clear_regression(self):
        result = fisher_exact_regression(95, 100, 60, 100)
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.current_rate < result.baseline_rate

    def test_effect_size_positive_on_regression(self):
        result = fisher_exact_regression(90, 100, 70, 100)
        assert result.effect_size > 0  # positive h means baseline > current

    def test_perfect_pass_rates(self):
        result = fisher_exact_regression(100, 100, 100, 100)
        assert result.significant is False
        assert result.effect_size == 0.0

    def test_all_fail(self):
        result = fisher_exact_regression(0, 100, 0, 100)
        assert result.significant is False

    def test_interpretation_contains_test_name(self):
        result = fisher_exact_regression(90, 100, 70, 100)
        assert "Fisher" in result.interpretation

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="0"):
            fisher_exact_regression(0, 0, 5, 10)

    def test_passes_exceed_n_raises(self):
        with pytest.raises(ValueError):
            fisher_exact_regression(15, 10, 5, 10)

    def test_small_sample_valid(self):
        # Fisher is designed for small samples
        result = fisher_exact_regression(3, 5, 1, 5)
        assert isinstance(result, RegressionTestResult)


# ===================================================================
# Chi-squared (structured API)
# ===================================================================


class TestChi2Regression:
    """Tests for chi2_regression."""

    def test_no_regression(self):
        result = chi2_regression(90, 100, 88, 100)
        assert result.significant is False

    def test_detects_clear_regression(self):
        result = chi2_regression(95, 100, 60, 100)
        assert result.significant is True

    def test_equal_rates_pvalue_is_half(self):
        result = chi2_regression(50, 100, 50, 100)
        assert abs(result.p_value - 0.5) < 0.01

    def test_current_higher_than_baseline(self):
        result = chi2_regression(50, 100, 80, 100)
        assert result.significant is False  # no regression


# ===================================================================
# KS regression (structured API)
# ===================================================================


class TestKSRegression:
    """Tests for ks_regression."""

    def test_identical_distributions_no_regression(self):
        scores = [0.8, 0.85, 0.9, 0.95, 0.88]
        result = ks_regression(scores, scores)
        assert result.significant is False

    def test_shifted_distribution_detects_regression(self):
        baseline = [0.9, 0.95, 0.88, 0.92, 0.91] * 10
        current = [0.4, 0.5, 0.45, 0.48, 0.42] * 10
        result = ks_regression(baseline, current)
        assert result.significant is True

    def test_single_baseline_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ks_regression([0.5], [0.5, 0.6])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            ks_regression([], [0.5])

    def test_effect_size_glass_delta(self):
        baseline = [0.9, 0.85, 0.88, 0.92, 0.87]
        current = [0.5, 0.55, 0.52, 0.48, 0.53]
        result = ks_regression(baseline, current)
        assert result.effect_size_name == "Glass's delta"


# ===================================================================
# Mann-Whitney (structured API)
# ===================================================================


class TestMannWhitneyRegression:
    """Tests for mann_whitney_regression."""

    def test_identical_no_regression(self):
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        result = mann_whitney_regression(scores, scores)
        assert result.significant is False

    def test_stochastic_dominance_detects_regression(self):
        baseline = [0.9, 0.95, 0.88, 0.92, 0.91] * 10
        current = [0.3, 0.35, 0.28, 0.32, 0.31] * 10
        result = mann_whitney_regression(baseline, current)
        assert result.significant is True
        # Effect size magnitude should be large regardless of sign convention
        assert abs(result.effect_size) > 0.5

    def test_effect_size_rank_biserial(self):
        baseline = [0.9, 0.85, 0.88]
        current = [0.5, 0.55, 0.52]
        result = mann_whitney_regression(baseline, current)
        assert result.effect_size_name == "rank-biserial"

    def test_single_observation_valid(self):
        result = mann_whitney_regression([0.9], [0.5])
        assert isinstance(result, RegressionTestResult)


# ===================================================================
# Legacy API: test_binary_regression
# ===================================================================


class TestLegacyBinaryRegression:
    """Tests for the legacy test_binary_regression API."""

    def test_returns_hypothesis_result(self):
        result = binary_regression_fn(90, 100, 80, 100)
        assert isinstance(result, HypothesisResult)

    def test_regression_detected_flag(self):
        result = binary_regression_fn(95, 100, 50, 100)
        assert result.regression_detected is True

    def test_no_regression_flag(self):
        result = binary_regression_fn(90, 100, 90, 100)
        assert bool(result.regression_detected) is False

    def test_invalid_trials_raises(self):
        with pytest.raises(ValueError, match="positive"):
            binary_regression_fn(5, 10, 5, 0)

    def test_successes_exceed_trials_raises(self):
        with pytest.raises(ValueError):
            binary_regression_fn(15, 10, 5, 10)

    def test_chi2_method(self):
        result = binary_regression_fn(90, 100, 70, 100, method="chi2")
        assert "chi" in result.test_name.lower() or "fisher" in result.test_name.lower()


# ===================================================================
# Legacy API: test_score_regression
# ===================================================================


class TestLegacyScoreRegression:
    """Tests for the legacy test_score_regression API."""

    def test_returns_hypothesis_result(self):
        result = score_regression_fn([0.9] * 20, [0.5] * 20)
        assert isinstance(result, HypothesisResult)

    def test_regression_detected(self):
        result = score_regression_fn([0.9] * 30, [0.3] * 30)
        assert result.regression_detected is True

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            score_regression_fn([], [0.5])
