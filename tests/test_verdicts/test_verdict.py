"""Tests for VerdictFunction and StochasticVerdict.

Tests evaluate_single (threshold test), evaluate_regression (cross-version),
evaluate_scores (continuous), VerdictStatus enum, StochasticVerdict properties.

Target: ~25 tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentassay.verdicts.verdict import (
    StochasticVerdict,
    VerdictFunction,
    VerdictStatus,
)


# ===================================================================
# VerdictStatus
# ===================================================================


class TestVerdictStatus:
    """Tests for the VerdictStatus enum."""

    def test_pass_value(self):
        assert VerdictStatus.PASS == "pass"

    def test_fail_value(self):
        assert VerdictStatus.FAIL == "fail"

    def test_inconclusive_value(self):
        assert VerdictStatus.INCONCLUSIVE == "inconclusive"

    def test_is_str_enum(self):
        assert isinstance(VerdictStatus.PASS, str)


# ===================================================================
# StochasticVerdict
# ===================================================================


class TestStochasticVerdict:
    """Tests for StochasticVerdict model."""

    def _make_verdict(self, **kwargs):
        defaults = {
            "status": VerdictStatus.PASS,
            "confidence": 0.95,
            "pass_rate": 0.9,
            "pass_rate_ci": (0.80, 0.96),
            "num_trials": 100,
            "num_passed": 90,
        }
        defaults.update(kwargs)
        return StochasticVerdict(**defaults)

    def test_is_definitive_pass(self):
        v = self._make_verdict(status=VerdictStatus.PASS)
        assert v.is_definitive is True

    def test_is_definitive_fail(self):
        v = self._make_verdict(status=VerdictStatus.FAIL)
        assert v.is_definitive is True

    def test_not_definitive_inconclusive(self):
        v = self._make_verdict(status=VerdictStatus.INCONCLUSIVE)
        assert v.is_definitive is False

    def test_margin_of_error(self):
        v = self._make_verdict(pass_rate_ci=(0.80, 0.96))
        expected_moe = (0.96 - 0.80) / 2.0
        assert v.margin_of_error == pytest.approx(expected_moe)

    def test_frozen_immutable(self):
        v = self._make_verdict()
        with pytest.raises(ValidationError):
            v.status = VerdictStatus.FAIL

    def test_num_passed_cannot_exceed_num_trials(self):
        with pytest.raises(ValidationError, match="cannot exceed"):
            self._make_verdict(num_trials=10, num_passed=15)

    def test_ci_bounds_validated(self):
        with pytest.raises(ValidationError, match="lower <= upper"):
            self._make_verdict(pass_rate_ci=(0.9, 0.5))


# ===================================================================
# VerdictFunction
# ===================================================================


class TestVerdictFunction:
    """Tests for VerdictFunction."""

    def test_create_with_defaults(self):
        vf = VerdictFunction()
        assert vf.alpha == 0.05
        assert vf.beta == 0.20
        assert vf.min_trials == 30
        assert vf.confidence_level == 0.95

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            VerdictFunction(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            VerdictFunction(alpha=1.0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            VerdictFunction(beta=0.0)

    def test_invalid_min_trials_raises(self):
        with pytest.raises(ValueError, match="min_trials"):
            VerdictFunction(min_trials=0)


# ===================================================================
# evaluate_single
# ===================================================================


class TestEvaluateSingle:
    """Tests for VerdictFunction.evaluate_single."""

    def test_pass_when_ci_above_threshold(self):
        vf = VerdictFunction(min_trials=1)
        results = [True] * 100
        verdict = vf.evaluate_single(results, threshold=0.50)
        assert verdict.status == VerdictStatus.PASS

    def test_fail_when_ci_below_threshold(self):
        vf = VerdictFunction(min_trials=1)
        results = [False] * 100
        verdict = vf.evaluate_single(results, threshold=0.50)
        assert verdict.status == VerdictStatus.FAIL

    def test_inconclusive_when_below_min_trials(self):
        vf = VerdictFunction(min_trials=50)
        results = [True] * 10
        verdict = vf.evaluate_single(results, threshold=0.80)
        assert verdict.status == VerdictStatus.INCONCLUSIVE

    def test_empty_results_inconclusive(self):
        vf = VerdictFunction()
        verdict = vf.evaluate_single([], threshold=0.80)
        assert verdict.status == VerdictStatus.INCONCLUSIVE
        assert verdict.num_trials == 0

    def test_threshold_out_of_range_raises(self):
        vf = VerdictFunction()
        with pytest.raises(ValueError, match="threshold"):
            vf.evaluate_single([True], threshold=1.5)

    def test_pass_rate_computed_correctly(self):
        vf = VerdictFunction(min_trials=1)
        results = [True] * 75 + [False] * 25
        verdict = vf.evaluate_single(results, threshold=0.50)
        assert verdict.pass_rate == pytest.approx(0.75)
        assert verdict.num_passed == 75
        assert verdict.num_trials == 100


# ===================================================================
# evaluate_regression
# ===================================================================


class TestEvaluateRegression:
    """Tests for VerdictFunction.evaluate_regression."""

    def test_detects_regression(self):
        vf = VerdictFunction(min_trials=1)
        baseline = [True] * 95 + [False] * 5
        current = [True] * 50 + [False] * 50
        verdict = vf.evaluate_regression(baseline, current)
        assert verdict.status == VerdictStatus.FAIL
        assert verdict.regression_detected is True

    def test_no_regression_identical(self):
        vf = VerdictFunction(min_trials=1)
        results = [True] * 90 + [False] * 10
        verdict = vf.evaluate_regression(results, results)
        # Identical rates => no regression
        assert verdict.regression_detected is False

    def test_empty_baseline_inconclusive(self):
        vf = VerdictFunction()
        verdict = vf.evaluate_regression([], [True] * 50)
        assert verdict.status == VerdictStatus.INCONCLUSIVE

    def test_empty_current_inconclusive(self):
        vf = VerdictFunction()
        verdict = vf.evaluate_regression([True] * 50, [])
        assert verdict.status == VerdictStatus.INCONCLUSIVE

    def test_regression_details_contain_baseline_info(self):
        vf = VerdictFunction(min_trials=1)
        baseline = [True] * 95 + [False] * 5
        current = [True] * 50 + [False] * 50
        verdict = vf.evaluate_regression(baseline, current)
        assert "baseline_pass_rate" in verdict.details
        assert "baseline_trials" in verdict.details


# ===================================================================
# evaluate_scores
# ===================================================================


class TestEvaluateScores:
    """Tests for VerdictFunction.evaluate_scores."""

    def test_detects_score_regression(self):
        vf = VerdictFunction(min_trials=1)
        baseline = [0.9, 0.85, 0.88, 0.92, 0.87] * 10
        current = [0.3, 0.35, 0.28, 0.32, 0.31] * 10
        verdict = vf.evaluate_scores(baseline, current)
        assert verdict.status == VerdictStatus.FAIL
        assert verdict.regression_detected is True

    def test_no_regression_identical_scores(self):
        vf = VerdictFunction(min_trials=1)
        scores = [0.8, 0.85, 0.9, 0.82, 0.88] * 10
        verdict = vf.evaluate_scores(scores, scores)
        assert verdict.regression_detected is False

    def test_empty_scores_inconclusive(self):
        vf = VerdictFunction()
        verdict = vf.evaluate_scores([], [0.5])
        assert verdict.status == VerdictStatus.INCONCLUSIVE

    def test_details_contain_score_stats(self):
        vf = VerdictFunction(min_trials=1)
        baseline = [0.9, 0.85, 0.88] * 10
        current = [0.3, 0.35, 0.28] * 10
        verdict = vf.evaluate_scores(baseline, current)
        assert "baseline_mean" in verdict.details
        assert "current_mean" in verdict.details
