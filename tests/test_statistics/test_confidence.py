"""Tests for confidence interval module.

Tests Wilson, Clopper-Pearson, and Normal interval methods from both
the structured (Pydantic) and legacy APIs.

Target: ~15 tests.
"""

from __future__ import annotations

import pytest

from agentassay.statistics.confidence import (
    ConfidenceInterval,
    ConfidenceMethod,
    binomial_confidence_interval,
    clopper_pearson_interval,
    minimum_sample_size,
    normal_interval,
    wilson_interval,
)


# ===================================================================
# Wilson interval (structured API)
# ===================================================================


class TestWilsonInterval:
    """Tests for wilson_interval."""

    def test_basic_interval(self):
        ci = wilson_interval(27, 30, confidence=0.95)
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == "wilson"
        assert 0.0 <= ci.lower <= ci.point_estimate <= ci.upper <= 1.0

    def test_all_pass(self):
        ci = wilson_interval(100, 100)
        assert ci.point_estimate == 1.0
        assert ci.upper == 1.0
        # Wilson shrinks toward 0.5, so lower < 1.0
        assert ci.lower < 1.0

    def test_all_fail(self):
        ci = wilson_interval(0, 100)
        assert ci.point_estimate == 0.0
        # Wilson shrinks toward 0.5, so lower is very close to 0 but may
        # have tiny floating point residue
        assert ci.lower == pytest.approx(0.0, abs=1e-10)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="n=0"):
            wilson_interval(0, 0)

    def test_successes_exceed_n_raises(self):
        with pytest.raises(ValueError):
            wilson_interval(15, 10)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 10, confidence=0.0)
        with pytest.raises(ValueError):
            wilson_interval(5, 10, confidence=1.0)

    def test_wider_at_higher_confidence(self):
        ci90 = wilson_interval(50, 100, confidence=0.90)
        ci99 = wilson_interval(50, 100, confidence=0.99)
        width90 = ci90.upper - ci90.lower
        width99 = ci99.upper - ci99.lower
        assert width99 > width90


# ===================================================================
# Clopper-Pearson interval (structured API)
# ===================================================================


class TestClopperPearsonInterval:
    """Tests for clopper_pearson_interval."""

    def test_exact_coverage(self):
        ci = clopper_pearson_interval(27, 30)
        assert ci.method == "clopper_pearson"
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_all_pass_upper_bound(self):
        ci = clopper_pearson_interval(100, 100)
        assert ci.upper == 1.0

    def test_all_fail_lower_bound(self):
        ci = clopper_pearson_interval(0, 100)
        assert ci.lower == 0.0

    def test_conservative_vs_wilson(self):
        """Clopper-Pearson should generally be wider than Wilson."""
        cp = clopper_pearson_interval(50, 100)
        w = wilson_interval(50, 100)
        cp_width = cp.upper - cp.lower
        w_width = w.upper - w.lower
        assert cp_width >= w_width - 0.01  # allow tiny numerical difference


# ===================================================================
# Normal (Wald) interval (structured API)
# ===================================================================


class TestNormalInterval:
    """Tests for normal_interval."""

    def test_basic_interval(self):
        ci = normal_interval(50, 100)
        assert ci.method == "normal"
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_all_pass_degenerate(self):
        ci = normal_interval(100, 100)
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_all_fail_degenerate(self):
        ci = normal_interval(0, 100)
        assert ci.lower == 0.0
        assert ci.upper == 0.0


# ===================================================================
# CI ordering invariant
# ===================================================================


class TestCIOrdering:
    """Test that lower <= point_estimate <= upper for all methods."""

    @pytest.mark.parametrize("successes,n", [
        (0, 10), (1, 10), (5, 10), (9, 10), (10, 10),
        (0, 100), (50, 100), (100, 100), (1, 1),
    ])
    @pytest.mark.parametrize("method_fn", [wilson_interval, clopper_pearson_interval, normal_interval])
    def test_ordering(self, successes, n, method_fn):
        ci = method_fn(successes, n)
        assert ci.lower <= ci.point_estimate + 1e-10
        assert ci.point_estimate <= ci.upper + 1e-10


# ===================================================================
# Legacy API
# ===================================================================


class TestLegacyBinomialCI:
    """Tests for the legacy binomial_confidence_interval API."""

    def test_returns_tuple(self):
        lo, hi = binomial_confidence_interval(50, 100)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert 0.0 <= lo <= hi <= 1.0

    def test_zero_trials_returns_full_range(self):
        lo, hi = binomial_confidence_interval(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    @pytest.mark.parametrize("method", ["wilson", "clopper_pearson", "agresti_coull", "wald"])
    def test_all_methods(self, method):
        lo, hi = binomial_confidence_interval(50, 100, method=method)
        assert 0.0 <= lo <= hi <= 1.0


class TestMinimumSampleSize:
    """Tests for minimum_sample_size."""

    def test_wilson_min(self):
        assert minimum_sample_size("wilson") == 5

    def test_clopper_pearson_min(self):
        assert minimum_sample_size("clopper_pearson") == 1

    def test_wald_min(self):
        assert minimum_sample_size("wald") == 30
