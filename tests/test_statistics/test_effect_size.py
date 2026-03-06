"""Tests for effect size calculations.

Tests Cohen's h, Glass's delta, rank-biserial, and interpretation.

Target: ~10 tests.
"""

from __future__ import annotations

import math

import pytest

from agentassay.statistics.effect_size import (
    cohens_h,
    glass_delta,
    interpret_effect_size,
    rank_biserial,
)


class TestCohensH:
    """Tests for cohens_h."""

    def test_identical_proportions(self):
        assert cohens_h(0.8, 0.8) == 0.0

    def test_known_direction(self):
        h = cohens_h(0.9, 0.5)
        assert h > 0  # p1 > p2 => positive

    def test_symmetric(self):
        h1 = cohens_h(0.9, 0.5)
        h2 = cohens_h(0.5, 0.9)
        assert abs(h1 + h2) < 1e-10  # opposite signs

    def test_extreme_values(self):
        h = cohens_h(1.0, 0.0)
        assert h == pytest.approx(math.pi, abs=0.001)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="p1"):
            cohens_h(-0.1, 0.5)
        with pytest.raises(ValueError, match="p2"):
            cohens_h(0.5, 1.1)


class TestGlassDelta:
    """Tests for glass_delta."""

    def test_identical_means_with_variance(self):
        d = glass_delta([0.5, 0.6, 0.4], [0.5, 0.6, 0.4])
        assert d == pytest.approx(0.0)

    def test_negative_on_regression(self):
        d = glass_delta([0.9, 0.85, 0.88], [0.5, 0.55, 0.52])
        assert d < 0  # treatment < baseline => negative

    def test_too_few_baseline_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            glass_delta([0.5], [0.5])

    def test_zero_std_raises(self):
        with pytest.raises(ValueError, match="zero"):
            glass_delta([0.5, 0.5, 0.5], [0.3])


class TestRankBiserial:
    """Tests for rank_biserial."""

    def test_perfect_dominance(self):
        # U = n1*n2 when baseline fully dominates => r = 1 - 2*n1*n2/(n1*n2) = -1
        # But for mannwhitneyu with alternative='greater', U = n1*n2 means
        # all baseline > current, so r should be -1
        r = rank_biserial(0.0, 5, 5)
        assert r == pytest.approx(1.0)

    def test_no_dominance(self):
        # U = n1*n2/2 => r = 0
        r = rank_biserial(12.5, 5, 5)
        assert r == pytest.approx(0.0)

    def test_n1_zero_raises(self):
        with pytest.raises(ValueError, match="n1"):
            rank_biserial(5.0, 0, 5)

    def test_n2_zero_raises(self):
        with pytest.raises(ValueError, match="n2"):
            rank_biserial(5.0, 5, 0)


class TestInterpretEffectSize:
    """Tests for interpret_effect_size."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, "negligible"),
            (0.1, "negligible"),
            (0.25, "small"),
            (0.55, "medium"),
            (0.85, "large"),
            (1.5, "large"),
        ],
    )
    def test_cohens_h_interpretation(self, value, expected):
        assert interpret_effect_size(value, "cohens_h") == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, "negligible"),
            (0.3, "small"),
            (0.6, "medium"),
            (0.9, "large"),
        ],
    )
    def test_glass_delta_interpretation(self, value, expected):
        assert interpret_effect_size(value, "glass_delta") == expected

    def test_negative_value_uses_absolute(self):
        assert interpret_effect_size(-0.85, "cohens_h") == "large"

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            interpret_effect_size(0.5, "unknown_metric")
