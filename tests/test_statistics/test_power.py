"""Tests for power analysis module.

Tests required_sample_size and achieved_power.

Target: ~10 tests.
"""

from __future__ import annotations

import pytest

from agentassay.statistics.power import achieved_power, required_sample_size


class TestRequiredSampleSize:
    """Tests for required_sample_size."""

    def test_basic_computation(self):
        n = required_sample_size(p0=0.90, p1=0.75)
        assert isinstance(n, int)
        assert n > 0

    def test_returns_integer(self):
        n = required_sample_size(p0=0.90, p1=0.80)
        assert isinstance(n, int)

    def test_smaller_effect_needs_more_samples(self):
        n_large = required_sample_size(p0=0.90, p1=0.50)
        n_small = required_sample_size(p0=0.90, p1=0.85)
        assert n_small > n_large

    def test_higher_power_needs_more_samples(self):
        n_80 = required_sample_size(p0=0.90, p1=0.75, power=0.80)
        n_95 = required_sample_size(p0=0.90, p1=0.75, power=0.95)
        assert n_95 > n_80

    def test_stricter_alpha_needs_more_samples(self):
        n_05 = required_sample_size(p0=0.90, p1=0.75, alpha=0.05)
        n_01 = required_sample_size(p0=0.90, p1=0.75, alpha=0.01)
        assert n_01 > n_05

    def test_p1_must_be_less_than_p0(self):
        with pytest.raises(ValueError, match="p1 must be < p0"):
            required_sample_size(p0=0.50, p1=0.90)

    def test_invalid_p0_raises(self):
        with pytest.raises(ValueError, match="p0"):
            required_sample_size(p0=0.0, p1=0.5)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            required_sample_size(p0=0.9, p1=0.7, alpha=1.0)


class TestAchievedPower:
    """Tests for achieved_power."""

    def test_power_increases_with_n(self):
        p_50 = achieved_power(p0=0.90, p1=0.70, n=50)
        p_200 = achieved_power(p0=0.90, p1=0.70, n=200)
        assert p_200 > p_50

    def test_power_in_valid_range(self):
        p = achieved_power(p0=0.90, p1=0.75, n=100)
        assert 0.0 <= p <= 1.0

    def test_round_trip_with_required_n(self):
        """n from required_sample_size should yield >= target power."""
        target_power = 0.80
        n = required_sample_size(p0=0.90, p1=0.75, power=target_power)
        actual = achieved_power(p0=0.90, p1=0.75, n=n)
        # Allow small numerical tolerance
        assert actual >= target_power - 0.05

    def test_p1_must_be_less_than_p0(self):
        with pytest.raises(ValueError, match="p1 must be < p0"):
            achieved_power(p0=0.50, p1=0.90, n=50)

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            achieved_power(p0=0.90, p1=0.70, n=0)

    def test_large_n_high_power(self):
        p = achieved_power(p0=0.90, p1=0.50, n=1000)
        assert p > 0.99
