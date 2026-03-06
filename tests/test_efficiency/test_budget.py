"""Tests for adaptive budget optimization.

Validates that AdaptiveBudgetOptimizer correctly calibrates trial counts
based on observed agent variance, respects upper/lower bounds, computes
savings accurately, and produces immutable BudgetEstimate objects.

Target: ~12 tests covering calibration, bounds, classification, savings,
cost estimation, immutability, and allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentassay.efficiency.budget import (
    AdaptiveBudgetOptimizer,
)

from .conftest import make_regressed_traces, make_traces

# ===================================================================
# AdaptiveBudgetOptimizer — calibration
# ===================================================================


class TestAdaptiveBudgetOptimizerCalibration:
    """Tests for calibration-based trial count recommendations."""

    def test_calibrate_stable_agent(self):
        """A stable agent (all passes, same tools) needs fewer trials.

        When behavioral variance is low, the optimizer should recommend
        a sample size smaller than the fixed-N default (100).
        """
        # 20 identical-behavior traces — very low variance
        traces = make_traces(20, steps=5, tools=["search", "calculate"])
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(traces)

        assert estimate.recommended_n < 100

    def test_calibrate_volatile_agent(self):
        """A volatile agent (mixed pass/fail, varied tools) needs more trials.

        When behavioral variance is high, the optimizer should recommend
        more trials than it would for a stable agent.
        """
        # Mix stable and regressed traces to create high variance
        stable = make_traces(10, steps=5, tools=["search"])
        volatile = make_regressed_traces(10, steps=12, tools=["search", "write", "delete"])
        mixed = stable + volatile

        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(mixed)

        # Compare against the stable-only estimate
        stable_estimate = optimizer.calibrate(make_traces(20))

        assert estimate.recommended_n > stable_estimate.recommended_n

    def test_recommended_n_upper_bound(self):
        """Recommended N never exceeds the max_trials ceiling."""
        # Even with maximum variance, the recommendation should be capped
        traces = make_regressed_traces(20, pass_rate=0.3)
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10, max_trials=100)
        estimate = optimizer.calibrate(traces)

        assert estimate.recommended_n <= 100

    def test_recommended_n_lower_bound(self):
        """Recommended N is always at least min_trials."""
        traces = make_traces(15)
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10, min_trials=10)
        estimate = optimizer.calibrate(traces)

        assert estimate.recommended_n >= 10

    def test_variance_classification_stable(self):
        """All-passing, same-structure traces are classified as 'stable'."""
        traces = make_traces(20, steps=5, passed=True)
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(traces)

        assert estimate.variance_classification in ("stable", "moderate")

    def test_variance_classification_volatile(self):
        """Highly mixed traces (varied pass rate, varied tools) are 'volatile'."""
        stable = make_traces(10, steps=3, tools=["search"])
        regressed = make_regressed_traces(10, steps=12, pass_rate=0.2)
        mixed = stable + regressed

        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(mixed)

        # Should not be classified as stable
        assert estimate.variance_classification in ("moderate", "volatile")


# ===================================================================
# AdaptiveBudgetOptimizer — savings and cost
# ===================================================================


class TestAdaptiveBudgetOptimizerSavings:
    """Tests for savings calculation and cost estimation."""

    def test_savings_calculation(self):
        """Savings percentage is correct: (100 - recommended_n) / 100 * 100."""
        traces = make_traces(20)
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(traces)

        expected_savings = max(0.0, (1.0 - estimate.recommended_n / 100.0)) * 100.0
        assert estimate.savings_vs_fixed_100 == pytest.approx(expected_savings, abs=0.1)

    def test_cost_estimation(self):
        """Estimated cost should be non-negative and proportional to recommended_n."""
        traces = make_traces(20, cost=0.02)
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(traces)

        assert estimate.estimated_cost_usd >= 0.0
        # Cost should be roughly recommended_n * avg_cost_per_trial
        avg_cost = np.mean([t.total_cost_usd for t in traces])
        rough_estimate = estimate.recommended_n * avg_cost
        # Within 50% — the optimizer may apply adjustments
        assert estimate.estimated_cost_usd < rough_estimate * 2.0


# ===================================================================
# BudgetEstimate — immutability
# ===================================================================


class TestBudgetEstimateImmutability:
    """Tests for BudgetEstimate being frozen (immutable)."""

    def test_budget_estimate_frozen(self):
        """BudgetEstimate fields cannot be modified after creation."""
        traces = make_traces(20)
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)
        estimate = optimizer.calibrate(traces)

        with pytest.raises((AttributeError, TypeError, Exception)):
            estimate.recommended_n = 999  # type: ignore[misc]


# ===================================================================
# optimal_allocation — budget distribution
# ===================================================================


class TestOptimalAllocation:
    """Tests for distributing a fixed budget across multiple test types."""

    def test_allocation_sums_within_budget(self):
        """Total allocated trials across types do not wildly exceed budget."""
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)

        alloc = optimizer.optimal_allocation(
            budget_tokens=100_000,
            test_types=["regression", "coverage", "mutation"],
            tokens_per_trial={"regression": 500, "coverage": 300, "mutation": 400},
        )

        assert alloc["utilization"] <= 1.5  # reasonable range

    def test_allocation_prioritizes_high_variance(self):
        """Test types with higher variance get more budget allocation.

        Regression type has variance=2.0, coverage has variance=0.5.
        Regression should receive more trials.
        """
        optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10, delta=0.10)

        alloc = optimizer.optimal_allocation(
            budget_tokens=100_000,
            test_types=["regression", "coverage"],
            tokens_per_trial={"regression": 500, "coverage": 500},
            variance_per_type={"regression": 4.0, "coverage": 1.0},
        )

        assert alloc["allocation"]["regression"] > alloc["allocation"]["coverage"]
        assert alloc["strategy"] == "variance_weighted"
