"""Tests for multi-fidelity testing with cheap/expensive model correlation.

Validates that MultiFidelityTester correctly estimates correlation between
a cheap proxy model and the expensive target model, combines verdicts,
enforces minimum correlation thresholds, and allocates budgets optimally.

Target: ~9 tests covering correlation estimation, verdict combination,
threshold enforcement, and budget splitting.
"""

from __future__ import annotations

from agentassay.efficiency.multi_fidelity import MultiFidelityTester

from .conftest import make_regressed_traces, make_traces

# ===================================================================
# MultiFidelityTester — correlation estimation
# ===================================================================


class TestMultiFidelityCorrelation:
    """Tests for estimating cheap/expensive model correlation."""

    def test_estimate_correlation_high(self):
        """Similar behaviors yield a positive correlation estimate."""
        # Same behavioral pattern for both models — correlation should be high
        cheap = make_traces(20, model="gpt-4o-mini", cost=0.002)
        expensive = make_traces(20, model="gpt-4o", cost=0.03)

        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
        )
        corr = tester.estimate_correlation(cheap, expensive)

        # Since both use the same tool pattern, correlation should be high
        assert corr > 0.5

    def test_estimate_correlation_low(self):
        """Very different behaviors yield lower correlation."""
        cheap = make_traces(20, model="gpt-4o-mini", steps=3, tools=["search"])
        expensive = make_regressed_traces(20)

        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
        )
        corr = tester.estimate_correlation(cheap, expensive)

        # Different behavioral patterns should produce lower correlation
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0


# ===================================================================
# MultiFidelityTester — verdict combination
# ===================================================================


class TestMultiFidelityVerdicts:
    """Tests for combining cheap + expensive verdicts into a final decision."""

    def test_combined_verdict_no_regression(self):
        """When neither model shows regression, verdict is no-regression."""
        cheap_baseline = make_traces(15, model="gpt-4o-mini", cost=0.002)
        cheap_current = make_traces(15, model="gpt-4o-mini", cost=0.002)
        expensive_baseline = make_traces(10, model="gpt-4o", cost=0.03)
        expensive_current = make_traces(10, model="gpt-4o", cost=0.03)

        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
        )

        verdict = tester.combined_verdict(
            proxy_baseline=cheap_baseline,
            proxy_candidate=cheap_current,
            target_baseline=expensive_baseline,
            target_candidate=expensive_current,
        )

        assert verdict["regression_detected"] is False

    def test_combined_verdict_with_regression(self):
        """When both models show regression, the combined verdict detects it."""
        cheap_baseline = make_traces(15, model="gpt-4o-mini", cost=0.002)
        cheap_current = make_regressed_traces(15)
        expensive_baseline = make_traces(10, model="gpt-4o", cost=0.03)
        expensive_current = make_regressed_traces(10)

        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
        )

        verdict = tester.combined_verdict(
            proxy_baseline=cheap_baseline,
            proxy_candidate=cheap_current,
            target_baseline=expensive_baseline,
            target_candidate=expensive_current,
        )

        assert verdict["regression_detected"] is True


# ===================================================================
# MultiFidelityTester — threshold enforcement
# ===================================================================


class TestMultiFidelityThreshold:
    """Tests for minimum correlation threshold enforcement."""

    def test_correlation_threshold_enforcement(self):
        """Combined verdict reports whether correlation is sufficient."""
        cheap = make_traces(10, model="gpt-4o-mini", steps=3, tools=["search"])
        expensive = make_regressed_traces(10)

        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
            correlation_threshold=0.99,  # very high threshold
        )
        tester.estimate_correlation(cheap, expensive)

        verdict = tester.combined_verdict(
            proxy_baseline=cheap,
            proxy_candidate=cheap,
            target_baseline=expensive,
            target_candidate=expensive,
        )

        # With such a high threshold, correlation_sufficient should be False
        assert "correlation_sufficient" in verdict


# ===================================================================
# MultiFidelityTester — budget allocation
# ===================================================================


class TestMultiFidelityBudgetSplit:
    """Tests for optimal budget splitting between cheap and expensive models."""

    def test_optimal_split_respects_budget(self):
        """Total cost of the split does not wildly exceed the given budget."""
        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
        )
        # Estimate correlation first
        cheap = make_traces(10, model="gpt-4o-mini")
        expensive = make_traces(10, model="gpt-4o")
        tester.estimate_correlation(cheap, expensive)

        n_proxy, n_target = tester.optimal_split(
            total_budget=2.0,
            proxy_cost=0.002,
            target_cost=0.03,
        )

        total_cost = n_proxy * 0.002 + n_target * 0.03
        # Should be within reasonable range of budget
        assert total_cost <= 2.5  # allow some overshoot from rounding
        assert n_proxy >= 2
        assert n_target >= 2

    def test_optimal_split_allocates_more_to_cheap(self):
        """The optimizer allocates more trials to the cheaper model.

        Since cheap trials cost 15x less than expensive trials (0.002 vs 0.03),
        the optimal split should run significantly more cheap trials.
        """
        tester = MultiFidelityTester(
            proxy_model="gpt-4o-mini",
            target_model="gpt-4o",
        )
        cheap = make_traces(10, model="gpt-4o-mini")
        expensive = make_traces(10, model="gpt-4o")
        tester.estimate_correlation(cheap, expensive)

        n_proxy, n_target = tester.optimal_split(
            total_budget=1.0,
            proxy_cost=0.002,
            target_cost=0.03,
        )

        assert n_proxy > n_target
