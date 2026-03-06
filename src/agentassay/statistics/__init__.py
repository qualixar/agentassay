"""AgentAssay statistics — formal statistical methods for agent regression testing.

This package provides the mathematical foundation for deciding whether an
agent has regressed.  Traditional software testing is binary (pass/fail);
agent testing is *stochastic* — the same agent on the same input can produce
different results.  The tools here handle that non-determinism rigorously.

Submodules
----------
hypothesis
    Regression hypothesis tests (Fisher, chi-squared, KS, Mann-Whitney).
confidence
    Confidence interval computation (Wilson, Clopper-Pearson, normal).
sprt
    Sequential Probability Ratio Test for adaptive stopping.
effect_size
    Effect size calculations (Cohen's h, Glass's delta, rank-biserial).
power
    Power analysis for sample size planning.

Quick Start
-----------
>>> from agentassay.statistics import wilson_interval, fisher_exact_regression
>>> ci = wilson_interval(successes=85, n=100)
>>> print(f"Pass rate: {ci.point_estimate:.0%} [{ci.lower:.2%}, {ci.upper:.2%}]")
Pass rate: 85% [76.86%, 90.87%]

>>> result = fisher_exact_regression(
...     baseline_passes=90, baseline_n=100,
...     current_passes=78, current_n=100,
... )
>>> print(result.interpretation)
"""

# ============================================================================
# Structured API — Pydantic models (recommended)
# ============================================================================

# Confidence intervals
from agentassay.statistics.confidence import (
    ConfidenceInterval,
    clopper_pearson_interval,
    normal_interval,
    wilson_interval,
)

# Effect sizes
from agentassay.statistics.effect_size import (
    cohens_h,
    glass_delta,
    interpret_effect_size,
    rank_biserial,
)

# Hypothesis tests
from agentassay.statistics.hypothesis import (
    RegressionTestResult,
    chi2_regression,
    fisher_exact_regression,
    ks_regression,
    mann_whitney_regression,
)

# Power analysis
from agentassay.statistics.power import (
    achieved_power,
    required_sample_size,
)

# SPRT
from agentassay.statistics.sprt import (
    SPRTResult,
    SPRTRunner,
)

# ============================================================================
# Legacy API — preserved for backward compatibility
# ============================================================================

from agentassay.statistics.confidence import (
    ConfidenceMethod,
    binomial_confidence_interval,
    minimum_sample_size,
)
from agentassay.statistics.hypothesis_legacy import (
    HypothesisResult,
    RegressionTest,
    ScoreTest,
    cohens_h as cohens_h,  # noqa: duplicate re-export is intentional
    rank_biserial_r,
    test_binary_regression,
    test_score_regression,
)

__all__ = [
    # --- Structured API (Pydantic) ---
    # Confidence
    "ConfidenceInterval",
    "wilson_interval",
    "clopper_pearson_interval",
    "normal_interval",
    # Effect size
    "cohens_h",
    "glass_delta",
    "rank_biserial",
    "interpret_effect_size",
    # Hypothesis tests
    "RegressionTestResult",
    "fisher_exact_regression",
    "chi2_regression",
    "ks_regression",
    "mann_whitney_regression",
    # Power
    "required_sample_size",
    "achieved_power",
    # SPRT
    "SPRTResult",
    "SPRTRunner",
    # --- Legacy API ---
    "ConfidenceMethod",
    "binomial_confidence_interval",
    "minimum_sample_size",
    "HypothesisResult",
    "RegressionTest",
    "ScoreTest",
    "rank_biserial_r",
    "test_binary_regression",
    "test_score_regression",
]
