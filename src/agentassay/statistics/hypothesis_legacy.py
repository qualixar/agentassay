# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Legacy hypothesis testing API — preserved for backward compatibility.

Prefer the structured API in hypothesis.py (fisher_exact_regression,
chi2_regression, etc.) which returns Pydantic models.

This module contains the original dataclass-based API: test_binary_regression
and test_score_regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from scipy import stats as sp_stats

from agentassay.statistics.effect_size import cohens_h as _cohens_h_standalone

class RegressionTest(StrEnum):
    """Supported hypothesis tests for binary regression detection."""

    FISHER = "fisher"
    CHI2 = "chi2"
    BARNARD = "barnard"


class ScoreTest(StrEnum):
    """Supported hypothesis tests for continuous score comparison."""

    MANN_WHITNEY = "mann_whitney"
    KS = "ks"
    WELCH_T = "welch_t"


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    """Result of a hypothesis test for regression detection (legacy API).

    Attributes:
        test_name: Name of the statistical test used.
        statistic: The test statistic value.
        p_value: p-value for the test (one-sided where applicable).
        effect_size: Standardized effect size (Cohen's h or rank-biserial r).
        effect_size_name: Name of the effect size metric.
        effect_size_interpretation: Qualitative interpretation (small/medium/large).
        power: Estimated statistical power (1 - beta), if computable.
        regression_detected: Whether p < alpha AND direction indicates regression.
        alpha: Significance level used.
    """

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    effect_size_interpretation: str
    power: float | None
    regression_detected: bool
    alpha: float


# --- Effect size interpretation thresholds ---

_COHENS_H_THRESHOLDS: Final[list[tuple[float, str]]] = [
    (0.8, "large"),
    (0.5, "medium"),
    (0.2, "small"),
    (0.0, "negligible"),
]

_RANK_BISERIAL_THRESHOLDS: Final[list[tuple[float, str]]] = [
    (0.5, "large"),
    (0.3, "medium"),
    (0.1, "small"),
    (0.0, "negligible"),
]


def _interpret_effect_size(
    value: float, thresholds: list[tuple[float, str]]
) -> str:
    """Map an absolute effect size to a qualitative interpretation."""
    abs_val = abs(value)
    for threshold, label in thresholds:
        if abs_val >= threshold:
            return label
    return "negligible"


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Delegates to :func:`agentassay.statistics.effect_size.cohens_h`
    which is the canonical implementation. This wrapper is kept for
    backward compatibility with code importing from this module.

    Args:
        p1: First proportion (e.g., baseline pass rate).
        p2: Second proportion (e.g., current pass rate).

    Returns:
        Cohen's h. Positive means p1 > p2 (regression if p1=baseline).
    """
    return _cohens_h_standalone(p1, p2)


def rank_biserial_r(u_statistic: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U.

    r = 1 - 2U / (n1 * n2)

    Args:
        u_statistic: Mann-Whitney U statistic.
        n1: Size of first sample.
        n2: Size of second sample.

    Returns:
        Rank-biserial r in [-1, 1].
    """
    if n1 * n2 == 0:
        return 0.0
    return 1.0 - (2.0 * u_statistic) / (n1 * n2)


def _estimate_power_proportions(
    p1: float, p2: float, n1: int, n2: int, alpha: float
) -> float | None:
    """Estimate power for a two-proportion z-test (approximate).

    Uses the normal approximation for power calculation.
    Returns None if the computation is not feasible.
    """
    try:
        h = cohens_h(p1, p2)
        if abs(h) < 1e-10:
            return None
        # Harmonic mean of sample sizes for unequal groups
        n_harmonic = 2.0 * n1 * n2 / (n1 + n2) if (n1 + n2) > 0 else 0.0
        if n_harmonic == 0:
            return None
        z_alpha = sp_stats.norm.ppf(1.0 - alpha)
        noncentrality = abs(h) * math.sqrt(n_harmonic)
        power = 1.0 - sp_stats.norm.cdf(z_alpha - noncentrality)
        return float(max(0.0, min(1.0, power)))
    except (ValueError, ZeroDivisionError):
        return None


def test_binary_regression(
    baseline_successes: int,
    baseline_trials: int,
    current_successes: int,
    current_trials: int,
    alpha: float = 0.05,
    method: str | RegressionTest = RegressionTest.FISHER,
) -> HypothesisResult:
    """Test whether binary pass rates have regressed (legacy API).

    .. note::
        Prefer :func:`fisher_exact_regression` or :func:`chi2_regression`
        which return richer :class:`RegressionTestResult` models.

    Performs a one-sided test: H1 is that the current rate is LOWER
    than the baseline rate (i.e., regression occurred).

    Args:
        baseline_successes: Passes in baseline.
        baseline_trials: Total trials in baseline.
        current_successes: Passes in current version.
        current_trials: Total trials in current version.
        alpha: Significance level (Type I error rate). Default 0.05.
        method: Test method. Default 'fisher'.

    Returns:
        HypothesisResult with test outcome.

    Raises:
        ValueError: If inputs are invalid.
    """
    # --- Validation ---
    if baseline_trials <= 0 or current_trials <= 0:
        raise ValueError("Trial counts must be positive")
    if baseline_successes < 0 or current_successes < 0:
        raise ValueError("Success counts must be non-negative")
    if baseline_successes > baseline_trials:
        raise ValueError("Baseline successes cannot exceed trials")
    if current_successes > current_trials:
        raise ValueError("Current successes cannot exceed trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    method = RegressionTest(method)

    p_baseline = baseline_successes / baseline_trials
    p_current = current_successes / current_trials

    # Build 2x2 contingency table:
    #               | Success | Failure |
    # Baseline      |   a     |   b     |
    # Current       |   c     |   d     |
    a = baseline_successes
    b = baseline_trials - baseline_successes
    c = current_successes
    d = current_trials - current_successes
    table = np.array([[a, b], [c, d]])

    if method == RegressionTest.FISHER:
        # Fisher's exact test, one-sided (alternative: current < baseline)
        _, p_value = sp_stats.fisher_exact(table, alternative="greater")
        test_name = "Fisher's exact test (one-sided)"
        statistic = float(p_baseline - p_current)

    elif method == RegressionTest.CHI2:
        # Chi-squared test (two-sided, then halve for one-sided)
        if min(a, b, c, d) < 5:
            # Fall back to Fisher for small expected counts
            _, p_value = sp_stats.fisher_exact(table, alternative="greater")
            test_name = "Fisher's exact test (one-sided, fallback from chi2)"
            statistic = float(p_baseline - p_current)
        else:
            chi2_stat, p_two_sided, _, _ = sp_stats.chi2_contingency(
                table, correction=True
            )
            # One-sided: halve and check direction
            if p_current < p_baseline:
                p_value = p_two_sided / 2.0
            else:
                p_value = 1.0 - p_two_sided / 2.0
            test_name = "Chi-squared test (one-sided)"
            statistic = float(chi2_stat)

    elif method == RegressionTest.BARNARD:
        # Barnard's exact test — more powerful than Fisher for 2x2
        result = sp_stats.barnard_exact(table, alternative="greater")
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
        test_name = "Barnard's exact test (one-sided)"
    else:
        raise ValueError(f"Unknown regression test: {method}")

    # Effect size: Cohen's h
    h = cohens_h(p_baseline, p_current)
    interpretation = _interpret_effect_size(h, _COHENS_H_THRESHOLDS)

    # Power estimate
    power = _estimate_power_proportions(
        p_baseline, p_current,
        baseline_trials, current_trials,
        alpha,
    )

    # Regression detected if statistically significant AND direction is negative
    regression_detected = (p_value < alpha) and (p_current < p_baseline)

    return HypothesisResult(
        test_name=test_name,
        statistic=statistic,
        p_value=float(p_value),
        effect_size=h,
        effect_size_name="Cohen's h",
        effect_size_interpretation=interpretation,
        power=power,
        regression_detected=regression_detected,
        alpha=alpha,
    )


def test_score_regression(
    baseline_scores: list[float] | np.ndarray,
    current_scores: list[float] | np.ndarray,
    alpha: float = 0.05,
    method: str | ScoreTest = ScoreTest.MANN_WHITNEY,
) -> HypothesisResult:
    """Test whether continuous score distributions have regressed (legacy API).

    .. note::
        Prefer :func:`ks_regression` or :func:`mann_whitney_regression`
        which return richer :class:`RegressionTestResult` models.

    Performs a two-sided test for distributional difference, then
    checks direction to determine if regression occurred.

    Args:
        baseline_scores: Score values from baseline version.
        current_scores: Score values from current version.
        alpha: Significance level. Default 0.05.
        method: Test method. Default 'mann_whitney'.

    Returns:
        HypothesisResult with test outcome.

    Raises:
        ValueError: If inputs are invalid.
    """
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    current = np.asarray(current_scores, dtype=np.float64)

    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("Score arrays must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    method = ScoreTest(method)
    n1 = len(baseline)
    n2 = len(current)

    if method == ScoreTest.MANN_WHITNEY:
        u_stat, p_value = sp_stats.mannwhitneyu(
            baseline, current, alternative="greater"
        )
        test_name = "Mann-Whitney U test (one-sided)"
        statistic = float(u_stat)
        effect = rank_biserial_r(u_stat, n1, n2)
        effect_name = "rank-biserial r"
        effect_interp = _interpret_effect_size(
            effect, _RANK_BISERIAL_THRESHOLDS
        )

    elif method == ScoreTest.KS:
        ks_stat, p_two_sided = sp_stats.ks_2samp(baseline, current)
        if float(np.median(current)) < float(np.median(baseline)):
            p_value = p_two_sided / 2.0
        else:
            p_value = 1.0 - p_two_sided / 2.0
        test_name = "Kolmogorov-Smirnov test (one-sided)"
        statistic = float(ks_stat)
        effect = float(ks_stat)
        effect_name = "KS statistic"
        effect_interp = _interpret_effect_size(
            ks_stat, _RANK_BISERIAL_THRESHOLDS
        )

    elif method == ScoreTest.WELCH_T:
        t_stat, p_value = sp_stats.ttest_ind(
            baseline, current, equal_var=False, alternative="greater"
        )
        test_name = "Welch's t-test (one-sided)"
        statistic = float(t_stat)
        pooled_std = math.sqrt(
            (float(np.var(baseline, ddof=1)) + float(np.var(current, ddof=1))) / 2.0
        )
        if pooled_std > 0:
            effect = (float(np.mean(baseline)) - float(np.mean(current))) / pooled_std
        else:
            effect = 0.0
        effect_name = "Cohen's d"
        effect_interp = _interpret_effect_size(effect, _COHENS_H_THRESHOLDS)
    else:
        raise ValueError(f"Unknown score test: {method}")

    current_worse = float(np.mean(current)) < float(np.mean(baseline))
    regression_detected = (p_value < alpha) and current_worse

    return HypothesisResult(
        test_name=test_name,
        statistic=statistic,
        p_value=float(p_value),
        effect_size=effect,
        effect_size_name=effect_name,
        effect_size_interpretation=effect_interp,
        power=None,
        regression_detected=regression_detected,
        alpha=alpha,
    )
