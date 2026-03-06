"""Hypothesis testing for agent regression detection.

Two API levels are provided:

1. **Structured API** (recommended) — returns :class:`RegressionTestResult`
   Pydantic models via :func:`fisher_exact_regression`,
   :func:`chi2_regression`, :func:`ks_regression`, and
   :func:`mann_whitney_regression`.

2. **Legacy API** — :func:`test_binary_regression` and
   :func:`test_score_regression` return :class:`HypothesisResult`
   dataclass instances.  Kept for backward compatibility.

Implements statistical tests to determine whether an agent has regressed
between a baseline version and a current version. Supports both binary
(pass/fail) and continuous (score) comparisons.

Mathematical Foundation:
    Regression testing is formulated as a hypothesis test:

    Binary case (Fisher exact / Chi-squared):
        H0: p_current >= p_baseline  (no regression)
        H1: p_current < p_baseline   (regression occurred)

    Continuous case (Mann-Whitney U / Kolmogorov-Smirnov):
        H0: F_current = F_baseline   (distributions are the same)
        H1: F_current != F_baseline  (distributions differ)

    The verdict maps test results to the (alpha, beta, n)-triple:
        - alpha: Type I error rate (false positive — claiming regression when none)
        - beta:  Type II error rate (false negative — missing real regression)
        - n:     Sample size

Effect Sizes:
    Cohen's h (binary): h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        Small: |h| ~ 0.2, Medium: |h| ~ 0.5, Large: |h| ~ 0.8

    Rank-biserial r (continuous): r = 1 - 2U / (n1 * n2)
        Small: |r| ~ 0.1, Medium: |r| ~ 0.3, Large: |r| ~ 0.5

References:
    - Fisher, R.A. (1922). "On the Interpretation of Chi-Squared from
      Contingency Tables." JRSS 85(1): 87-94.
    - Mann, H.B. & Whitney, D.R. (1947). "On a Test of Whether One of Two
      Random Variables Is Stochastically Larger than the Other."
      Annals of Mathematical Statistics 18(1): 50-60.
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral
      Sciences. 2nd ed. Lawrence Erlbaum.
    - Wald, A. (1947). Sequential Analysis. John Wiley & Sons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Sequence

import numpy as np
from pydantic import BaseModel
from scipy import stats as sp_stats

from agentassay.statistics.effect_size import (
    cohens_h as _cohens_h_standalone,
    glass_delta as _glass_delta_standalone,
    interpret_effect_size as _interpret_standalone,
    rank_biserial as _rank_biserial_standalone,
)


# ============================================================================
# Pydantic model for the structured (new) API
# ============================================================================

class RegressionTestResult(BaseModel, frozen=True):
    """Immutable result of a regression hypothesis test.

    Attributes
    ----------
    test_name : str
        Human-readable test identifier (e.g. ``"Fisher's exact test"``).
    statistic : float
        The raw test statistic (odds ratio, chi2, KS D, or U).
    p_value : float
        One-sided p-value in the regression direction.
    effect_size : float
        Magnitude of the difference.
    effect_size_name : str
        Name of the effect-size metric (e.g. ``"Cohen's h"``).
    baseline_rate : float
        Baseline pass rate (or mean score for continuous tests).
    current_rate : float
        Current pass rate (or mean score for continuous tests).
    baseline_n : int
        Number of baseline trials/observations.
    current_n : int
        Number of current trials/observations.
    significant : bool
        ``True`` if ``p_value < alpha``.
    alpha : float
        Significance level used.
    interpretation : str
        Plain-English sentence describing the result.
    """

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    baseline_rate: float
    current_rate: float
    baseline_n: int
    current_n: int
    significant: bool
    alpha: float
    interpretation: str


# ============================================================================
# Validation helpers for the structured API
# ============================================================================

def _validate_counts(passes: int, n: int, label: str) -> None:
    if n < 0:
        raise ValueError(f"{label}_n must be >= 0, got {n}")
    if n == 0:
        raise ValueError(
            f"{label}_n is 0 — cannot perform a hypothesis test with zero trials."
        )
    if not (0 <= passes <= n):
        raise ValueError(
            f"{label}_passes must be in [0, {label}_n], "
            f"got {label}_passes={passes}, {label}_n={n}"
        )


def _validate_scores(scores: Sequence[float], label: str) -> None:
    if len(scores) < 1:
        raise ValueError(f"{label} must have at least 1 observation, got 0")


def _build_interpretation(
    test_name: str,
    significant: bool,
    p_value: float,
    effect_size: float,
    effect_size_name: str,
    baseline_rate: float,
    current_rate: float,
) -> str:
    """Build a plain-English interpretation string."""
    es_key = effect_size_name.lower().replace("'", "").replace(" ", "_")
    magnitude = _interpret_standalone(effect_size, es_key)

    if significant:
        return (
            f"{test_name} detected a statistically significant regression "
            f"(p={p_value:.4f}). The current pass rate ({current_rate:.3f}) is "
            f"lower than the baseline ({baseline_rate:.3f}) with a {magnitude} "
            f"effect size ({effect_size_name}={effect_size:+.3f})."
        )
    return (
        f"{test_name} found no statistically significant regression "
        f"(p={p_value:.4f}). The current pass rate ({current_rate:.3f}) vs "
        f"baseline ({baseline_rate:.3f}) shows a {magnitude} effect size "
        f"({effect_size_name}={effect_size:+.3f})."
    )


# ============================================================================
# Structured API — Pydantic-based functions
# ============================================================================

def fisher_exact_regression(
    baseline_passes: int,
    baseline_n: int,
    current_passes: int,
    current_n: int,
    alpha: float = 0.05,
) -> RegressionTestResult:
    """Fisher's exact test for regression in pass rates.

    Constructs a 2x2 contingency table and applies Fisher's exact test with
    the one-sided alternative that the current version has a *lower* pass
    rate (regression direction).

    Best suited for small sample sizes (n < 20) where asymptotic chi-squared
    approximations break down.

    Parameters
    ----------
    baseline_passes : int
        Number of passed trials in the baseline run.
    baseline_n : int
        Total baseline trials.
    current_passes : int
        Number of passed trials in the current run.
    current_n : int
        Total current trials.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    RegressionTestResult
    """
    _validate_counts(baseline_passes, baseline_n, "baseline")
    _validate_counts(current_passes, current_n, "current")

    baseline_fails = baseline_n - baseline_passes
    current_fails = current_n - current_passes

    table = [[baseline_passes, baseline_fails], [current_passes, current_fails]]

    # alternative='greater' tests whether the odds ratio > 1,
    # i.e. baseline has higher odds of passing => current regressed.
    odds_ratio, p_value = sp_stats.fisher_exact(table, alternative="greater")

    baseline_rate = baseline_passes / baseline_n
    current_rate = current_passes / current_n
    h = _cohens_h_standalone(baseline_rate, current_rate)
    significant = p_value < alpha

    return RegressionTestResult(
        test_name="Fisher's exact test",
        statistic=float(odds_ratio),
        p_value=float(p_value),
        effect_size=h,
        effect_size_name="Cohen's h",
        baseline_rate=baseline_rate,
        current_rate=current_rate,
        baseline_n=baseline_n,
        current_n=current_n,
        significant=significant,
        alpha=alpha,
        interpretation=_build_interpretation(
            "Fisher's exact test",
            significant,
            float(p_value),
            h,
            "Cohen's h",
            baseline_rate,
            current_rate,
        ),
    )


def chi2_regression(
    baseline_passes: int,
    baseline_n: int,
    current_passes: int,
    current_n: int,
    alpha: float = 0.05,
) -> RegressionTestResult:
    """Chi-squared test for regression in pass rates.

    Uses ``scipy.stats.chi2_contingency`` on the same 2x2 table as
    :func:`fisher_exact_regression`.  The two-sided chi-squared p-value is
    converted to a one-sided p-value in the regression direction.

    Prefer this over Fisher's exact test when both groups have n >= 20
    and all expected cell counts are >= 5.

    Parameters
    ----------
    baseline_passes : int
        Number of passed trials in the baseline run.
    baseline_n : int
        Total baseline trials.
    current_passes : int
        Number of passed trials in the current run.
    current_n : int
        Total current trials.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    RegressionTestResult
    """
    _validate_counts(baseline_passes, baseline_n, "baseline")
    _validate_counts(current_passes, current_n, "current")

    baseline_fails = baseline_n - baseline_passes
    current_fails = current_n - current_passes

    table = [[baseline_passes, baseline_fails], [current_passes, current_fails]]

    # Small-cell fallback: when any cell count is < 5, the chi-squared
    # approximation is unreliable.  Fall back to Fisher's exact test,
    # matching the legacy API behaviour in hypothesis_legacy.py.
    a, b, c, d = baseline_passes, baseline_fails, current_passes, current_fails
    if min(a, b, c, d) < 5:
        return fisher_exact_regression(
            baseline_passes=baseline_passes,
            baseline_n=baseline_n,
            current_passes=current_passes,
            current_n=current_n,
            alpha=alpha,
        )

    chi2, p_two_sided, dof, expected = sp_stats.chi2_contingency(table, correction=True)

    baseline_rate = baseline_passes / baseline_n
    current_rate = current_passes / current_n

    # Convert two-sided to one-sided in the regression direction.
    if current_rate < baseline_rate:
        p_value = p_two_sided / 2.0
    elif current_rate > baseline_rate:
        p_value = 1.0 - p_two_sided / 2.0
    else:
        p_value = 0.5

    h = _cohens_h_standalone(baseline_rate, current_rate)
    significant = p_value < alpha

    return RegressionTestResult(
        test_name="Chi-squared test",
        statistic=float(chi2),
        p_value=float(p_value),
        effect_size=h,
        effect_size_name="Cohen's h",
        baseline_rate=baseline_rate,
        current_rate=current_rate,
        baseline_n=baseline_n,
        current_n=current_n,
        significant=significant,
        alpha=alpha,
        interpretation=_build_interpretation(
            "Chi-squared test",
            significant,
            float(p_value),
            h,
            "Cohen's h",
            baseline_rate,
            current_rate,
        ),
    )


def ks_regression(
    baseline_scores: Sequence[float],
    current_scores: Sequence[float],
    alpha: float = 0.05,
) -> RegressionTestResult:
    """Two-sample Kolmogorov-Smirnov test for score distribution shift.

    Tests whether the current score distribution has shifted *leftward*
    (lower scores = regression) compared to the baseline.  Uses the
    one-sided alternative ``"less"`` in ``scipy.stats.ks_2samp``, which
    tests whether F_current(x) > F_baseline(x).

    Effect size is Glass's delta (baseline-normalised mean difference).

    Parameters
    ----------
    baseline_scores : Sequence[float]
        Score distribution from the baseline version.  Must have len >= 2.
    current_scores : Sequence[float]
        Score distribution from the current version.  Must have len >= 1.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    RegressionTestResult
    """
    _validate_scores(baseline_scores, "baseline_scores")
    _validate_scores(current_scores, "current_scores")

    if len(baseline_scores) < 2:
        raise ValueError(
            "baseline_scores must have at least 2 observations for "
            "KS test and Glass's delta computation."
        )

    ks_stat, p_value = sp_stats.ks_2samp(
        baseline_scores, current_scores, alternative="less"
    )

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    current_mean = sum(current_scores) / len(current_scores)

    try:
        delta = _glass_delta_standalone(list(baseline_scores), list(current_scores))
    except ValueError:
        delta = 0.0

    significant = p_value < alpha

    return RegressionTestResult(
        test_name="Kolmogorov-Smirnov test",
        statistic=float(ks_stat),
        p_value=float(p_value),
        effect_size=delta,
        effect_size_name="Glass's delta",
        baseline_rate=baseline_mean,
        current_rate=current_mean,
        baseline_n=len(baseline_scores),
        current_n=len(current_scores),
        significant=significant,
        alpha=alpha,
        interpretation=_build_interpretation(
            "Kolmogorov-Smirnov test",
            significant,
            float(p_value),
            delta,
            "Glass's delta",
            baseline_mean,
            current_mean,
        ),
    )


def mann_whitney_regression(
    baseline_scores: Sequence[float],
    current_scores: Sequence[float],
    alpha: float = 0.05,
) -> RegressionTestResult:
    """Mann-Whitney U test for stochastic regression.

    Tests whether current scores are stochastically *smaller* than baseline
    scores.  Effect size is the rank-biserial correlation.

    Parameters
    ----------
    baseline_scores : Sequence[float]
        Score distribution from the baseline version.  Must have len >= 1.
    current_scores : Sequence[float]
        Score distribution from the current version.  Must have len >= 1.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    RegressionTestResult
    """
    _validate_scores(baseline_scores, "baseline_scores")
    _validate_scores(current_scores, "current_scores")

    u_stat, p_value = sp_stats.mannwhitneyu(
        baseline_scores, current_scores, alternative="greater"
    )

    n1 = len(baseline_scores)
    n2 = len(current_scores)
    r = _rank_biserial_standalone(float(u_stat), n1, n2)

    baseline_mean = sum(baseline_scores) / n1
    current_mean = sum(current_scores) / n2
    significant = p_value < alpha

    return RegressionTestResult(
        test_name="Mann-Whitney U test",
        statistic=float(u_stat),
        p_value=float(p_value),
        effect_size=r,
        effect_size_name="rank-biserial",
        baseline_rate=baseline_mean,
        current_rate=current_mean,
        baseline_n=n1,
        current_n=n2,
        significant=significant,
        alpha=alpha,
        interpretation=_build_interpretation(
            "Mann-Whitney U test",
            significant,
            float(p_value),
            r,
            "rank_biserial",
            baseline_mean,
            current_mean,
        ),
    )

