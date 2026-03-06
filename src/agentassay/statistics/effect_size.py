# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Effect size calculations for agent regression testing.

Effect sizes quantify the *magnitude* of a regression, not just its statistical
significance. A p-value tells you whether a difference is real; an effect size
tells you whether it matters. In agent testing this distinction is critical:
with enough trials almost any prompt tweak yields a significant p-value, but
only a meaningful effect size warrants blocking a deployment.

Three metrics are provided, each paired with a different hypothesis test:

    * Cohen's h   -- compares two proportions (pass rates)
    * Glass's delta -- compares two score distributions (baseline-normalized)
    * Rank-biserial r -- non-parametric effect size from Mann-Whitney U

Interpretation follows Cohen's (1988) conventions extended to the agent domain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Cohen's h — effect size for two proportions
# ---------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h for the difference between two proportions.

    Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    A positive h means p1 > p2.  In regression testing the convention is
    p1 = baseline rate, p2 = current rate, so a positive h indicates that
    the current version regressed (lower pass rate).

    Parameters
    ----------
    p1 : float
        First proportion (typically the baseline pass rate). Must be in [0, 1].
    p2 : float
        Second proportion (typically the current pass rate). Must be in [0, 1].

    Returns
    -------
    float
        Cohen's h.  Range is [-pi, pi] though practical values rarely exceed
        |h| > 1.5.

    Raises
    ------
    ValueError
        If either proportion is outside [0, 1].
    """
    if not (0.0 <= p1 <= 1.0):
        raise ValueError(f"p1 must be in [0, 1], got {p1}")
    if not (0.0 <= p2 <= 1.0):
        raise ValueError(f"p2 must be in [0, 1], got {p2}")

    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


# ---------------------------------------------------------------------------
# Glass's delta — baseline-normalised mean difference
# ---------------------------------------------------------------------------


def glass_delta(baseline: Sequence[float], treatment: Sequence[float]) -> float:
    """Compute Glass's delta between a treatment and a baseline distribution.

    Glass's delta = (mean(treatment) - mean(baseline)) / std(baseline)

    Unlike Cohen's d, Glass's delta uses only the baseline standard deviation
    as the denominator.  This is appropriate for regression testing because the
    baseline distribution is the reference and the current (treatment) distribution
    may have different variance if it has regressed.

    Parameters
    ----------
    baseline : Sequence[float]
        Score distribution from the baseline version.  Must have len >= 2.
    treatment : Sequence[float]
        Score distribution from the current version.  Must have len >= 1.

    Returns
    -------
    float
        Glass's delta.  Negative means the treatment scores are lower (regression).

    Raises
    ------
    ValueError
        If baseline has fewer than 2 observations (cannot compute std) or
        if baseline standard deviation is zero.
    """
    if len(baseline) < 2:
        raise ValueError(f"baseline must have at least 2 observations, got {len(baseline)}")
    if len(treatment) < 1:
        raise ValueError(f"treatment must have at least 1 observation, got {len(treatment)}")

    baseline_mean = sum(baseline) / len(baseline)
    treatment_mean = sum(treatment) / len(treatment)

    # Population-style std (ddof=0) matches Glass's original definition;
    # some implementations use ddof=1.  We use ddof=1 (Bessel's correction)
    # which is more conservative for small samples — appropriate for agent
    # testing where trial counts are typically 30-200.
    variance = sum((x - baseline_mean) ** 2 for x in baseline) / (len(baseline) - 1)
    baseline_std = math.sqrt(variance)

    if baseline_std == 0.0:
        raise ValueError(
            "baseline standard deviation is zero — all baseline observations are "
            "identical. Glass's delta is undefined when there is no variance in "
            "the reference distribution."
        )

    return (treatment_mean - baseline_mean) / baseline_std


# ---------------------------------------------------------------------------
# Rank-biserial correlation — effect size for Mann-Whitney U
# ---------------------------------------------------------------------------


def rank_biserial(u_statistic: float, n1: int, n2: int) -> float:
    """Compute the rank-biserial correlation from a Mann-Whitney U statistic.

    r = 1 - 2U / (n1 * n2)

    This transforms the U statistic into a [-1, 1] effect size where:
    - r =  1.0 means every baseline score exceeds every current score
    - r =  0.0 means no stochastic dominance (distributions overlap completely)
    - r = -1.0 means every current score exceeds every baseline score

    In the regression testing context a *positive* r indicates regression
    (baseline dominates current).

    Parameters
    ----------
    u_statistic : float
        The U statistic from ``scipy.stats.mannwhitneyu``.
    n1 : int
        Number of observations in the baseline group.
    n2 : int
        Number of observations in the current group.

    Returns
    -------
    float
        Rank-biserial correlation r in [-1, 1].

    Raises
    ------
    ValueError
        If n1 or n2 is less than 1.
    """
    if n1 < 1:
        raise ValueError(f"n1 must be >= 1, got {n1}")
    if n2 < 1:
        raise ValueError(f"n2 must be >= 1, got {n2}")

    product = n1 * n2
    if product == 0:
        raise ValueError("n1 * n2 must be > 0")

    return 1.0 - (2.0 * u_statistic) / product


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

# Cohen's (1988) conventions.  For rank-biserial we use the same thresholds
# because r maps onto a comparable scale.  Glass's delta and Cohen's h use
# the classic small/medium/large cut-offs.
_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "cohens_h": (0.20, 0.50, 0.80),
    "glass_delta": (0.20, 0.50, 0.80),
    "rank_biserial": (0.20, 0.50, 0.80),
}


def interpret_effect_size(value: float, metric: str) -> str:
    """Interpret an effect size value using Cohen's (1988) conventions.

    Parameters
    ----------
    value : float
        The absolute effect size value.
    metric : str
        One of ``"cohens_h"``, ``"glass_delta"``, or ``"rank_biserial"``.

    Returns
    -------
    str
        One of ``"negligible"``, ``"small"``, ``"medium"``, or ``"large"``.

    Raises
    ------
    ValueError
        If *metric* is not a recognised effect size name.
    """
    key = metric.lower().replace("'", "").replace(" ", "_")
    # Accept both snake_case and display names
    _ALIAS: dict[str, str] = {  # noqa: N806 - constant-like mapping
        "cohens_h": "cohens_h",
        "cohen_h": "cohens_h",
        "glasss_delta": "glass_delta",
        "glass_delta": "glass_delta",
        "rank_biserial": "rank_biserial",
        "rank-biserial": "rank_biserial",
    }

    canonical = _ALIAS.get(key)
    if canonical is None:
        raise ValueError(f"Unknown effect size metric '{metric}'. Supported: {list(_ALIAS.keys())}")

    small, medium, large = _THRESHOLDS[canonical]
    abs_value = abs(value)

    if abs_value < small:
        return "negligible"
    if abs_value < medium:
        return "small"
    if abs_value < large:
        return "medium"
    return "large"
