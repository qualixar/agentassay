# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Confidence interval computation for stochastic agent test verdicts.

Implements multiple confidence interval methods for binomial proportions,
used to determine whether an agent's observed pass rate meets a threshold
with statistical rigor.

Two API levels are provided:

1. **Structured API** (recommended) — returns :class:`ConfidenceInterval`
   Pydantic models via :func:`wilson_interval`, :func:`clopper_pearson_interval`,
   and :func:`normal_interval`.

2. **Legacy API** — :func:`binomial_confidence_interval` returns a plain
   ``(lower, upper)`` tuple.  Kept for backward compatibility.

Mathematical Foundation:
    For n independent Bernoulli trials with observed successes k,
    we compute a (1-alpha) confidence interval for the true proportion p.

    Wilson score interval (default):
        p_hat +/- z * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4 * n^2))
        ---------------------------------------------------------------
                            1 + z^2 / n

    This is preferred over the Wald (normal) interval because it has
    better coverage properties near 0 and 1, and never produces
    intervals outside [0, 1].

References:
    - Wilson, E.B. (1927). "Probable Inference, the Law of Succession,
      and Statistical Inference." JASA 22(158): 209-212.
    - Agresti, A. & Coull, B.A. (1998). "Approximate Is Better than 'Exact'
      for Interval Estimation of Binomial Proportions." The American
      Statistician 52(2): 119-126.
    - Clopper, C.J. & Pearson, E.S. (1934). "The Use of Confidence or
      Fiducial Limits Illustrated in the Case of the Binomial."
      Biometrika 26(4): 404-413.
"""

from __future__ import annotations

import math
from typing import Final

try:
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # Python 3.10 compat
        pass

from pydantic import BaseModel, model_validator
from scipy import stats as sp_stats

# ============================================================================
# Pydantic model for structured confidence interval results
# ============================================================================


class ConfidenceInterval(BaseModel, frozen=True):
    """Immutable result of a confidence interval computation.

    Attributes
    ----------
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.
    point_estimate : float
        The observed proportion (successes / n).
    confidence_level : float
        Confidence level used (e.g. 0.95 for 95 %).
    method : str
        Name of the method (``"wilson"``, ``"clopper_pearson"``, ``"normal"``).
    n : int
        Sample size (total number of trials).
    """

    lower: float
    upper: float
    point_estimate: float
    confidence_level: float
    method: str
    n: int

    @model_validator(mode="after")
    def _check_bounds(self) -> ConfidenceInterval:
        if self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) must be <= upper ({self.upper})")
        return self


# ============================================================================
# Input validation shared across all interval functions
# ============================================================================


def _validate_ci_inputs(successes: int, n: int, confidence: float) -> None:
    """Common input validation for the structured CI functions."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        raise ValueError(
            "Cannot compute a confidence interval with n=0 trials. Run at least 1 trial first."
        )
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, n], got successes={successes}, n={n}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


# ============================================================================
# Structured API — returns ConfidenceInterval Pydantic models
# ============================================================================


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson score confidence interval for a binomial proportion.

    This is the recommended default for agent testing.  The Wilson interval
    is centred on a *shrinkage* estimate (pulled toward 0.5) rather than
    the raw proportion, which gives it excellent coverage properties even
    when *n* is small or the true proportion is near 0 or 1.

    Formula (Wilson 1927)::

        centre = (p + z^2 / 2n) / (1 + z^2 / n)
        margin = z * sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n)
        CI = [centre - margin, centre + margin]

    Parameters
    ----------
    successes : int
        Number of passed trials.
    n : int
        Total number of trials.  Must be >= 1.
    confidence : float
        Confidence level in (0, 1).  Default is 0.95.

    Returns
    -------
    ConfidenceInterval
    """
    _validate_ci_inputs(successes, n, confidence)

    p_hat = successes / n
    z = sp_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z

    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denominator

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=p_hat,
        confidence_level=confidence,
        method="wilson",
        n=n,
    )


def clopper_pearson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> ConfidenceInterval:
    """Clopper-Pearson exact binomial confidence interval.

    This interval has *guaranteed* coverage — the true coverage probability
    is never below the nominal confidence level.  The price is that it is
    conservative (wider than necessary), especially for moderate *n*.

    The bounds are computed via the inverse regularised incomplete beta
    function (``scipy.stats.beta.ppf``), which is numerically robust for
    all valid inputs including the boundary cases (successes = 0 or n).

    Parameters
    ----------
    successes : int
        Number of passed trials.
    n : int
        Total number of trials.  Must be >= 1.
    confidence : float
        Confidence level in (0, 1).  Default is 0.95.

    Returns
    -------
    ConfidenceInterval
    """
    _validate_ci_inputs(successes, n, confidence)

    alpha = 1.0 - confidence
    p_hat = successes / n

    if successes == 0:
        lower = 0.0
    else:
        lower = float(sp_stats.beta.ppf(alpha / 2.0, successes, n - successes + 1))

    if successes == n:
        upper = 1.0
    else:
        upper = float(sp_stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=p_hat,
        confidence_level=confidence,
        method="clopper_pearson",
        n=n,
    )


def normal_interval(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Normal approximation (Wald) confidence interval.

    This is the textbook interval::

        p +/- z * sqrt(p * (1 - p) / n)

    It is adequate when *n* is large (>= 30) and *p* is not extreme.
    For agent testing this is often **not** the case — Wilson or
    Clopper-Pearson should be preferred.

    .. warning::

        The Wald interval can produce bounds outside [0, 1] or degenerate
        to a zero-width interval when p_hat is exactly 0 or 1.  Bounds
        are clamped to [0, 1] for safety.

    Parameters
    ----------
    successes : int
        Number of passed trials.
    n : int
        Total number of trials.  Must be >= 1.
    confidence : float
        Confidence level in (0, 1).  Default is 0.95.

    Returns
    -------
    ConfidenceInterval
    """
    _validate_ci_inputs(successes, n, confidence)

    p_hat = successes / n
    z = sp_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)

    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n)

    lower = max(0.0, p_hat - margin)
    upper = min(1.0, p_hat + margin)

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=p_hat,
        confidence_level=confidence,
        method="normal",
        n=n,
    )


# ============================================================================
# Legacy API — preserved for backward compatibility
# ============================================================================


class ConfidenceMethod(StrEnum):
    """Supported confidence interval methods for binomial proportions."""

    WILSON = "wilson"
    CLOPPER_PEARSON = "clopper_pearson"
    AGRESTI_COULL = "agresti_coull"
    WALD = "wald"


# Minimum sample sizes for each method to produce reliable intervals.
# Below these, intervals may have poor coverage properties.
_MIN_SAMPLE_SIZES: Final[dict[ConfidenceMethod, int]] = {
    ConfidenceMethod.WILSON: 5,  # type: ignore[dict-item]
    ConfidenceMethod.CLOPPER_PEARSON: 1,  # type: ignore[dict-item]
    ConfidenceMethod.AGRESTI_COULL: 10,  # type: ignore[dict-item]
    ConfidenceMethod.WALD: 30,  # type: ignore[dict-item]
}


def binomial_confidence_interval(
    n_successes: int,
    n_trials: int,
    confidence_level: float = 0.95,
    method: str | ConfidenceMethod = ConfidenceMethod.WILSON,
) -> tuple[float, float]:
    """Compute a confidence interval for a binomial proportion.

    .. note::
        This is the legacy API that returns a plain tuple.  Prefer the
        structured functions (:func:`wilson_interval`, etc.) that return
        :class:`ConfidenceInterval` models.

    Args:
        n_successes: Number of successes (passed trials).
        n_trials: Total number of trials.
        confidence_level: Confidence level in (0, 1). Default 0.95.
        method: CI method. One of 'wilson', 'clopper_pearson',
            'agresti_coull', 'wald'. Default 'wilson'.

    Returns:
        Tuple of (lower_bound, upper_bound), each in [0.0, 1.0].

    Raises:
        ValueError: If inputs are invalid (negative counts, successes > trials,
            confidence level out of range, unknown method).
    """
    # --- Input validation ---
    if n_trials < 0:
        raise ValueError(f"n_trials must be non-negative, got {n_trials}")
    if n_successes < 0:
        raise ValueError(f"n_successes must be non-negative, got {n_successes}")
    if n_successes > n_trials:
        raise ValueError(f"n_successes ({n_successes}) cannot exceed n_trials ({n_trials})")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    method = ConfidenceMethod(method)

    # --- Edge case: zero trials ---
    if n_trials == 0:
        return (0.0, 1.0)

    p_hat = n_successes / n_trials
    alpha = 1.0 - confidence_level
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    if method == ConfidenceMethod.WILSON:
        lower, upper = _wilson_interval_legacy(p_hat, n_trials, z)
    elif method == ConfidenceMethod.CLOPPER_PEARSON:
        lower, upper = _clopper_pearson_interval_legacy(n_successes, n_trials, alpha)
    elif method == ConfidenceMethod.AGRESTI_COULL:
        lower, upper = _agresti_coull_interval(n_successes, n_trials, z)
    elif method == ConfidenceMethod.WALD:
        lower, upper = _wald_interval_legacy(p_hat, n_trials, z)
    else:
        raise ValueError(f"Unknown confidence method: {method}")

    # Clamp to [0, 1] — some methods can produce out-of-range values
    return (max(0.0, lower), min(1.0, upper))


def _wilson_interval_legacy(p_hat: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval (1927) — legacy tuple API."""
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (center - half_width, center + half_width)


def _clopper_pearson_interval_legacy(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson exact interval (1934) — legacy tuple API."""
    if k == 0:
        lower = 0.0
    else:
        lower = sp_stats.beta.ppf(alpha / 2.0, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = sp_stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)

    return (float(lower), float(upper))


def _agresti_coull_interval(k: int, n: int, z: float) -> tuple[float, float]:
    """Agresti-Coull adjusted Wald interval (1998)."""
    n_tilde = n + z * z
    p_tilde = (k + z * z / 2.0) / n_tilde
    half_width = z * math.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
    return (p_tilde - half_width, p_tilde + half_width)


def _wald_interval_legacy(p_hat: float, n: int, z: float) -> tuple[float, float]:
    """Standard Wald (normal approximation) interval — legacy tuple API."""
    if n == 0:
        return (0.0, 1.0)
    se = math.sqrt(p_hat * (1.0 - p_hat) / n)
    return (p_hat - z * se, p_hat + z * se)


def minimum_sample_size(
    method: str | ConfidenceMethod = ConfidenceMethod.WILSON,
) -> int:
    """Return the minimum recommended sample size for reliable intervals.

    Args:
        method: The confidence interval method.

    Returns:
        Minimum number of trials recommended.
    """
    method = ConfidenceMethod(method)
    return _MIN_SAMPLE_SIZES.get(method, 5)
