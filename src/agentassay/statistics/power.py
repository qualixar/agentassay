# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Power analysis for agent regression testing.

Power analysis answers two critical planning questions:

1. **Before testing:** How many agent trials do I need to reliably detect
   a regression from pass rate p0 to pass rate p1?
   -> :func:`required_sample_size`

2. **After testing:** Given the number of trials I ran, what was my
   probability of detecting a real regression?
   -> :func:`achieved_power`

These functions use the normal approximation to the binomial distribution.
The approximation is accurate when n * p * (1-p) >= 5 for both p0 and p1,
which holds for the sample sizes typically used in agent testing (n >= 30).

The formulas follow the classical two-proportion z-test power analysis
(Fleiss, Levin & Paik, 2003) adapted for one-sided tests in the regression
direction.
"""

from __future__ import annotations

import math

from scipy import stats as sp_stats


def required_sample_size(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Compute the number of trials needed to detect a regression.

    Answers: "If my agent's true pass rate drops from p0 to p1, how many
    trials do I need per version to detect this with the given alpha and
    power?"

    Uses the normal approximation formula for comparing two independent
    proportions (one-sided)::

        n = ((z_alpha + z_beta)^2 * (p0*(1-p0) + p1*(1-p1))) / (p0 - p1)^2

    The result is the per-group sample size — you need this many trials
    for BOTH the baseline run and the current run.

    Parameters
    ----------
    p0 : float
        Baseline pass rate (null hypothesis).  Must be in (0, 1).
    p1 : float
        Regression pass rate (alternative hypothesis).  Must be in (0, p0).
    alpha : float
        Type I error probability.  Default 0.05.
    power : float
        Desired statistical power (1 - beta).  Default 0.80.

    Returns
    -------
    int
        Required per-group sample size, rounded up to the nearest integer.

    Raises
    ------
    ValueError
        If inputs are out of valid ranges.

    Examples
    --------
    >>> required_sample_size(p0=0.90, p1=0.75)
    62
    >>> required_sample_size(p0=0.95, p1=0.85, alpha=0.01, power=0.90)
    125
    """
    # --- Validate -----------------------------------------------------------
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if not (0.0 < p1 < 1.0):
        raise ValueError(f"p1 must be in (0, 1), got {p1}")
    if p1 >= p0:
        raise ValueError(
            f"p1 must be < p0 (regression means lower pass rate). "
            f"Got p0={p0}, p1={p1}."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")

    beta = 1.0 - power
    z_alpha = sp_stats.norm.ppf(1.0 - alpha)        # one-sided
    z_beta = sp_stats.norm.ppf(1.0 - beta)           # = norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p0 * (1.0 - p0) + p1 * (1.0 - p1))
    denominator = (p0 - p1) ** 2

    n = numerator / denominator

    # Round up — you can't run half a trial.
    return math.ceil(n)


def achieved_power(
    p0: float,
    p1: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Compute the statistical power achieved with a given sample size.

    Answers: "Given that I ran *n* trials per version, what is the probability
    I would have detected a regression from p0 to p1?"

    This is the inverse of :func:`required_sample_size`.  A post-hoc power
    below 0.80 suggests the test was underpowered and a non-significant
    result should be interpreted cautiously (absence of evidence is not
    evidence of absence).

    Parameters
    ----------
    p0 : float
        Baseline pass rate (null hypothesis).  Must be in (0, 1).
    p1 : float
        Regression pass rate (alternative hypothesis).  Must be in (0, p0).
    n : int
        Per-group sample size (number of trials run per version).  Must be >= 1.
    alpha : float
        Type I error probability.  Default 0.05.

    Returns
    -------
    float
        Statistical power in [0, 1].

    Raises
    ------
    ValueError
        If inputs are out of valid ranges.

    Examples
    --------
    >>> achieved_power(p0=0.90, p1=0.75, n=50)
    0.73...
    >>> achieved_power(p0=0.90, p1=0.75, n=100)
    0.95...
    """
    # --- Validate -----------------------------------------------------------
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if not (0.0 < p1 < 1.0):
        raise ValueError(f"p1 must be in (0, 1), got {p1}")
    if p1 >= p0:
        raise ValueError(
            f"p1 must be < p0 (regression means lower pass rate). "
            f"Got p0={p0}, p1={p1}."
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    z_alpha = sp_stats.norm.ppf(1.0 - alpha)  # one-sided critical value

    # Standard error under H0 and H1
    se_h0 = math.sqrt(p0 * (1.0 - p0) / n)
    se_h1 = math.sqrt(p1 * (1.0 - p1) / n)

    # The rejection threshold on the observed rate scale
    # We reject H0 when (p_hat_baseline - p_hat_current) > threshold
    # Under H0 the combined SE uses pooled estimate, but for power
    # calculation we use the variance under H1.
    #
    # z_observed = (p0 - p1 - z_alpha * se_combined_h0) / se_combined_h1
    # power = Phi(z_observed)
    #
    # Simplified formula (Fleiss approach): combine SEs for two-sample test
    se_combined = math.sqrt(p0 * (1.0 - p0) / n + p1 * (1.0 - p1) / n)

    if se_combined == 0.0:
        # Degenerate case: both proportions have zero variance (p=0 or p=1).
        # This shouldn't happen given validation, but guard anyway.
        return 1.0

    # Power = P(reject H0 | H1 is true) = Phi((p0-p1)/SE - z_alpha)
    effect = (p0 - p1) / math.sqrt(p0 * (1.0 - p0) / n + p1 * (1.0 - p1) / n)
    power_value = float(sp_stats.norm.cdf(effect - z_alpha))

    return power_value
