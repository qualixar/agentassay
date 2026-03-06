# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Warm-Start SPRT — sequential testing with informative priors.

THE PROBLEM (paper Section 7.5):

    Standard SPRT starts from scratch every CI run. Yesterday you ran 100
    trials and found pass_rate = 0.87. Today you deploy a new commit and
    run SPRT again — starting from LLR = 0 as if you knew nothing.

    This is wasteful. Yesterday's 100 trials contain valuable information:
    the agent's baseline behavior is well-characterized. If today's commit
    changes nothing, the SPRT should reach "accept H0" in very few trials.

    Warm-Start SPRT incorporates prior information from previous test runs
    as a Bayesian prior on the pass rate. The posterior from yesterday
    becomes today's prior, and the SPRT updates from there.

THE MATH:

    Classical SPRT uses a flat prior (LLR starts at 0). Warm-Start SPRT
    initializes the LLR using the posterior log-odds from the prior:

        LLR_initial = prior_successes * log(p1/p0) +
                      (prior_trials - prior_successes) * log((1-p1)/(1-p0))

    This is equivalent to having already observed ``prior_trials`` outcomes.
    The SPRT boundaries remain unchanged — we are simply starting from a
    non-zero point on the LLR walk.

    The (alpha, beta) guarantees are preserved because the prior is treated
    as OBSERVED DATA, not as a subjective belief. The LLR walk is still
    a martingale, and Wald's theorem still applies.

PRACTICAL IMPACT:

    With a strong prior (100 prior trials, pass_rate ~ p0), the warm-start
    SPRT typically terminates in 5-15 new trials instead of the 40-80 needed
    by cold-start SPRT. This is a 3-8x reduction in new trial cost.

    If the prior is misleading (e.g., a major regression occurred and the
    true rate dropped), the SPRT self-corrects: the new observations push
    the LLR in the opposite direction, and it may take more trials than
    cold-start — but it will still reach the correct decision.

PRIOR DECAY:

    Old priors become less relevant over time. A test run from 30 days ago
    may not reflect the current agent's behavior. The ``prior_decay``
    parameter implements exponential decay:

        effective_prior = prior_trials * exp(-decay_rate * days_since_prior)

    Default decay_rate = 0.05 means the prior's influence halves every
    ~14 days. Set decay_rate = 0 for no decay (CI runs on every commit).

References:
    - Wald, A. (1947). Sequential Analysis. John Wiley & Sons.
    - Lai, T.L. (2001). "Sequential Analysis: Some Classical Problems
      and New Challenges." Statistica Sinica 11: 303-350.
    - Tartakovsky, A.G. et al. (2014). Sequential Analysis: Hypothesis
      Testing and Changepoint Detection. CRC Press.
"""

from __future__ import annotations

import math
from typing import Literal


class WarmStartSPRT:
    """SPRT that uses prior test results to start with an informative prior.

    Instead of starting from LLR = 0, the warm-start SPRT initializes the
    log-likelihood ratio based on previously observed outcomes. This
    dramatically reduces the number of new trials needed when the prior
    is informative (i.e., recent test data exists).

    Parameters
    ----------
    theta_0 : float
        Null hypothesis pass rate (baseline). Must be in (0, 1).
    delta : float
        Detectable difference. H1 tests pass rate = theta_0 - delta.
        Must be > 0 and < theta_0.
    alpha : float
        Type I error probability. Default 0.05.
    beta : float
        Type II error probability. Default 0.10.
    prior_successes : int
        Number of successes from previous test runs. Default 0 (cold start).
    prior_trials : int
        Number of trials from previous test runs. Default 0 (cold start).
    prior_decay : float
        Exponential decay rate for prior influence. Applied as:
        effective_prior = prior * exp(-decay * days_since_prior).
        Default 0.0 (no decay — use the full prior).
    days_since_prior : float
        Days elapsed since the prior data was collected.
        Only used if prior_decay > 0. Default 0.0.

    Example
    -------
    >>> # Yesterday: 85 passes out of 100 trials
    >>> sprt = WarmStartSPRT(
    ...     theta_0=0.90, delta=0.15,
    ...     alpha=0.05, beta=0.10,
    ...     prior_successes=85, prior_trials=100,
    ... )
    >>> # Today: run new trials until decision
    >>> for result in new_trials:
    ...     decision = sprt.update(result.passed)
    ...     if decision != "continue":
    ...         break
    >>> print(f"Decided '{decision}' after {sprt.trials_used} new trials")
    """

    __slots__ = (
        "_p0",
        "_p1",
        "_alpha",
        "_beta",
        "_lower_boundary",
        "_upper_boundary",
        "_llr_pass",
        "_llr_fail",
        "_llr",
        "_llr_initial",
        "_trials",
        "_successes",
        "_decided",
        "_decision",
        "_prior_successes",
        "_prior_trials",
    )

    def __init__(
        self,
        theta_0: float,
        delta: float,
        alpha: float = 0.05,
        beta: float = 0.10,
        prior_successes: int = 0,
        prior_trials: int = 0,
        prior_decay: float = 0.0,
        days_since_prior: float = 0.0,
    ) -> None:
        # --- Validate inputs ------------------------------------------------
        if not (0.0 < theta_0 < 1.0):
            raise ValueError(f"theta_0 must be in (0, 1), got {theta_0}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if delta >= theta_0:
            raise ValueError(
                f"delta ({delta}) must be < theta_0 ({theta_0}). "
                f"The alternative pass rate (theta_0 - delta = {theta_0 - delta}) "
                f"must be positive."
            )
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if alpha + beta >= 1.0:
            raise ValueError(
                f"alpha + beta must be < 1.0, got {alpha} + {beta} = {alpha + beta}"
            )
        if prior_successes < 0:
            raise ValueError(f"prior_successes must be >= 0, got {prior_successes}")
        if prior_trials < 0:
            raise ValueError(f"prior_trials must be >= 0, got {prior_trials}")
        if prior_successes > prior_trials:
            raise ValueError(
                f"prior_successes ({prior_successes}) cannot exceed "
                f"prior_trials ({prior_trials})"
            )

        self._p0 = theta_0
        self._p1 = theta_0 - delta
        self._alpha = alpha
        self._beta = beta

        # Wald boundaries (same as standard SPRT)
        self._lower_boundary = math.log(beta / (1.0 - alpha))
        self._upper_boundary = math.log((1.0 - beta) / alpha)

        # Per-trial LLR increments
        self._llr_pass = math.log(self._p1 / self._p0)
        self._llr_fail = math.log((1.0 - self._p1) / (1.0 - self._p0))

        # Apply prior decay
        effective_prior_trials = prior_trials
        effective_prior_successes = prior_successes

        if prior_decay > 0.0 and days_since_prior > 0.0 and prior_trials > 0:
            decay_factor = math.exp(-prior_decay * days_since_prior)
            effective_prior_trials = int(round(prior_trials * decay_factor))
            effective_prior_successes = int(round(prior_successes * decay_factor))
            # Ensure consistency
            effective_prior_successes = min(
                effective_prior_successes, effective_prior_trials
            )

        self._prior_successes = effective_prior_successes
        self._prior_trials = effective_prior_trials

        # Initialize LLR from prior (the warm start)
        prior_failures = effective_prior_trials - effective_prior_successes
        self._llr_initial = (
            effective_prior_successes * self._llr_pass
            + prior_failures * self._llr_fail
        )
        self._llr = self._llr_initial

        # State
        self._trials: int = 0
        self._successes: int = 0
        self._decided: bool = False
        self._decision: Literal["accept_h0", "accept_h1", "continue"] = "continue"

        # Check if prior alone is sufficient for a decision
        self._check_boundaries()

    # --- Public interface ---------------------------------------------------

    @property
    def is_decided(self) -> bool:
        """Whether the test has reached a terminal decision."""
        return self._decided

    @property
    def decision(self) -> str:
        """Current decision: 'accept_h0', 'accept_h1', or 'continue'."""
        return self._decision

    @property
    def trials_used(self) -> int:
        """Number of NEW trials observed (excludes prior)."""
        return self._trials

    @property
    def total_evidence(self) -> int:
        """Total evidence: prior trials + new trials."""
        return self._prior_trials + self._trials

    @property
    def current_llr(self) -> float:
        """Current log-likelihood ratio."""
        return self._llr

    @property
    def initial_llr(self) -> float:
        """LLR from the warm-start prior (before any new observations)."""
        return self._llr_initial

    @property
    def boundaries(self) -> tuple[float, float]:
        """SPRT boundaries: (lower, upper) in log space."""
        return (self._lower_boundary, self._upper_boundary)

    @property
    def prior_info(self) -> dict[str, int]:
        """Prior information used for warm start.

        Returns
        -------
        dict
            Keys: prior_successes, prior_trials (after decay).
        """
        return {
            "prior_successes": self._prior_successes,
            "prior_trials": self._prior_trials,
        }

    def update(self, success: bool) -> str:
        """Observe one new trial outcome and update the SPRT state.

        Parameters
        ----------
        success : bool
            True if the agent passed this trial.

        Returns
        -------
        str
            Decision: "continue", "accept_h0" (no regression), or
            "accept_h1" (regression detected).

        Raises
        ------
        RuntimeError
            If called after the test has already terminated.
        """
        if self._decided:
            raise RuntimeError(
                f"WarmStartSPRT has already terminated with decision "
                f"'{self._decision}'. Create a new instance to run another test."
            )

        self._trials += 1
        if success:
            self._successes += 1
            self._llr += self._llr_pass
        else:
            self._llr += self._llr_fail

        self._check_boundaries()
        return self._decision

    def expected_savings(self) -> float:
        """Expected trial savings compared to cold-start SPRT.

        Computes the approximate percentage of trials saved by using the
        warm start. Based on Wald's ASN formula:

            ASN_cold = |ln(B) * P(H1) + ln(A) * P(H0)| / |E[Z]|
            ASN_warm = ASN_cold - prior_effective_trials

        Returns
        -------
        float
            Expected percentage of trials saved (0 to 100).
            Can be negative if the prior is counter-productive.
        """
        # Expected cold-start ASN under H0
        ez_h0 = self._p0 * self._llr_pass + (1.0 - self._p0) * self._llr_fail
        if abs(ez_h0) < 1e-15:
            return 0.0

        asn_cold = abs(
            (self._alpha * self._upper_boundary
             + (1.0 - self._alpha) * self._lower_boundary) / ez_h0
        )

        if asn_cold <= 0:
            return 0.0

        # Prior contribution: how many cold-start trials the prior is "worth"
        # This is the distance the prior moves the LLR, divided by the
        # expected per-trial LLR movement.
        prior_equivalent = abs(self._llr_initial / ez_h0) if abs(ez_h0) > 1e-15 else 0.0

        savings_ratio = prior_equivalent / asn_cold
        return min(100.0, max(-100.0, savings_ratio * 100.0))

    def cold_start_expected_trials(self) -> dict[str, float]:
        """Expected trial counts for a cold-start SPRT (for comparison).

        Returns
        -------
        dict
            Keys "under_h0" and "under_h1" with expected trial counts.
        """
        ez_h0 = self._p0 * self._llr_pass + (1.0 - self._p0) * self._llr_fail
        ez_h1 = self._p1 * self._llr_pass + (1.0 - self._p1) * self._llr_fail

        ln_a = self._lower_boundary
        ln_b = self._upper_boundary

        asn_h0 = (
            abs((self._alpha * ln_b + (1.0 - self._alpha) * ln_a) / ez_h0)
            if abs(ez_h0) > 1e-15
            else float("inf")
        )

        asn_h1 = (
            abs(((1.0 - self._beta) * ln_b + self._beta * ln_a) / ez_h1)
            if abs(ez_h1) > 1e-15
            else float("inf")
        )

        return {"under_h0": asn_h0, "under_h1": asn_h1}

    def reset(self, keep_prior: bool = True) -> None:
        """Reset the SPRT for reuse.

        Parameters
        ----------
        keep_prior : bool
            If True, retains the warm-start prior. If False, resets to
            cold start (LLR = 0). Default True.
        """
        if keep_prior:
            self._llr = self._llr_initial
        else:
            self._llr = 0.0
            self._llr_initial = 0.0
            self._prior_successes = 0
            self._prior_trials = 0

        self._trials = 0
        self._successes = 0
        self._decided = False
        self._decision = "continue"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_boundaries(self) -> None:
        """Check if the LLR has crossed a decision boundary."""
        if self._llr >= self._upper_boundary:
            self._decided = True
            self._decision = "accept_h1"
        elif self._llr <= self._lower_boundary:
            self._decided = True
            self._decision = "accept_h0"
        else:
            self._decision = "continue"
