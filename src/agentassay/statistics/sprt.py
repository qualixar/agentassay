# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Sequential Probability Ratio Test (SPRT) for cost-efficient agent testing.

Wald's SPRT (1947) is an adaptive stopping procedure: instead of fixing the
number of agent trials in advance, we observe results one at a time and stop
as soon as we have enough evidence to decide.

Why this matters for agent testing:
    Each agent trial costs real money (LLM API calls, tool invocations,
    compute time).  A fixed-sample test might require 200 trials to reach
    a conclusion.  SPRT often reaches the same conclusion in 40-80 trials
    when the true state is clear, cutting costs by 60-80 %.

    The tradeoff: SPRT takes *more* trials when the true rate is ambiguous
    (near the boundary between H0 and H1).  But in practice agent regressions
    are usually either obvious or absent — the ambiguous middle ground is rare.

Hypotheses:
    H0: pass rate = p0   (no regression, baseline performance maintained)
    H1: pass rate = p1   (regression occurred, p1 < p0)

After each trial the log-likelihood ratio (LLR) is updated and compared to
two boundaries derived from the desired Type I (alpha) and Type II (beta)
error probabilities.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel


class SPRTResult(BaseModel, frozen=True):
    """Immutable snapshot of the SPRT state after an update.

    Attributes
    ----------
    decision : str
        One of ``"accept_h0"`` (no regression), ``"accept_h1"`` (regression
        detected), or ``"continue"`` (keep testing).
    trials_used : int
        Number of trials observed so far.
    log_likelihood_ratio : float
        Current cumulative LLR.
    upper_boundary : float
        ln(B) — crossing this accepts H1 (regression).
    lower_boundary : float
        ln(A) — crossing this accepts H0 (no regression).
    p0 : float
        Null hypothesis pass rate.
    p1 : float
        Alternative hypothesis pass rate.
    """

    decision: Literal["accept_h0", "accept_h1", "continue"]
    trials_used: int
    log_likelihood_ratio: float
    upper_boundary: float
    lower_boundary: float
    p0: float
    p1: float


class SPRTRunner:
    """Wald's Sequential Probability Ratio Test for agent regression detection.

    Usage::

        runner = SPRTRunner(p0=0.90, p1=0.75, alpha=0.05, beta=0.20)
        for trial_result in run_agent_trials():
            result = runner.update(passed=trial_result.passed)
            if result.decision != "continue":
                break
        print(result)

    Parameters
    ----------
    p0 : float
        Baseline pass rate — null hypothesis.  Must be in (0, 1).
    p1 : float
        Regression pass rate — alternative hypothesis.  Must be in (0, p0).
        The further p1 is from p0, the fewer trials SPRT needs on average.
    alpha : float
        Type I error probability — probability of falsely declaring regression
        when none occurred.  Default 0.05.
    beta : float
        Type II error probability — probability of missing a real regression.
        Default 0.20 (power = 0.80).
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        alpha: float = 0.05,
        beta: float = 0.20,
    ) -> None:
        # --- Validate inputs ------------------------------------------------
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
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        self._p0 = p0
        self._p1 = p1
        self._alpha = alpha
        self._beta = beta

        # Wald boundaries (in log space)
        # A = beta / (1 - alpha)          — accept H0 boundary
        # B = (1 - beta) / alpha          — accept H1 boundary
        # ln(A) is the LOWER boundary, ln(B) is the UPPER boundary.
        self._lower_boundary = math.log(beta / (1.0 - alpha))
        self._upper_boundary = math.log((1.0 - beta) / alpha)

        # Incremental log-likelihood ratio contributions
        # For a pass:  log(p1 / p0)
        # For a fail:  log((1 - p1) / (1 - p0))
        self._llr_pass = math.log(p1 / p0)
        self._llr_fail = math.log((1.0 - p1) / (1.0 - p0))

        # State
        self._llr: float = 0.0
        self._trials: int = 0
        self._decided: bool = False
        self._decision: Literal["accept_h0", "accept_h1", "continue"] = "continue"

    # --- Public interface ---------------------------------------------------

    @property
    def is_decided(self) -> bool:
        """Whether SPRT has reached a terminal decision."""
        return self._decided

    def update(self, passed: bool) -> SPRTResult:
        """Observe one trial outcome and update the SPRT state.

        Parameters
        ----------
        passed : bool
            ``True`` if the agent passed this trial, ``False`` otherwise.

        Returns
        -------
        SPRTResult
            Snapshot of the SPRT state after this update.

        Raises
        ------
        RuntimeError
            If called after the test has already reached a terminal decision.
        """
        if self._decided:
            raise RuntimeError(
                f"SPRT has already terminated with decision '{self._decision}'. "
                f"Create a new SPRTRunner to run another test."
            )

        self._trials += 1

        if passed:
            self._llr += self._llr_pass
        else:
            self._llr += self._llr_fail

        # Check boundaries
        if self._llr >= self._upper_boundary:
            self._decided = True
            self._decision = "accept_h1"
        elif self._llr <= self._lower_boundary:
            self._decided = True
            self._decision = "accept_h0"
        else:
            self._decision = "continue"

        return SPRTResult(
            decision=self._decision,
            trials_used=self._trials,
            log_likelihood_ratio=self._llr,
            upper_boundary=self._upper_boundary,
            lower_boundary=self._lower_boundary,
            p0=self._p0,
            p1=self._p1,
        )

    def reset(self) -> None:
        """Reset the runner to its initial state for reuse."""
        self._llr = 0.0
        self._trials = 0
        self._decided = False
        self._decision = "continue"

    def expected_sample_size(self) -> dict[str, float]:
        """Compute the expected number of trials under H0 and H1.

        Wald's approximation for the expected sample size (ASN) is::

            E[N | H_i] = (P(accept H1 | H_i) * ln(B) + P(accept H0 | H_i) * ln(A))
                         / E[Z | H_i]

        where Z is the per-trial log-likelihood ratio increment.

        Returns
        -------
        dict
            Keys ``"under_h0"`` and ``"under_h1"`` with the expected trial
            counts.  These are theoretical approximations — actual trial
            counts will vary due to overshoot past the boundaries.
        """
        ln_a = self._lower_boundary
        ln_b = self._upper_boundary

        # Expected per-trial LLR increment under H0 (truth = p0)
        # E[Z | H0] = p0 * log(p1/p0) + (1-p0) * log((1-p1)/(1-p0))
        ez_h0 = self._p0 * self._llr_pass + (1.0 - self._p0) * self._llr_fail

        # Expected per-trial LLR increment under H1 (truth = p1)
        # E[Z | H1] = p1 * log(p1/p0) + (1-p1) * log((1-p1)/(1-p0))
        ez_h1 = self._p1 * self._llr_pass + (1.0 - self._p1) * self._llr_fail

        # Under H0: P(accept H1) ~= alpha, P(accept H0) ~= 1 - alpha
        # Under H1: P(accept H1) ~= 1 - beta, P(accept H0) ~= beta
        asn_h0 = (
            (self._alpha * ln_b + (1.0 - self._alpha) * ln_a) / ez_h0
            if ez_h0 != 0.0
            else float("inf")
        )

        asn_h1 = (
            ((1.0 - self._beta) * ln_b + self._beta * ln_a) / ez_h1
            if ez_h1 != 0.0
            else float("inf")
        )

        return {
            "under_h0": abs(asn_h0),
            "under_h1": abs(asn_h1),
        }
