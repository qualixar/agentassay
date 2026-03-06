# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Core stochastic test semantics for agent regression testing.

This module implements Definition 3.2 from the paper: the (alpha, beta, n)-test
triple that maps non-deterministic agent trial outcomes into formal verdicts
with statistical guarantees.

Key Insight:
    Traditional software testing is binary: a test passes or fails.
    Agent testing must be STOCHASTIC because:
    1. LLMs are inherently non-deterministic (temperature, sampling)
    2. Tool outputs vary (APIs, search results, file systems)
    3. Environment state changes between runs

    Therefore, an agent test verdict is a PROBABILISTIC JUDGMENT:
        - PASS:          Sufficient evidence of no regression
        - FAIL:          Sufficient evidence of regression
        - INCONCLUSIVE:  Insufficient evidence either way

    The verdict is backed by confidence intervals and hypothesis tests,
    not by a single binary outcome.

Formal Definition (Definition 3.2):
    A stochastic test verdict V is a function:

        V: (Results, alpha, beta, n) -> {PASS, FAIL, INCONCLUSIVE}

    where:
        Results = sequence of Bernoulli outcomes from n agent trials
        alpha   = Type I error rate (false positive: claiming regression when none)
        beta    = Type II error rate (false negative: missing real regression)
        n       = number of trials

    Single-version evaluation (threshold test):
        Given pass rate p_hat and threshold tau, with CI = [L, U]:
            V = PASS          if L >= tau
            V = FAIL          if U < tau
            V = INCONCLUSIVE  otherwise

    Cross-version evaluation (regression test):
        Given baseline results B and current results C:
            V = FAIL  if H0 rejected (p < alpha) AND current_rate < baseline_rate
            V = PASS  if H0 not rejected (p >= alpha) AND power >= 1-beta
            V = INCONCLUSIVE  if H0 not rejected but power < 1-beta

References:
    - Neyman, J. & Pearson, E.S. (1933). "On the Problem of the Most
      Efficient Tests of Statistical Hypotheses." Phil. Trans. Royal
      Society A 231: 289-337.
    - Wald, A. (1947). Sequential Analysis. John Wiley & Sons.
    - Wilson, E.B. (1927). "Probable Inference." JASA 22(158): 209-212.
"""

from __future__ import annotations

from datetime import datetime, timezone

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # Python 3.10 compat
        pass


from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from agentassay.statistics.confidence import (
    ConfidenceMethod,
    binomial_confidence_interval,
    minimum_sample_size,
)
from agentassay.statistics.hypothesis_legacy import (
    RegressionTest,
    ScoreTest,
    test_binary_regression,
    test_score_regression,
)


class VerdictStatus(StrEnum):
    """Stochastic test verdict status.

    Unlike traditional binary pass/fail, agent tests have three possible
    outcomes because non-deterministic systems require statistical evidence
    to make definitive claims.
    """

    PASS = "pass"
    """Sufficient statistical evidence of no regression."""

    FAIL = "fail"
    """Sufficient statistical evidence of regression."""

    INCONCLUSIVE = "inconclusive"
    """Insufficient evidence to determine pass or fail.

    This occurs when the confidence interval spans the threshold,
    or when statistical power is too low to detect a real regression.
    Typical remedy: increase the number of trials (n).
    """


class StochasticVerdict(BaseModel):
    """A statistically-backed verdict for an agent test.

    This is the primary output of the verdict system. Every field is
    computed from actual trial data — nothing is assumed or fabricated.

    Attributes:
        status: The verdict (PASS, FAIL, or INCONCLUSIVE).
        confidence: The confidence level used (e.g., 0.95).
        pass_rate: Observed pass rate across all trials.
        pass_rate_ci: Wilson (or other) confidence interval for the true pass rate.
        num_trials: Total number of trials executed.
        num_passed: Number of trials that passed.
        p_value: p-value from regression hypothesis test, if comparing versions.
        effect_size: Standardized effect size (Cohen's h or rank-biserial r).
        effect_size_interpretation: Human-readable effect size label.
        regression_detected: Whether a statistically significant regression was found.
        details: Additional metadata (test name, power, thresholds, etc.).
        timestamp: When this verdict was computed (UTC).
    """

    model_config = {"frozen": True}

    status: VerdictStatus
    confidence: float = Field(ge=0.0, le=1.0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    pass_rate_ci: tuple[float, float]
    num_trials: int = Field(ge=0)
    num_passed: int = Field(ge=0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    effect_size: float | None = None
    effect_size_interpretation: str | None = None
    regression_detected: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("pass_rate_ci")
    @classmethod
    def _validate_ci_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        lower, upper = v
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError(
                f"CI bounds must satisfy 0 <= lower <= upper <= 1, got ({lower}, {upper})"
            )
        return v

    @model_validator(mode="after")
    def _validate_counts(self) -> StochasticVerdict:
        if self.num_passed > self.num_trials:
            raise ValueError(
                f"num_passed ({self.num_passed}) cannot exceed num_trials ({self.num_trials})"
            )
        return self

    @property
    def is_definitive(self) -> bool:
        """True if the verdict is PASS or FAIL (not INCONCLUSIVE).

        A definitive verdict means we have enough statistical evidence
        to make a clear determination. An inconclusive result indicates
        that more trials are needed.
        """
        return self.status in (VerdictStatus.PASS, VerdictStatus.FAIL)

    @property
    def margin_of_error(self) -> float:
        """Half-width of the confidence interval.

        Smaller margin = more precise estimate of the true pass rate.
        A large margin of error suggests the number of trials is too low.
        """
        lower, upper = self.pass_rate_ci
        return (upper - lower) / 2.0


class VerdictFunction:
    """The (alpha, beta, n)-test triple for stochastic agent testing.

    This is Definition 3.2 from the paper. It encapsulates the statistical
    parameters that govern how trial outcomes are mapped to verdicts.

    The three parameters control the operating characteristics:
        alpha (significance level): Maximum acceptable false positive rate.
            Lower alpha = harder to wrongly claim regression.
        beta (Type II error): Maximum acceptable false negative rate.
            Lower beta = harder to miss a real regression.
        min_trials: Minimum number of trials before issuing a verdict.
            More trials = tighter confidence intervals = more definitive verdicts.

    Example:
        >>> vf = VerdictFunction(alpha=0.05, beta=0.20, min_trials=30)
        >>> results = [True] * 27 + [False] * 3  # 90% pass rate
        >>> verdict = vf.evaluate_single(results, threshold=0.80)
        >>> verdict.status
        <VerdictStatus.PASS: 'pass'>

    Args:
        alpha: Significance level for hypothesis tests. Default 0.05.
        beta: Type II error rate (1 - statistical power). Default 0.20.
        min_trials: Minimum trials required. Default 30.
        confidence_method: CI method ('wilson', 'clopper_pearson', etc.). Default 'wilson'.
        regression_test: Hypothesis test for binary regression ('fisher', 'chi2'). Default 'fisher'.
    """

    __slots__ = (
        "_alpha",
        "_beta",
        "_min_trials",
        "_confidence_method",
        "_regression_test",
        "_confidence_level",
    )

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.20,
        min_trials: int = 30,
        confidence_method: str = "wilson",
        regression_test: str = "fisher",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if min_trials < 1:
            raise ValueError(f"min_trials must be >= 1, got {min_trials}")

        self._alpha = alpha
        self._beta = beta
        self._min_trials = min_trials
        self._confidence_method = ConfidenceMethod(confidence_method)
        self._regression_test = RegressionTest(regression_test)
        self._confidence_level = 1.0 - alpha

    @property
    def alpha(self) -> float:
        """Significance level (Type I error rate)."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Type II error rate (1 - power)."""
        return self._beta

    @property
    def min_trials(self) -> int:
        """Minimum number of trials required for a verdict."""
        return self._min_trials

    @property
    def confidence_level(self) -> float:
        """Confidence level for intervals (1 - alpha)."""
        return self._confidence_level

    def evaluate_single(
        self,
        results: list[bool],
        threshold: float = 0.80,
    ) -> StochasticVerdict:
        """Evaluate a single version's trial results against a threshold.

        This is the simplest verdict: does the agent meet a minimum pass
        rate with statistical confidence?

        Decision Logic:
            Let CI = [L, U] be the confidence interval for the true pass rate.
            - PASS:          L >= threshold  (entire CI is above threshold)
            - FAIL:          U < threshold   (entire CI is below threshold)
            - INCONCLUSIVE:  otherwise       (CI straddles the threshold)

        Args:
            results: List of boolean trial outcomes (True = passed, False = failed).
            threshold: Minimum acceptable pass rate in [0, 1]. Default 0.80.

        Returns:
            StochasticVerdict with the determination.

        Raises:
            ValueError: If threshold is out of range or results is empty.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        n = len(results)
        if n == 0:
            return self._empty_verdict(threshold=threshold)

        k = sum(results)
        p_hat = k / n

        # Check minimum trial count
        min_required = max(
            self._min_trials,
            minimum_sample_size(self._confidence_method),
        )
        below_minimum = n < min_required

        # Compute confidence interval
        ci_lower, ci_upper = binomial_confidence_interval(
            n_successes=k,
            n_trials=n,
            confidence_level=self._confidence_level,
            method=self._confidence_method,
        )

        # Decision logic per Definition 3.2
        if below_minimum:
            status = VerdictStatus.INCONCLUSIVE
            reason = f"Insufficient trials: {n} < {min_required} minimum required"
        elif ci_lower >= threshold:
            status = VerdictStatus.PASS
            reason = f"CI lower bound ({ci_lower:.4f}) >= threshold ({threshold:.4f})"
        elif ci_upper < threshold:
            status = VerdictStatus.FAIL
            reason = f"CI upper bound ({ci_upper:.4f}) < threshold ({threshold:.4f})"
        else:
            status = VerdictStatus.INCONCLUSIVE
            reason = (
                f"CI [{ci_lower:.4f}, {ci_upper:.4f}] straddles "
                f"threshold ({threshold:.4f}). Need more trials."
            )

        return StochasticVerdict(
            status=status,
            confidence=self._confidence_level,
            pass_rate=p_hat,
            pass_rate_ci=(ci_lower, ci_upper),
            num_trials=n,
            num_passed=k,
            p_value=None,
            effect_size=None,
            effect_size_interpretation=None,
            regression_detected=(status == VerdictStatus.FAIL),
            details={
                "verdict_type": "single_threshold",
                "threshold": threshold,
                "confidence_method": str(self._confidence_method),
                "reason": reason,
                "below_minimum_trials": below_minimum,
                "min_trials_required": min_required,
            },
        )

    def evaluate_regression(
        self,
        baseline_results: list[bool],
        current_results: list[bool],
    ) -> StochasticVerdict:
        """Compare two versions' trial results for regression detection.

        This is the cross-version verdict: has the agent regressed from
        the baseline to the current version?

        Decision Logic:
            Run a hypothesis test (Fisher/chi2) comparing pass rates:
            H0: current_rate >= baseline_rate (no regression)
            H1: current_rate < baseline_rate  (regression occurred)

            - FAIL:          H0 rejected (p < alpha) AND current_rate < baseline_rate
            - PASS:          H0 not rejected AND estimated power >= (1 - beta)
            - INCONCLUSIVE:  H0 not rejected but power < (1 - beta)

            The INCONCLUSIVE case catches scenarios where we fail to reject
            H0 simply because we lack statistical power (too few trials),
            not because there is genuinely no regression.

        Args:
            baseline_results: Boolean outcomes from the baseline version.
            current_results: Boolean outcomes from the current version.

        Returns:
            StochasticVerdict with regression analysis.

        Raises:
            ValueError: If either result list is empty.
        """
        n_baseline = len(baseline_results)
        n_current = len(current_results)

        if n_baseline == 0 or n_current == 0:
            return self._empty_verdict(note="Cannot evaluate regression with empty results")

        k_baseline = sum(baseline_results)
        k_current = sum(current_results)
        p_baseline = k_baseline / n_baseline
        p_current = k_current / n_current

        # Check minimum trial counts
        min_required = max(
            self._min_trials,
            minimum_sample_size(self._confidence_method),
        )
        below_minimum = (n_baseline < min_required) or (n_current < min_required)

        # Confidence interval on the CURRENT version's pass rate
        ci_lower, ci_upper = binomial_confidence_interval(
            n_successes=k_current,
            n_trials=n_current,
            confidence_level=self._confidence_level,
            method=self._confidence_method,
        )

        # Hypothesis test for regression
        hyp_result = test_binary_regression(
            baseline_successes=k_baseline,
            baseline_trials=n_baseline,
            current_successes=k_current,
            current_trials=n_current,
            alpha=self._alpha,
            method=self._regression_test,
        )

        # Decision logic
        if below_minimum:
            status = VerdictStatus.INCONCLUSIVE
            reason = (
                f"Insufficient trials: baseline={n_baseline}, "
                f"current={n_current}, minimum={min_required}"
            )
        elif hyp_result.regression_detected:
            status = VerdictStatus.FAIL
            reason = (
                f"Regression detected: p={hyp_result.p_value:.6f} < "
                f"alpha={self._alpha}, current rate ({p_current:.4f}) < "
                f"baseline rate ({p_baseline:.4f}), "
                f"effect size={hyp_result.effect_size:.4f} "
                f"({hyp_result.effect_size_interpretation})"
            )
        elif hyp_result.power is not None and hyp_result.power >= (1.0 - self._beta):
            status = VerdictStatus.PASS
            reason = (
                f"No regression: p={hyp_result.p_value:.6f} >= "
                f"alpha={self._alpha}, power={hyp_result.power:.4f} >= "
                f"{1.0 - self._beta:.4f}"
            )
        elif hyp_result.power is not None and hyp_result.power < (1.0 - self._beta):
            status = VerdictStatus.INCONCLUSIVE
            reason = (
                f"Underpowered: p={hyp_result.p_value:.6f} >= "
                f"alpha={self._alpha}, but power={hyp_result.power:.4f} < "
                f"{1.0 - self._beta:.4f}. Increase trials for definitive result."
            )
        else:
            # Power could not be estimated — treat as inconclusive unless
            # the pass rates are identical (strong evidence of no regression)
            if abs(p_current - p_baseline) < 1e-10:
                status = VerdictStatus.PASS
                reason = (
                    f"No regression: pass rates are identical "
                    f"({p_baseline:.4f}), p={hyp_result.p_value:.6f}"
                )
            else:
                status = VerdictStatus.INCONCLUSIVE
                reason = (
                    f"Cannot estimate power. p={hyp_result.p_value:.6f} >= "
                    f"alpha={self._alpha}. Recommend more trials."
                )

        return StochasticVerdict(
            status=status,
            confidence=self._confidence_level,
            pass_rate=p_current,
            pass_rate_ci=(ci_lower, ci_upper),
            num_trials=n_current,
            num_passed=k_current,
            p_value=hyp_result.p_value,
            effect_size=hyp_result.effect_size,
            effect_size_interpretation=hyp_result.effect_size_interpretation,
            regression_detected=hyp_result.regression_detected,
            details={
                "verdict_type": "regression",
                "test_name": hyp_result.test_name,
                "statistic": hyp_result.statistic,
                "power": hyp_result.power,
                "alpha": self._alpha,
                "beta": self._beta,
                "baseline_pass_rate": p_baseline,
                "baseline_trials": n_baseline,
                "baseline_passed": k_baseline,
                "confidence_method": str(self._confidence_method),
                "regression_test": str(self._regression_test),
                "reason": reason,
                "below_minimum_trials": below_minimum,
            },
        )

    def evaluate_scores(
        self,
        baseline_scores: list[float],
        current_scores: list[float],
        score_test: str = "mann_whitney",
    ) -> StochasticVerdict:
        """Compare continuous score distributions for regression.

        When agent quality is measured by continuous scores (e.g., BLEU,
        ROUGE, similarity scores, latency) rather than binary pass/fail,
        this method tests whether the score distribution has degraded.

        Decision Logic:
            Run a non-parametric test (Mann-Whitney / KS / Welch's t):
            H0: score distributions are the same
            H1: current scores are worse (stochastically smaller)

            - FAIL:          H0 rejected (p < alpha) AND current mean < baseline mean
            - PASS:          H0 not rejected (p >= alpha)
            - INCONCLUSIVE:  insufficient data or edge cases

            Pass rate is synthesized from scores: the fraction of current scores
            that exceed the baseline median, providing a bridge between continuous
            and binary semantics.

        Args:
            baseline_scores: Score values from baseline version.
            current_scores: Score values from current version.
            score_test: Test method ('mann_whitney', 'ks', 'welch_t'). Default 'mann_whitney'.

        Returns:
            StochasticVerdict with score comparison analysis.

        Raises:
            ValueError: If either score list is empty.
        """
        if len(baseline_scores) == 0 or len(current_scores) == 0:
            return self._empty_verdict(note="Cannot evaluate scores with empty data")

        baseline_arr = np.asarray(baseline_scores, dtype=np.float64)
        current_arr = np.asarray(current_scores, dtype=np.float64)

        n_baseline = len(baseline_arr)
        n_current = len(current_arr)

        # Check minimum trial counts
        min_required = max(self._min_trials, 5)
        below_minimum = (n_baseline < min_required) or (n_current < min_required)

        # Synthesize a pass rate: fraction of current scores >= baseline median
        baseline_median = float(np.median(baseline_arr))
        if baseline_median == 0.0 and float(np.max(baseline_arr)) == 0.0:
            # All baseline scores are zero — use equality check
            scores_meeting = int(np.sum(current_arr >= baseline_median))
        else:
            scores_meeting = int(np.sum(current_arr >= baseline_median))
        synthetic_pass_rate = scores_meeting / n_current if n_current > 0 else 0.0

        # CI on synthetic pass rate
        ci_lower, ci_upper = binomial_confidence_interval(
            n_successes=scores_meeting,
            n_trials=n_current,
            confidence_level=self._confidence_level,
            method=self._confidence_method,
        )

        # Hypothesis test
        score_method = ScoreTest(score_test)
        hyp_result = test_score_regression(
            baseline_scores=baseline_scores,
            current_scores=current_scores,
            alpha=self._alpha,
            method=score_method,
        )

        # Decision logic
        current_mean = float(np.mean(current_arr))
        baseline_mean = float(np.mean(baseline_arr))

        if below_minimum:
            status = VerdictStatus.INCONCLUSIVE
            reason = (
                f"Insufficient samples: baseline={n_baseline}, "
                f"current={n_current}, minimum={min_required}"
            )
        elif hyp_result.regression_detected:
            status = VerdictStatus.FAIL
            reason = (
                f"Score regression detected: p={hyp_result.p_value:.6f} < "
                f"alpha={self._alpha}, current mean ({current_mean:.4f}) < "
                f"baseline mean ({baseline_mean:.4f}), "
                f"effect={hyp_result.effect_size:.4f} "
                f"({hyp_result.effect_size_interpretation})"
            )
        elif hyp_result.p_value >= self._alpha:
            status = VerdictStatus.PASS
            reason = f"No score regression: p={hyp_result.p_value:.6f} >= alpha={self._alpha}"
        else:
            status = VerdictStatus.INCONCLUSIVE
            reason = f"Ambiguous: p={hyp_result.p_value:.6f}, direction unclear"

        return StochasticVerdict(
            status=status,
            confidence=self._confidence_level,
            pass_rate=synthetic_pass_rate,
            pass_rate_ci=(ci_lower, ci_upper),
            num_trials=n_current,
            num_passed=scores_meeting,
            p_value=hyp_result.p_value,
            effect_size=hyp_result.effect_size,
            effect_size_interpretation=hyp_result.effect_size_interpretation,
            regression_detected=hyp_result.regression_detected,
            details={
                "verdict_type": "score_comparison",
                "test_name": hyp_result.test_name,
                "statistic": hyp_result.statistic,
                "score_test": str(score_method),
                "baseline_mean": baseline_mean,
                "baseline_median": baseline_median,
                "baseline_std": float(np.std(baseline_arr, ddof=1)) if n_baseline > 1 else 0.0,
                "current_mean": current_mean,
                "current_median": float(np.median(current_arr)),
                "current_std": float(np.std(current_arr, ddof=1)) if n_current > 1 else 0.0,
                "baseline_samples": n_baseline,
                "current_samples": n_current,
                "synthetic_pass_rate_method": "fraction >= baseline median",
                "alpha": self._alpha,
                "reason": reason,
                "below_minimum_trials": below_minimum,
            },
        )

    def _empty_verdict(
        self,
        threshold: float | None = None,
        note: str | None = None,
    ) -> StochasticVerdict:
        """Create an INCONCLUSIVE verdict for empty/missing data.

        Args:
            threshold: The threshold that was being tested, if any.
            note: Additional context about why data is empty.

        Returns:
            An INCONCLUSIVE StochasticVerdict with zero trials.
        """
        details: dict[str, Any] = {"verdict_type": "empty"}
        if threshold is not None:
            details["threshold"] = threshold
        if note is not None:
            details["note"] = note

        return StochasticVerdict(
            status=VerdictStatus.INCONCLUSIVE,
            confidence=self._confidence_level,
            pass_rate=0.0,
            pass_rate_ci=(0.0, 1.0),
            num_trials=0,
            num_passed=0,
            p_value=None,
            effect_size=None,
            effect_size_interpretation=None,
            regression_detected=False,
            details=details,
        )

    def __repr__(self) -> str:
        return (
            f"VerdictFunction(alpha={self._alpha}, beta={self._beta}, "
            f"min_trials={self._min_trials}, "
            f"confidence_method='{self._confidence_method}', "
            f"regression_test='{self._regression_test}')"
        )
