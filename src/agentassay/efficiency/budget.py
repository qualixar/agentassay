# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Adaptive Budget Optimization — minimum trials for maximum confidence.

THE PROBLEM (paper Section 7.2):

    Fixed-sample testing is wasteful. The default "run 100 trials" approach
    ignores a critical signal: how STABLE is the agent?

    A well-tuned search agent that consistently calls search->filter->select
    with 92% success rate does NOT need 100 trials. Its low behavioral
    variance means 20-25 trials suffice for (alpha=0.05, beta=0.10).

    Conversely, an experimental reasoning agent with high variance in tool
    choices and step counts NEEDS more trials — maybe 150 — to achieve the
    same guarantees.

    The Adaptive Budget Optimizer solves this by:
    1. Running a small calibration set (default: 10 traces)
    2. Computing behavioral variance from calibration fingerprints
    3. Calculating the optimal N for desired (alpha, beta) guarantees
    4. Returning the exact budget needed — no more, no less

THE MATH:

    Sample size for detecting a shift of magnitude delta in a
    d-dimensional multivariate distribution:

        N >= (z_alpha + z_beta)^2 * trace(Sigma) / delta^2

    where Sigma is the covariance matrix of the behavioral fingerprints.
    The trace(Sigma) = sum of variances = behavioral_variance.

    For a p-dimensional Hotelling test, the exact formula includes
    a correction factor based on the F-distribution:

        N >= (d * (n1 + n2 - 2)) / (n1 + n2 - d - 1) * critical_value

    In practice, the simpler formula provides a useful first approximation
    that we refine using the achieved-power iteration.

BUDGET ALLOCATION:

    When running multiple test types (regression, coverage, mutation,
    metamorphic), the optimizer allocates the total token budget to
    maximize information gain. This uses a simplified convex optimization:
    allocate more trials to higher-variance tests and fewer to low-variance
    ones, subject to minimum trial counts for statistical validity.

References:
    - Anderson, T.W. (2003). An Introduction to Multivariate Statistical
      Analysis. Chapter 5.
    - Fleiss, J.L., Levin, B. & Paik, M.C. (2003). Statistical Methods
      for Rates and Proportions. 3rd ed. Wiley.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as sp_stats

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.distribution import (
    FingerprintDistribution,
)
from agentassay.efficiency.fingerprint import (
    BehavioralFingerprint,
)

# ===================================================================
# BudgetEstimate — the frozen result of calibration
# ===================================================================


class BudgetEstimate(BaseModel):
    """Result of adaptive budget calibration.

    Contains the recommended trial count and supporting diagnostics.
    This is the object that tells the developer: "For your agent with
    this variance profile, run exactly N trials to get (alpha, beta)
    guarantees."

    Attributes
    ----------
    behavioral_variance : float
        Total variance of the fingerprint distribution (trace of covariance).
    recommended_n : int
        Minimum per-group trial count for the requested (alpha, beta).
    estimated_cost_usd : float
        Projected cost based on calibration per-trial cost.
    variance_classification : str
        Human-readable classification: "stable", "moderate", or "volatile".
    savings_vs_fixed_100 : float
        Percentage saved vs the default 100-trial approach.
    confidence_level : float
        The (1 - alpha) confidence level used.
    power_level : float
        The (1 - beta) power level used.
    per_trial_cost_usd : float
        Average cost per trial from calibration data.
    diagnostics : dict[str, float]
        Per-dimension variance breakdown.
    calibration_size : int
        Number of traces used for calibration.
    """

    model_config = ConfigDict(frozen=True)

    behavioral_variance: float = Field(ge=0.0)
    recommended_n: int = Field(ge=2)
    estimated_cost_usd: float = Field(ge=0.0)
    variance_classification: str
    savings_vs_fixed_100: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    power_level: float = Field(gt=0.0, lt=1.0)
    per_trial_cost_usd: float = Field(ge=0.0)
    diagnostics: dict[str, float] = Field(default_factory=dict)
    calibration_size: int = Field(ge=2)


# ===================================================================
# Variance classification thresholds
# ===================================================================

# These thresholds are empirically calibrated from the AgentAssay
# experiments (E1-E6). They partition the behavioral variance space
# into three regions that map to intuitive stability categories.
#
# The thresholds use the TRACE of the covariance matrix (sum of
# per-dimension variances). Since we have 14 dimensions, a "stable"
# agent has average per-dimension variance < 0.1 (total < 1.4),
# while a "volatile" agent has average per-dimension variance > 0.5
# (total > 7.0).

_VARIANCE_THRESHOLDS: dict[str, float] = {
    "stable": 1.5,  # total variance < 1.5
    "moderate": 5.0,  # total variance < 5.0
    # anything above 5.0 is "volatile"
}


def _classify_variance(variance: float) -> str:
    """Map total behavioral variance to a human-readable classification.

    Parameters
    ----------
    variance : float
        Total behavioral variance (trace of covariance).

    Returns
    -------
    str
        One of "stable", "moderate", or "volatile".
    """
    if variance < _VARIANCE_THRESHOLDS["stable"]:
        return "stable"
    if variance < _VARIANCE_THRESHOLDS["moderate"]:
        return "moderate"
    return "volatile"


# ===================================================================
# AdaptiveBudgetOptimizer
# ===================================================================


class AdaptiveBudgetOptimizer:
    """Computes the MINIMUM number of trials needed for (alpha, beta) guarantees.

    Instead of blindly running N=100 trials, the optimizer:
    1. Takes a small calibration set of traces (default: 10)
    2. Extracts behavioral fingerprints
    3. Computes the covariance structure of the agent's behavior
    4. Calculates the optimal N using multivariate power analysis
    5. Returns a BudgetEstimate with the exact budget needed

    For stable agents (low variance): N might be 15-25 instead of 100.
    For volatile agents (high variance): N might be 80-150.
    The variance itself is a diagnostic signal — a volatile agent may
    have underlying problems worth investigating.

    Parameters
    ----------
    alpha : float
        Type I error rate (false positive). Default 0.05.
    beta : float
        Type II error rate (false negative). Default 0.10.
    delta : float
        Minimum detectable effect size (Mahalanobis distance). Default 0.10.
        Smaller delta requires more trials.
    min_trials : int
        Floor on recommended trials (never recommend fewer). Default 10.
    max_trials : int
        Ceiling on recommended trials (never recommend more). Default 500.
    """

    __slots__ = ("_alpha", "_beta", "_delta", "_min_trials", "_max_trials")

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.10,
        delta: float = 0.10,
        min_trials: int = 10,
        max_trials: int = 500,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if min_trials < 2:
            raise ValueError(f"min_trials must be >= 2, got {min_trials}")
        if max_trials < min_trials:
            raise ValueError(f"max_trials ({max_trials}) must be >= min_trials ({min_trials})")

        self._alpha = alpha
        self._beta = beta
        self._delta = delta
        self._min_trials = min_trials
        self._max_trials = max_trials

    @property
    def alpha(self) -> float:
        """Type I error rate."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Type II error rate."""
        return self._beta

    @property
    def delta(self) -> float:
        """Minimum detectable effect size."""
        return self._delta

    def calibrate(self, traces: list[ExecutionTrace]) -> BudgetEstimate:
        """Run calibration from a set of execution traces.

        This is the main entry point. Given calibration traces (typically
        from a small exploratory run), computes the optimal trial count
        for the full test.

        Parameters
        ----------
        traces : list[ExecutionTrace]
            Calibration traces. Minimum 2 required. Recommended: 10.

        Returns
        -------
        BudgetEstimate
            Contains recommended_n, cost estimate, and diagnostics.

        Raises
        ------
        ValueError
            If fewer than 2 traces are provided.
        """
        if len(traces) < 2:
            raise ValueError(
                f"Calibration requires at least 2 traces, got {len(traces)}. "
                "Run at least 2 trials first."
            )

        # Step 1: Extract fingerprints
        fingerprints = [BehavioralFingerprint.from_trace(t) for t in traces]
        distribution = FingerprintDistribution(fingerprints)

        # Step 2: Compute behavioral variance
        bv = distribution.behavioral_variance
        per_dim = distribution.per_dimension_variance

        # Step 3: Compute optimal N
        recommended = self._compute_optimal_n(
            behavioral_variance=bv,
            dimensionality=distribution.dimensionality,
            n_calibration=len(traces),
        )

        # Step 4: Estimate per-trial cost
        costs = [t.total_cost_usd for t in traces]
        avg_cost = sum(costs) / len(costs) if costs else 0.0

        # Step 5: Compute savings
        savings = max(0.0, (1.0 - recommended / 100.0)) * 100.0

        return BudgetEstimate(
            behavioral_variance=bv,
            recommended_n=recommended,
            estimated_cost_usd=recommended * avg_cost,
            variance_classification=_classify_variance(bv),
            savings_vs_fixed_100=savings,
            confidence_level=1.0 - self._alpha,
            power_level=1.0 - self._beta,
            per_trial_cost_usd=avg_cost,
            diagnostics=per_dim,
            calibration_size=len(traces),
        )

    def calibrate_from_fingerprints(
        self,
        fingerprints: list[BehavioralFingerprint],
        per_trial_cost_usd: float = 0.0,
    ) -> BudgetEstimate:
        """Run calibration from pre-computed fingerprints.

        Use this when you already have fingerprints (e.g., from a
        TraceStore) and want to avoid re-extracting them.

        Parameters
        ----------
        fingerprints : list[BehavioralFingerprint]
            Pre-computed fingerprints. Minimum 2.
        per_trial_cost_usd : float
            Average cost per trial. Default 0.0.

        Returns
        -------
        BudgetEstimate
        """
        if len(fingerprints) < 2:
            raise ValueError(
                f"Calibration requires at least 2 fingerprints, got {len(fingerprints)}."
            )

        distribution = FingerprintDistribution(fingerprints)
        bv = distribution.behavioral_variance
        per_dim = distribution.per_dimension_variance

        recommended = self._compute_optimal_n(
            behavioral_variance=bv,
            dimensionality=distribution.dimensionality,
            n_calibration=len(fingerprints),
        )

        savings = max(0.0, (1.0 - recommended / 100.0)) * 100.0

        return BudgetEstimate(
            behavioral_variance=bv,
            recommended_n=recommended,
            estimated_cost_usd=recommended * per_trial_cost_usd,
            variance_classification=_classify_variance(bv),
            savings_vs_fixed_100=savings,
            confidence_level=1.0 - self._alpha,
            power_level=1.0 - self._beta,
            per_trial_cost_usd=per_trial_cost_usd,
            diagnostics=per_dim,
            calibration_size=len(fingerprints),
        )

    def optimal_allocation(
        self,
        budget_tokens: int,
        test_types: list[str],
        tokens_per_trial: dict[str, int] | None = None,
        variance_per_type: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Allocate a fixed token budget optimally across test types.

        Given a total token budget and a set of test types (e.g.,
        ["regression", "coverage", "mutation", "metamorphic"]), computes
        how many trials to allocate to each type to maximize total
        information gain.

        The allocation uses a variance-weighted proportional strategy:
        test types with higher variance get more trials (they need more
        data to reach stable conclusions).

        Parameters
        ----------
        budget_tokens : int
            Total token budget across all test types.
        test_types : list[str]
            Names of test types to allocate budget across.
        tokens_per_trial : dict[str, int] | None
            Average tokens per trial for each test type.
            If None, assumes equal cost across types.
        variance_per_type : dict[str, float] | None
            Behavioral variance for each test type.
            If None, assumes equal variance (uniform allocation).

        Returns
        -------
        dict[str, Any]
            Keys:
            - allocation (dict[str, int]): test_type -> recommended trials
            - total_tokens_used (int): total tokens consumed by allocation
            - utilization (float): fraction of budget used (0-1)
            - strategy (str): "variance_weighted" or "uniform"
        """
        if not test_types:
            raise ValueError("test_types must be non-empty")
        if budget_tokens <= 0:
            raise ValueError(f"budget_tokens must be > 0, got {budget_tokens}")

        n_types = len(test_types)

        # Default: equal cost per trial
        if tokens_per_trial is None:
            avg_tokens = budget_tokens // (n_types * 10)
            tokens_per_trial = {t: max(1, avg_tokens) for t in test_types}

        # Default: equal variance (uniform allocation)
        if variance_per_type is None:
            variance_per_type = {t: 1.0 for t in test_types}
            strategy = "uniform"
        else:
            strategy = "variance_weighted"

        # Compute variance-weighted allocation
        total_variance = sum(variance_per_type.get(t, 1.0) for t in test_types)
        if total_variance == 0.0:
            total_variance = float(n_types)

        allocation: dict[str, int] = {}
        total_used = 0

        for t in test_types:
            var_weight = variance_per_type.get(t, 1.0) / total_variance
            type_budget = int(budget_tokens * var_weight)
            tpt = tokens_per_trial.get(t, 1)
            if tpt <= 0:
                tpt = 1

            n_trials = max(self._min_trials, type_budget // tpt)
            n_trials = min(n_trials, self._max_trials)
            allocation[t] = n_trials
            total_used += n_trials * tpt

        utilization = total_used / budget_tokens if budget_tokens > 0 else 0.0

        return {
            "allocation": allocation,
            "total_tokens_used": total_used,
            "utilization": min(1.0, utilization),
            "strategy": strategy,
        }

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_optimal_n(
        self,
        behavioral_variance: float,
        dimensionality: int,
        n_calibration: int,
    ) -> int:
        """Compute the optimal per-group trial count.

        Uses the multivariate sample size formula for detecting a shift
        of magnitude delta in a d-dimensional space with known variance.

        The formula for Hotelling's T-squared test:

            N >= ((z_alpha + z_beta) / delta)^2 * average_variance + d + 1

        where average_variance = behavioral_variance / d is the average
        per-dimension variance. The "+ d + 1" correction accounts for the
        degrees of freedom consumed by covariance estimation in the
        multivariate case.

        The result is clamped to [min_trials, max_trials].

        Parameters
        ----------
        behavioral_variance : float
            Total variance (trace of covariance matrix).
        dimensionality : int
            Number of fingerprint dimensions.
        n_calibration : int
            Number of calibration samples (used for finite-sample correction).

        Returns
        -------
        int
            Recommended per-group trial count.
        """
        z_alpha = float(sp_stats.norm.ppf(1.0 - self._alpha))
        z_beta = float(sp_stats.norm.ppf(1.0 - self._beta))

        # Average per-dimension variance
        avg_var = behavioral_variance / max(dimensionality, 1)

        # Core formula: sample size for detecting effect size delta
        # in terms of average standard deviation
        z_sum_sq = (z_alpha + z_beta) ** 2
        n_raw = z_sum_sq * avg_var / (self._delta**2)

        # Correction for multivariate case: need extra samples to
        # estimate the covariance matrix reliably
        n_corrected = n_raw + dimensionality + 1

        # Finite-sample inflation: if calibration set is small, add
        # a safety margin. Uses the rule: inflate by sqrt(2/n_cal).
        if n_calibration < 30:
            inflation = math.sqrt(2.0 / n_calibration)
            n_corrected *= 1.0 + inflation

        n_final = math.ceil(n_corrected)

        return max(self._min_trials, min(n_final, self._max_trials))
