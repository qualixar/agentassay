# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Multi-Fidelity Proxy Testing — test expensive agents using cheap models.

THE PROBLEM (paper Section 7.4):

    Testing GPT-5.2 at $0.02/trial with 100 trials = $2.00 per scenario.
    Testing Claude Haiku at $0.001/trial with 100 trials = $0.10 per scenario.

    If the agent's behavioral pattern is similar across model tiers —
    meaning a regression in the GPT-5.2 version is also visible in the
    Haiku version — we can:
    1. Run many trials on the cheap proxy (Haiku)
    2. Run few trials on the expensive target (GPT-5.2)
    3. Combine evidence from both for a single verdict

    This is multi-fidelity testing: use low-fidelity (cheap) experiments
    to guide and augment high-fidelity (expensive) ones.

THE METHOD:

    1. **Correlation estimation:** Run a small calibration set on both
       proxy and target. Compute the Pearson correlation between their
       behavioral fingerprint vectors. If correlation > threshold (0.6),
       multi-fidelity is valid.

    2. **Optimal split:** Given a fixed budget, compute the optimal
       allocation between proxy and target trials that minimizes the
       expected error of the combined test.

    3. **Evidence combination:** Use Fisher's method (or Stouffer's
       weighted Z-method) to combine p-values from proxy and target
       regression tests. Weight by estimated correlation.

    The theoretical justification comes from meta-analysis: combining
    evidence from multiple studies (here: fidelity levels) gives higher
    power than any single study alone, provided the correlation is
    sufficient.

REQUIREMENTS:

    Multi-fidelity is only valid when proxy and target agents use the
    same tools and workflow structure. A cheap model running a completely
    different strategy provides no useful transfer signal.

    The ``estimate_correlation`` method verifies this before proceeding.

References:
    - Fisher, R.A. (1925). Statistical Methods for Research Workers.
      Section 21.1: Combining independent tests.
    - Stouffer, S.A. et al. (1949). The American Soldier, Vol. 1.
    - Peherstorfer, B. et al. (2018). "Survey of Multifidelity Methods
      in Uncertainty Quantification, Inference, and Optimization."
      SIAM Review 60(3): 550-591.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.fingerprint import (
    BehavioralFingerprint,
)
from agentassay.efficiency.distribution import (
    FingerprintDistribution,
)


# ===================================================================
# MultiFidelityTester
# ===================================================================


class MultiFidelityTester:
    """Test expensive agents using cheap models as proxies.

    The core idea: if a behavioral regression in GPT-5.2 is also visible
    in Haiku, we can run most trials on Haiku and only a few on GPT-5.2.
    The combined evidence is stronger (higher power) than running either
    alone with the same budget.

    Parameters
    ----------
    proxy_model : str
        Name of the cheap proxy model (e.g., "claude-haiku-3.5").
    target_model : str
        Name of the expensive target model (e.g., "gpt-5.2").
    correlation_threshold : float
        Minimum required correlation between proxy and target fingerprints.
        Below this threshold, multi-fidelity is rejected as unreliable.
        Default 0.6 (moderate correlation).

    Example
    -------
    >>> tester = MultiFidelityTester(
    ...     proxy_model="claude-haiku-3.5",
    ...     target_model="gpt-5.2",
    ... )
    >>> correlation = tester.estimate_correlation(proxy_traces, target_traces)
    >>> if correlation >= 0.6:
    ...     result = tester.combined_verdict(
    ...         proxy_baseline, proxy_candidate,
    ...         target_baseline, target_candidate,
    ...     )
    """

    __slots__ = ("_proxy_model", "_target_model", "_corr_threshold", "_estimated_corr")

    def __init__(
        self,
        proxy_model: str,
        target_model: str,
        correlation_threshold: float = 0.6,
    ) -> None:
        if not proxy_model:
            raise ValueError("proxy_model must be non-empty")
        if not target_model:
            raise ValueError("target_model must be non-empty")
        if not (0.0 < correlation_threshold <= 1.0):
            raise ValueError(
                f"correlation_threshold must be in (0, 1], got {correlation_threshold}"
            )

        self._proxy_model = proxy_model
        self._target_model = target_model
        self._corr_threshold = correlation_threshold
        self._estimated_corr: float | None = None

    @property
    def proxy_model(self) -> str:
        """Name of the cheap proxy model."""
        return self._proxy_model

    @property
    def target_model(self) -> str:
        """Name of the expensive target model."""
        return self._target_model

    @property
    def estimated_correlation(self) -> float | None:
        """Last estimated correlation, or None if not yet computed."""
        return self._estimated_corr

    # ------------------------------------------------------------------
    # Correlation estimation
    # ------------------------------------------------------------------

    def estimate_correlation(
        self,
        proxy_traces: list[ExecutionTrace],
        target_traces: list[ExecutionTrace],
    ) -> float:
        """Estimate behavioral correlation between proxy and target.

        Uses behavioral fingerprints (not raw outputs) to compute the
        Pearson correlation between the proxy and target mean vectors.

        For this to be meaningful, both trace sets should come from the
        same scenarios — so that we are comparing behavioral patterns
        on the SAME tasks.

        Parameters
        ----------
        proxy_traces : list[ExecutionTrace]
            Traces from running the proxy model. Minimum 2.
        target_traces : list[ExecutionTrace]
            Traces from running the target model. Minimum 2.

        Returns
        -------
        float
            Pearson correlation in [-1, 1]. Higher means more similar
            behavioral patterns.

        Raises
        ------
        ValueError
            If either trace list has fewer than 2 elements.
        """
        if len(proxy_traces) < 2:
            raise ValueError(
                f"Need at least 2 proxy traces, got {len(proxy_traces)}"
            )
        if len(target_traces) < 2:
            raise ValueError(
                f"Need at least 2 target traces, got {len(target_traces)}"
            )

        # Extract fingerprint vectors
        proxy_fps = [BehavioralFingerprint.from_trace(t) for t in proxy_traces]
        target_fps = [BehavioralFingerprint.from_trace(t) for t in target_traces]

        proxy_dist = FingerprintDistribution(proxy_fps)
        target_dist = FingerprintDistribution(target_fps)

        # Compute Pearson correlation of mean vectors
        proxy_mean = proxy_dist.mean_vector
        target_mean = target_dist.mean_vector

        # Correlation between the mean fingerprint vectors
        # This measures how similarly the two models approach the task.
        correlation = float(np.corrcoef(proxy_mean, target_mean)[0, 1])

        # Handle NaN (can happen if one vector is constant)
        if np.isnan(correlation):
            correlation = 0.0

        self._estimated_corr = correlation
        return correlation

    # ------------------------------------------------------------------
    # Combined verdict
    # ------------------------------------------------------------------

    def combined_verdict(
        self,
        proxy_baseline: list[ExecutionTrace],
        proxy_candidate: list[ExecutionTrace],
        target_baseline: list[ExecutionTrace],
        target_candidate: list[ExecutionTrace],
        alpha: float = 0.05,
        method: str = "stouffer",
    ) -> dict[str, Any]:
        """Combine evidence from proxy and target for a single verdict.

        Runs separate regression tests on proxy and target fingerprints,
        then combines the p-values using either Fisher's method or
        Stouffer's weighted Z-method.

        The weight assigned to the proxy evidence is proportional to the
        estimated correlation — if the proxy has low correlation with the
        target, its evidence gets less weight.

        Parameters
        ----------
        proxy_baseline : list[ExecutionTrace]
            Baseline traces from proxy model.
        proxy_candidate : list[ExecutionTrace]
            Candidate traces from proxy model.
        target_baseline : list[ExecutionTrace]
            Baseline traces from target model.
        target_candidate : list[ExecutionTrace]
            Candidate traces from target model.
        alpha : float
            Significance level. Default 0.05.
        method : str
            Combination method: "fisher" or "stouffer". Default "stouffer".

        Returns
        -------
        dict[str, Any]
            Keys:
            - regression_detected (bool): combined verdict
            - combined_p_value (float): combined p-value
            - proxy_p_value (float): p-value from proxy test alone
            - target_p_value (float): p-value from target test alone
            - proxy_distance (float): Mahalanobis distance from proxy
            - target_distance (float): Mahalanobis distance from target
            - correlation (float): estimated proxy-target correlation
            - correlation_sufficient (bool): whether correlation exceeds threshold
            - method (str): combination method used
            - cost_savings_estimate (float): estimated % saved vs target-only

        Raises
        ------
        ValueError
            If any trace list has fewer than 2 elements.
        """
        # Validate inputs
        for name, traces in [
            ("proxy_baseline", proxy_baseline),
            ("proxy_candidate", proxy_candidate),
            ("target_baseline", target_baseline),
            ("target_candidate", target_candidate),
        ]:
            if len(traces) < 2:
                raise ValueError(f"Need at least 2 traces for {name}, got {len(traces)}")

        # Estimate correlation if not already done
        if self._estimated_corr is None:
            self.estimate_correlation(proxy_baseline, target_baseline)

        correlation = self._estimated_corr or 0.0
        corr_sufficient = abs(correlation) >= self._corr_threshold

        # Run fingerprint regression tests separately
        proxy_result = _fingerprint_test(proxy_baseline, proxy_candidate, alpha)
        target_result = _fingerprint_test(target_baseline, target_candidate, alpha)

        proxy_p = proxy_result["p_value"]
        target_p = target_result["p_value"]

        # Combine p-values
        if method == "fisher":
            combined_p = _fisher_combine(proxy_p, target_p)
        elif method == "stouffer":
            # Weight by correlation: target gets full weight,
            # proxy gets weight proportional to |correlation|
            proxy_weight = abs(correlation) if corr_sufficient else 0.0
            target_weight = 1.0
            combined_p = _stouffer_combine(
                [proxy_p, target_p],
                [proxy_weight, target_weight],
            )
        else:
            raise ValueError(f"Unknown combination method: {method}. Use 'fisher' or 'stouffer'.")

        # Cost savings estimate
        proxy_n = len(proxy_baseline) + len(proxy_candidate)
        target_n = len(target_baseline) + len(target_candidate)
        total_n = proxy_n + target_n
        # Compare to doing everything on target
        target_only_n = total_n  # hypothetical target-only count
        # Rough estimate: proxy is ~10x cheaper
        effective_cost = proxy_n * 0.1 + target_n
        savings = max(0.0, (1.0 - effective_cost / target_only_n)) * 100.0

        return {
            "regression_detected": combined_p < alpha,
            "combined_p_value": combined_p,
            "proxy_p_value": proxy_p,
            "target_p_value": target_p,
            "proxy_distance": proxy_result["distance"],
            "target_distance": target_result["distance"],
            "correlation": correlation,
            "correlation_sufficient": corr_sufficient,
            "method": method,
            "cost_savings_estimate": savings,
            "proxy_changed_dimensions": proxy_result.get("changed_dimensions", []),
            "target_changed_dimensions": target_result.get("changed_dimensions", []),
        }

    # ------------------------------------------------------------------
    # Optimal budget split
    # ------------------------------------------------------------------

    def optimal_split(
        self,
        total_budget: float,
        proxy_cost: float,
        target_cost: float,
        correlation: float | None = None,
    ) -> tuple[int, int]:
        """Compute optimal split of budget between proxy and target trials.

        Minimizes the expected error of the combined test subject to the
        budget constraint.

        The optimal allocation gives more trials to the cheaper model,
        weighted by the correlation. With perfect correlation (r=1.0),
        almost all budget goes to the proxy. With low correlation,
        more budget goes to the target.

        The derivation follows from minimizing the variance of the
        combined estimator under a linear budget constraint:

            min   Var(theta_combined)
            s.t.  n_proxy * c_proxy + n_target * c_target <= Budget

        Solution:
            n_proxy / n_target = (r * sqrt(c_target)) / sqrt(c_proxy)

        where r is the correlation and c_x is the per-trial cost.

        Parameters
        ----------
        total_budget : float
            Total budget in USD.
        proxy_cost : float
            Cost per proxy trial in USD.
        target_cost : float
            Cost per target trial in USD.
        correlation : float | None
            Override the estimated correlation. If None, uses the last
            estimate from ``estimate_correlation``.

        Returns
        -------
        tuple[int, int]
            (n_proxy, n_target) — recommended trial counts.

        Raises
        ------
        ValueError
            If budget or costs are non-positive, or no correlation available.
        """
        if total_budget <= 0:
            raise ValueError(f"total_budget must be > 0, got {total_budget}")
        if proxy_cost <= 0:
            raise ValueError(f"proxy_cost must be > 0, got {proxy_cost}")
        if target_cost <= 0:
            raise ValueError(f"target_cost must be > 0, got {target_cost}")

        r = correlation if correlation is not None else self._estimated_corr
        if r is None:
            raise ValueError(
                "No correlation available. Call estimate_correlation first, "
                "or pass correlation explicitly."
            )

        # Effective correlation (use absolute value)
        r_eff = abs(r)

        if r_eff < 0.01:
            # Essentially zero correlation — put everything on target
            n_target = max(2, int(total_budget / target_cost))
            return (0, n_target)

        # Optimal ratio: n_proxy / n_target = r * sqrt(c_target / c_proxy)
        ratio = r_eff * math.sqrt(target_cost / proxy_cost)

        # Solve: n_proxy * c_proxy + n_target * c_target = budget
        # with n_proxy = ratio * n_target
        # => n_target * (ratio * c_proxy + c_target) = budget
        n_target_float = total_budget / (ratio * proxy_cost + target_cost)
        n_target = max(2, int(n_target_float))

        n_proxy_float = ratio * n_target
        n_proxy = max(2, int(n_proxy_float))

        # Verify budget constraint (may need to reduce)
        while (n_proxy * proxy_cost + n_target * target_cost) > total_budget * 1.1:
            if n_proxy > n_target:
                n_proxy -= 1
            else:
                n_target -= 1
            if n_proxy < 2 or n_target < 2:
                break

        return (max(2, n_proxy), max(2, n_target))


# ===================================================================
# Internal helper functions
# ===================================================================


def _fingerprint_test(
    baseline: list[ExecutionTrace],
    candidate: list[ExecutionTrace],
    alpha: float,
) -> dict[str, Any]:
    """Run a fingerprint regression test (internal helper).

    Parameters
    ----------
    baseline : list[ExecutionTrace]
    candidate : list[ExecutionTrace]
    alpha : float

    Returns
    -------
    dict
        Same structure as FingerprintDistribution.regression_test.
    """
    baseline_fps = [BehavioralFingerprint.from_trace(t) for t in baseline]
    candidate_fps = [BehavioralFingerprint.from_trace(t) for t in candidate]

    baseline_dist = FingerprintDistribution(baseline_fps)
    candidate_dist = FingerprintDistribution(candidate_fps)

    return baseline_dist.regression_test(candidate_dist, alpha=alpha)


def _fisher_combine(p1: float, p2: float) -> float:
    """Combine two p-values using Fisher's method.

    Fisher's combined test statistic:
        X^2 = -2 * (ln(p1) + ln(p2))

    Under H0 (both nulls are true), X^2 follows a chi-squared distribution
    with 2k degrees of freedom (k=2 studies).

    Parameters
    ----------
    p1, p2 : float
        Individual p-values. Must be in (0, 1].

    Returns
    -------
    float
        Combined p-value.
    """
    # Clamp to avoid log(0)
    p1 = max(p1, 1e-300)
    p2 = max(p2, 1e-300)

    chi2_stat = -2.0 * (math.log(p1) + math.log(p2))
    combined_p = float(1.0 - sp_stats.chi2.cdf(chi2_stat, df=4))  # df = 2k = 4

    return combined_p


def _stouffer_combine(
    p_values: list[float],
    weights: list[float],
) -> float:
    """Combine p-values using Stouffer's weighted Z-method.

    Each p-value is converted to a Z-score, weighted, and combined:
        Z_combined = sum(w_i * Z_i) / sqrt(sum(w_i^2))

    This method allows differential weighting of evidence sources.
    In multi-fidelity testing, the proxy gets lower weight when
    correlation is imperfect.

    Parameters
    ----------
    p_values : list[float]
        Individual p-values.
    weights : list[float]
        Non-negative weights for each p-value.

    Returns
    -------
    float
        Combined p-value.
    """
    if len(p_values) != len(weights):
        raise ValueError(
            f"p_values and weights must have same length: "
            f"{len(p_values)} vs {len(weights)}"
        )

    # Convert p-values to Z-scores
    z_scores = []
    valid_weights = []
    for p, w in zip(p_values, weights):
        if w <= 0.0:
            continue
        # Clamp p to valid range for inverse normal
        p_clamped = max(min(p, 1.0 - 1e-15), 1e-15)
        z = float(sp_stats.norm.ppf(1.0 - p_clamped))
        z_scores.append(z)
        valid_weights.append(w)

    if not z_scores:
        return 1.0  # No valid evidence

    # Weighted combination
    w_arr = np.array(valid_weights)
    z_arr = np.array(z_scores)

    z_combined = float(np.sum(w_arr * z_arr) / np.sqrt(np.sum(w_arr**2)))

    # Convert back to p-value (one-sided)
    combined_p = float(1.0 - sp_stats.norm.cdf(z_combined))

    return combined_p
