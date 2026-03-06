# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Fingerprint Distribution — statistical operations on fingerprint collections.

Provides FingerprintDistribution: multivariate statistics (mean, covariance,
Mahalanobis distance, Hotelling's T-squared) over collections of
BehavioralFingerprints for regression detection.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from agentassay.efficiency.fingerprint import (
    _DIMENSION_NAMES,
    BehavioralFingerprint,
)

# ===================================================================
# FingerprintDistribution — statistics over multiple fingerprints
# ===================================================================


class FingerprintDistribution:
    """Statistical distribution over a collection of BehavioralFingerprints.

    Computes mean vector, covariance matrix, behavioral variance, Mahalanobis
    distance, and Hotelling's T-squared test for regression detection.
    Requires at least 2 fingerprints.
    """

    __slots__ = ("_fingerprints", "_matrix", "_mean", "_cov", "_n", "_d")

    def __init__(self, fingerprints: list[BehavioralFingerprint]) -> None:
        if len(fingerprints) < 2:
            raise ValueError(
                f"FingerprintDistribution requires at least 2 fingerprints, "
                f"got {len(fingerprints)}. Run more trials."
            )

        self._fingerprints = fingerprints
        self._n = len(fingerprints)
        self._d = BehavioralFingerprint.vector_dimension()

        # Build the n x d data matrix
        self._matrix = np.array([fp.to_vector() for fp in fingerprints], dtype=np.float64)

        # Mean vector and covariance matrix
        self._mean = np.mean(self._matrix, axis=0)
        # Use ddof=1 for unbiased sample covariance (Bessel's correction).
        # Add small ridge for numerical stability when dimensions are
        # nearly collinear.
        self._cov = np.cov(self._matrix, rowvar=False, ddof=1)
        # Regularize: add epsilon * I to prevent singular covariance
        self._cov += np.eye(self._d) * 1e-8

    @property
    def n_samples(self) -> int:
        """Number of fingerprints in this distribution."""
        return self._n

    @property
    def dimensionality(self) -> int:
        """Dimensionality of the fingerprint vector."""
        return self._d

    @property
    def mean_vector(self) -> np.ndarray:
        """Mean fingerprint vector, shape (d,) float64."""
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Sample covariance matrix, shape (d, d) positive semi-definite."""
        return self._cov.copy()

    @property
    def behavioral_variance(self) -> float:
        """Total variance (trace of covariance matrix). Low = stable agent."""
        return float(np.trace(self._cov))

    @property
    def per_dimension_variance(self) -> dict[str, float]:
        """Variance for each fingerprint dimension individually."""
        diag = np.diag(self._cov)
        return {name: float(diag[i]) for i, name in enumerate(_DIMENSION_NAMES)}

    def distance_to(self, other: FingerprintDistribution) -> float:
        """Mahalanobis distance between two fingerprint distributions."""
        pooled_cov = _pooled_covariance(self, other)
        diff = self._mean - other._mean

        try:
            inv_cov = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            # Singular covariance — use pseudoinverse
            inv_cov = np.linalg.pinv(pooled_cov)

        maha_sq = float(diff @ inv_cov @ diff)
        return math.sqrt(max(0.0, maha_sq))

    def regression_test(
        self,
        other: FingerprintDistribution,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Hotelling's T-squared test for multivariate regression detection.

        Tests H0: mu_baseline = mu_candidate against H1: mu_baseline != mu_candidate.
        Falls back to per-dimension Bonferroni-corrected t-tests when sample size
        is insufficient for full-rank Hotelling.

        Returns dict with keys: regression_detected, p_value, t_squared,
        f_statistic, distance, changed_dimensions, confidence, df1, df2.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        n1 = self._n
        n2 = other._n
        p = self._d

        # Check if we have enough samples for the test
        if n1 + n2 - 2 < p:
            # Not enough samples for full-rank pooled covariance.
            # Fall back to dimension-wise t-tests with Bonferroni correction.
            return self._dimension_wise_fallback(other, alpha)

        pooled_cov = _pooled_covariance(self, other)
        diff = self._mean - other._mean

        try:
            inv_cov = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(pooled_cov)

        # Hotelling's T-squared
        scaling = (n1 * n2) / (n1 + n2)
        t_squared = float(scaling * diff @ inv_cov @ diff)

        # Convert to F-statistic
        df1 = p
        df2 = n1 + n2 - p - 1
        if df2 <= 0:
            return self._dimension_wise_fallback(other, alpha)

        f_stat = t_squared * df2 / ((n1 + n2 - 2) * p)
        p_value = float(1.0 - sp_stats.f.cdf(f_stat, df1, df2))

        # Distance
        maha_dist = math.sqrt(max(0.0, float(diff @ inv_cov @ diff)))

        # Identify which dimensions changed the most
        changed = _identify_changed_dimensions(self, other, alpha)

        return {
            "regression_detected": p_value < alpha,
            "p_value": p_value,
            "t_squared": t_squared,
            "f_statistic": f_stat,
            "distance": maha_dist,
            "changed_dimensions": changed,
            "confidence": 1.0 - p_value,
            "df1": df1,
            "df2": df2,
        }

    def _dimension_wise_fallback(
        self,
        other: FingerprintDistribution,
        alpha: float,
    ) -> dict[str, Any]:
        """Bonferroni-corrected per-dimension t-test fallback for small samples."""
        p = self._d
        bonferroni_alpha = alpha / p
        min_p = 1.0
        changed: list[str] = []

        for i in range(p):
            baseline_vals = self._matrix[:, i]
            candidate_vals = other._matrix[:, i]

            _, p_val = sp_stats.ttest_ind(baseline_vals, candidate_vals, equal_var=False)
            p_val = float(p_val)

            if p_val < min_p:
                min_p = p_val

            if p_val < bonferroni_alpha:
                changed.append(_DIMENSION_NAMES[i])

        # Use the minimum p-value adjusted by Bonferroni
        adjusted_p = min(min_p * p, 1.0)

        diff = self._mean - other._mean
        euclidean_dist = float(np.linalg.norm(diff))

        return {
            "regression_detected": adjusted_p < alpha,
            "p_value": adjusted_p,
            "t_squared": float("nan"),
            "f_statistic": float("nan"),
            "distance": euclidean_dist,
            "changed_dimensions": changed,
            "confidence": 1.0 - adjusted_p,
            "df1": p,
            "df2": 0,
        }


# ===================================================================
# Internal helper functions for distribution operations
# ===================================================================


def _pooled_covariance(
    dist1: FingerprintDistribution,
    dist2: FingerprintDistribution,
) -> np.ndarray:
    """Pooled covariance: ((n1-1)*S1 + (n2-1)*S2) / (n1 + n2 - 2)."""
    n1 = dist1.n_samples
    n2 = dist2.n_samples
    s1 = dist1.covariance
    s2 = dist2.covariance

    denom = n1 + n2 - 2
    if denom <= 0:
        # Degenerate: return average
        return (s1 + s2) / 2.0

    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / denom
    return pooled


def _identify_changed_dimensions(
    baseline: FingerprintDistribution,
    candidate: FingerprintDistribution,
    alpha: float,
) -> list[str]:
    """Per-dimension Bonferroni-corrected t-tests to find changed dimensions."""
    p = baseline.dimensionality
    bonferroni_alpha = alpha / p
    changed: list[str] = []

    for i in range(p):
        baseline_vals = baseline._matrix[:, i]
        candidate_vals = candidate._matrix[:, i]

        # Welch's t-test (unequal variance assumed)
        _, p_val = sp_stats.ttest_ind(baseline_vals, candidate_vals, equal_var=False)

        if p_val < bonferroni_alpha:
            changed.append(_DIMENSION_NAMES[i])

    return changed
