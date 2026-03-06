# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Fingerprint Regression Detection — high-level API for behavioral regression testing.

This module provides the top-level ``fingerprint_regression_test`` function,
which is the recommended entry point for detecting behavioral regressions
between two versions of an agent. It orchestrates the full pipeline:

1. Extract behavioral fingerprints from raw execution traces
2. Build fingerprint distributions for baseline and candidate
3. Run Hotelling's T-squared test (or Bonferroni fallback)
4. Return a rich result with p-value, distance, and changed dimensions

Usage::

    from agentassay.efficiency.regression import fingerprint_regression_test

    result = fingerprint_regression_test(baseline_traces, candidate_traces)
    if result["regression_detected"]:
        print(f"Regression in: {result['changed_dimensions']}")
"""

from __future__ import annotations

from typing import Any

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.distribution import FingerprintDistribution
from agentassay.efficiency.fingerprint import BehavioralFingerprint


def fingerprint_regression_test(
    baseline_traces: list[ExecutionTrace],
    candidate_traces: list[ExecutionTrace],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Detect behavioral regression from raw execution traces.

    This is the recommended high-level API. Given baseline and candidate
    traces, it:
    1. Extracts behavioral fingerprints from all traces
    2. Builds fingerprint distributions for both versions
    3. Runs Hotelling's T-squared test
    4. Returns a rich result with p-value, distance, and changed dimensions

    Parameters
    ----------
    baseline_traces : list[ExecutionTrace]
        Traces from the baseline (known-good) version. Minimum 2.
    candidate_traces : list[ExecutionTrace]
        Traces from the candidate (new) version. Minimum 2.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        Keys:
        - regression_detected (bool)
        - p_value (float)
        - distance (float): Mahalanobis distance
        - changed_dimensions (list[str]): which behavioral aspects changed
        - confidence (float): 1 - p_value
        - baseline_variance (float): behavioral variance of baseline
        - candidate_variance (float): behavioral variance of candidate
        - baseline_n (int): number of baseline traces
        - candidate_n (int): number of candidate traces

    Raises
    ------
    ValueError
        If either trace list has fewer than 2 elements.
    """
    if len(baseline_traces) < 2:
        raise ValueError(f"Need at least 2 baseline traces, got {len(baseline_traces)}")
    if len(candidate_traces) < 2:
        raise ValueError(f"Need at least 2 candidate traces, got {len(candidate_traces)}")

    baseline_fps = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
    candidate_fps = [BehavioralFingerprint.from_trace(t) for t in candidate_traces]

    baseline_dist = FingerprintDistribution(baseline_fps)
    candidate_dist = FingerprintDistribution(candidate_fps)

    result = baseline_dist.regression_test(candidate_dist, alpha=alpha)
    result["baseline_variance"] = baseline_dist.behavioral_variance
    result["candidate_variance"] = candidate_dist.behavioral_variance
    result["baseline_n"] = len(baseline_traces)
    result["candidate_n"] = len(candidate_traces)

    return result
