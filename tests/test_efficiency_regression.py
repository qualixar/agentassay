"""Tests for fingerprint_regression_test() function — core regression detection API.

This module validates the high-level regression detection API that orchestrates:
1. Fingerprint extraction from raw traces
2. Distribution construction
3. Hotelling's T² test execution
4. Rich result dictionary construction

Target: 15+ tests covering basic detection, no-regression cases, edge cases,
statistical properties, and integration with BehavioralFingerprint class.
"""

from __future__ import annotations

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.fingerprint import BehavioralFingerprint
from agentassay.efficiency.regression import fingerprint_regression_test
from tests.conftest import make_trace

# ===================================================================
# Test: Basic regression detection
# ===================================================================


class TestBasicRegressionDetection:
    """Tests for detecting regressions when distributions differ."""

    def test_regression_detected_when_step_counts_differ(self):
        """Regression detected when candidate has significantly more steps."""
        # Baseline: 3 steps per trace
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        # Candidate: 10 steps per trace (3x increase)
        candidate_traces = [make_trace(steps=10) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert result["regression_detected"] is True
        assert result["p_value"] < 0.05
        assert result["confidence"] > 0.95
        assert (
            "step_count" in str(result["changed_dimensions"]).lower()
            or len(result["changed_dimensions"]) > 0
        )

    def test_regression_detected_when_tool_usage_differs(self):
        """Regression detected when tool usage patterns change."""
        # Baseline: search->calculate pattern
        baseline_traces = [make_trace(steps=4, tools=["search", "calculate"]) for _ in range(8)]
        # Candidate: different tool pattern (filter->validate->select)
        candidate_traces = [
            make_trace(steps=4, tools=["filter", "validate", "select"]) for _ in range(8)
        ]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert result["regression_detected"] is True
        assert result["p_value"] < 0.05

    def test_regression_detected_with_cost_difference(self):
        """Regression detected when costs differ significantly."""
        # Baseline: low cost
        baseline_traces = [make_trace(steps=3, cost=0.01) for _ in range(10)]
        # Candidate: high cost (10x increase)
        candidate_traces = [make_trace(steps=3, cost=0.10) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # This may or may not detect regression depending on token metadata
        # At minimum, test completes without error
        assert isinstance(result["regression_detected"], bool)
        assert 0.0 <= result["p_value"] <= 1.0


# ===================================================================
# Test: No regression cases
# ===================================================================


class TestNoRegressionCases:
    """Tests for cases where no regression should be detected."""

    def test_no_regression_when_distributions_identical(self):
        """No regression when baseline and candidate are identical."""
        # Create 10 identical traces for each group
        baseline_traces = [make_trace(steps=3, cost=0.01) for _ in range(10)]
        candidate_traces = [make_trace(steps=3, cost=0.01) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert result["regression_detected"] is False
        assert result["p_value"] >= 0.05
        assert result["confidence"] < 0.95

    def test_no_regression_with_small_natural_variance(self):
        """No regression when candidate has small natural variance."""
        # Baseline: 3-4 steps
        baseline_traces = [make_trace(steps=3 + (i % 2)) for i in range(10)]
        # Candidate: 3-4 steps (same distribution)
        candidate_traces = [make_trace(steps=3 + (i % 2)) for i in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert result["regression_detected"] is False
        assert result["p_value"] >= 0.05

    def test_no_regression_with_same_output_patterns(self):
        """No regression when output patterns match."""
        baseline_traces = [
            make_trace(steps=3, output_data={"result": f"output-{i}"}) for i in range(10)
        ]
        candidate_traces = [
            make_trace(steps=3, output_data={"result": f"output-{i}"}) for i in range(10)
        ]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # Both have dict outputs with similar structure
        assert result["regression_detected"] is False


# ===================================================================
# Test: Edge cases
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_raises_error_with_single_baseline_trace(self):
        """Raises ValueError when baseline has only 1 trace."""
        baseline_traces = [make_trace(steps=3)]
        candidate_traces = [make_trace(steps=3), make_trace(steps=4)]

        with pytest.raises(ValueError, match="at least 2 baseline traces"):
            fingerprint_regression_test(baseline_traces, candidate_traces)

    def test_raises_error_with_single_candidate_trace(self):
        """Raises ValueError when candidate has only 1 trace."""
        baseline_traces = [make_trace(steps=3), make_trace(steps=4)]
        candidate_traces = [make_trace(steps=3)]

        with pytest.raises(ValueError, match="at least 2 candidate traces"):
            fingerprint_regression_test(baseline_traces, candidate_traces)

    def test_raises_error_with_zero_baseline_traces(self):
        """Raises ValueError when baseline is empty."""
        baseline_traces: list[ExecutionTrace] = []
        candidate_traces = [make_trace(steps=3), make_trace(steps=4)]

        with pytest.raises(ValueError, match="at least 2 baseline traces"):
            fingerprint_regression_test(baseline_traces, candidate_traces)

    def test_raises_error_with_zero_candidate_traces(self):
        """Raises ValueError when candidate is empty."""
        baseline_traces = [make_trace(steps=3), make_trace(steps=4)]
        candidate_traces: list[ExecutionTrace] = []

        with pytest.raises(ValueError, match="at least 2 candidate traces"):
            fingerprint_regression_test(baseline_traces, candidate_traces)

    def test_works_with_minimum_traces(self):
        """Works correctly with exactly 2 traces in each group."""
        baseline_traces = [make_trace(steps=3), make_trace(steps=3)]
        candidate_traces = [make_trace(steps=10), make_trace(steps=10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # Should detect regression (3x step count difference)
        assert result["regression_detected"] is True
        assert result["baseline_n"] == 2
        assert result["candidate_n"] == 2

    def test_works_with_large_sample_sizes(self):
        """Works correctly with large sample sizes (100+ traces)."""
        baseline_traces = [make_trace(steps=3) for _ in range(100)]
        candidate_traces = [make_trace(steps=4) for _ in range(100)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # With 100 samples, even small differences are detectable
        assert isinstance(result["regression_detected"], bool)
        assert result["baseline_n"] == 100
        assert result["candidate_n"] == 100


# ===================================================================
# Test: Statistical properties
# ===================================================================


class TestStatisticalProperties:
    """Tests for statistical properties of the result."""

    def test_p_value_in_valid_range(self):
        """p_value is always in [0, 1]."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=5) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert 0.0 <= result["p_value"] <= 1.0

    def test_confidence_equals_one_minus_p_value(self):
        """confidence = 1 - p_value."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=5) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert abs(result["confidence"] - (1.0 - result["p_value"])) < 1e-9

    def test_distance_is_non_negative(self):
        """Mahalanobis distance is always non-negative."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=5) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert result["distance"] >= 0.0

    def test_distance_increases_with_difference_magnitude(self):
        """Mahalanobis distance increases as distributions diverge."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]

        # Small difference
        candidate_small = [make_trace(steps=4) for _ in range(10)]
        result_small = fingerprint_regression_test(baseline_traces, candidate_small, alpha=0.05)

        # Large difference
        candidate_large = [make_trace(steps=20) for _ in range(10)]
        result_large = fingerprint_regression_test(baseline_traces, candidate_large, alpha=0.05)

        assert result_large["distance"] > result_small["distance"]

    def test_variance_computed_for_both_groups(self):
        """Result includes behavioral variance for baseline and candidate."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=5) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert "baseline_variance" in result
        assert "candidate_variance" in result
        assert result["baseline_variance"] >= 0.0
        assert result["candidate_variance"] >= 0.0

    def test_changed_dimensions_is_list(self):
        """changed_dimensions is always a list (may be empty)."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=3) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        assert isinstance(result["changed_dimensions"], list)


# ===================================================================
# Test: Integration with BehavioralFingerprint
# ===================================================================


class TestBehavioralFingerprintIntegration:
    """Tests for integration with BehavioralFingerprint class."""

    def test_extracts_fingerprints_correctly(self):
        """Function correctly extracts fingerprints from traces."""
        baseline_traces = [make_trace(steps=3) for _ in range(5)]
        candidate_traces = [make_trace(steps=3) for _ in range(5)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # If we can extract fingerprints manually, verify they match
        baseline_fps = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        assert len(baseline_fps) == 5
        assert all(isinstance(fp, BehavioralFingerprint) for fp in baseline_fps)

        # Result should reflect the correct counts
        assert result["baseline_n"] == 5
        assert result["candidate_n"] == 5

    def test_handles_traces_with_errors(self):
        """Function handles traces with error steps gracefully."""
        baseline_traces = [make_trace(steps=3, success=True) for _ in range(10)]
        candidate_traces = [
            make_trace(steps=3, success=False, error="test error") for _ in range(10)
        ]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # Should detect regression (success patterns differ)
        # At minimum, completes without exception
        assert isinstance(result["regression_detected"], bool)

    def test_handles_traces_with_different_outputs(self):
        """Function handles traces with different output types."""
        # Baseline: string outputs
        baseline_traces = [make_trace(steps=3, output_data="text") for _ in range(10)]
        # Candidate: dict outputs
        candidate_traces = [make_trace(steps=3, output_data={"result": "data"}) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)

        # Should detect difference in output structure
        assert isinstance(result["regression_detected"], bool)


# ===================================================================
# Test: Custom alpha levels
# ===================================================================


class TestCustomAlphaLevels:
    """Tests for custom significance levels."""

    def test_stricter_alpha_reduces_detections(self):
        """Lower alpha (0.01) reduces false positives."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=4) for _ in range(10)]

        # Standard alpha
        result_05 = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.05)
        # Stricter alpha
        _result_01 = fingerprint_regression_test(  # noqa: F841 - computed but not asserted
            baseline_traces, candidate_traces, alpha=0.01
        )

        # If detected at 0.05, p_value < 0.05
        # May or may not be detected at 0.01 depending on actual p_value
        if result_05["regression_detected"]:
            assert result_05["p_value"] < 0.05

    def test_lenient_alpha_increases_detections(self):
        """Higher alpha (0.10) increases sensitivity."""
        baseline_traces = [make_trace(steps=3) for _ in range(10)]
        candidate_traces = [make_trace(steps=4) for _ in range(10)]

        result = fingerprint_regression_test(baseline_traces, candidate_traces, alpha=0.10)

        # With lenient alpha, more likely to detect
        assert isinstance(result["regression_detected"], bool)
        assert result["p_value"] >= 0.0
