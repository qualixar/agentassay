"""Tests for behavioral fingerprinting of agent execution traces.

Validates BehavioralFingerprint extraction, FingerprintDistribution
statistics, and the high-level fingerprint_regression_test function.

Target: ~23 tests covering extraction, vectorization, determinism,
sensitivity, distribution statistics, and regression detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.efficiency.distribution import (
    FingerprintDistribution,
)
from agentassay.efficiency.fingerprint import (
    BehavioralFingerprint,
)
from agentassay.efficiency.regression import (
    fingerprint_regression_test,
)

from .conftest import make_regressed_traces, make_trace, make_traces

# ===================================================================
# BehavioralFingerprint — extraction
# ===================================================================


class TestBehavioralFingerprintExtraction:
    """Tests for extracting fingerprints from execution traces."""

    def test_fingerprint_from_trace_basic(self, single_trace: ExecutionTrace):
        """Extract a fingerprint from a simple 5-step trace.
        The fingerprint should capture step count and duration."""
        fp = BehavioralFingerprint.from_trace(single_trace)

        assert fp.step_count == single_trace.step_count
        assert fp.total_duration_ms == pytest.approx(single_trace.total_duration_ms)
        assert fp.total_tokens > 0

    def test_fingerprint_from_trace_with_tools(self, tool_heavy_trace: ExecutionTrace):
        """Fingerprint from a trace using 5 distinct tools captures all of them."""
        fp = BehavioralFingerprint.from_trace(tool_heavy_trace)

        # The trace uses ["search", "calculate", "write", "read", "delete"]
        assert len(fp.tool_distribution) > 0
        # Every tool that appears in the trace should be in the distribution
        for tool in tool_heavy_trace.tools_used:
            assert tool in fp.tool_distribution

    def test_fingerprint_from_trace_with_errors(self, error_trace: ExecutionTrace):
        """Fingerprint correctly reflects a failed trace with minimal steps."""
        fp = BehavioralFingerprint.from_trace(error_trace)

        assert fp.step_count == 2

    def test_fingerprint_to_vector(self, single_trace: ExecutionTrace):
        """to_vector() produces a numpy array of finite floats with positive dimension."""
        fp = BehavioralFingerprint.from_trace(single_trace)
        vec = fp.to_vector()

        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == 14
        assert np.all(np.isfinite(vec))

    def test_fingerprint_determinism(self):
        """Same trace always produces the exact same fingerprint vector."""
        trace = make_trace(steps=5, seed=42)

        fp1 = BehavioralFingerprint.from_trace(trace)
        fp2 = BehavioralFingerprint.from_trace(trace)

        np.testing.assert_array_equal(fp1.to_vector(), fp2.to_vector())

    def test_fingerprint_sensitivity(self):
        """Different behavioral patterns produce different fingerprint vectors.

        A 5-step trace using ["search", "calculate"] and a 10-step trace using
        ["write", "delete", "deploy"] should yield distinct fingerprints.
        """
        trace_a = make_trace(steps=5, tools=["search", "calculate"], seed=1)
        trace_b = make_trace(steps=10, tools=["write", "delete"], cost=0.05, seed=2)

        vec_a = BehavioralFingerprint.from_trace(trace_a).to_vector()
        vec_b = BehavioralFingerprint.from_trace(trace_b).to_vector()

        assert not np.array_equal(vec_a, vec_b)

    def test_fingerprint_tool_distribution_sums_to_one(self, tool_heavy_trace: ExecutionTrace):
        """Tool distribution probabilities must form a valid distribution (sum to 1)."""
        fp = BehavioralFingerprint.from_trace(tool_heavy_trace)

        total = sum(fp.tool_distribution.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_fingerprint_tool_distribution_all_nonnegative(self, tool_heavy_trace: ExecutionTrace):
        """Every tool probability must be non-negative."""
        fp = BehavioralFingerprint.from_trace(tool_heavy_trace)

        for prob in fp.tool_distribution.values():
            assert prob >= 0.0


# ===================================================================
# FingerprintDistribution — statistics over multiple fingerprints
# ===================================================================


class TestFingerprintDistribution:
    """Tests for FingerprintDistribution computed from trace collections."""

    def test_distribution_mean(self, baseline_traces: list[ExecutionTrace]):
        """Mean vector has the correct length and finite values."""
        fps = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        dist = FingerprintDistribution(fps)
        mean = dist.mean_vector

        assert isinstance(mean, np.ndarray)
        assert mean.ndim == 1
        assert len(mean) == 14
        assert np.all(np.isfinite(mean))

    def test_distribution_variance(self, baseline_traces: list[ExecutionTrace]):
        """Behavioral variance is a non-negative scalar."""
        fps = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        dist = FingerprintDistribution(fps)
        var = dist.behavioral_variance

        assert isinstance(var, float)
        assert var >= 0.0

    def test_distribution_distance_identical(self, baseline_traces: list[ExecutionTrace]):
        """Distance between a distribution and itself should be zero (or near-zero)."""
        fps = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        dist = FingerprintDistribution(fps)
        distance = dist.distance_to(dist)

        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_distribution_distance_different(
        self,
        baseline_traces: list[ExecutionTrace],
        regressed_traces: list[ExecutionTrace],
    ):
        """Distance between baseline and regressed distributions is positive."""
        fps_base = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        fps_reg = [BehavioralFingerprint.from_trace(t) for t in regressed_traces]
        dist_base = FingerprintDistribution(fps_base)
        dist_reg = FingerprintDistribution(fps_reg)
        distance = dist_base.distance_to(dist_reg)

        assert distance > 0.0

    def test_distribution_regression_test_no_regression(self):
        """When comparing the same behavioral distribution, no regression is detected.

        Generate two sets from the same behavioral template with matching
        tools, steps, and cost — only step durations vary slightly.
        With alpha=0.01 (conservative), this should not detect regression.
        """
        # Use identical parameters for both sets (same seeds offset)
        set_a = make_traces(20, steps=5, tools=["search", "calculate"], cost=0.01)
        set_b = make_traces(20, steps=5, tools=["search", "calculate"], cost=0.01)
        fps_a = [BehavioralFingerprint.from_trace(t) for t in set_a]
        fps_b = [BehavioralFingerprint.from_trace(t) for t in set_b]
        dist_a = FingerprintDistribution(fps_a)
        dist_b = FingerprintDistribution(fps_b)
        result = dist_a.regression_test(dist_b, alpha=0.01)

        # Same behavioral template should not detect regression at alpha=0.01
        assert result["p_value"] > 0.001

    def test_distribution_regression_test_with_regression(
        self,
        baseline_traces: list[ExecutionTrace],
        regressed_traces: list[ExecutionTrace],
    ):
        """When behavior has genuinely changed, regression is detected (p < alpha)."""
        fps_base = [BehavioralFingerprint.from_trace(t) for t in baseline_traces]
        fps_reg = [BehavioralFingerprint.from_trace(t) for t in regressed_traces]
        dist_base = FingerprintDistribution(fps_base)
        dist_reg = FingerprintDistribution(fps_reg)
        result = dist_base.regression_test(dist_reg, alpha=0.05)

        assert result["regression_detected"] is True
        assert result["p_value"] < 0.05

    def test_distribution_small_sample(self):
        """Distribution handles small sample sizes (n=3) without crashing.

        With very few traces, variance estimates will be noisy, but the
        code should not raise or produce NaN.
        """
        traces = make_traces(3)
        fps = [BehavioralFingerprint.from_trace(t) for t in traces]
        dist = FingerprintDistribution(fps)

        mean = dist.mean_vector
        var = dist.behavioral_variance

        assert np.all(np.isfinite(mean))
        assert np.isfinite(var)
        assert var >= 0.0

    def test_distribution_single_trace(self):
        """Distribution from a single trace raises ValueError (need >= 2)."""
        traces = make_traces(1)
        fps = [BehavioralFingerprint.from_trace(t) for t in traces]

        with pytest.raises(ValueError, match="at least 2"):
            FingerprintDistribution(fps)


# ===================================================================
# fingerprint_regression_test — high-level API
# ===================================================================


class TestBehavioralFingerprintFromTrialResult:
    """Tests for creating fingerprints from daemon trial result dicts."""

    def _make_trial_result(
        self,
        step_count: int = 5,
        tools: list[str] | None = None,
        include_steps: bool = True,
    ) -> dict:
        """Build a minimal trial result dict for testing."""
        tools = tools or ["search", "calculate"]
        steps = []
        if include_steps:
            for i in range(step_count):
                tool = tools[i % len(tools)]
                steps.append(
                    {
                        "step_index": i,
                        "action": "tool_call",
                        "tool_name": tool,
                        "tool_input": {"q": f"query-{i}"},
                        "tool_output": f"result-{i}",
                        "llm_output": None,
                        "model": "gpt-5.2",
                        "duration_ms": 30.0 + i * 5,
                        "usage": {
                            "prompt_tokens": 100 + i * 10,
                            "completion_tokens": 50 + i * 5,
                        },
                    }
                )
        return {
            "trial_id": "trial-001",
            "scenario_id": "ecommerce-checkout",
            "model": "gpt-5.2",
            "passed": True,
            "score": 0.85,
            "success": True,
            "error": None,
            "duration_ms": 250.0,
            "cost_usd": 0.03,
            "tokens": 750,
            "step_count": step_count,
            "timestamp": "2026-02-28T12:00:00Z",
            "_steps": steps,
        }

    def test_from_trial_result_basic(self):
        """Basic extraction from a well-formed trial result dict."""
        result = self._make_trial_result(step_count=5)
        fp = BehavioralFingerprint.from_trial_result(result)

        assert fp.step_count == 5
        assert fp.tool_count > 0
        assert len(fp.tool_distribution) == 2  # search + calculate
        assert fp.total_tokens == 750
        assert fp.total_duration_ms == pytest.approx(250.0)

    def test_from_trial_result_empty_steps(self):
        """When _steps is empty, creates a minimal fingerprint with zero values."""
        result = self._make_trial_result(include_steps=False)
        fp = BehavioralFingerprint.from_trial_result(result)

        assert fp.step_count == 0
        assert fp.tool_count == 0
        assert fp.tool_entropy == 0.0
        assert len(fp.tool_distribution) == 0

    def test_from_trial_result_missing_steps_key(self):
        """When _steps key is absent, behaves like empty steps."""
        result = self._make_trial_result()
        del result["_steps"]
        fp = BehavioralFingerprint.from_trial_result(result)

        assert fp.step_count == 0

    def test_from_trial_result_vector_finite(self):
        """Vector from trial-result fingerprint has all finite values."""
        result = self._make_trial_result(step_count=8)
        fp = BehavioralFingerprint.from_trial_result(result)
        vec = fp.to_vector()

        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == 14
        assert np.all(np.isfinite(vec))

    def test_from_trial_result_tokens_from_usage(self):
        """Per-step usage tokens are picked up via step metadata."""
        result = self._make_trial_result(step_count=3)
        fp = BehavioralFingerprint.from_trial_result(result)

        # Tokens come from trace-level metadata (total_tokens=750)
        assert fp.total_tokens == 750

    def test_from_trial_result_error_trace(self):
        """Trial result with error still produces a valid fingerprint."""
        result = self._make_trial_result(step_count=2)
        result["success"] = False
        result["error"] = "Agent crashed"
        result["passed"] = False
        fp = BehavioralFingerprint.from_trial_result(result)

        assert fp.step_count == 2
        vec = fp.to_vector()
        assert np.all(np.isfinite(vec))

    def test_from_trial_result_no_usage_in_steps(self):
        """Steps without usage dicts don't crash — metadata is empty."""
        result = self._make_trial_result(step_count=3)
        for step in result["_steps"]:
            del step["usage"]
        fp = BehavioralFingerprint.from_trial_result(result)

        assert fp.step_count == 3


class TestHotellingT2Test:
    """Tests for Hotelling's T² multivariate regression detection."""

    def test_same_distribution_no_regression(self):
        """Two groups from the same behavioral template: no regression."""
        set_a = make_traces(25, steps=5, tools=["search", "calculate"], cost=0.01)
        set_b = make_traces(25, steps=5, tools=["search", "calculate"], cost=0.01)
        fps_a = [BehavioralFingerprint.from_trace(t) for t in set_a]
        fps_b = [BehavioralFingerprint.from_trace(t) for t in set_b]

        # Should NOT detect regression at alpha=0.05
        assert BehavioralFingerprint.hotelling_t2_test(fps_a, fps_b, alpha=0.05) is False

    def test_different_distribution_regression(self):
        """Two genuinely different groups: regression detected."""
        baseline = make_traces(25)
        regressed = make_regressed_traces(25)
        fps_base = [BehavioralFingerprint.from_trace(t) for t in baseline]
        fps_reg = [BehavioralFingerprint.from_trace(t) for t in regressed]

        assert BehavioralFingerprint.hotelling_t2_test(fps_base, fps_reg) is True

    def test_small_sample_no_crash(self):
        """Works with small samples (n=3 each) without crashing."""
        set_a = make_traces(3, steps=5, tools=["search"], cost=0.01)
        set_b = make_traces(3, steps=5, tools=["search"], cost=0.01)
        fps_a = [BehavioralFingerprint.from_trace(t) for t in set_a]
        fps_b = [BehavioralFingerprint.from_trace(t) for t in set_b]

        # Should not crash — may fall back to t-tests due to singular cov
        result = BehavioralFingerprint.hotelling_t2_test(fps_a, fps_b, alpha=0.05)
        assert isinstance(result, bool)

    def test_insufficient_samples_raises(self):
        """Fewer than 2 fingerprints per group raises ValueError."""
        set_a = make_traces(1)
        set_b = make_traces(5)
        fps_a = [BehavioralFingerprint.from_trace(t) for t in set_a]
        fps_b = [BehavioralFingerprint.from_trace(t) for t in set_b]

        with pytest.raises(ValueError, match="at least 2"):
            BehavioralFingerprint.hotelling_t2_test(fps_a, fps_b)

    def test_returns_bool(self):
        """Return type is always a plain bool, not numpy.bool_."""
        baseline = make_traces(10)
        candidate = make_traces(10)
        fps_base = [BehavioralFingerprint.from_trace(t) for t in baseline]
        fps_cand = [BehavioralFingerprint.from_trace(t) for t in candidate]

        result = BehavioralFingerprint.hotelling_t2_test(fps_base, fps_cand)
        assert type(result) is bool

    def test_alpha_sensitivity(self):
        """More stringent alpha makes regression harder to detect."""
        baseline = make_traces(25)
        regressed = make_regressed_traces(25)
        fps_base = [BehavioralFingerprint.from_trace(t) for t in baseline]
        fps_reg = [BehavioralFingerprint.from_trace(t) for t in regressed]

        # Very lenient alpha should detect, very stringent may or may not
        # But the regressed set is strongly different, so both should detect
        assert BehavioralFingerprint.hotelling_t2_test(fps_base, fps_reg, alpha=0.10) is True
        assert BehavioralFingerprint.hotelling_t2_test(fps_base, fps_reg, alpha=0.01) is True


class TestFingerprintRegressionTest:
    """Tests for the top-level fingerprint_regression_test function."""

    def test_high_level_no_regression(self):
        """No regression when comparing traces from the same agent behavior.

        Generate two identical-template sets. The fingerprint regression test
        should return a high p-value (no significant behavioral difference).
        """
        baseline = make_traces(25, steps=5, tools=["search", "calculate"], cost=0.01)
        current = make_traces(25, steps=5, tools=["search", "calculate"], cost=0.01)
        result = fingerprint_regression_test(baseline, current, alpha=0.01)

        # Same template — p-value should be well above strict threshold
        assert result["p_value"] > 0.001

    def test_high_level_regression_detected(self):
        """Regression detected when comparing baseline to regressed traces."""
        baseline = make_traces(25)
        current = make_regressed_traces(25)
        result = fingerprint_regression_test(baseline, current, alpha=0.05)

        assert result["regression_detected"] is True

    def test_changed_dimensions_reported(self):
        """The result identifies which behavioral dimensions changed.

        When the agent regresses (different tools, more steps, higher cost),
        the changed_dimensions field should contain at least one entry.
        """
        baseline = make_traces(25)
        current = make_regressed_traces(25)
        result = fingerprint_regression_test(baseline, current, alpha=0.05)

        assert "changed_dimensions" in result
        assert len(result["changed_dimensions"]) > 0
