"""Tests for AgentAssay pytest plugin.

Verifies that the plugin registers correctly with pytest, that the
``agentassay`` marker is available, and that assertion helpers work.

Target: 10+ tests.  Max 200 lines.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentassay.plugin.pytest_plugin import (
    MARKER_NAME,
    assert_no_regression,
    assert_pass_rate,
    assert_verdict_passes,
    pytest_configure,
)
from agentassay.core.models import AssayConfig
from agentassay.statistics.confidence import ConfidenceInterval


# ===================================================================
# Plugin registration
# ===================================================================


class TestPluginRegistration:
    """Verify the plugin hooks register correctly."""

    def test_marker_name_constant(self) -> None:
        assert MARKER_NAME == "agentassay"

    def test_marker_is_registered(self, pytestconfig: pytest.Config) -> None:
        """The agentassay marker should be known to pytest."""
        markers = pytestconfig.getini("markers")
        found = any("agentassay" in str(m) for m in markers)
        assert found, "agentassay marker not registered"


# ===================================================================
# Public imports from agentassay.plugin
# ===================================================================


class TestPublicImports:
    """Verify the plugin __init__.py re-exports work."""

    def test_import_assert_no_regression(self) -> None:
        from agentassay.plugin import assert_no_regression as anr
        assert callable(anr)

    def test_import_assert_pass_rate(self) -> None:
        from agentassay.plugin import assert_pass_rate as apr
        assert callable(apr)

    def test_import_assert_verdict_passes(self) -> None:
        from agentassay.plugin import assert_verdict_passes as avp
        assert callable(avp)


# ===================================================================
# assert_pass_rate
# ===================================================================


class TestAssertPassRate:
    """Tests for the assert_pass_rate helper."""

    def test_pass_rate_above_threshold(self) -> None:
        # 95% pass rate with 100 trials -- Wilson CI lower bound is
        # well above 0.80, so this should always pass.
        results = [True] * 95 + [False] * 5
        ci = assert_pass_rate(results, threshold=0.80)
        assert isinstance(ci, ConfidenceInterval)
        assert ci.point_estimate == pytest.approx(0.95, abs=0.01)

    def test_pass_rate_below_threshold_raises(self) -> None:
        results = [True] * 10 + [False] * 20  # 33% pass rate
        with pytest.raises(AssertionError, match="pass rate below threshold"):
            assert_pass_rate(results, threshold=0.80)

    def test_pass_rate_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            assert_pass_rate([], threshold=0.80)

    def test_pass_rate_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            assert_pass_rate([True], threshold=1.5)


# ===================================================================
# assert_no_regression
# ===================================================================


class TestAssertNoRegression:
    """Tests for the assert_no_regression helper."""

    def test_no_regression_when_similar(self) -> None:
        baseline = [True] * 90 + [False] * 10
        current = [True] * 88 + [False] * 12
        result = assert_no_regression(baseline, current)
        assert result is not None
        assert not result.significant

    def test_regression_detected_raises(self) -> None:
        baseline = [True] * 95 + [False] * 5
        current = [True] * 50 + [False] * 50
        with pytest.raises(AssertionError, match="regression detected"):
            assert_no_regression(baseline, current)

    def test_empty_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            assert_no_regression([], [True, False])

    def test_empty_current_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            assert_no_regression([True, False], [])
