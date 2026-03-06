"""Tests for coverage metrics module.

Tests ToolCoverageTracker, PathCoverageTracker, StateCoverageTracker,
BoundaryCoverageTracker, ModelCoverageTracker, CoverageTuple, and
AgentCoverageCollector.

Target: ~20 tests.
"""

from __future__ import annotations

import pytest

from agentassay.coverage.aggregate import (
    AgentCoverageCollector,
    CoverageTuple,
    _normalize_state_count,
)
from agentassay.coverage.boundary_coverage import BoundaryCoverageTracker
from agentassay.coverage.model_coverage import ModelCoverageTracker
from agentassay.coverage.path_coverage import PathCoverageTracker
from agentassay.coverage.state_coverage import StateCoverageTracker
from agentassay.coverage.tool_coverage import ToolCoverageTracker
from tests.conftest import make_trace

# ===================================================================
# ToolCoverageTracker
# ===================================================================


class TestToolCoverageTracker:
    """Tests for ToolCoverageTracker."""

    def test_initial_coverage_zero(self):
        tracker = ToolCoverageTracker({"search", "calculate", "write"})
        assert tracker.coverage_ratio() == 0.0

    def test_full_coverage(self):
        tracker = ToolCoverageTracker({"search", "calculate"})
        trace = make_trace(steps=2, tools=["search", "calculate"])
        tracker.update(trace)
        assert tracker.coverage_ratio() == 1.0

    def test_partial_coverage(self):
        tracker = ToolCoverageTracker({"search", "calculate", "write"})
        trace = make_trace(steps=2, tools=["search", "calculate"])
        tracker.update(trace)
        assert tracker.coverage_ratio() == pytest.approx(2.0 / 3.0)

    def test_uncovered_tools(self):
        tracker = ToolCoverageTracker({"search", "calculate", "write"})
        trace = make_trace(steps=1, tools=["search"])
        tracker.update(trace)
        assert "write" in tracker.uncovered_tools()
        assert "calculate" in tracker.uncovered_tools()

    def test_empty_known_tools_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ToolCoverageTracker(set())

    def test_reset_clears_observations(self):
        tracker = ToolCoverageTracker({"search"})
        trace = make_trace(steps=1, tools=["search"])
        tracker.update(trace)
        assert tracker.coverage_ratio() == 1.0
        tracker.reset()
        assert tracker.coverage_ratio() == 0.0


# ===================================================================
# PathCoverageTracker
# ===================================================================


class TestPathCoverageTracker:
    """Tests for PathCoverageTracker."""

    def test_initial_coverage_zero(self):
        tracker = PathCoverageTracker(max_path_depth=5)
        assert tracker.coverage_ratio() == 0.0

    def test_unique_paths_recorded(self):
        tracker = PathCoverageTracker(max_path_depth=5)
        # trace1: tool_call, tool_call (2 steps, both tool_call action)
        trace1 = make_trace(steps=2, tools=["search", "calculate"])
        # trace2: 3 steps gives a different path length -> different path
        trace2 = make_trace(steps=3, tools=["search", "calculate"])
        tracker.update(trace1)
        tracker.update(trace2)
        assert len(tracker.unique_paths()) == 2

    def test_identical_paths_counted_once(self):
        tracker = PathCoverageTracker(max_path_depth=5)
        trace = make_trace(steps=2, tools=["search", "calculate"])
        tracker.update(trace)
        tracker.update(trace)
        assert len(tracker.unique_paths()) == 1

    def test_max_depth_truncation(self):
        tracker = PathCoverageTracker(max_path_depth=2)
        trace = make_trace(steps=5, tools=["search"])
        tracker.update(trace)
        paths = tracker.unique_paths()
        for p in paths:
            assert len(p) <= 2

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError):
            PathCoverageTracker(max_path_depth=0)


# ===================================================================
# StateCoverageTracker
# ===================================================================


class TestStateCoverageTracker:
    """Tests for StateCoverageTracker."""

    def test_initial_count_zero(self):
        tracker = StateCoverageTracker()
        assert tracker.coverage_count() == 0

    def test_states_from_tool_steps(self):
        tracker = StateCoverageTracker()
        trace = make_trace(steps=3, tools=["search", "calculate"])
        tracker.update(trace)
        assert tracker.coverage_count() > 0

    def test_reset_clears(self):
        tracker = StateCoverageTracker()
        trace = make_trace(steps=3)
        tracker.update(trace)
        tracker.reset()
        assert tracker.coverage_count() == 0


# ===================================================================
# BoundaryCoverageTracker
# ===================================================================


class TestBoundaryCoverageTracker:
    """Tests for BoundaryCoverageTracker."""

    def test_no_boundaries_full_coverage(self):
        tracker = BoundaryCoverageTracker()
        assert tracker.coverage_ratio() == 1.0

    def test_untested_boundary_zero_coverage(self):
        tracker = BoundaryCoverageTracker(boundaries={"latency": (0.0, 5000.0)})
        assert tracker.coverage_ratio() == 0.0

    def test_near_both_ends_full_coverage(self):
        tracker = BoundaryCoverageTracker(boundaries={"latency": (0.0, 100.0)})
        # Create traces with metadata near boundaries
        low_trace = make_trace(steps=1, metadata={"latency": 5.0})
        high_trace = make_trace(steps=1, metadata={"latency": 95.0})
        tracker.update(low_trace)
        tracker.update(high_trace)
        assert tracker.coverage_ratio() == 1.0

    def test_invalid_boundary_raises(self):
        with pytest.raises(ValueError, match="min >= max"):
            BoundaryCoverageTracker(boundaries={"x": (100.0, 50.0)})

    def test_non_finite_boundary_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            BoundaryCoverageTracker(boundaries={"x": (float("inf"), 100.0)})


# ===================================================================
# ModelCoverageTracker
# ===================================================================


class TestModelCoverageTracker:
    """Tests for ModelCoverageTracker."""

    def test_discovery_mode_full_coverage(self):
        tracker = ModelCoverageTracker()
        assert tracker.coverage_ratio() == 1.0

    def test_known_models_tracking(self):
        tracker = ModelCoverageTracker({"gpt-4o", "claude-opus"})
        trace = make_trace(model="gpt-4o")
        tracker.update(trace)
        assert tracker.coverage_ratio() == 0.5

    def test_untested_models(self):
        tracker = ModelCoverageTracker({"gpt-4o", "claude-opus"})
        trace = make_trace(model="gpt-4o")
        tracker.update(trace)
        assert "claude-opus" in tracker.untested_models()

    def test_full_model_coverage(self):
        tracker = ModelCoverageTracker({"gpt-4o", "claude"})
        t1 = make_trace(model="gpt-4o")
        t2 = make_trace(model="claude")
        tracker.update(t1)
        tracker.update(t2)
        assert tracker.coverage_ratio() == 1.0

    def test_reset_preserves_known(self):
        tracker = ModelCoverageTracker({"gpt-4o"})
        trace = make_trace(model="gpt-4o")
        tracker.update(trace)
        tracker.reset()
        assert tracker.coverage_ratio() == 0.0
        assert "gpt-4o" in tracker.untested_models()


# ===================================================================
# CoverageTuple
# ===================================================================


class TestCoverageTuple:
    """Tests for CoverageTuple."""

    def test_geometric_mean_perfect(self):
        ct = CoverageTuple(tool=1.0, path=1.0, state=1.0, boundary=1.0, model=1.0)
        assert ct.overall == pytest.approx(1.0)

    def test_geometric_mean_zero_dimension(self):
        ct = CoverageTuple(tool=0.0, path=1.0, state=1.0, boundary=1.0, model=1.0)
        assert ct.overall == 0.0

    def test_weakest_dimension(self):
        ct = CoverageTuple(tool=0.9, path=0.3, state=0.8, boundary=0.7, model=0.6)
        name, value = ct.weakest
        assert name == "path"
        assert value == 0.3

    def test_dimensions_dict(self):
        ct = CoverageTuple(tool=0.5, path=0.5, state=0.5, boundary=0.5, model=0.5)
        dims = ct.dimensions
        assert len(dims) == 5
        assert all(v == 0.5 for v in dims.values())

    def test_to_vector(self):
        ct = CoverageTuple(tool=0.1, path=0.2, state=0.3, boundary=0.4, model=0.5)
        assert ct.to_vector() == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_frozen(self):
        ct = CoverageTuple(tool=0.5, path=0.5, state=0.5, boundary=0.5, model=0.5)
        with pytest.raises(Exception):
            ct.tool = 0.9


# ===================================================================
# Normalize state count
# ===================================================================


class TestNormalizeStateCount:
    """Tests for _normalize_state_count helper."""

    def test_zero_gives_zero(self):
        assert _normalize_state_count(0) == 0.0

    def test_one_gives_half(self):
        assert _normalize_state_count(1) == 0.5

    def test_nine_gives_point_nine(self):
        assert _normalize_state_count(9) == pytest.approx(0.9)

    def test_negative_gives_zero(self):
        assert _normalize_state_count(-5) == 0.0

    def test_monotonically_increasing(self):
        prev = -1.0
        for n in range(0, 100):
            val = _normalize_state_count(n)
            assert val > prev or n == 0
            prev = val


# ===================================================================
# AgentCoverageCollector
# ===================================================================


class TestAgentCoverageCollector:
    """Tests for AgentCoverageCollector."""

    def test_initial_snapshot(self):
        collector = AgentCoverageCollector(
            known_tools={"search"},
            known_models={"gpt-4o"},
        )
        snap = collector.snapshot()
        assert snap.tool == 0.0
        assert snap.model == 0.0

    def test_update_and_snapshot(self):
        collector = AgentCoverageCollector(
            known_tools={"search", "calculate"},
            known_models={"test-model"},
        )
        trace = make_trace(steps=2, tools=["search", "calculate"])
        collector.update(trace)
        snap = collector.snapshot()
        assert snap.tool == 1.0
        assert snap.model == 1.0

    def test_reset(self):
        collector = AgentCoverageCollector(
            known_tools={"search"},
            known_models={"test-model"},
        )
        trace = make_trace(steps=1, tools=["search"])
        collector.update(trace)
        collector.reset()
        snap = collector.snapshot()
        assert snap.tool == 0.0
