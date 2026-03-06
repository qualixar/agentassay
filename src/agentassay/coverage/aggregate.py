# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Aggregate coverage — the 5-dimensional coverage vector.

Implements Definition 4.1 from the paper:

    C = (C_tool, C_path, C_state, C_boundary, C_model) ∈ [0,1]^5

The ``CoverageTuple`` is the frozen snapshot, and ``AgentCoverageCollector``
orchestrates all five individual trackers and produces snapshots.

The overall coverage score uses the **geometric mean** of all five
dimensions. The geometric mean is chosen over the arithmetic mean
because:
    1. It penalizes imbalance — a single zero dimension zeroes the score.
    2. It reflects the multiplicative nature of coverage adequacy:
       high tool coverage is meaningless if path coverage is zero.
    3. It matches how coverage is used in practice: all dimensions
       must be reasonably exercised for the suite to be trusted.

State coverage normalization:
    State space is theoretically unbounded, so we normalize using a
    diminishing-returns function: C_state = 1 - 1/(1 + count).
    This maps count=0 -> 0.0 and approaches 1.0 as count grows.
    At count=9 it reaches 0.9; at count=99 it reaches ~0.99.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from agentassay.core.models import ExecutionTrace
from agentassay.coverage.boundary_coverage import BoundaryCoverageTracker
from agentassay.coverage.model_coverage import ModelCoverageTracker
from agentassay.coverage.path_coverage import PathCoverageTracker
from agentassay.coverage.state_coverage import StateCoverageTracker
from agentassay.coverage.tool_coverage import ToolCoverageTracker

# ===================================================================
# CoverageTuple — the frozen 5D snapshot
# ===================================================================


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_state_count(count: int) -> float:
    """Normalize unbounded state count to [0, 1] using diminishing returns.

    f(n) = 1 - 1/(1 + n)

    Properties:
        f(0) = 0.0
        f(1) = 0.5
        f(9) = 0.9
        f(99) ≈ 0.99
        lim n→∞ f(n) = 1.0

    Monotonically increasing, concave — more states always help
    but with diminishing marginal value.
    """
    if count < 0:
        return 0.0
    return 1.0 - 1.0 / (1.0 + count)


class CoverageTuple(BaseModel):
    """The 5-dimensional agent coverage vector (paper Definition 4.1).

    Frozen after creation — coverage snapshots are immutable records.

    Attributes
    ----------
    tool : float
        Tool coverage ratio in [0, 1].
    path : float
        Decision path coverage ratio in [0, 1].
    state : float
        Normalized state space coverage in [0, 1].
    boundary : float
        Boundary condition coverage ratio in [0, 1].
    model : float
        Model coverage ratio in [0, 1].
    timestamp : datetime
        When this snapshot was taken (UTC).
    """

    model_config = ConfigDict(frozen=True)

    tool: float = Field(ge=0.0, le=1.0)
    path: float = Field(ge=0.0, le=1.0)
    state: float = Field(ge=0.0, le=1.0)
    boundary: float = Field(ge=0.0, le=1.0)
    model: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=_now_utc)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def overall(self) -> float:
        """Geometric mean of all 5 coverage dimensions.

        Returns 0.0 if ANY dimension is 0.0 — this is intentional.
        A test suite with zero coverage on any dimension is not
        trustworthy regardless of other dimensions.

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        values = [self.tool, self.path, self.state, self.boundary, self.model]
        if any(v == 0.0 for v in values):
            return 0.0
        product = math.prod(values)
        return product ** (1.0 / len(values))

    @property
    def dimensions(self) -> dict[str, float]:
        """All 5 dimensions as a dictionary.

        Returns
        -------
        dict[str, float]
        """
        return {
            "tool": self.tool,
            "path": self.path,
            "state": self.state,
            "boundary": self.boundary,
            "model": self.model,
        }

    @property
    def weakest(self) -> tuple[str, float]:
        """The dimension with the lowest coverage score.

        When multiple dimensions tie for lowest, returns the first
        in canonical order (tool, path, state, boundary, model).

        Returns
        -------
        tuple[str, float]
            (dimension_name, score)
        """
        dims = self.dimensions
        return min(dims.items(), key=lambda kv: kv[1])

    def to_vector(self) -> list[float]:
        """Coverage as a plain list in canonical order.

        Order: [tool, path, state, boundary, model]

        Returns
        -------
        list[float]
        """
        return [self.tool, self.path, self.state, self.boundary, self.model]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CoverageTuple("
            f"tool={self.tool:.2f}, "
            f"path={self.path:.2f}, "
            f"state={self.state:.2f}, "
            f"boundary={self.boundary:.2f}, "
            f"model={self.model:.2f}, "
            f"overall={self.overall:.2f})"
        )


# ===================================================================
# AgentCoverageCollector — orchestrates all 5 trackers
# ===================================================================


class AgentCoverageCollector:
    """Manages all 5 coverage trackers and produces snapshots.

    This is the primary entry point for coverage measurement. Feed
    it execution traces, then call ``snapshot()`` to get the current
    5-dimensional coverage vector.

    Parameters
    ----------
    known_tools : set[str]
        Tools the agent is configured to use.
    known_models : set[str]
        Models the agent may use. Empty = discovery mode.
    boundaries : dict[str, tuple[float, float]]
        Named operational boundaries with (min, max) ranges.
    max_path_depth : int
        Maximum depth for decision path tracking.

    Example
    -------
    >>> collector = AgentCoverageCollector(
    ...     known_tools={"search", "calculate", "write"},
    ...     known_models={"gpt-4o", "claude-opus-4-6"},
    ...     boundaries={"latency_ms": (0, 5000)},
    ...     max_path_depth=8,
    ... )
    >>> collector.update(trace)
    >>> coverage = collector.snapshot()
    >>> print(coverage.overall)
    0.72
    """

    __slots__ = (
        "_tool_tracker",
        "_path_tracker",
        "_state_tracker",
        "_boundary_tracker",
        "_model_tracker",
    )

    def __init__(
        self,
        known_tools: set[str],
        known_models: set[str] | None = None,
        boundaries: dict[str, tuple[float, float]] | None = None,
        max_path_depth: int = 10,
    ) -> None:
        self._tool_tracker = ToolCoverageTracker(known_tools)
        self._path_tracker = PathCoverageTracker(max_path_depth)
        self._state_tracker = StateCoverageTracker()
        self._boundary_tracker = BoundaryCoverageTracker(boundaries)
        self._model_tracker = ModelCoverageTracker(known_models)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trace: ExecutionTrace) -> None:
        """Feed a trace to all 5 trackers.

        Parameters
        ----------
        trace : ExecutionTrace
            A completed agent execution trace.
        """
        self._tool_tracker.update(trace)
        self._path_tracker.update(trace)
        self._state_tracker.update(trace)
        self._boundary_tracker.update(trace)
        self._model_tracker.update(trace)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> CoverageTuple:
        """Capture the current 5-dimensional coverage vector.

        State coverage is normalized from raw count to [0, 1]
        using the diminishing-returns function.

        Returns
        -------
        CoverageTuple
            Frozen, immutable coverage record.
        """
        state_raw = self._state_tracker.coverage_count()
        return CoverageTuple(
            tool=self._tool_tracker.coverage_ratio(),
            path=self._path_tracker.coverage_ratio(),
            state=_normalize_state_count(state_raw),
            boundary=self._boundary_tracker.coverage_ratio(),
            model=self._model_tracker.coverage_ratio(),
        )

    # ------------------------------------------------------------------
    # Direct tracker access (for advanced queries)
    # ------------------------------------------------------------------

    @property
    def tool_tracker(self) -> ToolCoverageTracker:
        """Direct access to the tool coverage tracker."""
        return self._tool_tracker

    @property
    def path_tracker(self) -> PathCoverageTracker:
        """Direct access to the path coverage tracker."""
        return self._path_tracker

    @property
    def state_tracker(self) -> StateCoverageTracker:
        """Direct access to the state coverage tracker."""
        return self._state_tracker

    @property
    def boundary_tracker(self) -> BoundaryCoverageTracker:
        """Direct access to the boundary coverage tracker."""
        return self._boundary_tracker

    @property
    def model_tracker(self) -> ModelCoverageTracker:
        """Direct access to the model coverage tracker."""
        return self._model_tracker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all 5 trackers. Configurations are preserved."""
        self._tool_tracker.reset()
        self._path_tracker.reset()
        self._state_tracker.reset()
        self._boundary_tracker.reset()
        self._model_tracker.reset()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        snap = self.snapshot()
        return f"AgentCoverageCollector(overall={snap.overall:.2f})"
