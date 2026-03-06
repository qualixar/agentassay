"""AgentAssay coverage — 5-dimensional agent coverage metrics.

Implements Definition 4.1 from the paper: the coverage vector
C = (C_tool, C_path, C_state, C_boundary, C_model) ∈ [0,1]^5.

Individual trackers measure each dimension independently.
``AgentCoverageCollector`` orchestrates all five and produces
``CoverageTuple`` snapshots with the geometric-mean overall score.
"""

from agentassay.coverage.aggregate import (
    AgentCoverageCollector,
    CoverageTuple,
)
from agentassay.coverage.boundary_coverage import BoundaryCoverageTracker
from agentassay.coverage.model_coverage import ModelCoverageTracker
from agentassay.coverage.path_coverage import PathCoverageTracker
from agentassay.coverage.state_coverage import StateCoverageTracker
from agentassay.coverage.tool_coverage import ToolCoverageTracker

__all__ = [
    # Individual trackers
    "ToolCoverageTracker",
    "PathCoverageTracker",
    "StateCoverageTracker",
    "BoundaryCoverageTracker",
    "ModelCoverageTracker",
    # Aggregate
    "CoverageTuple",
    "AgentCoverageCollector",
]
