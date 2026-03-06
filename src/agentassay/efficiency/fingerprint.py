"""Behavioral Fingerprinting — extraction of fixed-size behavioral vectors from traces.

Provides BehavioralFingerprint: a compact, fixed-size vector extracted from an
ExecutionTrace that captures the agent's behavioral pattern (tool usage,
structural complexity, output characteristics, reasoning depth, error/recovery,
efficiency) independent of specific text produced.

Formal definition (paper Definition 7.1): phi: Trace -> R^d where phi is
deterministic, Lipschitz-continuous in edit distance, and behaviorally
equivalent traces have small ||phi(T1) - phi(T2)||.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from agentassay.core.models import ExecutionTrace, StepTrace

logger = logging.getLogger(__name__)


# ===================================================================
# Fingerprint dimension names — canonical ordering for vector ops
# ===================================================================

_DIMENSION_NAMES: list[str] = [
    "tool_entropy",
    "tool_count",
    "unique_tools_ratio",
    "step_count",
    "max_chain_depth",
    "backtrack_ratio",
    "output_length_log",
    "has_structured_output",
    "sequence_complexity",
    "reasoning_depth",
    "error_rate",
    "recovery_rate",
    "tokens_per_step",
    "duration_per_step",
]


# ===================================================================
# BehavioralFingerprint — fixed-size vector from a single trace
# ===================================================================


class BehavioralFingerprint(BaseModel):
    """Fixed-size behavioral vector extracted from an ExecutionTrace.

    Captures six dimensions: (1) tool usage distribution, (2) structural
    complexity, (3) output characteristics, (4) reasoning patterns,
    (5) error/recovery behavior, (6) efficiency metrics. All dimensions
    are normalized to comparable scales for multivariate testing.
    """

    model_config = ConfigDict(frozen=True)

    # Dimension 1: Tool usage distribution
    tool_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Tool name to frequency (0-1). Sums to 1.0 if non-empty.",
    )
    tool_entropy: float = Field(
        ge=0.0,
        description="Normalized Shannon entropy of tool distribution. 0=single tool, 1=uniform.",
    )
    tool_count: int = Field(ge=0, description="Total tool calls in this trace.")
    unique_tools: int = Field(ge=0, description="Distinct tools used.")

    # Dimension 2: Structural complexity
    step_count: int = Field(ge=0, description="Total steps in the trace.")
    max_chain_depth: int = Field(
        ge=0, description="Longest sequential chain of tool calls without non-tool steps."
    )
    backtrack_count: int = Field(ge=0, description="Number of retries or corrections detected.")

    # Dimension 3: Output characteristics
    output_length: int = Field(ge=0, description="Character count of final output.")
    output_structure_hash: str = Field(
        description="SHA-256 of the output's structural skeleton (keys/format)."
    )
    has_structured_output: bool = Field(
        description="True if output is JSON/dict, False if free text."
    )

    # Dimension 4: Reasoning patterns
    tool_sequence_signature: str = Field(
        description="Ordered tool call pattern (e.g., 'search->filter->select')."
    )
    reasoning_depth: float = Field(
        ge=0.0, description="Average characters per step (proxy for reasoning effort)."
    )

    # Dimension 5: Error/recovery behavior
    error_count: int = Field(ge=0, description="Number of error/failed steps.")
    recovery_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of errors followed by a successful retry.",
    )

    # Dimension 6: Efficiency metrics
    total_tokens: int = Field(ge=0, description="Total tokens consumed (from metadata).")
    tokens_per_step: float = Field(ge=0.0, description="Tokens / step_count.")
    total_duration_ms: float = Field(ge=0.0, description="Total execution time in ms.")
    duration_per_step: float = Field(ge=0.0, description="Duration / step_count.")

    def to_vector(self) -> np.ndarray:
        """Convert to a 14-dimensional float64 vector for statistical operations.

        Ordering follows ``_DIMENSION_NAMES`` for reproducibility.
        """
        step_count = max(self.step_count, 1)
        tool_count = max(self.tool_count, 1)

        return np.array(
            [
                self.tool_entropy,
                # Log-scale tool count to compress range
                math.log1p(self.tool_count),
                # Ratio of unique tools to total calls (0-1)
                self.unique_tools / tool_count if self.tool_count > 0 else 0.0,
                # Log-scale step count
                math.log1p(self.step_count),
                # Chain depth normalized by step count
                self.max_chain_depth / step_count,
                # Backtrack ratio
                self.backtrack_count / step_count,
                # Log-scale output length
                math.log1p(self.output_length),
                # Binary: structured output
                1.0 if self.has_structured_output else 0.0,
                # Sequence complexity: number of unique bigrams / total bigrams
                _sequence_complexity(self.tool_sequence_signature),
                # Log-scale reasoning depth
                math.log1p(self.reasoning_depth),
                # Error rate
                self.error_count / step_count,
                # Recovery rate (already in [0,1])
                self.recovery_rate,
                # Log-scale tokens per step
                math.log1p(self.tokens_per_step),
                # Log-scale duration per step
                math.log1p(self.duration_per_step),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_trace(cls, trace: ExecutionTrace) -> BehavioralFingerprint:
        """Extract a behavioral fingerprint from an execution trace."""
        steps = trace.steps

        # --- Tool distribution ---
        tool_calls = [s.tool_name for s in steps if s.tool_name is not None]
        tool_counts = Counter(tool_calls)
        total_tool_calls = sum(tool_counts.values())

        tool_distribution: dict[str, float] = {}
        if total_tool_calls > 0:
            tool_distribution = {
                name: count / total_tool_calls for name, count in tool_counts.items()
            }

        tool_entropy = _normalized_entropy(list(tool_distribution.values()))
        unique_tools = len(tool_counts)

        # --- Structural complexity ---
        step_count = len(steps)
        max_chain_depth = _max_tool_chain(steps)
        backtrack_count = _count_backtracks(steps)

        # --- Output characteristics ---
        output_data = trace.output_data
        output_str = _to_string(output_data)
        output_length = len(output_str)
        has_structured = isinstance(output_data, (dict, list))
        structure_hash = _structure_hash(output_data)

        # --- Reasoning patterns ---
        tool_sequence = "->".join(
            s.tool_name for s in steps if s.tool_name is not None
        )
        reasoning_depth = _avg_step_content_length(steps)

        # --- Error / recovery ---
        error_steps = [s for s in steps if s.action == "error" or s.metadata.get("error")]
        error_count = len(error_steps)
        recovery_rate = _compute_recovery_rate(steps)

        # --- Efficiency ---
        total_tokens = _extract_total_tokens(trace)
        tokens_per_step = total_tokens / max(step_count, 1)
        total_dur = trace.total_duration_ms
        duration_per_step = total_dur / max(step_count, 1)

        return cls(
            tool_distribution=tool_distribution,
            tool_entropy=tool_entropy,
            tool_count=total_tool_calls,
            unique_tools=unique_tools,
            step_count=step_count,
            max_chain_depth=max_chain_depth,
            backtrack_count=backtrack_count,
            output_length=output_length,
            output_structure_hash=structure_hash,
            has_structured_output=has_structured,
            tool_sequence_signature=tool_sequence if tool_sequence else "",
            reasoning_depth=reasoning_depth,
            error_count=error_count,
            recovery_rate=recovery_rate,
            total_tokens=total_tokens,
            tokens_per_step=tokens_per_step,
            total_duration_ms=total_dur,
            duration_per_step=duration_per_step,
        )

    @classmethod
    def vector_dimension(cls) -> int:
        """Return the dimensionality of the fingerprint vector (14)."""
        return len(_DIMENSION_NAMES)

    @classmethod
    def dimension_names(cls) -> list[str]:
        """Return the canonical ordered list of dimension names."""
        return list(_DIMENSION_NAMES)

    @classmethod
    def from_trial_result(cls, trial_result: dict[str, Any]) -> BehavioralFingerprint:
        """Create a BehavioralFingerprint from the daemon's trial result dict.

        The daemon (``experiments/runner/azure_adapter.py``) emits trial
        results with a ``_steps`` list containing raw step dicts.  This
        classmethod converts that dict into an ``ExecutionTrace`` and
        delegates to ``from_trace``.

        Parameters
        ----------
        trial_result : dict
            Trial result dict with keys: trial_id, scenario_id, model,
            passed, score, success, error, duration_ms, cost_usd, tokens,
            step_count, timestamp, and ``_steps`` (list of step dicts).

        Returns
        -------
        BehavioralFingerprint
            The extracted fingerprint.  If ``_steps`` is empty, returns a
            minimal fingerprint with zero values.
        """
        raw_steps: list[dict[str, Any]] = trial_result.get("_steps", [])

        # -- Build StepTrace objects from raw step dicts --
        step_traces: list[StepTrace] = []
        for raw in raw_steps:
            action = raw.get("action", "tool_call")
            tool_name = raw.get("tool_name")

            # Collect token usage into metadata
            meta: dict[str, Any] = {}
            usage = raw.get("usage")
            if isinstance(usage, dict):
                prompt_tok = usage.get("prompt_tokens", 0)
                completion_tok = usage.get("completion_tokens", 0)
                meta["tokens"] = prompt_tok + completion_tok
                meta["prompt_tokens"] = prompt_tok
                meta["completion_tokens"] = completion_tok

            step_traces.append(
                StepTrace(
                    step_index=raw.get("step_index", len(step_traces)),
                    action=action,
                    tool_name=tool_name if action == "tool_call" else None,
                    tool_input=raw.get("tool_input"),
                    tool_output=raw.get("tool_output"),
                    llm_output=raw.get("llm_output"),
                    model=raw.get("model"),
                    duration_ms=float(raw.get("duration_ms", 0.0)),
                    metadata=meta,
                )
            )

        # -- Build ExecutionTrace --
        total_dur = float(trial_result.get("duration_ms", 0.0))
        total_tokens = int(trial_result.get("tokens", 0))

        trace = ExecutionTrace(
            trace_id=trial_result.get("trial_id", "unknown"),
            scenario_id=trial_result.get("scenario_id", "unknown"),
            steps=step_traces,
            input_data={},
            output_data=None,
            success=bool(trial_result.get("success", False)),
            error=trial_result.get("error"),
            total_duration_ms=total_dur,
            total_cost_usd=float(trial_result.get("cost_usd", 0.0)),
            model=trial_result.get("model", "unknown"),
            framework="azure_daemon",
            metadata={"total_tokens": total_tokens},
        )

        return cls.from_trace(trace)

    @staticmethod
    def hotelling_t2_test(
        baseline_fps: list[BehavioralFingerprint],
        candidate_fps: list[BehavioralFingerprint],
        alpha: float = 0.05,
    ) -> bool:
        """Hotelling's T-squared test for multivariate regression detection.

        Compares two groups of behavioral fingerprints to determine whether
        their multivariate means differ significantly — i.e., whether the
        candidate agent has *regressed* relative to the baseline.

        Parameters
        ----------
        baseline_fps : list[BehavioralFingerprint]
            Fingerprints from the known-good baseline (n1 >= 2).
        candidate_fps : list[BehavioralFingerprint]
            Fingerprints from the candidate build (n2 >= 2).
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        bool
            ``True`` if the two groups are statistically different at the
            given alpha (regression detected); ``False`` otherwise.

        Notes
        -----
        The test converts T^2 to an F-statistic:

            F = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)  *  T^2

        which follows F(p, n1 + n2 - p - 1) under H0.

        If the pooled covariance matrix is singular (common when n1 + n2
        is close to p, or features are collinear), the method falls back
        to dimension-wise Welch t-tests with Bonferroni correction.
        """
        if len(baseline_fps) < 2 or len(candidate_fps) < 2:
            raise ValueError(
                "Hotelling's T² requires at least 2 fingerprints per group; "
                f"got n1={len(baseline_fps)}, n2={len(candidate_fps)}"
            )

        # -- Vectorize --
        X1 = np.array([fp.to_vector() for fp in baseline_fps], dtype=np.float64)
        X2 = np.array([fp.to_vector() for fp in candidate_fps], dtype=np.float64)

        n1, p = X1.shape
        n2 = X2.shape[0]

        mean1 = X1.mean(axis=0)
        mean2 = X2.mean(axis=0)
        diff = mean1 - mean2

        # -- Pooled covariance --
        # Using unbiased estimators (ddof=1), then pooling
        S1 = np.cov(X1, rowvar=False, ddof=1)
        S2 = np.cov(X2, rowvar=False, ddof=1)

        # Handle edge case: np.cov returns a scalar when p==1
        if S1.ndim == 0:
            S1 = np.array([[float(S1)]])
            S2 = np.array([[float(S2)]])

        S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

        # -- Attempt inversion --
        try:
            S_inv = np.linalg.inv(S_pooled)

            # Hotelling's T²
            t2 = (n1 * n2) / (n1 + n2) * diff @ S_inv @ diff

            # Convert to F-statistic
            denom_df = n1 + n2 - p - 1
            if denom_df <= 0:
                # Not enough samples for the F-approximation; fall back
                raise np.linalg.LinAlgError("Insufficient df")

            f_stat = denom_df / ((n1 + n2 - 2) * p) * t2
            p_value = 1.0 - scipy_stats.f.cdf(f_stat, dfn=p, dfd=denom_df)

            return bool(p_value < alpha)

        except np.linalg.LinAlgError:
            # Singular covariance — fall back to dimension-wise Welch t-tests
            # with Bonferroni correction
            logger.debug(
                "Pooled covariance singular; falling back to "
                "dimension-wise Welch t-tests with Bonferroni correction"
            )
            corrected_alpha = alpha / p
            for dim in range(p):
                _, pval = scipy_stats.ttest_ind(
                    X1[:, dim], X2[:, dim], equal_var=False
                )
                if pval < corrected_alpha:
                    return True

            return False


def _normalized_entropy(probabilities: list[float]) -> float:
    """Normalized Shannon entropy in [0, 1]. 0=single outcome, 1=uniform."""
    if not probabilities or len(probabilities) <= 1:
        return 0.0

    # Shannon entropy
    h = 0.0
    for p in probabilities:
        if p > 0:
            h -= p * math.log2(p)

    # Maximum entropy for k outcomes
    h_max = math.log2(len(probabilities))
    if h_max == 0.0:
        return 0.0

    return h / h_max


def _max_tool_chain(steps: list[StepTrace]) -> int:
    """Longest consecutive sequence of tool_call steps."""
    max_depth = 0
    current_depth = 0

    for step in steps:
        if step.action == "tool_call":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        else:
            current_depth = 0

    return max_depth


def _count_backtracks(steps: list[StepTrace]) -> int:
    """Count retry/correction patterns (explicit actions, metadata flags, or consecutive same-tool calls)."""
    count = 0

    for i, step in enumerate(steps):
        # Explicit retry/correction actions
        if step.action in ("retry", "correction", "backtrack"):
            count += 1
            continue

        # Metadata-based detection
        if step.metadata.get("retry") or step.metadata.get("backtrack"):
            count += 1
            continue

        # Same tool called consecutively (implicit retry)
        if (
            i > 0
            and step.action == "tool_call"
            and steps[i - 1].action == "tool_call"
            and step.tool_name == steps[i - 1].tool_name
            and step.tool_name is not None
        ):
            count += 1

    return count


def _to_string(data: Any) -> str:
    """Convert arbitrary output data to a string representation."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (dict, list)):
        return json.dumps(data, default=str)
    return str(data)


def _structure_hash(data: Any) -> str:
    """SHA-256 (first 16 chars) of the output's structural skeleton."""
    if data is None:
        skeleton = "none"
    elif isinstance(data, dict):
        skeleton = "dict:" + ",".join(sorted(data.keys()))
    elif isinstance(data, list):
        type_pattern = [type(item).__name__ for item in data[:20]]
        skeleton = f"list:{len(data)}:" + ",".join(type_pattern)
    elif isinstance(data, str):
        # Bucket by order of magnitude: 0-9, 10-99, 100-999, etc.
        bucket = len(str(len(data)))
        skeleton = f"string:magnitude_{bucket}"
    else:
        skeleton = f"other:{type(data).__name__}"

    return hashlib.sha256(skeleton.encode()).hexdigest()[:16]


def _avg_step_content_length(steps: list[StepTrace]) -> float:
    """Average content length across all steps (proxy for reasoning depth)."""
    if not steps:
        return 0.0

    total = 0
    for step in steps:
        if step.llm_output:
            total += len(step.llm_output)
        elif step.tool_input:
            total += len(json.dumps(step.tool_input, default=str))
        elif step.tool_output is not None:
            total += len(str(step.tool_output))

    return total / len(steps)


def _compute_recovery_rate(steps: list[StepTrace]) -> float:
    """Fraction of error steps followed by a successful step. Returns 0.0 if no errors."""
    error_indices: list[int] = []
    for i, step in enumerate(steps):
        if step.action == "error" or step.metadata.get("error"):
            error_indices.append(i)

    if not error_indices:
        return 0.0

    recovered = 0
    for idx in error_indices:
        # Check if next step exists and is not an error
        if idx + 1 < len(steps):
            next_step = steps[idx + 1]
            if next_step.action != "error" and not next_step.metadata.get("error"):
                recovered += 1

    return recovered / len(error_indices)


def _extract_total_tokens(trace: ExecutionTrace) -> int:
    """Extract total token count from trace-level or step-level metadata."""
    # Check trace-level metadata
    if "total_tokens" in trace.metadata:
        return int(trace.metadata["total_tokens"])

    # Sum step-level tokens
    total = 0
    for step in trace.steps:
        if "tokens" in step.metadata:
            total += int(step.metadata["tokens"])
        elif "prompt_tokens" in step.metadata:
            total += int(step.metadata.get("prompt_tokens", 0))
            total += int(step.metadata.get("completion_tokens", 0))

    return total


def _sequence_complexity(signature: str) -> float:
    """Complexity of a tool sequence: (unique bigrams) / (total bigrams)."""
    if not signature:
        return 0.0

    parts = signature.split("->")
    if len(parts) < 2:
        return 0.0

    bigrams = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
    unique_bigrams = set(bigrams)

    return len(unique_bigrams) / len(bigrams)
