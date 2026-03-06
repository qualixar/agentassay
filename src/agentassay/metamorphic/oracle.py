"""Oracle-based metamorphic relations.

Family 4 of the metamorphic relation taxonomy. Tests known invariants
that must hold across agent executions without requiring ground-truth
labels.

Relations:
    - ``ConsistencyRelation`` -- same input, same output (self-consistency).
    - ``MonotonicityRelation`` -- more information leads to equal or
      better results (monotonicity property).
"""

from __future__ import annotations

from typing import Any

from agentassay.core.models import ExecutionTrace, TestScenario
from agentassay.metamorphic.base import (
    MetamorphicRelation,
    MetamorphicResult,
    _deep_copy_scenario,
    _exact_match,
    _stringify,
    _text_similarity,
)


class ConsistencyRelation(MetamorphicRelation):
    """Tests self-consistency: same input, same output.

    The simplest metamorphic relation: if you run the same input
    twice, the outputs should be consistent. For deterministic systems
    this is trivially true, but for stochastic LLM agents, consistency
    violations reveal excessive non-determinism.

    Transform
        Identity (return the same scenario unchanged).

    Relation
        Output similarity >= ``threshold``.

    Parameters
    ----------
    threshold
        Minimum similarity for the relation to hold. Default 0.7 --
        lower than other relations because some variation in wording
        is expected for LLM-based agents.
    seed
        RNG seed (unused for this relation but accepted for API
        consistency).
    """

    def __init__(
        self,
        threshold: float = 0.7,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="consistency",
            family="oracle",
            description=(
                "Run the same input multiple times; outputs should be "
                "consistent above a threshold."
            ),
            seed=seed,
        )
        self.threshold = threshold

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Identity transform -- return the same scenario (new ID)."""
        return _deep_copy_scenario(scenario)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check output consistency between two runs of the same input."""
        sim = _text_similarity(source_trace.output_data, followup_trace.output_data)
        return MetamorphicResult(
            relation_name=self.name,
            relation_family=self.family,
            holds=sim >= self.threshold,
            source_output=source_trace.output_data,
            followup_output=followup_trace.output_data,
            similarity_score=sim,
            details={
                "threshold": self.threshold,
                "transform": "identity",
                "exact_match": _exact_match(
                    source_trace.output_data, followup_trace.output_data
                ),
            },
        )


class MonotonicityRelation(MetamorphicRelation):
    """Tests that more information leads to equal or better results.

    For tasks where additional relevant context should help, adding
    information should not *decrease* the quality of the output. This
    is the monotonicity property.

    Transform
        Enrich the input with additional relevant context from
        ``additional_context``.

    Relation
        Follow-up output quality >= source output quality. Quality is
        measured by the ``quality_fn`` if provided, otherwise falls
        back to output length as a proxy (longer = more detailed).

    Parameters
    ----------
    additional_context
        Text to append as additional context.
    quality_fn
        Optional callable that takes an output value and returns a
        float in [0, 1]. If not provided, uses normalized output
        length as a rough proxy.
    tolerance
        Acceptable quality decrease before flagging a violation.
        Default 0.1 (10% decrease is tolerated).
    seed
        RNG seed.
    """

    def __init__(
        self,
        additional_context: str = "",
        quality_fn: Any | None = None,
        tolerance: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="monotonicity",
            family="oracle",
            description=(
                "Adding relevant context should not decrease output "
                "quality (monotonicity property)."
            ),
            seed=seed,
        )
        self.additional_context = additional_context
        self.quality_fn = quality_fn
        self.tolerance = tolerance

    def _default_quality(self, output: Any) -> float:
        """Heuristic quality measure based on output length.

        Longer, more detailed outputs are assumed to be higher quality.
        This is a rough proxy -- users should provide ``quality_fn``
        for domain-specific quality measures.

        Caps at 1.0 for outputs >= 500 characters.
        """
        text = _stringify(output)
        if not text:
            return 0.0
        # Normalize: 0 chars -> 0.0, 500+ chars -> 1.0
        return min(len(text) / 500.0, 1.0)

    def _measure_quality(self, output: Any) -> float:
        """Measure output quality using quality_fn or default heuristic."""
        if self.quality_fn is not None:
            score = self.quality_fn(output)
            return max(0.0, min(1.0, float(score)))
        return self._default_quality(output)

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Enrich the input with additional relevant context."""
        if not self.additional_context:
            return _deep_copy_scenario(scenario)

        followup = _deep_copy_scenario(scenario)
        data = followup.input_data

        text_key = next(
            (k for k in ("query", "input", "prompt", "text", "question") if k in data),
            None,
        )

        if text_key and isinstance(data[text_key], str):
            data[text_key] = (
                data[text_key] + "\n\nAdditional context: " + self.additional_context
            )
            return _deep_copy_scenario(scenario, input_data=data)

        # If no text key, add as a separate field
        data["additional_context"] = self.additional_context
        return _deep_copy_scenario(scenario, input_data=data)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check that follow-up quality >= source quality - tolerance."""
        source_quality = self._measure_quality(source_trace.output_data)
        followup_quality = self._measure_quality(followup_trace.output_data)

        holds = followup_quality >= (source_quality - self.tolerance)
        # Similarity score: 1.0 if quality maintained/improved, scaled down otherwise
        if source_quality == 0.0:
            sim = 1.0 if followup_quality >= 0.0 else 0.0
        else:
            sim = min(followup_quality / source_quality, 1.0) if source_quality > 0 else 1.0

        return MetamorphicResult(
            relation_name=self.name,
            relation_family=self.family,
            holds=holds,
            source_output=source_trace.output_data,
            followup_output=followup_trace.output_data,
            similarity_score=sim,
            details={
                "source_quality": source_quality,
                "followup_quality": followup_quality,
                "quality_delta": followup_quality - source_quality,
                "tolerance": self.tolerance,
                "transform": "context_enrichment",
                "used_custom_quality_fn": self.quality_fn is not None,
            },
        )
