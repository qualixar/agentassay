# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Composition consistency metamorphic relations.

Family 3 of the metamorphic relation taxonomy. Tests whether agents
produce consistent results when a complex task is decomposed into
subtasks and the sub-results are composed back together.

Relations:
    - ``DecompositionRelation`` -- decomposes a scenario into parts,
      runs each independently, composes results, and checks consistency
      against the direct (un-decomposed) execution.
"""

from __future__ import annotations

import copy
import re
import uuid
from typing import Any

from agentassay.core.models import ExecutionTrace, TestScenario
from agentassay.metamorphic.base import (
    MetamorphicRelation,
    MetamorphicResult,
    _deep_copy_scenario,
    _stringify,
    _text_similarity,
)


class DecompositionRelation(MetamorphicRelation):
    """Tests that decomposed subtasks produce consistent results.

    For complex tasks, an agent should produce the same answer whether
    it solves the problem in one shot or as a series of sub-problems.
    This relation decomposes a complex scenario into simpler parts
    (using a user-provided decomposition function) and checks
    consistency.

    Transform
        Apply ``decompose_fn`` to create a list of simpler scenarios.
        Each sub-scenario is run independently. The user provides a
        ``compose_fn`` to aggregate sub-results into a final answer.

    Relation
        Composed output similarity >= ``threshold`` compared to the
        direct (un-decomposed) output.

    Parameters
    ----------
    decompose_fn
        Callable that takes a ``TestScenario`` and returns a list of
        simpler ``TestScenario`` objects. If not provided, a default
        decomposition is attempted by splitting the query on "and"
        conjunctions.
    compose_fn
        Callable that takes a list of output values (from sub-scenarios)
        and returns a single composed output. If not provided, defaults
        to newline-joined concatenation.
    threshold
        Minimum similarity for the relation to hold.
    seed
        RNG seed.
    """

    def __init__(
        self,
        decompose_fn: Any | None = None,
        compose_fn: Any | None = None,
        threshold: float = 0.7,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="decomposition",
            family="composition",
            description=(
                "Break a complex task into subtasks; composed result "
                "should match the direct result."
            ),
            seed=seed,
        )
        self.decompose_fn = decompose_fn
        self.compose_fn = compose_fn or self._default_compose
        self.threshold = threshold

    @staticmethod
    def _default_compose(outputs: list[Any]) -> str:
        """Default composition: join stringified outputs."""
        return "\n".join(_stringify(o) for o in outputs if o is not None)

    def _default_decompose(self, scenario: TestScenario) -> list[TestScenario]:
        """Default decomposition: split query on 'and' conjunctions.

        This is a heuristic that works for queries like:
        "What is the capital of France and what is its population?"
        -> ["What is the capital of France?", "What is its population?"]
        """
        data = scenario.input_data
        text_key = next(
            (k for k in ("query", "input", "prompt", "text", "question") if k in data),
            None,
        )
        if not text_key or not isinstance(data.get(text_key), str):
            return [scenario]

        text = data[text_key]
        # Split on " and " (case-insensitive) but only at word boundaries
        parts = re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
        if len(parts) <= 1:
            return [scenario]

        sub_scenarios: list[TestScenario] = []
        for i, part in enumerate(parts):
            sub_data = copy.deepcopy(data)
            # Clean up: ensure each part ends with appropriate punctuation
            part = part.strip()
            if not part.endswith((".", "?", "!")):
                part = part + "?"
            sub_data[text_key] = part
            sub_scenario = TestScenario(
                scenario_id=f"{scenario.scenario_id}__sub_{i}_{uuid.uuid4().hex[:6]}",
                name=f"{scenario.name} (sub-task {i})",
                description=f"Sub-task {i} of decomposed scenario",
                input_data=sub_data,
                expected_properties=scenario.expected_properties,
                tags=[*scenario.tags, "decomposed", f"sub_{i}"],
                timeout_seconds=scenario.timeout_seconds,
                metadata={**scenario.metadata, "parent_scenario": scenario.scenario_id},
            )
            sub_scenarios.append(sub_scenario)

        return sub_scenarios

    def decompose(self, scenario: TestScenario) -> list[TestScenario]:
        """Decompose a scenario into sub-scenarios.

        Uses the user-provided ``decompose_fn`` if available, otherwise
        falls back to the default heuristic.
        """
        if self.decompose_fn is not None:
            return self.decompose_fn(scenario)
        return self._default_decompose(scenario)

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Return the first sub-scenario as the follow-up.

        The ``MetamorphicRunner`` calls ``transform_input`` to get a
        single follow-up scenario. For decomposition, the runner uses
        ``decompose()`` directly via ``test_decomposition()``.
        This method exists to satisfy the abstract interface and returns
        the first sub-task.
        """
        sub_scenarios = self.decompose(scenario)
        if sub_scenarios:
            return sub_scenarios[0]
        return _deep_copy_scenario(scenario)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check similarity between direct output and composed output.

        For full decomposition testing, use ``check_composed_relation``
        which accepts the list of sub-trace outputs.
        """
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
                "transform": "decomposition",
                "mode": "single_subtask",
            },
        )

    def check_composed_relation(
        self,
        source_trace: ExecutionTrace,
        sub_traces: list[ExecutionTrace],
    ) -> MetamorphicResult:
        """Check the full decomposition relation.

        Composes the outputs of all sub-traces using ``compose_fn``
        and compares the composed result against the direct (source)
        output.

        Parameters
        ----------
        source_trace
            Execution trace from the direct (un-decomposed) run.
        sub_traces
            Execution traces from each sub-scenario run.

        Returns
        -------
        MetamorphicResult
            Whether the composed output is similar enough to the
            direct output.
        """
        sub_outputs = [t.output_data for t in sub_traces]
        composed_output = self.compose_fn(sub_outputs)
        sim = _text_similarity(source_trace.output_data, composed_output)
        return MetamorphicResult(
            relation_name=self.name,
            relation_family=self.family,
            holds=sim >= self.threshold,
            source_output=source_trace.output_data,
            followup_output=composed_output,
            similarity_score=sim,
            details={
                "threshold": self.threshold,
                "transform": "decomposition",
                "mode": "full_composition",
                "num_subtasks": len(sub_traces),
                "sub_outputs": sub_outputs,
            },
        )
