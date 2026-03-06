# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Permutation invariance metamorphic relations.

Family 1 of the metamorphic relation taxonomy. Tests whether agents
produce semantically equivalent outputs when input ordering is changed.

Relations:
    - ``InputPermutationRelation`` -- shuffles items or sentences in the input.
    - ``ToolOrderRelation`` -- shuffles the list of available tools.
"""

from __future__ import annotations

import re

from agentassay.core.models import ExecutionTrace, TestScenario
from agentassay.metamorphic.base import (
    MetamorphicRelation,
    MetamorphicResult,
    _deep_copy_scenario,
    _text_similarity,
)


class InputPermutationRelation(MetamorphicRelation):
    """Tests invariance to the ordering of items in the input.

    Many agents receive lists (search results, document sections, tool
    outputs). A reliable agent should produce semantically equivalent
    answers regardless of the order those items are presented in.

    Transform
        Shuffle ``input_data["items"]`` if present. Otherwise, split
        the ``"query"`` or ``"input"`` text on sentence boundaries and
        shuffle the sentences.

    Relation
        Output similarity >= ``threshold`` (default 0.8).

    Parameters
    ----------
    threshold
        Minimum similarity score (0.0-1.0) for the relation to hold.
    seed
        RNG seed for reproducible shuffles.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="input_permutation",
            family="permutation",
            description=(
                "Reorder items/sections in the input; output should be semantically equivalent."
            ),
            seed=seed,
        )
        self.threshold = threshold

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Shuffle list items or text sections in the input."""
        followup = _deep_copy_scenario(scenario)
        data = followup.input_data

        # Strategy 1: shuffle an explicit "items" list
        if "items" in data and isinstance(data["items"], list) and len(data["items"]) > 1:
            items = list(data["items"])
            self._rng.shuffle(items)
            data["items"] = items
            return _deep_copy_scenario(scenario, input_data=data)

        # Strategy 2: shuffle sentences in "query" or "input" text
        text_key = next(
            (k for k in ("query", "input", "prompt", "text", "question") if k in data),
            None,
        )
        if text_key and isinstance(data[text_key], str):
            text = data[text_key]
            # Split on sentence boundaries (period/exclamation/question + space)
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 1:
                self._rng.shuffle(sentences)
                data[text_key] = " ".join(sentences)
                return _deep_copy_scenario(scenario, input_data=data)

        # No permutable content found -- return identity (no-op)
        return _deep_copy_scenario(scenario)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check output similarity after input permutation."""
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
                "transform": "input_item_permutation",
            },
        )


class ToolOrderRelation(MetamorphicRelation):
    """Tests invariance to the ordering of available tools.

    When an agent is presented with a list of tools, the order should
    not influence which tools it chooses or what final answer it
    produces. Ordering bias is a known LLM failure mode (position
    bias / primacy effect).

    Transform
        Shuffle ``input_data["tools"]`` list.

    Relation
        Final output similarity >= ``threshold``.

    Parameters
    ----------
    threshold
        Minimum similarity for the relation to hold.
    seed
        RNG seed for reproducible shuffles.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="tool_order",
            family="permutation",
            description=(
                "Reorder available tools; agent should reach the same "
                "conclusion regardless of tool presentation order."
            ),
            seed=seed,
        )
        self.threshold = threshold

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Shuffle the tools list in input_data."""
        followup = _deep_copy_scenario(scenario)
        data = followup.input_data

        tools_key = next(
            (k for k in ("tools", "available_tools", "tool_list") if k in data),
            None,
        )
        if tools_key and isinstance(data[tools_key], list) and len(data[tools_key]) > 1:
            tools = list(data[tools_key])
            self._rng.shuffle(tools)
            data[tools_key] = tools
            return _deep_copy_scenario(scenario, input_data=data)

        # No tools list found -- return identity
        return _deep_copy_scenario(scenario)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check output similarity after tool reordering."""
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
                "transform": "tool_order_permutation",
                "source_tools_used": sorted(source_trace.tools_used),
                "followup_tools_used": sorted(followup_trace.tools_used),
            },
        )
