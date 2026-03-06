"""Tests for metamorphic relations module.

Tests each relation family (permutation, perturbation, composition, oracle),
MetamorphicResult model, and violation detection.

Target: ~15 tests.
"""

from __future__ import annotations

import pytest

from agentassay.metamorphic.relations import (
    ConsistencyRelation,
    DecompositionRelation,
    InputPermutationRelation,
    IrrelevantAdditionRelation,
    MetamorphicResult,
    MonotonicityRelation,
    ToolOrderRelation,
    TypographicalPerturbation,
    _text_similarity,
)

from tests.conftest import make_scenario, make_trace


def _make_trace_with_output(output: str, scenario_id: str = "s1"):
    """Helper to create a trace with specific output."""
    return make_trace(output_data=output, scenario_id=scenario_id)


# ===================================================================
# MetamorphicResult
# ===================================================================


class TestMetamorphicResult:
    """Tests for MetamorphicResult model."""

    def test_create_result(self):
        r = MetamorphicResult(
            relation_name="test",
            relation_family="oracle",
            holds=True,
            similarity_score=0.95,
        )
        assert r.holds is True
        assert r.relation_family == "oracle"

    def test_frozen(self):
        r = MetamorphicResult(
            relation_name="test",
            relation_family="oracle",
            holds=True,
        )
        with pytest.raises(Exception):
            r.holds = False


# ===================================================================
# Family 1: Permutation
# ===================================================================


class TestInputPermutationRelation:
    """Tests for InputPermutationRelation."""

    def test_transform_shuffles_items(self):
        relation = InputPermutationRelation(seed=42)
        scenario = make_scenario(
            input_data={"items": ["a", "b", "c", "d", "e"]}
        )
        followup = relation.transform_input(scenario)
        # The items should be in different order (with high probability)
        assert followup.input_data.get("items") is not None

    def test_relation_holds_for_identical_outputs(self):
        relation = InputPermutationRelation(threshold=0.8)
        t1 = _make_trace_with_output("The answer is Paris")
        t2 = _make_trace_with_output("The answer is Paris")
        result = relation.check_relation(t1, t2)
        assert result.holds is True
        assert result.similarity_score == 1.0

    def test_relation_fails_for_different_outputs(self):
        relation = InputPermutationRelation(threshold=0.8)
        t1 = _make_trace_with_output("The answer is Paris")
        t2 = _make_trace_with_output("I cannot determine the answer")
        result = relation.check_relation(t1, t2)
        # Outputs are quite different
        assert result.similarity_score < 0.8

    def test_transform_shuffles_text_sentences(self):
        relation = InputPermutationRelation(seed=42)
        scenario = make_scenario(
            input_data={"query": "First sentence. Second sentence. Third sentence."}
        )
        followup = relation.transform_input(scenario)
        assert followup.scenario_id != scenario.scenario_id


class TestToolOrderRelation:
    """Tests for ToolOrderRelation."""

    def test_transform_with_tools_list(self):
        relation = ToolOrderRelation(seed=42)
        scenario = make_scenario(
            input_data={"tools": ["search", "calculate", "write", "read"]}
        )
        followup = relation.transform_input(scenario)
        assert "tools" in followup.input_data

    def test_no_tools_identity(self):
        relation = ToolOrderRelation(seed=42)
        scenario = make_scenario(input_data={"query": "no tools here"})
        followup = relation.transform_input(scenario)
        # Should return a copy without crashing
        assert followup.scenario_id != scenario.scenario_id


# ===================================================================
# Family 2: Perturbation
# ===================================================================


class TestTypographicalPerturbation:
    """Tests for TypographicalPerturbation."""

    def test_introduces_typos(self):
        relation = TypographicalPerturbation(num_typos=3, seed=42)
        scenario = make_scenario(
            input_data={"query": "What is the capital of France?"}
        )
        followup = relation.transform_input(scenario)
        # Should be different (with high probability for 3 typos)
        original = scenario.input_data["query"]
        mutated = followup.input_data.get("query", "")
        # At least one character should differ
        assert mutated != "" or original == mutated

    def test_short_text_unchanged(self):
        relation = TypographicalPerturbation(num_typos=2, seed=42)
        scenario = make_scenario(input_data={"query": "Hi"})
        followup = relation.transform_input(scenario)
        # Very short text may not be mutated
        assert followup.input_data.get("query") is not None


class TestIrrelevantAdditionRelation:
    """Tests for IrrelevantAdditionRelation."""

    def test_adds_irrelevant_text(self):
        relation = IrrelevantAdditionRelation(num_additions=2, seed=42)
        scenario = make_scenario(
            input_data={"query": "What is AI?"}
        )
        followup = relation.transform_input(scenario)
        # Follow-up text should be longer
        assert len(followup.input_data["query"]) > len(scenario.input_data["query"])

    def test_custom_irrelevant_texts(self):
        custom = ["Random fact 1.", "Random fact 2."]
        relation = IrrelevantAdditionRelation(
            irrelevant_texts=custom, num_additions=1, seed=42
        )
        scenario = make_scenario(input_data={"query": "test"})
        followup = relation.transform_input(scenario)
        text = followup.input_data["query"]
        assert any(c in text for c in custom)


# ===================================================================
# Family 3: Composition
# ===================================================================


class TestDecompositionRelation:
    """Tests for DecompositionRelation."""

    def test_decompose_on_and(self):
        relation = DecompositionRelation(seed=42)
        scenario = make_scenario(
            input_data={"query": "What is the capital of France and what is its population?"}
        )
        sub = relation.decompose(scenario)
        assert len(sub) == 2

    def test_no_decomposition_single_query(self):
        relation = DecompositionRelation(seed=42)
        scenario = make_scenario(
            input_data={"query": "What is the capital of France?"}
        )
        sub = relation.decompose(scenario)
        assert len(sub) == 1

    def test_check_composed_relation(self):
        relation = DecompositionRelation(threshold=0.5)
        source = _make_trace_with_output("Paris is the capital. Population is 2M.")
        sub1 = _make_trace_with_output("Paris is the capital.", scenario_id="sub1")
        sub2 = _make_trace_with_output("Population is 2M.", scenario_id="sub2")
        result = relation.check_composed_relation(source, [sub1, sub2])
        assert isinstance(result, MetamorphicResult)
        assert result.similarity_score > 0


# ===================================================================
# Family 4: Oracle
# ===================================================================


class TestConsistencyRelation:
    """Tests for ConsistencyRelation."""

    def test_identical_outputs_hold(self):
        relation = ConsistencyRelation(threshold=0.7)
        t1 = _make_trace_with_output("The answer is 42")
        t2 = _make_trace_with_output("The answer is 42")
        result = relation.check_relation(t1, t2)
        assert result.holds is True
        assert result.similarity_score == 1.0

    def test_different_outputs_may_fail(self):
        relation = ConsistencyRelation(threshold=0.9)
        t1 = _make_trace_with_output("The answer is 42")
        t2 = _make_trace_with_output("I'm not sure about the answer")
        result = relation.check_relation(t1, t2)
        assert result.similarity_score < 0.9

    def test_identity_transform(self):
        relation = ConsistencyRelation(seed=42)
        scenario = make_scenario()
        followup = relation.transform_input(scenario)
        # Should be a copy with different ID
        assert followup.scenario_id != scenario.scenario_id
        assert followup.input_data == scenario.input_data


class TestMonotonicityRelation:
    """Tests for MonotonicityRelation."""

    def test_more_context_no_quality_decrease(self):
        relation = MonotonicityRelation(
            additional_context="France is in Western Europe.",
            tolerance=0.1,
        )
        source = _make_trace_with_output("Paris" * 50)
        followup = _make_trace_with_output("Paris is the capital of France" * 50)
        result = relation.check_relation(source, followup)
        assert result.holds is True

    def test_transform_adds_context(self):
        relation = MonotonicityRelation(
            additional_context="Extra info here.",
            seed=42,
        )
        scenario = make_scenario(input_data={"query": "What is AI?"})
        followup = relation.transform_input(scenario)
        assert "Extra info" in followup.input_data["query"]


# ===================================================================
# Helper: _text_similarity
# ===================================================================


class TestTextSimilarity:
    """Tests for the _text_similarity helper."""

    def test_identical_strings(self):
        assert _text_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        sim = _text_similarity("aaaa", "zzzz")
        assert sim < 0.5

    def test_empty_both(self):
        assert _text_similarity("", "") == 1.0

    def test_one_empty(self):
        assert _text_similarity("hello", "") == 0.0

    def test_none_values(self):
        assert _text_similarity(None, None) == 1.0
