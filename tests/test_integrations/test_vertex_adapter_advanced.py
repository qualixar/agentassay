"""Advanced tests for the Google Vertex AI Agents adapter.

Covers token usage edge cases, grounding metadata edge cases,
and complex multi-part response scenarios.

All tests use mock objects -- ``google-cloud-aiplatform`` is NOT required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from agentassay.integrations.vertex_adapter import (
    VertexAIAgentsAdapter,
)

# Import helpers from the core test module
from tests.test_integrations.test_vertex_adapter import (
    _FAKE_VERTEXAI,
    _make_candidate,
    _make_function_call_part,
    _make_function_response_part,
    _make_response,
    _make_text_part,
    _make_usage_metadata,
)

# ===================================================================
# Tests: Token usage edge cases
# ===================================================================


class TestVertexAdapterTokenUsage:
    """Tests for token usage extraction and cost estimation edge cases."""

    def test_zero_total_auto_computed(self) -> None:
        """When total_token_count is 0, it's computed from prompt + completion."""
        usage = _make_usage_metadata(100, 50, 0)
        response = MagicMock()
        response.usage_metadata = usage

        result = VertexAIAgentsAdapter._extract_token_usage(response)

        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_none_usage_metadata(self) -> None:
        """None usage_metadata returns all zeros."""
        response = MagicMock()
        response.usage_metadata = None

        result = VertexAIAgentsAdapter._extract_token_usage(response)

        assert result == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_zero_tokens_zero_cost(self) -> None:
        """Zero tokens produce zero cost."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        cost = adapter._estimate_cost(
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )

        assert cost == 0.0


# ===================================================================
# Tests: Grounding metadata edge cases
# ===================================================================


class TestVertexAdapterGroundingEdgeCases:
    """Tests for grounding metadata edge cases."""

    def test_grounding_with_retrieved_context(self) -> None:
        """Grounding chunk with retrieved_context (not web) is handled."""
        meta = MagicMock()

        chunk = MagicMock()
        chunk.web = None
        retrieved = MagicMock()
        retrieved.uri = "gs://bucket/doc.pdf"
        retrieved.title = "Internal Doc"
        chunk.retrieved_context = retrieved

        meta.grounding_chunks = [chunk]
        meta.grounding_supports = [MagicMock()]
        meta.retrieval_metadata = None

        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        step = adapter._extract_grounding_step(meta, 0, 10.0, 0)

        assert step.action == "retrieval"
        sources = step.metadata["vertex_grounding_sources"]
        assert len(sources) == 1
        assert sources[0]["uri"] == "gs://bucket/doc.pdf"
        assert sources[0]["title"] == "Internal Doc"

    def test_grounding_no_retrieval_metadata(self) -> None:
        """Grounding without retrieval_metadata sets quality_score to None."""
        meta = MagicMock()
        meta.grounding_chunks = []
        meta.grounding_supports = []
        meta.retrieval_metadata = None

        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        step = adapter._extract_grounding_step(meta, 0, 5.0, 0)

        assert step.metadata["vertex_grounding_quality_score"] is None


# ===================================================================
# Tests: Complex multi-part scenarios
# ===================================================================


class TestVertexAdapterComplexScenario:
    """Integration-style tests with complex multi-part responses."""

    def test_full_tool_use_cycle(self) -> None:
        """Simulates a full cycle: text -> function_call -> function_response -> text."""
        parts = [
            _make_text_part("I'll look up the weather."),
            _make_function_call_part("get_weather", {"city": "Paris"}),
            _make_function_response_part("get_weather", {"temp": 18, "condition": "cloudy"}),
            _make_text_part("The weather in Paris is 18C and cloudy."),
        ]
        candidate = _make_candidate(parts)
        usage = _make_usage_metadata(300, 100, 400)
        response = _make_response(
            [candidate],
            usage,
            text="I'll look up the weather.\nThe weather in Paris is 18C and cloudy.",
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model._model_name = None
        mock_model.model_name = None

        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            trace = adapter.run({"query": "What's the weather in Paris?"})

        assert trace.success is True
        assert len(trace.steps) == 4
        assert trace.steps[0].action == "llm_response"
        assert trace.steps[1].action == "tool_call"
        assert trace.steps[1].tool_name == "get_weather"
        assert trace.steps[2].action == "observation"
        assert trace.steps[3].action == "llm_response"

        # Verify monotonic step indices
        indices = [s.step_index for s in trace.steps]
        assert indices == [0, 1, 2, 3]

        # Verify cost was calculated
        assert trace.total_cost_usd > 0.0

    def test_candidate_with_no_content(self) -> None:
        """Candidate with content=None is gracefully skipped."""
        good_part = _make_text_part("Good answer")
        good_candidate = _make_candidate([good_part])

        bad_candidate = MagicMock()
        bad_candidate.content = None
        bad_candidate.grounding_metadata = None

        response = _make_response([bad_candidate, good_candidate], text="Good answer")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model._model_name = None
        mock_model.model_name = None

        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            trace = adapter.run({"query": "test"})

        assert trace.success is True
        # Only the good candidate's part is extracted
        assert len(trace.steps) == 1
        assert trace.steps[0].llm_output == "Good answer"

    def test_function_call_with_non_dict_args(self) -> None:
        """Function call with args that cannot be dict() produces raw string."""
        fc = MagicMock()
        fc.name = "search"

        # Create an object where dict() raises TypeError.
        # MagicMock implements __iter__ so dict() succeeds on it.
        # Use a real non-iterable, non-mapping object instead.
        class _NonDictArgs:
            """Object that causes dict() to raise TypeError."""

            def __str__(self) -> str:
                return "non-dict-args-value"

        fc.args = _NonDictArgs()

        part = MagicMock()
        part.function_call = fc
        part.function_response = None
        part.text = None
        part.inline_data = None

        candidate = _make_candidate([part])
        response = _make_response([candidate])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model._model_name = None
        mock_model.model_name = None

        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            trace = adapter.run({"query": "test"})

        assert trace.success is True
        assert trace.steps[0].action == "tool_call"
        assert "raw" in trace.steps[0].tool_input
