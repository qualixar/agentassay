"""Tests for the Google Vertex AI Agents adapter.

All tests use mock objects -- ``google-cloud-aiplatform`` is NOT required.
Tests verify:
- Adapter construction with various parameter combinations.
- Install check raises ``FrameworkNotInstalledError``.
- ``run()`` with mocked text, function_call, and mixed responses.
- Grounding metadata extraction into retrieval steps.
- Token usage metadata extraction and cost estimation.
- Error handling (API errors produce ``success=False`` traces).
- ``to_callable()`` returns a callable that delegates to ``run()``.
- ``get_config()`` produces a valid ``AgentConfig``.
- ``scenario_id`` extraction from input data.
- Empty response and multiple-candidate handling.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from agentassay.core.models import AgentConfig, ExecutionTrace
from agentassay.integrations.base import FrameworkNotInstalledError

# ---------------------------------------------------------------------------
# We must be able to import the adapter WITHOUT google-cloud-aiplatform.
# The module uses lazy imports only at run-time, so top-level import works.
# ---------------------------------------------------------------------------
from agentassay.integrations.vertex_adapter import (
    _DEFAULT_PRICING,
    VertexAIAgentsAdapter,
    _check_vertex_installed,
)

# ===================================================================
# Helpers -- mock Vertex AI response objects
# ===================================================================


def _make_text_part(text: str) -> MagicMock:
    """Create a mock Part with a text attribute."""
    part = MagicMock()
    part.text = text
    part.function_call = None
    part.function_response = None
    part.inline_data = None
    return part


def _make_function_call_part(name: str, args: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock Part with a function_call attribute."""
    fc = MagicMock()
    fc.name = name
    fc.args = args or {}

    part = MagicMock()
    part.function_call = fc
    part.function_response = None
    part.text = None
    part.inline_data = None
    return part


def _make_function_response_part(name: str, response: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock Part with a function_response attribute."""
    fr = MagicMock()
    fr.name = name
    fr.response = response or {}

    part = MagicMock()
    part.function_response = fr
    part.function_call = None
    part.text = None
    part.inline_data = None
    return part


def _make_inline_data_part(mime_type: str = "image/png") -> MagicMock:
    """Create a mock Part with an inline_data attribute."""
    inline = MagicMock()
    inline.mime_type = mime_type

    part = MagicMock()
    part.inline_data = inline
    part.function_call = None
    part.function_response = None
    part.text = None
    return part


def _make_candidate(
    parts: list[MagicMock],
    grounding_metadata: Any = None,
) -> MagicMock:
    """Create a mock Candidate with content.parts."""
    content = MagicMock()
    content.parts = parts

    candidate = MagicMock()
    candidate.content = content
    candidate.grounding_metadata = grounding_metadata
    return candidate


def _make_usage_metadata(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
) -> MagicMock:
    """Create a mock usage_metadata object."""
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = total_tokens
    return usage


def _make_response(
    candidates: list[MagicMock] | None = None,
    usage_metadata: Any = None,
    text: str | None = None,
) -> MagicMock:
    """Create a mock GenerateContentResponse."""
    response = MagicMock()
    response.candidates = candidates or []
    response.usage_metadata = usage_metadata

    if text is not None:
        type(response).text = PropertyMock(return_value=text)
    else:
        # If no text shortcut, make .text raise ValueError (like the real SDK
        # does when the response contains non-text parts)
        if candidates:
            # Try to extract text from parts
            text_parts = []
            for c in candidates:
                for p in getattr(getattr(c, "content", None), "parts", None) or []:
                    if getattr(p, "text", None):
                        text_parts.append(p.text)
            if text_parts:
                type(response).text = PropertyMock(return_value="\n".join(text_parts))
            else:
                type(response).text = PropertyMock(side_effect=ValueError("No text parts"))
        else:
            type(response).text = PropertyMock(side_effect=ValueError("No candidates"))

    return response


def _make_grounding_metadata(
    sources: list[dict[str, str]] | None = None,
    supports_count: int = 2,
    quality_score: float = 0.85,
) -> MagicMock:
    """Create a mock GroundingMetadata."""
    meta = MagicMock()

    # Build grounding chunks
    chunks = []
    for src in sources or []:
        chunk = MagicMock()
        web = MagicMock()
        web.uri = src.get("uri", "")
        web.title = src.get("title", "")
        chunk.web = web
        chunk.retrieved_context = None
        chunks.append(chunk)

    meta.grounding_chunks = chunks

    # Build supports
    meta.grounding_supports = [MagicMock() for _ in range(supports_count)]

    # Retrieval metadata
    retrieval_meta = MagicMock()
    retrieval_meta.google_search_dynamic_retrieval_score = quality_score
    meta.retrieval_metadata = retrieval_meta

    return meta


# Fake vertexai module for patching
_FAKE_VERTEXAI = ModuleType("vertexai")


# ===================================================================
# Tests
# ===================================================================


class TestVertexAdapterCreation:
    """Tests for adapter construction and attribute resolution."""

    def test_basic_creation(self) -> None:
        """Adapter can be created with a mock model and explicit model name."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")
        assert adapter.framework == "vertex"
        assert adapter.model == "gemini-2.0-flash"
        assert adapter.agent_name == "vertex-agent"

    def test_model_inference_from_object(self) -> None:
        """Model name is inferred from generative_model._model_name."""
        mock_model = MagicMock()
        mock_model._model_name = "publishers/google/models/gemini-2.0-pro"
        adapter = VertexAIAgentsAdapter(mock_model)
        # Should strip prefix and use the last segment
        assert adapter.model == "gemini-2.0-pro"

    def test_model_inference_no_prefix(self) -> None:
        """Model name without slash prefix is used as-is."""
        mock_model = MagicMock()
        mock_model._model_name = "gemini-1.5-flash"
        mock_model.model_name = None
        adapter = VertexAIAgentsAdapter(mock_model)
        assert adapter.model == "gemini-1.5-flash"

    def test_custom_agent_name(self) -> None:
        """Custom agent_name is preserved."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(
            mock_model,
            model="gemini-2.0-flash",
            agent_name="my-vertex-agent",
        )
        assert adapter.agent_name == "my-vertex-agent"

    def test_metadata_and_project(self) -> None:
        """Project ID, location, and metadata are stored."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(
            mock_model,
            model="gemini-2.0-flash",
            project_id="my-project-123",
            location="europe-west4",
            metadata={"env": "test"},
        )
        assert adapter._project_id == "my-project-123"
        assert adapter._location == "europe-west4"
        assert adapter._metadata == {"env": "test"}

    def test_tools_stored(self) -> None:
        """Tools list is stored for later use in generate_content."""
        mock_model = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash", tools=mock_tools)
        assert adapter._tools is mock_tools
        assert len(adapter._tools) == 2

    def test_repr(self) -> None:
        """Repr includes framework, model, and agent name."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")
        r = repr(adapter)
        assert "VertexAIAgentsAdapter" in r
        assert "vertex" in r
        assert "gemini-2.0-flash" in r

    def test_default_pricing_applied(self) -> None:
        """Default pricing is set when no custom pricing provided."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")
        assert adapter._pricing["prompt_per_token"] == _DEFAULT_PRICING["prompt_per_token"]
        assert adapter._pricing["completion_per_token"] == _DEFAULT_PRICING["completion_per_token"]

    def test_custom_pricing_override(self) -> None:
        """User-supplied pricing overrides defaults."""
        mock_model = MagicMock()
        custom_pricing = {"prompt_per_token": 0.001, "completion_per_token": 0.002}
        adapter = VertexAIAgentsAdapter(
            mock_model,
            model="gemini-2.0-flash",
            metadata={"pricing": custom_pricing},
        )
        assert adapter._pricing["prompt_per_token"] == 0.001
        assert adapter._pricing["completion_per_token"] == 0.002


class TestCheckVertexInstalled:
    """Tests for the _check_vertex_installed guard."""

    def test_raises_when_not_installed(self) -> None:
        """FrameworkNotInstalledError raised when vertexai is absent."""
        with patch.dict(sys.modules, {"vertexai": None}):
            with pytest.raises(FrameworkNotInstalledError, match="google-cloud-aiplatform"):
                _check_vertex_installed()

    def test_passes_when_installed(self) -> None:
        """No error when vertexai is importable."""
        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            _check_vertex_installed()  # Should not raise


class TestVertexAdapterRun:
    """Tests for the run() method with mocked generate_content."""

    def _run_with_response(
        self,
        response: MagicMock,
        input_data: dict[str, Any] | None = None,
        **adapter_kwargs: Any,
    ) -> ExecutionTrace:
        """Helper: create adapter with mocked model and run."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model._model_name = None
        mock_model.model_name = None

        adapter = VertexAIAgentsAdapter(
            mock_model,
            model=adapter_kwargs.pop("model", "gemini-2.0-flash"),
            **adapter_kwargs,
        )

        data = input_data or {"query": "test query"}
        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            return adapter.run(data)

    def test_text_response(self) -> None:
        """Text-only response produces a single llm_response step."""
        text_part = _make_text_part("Paris is the capital of France.")
        candidate = _make_candidate([text_part])
        usage = _make_usage_metadata(100, 20, 120)
        response = _make_response([candidate], usage, text="Paris is the capital of France.")

        trace = self._run_with_response(response)

        assert trace.success is True
        assert trace.framework == "vertex"
        assert trace.model == "gemini-2.0-flash"
        assert len(trace.steps) == 1
        assert trace.steps[0].action == "llm_response"
        assert "Paris" in trace.steps[0].llm_output
        assert trace.output_data == "Paris is the capital of France."
        assert trace.total_cost_usd > 0.0

    def test_function_call_response(self) -> None:
        """Function call part produces a tool_call step."""
        fc_part = _make_function_call_part("get_weather", {"city": "Tokyo", "unit": "celsius"})
        candidate = _make_candidate([fc_part])
        response = _make_response([candidate])

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.action == "tool_call"
        assert step.tool_name == "get_weather"
        assert step.tool_input == {"city": "Tokyo", "unit": "celsius"}

    def test_mixed_response_text_and_function_call(self) -> None:
        """Response with both text and function_call produces two steps."""
        text_part = _make_text_part("Let me check the weather.")
        fc_part = _make_function_call_part("get_weather", {"city": "NYC"})
        candidate = _make_candidate([text_part, fc_part])
        response = _make_response([candidate], text="Let me check the weather.")

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 2
        assert trace.steps[0].action == "llm_response"
        assert trace.steps[1].action == "tool_call"
        assert trace.steps[0].step_index == 0
        assert trace.steps[1].step_index == 1

    def test_function_response_part(self) -> None:
        """Function response part produces an observation step."""
        fr_part = _make_function_response_part(
            "get_weather", {"temperature": 25, "condition": "sunny"}
        )
        candidate = _make_candidate([fr_part])
        response = _make_response([candidate])

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.action == "observation"
        assert step.tool_name == "get_weather"

    def test_grounding_metadata_extraction(self) -> None:
        """Grounding metadata produces a retrieval step."""
        text_part = _make_text_part("According to sources...")
        grounding = _make_grounding_metadata(
            sources=[
                {"uri": "https://example.com/1", "title": "Source One"},
                {"uri": "https://example.com/2", "title": "Source Two"},
            ],
            supports_count=3,
            quality_score=0.92,
        )
        candidate = _make_candidate([text_part], grounding_metadata=grounding)
        response = _make_response([candidate], text="According to sources...")

        trace = self._run_with_response(response)

        assert trace.success is True
        # 1 text step + 1 grounding step
        assert len(trace.steps) == 2
        retrieval_step = trace.steps[1]
        assert retrieval_step.action == "retrieval"
        assert retrieval_step.tool_name == "vertex_grounding"
        assert "2 source" in retrieval_step.llm_output
        meta = retrieval_step.metadata
        assert len(meta["vertex_grounding_sources"]) == 2
        assert meta["vertex_grounding_supports_count"] == 3
        assert meta["vertex_grounding_quality_score"] == 0.92

    def test_token_usage_extraction(self) -> None:
        """Token usage metadata is extracted into trace metadata."""
        text_part = _make_text_part("Hello")
        candidate = _make_candidate([text_part])
        usage = _make_usage_metadata(500, 200, 700)
        response = _make_response([candidate], usage, text="Hello")

        trace = self._run_with_response(response)

        token_usage = trace.metadata.get("vertex_token_usage", {})
        assert token_usage["prompt_tokens"] == 500
        assert token_usage["completion_tokens"] == 200
        assert token_usage["total_tokens"] == 700

    def test_cost_calculation(self) -> None:
        """Cost is estimated from token counts and pricing."""
        text_part = _make_text_part("Answer")
        candidate = _make_candidate([text_part])
        usage = _make_usage_metadata(1000, 500, 1500)
        response = _make_response([candidate], usage, text="Answer")

        trace = self._run_with_response(response)

        expected_cost = (
            1000 * _DEFAULT_PRICING["prompt_per_token"]
            + 500 * _DEFAULT_PRICING["completion_per_token"]
        )
        assert abs(trace.total_cost_usd - expected_cost) < 1e-12

    def test_cost_with_custom_pricing(self) -> None:
        """Custom pricing in metadata overrides default pricing."""
        text_part = _make_text_part("Answer")
        candidate = _make_candidate([text_part])
        usage = _make_usage_metadata(1000, 500, 1500)
        response = _make_response([candidate], usage, text="Answer")

        trace = self._run_with_response(
            response,
            metadata={"pricing": {"prompt_per_token": 0.01, "completion_per_token": 0.02}},
        )

        expected_cost = 1000 * 0.01 + 500 * 0.02
        assert abs(trace.total_cost_usd - expected_cost) < 1e-12

    def test_error_handling(self) -> None:
        """API errors produce success=False with error message."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = RuntimeError("API quota exceeded")
        mock_model._model_name = None
        mock_model.model_name = None

        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            trace = adapter.run({"query": "test"})

        assert trace.success is False
        assert "RuntimeError" in trace.error
        assert "API quota exceeded" in trace.error
        assert trace.steps == []
        assert trace.output_data is None

    def test_framework_not_installed_propagates(self) -> None:
        """FrameworkNotInstalledError is not swallowed by error handler."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": None}):
            with pytest.raises(FrameworkNotInstalledError):
                adapter.run({"query": "test"})

    def test_scenario_id_extraction(self) -> None:
        """scenario_id is extracted from input_data."""
        text_part = _make_text_part("response")
        candidate = _make_candidate([text_part])
        response = _make_response([candidate], text="response")

        trace = self._run_with_response(
            response,
            input_data={"query": "test", "scenario_id": "ecommerce-001"},
        )

        assert trace.scenario_id == "ecommerce-001"

    def test_default_scenario_id(self) -> None:
        """Default scenario_id is 'default' when not provided."""
        text_part = _make_text_part("response")
        candidate = _make_candidate([text_part])
        response = _make_response([candidate], text="response")

        trace = self._run_with_response(response, input_data={"query": "test"})

        assert trace.scenario_id == "default"

    def test_empty_response_handling(self) -> None:
        """Empty response (no candidates) produces an observation step."""
        response = _make_response(candidates=[], text=None)

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].action == "observation"
        assert "empty response" in trace.steps[0].llm_output.lower()

    def test_empty_candidates_none(self) -> None:
        """Response with candidates=None produces an observation step."""
        response = MagicMock()
        response.candidates = None
        response.usage_metadata = None
        type(response).text = PropertyMock(side_effect=ValueError("No text"))

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].action == "observation"

    def test_multiple_candidates(self) -> None:
        """Multiple candidates produce steps from all of them."""
        text_part1 = _make_text_part("Answer A")
        text_part2 = _make_text_part("Answer B")
        cand1 = _make_candidate([text_part1])
        cand2 = _make_candidate([text_part2])
        response = _make_response([cand1, cand2], text="Answer A\nAnswer B")

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 2
        assert trace.steps[0].action == "llm_response"
        assert trace.steps[1].action == "llm_response"
        assert trace.steps[0].llm_output == "Answer A"
        assert trace.steps[1].llm_output == "Answer B"
        # Step indices are monotonically increasing
        assert trace.steps[0].step_index == 0
        assert trace.steps[1].step_index == 1

    def test_inline_data_part(self) -> None:
        """Inline data part produces an observation step with mime type."""
        inline_part = _make_inline_data_part("image/jpeg")
        candidate = _make_candidate([inline_part])
        response = _make_response([candidate])

        trace = self._run_with_response(response)

        assert trace.success is True
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.action == "observation"
        assert "image/jpeg" in step.llm_output
        assert step.metadata["vertex_mime_type"] == "image/jpeg"

    def test_tools_passed_to_generate_content(self) -> None:
        """Tools are forwarded to generate_content() when provided."""
        mock_model = MagicMock()
        text_part = _make_text_part("Using tools")
        candidate = _make_candidate([text_part])
        response = _make_response([candidate], text="Using tools")
        mock_model.generate_content.return_value = response
        mock_model._model_name = None
        mock_model.model_name = None

        mock_tools = [MagicMock(name="tool1")]
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash", tools=mock_tools)

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            adapter.run({"query": "test"})

        mock_model.generate_content.assert_called_once()
        call_kwargs = mock_model.generate_content.call_args
        assert call_kwargs.kwargs.get("tools") is mock_tools

    def test_no_tools_not_passed(self) -> None:
        """When no tools provided, tools kwarg is not passed."""
        mock_model = MagicMock()
        text_part = _make_text_part("No tools")
        candidate = _make_candidate([text_part])
        response = _make_response([candidate], text="No tools")
        mock_model.generate_content.return_value = response
        mock_model._model_name = None
        mock_model.model_name = None

        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            adapter.run({"query": "test"})

        call_kwargs = mock_model.generate_content.call_args
        assert "tools" not in call_kwargs.kwargs

    def test_project_id_in_trace_metadata(self) -> None:
        """Project ID and location appear in trace metadata."""
        text_part = _make_text_part("Hello")
        candidate = _make_candidate([text_part])
        response = _make_response([candidate], text="Hello")

        trace = self._run_with_response(response, project_id="test-project", location="asia-east1")

        assert trace.metadata["vertex_project_id"] == "test-project"
        assert trace.metadata["vertex_location"] == "asia-east1"

    def test_no_usage_metadata_zero_cost(self) -> None:
        """When usage_metadata is None, cost is 0.0."""
        text_part = _make_text_part("Hello")
        candidate = _make_candidate([text_part])
        response = _make_response([candidate], usage_metadata=None, text="Hello")

        trace = self._run_with_response(response)

        assert trace.total_cost_usd == 0.0
        token_usage = trace.metadata.get("vertex_token_usage", {})
        assert token_usage["prompt_tokens"] == 0
        assert token_usage["completion_tokens"] == 0


class TestVertexAdapterBuildPrompt:
    """Tests for the _build_user_prompt static method."""

    def test_query_key(self) -> None:
        """'query' key is extracted first."""
        prompt = VertexAIAgentsAdapter._build_user_prompt({"query": "hello", "input": "world"})
        assert prompt == "hello"

    def test_input_key(self) -> None:
        """'input' key is used when 'query' is absent."""
        prompt = VertexAIAgentsAdapter._build_user_prompt({"input": "hello"})
        assert prompt == "hello"

    def test_prompt_key(self) -> None:
        """'prompt' key is used."""
        prompt = VertexAIAgentsAdapter._build_user_prompt({"prompt": "hello"})
        assert prompt == "hello"

    def test_message_key(self) -> None:
        """'message' key is used."""
        prompt = VertexAIAgentsAdapter._build_user_prompt({"message": "hello"})
        assert prompt == "hello"

    def test_single_value_fallback(self) -> None:
        """Single non-reserved key is used directly."""
        prompt = VertexAIAgentsAdapter._build_user_prompt(
            {"custom_field": "hello", "scenario_id": "s1"}
        )
        assert prompt == "hello"

    def test_multi_value_json_fallback(self) -> None:
        """Multiple non-reserved keys are serialized as JSON."""
        prompt = VertexAIAgentsAdapter._build_user_prompt({"a": 1, "b": 2})
        assert '"a"' in prompt
        assert '"b"' in prompt


class TestVertexAdapterToCallable:
    """Tests for the to_callable() method."""

    def test_returns_callable(self) -> None:
        """to_callable returns a callable."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": _FAKE_VERTEXAI}):
            fn = adapter.to_callable()

        assert callable(fn)
        # Bound methods are not identity-equal on each access, so check
        # that the callable is functionally the adapter's run method.
        assert fn.__func__ is VertexAIAgentsAdapter.run
        assert fn.__self__ is adapter

    def test_raises_when_not_installed(self) -> None:
        """to_callable raises FrameworkNotInstalledError when SDK missing."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(mock_model, model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"vertexai": None}):
            with pytest.raises(FrameworkNotInstalledError):
                adapter.to_callable()


class TestVertexAdapterGetConfig:
    """Tests for get_config() override."""

    def test_produces_valid_agent_config(self) -> None:
        """get_config returns a valid AgentConfig with framework='custom'."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(
            mock_model,
            model="gemini-2.0-flash",
            project_id="proj-xyz",
            location="us-central1",
        )

        config = adapter.get_config()

        assert isinstance(config, AgentConfig)
        assert config.framework == "vertex"
        assert config.model == "gemini-2.0-flash"
        assert config.name == "vertex-agent"
        assert config.metadata["vertex_project_id"] == "proj-xyz"
        assert config.metadata["vertex_location"] == "us-central1"
        # agent_id is a valid UUID string
        assert len(config.agent_id) == 36

    def test_config_with_custom_metadata(self) -> None:
        """User metadata is merged into get_config metadata."""
        mock_model = MagicMock()
        adapter = VertexAIAgentsAdapter(
            mock_model,
            model="gemini-2.0-flash",
            metadata={"team": "ml-platform"},
        )

        config = adapter.get_config()

        assert config.metadata["team"] == "ml-platform"
