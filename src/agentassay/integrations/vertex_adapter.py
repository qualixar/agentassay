# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Google Vertex AI Agents adapter for AgentAssay.

Wraps a Vertex AI ``GenerativeModel`` instance and captures its execution
as an ``ExecutionTrace``. The Google Cloud AI Platform SDK provides access
to Gemini models with tool-use capabilities, grounding, and Vertex AI
Agent Builder applications.

The adapter parses ``generate_content()`` responses to extract:

- ``function_call`` parts as tool-call steps.
- ``text`` parts as LLM-response steps.
- ``grounding_metadata`` as retrieval steps.
- ``usage_metadata`` for token counts and cost estimation.

All ``google.cloud`` and ``vertexai`` imports are **lazy** -- this module
can be imported even when ``google-cloud-aiplatform`` is not installed.
The ``ImportError`` is raised only when ``run()`` or ``to_callable()`` is
actually called.

Usage
-----
>>> from agentassay.integrations.vertex_adapter import VertexAIAgentsAdapter
>>> adapter = VertexAIAgentsAdapter(model_instance, model="gemini-2.0-flash")
>>> trace = adapter.run({"query": "Summarize recent earnings"})
>>> runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from agentassay.core.models import AgentConfig, ExecutionTrace, StepTrace
from agentassay.integrations.base import (
    AgentAdapter,
    FrameworkNotInstalledError,
)
from agentassay.integrations.vertex_helpers import (
    classify_part as _classify_part,
)
from agentassay.integrations.vertex_helpers import (
    extract_grounding_step as _extract_grounding_step,
)
from agentassay.integrations.vertex_helpers import (
    extract_steps as _extract_steps,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Vertex AI adapter requires google-cloud-aiplatform>=1.60. "
    "Install with: pip install agentassay[vertex]"
)

# ---------------------------------------------------------------------------
# Default per-token pricing (USD) -- Gemini 2.0 Flash tier as baseline.
# These are conservative estimates for cost tracking. Users can override
# via the ``pricing`` metadata key.
# ---------------------------------------------------------------------------
_DEFAULT_PRICING: dict[str, float] = {
    "prompt_per_token": 0.000_000_075,  # $0.075 per 1M input tokens
    "completion_per_token": 0.000_000_30,  # $0.30  per 1M output tokens
}


def _check_vertex_installed() -> None:
    """Verify that the Vertex AI SDK is available.

    Raises
    ------
    FrameworkNotInstalledError
        If ``google-cloud-aiplatform`` (which provides the ``vertexai``
        namespace) is not installed.
    """
    try:
        import vertexai  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class VertexAIAgentsAdapter(AgentAdapter):
    """Adapter for Google Vertex AI generative models with tool support.

    Wraps a Vertex AI ``GenerativeModel`` (from the ``vertexai.generative_models``
    package) and captures each ``generate_content()`` invocation as an
    ``ExecutionTrace`` with fine-grained per-part step extraction.

    Parameters
    ----------
    generative_model
        A Vertex AI ``GenerativeModel`` instance, pre-configured with
        system instructions, safety settings, and generation config.
    tools
        Optional list of Vertex AI ``Tool`` objects (function declarations,
        grounding tools, retrieval tools). Passed to ``generate_content()``.
    project_id
        Google Cloud project ID. Informational -- attached to trace metadata.
    location
        Google Cloud region (e.g. ``"us-central1"``). Informational.
    model
        LLM model identifier (e.g. ``"gemini-2.0-flash"``). If ``"unknown"``,
        the adapter attempts to infer from the model object's ``_model_name``
        attribute.
    agent_name
        Human-readable name. Defaults to ``"vertex-agent"``.
    metadata
        Arbitrary metadata attached to every trace. May include a
        ``"pricing"`` key to override default per-token cost estimates.

    Examples
    --------
    >>> import vertexai
    >>> from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
    >>> vertexai.init(project="my-project", location="us-central1")
    >>> model = GenerativeModel("gemini-2.0-flash")
    >>> adapter = VertexAIAgentsAdapter(model, model="gemini-2.0-flash")
    >>> trace = adapter.run({"query": "What is the weather in Tokyo?"})
    """

    framework: str = "vertex"

    def __init__(
        self,
        generative_model: Any,
        *,
        tools: list[Any] | None = None,
        project_id: str | None = None,
        location: str = "us-central1",
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Attempt to resolve model name from the object if not provided
        resolved_model = model
        if model == "unknown":
            resolved_model = (
                getattr(generative_model, "_model_name", None)
                or getattr(generative_model, "model_name", None)
                or "unknown"
            )
            # Strip prefix like "publishers/google/models/"
            if isinstance(resolved_model, str) and "/" in resolved_model:
                resolved_model = resolved_model.rsplit("/", 1)[-1]

        super().__init__(model=resolved_model, agent_name=agent_name, metadata=metadata)
        self._generative_model = generative_model
        self._tools = tools
        self._project_id = project_id
        self._location = location

        # Merge user-supplied pricing into defaults
        user_pricing = (self._metadata or {}).get("pricing", {})
        self._pricing = {**_DEFAULT_PRICING, **user_pricing}

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the Vertex AI model and capture an ExecutionTrace.

        Calls ``generative_model.generate_content()`` with the user prompt
        and optional tools, then parses the response candidates into
        individual ``StepTrace`` objects.

        Parameters
        ----------
        input_data
            The scenario input. Expects a ``"query"``, ``"input"``,
            ``"prompt"``, or ``"message"`` key with the user content.
            The entire dict is serialized as the prompt if no known key
            is found.

        Returns
        -------
        ExecutionTrace
            Full trace with tool calls, LLM responses, retrieval steps,
            token usage, and cost estimates. On failure, ``success=False``
            with the error message.
        """
        _check_vertex_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            user_prompt = self._build_user_prompt(input_data)

            # Build keyword arguments for generate_content
            gen_kwargs: dict[str, Any] = {}
            if self._tools:
                gen_kwargs["tools"] = self._tools

            call_start = time.perf_counter()
            response = self._generative_model.generate_content(user_prompt, **gen_kwargs)
            call_duration_ms = (time.perf_counter() - call_start) * 1000.0

            # Extract steps from response candidates
            steps = self._extract_steps(response, call_duration_ms)

            # Extract final text output
            output = self._extract_output(response)

            # Extract token usage and compute cost
            token_meta = self._extract_token_usage(response)
            cost_usd = self._estimate_cost(token_meta)

            total_ms = (time.perf_counter() - overall_start) * 1000.0

            # Build trace metadata with Vertex-specific details
            trace_metadata = {
                **self._metadata,
                "vertex_project_id": self._project_id,
                "vertex_location": self._location,
                "vertex_token_usage": token_meta,
            }

            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=steps,
                input_data=input_data,
                output_data=output,
                success=True,
                error=None,
                total_duration_ms=total_ms,
                total_cost_usd=cost_usd,
                model=self._model,
                framework=self.framework,
                metadata=trace_metadata,
            )

        except FrameworkNotInstalledError:
            raise

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Vertex AI adapter failed: %s\n%s",
                error_msg,
                traceback.format_exc(),
            )
            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=[],
                input_data=input_data,
                output_data=None,
                success=False,
                error=error_msg,
                total_duration_ms=total_ms,
                total_cost_usd=0.0,
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

    def to_callable(self) -> Callable[[dict[str, Any]], ExecutionTrace]:
        """Return a TrialRunner-compatible callable.

        Returns
        -------
        Callable[[dict[str, Any]], ExecutionTrace]
            A reference to ``self.run``.

        Raises
        ------
        FrameworkNotInstalledError
            If ``google-cloud-aiplatform`` is not installed.
        """
        _check_vertex_installed()
        return self.run

    def get_config(self) -> AgentConfig:
        """Build an ``AgentConfig`` describing this adapter's agent.

        Overrides the base implementation because ``AgentConfig.framework``
        is a Literal constrained to the six built-in frameworks. The Vertex
        adapter maps to ``"custom"`` in ``AgentConfig`` while retaining
        ``"vertex"`` as the framework identifier in ``ExecutionTrace``.

        Returns
        -------
        AgentConfig
            Configuration with ``framework="custom"`` and Vertex-specific
            metadata including ``actual_framework``, ``project_id``, and
            ``location``.
        """
        return AgentConfig(
            agent_id=str(uuid.uuid4()),
            name=self._agent_name,
            framework="vertex",
            model=self._model,
            metadata={
                **self._metadata,
                "vertex_project_id": self._project_id,
                "vertex_location": self._location,
            },
        )

    # -- Internal: prompt construction ----------------------------------------

    @staticmethod
    def _build_user_prompt(input_data: dict[str, Any]) -> str:
        """Extract or construct the user prompt from input_data.

        Checks keys in priority order: ``query``, ``input``, ``prompt``,
        ``message``. Falls back to serializing the entire dict (minus
        internal keys).

        Parameters
        ----------
        input_data
            The scenario input dictionary.

        Returns
        -------
        str
            The user prompt string.
        """
        for key in ("query", "input", "prompt", "message"):
            if key in input_data:
                return str(input_data[key])

        # Filter internal keys before serializing
        filtered = {k: v for k, v in input_data.items() if k not in ("scenario_id", "metadata")}
        if len(filtered) == 1:
            return str(next(iter(filtered.values())))

        import json

        return json.dumps(filtered, default=str)

    # -- Internal: step extraction (delegates to vertex_helpers) ---------------

    def _extract_steps(self, response: Any, total_call_ms: float) -> list[StepTrace]:
        """Extract StepTrace objects from a Vertex AI response.

        Delegates to ``vertex_helpers.extract_steps()`` for the actual
        parsing logic.

        Parameters
        ----------
        response
            The ``GenerateContentResponse`` from ``generate_content()``.
        total_call_ms
            Total wall-clock time for the API call in milliseconds.

        Returns
        -------
        list[StepTrace]
            Ordered list of steps extracted from the response.
        """
        return _extract_steps(response, total_call_ms, self._model, self._extract_output)

    def _classify_part(
        self,
        part: Any,
        step_index: int,
        per_item_ms: float,
        candidate_index: int,
    ) -> StepTrace | None:
        """Classify a single response part into a StepTrace.

        Backward-compatibility shim -- delegates to ``vertex_helpers``.

        Parameters
        ----------
        part
            A Vertex AI ``Part`` object.
        step_index
            Current step index.
        per_item_ms
            Estimated duration per item in milliseconds.
        candidate_index
            Index of the parent candidate.

        Returns
        -------
        StepTrace or None
            A step trace, or ``None`` if the part cannot be classified.
        """
        return _classify_part(part, step_index, per_item_ms, candidate_index, self._model)

    def _extract_grounding_step(
        self,
        grounding_metadata: Any,
        step_index: int,
        per_item_ms: float,
        candidate_index: int,
    ) -> StepTrace:
        """Extract a retrieval step from grounding metadata.

        Backward-compatibility shim -- delegates to ``vertex_helpers``.

        Parameters
        ----------
        grounding_metadata
            The ``GroundingMetadata`` from a response candidate.
        step_index
            Current step index.
        per_item_ms
            Estimated duration per item.
        candidate_index
            Index of the parent candidate.

        Returns
        -------
        StepTrace
            A retrieval step.
        """
        return _extract_grounding_step(
            grounding_metadata,
            step_index,
            per_item_ms,
            candidate_index,
            self._model,
        )

    # -- Internal: output extraction ------------------------------------------

    @staticmethod
    def _extract_output(response: Any) -> Any:
        """Extract the final text output from the response.

        Tries ``response.text`` first (convenience property), then iterates
        candidates to concatenate text parts.

        Parameters
        ----------
        response
            The ``GenerateContentResponse`` object.

        Returns
        -------
        str or None
            The concatenated text output, or ``None`` if no text is found.
        """
        # Try the convenience .text property
        try:
            text = response.text
            if text:
                return text
        except (AttributeError, ValueError):
            pass

        # Fall back to iterating candidates
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None

        text_parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts) if text_parts else None

    # -- Internal: token usage & cost -----------------------------------------

    @staticmethod
    def _extract_token_usage(response: Any) -> dict[str, int]:
        """Extract token usage metadata from the response.

        Parameters
        ----------
        response
            The ``GenerateContentResponse`` object.

        Returns
        -------
        dict[str, int]
            Token counts keyed as ``prompt_tokens``, ``completion_tokens``,
            ``total_tokens``. Returns zeros if metadata is unavailable.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0

        # If total is missing, compute it
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

    def _estimate_cost(self, token_usage: dict[str, int]) -> float:
        """Estimate USD cost from token usage and pricing table.

        Parameters
        ----------
        token_usage
            Token counts from ``_extract_token_usage()``.

        Returns
        -------
        float
            Estimated cost in USD. Returns ``0.0`` if no tokens were used.
        """
        prompt_cost = token_usage.get("prompt_tokens", 0) * self._pricing["prompt_per_token"]
        completion_cost = (
            token_usage.get("completion_tokens", 0) * self._pricing["completion_per_token"]
        )
        return prompt_cost + completion_cost
