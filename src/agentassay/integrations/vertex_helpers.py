"""Response parsing helpers for the Vertex AI adapter.

Contains pure functions for extracting ``StepTrace`` objects from
Vertex AI ``GenerateContentResponse`` candidates, parts, and grounding
metadata.  Separated from the main adapter to keep each file focused
on a single responsibility.

This module is imported by ``vertex_adapter.py`` -- it is NOT intended
for direct consumption by end users.
"""

from __future__ import annotations

from typing import Any

from agentassay.core.models import StepTrace


def extract_steps(
    response: Any,
    total_call_ms: float,
    model: str,
    extract_output_fn: Any,
) -> list[StepTrace]:
    """Extract StepTrace objects from a Vertex AI generate_content response.

    Parses response candidates and their content parts. Each part can be:

    - A ``text`` part -> ``llm_response`` step.
    - A ``function_call`` part -> ``tool_call`` step.
    - A ``function_response`` part -> ``observation`` step.

    After processing parts, grounding metadata (if present) is emitted
    as additional ``retrieval`` steps.

    Parameters
    ----------
    response
        The ``GenerateContentResponse`` from ``generate_content()``.
    total_call_ms
        Total wall-clock time for the API call in milliseconds.
    model
        LLM model identifier for step metadata.
    extract_output_fn
        Callable to extract text output from the response (used as
        fallback when no steps can be extracted).

    Returns
    -------
    list[StepTrace]
        Ordered list of steps extracted from the response.
    """
    steps: list[StepTrace] = []
    step_index = 0

    # Get candidates from response
    candidates = getattr(response, "candidates", None)
    if not candidates:
        # Empty response -- create a single observation step
        steps.append(
            StepTrace(
                step_index=0,
                action="observation",
                llm_output="(empty response -- no candidates)",
                duration_ms=total_call_ms,
                model=model,
                metadata={"vertex_empty_response": True},
            )
        )
        return steps

    # Count total parts across all candidates for time distribution
    total_parts = sum(
        len(getattr(getattr(c, "content", None), "parts", []) or [])
        for c in candidates
    )
    # Add grounding steps count estimate
    grounding_count = sum(
        1
        for c in candidates
        if getattr(c, "grounding_metadata", None)
    )
    total_items = max(total_parts + grounding_count, 1)
    per_item_ms = total_call_ms / total_items

    for cand_idx, candidate in enumerate(candidates):
        content = getattr(candidate, "content", None)
        if content is None:
            continue

        parts = getattr(content, "parts", None)
        if not parts:
            continue

        for part in parts:
            step = classify_part(part, step_index, per_item_ms, cand_idx, model)
            if step is not None:
                steps.append(step)
                step_index += 1

        # Extract grounding metadata if present
        grounding_meta = getattr(candidate, "grounding_metadata", None)
        if grounding_meta:
            grounding_step = extract_grounding_step(
                grounding_meta, step_index, per_item_ms, cand_idx, model
            )
            steps.append(grounding_step)
            step_index += 1

    # Fallback if no steps were extracted
    if not steps:
        steps.append(
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=extract_output_fn(response),
                duration_ms=total_call_ms,
                model=model,
                metadata={"vertex_fallback": True},
            )
        )

    return steps


def classify_part(
    part: Any,
    step_index: int,
    per_item_ms: float,
    candidate_index: int,
    model: str,
) -> StepTrace | None:
    """Classify a single response part into a StepTrace.

    Parameters
    ----------
    part
        A Vertex AI ``Part`` object from a candidate's content.
    step_index
        Current step index in the trace.
    per_item_ms
        Estimated duration per item in milliseconds.
    candidate_index
        Index of the parent candidate (for metadata).
    model
        LLM model identifier.

    Returns
    -------
    StepTrace or None
        A step trace, or ``None`` if the part cannot be classified.
    """
    # -- Function call part -----------------------------------------------
    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        fc_name = getattr(function_call, "name", "unknown_function")
        fc_args = getattr(function_call, "args", None)

        # Convert proto Map/Struct to plain dict
        args_dict: dict[str, Any] = {}
        if fc_args is not None:
            try:
                args_dict = dict(fc_args)
            except (TypeError, ValueError):
                args_dict = {"raw": str(fc_args)}

        return StepTrace(
            step_index=step_index,
            action="tool_call",
            tool_name=str(fc_name),
            tool_input=args_dict,
            tool_output=None,
            duration_ms=per_item_ms,
            model=model,
            metadata={
                "vertex_part_type": "function_call",
                "vertex_candidate_index": candidate_index,
            },
        )

    # -- Function response part -------------------------------------------
    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        fr_name = getattr(function_response, "name", "unknown_function")
        fr_response = getattr(function_response, "response", None)

        response_data: dict[str, Any] = {}
        if fr_response is not None:
            try:
                response_data = dict(fr_response)
            except (TypeError, ValueError):
                response_data = {"raw": str(fr_response)}

        return StepTrace(
            step_index=step_index,
            action="observation",
            tool_name=str(fr_name),
            tool_input={"source": "function_response"},
            tool_output=response_data,
            llm_output=str(response_data),
            duration_ms=per_item_ms,
            model=model,
            metadata={
                "vertex_part_type": "function_response",
                "vertex_candidate_index": candidate_index,
            },
        )

    # -- Text part --------------------------------------------------------
    text = getattr(part, "text", None)
    if text is not None:
        return StepTrace(
            step_index=step_index,
            action="llm_response",
            llm_output=str(text),
            duration_ms=per_item_ms,
            model=model,
            metadata={
                "vertex_part_type": "text",
                "vertex_candidate_index": candidate_index,
            },
        )

    # -- Inline data or other part types ----------------------------------
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        mime_type = getattr(inline_data, "mime_type", "unknown")
        return StepTrace(
            step_index=step_index,
            action="observation",
            llm_output=f"(inline data: {mime_type})",
            duration_ms=per_item_ms,
            model=model,
            metadata={
                "vertex_part_type": "inline_data",
                "vertex_mime_type": mime_type,
                "vertex_candidate_index": candidate_index,
            },
        )

    return None


def extract_grounding_step(
    grounding_metadata: Any,
    step_index: int,
    per_item_ms: float,
    candidate_index: int,
    model: str,
) -> StepTrace:
    """Extract a retrieval step from Vertex AI grounding metadata.

    Grounding metadata appears when the model uses Google Search or
    Vertex AI Search as grounding sources. Contains:

    - ``grounding_chunks``: list of source chunks with URIs and titles.
    - ``grounding_supports``: which parts of the response are grounded.
    - ``search_entry_point``: the rendered search widget HTML.
    - ``retrieval_metadata``: quality scores.

    Parameters
    ----------
    grounding_metadata
        The ``GroundingMetadata`` from a response candidate.
    step_index
        Current step index.
    per_item_ms
        Estimated duration per item in milliseconds.
    candidate_index
        Index of the parent candidate.
    model
        LLM model identifier.

    Returns
    -------
    StepTrace
        A retrieval step with source information in metadata.
    """
    # Extract grounding chunks (sources)
    chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
    sources: list[dict[str, str]] = []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web is not None:
            sources.append({
                "uri": getattr(web, "uri", ""),
                "title": getattr(web, "title", ""),
            })
        retrieved_context = getattr(chunk, "retrieved_context", None)
        if retrieved_context is not None:
            sources.append({
                "uri": getattr(retrieved_context, "uri", ""),
                "title": getattr(retrieved_context, "title", ""),
            })

    # Extract grounding supports count
    supports = getattr(grounding_metadata, "grounding_supports", None) or []
    num_supports = len(supports)

    # Extract retrieval quality score if available
    retrieval_meta = getattr(
        grounding_metadata, "retrieval_metadata", None
    )
    quality_score = (
        getattr(retrieval_meta, "google_search_dynamic_retrieval_score", None)
        if retrieval_meta
        else None
    )

    source_summary = "; ".join(
        s.get("title", s.get("uri", "unknown")) for s in sources[:5]
    )

    return StepTrace(
        step_index=step_index,
        action="retrieval",
        tool_name="vertex_grounding",
        tool_input={"source_count": len(sources)},
        tool_output=source_summary or "(no sources)",
        llm_output=f"Grounded in {len(sources)} source(s)",
        duration_ms=per_item_ms,
        model=model,
        metadata={
            "vertex_part_type": "grounding",
            "vertex_candidate_index": candidate_index,
            "vertex_grounding_sources": sources,
            "vertex_grounding_supports_count": num_supports,
            "vertex_grounding_quality_score": quality_score,
        },
    )
