# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Anthropic client execution mode for the MCP Tools adapter.

Handles the agentic loop that sends messages via the Anthropic messages
API with MCP tool definitions converted to Anthropic's tool format.
Separated from the main MCP adapter to maintain single-responsibility
per file.

This module is imported by ``mcp_adapter.py`` -- it is NOT intended for
direct consumption by end users.
"""

from __future__ import annotations

import time
from typing import Any

from agentassay.core.models import StepTrace


def convert_mcp_tools_to_anthropic(
    mcp_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool format.

    MCP tools have ``name``, ``description``, ``inputSchema``.
    Anthropic tools have ``name``, ``description``, ``input_schema``.

    Parameters
    ----------
    mcp_tools
        List of MCP tool definition dicts.

    Returns
    -------
    list[dict[str, Any]]
        Tools in Anthropic API format.
    """
    anthropic_tools = []
    for tool in mcp_tools:
        anthropic_tools.append(
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get(
                    "inputSchema",
                    tool.get("input_schema", {"type": "object", "properties": {}}),
                ),
            }
        )
    return anthropic_tools


def run_anthropic(
    *,
    client: Any,
    tools: list[dict[str, Any]],
    model: str,
    input_data: dict[str, Any],
    user_input: str,
    execute_tool_fn: Any,
    safe_serialize_fn: Any,
) -> tuple[list[StepTrace], Any]:
    """Execute using an Anthropic client with MCP tool definitions.

    Sends a messages API call with MCP tools converted to Anthropic
    tool format.  Iterates through content blocks extracting tool_use
    and text blocks.

    Parameters
    ----------
    client
        An Anthropic ``Client`` instance with a ``messages.create`` method.
    tools
        MCP tool definitions (will be converted to Anthropic format).
    model
        LLM model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    input_data
        The scenario input dictionary.
    user_input
        Pre-extracted user prompt string.
    execute_tool_fn
        Callable ``(tool_name, tool_input) -> (output, duration_ms)``
        for executing tool calls via MCP.
    safe_serialize_fn
        Callable ``(value) -> serialized`` for safe JSON serialization.

    Returns
    -------
    tuple[list[StepTrace], Any]
        (ordered list of steps, final output text).
    """
    steps: list[StepTrace] = []
    step_index = 0

    # Convert MCP tools to Anthropic tool format
    anthropic_tools = convert_mcp_tools_to_anthropic(tools)

    # Initial message
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]

    # Agentic loop: call messages API, handle tool_use, repeat
    max_iterations = 10
    final_text = ""

    for _iteration in range(max_iterations):
        step_start = time.perf_counter()

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = client.messages.create(**create_kwargs)
        duration_ms = (time.perf_counter() - step_start) * 1000.0

        # Extract content blocks
        tool_use_blocks = []
        text_blocks = []

        content = getattr(response, "content", [])
        for block in content:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use":
                tool_use_blocks.append(block)
            elif block_type == "text":
                text_blocks.append(getattr(block, "text", ""))

        # Record text response step
        if text_blocks:
            text_content = "\n".join(text_blocks)
            final_text = text_content
            steps.append(
                StepTrace(
                    step_index=step_index,
                    action="llm_response",
                    llm_input=user_input if step_index == 0 else None,
                    llm_output=text_content,
                    duration_ms=duration_ms / max(len(content), 1),
                    model=model,
                    metadata={"mcp_method": "messages.create"},
                )
            )
            step_index += 1

        # Record tool use steps
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = getattr(tool_block, "name", "unknown")
            tool_input_data = getattr(tool_block, "input", {})
            tool_id = getattr(tool_block, "id", "")

            # Execute tool via MCP client if available
            tool_output, tool_duration = execute_tool_fn(tool_name, tool_input_data)

            steps.append(
                StepTrace(
                    step_index=step_index,
                    action="tool_call",
                    tool_name=tool_name,
                    tool_input=tool_input_data
                    if isinstance(tool_input_data, dict)
                    else {"raw": tool_input_data},
                    tool_output=tool_output,
                    duration_ms=tool_duration,
                    model=model,
                    metadata={
                        "mcp_method": "tool_use",
                        "tool_use_id": tool_id,
                    },
                )
            )
            step_index += 1

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": safe_serialize_fn(tool_output),
                }
            )

        # Check stop reason
        stop_reason = getattr(response, "stop_reason", "end_turn")
        if stop_reason != "tool_use" or not tool_results:
            break

        # Continue the conversation with tool results
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": tool_results})

    return steps, final_text
