# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Framework adapters for AgentAssay.

Each adapter wraps a specific AI agent framework and translates its
execution into the uniform ``ExecutionTrace`` format that the rest of
AgentAssay (TrialRunner, statistics, coverage, verdicts) consumes.

All framework dependencies are **optional** — only the base classes and
the ``CustomAdapter`` are guaranteed to import without extra packages.
Framework-specific adapters raise a clear ``FrameworkNotInstalledError``
with ``pip install`` instructions when used without the framework.

Quick start
-----------
>>> from agentassay.integrations import create_adapter
>>> adapter = create_adapter("langgraph", my_graph, model="gpt-4o")
>>> trace = adapter.run({"query": "Hello"})

Or use an adapter directly::

>>> from agentassay.integrations import LangGraphAdapter
>>> adapter = LangGraphAdapter(graph=my_graph, model="gpt-4o")
"""

from __future__ import annotations

from typing import Any

from agentassay.integrations.autogen_adapter import AutoGenAdapter
from agentassay.integrations.base import (
    AdapterError,
    AgentAdapter,
    FrameworkNotInstalledError,
)
from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter
from agentassay.integrations.crewai_adapter import CrewAIAdapter
from agentassay.integrations.custom_adapter import CustomAdapter

# Framework adapters — imported at module level so they appear in
# autocomplete, but their framework dependencies are lazy.
from agentassay.integrations.langgraph_adapter import LangGraphAdapter
from agentassay.integrations.mcp_adapter import MCPToolsAdapter
from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter
from agentassay.integrations.semantic_kernel_adapter import SemanticKernelAdapter
from agentassay.integrations.smolagents_adapter import SmolAgentsAdapter
from agentassay.integrations.vertex_adapter import VertexAIAgentsAdapter

# ── Registry: framework name → adapter class ──────────────────────────
_ADAPTER_REGISTRY: dict[str, type[AgentAdapter]] = {
    "langgraph": LangGraphAdapter,
    "crewai": CrewAIAdapter,
    "openai": OpenAIAgentsAdapter,
    "autogen": AutoGenAdapter,
    "smolagents": SmolAgentsAdapter,
    "semantic_kernel": SemanticKernelAdapter,
    "bedrock": BedrockAgentsAdapter,
    "mcp": MCPToolsAdapter,
    "vertex": VertexAIAgentsAdapter,
    "custom": CustomAdapter,
}

# Alternative names that users might reasonably use
_ALIASES: dict[str, str] = {
    "langchain": "langgraph",
    "lang_graph": "langgraph",
    "lang-graph": "langgraph",
    "crew_ai": "crewai",
    "crew-ai": "crewai",
    "crew": "crewai",
    "openai_agents": "openai",
    "openai-agents": "openai",
    "openai_sdk": "openai",
    "auto_gen": "autogen",
    "auto-gen": "autogen",
    "ag2": "autogen",
    "pyautogen": "autogen",
    "smol_agents": "smolagents",
    "smol-agents": "smolagents",
    "smol": "smolagents",
    "huggingface": "smolagents",
    # Semantic Kernel aliases
    "semantic-kernel": "semantic_kernel",
    "sk": "semantic_kernel",
    "microsoft": "semantic_kernel",
    "azure-ai": "semantic_kernel",
    # Bedrock aliases
    "bedrock-agents": "bedrock",
    "bedrock_agents": "bedrock",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    # MCP aliases
    "mcp-tools": "mcp",
    "claude-mcp": "mcp",
    "anthropic-mcp": "mcp",
    "anthropic": "mcp",
    # Vertex AI aliases
    "vertex-ai": "vertex",
    "vertex_ai": "vertex",
    "google": "vertex",
    "gcp": "vertex",
    "google-vertex": "vertex",
}


def create_adapter(
    framework: str,
    agent: Any,
    **kwargs: Any,
) -> AgentAdapter:
    """Factory function to create the right adapter for a framework.

    This is the recommended entry point when you want AgentAssay to
    auto-detect the adapter class based on a framework name string.

    Parameters
    ----------
    framework
        Framework identifier: ``"langgraph"``, ``"crewai"``, ``"openai"``,
        ``"autogen"``, ``"smolagents"``, or ``"custom"``.
        Common aliases are also supported (e.g. ``"crew"``, ``"ag2"``).
    agent
        The framework-specific agent object. For ``"custom"``, this must
        be a callable with signature ``(dict[str, Any]) -> Any``.
    **kwargs
        Additional keyword arguments forwarded to the adapter constructor.
        Common options: ``model``, ``agent_name``, ``metadata``.

    Returns
    -------
    AgentAdapter
        A configured adapter instance ready for ``run()`` or ``to_callable()``.

    Raises
    ------
    ValueError
        If ``framework`` is not recognized.
    FrameworkNotInstalledError
        If the required framework package is not installed (raised lazily
        on ``run()`` or ``to_callable()``, NOT during creation).

    Examples
    --------
    >>> adapter = create_adapter("langgraph", my_graph, model="gpt-4o")
    >>> adapter = create_adapter("crewai", my_crew, model="claude-3.5-sonnet")
    >>> adapter = create_adapter("custom", my_fn, model="local-llm")
    """
    # Normalize framework name
    normalized = framework.strip().lower().replace(" ", "")

    # Resolve aliases
    resolved = _ALIASES.get(normalized, normalized)

    # Look up adapter class
    adapter_cls = _ADAPTER_REGISTRY.get(resolved)
    if adapter_cls is None:
        supported = sorted(_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown framework {framework!r}. "
            f"Supported frameworks: {', '.join(supported)}. "
            f"Use 'custom' with a callable for unsupported frameworks."
        )

    # Special handling for custom adapter: first positional arg is callable_fn
    if resolved == "custom":
        return CustomAdapter(callable_fn=agent, **kwargs)

    # All other adapters: first positional arg is the agent/graph/crew object
    # The parameter name varies by adapter, but they all accept it as
    # the first positional argument after self.
    return adapter_cls(agent, **kwargs)  # type: ignore[call-arg]


def list_adapters() -> dict[str, type[AgentAdapter]]:
    """Return the registry of available adapter classes.

    Returns
    -------
    dict[str, type[AgentAdapter]]
        Mapping from framework name to adapter class.
    """
    return dict(_ADAPTER_REGISTRY)


def list_aliases() -> dict[str, str]:
    """Return all framework name aliases.

    Returns
    -------
    dict[str, str]
        Mapping from alias to canonical framework name.
    """
    return dict(_ALIASES)


__all__ = [
    # Base
    "AgentAdapter",
    "AdapterError",
    "FrameworkNotInstalledError",
    # Concrete adapters
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
    "AutoGenAdapter",
    "SmolAgentsAdapter",
    "SemanticKernelAdapter",
    "BedrockAgentsAdapter",
    "MCPToolsAdapter",
    "VertexAIAgentsAdapter",
    "CustomAdapter",
    # Factory
    "create_adapter",
    "list_adapters",
    "list_aliases",
]
