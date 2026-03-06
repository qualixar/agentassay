# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Tool mutation operators (M_tool) for AgentAssay.

Implements three operators that target the tool specification surface
of the agent configuration:

- ``ToolRemovalMutator``: Remove one tool from the available set.
- ``ToolReorderMutator``: Shuffle tool presentation order.
- ``ToolNoiseMutator``: Corrupt tool descriptions with typos.

Each operator tests a distinct dimension of tool-related fragility.
"""

from __future__ import annotations

import copy
import string
from typing import Any

from agentassay.core.models import AgentConfig, TestScenario
from agentassay.mutation.base import (
    MutationOperator,
    _deep_copy_config,
    _deep_copy_scenario,
)


# ===================================================================
# ToolRemovalMutator
# ===================================================================


class ToolRemovalMutator(MutationOperator):
    """Remove one tool from the agent's available tool set.

    Tests whether the agent can recover when a tool is unavailable ---
    or whether it fails silently, hallucinates tool calls, or degrades
    output quality.

    Parameters
    ----------
    tool_key
        The key in ``config.parameters`` containing the tool list.
    seed
        Optional RNG seed.
    """

    name = "tool_removal"
    category = "tool"
    description = "Remove one tool from available tools"

    def __init__(
        self,
        *,
        tool_key: str = "tools",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._tool_key = tool_key
        self._removed_tool: str = ""

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Remove one random tool from config.parameters[tool_key]."""
        new_config = _deep_copy_config(config)
        new_scenario = _deep_copy_scenario(scenario)
        self._removed_tool = ""

        tools = new_config.parameters.get(self._tool_key)
        if not tools or not isinstance(tools, list) or len(tools) <= 1:
            return new_config, new_scenario

        # Deep-copy the tools list to avoid aliasing
        tools = copy.deepcopy(tools)
        remove_idx = self._rng.randint(0, len(tools) - 1)
        removed = tools.pop(remove_idx)

        # Extract a human-readable name from the removed tool
        self._removed_tool = self._extract_tool_name(removed)
        new_config.parameters[self._tool_key] = tools

        return new_config, new_scenario

    def describe_mutation(self) -> str:
        if not self._removed_tool:
            return "Tool removal mutation (no tool removed --- list empty or has only 1)"
        return f"Removed tool: '{self._removed_tool}'"

    @staticmethod
    def _extract_tool_name(tool: Any) -> str:
        """Best-effort extraction of a tool name from various formats."""
        if isinstance(tool, str):
            return tool
        if isinstance(tool, dict):
            for key in ("name", "function_name", "tool_name", "id"):
                if key in tool:
                    return str(tool[key])
            # Nested OpenAI-style: {"function": {"name": "..."}}
            if "function" in tool and isinstance(tool["function"], dict):
                return str(tool["function"].get("name", "<unnamed>"))
        return str(tool)[:50]


# ===================================================================
# ToolReorderMutator
# ===================================================================


class ToolReorderMutator(MutationOperator):
    """Change the presentation order of available tools.

    LLMs exhibit position bias --- tools listed first are often preferred.
    This operator tests whether the agent's tool selection is robust to
    ordering effects.

    Parameters
    ----------
    tool_key
        The key in ``config.parameters`` containing the tool list.
    seed
        Optional RNG seed.
    """

    name = "tool_reorder"
    category = "tool"
    description = "Shuffle the order of available tools"

    def __init__(
        self,
        *,
        tool_key: str = "tools",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._tool_key = tool_key
        self._original_count: int = 0

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Shuffle the tool list order."""
        new_config = _deep_copy_config(config)
        new_scenario = _deep_copy_scenario(scenario)
        self._original_count = 0

        tools = new_config.parameters.get(self._tool_key)
        if not tools or not isinstance(tools, list) or len(tools) <= 1:
            return new_config, new_scenario

        tools = copy.deepcopy(tools)
        self._original_count = len(tools)

        # Shuffle until we get a different ordering (bounded)
        original = list(tools)
        for _ in range(20):
            self._rng.shuffle(tools)
            if tools != original:
                break

        new_config.parameters[self._tool_key] = tools
        return new_config, new_scenario

    def describe_mutation(self) -> str:
        if self._original_count == 0:
            return "Tool reorder mutation (no tools to reorder)"
        return f"Reordered {self._original_count} tools"


# ===================================================================
# ToolNoiseMutator
# ===================================================================


class ToolNoiseMutator(MutationOperator):
    """Corrupt tool descriptions slightly.

    Tests whether the agent relies on exact tool description wording
    or understands tool semantics. In production, tool descriptions
    change across versions --- agents must be robust to minor variations.

    Parameters
    ----------
    tool_key
        The key in ``config.parameters`` containing the tool list.
    noise_rate
        Probability of corrupting each word in descriptions.
    seed
        Optional RNG seed.
    """

    name = "tool_noise"
    category = "tool"
    description = "Add noise to tool descriptions"

    def __init__(
        self,
        *,
        tool_key: str = "tools",
        noise_rate: float = 0.15,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._tool_key = tool_key
        self._noise_rate = max(0.0, min(1.0, noise_rate))
        self._num_tools_corrupted: int = 0

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Add character-level noise to tool descriptions."""
        new_config = _deep_copy_config(config)
        new_scenario = _deep_copy_scenario(scenario)
        self._num_tools_corrupted = 0

        tools = new_config.parameters.get(self._tool_key)
        if not tools or not isinstance(tools, list):
            return new_config, new_scenario

        tools = copy.deepcopy(tools)
        for tool in tools:
            if isinstance(tool, dict):
                corrupted = self._corrupt_description(tool)
                if corrupted:
                    self._num_tools_corrupted += 1

        new_config.parameters[self._tool_key] = tools
        return new_config, new_scenario

    def describe_mutation(self) -> str:
        return f"Corrupted descriptions of {self._num_tools_corrupted} tools"

    def _corrupt_description(self, tool: dict[str, Any]) -> bool:
        """Corrupt the description field of a tool dict in-place.

        Returns True if any corruption was applied.
        """
        # Try common description field names
        desc_key: str | None = None
        for key in ("description", "desc", "summary"):
            if key in tool and isinstance(tool[key], str):
                desc_key = key
                break

        # Also check nested OpenAI-style format
        if desc_key is None and "function" in tool and isinstance(tool["function"], dict):
            func = tool["function"]
            for key in ("description", "desc"):
                if key in func and isinstance(func[key], str):
                    return self._corrupt_text_inplace(func, key)
            return False

        if desc_key is None:
            return False

        return self._corrupt_text_inplace(tool, desc_key)

    def _corrupt_text_inplace(
        self, container: dict[str, Any], key: str
    ) -> bool:
        """Replace words in container[key] with typo-corrupted versions."""
        text = str(container[key])
        words = text.split()
        changed = False

        for i, word in enumerate(words):
            if len(word) > 3 and self._rng.random() < self._noise_rate:
                idx = self._rng.randint(1, len(word) - 2)
                chars = list(word)
                chars[idx] = self._rng.choice(string.ascii_lowercase)
                words[i] = "".join(chars)
                changed = True

        if changed:
            container[key] = " ".join(words)
        return changed


__all__ = [
    "ToolRemovalMutator",
    "ToolReorderMutator",
    "ToolNoiseMutator",
]
