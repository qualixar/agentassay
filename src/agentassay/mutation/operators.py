# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Agent mutation operators for AgentAssay (re-export shim).

This module re-exports all mutation operator classes and shared utilities
from their respective sub-modules. Existing imports from
``agentassay.mutation.operators`` continue to work unchanged.

Operator categories:

    M = {M_prompt, M_tool, M_model, M_context}

See individual modules for implementation details:

- ``base.py`` --- MutationOperator base class and shared helpers.
- ``prompt_ops.py`` --- Prompt mutation operators (4 classes).
- ``tool_ops.py`` --- Tool mutation operators (3 classes).
- ``model_ops.py`` --- Model mutation operators (2 classes).
- ``context_ops.py`` --- Context mutation operators (3 classes).
"""

# Base class and shared utilities
from agentassay.mutation.base import (  # noqa: F401
    MutationOperator,
    _deep_copy_config,
    _deep_copy_scenario,
    _rebuild_scenario,
    _split_sentences,
)

# Prompt operators (M_prompt)
from agentassay.mutation.prompt_ops import (  # noqa: F401
    PromptDropMutator,
    PromptNoiseMutator,
    PromptOrderMutator,
    PromptSynonymMutator,
)

# Tool operators (M_tool)
from agentassay.mutation.tool_ops import (  # noqa: F401
    ToolNoiseMutator,
    ToolRemovalMutator,
    ToolReorderMutator,
)

# Model operators (M_model)
from agentassay.mutation.model_ops import (  # noqa: F401
    ModelSwapMutator,
    ModelVersionMutator,
)

# Context operators (M_context)
from agentassay.mutation.context_ops import (  # noqa: F401
    ContextNoiseMutator,
    ContextPermutationMutator,
    ContextTruncationMutator,
)

__all__ = [
    # Abstract base
    "MutationOperator",
    # Prompt operators (4)
    "PromptSynonymMutator",
    "PromptOrderMutator",
    "PromptNoiseMutator",
    "PromptDropMutator",
    # Tool operators (3)
    "ToolRemovalMutator",
    "ToolReorderMutator",
    "ToolNoiseMutator",
    # Model operators (2)
    "ModelSwapMutator",
    "ModelVersionMutator",
    # Context operators (3)
    "ContextTruncationMutator",
    "ContextNoiseMutator",
    "ContextPermutationMutator",
]
