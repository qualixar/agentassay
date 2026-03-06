# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Agent mutation testing for AgentAssay.

Implements novel agent-specific mutation operators (paper Definition 5.1)
across four specification dimensions:

    M = {M_prompt, M_tool, M_model, M_context}

Each operator applies a single atomic transformation to the agent's
configuration or test scenario. The ``MutationRunner`` executes both
the original and mutated agent, compares outcomes, and computes the
mutation score — the fraction of mutants that the test suite detects.

Usage
-----
>>> from agentassay.mutation import MutationRunner, DEFAULT_OPERATORS
>>> runner = MutationRunner(my_agent, config, assay_config)
>>> result = runner.run_suite(scenario)
>>> print(f"Mutation score: {result.mutation_score:.1%}")
"""

from agentassay.mutation.operators import (
    ContextNoiseMutator,
    ContextPermutationMutator,
    # Context operators (M_context)
    ContextTruncationMutator,
    # Model operators (M_model)
    ModelSwapMutator,
    ModelVersionMutator,
    # Abstract base
    MutationOperator,
    PromptDropMutator,
    PromptNoiseMutator,
    PromptOrderMutator,
    # Prompt operators (M_prompt)
    PromptSynonymMutator,
    ToolNoiseMutator,
    # Tool operators (M_tool)
    ToolRemovalMutator,
    ToolReorderMutator,
)
from agentassay.mutation.runner import (
    DEFAULT_OPERATORS,
    MutationResult,
    MutationRunner,
    MutationSuiteResult,
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
    # Runner
    "MutationRunner",
    "MutationResult",
    "MutationSuiteResult",
    "DEFAULT_OPERATORS",
]
