# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Base class and shared utilities for agent mutation operators.

Defines the ``MutationOperator`` abstract base class and helper functions
used across all four operator categories (prompt, tool, model, context).

Every concrete mutation operator inherits from ``MutationOperator`` and
implements ``mutate()`` and ``describe_mutation()``.

Uses ``random.Random`` with optional seeding for reproducible mutations.
"""

from __future__ import annotations

import copy
import random as _random_module
import re
from abc import ABC, abstractmethod
from typing import Any

from agentassay.core.models import AgentConfig, TestScenario


# ===================================================================
# Abstract Base Class
# ===================================================================


class MutationOperator(ABC):
    """Abstract base class for all agent mutation operators.

    A mutation operator defines a single, atomic transformation over
    the agent's specification surface (config + scenario). The operator
    does NOT execute the agent --- it only produces the mutated inputs.

    Attributes
    ----------
    name
        Short identifier (e.g., ``"prompt_synonym"``).
    category
        One of ``"prompt"``, ``"tool"``, ``"model"``, ``"context"``.
    description
        Human-readable explanation of the mutation strategy.

    Parameters
    ----------
    seed
        Optional RNG seed for reproducible mutations.
    """

    name: str
    category: str
    description: str

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = _random_module.Random(seed)

    @abstractmethod
    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Apply this mutation, returning modified (config, scenario).

        Implementations MUST deep-copy the inputs before modifying them
        so the originals are never altered.

        Parameters
        ----------
        config
            The agent configuration (model, tools, parameters).
        scenario
            The test scenario (input data, expected properties).

        Returns
        -------
        tuple[AgentConfig, TestScenario]
            A new (config, scenario) pair with the mutation applied.
        """

    @abstractmethod
    def describe_mutation(self) -> str:
        """Return a human-readable description of the last mutation applied.

        Called after ``mutate()`` to record what changed in the
        ``MutationResult``. Must not raise if ``mutate()`` has not been
        called yet (return a generic description in that case).
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, category={self.category!r})"


# ===================================================================
# Helper: deep-copy with Pydantic v2 frozen models
# ===================================================================


def _deep_copy_config(config: AgentConfig) -> AgentConfig:
    """Deep-copy an AgentConfig (mutable model --- standard copy works)."""
    return config.model_copy(deep=True)


def _deep_copy_scenario(scenario: TestScenario) -> TestScenario:
    """Deep-copy a frozen TestScenario by reconstructing it.

    Pydantic v2 frozen models cannot be mutated in-place, so we dump
    to dict, modify the dict, and reconstruct. The caller will modify
    the dict before reconstruction.
    """
    return scenario.model_copy(deep=True)


def _rebuild_scenario(scenario: TestScenario, **overrides: Any) -> TestScenario:
    """Rebuild a frozen TestScenario with field overrides.

    Dumps to dict, applies overrides, reconstructs. This is the
    canonical way to 'mutate' a frozen Pydantic v2 model.
    """
    data = scenario.model_dump()
    data.update(overrides)
    return TestScenario(**data)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic.

    Handles period, exclamation, and question mark delimiters.
    Preserves the delimiter at the end of each sentence.
    """
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


__all__ = [
    "MutationOperator",
    "_deep_copy_config",
    "_deep_copy_scenario",
    "_rebuild_scenario",
    "_split_sentences",
]
