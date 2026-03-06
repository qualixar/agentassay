"""Model mutation operators (M_model) for AgentAssay.

Implements two operators that target the model specification surface
of the agent configuration:

- ``ModelSwapMutator``: Replace the model with a different one.
- ``ModelVersionMutator``: Change to a different version within the
  same model family.

Each operator tests model-sensitivity: whether behavior is tightly
coupled to a specific LLM or generalizes across providers and sizes.
"""

from __future__ import annotations

from agentassay.core.models import AgentConfig, TestScenario
from agentassay.mutation.base import (
    MutationOperator,
    _deep_copy_config,
    _deep_copy_scenario,
)


# ---------------------------------------------------------------------------
# Model family tables for ModelVersionMutator
# ---------------------------------------------------------------------------

_MODEL_FAMILIES: dict[str, list[str]] = {
    "gpt-4o": ["gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
    "gpt-4o-mini": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "gpt-4-turbo": ["gpt-4o", "gpt-4o-mini", "gpt-4"],
    "gpt-4": ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
    "gpt-3.5-turbo": ["gpt-4o-mini", "gpt-4o"],
    "claude-3-opus-20240229": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "claude-3-sonnet-20240229": ["claude-3-opus-20240229", "claude-3-haiku-20240307"],
    "claude-3-haiku-20240307": ["claude-3-sonnet-20240229", "claude-3-opus-20240229"],
    "claude-3.5-sonnet": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "claude-opus-4": ["claude-sonnet-4", "claude-3.5-sonnet"],
    "claude-sonnet-4": ["claude-opus-4", "claude-3.5-sonnet"],
}


# ===================================================================
# ModelSwapMutator
# ===================================================================


class ModelSwapMutator(MutationOperator):
    """Replace the agent's model with a different one from a provided list.

    Tests model-sensitivity: whether the agent's behavior is tightly
    coupled to a specific LLM or generalizes across providers/sizes.

    Parameters
    ----------
    alternative_models
        Candidate models to swap in. One is chosen at random.
    seed
        Optional RNG seed.

    Raises
    ------
    ValueError
        If ``alternative_models`` is empty.
    """

    name = "model_swap"
    category = "model"
    description = "Replace the model with a different one"

    def __init__(
        self,
        *,
        alternative_models: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._alternatives = alternative_models or [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "claude-3-haiku-20240307",
        ]
        self._original_model: str = ""
        self._new_model: str = ""

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Swap config.model to a random alternative."""
        new_config = _deep_copy_config(config)
        new_scenario = _deep_copy_scenario(scenario)
        self._original_model = config.model
        self._new_model = ""

        # Filter out the current model to ensure an actual swap
        candidates = [m for m in self._alternatives if m != config.model]
        if not candidates:
            # All alternatives are the same as current --- no-op
            self._new_model = config.model
            return new_config, new_scenario

        self._new_model = self._rng.choice(candidates)
        new_config.model = self._new_model
        return new_config, new_scenario

    def describe_mutation(self) -> str:
        if not self._original_model:
            return "Model swap mutation (not yet applied)"
        if self._original_model == self._new_model:
            return f"Model swap mutation (no-op: '{self._original_model}' is the only option)"
        return f"Swapped model: '{self._original_model}' -> '{self._new_model}'"


# ===================================================================
# ModelVersionMutator
# ===================================================================


class ModelVersionMutator(MutationOperator):
    """Change model version within the same family.

    Uses a built-in family table to find related models. For example,
    ``gpt-4o`` might mutate to ``gpt-4o-mini`` or ``gpt-4-turbo``.

    Parameters
    ----------
    family_table
        Custom model family mapping. Defaults to a built-in table
        covering OpenAI and Anthropic model families.
    seed
        Optional RNG seed.
    """

    name = "model_version"
    category = "model"
    description = "Change to a different version of the same model family"

    def __init__(
        self,
        *,
        family_table: dict[str, list[str]] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._families = family_table if family_table is not None else _MODEL_FAMILIES
        self._original_model: str = ""
        self._new_model: str = ""

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Change model version within its family."""
        new_config = _deep_copy_config(config)
        new_scenario = _deep_copy_scenario(scenario)
        self._original_model = config.model
        self._new_model = ""

        # Look up family members
        family = self._families.get(config.model)
        if not family:
            # Unknown model --- try prefix matching
            family = self._find_family_by_prefix(config.model)

        if not family:
            self._new_model = config.model
            return new_config, new_scenario

        self._new_model = self._rng.choice(family)
        new_config.model = self._new_model
        return new_config, new_scenario

    def describe_mutation(self) -> str:
        if not self._original_model:
            return "Model version mutation (not yet applied)"
        if self._original_model == self._new_model:
            return (
                f"Model version mutation (no-op: '{self._original_model}' "
                f"has no known family members)"
            )
        return f"Changed model version: '{self._original_model}' -> '{self._new_model}'"

    def _find_family_by_prefix(self, model: str) -> list[str] | None:
        """Try to match a model to a family by shared prefix."""
        for known_model, family in self._families.items():
            # Check if they share a meaningful prefix (at least 5 chars)
            prefix_len = len(_common_prefix(model, known_model))
            if prefix_len >= 5:
                return [m for m in family if m != model]
        return None


def _common_prefix(a: str, b: str) -> str:
    """Return the longest common prefix of two strings."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


__all__ = [
    "ModelSwapMutator",
    "ModelVersionMutator",
]
