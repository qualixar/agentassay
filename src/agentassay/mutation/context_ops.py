"""Context mutation operators (M_context) for AgentAssay.

Implements three operators that target the context/input data surface
of the agent specification:

- ``ContextTruncationMutator``: Truncate context to a fraction of its
  original length.
- ``ContextNoiseMutator``: Inject irrelevant distractor information.
- ``ContextPermutationMutator``: Reorder sections within the context.

Each operator tests a distinct dimension of context-related fragility
that agents encounter in production (RAG truncation, noisy retrieval,
and ordering sensitivity).
"""

from __future__ import annotations

import copy
from typing import Any

from agentassay.core.models import AgentConfig, TestScenario
from agentassay.mutation.base import (
    MutationOperator,
    _deep_copy_config,
    _rebuild_scenario,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# Distractor sentences for ContextNoiseMutator
# ---------------------------------------------------------------------------

_DISTRACTOR_SENTENCES: list[str] = [
    "The weather in Tokyo is usually mild during spring.",
    "Python was first released in 1991 by Guido van Rossum.",
    "The speed of light is approximately 299,792 km/s.",
    "Coffee is one of the most traded commodities worldwide.",
    "The Great Wall of China is visible from low Earth orbit.",
    "Elephants are the largest living land animals.",
    "The Fibonacci sequence appears frequently in nature.",
    "Marie Curie won two Nobel Prizes in different sciences.",
    "Bananas are technically berries, but strawberries are not.",
    "The Pacific Ocean covers more area than all land combined.",
]


# ===================================================================
# ContextTruncationMutator
# ===================================================================


class ContextTruncationMutator(MutationOperator):
    """Truncate input context to a percentage of its original length.

    Tests whether the agent degrades gracefully when given incomplete
    context --- a real-world condition when token limits are hit, RAG
    retrieval is partial, or upstream systems produce truncated output.

    Parameters
    ----------
    keep_ratio
        Fraction of the context to keep (0.0 to 1.0). Default 0.5
        keeps the first half.
    target_keys
        Keys in ``scenario.input_data`` to truncate. If None, targets
        all string-valued keys except ``"prompt"`` and ``"instructions"``.
    seed
        Optional RNG seed.
    """

    name = "context_truncation"
    category = "context"
    description = "Truncate input context to a fraction of its length"

    def __init__(
        self,
        *,
        keep_ratio: float = 0.5,
        target_keys: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._keep_ratio = max(0.0, min(1.0, keep_ratio))
        self._target_keys = target_keys
        self._truncated_keys: list[str] = []

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Truncate string values in scenario.input_data."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._truncated_keys = []

        keys_to_truncate = self._resolve_target_keys(input_data)

        for key in keys_to_truncate:
            value = input_data[key]
            if isinstance(value, str) and len(value) > 0:
                new_length = max(1, int(len(value) * self._keep_ratio))
                input_data[key] = value[:new_length]
                self._truncated_keys.append(key)
            elif isinstance(value, list) and len(value) > 0:
                new_length = max(1, int(len(value) * self._keep_ratio))
                input_data[key] = value[:new_length]
                self._truncated_keys.append(key)

        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        if not self._truncated_keys:
            return f"Context truncation mutation (keep_ratio={self._keep_ratio}, nothing truncated)"
        keys = ", ".join(self._truncated_keys)
        return f"Truncated to {self._keep_ratio:.0%} of original: [{keys}]"

    def _resolve_target_keys(self, input_data: dict[str, Any]) -> list[str]:
        """Determine which keys in input_data to truncate."""
        if self._target_keys is not None:
            return [k for k in self._target_keys if k in input_data]

        # Auto-detect: all string or list keys except prompt/instructions
        skip = {"prompt", "instructions", "system_prompt", "query", "input", "message"}
        return [
            k
            for k, v in input_data.items()
            if k not in skip and isinstance(v, (str, list))
        ]


# ===================================================================
# ContextNoiseMutator
# ===================================================================


class ContextNoiseMutator(MutationOperator):
    """Add irrelevant distractor information to the context.

    Tests the agent's ability to distinguish signal from noise. In
    production, RAG systems often inject partially relevant or
    completely irrelevant passages alongside useful context.

    Parameters
    ----------
    num_distractors
        Number of irrelevant sentences to inject.
    distractor_pool
        Custom distractor sentences. Defaults to a built-in pool.
    target_key
        Specific key in ``scenario.input_data`` to inject into.
        Defaults to ``"context"``, then falls back to the first
        string-valued key that is not a prompt field.
    seed
        Optional RNG seed.
    """

    name = "context_noise"
    category = "context"
    description = "Inject irrelevant information into the context"

    def __init__(
        self,
        *,
        num_distractors: int = 3,
        distractor_pool: list[str] | None = None,
        target_key: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._num_distractors = max(1, num_distractors)
        self._distractors = distractor_pool or _DISTRACTOR_SENTENCES
        self._target_key = target_key
        self._injected_key: str = ""
        self._num_injected: int = 0

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Inject distractor sentences into the context."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._injected_key = ""
        self._num_injected = 0

        key = self._resolve_target_key(input_data)
        if key is None:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        self._injected_key = key
        n = min(self._num_distractors, len(self._distractors))
        chosen = self._rng.sample(self._distractors, n)
        self._num_injected = len(chosen)

        value = input_data[key]
        if isinstance(value, str):
            # Interleave distractors into the text
            sentences = _split_sentences(value)
            for distractor in chosen:
                insert_pos = self._rng.randint(0, len(sentences))
                sentences.insert(insert_pos, distractor)
            input_data[key] = " ".join(sentences)
        elif isinstance(value, list):
            # Insert distractor items into the list
            for distractor in chosen:
                insert_pos = self._rng.randint(0, len(value))
                value.insert(insert_pos, distractor)
            input_data[key] = value

        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        if not self._injected_key:
            return "Context noise mutation (no suitable key found)"
        return f"Injected {self._num_injected} distractors into '{self._injected_key}'"

    def _resolve_target_key(self, input_data: dict[str, Any]) -> str | None:
        """Find the best key to inject distractors into."""
        if self._target_key and self._target_key in input_data:
            return self._target_key

        # Preferred keys
        for key in ("context", "documents", "passages", "references", "background"):
            if key in input_data and isinstance(input_data[key], (str, list)):
                return key

        # Fallback: first string/list key that is not a prompt field
        prompt_keys = {"prompt", "instructions", "system_prompt", "query", "input", "message"}
        for key, value in input_data.items():
            if key not in prompt_keys and isinstance(value, (str, list)):
                return key

        return None


# ===================================================================
# ContextPermutationMutator
# ===================================================================


class ContextPermutationMutator(MutationOperator):
    """Reorder sections or paragraphs within the context.

    Tests whether the agent is sensitive to the ordering of contextual
    information --- e.g., whether it exhibits recency bias (preferring
    later context) or primacy bias (preferring earlier context).

    Sections are detected by double-newline boundaries or by list
    elements if the context is a list.

    Parameters
    ----------
    target_key
        Specific key in ``scenario.input_data`` to permute.
        Resolution logic matches ``ContextNoiseMutator``.
    seed
        Optional RNG seed.
    """

    name = "context_permutation"
    category = "context"
    description = "Reorder sections within the context"

    def __init__(
        self,
        *,
        target_key: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._target_key = target_key
        self._permuted_key: str = ""
        self._num_sections: int = 0

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Shuffle context sections."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._permuted_key = ""
        self._num_sections = 0

        # Reuse the same resolution logic as ContextNoiseMutator
        key = ContextNoiseMutator(
            target_key=self._target_key, seed=None
        )._resolve_target_key(input_data)

        if key is None:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        value = input_data[key]
        self._permuted_key = key

        if isinstance(value, list):
            if len(value) <= 1:
                return new_config, _rebuild_scenario(scenario, input_data=input_data)
            self._num_sections = len(value)
            shuffled = list(value)
            for _ in range(20):
                self._rng.shuffle(shuffled)
                if shuffled != value:
                    break
            input_data[key] = shuffled

        elif isinstance(value, str):
            # Split on double-newline (paragraph boundaries)
            sections = [s.strip() for s in value.split("\n\n") if s.strip()]
            if len(sections) <= 1:
                # Try single-newline as fallback
                sections = [s.strip() for s in value.split("\n") if s.strip()]
            if len(sections) <= 1:
                return new_config, _rebuild_scenario(scenario, input_data=input_data)

            self._num_sections = len(sections)
            original = list(sections)
            for _ in range(20):
                self._rng.shuffle(sections)
                if sections != original:
                    break
            input_data[key] = "\n\n".join(sections)

        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        if not self._permuted_key:
            return "Context permutation mutation (no suitable key found)"
        return f"Permuted {self._num_sections} sections in '{self._permuted_key}'"


__all__ = [
    "ContextTruncationMutator",
    "ContextNoiseMutator",
    "ContextPermutationMutator",
]
