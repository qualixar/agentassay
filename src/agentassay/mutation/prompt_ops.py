"""Prompt mutation operators (M_prompt) for AgentAssay.

Implements four operators that target the prompt/instruction surface
of the agent specification:

- ``PromptSynonymMutator``: Replace key words with synonyms.
- ``PromptOrderMutator``: Reorder sentences in instructions.
- ``PromptNoiseMutator``: Inject typos and character-level noise.
- ``PromptDropMutator``: Remove a random sentence.

Each operator tests a distinct dimension of prompt fragility.
"""

from __future__ import annotations

import copy
import re
import string
from typing import Any

from agentassay.core.models import AgentConfig, TestScenario
from agentassay.mutation.base import (
    MutationOperator,
    _deep_copy_config,
    _rebuild_scenario,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# Synonym table for PromptSynonymMutator
# ---------------------------------------------------------------------------

_SYNONYM_TABLE: dict[str, list[str]] = {
    "analyze": ["examine", "investigate", "evaluate", "assess", "review"],
    "calculate": ["compute", "determine", "figure out", "work out"],
    "create": ["generate", "produce", "build", "construct", "make"],
    "describe": ["explain", "outline", "detail", "characterize"],
    "find": ["locate", "identify", "discover", "search for", "detect"],
    "help": ["assist", "support", "aid", "guide"],
    "list": ["enumerate", "itemize", "catalogue", "outline"],
    "provide": ["supply", "furnish", "deliver", "give", "offer"],
    "search": ["look up", "query", "retrieve", "look for"],
    "summarize": ["condense", "recap", "brief", "digest"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "write": ["compose", "draft", "author", "produce"],
    "important": ["critical", "essential", "vital", "crucial"],
    "always": ["consistently", "invariably", "at all times"],
    "never": ["under no circumstances", "at no point", "do not"],
    "must": ["shall", "is required to", "needs to", "has to"],
    "should": ["ought to", "is expected to", "is recommended to"],
    "ensure": ["verify", "confirm", "guarantee", "make sure"],
    "return": ["output", "produce", "yield", "respond with"],
    "correct": ["accurate", "right", "proper", "precise"],
}


# ===================================================================
# PromptSynonymMutator
# ===================================================================


class PromptSynonymMutator(MutationOperator):
    """Replace key words in the prompt/instructions with synonyms.

    Targets the ``"prompt"`` or ``"instructions"`` key in
    ``scenario.input_data``. If neither key exists, the mutation is a
    no-op (identity mutation).

    Parameters
    ----------
    max_replacements
        Maximum number of synonym substitutions per mutation.
    synonym_table
        Custom synonym mapping. Defaults to a built-in table of
        common instruction verbs and modifiers.
    seed
        Optional RNG seed.
    """

    name = "prompt_synonym"
    category = "prompt"
    description = "Replace key words in prompt with synonyms"

    def __init__(
        self,
        *,
        max_replacements: int = 3,
        synonym_table: dict[str, list[str]] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._max_replacements = max(1, max_replacements)
        self._synonyms = synonym_table if synonym_table is not None else _SYNONYM_TABLE
        self._last_replacements: list[tuple[str, str]] = []

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Apply synonym substitutions to the prompt text."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._last_replacements = []

        text_key = self._find_text_key(input_data)
        if text_key is None:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        text = str(input_data[text_key])
        text_lower = text.lower()

        # Find all replaceable words present in the text
        candidates: list[str] = [
            word for word in self._synonyms if word in text_lower
        ]

        if not candidates:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        # Pick up to max_replacements candidates
        n = min(self._max_replacements, len(candidates))
        chosen = self._rng.sample(candidates, n)

        for word in chosen:
            synonym = self._rng.choice(self._synonyms[word])
            # Case-insensitive replacement (first occurrence only)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(synonym, text, count=1)
            self._last_replacements.append((word, synonym))

        input_data[text_key] = text
        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        if not self._last_replacements:
            return "Prompt synonym mutation (no replacements applied)"
        pairs = ", ".join(f"'{w}'->'{s}'" for w, s in self._last_replacements)
        return f"Prompt synonym substitution: {pairs}"

    @staticmethod
    def _find_text_key(input_data: dict[str, Any]) -> str | None:
        """Find the prompt/instructions key in input_data."""
        for key in ("prompt", "instructions", "system_prompt", "query", "input", "message"):
            if key in input_data and isinstance(input_data[key], str):
                return key
        return None


# ===================================================================
# PromptOrderMutator
# ===================================================================


class PromptOrderMutator(MutationOperator):
    """Reorder sentences in the prompt/instructions.

    Tests whether the agent's behavior is sensitive to instruction
    ordering --- a known fragility in LLM-based systems.

    Parameters
    ----------
    seed
        Optional RNG seed.
    """

    name = "prompt_order"
    category = "prompt"
    description = "Reorder sentences in the prompt/instructions"

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._original_order: list[str] = []
        self._new_order: list[str] = []

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Shuffle sentence order in the prompt."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._original_order = []
        self._new_order = []

        text_key = PromptSynonymMutator._find_text_key(input_data)
        if text_key is None:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        sentences = _split_sentences(str(input_data[text_key]))
        if len(sentences) <= 1:
            # Nothing to reorder
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        self._original_order = list(sentences)
        shuffled = list(sentences)

        # Shuffle until we get a different ordering (bounded attempts)
        for _ in range(20):
            self._rng.shuffle(shuffled)
            if shuffled != sentences:
                break

        self._new_order = list(shuffled)
        input_data[text_key] = " ".join(shuffled)
        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        if not self._original_order:
            return "Prompt order mutation (no sentences to reorder)"
        return (
            f"Reordered {len(self._original_order)} sentences in prompt "
            f"(original order -> shuffled)"
        )


# ===================================================================
# PromptNoiseMutator
# ===================================================================


class PromptNoiseMutator(MutationOperator):
    """Inject typos and character-level noise into the prompt.

    Tests robustness to imperfect user input --- a common real-world
    condition that agents must handle gracefully.

    Parameters
    ----------
    noise_rate
        Probability of corrupting each word (0.0 to 1.0).
    seed
        Optional RNG seed.
    """

    name = "prompt_noise"
    category = "prompt"
    description = "Add typos and character-level noise to the prompt"

    def __init__(
        self,
        *,
        noise_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._noise_rate = max(0.0, min(1.0, noise_rate))
        self._num_corrupted: int = 0

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Inject character-level noise into prompt words."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._num_corrupted = 0

        text_key = PromptSynonymMutator._find_text_key(input_data)
        if text_key is None:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        text = str(input_data[text_key])
        words = text.split()
        corrupted_words: list[str] = []

        for word in words:
            if len(word) > 2 and self._rng.random() < self._noise_rate:
                corrupted_words.append(self._corrupt_word(word))
                self._num_corrupted += 1
            else:
                corrupted_words.append(word)

        input_data[text_key] = " ".join(corrupted_words)
        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        return f"Injected typos into {self._num_corrupted} words (rate={self._noise_rate})"

    def _corrupt_word(self, word: str) -> str:
        """Apply a random character-level corruption to a word."""
        corruption = self._rng.choice(["swap", "delete", "insert", "substitute"])
        chars = list(word)

        if corruption == "swap" and len(chars) >= 3:
            # Swap two adjacent characters (not first/last for readability)
            idx = self._rng.randint(1, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        elif corruption == "delete" and len(chars) >= 3:
            idx = self._rng.randint(1, len(chars) - 2)
            chars.pop(idx)

        elif corruption == "insert":
            idx = self._rng.randint(1, len(chars) - 1)
            chars.insert(idx, self._rng.choice(string.ascii_lowercase))

        elif corruption == "substitute" and len(chars) >= 3:
            idx = self._rng.randint(1, len(chars) - 2)
            chars[idx] = self._rng.choice(string.ascii_lowercase)

        return "".join(chars)


# ===================================================================
# PromptDropMutator
# ===================================================================


class PromptDropMutator(MutationOperator):
    """Remove a random sentence from the prompt/instructions.

    Tests whether the agent's behavior depends on ALL instructions
    being present, or if it can operate with partial specifications.
    This is a critical fragility dimension --- production prompts often
    accumulate instructions, and removing one reveals which are load-
    bearing.

    Parameters
    ----------
    seed
        Optional RNG seed.
    """

    name = "prompt_drop"
    category = "prompt"
    description = "Remove a random sentence from the prompt"

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._dropped_sentence: str = ""

    def mutate(
        self,
        config: AgentConfig,
        scenario: TestScenario,
    ) -> tuple[AgentConfig, TestScenario]:
        """Drop one random sentence from the prompt."""
        new_config = _deep_copy_config(config)
        input_data = copy.deepcopy(scenario.input_data)
        self._dropped_sentence = ""

        text_key = PromptSynonymMutator._find_text_key(input_data)
        if text_key is None:
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        sentences = _split_sentences(str(input_data[text_key]))
        if len(sentences) <= 1:
            # Cannot drop from a single-sentence prompt
            return new_config, _rebuild_scenario(scenario, input_data=input_data)

        drop_idx = self._rng.randint(0, len(sentences) - 1)
        self._dropped_sentence = sentences[drop_idx]
        remaining = sentences[:drop_idx] + sentences[drop_idx + 1 :]

        input_data[text_key] = " ".join(remaining)
        return new_config, _rebuild_scenario(scenario, input_data=input_data)

    def describe_mutation(self) -> str:
        if not self._dropped_sentence:
            return "Prompt drop mutation (no sentence dropped)"
        preview = (
            self._dropped_sentence[:80] + "..."
            if len(self._dropped_sentence) > 80
            else self._dropped_sentence
        )
        return f"Dropped sentence: \"{preview}\""


__all__ = [
    "PromptSynonymMutator",
    "PromptOrderMutator",
    "PromptNoiseMutator",
    "PromptDropMutator",
]
