"""Perturbation robustness metamorphic relations.

Family 2 of the metamorphic relation taxonomy. Tests whether agents
produce consistent outputs when small, semantically irrelevant changes
are introduced to the input.

Relations:
    - ``TypographicalPerturbation`` -- introduces controlled character-level
      mutations (swaps, deletions, transpositions).
    - ``IrrelevantAdditionRelation`` -- appends unrelated filler sentences.
"""

from __future__ import annotations

from agentassay.core.models import ExecutionTrace, TestScenario
from agentassay.metamorphic.base import (
    MetamorphicRelation,
    MetamorphicResult,
    _deep_copy_scenario,
    _text_similarity,
)


# Character-swap pairs for typographical perturbation
_TYPO_SWAPS: list[tuple[str, str]] = [
    ("a", "s"),
    ("e", "r"),
    ("i", "o"),
    ("t", "y"),
    ("n", "m"),
    ("h", "j"),
    ("d", "f"),
    ("l", "k"),
]


class TypographicalPerturbation(MetamorphicRelation):
    """Tests robustness to minor typographical errors in the input.

    Real users make typos. A robust agent should handle minor spelling
    errors gracefully. This relation introduces a controlled number of
    character-level mutations (swaps, deletions, transpositions) and
    checks that the output remains substantively the same.

    Transform
        Apply ``num_typos`` character-level mutations to the primary
        text field.

    Relation
        Output similarity >= ``threshold``.

    Parameters
    ----------
    threshold
        Minimum similarity for the relation to hold.
    num_typos
        Number of character-level mutations to introduce.
    seed
        RNG seed for reproducible mutations.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_typos: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="typographical_perturbation",
            family="perturbation",
            description=(
                "Add minor typos to the input; output should be "
                "unaffected by minor typographical errors."
            ),
            seed=seed,
        )
        self.threshold = threshold
        self.num_typos = max(1, num_typos)

    def _introduce_typos(self, text: str) -> str:
        """Introduce controlled character-level mutations.

        Mutation types:
        1. **Swap** -- replace a character with an adjacent-keyboard char
        2. **Delete** -- remove a character
        3. **Transpose** -- swap two adjacent characters

        Only mutates alphabetic characters. Preserves word boundaries,
        punctuation, and whitespace.
        """
        if len(text) < 3:
            return text

        chars = list(text)
        # Find positions of alphabetic characters (candidates for mutation)
        alpha_positions = [i for i, c in enumerate(chars) if c.isalpha()]
        if not alpha_positions:
            return text

        applied = 0
        max_attempts = self.num_typos * 5  # safety valve
        attempts = 0

        while applied < self.num_typos and attempts < max_attempts:
            attempts += 1
            pos = self._rng.choice(alpha_positions)
            mutation = self._rng.choice(["swap", "delete", "transpose"])

            if mutation == "swap":
                original = chars[pos].lower()
                # Find a swap pair
                swap_char = None
                for a, b in _TYPO_SWAPS:
                    if original == a:
                        swap_char = b
                        break
                    if original == b:
                        swap_char = a
                        break
                if swap_char:
                    # Preserve case
                    if chars[pos].isupper():
                        swap_char = swap_char.upper()
                    chars[pos] = swap_char
                    applied += 1

            elif mutation == "delete" and len(chars) > 5:
                chars.pop(pos)
                # Rebuild alpha positions
                alpha_positions = [i for i, c in enumerate(chars) if c.isalpha()]
                applied += 1

            elif mutation == "transpose":
                if pos + 1 < len(chars) and chars[pos + 1].isalpha():
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                    applied += 1

        return "".join(chars)

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Introduce typos into the primary text field."""
        followup = _deep_copy_scenario(scenario)
        data = followup.input_data

        text_key = next(
            (k for k in ("query", "input", "prompt", "text", "question") if k in data),
            None,
        )
        if text_key and isinstance(data[text_key], str) and len(data[text_key]) >= 3:
            data[text_key] = self._introduce_typos(data[text_key])
            return _deep_copy_scenario(scenario, input_data=data)

        return _deep_copy_scenario(scenario)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check output similarity after typographical perturbation."""
        sim = _text_similarity(source_trace.output_data, followup_trace.output_data)
        return MetamorphicResult(
            relation_name=self.name,
            relation_family=self.family,
            holds=sim >= self.threshold,
            source_output=source_trace.output_data,
            followup_output=followup_trace.output_data,
            similarity_score=sim,
            details={
                "threshold": self.threshold,
                "num_typos": self.num_typos,
                "transform": "typographical_perturbation",
            },
        )


class IrrelevantAdditionRelation(MetamorphicRelation):
    """Tests robustness to irrelevant information added to the input.

    A reliable agent should not be distracted by information that is
    unrelated to the task at hand. This relation appends irrelevant
    context (filler sentences, unrelated facts) and checks that the
    output remains substantively unchanged.

    Transform
        Append irrelevant context to the primary text field.

    Relation
        Output similarity >= ``threshold``.

    Parameters
    ----------
    threshold
        Minimum similarity for the relation to hold.
    irrelevant_texts
        Optional list of irrelevant sentences to sample from. If not
        provided, uses built-in filler sentences.
    num_additions
        Number of irrelevant sentences to add.
    seed
        RNG seed.
    """

    _DEFAULT_IRRELEVANT: list[str] = [
        "The weather in Tokyo is expected to be sunny this weekend.",
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "Bananas are technically classified as berries by botanists.",
        "The speed of light in a vacuum is approximately 299,792 km/s.",
        "The Great Wall of China is not actually visible from space with the naked eye.",
        "Octopuses have three hearts and blue blood.",
        "The population of Iceland is approximately 370,000 people.",
        "Coffee beans are actually the seeds of a fruit called a coffee cherry.",
        "A day on Venus is longer than a year on Venus.",
        "The shortest war in history lasted 38 minutes between Britain and Zanzibar.",
    ]

    def __init__(
        self,
        threshold: float = 0.8,
        irrelevant_texts: list[str] | None = None,
        num_additions: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name="irrelevant_addition",
            family="perturbation",
            description=(
                "Add irrelevant information to the input; output should "
                "be unaffected by distracting additions."
            ),
            seed=seed,
        )
        self.threshold = threshold
        self.irrelevant_texts = irrelevant_texts or self._DEFAULT_IRRELEVANT
        self.num_additions = max(1, num_additions)

    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Append irrelevant sentences to the primary text field."""
        followup = _deep_copy_scenario(scenario)
        data = followup.input_data

        text_key = next(
            (k for k in ("query", "input", "prompt", "text", "question") if k in data),
            None,
        )
        if text_key and isinstance(data[text_key], str):
            additions = self._rng.sample(
                self.irrelevant_texts,
                k=min(self.num_additions, len(self.irrelevant_texts)),
            )
            original_text = data[text_key]
            data[text_key] = original_text + " " + " ".join(additions)
            return _deep_copy_scenario(scenario, input_data=data)

        return _deep_copy_scenario(scenario)

    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check output similarity after irrelevant additions."""
        sim = _text_similarity(source_trace.output_data, followup_trace.output_data)
        return MetamorphicResult(
            relation_name=self.name,
            relation_family=self.family,
            holds=sim >= self.threshold,
            source_output=source_trace.output_data,
            followup_output=followup_trace.output_data,
            similarity_score=sim,
            details={
                "threshold": self.threshold,
                "num_additions": self.num_additions,
                "transform": "irrelevant_addition",
            },
        )
