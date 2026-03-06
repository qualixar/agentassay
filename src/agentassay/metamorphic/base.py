# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Base classes and shared utilities for metamorphic relations.

Defines the ``MetamorphicRelation`` abstract base class, the
``MetamorphicResult`` frozen model, and internal helper functions
used across all relation families (permutation, perturbation,
composition, oracle).

Pattern note: This follows the **Strategy pattern** -- each relation is
an interchangeable strategy that the ``MetamorphicRunner`` applies
uniformly.  New relations can be added without touching the runner.
"""

from __future__ import annotations

import copy
import random
import uuid
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentassay.core.models import ExecutionTrace, TestScenario

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class MetamorphicResult(BaseModel):
    """Outcome of checking a single metamorphic relation.

    Frozen (immutable) to guarantee that results are never silently
    mutated after creation -- important for statistical aggregation
    downstream.

    Attributes
    ----------
    relation_name
        Human-readable name of the relation that was tested.
    relation_family
        One of ``"permutation"``, ``"perturbation"``, ``"composition"``,
        ``"oracle"``.
    holds
        ``True`` if the metamorphic relation held between source and
        follow-up outputs.
    source_output
        The ``output_data`` from the source execution trace.
    followup_output
        The ``output_data`` from the follow-up execution trace.
    similarity_score
        Continuous measure in [0.0, 1.0] indicating how close the
        source and follow-up outputs are. 1.0 means identical.
    details
        Arbitrary metadata about the check (thresholds used, transform
        description, intermediate computations, etc.).
    """

    model_config = ConfigDict(frozen=True)

    relation_name: str
    relation_family: str
    holds: bool
    source_output: Any = None
    followup_output: Any = None
    similarity_score: float = Field(ge=0.0, le=1.0, default=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Similarity helpers (internal)
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    """Coerce a value to a string for comparison.

    Handles ``None``, primitives, lists, and dicts gracefully.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _text_similarity(a: Any, b: Any) -> float:
    """Compute sequence-matcher similarity between two values.

    Uses ``difflib.SequenceMatcher`` (Ratcliff/Obershelp algorithm)
    which produces a ratio in [0.0, 1.0]. Both values are stringified
    and lowercased before comparison.
    """
    str_a = _stringify(a).lower().strip()
    str_b = _stringify(b).lower().strip()
    if not str_a and not str_b:
        return 1.0
    if not str_a or not str_b:
        return 0.0
    return SequenceMatcher(None, str_a, str_b).ratio()


def _exact_match(a: Any, b: Any) -> bool:
    """Check exact equality between two values, stringified and stripped."""
    return _stringify(a).strip().lower() == _stringify(b).strip().lower()


def _deep_copy_scenario(scenario: TestScenario, **overrides: Any) -> TestScenario:
    """Create a deep copy of a scenario with optional field overrides.

    Since ``TestScenario`` is frozen, we must reconstruct it from
    a dict. The ``input_data`` is deep-copied so mutations to the
    follow-up do not affect the source.
    """
    data = scenario.model_dump()
    # Deep-copy input_data to avoid aliasing
    data["input_data"] = copy.deepcopy(data["input_data"])
    # Generate a unique scenario_id for the follow-up
    data["scenario_id"] = f"{scenario.scenario_id}__followup__{uuid.uuid4().hex[:8]}"
    data.update(overrides)
    return TestScenario(**data)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class MetamorphicRelation(ABC):
    """Abstract base class for metamorphic relations.

    A metamorphic relation defines a **transformation** on the input
    and a **predicate** on the (source output, follow-up output) pair.
    If the predicate fails, the agent's behavior is inconsistent with
    the stated relation -- a potential bug.

    Subclasses must implement ``transform_input`` (how to create the
    follow-up test case) and ``check_relation`` (whether the relation
    holds between outputs).

    Parameters
    ----------
    name
        Human-readable identifier (e.g., ``"input_permutation"``).
    family
        Relation family: one of ``"permutation"``, ``"perturbation"``,
        ``"composition"``, ``"oracle"``.
    description
        One-sentence description of what this relation tests.
    seed
        Optional RNG seed for reproducible transformations.
    """

    def __init__(
        self,
        name: str,
        family: str,
        description: str,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.family = family
        self.description = description
        self._rng = random.Random(seed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, family={self.family!r})"

    @abstractmethod
    def transform_input(self, scenario: TestScenario) -> TestScenario:
        """Create a follow-up test case by transforming the source.

        Must return a new ``TestScenario`` (the original is frozen and
        must not be mutated). Use ``_deep_copy_scenario`` helper.
        """

    @abstractmethod
    def check_relation(
        self,
        source_trace: ExecutionTrace,
        followup_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Check whether the metamorphic relation holds.

        Compares the source execution trace with the follow-up
        execution trace and returns a ``MetamorphicResult``.
        """
