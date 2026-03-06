# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Metamorphic test runner for AgentAssay.

Orchestrates the execution of metamorphic relations against an agent
callable. For each relation:

1. Run the **source** scenario through the agent -> source trace
2. **Transform** the input using the relation's transform function
3. Run the **follow-up** scenario through the agent -> follow-up trace
4. **Check** whether the metamorphic relation holds

The runner aggregates results per-family and computes a violation rate
(the fraction of relations that did NOT hold). A high violation rate
indicates the agent has behavioral inconsistencies.

The ``DecompositionRelation`` receives special handling: instead of a
single follow-up, it decomposes the scenario into multiple sub-scenarios,
runs each one, composes the results, and checks the composed relation.

Pattern note: This is the **Mediator pattern** -- the runner mediates
between the agent callable and the relation strategies without either
needing to know about the other.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentassay.core.models import (
    AgentConfig,
    ExecutionTrace,
    TestScenario,
)
from agentassay.metamorphic.relations import (
    ConsistencyRelation,
    DecompositionRelation,
    InputPermutationRelation,
    IrrelevantAdditionRelation,
    MetamorphicRelation,
    MetamorphicResult,
    MonotonicityRelation,
    ToolOrderRelation,
    TypographicalPerturbation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default relations (sensible out-of-the-box configuration)
# ---------------------------------------------------------------------------

DEFAULT_RELATIONS: list[MetamorphicRelation] = [
    # Family 1: Permutation
    InputPermutationRelation(threshold=0.8, seed=42),
    ToolOrderRelation(threshold=0.8, seed=42),
    # Family 2: Perturbation
    TypographicalPerturbation(threshold=0.8, num_typos=2, seed=42),
    IrrelevantAdditionRelation(threshold=0.8, num_additions=2, seed=42),
    # Family 3: Composition
    DecompositionRelation(threshold=0.7, seed=42),
    # Family 4: Oracle
    ConsistencyRelation(threshold=0.7, seed=42),
    MonotonicityRelation(tolerance=0.1, seed=42),
]
"""Default set of all 7 metamorphic relations with production-ready thresholds.

These thresholds are deliberately conservative -- a similarity of 0.7-0.8
allows for natural variation in LLM wording while still catching
significant behavioral regressions.
"""


# ---------------------------------------------------------------------------
# Aggregate result model
# ---------------------------------------------------------------------------


class MetamorphicTestResult(BaseModel):
    """Aggregated result from running multiple metamorphic relations.

    Frozen to guarantee immutability after creation.

    Attributes
    ----------
    results
        Individual ``MetamorphicResult`` for each relation tested.
    total_relations
        Total number of relations that were tested.
    violations
        Count of relations that did NOT hold.
    violation_rate
        Fraction of relations that violated (violations / total).
    per_family
        Per-family breakdown with counts and rates.
    timestamp
        When the metamorphic test session completed (UTC).
    """

    model_config = ConfigDict(frozen=True)

    results: list[MetamorphicResult] = Field(default_factory=list)
    total_relations: int = Field(ge=0, default=0)
    violations: int = Field(ge=0, default=0)
    violation_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    per_family: dict[str, dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MetamorphicRunner:
    """Executes metamorphic testing against an agent callable.

    The runner is framework-agnostic: any function that accepts
    ``dict[str, Any]`` and returns an ``ExecutionTrace`` can be tested.

    Parameters
    ----------
    agent_callable
        The agent under test. Takes ``dict[str, Any]`` (input data)
        and returns an ``ExecutionTrace``.
    config
        Agent configuration (framework, model, version metadata).
    relations
        Metamorphic relations to test. If ``None``, uses
        ``DEFAULT_RELATIONS``.

    Example
    -------
    >>> from agentassay.metamorphic import MetamorphicRunner, DEFAULT_RELATIONS
    >>> runner = MetamorphicRunner(my_agent, agent_config)
    >>> result = runner.test_all(scenario)
    >>> print(f"Violation rate: {result.violation_rate:.1%}")
    """

    def __init__(
        self,
        agent_callable: Callable[[dict[str, Any]], ExecutionTrace],
        config: AgentConfig,
        relations: list[MetamorphicRelation] | None = None,
    ) -> None:
        self._agent_callable = agent_callable
        self._config = config
        self._relations = relations if relations is not None else list(DEFAULT_RELATIONS)

    # -- Public properties ---------------------------------------------------

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def relations(self) -> list[MetamorphicRelation]:
        """Currently configured metamorphic relations."""
        return list(self._relations)

    # -- Core execution: run agent -------------------------------------------

    def _run_agent(self, scenario: TestScenario) -> ExecutionTrace:
        """Run the agent callable on a scenario and return the trace.

        On failure, returns a fallback trace with ``success=False``
        and the error recorded.
        """
        start_ms = time.monotonic() * 1000.0
        try:
            trace = self._agent_callable(scenario.input_data)
            return trace
        except Exception as exc:
            elapsed_ms = (time.monotonic() * 1000.0) - start_ms
            logger.error(
                "Agent failed on scenario %s: %s\n%s",
                scenario.scenario_id,
                exc,
                traceback.format_exc(),
            )
            return ExecutionTrace(
                trace_id=str(uuid.uuid4()),
                scenario_id=scenario.scenario_id,
                steps=[],
                input_data=scenario.input_data,
                output_data=None,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                total_duration_ms=elapsed_ms,
                total_cost_usd=0.0,
                model=self._config.model,
                framework=self._config.framework,
            )

    # -- Single relation test ------------------------------------------------

    def test_relation(
        self,
        scenario: TestScenario,
        relation: MetamorphicRelation,
    ) -> MetamorphicResult:
        """Test a single metamorphic relation against a scenario.

        Steps:
        1. Run the source scenario through the agent.
        2. Transform the input using the relation.
        3. Run the follow-up scenario through the agent.
        4. Check whether the relation holds.

        For ``DecompositionRelation``, uses the full decomposition
        path (multiple sub-scenarios + composition).

        Parameters
        ----------
        scenario
            The source test scenario.
        relation
            The metamorphic relation to test.

        Returns
        -------
        MetamorphicResult
            Whether the relation held, with similarity score and details.
        """
        logger.info(
            "Testing relation '%s' (%s) on scenario '%s'",
            relation.name,
            relation.family,
            scenario.name,
        )

        # 1. Run source
        source_trace = self._run_agent(scenario)

        # 2. Handle DecompositionRelation specially
        if isinstance(relation, DecompositionRelation):
            return self._test_decomposition(scenario, relation, source_trace)

        # 3. Transform input
        followup_scenario = relation.transform_input(scenario)

        # 4. Run follow-up
        followup_trace = self._run_agent(followup_scenario)

        # 5. Check relation
        result = relation.check_relation(source_trace, followup_trace)

        if not result.holds:
            logger.warning(
                "Relation '%s' VIOLATED on scenario '%s' "
                "(similarity=%.3f)",
                relation.name,
                scenario.name,
                result.similarity_score,
            )

        return result

    def _test_decomposition(
        self,
        scenario: TestScenario,
        relation: DecompositionRelation,
        source_trace: ExecutionTrace,
    ) -> MetamorphicResult:
        """Handle the full decomposition path.

        1. Decompose the scenario into sub-scenarios.
        2. Run each sub-scenario through the agent.
        3. Compose the sub-results.
        4. Check the composed relation against the direct result.

        If decomposition produces only one sub-scenario (the original),
        falls back to the standard single-followup path.
        """
        sub_scenarios = relation.decompose(scenario)

        if len(sub_scenarios) <= 1:
            # Decomposition didn't produce meaningful subtasks;
            # fall back to standard path
            logger.debug(
                "Decomposition of '%s' produced %d sub-scenarios; "
                "falling back to standard path.",
                scenario.name,
                len(sub_scenarios),
            )
            followup_scenario = relation.transform_input(scenario)
            followup_trace = self._run_agent(followup_scenario)
            return relation.check_relation(source_trace, followup_trace)

        # Run all sub-scenarios
        sub_traces: list[ExecutionTrace] = []
        for i, sub in enumerate(sub_scenarios):
            logger.debug(
                "Running sub-scenario %d/%d: '%s'",
                i + 1,
                len(sub_scenarios),
                sub.name,
            )
            sub_trace = self._run_agent(sub)
            sub_traces.append(sub_trace)

        # Check the composed relation
        result = relation.check_composed_relation(source_trace, sub_traces)

        if not result.holds:
            logger.warning(
                "Decomposition relation VIOLATED on scenario '%s' "
                "(similarity=%.3f, %d subtasks)",
                scenario.name,
                result.similarity_score,
                len(sub_traces),
            )

        return result

    # -- Full test suite -----------------------------------------------------

    def test_all(
        self,
        scenario: TestScenario,
        relations: list[MetamorphicRelation] | None = None,
    ) -> MetamorphicTestResult:
        """Test all configured metamorphic relations against a scenario.

        Parameters
        ----------
        scenario
            The source test scenario.
        relations
            Optional override list of relations. If ``None``, uses the
            runner's configured relations.

        Returns
        -------
        MetamorphicTestResult
            Aggregated results with per-family breakdown and overall
            violation rate.
        """
        rels = relations if relations is not None else self._relations
        if not rels:
            logger.warning("No metamorphic relations configured; returning empty result.")
            return MetamorphicTestResult()

        logger.info(
            "Running %d metamorphic relations on scenario '%s'",
            len(rels),
            scenario.name,
        )

        results: list[MetamorphicResult] = []

        for relation in rels:
            try:
                result = self.test_relation(scenario, relation)
                results.append(result)
            except Exception as exc:
                # Record a failed relation check as a violation
                logger.error(
                    "Relation '%s' raised an error: %s\n%s",
                    relation.name,
                    exc,
                    traceback.format_exc(),
                )
                results.append(
                    MetamorphicResult(
                        relation_name=relation.name,
                        relation_family=relation.family,
                        holds=False,
                        source_output=None,
                        followup_output=None,
                        similarity_score=0.0,
                        details={
                            "error": f"{type(exc).__name__}: {exc}",
                            "transform": "error_during_test",
                        },
                    )
                )

        # Aggregate
        total = len(results)
        violations = sum(1 for r in results if not r.holds)
        violation_rate = violations / total if total > 0 else 0.0

        # Per-family aggregation
        families: dict[str, list[MetamorphicResult]] = {}
        for r in results:
            families.setdefault(r.relation_family, []).append(r)

        per_family: dict[str, dict[str, Any]] = {}
        for family, family_results in families.items():
            family_total = len(family_results)
            family_violations = sum(1 for r in family_results if not r.holds)
            per_family[family] = {
                "tested": family_total,
                "violations": family_violations,
                "rate": family_violations / family_total if family_total > 0 else 0.0,
                "relations": [r.relation_name for r in family_results],
            }

        return MetamorphicTestResult(
            results=results,
            total_relations=total,
            violations=violations,
            violation_rate=violation_rate,
            per_family=per_family,
        )
