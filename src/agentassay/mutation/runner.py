"""Mutation testing runner for AgentAssay.

Orchestrates the execution of mutation operators against an agent-under-test,
collecting kill/survive verdicts and computing the mutation score.

The mutation score is the primary metric:

    MS = |killed_mutants| / |total_mutants|

A high mutation score (> 0.8) means the test suite is sensitive to changes
in prompts, tools, models, and context — i.e., it's a *strong* test suite.
A low score means the tests pass regardless of perturbations — either the
tests are too weak or the agent is exceptionally robust.

Architecture note: The runner follows the same Strategy pattern as
``TrialRunner`` — the agent is injected as a callable, making the
mutation runner framework-agnostic.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentassay.core.models import (
    AgentConfig,
    AssayConfig,
    ExecutionTrace,
    TestScenario,
)
from agentassay.mutation.operators import (
    ContextNoiseMutator,
    ContextPermutationMutator,
    ContextTruncationMutator,
    ModelSwapMutator,
    ModelVersionMutator,
    MutationOperator,
    PromptDropMutator,
    PromptNoiseMutator,
    PromptOrderMutator,
    PromptSynonymMutator,
    ToolNoiseMutator,
    ToolRemovalMutator,
    ToolReorderMutator,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Default operator set
# ===================================================================

def _build_default_operators(seed: int | None = None) -> list[MutationOperator]:
    """Construct the full set of 12 default mutation operators.

    Each operator category (prompt, tool, model, context) is
    represented with multiple specific operators. Default parameters
    are tuned for general-purpose agent testing.

    Parameters
    ----------
    seed
        Optional base seed. Each operator receives ``seed + offset``
        for reproducible but independent randomness.
    """
    def _seed(offset: int) -> int | None:
        return seed + offset if seed is not None else None

    return [
        # Prompt operators (4)
        PromptSynonymMutator(max_replacements=3, seed=_seed(0)),
        PromptOrderMutator(seed=_seed(1)),
        PromptNoiseMutator(noise_rate=0.1, seed=_seed(2)),
        PromptDropMutator(seed=_seed(3)),
        # Tool operators (3)
        ToolRemovalMutator(seed=_seed(4)),
        ToolReorderMutator(seed=_seed(5)),
        ToolNoiseMutator(noise_rate=0.15, seed=_seed(6)),
        # Model operators (2)
        ModelSwapMutator(seed=_seed(7)),
        ModelVersionMutator(seed=_seed(8)),
        # Context operators (3)
        ContextTruncationMutator(keep_ratio=0.5, seed=_seed(9)),
        ContextNoiseMutator(num_distractors=3, seed=_seed(10)),
        ContextPermutationMutator(seed=_seed(11)),
    ]


DEFAULT_OPERATORS: list[MutationOperator] = _build_default_operators()
"""All 12 default mutation operators with sensible defaults (unseeded)."""


# ===================================================================
# Result models (frozen)
# ===================================================================


class MutationResult(BaseModel):
    """Result of applying a single mutation operator to a single scenario.

    Captures both the original and mutant execution outcomes so the
    caller can analyse sensitivity per-operator.

    Attributes
    ----------
    killed
        ``True`` if the mutation changed the test outcome (mutant_passed
        differs from original_passed). A killed mutant proves the test
        suite detected the perturbation.
    score_delta
        ``mutant_score - original_score``. Negative means the mutant
        degraded quality; positive means it (surprisingly) improved.
    """

    model_config = ConfigDict(frozen=True)

    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operator_name: str = Field(min_length=1)
    operator_category: str = Field(min_length=1)
    mutation_description: str = ""

    original_passed: bool = False
    mutant_passed: bool = False
    killed: bool = False

    original_score: float = Field(ge=0.0, le=1.0, default=0.0)
    mutant_score: float = Field(ge=0.0, le=1.0, default=0.0)
    score_delta: float = Field(ge=-1.0, le=1.0, default=0.0)

    original_trace_id: str = ""
    mutant_trace_id: str = ""

    error: str | None = None
    duration_ms: float = Field(ge=0.0, default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MutationSuiteResult(BaseModel):
    """Aggregate result of running a full mutation testing suite.

    The key metric is ``mutation_score``:

        mutation_score = killed_mutants / total_mutants

    ``per_category`` breaks this down by operator category (prompt,
    tool, model, context) to identify which specification dimensions
    the test suite covers well vs. poorly.

    Attributes
    ----------
    per_category
        Mutation score per operator category. A category with 0.0
        means the test suite is completely blind to that mutation
        dimension.
    """

    model_config = ConfigDict(frozen=True)

    suite_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: list[MutationResult] = Field(default_factory=list)

    total_mutants: int = Field(ge=0, default=0)
    killed_mutants: int = Field(ge=0, default=0)
    survived_mutants: int = Field(ge=0, default=0)
    errored_mutants: int = Field(ge=0, default=0)
    mutation_score: float = Field(ge=0.0, le=1.0, default=0.0)

    per_category: dict[str, float] = Field(default_factory=dict)

    total_duration_ms: float = Field(ge=0.0, default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    scenario_id: str = ""
    agent_id: str = ""


# ===================================================================
# Mutation Runner
# ===================================================================


class MutationRunner:
    """Orchestrates mutation testing for an agent.

    Applies mutation operators to the agent's configuration and
    scenarios, executes both original and mutant versions, and
    computes the mutation score.

    Parameters
    ----------
    agent_callable
        A function that takes ``(AgentConfig, dict[str, Any])`` and
        returns an ``ExecutionTrace``. The first argument is the agent
        config (which may be mutated), the second is the scenario
        input data.

        **Why two arguments?** Unlike ``TrialRunner`` which uses a
        fixed config, mutation testing needs to swap models and tools
        per-run. The callable must accept the (potentially mutated)
        config to honor model/tool changes.

        If your agent callable only accepts ``dict[str, Any]``, wrap
        it::

            runner = MutationRunner(
                agent_callable=lambda cfg, inp: my_agent(inp),
                config=agent_config,
                assay_config=assay_config,
            )

    config
        The baseline agent configuration.
    assay_config
        Statistical and resource configuration.
    operators
        Mutation operators to use. Defaults to all 12 standard
        operators.
    seed
        Optional master seed for reproducible operator construction
        when ``operators`` is None.

    Example
    -------
    >>> runner = MutationRunner(my_agent, config, assay_config)
    >>> result = runner.run_suite(scenario)
    >>> print(f"Mutation score: {result.mutation_score:.1%}")
    >>> for cat, score in result.per_category.items():
    ...     print(f"  {cat}: {score:.1%}")
    """

    def __init__(
        self,
        agent_callable: Callable[[AgentConfig, dict[str, Any]], ExecutionTrace],
        config: AgentConfig,
        assay_config: AssayConfig,
        operators: list[MutationOperator] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._agent_callable = agent_callable
        self._config = config
        self._assay_config = assay_config

        if operators is not None:
            self._operators = list(operators)
        else:
            self._operators = _build_default_operators(seed=seed)

    # -- Public properties ---------------------------------------------------

    @property
    def config(self) -> AgentConfig:
        """The baseline agent configuration."""
        return self._config

    @property
    def assay_config(self) -> AssayConfig:
        """Statistical configuration for the assay."""
        return self._assay_config

    @property
    def operators(self) -> list[MutationOperator]:
        """The mutation operators that will be applied."""
        return list(self._operators)

    # -- Single mutation run -------------------------------------------------

    def run_mutation(
        self,
        scenario: TestScenario,
        operator: MutationOperator,
        evaluator: Callable[[ExecutionTrace], float] | None = None,
    ) -> MutationResult:
        """Run a single mutation: execute original, execute mutant, compare.

        Parameters
        ----------
        scenario
            The test scenario to mutate and execute.
        operator
            The mutation operator to apply.
        evaluator
            Optional function that scores an ``ExecutionTrace``,
            returning a float in [0.0, 1.0]. If ``None``, uses
            ``trace.success`` as a binary pass/fail (1.0 / 0.0).

        Returns
        -------
        MutationResult
            Contains original vs. mutant outcomes and whether the
            mutant was killed.
        """
        start_time = time.monotonic()

        # --- Step 1: Execute original -----------------------------------------
        original_trace, original_error = self._execute_agent(
            self._config, scenario.input_data
        )
        original_score = self._evaluate(original_trace, evaluator)
        original_passed = original_score >= 0.5 if original_trace else False

        # --- Step 2: Apply mutation -------------------------------------------
        try:
            mutated_config, mutated_scenario = operator.mutate(
                self._config, scenario
            )
            mutation_desc = operator.describe_mutation()
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "Mutation operator '%s' raised: %s\n%s",
                operator.name,
                exc,
                traceback.format_exc(),
            )
            return MutationResult(
                operator_name=operator.name,
                operator_category=operator.category,
                mutation_description=f"ERROR: {exc}",
                original_passed=original_passed,
                mutant_passed=False,
                killed=True,
                original_score=original_score,
                mutant_score=0.0,
                score_delta=-original_score,
                original_trace_id=original_trace.trace_id if original_trace else "",
                mutant_trace_id="",
                error=str(exc),
                duration_ms=elapsed_ms,
            )

        # --- Step 3: Execute mutant -------------------------------------------
        mutant_trace, mutant_error = self._execute_agent(
            mutated_config, mutated_scenario.input_data
        )
        mutant_score = self._evaluate(mutant_trace, evaluator)
        mutant_passed = mutant_score >= 0.5 if mutant_trace else False

        # --- Step 4: Compare --------------------------------------------------
        killed = mutant_passed != original_passed
        score_delta = mutant_score - original_score

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        error_msg: str | None = None
        if original_error and mutant_error:
            error_msg = f"Both failed — original: {original_error}; mutant: {mutant_error}"
        elif original_error:
            error_msg = f"Original failed: {original_error}"
        elif mutant_error:
            error_msg = f"Mutant failed: {mutant_error}"

        return MutationResult(
            operator_name=operator.name,
            operator_category=operator.category,
            mutation_description=mutation_desc,
            original_passed=original_passed,
            mutant_passed=mutant_passed,
            killed=killed,
            original_score=original_score,
            mutant_score=mutant_score,
            score_delta=round(score_delta, 6),
            original_trace_id=original_trace.trace_id if original_trace else "",
            mutant_trace_id=mutant_trace.trace_id if mutant_trace else "",
            error=error_msg,
            duration_ms=round(elapsed_ms, 2),
        )

    # -- Full suite run ------------------------------------------------------

    def run_suite(
        self,
        scenario: TestScenario,
        operators: list[MutationOperator] | None = None,
        evaluator: Callable[[ExecutionTrace], float] | None = None,
    ) -> MutationSuiteResult:
        """Run all mutation operators and compute the mutation score.

        Parameters
        ----------
        scenario
            The test scenario to mutate.
        operators
            Override operators for this run. If ``None``, uses the
            operators configured at construction time.
        evaluator
            Optional scoring function. See ``run_mutation``.

        Returns
        -------
        MutationSuiteResult
            Aggregate result with mutation score, per-category
            breakdown, and individual ``MutationResult`` objects.
        """
        ops = operators if operators is not None else self._operators

        if not ops:
            logger.warning("No mutation operators provided — returning empty result.")
            return MutationSuiteResult(
                scenario_id=scenario.scenario_id,
                agent_id=self._config.agent_id,
            )

        suite_start = time.monotonic()
        results: list[MutationResult] = []

        logger.info(
            "Starting mutation suite: %d operators on scenario '%s'",
            len(ops),
            scenario.name,
        )

        for i, op in enumerate(ops, 1):
            logger.debug(
                "Running mutation %d/%d: %s (%s)",
                i, len(ops), op.name, op.category,
            )
            result = self.run_mutation(scenario, op, evaluator=evaluator)
            results.append(result)

            status = "KILLED" if result.killed else "SURVIVED"
            logger.debug(
                "  %s [%s] — original=%.2f, mutant=%.2f (delta=%.3f)",
                status, op.name, result.original_score,
                result.mutant_score, result.score_delta,
            )

        # --- Aggregate statistics ---------------------------------------------
        total = len(results)
        killed = sum(1 for r in results if r.killed and r.error is None)
        errored = sum(1 for r in results if r.error is not None)
        survived = total - killed - errored
        mutation_score = killed / total if total > 0 else 0.0

        # Per-category breakdown
        category_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "killed": 0}
        )
        for r in results:
            if r.error is None:
                category_counts[r.operator_category]["total"] += 1
                if r.killed:
                    category_counts[r.operator_category]["killed"] += 1

        per_category: dict[str, float] = {}
        for cat, counts in sorted(category_counts.items()):
            if counts["total"] > 0:
                per_category[cat] = round(
                    counts["killed"] / counts["total"], 4
                )
            else:
                per_category[cat] = 0.0

        suite_duration_ms = (time.monotonic() - suite_start) * 1000.0

        suite_result = MutationSuiteResult(
            results=results,
            total_mutants=total,
            killed_mutants=killed,
            survived_mutants=survived,
            errored_mutants=errored,
            mutation_score=round(mutation_score, 4),
            per_category=per_category,
            total_duration_ms=round(suite_duration_ms, 2),
            scenario_id=scenario.scenario_id,
            agent_id=self._config.agent_id,
        )

        logger.info(
            "Mutation suite complete: %d/%d killed (score=%.1f%%), "
            "%d errored, %.0fms total",
            killed, total, mutation_score * 100,
            errored, suite_duration_ms,
        )

        return suite_result

    # -- Internal: agent execution -------------------------------------------

    def _execute_agent(
        self,
        config: AgentConfig,
        input_data: dict[str, Any],
    ) -> tuple[ExecutionTrace | None, str | None]:
        """Execute the agent callable with error handling.

        Returns (trace, error_message). If execution succeeds,
        error_message is None. If it fails, trace may be None.
        """
        try:
            trace = self._agent_callable(config, input_data)
            return trace, None
        except Exception as exc:
            logger.warning(
                "Agent execution failed: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            return None, f"{type(exc).__name__}: {exc}"

    # -- Internal: evaluation ------------------------------------------------

    @staticmethod
    def _evaluate(
        trace: ExecutionTrace | None,
        evaluator: Callable[[ExecutionTrace], float] | None,
    ) -> float:
        """Score an execution trace.

        If ``evaluator`` is provided, calls it with the trace and
        returns its score (clamped to [0, 1]). Otherwise, uses
        ``trace.success`` as binary 1.0 / 0.0.

        Returns 0.0 if trace is None (agent failed to execute).
        """
        if trace is None:
            return 0.0

        if evaluator is not None:
            try:
                score = evaluator(trace)
                return max(0.0, min(1.0, float(score)))
            except Exception as exc:
                logger.warning("Evaluator raised: %s", exc)
                return 0.0

        return 1.0 if trace.success else 0.0
