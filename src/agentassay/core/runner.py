"""Trial runner for AgentAssay.

Executes agent callables against test scenarios, collecting
``TrialResult`` objects with full execution traces. Each trial is
independent — no shared mutable state between invocations.

Pattern note: The runner follows the **Strategy pattern** — the
agent-under-test is injected as a callable, making the runner
framework-agnostic. Any function that accepts ``dict[str, Any]``
and returns an ``ExecutionTrace`` can be tested.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from agentassay.core.models import (
    AgentConfig,
    AssayConfig,
    ExecutionTrace,
    TestScenario,
    TrialResult,
)

logger = logging.getLogger(__name__)


class TrialTimeoutError(Exception):
    """Raised when a single trial exceeds its timeout budget."""


class CostBudgetExceededError(Exception):
    """Raised when cumulative trial cost exceeds ``max_cost_usd``."""


class TrialRunner:
    """Executes stochastic trials of an agent against test scenarios.

    Parameters
    ----------
    agent_callable
        A function that takes ``dict[str, Any]`` (the scenario input)
        and returns an ``ExecutionTrace``. This is the agent-under-test.
    config
        Statistical and resource configuration for the assay.
    agent_config
        Metadata describing the agent (framework, model, version).

    Example
    -------
    >>> def my_agent(input_data: dict) -> ExecutionTrace:
    ...     # invoke your LangGraph / CrewAI / custom agent here
    ...     ...
    >>> runner = TrialRunner(my_agent, AssayConfig(), AgentConfig(...))
    >>> results = runner.run_trials(scenario)
    """

    def __init__(
        self,
        agent_callable: Callable[[dict[str, Any]], ExecutionTrace],
        config: AssayConfig,
        agent_config: AgentConfig,
    ) -> None:
        self._agent_callable = agent_callable
        self._config = config
        self._agent_config = agent_config
        self._cumulative_cost_usd: float = 0.0
        self._cost_lock = __import__("threading").Lock()

    # -- Public properties ---------------------------------------------------

    @property
    def config(self) -> AssayConfig:
        return self._config

    @property
    def agent_config(self) -> AgentConfig:
        return self._agent_config

    @property
    def cumulative_cost_usd(self) -> float:
        return self._cumulative_cost_usd

    # -- Single trial --------------------------------------------------------

    def run_trial(self, scenario: TestScenario) -> TrialResult:
        """Run a single trial of the agent on the given scenario.

        The trial is fully self-contained: it catches all exceptions,
        records timing, and returns a ``TrialResult`` regardless of
        whether the agent succeeded or failed.

        Raises
        ------
        CostBudgetExceededError
            If cumulative cost across all trials exceeds
            ``config.max_cost_usd``.
        """
        trial_id = str(uuid.uuid4())
        trial_start = time.monotonic()

        # Pre-flight: check cost budget (thread-safe)
        with self._cost_lock:
            if self._cumulative_cost_usd >= self._config.max_cost_usd:
                raise CostBudgetExceededError(
                    f"Cumulative cost ${self._cumulative_cost_usd:.4f} "
                    f"exceeds budget ${self._config.max_cost_usd:.2f}"
                )

        trace: ExecutionTrace | None = None
        passed = False
        score = 0.0
        evaluation_details: dict[str, Any] = {}
        error_msg: str | None = None

        try:
            trace = self._execute_with_timeout(
                scenario=scenario,
                timeout_seconds=min(
                    scenario.timeout_seconds,
                    self._config.timeout_seconds,
                ),
            )

            # Accumulate cost (thread-safe for parallel_trials > 1)
            with self._cost_lock:
                self._cumulative_cost_usd += trace.total_cost_usd

            # Evaluate the trace against expected properties
            passed, score, evaluation_details = self._evaluate(
                trace=trace,
                scenario=scenario,
            )

        except TrialTimeoutError:
            error_msg = (
                f"Trial timed out after "
                f"{min(scenario.timeout_seconds, self._config.timeout_seconds):.1f}s"
            )
            logger.warning("Trial %s: %s", trial_id, error_msg)

        except CostBudgetExceededError:
            raise  # Re-raise cost errors — caller must handle

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Trial %s failed: %s\n%s",
                trial_id,
                error_msg,
                traceback.format_exc(),
            )

        # Build a fallback trace if the agent callable raised
        if trace is None:
            elapsed_ms = (time.monotonic() - trial_start) * 1000.0
            trace = ExecutionTrace(
                trace_id=str(uuid.uuid4()),
                scenario_id=scenario.scenario_id,
                steps=[],
                input_data=scenario.input_data,
                output_data=None,
                success=False,
                error=error_msg,
                total_duration_ms=elapsed_ms,
                total_cost_usd=0.0,
                model=self._agent_config.model,
                framework=self._agent_config.framework,
            )

        return TrialResult(
            trial_id=trial_id,
            scenario_id=scenario.scenario_id,
            trace=trace,
            passed=passed,
            score=score,
            evaluation_details=evaluation_details,
            timestamp=datetime.now(timezone.utc),
        )

    # -- Batch trials --------------------------------------------------------

    def run_trials(
        self,
        scenario: TestScenario,
        n: int | None = None,
    ) -> list[TrialResult]:
        """Run N independent trials of the agent on the given scenario.

        Parameters
        ----------
        scenario
            The test scenario to execute.
        n
            Number of trials. Defaults to ``config.num_trials``.

        Returns
        -------
        list[TrialResult]
            One result per trial, in completion order when running in
            parallel or in sequential order otherwise.

        Raises
        ------
        CostBudgetExceededError
            Propagated if the cumulative cost budget is exhausted
            mid-run. Results collected so far are NOT returned — the
            caller should catch this and inspect ``cumulative_cost_usd``.
        """
        num = n if n is not None else self._config.num_trials
        logger.info(
            "Starting %d trials for scenario '%s' (agent=%s, model=%s)",
            num,
            scenario.name,
            self._agent_config.name,
            self._agent_config.model,
        )

        if self._config.parallel_trials <= 1:
            return self._run_sequential(scenario, num)
        return self._run_parallel(scenario, num)

    # -- Internal: execution -------------------------------------------------

    # Timeout warning flag -- shared across all TrialRunner instances.
    # We only log the CPython limitation warning once per process to
    # avoid flooding the logs in long-running assays.
    _warned_timeout: bool = False

    def _execute_with_timeout(
        self,
        scenario: TestScenario,
        timeout_seconds: float,
    ) -> ExecutionTrace:
        """Invoke the agent callable with a wall-clock timeout.

        Uses a thread-pool with a single worker so we can enforce a
        hard timeout on the callable. The agent callable MUST be
        thread-safe (which is trivially true when each trial is
        independent, since we only run one at a time per worker).

        .. note:: **Known CPython limitation**

           ``future.cancel()`` only prevents a *pending* future from
           starting -- it cannot interrupt an already-running thread in
           CPython because CPython has no mechanism to asynchronously
           raise an exception inside an arbitrary thread.  If the agent
           callable is blocked (e.g. waiting on an HTTP response), the
           thread will continue running in the background after the
           ``TrialTimeoutError`` is raised.  For hard-kill semantics,
           use ``ProcessPoolExecutor`` instead (at the cost of IPC
           overhead and serialization constraints).
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._agent_callable,
                scenario.input_data,
            )
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError as exc:
                future.cancel()
                if not TrialRunner._warned_timeout:
                    TrialRunner._warned_timeout = True
                    logger.warning(
                        "Warning: agent callable may continue running "
                        "after timeout due to CPython thread limitations. "
                        "Use ProcessPoolExecutor for hard kill semantics."
                    )
                raise TrialTimeoutError(
                    f"Agent did not return within {timeout_seconds:.1f}s"
                ) from exc

    # -- Internal: evaluation ------------------------------------------------

    def _evaluate(
        self,
        trace: ExecutionTrace,
        scenario: TestScenario,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Evaluate an execution trace against a scenario's expected properties.

        Returns (passed, score, details).

        The built-in evaluator checks declarative properties from
        ``scenario.expected_properties``. If ``scenario.evaluator`` is
        set, it is expected to be resolved externally (plugin system);
        for now we log a warning and fall back to the built-in logic.
        """
        if scenario.evaluator is not None:
            logger.debug(
                "Custom evaluator '%s' requested but plugin resolution "
                "is not yet implemented — using built-in evaluation.",
                scenario.evaluator,
            )

        # Baseline: if the agent reported success, start from there
        if not trace.success:
            return False, 0.0, {"reason": trace.error or "agent reported failure"}

        checks: dict[str, bool] = {}
        props = scenario.expected_properties

        # --- max_steps ---
        if "max_steps" in props:
            limit = int(props["max_steps"])
            ok = trace.step_count <= limit
            checks["max_steps"] = ok

        # --- must_use_tools ---
        if "must_use_tools" in props:
            required: set[str] = set(props["must_use_tools"])
            ok = required.issubset(trace.tools_used)
            checks["must_use_tools"] = ok

        # --- must_not_use_tools ---
        if "must_not_use_tools" in props:
            forbidden: set[str] = set(props["must_not_use_tools"])
            ok = forbidden.isdisjoint(trace.tools_used)
            checks["must_not_use_tools"] = ok

        # --- output_contains ---
        if "output_contains" in props:
            needle = str(props["output_contains"])
            haystack = str(trace.output_data) if trace.output_data is not None else ""
            ok = needle.lower() in haystack.lower()
            checks["output_contains"] = ok

        # --- max_cost_usd ---
        if "max_cost_usd" in props:
            limit_cost = float(props["max_cost_usd"])
            ok = trace.total_cost_usd <= limit_cost
            checks["max_cost_usd"] = ok

        # --- max_duration_ms ---
        if "max_duration_ms" in props:
            limit_dur = float(props["max_duration_ms"])
            ok = trace.total_duration_ms <= limit_dur
            checks["max_duration_ms"] = ok

        # Aggregate verdict
        if not checks:
            # No declarative properties — pass if agent succeeded
            return True, 1.0, {"reason": "no expected_properties; agent succeeded"}

        all_passed = all(checks.values())
        score = sum(checks.values()) / len(checks) if checks else 0.0

        return all_passed, score, {"checks": checks}

    # -- Internal: sequential / parallel execution ---------------------------

    def _run_sequential(
        self,
        scenario: TestScenario,
        n: int,
    ) -> list[TrialResult]:
        results: list[TrialResult] = []
        for i in range(n):
            logger.debug("Running trial %d/%d", i + 1, n)
            result = self.run_trial(scenario)
            results.append(result)
        return results

    def _run_parallel(
        self,
        scenario: TestScenario,
        n: int,
    ) -> list[TrialResult]:
        max_workers = min(self._config.parallel_trials, n)
        results: list[TrialResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.run_trial, scenario): i
                for i in range(n)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except CostBudgetExceededError:
                    logger.warning(
                        "Cost budget exhausted at trial %d/%d — "
                        "stopping remaining trials.",
                        len(results),
                        n,
                    )
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    raise
                except Exception as exc:
                    logger.error("Trial %d raised unexpected error: %s", idx, exc)
                    raise

        return results
