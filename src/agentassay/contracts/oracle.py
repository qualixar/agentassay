"""Contract oracle for AgentAssay.

Evaluates agent ``ExecutionTrace`` objects against behavioral contracts
and produces ``ContractEvaluation`` results with per-constraint details.

AgentAssert is OPTIONAL. The oracle parses ContractSpec YAML independently
using a safe condition evaluator (no dynamic code). ``ContractViolation``
and ``ContractEvaluation`` are re-exported here for backward compatibility;
canonical definitions live in ``agentassay.contracts.evaluation``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentassay.contracts.evaluation import (  # noqa: F401 -- re-export
    ContractEvaluation,
    ContractViolation,
)
from agentassay.contracts.loader import ContractLoader, ContractLoadError  # noqa: F401
from agentassay.contracts.parser import build_trace_context, evaluate_condition
from agentassay.core.models import ExecutionTrace

logger = logging.getLogger(__name__)

_HAS_AGENTASSERT = False
try:
    import agentassert  # noqa: F401
    _HAS_AGENTASSERT = True
except ImportError:
    pass

# Re-export parser internals for backward compatibility (private API)
from agentassay.contracts.parser import (  # noqa: F401, E402
    _BARE_IDENT_RE,
    _call_builtin,
    _COMPARISON_OPS,
    _COMPARISON_RE,
    _evaluate_comparison,
    _FUNC_CALL_RE,
    _NOT_BARE_RE,
    _NOT_FUNC_RE,
    _resolve_value,
    build_trace_context as _build_trace_context,
    evaluate_condition as _evaluate_condition,
)


# ===================================================================
# ContractOracle
# ===================================================================


class ContractOracle:
    """Evaluates agent execution traces against behavioral contracts.

    Initialize with a ContractSpec YAML path or a pre-loaded dict.
    AgentAssert is NOT required -- uses built-in safe evaluation.

    Parameters
    ----------
    contract_path
        Path to a ContractSpec YAML file (mutually exclusive with ``contract_dict``).
    contract_dict
        Pre-loaded contract dict with a ``contract`` key.
    """

    __slots__ = ("_contract_data", "_contract_name", "_constraints")

    def __init__(
        self,
        contract_path: str | Path | None = None,
        contract_dict: dict[str, Any] | None = None,
    ) -> None:
        if contract_path is not None and contract_dict is not None:
            raise ValueError(
                "Provide either contract_path or contract_dict, not both"
            )
        if contract_path is None and contract_dict is None:
            raise ValueError(
                "Provide either contract_path or contract_dict"
            )

        if contract_path is not None:
            self._contract_data = ContractLoader.load_yaml(contract_path)
        else:
            assert contract_dict is not None  # for type narrowing
            self._contract_data = ContractLoader.load_dict(contract_dict)

        contract = self._contract_data["contract"]
        self._contract_name: str = contract["name"]
        self._constraints: list[dict[str, Any]] = contract["constraints"]

        logger.info(
            "ContractOracle initialized: contract='%s', constraints=%d",
            self._contract_name,
            len(self._constraints),
        )

    # -- Public properties ---------------------------------------------------

    @property
    def contract_name(self) -> str:
        """Name of the loaded contract."""
        return self._contract_name

    @property
    def num_constraints(self) -> int:
        """Number of constraints in the contract."""
        return len(self._constraints)

    # -- Evaluation ----------------------------------------------------------

    def evaluate(self, trace: ExecutionTrace) -> ContractEvaluation:
        """Evaluate an execution trace against all contract constraints.

        Constraint evaluation order:
            1. **Preconditions** -- checked against ``trace.input_data``
               and trace state before the first step.
            2. **Invariants** -- checked at every step in the trace.
            3. **Guardrails** -- checked against trace-level aggregates
               (cost, duration, step count).
            4. **Postconditions** -- checked against the final state
               (output, tools used, success status).

        Parameters
        ----------
        trace
            A complete execution trace from one agent invocation.

        Returns
        -------
        ContractEvaluation
            Evaluation result with violations, score, and pass/fail verdict.
        """
        context = build_trace_context(trace)
        violations: list[ContractViolation] = []

        for constraint in self._constraints:
            c_name: str = constraint["name"]
            c_type: str = constraint["type"]
            c_severity: str = constraint["severity"]
            c_condition: str = constraint["condition"]

            if c_type == "precondition":
                self._check_precondition(
                    trace, context, c_name, c_condition, c_severity, violations
                )
            elif c_type == "postcondition":
                self._check_postcondition(
                    trace, context, c_name, c_condition, c_severity, violations
                )
            elif c_type == "invariant":
                self._check_invariant(
                    trace, context, c_name, c_condition, c_severity, violations
                )
            elif c_type == "guardrail":
                self._check_guardrail(
                    trace, context, c_name, c_condition, c_severity, violations
                )

        # --- Compute score ---
        score = self._compute_score(violations)

        # --- Pass/fail: hard violations cause failure ---
        has_hard_violations = any(
            v.severity == "hard" for v in violations
        )

        return ContractEvaluation(
            contract_name=self._contract_name,
            passed=not has_hard_violations,
            violations=violations,
            score=score,
            trace_id=trace.trace_id,
        )

    def evaluate_batch(
        self,
        traces: list[ExecutionTrace],
    ) -> list[ContractEvaluation]:
        """Evaluate multiple execution traces against the contract.

        Parameters
        ----------
        traces
            List of execution traces to evaluate.

        Returns
        -------
        list[ContractEvaluation]
            One evaluation per trace, in the same order.
        """
        return [self.evaluate(trace) for trace in traces]

    def as_evaluator(self) -> Callable[[ExecutionTrace], float]:
        """Return a function compatible with TrialRunner's evaluator interface.

        The returned callable takes an ``ExecutionTrace`` and returns a
        float score in [0.0, 1.0] from the ``ContractEvaluation``.

        This bridges the contracts module with the trial runner: you can
        use a behavioral contract as the pass/fail oracle for stochastic
        regression testing.

        Returns
        -------
        Callable[[ExecutionTrace], float]
            A function ``f(trace) -> score``.

        Example
        -------
        >>> oracle = ContractOracle(contract_path="contract.yaml")
        >>> evaluator = oracle.as_evaluator()
        >>> score = evaluator(trace)  # 0.0 to 1.0
        """

        def _evaluator(trace: ExecutionTrace) -> float:
            evaluation = self.evaluate(trace)
            return evaluation.score

        return _evaluator

    # -- Internal: constraint type handlers ----------------------------------

    def _check_precondition(
        self,
        trace: ExecutionTrace,
        context: dict[str, Any],
        name: str,
        condition: str,
        severity: str,
        violations: list[ContractViolation],
    ) -> None:
        """Check a precondition against the trace's initial state."""
        satisfied, actual = evaluate_condition(condition, context)
        if not satisfied:
            violations.append(
                ContractViolation(
                    contract_name=self._contract_name,
                    constraint_name=name,
                    constraint_type="precondition",
                    violated_at_step=None,
                    expected=f"precondition: {condition}",
                    actual=actual,
                    severity=severity,  # type: ignore[arg-type]
                )
            )

    def _check_postcondition(
        self,
        trace: ExecutionTrace,
        context: dict[str, Any],
        name: str,
        condition: str,
        severity: str,
        violations: list[ContractViolation],
    ) -> None:
        """Check a postcondition against the trace's final state."""
        satisfied, actual = evaluate_condition(condition, context)
        if not satisfied:
            last_step = trace.step_count - 1 if trace.step_count > 0 else None
            violations.append(
                ContractViolation(
                    contract_name=self._contract_name,
                    constraint_name=name,
                    constraint_type="postcondition",
                    violated_at_step=last_step,
                    expected=f"postcondition: {condition}",
                    actual=actual,
                    severity=severity,  # type: ignore[arg-type]
                )
            )

    def _check_invariant(
        self,
        trace: ExecutionTrace,
        context: dict[str, Any],
        name: str,
        condition: str,
        severity: str,
        violations: list[ContractViolation],
    ) -> None:
        """Check an invariant that must hold across the execution."""
        satisfied, actual = evaluate_condition(condition, context)
        if not satisfied:
            last_step = trace.step_count - 1 if trace.step_count > 0 else None
            violations.append(
                ContractViolation(
                    contract_name=self._contract_name,
                    constraint_name=name,
                    constraint_type="invariant",
                    violated_at_step=last_step,
                    expected=f"invariant: {condition}",
                    actual=actual,
                    severity=severity,  # type: ignore[arg-type]
                )
            )

    def _check_guardrail(
        self,
        trace: ExecutionTrace,
        context: dict[str, Any],
        name: str,
        condition: str,
        severity: str,
        violations: list[ContractViolation],
    ) -> None:
        """Check a guardrail against trace-level aggregates."""
        satisfied, actual = evaluate_condition(condition, context)
        if not satisfied:
            violations.append(
                ContractViolation(
                    contract_name=self._contract_name,
                    constraint_name=name,
                    constraint_type="guardrail",
                    violated_at_step=None,
                    expected=f"guardrail: {condition}",
                    actual=actual,
                    severity=severity,  # type: ignore[arg-type]
                )
            )

    # -- Internal: scoring ---------------------------------------------------

    def _compute_score(
        self,
        violations: list[ContractViolation],
    ) -> float:
        """Compute a normalized compliance score from violations.

        Scoring logic:
            - Start at 1.0 (perfect compliance).
            - Each hard violation reduces by ``1.0 / total_constraints``.
            - Each soft violation reduces by ``0.5 / total_constraints``.
            - Score is clamped to [0.0, 1.0].

        If there are no constraints, score is 1.0 (vacuously true).
        """
        if not self._constraints:
            return 1.0

        if not violations:
            return 1.0

        total = len(self._constraints)
        penalty = 0.0

        for v in violations:
            if v.severity == "hard":
                penalty += 1.0 / total
            else:
                penalty += 0.5 / total

        return max(0.0, min(1.0, 1.0 - penalty))

    # -- Representation ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ContractOracle(contract='{self._contract_name}', "
            f"constraints={len(self._constraints)})"
        )
