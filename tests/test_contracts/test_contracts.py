"""Tests for contracts module (ContractOracle and ContractLoader).

Tests contract loading, evaluation, constraint types, scoring logic,
and the as_evaluator interface.

Target: ~15 tests.
"""

from __future__ import annotations

import pytest

from agentassay.contracts.loader import ContractLoadError, ContractLoader
from agentassay.contracts.oracle import (
    ContractEvaluation,
    ContractOracle,
    ContractViolation,
)

from tests.conftest import make_contract_dict, make_trace


# ===================================================================
# ContractLoader
# ===================================================================


class TestContractLoader:
    """Tests for ContractLoader."""

    def test_load_valid_dict(self):
        data = make_contract_dict()
        result = ContractLoader.load_dict(data)
        assert result["contract"]["name"] == "test_contract"
        assert len(result["contract"]["constraints"]) == 3

    def test_missing_contract_key_raises(self):
        with pytest.raises(ContractLoadError, match="contract"):
            ContractLoader.load_dict({"not_contract": {}})

    def test_missing_name_raises(self):
        with pytest.raises(ContractLoadError, match="name"):
            ContractLoader.load_dict({"contract": {"constraints": []}})

    def test_invalid_constraint_type_raises(self):
        data = {
            "contract": {
                "name": "test",
                "constraints": [
                    {
                        "name": "c1",
                        "type": "invalid_type",
                        "severity": "hard",
                        "condition": "success",
                    }
                ],
            }
        }
        with pytest.raises(ContractLoadError, match="invalid type"):
            ContractLoader.load_dict(data)

    def test_invalid_severity_raises(self):
        data = {
            "contract": {
                "name": "test",
                "constraints": [
                    {
                        "name": "c1",
                        "type": "invariant",
                        "severity": "critical",  # invalid
                        "condition": "success",
                    }
                ],
            }
        }
        with pytest.raises(ContractLoadError, match="severity"):
            ContractLoader.load_dict(data)

    def test_duplicate_constraint_name_raises(self):
        data = {
            "contract": {
                "name": "test",
                "constraints": [
                    {"name": "same", "type": "invariant", "severity": "hard", "condition": "success"},
                    {"name": "same", "type": "guardrail", "severity": "soft", "condition": "step_count <= 10"},
                ],
            }
        }
        with pytest.raises(ContractLoadError, match="duplicate"):
            ContractLoader.load_dict(data)

    def test_empty_constraints_valid(self):
        data = {"contract": {"name": "test", "constraints": []}}
        result = ContractLoader.load_dict(data)
        assert len(result["contract"]["constraints"]) == 0


# ===================================================================
# ContractOracle
# ===================================================================


class TestContractOracle:
    """Tests for ContractOracle."""

    def test_create_from_dict(self):
        oracle = ContractOracle(contract_dict=make_contract_dict())
        assert oracle.contract_name == "test_contract"
        assert oracle.num_constraints == 3

    def test_both_path_and_dict_raises(self):
        with pytest.raises(ValueError, match="not both"):
            ContractOracle(
                contract_path="fake.yaml",
                contract_dict=make_contract_dict(),
            )

    def test_neither_path_nor_dict_raises(self):
        with pytest.raises(ValueError, match="either"):
            ContractOracle()

    def test_evaluate_passing_trace(self):
        oracle = ContractOracle(contract_dict=make_contract_dict())
        trace = make_trace(steps=3, success=True, cost=0.01)
        evaluation = oracle.evaluate(trace)
        assert isinstance(evaluation, ContractEvaluation)
        assert evaluation.passed is True
        assert evaluation.score == 1.0

    def test_evaluate_step_count_violation(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "max_steps",
                    "type": "guardrail",
                    "severity": "hard",
                    "condition": "step_count <= 1",
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(steps=5, success=True)
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is False
        assert len(evaluation.violations) == 1
        assert evaluation.violations[0].severity == "hard"

    def test_evaluate_cost_violation(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "cost_limit",
                    "type": "guardrail",
                    "severity": "hard",
                    "condition": "total_cost_usd <= 0.001",
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(steps=3, cost=0.5)
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is False

    def test_soft_violation_still_passes(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "preferred_steps",
                    "type": "guardrail",
                    "severity": "soft",
                    "condition": "step_count <= 1",
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(steps=3, success=True)
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is True  # soft violations don't cause failure
        assert len(evaluation.violations) == 1
        assert evaluation.score < 1.0

    def test_uses_tool_condition(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "must_search",
                    "type": "postcondition",
                    "severity": "hard",
                    "condition": "uses_tool('search')",
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(steps=2, tools=["search", "calculate"])
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is True

    def test_not_uses_tool_condition(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "no_delete",
                    "type": "invariant",
                    "severity": "hard",
                    "condition": "not uses_tool('delete_database')",
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(steps=2, tools=["search"])
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is True

    def test_output_contains_condition(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "has_answer",
                    "type": "postcondition",
                    "severity": "hard",
                    "condition": "output_contains('test output')",
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(output_data="test output result")
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is True

    def test_as_evaluator_returns_callable(self):
        oracle = ContractOracle(contract_dict=make_contract_dict())
        evaluator = oracle.as_evaluator()
        trace = make_trace(steps=3, success=True, cost=0.01)
        score = evaluator(trace)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_batch(self):
        oracle = ContractOracle(contract_dict=make_contract_dict())
        traces = [make_trace(steps=2, success=True) for _ in range(5)]
        results = oracle.evaluate_batch(traces)
        assert len(results) == 5
        assert all(isinstance(r, ContractEvaluation) for r in results)

    def test_score_computation_no_constraints(self):
        contract = make_contract_dict(constraints=[])
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(steps=3)
        evaluation = oracle.evaluate(trace)
        assert evaluation.score == 1.0

    def test_precondition_evaluation(self):
        contract = make_contract_dict(
            constraints=[
                {
                    "name": "has_input",
                    "type": "precondition",
                    "severity": "hard",
                    "condition": "success",  # checks the success field
                },
            ]
        )
        oracle = ContractOracle(contract_dict=contract)
        trace = make_trace(success=True)
        evaluation = oracle.evaluate(trace)
        assert evaluation.passed is True
