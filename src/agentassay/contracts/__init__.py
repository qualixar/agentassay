# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AgentAssay contracts -- behavioral contract integration for test oracles.

Adapts AgentAssert (ABC) behavioral contracts as test oracles for
stochastic regression testing. AgentAssert is an OPTIONAL dependency:
this module parses ContractSpec YAML independently and evaluates
constraints using a built-in safe expression parser.

Public API:
    - ``ContractOracle``     -- evaluates traces against contracts
    - ``ContractEvaluation`` -- evaluation result with violations and score
    - ``ContractViolation``  -- a single constraint violation
    - ``ContractLoader``     -- loads and validates contract YAML/dicts
"""

from agentassay.contracts.evaluation import (
    ContractEvaluation,
    ContractViolation,
)
from agentassay.contracts.loader import ContractLoader, ContractLoadError
from agentassay.contracts.oracle import ContractOracle

__all__ = [
    "ContractOracle",
    "ContractEvaluation",
    "ContractViolation",
    "ContractLoader",
    "ContractLoadError",
]
