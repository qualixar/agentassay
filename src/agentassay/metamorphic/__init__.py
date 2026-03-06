# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AgentAssay metamorphic testing -- agent-specific metamorphic relations.

Metamorphic testing (Chen et al., 2018) solves the oracle problem for
non-deterministic agents by defining *relations* between source and
follow-up test cases. If an input transformation produces a predictable
output transformation, the relation **holds**; otherwise, the agent's
behavior is inconsistent -- a potential regression.

Four families of relations are provided:

- **Permutation** (``InputPermutationRelation``, ``ToolOrderRelation``):
  invariance to input ordering.
- **Perturbation** (``TypographicalPerturbation``, ``IrrelevantAdditionRelation``):
  robustness to small input changes.
- **Composition** (``DecompositionRelation``): decomposed tasks produce
  consistent results.
- **Oracle** (``ConsistencyRelation``, ``MonotonicityRelation``): known
  invariants that must hold across executions.

Quick start::

    from agentassay.metamorphic import MetamorphicRunner, DEFAULT_RELATIONS

    runner = MetamorphicRunner(my_agent, agent_config)
    result = runner.test_all(scenario)
    print(f"Violations: {result.violations}/{result.total_relations}")
"""

from agentassay.metamorphic.relations import (
    # Base class
    MetamorphicRelation,
    # Result model
    MetamorphicResult,
    # Family 1: Permutation
    InputPermutationRelation,
    ToolOrderRelation,
    # Family 2: Perturbation
    TypographicalPerturbation,
    IrrelevantAdditionRelation,
    # Family 3: Composition
    DecompositionRelation,
    # Family 4: Oracle
    ConsistencyRelation,
    MonotonicityRelation,
)
from agentassay.metamorphic.runner import (
    DEFAULT_RELATIONS,
    MetamorphicRunner,
    MetamorphicTestResult,
)

__all__ = [
    # Base
    "MetamorphicRelation",
    "MetamorphicResult",
    # Family 1: Permutation
    "InputPermutationRelation",
    "ToolOrderRelation",
    # Family 2: Perturbation
    "TypographicalPerturbation",
    "IrrelevantAdditionRelation",
    # Family 3: Composition
    "DecompositionRelation",
    # Family 4: Oracle
    "ConsistencyRelation",
    "MonotonicityRelation",
    # Runner
    "MetamorphicRunner",
    "MetamorphicTestResult",
    # Defaults
    "DEFAULT_RELATIONS",
]
