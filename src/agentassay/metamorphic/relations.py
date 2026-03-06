# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Re-export shim for metamorphic relations (backward compatibility).

All relation classes and helpers were split into focused submodules:

- ``base.py`` -- MetamorphicRelation base class, MetamorphicResult, helpers
- ``permutation.py`` -- InputPermutationRelation, ToolOrderRelation
- ``perturbation.py`` -- TypographicalPerturbation, IrrelevantAdditionRelation
- ``composition.py`` -- DecompositionRelation
- ``oracle.py`` -- ConsistencyRelation, MonotonicityRelation

This shim re-exports every public symbol so that existing imports like
``from agentassay.metamorphic.relations import X`` continue to work
unchanged.
"""

# Base class, result model, and helpers
from agentassay.metamorphic.base import (  # noqa: F401
    MetamorphicRelation,
    MetamorphicResult,
    _deep_copy_scenario,
    _exact_match,
    _stringify,
    _text_similarity,
)

# Family 1: Permutation
from agentassay.metamorphic.permutation import (  # noqa: F401
    InputPermutationRelation,
    ToolOrderRelation,
)

# Family 2: Perturbation
from agentassay.metamorphic.perturbation import (  # noqa: F401
    IrrelevantAdditionRelation,
    TypographicalPerturbation,
)

# Family 3: Composition
from agentassay.metamorphic.composition import (  # noqa: F401
    DecompositionRelation,
)

# Family 4: Oracle
from agentassay.metamorphic.oracle import (  # noqa: F401
    ConsistencyRelation,
    MonotonicityRelation,
)

__all__ = [
    # Base
    "MetamorphicRelation",
    "MetamorphicResult",
    # Helpers (internal but imported by tests)
    "_deep_copy_scenario",
    "_exact_match",
    "_stringify",
    "_text_similarity",
    # Permutation
    "InputPermutationRelation",
    "ToolOrderRelation",
    # Perturbation
    "TypographicalPerturbation",
    "IrrelevantAdditionRelation",
    # Composition
    "DecompositionRelation",
    # Oracle
    "ConsistencyRelation",
    "MonotonicityRelation",
]
