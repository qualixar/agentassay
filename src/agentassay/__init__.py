# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AgentAssay: Formal regression testing for non-deterministic AI agent workflows.

Core Classes
------------
TrialRunner
    Execute stochastic trials and collect execution traces.
ExecutionTrace
    Captures multi-step agent execution with costs and timings.
TestScenario
    Defines a test scenario with input, oracle, and metamorphic relations.

Verdicts & Gates
----------------
VerdictFunction
    Compute three-valued verdicts (PASS/FAIL/INCONCLUSIVE).
DeploymentGate
    Gate deployment based on confidence intervals and thresholds.

Attribution
-----------
QualixarSigner
    Cryptographic signing for output provenance (Layer 2).
QualixarWatermark
    Steganographic watermarking for IP protection (Layer 3).
"""

__version__ = "0.1.0"

# Core classes
from agentassay.core import ExecutionTrace, TestScenario, TrialRunner

# Verdicts and gates
from agentassay.verdicts import DeploymentGate, VerdictFunction

# Attribution (Qualixar 3-Layer System)
from agentassay.attribution import QualixarSigner, QualixarWatermark

__all__ = [
    # Core
    "TrialRunner",
    "ExecutionTrace",
    "TestScenario",
    # Verdicts
    "VerdictFunction",
    "DeploymentGate",
    # Attribution
    "QualixarSigner",
    "QualixarWatermark",
]
