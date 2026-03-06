# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Token-Efficient Testing — the core innovation of AgentAssay.

This module solves the fundamental economic problem of AI agent testing:
each test trial costs real money (LLM API calls, tool invocations, compute).
Running 100 trials at $0.02 each = $2.00 per scenario. With 50 scenarios
per CI run, that is $100 per commit — untenable for any team.

The insight: agent outputs are stochastic (different text every time), but
agent BEHAVIORS are structured. A well-built search agent will always call
search, filter results, then synthesize — even though the specific search
queries, filter criteria, and synthesized text vary across runs.

This behavioral regularity lives on a low-dimensional manifold inside the
high-dimensional output space. By testing on that manifold instead of raw
outputs, we achieve the same statistical guarantees with exponentially fewer
samples. This module provides five complementary mechanisms:

1. **Behavioral Fingerprinting** (``fingerprint``)
   Extract fixed-size behavioral vectors from execution traces and detect
   regressions via multivariate Hotelling's T-squared test — more powerful
   than univariate pass-rate testing.

2. **Adaptive Budget Optimization** (``budget``)
   Compute the minimum number of trials needed for (alpha, beta) guarantees
   using a small calibration set. Stable agents need 15-25 trials instead
   of the default 100.

3. **Trace-First Testing** (``trace_store``)
   Record production traces and analyze them offline at zero additional token
   cost. Coverage, contracts, and metamorphic analysis run on stored data.

4. **Multi-Fidelity Proxy Testing** (``multi_fidelity``)
   Test expensive models using cheap proxy models. Combine evidence from
   both fidelity levels via meta-analytic methods for a single verdict.

5. **Warm-Start SPRT** (``warm_start``)
   Sequential testing with Bayesian prior from previous runs. Yesterday's
   100-trial result becomes today's informative prior, dramatically reducing
   the number of new trials needed.

Theoretical contribution (paper Section 7):
    These five mechanisms are proven to reduce total testing cost by 3-12x
    while maintaining identical (alpha, beta) statistical guarantees. The
    savings compound: fingerprinting reduces dimensionality, budget
    optimization reduces trial count, trace-first eliminates redundant
    API calls, multi-fidelity reduces per-trial cost, and warm-start
    amortizes information across CI runs.

References:
    - Hotelling, H. (1931). "The Generalization of Student's Ratio."
      Annals of Mathematical Statistics 2(3): 360-378.
    - Wald, A. (1947). Sequential Analysis. John Wiley & Sons.
    - Fisher, R.A. (1925). Statistical Methods for Research Workers.
    - Pham-Gia, T. & Hung, T.L. (2001). "The Mean and Median Absolute
      Deviations." Mathematical and Computer Modelling 34(7-8): 921-936.
"""

# ---------------------------------------------------------------------------
# Behavioral Fingerprinting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Adaptive Budget Optimization
# ---------------------------------------------------------------------------
from agentassay.efficiency.budget import (
    AdaptiveBudgetOptimizer,
    BudgetEstimate,
)
from agentassay.efficiency.distribution import (
    FingerprintDistribution,
)
from agentassay.efficiency.fingerprint import (
    BehavioralFingerprint,
)

# ---------------------------------------------------------------------------
# Multi-Fidelity Proxy Testing
# ---------------------------------------------------------------------------
from agentassay.efficiency.multi_fidelity import (
    MultiFidelityTester,
)
from agentassay.efficiency.regression import (
    fingerprint_regression_test,
)

# ---------------------------------------------------------------------------
# Trace-First Testing (persistent trace store)
# ---------------------------------------------------------------------------
from agentassay.efficiency.trace_store import (
    TraceStore,
)

# ---------------------------------------------------------------------------
# Warm-Start SPRT
# ---------------------------------------------------------------------------
from agentassay.efficiency.warm_start import (
    WarmStartSPRT,
)

__all__ = [
    # Fingerprinting
    "BehavioralFingerprint",
    "FingerprintDistribution",
    "fingerprint_regression_test",
    # Budget
    "AdaptiveBudgetOptimizer",
    "BudgetEstimate",
    # Trace store
    "TraceStore",
    # Multi-fidelity
    "MultiFidelityTester",
    # Warm-start
    "WarmStartSPRT",
]
