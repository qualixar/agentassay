# Architecture Overview

This page describes the high-level architecture of AgentAssay. It covers the layer structure, data flow, and design principles without exposing implementation details.

## Design Principles

1. **Statistics, not luck.** Every verdict is backed by formal hypothesis tests and confidence intervals. No single-run verdicts.

2. **Three-valued logic.** PASS / FAIL / INCONCLUSIVE. The system never forces a binary answer when evidence is insufficient.

3. **Token efficiency by default.** Every testing operation is optimized to extract maximum statistical signal from minimum token expenditure. Offline-first, adaptive-second, full-run-last.

4. **Framework-agnostic.** Agent frameworks come and go. AgentAssay defines an abstract execution trace format and adapts to any framework through adapters.

5. **Immutable traces.** Execution traces and results are immutable after creation. This guarantees reproducibility and prevents accidental modification during analysis.

6. **Safe evaluation.** Contract conditions are evaluated through safe pattern matching. No arbitrary code evaluation.

7. **Lazy dependencies.** Framework-specific adapters only import their framework when used. Installing AgentAssay does not require installing every agent framework.

## Layer Architecture

AgentAssay is organized into six layers. Each layer depends only on the layers below it.

```
+-------------------------------------------------------------------+
|  Layer 6: Efficiency                                               |
|                                                                     |
|  Behavioral Fingerprinting   Adaptive Budget Optimization           |
|  Trace-First Offline Analysis                                       |
|  Multi-Fidelity Proxy Testing   Warm-Start Sequential Testing       |
+-------------------------------------------------------------------+
|  Layer 5: Integration                                              |
|                                                                     |
|  Framework Adapters   pytest Plugin   CLI (6 commands)              |
|  Console Reporter   HTML Reporter   JSON Exporter                   |
|  Dashboard   SQLite Persistence   WebSocket Event Bus               |
+-------------------------------------------------------------------+
|  Layer 4: Analysis                                                 |
|                                                                     |
|  Coverage (5D)   Mutation Testing   Metamorphic Testing             |
|  Contract Oracle                                                    |
+-------------------------------------------------------------------+
|  Layer 3: Verdicts                                                 |
|                                                                     |
|  Stochastic Verdict System   Deployment Gates                       |
+-------------------------------------------------------------------+
|  Layer 2: Statistics                                               |
|                                                                     |
|  Hypothesis Tests   Confidence Intervals   Sequential Testing       |
|  Effect Size Computation   Power Analysis                           |
+-------------------------------------------------------------------+
|  Layer 1: Core                                                     |
|                                                                     |
|  Data Models   Execution Engine   Trace Format                      |
+-------------------------------------------------------------------+
```

### Layer Descriptions

**Layer 1 -- Core.** Defines the data models (traces, scenarios, results, configurations) and the execution engine that runs agents and collects traces. Everything above builds on these immutable data structures.

**Layer 2 -- Statistics.** The mathematical backbone. Hypothesis testing for regression detection, confidence interval estimation (multiple methods), sequential probability ratio test for adaptive sampling, effect size quantification, and power analysis. All computations use established, peer-reviewed statistical methods.

**Layer 3 -- Verdicts.** Transforms raw statistical outputs into actionable three-valued verdicts (PASS / FAIL / INCONCLUSIVE) and deployment gate decisions (ALLOW / BLOCK). This is the decision layer.

**Layer 4 -- Analysis.** Higher-order testing techniques: five-dimensional coverage metrics, mutation testing with operators across four categories, metamorphic testing with relations across four families, and contract compliance checking.

**Layer 5 -- Integration.** Connects AgentAssay to the outside world: framework adapters that normalize agent invocations, a pytest plugin for test discovery and execution, a CLI for command-line workflows, and reporters for human-readable and machine-readable output.

**Layer 6 -- Efficiency.** The differentiating layer. Sits atop the full statistical stack and optimizes *how many* runs are needed while Layers 1-5 ensure *every* run produces rigorous results. This layer contains:

- **Behavioral Fingerprinting** -- Extracts low-dimensional behavioral signatures from execution traces. Comparing fingerprints requires fewer samples than comparing raw outputs.
- **Adaptive Budget Optimization** -- Runs a calibration phase to measure scenario variance, then computes the exact minimum trials needed for target confidence.
- **Trace-First Offline Analysis** -- Routes coverage, contract, and metamorphic analyses to existing production traces instead of requiring new agent invocations.
- **Multi-Fidelity Proxy Testing** -- Screens scenarios with cheaper models first, escalating only uncertain cases to full-price models.
- **Warm-Start Sequential Testing** -- Incorporates prior test results as Bayesian priors, allowing sequential tests to reach conclusions faster.

## Data Flow

### Primary Flow: Token-Efficient Testing

```
  Production Traces ---------> Trace Store ---------> Offline Analysis
  (already paid for)                                   (coverage, contracts,
                                                        metamorphic -- FREE)
                                                              |
  New Agent Version ---------> Calibration ----------> Budget Estimate
                               (5-10 runs)             (exact minimum N)
                                                              |
                            Targeted Testing ----------> Fingerprint
                            (optimal N runs)              Comparison
                                                              |
                                                    Statistical Verdict
                                                    (5-20x cheaper)
```

### Detailed Flow: Full Pipeline

```
  Test Input
      |
      v
  Budget Optimizer ---- calibrate (5-10 runs) ----> Recommended N
      |
      v
  Trial Runner ---- runs agent N times ----> [Trace_1, Trace_2, ..., Trace_N]
      |                                              |
      |                                    +---------+---------+-----------+
      |                                    |         |         |           |
      |                              Evaluator  Coverage  Mutation  Fingerprint
      |                                    |    Collector   Runner   Extractor
      |                                    v         |         |           |
      |                             [Result_1..N]    |         |           |
      |                                    |         |         |           |
      v                                    v         v         v           v
  Verdict Function              Stochastic  Coverage  Mutation  Behavioral
  (alpha, beta, n)              Verdict     Tuple     Score    Distance
      |                            |            |         |           |
      v                            v            v         v           v
  Deployment Gate  <----- aggregates all evidence --------+-----------+
      |
      v
  Gate Decision: ALLOW or BLOCK
      |
      v
  CI/CD Pipeline (exit code 0 or 1)
```

### Step-by-Step

1. **Budget:** The adaptive budget optimizer runs a small calibration (5-10 trials) to measure variance and computes the minimum number of additional trials needed for the target confidence level.
2. **Execution:** The Trial Runner invokes the agent N times (where N is the optimizer's recommendation), producing N execution traces.
3. **Evaluation:** Each trace is evaluated against the expected properties, producing N trial results (pass/fail + score).
4. **Analysis:** The trial results flow to multiple analysis modules in parallel:
   - The **verdict function** computes confidence intervals and produces a three-valued verdict.
   - The **coverage collector** analyzes the traces to compute the five-dimensional coverage tuple.
   - The **mutation runner** (if enabled) re-runs trials with perturbed agent configurations.
   - The **metamorphic runner** (if enabled) verifies behavioral invariants.
   - The **fingerprint extractor** computes behavioral fingerprints and compares against the baseline.
5. **Offline analysis:** Coverage, contract, and metamorphic checks also run against existing traces from the trace store, at zero additional token cost.
6. **Reporting:** Results are formatted for the console, HTML, or JSON.
7. **Gating:** The deployment gate aggregates all verdicts and evidence, then issues a binary ALLOW/BLOCK decision.

## Key Data Models

| Model | Purpose | Mutability |
|-------|---------|------------|
| `StepTrace` | One atomic action in an agent run | Immutable |
| `ExecutionTrace` | Complete agent execution (ordered steps) | Immutable |
| `TestScenario` | Test case definition (input + expected properties) | Immutable |
| `TrialResult` | One trial outcome (trace + verdict) | Immutable |
| `BehavioralFingerprint` | Compact behavioral signature of an execution | Immutable |
| `BudgetEstimate` | Recommended N, cost estimate, savings | Immutable |
| `AgentConfig` | Agent-under-test description | Mutable |
| `AssayConfig` | Statistical parameters and resource limits | Mutable |
| `StochasticVerdict` | Three-valued verdict with statistical backing | Immutable |
| `CoverageTuple` | Five-dimensional coverage vector | Immutable |

Result models are immutable to guarantee that analysis never accidentally modifies captured data. Configuration models are mutable because they are set up before the test run begins.

## Adapter Pattern

AgentAssay does not depend on any specific agent framework at the core level. Integration happens through the adapter pattern:

```
Your Agent Framework             AgentAssay Core
+------------------+            +------------------+
|                  |            |                  |
|  Framework A     |---+       |  Trial Runner    |
|  Framework B     |   |       |                  |
|  Framework C     |   +------>|  Expects:        |
|  Framework D     |   |       |  f(dict) -> Trace|
|  Framework E     |   |       |                  |
|  Custom          |---+       +------------------+
|                  |
+------------------+
       Adapter
```

Each adapter converts a framework-specific agent into a simple callable: `Callable[[dict], ExecutionTrace]`. The core engine does not know or care which framework is being used.

## Statistical Foundation

The statistical layer is the backbone of AgentAssay. It provides:

- **Hypothesis testing:** One-sided tests for regression detection (comparing two versions)
- **Confidence intervals:** Multiple estimation methods for different sample size regimes
- **Sequential testing:** Adaptive early stopping that maintains error rate guarantees
- **Effect size:** Quantifying the magnitude of detected regressions
- **Power analysis:** Determining whether the test had enough power to detect a given effect

All statistical computations use established, peer-reviewed methods.

## Efficiency Layer

The efficiency layer is what separates AgentAssay from evaluation frameworks. It answers the question: **how do I get the same statistical confidence with fewer token-consuming runs?**

```
Traditional:     Run 100 trials ---------> Analyze ---------> Verdict
                 (high cost)

AgentAssay:      Calibrate (10 runs) ---> Budget (N=17) ---> Run 17 ---> Verdict
                 + Offline traces ------> Coverage, Contracts (free)
                 + Fingerprints ---------> Regression (fewer samples)
                 + Prior results ---------> Warm-start (faster convergence)
                 (5-20x less cost, same confidence)
```

For detailed explanation of each efficiency technique, see [Token-Efficient Testing](../concepts/token-efficient-testing.md).

## Extension Points

AgentAssay is designed to be extended at several points:

| Extension Point | How |
|-----------------|-----|
| New agent framework | Write an adapter function that returns `ExecutionTrace` |
| Custom evaluator | Register a function that checks `expected_properties` |
| New mutation operator | Subclass the mutation operator base and implement the perturbation |
| New metamorphic relation | Subclass the metamorphic relation base and define the invariant |
| Custom reporter | Implement the reporter interface for new output formats |
| Custom fingerprint extractor | Define domain-specific behavioral features for fingerprinting |
| Custom budget strategy | Implement a budget optimizer with domain-specific variance models |

## Dashboard and Observability

AgentAssay includes a built-in dashboard that provides real-time visualization of test results, historical trends, behavioral fingerprint comparison, and deployment gate monitoring.

The dashboard connects to the same data layer that powers the CLI reporters:

```
Test Runner --> Event Bus --> Persistence Layer --> Dashboard
                                   |
                                   +--> CLI Reporters (console, HTML, JSON)
                                   |
                                   +--> Metrics Exporter
```

The persistence layer stores all test runs, trials, verdicts, coverage, fingerprints, and gate decisions. Historical data enables trend analysis and cross-run comparison that the one-shot CLI reporters cannot provide.
