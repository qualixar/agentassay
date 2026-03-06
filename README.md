<p align="center">
  <h1 align="center">AgentAssay</h1>
  <p align="center"><strong>Test More. Spend Less. Ship Confident.</strong></p>
  <p align="center">The first agent testing framework that delivers statistical guarantees WITHOUT burning your token budget.</p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.02601"><img src="https://img.shields.io/badge/arXiv-2603.02601-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://doi.org/10.5281/zenodo.18842011"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18842011.svg" alt="DOI"></a>
  <a href="https://github.com/qualixar/agentassay/actions"><img src="https://img.shields.io/github/actions/workflow/status/qualixar/agentassay/ci.yml?branch=main&style=flat-square" alt="Build"></a>
  <a href="https://codecov.io/gh/qualixar/agentassay"><img src="https://img.shields.io/codecov/c/github/qualixar/agentassay?style=flat-square" alt="Coverage"></a>
  <a href="https://pypi.org/project/agentassay/"><img src="https://img.shields.io/pypi/v/agentassay?style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/agentassay/"><img src="https://img.shields.io/pypi/pyversions/agentassay?style=flat-square" alt="Python"></a>
  <a href="https://github.com/qualixar/agentassay/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License"></a>
</p>

---

## The Problem

Every time you change a prompt, swap a model, or update a tool, you need to know: **does my agent still work?**

Today, answering that question is painfully expensive. Run 100 trials across 20 scenarios, and you've burned thousands of tokens just to check for a regression. Most teams either:

- **Over-test:** Run fixed-N trials and waste budget on scenarios that don't need it.
- **Under-test:** Skip testing because the cost is too high, and ship broken agents.
- **Guess:** Run a few trials, eyeball the results, and hope for the best.

None of these are engineering. They are gambling.

## The Solution

AgentAssay introduces **token-efficient agent testing** -- three techniques that deliver the same statistical confidence at a fraction of the cost:

### 1. Behavioral Fingerprinting

Instead of comparing raw text outputs (high-dimensional, noisy, expensive), AgentAssay extracts **behavioral fingerprints** -- compact representations of what the agent *did* rather than what it *said*. Tool sequences, state transitions, decision patterns. Low-dimensional signals need fewer samples to detect change.

### 2. Adaptive Budget Optimization

No more guessing how many trials to run. AgentAssay runs a small calibration set (5-10 runs), measures behavioral variance, and computes the **exact minimum number of trials** needed for your target confidence level. High-variance scenarios get more trials. Stable scenarios get fewer. Zero waste.

### 3. Trace-First Offline Analysis

Coverage metrics, contract checks, metamorphic relations, and mutation analysis can all run on **production traces you already have** -- at zero additional token cost. Why re-run your agent when you can analyze runs that already happened?

**Result: Same confidence. 83% less cost.**

---

## Install

```bash
pip install agentassay
```

```bash
# With framework adapters
pip install agentassay[all]
```

```bash
# Development
pip install agentassay[dev]
```

---

## Quick Example: Token-Efficient Testing

```python
from agentassay.efficiency import BehavioralFingerprint, AdaptiveBudgetOptimizer
from agentassay.core.runner import TrialRunner
from agentassay.verdicts import VerdictFunction

# Step 1: Calibrate -- run just 10 trials to measure variance
optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10)
estimate = optimizer.calibrate(calibration_traces)

print(f"Recommended trials: {estimate.recommended_n}")   # e.g., 17 (not 100)
print(f"Estimated cost: ${estimate.estimated_cost_usd:.2f}")  # e.g., $0.34
print(f"Savings vs fixed-100: {estimate.savings_vs_fixed_100:.0%}")  # e.g., 83%

# Step 2: Run only the trials you need
runner = TrialRunner(agent_fn=my_agent, config=config)
results = runner.run_trials(scenario, n=estimate.recommended_n)

# Step 3: Compare fingerprints for regression detection
baseline_fp = BehavioralFingerprint.from_traces(baseline_traces)
current_fp = BehavioralFingerprint.from_traces(current_traces)
drift = baseline_fp.distance(current_fp)

# Step 4: Get a statistically-backed verdict
verdict = VerdictFunction(alpha=0.05).evaluate(results)
print(f"Verdict: {verdict.status}")  # PASS / FAIL / INCONCLUSIVE
print(f"Pass rate: {verdict.pass_rate:.1%} [{verdict.ci_lower:.1%}, {verdict.ci_upper:.1%}]")
```

---

## How It Works

```
                      Token-Efficient Testing Pipeline
  +-----------------------------------------------------------------+
  |                                                                   |
  |  Production Traces -----> Trace Store -----> Offline Analysis     |
  |  (already paid for)                          (coverage, contracts,|
  |                                               metamorphic -- FREE)|
  |                                                     |             |
  |  New Agent Version --> Calibration (5-10 runs) --> Budget Estimate|
  |                                                     |             |
  |                   Targeted Testing (optimal N) --> Fingerprint    |
  |                                                    Comparison     |
  |                                                     |             |
  |                                          Statistical Verdict      |
  |                                          (5-20x cheaper)          |
  +-----------------------------------------------------------------+
```

The core insight: **most of the information you need to test an agent is already in traces you have collected.** AgentAssay extracts maximum signal from minimum runs.

---

## Feature Matrix

| Feature | Description |
|---------|-------------|
| **Behavioral fingerprinting** | Detect regression from behavioral patterns, not raw text. Fewer samples needed. |
| **Adaptive budget optimization** | Calibrate variance, compute exact minimum N. No over-testing. |
| **Trace-first offline analysis** | Run coverage, contracts, and metamorphic checks on existing traces. Zero token cost. |
| **Multi-fidelity proxy testing** | Use cheaper models for initial screening, expensive models only for confirmation. |
| **Warm-start sequential testing** | Incorporate prior results to reach verdicts faster. |
| **Three-valued verdicts** | PASS, FAIL, or INCONCLUSIVE -- never a misleading binary answer. |
| **Confidence intervals** | Know the true pass rate range, not a point estimate. |
| **Statistical regression detection** | Hypothesis tests catch regressions before production. |
| **5D coverage metrics** | Measure tool, path, state, boundary, and model coverage. |
| **Mutation testing** | Perturb your agent to validate test sensitivity. |
| **Metamorphic testing** | Verify behavioral invariants across input transformations. |
| **Contract oracle** | Check behavioral specifications from AgentAssert contracts. |
| **Deployment gates** | Block broken deployments in CI/CD with statistical evidence. |
| **Framework adapters** | Works with popular agent frameworks out of the box. |
| **pytest integration** | Use familiar pytest conventions with statistical assertions. |
| **CLI** | Five commands: `run`, `compare`, `mutate`, `coverage`, `report`. |

---

## Comparison

| Feature | AgentAssay | deepeval | agentrial | LangSmith |
|---------|:----------:|:--------:|:---------:|:---------:|
| Statistical regression testing | :white_check_mark: | :x: | :warning: | :x: |
| Three-valued verdicts | :white_check_mark: | :x: | :x: | :x: |
| **Token-efficient testing** | **:white_check_mark:** | **:x:** | **:x:** | **:x:** |
| **Behavioral fingerprinting** | **:white_check_mark:** | **:x:** | **:x:** | **:x:** |
| **Adaptive budget optimization** | **:white_check_mark:** | **:x:** | **:x:** | **:x:** |
| **Trace-first offline analysis** | **:white_check_mark:** | **:x:** | **:x:** | **:x:** |
| 5D coverage metrics | :white_check_mark: | :x: | :x: | :x: |
| Mutation testing | :white_check_mark: | :x: | :x: | :x: |
| Metamorphic testing | :white_check_mark: | :x: | :x: | :x: |
| CI/CD deployment gates | :white_check_mark: | :x: | :white_check_mark: | :x: |
| Published research paper | :white_check_mark: | :x: | :x: | :x: |

---

## Architecture

```
+-------------------------------------------------------------------+
|  Layer 6: Efficiency                                               |
|  Fingerprinting | Budget Optimization | Trace Analysis             |
|  Multi-Fidelity | Warm-Start Sequential                           |
+-------------------------------------------------------------------+
|  Layer 5: Integration                                              |
|  Framework Adapters | pytest Plugin | CLI | Reporting              |
+-------------------------------------------------------------------+
|  Layer 4: Analysis                                                 |
|  Coverage (5D) | Mutation | Metamorphic | Contract Oracle          |
+-------------------------------------------------------------------+
|  Layer 3: Verdicts                                                 |
|  Stochastic Verdicts | Deployment Gates                           |
+-------------------------------------------------------------------+
|  Layer 2: Statistics                                               |
|  Hypothesis Tests | Confidence Intervals | SPRT | Effect Size      |
+-------------------------------------------------------------------+
|  Layer 1: Core                                                     |
|  Data Models | Execution Engine | Trace Format                    |
+-------------------------------------------------------------------+
```

Layer 6 (Efficiency) is the differentiator. It sits atop the full statistical testing stack, optimizing *how many* runs are needed while Layers 1-5 ensure *every* run produces rigorous results.

---

## Usage with pytest

```python
import pytest

@pytest.mark.agentassay(n=30, threshold=0.80)
def test_agent_booking_flow(trial_runner):
    runner = trial_runner(my_agent)
    scenario = TestScenario(
        scenario_id="booking",
        name="Flight booking",
        input_data={"task": "Book a flight from NYC to London"},
        expected_properties={"max_steps": 10, "must_use_tools": ["search", "book"]},
    )
    results = runner.run_trials(scenario)
    assert_pass_rate(results, threshold=0.80, confidence=0.95)
```

```bash
python -m pytest tests/ -v --agentassay
```

---

## CLI

```bash
# Run trials with adaptive budget
agentassay run --scenario booking.yaml --budget-mode adaptive

# Compare two versions for regression
agentassay compare --baseline v1.json --current v2.json

# Analyze coverage from existing traces
agentassay coverage --traces production-traces/ --tools search,book,cancel

# Mutation testing
agentassay mutate --scenario booking.yaml --operators prompt,tool,model

# Generate report
agentassay report --results trials.json --output report.html
```

---

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Token-Efficient Testing](docs/concepts/token-efficient-testing.md) -- The core concept
- [Stochastic Testing](docs/concepts/stochastic-testing.md) -- Why agent testing needs statistics
- [Coverage Metrics](docs/concepts/coverage.md) -- Five-dimensional coverage model
- [Architecture Overview](docs/architecture/overview.md)
- [CLI Reference](docs/reference/cli.md)

---

## Research

AgentAssay is backed by a published research paper with formal definitions, theorems, and proofs.

**Paper:** [arXiv:2603.02601](https://arxiv.org/abs/2603.02601) (cs.AI + cs.SE)
**DOI:** [10.5281/zenodo.18842011](https://doi.org/10.5281/zenodo.18842011)

```bibtex
@article{bhardwaj2026agentassay,
  title={AgentAssay: Formal Regression Testing for Non-Deterministic AI Agent Workflows},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.02601},
  year={2026},
  doi={10.5281/zenodo.18842011}
}
```

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
