# Quickstart

Test your first AI agent in 5 minutes -- with token-efficient budget optimization.

## 1. Install AgentAssay

```bash
pip install agentassay
```

## 2. Define Your Agent as a Callable

AgentAssay tests any agent that can be wrapped as a function. The function takes a dictionary of inputs and returns an `ExecutionTrace` that records what the agent did.

```python
from agentassay.core.models import ExecutionTrace, StepTrace

def my_agent(input_data: dict) -> ExecutionTrace:
    """Your agent logic goes here.

    In practice, this wraps your real agent framework.
    For this example, we simulate a simple agent.
    """
    prompt = input_data.get("prompt", "")

    # Simulate the agent doing work
    steps = [
        StepTrace(
            step_index=0,
            action="llm_response",
            llm_input=prompt,
            llm_output=f"Response to: {prompt}",
            model="gpt-4o",
            duration_ms=250.0,
        ),
    ]

    return ExecutionTrace(
        scenario_id="quickstart",
        steps=steps,
        input_data=input_data,
        output_data={"response": f"Response to: {prompt}"},
        success=True,
        total_duration_ms=250.0,
        total_cost_usd=0.001,
        model="gpt-4o",
        framework="custom",
    )
```

## 3. Token-Efficient Testing: Know How Many Trials You Actually Need

Before running a full test suite, let AgentAssay tell you the minimum number of trials each scenario requires. This avoids over-testing (wasting tokens) and under-testing (missing regressions).

```python
from agentassay.efficiency import BehavioralFingerprint, AdaptiveBudgetOptimizer

# Step 1: Run a small calibration set (10 trials)
calibration_traces = [my_agent({"prompt": "Hello"}) for _ in range(10)]

# Step 2: Let the optimizer compute the minimum N
optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10)
estimate = optimizer.calibrate(calibration_traces)

print(f"Recommended trials: {estimate.recommended_n}")         # e.g., 17
print(f"Estimated cost: ${estimate.estimated_cost_usd:.2f}")   # e.g., $0.34
print(f"Savings vs fixed-100: {estimate.savings_vs_fixed_100:.0%}")  # e.g., 83%
```

The optimizer measures behavioral variance from the calibration runs and computes the exact minimum N for your target confidence level. Low-variance scenarios need fewer trials. High-variance scenarios get more.

## 4. Write a pytest Test

```python
import pytest
from agentassay.core.models import TestScenario
from agentassay.plugin.pytest_plugin import assert_pass_rate

@pytest.mark.agentassay(n=30, threshold=0.80)
def test_my_agent_pass_rate(trial_runner):
    """Test that the agent passes at least 80% of trials."""
    runner = trial_runner(my_agent)

    scenario = TestScenario(
        scenario_id="greeting",
        name="Greeting test",
        description="Agent should respond to a greeting",
        input_data={"prompt": "Hello, how are you?"},
        expected_properties={"max_steps": 5},
    )

    results = runner.run_trials(scenario)
    passed = [r.passed for r in results]

    assert_pass_rate(passed, threshold=0.80, confidence=0.95)
```

## 5. Run the Test

```bash
python -m pytest test_my_agent.py -v
```

You will see output like:

```
test_my_agent.py::test_my_agent_pass_rate PASSED

============================= AgentAssay Summary ==============================
  assert_pass_rate: rate=100.00% [93.02%, 100.00%] (30/30 trials) verdict=PASS
  AgentAssay totals: 1 tests -- 1 PASS, 0 FAIL, 0 INCONCLUSIVE
```

## 6. Detect a Regression with Behavioral Fingerprinting

Instead of comparing raw outputs (expensive, noisy), compare behavioral fingerprints:

```python
from agentassay.efficiency import BehavioralFingerprint
from agentassay.plugin.pytest_plugin import assert_no_regression

def test_no_regression():
    """Compare baseline and current agent using behavioral fingerprints."""
    # Run both versions
    baseline_traces = [baseline_agent({"prompt": "Book a flight"}) for _ in range(20)]
    current_traces = [current_agent({"prompt": "Book a flight"}) for _ in range(20)]

    # Extract fingerprints -- compact behavioral signatures
    baseline_fp = BehavioralFingerprint.from_traces(baseline_traces)
    current_fp = BehavioralFingerprint.from_traces(current_traces)

    # Compare: low distance = no regression, high distance = regression
    drift = baseline_fp.distance(current_fp)
    print(f"Behavioral drift: {drift:.4f}")

    # Or use the statistical regression test
    baseline_results = [True] * 28 + [False] * 2   # 93% baseline
    current_results = [True] * 20 + [False] * 10    # 67% current
    assert_no_regression(baseline_results, current_results, alpha=0.05)
```

## 7. Analyze Existing Traces Offline (Zero Token Cost)

Run coverage and contract checks on traces you already have -- no new agent invocations needed:

```python
from agentassay.coverage import AgentCoverageCollector

# Load traces from your production monitoring or previous test runs
traces = load_traces_from_store("production-traces/2026-02-28/")

# Compute coverage metrics -- completely offline, zero token cost
collector = AgentCoverageCollector(
    known_tools=["search", "book", "cancel", "confirm"],
)
coverage = collector.compute(traces)

print(f"Tool coverage: {coverage.tool:.1%}")
print(f"Path coverage: {coverage.path:.1%}")
print(f"State coverage: {coverage.state:.1%}")
print(f"Boundary coverage: {coverage.boundary:.1%}")
print(f"Model coverage: {coverage.model:.1%}")
```

## 8. Use the CLI

```bash
# Run with adaptive budget optimization
agentassay run --scenario booking.yaml --budget-mode adaptive

# Compare two result files for regression
agentassay compare --baseline v1-results.json --current v2-results.json

# View coverage metrics from existing traces
agentassay coverage --traces production-traces/ --tools search,book,cancel

# Generate an HTML report
agentassay report --results trials.json --output report.html
```

## What's Next?

- [Token-Efficient Testing](../concepts/token-efficient-testing.md) -- Deep dive into behavioral fingerprinting, adaptive budgeting, and trace-first analysis
- [Stochastic Testing Concepts](../concepts/stochastic-testing.md) -- Understand why three-valued verdicts matter
- [Coverage Metrics](../concepts/coverage.md) -- Learn what the five coverage dimensions measure
- [pytest Plugin Guide](../guides/pytest-plugin.md) -- Deeper dive into test writing
- [CI/CD Integration](../guides/ci-cd-integration.md) -- Set up automated deployment gates
