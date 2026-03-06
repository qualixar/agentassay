# pytest Plugin Guide

AgentAssay integrates with pytest through a plugin that is automatically registered when you install the package. No configuration needed -- just write tests and add the `@pytest.mark.agentassay` marker.

## Basic Usage

### The agentassay Marker

Mark any test function with `@pytest.mark.agentassay` to enable stochastic testing features:

```python
import pytest

@pytest.mark.agentassay(n=50, threshold=0.85, alpha=0.05)
def test_my_agent(trial_runner, assay_config):
    ...
```

### Marker Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 30 | Number of stochastic trials to run |
| `alpha` | float | 0.05 | Significance level (Type I error rate) |
| `threshold` | float | 0.80 | Minimum acceptable pass rate |
| `confidence_method` | str | "wilson" | Confidence interval method |
| `regression_test` | str | "fisher" | Hypothesis test for regression |
| `use_sprt` | bool | False | Enable adaptive early stopping |
| `power` | float | 0.80 | Statistical power (1 - Type II error) |
| `seed` | int | None | Random seed for reproducibility |
| `parallel` | int | 1 | Number of parallel trials |

## Fixtures

### `assay_config`

Provides an `AssayConfig` object configured from the marker parameters:

```python
@pytest.mark.agentassay(n=100, alpha=0.01)
def test_strict_agent(assay_config):
    print(assay_config.num_trials)         # 100
    print(assay_config.significance_level)  # 0.01
```

### `trial_runner`

Factory fixture that creates a `TrialRunner` for executing agent trials:

```python
@pytest.mark.agentassay(n=30)
def test_with_runner(trial_runner):
    # Create a runner for your agent
    runner = trial_runner(my_agent_callable)

    # Optionally provide agent configuration
    from agentassay.core.models import AgentConfig
    config = AgentConfig(
        agent_id="my-agent",
        name="My Agent",
        framework="custom",
        model="gpt-4o",
    )
    runner = trial_runner(my_agent_callable, agent_config=config)
```

## Custom Assertions

### `assert_pass_rate`

Asserts that the pass rate meets a threshold with statistical confidence:

```python
from agentassay.plugin.pytest_plugin import assert_pass_rate

def test_agent_reliability(trial_runner):
    runner = trial_runner(my_agent)
    results = runner.run_trials(scenario)
    passed = [r.passed for r in results]

    # Fails if the Wilson CI lower bound is below 85%
    assert_pass_rate(passed, threshold=0.85, confidence=0.95)
```

What this checks:
- Computes a confidence interval for the true pass rate
- Verifies that the **lower bound** of the CI is at or above the threshold
- This is stronger than checking the raw percentage -- it accounts for sample size

### `assert_no_regression`

Compares two sets of results and asserts no statistically significant regression:

```python
from agentassay.plugin.pytest_plugin import assert_no_regression

def test_no_regression_after_update():
    baseline = [True] * 45 + [False] * 5   # 90% (old version)
    current = [True] * 42 + [False] * 8    # 84% (new version)

    # Uses Fisher's exact test at alpha=0.05
    assert_no_regression(baseline, current, alpha=0.05)
```

When regression is detected, the error message includes:
- Baseline and current pass rates
- p-value and statistical test name
- Effect size with interpretation
- Detailed recommendation

### `assert_verdict_passes`

Uses the full three-valued verdict system for the strongest guarantee:

```python
from agentassay.plugin.pytest_plugin import assert_verdict_passes

def test_agent_with_verdict():
    results = [True] * 27 + [False] * 3  # 90% pass rate

    # Fails on FAIL or INCONCLUSIVE -- only accepts PASS
    assert_verdict_passes(results, threshold=0.80, alpha=0.05, min_trials=30)
```

This assertion is stricter than `assert_pass_rate` because it also considers whether the sample size is sufficient for a definitive verdict.

## Running Tests

### Run all agentassay-marked tests

```bash
python -m pytest -m agentassay -v
```

### Run with verbose AgentAssay summary

The plugin automatically adds a summary section to pytest output showing pass rates, confidence intervals, and verdicts for every agentassay test.

```
============================= AgentAssay Summary ==============================
  test_greeting: rate=93.33% [79.26%, 98.15%] (28/30 trials) verdict=PASS
  test_calculation: rate=86.67% [70.32%, 94.69%] (26/30 trials) verdict=PASS
  AgentAssay totals: 2 tests -- 2 PASS, 0 FAIL, 0 INCONCLUSIVE
```

### Skip slow stochastic tests during development

```bash
# Skip tests marked as slow
python -m pytest -m "not slow"

# Run only quick tests (reduce trial count via marker)
python -m pytest -m agentassay -v
```

## Complete Example

```python
import pytest
from agentassay.core.models import TestScenario, AgentConfig
from agentassay.plugin.pytest_plugin import (
    assert_pass_rate,
    assert_no_regression,
    assert_verdict_passes,
)


# -- Agent callable (wraps your real agent) --

def my_qa_agent(input_data: dict) -> "ExecutionTrace":
    """Wrap your agent framework here."""
    # ... your agent logic ...
    pass


# -- Test: pass rate meets threshold --

@pytest.mark.agentassay(n=50, threshold=0.85)
def test_qa_pass_rate(trial_runner):
    runner = trial_runner(my_qa_agent)
    scenario = TestScenario(
        scenario_id="qa-basic",
        name="Basic QA",
        input_data={"question": "What is the capital of France?"},
        expected_properties={"output_contains": "Paris"},
    )
    results = runner.run_trials(scenario)
    assert_pass_rate([r.passed for r in results], threshold=0.85)


# -- Test: no regression from baseline --

@pytest.mark.agentassay(n=50)
def test_qa_no_regression(trial_runner):
    runner = trial_runner(my_qa_agent)
    scenario = TestScenario(
        scenario_id="qa-baseline",
        name="Baseline comparison",
        input_data={"question": "Explain photosynthesis"},
    )

    # Run baseline (or load from saved results)
    baseline_results = runner.run_trials(scenario)
    baseline_passed = [r.passed for r in baseline_results]

    # Run current version
    current_results = runner.run_trials(scenario)
    current_passed = [r.passed for r in current_results]

    assert_no_regression(baseline_passed, current_passed)


# -- Test: strict verdict --

@pytest.mark.agentassay(n=100, alpha=0.01)
def test_qa_strict_verdict(trial_runner):
    runner = trial_runner(my_qa_agent)
    scenario = TestScenario(
        scenario_id="qa-strict",
        name="Strict reliability check",
        input_data={"question": "Translate 'hello' to French"},
        expected_properties={"output_contains": "bonjour"},
    )
    results = runner.run_trials(scenario)
    assert_verdict_passes(
        [r.passed for r in results],
        threshold=0.90,
        alpha=0.01,
        min_trials=100,
    )
```
