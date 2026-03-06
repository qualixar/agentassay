# CI/CD Integration

AgentAssay is designed to run in continuous integration pipelines. Its deployment gate system blocks releases when statistical evidence shows the agent has regressed.

## Deployment Gates

A deployment gate evaluates a suite of stochastic test results and produces a binary decision: **ALLOW** or **BLOCK** the deployment.

### How it Works

1. Run your agent tests with AgentAssay (multiple trials per scenario).
2. Each test produces a stochastic verdict (PASS / FAIL / INCONCLUSIVE).
3. The deployment gate aggregates all verdicts and decides:
   - **ALLOW:** All tests passed. Deploy with confidence.
   - **BLOCK:** One or more tests failed or were inconclusive. Do not deploy.

### Gate Policies

The deployment gate supports multiple policies for handling inconclusive results:

| Policy | Behavior |
|--------|----------|
| **strict** | Block on any FAIL or INCONCLUSIVE. Only deploy when every test is a definitive PASS. |
| **moderate** | Block on FAIL. Allow INCONCLUSIVE (treat as non-blocking warnings). |
| **permissive** | Only block on FAIL. INCONCLUSIVE tests are allowed. |

For production deployments, the **strict** policy is recommended. Use **moderate** during development when running with fewer trials.

## GitHub Actions Example

```yaml
name: Agent Regression Test

on:
  pull_request:
    branches: [main]

jobs:
  agent-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run agent regression tests
        run: python -m pytest tests/ -m agentassay -v --tb=short

      - name: Compare against baseline
        run: |
          agentassay compare \
            --baseline tests/baseline-results.json \
            --current tests/current-results.json \
            --alpha 0.05
```

The `agentassay compare` command exits with code 1 if a regression is detected, which causes the CI job to fail and block the pull request.

## Saving and Loading Baselines

A common pattern is to save baseline results from a known-good version and compare against them in CI:

```python
import json
from agentassay.core.runner import TrialRunner

# Generate baseline (run once, save results)
results = runner.run_trials(scenario)
baseline_data = [{"passed": r.passed, "score": r.score} for r in results]

with open("tests/baseline-results.json", "w") as f:
    json.dump(baseline_data, f, indent=2)
```

Commit the baseline file to your repository. On every PR, AgentAssay compares the new results against this saved baseline.

## Handling Cost in CI

Agent tests cost money when they invoke real LLM APIs. AgentAssay provides controls to manage this:

### Budget Limits

Set a maximum cost per test run:

```yaml
# In your assay config
max_cost_usd: 5.00
timeout_seconds: 300
```

### SPRT for Cost Reduction

Enable adaptive early stopping to reduce the average number of trials:

```python
@pytest.mark.agentassay(n=100, use_sprt=True)
def test_agent_with_sprt(trial_runner):
    # SPRT may stop after 30-50 trials if the answer is clear
    ...
```

### Tiered CI Strategy

Run different levels of testing at different stages:

| Stage | Trials | Purpose |
|-------|--------|---------|
| Pre-commit (local) | 10 | Quick sanity check |
| Pull request CI | 30 | Standard regression check |
| Pre-production | 100+ | Full statistical validation |

## Exit Codes

The CLI uses exit codes compatible with CI systems:

| Exit Code | Meaning |
|-----------|---------|
| 0 | No regression detected / all tests passed |
| 1 | Regression detected / test failure |

## HTML Reports as CI Artifacts

Generate an HTML report and upload it as a CI artifact for review:

```yaml
      - name: Generate report
        if: always()
        run: agentassay report --results tests/current-results.json --output report.html

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: agentassay-report
          path: report.html
```

## Integrating with Deployment Pipelines

### Block a Deployment

```python
from agentassay.verdicts.gate import DeploymentGate

gate = DeploymentGate(policy="strict")

# Feed in all test verdicts
for verdict in test_verdicts:
    gate.add_verdict(verdict)

# Get the decision
decision = gate.evaluate()

if decision.blocked:
    print(f"DEPLOYMENT BLOCKED: {decision.reason}")
    sys.exit(1)
else:
    print("DEPLOYMENT ALLOWED")
    sys.exit(0)
```

### Notify on Regression

Combine AgentAssay with your notification system:

```python
if decision.blocked:
    # Send alert to Slack, PagerDuty, etc.
    notify_team(
        message=f"Agent regression detected. "
        f"Pass rate dropped from {decision.baseline_rate:.0%} "
        f"to {decision.current_rate:.0%}. "
        f"p-value: {decision.p_value:.6f}"
    )
```
