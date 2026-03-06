# Configuration Reference

AgentAssay is configured through YAML files, Python objects, or pytest markers.

## AssayConfig (Test Run Configuration)

Controls the statistical parameters, resource limits, and execution strategy for a test run.

### YAML Format

```yaml
# assay-config.yaml

# -- Statistical parameters --
num_trials: 50              # Number of stochastic trials per scenario
significance_level: 0.05    # Alpha: Type I error rate (false positive)
power: 0.80                 # 1 - beta: probability of detecting a true regression
effect_size_threshold: 0.10 # Minimum pass rate drop to flag as regression

# -- Confidence interval method --
# Options: wilson, clopper-pearson, normal
confidence_method: wilson

# -- Regression hypothesis test --
# Options: fisher, chi2, ks, mann-whitney
regression_test: fisher

# -- Sequential testing (SPRT) --
use_sprt: false             # Enable adaptive early stopping
sprt_strength: 0.0          # Log-likelihood ratio threshold (0 = auto-derive)

# -- Reproducibility --
seed: 42                    # Random seed (optional)

# -- Resource limits --
timeout_seconds: 600        # Maximum time per trial run
max_cost_usd: 50.00         # Maximum total cost
parallel_trials: 1          # Number of parallel trials (1-256)
```

### Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_trials` | int | 30 | 1 - 10,000 | Trials per scenario. More trials = tighter confidence intervals. |
| `significance_level` | float | 0.05 | (0, 1) | Alpha. Lower = fewer false positives but needs more trials. |
| `power` | float | 0.80 | (0, 1) | 1 - beta. Higher = better regression detection but needs more trials. |
| `effect_size_threshold` | float | 0.10 | [0, 1] | Minimum meaningful drop in pass rate. |
| `confidence_method` | str | "wilson" | See below | Method for computing confidence intervals. |
| `regression_test` | str | "fisher" | See below | Statistical test for regression comparison. |
| `use_sprt` | bool | false | | Enable SPRT adaptive stopping. |
| `sprt_strength` | float | 0.0 | >= 0 | SPRT threshold. 0 = auto-derive from alpha/beta. |
| `seed` | int | null | | Random seed for reproducibility. |
| `timeout_seconds` | float | 600.0 | > 0 | Max seconds per test run. |
| `max_cost_usd` | float | 50.0 | >= 0 | Max total cost in USD. |
| `parallel_trials` | int | 1 | 1 - 256 | Number of parallel trial executions. |

### Confidence Interval Methods

| Method | Best for | Notes |
|--------|----------|-------|
| `wilson` | **Recommended default.** Small to medium sample sizes. | Good coverage properties even with small N. |
| `clopper-pearson` | Conservative estimates. | Exact method. Wider intervals (more conservative). |
| `normal` | Large sample sizes (N > 100). | Asymptotic. May be inaccurate for small N or extreme pass rates. |

### Regression Test Methods

| Method | Best for | Notes |
|--------|----------|-------|
| `fisher` | **Recommended default.** Small sample sizes. | Exact test. No assumptions. |
| `chi2` | Large sample sizes. | Asymptotic. Faster for large N. |
| `ks` | Comparing score distributions. | Non-parametric. Sensitive to any distributional change. |
| `mann-whitney` | Comparing continuous scores. | Non-parametric. Tests stochastic dominance. |

## AgentConfig (Agent Under Test)

Describes the agent being tested.

### YAML Format

```yaml
# agent-config.yaml
agent_id: research-agent-v2
name: Research Agent
framework: langgraph        # langgraph, crewai, autogen, openai, smolagents, custom
model: gpt-4o
version: 2.1.0
parameters:
  temperature: 0.7
  max_tokens: 4096
metadata:
  team: platform
  environment: staging
```

### Supported Frameworks

| Framework | Value |
|-----------|-------|
| LangGraph | `langgraph` |
| CrewAI | `crewai` |
| AutoGen | `autogen` |
| OpenAI Agents SDK | `openai` |
| smolagents | `smolagents` |
| Custom | `custom` |

## TestScenario (Test Case Definition)

Defines a single test case for an agent.

### YAML Format

```yaml
# scenario.yaml
scenario_id: qa-capital-cities
name: Capital City QA
description: Agent should correctly answer questions about capital cities
timeout_seconds: 120

input_data:
  prompt: "What is the capital of Japan?"
  context: "Geography questions"

expected_properties:
  output_contains: "Tokyo"
  max_steps: 10
  max_cost_usd: 0.05
  must_use_tools:
    - search

tags:
  - geography
  - qa
  - basic
```

### Expected Properties

| Property | Type | Description |
|----------|------|-------------|
| `output_contains` | str | Agent output must contain this substring |
| `max_steps` | int | Maximum number of execution steps allowed |
| `max_cost_usd` | float | Maximum cost per trial |
| `must_use_tools` | list[str] | Tools that must be invoked during execution |

You can define any key-value pairs in `expected_properties`. Custom evaluator functions can check these properties.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AGENTASSAY_CONFIG` | Path to default AssayConfig YAML file |
| `AGENTASSAY_LOG_LEVEL` | Logging level: DEBUG, INFO, WARNING, ERROR |

## pytest Configuration

AgentAssay's pytest plugin is configured through `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "agentassay: marks tests as agent assay tests",
    "slow: marks tests as slow",
]
```

Use the `@pytest.mark.agentassay` marker to configure individual tests. See the [pytest Plugin Guide](../guides/pytest-plugin.md) for details.
