# Dashboard

AgentAssay includes a built-in dashboard for visualizing test results, tracking quality trends, and monitoring token costs across your agent test suites.

## Quick Start

```bash
agentassay dashboard
```

Opens the dashboard at `http://localhost:8501`. The dashboard reads from your local test results database.

## Prerequisites

Install AgentAssay with dashboard extras:

```bash
pip install agentassay[dashboard]
```

You need at least one completed test run to have data to visualize:

```bash
agentassay run --config agentassay.yaml
agentassay dashboard
```

## Views

### Overview

The landing page. Displays summary cards for total runs, pass rate, 5D coverage score, and total token cost. Below the cards: a recent runs list, 7-day trend sparklines for key metrics, and a gate decision summary showing how many deployments were approved or blocked.

### Test Run (Live)

Watch a test run in progress. Shows a live progress bar, per-scenario results table with confidence intervals updating in real-time, SPRT status indicator (continue/accept/reject), and a running cost accumulator.

To use live view, start a test run in one terminal and open the dashboard in another:

```bash
# Terminal 1
agentassay run --config agentassay.yaml

# Terminal 2
agentassay dashboard
```

The live view auto-detects running test sessions.

### History

Trend analysis across runs. Three main charts: pass rate over time, cost per run over time, and coverage score over time. Filterable by agent, model, and date range. Click any data point to drill into that specific run's details.

### Behavioral Fingerprints

The most distinctive view. Displays a behavioral heatmap (dimensions x scenarios), a drift detection table with Hotelling's T-squared statistics, and a dimension drill-down with distribution comparisons between baseline and current runs.

Use this view to understand _how_ your agent's behavior changed, not just _whether_ it regressed.

### Coverage

5D coverage radar chart showing tool, path, state, boundary, and model coverage. Includes a tool coverage map showing which tools have been exercised and a path coverage tree highlighting execution paths with gap indicators.

For details on each coverage dimension, see [Coverage](../concepts/coverage.md).

### CI/CD Gates

Deployment gate decision timeline showing approve/block history. Includes a gate configuration viewer, blocked deployment detail with the regression analysis that triggered the block, and an SLA compliance summary.

For gate configuration, see [CI/CD Integration](ci-cd-integration.md).

### Settings

Configure model endpoints, budget limits, alert thresholds, and per-pipeline gate rules directly from the dashboard. Changes are written back to your `agentassay.yaml`.

## Configuration

The dashboard reads from the AgentAssay results database. Default location: `~/.agentassay/results.db`. Override with the `AGENTASSAY_DB_PATH` environment variable.

```yaml
# agentassay.yaml
dashboard:
  port: 8501
  host: 0.0.0.0
  theme: dark  # dark | light
  refresh_interval: 5  # seconds for live view updates
```

## CLI Options

```bash
agentassay dashboard [OPTIONS]

Options:
  --port INTEGER    Port to run dashboard on (default: 8501)
  --host TEXT       Host to bind to (default: localhost)
  --no-browser      Don't auto-open browser
  --theme TEXT       Color theme: dark, light (default: dark)
```

## Notes

- The dashboard is read-only for test data. It visualizes results but does not modify test configurations (except through the Settings view).
- For CI/CD environments, use the CLI commands (`agentassay run`, `agentassay compare`, `agentassay gate`) instead of the dashboard.
- All statistical values shown in the dashboard (confidence intervals, p-values, effect sizes) use the same methods as the CLI. See [Stochastic Testing](../concepts/stochastic-testing.md) for the underlying methodology.
- Token cost tracking requires that your framework adapter reports `total_cost_usd` in the `ExecutionTrace`. See [Framework Adapters](framework-adapters.md) for setup.
