# CLI Reference

AgentAssay provides a command-line interface with six commands.

```bash
agentassay --help
agentassay --version
```

---

## `agentassay run`

Run agent assay trials against test scenarios.

```bash
agentassay run [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config PATH` | `-c` | YAML config file with AssayConfig parameters |
| `--scenario PATH` | `-s` | YAML scenario file with TestScenario definition |
| `--n INTEGER` | `-n` | Number of trials (overrides config file) |
| `--output PATH` | `-o` | Output JSON file for results |

### Examples

```bash
# Validate config and scenario files
agentassay run --config assay.yaml --scenario qa.yaml

# Override trial count
agentassay run -c config.yaml -s scenario.yaml -n 50

# Save validated config to JSON
agentassay run -c config.yaml -s scenario.yaml -o validated.json
```

### Notes

The `run` command validates configuration files and displays parameters. For full trial execution with real agents, use the Python API (`TrialRunner`). The CLI is designed for configuration validation, result analysis, and reporting.

---

## `agentassay compare`

Compare baseline vs. current results for regression detection.

```bash
agentassay compare [OPTIONS]
```

### Options

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--baseline PATH` | `-b` | Yes | JSON file with baseline trial results |
| `--current PATH` | `-c` | Yes | JSON file with current trial results |
| `--alpha FLOAT` | `-a` | No | Significance level (default: 0.05) |
| `--output PATH` | `-o` | No | Output JSON file for comparison results |

### Input Format

The JSON files must contain trial results in one of these formats:

```json
[{"passed": true}, {"passed": false}, {"passed": true}]
```

or:

```json
{"results": [{"passed": true}, {"passed": false}]}
```

or:

```json
{"trials": [{"success": true}, {"success": false}]}
```

### Examples

```bash
# Basic regression comparison
agentassay compare --baseline v1.json --current v2.json

# Stricter significance level
agentassay compare -b baseline.json -c current.json --alpha 0.01

# Save comparison results
agentassay compare -b v1.json -c v2.json -o comparison.json
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No regression detected |
| 1 | Regression detected |

### Output

Displays a table with:
- Baseline and current trial counts, pass counts, and pass rates
- 95% confidence intervals for both versions
- Hypothesis test result (test name, statistic, p-value, effect size)
- Verdict: REGRESSION DETECTED or NO REGRESSION

---

## `agentassay mutate`

Run mutation testing on an agent to evaluate test suite sensitivity.

```bash
agentassay mutate [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config PATH` | `-c` | YAML config file with AgentConfig parameters |
| `--scenario PATH` | `-s` | YAML scenario file with TestScenario definition |
| `--operators TEXT` | | Comma-separated operator categories: `prompt`, `tool`, `model`, `context` |
| `--output PATH` | `-o` | Output JSON file for mutation results |

### Operator Categories

| Category | Operators |
|----------|-----------|
| `prompt` | Synonym substitution, instruction order, noise injection, instruction drop |
| `tool` | Tool removal, tool reorder, tool noise |
| `model` | Model swap, model version |
| `context` | Context truncation, context noise, context permutation |

### Examples

```bash
# List all mutation operators
agentassay mutate --config agent.yaml --scenario qa.yaml

# Only prompt and tool mutations
agentassay mutate -c agent.yaml -s qa.yaml --operators prompt,tool

# Save operator list
agentassay mutate -c agent.yaml -s qa.yaml -o mutations.json
```

---

## `agentassay coverage`

Compute and display the five-dimensional coverage metrics from trial results.

```bash
agentassay coverage [OPTIONS]
```

### Options

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--results PATH` | `-r` | Yes | JSON file with trial results containing execution traces |
| `--tools TEXT` | | No | Comma-separated list of known tool names |
| `--models TEXT` | | No | Comma-separated list of known model names |

### Examples

```bash
# Basic coverage analysis
agentassay coverage --results trials.json

# With known tools for accurate tool coverage
agentassay coverage -r trials.json --tools search,calculate,write_file

# With known models
agentassay coverage -r trials.json --tools search --models gpt-4o,claude-opus-4-6
```

### Output

Displays:
- Five-dimensional coverage vector with visual bars
- Overall score (geometric mean)
- Weakest dimension identification
- Summary statistics (traces analyzed, tools observed, unique paths)

### Coverage Interpretation

| Score | Status |
|-------|--------|
| >= 80% | GOOD |
| 50-79% | MODERATE |
| < 50% | LOW |

---

## `agentassay report`

Generate a self-contained HTML report from trial results.

```bash
agentassay report [OPTIONS]
```

### Options

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--results PATH` | `-r` | Yes | JSON file with trial results |
| `--output PATH` | `-o` | No | Output HTML file path (default: `agentassay-report.html`) |

### Examples

```bash
# Generate with default output filename
agentassay report --results trials.json

# Custom output path
agentassay report -r results.json -o reports/latest.html
```

### Report Contents

The generated HTML report includes:
- Overall verdict (PASS / FAIL / INCONCLUSIVE / NO DATA)
- Summary table: total trials, passed, failed, pass rate, confidence interval
- Visual pass rate bar
- Methodology section: framework version, CI method, regression test, verdict semantics

The report is self-contained (no external CSS or JavaScript dependencies) and can be opened in any browser.

---

## `agentassay dashboard`

Launch the interactive dashboard for visualizing test results, trends, and behavioral fingerprints.

```bash
agentassay dashboard [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--port INTEGER` | `-p` | Port to run dashboard on (default: 8501) |
| `--host TEXT` | | Host to bind to (default: localhost) |
| `--no-browser` | | Don't auto-open browser |
| `--theme TEXT` | | Color theme: dark, light (default: dark) |

### Prerequisites

Install with dashboard extras:
```bash
pip install agentassay[dashboard]
```

### Examples

```bash
# Launch with defaults (opens browser automatically)
agentassay dashboard

# Custom port, no auto-open
agentassay dashboard --port 9000 --no-browser

# Light theme
agentassay dashboard --theme light

# Bind to all interfaces (for remote access)
agentassay dashboard --host 0.0.0.0
```

### Notes

The dashboard reads from the local results database. Run `agentassay run` at least once to have data to visualize. See [Dashboard Guide](../guides/dashboard.md) for full documentation.
