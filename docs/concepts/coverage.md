# Coverage Metrics

## Why Agent Coverage Is Different

In traditional software testing, code coverage measures which lines, branches, or functions were exercised. For AI agents, there is no static source code to cover. Instead, coverage must measure how thoroughly the test suite exercises the agent's **behavioral space**.

AgentAssay defines a five-dimensional coverage model. Each dimension measures a distinct aspect of agent behavior, and together they form a **coverage tuple** that gives a holistic view of test thoroughness.

## The Five Dimensions

### 1. Tool Coverage (C_tool)

**What it measures:** The fraction of the agent's available tools that were invoked during testing.

**Why it matters:** If your agent has access to 10 tools but your tests only trigger 3 of them, you have no confidence that the other 7 tools work correctly after a change.

**Calculation:**

```
C_tool = |tools invoked in tests| / |total known tools|
```

**Example:** An agent has tools: `search`, `calculate`, `write_file`, `read_file`, `send_email`. Tests invoke `search`, `calculate`, and `read_file`. C_tool = 3/5 = 0.60.

### 2. Path Coverage (C_path)

**What it measures:** The fraction of distinct behavioral paths (sequences of actions) observed during testing relative to all paths seen.

**Why it matters:** Agents take different reasoning paths depending on the input. If your tests only exercise the "happy path," you miss edge-case behaviors like error recovery, tool retries, or fallback strategies.

**Calculation:**

```
C_path = |unique action sequences observed| / |total unique paths across all runs|
```

A "path" is the ordered sequence of action types in an execution trace (e.g., `llm_response -> tool_call -> llm_response -> tool_call`).

### 3. State Coverage (C_state)

**What it measures:** The fraction of distinct intermediate states the agent visited during testing.

**Why it matters:** Agents maintain internal state across steps (memory, accumulated context, tool outputs). A regression might only manifest when the agent reaches a specific state that your tests never trigger.

**Calculation:**

```
C_state = |unique states visited| / |total states observed across all runs|
```

States are derived from the metadata and tool outputs at each step of the execution trace.

### 4. Boundary Coverage (C_boundary)

**What it measures:** How well the test suite exercises edge cases and boundary conditions: timeouts, cost limits, maximum step counts, empty inputs, and error conditions.

**Why it matters:** Most agent failures occur at boundaries -- when the context window is full, when a tool returns an error, when the budget runs out. If tests only use "normal" inputs, boundary behavior is untested.

**Boundary conditions tracked:**
- Maximum step count reached
- Timeout triggered
- Cost limit approached or exceeded
- Empty or minimal inputs
- Error/exception paths
- Tool failure handling

### 5. Model Coverage (C_model)

**What it measures:** The fraction of model variants the agent is tested against.

**Why it matters:** An agent that works with one model may fail with another. If you switch from GPT-4o to Claude Opus, do your tests verify that the agent still works? Model coverage tracks whether your test suite validates behavior across all supported model backends.

**Calculation:**

```
C_model = |models tested| / |total known models|
```

## The Coverage Tuple

The five dimensions combine into a single coverage tuple:

```
C = (C_tool, C_path, C_state, C_boundary, C_model)
```

**Example:**

```
C = (0.80, 0.65, 0.72, 0.40, 0.50)
```

This tells you at a glance: tool coverage is strong (80%), boundary coverage is weak (40%), and model coverage is limited (50%).

## Overall Coverage Score

The overall score is the geometric mean of all five dimensions:

```
C_overall = (C_tool * C_path * C_state * C_boundary * C_model) ^ (1/5)
```

The geometric mean is used instead of the arithmetic mean because it penalizes low scores in any single dimension. A test suite with 100% tool coverage but 0% boundary coverage should not get a 80% average -- it should get 0%.

## Interpreting Coverage

| Score | Interpretation |
|-------|---------------|
| >= 0.80 | Strong coverage. The test suite exercises the agent thoroughly in this dimension. |
| 0.50 - 0.79 | Moderate coverage. Some behavioral aspects are untested. Consider adding scenarios. |
| < 0.50 | Weak coverage. Significant gaps exist. A regression in this area would likely go undetected. |

## Improving Coverage

For each dimension, there are specific strategies:

| Dimension | How to improve |
|-----------|---------------|
| Tool | Add scenarios that require different tool combinations |
| Path | Add scenarios that trigger alternative reasoning paths (error cases, ambiguous inputs) |
| State | Add scenarios that produce different intermediate states (multi-step chains, varied contexts) |
| Boundary | Add edge-case scenarios: empty inputs, max-length inputs, timeout-inducing tasks |
| Model | Run the same test suite against multiple model backends |

## Using Coverage from the CLI

```bash
# View coverage for a results file
agentassay coverage --results trials.json --tools search,calculate,write

# Specify known models for model coverage
agentassay coverage --results trials.json --tools search,calculate --models gpt-4o,claude-opus-4-6
```

## Using Coverage from Python

```python
from agentassay.coverage import AgentCoverageCollector

collector = AgentCoverageCollector(
    known_tools={"search", "calculate", "write_file"},
    known_models={"gpt-4o", "claude-opus-4-6"},
)

# Feed execution traces into the collector
for trace in execution_traces:
    collector.update(trace)

# Get the coverage snapshot
snapshot = collector.snapshot()
print(f"Overall: {snapshot.overall:.2%}")
print(f"Weakest: {snapshot.weakest}")
print(f"Dimensions: {snapshot.dimensions}")
```
