# LangGraph Integration

> pip install agentassay[langgraph]

## Quick Start (2 minutes)

```python
from langgraph.graph import StateGraph
from agentassay.integrations import LangGraphAdapter
from agentassay.core.trial_runner import TrialRunner
from agentassay.core.models import AssayConfig, TestScenario

# 1. Your existing LangGraph
graph = StateGraph(YourState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
compiled_graph = graph.compile()

# 2. Wrap with adapter
adapter = LangGraphAdapter(graph=compiled_graph, model="gpt-4o")

# 3. Define test scenario
scenario = TestScenario(
    scenario_id="test-1",
    name="Query Test",
    input_data={"query": "What is the capital of France?"}
)

# 4. Run stochastic trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## How It Works

The LangGraphAdapter uses `graph.stream()` to capture per-node execution. Each node's output becomes a `StepTrace`:

- **Node executions** → `StepTrace` with `action="tool_call"` or `action="llm_response"`
- **Messages** → Classified by type (AIMessage, ToolMessage, HumanMessage)
- **Tool calls** → Extracted from `tool_calls` attribute on messages
- **Timing** → Per-node duration measured via `time.perf_counter()`

## Full Example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from agentassay.integrations import LangGraphAdapter
from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner

# Define state
class AgentState(TypedDict):
    messages: list
    query: str
    result: str

# Define nodes
def agent_node(state: AgentState):
    query = state["query"]
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([HumanMessage(content=query)])
    return {
        "messages": state["messages"] + [response],
        "result": response.content
    }

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")
compiled_graph = graph.compile()

# Wrap and test
adapter = LangGraphAdapter(
    graph=compiled_graph,
    model="gpt-4o",
    use_stream=True,  # Per-node granularity
    agent_name="my-agent"
)

scenario = TestScenario(
    scenario_id="basic-test",
    name="Basic Query",
    input_data={"query": "What is 2+2?"},
    expected_properties={"max_steps": 5}
)

assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])

# Check verdict
result = verdict["basic-test"]
print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Decision: {result.decision}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `CompiledGraph` | **required** | Your compiled LangGraph instance |
| `model` | `str` | `"unknown"` | LLM model identifier (e.g., `"gpt-4o"`) |
| `config` | `dict | None` | `None` | LangGraph invocation config (`recursion_limit`, `configurable`, etc.) |
| `use_stream` | `bool` | `True` | Use `graph.stream()` for per-node traces. If `False`, uses `graph.invoke()` (single step) |
| `agent_name` | `str | None` | `"langgraph-agent"` | Human-readable name for the agent |
| `metadata` | `dict | None` | `None` | Arbitrary metadata attached to every trace |

## Common Patterns

### Pattern 1: Basic Regression Testing

```python
from agentassay.efficiency.regression import behavioral_regression_test

# Collect baseline traces (yesterday's production)
baseline_traces = [adapter.run(scenario.input_data) for _ in range(10)]
baseline_fp = [compute_behavioral_fingerprint(t) for t in baseline_traces]

# Collect current traces (today's candidate)
current_traces = [adapter.run(scenario.input_data) for _ in range(10)]
current_fp = [compute_behavioral_fingerprint(t) for t in current_traces]

# Statistical regression test
result = behavioral_regression_test(baseline_fp, current_fp, alpha=0.05)
if result.regressed:
    raise ValueError(f"Regression detected! T²={result.t_squared:.2f}, p={result.p_value:.4f}")
```

### Pattern 2: With Behavioral Fingerprinting

```python
from agentassay.efficiency.fingerprint import compute_behavioral_fingerprint
from agentassay.efficiency.budget import AdaptiveBudgetOptimizer

# Run pilot trials
pilot_traces = [adapter.run(scenario.input_data) for _ in range(5)]
pilot_fingerprints = [compute_behavioral_fingerprint(t) for t in pilot_traces]

# Compute optimal N based on observed variance
pilot_pass_rate = sum(t.success for t in pilot_traces) / len(pilot_traces)
pilot_variance = np.var([1 if t.success else 0 for t in pilot_traces])

optimizer = AdaptiveBudgetOptimizer(
    alpha=0.05,
    power=0.80,
    effect_size=0.10,
    pilot_pass_rate=pilot_pass_rate,
    pilot_variance=pilot_variance
)

optimal_n = optimizer.compute_optimal_trials()
print(f"Run {optimal_n} trials instead of typical 100+ (2-4x savings)")

# Run with optimal budget
assay_config = AssayConfig(num_trials=optimal_n)
verdict = runner.run_trials([scenario])
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Framework not installed | `pip install agentassay[langgraph]` |
| Import error: `langgraph` module | Ensure `langgraph>=0.2.0` is installed |
| No steps captured | Set `use_stream=True` in adapter constructor |
| Empty `tool_name` in steps | LangGraph node doesn't expose tool metadata — check node implementation |
| Cost not tracked | LangGraph doesn't expose token costs natively. Set `total_cost_usd` manually in traces if needed. |

## Next Steps

- [Quickstart Guide](../getting-started/quickstart.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
- [CI/CD Integration](../ci-cd-integration.md)
- [Full API Reference](../api/integrations.md#langgraph-adapter)
