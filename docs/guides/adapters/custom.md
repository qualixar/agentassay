# Custom Adapter — Test ANY Agent

> pip install agentassay

The CustomAdapter is the universal escape hatch: wrap **any** Python callable, regardless of framework.

## Quick Start (2 minutes)

```python
from agentassay.integrations import CustomAdapter
from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner

# 1. Your agent (any callable)
def my_agent(input_data: dict) -> str:
    return f"Response to {input_data['query']}"

# 2. Wrap with CustomAdapter
adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")

# 3. Run trials
scenario = TestScenario(
    scenario_id="test-1",
    name="Basic Test",
    input_data={"query": "What is 2+2?"}
)

assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## How It Works

CustomAdapter accepts **three return types** from your callable:

### 1. ExecutionTrace (zero overhead)

```python
from agentassay.core.models import ExecutionTrace, StepTrace

def my_agent(input_data: dict) -> ExecutionTrace:
    steps = [
        StepTrace(step_index=0, action="llm_response", duration_ms=100.0, llm_output="Thinking..."),
        StepTrace(step_index=1, action="tool_call", tool_name="search", duration_ms=200.0, tool_output="Results"),
    ]
    return ExecutionTrace(
        trace_id="abc",
        scenario_id=input_data.get("scenario_id", "default"),
        steps=steps,
        input_data=input_data,
        output_data="Final answer",
        success=True,
        total_duration_ms=300.0,
        total_cost_usd=0.0012,
        model="gpt-4o",
        framework="custom"
    )

adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")
```

### 2. dict (wrapped automatically)

```python
def my_agent(input_data: dict) -> dict:
    return {
        "output": "Final answer",
        "steps": [
            {"action": "llm_response", "llm_output": "Thinking..."},
            {"action": "tool_call", "tool_name": "search", "tool_output": "Results"}
        ],
        "cost": 0.0012,
        "success": True
    }

adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")
```

**Dict schema:**
- `output` or `result` (required): Agent output
- `steps` (optional): List of step dicts
- `cost` (optional): USD cost
- `success` (optional): Boolean, default `True`
- `error` (optional): Error message string

### 3. str (single-step trace)

```python
def my_agent(input_data: dict) -> str:
    return "The answer is 42"

adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")
```

Wrapped as a single `llm_response` step.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callable_fn` | `Callable[[dict], Any]` | **required** | Your agent function |
| `framework` | `str` | `"custom"` | Framework identifier |
| `model` | `str` | `"unknown"` | LLM model identifier |
| `agent_name` | `str | None` | `"custom-agent"` | Human-readable name |
| `metadata` | `dict | None` | `None` | Arbitrary metadata |

## Full Example

```python
from agentassay.integrations import CustomAdapter
from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner
import openai

def my_openai_agent(input_data: dict) -> dict:
    """Agent using raw OpenAI API."""
    query = input_data["query"]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )

    return {
        "output": response.choices[0].message.content,
        "steps": [
            {
                "action": "llm_response",
                "llm_input": query,
                "llm_output": response.choices[0].message.content,
                "duration_ms": 1500.0
            }
        ],
        "cost": response.usage.total_tokens * 0.00001,
        "success": True
    }

# Wrap and test
adapter = CustomAdapter(my_openai_agent, framework="openai", model="gpt-4o")

scenario = TestScenario(
    scenario_id="openai-test",
    name="OpenAI Query",
    input_data={"query": "What is the capital of France?"}
)

assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])

print(f"Pass rate: {verdict['openai-test'].pass_rate:.1%}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| TypeError: callable_fn must be callable | Ensure you pass a function, not a string or object |
| Missing `output` key in dict | Return `{"output": "..."}` or `{"result": "..."}` |
| Steps not captured | Provide `"steps"` key in returned dict |

## Next Steps

- [Quickstart Guide](../getting-started/quickstart.md)
- [All Framework Adapters](../api/integrations.md)
