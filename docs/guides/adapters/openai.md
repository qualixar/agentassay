# OpenAI Agents SDK Integration

> pip install agentassay[openai]

## Quick Start

```python
from agents import Agent
from agentassay.integrations import OpenAIAgentsAdapter

agent = Agent(name="assistant", model="gpt-4o", tools=[...])
adapter = OpenAIAgentsAdapter(agent=agent, model="gpt-4o")

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | OpenAI Agents SDK agent instance |
| `model` | `str` | Model identifier (auto-detected from agent if not provided) |

The adapter uses `Runner.run_sync()` and extracts steps from `RunResult.new_items` (messages, tool calls, handoffs).

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
