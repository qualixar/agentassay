# smolagents (HuggingFace) Integration

> pip install agentassay[smolagents]

## Quick Start

```python
from smolagents import CodeAgent, HfApiModel
from agentassay.integrations import SmolAgentsAdapter

model = HfApiModel(model_id="Qwen/Qwen2.5-72B")
agent = CodeAgent(tools=[], model=model)
adapter = SmolAgentsAdapter(agent=agent, model="Qwen/Qwen2.5-72B")

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `CodeAgent | ToolCallingAgent | MultiStepAgent` | smolagents agent |
| `model` | `str` | Model identifier (auto-detected from `agent.model` if not provided) |

The adapter extracts steps from `agent.logs` (reasoning, tool calls, observations).

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
