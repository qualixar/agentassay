# AutoGen/AG2 Integration

> pip install agentassay[autogen]

## Quick Start

```python
from autogen_agentchat import AssistantAgent
from agentassay.integrations import AutoGenAdapter

agent = AssistantAgent(name="assistant", llm_config={"model": "gpt-4o"})
adapter = AutoGenAdapter(agent=agent, model="gpt-4o")

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | AutoGen agent (AssistantAgent, ConversableAgent, etc.) |
| `user_proxy` | `Agent | None` | Optional UserProxyAgent for `initiate_chat()` pattern |
| `model` | `str` | Model identifier (auto-detected from `llm_config` if not provided) |

Supports both AutoGen v0.4+ (`run()`) and legacy (`initiate_chat()`) patterns.

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
