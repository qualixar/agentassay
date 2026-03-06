# AWS Bedrock Agents Integration

> pip install agentassay[bedrock]

## Quick Start

```python
from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter

adapter = BedrockAgentsAdapter(
    agent_id="ABCDEFGHIJ",
    agent_alias_id="TSTALIASID",
    region="us-east-1",
    model="anthropic.claude-3-sonnet"
)

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | Bedrock Agent ID (10-char alphanumeric) |
| `agent_alias_id` | `str` | Agent alias ID (`TSTALIASID` for draft) |
| `session_id` | `str | None` | Session ID (auto-generated if not provided) |
| `region` | `str | None` | AWS region (uses boto3 defaults if not set) |
| `model` | `str` | Model identifier (informational) |

The adapter parses EventStream responses with `enableTrace=True` to capture rationale, action group invocations, and knowledge base lookups.

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
