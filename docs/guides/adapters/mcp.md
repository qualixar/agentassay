# MCP (Model Context Protocol) Integration

> pip install agentassay[mcp]

## Quick Start

```python
from agentassay.integrations.mcp_adapter import MCPToolsAdapter

# Option 1: Direct MCP client
adapter = MCPToolsAdapter(client=my_mcp_client, tools=[], model="claude-sonnet-4")

# Option 2: Anthropic client + MCP tools
from anthropic import Anthropic
adapter = MCPToolsAdapter.from_anthropic_client(
    client=Anthropic(),
    tools=mcp_tool_list,
    model="claude-sonnet-4-20250514"
)

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | MCP Client or Anthropic Client | Client instance |
| `tools` | `list[dict] | None` | MCP tool definitions (required for Anthropic mode) |
| `model` | `str` | Model identifier |

Captures `tools/call` as `tool_call` steps and `resources/read` as `retrieval` steps.

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
