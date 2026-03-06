# Semantic Kernel (Microsoft) Integration

> pip install agentassay[semantic-kernel]

## Quick Start

```python
from semantic_kernel import Kernel
from agentassay.integrations.semantic_kernel_adapter import SemanticKernelAdapter

kernel = Kernel()
# Add plugins and services...

adapter = SemanticKernelAdapter(
    kernel=kernel,
    model="gpt-4o",
    plugin_name="chat",
    function_name="respond"
)

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | `Kernel` | Semantic Kernel instance |
| `model` | `str` | LLM model identifier |
| `plugin_name` | `str | None` | Plugin to invoke (optional) |
| `function_name` | `str | None` | Function to invoke (required if `plugin_name` is set) |
| `config` | `dict | None` | Extra kwargs for `kernel.invoke()` |

Uses `FunctionInvocationFilter` for per-function trace capture when available.

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
