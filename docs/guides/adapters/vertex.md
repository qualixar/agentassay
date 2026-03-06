# Google Vertex AI Agents Integration

> pip install agentassay[vertex]

## Quick Start

```python
import vertexai
from vertexai.generative_models import GenerativeModel
from agentassay.integrations.vertex_adapter import VertexAIAgentsAdapter

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-2.0-flash")

adapter = VertexAIAgentsAdapter(
    generative_model=model,
    model="gemini-2.0-flash",
    project_id="my-project",
    location="us-central1"
)

# Run trials
assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `generative_model` | `GenerativeModel` | Vertex AI model instance |
| `tools` | `list[Tool] | None` | Vertex AI tools (function declarations, grounding, retrieval) |
| `project_id` | `str | None` | GCP project ID (informational) |
| `location` | `str` | GCP region (default: `us-central1`) |
| `model` | `str` | Model identifier (auto-detected from model object if not provided) |

Extracts `function_call`, `text`, and `grounding_metadata` from response parts as separate steps. Includes token usage and cost estimation.

## Next Steps
- [Custom Adapter](custom.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
