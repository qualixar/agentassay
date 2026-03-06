# CrewAI Integration

> pip install agentassay[crewai]

## Quick Start (2 minutes)

```python
from crewai import Agent, Task, Crew
from agentassay.integrations import CrewAIAdapter
from agentassay.core.trial_runner import TrialRunner
from agentassay.core.models import AssayConfig, TestScenario

# 1. Your existing CrewAI crew
researcher = Agent(role="Researcher", goal="Research topics")
task = Task(description="Research {topic}", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])

# 2. Wrap with adapter
adapter = CrewAIAdapter(crew=crew, model="gpt-4o")

# 3. Run stochastic trials
scenario = TestScenario(
    scenario_id="test-1",
    name="Research Test",
    input_data={"topic": "AI testing"}
)

assay_config = AssayConfig(num_trials=20, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## How It Works

CrewAI organizes work as tasks assigned to agents. The adapter:

- **Task executions** → Each task becomes a `StepTrace`
- **Agent attribution** → `metadata["crewai_agent"]` tracks which agent ran the task
- **Tool usage** → Extracted from `tools_used` attribute on `TaskOutput`
- **Cost tracking** → Reads `token_usage.total_cost` if available

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crew` | `Crew` | **required** | Your CrewAI crew instance |
| `model` | `str` | `"unknown"` | LLM model identifier |
| `agent_name` | `str | None` | `"crewai-agent"` | Human-readable name |
| `metadata` | `dict | None` | `None` | Arbitrary metadata |

## Full Example

```python
from crewai import Agent, Task, Crew
from agentassay.integrations import CrewAIAdapter

# Create crew
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with 10 years experience",
)

writer = Agent(
    role="Writer",
    goal="Write clear reports",
    backstory="Technical writer",
)

research_task = Task(
    description="Research: {topic}",
    agent=researcher,
    expected_output="Research findings"
)

write_task = Task(
    description="Write report on research",
    agent=writer,
    expected_output="Final report"
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])

# Test
adapter = CrewAIAdapter(crew=crew, model="gpt-4o")
scenario = TestScenario(
    scenario_id="crew-test",
    name="Research + Write",
    input_data={"topic": "AI agent reliability"}
)

assay_config = AssayConfig(num_trials=15, use_sprt=True)
runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
verdict = runner.run_trials([scenario])
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Framework not installed | `pip install agentassay[crewai]` |
| No per-task steps | Ensure CrewAI v0.80+ (earlier versions had different `TaskOutput` API) |

## Next Steps

- [Quickstart Guide](../getting-started/quickstart.md)
- [Token-Efficient Testing](../concepts/token-efficient-testing.md)
