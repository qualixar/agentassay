# Framework Adapters

AgentAssay works with any AI agent framework through adapters. An adapter translates your framework's execution into the standardized `ExecutionTrace` format that AgentAssay understands.

> **Looking for a specific framework?** Jump to:
> [LangGraph](adapters/langgraph.md) | [CrewAI](adapters/crewai.md) | [AutoGen](adapters/autogen.md) | [OpenAI Agents](adapters/openai.md) | [smolagents](adapters/smolagents.md) | [Semantic Kernel](adapters/semantic-kernel.md) | [AWS Bedrock](adapters/bedrock.md) | [MCP](adapters/mcp.md) | [Vertex AI](adapters/vertex.md) | [Custom](adapters/custom.md)

## Supported Frameworks

| Framework | Install Extra | Adapter Class | Status |
|-----------|--------------|---------------|--------|
| LangGraph | `agentassay[langgraph]` | `LangGraphAdapter` | ✅ Production |
| CrewAI | `agentassay[crewai]` | `CrewAIAdapter` | ✅ Production |
| AutoGen | `agentassay[autogen]` | `AutoGenAdapter` | ✅ Production |
| OpenAI Agents SDK | `agentassay[openai]` | `OpenAIAgentsAdapter` | ✅ Production |
| smolagents | `agentassay[smolagents]` | `SmolAgentsAdapter` | ✅ Production |
| Semantic Kernel (Microsoft) | `agentassay[semantic-kernel]` | `SemanticKernelAdapter` | ✅ Production |
| AWS Bedrock Agents | `agentassay[bedrock]` | `BedrockAgentsAdapter` | ✅ Production |
| Anthropic MCP | `agentassay[mcp]` | `MCPToolsAdapter` | ✅ Production |
| Google Vertex AI Agents | `agentassay[vertex]` | `VertexAIAgentsAdapter` | ✅ Production |
| Custom | (included) | `CustomAdapter` | ✅ Production |

> All 10 adapters are production-ready. Any framework can be integrated using `CustomAdapter`.

## Using the Adapter Factory

The simplest way to create an adapter is through the factory function:

```python
from agentassay.integrations import create_adapter

# Automatically selects the right adapter based on framework name
adapter = create_adapter("langgraph", your_langgraph_agent)

# The adapter returns a callable that produces ExecutionTrace objects
trace = adapter({"prompt": "What is 2+2?"})
```

## Custom Agents

If your agent does not use one of the supported frameworks, use the custom adapter:

```python
from agentassay.core.models import ExecutionTrace, StepTrace

def my_custom_agent(input_data: dict) -> ExecutionTrace:
    """Wrap your agent logic and return an ExecutionTrace."""
    # Run your agent
    response = your_agent.run(input_data["prompt"])

    # Build step traces manually
    steps = [
        StepTrace(
            step_index=0,
            action="llm_response",
            llm_input=input_data["prompt"],
            llm_output=response.text,
            model=response.model,
            duration_ms=response.latency_ms,
        ),
    ]

    # Add tool call steps if your agent used tools
    for i, tool_call in enumerate(response.tool_calls, start=1):
        steps.append(StepTrace(
            step_index=i,
            action="tool_call",
            tool_name=tool_call.name,
            tool_input=tool_call.args,
            tool_output=tool_call.result,
            duration_ms=tool_call.latency_ms,
        ))

    return ExecutionTrace(
        scenario_id=input_data.get("scenario_id", "default"),
        steps=steps,
        input_data=input_data,
        output_data={"response": response.text},
        success=not response.error,
        error=response.error,
        total_duration_ms=response.total_latency_ms,
        total_cost_usd=response.cost,
        model=response.model,
        framework="custom",
    )
```

## Using Adapters with the Trial Runner

```python
from agentassay.core.models import AgentConfig, AssayConfig, TestScenario
from agentassay.core.runner import TrialRunner
from agentassay.integrations import create_adapter

# Create the adapter
agent_callable = create_adapter("crewai", your_crew)

# Configure the agent under test
agent_config = AgentConfig(
    agent_id="research-crew",
    name="Research Crew",
    framework="crewai",
    model="gpt-4o",
    version="1.2.0",
)

# Configure the test run
assay_config = AssayConfig(
    num_trials=50,
    significance_level=0.05,
    power=0.80,
)

# Create the runner
runner = TrialRunner(
    agent_callable=agent_callable,
    config=assay_config,
    agent_config=agent_config,
)

# Define a scenario and run
scenario = TestScenario(
    scenario_id="research-task",
    name="Research a topic",
    input_data={"prompt": "Summarize recent advances in protein folding"},
    expected_properties={"max_steps": 20, "max_cost_usd": 0.50},
)

results = runner.run_trials(scenario)
```

## Using Adapters with pytest

The `trial_runner` fixture accepts any callable that returns an `ExecutionTrace`:

```python
import pytest
from agentassay.integrations import create_adapter

@pytest.fixture
def my_agent():
    """Create the agent under test."""
    agent = create_adapter("langgraph", your_langgraph_app)
    return agent

@pytest.mark.agentassay(n=30, threshold=0.80)
def test_langgraph_agent(trial_runner, my_agent):
    from agentassay.core.models import AgentConfig
    config = AgentConfig(
        agent_id="my-lg-agent",
        name="LangGraph Agent",
        framework="langgraph",
        model="gpt-4o",
    )
    runner = trial_runner(my_agent, agent_config=config)
    results = runner.run_trials(scenario)
    assert_pass_rate([r.passed for r in results], threshold=0.80)
```

## Writing a Custom Adapter

If you need to integrate with a framework not listed above, implement a function that takes your agent and returns a callable:

```python
from agentassay.core.models import ExecutionTrace, StepTrace

def my_framework_adapter(agent) -> callable:
    """Adapt MyFramework agents to AgentAssay's ExecutionTrace format."""

    def run(input_data: dict) -> ExecutionTrace:
        # Execute the agent using your framework's API
        result = agent.invoke(input_data)

        # Convert your framework's output to StepTrace objects
        steps = []
        for i, step in enumerate(result.steps):
            steps.append(StepTrace(
                step_index=i,
                action=step.type,
                tool_name=step.tool if step.type == "tool_call" else None,
                tool_input=step.tool_args if step.type == "tool_call" else None,
                tool_output=step.tool_result if step.type == "tool_call" else None,
                llm_input=step.prompt if step.type == "llm_response" else None,
                llm_output=step.response if step.type == "llm_response" else None,
                model=step.model,
                duration_ms=step.duration_ms,
            ))

        return ExecutionTrace(
            scenario_id=input_data.get("scenario_id", "default"),
            steps=steps,
            input_data=input_data,
            output_data=result.final_output,
            success=result.success,
            error=str(result.error) if result.error else None,
            total_duration_ms=result.total_duration_ms,
            total_cost_usd=result.total_cost,
            model=result.model,
            framework="my_framework",
        )

    return run
```

The key requirement is that your adapter returns an `ExecutionTrace` with accurate `StepTrace` entries. The more detail you capture in the steps, the more coverage metrics and mutation testing can analyze.
