# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""pytest Integration Example for AgentAssay.

Shows how to use AgentAssay's pytest plugin with markers like
@pytest.mark.agentassay for statistical agent testing.

Prerequisites:
    pip install agentassay[dev]

Run this example:
    pytest examples/pytest-plugin/test_my_agent.py -v
"""

import pytest
from agentassay.core.models import ExecutionTrace, StepTrace
from agentassay.integrations.custom_adapter import CustomAdapter


def my_agent(input_data: dict) -> dict:
    """Simple agent for testing."""
    query = input_data.get("query", "")
    return {
        "output": f"Response: {query}",
        "success": True,
        "steps": [
            {"action": "llm_response", "llm_output": f"Thinking about {query}"}
        ]
    }


@pytest.mark.agentassay(trials=20, alpha=0.05, power=0.80)
def test_agent_basic_query():
    """Test agent responds to basic queries with statistical rigor."""
    adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")

    input_data = {"query": "What is 2+2?"}
    trace = adapter.run(input_data)

    # AgentAssay markers run this test 20 times and compute statistical verdict
    assert trace.success is True
    assert trace.output_data is not None


@pytest.mark.agentassay(trials=15, use_sprt=True)
def test_agent_handles_errors():
    """Test agent gracefully handles edge cases."""
    adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")

    input_data = {"query": ""}  # Edge case: empty query
    trace = adapter.run(input_data)

    assert trace.success is True  # Should not crash


def test_agent_trace_structure():
    """Standard unit test (not stochastic, runs once)."""
    adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")
    trace = adapter.run({"query": "Test"})

    assert isinstance(trace, ExecutionTrace)
    assert len(trace.steps) > 0
    assert trace.steps[0].action == "llm_response"


@pytest.mark.agentassay(trials=25, regression=True)
def test_no_behavioral_regression():
    """Regression test: compare against baseline traces."""
    adapter = CustomAdapter(my_agent, framework="custom", model="gpt-4o")

    # This marker instructs AgentAssay to:
    # 1. Load baseline traces from previous run
    # 2. Compute behavioral fingerprints
    # 3. Run Hotelling's T² test for regression
    trace = adapter.run({"query": "Production query"})

    assert trace.success is True


# To run with coverage:
#   pytest examples/pytest-plugin/test_my_agent.py --cov=agentassay --cov-report=html
