# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""LangGraph Integration Example.

Shows how to test a LangGraph agent with AgentAssay. The adapter
automatically captures per-node execution as individual steps.

Prerequisites:
    pip install agentassay[langgraph]

Run this example:
    python examples/framework-specific/langgraph_example.py
"""

from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner
from agentassay.integrations.langgraph_adapter import LangGraphAdapter
from rich.console import Console

console = Console()


def create_mock_graph():
    """Create a mock LangGraph for demonstration.

    In a real application, replace this with your actual compiled graph:

    >>> from langgraph.graph import StateGraph
    >>> graph = StateGraph(YourState)
    >>> graph.add_node("agent", agent_node)
    >>> graph.add_node("tools", tool_node)
    >>> compiled = graph.compile()
    """
    try:
        from langgraph.graph import StateGraph
        from typing_extensions import TypedDict

        class AgentState(TypedDict):
            messages: list
            query: str
            result: str

        def agent_node(state):
            query = state.get("query", "")
            return {"result": f"Processed: {query}", "messages": state.get("messages", []) + ["agent_response"]}

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.set_finish_point("agent")

        return graph.compile()

    except ImportError:
        # LangGraph not installed — return a mock object
        console.print("[yellow]LangGraph not installed. Using mock graph.[/yellow]")
        console.print("[dim]Install with: pip install agentassay[langgraph][/dim]\n")

        class MockGraph:
            def stream(self, input_data, config=None):
                # Simulate streaming per-node output
                yield {"agent": {"result": f"Response to {input_data.get('query', 'query')}"}}

        return MockGraph()


def main():
    console.print("[bold cyan]LangGraph + AgentAssay Integration[/bold cyan]\n")

    # Step 1: Create your LangGraph
    console.print("[yellow]Step 1:[/yellow] Create LangGraph (compiled)")
    compiled_graph = create_mock_graph()
    console.print("✓ Graph compiled\n")

    # Step 2: Wrap with LangGraphAdapter
    console.print("[yellow]Step 2:[/yellow] Wrap graph with adapter")
    adapter = LangGraphAdapter(
        graph=compiled_graph,
        model="gpt-4o",
        use_stream=True,  # Capture per-node granularity
        agent_name="my-langgraph-agent"
    )
    console.print("✓ Adapter created (streaming mode)\n")

    # Step 3: Define test scenario
    console.print("[yellow]Step 3:[/yellow] Define test scenario")
    scenario = TestScenario(
        scenario_id="langgraph-test",
        name="Query Processing Test",
        input_data={"query": "What is the capital of France?"},
        expected_properties={"max_steps": 5}
    )
    console.print(f"✓ Scenario: {scenario.name}\n")

    # Step 4: Configure assay
    console.print("[yellow]Step 4:[/yellow] Configure statistical testing")
    assay_config = AssayConfig(
        num_trials=15,
        use_sprt=True,
        significance_level=0.05,
        power=0.80
    )
    console.print(f"✓ Config: {assay_config.num_trials} trials max, SPRT enabled\n")

    # Step 5: Run trials
    console.print("[yellow]Step 5:[/yellow] Run stochastic trials")
    runner = TrialRunner(
        agent_callable=adapter.to_callable(),
        assay_config=assay_config,
        agent_config=adapter.get_config()
    )

    with console.status("[bold green]Running trials..."):
        verdict = runner.run_trials([scenario])

    console.print("✓ Trials complete\n")

    # Step 6: Check verdict
    console.print("[yellow]Step 6:[/yellow] Statistical verdict")
    if scenario.scenario_id in verdict:
        result = verdict[scenario.scenario_id]
        console.print(f"[bold]Pass Rate:[/bold] {result.pass_rate:.1%}")
        console.print(f"[bold]Trials Run:[/bold] {result.num_trials}")
        console.print(f"[bold]Decision:[/bold] {result.decision}")

    console.print("\n[dim]Per-node steps are captured automatically via graph.stream()[/dim]")
    console.print("[dim]Each LangGraph node becomes a StepTrace in the ExecutionTrace.[/dim]")


if __name__ == "__main__":
    main()
