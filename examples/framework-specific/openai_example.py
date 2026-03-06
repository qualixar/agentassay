# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""OpenAI Agents SDK Integration Example.

Prerequisites: pip install agentassay[openai]
Run: python examples/framework-specific/openai_example.py
"""

from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner
from agentassay.integrations.openai_adapter import OpenAIAgentsAdapter
from rich.console import Console

console = Console()


def create_mock_agent():
    """Mock OpenAI Agent. Replace with your actual agent:
    >>> from agents import Agent
    >>> agent = Agent(name="assistant", model="gpt-4o", tools=[...])
    """
    try:
        from agents import Agent
        return Agent(name="assistant", model="gpt-4o")
    except ImportError:
        console.print("[yellow]OpenAI Agents SDK not installed.[/yellow]")
        console.print("[dim]Install: pip install agentassay[openai][/dim]\n")

        class MockAgent:
            name = "mock-agent"
            model = "gpt-4o"
        return MockAgent()


def main():
    console.print("[bold cyan]OpenAI Agents + AgentAssay[/bold cyan]\n")

    agent = create_mock_agent()
    adapter = OpenAIAgentsAdapter(agent=agent, model="gpt-4o")

    scenario = TestScenario(
        scenario_id="openai-test",
        name="Assistant Query",
        input_data={"query": "Summarize this document"},
    )

    assay_config = AssayConfig(num_trials=15, use_sprt=True)
    runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())

    with console.status("[green]Running..."):
        verdict = runner.run_trials([scenario])

    result = verdict.get(scenario.scenario_id)
    if result:
        console.print(f"[bold]Pass Rate:[/bold] {result.pass_rate:.1%}")
        console.print(f"[bold]Decision:[/bold] {result.decision}")


if __name__ == "__main__":
    main()
