# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AutoGen/AG2 Integration Example.

Prerequisites: pip install agentassay[autogen]
Run: python examples/framework-specific/autogen_example.py
"""

from agentassay.core.models import AssayConfig, TestScenario
from agentassay.core.trial_runner import TrialRunner
from agentassay.integrations.autogen_adapter import AutoGenAdapter
from rich.console import Console

console = Console()


def create_mock_agent():
    """Mock AutoGen agent. Replace with your actual agent:
    >>> from autogen_agentchat import AssistantAgent
    >>> agent = AssistantAgent(name="assistant", llm_config={...})
    """
    try:
        from autogen_agentchat import AssistantAgent
        return AssistantAgent(name="assistant")
    except ImportError:
        console.print("[yellow]AutoGen not installed.[/yellow]")
        console.print("[dim]Install: pip install agentassay[autogen][/dim]\n")

        class MockAgent:
            name = "mock-agent"
            llm_config = {"model": "gpt-4o"}
        return MockAgent()


def main():
    console.print("[bold cyan]AutoGen + AgentAssay[/bold cyan]\n")

    agent = create_mock_agent()
    adapter = AutoGenAdapter(agent=agent, model="gpt-4o")

    scenario = TestScenario(
        scenario_id="autogen-test",
        name="Math Problem",
        input_data={"message": "Solve: 2x + 5 = 15"},
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
